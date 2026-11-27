"""Controlled CClinguist baseline / multi-step evidence fusion (MEF).
Uses the original frozen classifier, original actor/critic architecture and
full-DTW reward, with common PPO and offline evaluation correctness fixes.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import f1_score

# The upstream classifier module sets CUDA_VISIBLE_DEVICES at import time.
# Restore the caller's setting before constructing any CUDA tensors.
_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
from model_classify_unknown import Classifier
if _visible is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible
from RL import ActorCritic
from problem import Dataset

K = 12
EPS = 1e-8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_dtw(shared, path, split, device):
    meta = json.loads((shared / "manifest.json").read_text())
    envs = meta[split + "_envs"]
    assert os.listdir(path) == envs, "Profile order differs from DTW manifest"
    assert str(path.resolve()) == meta[split + "_path"]
    assert meta["test_traces_used"] is False
    table = np.loadtxt(shared / (split + "_dtw.txt"), delimiter="\t", ndmin=2)
    assert table.shape == (len(envs)*78, 4)
    matrix = np.full((len(envs), K, K), np.nan, dtype=np.float32)
    for e, i, j, value in table:
        assert e == int(e) and i == int(i) and j == int(j)
        e, i, j = int(e)-1, int(i), int(j)
        assert 0 <= e < len(envs) and 0 <= i <= j < K
        assert np.isnan(matrix[e,i,j]) and np.isfinite(value) and value >= 0
        matrix[e,i,j] = matrix[e,j,i] = value
    assert np.isfinite(matrix).all()
    assert np.allclose(np.diagonal(matrix, axis1=1, axis2=2), 0)
    return torch.tensor(matrix, device=device), envs


def normalize(p):
    return p / p.sum(-1, keepdim=True).clamp_min(EPS)


def update_evidence(p, previous, mask, mode, decay, first):
    if mode == "baseline":
        return normalize(p * mask)
    if mode == "no_mask" or first:
        return p
    fused = torch.softmax(decay * previous.clamp_min(EPS).log()
                          + p.clamp_min(EPS).log(), dim=-1)
    if mode == "fusion_mask":
        return normalize(fused * mask)
    return fused


def next_state(p, previous, mask, mode, decay, first, threshold=0.7):
    belief = update_evidence(p, previous, mask, mode, decay, first)
    pred = belief.argmax(-1)
    confident = belief.max(-1).values >= threshold
    if mode in ("baseline", "fusion_mask"):
        keep = (belief > 0.3 / K) | F.one_hot(pred, K).bool()
        new_mask = torch.where(confident[:,None], mask, mask & keep)
        state = normalize(belief * new_mask)
    else:
        new_mask = mask
        state = belief
    return state, new_mask, pred, confident, belief


def distribution(policy, state, dtw, blocked):
    # Same D @ p representation as upstream einsum, without redundant
    # input repeats. CNN architecture and actor/critic heads are unchanged.
    b, e = len(state), len(dtw)
    v = torch.einsum("eij,bj->bei", dtw, state)
    x = v.unsqueeze(-1).expand(b,e,K,K).reshape(b*e,1,K,K)
    features_a = policy.cnn_layer_actor(x).reshape(b,e,-1)
    features_c = policy.cnn_layer_critic(x).reshape(b,e,-1)
    logits = policy.actor[:-1](features_a).squeeze(-1)
    assert (~blocked).any(-1).all()
    assert torch.isfinite(logits).all(), "Non-finite actor logits"
    logits = logits.masked_fill(blocked, -torch.inf)
    logits = logits - logits.max(-1,keepdim=True).values
    dist = Categorical(logits=logits)
    value = policy.critic(features_c).squeeze(-1).mean(-1)
    return dist, value


def make_policy(device):
    cfg = SimpleNamespace(train=True, kernel_size=2, stride=2,
                          actor_h1=128, actor_h2=64)
    return ActorCritic(K,32,cfg).to(device)


def load_observations(args, dtw, envs, device, training, index):
    path = args.train_data if training else args.test_data
    n = 7 if training else 1
    ds = Dataset(args.batch_size, str(path), None, [index])
    ds.make_dataset_1209(args.batch_size, train=training,
                         train_data_num=7 if training else 5)
    assert os.listdir(path) == envs
    assert ds.input_data_num == n
    data = ds.input
    m = K*n
    assert len(data) == len(envs)*m
    labels = data[:,4].long().reshape(len(envs),m)
    assert (labels == labels[:1]).all(), "Class/sample ordering mismatch"
    assert (torch.bincount(labels[0], minlength=K) == n).all()
    assert torch.equal(data[:,1].long().reshape(len(envs),m),
                       torch.arange(1,len(envs)+1)[:,None].expand(-1,m))
    assert torch.equal(data[:,5].long().reshape(len(envs),m),
                       torch.arange(m)[None,:].expand(len(envs),-1))
    max_len = (data.shape[1]-6)//2
    assert data.shape[1] == 6 + 2*max_len
    assert ((data[:,0] > 0) & (data[:,0] <= max_len)).all()

    # Classifier is frozen. Precompute each deterministic trace prediction
    # once, preserving upstream dataset-wide padding and original features.
    clf = Classifier(K,128,8,8,2,0.2,1.2905981663049348,
                     0.00485668544506156).to(device)
    weights = torch.load(args.classifier, map_location=device)
    clf.load_state_dict(weights["state_dict"])
    clf.eval()
    pred = []
    with torch.no_grad():
        for chunk in data.split(args.batch_size):
            chunk = chunk.to(device)
            logits = clf(chunk[:,6:].reshape(-1,max_len,2),
                         chunk[:,0].long(),chunk[:,2:4],max_len)
            pred.append(logits.softmax(-1))
    probs = torch.cat(pred).reshape(len(envs),m,K)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1),torch.ones_like(probs[:,:,0]),atol=1e-5)
    print(f"[MEF] Observations PASS: {tuple(probs.shape)} index={index}",flush=True)
    return probs, labels[0].to(device)


def rollout(policy, dtw, probs, labels, ids, args, training):
    device = probs.device
    m = probs.shape[1]
    env = ids // m
    slot = ids % m
    true = labels[slot]
    b = len(ids)
    active = torch.ones(b,dtype=torch.bool,device=device)
    state = torch.full((b,K),1/K,device=device)
    mask = torch.ones_like(state,dtype=torch.bool)
    visited = torch.zeros(b,len(dtw),dtype=torch.bool,device=device)
    steps = torch.zeros(b,dtype=torch.long,device=device)
    final = torch.zeros_like(true)
    wrong_eliminated = torch.zeros_like(active)
    first_pred = torch.zeros_like(true)
    buf = []
    action_history = torch.full((b,args.max_steps),-1,device=device,dtype=torch.long)
    with torch.no_grad():
        for t in range(args.max_steps):
            # Same full history mask for both methods: repeated offline
            # profile/sample lookup returns exactly the same observation.
            blocked = visited.clone()
            dist, val = distribution(policy,state,dtw,blocked)
            a = env if t == 0 else (dist.sample() if training else dist.probs.argmax(-1))
            lp = dist.log_prob(a)
            p = probs[a,slot]
            new, new_mask, pred, confident, belief = next_state(
                p,state,mask,args.mode,args.decay,t==0,args.threshold)
            cont = active & ~confident
            correct = pred == true
            if t == 0:
                first_pred = pred.clone()
            eliminated = ~new_mask.gather(1,true[:,None]).squeeze(1)
            wrong_eliminated |= active & eliminated
            # Original reward coefficients and functional form, on active
            # observations only. Labels are not used by state/stop/action.
            reward = (torch.einsum("bi,bij->bj",new,dtw[a]).mean(-1)
                      - 4.0*cont.float()*t + 13.0*correct.float()
                      + 20.0*confident.float())
            if t == 0:
                reward -= 30.0*(~correct).float()
            if t == args.max_steps-1:
                reward -= 15.0*cont.float()
            buf.append((state,blocked,a,lp,val,active.clone(),reward*active,t>0))
            action_history[active,t] = a[active]+1
            steps += active.long()
            final = torch.where(active,pred,final)
            visited[torch.arange(b,device=device),a] |= active
            state = torch.where(active[:,None],new,state)
            mask = torch.where(active[:,None],new_mask,mask)
            active = cont
            if not active.any():
                break
    rows = torch.stack((true,final,steps,env+1,slot,first_pred,
                        wrong_eliminated.long()),dim=1).cpu().numpy()
    return buf, rows, action_history.cpu().numpy()


def update(policy, dtw, buf, actor_opt, critic_opt, args):
    ret = torch.zeros_like(buf[0][6])
    returns = []
    for item in reversed(buf):
        ret = (item[6]+0.6*ret)*item[5]
        returns.insert(0,ret)
    state,blocked,a,lp,val,active,_,learn = zip(*buf)
    active = torch.cat(active)
    old_lp = torch.cat(lp)[active]
    val = torch.cat(val)[active]
    returns = torch.cat(returns)[active]
    learn = torch.cat([torch.full_like(item[5],item[7]) for item in buf])[active]
    advantage = returns-val
    if learn.sum() > 1:
        adv = advantage[learn]
        advantage = advantage.clone()
        advantage[learn] = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    # The initially forced profile is not an action sampled by PPO.
    def replay():
        # Preserve the rollout batch shape for each timestep: convolution and
        # matrix kernels may otherwise change floating-point reduction order.
        logs, values, entropies = [], [], []
        for item in buf:
            d, v = distribution(policy,item[0],dtw,item[1])
            logs.append(d.log_prob(item[2]))
            values.append(v)
            entropies.append(d.entropy())
        return tuple(torch.cat(x)[active] for x in (logs,values,entropies))

    with torch.no_grad():
        lp_new,v,_ = replay()
        lp_err = (lp_new-old_lp).abs().max().item()
        v_err = (v-val).abs().max().item()
        assert torch.allclose(lp_new,old_lp,atol=2e-5,rtol=2e-5), f"LogProb replay mismatch: max_abs={lp_err:.8g}"
        assert torch.allclose(v,val,atol=2e-5,rtol=2e-5), f"Value replay mismatch: max_abs={v_err:.8g}"
    loss_a = 0.0
    grad_a = 0.0
    for _ in range(args.k_epochs):
        new_lp,value,entropy = replay()
        loss_c = F.l1_loss(value,returns)
        critic_opt.zero_grad(set_to_none=True)
        loss_c.backward()
        grad_c = torch.nn.utils.clip_grad_norm_(
            list(policy.critic.parameters())+list(policy.cnn_layer_critic.parameters()),1.0)
        assert torch.isfinite(grad_c)
        critic_opt.step()
        if learn.any():
            ratio = (new_lp[learn]-old_lp[learn]).exp()
            adv = advantage[learn]
            loss = -torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
            loss -= .01*entropy[learn].mean()
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_a = torch.nn.utils.clip_grad_norm_(
                list(policy.actor.parameters())+list(policy.cnn_layer_actor.parameters()),1.0)
            assert torch.isfinite(grad_a)
            actor_opt.step()
            loss_a = float(loss.detach())
    return loss_a,float(loss_c.detach()),float(grad_a),float(grad_c)


def metrics(rows):
    y,p,steps = rows[:,0],rows[:,1],rows[:,2]
    multi = steps>1
    votes = [np.mean(p[y==c]==c)>=2/3 for c in range(K) if (y==c).any()]
    return dict(n=int(len(rows)),accuracy=float(np.mean(y==p)),
                macro_f1=float(f1_score(y,p,labels=list(range(K)),average="macro",zero_division=0)),
                mean_steps=float(steps.mean()),multi_n=int(multi.sum()),
                multi_accuracy=float(np.mean(y[multi]==p[multi])) if multi.any() else None,
                wrong_elimination_rate=float(rows[:,6].mean()),
                first_accuracy=float(np.mean(y==rows[:,5])),
                voting_accuracy=float(np.mean(votes)))


def save_rows(path,rows,actions):
    with path.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["label","prediction","steps","initial_env","sample_slot",
                    "first_prediction","true_class_eliminated","actions"])
        for row,act in zip(rows,actions):
            w.writerow([*row.tolist()," ".join(str(a) for a in act if a>0)])


def self_test(device):
    p = torch.tensor([[.6,.39,.01]],device=device)
    q = torch.tensor([[.2,.7,.1]],device=device)
    ones = torch.ones_like(p,dtype=torch.bool)
    assert torch.allclose(update_evidence(p,p,ones,"mef",.5,True),p)
    f = update_evidence(q,p,ones,"mef",.5,False)
    assert torch.allclose(f,normalize(p.sqrt()*q),atol=1e-6)
    assert (f>0).all()
    hard = torch.tensor([[True, True, False]], device=device)
    assert torch.allclose(update_evidence(q,p,hard,"no_mask",.5,False),q)
    assert torch.allclose(update_evidence(q,p,hard,"baseline",.5,False),normalize(q*hard))
    fm = update_evidence(q,p,hard,"fusion_mask",.5,False)
    assert torch.allclose(fm,normalize(f*hard),atol=1e-6)
    assert fm[0,2] == 0
    # Check pruning behavior on the actual twelve-class state size.
    p12 = torch.tensor([[.6,.39]+[.001]*10],device=device)
    mask12 = torch.ones_like(p12,dtype=torch.bool)
    for mode in ("baseline","no_mask","fusion_mask","mef"):
        s,m,_,_,_ = next_state(p12,p12,mask12,mode,.5,True)
        assert torch.allclose(s.sum(-1),torch.ones(1,device=device))
        assert bool(m[0,2]) == (mode in ("no_mask","mef"))
    print("[MEF] Evidence fusion unit checks PASS",flush=True)


def main(args):
    assert torch.cuda.is_available()
    assert args.max_steps <= 32 and 0 <= args.decay <= 1 and 0 < args.threshold < 1
    device = torch.device("cuda:0")
    seed_all(args.seed)
    self_test(device)
    args.out.mkdir(parents=True,exist_ok=True)
    training = args.phase == "train"
    path = args.train_data if training else args.test_data
    split = "train" if training else "test"
    dtw,envs = load_dtw(args.shared,path,split,device)
    meta = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    meta["classifier_sha256"] = sha256(args.classifier)
    meta["dtw_sha256"] = sha256(args.shared/(split+"_dtw.txt"))
    (args.out/"config.json").write_text(json.dumps(meta,indent=2)+"\n")
    policy = make_policy(device)
    if training:
        probs,labels = load_observations(args,dtw,envs,device,True,0)
        # Reset RNG after loading frozen classifier, for matched policy sampling.
        seed_all(args.seed)
        actor_opt = torch.optim.Adagrad(
            list(policy.actor.parameters())+list(policy.cnn_layer_actor.parameters()),lr=.05,eps=1e-5)
        critic_opt = torch.optim.Adagrad(
            list(policy.critic.parameters())+list(policy.cnn_layer_critic.parameters()),lr=.02,eps=1e-5)
        history=[]
        for ep in range(1,args.epochs+1):
            start=time.perf_counter()
            for opt,lr in [(actor_opt,.05),(critic_opt,.02)]:
                for group in opt.param_groups:
                    group["lr"]=lr*(1-(ep-1)/args.epochs)
            rows=[]; actions=[]; losses=[]
            ids=torch.randperm(probs.shape[0]*probs.shape[1],device=device)
            for bi,batch in enumerate(ids.split(args.batch_size)):
                if args.max_batches and bi>=args.max_batches:
                    break
                buf,r,a=rollout(policy,dtw,probs,labels,batch,args,True)
                losses.append(update(policy,dtw,buf,actor_opt,critic_opt,args))
                rows.append(r); actions.append(a)
            rows=np.concatenate(rows); actions=np.concatenate(actions)
            s=metrics(rows)
            s.update(epoch=ep,elapsed_s=time.perf_counter()-start,
                     actor_loss=float(np.mean([l[0] for l in losses])),
                     critic_loss=float(np.mean([l[1] for l in losses])),
                     actor_grad_max=float(max(l[2] for l in losses)),
                     critic_grad_max=float(max(l[3] for l in losses)))
            history.append(s)
            save_rows(args.out/f"epoch-{ep}.csv",rows,actions)
            (args.out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
            torch.save({"policy":policy.state_dict(),"mode":args.mode,"config":meta},
                       args.out/f"epoch-{ep}.pt")
            print("[MEF] EPOCH",json.dumps(s),flush=True)
    else:
        ckpt=torch.load(args.checkpoint,map_location=device)
        assert ckpt["mode"]==args.mode
        assert ckpt["config"]["decay"]==args.decay and ckpt["config"]["threshold"]==args.threshold
        assert ckpt["config"]["max_steps"]==args.max_steps
        assert ckpt["config"]["classifier_sha256"]==meta["classifier_sha256"]
        assert ckpt["config"]["dtw_sha256"]==sha256(args.shared/"train_dtw.txt")
        policy.load_state_dict(ckpt["policy"])
        results=[]; all_rows=[]
        for idx in args.indices:
            probs,labels=load_observations(args,dtw,envs,device,False,idx)
            ids=torch.arange(probs.shape[0]*probs.shape[1],device=device)
            rows=[]; actions=[]
            for batch in ids.split(args.batch_size):
                _,r,a=rollout(policy,dtw,probs,labels,batch,args,False)
                rows.append(r); actions.append(a)
            rows=np.concatenate(rows); actions=np.concatenate(actions)
            s=metrics(rows); s["index"]=idx
            results.append(s); all_rows.append(rows)
            save_rows(args.out/f"test-{idx}.csv",rows,actions)
            print("[MEF] TEST",json.dumps(s),flush=True)
        aggregate=metrics(np.concatenate(all_rows))
        aggregate["e1_voting_accuracy"]=float(np.mean([r["voting_accuracy"] for r in results]))
        (args.out/"test_metrics.json").write_text(json.dumps({"per_index":results,"aggregate":aggregate},indent=2)+"\n")
        print("[MEF] FINAL",json.dumps(aggregate),flush=True)
    print("[MEF] ALL PASS",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["train","test"],default="train")
    p.add_argument("--mode",choices=["baseline","no_mask","fusion_mask","mef"],required=True)
    for key in ["train-data","test-data","shared","classifier","out"]:
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--seed",type=int,default=3402)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=36)
    p.add_argument("--k-epochs",type=int,default=10)
    p.add_argument("--max-steps",type=int,default=7)
    p.add_argument("--max-batches",type=int,default=0)
    p.add_argument("--decay",type=float,default=.5)
    p.add_argument("--threshold",type=float,default=.7)
    p.add_argument("--indices",type=int,nargs="+",default=list(range(9)))
    args=p.parse_args()
    if args.phase=="test" and args.checkpoint is None:
        p.error("--checkpoint required for test")
    main(args)
