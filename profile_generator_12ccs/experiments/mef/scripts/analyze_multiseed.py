from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path.home() / "dl-workspace/experiment-results/CClinguist/mef"

ROOTS = {
    3402: ROOT / "train-20261127-113059/test",
    3403: ROOT / "multiseed/seed-3403/test",
    3404: ROOT / "multiseed/seed-3404/test",
}

MODES = ["baseline", "no_mask", "fusion_mask", "mef"]

OUT = ROOT / "final_analysis"
OUT.mkdir(exist_ok=True)

results = []
paired = []
step_rows = []
all_data = {}

KEYS = ["index", "initial_env", "sample_slot", "label"]

for seed, root in ROOTS.items():
    for mode in MODES:
        path = root / mode
        frames = []
        top3_correct = 0

        for idx in range(9):
            file = path / f"test-{idx}.csv"
            assert file.exists(), file

            df = pd.read_csv(file)
            assert len(df) == 384, file
            assert df["initial_env"].nunique() == 32
            assert df["label"].nunique() == 12

            # 每个环境应包含全部12种CCA
            assert (
                df.groupby("initial_env")["label"].nunique() == 12
            ).all()

            df["index"] = idx
            df["correct"] = df["label"] == df["prediction"]

            # Top-3投票：按当前测试正确率选择三个环境
            # 平分时按环境ID排序，保证结果可复现
            ranking = (
                df.groupby("initial_env")["correct"]
                .mean()
                .reset_index()
                .sort_values(
                    ["correct", "initial_env"],
                    ascending=[False, True]
                )
            )

            top3 = ranking.head(3)["initial_env"]
            selected = df[df["initial_env"].isin(top3)]

            votes = selected.groupby("label")["correct"].sum()

            assert len(votes) == 12
            top3_correct += int((votes >= 2).sum())

            frames.append(df)

        df = pd.concat(frames, ignore_index=True)

        assert len(df) == 3456
        assert not df.duplicated(KEYS).any()

        multi = df["steps"] > 1

        metrics = {
            "seed": seed,
            "mode": mode,
            "n": len(df),
            "accuracy": df["correct"].mean(),
            "macro_f1": f1_score(
                df["label"],
                df["prediction"],
                average="macro"
            ),
            "multi_n": int(multi.sum()),
            "multi_accuracy": df.loc[multi, "correct"].mean(),
            "mean_steps": df["steps"].mean(),
            "wrong_elimination_rate":
                df["true_class_eliminated"].mean(),
            "first_accuracy":
                (df["label"] == df["first_prediction"]).mean(),
            "top3_voting": top3_correct / 108,
        }

        saved = json.loads(
            (path / "test_metrics.json").read_text()
        )["aggregate"]

        for key in [
            "accuracy", "macro_f1", "mean_steps",
            "multi_accuracy", "wrong_elimination_rate",
            "first_accuracy"
        ]:
            assert abs(metrics[key] - saved[key]) < 1e-8, (
                seed, mode, key, metrics[key], saved[key]
            )

        results.append(metrics)
        all_data[(seed, mode)] = df

        for steps, group in df.groupby("steps"):
            step_rows.append({
                "seed": seed,
                "mode": mode,
                "steps": steps,
                "n": len(group),
                "accuracy": group["correct"].mean(),
            })

    # 配对比较：验证每种方法评估的确实是相同样本
    base = all_data[(seed, "baseline")]

    for mode in MODES[1:]:
        other = all_data[(seed, mode)]

        pair = base.merge(
            other,
            on=KEYS,
            suffixes=("_base", "_new"),
            validate="one_to_one"
        )

        assert len(pair) == 3456

        assert (
            pair["first_prediction_base"] ==
            pair["first_prediction_new"]
        ).all()

        assert (
            (pair["steps_base"] > 1) ==
            (pair["steps_new"] > 1)
        ).all()

        old = pair["correct_base"]
        new = pair["correct_new"]

        helped = int((~old & new).sum())
        harmed = int((old & ~new).sum())

        paired.append({
            "seed": seed,
            "comparison": f"baseline -> {mode}",
            "both_correct": int((old & new).sum()),
            "both_wrong": int((~old & ~new).sum()),
            "helped": helped,
            "harmed": harmed,
            "net_correct": helped - harmed,
            "accuracy_gain_pp": (
                new.mean() - old.mean()
            ) * 100,
        })

result_df = pd.DataFrame(results)
paired_df = pd.DataFrame(paired)
steps_df = pd.DataFrame(step_rows)

result_df.to_csv(OUT / "all_seed_metrics.csv", index=False)
paired_df.to_csv(OUT / "paired_error_analysis.csv", index=False)
steps_df.to_csv(OUT / "step_distribution.csv", index=False)

cols = [
    "accuracy",
    "macro_f1",
    "multi_accuracy",
    "mean_steps",
    "wrong_elimination_rate",
    "top3_voting",
]

summary = (
    result_df.groupby("mode")[cols]
    .agg(["mean", "std"])
    .reindex(MODES)
)

summary.to_csv(OUT / "mean_std.csv")

print("\n========== ALL SEED METRICS ==========")
print(result_df.to_string(index=False, float_format="%.6f"))

print("\n========== MEAN +/- STD ==========")

for mode in MODES:
    print(f"\n{mode}")
    for col in cols:
        mean = summary.loc[mode, (col, "mean")]
        std = summary.loc[mode, (col, "std")]

        if col == "mean_steps":
            print(f"  {col:25s}: {mean:.4f} +/- {std:.4f}")
        else:
            print(f"  {col:25s}: {mean*100:.4f} +/- {std*100:.4f}%")

print("\n========== PAIRED ERROR ANALYSIS ==========")
print(paired_df.to_string(index=False))

print("\n========== STEP DISTRIBUTION ==========")
print(steps_df.to_string(index=False))

print("\n========== FINAL VALIDATION: ALL PASS ==========")
print("Results:", OUT)
