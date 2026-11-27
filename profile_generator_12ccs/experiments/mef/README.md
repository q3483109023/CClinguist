# MEF: Multi-step Evidence Fusion for Reliable CCA Identification

## 1. Research Motivation

原始 CClinguist 通过主动网络探测、CCA 分类器和 PPO 策略完成拥塞控制算法识别。

本实验针对多步识别决策过程中的两个问题：

1. 历史探测概率证据利用不足。
2. 不可逆的候选类别硬掩码可能错误排除真实类别。

## 2. Proposed Method

MEF 包含两个核心机制：

### Multi-step Evidence Fusion

融合上一轮累积分类概率 q 和当前分类概率 p：

q_t = softmax(0.5 * log(q_{t-1}) + log(p_t))

首次探测直接使用分类器输出概率。

### Removal of Hard Category Pruning

取消不可逆的类别硬掩码，保留全部候选 CCA，
允许后续探测证据修正早期错误判断。

本方法不修改冻结的原始 CCA 分类器，
主要改进 PPO 多步识别与决策阶段。

## 3. Experimental Setup

- CCA classes: 12
- Training environments: 43
- Testing environments: 32
- Training samples: 3612
- Test samples per run: 3456
- Multi-step samples per run: 646
- Random seeds: 3402, 3403, 3404
- PPO training epochs: 5
- Evidence discount coefficient: 0.5
- Stopping confidence threshold: 0.7
- Classifier: frozen original pretrained classifier

四种方法共享分类器、数据集、修正后的 DTW 矩阵、
奖励参数和共同实验协议，并分别训练 PPO 策略。

共同实验协议包含修正后的 DTW 类别对应关系、
无标签停止判断及统一的 PPO 实现修正。
因此 Baseline 不等同于未经修改的官方实现。

## 4. Ablation Study

| Method | Evidence Fusion | Hard Pruning |
|---|---|---|
| Baseline | No | Yes |
| No Mask | No | No |
| Fusion + Mask | Yes | Yes |
| MEF | Yes | No |

## 5. Main Results

Mean +/- sample standard deviation over three random seeds.

| Metric | Baseline | No Mask | Fusion + Mask | MEF |
|---|---:|---:|---:|---:|
| Accuracy (%) | 60.00 +/- 4.45 | 62.37 +/- 0.49 | 62.02 +/- 0.99 | 63.38 +/- 0.42 |
| Macro-F1 (%) | 59.53 +/- 5.05 | 61.91 +/- 0.54 | 61.86 +/- 1.06 | 63.08 +/- 0.48 |
| Multi-step Accuracy (%) | 60.32 +/- 23.81 | 72.96 +/- 2.60 | 71.10 +/- 5.29 | 78.38 +/- 2.26 |
| Mean probing steps | 1.3198 | 1.2126 | 1.2089 | 1.2044 |
| Wrong elimination rate (%) | 3.55 | 0.00 | 3.13 | 0.00 |
| Top-3 voting (%) | 97.22 | 95.37 | 96.91 | 95.68 |

MEF achieves:

- Overall Accuracy: +3.38 percentage points.
- Macro-F1: +3.55 percentage points.
- Multi-step Accuracy: +18.06 percentage points.
- Mean probing steps: -8.75%.
- Wrong hard elimination rate: reduced to zero by design.
- Top-3 voting Accuracy: -1.54 percentage points.

## 6. Per-seed Results

| Seed | Baseline Accuracy | MEF Accuracy | Gain (pp) |
|---|---:|---:|---:|
| 3402 | 63.0208% | 63.8600% | +0.8391 |
| 3403 | 54.8900% | 63.0787% | +8.1887 |
| 3404 | 62.0949% | 63.1944% | +1.0995 |

MEF 在三个随机种子下均取得整体识别准确率提升。

## 7. Paired Error Analysis

| Seed | Corrected errors | Introduced errors | Net improvement |
|---|---:|---:|---:|
| 3402 | 89 | 60 | 29 |
| 3403 | 313 | 30 | 283 |
| 3404 | 117 | 79 | 38 |
| Total | 519 | 169 | 350 |

合计为同一测试集在三个随机种子下的重复评估结果，
并非独立测试样本的数量。

## 8. Findings and Limitations

1. MEF 在三个随机种子下均优于其他消融变体的整体 Accuracy。
2. 收益集中于进入多步探测的样本，首次分类结果保持一致。
3. MEF 改善了多步识别准确率和平均探测效率。
4. 平均性能增益受到 Seed 3403 基线性能退化的较大影响。
5. 仅使用三个随机种子，尚不足以证明统计显著性。
6. 不同变体分别训练 PPO，尚不能将全部增益独立归因于融合公式。
7. Top-3 投票指标使用测试标签筛选初始环境，属于诊断指标，
   不代表无标签部署性能。
8. 尚未验证相对于未经修改的官方 CClinguist 完整 E1 系统的性能提升。
9. 尚未与 LATFF 等分类器改进进行联合实验。

## 9. Source Code

- ../../mef_experiment.py
- ../../prepare_mef_dtw.py

## 10. Experiment Archive

- scripts/: training and analysis scripts.
- results/shared/: shared DTW and experiment resources.
- results/seed3402/: primary experiment and ablations.
- results/multiseed/: additional seed experiments.
- results/final_analysis/: aggregated metrics and error analysis.
- checkpoint_manifest.csv: checkpoint size and SHA256 records.

模型权重保存在实验服务器，不直接提交至 GitHub。

归档脚本保留原始服务器实验路径，
在其他机器复现时需要根据实际目录调整路径。

## 11. Conclusion

MEF 在统一受控实验条件下改善了 CClinguist 的多步识别可靠性、
整体识别准确率及平均探测效率，并表现出较低的跨种子性能波动。

现有证据支持其作为多步识别决策机制的改进，
但不支持宣称其全面超越原始 CClinguist 系统。
