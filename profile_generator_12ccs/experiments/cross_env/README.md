# CClinguist 跨环境与多尺度分类实验

## 改进点

现有 A/B/C 的正式实验中，B 的测试 Accuracy 为 53.99%，C 为 54.08%。本组实验以 **B 的 Masked Pooling** 为共同起点，检验两种进一步改进能否提高跨网络路径测试表现。

| 实验 | 模型与训练变化 | 分类头输入 |
|---|---|---|
| B | 原 Masked Pooling 对照。使用交叉熵训练 | 176 维 |
| D | B + 跨环境对比损失。同一 CCA、不同 RTT/带宽环境为正样本；不同 CCA 为负样本，同环境负样本额外增加 logit 0.1 | 176 维 |
| E | B + 多尺度时序汇总。提取前后半段和首末四分段的均值差，通过零初始化线性层修正原 Masked Mean | 176 维 |
| F | 同时采用 D 的训练损失和 E 的多尺度汇总 | 176 维 |

对比损失仅使用**训练 batch** 中的样本。缺少跨环境同类正样本的 anchor 不参与该损失。`epoch_metrics.csv` 记录每轮参与对比损失的 anchor 数量。E/F 的新投影层零初始化，初始输出与 B 相同。所有模型均加载原始分类器 checkpoint 中可复用的权重。

## 文件作用

| 文件 | 作用 |
|---|---|
| `experiment_models.py` | B/D/E/F 模型映射和多尺度汇总模块 |
| `run.py` | 训练、验证、选择最佳权重并测试 |
| `evaluate.py` | 从保存的权重重新评估独立测试集 |
| `README.md` | 实验说明与命令 |

脚本复用 `../latff/` 中的 CSV 预处理、12 类标签、环境划分、指标计算和原 checkpoint 加载器。需要原始数据目录 `profile_generator_12ccs/data_12ccs/` 和原始权重 `profile_generator_12ccs/models/best_model_accuracy.pth.tar`。

## 如何运行

在 Linux 项目根目录执行。PyTorch、NumPy、Pandas、scikit-learn 和 SciPy 应已安装。默认设备是 CUDA。以下命令固定种子、轮数及优化器配置，以便公平比较。

```bash
cd profile_generator_12ccs/experiments/cross_env

python run.py --model B --seed 42 --epochs 20 --batch-size 32 --learning-rate 1e-4 --device cuda
python run.py --model D --seed 42 --epochs 20 --batch-size 32 --learning-rate 1e-4 --device cuda
python run.py --model E --seed 42 --epochs 20 --batch-size 32 --learning-rate 1e-4 --device cuda
python run.py --model F --seed 42 --epochs 20 --batch-size 32 --learning-rate 1e-4 --device cuda
```

先用少量**真实数据**检查流程时，可对每个模型运行 `python run.py --model D --quick --device cuda`，并把 `D` 替换为 `B`、`E`、`F`。快速模式从两个训练环境中为每类各选一条样本，确保 D/F 有跨环境正样本；四个模型使用相同的快速训练样本。它只运行 1 个 epoch，其指标不用于正式结论。

每条正式命令按验证集 Macro-F1 选择 checkpoint，再重新加载它评估独立测试集。可使用 `python evaluate.py --checkpoint runs/<某次运行>/best_model.pt --device cuda` 重复评估。

## 输出与判断

每次运行在 `runs/full_<模型>_seed<种子>_<时间>/` 保存 `config.json`、`split.json`、训练/验证/测试样本清单、`initial_checkpoint_load.json`、`epoch_metrics.csv`、`best_model.pt`、`test_metrics.json`、`test_confusion_matrix.csv` 和 `summary.json`。

测试指标包括 Accuracy、Macro-F1、12 类各自的 Precision/Recall/F1/Support，以及 12×12 混淆矩阵。对比 D 与 B 可衡量跨环境对比损失的贡献，对比 E 与 B 可衡量多尺度汇总的贡献，对比 F 与 D/E 可检验两者是否互补。理想情况是测试集 Macro-F1 和 Accuracy 均提高，低 Recall 类别改善，并且不同随机种子下趋势一致。若 `contrast_active_anchors` 很少，应先检查 batch 内跨环境同类配对是否充足。

建议再用种子 `123`、`2026` 重复四组正式命令，报告均值和标准差。测试集只用于固定方案后的最终评价，模型选择依据始终是验证集 Macro-F1。
