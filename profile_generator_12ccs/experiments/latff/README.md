# CClinguist 12 CCA 分类器改进实验

## 创新点

原分类器将 Transformer 输出的固定末位置、普通均值和普通最大值拼接。变长序列补零后，固定末位置可能落在 Padding 上，普通均值也可能受补齐位置影响。本项目提出长度感知的自适应时序特征融合（LATFF）：

| 版本 | 改动 |
|---|---|
| A | 原始池化：固定末位置 + Mean + Max |
| B | 用有效长度计算 Masked Last、Masked Mean、Masked Max |
| C | 在 B 上增加长度门控：`length/512 → Linear(1,3) → Softmax`，三路加权后拼接 |

三个版本保留相同 Transformer、环境特征和分类头。C 的 gate 零初始化，初始三路权重均为 1。实验要验证两个问题：排除 Padding 是否改善分类（B 对 A）；按长度调整池化权重是否进一步改善分类（C 对 B）。

## 新增文件

| 文件 | 作用 |
|---|---|
| `models.py` | A/B/C 分类器、长度校验、masked pooling 和门控融合 |
| `dataset.py` | 读取真实 CSV，沿用原标签及预处理，按环境分训练/验证并构造独立测试集 |
| `config.py` | 数据路径、训练参数、快速模式等统一配置 |
| `train.py` | 训练一个模型，按验证集 Macro-F1 选 checkpoint，再评估独立测试集 |
| `evaluate.py` | 对已保存 checkpoint 重新评估 |
| `checkpoint.py` | 加载原始分类器权重并检查缺失键、意外键和形状 |
| `utils.py` | 随机种子、指标计算和结果保存 |
| `test_models.py` | 用人工张量检查模型逻辑与 checkpoint 兼容性；不是实验数据 |

原始模型、数据和 checkpoint 均不修改。训练初始化权重为 `profile_generator_12ccs/models/best_model_accuracy.pth.tar`。

## 如何运行

在 Linux 虚拟机准备包含 PyTorch、NumPy、Pandas、scikit-learn、SciPy 的环境。确认真实数据位于：

```text
profile_generator_12ccs/data_12ccs/simulation_hk2jp_1201to1203/  # 训练来源
profile_generator_12ccs/data_12ccs/simulation_pk2hk_12_20/       # 独立测试来源
```

进入本目录后先检查模型与数据：

```bash
cd profile_generator_12ccs/experiments/latff
python test_models.py
python dataset.py ../../data_12ccs/simulation_hk2jp_1201to1203
python dataset.py ../../data_12ccs/simulation_pk2hk_12_20
```

快速模式使用少量真实 CSV、每类少量样本、训练 1 个 epoch，并完成验证、测试、保存和重载 checkpoint：

```bash
python train.py --model A --quick
python train.py --model B --quick
python train.py --model C --quick
```

默认使用 CUDA；需要时可显式指定 `--device cpu`。快速模式只验证程序能否端到端运行，样本少，不能作为正式结论。

**正式对照使用完全相同的数据划分和参数。每轮用验证集选模型，最终测试集只评估一次：**

```bash
cd profile_generator_12ccs/experiments/latff
python train.py --model A --device cuda
python train.py --model B --device cuda
python train.py --model C --device cuda
```
默认参数与原项目对应：seed 3402、Adam、学习率 1e-4、batch size 32、1 epoch。每条命令训练完成后会自动用验证集选 checkpoint、重载，并评估独立测试集；结果保存在各自的 runs/ 子目录。



## 评价指标与输出

测试集输出：

- Accuracy、Macro-F1、测试损失；
- 每类 Precision、Recall、F1、Support；
- 12×12 混淆矩阵。

每个运行目录包含 `config.json`（配置和种子）、`split.json` 与 `*_manifest.json`（数据划分及样本路径）、`epoch_metrics.csv`（每轮训练/验证指标）、`initial_checkpoint_load.json`（权重加载检查）、`best_model.pt`、`test_metrics.json`、`test_confusion_matrix.csv` 和 `summary.json`。

三随机种子结果应报告均值和标准差。短/中/长序列分段分析可根据测试清单里的有效长度进行；当前脚本尚未自动输出分段指标。

## 怎样判断 Idea 是否有效

理想结果是 **B 的 Macro-F1/Accuracy 高于 A，C 又高于 B**，且多随机种子趋势一致；混淆矩阵中易混类别错误减少，低召回类别有所改善。如果 B 优于 A，说明屏蔽 Padding 有帮助；如果 C 优于 B，说明长度门控提供额外收益。如果差距很小或跨种子不稳定，则当前数据和设置下改进证据不足；若 C 变差，需检查 gate 权重是否偏向单一路径或出现过拟合。

当前快速验证三者都得到 Accuracy 0.875、Macro-F1 0.8667，但每类只有 2 个测试样本，只能说明流程跑通，不能说明改进有效或无效。不要把 RL 搜索日志里的预测率当成独立分类器指标。

## 对照边界

A 保留原始分类器前向和池化，B 只改池化，C 只在 B 上增加 gate。独立训练脚本对三者统一使用交叉熵微调并从同一原始权重初始化；这使 A/B/C 内部可比较，但训练损失与原通用训练代码中的交叉熵加类内/类间距离损失不同，报告中应注明这一点。
