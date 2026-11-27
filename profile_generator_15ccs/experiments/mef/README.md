# Multi-round Evidence Fusion (MEF): 15CCs Extension

## 1. Research Objective

This experiment investigates two mechanisms for improving
CClinguist's congestion control algorithm identification:

- Removing irreversible hard candidate elimination.
- Aggregating historical classification probabilities
  through discounted multi-round evidence fusion.

The 15CCs experiment extends the controlled MEF study
previously conducted on the 12CCs dataset.

## 2. Ablation Design

| Method | Hard Mask | Evidence Fusion |
|---|---|---|
| Baseline | Yes | No |
| No Mask | No | No |
| Fusion + Mask | Yes | Yes |
| MEF | No | Yes |

Each variant independently trains its PPO policy while
sharing the same frozen classifier and DTW matrices.

The baseline is the controlled MEF experimental baseline,
not an exact reproduction of the original paper's
complete evaluation protocol.

## 3. Experimental Configuration

- Training dataset: simulation_pk2hk_15CCs
- Testing dataset: simulation_15CCs
- Number of CCA classes: 15
- Number of network environments: 16
- Training samples per environment and class: 7
- Training samples per epoch: 1680
- Test samples per run: 2400
- Random seeds: 3402, 3403, 3404
- Training epochs: 5
- Batch size: 36
- PPO update epochs: 10
- Maximum probing steps: 7
- Confidence threshold: 0.7
- Evidence discount factor: 0.5
- Frozen classifier: models/checkpoint_9000.pth.tar
- Reward coefficients: inherited from the controlled
  12CCs MEF protocol rather than the official 15CCs defaults.

All variants within each seed were trained on the same GPU.

DTW matrices were computed using training traces only,
with explicit class-ID alignment and seven
duration-rank-matched traces per class.

## 4. Computational Optimization

Frozen-classifier observations were cached and shared
across all experimental variants and random seeds.

The cache contains one training probability tensor
and ten testing probability tensors.

Independent recomputation verified exact probability
and label equality for the training observations,
test index 0 on GPU 0, and test index 9 on GPU 1.

All eleven cached files passed SHA256 integrity checks.

No experimental samples, training epochs, random seeds
or PPO update epochs were removed for acceleration.

## 5. Main Results

Mean +/- sample standard deviation over three seeds.

| Method | Accuracy (%) | Macro-F1 (%) | Mean Steps |
|---|---:|---:|---:|
| Baseline | 74.79 +/- 2.20 | 73.82 +/- 2.66 | 1.7550 |
| No Mask | 77.32 +/- 0.80 | 76.64 +/- 0.79 | 1.7551 |
| Fusion + Mask | 75.50 +/- 4.27 | 74.75 +/- 4.43 | 1.7249 |
| MEF | 75.86 +/- 2.95 | 74.61 +/- 3.56 | 1.7244 |

| Method | Multi-round Accuracy (%) | Wrong Elimination Rate (%) |
|---|---:|---:|
| Baseline | 66.87 | 5.57 |
| No Mask | 72.25 | 0.00 |
| Fusion + Mask | 68.38 | 4.79 |
| MEF | 69.15 | 0.00 |

Compared with the controlled baseline, MEF improves
accuracy by 1.07 percentage points and reduces the
mean probing cost by approximately 1.74%.

However, No Mask achieves the highest mean accuracy
and Macro-F1 on the 15CCs dataset.

## 6. Paired Error Analysis

Aggregated prediction transitions across three seeds:

| Reference | Method | Helped | Harmed | Net |
|---|---|---:|---:|---:|
| Baseline | No Mask | 490 | 308 | +182 |
| Baseline | Fusion + Mask | 309 | 258 | +51 |
| Baseline | MEF | 452 | 375 | +77 |
| No Mask | MEF | 345 | 450 | -105 |
| Fusion + Mask | MEF | 382 | 356 | +26 |

The same test dataset is evaluated under three random
seeds; these counts are not independent test samples.

Compared with No Mask, MEF produces its largest
per-class accuracy losses on:

- pccLoss: -26.04 percentage points.
- scalable: -18.33 percentage points.

## 7. Cross-Dataset Findings

| MEF vs. Baseline | 12CCs | 15CCs |
|---|---:|---:|
| Accuracy gain (pp) | +3.38 | +1.07 |
| Macro-F1 gain (pp) | +3.55 | +0.79 |
| Multi-round accuracy gain (pp) | +18.06 | +2.27 |
| Mean probing cost reduction (%) | 8.75 | 1.74 |
| Seeds with positive accuracy gains | 3/3 | 2/3 |

Removing hard elimination improves mean accuracy
on both datasets.

Discounted evidence fusion provides additional
benefits on 12CCs but does not outperform No Mask
on 15CCs in terms of mean accuracy.

## 8. Limitations

- Only three random seeds were evaluated.
- Statistical significance has not been established.
- Dataset, environment and classifier differences
  prevent attributing cross-dataset performance
  changes exclusively to the number of classes.
- Independently trained PPO policies prevent
  attributing all differences directly to fusion.
- Evidence reliability is not explicitly modeled.
- Lower probing cost does not necessarily imply
  that only unnecessary probes have been eliminated.
- E1-style voting diagnostics use test-label-based
  environment selection and are not deployment metrics.

## 9. Archive Contents

The archive preserves experimental scripts,
training histories, prediction CSV files,
evaluation metrics, cache verification records,
paired error analyses and per-class results.

Model checkpoints remain on the experiment server.
Their paths and SHA256 hashes are recorded in
checkpoint_manifest.csv.

Raw datasets, model weights and probability cache
tensors are intentionally excluded from Git.

The archived launch and verification scripts retain
the original server paths and require adaptation
when executed in a different environment.

All 12 formal runs completed successfully:
60 training epochs and 28,800 test predictions.
