# DTW v2 Optimization Experiment

## Method

Compare full DTW (Original) against constrained-window
DTW (CBand) using the same underlying DTW library.

The window radius is approximately 10% of the longer
sequence, enlarged when necessary for length differences.

Both methods construct class-aligned distance matrices
using representative training traces. Test matrices
are mapped from training matrices by environment name.

## Experimental Setup

- Datasets: 12CCs and 15CCs
- Methods: Original and CBand
- Seeds: 3402, 3403, 3404
- PPO epochs: 5
- Batch size: 36
- PPO update epochs: 10
- Maximum probing steps: 7
- Confidence threshold: 0.7
- Shared frozen classifier observations
- Independently trained PPO policies
- Controlled baseline without MEF evidence fusion
- Test traces excluded from DTW matrix construction

## DTW Computational Efficiency

Median wall time over three timing repetitions.

| Dataset | Original (s) | CBand (s) | Speedup |
|---|---:|---:|---:|
| 12CCs | 16.479 | 11.291 | 1.46x |
| 15CCs | 4.907 | 3.677 | 1.33x |

## Recognition Performance

Mean +/- sample standard deviation over three seeds.

| Dataset | Metric | Original | CBand |
|---|---|---:|---:|
| 12CCs | Accuracy (%) | 61.15 +/- 1.47 | 62.34 +/- 0.06 |
| 12CCs | Macro-F1 (%) | 60.91 +/- 1.57 | 62.13 +/- 0.07 |
| 12CCs | Multi-step Accuracy (%) | 66.46 +/- 7.87 | 72.81 +/- 0.32 |
| 12CCs | Mean Steps | 1.2105 | 1.2117 |
| 12CCs | Wrong Elimination (%) | 3.22 | 3.34 |
| 15CCs | Accuracy (%) | 76.97 +/- 5.25 | 76.47 +/- 5.99 |
| 15CCs | Macro-F1 (%) | 76.50 +/- 5.32 | 76.02 +/- 6.27 |
| 15CCs | Multi-step Accuracy (%) | 71.51 +/- 11.16 | 70.45 +/- 12.74 |
| 15CCs | Mean Steps | 1.7501 | 1.7675 |
| 15CCs | Wrong Elimination (%) | 5.26 | 4.63 |

## Findings

1. CBand reduces DTW matrix construction time by
   approximately 31.49% on 12CCs and 25.06% on 15CCs.
2. On 12CCs, mean recognition accuracy increases
   by 1.19 percentage points, with lower seed variance.
3. On 15CCs, mean recognition accuracy decreases
   by 0.50 percentage points.
4. Accuracy improves in two out of three seeds
   on each dataset.
5. The constrained window changes DTW distances,
   which may affect downstream PPO decisions.

## Limitations

- Only three random seeds were evaluated.
- Statistical significance and accuracy equivalence
  have not been established.
- DTW construction speedup is not end-to-end speedup.
- Results do not establish that CBand universally
  outperforms full DTW.
- The controlled baseline is not the exact original
  paper's E1 evaluation protocol.
- No joint experiment with MEF or LATFF was conducted.

## Archive

- 12 completed PPO training runs
- 60 training epochs
- 60 checkpoint SHA256 records
- 114 test prediction CSV files
- 35136 test prediction records
- 12 aggregate evaluation JSON files

Repeated evaluation across seeds does not create
independent test samples.

Raw datasets, model checkpoints and probability
cache tensors remain on the experiment server.
