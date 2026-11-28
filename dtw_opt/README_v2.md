# DTW v2 controlled evaluation

The teammate's six profile files and core distance modules are preserved. CRLF source files were normalized to LF. The alternative C-library window implementation and label-indexed matrices are the experimental changes; test-trace-derived matrices must not be used. The legacy entry points default to band and may generate matrices from labeled test traces if invoked without an explicit matrix. Use run_v2.sh instead.

The controlled runners reuse the common PPO implementation from experiment/ppo-opt and experiment/mef-15ccs, **baseline mode only**, without MEF fusion or no-mask. Both modes use identical frozen classifier probabilities and matched initialization; only DTW matrices differ. A single median-duration representative is selected per class in each environment using the teammate's labeled_matrix.py. Test matrices are mapped by environment name from training matrices. Both C-library modes use keep_internals=True and symmetric2; the 10% window is enlarged when necessary for length differences. Three alternating timing repeats measure the same real training trace pairs. This is not a direct reproduction of the upstream E1 evaluation or the earlier incorrect-index experiment.

The first run uses seed 3402, five epochs, batch size 36, ten PPO updates, threshold .7, and the same controlled reward coefficients for both variants. Further seeds are needed before claiming stability. For 12CCs 3456 test records and for 15CCs 2400 are expected per mode. Actual speedup and accuracy must be measured rather than assumed.

Use separate output roots; no raw datasets or weights are committed. The auxiliary legacy comparison scripts are not required for this controlled runner.
