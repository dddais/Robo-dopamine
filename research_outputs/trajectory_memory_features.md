# Trajectory Memory Feature Analysis

This report tests whether simple memory-like features computed only from cached GRM score trajectories can improve success/failure judgment.

Important: these features do not inspect visual event content; they only use scalar progress and hop curves. A failure here motivates explicit event memory rather than scalar smoothing.

## Data

- Complete cases: 90
- Goal-success positives: 6
- Negatives: 84

## Metrics

| feature | AUROC | best F1 | accuracy | threshold | FP | FN | weights |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_final_avg | 99.0% | 0.800 | 0.978 | 93.22 | 0 | 2 |  |
| trajectory_mean_avg | 100.0% | 1.000 | 1.000 | 42.32 | 0 | 0 |  |
| minus_drawdown_0.5 | 98.6% | 0.800 | 0.967 | 61.36 | 3 | 0 |  |
| minus_neg_hop_0.25 | 98.6% | 0.800 | 0.967 | 50.73 | 3 | 0 |  |
| grid_memory_score | 99.2% | 0.833 | 0.978 | 86.97 | 1 | 1 | drawdown=0.0, neg=0.05 |

## Remaining False Positives Under Best Scalar-Memory Score

| score | data | scene | task | interval | final_avg | drawdown_avg | neg_hop_avg |
|---:|---|---|---|---:|---:|---:|---:|
| 88.74 | pick3fail_5_cube | cube | cube | 20 | 92.64 | 23.16 | 78.07 |

## Interpretation

Scalar trajectory memory helps only if failures leave visible traces in the GRM score curve, such as large regressions or unstable progress. Some cached failure episodes still receive very high final progress and remain difficult to reject with score-only memory. The proposed research should therefore add explicit visual/event memory, e.g. remembering order, forbidden contacts, dropped objects, and transient violations.
