# Cached GRM Aggregation

This report aggregates cached Robo-Dopamine summary files. `goal_success` is true only when the video is a success episode and the task prompt matches the object in the video.

## Data

- Records: 138
- Goal-success positives: 6
- Negatives: 132

## Best Threshold Metrics

| source | model | interval | score | positives | AUROC | best F1 | accuracy | threshold | FP | FN |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| summary3 | multi_task | 10 | avg_progress | 0 | n/a | 0.000 | 1.000 | 72.77 | 0 | 0 |
| summary3 | multi_task | 10 | forward_progress | 0 | n/a | 0.000 | 1.000 | 75.80 | 0 | 0 |
| summary3 | multi_task | 10 | incremental_progress | 0 | n/a | 0.000 | 1.000 | 84.20 | 0 | 0 |
| summary3 | multi_task | 10 | backward_progress | 0 | n/a | 0.000 | 1.000 | 63.30 | 0 | 0 |
| summary3 | multi_task | 20 | avg_progress | 0 | n/a | 0.000 | 1.000 | 70.87 | 0 | 0 |
| summary3 | multi_task | 20 | forward_progress | 0 | n/a | 0.000 | 1.000 | 75.80 | 0 | 0 |
| summary3 | multi_task | 20 | incremental_progress | 0 | n/a | 0.000 | 1.000 | 77.20 | 0 | 0 |
| summary3 | multi_task | 20 | backward_progress | 0 | n/a | 0.000 | 1.000 | 63.30 | 0 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | avg_progress | 3 | 98.4% | 0.857 | 0.978 | 77.80 | 1 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | forward_progress | 3 | 92.1% | 0.545 | 0.889 | 91.20 | 5 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | incremental_progress | 3 | 92.9% | 0.571 | 0.933 | 73.73 | 2 | 1 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | backward_progress | 3 | 98.8% | 0.857 | 0.978 | 94.75 | 1 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | avg_progress | 3 | 100.0% | 1.000 | 1.000 | 93.22 | 0 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | forward_progress | 3 | 92.5% | 0.545 | 0.889 | 93.05 | 5 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | incremental_progress | 3 | 97.6% | 0.857 | 0.978 | 85.89 | 1 | 0 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | backward_progress | 3 | 98.8% | 0.857 | 0.978 | 95.00 | 1 | 0 |

## Score Distributions

| source | model | interval | goal_success | video_success | task_match | n | mean | median | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| summary3 | multi_task | 10 | False | False | False | 16 | 27.04 | 19.76 | 8.70 | 61.24 |
| summary3 | multi_task | 10 | False | False | True | 8 | 29.08 | 18.89 | 6.71 | 72.77 |
| summary3 | multi_task | 20 | False | False | False | 16 | 26.18 | 18.36 | 7.41 | 60.90 |
| summary3 | multi_task | 20 | False | False | True | 8 | 28.49 | 18.20 | 6.06 | 70.87 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | False | False | False | 24 | 14.29 | 4.83 | 1.94 | 54.32 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | False | False | True | 12 | 23.56 | 7.67 | 2.64 | 90.89 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | False | True | False | 6 | 32.51 | 31.35 | 26.54 | 45.30 |
| summary_GRM8B_new | GRM-2.0-8B | 10 | True | True | True | 3 | 90.09 | 90.42 | 82.23 | 97.61 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | False | False | False | 24 | 14.84 | 4.37 | 2.61 | 59.76 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | False | False | True | 12 | 24.03 | 5.82 | 2.22 | 92.64 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | False | True | False | 6 | 40.12 | 35.55 | 34.94 | 55.99 |
| summary_GRM8B_new | GRM-2.0-8B | 20 | True | True | True | 3 | 94.06 | 93.82 | 93.80 | 94.55 |

## Highest-Scoring Negatives

| source | model | data | scene | task | interval | avg | forward | inc | backward |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_5_cube | cube | cube | 20 | 92.64 | 94.70 | 83.22 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_5_cube | cube | cube | 10 | 90.89 | 94.70 | 77.96 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_4 | carrot | carrot | 20 | 89.66 | 95.20 | 83.79 | 90.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_4 | carrot | carrot | 10 | 73.37 | 95.20 | 35.42 | 89.50 |
| summary3 | multi_task | pick3fail_6_bottle | bottle | bottle | 10 | 72.77 | 75.80 | 84.20 | 58.30 |
| summary3 | multi_task | pick3fail_5_cube | cube | cube | 20 | 70.87 | 75.80 | 73.52 | 63.30 |
| summary3 | multi_task | pick3fail_6_bottle | bottle | bottle | 20 | 70.43 | 75.80 | 77.20 | 58.30 |
| summary3 | multi_task | pick3fail_5_cube | cube | cube | 10 | 68.93 | 75.80 | 67.69 | 63.30 |
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_6_bottle | bottle | bottle | 10 | 63.99 | 95.20 | 96.76 | 0.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3fail_6_bottle | bottle | bottle | 20 | 61.67 | 95.00 | 90.02 | 0.00 |
| summary3 | multi_task | pick3fail_6_bottle | bottle | cube | 10 | 61.24 | 72.70 | 67.83 | 43.20 |
| summary3 | multi_task | pick3fail_6_bottle | bottle | cube | 20 | 60.90 | 72.70 | 66.81 | 43.20 |

## Lowest-Scoring Positives

| source | model | data | scene | task | interval | avg | forward | inc | backward |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_1 | carrot | carrot | 10 | 82.23 | 95.50 | 51.20 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_4_cube | cube | cube | 10 | 90.42 | 94.40 | 76.87 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_4_cube | cube | cube | 20 | 93.80 | 94.10 | 87.31 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_3_bottle | bottle | bottle | 20 | 93.82 | 95.20 | 86.26 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_1 | carrot | carrot | 20 | 94.55 | 95.50 | 88.16 | 100.00 |
| summary_GRM8B_new | GRM-2.0-8B | pick3suc_3_bottle | bottle | bottle | 10 | 97.61 | 95.20 | 97.63 | 100.00 |
