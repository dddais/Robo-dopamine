# Benchmark v0 Candidate Cases / Benchmark v0 候选案例

## English Summary

This report selects cached cases for qualitative inspection and baseline diagnosis. It uses only existing CSV outputs and does not rerun GRM. These candidates are not current non-Markovian benchmark labels.

Selection policy:

- high-scoring negatives: failed/non-goal-success episodes that GRM scores highly;
- large score regressions: episodes with strong trajectory drawdown that may contain transient events.

## 中文总结

本报告从已有缓存结果中筛选定性检查和 baseline 诊断候选案例；只读取现有 CSV，不重新运行 GRM。这些候选不是当前非马尔可夫 benchmark 标签。

筛选策略：

- high-scoring negatives：失败或非目标成功轨迹，但 GRM 给出高分；
- large score regressions：分数轨迹有明显回落，可能包含 slip、drop、recovery 等短暂事件。

## Candidates / 候选列表

| Priority | Data | Task | Interval | Main score | Why inspect |
|---|---|---|---:|---:|---|
| high_scoring_negative | `pick3fail_5_cube` | `cube` | 20 | 92.64 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| high_scoring_negative | `pick3fail_5_cube` | `cube` | 10 | 90.89 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| high_scoring_negative | `pick3fail_4` | `carrot` | 20 | 89.66 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| high_scoring_negative | `pick3fail_4` | `carrot` | 10 | 73.37 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| high_scoring_negative | `pick3fail_6_bottle` | `bottle` | 10 | 63.99 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| high_scoring_negative | `pick3fail_6_bottle` | `bottle` | 20 | 61.67 | Failed or non-goal-success episode with high final fused GRM score; inspect for visually plausible failure, transient violation, or missing event labels. |
| large_score_regression | `pick3fail_11` | `carrot` | 20 | 47.42 | Trajectory has large progress drawdown; inspect for slip, drop, recovery, or other transient events that scalar final progress may miss. |
| large_score_regression | `pick3fail_6_bottle` | `bottle` | 20 | 47.19 | Trajectory has large progress drawdown; inspect for slip, drop, recovery, or other transient events that scalar final progress may miss. |
| large_score_regression | `pick3fail_9_cube` | `cube` | 20 | 45.58 | Trajectory has large progress drawdown; inspect for slip, drop, recovery, or other transient events that scalar final progress may miss. |
| large_score_regression | `pick3fail_9_cube` | `cube` | 10 | 45.58 | Trajectory has large progress drawdown; inspect for slip, drop, recovery, or other transient events that scalar final progress may miss. |

## Recommended Next Manual Step / 建议下一步人工检查

English: Start with the top 3 high-scoring negatives and the top 3 large-regression cases. For each case, extract/contact-sheet keyframes and mark visible events. Keep a case as non-Markovian only if manual inspection proves that history changes the success/failure label under similar final visual evidence; otherwise keep it as a visible-state baseline diagnostic.

中文：优先检查前 3 个高分负例和前 3 个大回落案例。每个案例先抽关键帧/contact sheet，再标注可见事件。只有当人工检查证明在相似终态视觉证据下历史会改变成败标签时，才纳入非马尔可夫 benchmark；否则保留为可见状态 baseline 诊断。
