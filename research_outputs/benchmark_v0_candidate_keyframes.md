# Candidate Keyframes / 候选案例关键帧

## English Summary

This report indexes contact sheets for manual inspection and baseline diagnosis. It uses existing local videos only and does not rerun GRM. These candidates are not current non-Markovian benchmark labels.

## 中文总结

本报告索引人工检查和 baseline 诊断所需的 contact sheet。它只使用已有本地视频，不重新运行 GRM。这些候选不是当前非马尔可夫 benchmark 标签。

## Extracted Episodes / 已抽帧案例

| Data | Priority | Task | Frames | Contact sheet | Manifest |
|---|---|---|---:|---|---|
| `pick3fail_5_cube` | high_scoring_negative | `cube` | 12 | `benchmark_v0/keyframes/candidate_cases/pick3fail_5_cube/contact_sheet.png` | `benchmark_v0/keyframes/candidate_cases/pick3fail_5_cube/manifest.json` |
| `pick3fail_4` | high_scoring_negative | `carrot` | 12 | `benchmark_v0/keyframes/candidate_cases/pick3fail_4/contact_sheet.png` | `benchmark_v0/keyframes/candidate_cases/pick3fail_4/manifest.json` |
| `pick3fail_6_bottle` | high_scoring_negative | `bottle` | 12 | `benchmark_v0/keyframes/candidate_cases/pick3fail_6_bottle/contact_sheet.png` | `benchmark_v0/keyframes/candidate_cases/pick3fail_6_bottle/manifest.json` |
| `pick3fail_11` | large_score_regression | `carrot` | 12 | `benchmark_v0/keyframes/candidate_cases/pick3fail_11/contact_sheet.png` | `benchmark_v0/keyframes/candidate_cases/pick3fail_11/manifest.json` |
| `pick3fail_9_cube` | large_score_regression | `cube` | 12 | `benchmark_v0/keyframes/candidate_cases/pick3fail_9_cube/contact_sheet.png` | `benchmark_v0/keyframes/candidate_cases/pick3fail_9_cube/manifest.json` |

## Manual Inspection Checklist / 人工检查清单

English:

- Decide whether the failure is visible from the final state or depends on history.
- Mark candidate events such as `grasp`, `lift`, `drop`, `slip`, `place`, `release`, `wrong_object`, and `wrong_target`.
- Keep the case as non-Markovian only if event history changes the success/failure label under similar final visual evidence.
- Otherwise, keep the case as a visible-state baseline diagnostic.

中文：

- 判断失败是否仅靠终态可见，还是依赖历史事件。
- 标注候选事件，例如 `grasp`, `lift`, `drop`, `slip`, `place`, `release`, `wrong_object`, `wrong_target`。
- 只有当历史事件会在相似终态视觉证据下改变成败标签时，才把该案例作为非马尔可夫样本。
- 否则保留为可见状态 baseline 诊断样本。
