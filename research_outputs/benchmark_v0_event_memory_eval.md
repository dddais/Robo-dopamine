# Benchmark v0 EventMemory Evaluation / Benchmark v0 事件记忆评估

## English Summary

This report evaluates the current Benchmark v0 through a generic episode loader. It reuses cached GRM summaries and human-verified event labels; it does not rerun GRM and does not claim automatic event detection.

- Episodes in `benchmark_v0/episodes.json`: 1
- Evaluated episodes: 1
- Labeled episodes: 1
- Current non-Markovian data scope: only `xzx_radio_sub23` is counted as a verified non-Markovian benchmark episode.
- Cached carrot/cube candidates remain qualitative inspection material, not non-Markovian benchmark labels.

## 中文总结

本报告通过通用 episode loader 评估当前 Benchmark v0。它复用已有 GRM summary 和人工核验事件标签；不重新运行 GRM，也不声称自动事件检测已经完成。

- `benchmark_v0/episodes.json` 中 episode 数量：1
- 已评估 episode 数量：1
- 有成功标签的 episode 数量：1
- 当前非马尔可夫数据范围：只有 `xzx_radio_sub23` 被计入已核验非马尔可夫 benchmark episode。
- 已缓存 carrot/cube 候选只作为定性检查材料，不作为当前非马尔可夫 benchmark 标签。

## Aggregate Results / 汇总结果

| Monitor | Accuracy on labeled Benchmark v0 |
|---|---:|
| Final-only GRM / 只看终态 GRM | 0.00% |
| Score-memory GRM / 分数轨迹记忆 GRM | 100.00% |
| Event-Latched GRM / 事件锁存 GRM | 100.00% |

- Mean fused final progress: 34.17%
- Mean fused peak progress: 71.43%

## Episode Results / 单 episode 结果

| Episode | Rule | Label | Final | Peak | Final-only | Score-memory | Event-latched | Non-Markovian evidence |
|---|---|---:|---:|---:|---|---|---|---|
| `xzx_radio_sub23` | `hidden_intermediate_success_event` | true | 34.17% | 71.43% | not success (wrong) | success (correct) | success (correct) | button_press: frame 615; indicator_green: frame 630 |

## Reproducibility / 可复现性

Command:

```bash
conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py
```

Inputs:

- `benchmark_v0/episodes.json`
- `results/xzx_episode_1_sub23_memory_grm/run_summary.json`
- `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`

Outputs:

- `research_outputs/benchmark_v0_event_memory_eval.md`
- `research_outputs/benchmark_v0_event_memory_eval.json`

Limitation / 限制：the current evaluated benchmark has one verified non-Markovian episode. The next publishable step is collecting or constructing additional final-state-similar histories, not re-labeling ordinary visible-state failures as non-Markovian cases.
