# Radio Intake Candidates / Radio 待接入候选

This inventory lists local radio-like videos that could be inspected next. It does not add benchmark labels and does not change `benchmark_v0/episodes.json`.

当前 live verified non-Markovian benchmark 仍然只有 `xzx_radio_sub23`。

## Summary

- Candidates found: `5`
- Pending candidates: `4`
- Already live verified entries: `1`
- Adds benchmark labels: `False`

## Candidates

| Candidate | Status | Frames | FPS | Cached preds | Modes | Next step |
|---|---|---:|---:|---:|---|---|
| `xzx_episode_1` | `pending_human_verification` | 1956 | 30.00 | 3 | `backward, forward, incremental` | Scaffold outside live benchmark, extract keyframes, and human-verify button_press/indicator_green before any benchmark inclusion. |
| `xzx_episode_1_sub1` | `pending_human_verification` | 750 | 30.00 | 9 | `backward, forward, incremental` | Scaffold outside live benchmark, extract keyframes, and human-verify button_press/indicator_green before any benchmark inclusion. |
| `xzx_episode_1_sub2` | `pending_human_verification` | 650 | 30.00 | 18 | `backward, forward, incremental` | Scaffold outside live benchmark, extract keyframes, and human-verify button_press/indicator_green before any benchmark inclusion. |
| `xzx_episode_1_sub23` | `already_live_verified_as_xzx_radio_sub23` | 850 | 30.00 | 3 | `backward, forward, incremental` | Already represented by xzx_radio_sub23; do not duplicate. |
| `xzx_episode_2` | `pending_human_verification` | 2340 | 30.00 | 0 | `` | Scaffold outside live benchmark, extract keyframes, and human-verify button_press/indicator_green before any benchmark inclusion. |

## Required Gate Before Benchmark Inclusion

- Create scaffold files under `research_outputs/scaffolded_radio_episodes/`.
- Extract event-window keyframes.
- Human-verify `button_press` and `indicator_green`, including frame ids and evidence views.
- Fill success label and cached GRM paths.
- Only then copy finalized files into `benchmark_v0/` and run `research/validate_benchmark_v0.py`.

## Reproducibility

```bash
conda run -n robo-dopamine python research/inventory_radio_intake_candidates.py
```
