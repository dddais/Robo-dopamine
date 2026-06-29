# Paper-Ready Results Summary / 论文结果汇总

## Data Scope / 数据边界

English: The current verified non-Markovian benchmark contains one episode: `xzx_radio_sub23` / turn-on-radio. Cached carrot/cube/bottle cases are used only as visible-state baseline diagnostics and candidate material.

中文：当前已核验非马尔可夫 benchmark 只有一个 episode：`xzx_radio_sub23` / turn-on-radio。已缓存 carrot/cube/bottle 案例只作为可见状态 baseline 诊断和候选材料。

## Table 1: Visible-State GRM Baseline / 可见状态 GRM baseline

| Setting | AUROC | Best F1 | Accuracy |
|---|---:|---:|---:|
| fused final, interval 10 | 98.4% | 0.857 | 97.8% |
| fused final, interval 20 | 100.0% | 1.000 | 100.0% |
| forward only, interval 10 | 92.1% | 0.545 | 88.9% |
| forward only, interval 20 | 92.5% | 0.545 | 88.9% |

Interpretation: Robo-Dopamine GRM-2.0-8B is already strong for ordinary visible-state success/failure separation. The paper should not claim that GRM is broadly weak.

## Table 2: Score-Only Temporal Memory / 仅分数轨迹记忆

| Feature | AUROC | Best F1 | Accuracy | FP | FN |
|---|---:|---:|---:|---:|---:|
| baseline final average | 99.0% | 0.800 | 97.8% | 0 | 2 |
| trajectory mean average | 100.0% | 1.000 | 100.0% | 0 | 0 |
| final - 0.5 drawdown | 98.6% | 0.800 | 96.7% | 3 | 0 |
| final - 0.25 negative-hop | 98.6% | 0.800 | 96.7% | 3 | 0 |
| best grid scalar-memory score | 99.2% | 0.833 | 97.8% | 1 | 1 |

Interpretation: Scalar trajectory evidence helps on visible-state cached data, but it cannot encode hidden event predicates when score evidence is unchanged.

## Table 3: Current Benchmark v0 Radio Result / 当前 Benchmark v0 radio 结果

| Monitor | Accuracy on labeled Benchmark v0 | Decision | Evidence |
|---|---:|---|---|
| Final-only GRM | 0.0% | not success | final fused progress 34.17% |
| Score-memory GRM | 100.0% | success | peak fused progress 71.43% |
| Event-Latched GRM | 100.0% | success | button_press: frame 615; indicator_green: frame 630 |

Interpretation: On the verified turn-on-radio episode, final-only GRM misses the success because the decisive evidence is an intermediate event. Event-Latched GRM recovers the success by remembering `button_press` and `indicator_green`.

## Table 4: Radio Event Counterfactuals / Radio 事件反事实

| Monitor | Accuracy | Balanced accuracy | Valid-history recall | Invalid-history recall |
|---|---:|---:|---:|---:|
| Final-only GRM | 80.0% | 50.0% | 0.0% | 100.0% |
| Score-memory GRM | 20.0% | 50.0% | 100.0% | 0.0% |
| Event-Latched GRM | 100.0% | 100.0% | 100.0% | 100.0% |

Interpretation: All counterfactual variants share the same GRM score curve. Final-only and score-memory monitors collapse to constant decisions, while Event-Latched GRM separates valid and invalid event histories.

## Claims Supported Now / 当前可支撑的论文主张

- GRM is a strong visible-state progress baseline.
- A single verified turn-on-radio case demonstrates hidden intermediate success evidence.
- Event-latched memory can represent required events, order violations, and negative latches under identical scalar score evidence.
- The current evidence is a feasibility study, not a statistically complete non-Markovian benchmark.

## Claims Not Yet Supported / 当前不能声称

- Do not claim a completed large-scale non-Markovian benchmark.
- Do not claim automatic event detection; current key radio events are human-verified.
- Do not count carrot/cube/bottle cached candidates as non-Markovian labels without manual proof of final-state-similar, history-different episodes.

## Reproducibility / 可复现性

Generated from:

- `cached_grm_metrics`: `research_outputs/cached_grm_metrics.csv`
- `trajectory_memory_metrics`: `research_outputs/trajectory_memory_metrics.csv`
- `benchmark_v0_eval`: `research_outputs/benchmark_v0_event_memory_eval.json`
- `radio_counterfactuals`: `research_outputs/radio_event_counterfactuals.json`

Command:

```bash
conda run -n robo-dopamine python research/make_paper_ready_summary.py
```
