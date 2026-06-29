# Radio Event Counterfactual Stress Test / Radio 事件反事实压力测试

## English Summary

This stress test reuses the same radio trajectory and the same cached GRM score curve, then changes only the event-memory labels. It checks whether a monitor can distinguish missing required events, order violations, and negative latches when scalar visual progress evidence is identical.

- Source trajectory: `results/xzx_episode_1_sub23_memory_grm/run_summary.json`
- Source event labels: `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`
- Counterfactual variants: 5
- Shared fused final progress: 34.17%
- Shared fused peak progress: 71.43%

## 中文总结

该压力测试复用同一条 radio 轨迹和同一条 GRM 分数曲线，只改变事件记忆标签。它用于检查：当标量视觉进度证据完全相同时，监控器能否区分缺失必要事件、顺序违规和负事件锁存。

- 源轨迹：`results/xzx_episode_1_sub23_memory_grm/run_summary.json`
- 源事件标签：`benchmark_v0/event_annotations/xzx_radio_sub23_events.json`
- 反事实变体数量：5
- 共享 fused final progress：34.17%
- 共享 fused peak progress：71.43%

## Aggregate Results / 汇总结果

| Monitor | Accuracy | Balanced accuracy | Valid-history recall | Invalid-history recall |
|---|---:|---:|---:|---:|
| Final-only GRM / 只看终态 GRM | 80.00% | 50.00% | 0.00% | 100.00% |
| Score-memory GRM / 分数轨迹记忆 GRM | 20.00% | 50.00% | 100.00% | 0.00% |
| Event-Latched GRM / 事件锁存 GRM | 100.00% | 100.00% | 100.00% | 100.00% |

## Variant Results / 变体结果

| Variant | Label | Final-only | Score-memory | Event-latched | Counterfactual meaning |
|---|---:|---|---|---|---|
| `observed_success` | true | not success (wrong) | success (correct) | success (correct) | Human-verified observed radio success event chain. |
| `missing_indicator_green` | false | not success (correct) | success (wrong) | not success (correct) | Synthetic invalid history: the green indicator event is absent. |
| `missing_button_press` | false | not success (correct) | success (wrong) | not success (correct) | Synthetic invalid history: the required switch press event is absent. |
| `indicator_before_button_press` | false | not success (correct) | success (wrong) | not success (correct) | Synthetic invalid history: required event order is violated. |
| `forbidden_contact_latched` | false | not success (correct) | success (wrong) | not success (correct) | Synthetic invalid history: a negative event latch is active. |

## Interpretation / 解释

English: Final-only GRM makes the same decision for all histories because the final score is unchanged. Its high raw accuracy here is a class-imbalance artifact because four of five variants are invalid. Score-memory GRM also makes the same decision for all histories because the peak score is unchanged. Event-Latched GRM changes its decision when required events are missing, out of order, or invalidated by a negative latch.

中文：Final-only GRM 对所有历史给出同一判定，因为最终分数不变。这里 raw accuracy 较高只是类别不均衡造成的表象，因为 5 个变体中 4 个是 invalid。Score-memory GRM 也对所有历史给出同一判定，因为峰值分数不变。Event-Latched GRM 会在必要事件缺失、顺序错误或负事件锁存时改变判定。

## Limitation / 限制

These are event-label counterfactuals over one observed trajectory. They test monitor logic under identical GRM score evidence; they are not additional real robot episodes.

Command:

```bash
conda run -n robo-dopamine python research/evaluate_radio_event_counterfactuals.py
```
