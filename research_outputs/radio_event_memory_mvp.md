# Radio EventMemory MVP / Radio 事件记忆最小验证

## English Summary

This MVP uses cached Robo-Dopamine GRM outputs and human-verified keyframe events. It does not rerun GRM and does not claim automatic event detection.

- Episode: `xzx_radio_sub23`
- Final fused GRM progress: 34.17%
- Peak fused GRM progress: 71.43% at frame 450 (15.0s)
- Peak-final gap: 37.26%
- Core finding: final-only GRM gives a low success signal, while Event-Latched GRM recovers success by remembering `button_press` and `indicator_green`.

## 中文总结

该 MVP 使用已有 Robo-Dopamine GRM 缓存结果和人工核验关键帧事件；没有重新跑 GRM，也不声称已经实现自动事件检测。

- Episode: `xzx_radio_sub23`
- GRM fused final progress: 34.17%
- GRM fused peak progress: 71.43%，峰值在 frame 450 (15.0s)
- peak-final gap: 37.26%
- 核心结论：final-only GRM 给出较低成功信号，而 Event-Latched GRM 通过记住 `button_press` 和 `indicator_green` 恢复成功判断。

## Decision Comparison / 判定对比

| Monitor | Decision | Evidence |
|---|---|---|
| Final-only GRM / 只看终态 GRM | not success | final fused progress 34.17% < 70.00% |
| Score-memory GRM / 分数轨迹记忆 GRM | success | peak fused progress 71.43% >= 70.00% |
| Event-Latched GRM / 事件锁存 GRM | success | all required events are present in order and no negative latch is active |

## Event Timeline / 事件时间线

| Event | Frame | Time | Confidence | Evidence |
|---|---:|---:|---:|---|
| `grasp` | 420 | 14.0s | 0.85 | `benchmark_v0/keyframes/xzx_radio_sub23/frame_000420_cam_high.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23/frame_000420_cam_left_wrist.png` |
| `lift` | 450 | 15.0s | 0.90 | `benchmark_v0/keyframes/xzx_radio_sub23/frame_000450_cam_high.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23/frame_000450_cam_left_wrist.png` |
| `button_press` | 615 | 20.5s | 0.78 | `benchmark_v0/keyframes/xzx_radio_sub23_event_window/frame_000600_cam_left_wrist.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23_event_window/frame_000615_cam_left_wrist.png` |
| `indicator_green` | 630 | 21.0s | 0.98 | `benchmark_v0/keyframes/xzx_radio_sub23_event_window/frame_000630_cam_left_wrist.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23/frame_000630_cam_left_wrist.png` |
| `place` | 660 | 22.0s | 0.82 | `benchmark_v0/keyframes/xzx_radio_sub23_event_window/frame_000660_cam_high.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23_event_window/frame_000660_cam_left_wrist.png` |
| `release` | 840 | 28.0s | 0.85 | `benchmark_v0/keyframes/xzx_radio_sub23/frame_000840_cam_high.png`<br>`benchmark_v0/keyframes/xzx_radio_sub23/frame_000840_cam_left_wrist.png` |

## Interpretation / 解释

English: The final frame no longer exposes the key success evidence clearly. A monitor that only thresholds final GRM progress can therefore under-estimate this successful non-Markovian episode. The event-memory rule stores the intermediate `button_press` and `indicator_green` events, so the final decision remains successful after the radio is put down.

中文：终态画面不再清楚呈现关键成功证据。因此，只对 final GRM progress 设阈值的监控器会低估这个成功的非马尔可夫任务。事件记忆规则会在中段锁存 `button_press` 和 `indicator_green`，所以 radio 放下后仍能保留成功判断。

## Reproducibility / 可复现性

Command:

```bash
python research/evaluate_radio_event_memory_mvp.py
```

Inputs:

- `results/xzx_episode_1_sub23_memory_grm/run_summary.json`
- `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`

Outputs:

- `research_outputs/radio_event_memory_mvp.md`
- `research_outputs/radio_event_memory_mvp.json`

Limitation / 限制：event labels are human-verified from keyframes; automatic VLM/event-head detection remains future work. 事件标签来自人工关键帧核验，自动 VLM/event-head 检测仍是下一步工作。
