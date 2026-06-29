# Scaffolded Radio Benchmark Episode / Radio Benchmark 样本模板

This scaffold is intentionally written outside the live benchmark by default. Review and complete it before copying into `benchmark_v0/episodes.json` and `benchmark_v0/event_annotations/`.

该模板默认写入 research_outputs，不直接修改 live benchmark。请先完成人工核验和 GRM 路径替换，再合并到 `benchmark_v0/episodes.json` 和 `benchmark_v0/event_annotations/`。

## Episode / 样本

- Episode id: `xzx_radio_sub2_pending_review`
- Case type: `radio_success_hidden_green`
- Initial label: `unknown`
- Data dir: `aligned_data/xzx_episode_1_sub2`
- Episode template: `research_outputs/scaffolded_radio_episodes/xzx_radio_sub2_pending_review/episode_entry.json`
- Event template: `research_outputs/scaffolded_radio_episodes/xzx_radio_sub2_pending_review/xzx_radio_sub2_pending_review_events.json`

## Video Metadata / 视频元数据

| View | Frames | FPS | Resolution |
|---|---:|---:|---|
| `front` | 650 | 30.00 | 720x720 |
| `left_wrist` | 650 | 30.00 | 480x480 |
| `right_wrist` | 650 | 30.00 | 480x480 |

## Required Next Steps / 后续步骤

1. Extract event-window keyframes for `grasp`, `lift`, `button_press`, `indicator_green`, `place`, and `release`.
2. Fill frame ids, timestamps, evidence paths, confidence, and notes in the event template.
3. Set `success_label` and `label_status` only after human verification.
4. Run GRM for forward/incremental/backward modes and replace `cached_pred_path` TODOs.
5. Copy finalized entries into the live benchmark.
6. Run validation and evaluation:

```bash
conda run -n robo-dopamine python research/validate_benchmark_v0.py
conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py
conda run -n robo-dopamine python research/make_paper_ready_summary.py
```
