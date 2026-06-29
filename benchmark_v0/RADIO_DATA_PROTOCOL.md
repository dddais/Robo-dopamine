# Turn-On-Radio Data Protocol / Radio 非马尔可夫数据协议

This protocol defines the next data needed to turn the current one-case
Memory-GRM feasibility result into a publishable non-Markovian benchmark.

本协议定义下一批需要采集/构造的数据，用于把当前 one-case Memory-GRM
可行性结果扩展为可投稿的非马尔可夫 benchmark。

## Current Scope / 当前范围

- Verified non-Markovian episode: `xzx_radio_sub23`.
- Task: pick up the red radio, press the switch with the left hand until the
  indicator turns green, then put the radio down.
- Key hidden events: `button_press`, `indicator_green`.
- Current limitation: one real success episode plus event-label counterfactuals.

Do not add carrot/cube/bottle episodes as non-Markovian labels unless manual
inspection proves final-state-similar histories with different history labels.

不要把 carrot/cube/bottle episode 直接加入非马尔可夫标签，除非人工检查证明它们
存在“终态视觉相似但历史标签不同”的历史依赖。

## Required Episode Types / 需要的 episode 类型

Collect or construct matched histories where the final visual state is as
similar as possible, but event histories differ.

| Type | Target label | Required history | Purpose |
|---|---:|---|---|
| `radio_success_hidden_green` | true | `grasp -> lift -> button_press -> indicator_green -> place -> release` | Positive case with hidden intermediate success evidence. |
| `radio_no_green` | false | `grasp -> lift -> button_press? -> place -> release`, no `indicator_green` | Tests whether memory checks the green indicator, not only manipulation progress. |
| `radio_no_press` | false | `grasp -> lift -> place -> release`, no valid `button_press` | Tests missing required event. |
| `radio_wrong_order` | false | `place/release` before `indicator_green`, or press after release | Tests order constraints. |
| `radio_negative_latch` | false | required events may occur, but `drop`, `collision`, or `forbidden_contact` is active | Tests irreversible violations. |
| `radio_visual_decoy` | false | final pose resembles success, but switch/indicator event did not happen | Tests final-state-similar negative histories. |

Recommended minimum for a first paper-quality benchmark:

- 5 successful radio episodes.
- 5 failed `no_green` or `no_press` episodes.
- 3 wrong-order or premature-release episodes.
- 3 negative-latch episodes.
- At least 5 final-state-similar positive/negative pairs.

## Required Files / 必需文件

Each episode should have synchronized views:

```text
aligned_data/<episode_id>/cam_high.mp4
aligned_data/<episode_id>/cam_left_wrist.mp4
aligned_data/<episode_id>/cam_right_wrist.mp4
```

Expected video metadata:

- FPS: preferably 30.
- Views: front/high, left wrist, right wrist.
- Resolution can vary, but record it in `benchmark_v0/episodes.json`.

## Annotation Schema / 标注格式

Add one event file per episode:

```text
benchmark_v0/event_annotations/<episode_id>_events.json
```

Minimum event file:

```json
{
  "episode_id": "<episode_id>",
  "annotation_status": "human_verified_from_keyframes",
  "task": "pick up the red radio, press the switch with the left hand until the indicator light turns green, then put the radio down",
  "success_rule": {
    "type": "ordered_required_events",
    "required_order": ["grasp", "lift", "button_press", "indicator_green", "place", "release"],
    "non_markovian_events": ["button_press", "indicator_green"],
    "description": "The episode is successful only if the switch is pressed, the indicator turns green, and the radio is released after those events."
  },
  "events": [],
  "negative_event_latches": {
    "drop": false,
    "slip": false,
    "wrong_object": false,
    "wrong_target": false,
    "collision": false,
    "forbidden_contact": false,
    "order_violation": false
  },
  "human_correction_notes": ""
}
```

Each event must include:

```json
{
  "event": "indicator_green",
  "time_index": 21.0,
  "frame_id": 630,
  "view_evidence": [
    "benchmark_v0/keyframes/<episode_id>/frame_000630_cam_left_wrist.png"
  ],
  "confidence": 0.98,
  "source": "human_verified_from_keyframes",
  "notes": "Left-wrist view shows the indicator turned green."
}
```

## Label Rules / 标签规则

Success is true only when all conditions hold:

- `grasp`, `lift`, `button_press`, `indicator_green`, `place`, `release` are
  present.
- Required events are in frame order.
- No negative latch is active.

Success is false if any of these hold:

- `button_press` is missing.
- `indicator_green` is missing.
- `indicator_green` occurs before `button_press`.
- `place` or `release` occurs before `indicator_green`.
- Any irreversible negative latch is active.

## Processing Checklist / 处理流程

1. Add videos under `aligned_data/<episode_id>/`.
2. Generate scaffold files outside the live benchmark:

```bash
conda run -n robo-dopamine python research/scaffold_radio_benchmark_episode.py \
  --episode-id <episode_id> \
  --data-dir aligned_data/<episode_id> \
  --label unknown \
  --case-type radio_success_hidden_green
```

The scaffold is written to:

```text
research_outputs/scaffolded_radio_episodes/<episode_id>/
```

3. Extract keyframes around candidate event windows.
4. Fill the scaffolded event JSON with frame ids, timestamps, evidence paths,
   confidence, and notes.
5. Set `success_label` and `label_status` only after human verification.
6. Run GRM inference or reuse existing predictions, then replace `cached_pred_path`
   TODOs in the scaffolded episode entry.
7. Copy the finalized episode entry into `benchmark_v0/episodes.json` and the
   finalized event JSON into `benchmark_v0/event_annotations/`.
8. Run the benchmark validator:

```bash
conda run -n robo-dopamine python research/validate_benchmark_v0.py
```

9. Run or reuse GRM inference. For new radio episodes, adapt:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n robo-dopamine python research/run_xzx_radio_sub23_grm.py
```

10. Run benchmark evaluation:

```bash
conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py
conda run -n robo-dopamine python research/make_paper_ready_summary.py
```

## Paper Reporting Rules / 论文报告规则

- Report real episodes separately from event-label counterfactuals.
- Report human-verified labels separately from automatic event proposals.
- Use balanced accuracy when counterfactual labels are class-imbalanced.
- Keep the claim calibrated: current results support feasibility until enough
  matched positive/negative histories are collected.
