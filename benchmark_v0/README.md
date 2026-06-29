# Benchmark v0: Non-Markovian Robot Task Monitoring

This directory records the first auditable benchmark cases for Memory-GRM.

The benchmark is intentionally small at this stage. Its purpose is to create
history-dependent examples where final visual state alone is insufficient for
success/failure judgment.

Current data scope: `xzx_radio_sub23` is the only verified non-Markovian
episode in the available data. Cached carrot/cube/bottle cases are retained as
baseline diagnostics and qualitative inspection material; they are not counted
as non-Markovian benchmark labels.

## First Case: `xzx_radio_sub23`

Task:

> Pick up the red radio, press the switch with the left hand until the
> indicator light turns green, then put the radio down.

Why this is non-Markovian:

- The final state can look similar to the initial state or to a failed ending.
- The indicator light may be invisible at the end after the radio is put down.
- Success depends on an intermediate event: the switch was pressed and the
  indicator light turned green.

Video source:

- `aligned_data/xzx_episode_1_sub23/cam_high.mp4`
- `aligned_data/xzx_episode_1_sub23/cam_left_wrist.mp4`
- `aligned_data/xzx_episode_1_sub23/cam_right_wrist.mp4`

Current label status:

- User-described as a successful turn-on episode.
- Required event frames have been human-verified from keyframes.
- `button_press` is assigned to frame 615.
- `indicator_green` is clearly visible at frame 630 in the left-wrist view.
- Event labels are recorded in
  `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`.
- This is not yet an automatic event detector; VLM/event-head prelabeling is future work.

## Validation

Command:

```bash
conda run -n robo-dopamine python research/validate_benchmark_v0.py
```

Outputs:

- `research_outputs/benchmark_v0_validation.md`
- `research_outputs/benchmark_v0_validation.json`

Current result:

- Status: PASS.
- Episodes: 1.
- Errors: 0.
- Warnings: 0.
- Non-Markovian events verified for `xzx_radio_sub23`: `button_press`, `indicator_green`.

## Radio Data Protocol

The next data collection/annotation protocol is documented in:

- `benchmark_v0/RADIO_DATA_PROTOCOL.md`

It defines required turn-on-radio success/failure histories, annotation fields,
label rules, and processing commands for growing Benchmark v0 without violating
the current data-scope boundary.

Scaffold command for a new radio episode:

```bash
conda run -n robo-dopamine python research/scaffold_radio_benchmark_episode.py \
  --episode-id <episode_id> \
  --data-dir aligned_data/<episode_id> \
  --label unknown \
  --case-type radio_success_hidden_green
```

## Event Vocabulary

- `grasp`
- `lift`
- `place`
- `release`
- `drop`
- `slip`
- `wrong_object`
- `wrong_target`
- `collision`
- `forbidden_contact`
- `order_violation`
- `button_press`
- `indicator_green`

The last two events are task-specific extensions for the radio case.

## GRM Baseline Result

Script:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n robo-dopamine python research/run_xzx_radio_sub23_grm.py
```

Output:

- `results/xzx_episode_1_sub23_memory_grm/run_summary.json`
- `results/xzx_episode_1_sub23_memory_grm/progress_curve.png`
- `benchmark_v0/keyframes/xzx_radio_sub23/manifest.json`

Result with GRM-2.0-8B, blank goal, frame interval 30:

| Mode | Final progress | Mean progress | Max progress |
|---|---:|---:|---:|
| forward | 40.00% | 36.33% | 75.00% |
| incremental | 12.51% | 28.55% | 83.29% |
| backward | 50.00% | 40.90% | 75.00% |
| fused average | 34.17% | 35.26% | 71.43% |

Interpretation:

The score peaks in the middle of the episode and drops by the final frame.
This supports the Memory-GRM hypothesis: success evidence is an intermediate
event (`button_press` and `indicator_green`) that should be latched in memory.

## EventMemory MVP Result

Command:

```bash
conda run -n robo-dopamine python research/evaluate_radio_event_memory_mvp.py
```

Outputs:

- `research_outputs/radio_event_memory_mvp.md`
- `research_outputs/radio_event_memory_mvp.json`

Decision comparison:

| Monitor | Decision | Evidence |
|---|---|---|
| Final-only GRM | not success | final fused progress 34.17% < 70.00% |
| Score-memory GRM | success | peak fused progress 71.43% >= 70.00% |
| Event-Latched GRM | success | required event chain is complete and no negative latch is active |

## Generic Benchmark v0 Evaluation

Command:

```bash
conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py
```

Outputs:

- `research_outputs/benchmark_v0_event_memory_eval.md`
- `research_outputs/benchmark_v0_event_memory_eval.json`

Current result:

| Monitor | Accuracy on labeled Benchmark v0 |
|---|---:|
| Final-only GRM | 0.00% |
| Score-memory GRM | 100.00% |
| Event-Latched GRM | 100.00% |

This is a one-episode feasibility result over the current verified
non-Markovian benchmark, not a statistically complete benchmark.

## Event Counterfactual Stress Test

Command:

```bash
conda run -n robo-dopamine python research/evaluate_radio_event_counterfactuals.py
```

Outputs:

- `research_outputs/radio_event_counterfactuals.md`
- `research_outputs/radio_event_counterfactuals.json`

This stress test reuses the same radio GRM score curve and changes only
event-memory labels. It is not additional real robot data.

| Monitor | Accuracy | Balanced accuracy |
|---|---:|---:|
| Final-only GRM | 80.00% | 50.00% |
| Score-memory GRM | 20.00% | 50.00% |
| Event-Latched GRM | 100.00% | 100.00% |

## Paper Figure

Command:

```bash
conda run -n robo-dopamine python research/make_radio_case_figure.py
```

Outputs:

- `research_outputs/figures/radio_memory_grm_case.png`
- `research_outputs/figures/radio_memory_grm_case.pdf`
- `research_outputs/figures/radio_memory_grm_case.tex`
- `research_outputs/figures/radio_memory_grm_case.md`
- `research_outputs/figures/radio_memory_grm_case.json`

The figure shows selected verified keyframes, GRM progress curves, the low
final-only decision, and the event-memory latch over `button_press` and
`indicator_green`. It is a visualization of the single verified radio case, not
additional benchmark data.

## Paper Artifact Reproduction

No-GPU reproduction command:

```bash
conda run -n robo-dopamine python research/reproduce_current_paper_results.py
```

Outputs:

- `research_outputs/current_paper_reproduction.md`
- `research_outputs/current_paper_reproduction.json`
- `research_outputs/paper_ready_results_summary.md`
- `research_outputs/paper_ready_results_summary.json`
- `research_outputs/paper_tables.tex`
- `research_outputs/figures/radio_memory_grm_case.png`
- `research_outputs/figures/radio_memory_grm_case.pdf`

This pipeline regenerates the current cached-result paper artifacts and checks
the data-scope invariant. It does not rerun GRM inference and does not add
pending labels to Benchmark v0.

## Next Candidate Cases

The next cached cases for manual inspection are listed in:

- `research_outputs/benchmark_v0_candidate_cases.md`
- `research_outputs/benchmark_v0_candidate_cases.json`
- `research_outputs/benchmark_v0_candidate_keyframes.md`

These candidates are not benchmark labels. They are selected for keyframe inspection because they are either high-scoring negatives or large score-regression cases. Under the current data constraint, they should not be described as non-Markovian evidence unless manual review proves genuinely history-dependent labels with final-state-similar histories.

Contact sheets for the prepared candidate episodes are under:

- `benchmark_v0/keyframes/candidate_cases/`
