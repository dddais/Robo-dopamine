# Current Paper Reproduction / 当前论文结果复现

This pipeline reruns the current no-GPU Memory-GRM artifacts from cached results and human-verified event labels. It does not rerun Robo-Dopamine GRM inference and does not add benchmark labels.

当前 pipeline 基于已有缓存结果和人工核验事件标签，重新生成当前 Memory-GRM 论文产物。它不会重新运行 Robo-Dopamine GRM 推理，也不会新增 benchmark 标签。

## Data Scope / 数据边界

- Verified non-Markovian benchmark episode: `xzx_radio_sub23` only.
- Cached carrot/cube/bottle cases are visible-state baseline diagnostics and candidate material only.

## Steps / 步骤

| Step | Status | Seconds | Key outputs |
|---|---:|---:|---|
| `cached_grm_aggregation` | PASS | 0.05 | `research_outputs/cached_grm_aggregation.md`<br>`research_outputs/cached_grm_metrics.csv`<br>`research_outputs/cached_grm_rows.csv` |
| `trajectory_memory_features` | PASS | 0.68 | `research_outputs/trajectory_memory_features.md`<br>`research_outputs/trajectory_memory_metrics.csv`<br>`research_outputs/trajectory_memory_cases.csv` |
| `benchmark_v0_validation` | PASS | 0.04 | `research_outputs/benchmark_v0_validation.md`<br>`research_outputs/benchmark_v0_validation.json` |
| `radio_event_memory_mvp` | PASS | 0.04 | `research_outputs/radio_event_memory_mvp.md`<br>`research_outputs/radio_event_memory_mvp.json` |
| `benchmark_v0_event_memory` | PASS | 0.04 | `research_outputs/benchmark_v0_event_memory_eval.md`<br>`research_outputs/benchmark_v0_event_memory_eval.json` |
| `radio_event_counterfactuals` | PASS | 0.04 | `research_outputs/radio_event_counterfactuals.md`<br>`research_outputs/radio_event_counterfactuals.json` |
| `paper_ready_summary` | PASS | 0.03 | `research_outputs/paper_ready_results_summary.md`<br>`research_outputs/paper_ready_results_summary.json` |
| `claim_evidence_ledger` | PASS | 0.03 | `research_outputs/claim_evidence_ledger.md`<br>`research_outputs/claim_evidence_ledger.json` |
| `radio_intake_inventory` | PASS | 0.29 | `research_outputs/radio_intake_candidates.md`<br>`research_outputs/radio_intake_candidates.json` |
| `radio_intake_keyframes` | PASS | 8.29 | `research_outputs/radio_intake_keyframes.md`<br>`research_outputs/radio_intake_keyframes.json` |
| `pending_radio_event_window` | PASS | 1.92 | `research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_event_window.json`<br>`research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_event_window.md`<br>`research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_window_540_649/contact_sheet.png` |
| `paper_latex_tables` | PASS | 0.03 | `research_outputs/paper_tables.tex` |
| `radio_case_figure` | PASS | 1.88 | `research_outputs/figures/radio_memory_grm_case.png`<br>`research_outputs/figures/radio_memory_grm_case.pdf`<br>`research_outputs/figures/radio_memory_grm_case.tex`<br>`research_outputs/figures/radio_memory_grm_case.md`<br>`research_outputs/figures/radio_memory_grm_case.json` |
| `manuscript_package_check` | PASS | 0.04 | `research_outputs/manuscript_package_check.md`<br>`research_outputs/manuscript_package_check.json` |

## Invariants / 不变量

- Benchmark v0 valid: `True`
- Benchmark v0 errors/warnings: `0` / `0`
- Evaluated Benchmark v0 episodes: `1`
- Live benchmark episode ids: `xzx_radio_sub23`
- Verified non-Markovian episodes in paper summary: `xzx_radio_sub23`
- Manuscript package valid: `True`
- Manuscript package errors/warnings: `0` / `0`
- Claim ledger valid / adds labels: `True` / `False`
- Radio intake candidates / pending / adds labels: `5` / `4` / `False`
- Radio intake keyframe sheets / adds labels: `4` / `False`
- Pending radio event window / frames / adds labels: `xzx_episode_1_sub2` / `11` / `False`

## Reproducibility / 可复现性

Command:

```bash
conda run -n robo-dopamine python research/reproduce_current_paper_results.py
```
