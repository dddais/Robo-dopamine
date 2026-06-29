# Manuscript Package Check / Paper Package Check

This checker validates the current Memory-GRM manuscript skeleton and generated paper artifacts without adding benchmark labels or rerunning GRM inference.

## Status

- Valid: `True`
- Errors: `0`
- Warnings: `0`
- Verified non-Markovian Benchmark v0 episode: `xzx_radio_sub23` only.

## Checks

| Check | Status | Detail |
|---|---:|---|
| `required_file:manuscript/main.tex` | PASS | size=15014 |
| `required_file:manuscript/references.bib` | PASS | size=1796 |
| `required_file:manuscript/README.md` | PASS | size=1526 |
| `required_file:research_outputs/paper_tables.tex` | PASS | size=2269 |
| `required_file:research_outputs/figures/radio_memory_grm_case.tex` | PASS | size=641 |
| `required_file:research_outputs/figures/radio_memory_grm_case.pdf` | PASS | size=139990 |
| `required_file:research_outputs/paper_ready_results_summary.json` | PASS | size=3262 |
| `required_file:research_outputs/benchmark_v0_validation.json` | PASS | size=824 |
| `main_tex:generated_figure_input` | PASS |  |
| `main_tex:generated_table_input` | PASS |  |
| `main_tex:bibliography_path` | PASS |  |
| `main_tex:one_case_language` | PASS |  |
| `main_tex:feasibility_language` | PASS |  |
| `main_tex:radio_episode_id` | PASS |  |
| `main_tex:all_required_citations_used` | PASS | missing=[] |
| `references:required_bib_keys` | PASS | missing=[] |
| `paper_tables:label:tab:benchmark_v0_radio` | PASS |  |
| `paper_tables:label:tab:radio_counterfactuals` | PASS |  |
| `paper_tables:label:tab:score_memory` | PASS |  |
| `paper_tables:label:tab:visible_state_grm` | PASS |  |
| `figure:caption_names_verified_episode` | PASS |  |
| `paper_summary:verified_non_markovian_scope` | PASS | verified=['xzx_radio_sub23'] |
| `benchmark_validation:valid_zero_errors` | PASS | valid=True, errors=0, warnings=0 |

## Reproducibility

```bash
conda run -n robo-dopamine python research/check_manuscript_package.py
```
