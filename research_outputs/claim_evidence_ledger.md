# Claim/Evidence Ledger / 论文主张与证据台账

This ledger records what the current Memory-GRM package can and cannot claim. It does not add benchmark labels or rerun GRM inference.

## Data Scope

- Valid: `True`
- Live benchmark episode ids: `xzx_radio_sub23`
- Verified non-Markovian episodes: `xzx_radio_sub23`
- Adds benchmark labels: `False`

## Supported Claims

| ID | Status | Claim | Evidence | Scope limit |
|---|---|---|---|---|
| `C1` | `supported_visible_state_only` | Robo-Dopamine GRM-2.0-8B is a strong visible-state progress baseline on cached local runs. | Fused final interval 10 AUROC 98.4%, best F1 0.857, accuracy 97.8%.<br>Fused final interval 20 AUROC 100.0%, best F1 1.000, accuracy 100.0%. | Cached visible-state diagnostic data, not non-Markovian benchmark evidence. |
| `C2` | `supported_one_case_feasibility` | The verified turn-on-radio episode demonstrates hidden intermediate success evidence. | Verified non-Markovian episode list: ['xzx_radio_sub23'].<br>Fused final progress 34.17%, fused peak progress 71.43%.<br>Human-verified event labels include button_press frame 615 and indicator_green frame 630. | One real verified episode only. |
| `C3` | `supported_logic_stress_test` | Score-only memory cannot represent missing, out-of-order, or forbidden event predicates when scalar score evidence is identical. | Radio counterfactual final-only balanced accuracy 50.0%.<br>Radio counterfactual score-memory balanced accuracy 50.0%.<br>All counterfactual variants reuse the same GRM score curve and differ only in event labels. | Synthetic label variants over one trajectory. |
| `C4` | `supported_logic_stress_test` | Event-Latched GRM separates valid and invalid radio event histories under identical scalar score evidence. | Event-Latched GRM counterfactual balanced accuracy 100.0%.<br>Current Benchmark v0 Event-Latched GRM accuracy 100.0% on the single labeled episode. | Does not prove automatic event detection or statistical benchmark superiority. |
| `C5` | `supported_guardrail` | Benchmark v0 currently has exactly one verified non-Markovian episode and the repository enforces that boundary. | Live benchmark ids: ['xzx_radio_sub23'].<br>Benchmark validation valid=True, errors=0, warnings=0. | The benchmark is not yet large enough for statistical claims. |

## Unsupported Claims

| ID | Do not claim | Reason | Required next evidence |
|---|---|---|---|
| `U1` | A completed large-scale non-Markovian robot benchmark. | Only one human-verified non-Markovian episode is live. | Additional human-verified turn-on-radio-like positive/negative histories, preferably final-state-similar pairs. |
| `U2` | Automatic event detection. | Current radio events are human verified from keyframes; no event detector is evaluated. | VLM/event-head predictions compared against human event labels. |
| `U3` | Carrot/cube/bottle cached cases are non-Markovian benchmark labels. | They are baseline diagnostics and candidate inspection material only. | Manual proof of final-state-similar histories where hidden events change the label. |
| `U4` | Statistical superiority of Memory-GRM on non-Markovian robot tasks. | The current real non-Markovian benchmark has one labeled episode. | A larger labeled Benchmark v0 with matched positive/negative histories and confidence intervals. |

## Reproducibility

```bash
conda run -n robo-dopamine python research/make_claim_evidence_ledger.py
```
