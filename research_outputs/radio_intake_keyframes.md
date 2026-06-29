# Radio Intake Keyframes / Radio Pending-Candidate Keyframes

This report indexes contact sheets for pending radio-like candidates. These files are review material only and do not add benchmark labels.

Current live Benchmark v0 still contains only `xzx_radio_sub23`. The candidates below must be human-verified for `button_press` and `indicator_green` before any benchmark inclusion.

## Summary

- Extracted candidates: `4`
- Adds benchmark labels: `False`

## Extracted Candidates

| Candidate | Video frames | Sampled frames | Contact sheet | Manifest |
|---|---:|---:|---|---|
| `xzx_episode_1` | 1956 | 14 | `research_outputs/radio_intake_keyframes/xzx_episode_1/contact_sheet.png` | `research_outputs/radio_intake_keyframes/xzx_episode_1/manifest.json` |
| `xzx_episode_1_sub1` | 750 | 14 | `research_outputs/radio_intake_keyframes/xzx_episode_1_sub1/contact_sheet.png` | `research_outputs/radio_intake_keyframes/xzx_episode_1_sub1/manifest.json` |
| `xzx_episode_1_sub2` | 650 | 14 | `research_outputs/radio_intake_keyframes/xzx_episode_1_sub2/contact_sheet.png` | `research_outputs/radio_intake_keyframes/xzx_episode_1_sub2/manifest.json` |
| `xzx_episode_2` | 2340 | 14 | `research_outputs/radio_intake_keyframes/xzx_episode_2/contact_sheet.png` | `research_outputs/radio_intake_keyframes/xzx_episode_2/manifest.json` |

## Manual Review Checklist

- Locate candidate `grasp`, `lift`, `button_press`, `indicator_green`, `place`, and `release` events.
- Record exact frame ids and evidence views before setting a success label.
- Mark the candidate as non-Markovian only if the task label depends on hidden intermediate events or event order.
- Keep the candidate outside `benchmark_v0/episodes.json` until event labels and cached GRM paths are complete.

## Reproducibility

```bash
conda run -n robo-dopamine python research/extract_radio_intake_keyframes.py
```
