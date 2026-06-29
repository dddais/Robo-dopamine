# Radio Memory-GRM Case Figure

This figure is generated from cached Robo-Dopamine GRM outputs and human-verified radio keyframe events. It does not rerun GRM and does not add any benchmark labels.

- Episode: `xzx_radio_sub23`
- Fused peak progress: 71.43% at frame 450 (15.0s)
- Fused final progress: 34.17%
- Final-only GRM decision: not success under the 70% threshold.
- Event-Latched GRM decision: success because `button_press` and `indicator_green` are latched.

Outputs:

- PNG: `research_outputs/figures/radio_memory_grm_case.png`
- PDF: `research_outputs/figures/radio_memory_grm_case.pdf`
- LaTeX snippet: `research_outputs/figures/radio_memory_grm_case.tex`
- JSON: `research_outputs/figures/radio_memory_grm_case.json`

Suggested caption:

Figure: Hidden-intermediate-event radio case. GRM progress peaks while the radio is being manipulated, but the fused final progress falls to 34.17% after the radio is put down. Event memory latches the human-verified button press and green-indicator events, preserving the success evidence through the final frame.

Limitation: this is one human-verified non-Markovian episode, not a completed large-scale benchmark.
