# Robo-Dopamine instruction grounding

This directory contains an offline, resumable grounding pipeline for
RoboRewardBench videos:

```text
instruction
  -> Qwen3-4B structured target parse
  -> Qwen2.5-7B independent consistency parse
  -> target-only GroundingDINO query variants
  -> before/after endpoint bboxes
  -> confidence, endpoint-consistency, visualization, and summary artifacts
```

The model inputs never contain `reward` or `gpt5_mini_check`.  Those fields are
also excluded from parse/detection output records.  Runtime uses only the video,
instruction, and file identity.

人工审核每条 grounding 的判据、要看的文件、annotation JSONL 格式和 stale-fingerprint
处理见 [`MANUAL_AUDIT.md`](MANUAL_AUDIT.md)。自动 `steering_ready` 不能替代这一步。

本地 Qwen3-VL 自动审查流水线见 [`VLM_AUDIT.md`](VLM_AUDIT.md)。它对同一冻结 sample
生成独立的 `vlm_audit_*` 产物（默认不会覆盖人工结果）；Qwen3-VL-8B 用于正式运行，4B 可用于
smoke test。只有在独立人工校准后才使用该命令的 `--promote` 将机器标签接入旧的
`manual_audit.jsonl` 下游契约。

## Default full run

From the Robo-Dopamine repository root:

```bash
conda run -n robo-dopamine python -m roborewardbench.dopamine_eval.pipeline all \
  --dataset-root /home/dais/workspace/data/RoboRewardBench_counterfactual_reward1 \
  --primary-llm /home/dais/workspace/model/Qwen3-4B-Instruct-2507 \
  --secondary-llm /home/dais/workspace/model/Qwen2.5-7B-Instruct \
  --grounding-model /home/dais/workspace/model/grounding-dino-base \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1
```

The command is resumable.  Existing per-model parse and per-frame detector
records are skipped only when their model, task/query, frame file, and inference
parameter signatures match.  Changing any of those inputs invalidates the
corresponding cache instead of silently mixing old results with a new manifest.
Merged JSONL/CSV/summary files are regenerated deterministically.

For a smoke test, use a separate output directory:

```bash
conda run -n robo-dopamine python -m roborewardbench.dopamine_eval.pipeline all \
  --max-samples 4 \
  --output-dir roborewardbench/dopamine_eval/outputs/smoke
```

Stages can also be run separately: `parse`, `ground`, and `summarize`.

## Output contract

- `run_manifest.json`: command, model inventory, data hash, environment, status.
- `raw_parses/*.jsonl`: raw/validated output from each local LLM.
- `instruction_parses.jsonl`: agreement diagnostics and selected target parse.
- `frame_manifest.jsonl`: exact decoded endpoint frame indices.
- `frames/`: PNG endpoint frames, matching the forward benchmark protocol.
- `grounding_frames.jsonl`: GroundingDINO top-k candidates per endpoint.
- `grounding_results.jsonl` / `.csv`: paired per-example result.
- `visualizations/`: before/after boxes; green is selected, orange is alternate.
- `contact_sheets/`: subset-grouped visual audit sheets.
- `summary.json` / `.md`: descriptive coverage and consistency metrics.
- `audit_sample.jsonl` / `audit_sheets/`: frozen subset-stratified human-review sample.
- `manual_audit_annotations.jsonl`: one human label, failure category, and reason per sampled example.
- `audit_grounding_fingerprints.jsonl`: immutable link from those labels to the exact selected boxes that were reviewed.
- `manual_audit.jsonl` / `.csv`, `manual_audit_summary.json`, `audit_report.md`: validated audit records and descriptive correct-object precision.

`accepted` means the selected GroundingDINO score is at least 0.25 by default.
It does **not** mean the box is known to be the correct object.  Endpoint
consistency is also a proxy, not ground truth.  Correct-object precision must be
measured on a manually annotated/audited subset before steering conclusions are
reported.

After reviewing every row in `audit_sample.jsonl`, rebuild the checked audit
artifacts with:

```bash
python -m roborewardbench.dopamine_eval.audit \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1
```

The first successful audit command freezes selected-box fingerprints.  Later
runs fail on missing, duplicate, unexpected, invalid, or stale labels, so a
partial review—or labels left over after a grounding change—cannot silently
appear complete.  Delete/recreate the fingerprint file only when intentionally
starting a new human review.

## Tests

Pure utility tests do not load model weights:

```bash
python -m unittest tests.test_dopamine_eval_grounding -v
```
