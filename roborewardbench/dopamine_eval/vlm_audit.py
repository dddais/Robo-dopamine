#!/usr/bin/env python3
"""Audit selected GroundingDINO boxes with a local Qwen3-VL model.

This is deliberately a *machine* audit rather than a renamed manual audit.
It reads the frozen ``audit_sample.jsonl`` and emits records with the same
annotation contract, but keeps them under ``vlm_audit_*`` names.  A VLM is
shown the instruction, structured target parse, and the exact selected boxes
drawn in green on the endpoint images; detector confidence and alternate
candidate boxes are intentionally withheld.

The run is resumable.  A cached decision is reused only if the grounding-box
fingerprint, prompt version, review mode, model inventory, and generation
parameters all still match.  Use ``--promote`` only after validating the VLM
on a held-out, human-labelled calibration set: it writes the compatible
``manual_audit_*`` artifacts required by the attention-mask code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .audit import (
    AUDIT_LABELS,
    grounding_result_fingerprint,
    merge_manual_annotations,
    summarize_manual_audit,
    write_manual_audit_csv,
)
from .report import read_jsonl, write_jsonl


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "counterfactual_reward1"
DEFAULT_MODEL = Path("/home/dais/workspace/model/Qwen3-VL-8B-Instruct")
PROMPT_VERSION = "qwen3_vl_grounding_audit_v1"
FAILURE_CATEGORIES = {
    "wrong_target_parse",
    "wrong_object",
    "reference_object_confusion",
    "same_category_instance_confusion",
    "object_part_confusion",
    "robot_or_gripper_confusion",
    "background_scene_box",
    "endpoint_identity_switch",
    "empty_or_severely_misaligned_box",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _model_signature(model_path: str | Path) -> dict[str, Any]:
    """A cheap local model identity that invalidates cached generations."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VLM model directory not found: {root}")
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            stat = path.stat()
            files.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"path": str(root), "inventory_sha256": _canonical_hash({"files": files}), "files": files}


def _require_current_sample(
    sample_rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Check the frozen audit list still describes the current grounding run."""

    by_id = {str(row.get("example_id")): row for row in results}
    seen: set[str] = set()
    expected_fields = ("task", "selected_parse", "steering_ready", "status", "visualization_file")
    for sample in sample_rows:
        example_id = str(sample.get("example_id", ""))
        if not example_id:
            raise ValueError("audit sample contains an empty example_id")
        if example_id in seen:
            raise ValueError(f"duplicate audit sample id: {example_id}")
        seen.add(example_id)
        result = by_id.get(example_id)
        if result is None:
            raise ValueError(f"audit sample missing from grounding results: {example_id}")
        if any(sample.get(field) != result.get(field) for field in expected_fields):
            raise ValueError(f"audit sample is stale relative to grounding result: {example_id}")
    return by_id


def _box(frame: Mapping[str, Any] | None) -> list[float] | None:
    selected = (frame or {}).get("selected")
    value = selected.get("bbox") if isinstance(selected, Mapping) else None
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _draw_selected_box(image: Image.Image, bbox: Sequence[float] | None, label: str) -> Image.Image:
    """Render the exact reviewed box, without exposing detector alternatives."""

    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    if bbox is None:
        draw.rectangle((0, 0, rendered.width - 1, rendered.height - 1), outline=(230, 40, 40), width=max(2, rendered.width // 100))
        draw.text((4, 4), f"{label}: NO SELECTED BOX", fill=(255, 50, 50), stroke_width=1, stroke_fill=(0, 0, 0))
        return rendered
    x1, y1, x2, y2 = bbox
    width = max(2, round(min(rendered.width, rendered.height) / 80))
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=width)
    draw.text((max(2, x1), max(2, y1 - 13)), f"{label}: SELECTED", fill=(0, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    return rendered


def _selected_crop(image: Image.Image, bbox: Sequence[float] | None, *, scale: float = 1.25) -> Image.Image:
    """Return a padded crop so small selected objects remain inspectable."""

    image = image.convert("RGB")
    if bbox is None:
        return image.copy()
    x1, y1, x2, y2 = bbox
    width, height = image.size
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = max((x2 - x1) * scale, width * 0.15) / 2.0
    half_height = max((y2 - y1) * scale, height * 0.15) / 2.0
    left, top = max(0, int(center_x - half_width)), max(0, int(center_y - half_height))
    right, bottom = min(width, int(center_x + half_width)), min(height, int(center_y + half_height))
    if right <= left or bottom <= top:
        return image.copy()
    crop = image.crop((left, top, right, bottom))
    # Nearest-neighbour would make tiny robotics images look blocky to a VLM.
    return crop.resize((max(224, crop.width * 4), max(224, crop.height * 4)), Image.Resampling.LANCZOS)


def _video_contact_sheet(video_path: str | Path, *, frames: int) -> Image.Image:
    """Sample the entire video uniformly into one provenance-labelled image."""

    if frames <= 0:
        raise ValueError("video frames must be positive")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("video review requires opencv-python") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        indices = [round(index * (total - 1) / max(1, frames - 1)) for index in range(frames)]
        extracted: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            extracted.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not extracted:
        raise RuntimeError(f"no frames decoded from video: {video_path}")
    tile_width, tile_height = 224, 168
    columns = min(4, len(extracted))
    rows = (len(extracted) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + 18)), "black")
    draw = ImageDraw.Draw(sheet)
    for position, frame in enumerate(extracted):
        frame.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x, y = (position % columns) * tile_width, (position // columns) * (tile_height + 18)
        sheet.paste(frame, (x + (tile_width - frame.width) // 2, y + (tile_height - frame.height) // 2))
        draw.text((x + 3, y + tile_height + 2), f"video frame {indices[position]}", fill="white")
    return sheet


def _review_images(result: Mapping[str, Any], *, review_mode: str, video_frames: int) -> tuple[list[Image.Image], str]:
    """Make image inputs.  The returned basis records exactly what was supplied."""

    before = result.get("before") or {}
    after = result.get("after") or {}
    before_path, after_path = Path(str(before.get("image_path", ""))), Path(str(after.get("image_path", "")))
    if not before_path.is_file() or not after_path.is_file():
        raise FileNotFoundError(f"missing endpoint image(s): {before_path}, {after_path}")
    with Image.open(before_path) as source:
        before_image = source.convert("RGB")
    with Image.open(after_path) as source:
        after_image = source.convert("RGB")
    before_box, after_box = _box(before), _box(after)
    images = [
        _draw_selected_box(before_image, before_box, "BEFORE"),
        _draw_selected_box(after_image, after_box, "AFTER"),
        _selected_crop(before_image, before_box),
        _selected_crop(after_image, after_box),
    ]
    basis = "endpoint_visualization"
    if review_mode == "video":
        video_path = ((result.get("frame_manifest") or {}).get("video_path"))
        if video_path:
            try:
                images.append(_video_contact_sheet(str(video_path), frames=video_frames))
                basis = "video_keyframes"
            except Exception:
                # Endpoint review remains valid; the error is recorded in the raw output.
                pass
    return images, basis


def _attach_frame_manifests(results: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frames_by_id = {str(row.get("example_id")): row for row in frames}
    return [{**dict(result), "frame_manifest": frames_by_id.get(str(result.get("example_id")))} for result in results]


def build_prompt(result: Mapping[str, Any], *, review_basis: str) -> str:
    """The evidence-only prompt.  Scores, reward labels and outcomes never enter it."""

    parse = result.get("selected_parse") or {}
    before_box, after_box = _box(result.get("before")), _box(result.get("after"))
    target_json = json.dumps(parse, ensure_ascii=False, sort_keys=True)
    return f"""You are a strict visual auditor for robot-manipulation grounding.

Decide whether the GREEN SELECTED rectangle identifies the instruction's direct manipulation target in BOTH endpoint images. The first two images are BEFORE and AFTER with the selected rectangle. The next two images are enlarged crops of those rectangles. {'A final image is a uniformly sampled contact sheet from the source video.' if review_basis == 'video_keyframes' else 'No source-video evidence is available.'}

Instruction: {result.get('task', '')}
Target parse proposed by a separate text model: {target_json}
Selected BEFORE rectangle [x1,y1,x2,y2]: {before_box}
Selected AFTER rectangle [x1,y1,x2,y2]: {after_box}

Rules:
- Determine the direct target from the instruction yourself. A destination, peg, tray, table, or reference object is usually not the target.
- The target parse is part of the claim: if it names the wrong entity, choose incorrect even if the rectangle matches that wrong phrase.
- `correct` requires both green rectangles to show the same intended entity, including required part, colour/size and same-instance identity.
- `incorrect` covers a wrong object, reference object, robot/gripper, background, wrong part, identity switch, missing box, or materially misaligned box.
- `uncertain` is required when the available images cannot establish the identity. Do not guess from typical detector behaviour.
- Ignore any visual salience outside the green selected rectangle; there are no alternate detector candidates to judge.

Return exactly one JSON object and no markdown:
{{"manual_label":"correct|incorrect|uncertain","failure_category":null|"one allowed category","reason":"specific visual evidence in one or two sentences","confidence":0.0}}
For `incorrect`, failure_category must be exactly one of: {', '.join(sorted(FAILURE_CATEGORIES))}.
For `correct` or `uncertain`, failure_category must be null. confidence must be a number from 0 to 1."""


def _extract_json(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("VLM response did not contain a JSON object")


def normalize_vlm_annotation(raw: str, *, review_basis: str) -> dict[str, Any]:
    """Validate a model response before it can enter an audit artifact."""

    value = _extract_json(raw)
    label = str(value.get("manual_label", "")).strip().lower()
    if label not in AUDIT_LABELS:
        raise ValueError(f"invalid VLM manual_label: {label!r}")
    category = value.get("failure_category")
    category = str(category).strip() if category is not None else None
    if label == "incorrect" and category not in FAILURE_CATEGORIES:
        raise ValueError(f"invalid failure_category for incorrect: {category!r}")
    if label != "incorrect":
        category = None
    reason = str(value.get("reason", "")).strip()
    if not reason:
        raise ValueError("VLM response has an empty reason")
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM confidence must be in [0, 1]")
    return {
        "manual_label": label,
        "failure_category": category,
        "reason": reason,
        "review_basis": review_basis,
        "vlm_confidence": confidence,
    }


class Qwen3VLAuditor:
    """Small transformers wrapper kept separate from pure audit utilities."""

    def __init__(self, model_path: str | Path, *, device: str, dtype: str, max_pixels: int) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = self._dtype(dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "image_processor") and max_pixels > 0:
            self.processor.image_processor.max_pixels = max_pixels
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": None}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=self.dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=self.dtype, **kwargs)
        except Exception as first_error:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if getattr(config, "model_type", "") != "qwen3_vl":
                raise
            from transformers import Qwen3VLForConditionalGeneration

            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=self.dtype, **kwargs)
            except TypeError:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype, **kwargs)
            except Exception:
                raise first_error
        self.model.eval().to(device)

    def _dtype(self, value: str):
        if value == "auto":
            return self.torch.bfloat16 if self.device.startswith("cuda") else self.torch.float32
        choices = {"bfloat16": self.torch.bfloat16, "float16": self.torch.float16, "float32": self.torch.float32}
        if value not in choices:
            raise ValueError(f"unsupported dtype: {value}")
        return choices[value]

    def generate(self, prompt: str, images: Sequence[Image.Image], *, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=list(images), return_tensors="pt", padding=True)
        moved = {}
        for key, value in inputs.items():
            if not self.torch.is_tensor(value):
                moved[key] = value
            elif key.startswith("pixel_values"):
                moved[key] = value.to(self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**moved, do_sample=False, max_new_tokens=max_new_tokens)
        prompt_length = moved["input_ids"].shape[1]
        return self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()


def _cache_matches(row: Mapping[str, Any], *, fingerprint: str, signature: str, prompt_hash: str) -> bool:
    return bool(
        row.get("complete")
        and not row.get("error")
        and str(row.get("grounding_fingerprint")) == fingerprint
        and str(row.get("model_signature")) == signature
        and str(row.get("prompt_hash")) == prompt_hash
        and isinstance(row.get("annotation"), Mapping)
    )


def _error_annotation(error: Exception, *, review_basis: str) -> dict[str, Any]:
    return {
        "manual_label": "uncertain",
        "failure_category": None,
        "reason": f"VLM audit could not produce a validated visual decision: {type(error).__name__}: {error}",
        "review_basis": review_basis,
        "vlm_confidence": 0.0,
    }


def vlm_audit_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, model_path: str) -> str:
    failures = [row for row in rows if row.get("manual_label") == "incorrect"]
    uncertainty = [row for row in rows if row.get("manual_label") == "uncertain"]
    counts = Counter(str(row.get("review_basis")) for row in rows)
    lines = [
        "# Grounding VLM 审计报告",
        "",
        "## 方法与边界",
        "",
        f"- 审核模型：`{model_path}`；提示词版本：`{PROMPT_VERSION}`。",
        "- 模型仅接收 instruction、target parse、首末原图及绿色 selected bbox（加上可选的视频均匀采样联系表）；不接收 reward、GRM 分数、GroundingDINO score 或候选框。",
        "- 这是自动视觉审查，不是人工 ground truth。应先在独立人工标注的校准集报告 VLM-人工一致性，再将其用于正式筛选。",
        f"- 审查依据：{dict(sorted(counts.items()))}。`video_keyframes` 是均匀抽帧，不等同于人工逐帧观看完整视频。",
        "",
        "## 结果",
        "",
        f"- 样本数：{summary['audit_sample_size']}；correct：{summary['overall']['correct']}；incorrect：{summary['overall']['incorrect']}；uncertain：{summary['overall']['uncertain']}。",
        f"- steering-ready：correct {summary['steering_ready']['correct']} / evaluated {summary['steering_ready']['evaluated']}。",
        "",
        "## 自动判为错误的样本",
        "",
        "| index | instruction | category | VLM evidence |",
        "|---:|---|---|---|",
    ]
    for row in failures:
        task = str(row.get("task", "")).replace("|", "\\|")
        reason = str(row.get("reason", "")).replace("|", "\\|")
        lines.append(f"| {row.get('index')} | {task} | `{row.get('failure_category')}` | {reason} |")
    if not failures:
        lines.append("| — | — | — | No incorrect decisions |")
    lines.extend(["", "## 不确定样本", ""])
    for row in uncertainty:
        lines.append(f"- `{row.get('example_id')}`: {row.get('reason')}")
    if not uncertainty:
        lines.append("- 无。")
    return "\n".join(lines) + "\n"


def run_vlm_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir).expanduser().resolve()
    sample = read_jsonl(root / "audit_sample.jsonl")
    results = _attach_frame_manifests(read_jsonl(root / "grounding_results.jsonl"), read_jsonl(root / "frame_manifest.jsonl"))
    result_by_id = _require_current_sample(sample, results)
    signature = _model_signature(args.model_path)
    parameters = {
        "prompt_version": PROMPT_VERSION,
        "review_mode": args.review_mode,
        "video_frames": args.video_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "dtype": args.dtype,
    }
    signature_hash = _canonical_hash({"model": signature["inventory_sha256"], "parameters": parameters})
    raw_path = root / "vlm_audit_raw.jsonl"
    existing = {str(row.get("example_id")): row for row in read_jsonl(raw_path)}
    pending = list(sample[: args.max_samples or None])
    auditor: Qwen3VLAuditor | None = None
    output: list[dict[str, Any]] = []
    for position, sample_row in enumerate(pending, 1):
        example_id = str(sample_row["example_id"])
        result = result_by_id[example_id]
        fingerprint = grounding_result_fingerprint(result)
        # The prompt hash deliberately includes box fields via the result fingerprint.
        prompt_hash = _canonical_hash({"fingerprint": fingerprint, "parameters": parameters})
        cached = existing.get(example_id)
        if _cache_matches(cached or {}, fingerprint=fingerprint, signature=signature_hash, prompt_hash=prompt_hash):
            output.append(dict(cached))
            print(f"[vlm-audit] cached {position}/{len(pending)} {example_id}", flush=True)
            continue
        review_basis = "endpoint_visualization"
        try:
            images, review_basis = _review_images(result, review_mode=args.review_mode, video_frames=args.video_frames)
            prompt = build_prompt(result, review_basis=review_basis)
            if auditor is None:
                print(f"[vlm-audit] loading {args.model_path} on {args.device}", flush=True)
                auditor = Qwen3VLAuditor(args.model_path, device=args.device, dtype=args.dtype, max_pixels=args.max_pixels)
            raw_response = auditor.generate(prompt, images, max_new_tokens=args.max_new_tokens)
            annotation = normalize_vlm_annotation(raw_response, review_basis=review_basis)
            error = None
        except Exception as exc:  # Keep the sample auditable and explicit rather than dropping it.
            raw_response = None
            annotation = _error_annotation(exc, review_basis=review_basis)
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "example_id": example_id,
            "grounding_fingerprint": fingerprint,
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "model_signature": signature_hash,
            "prompt_hash": prompt_hash,
            "parameters": parameters,
            "reviewed_at": _utc_now(),
            "raw_response": raw_response,
            "annotation": annotation,
            "error": error,
            "complete": True,
        }
        existing[example_id] = record
        output.append(record)
        # Persist each completed item so interruption does not lose expensive work.
        write_jsonl(raw_path, [existing[str(row["example_id"])] for row in pending if str(row["example_id"]) in existing])
        print(f"[vlm-audit] {position}/{len(pending)} {example_id}: {annotation['manual_label']}", flush=True)
    if len(pending) != len(sample):
        # This mode is intentionally useful for confirming model loading and
        # prompt compatibility, but never fabricates a complete audit report.
        return {
            "status": "partial_raw_cache_only",
            "processed": len(pending),
            "expected": len(sample),
            "raw_cache": str(raw_path),
            "message": "No vlm_audit annotations or summaries were written; rerun without --max-samples.",
        }

    annotations = [{"example_id": row["example_id"], **dict(row["annotation"]), "audit_source": "qwen3_vl", "vlm_model_path": row["model_path"]} for row in output]
    merged = merge_manual_annotations(sample, annotations)
    for row, raw in zip(merged, output):
        row["grounding_fingerprint"] = raw["grounding_fingerprint"]
        row["audit_source"] = "qwen3_vl"
        row["vlm_model_path"] = raw["model_path"]
        row["vlm_model_signature"] = raw["model_signature"]
        row["vlm_prompt_version"] = PROMPT_VERSION
        row["vlm_error"] = raw["error"]
    summary = summarize_manual_audit(merged, population_total=len(results), population_steering_ready=sum(bool(row.get("steering_ready")) for row in results))
    summary.update({"audit_source": "qwen3_vl", "model": signature, "parameters": parameters, "errors": sum(bool(row.get("vlm_error")) for row in merged)})
    write_jsonl(root / "vlm_audit_annotations.jsonl", annotations)
    write_jsonl(root / "vlm_audit.jsonl", merged)
    write_manual_audit_csv(merged, root / "vlm_audit.csv")
    (root / "vlm_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (root / "vlm_audit_report.md").write_text(vlm_audit_markdown(summary, merged, model_path=str(Path(args.model_path).expanduser().resolve())), encoding="utf-8")
    if args.promote:
        # Explicit opt-in: downstream code only knows the historical manual contract.
        from .audit import build_audit_artifacts

        write_jsonl(root / "manual_audit_annotations.jsonl", annotations)
        build_audit_artifacts(root)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL), help="Qwen3-VL-8B by default; Qwen3-VL-4B is supported for smoke tests.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--review-mode", choices=("endpoints", "video"), default="video")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0, help="Only for connectivity testing; final artifacts require the full sample.")
    parser.add_argument("--promote", action="store_true", help="Explicitly write VLM labels into manual_audit_* compatibility artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.video_frames <= 0 or args.max_new_tokens <= 0 or args.max_pixels <= 0 or args.max_samples < 0:
        raise ValueError("frame, token, pixel, and sample limits must be positive")
    summary = run_vlm_audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
