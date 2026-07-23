"""Artifacts, visualizations, and descriptive summaries for grounding runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {source}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(destination)


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        handle.flush()


def _safe_mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _selected_score(frame: Mapping[str, Any] | None) -> float | None:
    if not frame or not frame.get("selected"):
        return None
    return float(frame["selected"]["score"])


def summarize_run(
    parses: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parse_by_id = {str(row["example_id"]): row for row in parses}
    result_by_id = {str(row["example_id"]): row for row in results}
    ids = sorted(set(parse_by_id) | set(result_by_id))
    parse_rows = [parse_by_id[example_id] for example_id in ids if example_id in parse_by_id]
    result_rows = [result_by_id[example_id] for example_id in ids if example_id in result_by_id]

    agreement_counts = Counter(str(row.get("agreement_level", "unavailable")) for row in parse_rows)
    selected_source_counts = Counter(str(row.get("selected_source", "none")) for row in parse_rows)
    target_type_counts = Counter(
        str((row.get("selected_parse") or {}).get("target_type", "missing")) for row in parse_rows
    )
    primary_valid = sum(bool(row.get("primary", {}).get("valid")) for row in parse_rows)
    secondary_valid = sum(bool(row.get("secondary", {}).get("valid")) for row in parse_rows)
    ambiguous = sum(bool((row.get("selected_parse") or {}).get("ambiguous")) for row in parse_rows)

    frame_roles = ("before", "after")
    frame_summary: dict[str, Any] = {}
    for role in frame_roles:
        frames = [row.get(role) for row in result_rows if row.get(role) is not None]
        scores = [score for frame in frames if (score := _selected_score(frame)) is not None]
        detected = sum(bool(frame.get("detected")) for frame in frames)
        accepted = sum(bool(frame.get("accepted")) for frame in frames)
        frame_summary[role] = {
            "available": len(frames),
            "detected": detected,
            "detected_rate": _fraction(detected, len(frames)),
            "accepted": accepted,
            "accepted_rate": _fraction(accepted, len(frames)),
            "score_mean": _safe_mean(scores),
            "score_median": _safe_median(scores),
            "coverage_at_score": {
                str(threshold): _fraction(sum(score >= threshold for score in scores), len(frames))
                for threshold in (0.20, 0.25, 0.30, 0.35)
            },
        }

    both_detected = 0
    both_accepted = 0
    consistency_available = 0
    consistent = 0
    steering_ready = 0
    joint_selection_changed = 0
    ious: list[float] = []
    center_distances: list[float] = []
    statuses = Counter()
    by_subset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in result_rows:
        before = row.get("before")
        after = row.get("after")
        before_detected = bool(before and before.get("detected"))
        after_detected = bool(after and after.get("detected"))
        before_accepted = bool(before and before.get("accepted"))
        after_accepted = bool(after and after.get("accepted"))
        both_detected += before_detected and after_detected
        both_accepted += before_accepted and after_accepted
        status = str(row.get("status", "unknown"))
        statuses[status] += 1
        steering_ready += int(bool(row.get("steering_ready")))
        if before and before.get("raw_selected") and before.get("selected"):
            joint_selection_changed += int(before["raw_selected"].get("bbox") != before["selected"].get("bbox"))
        subset = str(row.get("subset", "unknown"))
        by_subset[subset]["total"] += 1
        by_subset[subset]["before_accepted"] += int(before_accepted)
        by_subset[subset]["after_accepted"] += int(after_accepted)
        by_subset[subset]["both_accepted"] += int(before_accepted and after_accepted)
        pair = row.get("pair_consistency") or {}
        if pair.get("available"):
            consistency_available += 1
            consistent += bool(pair.get("consistent"))
            ious.append(float(pair["iou"]))
            center_distances.append(float(pair["center_distance"]))
            by_subset[subset]["pair_available"] += 1
            by_subset[subset]["pair_consistent"] += int(bool(pair.get("consistent")))

    subset_summary: dict[str, Any] = {}
    for subset, counts in sorted(by_subset.items()):
        total = counts["total"]
        pair_available = counts["pair_available"]
        subset_summary[subset] = {
            "total": total,
            "before_accepted": counts["before_accepted"],
            "after_accepted": counts["after_accepted"],
            "both_accepted": counts["both_accepted"],
            "both_accepted_rate": _fraction(counts["both_accepted"], total),
            "pair_consistent": counts["pair_consistent"],
            "pair_available": pair_available,
            "pair_consistency_rate": _fraction(counts["pair_consistent"], pair_available),
        }

    return {
        "num_parse_records": len(parse_rows),
        "num_grounding_records": len(result_rows),
        "identity_match": set(parse_by_id) == set(result_by_id),
        "missing_grounding_ids": sorted(set(parse_by_id) - set(result_by_id)),
        "unexpected_grounding_ids": sorted(set(result_by_id) - set(parse_by_id)),
        "instruction_parsing": {
            "primary_valid": primary_valid,
            "primary_valid_rate": _fraction(primary_valid, len(parse_rows)),
            "secondary_valid": secondary_valid,
            "secondary_valid_rate": _fraction(secondary_valid, len(parse_rows)),
            "agreement_counts": dict(sorted(agreement_counts.items())),
            "selected_source_counts": dict(sorted(selected_source_counts.items())),
            "target_type_counts": dict(sorted(target_type_counts.items())),
            "ambiguous": ambiguous,
        },
        "grounding": {
            "frames": frame_summary,
            "both_detected": both_detected,
            "both_detected_rate": _fraction(both_detected, len(result_rows)),
            "both_accepted": both_accepted,
            "both_accepted_rate": _fraction(both_accepted, len(result_rows)),
            "pair_consistency_available": consistency_available,
            "pair_consistent": consistent,
            "pair_consistency_rate": _fraction(consistent, consistency_available),
            "steering_ready": steering_ready,
            "steering_ready_rate": _fraction(steering_ready, len(result_rows)),
            "joint_selection_changed_before_top1": joint_selection_changed,
            "pair_iou_mean": _safe_mean(ious),
            "pair_iou_median": _safe_median(ious),
            "center_distance_mean": _safe_mean(center_distances),
            "status_counts": dict(sorted(statuses.items())),
        },
        "by_subset": subset_summary,
        "interpretation_boundary": (
            "Detection confidence and endpoint consistency are quality proxies, not bbox accuracy. "
            "Correct-object precision requires a human or independently annotated bbox audit."
        ),
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _number(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def summary_markdown(summary: Mapping[str, Any], *, command: str, output_dir: str) -> str:
    parsing = summary["instruction_parsing"]
    grounding = summary["grounding"]
    before = grounding["frames"]["before"]
    after = grounding["frames"]["after"]
    agreement = parsing["agreement_counts"]
    exact_or_compatible = int(agreement.get("exact", 0)) + int(agreement.get("compatible", 0))
    total = int(summary["num_parse_records"])
    lines = [
        "# LLM + GroundingDINO grounding 结果",
        "",
        "## Material Passport",
        "",
        f"- 样本数：{total}",
        f"- 输出目录：`{output_dir}`",
        "- 数据使用边界：运行时仅读取 `file_name` 与 `task`；未向解析器或检测器提供 `reward` 或 `gpt5_mini_check`。",
        f"- 运行命令：`{command}`",
        "",
        "## 结果概览",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Qwen3 有效 JSON | {parsing['primary_valid']}/{total} ({_percent(parsing['primary_valid_rate'])}) |",
        f"| Qwen2.5 有效 JSON | {parsing['secondary_valid']}/{total} ({_percent(parsing['secondary_valid_rate'])}) |",
        f"| 双模型 exact/compatible | {exact_or_compatible}/{total} ({_percent(_fraction(exact_or_compatible, total))}) |",
        f"| 首帧存在候选框 | {before['detected']}/{before['available']} ({_percent(before['detected_rate'])}) |",
        f"| 终帧存在候选框 | {after['detected']}/{after['available']} ({_percent(after['detected_rate'])}) |",
        f"| 首帧通过置信度门槛 | {before['accepted']}/{before['available']} ({_percent(before['accepted_rate'])}) |",
        f"| 终帧通过置信度门槛 | {after['accepted']}/{after['available']} ({_percent(after['accepted_rate'])}) |",
        f"| 首末帧均通过门槛 | {grounding['both_accepted']}/{summary['num_grounding_records']} ({_percent(grounding['both_accepted_rate'])}) |",
        f"| 双帧检测时序一致 | {grounding['pair_consistent']}/{grounding['pair_consistency_available']} ({_percent(grounding['pair_consistency_rate'])}) |",
        f"| 可进入 steering（双帧高置信且一致） | {grounding['steering_ready']}/{summary['num_grounding_records']} ({_percent(grounding['steering_ready_rate'])}) |",
        f"| 双帧 bbox IoU（均值/中位数） | {_number(grounding['pair_iou_mean'])} / {_number(grounding['pair_iou_median'])} |",
    ]
    manual_audit = summary.get("manual_audit")
    if isinstance(manual_audit, Mapping):
        audited_overall = manual_audit.get("overall") or {}
        audited_ready = manual_audit.get("steering_ready") or {}
        audited_nonready = manual_audit.get("not_steering_ready") or {}
        lines.extend(
            [
                "",
                "## 人工 correct-object 审计",
                "",
                "| 审计组 | 正确/有效审计数 | observed precision |",
                "|---|---:|---:|",
                f"| 全部审计样本 | {audited_overall.get('correct')}/{audited_overall.get('evaluated')} | {_percent(audited_overall.get('observed_correct_object_precision'))} |",
                f"| steering-ready | {audited_ready.get('correct')}/{audited_ready.get('evaluated')} | {_percent(audited_ready.get('observed_correct_object_precision'))} |",
                f"| 非 steering-ready | {audited_nonready.get('correct')}/{audited_nonready.get('evaluated')} | {_percent(audited_nonready.get('observed_correct_object_precision'))} |",
                "",
                "该审计覆盖全部 21 个来源子集，但属于单人、分层目的性抽样，不是简单随机样本；比例不能作为 228 条全体的无偏准确率。完整逐条判断见 `audit_report.md`。",
            ]
        )
    lines.extend(
        [
        "",
        "## 解析分布",
        "",
        f"- agreement：`{json.dumps(parsing['agreement_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- selected source：`{json.dumps(parsing['selected_source_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- target type：`{json.dumps(parsing['target_type_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- ambiguous：{parsing['ambiguous']}",
        "",
        "## Grounding 状态",
        "",
        f"- status：`{json.dumps(grounding['status_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- 首帧 score 均值/中位数：{_number(before['score_mean'])} / {_number(before['score_median'])}",
        f"- 终帧 score 均值/中位数：{_number(after['score_mean'])} / {_number(after['score_median'])}",
        "",
        "## 解释限制",
        "",
        summary["interpretation_boundary"],
        "置信度覆盖率和首末帧一致性不能证明框中了正确物体；正式的 correct-object precision 需要人工 bbox audit。",
        "",
        ]
    )
    return "\n".join(lines)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_candidates(
    image: Image.Image,
    frame: Mapping[str, Any] | None,
    *,
    accepted_color: tuple[int, int, int] = (0, 220, 0),
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    if not frame:
        return canvas
    draw = ImageDraw.Draw(canvas)
    candidates = frame.get("candidates") or []
    selected = frame.get("selected")
    for candidate in reversed(candidates[:5]):
        box = [float(value) for value in candidate["bbox"]]
        draw.rectangle(box, outline=(255, 165, 0), width=2)
    if selected:
        box = [float(value) for value in selected["bbox"]]
        draw.rectangle(box, outline=accepted_color, width=4)
        label = f"{selected.get('label', '')} {float(selected.get('score', 0.0)):.3f}"
        draw.text((box[0] + 3, max(2, box[1] - 22)), label, fill=accepted_color, font=_load_font(16))
    return canvas


def render_sample_visualization(result: Mapping[str, Any], destination: str | Path) -> None:
    before_frame = result.get("before")
    after_frame = result.get("after")
    if not before_frame or not after_frame:
        return
    with Image.open(before_frame["image_path"]) as before_image, Image.open(after_frame["image_path"]) as after_image:
        before = _draw_candidates(before_image, before_frame)
        after = _draw_candidates(after_image, after_frame)
    target_height = 360
    if before.height != target_height:
        before = before.resize(
            (round(before.width * target_height / before.height), target_height),
            Image.Resampling.LANCZOS,
        )
    if after.height != target_height:
        after = after.resize(
            (round(after.width * target_height / after.height), target_height),
            Image.Resampling.LANCZOS,
        )
    header_height = 115
    canvas = Image.new("RGB", (before.width + after.width, target_height + header_height), "white")
    canvas.paste(before, (0, header_height))
    canvas.paste(after, (before.width, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    small_font = _load_font(15)
    instruction = str(result.get("task", ""))
    target = str((result.get("selected_parse") or {}).get("target_phrase", ""))
    queries = ", ".join((before_frame or {}).get("queries", []))
    pair = result.get("pair_consistency") or {}
    lines = [
        f"Instruction: {instruction}",
        f"Target: {target} | Queries: {queries}",
        f"Parse agreement: {result.get('agreement_level')} | Pair: {pair.get('consistent')} | IoU: {pair.get('iou')}",
    ]
    y = 6
    for index, line in enumerate(lines):
        wrapped = textwrap.wrap(line, width=135) or [""]
        for piece in wrapped[:2]:
            draw.text((8, y), piece, fill="black", font=title_font if index == 0 else small_font)
            y += 24 if index == 0 else 20
    draw.text((8, header_height + 4), "BEFORE", fill=(255, 255, 255), stroke_width=2, stroke_fill="black", font=title_font)
    draw.text((before.width + 8, header_height + 4), "AFTER", fill=(255, 255, 255), stroke_width=2, stroke_fill="black", font=title_font)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def create_contact_sheets(
    results: Sequence[Mapping[str, Any]],
    visualization_dir: str | Path,
    output_dir: str | Path,
    *,
    rows_per_sheet: int = 4,
) -> list[Path]:
    """Create subset-grouped sheets with two samples per row."""

    visualization_dir = Path(visualization_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("subset", "unknown"))].append(result)
    written: list[Path] = []
    cell_width, cell_height = 640, 250
    per_sheet = rows_per_sheet * 2
    font = _load_font(20)
    for subset, subset_rows in sorted(grouped.items()):
        for page_index in range(0, len(subset_rows), per_sheet):
            page_rows = subset_rows[page_index : page_index + per_sheet]
            canvas = Image.new("RGB", (cell_width * 2, cell_height * rows_per_sheet + 34), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"{subset} | page {page_index // per_sheet + 1}", fill="black", font=font)
            for position, result in enumerate(page_rows):
                name = str(result["visualization_file"])
                source = visualization_dir / name
                if not source.is_file():
                    continue
                with Image.open(source) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                    x = (position % 2) * cell_width
                    y = 34 + (position // 2) * cell_height
                    canvas.paste(thumb, (x, y))
            safe_subset = "".join(char if char.isalnum() or char in "-_" else "_" for char in subset)
            destination = output_dir / f"{safe_subset}_{page_index // per_sheet + 1:02d}.jpg"
            canvas.save(destination, quality=90)
            written.append(destination)
    return written


def write_flat_csv(results: Sequence[Mapping[str, Any]], destination: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "example_id": result.get("example_id"),
            "subset": result.get("subset"),
            "task": result.get("task"),
            "target_phrase": (result.get("selected_parse") or {}).get("target_phrase"),
            "target_head": (result.get("selected_parse") or {}).get("target_head"),
            "target_type": (result.get("selected_parse") or {}).get("target_type"),
            "agreement_level": result.get("agreement_level"),
            "status": result.get("status"),
            "steering_ready": result.get("steering_ready"),
            "visualization_file": result.get("visualization_file"),
        }
        for role in ("before", "after"):
            frame = result.get(role) or {}
            selected = frame.get("selected") or {}
            row[f"{role}_detected"] = frame.get("detected")
            row[f"{role}_accepted"] = frame.get("accepted")
            row[f"{role}_score"] = selected.get("score")
            row[f"{role}_label"] = selected.get("label")
            bbox = selected.get("bbox") or [None] * 4
            for coordinate, value in zip(("x1", "y1", "x2", "y2"), bbox):
                row[f"{role}_{coordinate}"] = value
        pair = result.get("pair_consistency") or {}
        row["pair_available"] = pair.get("available")
        row["pair_consistent"] = pair.get("consistent")
        row["pair_iou"] = pair.get("iou")
        row["pair_center_distance"] = pair.get("center_distance")
        rows.append(row)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
