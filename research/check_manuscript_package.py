#!/usr/bin/env python3
# Check the current Memory-GRM manuscript package for paper-sharing readiness.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_MD = ROOT / "research_outputs/manuscript_package_check.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/manuscript_package_check.json"

REQUIRED_FILES = [
    "manuscript/main.tex",
    "manuscript/references.bib",
    "manuscript/README.md",
    "research_outputs/paper_tables.tex",
    "research_outputs/figures/radio_memory_grm_case.tex",
    "research_outputs/figures/radio_memory_grm_case.pdf",
    "research_outputs/paper_ready_results_summary.json",
    "research_outputs/benchmark_v0_validation.json",
]

REQUIRED_BIB_KEYS = {
    "robo_dopamine",
    "robo_reward",
    "large_reward_models",
    "aha_failure",
    "i_failsense",
    "memoryvla",
    "map_vla",
    "remem_vla",
    "optimus_vla",
    "reward_machine_inference",
}

REQUIRED_TABLE_LABELS = {
    "tab:visible_state_grm",
    "tab:score_memory",
    "tab:benchmark_v0_radio",
    "tab:radio_counterfactuals",
}

REQUIRED_MAIN_SNIPPETS = {
    "generated_figure_input": r"\input{research_outputs/figures/radio_memory_grm_case.tex}",
    "generated_table_input": r"\input{research_outputs/paper_tables.tex}",
    "bibliography_path": r"\bibliography{manuscript/references}",
    "one_case_language": "one human-verified non-Markovian episode",
    "feasibility_language": "feasibility study",
    "radio_episode_id": r"xzx\_radio\_sub23",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the current Memory-GRM manuscript package.")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def extract_cite_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\w*\s*\{([^}]+)\}", text):
        keys.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return keys


def add_check(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def build_payload() -> dict:
    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    for rel_path in REQUIRED_FILES:
        path = ROOT / rel_path
        ok = path.exists() and path.stat().st_size > 0
        size = path.stat().st_size if path.exists() else None
        add_check(checks, f"required_file:{rel_path}", ok, f"size={size}")
        if not ok:
            errors.append(f"missing or empty required file: {rel_path}")

    main_path = ROOT / "manuscript/main.tex"
    bib_path = ROOT / "manuscript/references.bib"
    tables_path = ROOT / "research_outputs/paper_tables.tex"
    figure_tex_path = ROOT / "research_outputs/figures/radio_memory_grm_case.tex"
    summary_json_path = ROOT / "research_outputs/paper_ready_results_summary.json"
    validation_json_path = ROOT / "research_outputs/benchmark_v0_validation.json"

    main_text = load_text(main_path) if main_path.exists() else ""
    for name, snippet in REQUIRED_MAIN_SNIPPETS.items():
        ok = snippet in main_text
        add_check(checks, f"main_tex:{name}", ok)
        if not ok:
            errors.append(f"main.tex missing required snippet for {name}: {snippet}")

    cite_keys = extract_cite_keys(main_text)
    missing_cites = sorted(REQUIRED_BIB_KEYS - cite_keys)
    add_check(checks, "main_tex:all_required_citations_used", not missing_cites, f"missing={missing_cites}")
    if missing_cites:
        errors.append(f"main.tex does not cite required bibliography keys: {missing_cites}")

    bib_text = load_text(bib_path) if bib_path.exists() else ""
    bib_keys = extract_bib_keys(bib_text)
    missing_bib = sorted(REQUIRED_BIB_KEYS - bib_keys)
    add_check(checks, "references:required_bib_keys", not missing_bib, f"missing={missing_bib}")
    if missing_bib:
        errors.append(f"references.bib missing required keys: {missing_bib}")

    extra_bib = sorted(bib_keys - REQUIRED_BIB_KEYS)
    if extra_bib:
        warnings.append(f"references.bib has extra keys not required by this checker: {extra_bib}")

    tables_text = load_text(tables_path) if tables_path.exists() else ""
    for label in sorted(REQUIRED_TABLE_LABELS):
        ok = rf"\label{{{label}}}" in tables_text
        add_check(checks, f"paper_tables:label:{label}", ok)
        if not ok:
            errors.append(f"paper_tables.tex missing label: {label}")

    figure_text = load_text(figure_tex_path) if figure_tex_path.exists() else ""
    figure_caption_has_episode = r"\texttt{xzx\_radio\_sub23}" in figure_text
    add_check(checks, "figure:caption_names_verified_episode", figure_caption_has_episode)
    if not figure_caption_has_episode:
        errors.append("radio case figure caption does not name texttt xzx_radio_sub23")

    if summary_json_path.exists():
        summary = load_json(summary_json_path)
        verified = summary.get("data_scope", {}).get("verified_non_markovian_episodes") if isinstance(summary, dict) else None
        ok = verified == ["xzx_radio_sub23"]
        add_check(checks, "paper_summary:verified_non_markovian_scope", ok, f"verified={verified}")
        if not ok:
            errors.append("paper_ready_results_summary.json must list only xzx_radio_sub23 as verified non-Markovian")

    if validation_json_path.exists():
        validation = load_json(validation_json_path)
        valid = validation.get("valid") if isinstance(validation, dict) else None
        num_errors = validation.get("num_errors") if isinstance(validation, dict) else None
        num_warnings = validation.get("num_warnings") if isinstance(validation, dict) else None
        ok = valid is True and num_errors == 0
        detail = f"valid={valid}, errors={num_errors}, warnings={num_warnings}"
        add_check(checks, "benchmark_validation:valid_zero_errors", ok, detail)
        if not ok:
            errors.append("benchmark_v0_validation.json must be valid with zero errors")
        if num_warnings not in (0, None):
            warnings.append(f"benchmark validation has warnings: {num_warnings}")

    return {
        "valid": not errors,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "data_scope": {
            "verified_non_markovian_episodes": ["xzx_radio_sub23"],
            "note": "This checker intentionally enforces the current one-case Benchmark v0 boundary.",
        },
    }


def write_report(payload: dict, out_md: Path) -> None:
    valid = payload["valid"]
    num_errors = payload["num_errors"]
    num_warnings = payload["num_warnings"]
    lines = [
        "# Manuscript Package Check / Paper Package Check",
        "",
        "This checker validates the current Memory-GRM manuscript skeleton and generated paper artifacts without adding benchmark labels or rerunning GRM inference.",
        "",
        "## Status",
        "",
        f"- Valid: `{valid}`",
        f"- Errors: `{num_errors}`",
        f"- Warnings: `{num_warnings}`",
        "- Verified non-Markovian Benchmark v0 episode: `xzx_radio_sub23` only.",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        detail = str(check.get("detail", "")).replace("|", "\\|")
        check_name = check["name"]
        lines.append(f"| `{check_name}` | {status} | {detail} |")

    if payload["errors"]:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {item}" for item in payload["errors"])
    if payload["warnings"]:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {item}" for item in payload["warnings"])

    lines += [
        "",
        "## Reproducibility",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/check_manuscript_package.py",
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out_md)
    valid = payload["valid"]
    num_errors = payload["num_errors"]
    num_warnings = payload["num_warnings"]
    print(f"valid={str(valid).lower()}")
    print(f"errors={num_errors}")
    print(f"warnings={num_warnings}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
