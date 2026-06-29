#!/usr/bin/env python3
"""Build a claim/evidence ledger for the current Memory-GRM paper package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_MD = ROOT / "research_outputs/claim_evidence_ledger.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/claim_evidence_ledger.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def build_payload() -> dict:
    summary = load_json(ROOT / "research_outputs/paper_ready_results_summary.json")
    validation = load_json(ROOT / "research_outputs/benchmark_v0_validation.json")
    benchmark = load_json(ROOT / "research_outputs/benchmark_v0_event_memory_eval.json")
    counterfactual = load_json(ROOT / "research_outputs/radio_event_counterfactuals.json")
    radio = load_json(ROOT / "research_outputs/radio_event_memory_mvp.json")
    figure = load_json(ROOT / "research_outputs/figures/radio_memory_grm_case.json")
    episodes = load_json(ROOT / "benchmark_v0/episodes.json")

    live_ids = [item.get("episode_id") for item in episodes]
    verified = summary.get("data_scope", {}).get("verified_non_markovian_episodes")
    errors: list[str] = []
    if live_ids != ["xzx_radio_sub23"]:
        errors.append(f"live benchmark episodes changed: {live_ids}")
    if verified != ["xzx_radio_sub23"]:
        errors.append(f"paper summary verified episodes changed: {verified}")
    if validation.get("valid") is not True or validation.get("num_errors") != 0:
        errors.append("Benchmark v0 validation is not clean")

    visible = summary["cached_grm_visible_state"]
    interval10 = next(row for row in visible if row["setting"] == "fused final, interval 10")
    interval20 = next(row for row in visible if row["setting"] == "fused final, interval 20")
    bench_accuracy = benchmark["aggregate"]["accuracy"]
    cf_balanced = counterfactual["balanced_accuracy"]

    claims = [
        {
            "id": "C1",
            "status": "supported_visible_state_only",
            "claim": "Robo-Dopamine GRM-2.0-8B is a strong visible-state progress baseline on cached local runs.",
            "evidence": [
                f"Fused final interval 10 AUROC {pct(interval10['auroc'])}, best F1 {interval10['best_f1']:.3f}, accuracy {pct(interval10['accuracy'])}.",
                f"Fused final interval 20 AUROC {pct(interval20['auroc'])}, best F1 {interval20['best_f1']:.3f}, accuracy {pct(interval20['accuracy'])}.",
            ],
            "source_files": [
                "research_outputs/paper_ready_results_summary.json",
                "research_outputs/cached_grm_metrics.csv",
            ],
            "paper_wording": "Use as visible-state baseline evidence; do not claim GRM is broadly weak.",
            "scope_limit": "Cached visible-state diagnostic data, not non-Markovian benchmark evidence.",
        },
        {
            "id": "C2",
            "status": "supported_one_case_feasibility",
            "claim": "The verified turn-on-radio episode demonstrates hidden intermediate success evidence.",
            "evidence": [
                f"Verified non-Markovian episode list: {verified}.",
                f"Fused final progress {radio['grm']['fused_final_progress']:.2f}%, fused peak progress {radio['grm']['fused_peak_progress']:.2f}%.",
                "Human-verified event labels include button_press frame 615 and indicator_green frame 630.",
            ],
            "source_files": [
                "benchmark_v0/event_annotations/xzx_radio_sub23_events.json",
                "research_outputs/radio_event_memory_mvp.json",
                "research_outputs/figures/radio_memory_grm_case.json",
            ],
            "paper_wording": "Present as one verified non-Markovian case study and feasibility evidence.",
            "scope_limit": "One real verified episode only.",
        },
        {
            "id": "C3",
            "status": "supported_logic_stress_test",
            "claim": "Score-only memory cannot represent missing, out-of-order, or forbidden event predicates when scalar score evidence is identical.",
            "evidence": [
                f"Radio counterfactual final-only balanced accuracy {pct(cf_balanced['final_only_grm'])}.",
                f"Radio counterfactual score-memory balanced accuracy {pct(cf_balanced['score_memory_grm'])}.",
                "All counterfactual variants reuse the same GRM score curve and differ only in event labels.",
            ],
            "source_files": [
                "research_outputs/radio_event_counterfactuals.json",
                "research_outputs/radio_event_counterfactuals.md",
            ],
            "paper_wording": "Use as event-label counterfactual stress test, not as additional real robot data.",
            "scope_limit": "Synthetic label variants over one trajectory.",
        },
        {
            "id": "C4",
            "status": "supported_logic_stress_test",
            "claim": "Event-Latched GRM separates valid and invalid radio event histories under identical scalar score evidence.",
            "evidence": [
                f"Event-Latched GRM counterfactual balanced accuracy {pct(cf_balanced['event_latched_grm'])}.",
                f"Current Benchmark v0 Event-Latched GRM accuracy {pct(bench_accuracy['event_latched_grm'])} on the single labeled episode.",
            ],
            "source_files": [
                "research_outputs/radio_event_counterfactuals.json",
                "research_outputs/benchmark_v0_event_memory_eval.json",
            ],
            "paper_wording": "Describe as rule-based EventMemory MVP over human-verified labels.",
            "scope_limit": "Does not prove automatic event detection or statistical benchmark superiority.",
        },
        {
            "id": "C5",
            "status": "supported_guardrail",
            "claim": "Benchmark v0 currently has exactly one verified non-Markovian episode and the repository enforces that boundary.",
            "evidence": [
                f"Live benchmark ids: {live_ids}.",
                f"Benchmark validation valid={validation.get('valid')}, errors={validation.get('num_errors')}, warnings={validation.get('num_warnings')}.",
            ],
            "source_files": [
                "benchmark_v0/episodes.json",
                "research_outputs/benchmark_v0_validation.json",
                "research/validate_benchmark_v0.py",
            ],
            "paper_wording": "State this explicitly in data scope, limitations, and captions.",
            "scope_limit": "The benchmark is not yet large enough for statistical claims.",
        },
    ]

    unsupported = [
        {
            "id": "U1",
            "claim": "A completed large-scale non-Markovian robot benchmark.",
            "reason": "Only one human-verified non-Markovian episode is live.",
            "required_next_evidence": "Additional human-verified turn-on-radio-like positive/negative histories, preferably final-state-similar pairs.",
        },
        {
            "id": "U2",
            "claim": "Automatic event detection.",
            "reason": "Current radio events are human verified from keyframes; no event detector is evaluated.",
            "required_next_evidence": "VLM/event-head predictions compared against human event labels.",
        },
        {
            "id": "U3",
            "claim": "Carrot/cube/bottle cached cases are non-Markovian benchmark labels.",
            "reason": "They are baseline diagnostics and candidate inspection material only.",
            "required_next_evidence": "Manual proof of final-state-similar histories where hidden events change the label.",
        },
        {
            "id": "U4",
            "claim": "Statistical superiority of Memory-GRM on non-Markovian robot tasks.",
            "reason": "The current real non-Markovian benchmark has one labeled episode.",
            "required_next_evidence": "A larger labeled Benchmark v0 with matched positive/negative histories and confidence intervals.",
        },
    ]

    return {
        "valid": not errors,
        "errors": errors,
        "data_scope": {
            "live_benchmark_episode_ids": live_ids,
            "verified_non_markovian_episodes": verified,
            "adds_benchmark_labels": False,
        },
        "claims": claims,
        "unsupported_claims": unsupported,
        "figure_consistency": {
            "fused_final_progress": figure["grm"]["fused_final_progress"],
            "fused_peak_progress": figure["grm"]["fused_peak_progress"],
        },
    }


def write_report(payload: dict, out_md: Path) -> None:
    lines = [
        "# Claim/Evidence Ledger / 论文主张与证据台账",
        "",
        "This ledger records what the current Memory-GRM package can and cannot claim. It does not add benchmark labels or rerun GRM inference.",
        "",
        "## Data Scope",
        "",
        f"- Valid: `{payload['valid']}`",
        f"- Live benchmark episode ids: `{', '.join(payload['data_scope']['live_benchmark_episode_ids'])}`",
        f"- Verified non-Markovian episodes: `{', '.join(payload['data_scope']['verified_non_markovian_episodes'] or [])}`",
        f"- Adds benchmark labels: `{payload['data_scope']['adds_benchmark_labels']}`",
        "",
        "## Supported Claims",
        "",
        "| ID | Status | Claim | Evidence | Scope limit |",
        "|---|---|---|---|---|",
    ]
    for item in payload["claims"]:
        evidence = "<br>".join(item["evidence"])
        lines.append(
            f"| `{item['id']}` | `{item['status']}` | {item['claim']} | {evidence} | {item['scope_limit']} |"
        )
    lines += [
        "",
        "## Unsupported Claims",
        "",
        "| ID | Do not claim | Reason | Required next evidence |",
        "|---|---|---|---|",
    ]
    for item in payload["unsupported_claims"]:
        lines.append(
            f"| `{item['id']}` | {item['claim']} | {item['reason']} | {item['required_next_evidence']} |"
        )
    if payload["errors"]:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {item}" for item in payload["errors"])
    lines += [
        "",
        "## Reproducibility",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/make_claim_evidence_ledger.py",
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
    print(f"valid={str(payload['valid']).lower()}")
    print(f"claims={len(payload['claims'])}")
    print(f"unsupported={len(payload['unsupported_claims'])}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")
    if not payload["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
