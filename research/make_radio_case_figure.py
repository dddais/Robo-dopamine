#!/usr/bin/env python3
"""Create a paper-ready figure for the verified radio Memory-GRM case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/xzx_episode_1_sub23_memory_grm/run_summary.json"
DEFAULT_EVENTS = ROOT / "benchmark_v0/event_annotations/xzx_radio_sub23_events.json"
DEFAULT_OUT_DIR = ROOT / "research_outputs/figures"
DEFAULT_OUT_PNG = DEFAULT_OUT_DIR / "radio_memory_grm_case.png"
DEFAULT_OUT_PDF = DEFAULT_OUT_DIR / "radio_memory_grm_case.pdf"
DEFAULT_OUT_TEX = DEFAULT_OUT_DIR / "radio_memory_grm_case.tex"
DEFAULT_OUT_MD = DEFAULT_OUT_DIR / "radio_memory_grm_case.md"
DEFAULT_OUT_JSON = DEFAULT_OUT_DIR / "radio_memory_grm_case.json"

EVENTS_TO_SHOW = ["lift", "button_press", "indicator_green", "release"]
PREFERRED_VIEW = {
    "lift": "cam_high",
    "button_press": "cam_left_wrist",
    "indicator_green": "cam_left_wrist",
    "release": "cam_high",
}
EVENT_LABELS = {
    "grasp": "grasp",
    "lift": "lift",
    "button_press": "button press",
    "indicator_green": "indicator green",
    "place": "place",
    "release": "release",
}
EVENT_COLORS = {
    "grasp": "#6b7280",
    "lift": "#2563eb",
    "button_press": "#c2410c",
    "indicator_green": "#16a34a",
    "place": "#7c3aed",
    "release": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    parser.add_argument("--out-tex", type=Path, default=DEFAULT_OUT_TEX)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def root_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def fused_series(summary: dict) -> list[float]:
    mode_series = [
        summary["modes"][mode]["progress_series"]
        for mode in ("forward", "incremental", "backward")
    ]
    min_len = min(len(series) for series in mode_series)
    return [mean(series[i] for series in mode_series) for i in range(min_len)]


def choose_event_image(event: dict) -> Path:
    evidence = [root_path(path) for path in event.get("view_evidence", [])]
    preferred = PREFERRED_VIEW.get(event["event"])
    if preferred:
        for path in evidence:
            if preferred in path.name and path.exists():
                return path
    for path in evidence:
        if path.exists():
            return path
    raise FileNotFoundError(f"No evidence image found for event {event['event']}")


def crop_center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def plot_event_thumbnail(ax: plt.Axes, event: dict) -> Path:
    path = choose_event_image(event)
    image = crop_center_square(Image.open(path).convert("RGB"))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    color = EVENT_COLORS.get(event["event"], "#111827")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2.6)
    ax.set_title(
        f"{EVENT_LABELS.get(event['event'], event['event'])}\n"
        f"frame {event['frame_id']} / {event['time_index']:.1f}s",
        fontsize=9,
        pad=6,
    )
    return path


def build_payload(summary: dict, events_doc: dict) -> dict:
    fused = fused_series(summary)
    frame_interval = int(summary["frame_interval"])
    frames = [i * frame_interval for i in range(len(fused))]
    peak_progress = max(fused)
    peak_index = fused.index(peak_progress)
    final_progress = fused[-1]
    events = {event["event"]: event for event in events_doc["events"]}
    shown = []
    for name in EVENTS_TO_SHOW:
        event = events[name]
        image_path = choose_event_image(event)
        shown.append(
            {
                "event": name,
                "frame_id": event["frame_id"],
                "time_index": event["time_index"],
                "image_path": rel(image_path),
            }
        )
    return {
        "episode_id": summary["episode_id"],
        "task": summary["task"],
        "data_scope": "single verified non-Markovian radio episode",
        "frame_interval": frame_interval,
        "frames": frames,
        "grm": {
            "fused_final_progress": final_progress,
            "fused_peak_progress": peak_progress,
            "fused_peak_frame": frames[peak_index],
            "fused_peak_time_sec": frames[peak_index] / 30.0,
            "final_success_threshold": 70.0,
        },
        "event_rule": events_doc["success_rule"],
        "shown_events": shown,
        "decisions": {
            "final_only_grm": "not success",
            "event_latched_grm": "success",
        },
        "limitation": (
            "This figure is generated from cached GRM outputs and human-verified "
            "radio event labels; it is not an automatic event detector and not "
            "additional robot data."
        ),
    }


def make_figure(summary: dict, events_doc: dict, out_png: Path, out_pdf: Path) -> None:
    modes = summary["modes"]
    fused = fused_series(summary)
    frame_interval = int(summary["frame_interval"])
    frames = [i * frame_interval for i in range(len(fused))]
    times = [frame / 30.0 for frame in frames]
    events = {event["event"]: event for event in events_doc["events"]}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig = plt.figure(figsize=(11.2, 7.4), constrained_layout=False)
    grid = GridSpec(
        nrows=3,
        ncols=4,
        figure=fig,
        height_ratios=[1.25, 1.65, 0.95],
        hspace=0.52,
        wspace=0.18,
    )

    for col, name in enumerate(EVENTS_TO_SHOW):
        ax = fig.add_subplot(grid[0, col])
        plot_event_thumbnail(ax, events[name])

    ax_progress = fig.add_subplot(grid[1, :])
    mode_styles = {
        "forward": ("#2563eb", "-"),
        "incremental": ("#c2410c", "--"),
        "backward": ("#7c3aed", "-."),
    }
    for mode, (color, linestyle) in mode_styles.items():
        series = modes[mode]["progress_series"][: len(times)]
        ax_progress.plot(
            times,
            series,
            label=mode,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            alpha=0.82,
        )
    ax_progress.plot(
        times,
        fused,
        label="fused average",
        color="#111827",
        linewidth=2.6,
        zorder=5,
    )
    ax_progress.axhline(
        70.0,
        color="#991b1b",
        linestyle=":",
        linewidth=1.3,
        label="70% success threshold",
    )

    peak_value = max(fused)
    peak_index = fused.index(peak_value)
    final_value = fused[-1]
    ax_progress.scatter(
        [times[peak_index]],
        [peak_value],
        color="#16a34a",
        edgecolor="white",
        linewidth=1.0,
        s=70,
        zorder=7,
    )
    ax_progress.scatter(
        [times[-1]],
        [final_value],
        color="#991b1b",
        edgecolor="white",
        linewidth=1.0,
        s=70,
        zorder=7,
    )
    ax_progress.annotate(
        f"peak {peak_value:.1f}%\nframe {frames[peak_index]}",
        xy=(times[peak_index], peak_value),
        xytext=(times[peak_index] + 1.2, min(96, peak_value + 16)),
        arrowprops={"arrowstyle": "->", "color": "#166534", "lw": 1.0},
        fontsize=8,
        color="#166534",
    )
    ax_progress.annotate(
        f"final {final_value:.1f}%\nfinal-only: fail",
        xy=(times[-1], final_value),
        xytext=(times[-1] - 6.3, final_value - 24),
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "lw": 1.0},
        fontsize=8,
        color="#7f1d1d",
    )

    for event in events_doc["events"]:
        event_name = event["event"]
        event_time = float(event["time_index"])
        color = EVENT_COLORS.get(event_name, "#4b5563")
        ax_progress.axvline(
            event_time,
            color=color,
            linewidth=0.9,
            alpha=0.45,
            zorder=0,
        )
        if event_name in {"button_press", "indicator_green"}:
            ax_progress.text(
                event_time + 0.12,
                4,
                EVENT_LABELS[event_name],
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=8,
                color=color,
            )

    ax_progress.set_xlim(min(times), max(times))
    ax_progress.set_ylim(0, 100)
    ax_progress.set_xlabel("time (s)")
    ax_progress.set_ylabel("GRM progress (%)")
    ax_progress.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax_progress.set_title(
        "GRM progress peaks during manipulation, but final progress is low after the radio is put down",
        loc="left",
        fontweight="bold",
    )
    ax_progress.legend(loc="upper left", ncols=3, frameon=False)

    ax_memory = fig.add_subplot(grid[2, :])
    ax_memory.set_xlim(min(times), max(times))
    ax_memory.set_ylim(-0.35, 1.35)
    ax_memory.set_yticks([0, 1])
    ax_memory.set_yticklabels(["not latched", "latched"])
    ax_memory.set_xlabel("time (s)")
    ax_memory.set_title(
        "Event memory latches hidden success evidence through the final frame",
        loc="left",
        fontweight="bold",
    )
    memory_label_y = {
        "grasp": 0.56,
        "lift": 0.56,
        "button_press": 0.72,
        "indicator_green": 1.28,
        "place": 0.56,
        "release": 0.56,
    }
    memory_label_rotation = {
        "button_press": 0,
        "indicator_green": 0,
    }
    memory_label_va = {
        "indicator_green": "bottom",
    }
    for event_name in events_doc["success_rule"]["required_order"]:
        event = events[event_name]
        event_time = float(event["time_index"])
        color = EVENT_COLORS.get(event_name, "#4b5563")
        ax_memory.scatter(
            [event_time],
            [1.0],
            marker="o",
            s=65 if event_name in {"button_press", "indicator_green"} else 40,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax_memory.text(
            event_time,
            memory_label_y.get(event_name, 0.56),
            EVENT_LABELS.get(event_name, event_name),
            ha="center",
            va=memory_label_va.get(event_name, "top"),
            fontsize=8,
            color=color,
            rotation=memory_label_rotation.get(event_name, 22),
        )
    for event_name in ("button_press", "indicator_green"):
        event_time = float(events[event_name]["time_index"])
        ax_memory.hlines(
            y=1.0,
            xmin=event_time,
            xmax=max(times),
            color=EVENT_COLORS[event_name],
            linewidth=4.0,
            alpha=0.25,
        )
    ax_memory.text(
        max(times),
        1.18,
        "Event-Latched GRM: success",
        ha="right",
        va="center",
        color="#166534",
        fontweight="bold",
    )
    ax_memory.text(
        max(times),
        -0.16,
        f"Final-only GRM: not success ({final_value:.1f}% < 70%)",
        ha="right",
        va="center",
        color="#7f1d1d",
        fontweight="bold",
    )
    ax_memory.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax_memory.spines["top"].set_visible(False)
    ax_memory.spines["right"].set_visible(False)

    fig.suptitle(
        "Radio hidden-intermediate-event case: final state misses the decisive success evidence",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Generated from cached GRM outputs and human-verified keyframe events for xzx_radio_sub23 only.",
        ha="center",
        fontsize=8,
        color="#374151",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_figure_tex(payload: dict, out_tex: Path, out_pdf: Path) -> None:
    lines = [
        "% Auto-generated by research/make_radio_case_figure.py",
        "% Requires: \\usepackage{graphicx}",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{rel(out_pdf)}}}",
        "\\caption{Hidden-intermediate-event radio case. GRM progress peaks while the radio is being manipulated, but the fused final progress falls to 34.17\\% after the radio is put down. Event memory latches the human-verified button press and green-indicator events, preserving the success evidence through the final frame. This figure uses only the verified \\texttt{xzx\\_radio\\_sub23} episode.}",
        "\\label{fig:radio_memory_grm_case}",
        "\\end{figure}",
        "",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_report(payload: dict, out_md: Path, out_png: Path, out_pdf: Path, out_tex: Path) -> None:
    grm = payload["grm"]
    lines = [
        "# Radio Memory-GRM Case Figure",
        "",
        "This figure is generated from cached Robo-Dopamine GRM outputs and human-verified radio keyframe events. It does not rerun GRM and does not add any benchmark labels.",
        "",
        f"- Episode: `{payload['episode_id']}`",
        f"- Fused peak progress: {grm['fused_peak_progress']:.2f}% at frame {grm['fused_peak_frame']} ({grm['fused_peak_time_sec']:.1f}s)",
        f"- Fused final progress: {grm['fused_final_progress']:.2f}%",
        "- Final-only GRM decision: not success under the 70% threshold.",
        "- Event-Latched GRM decision: success because `button_press` and `indicator_green` are latched.",
        "",
        "Outputs:",
        "",
        f"- PNG: `{rel(out_png)}`",
        f"- PDF: `{rel(out_pdf)}`",
        f"- LaTeX snippet: `{rel(out_tex)}`",
        f"- JSON: `{rel(DEFAULT_OUT_JSON)}`",
        "",
        "Suggested caption:",
        "",
        "Figure: Hidden-intermediate-event radio case. GRM progress peaks while the radio is being manipulated, but the fused final progress falls to 34.17% after the radio is put down. Event memory latches the human-verified button press and green-indicator events, preserving the success evidence through the final frame.",
        "",
        "Limitation: this is one human-verified non-Markovian episode, not a completed large-scale benchmark.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    events_doc = load_json(args.events)
    payload = build_payload(summary, events_doc)
    make_figure(summary, events_doc, args.out_png, args.out_pdf)
    write_figure_tex(payload, args.out_tex, args.out_pdf)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out_md, args.out_png, args.out_pdf, args.out_tex)
    print(f"wrote={args.out_png}")
    print(f"wrote={args.out_pdf}")
    print(f"wrote={args.out_tex}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")
    print(f"fused_final={payload['grm']['fused_final_progress']:.2f}")
    print(f"fused_peak={payload['grm']['fused_peak_progress']:.2f}")


if __name__ == "__main__":
    main()
