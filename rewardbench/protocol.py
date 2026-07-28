from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schemas import EpisodeRecord, FrameRecord

IMAGE_LABELS = (
    "reference_start",
    "reference_end",
    "before_cam_high",
    "before_cam_left_wrist",
    "before_cam_right_wrist",
    "after_cam_high",
    "after_cam_left_wrist",
    "after_cam_right_wrist",
)

OFFICIAL_SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


SIMPLIFIED_SYSTEM_PROMPT = """You are a rigorous, impartial vision evaluator for robot task progress.
Judge whether AFTER moves closer to the task objective than BEFORE.

Task: {task}

REFERENCE START: <image>
REFERENCE END (a blank placeholder; do not infer a visual goal from it): <image>
BEFORE front: <image>
BEFORE left wrist: <image>
BEFORE right wrist: <image>
AFTER front: <image>
AFTER left wrist: <image>
AFTER right wrist: <image>

Use task-aligned changes, completeness, contact, placement, orientation, stability,
and failures. The dataset is single-view; duplicated camera slots are identical
and must not be treated as independent evidence. Return ONLY:
<score>+NN%</score>, <score>-NN%</score>, or <score>0%</score>.
"""

PROMPT_MODES = {
    "official": OFFICIAL_SYSTEM_PROMPT,
    "simplified": SIMPLIFIED_SYSTEM_PROMPT,
}


def system_prompt(prompt_mode: str = "simplified") -> str:
    """Return a named, versioned raw-evaluation prompt.

    ``official`` is byte-for-byte the ``SYSTEM_PROMPT`` literal in
    ``examples/inference.py``.  ``simplified`` is the single-view/blank-goal
    adaptation used by the first RoboRewardBench raw run.
    """
    try:
        return PROMPT_MODES[prompt_mode]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_MODES))
        raise ValueError(f"Unknown prompt_mode {prompt_mode!r}; choose one of {choices}") from exc

SCORE_RE = re.compile(r"\A<score>([+-]?(?:0|[1-9]\d?|100)(?:\.\d+)?)%</score>\Z")


def native_endpoint_payload(
    episode: EpisodeRecord,
    frames: FrameRecord,
    blank_goal: str | Path,
    *,
    prompt_mode: str = "simplified",
) -> dict[str, Any]:
    blank = str(Path(blank_goal).resolve())
    if not Path(blank).is_file():
        raise FileNotFoundError(blank)
    images = [
        frames.first_path,
        blank,
        frames.first_path,
        frames.first_path,
        frames.first_path,
        frames.last_path,
        frames.last_path,
        frames.last_path,
    ]
    payload = episode.model_payload()
    payload.update(
        {
            "protocol": "native_endpoint_forward_v1",
            "image": images,
            "image_labels": list(IMAGE_LABELS),
            "prompt_mode": prompt_mode,
            "prompt": system_prompt(prompt_mode).format(task=episode.task),
        }
    )
    return payload


def chat_messages(task: str, prompt_mode: str = "simplified") -> list[dict[str, Any]]:
    chunks = system_prompt(prompt_mode).format(task=task).split("<image>")
    content: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if index < len(chunks) - 1:
            content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def temporal_chat_messages(task: str, image_count: int) -> list[dict[str, Any]]:
    if not 2 <= image_count <= 8:
        raise ValueError("Temporal ablation requires 2..8 frames")
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "You are evaluating one single-view robot episode. The images are "
                "uniformly ordered from start to end; they are temporal frames, not "
                "different cameras. Task: "
                + task
                + "\nFrames:\n"
            ),
        }
    ]
    for index in range(image_count):
        content.extend(
            (
                {"type": "text", "text": f"t{index}: "},
                {"type": "image"},
                {"type": "text", "text": "\n"},
            )
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Estimate end-of-episode task progress relative to the first frame. "
                "Return ONLY <score>+NN%</score>, <score>-NN%</score>, or <score>0%</score>."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def parse_score(text: str) -> float:
    match = SCORE_RE.fullmatch(text.strip())
    if not match:
        raise ValueError(f"Invalid GRM score output: {text!r}")
    value = float(match.group(1))
    if abs(value) > 100:
        raise ValueError(f"Score is outside [-100,100]: {value}")
    return value / 100.0


def progress(signed_score: float) -> float:
    return min(1.0, max(0.0, float(signed_score)))


def progress_to_reward(value: float) -> int:
    thresholds = (0.125, 0.375, 0.625, 0.875)
    return 1 + sum(float(value) >= threshold for threshold in thresholds)
