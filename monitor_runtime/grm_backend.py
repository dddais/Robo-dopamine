"""GRM-backed monitor: runs Robo-Dopamine-GRM online inference on each poll.

Mirrors the contract of ``DeterministicMonitorBackend`` (``start`` / ``status``
/ ``stop`` / ``health``) so it can be a drop-in replacement in ``service.py``.

On every ``status`` poll it:
  1. pulls the latest multi-view JPEG frames from Robot Runtime,
  2. (optionally) undistorts the fisheye wrist cameras,
  3. builds forward / incremental / backward samples,
  4. runs one GRM ``inference_batch`` over all active modes,
  5. fuses the per-mode progress and feeds it to ``MonitorState`` to decide
     ``running`` / ``success`` / ``failed``.

The GRM model is loaded exactly once when the backend is constructed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from monitor_runtime.core import (
    MONITOR_STATUS_FAIL,
    MONITOR_STATUS_RUNNING,
    MONITOR_STATUS_SUCCESS,
    MonitorState,
    MonitorSession,
    clamp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = (
    "/home/ubuntu/dais/Robo-dopamine/pretrained_models/"
    "Robo-Dopamine-GRM-2.0-4B-Preview"
)
DEFAULT_GOAL_IMAGE = str(REPO_ROOT / "examples" / "blank_goal.png")

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FISHEYE_KEYS = ("cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")


# ---------------------------------------------------------------------------
# Fisheye undistortion (lazy, optional)
# ---------------------------------------------------------------------------

@dataclass
class FisheyeRemap:
    """Pre-computed fisheye -> pinhole remap table, or None when disabled."""

    map_x: np.ndarray
    map_y: np.ndarray
    interp: int
    border_mode: int
    border_value: int


def init_fisheye_remap(config_path: str) -> FisheyeRemap:
    """Build the fisheye remap table from a fisheye_process config.yaml.

    ``fisheye_process`` lives outside this repo; its ``convert`` module is
    imported on demand so the monitor service can run without it when
    undistortion is disabled.
    """
    config_file = Path(config_path).resolve()
    sys.path.insert(0, str(config_file.parent))
    from convert import (  # type: ignore[import-not-found]
        build_remap_table,
        compute_extrinsics,
        get_border_flag,
        get_interp_flag,
        load_config,
        load_pinhole_intrinsics,
    )

    cfg = load_config(str(config_file))
    config_dir = str(config_file.parent)
    load_pinhole_intrinsics(cfg, config_dir)
    compute_extrinsics(cfg, config_dir)
    cfg.setdefault("depth", {})
    cfg["depth"]["enabled"] = False

    map_x, map_y = build_remap_table(cfg)
    proc = cfg.get("processing", {})
    interp = get_interp_flag(proc.get("interpolation", "LINEAR"))
    border_mode = get_border_flag(proc.get("border_mode", "CONSTANT"))
    border_value = proc.get("border_value", 0)
    print(f"[GRM] Fisheye remap built: {cfg['pinhole']['image_width']}x"
          f"{cfg['pinhole']['image_height']}")
    return FisheyeRemap(
        map_x=map_x,
        map_y=map_y,
        interp=interp,
        border_mode=border_mode,
        border_value=border_value,
    )


def undistort_fisheye(img: np.ndarray, remap: FisheyeRemap) -> np.ndarray:
    return cv2.remap(
        img,
        remap.map_x,
        remap.map_y,
        remap.interp,
        borderMode=remap.border_mode,
        borderValue=(remap.border_value, remap.border_value, remap.border_value),
    )


# ---------------------------------------------------------------------------
# Progress tracking (per-mode, cumulative)
# ---------------------------------------------------------------------------

@dataclass
class ProgressTracker:
    """Accumulates per-mode progress across successive polls for one subtask.

    forward    -> progress == score (absolute w.r.t. ref_start)
    backward   -> progress == clamp(1 + score, 0, 1) (absolute w.r.t. ref_end)
    incremental-> progress recurses toward 1.0 (or 0.0 on negative score)
    """

    prev_progress: dict[str, float] = field(
        default_factory=lambda: {m: 0.0 for m in VALID_MODES}
    )
    counts: dict[str, int] = field(
        default_factory=lambda: {m: 0 for m in VALID_MODES}
    )

    def reset(self) -> None:
        self.prev_progress = {m: 0.0 for m in VALID_MODES}
        self.counts = {m: 0 for m in VALID_MODES}

    def update(self, mode: str, score: float) -> dict[str, float]:
        prev = self.prev_progress[mode]
        if mode == "incremental":
            if self.counts[mode] == 0:
                progress = score
            elif score >= 0:
                progress = prev + (1.0 - prev) * score
            else:
                progress = prev + prev * score
            hop = score
        elif mode == "forward":
            progress = score
            hop = progress - prev
        elif mode == "backward":
            progress = clamp(1.0 + score, 0.0, 1.0)
            hop = progress - prev
        else:
            raise ValueError(f"Unknown eval mode: {mode}")
        self.prev_progress[mode] = progress
        self.counts[mode] += 1
        return {"score": score, "hop": hop, "progress": progress}


def parse_score(pred_text: str) -> float:
    """Parse a GRM prediction into a [-1, 1] score."""
    try:
        match = re.search(r"<score>(.*?)</score>", pred_text)
        if match:
            value = match.group(1).replace("%", "").strip()
        else:
            matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", pred_text)
            value = matches[-1] if matches else "0"
        return clamp(float(value), -100.0, 100.0) / 100.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_online_samples(
    task: str,
    step: int,
    ref_start: dict[str, str],
    ref_end_path: str,
    previous: dict[str, str],
    current: dict[str, str],
    modes: list[str],
) -> list[dict[str, Any]]:
    """Build the vLLM batch items for all active modes at one step."""
    samples: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "incremental":
            before = previous
            before_id = f"prev_{step - 1:06d}"
        elif mode == "forward":
            before = ref_start
            before_id = "start_000000"
        elif mode == "backward":
            before = {k: ref_end_path for k in CAMERA_KEYS}
            before_id = "goal"
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        samples.append(
            {
                "id": f"grm-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
                "task": task,
                "eval_mode": mode,
                "image": [
                    ref_start["cam_high"],
                    ref_end_path,
                    before["cam_high"],
                    before["cam_left_wrist"],
                    before["cam_right_wrist"],
                    current["cam_high"],
                    current["cam_left_wrist"],
                    current["cam_right_wrist"],
                ],
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Per-subtask state
# ---------------------------------------------------------------------------

@dataclass
class _SubtaskState:
    """All mutable state tied to one monitor_id / subtask.

    Inference runs on a dedicated background thread at a fixed cadence;
    ``status()`` only reads ``latest`` so polling never blocks on the model.
    """

    subtask: str
    # ref_start/previous are None until the background thread captures the
    # initial reference frame; status() must tolerate this warm-up window.
    ref_start: Optional[dict[str, str]] = None
    previous: Optional[dict[str, str]] = None
    step: int = 0
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    monitor: MonitorState = field(default_factory=MonitorState)
    # latest inference snapshot consumed by status(); guarded by the backend lock
    latest: dict[str, Any] = field(default_factory=dict)
    # most recent inference error (None when healthy)
    error: Optional[str] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    def reset_progress(self) -> None:
        self.tracker.reset()
        self.monitor.reset()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class GRMMonitorBackend:
    """Monitor backend driven by online GRM inference."""

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        goal_image: str = DEFAULT_GOAL_IMAGE,
        runtime_url: str,
        cameras: Optional[list[str]] = None,
        observation_timeout: float = 3.0,
        fisheye_remap: Optional[FisheyeRemap] = None,
        active_modes: Optional[list[str]] = None,
        interval: float = 1.0,
        success_threshold: float = 0.60,
        success_stable_steps: int = 5,
        success_max_drift: float = 0.02,
        fail_stable_steps: int = 8,
        fail_min_progress: float = 0.01,
    ) -> None:
        self.model_path = model_path
        self.goal_image = goal_image
        self.runtime_url = runtime_url.rstrip("/")
        self.cameras = list(cameras or CAMERA_KEYS)
        self.observation_timeout = observation_timeout
        self.fisheye_remap = fisheye_remap
        self.active_modes = list(active_modes or VALID_MODES)
        self.interval = max(0.1, float(interval))

        self.success_threshold = success_threshold
        self.success_stable_steps = success_stable_steps
        self.success_max_drift = success_max_drift
        self.fail_stable_steps = fail_stable_steps
        self.fail_min_progress = fail_min_progress

        self._ref_end_path = self._materialize_goal_image(goal_image)
        self._cache_root = Path(tempfile.mkdtemp(prefix="grm_monitor_"))
        self._lock = threading.Lock()
        self.sessions: dict[str, _SubtaskState] = {}

        # Heavy import is deferred to construction so the module can be
        # imported (and unit-tested) without vLLM / torch installed.
        print(f"[GRM] Loading GRM model: {model_path}")
        from examples.inference import GRMInference

        self.model = GRMInference(model_path)
        print("[GRM] Model loaded.")

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _materialize_goal_image(goal_image: str) -> str:
        src = Path(goal_image).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Goal image not found: {src}")
        return str(src)

    def _cam_dirs(self, subtask: str) -> dict[str, Path]:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", subtask).strip("_")[:80] or "task"
        root = self._cache_root / safe
        dirs = {}
        for key in CAMERA_KEYS:
            d = root / key
            d.mkdir(parents=True, exist_ok=True)
            dirs[key] = d
        return dirs

    def _fetch_bytes(self, path: str) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(self.runtime_url + path, method="GET")
        with urllib.request.urlopen(
            request, timeout=self.observation_timeout
        ) as response:
            data = response.read()
            headers = {k: v for k, v in response.headers.items()}
        return data, headers

    def _fetch_json(self, path: str) -> dict[str, Any]:
        data, _ = self._fetch_bytes(path)
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("robot runtime returned non-object JSON")

    def _snapshot_current(self, subtask: str, step: int) -> dict[str, str]:
        """Pull latest frames from Robot Runtime, undistort, save to cache."""
        metadata = self._fetch_json("/observations/latest/metadata")
        endpoints = metadata.get("binary_endpoints") if isinstance(metadata, dict) else None
        if not isinstance(endpoints, dict):
            raise RuntimeError("observation metadata missing binary_endpoints")

        cam_dirs = self._cam_dirs(subtask)
        saved: dict[str, str] = {}
        for camera in CAMERA_KEYS:
            path = endpoints.get(camera) or f"/observations/latest/{camera}.jpg"
            data, _ = self._fetch_bytes(path)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"failed to decode JPEG for {camera}")

            if camera in FISHEYE_KEYS and self.fisheye_remap is not None:
                img = undistort_fisheye(img, self.fisheye_remap)

            out_path = cam_dirs[camera] / f"frame_{step:06d}.png"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            saved[camera] = str(out_path)
        return saved

    def _run_one_step(self, state: _SubtaskState) -> dict[str, Any]:
        """One GRM inference step. Updates state in place, returns the record.

        Also prints a real-time one-liner matching the reference agent script:
            [GRM] "<task>" step=NNNNNN fused=FF.FF% fwd=.. inc=.. bwd=.. lat=L.LLs [status]
        """
        current = self._snapshot_current(state.subtask, state.step)
        infer_start = time.time()

        samples = build_online_samples(
            task=state.subtask,
            step=state.step,
            ref_start=state.ref_start,
            ref_end_path=self._ref_end_path,
            previous=state.previous,
            current=current,
            modes=self.active_modes,
        )
        outputs = self.model.inference_batch(samples)

        mode_results: dict[str, dict[str, Any]] = {}
        for item in outputs:
            mode = item["eval_mode"]
            score = parse_score(item.get("pred", ""))
            stats = state.tracker.update(mode, score)
            mode_results[mode] = {
                "pred": item.get("pred", ""),
                "score": stats["score"],
                "hop": stats["hop"],
                "progress": stats["progress"],
            }

        fused = clamp(
            sum(float(mode_results[m]["progress"]) for m in self.active_modes)
            / len(self.active_modes),
            0.0,
            1.0,
        )
        status = state.monitor.update(fused)
        latency = time.time() - infer_start

        state.previous = current
        this_step = state.step
        state.step += 1

        record = {
            "step": this_step,
            "progress": fused,
            "progress_percent": fused * 100.0,
            "latency_s": latency,
            "fused": fused,
            "status": status,
            "modes": mode_results,
        }

        # Real-time one-liner (same shape as examples/online_inference_local_img_agent.py)
        parts = [
            '[GRM] "{task}" step={step:06d} fused={fused:6.2f}%'.format(
                task=state.subtask, step=this_step, fused=fused * 100.0
            )
        ]
        if "forward" in mode_results:
            parts.append("fwd={:.2f}%".format(float(mode_results["forward"]["progress"]) * 100.0))
        if "incremental" in mode_results:
            parts.append("inc={:.2f}%".format(float(mode_results["incremental"]["progress"]) * 100.0))
        if "backward" in mode_results:
            parts.append("bwd={:.2f}%".format(float(mode_results["backward"]["progress"]) * 100.0))
        parts.append("lat={:.2f}s".format(latency))
        parts.append("[{}]".format(status))
        print(" ".join(parts), flush=True)

        return record

    def _inference_loop(self, monitor_id: str, state: _SubtaskState) -> None:
        """Background thread: run GRM at a fixed cadence until terminal/stop.

        - sleeps ``interval`` between steps
        - stops once MonitorState reaches success/failed
        - publishes each step into ``state.latest`` under the backend lock so
          ``status()`` can read a consistent snapshot without blocking
        """
        print(f"[GRM] background inference started for monitor={monitor_id} "
              f"subtask=\"{state.subtask}\" interval={self.interval}s", flush=True)

        # Warm-up: capture the reference start frame with retries. This keeps
        # transient Robot Runtime outages from failing /monitors/start; the
        # loop just keeps trying until the runtime is reachable.
        while not state.stop_event.is_set() and state.ref_start is None:
            try:
                ref = self._snapshot_current(state.subtask, 0)
                with self._lock:
                    state.ref_start = ref
                    state.previous = ref
                    state.error = None
                print(f"[GRM] monitor={monitor_id} reference start captured.", flush=True)
            except Exception as exc:
                with self._lock:
                    state.error = f"failed to capture reference frame: {exc}"
                print(f"[GRM] monitor={monitor_id} reference capture error: {exc}; "
                      f"retrying in {self.interval}s", flush=True)
                if state.stop_event.wait(self.interval):
                    return

        while not state.stop_event.is_set():
            # Terminal: keep last result, stop burning GPU.
            if state.monitor.is_finished:
                print(f"[GRM] monitor={monitor_id} reached terminal state "
                      f"[{state.monitor.status}], inference loop exiting.", flush=True)
                return

            try:
                record = self._run_one_step(state)
                with self._lock:
                    state.latest = record
                    state.error = None
            except Exception as exc:  # network/model hiccup: log + keep looping
                with self._lock:
                    state.error = str(exc)
                print(f"[GRM] monitor={monitor_id} step error: {exc}", flush=True)
                # avoid tight error loop if the runtime is down
                if state.stop_event.wait(self.interval):
                    return
                continue

            if state.monitor.is_finished:
                print(f"[GRM] monitor={monitor_id} reached terminal state "
                      f"[{state.monitor.status}], inference loop exiting.", flush=True)
                return

            # wait for the next beat (or early stop)
            if state.stop_event.wait(self.interval):
                return
        print(f"[GRM] monitor={monitor_id} inference loop stopped.", flush=True)

    # -- contract ------------------------------------------------------

    def start(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        execution_id = str(payload.get("execution_id") or "")
        subtask = str(payload.get("subtask") or "")
        if not monitor_id or not execution_id or not subtask:
            raise ValueError("monitor_id, execution_id, and subtask are required")

        # NOTE: deliberately no network I/O here. The reference frame is
        # captured by the background thread (_inference_loop warm-up), so a
        # Robot Runtime hiccup never fails /monitors/start with a 500.
        with self._lock:
            state = _SubtaskState(subtask=subtask)
            state.monitor = MonitorState(
                success_threshold=self.success_threshold,
                success_stable_steps=self.success_stable_steps,
                success_max_drift=self.success_max_drift,
                fail_stable_steps=self.fail_stable_steps,
                fail_min_progress=self.fail_min_progress,
            )
            self.sessions[monitor_id] = state
            thread = threading.Thread(
                target=self._inference_loop,
                args=(monitor_id, state),
                name=f"grm-infer-{monitor_id}",
                daemon=True,
            )
            state.thread = thread

        thread.start()

        return MonitorSession(
            monitor_id=monitor_id,
            execution_id=execution_id,
            subtask=subtask,
            subtask_index=payload.get("subtask_index"),
            status=MONITOR_STATUS_RUNNING,
            message="grm monitor backend (warming up)",
            result={
                "provider": "grm",
                "model": self.model_path,
                "active_modes": list(self.active_modes),
                "fisheye_enabled": self.fisheye_remap is not None,
                "interval": self.interval,
            },
        )

    def status(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        # Read a consistent snapshot of the latest background inference result.
        with self._lock:
            state = self.sessions.get(monitor_id)
            if state is None:
                raise KeyError(f"unknown monitor_id: {monitor_id}")
            latest = dict(state.latest)
            last_error = state.error
            warming_up = state.ref_start is None
            status_val = state.monitor.status
            progress_val = (
                state.monitor.progress_history[-1]
                if state.monitor.progress_history
                else 0.0
            )
            poll_count = state.step
            is_finished = state.monitor.is_finished
            history = list(state.monitor.progress_history)

        session = MonitorSession(
            monitor_id=monitor_id,
            execution_id=str(payload.get("execution_id") or ""),
            subtask=state.subtask,
            status=status_val,
            progress=progress_val,
            poll_count=poll_count,
            updated_at=time.time(),
            message="grm monitor backend",
            result={"provider": "grm"},
        )

        expected_execution_id = payload.get("execution_id")
        if expected_execution_id and str(expected_execution_id) != session.execution_id:
            raise ValueError("monitor_id does not belong to execution_id")

        # Merge the latest inference snapshot (modes/latency/...) into the response.
        result: dict[str, Any] = {**session.result}
        if warming_up:
            result["warming_up"] = True
            session.message = "grm monitor backend (warming up: capturing reference frame)"
        if latest:
            result.update(
                {
                    "progress_percent": latest.get("progress_percent"),
                    "latency_s": latest.get("latency_s"),
                    "modes": latest.get("modes"),
                    "step": latest.get("step"),
                }
            )
        if last_error:
            result["error"] = last_error
        if is_finished:
            result["final_status"] = status_val
            result["progress_history"] = history
        session.result = result
        return session

    def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        monitor_id = str(payload.get("monitor_id") or "")
        with self._lock:
            state = self.sessions.pop(monitor_id, None)
        if state is not None:
            state.stop_event.set()
            thread = state.thread
        else:
            thread = None
        # Join outside the lock so we don't block other sessions.
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if state is not None:
            # best-effort cache cleanup for the subtask
            shutil.rmtree(
                self._cache_root
                / re.sub(r"[^A-Za-z0-9_]+", "_", state.subtask).strip("_")[:80],
                ignore_errors=True,
            )
        return {"stopped": True, "monitor_id": monitor_id}

    def health(self) -> dict[str, Any]:
        return {
            "status": MONITOR_STATUS_RUNNING,
            "provider": "grm",
            "model": self.model_path,
            "runtime_url": self.runtime_url,
            "cameras": list(self.cameras),
            "active_modes": list(self.active_modes),
            "fisheye_enabled": self.fisheye_remap is not None,
            "interval": self.interval,
            "sessions": len(self.sessions),
        }
