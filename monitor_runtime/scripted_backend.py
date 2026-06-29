"""Scripted monitor backend: replays a predefined progress curve.

Mirrors the contract of ``DeterministicMonitorBackend`` /
``GRMMonitorBackend`` (``start`` / ``status`` / ``stop`` / ``health``) so it
is a drop-in replacement in ``service.py``. Instead of loading a model or
contacting Robot Runtime, it consumes a YAML script that fully defines the
monitor's behaviour:

  - the per-step progress curve to feed into ``MonitorState``
  - optional per-step ``status`` / ``modes`` / ``latency`` payload
  - the cadence (``interval``) between steps
  - the same success/fail thresholds the GRM backend uses

This is convenient for integration tests, demos, and upper-layer (Robot
Runtime / MCP adapter) dry-runs where the result needs to be deterministic
and offline.

Script shape (see ``configs/monitor_scripted.yaml`` for a working example):

    backend: scripted
    interval: 0.5
    scripted:
      # Either a flat list of fused-progress floats ...
      progress: [0.05, 0.15, 0.30, 0.45, 0.60, 0.62, 0.63, 0.63]

      # ... or an explicit per-step list. Each step can override status /
      # modes / latency. ``progress`` is still the value fed to MonitorState.
      steps:
        - progress: 0.05
          status: running
        - progress: 0.60
          latency_s: 0.18

      # Optional: shortcut presets ("success" / "fail" / "plateau")
      scenario: success    # ignored once progress/steps is non-empty

      # Hold the last progress value after the script ends (default: true).
      # When false, the backend stops stepping once the script is consumed
      # and keeps reporting the final snapshot.
      hold_last: true
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from monitor_runtime.core import (
    MONITOR_STATUS_FAIL,
    MONITOR_STATUS_RUNNING,
    MONITOR_STATUS_SUCCESS,
    MonitorState,
    MonitorSession,
    clamp,
)

VALID_MODES = ("forward", "incremental", "backward")


# ---------------------------------------------------------------------------
# Script normalization
# ---------------------------------------------------------------------------

@dataclass
class ScriptedStep:
    """One row of a predefined monitor script."""

    progress: float
    status_override: Optional[str] = None
    latency_s: Optional[float] = None
    modes: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _preset_curve(scenario: str, length: int = 12) -> list[float]:
    """Return the raw progress curve for a named preset.

    Returned values are clamped floats in [0, 1]; callers wrap them through
    ``normalize_script`` to get ``ScriptedStep`` objects.
    """
    scenario = (scenario or "").lower().strip()
    if scenario in ("success", "ok"):
        # Ramp past the success threshold and stabilize there.
        curve = [0.05, 0.15, 0.30, 0.45, 0.55, 0.62, 0.65, 0.66, 0.66, 0.66]
    elif scenario in ("fail", "failed", "plateau"):
        # Stay low and flat so MonitorState trips the fail branch.
        curve = [0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
    elif scenario in ("running", "loop", "steady"):
        # Sub-threshold but slowly improving; never terminates within length.
        curve = [0.05 + 0.03 * i for i in range(length)]
    elif scenario in ("oscillate",):
        # Bounce around the threshold to demonstrate drift gating.
        curve = [0.40, 0.62, 0.45, 0.63, 0.50, 0.64, 0.48, 0.65, 0.66, 0.66]
    elif not scenario:
        # Default: a short, mildly rising curve that ends in success.
        curve = [0.05, 0.20, 0.40, 0.55, 0.62, 0.65, 0.66, 0.66]
    else:
        raise ValueError(f"unknown scripted scenario: {scenario!r}")

    return [clamp(float(p), 0.0, 1.0) for p in curve]


# Backwards-compat alias for callers that used the old name.
def _build_preset(scenario: str, length: int = 12) -> list[ScriptedStep]:
    return [ScriptedStep(progress=p) for p in _preset_curve(scenario, length)]


def _normalize_scenario_queue(raw: Any, *, known: tuple[str, ...]) -> list[str]:
    """Normalize the YAML ``scenario_queue`` field.

    Accepts:
      - a list of scenario names, e.g. [success, fail, fail, success]
      - a comma-separated string, e.g. "success,fail,fail,success"
      - a single scenario name, e.g. "fail"  (every subtask uses fail)
      - None / empty -> []  (defaults to "success" for every subtask)

    Short aliases ("suc"/"ok" -> success, "f"/"ko" -> fail) are accepted.
    Unknown names fall back to "success" with a warning printed.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(s).strip().lower() for s in raw if str(s).strip()]
    else:
        raise ValueError(
            f"scenario_queue must be a list or string, got {type(raw).__name__}"
        )

    aliases = {
        "suc": "success", "succ": "success", "ok": "success", "s": "success",
        "fail": "fail", "failed": "fail", "ko": "fail", "f": "fail",
    }
    out: list[str] = []
    for item in items:
        name = aliases.get(item, item)
        if name not in known:
            print(f"[SCRIPTED] unknown scenario name {item!r} in queue, "
                  f"falling back to 'success'", flush=True)
            name = "success"
        out.append(name)
    return out


def normalize_script(raw: Any) -> list[ScriptedStep]:
    """Turn the ``scripted:`` block of the YAML into a list of steps.

    Accepted shapes:
      - ``progress: [0.1, 0.2, ...]``  (shorthand list of floats)
      - ``steps: [{progress: .., status: .., latency_s: .., modes: ..}]``
      - ``scenario: success|fail|plateau|running|oscillate`` (preset)
      - a raw list of floats (treated as the ``progress`` shorthand)
    """
    if raw is None:
        return _build_preset("")
    if isinstance(raw, (int, float)):
        raise ValueError("scripted block must be a mapping or a list, not a scalar")
    if isinstance(raw, list):
        return _normalize_progress_list(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"unsupported scripted block type: {type(raw).__name__}")

    if raw.get("steps"):
        return _normalize_steps(raw["steps"])
    if raw.get("progress"):
        return _normalize_progress_list(raw["progress"])
    if raw.get("scenario"):
        return _build_preset(str(raw["scenario"]))
    return _build_preset("")


def _normalize_progress_list(items: Any) -> list[ScriptedStep]:
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("progress list must be a non-empty sequence")
    steps: list[ScriptedStep] = []
    for idx, item in enumerate(items):
        if isinstance(item, (int, float)):
            steps.append(ScriptedStep(progress=clamp(float(item), 0.0, 1.0)))
            continue
        if isinstance(item, dict):
            steps.extend(_normalize_steps([item]))
            continue
        raise ValueError(f"progress[{idx}] must be a number or mapping, got {type(item).__name__}")
    return steps


def _normalize_steps(items: Any) -> list[ScriptedStep]:
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("steps must be a non-empty sequence")
    steps: list[ScriptedStep] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"steps[{idx}] must be a mapping, got {type(item).__name__}")
        if "progress" not in item:
            raise ValueError(f"steps[{idx}] is missing required 'progress' field")
        progress = clamp(float(item["progress"]), 0.0, 1.0)
        status_override = item.get("status")
        if status_override is not None:
            status_override = str(status_override).lower()
            if status_override not in (MONITOR_STATUS_RUNNING, MONITOR_STATUS_SUCCESS, MONITOR_STATUS_FAIL):
                raise ValueError(
                    f"steps[{idx}].status must be running|success|failed, got {status_override!r}"
                )
        latency = item.get("latency_s")
        if latency is not None:
            latency = float(latency)
        modes = item.get("modes") or {}
        if not isinstance(modes, dict):
            raise ValueError(f"steps[{idx}].modes must be a mapping")
        extra = {k: v for k, v in item.items()
                 if k not in {"progress", "status", "latency_s", "modes"}}
        steps.append(ScriptedStep(
            progress=progress,
            status_override=status_override,
            latency_s=latency,
            modes=dict(modes),
            extra=dict(extra),
        ))
    return steps


# ---------------------------------------------------------------------------
# Per-subtask state
# ---------------------------------------------------------------------------

@dataclass
class _SubtaskState:
    subtask: str
    script: list[ScriptedStep]
    interval: float
    hold_last: bool
    monitor: MonitorState = field(default_factory=MonitorState)
    cursor: int = 0
    latest: dict[str, Any] = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class ScriptedMonitorBackend:
    """Monitor backend that replays a predefined progress script."""

    def __init__(
        self,
        *,
        success_curve: Any = None,
        fail_curve: Any = None,
        scenario_queue: Any = None,
        interval: float = 0.5,
        hold_last: bool = True,
        success_threshold: float = 0.60,
        success_stable_steps: int = 5,
        success_max_drift: float = 0.02,
        fail_stable_steps: int = 8,
        fail_min_progress: float = 0.01,
    ) -> None:
        # Two named curves: per-subtask selection comes from scenario_queue.
        self.curves: dict[str, list[ScriptedStep]] = {
            "success": normalize_script(success_curve or _preset_curve("success")),
            "fail": normalize_script(fail_curve or _preset_curve("fail")),
        }
        # Ordered queue of scenario names, consumed one per /monitors/start.
        # Names not in self.curves fall back to "success". When the queue is
        # exhausted, the last entry repeats (or "success" if queue was empty).
        self.scenario_queue: list[str] = _normalize_scenario_queue(
            scenario_queue, known=tuple(self.curves.keys())
        )
        self._scenario_cursor: int = 0
        # Kept for backwards compatibility / health() introspection.
        self.script = self.curves["success"]
        self.interval: float = max(0.05, float(interval))
        self.hold_last: bool = bool(hold_last)

        self.success_threshold = success_threshold
        self.success_stable_steps = success_stable_steps
        self.success_max_drift = success_max_drift
        self.fail_stable_steps = fail_stable_steps
        self.fail_min_progress = fail_min_progress

        self._lock = threading.Lock()
        self.sessions: dict[str, _SubtaskState] = {}

    def _next_scenario(self) -> str:
        """Pop the next scenario from the queue.

        Caller must already hold ``self._lock`` (this is a non-reentrant Lock).
        """
        queue = self.scenario_queue
        if not queue:
            return "success"
        idx = min(self._scenario_cursor, len(queue) - 1)
        self._scenario_cursor += 1
        return queue[idx]

    # -- helpers --------------------------------------------------------

    def _run_one_step(self, state: _SubtaskState) -> dict[str, Any]:
        """Consume one row of the script and feed MonitorState."""
        step_idx = state.cursor
        script = state.script

        if step_idx < len(script):
            row = script[step_idx]
            progress = row.progress
        elif self.hold_last and script:
            row = script[-1]
            progress = row.progress
        else:
            # Script exhausted and not holding: do nothing.
            return {}

        state.cursor += 1

        status = state.monitor.update(progress)
        if row.status_override is not None:
            # Explicit override wins; useful for forcing a terminal state
            # regardless of the MonitorState thresholds.
            status = row.status_override
            state.monitor.status = status

        latency = row.latency_s if row.latency_s is not None else 0.0
        modes = self._materialize_modes(row, progress)

        record = {
            "step": step_idx,
            "progress": progress,
            "progress_percent": progress * 100.0,
            "latency_s": latency,
            "fused": progress,
            "status": status,
            "modes": modes,
            **row.extra,
        }

        parts = [
            '[SCRIPTED] "{task}" step={step:06d} fused={fused:6.2f}%'.format(
                task=state.subtask, step=step_idx, fused=progress * 100.0
            )
        ]
        for mode in VALID_MODES:
            if mode in modes:
                parts.append("{name}={value:6.2f}%".format(
                    name=mode[:3],
                    value=float(modes[mode].get("progress", 0.0)) * 100.0,
                ))
        parts.append("lat={:.2f}s".format(latency))
        parts.append("[{}]".format(status))
        print(" ".join(parts), flush=True)
        return record

    @staticmethod
    def _materialize_modes(row: ScriptedStep, progress: float) -> dict[str, Any]:
        """Default per-mode progress to the fused value when not given."""
        modes = {m: {
            "score": progress,
            "hop": 0.0,
            "progress": progress,
        } for m in VALID_MODES}
        for mode_name, mode_val in row.modes.items():
            if isinstance(mode_val, dict):
                modes[mode_name] = {
                    "score": float(mode_val.get("score", progress)),
                    "hop": float(mode_val.get("hop", 0.0)),
                    "progress": float(mode_val.get("progress", progress)),
                }
            elif isinstance(mode_val, (int, float)):
                modes[mode_name] = {
                    "score": float(mode_val),
                    "hop": 0.0,
                    "progress": clamp(float(mode_val), 0.0, 1.0),
                }
            else:
                modes[mode_name] = {"value": mode_val}
        return modes

    def _loop(self, monitor_id: str, state: _SubtaskState) -> None:
        """Background thread: replay the script at a fixed cadence."""
        print(
            f"[SCRIPTED] background replay started for monitor={monitor_id} "
            f"subtask=\"{state.subtask}\" interval={state.interval}s "
            f"steps={len(state.script)}",
            flush=True,
        )
        while not state.stop_event.is_set():
            if state.monitor.is_finished:
                print(
                    f"[SCRIPTED] monitor={monitor_id} reached terminal state "
                    f"[{state.monitor.status}], replay loop exiting.",
                    flush=True,
                )
                return

            record: dict[str, Any] = {}
            try:
                record = self._run_one_step(state)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[SCRIPTED] monitor={monitor_id} step error: {exc}", flush=True)

            if record:
                with self._lock:
                    state.latest = record

            if state.monitor.is_finished:
                print(
                    f"[SCRIPTED] monitor={monitor_id} reached terminal state "
                    f"[{state.monitor.status}], replay loop exiting.",
                    flush=True,
                )
                return

            consumed_all = state.cursor >= len(state.script)
            if consumed_all and not self.hold_last:
                print(
                    f"[SCRIPTED] monitor={monitor_id} script consumed "
                    f"(hold_last=False), replay loop exiting.",
                    flush=True,
                )
                return

            if state.stop_event.wait(state.interval):
                return

        print(f"[SCRIPTED] monitor={monitor_id} replay loop stopped.", flush=True)

    # -- contract ------------------------------------------------------

    def start(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        execution_id = str(payload.get("execution_id") or "")
        subtask = str(payload.get("subtask") or "")
        if not monitor_id or not execution_id or not subtask:
            raise ValueError("monitor_id, execution_id, and subtask are required")

        with self._lock:
            curve_name = self._next_scenario()
            state = _SubtaskState(
                subtask=subtask,
                script=list(self.curves[curve_name]),
                interval=self.interval,
                hold_last=self.hold_last,
                monitor=MonitorState(
                    success_threshold=self.success_threshold,
                    success_stable_steps=self.success_stable_steps,
                    success_max_drift=self.success_max_drift,
                    fail_stable_steps=self.fail_stable_steps,
                    fail_min_progress=self.fail_min_progress,
                ),
            )
            self.sessions[monitor_id] = state
            thread = threading.Thread(
                target=self._loop,
                args=(monitor_id, state),
                name=f"scripted-replay-{monitor_id}",
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
            message="scripted monitor backend (replaying)",
            result={
                "provider": "scripted",
                "interval": self.interval,
                "hold_last": self.hold_last,
                "scenario": curve_name,
                "queue_position": self._scenario_cursor,
                "script_steps": len(state.script),
            },
        )

    def status(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        with self._lock:
            state = self.sessions.get(monitor_id)
            if state is None:
                raise KeyError(f"unknown monitor_id: {monitor_id}")
            latest = dict(state.latest)
            status_val = state.monitor.status
            progress_val = (
                state.monitor.progress_history[-1]
                if state.monitor.progress_history
                else 0.0
            )
            cursor = state.cursor
            is_finished = state.monitor.is_finished
            history = list(state.monitor.progress_history)

        session = MonitorSession(
            monitor_id=monitor_id,
            execution_id=str(payload.get("execution_id") or ""),
            subtask=state.subtask,
            status=status_val,
            progress=progress_val,
            poll_count=cursor,
            updated_at=time.time(),
            message="scripted monitor backend",
            result={"provider": "scripted"},
        )

        expected_execution_id = payload.get("execution_id")
        if expected_execution_id and str(expected_execution_id) != session.execution_id:
            raise ValueError("monitor_id does not belong to execution_id")

        result: dict[str, Any] = {**session.result}
        if latest:
            result.update({
                "progress_percent": latest.get("progress_percent"),
                "latency_s": latest.get("latency_s"),
                "modes": latest.get("modes"),
                "step": latest.get("step"),
            })
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
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return {"stopped": True, "monitor_id": monitor_id}

    def health(self) -> dict[str, Any]:
        with self._lock:
            queue = list(self.scenario_queue)
            cursor = self._scenario_cursor
        return {
            "status": MONITOR_STATUS_RUNNING,
            "provider": "scripted",
            "interval": self.interval,
            "hold_last": self.hold_last,
            "curves": {name: len(steps) for name, steps in self.curves.items()},
            "scenario_queue": queue,
            "queue_position": cursor,
            "sessions": len(self.sessions),
        }


# ---------------------------------------------------------------------------
# Convenience CLI for running this backend standalone (without service.py)
# ---------------------------------------------------------------------------

def _demo_client(host: str, port: int) -> None:
    """Tiny client loop that starts a monitor and polls it until terminal."""
    base = f"http://{host}:{port}"
    monitor_id = f"scripted-demo-{int(time.time())}"
    execution_id = "demo-exec"
    subtask = "demo subtask"

    post_json(base + "/monitors/start", {
        "monitor_id": monitor_id,
        "execution_id": execution_id,
        "subtask": subtask,
        "subtask_index": 0,
    })
    print(f"[demo] monitor started: {monitor_id}")

    while True:
        time.sleep(0.5)
        data = post_json(base + "/monitors/status", {
            "monitor_id": monitor_id,
            "execution_id": execution_id,
        })
        session = (data or {}).get("data") or {}
        status = session.get("status")
        progress = session.get("progress")
        step = (session.get("result") or {}).get("step")
        print(f"[demo] status={status} progress={progress} step={step}")
        if status in (MONITOR_STATUS_SUCCESS, MONITOR_STATUS_FAIL):
            break

    post_json(base + "/monitors/stop", {"monitor_id": monitor_id})
    print("[demo] monitor stopped.")


def post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc
    return json.loads(raw.decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Run the scripted monitor service standalone.

    Equivalent to ``python -m monitor_runtime.service --backend scripted``,
    but kept here so the backend can be exercised on its own without the
    FastAPI plumbing in service.py.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Robo-Dopamine scripted monitor service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--config", default=None, help="YAML config (same shape as monitor.yaml)")
    parser.add_argument("--interval", type=float, default=None, help="Override config interval")
    parser.add_argument("--scenario", default=None,
                        help="Preset: success|fail|plateau|running|oscillate")
    parser.add_argument("--demo-client", action="store_true",
                        help="After starting the server, run a tiny client that starts a "
                             "monitor and polls it to completion.")
    args = parser.parse_args(argv)

    script_raw: Any = None
    cfg_interval: float | None = None
    hold_last = True
    thresholds: dict[str, Any] = {}

    if args.config:
        import yaml
        from pathlib import Path
        cfg_path = Path(args.config).expanduser().resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("config root must be a mapping")
        script_raw = (cfg.get("scripted") or {}).get("script") if isinstance(cfg.get("scripted"), dict) else cfg.get("scripted")
        cfg_interval = cfg.get("interval")
        hold_last = bool(cfg.get("hold_last", True))
        for key in ("success_threshold", "success_stable_steps", "success_max_drift",
                    "fail_stable_steps", "fail_min_progress"):
            if key in cfg:
                thresholds[key] = cfg[key]

    if args.scenario:
        script_raw = {"scenario": args.scenario}

    interval = args.interval if args.interval is not None else (cfg_interval or 0.5)

    backend = ScriptedMonitorBackend(
        script=script_raw,
        interval=interval,
        hold_last=hold_last,
        **thresholds,
    )

    from monitor_runtime.service import create_app
    import uvicorn

    if args.demo_client:
        import threading
        threading.Thread(
            target=_demo_client,
            args=(args.host, args.port),
            daemon=True,
        ).start()

    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
