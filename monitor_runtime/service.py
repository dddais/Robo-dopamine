#!/usr/bin/env python3
"""HTTP monitor service implementing the Robot Runtime monitor contract.

This is a lightweight service skeleton. It provides the stable distributed
interface now; the GRM backend can later replace ``DeterministicMonitorBackend``
without changing the robot runtime or MCP adapter.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from monitor_runtime.core import MONITOR_STATUS_FAIL, MONITOR_STATUS_RUNNING, MONITOR_STATUS_SUCCESS, MonitorSession

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse


class DeterministicMonitorBackend:
    def __init__(
        self,
        *,
        auto_success_after_polls: int = 0,
        observation_client: "RobotRuntimeObservationClient | None" = None,
    ) -> None:
        self.auto_success_after_polls = max(0, auto_success_after_polls)
        self.observation_client = observation_client
        self.sessions: dict[str, MonitorSession] = {}

    def start(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        execution_id = str(payload.get("execution_id") or "")
        subtask = str(payload.get("subtask") or "")
        if not monitor_id or not execution_id or not subtask:
            raise ValueError("monitor_id, execution_id, and subtask are required")
        session = MonitorSession(
            monitor_id=monitor_id,
            execution_id=execution_id,
            subtask=subtask,
            subtask_index=payload.get("subtask_index"),
            message="deterministic monitor backend",
            result={"provider": "deterministic"},
        )
        self.sessions[monitor_id] = session
        return session

    def status(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        if monitor_id not in self.sessions:
            raise KeyError(f"unknown monitor_id: {monitor_id}")
        session = self.sessions[monitor_id]
        expected_execution_id = payload.get("execution_id")
        if expected_execution_id and str(expected_execution_id) != session.execution_id:
            raise ValueError("monitor_id does not belong to execution_id")

        poll_count = session.poll_count + 1
        status = session.status
        progress = session.progress
        observation_result: dict[str, Any] = {}
        if self.observation_client is not None:
            observation_result = self.observation_client.fetch()
        if self.auto_success_after_polls and poll_count >= self.auto_success_after_polls:
            status = MONITOR_STATUS_SUCCESS
            progress = 1.0
        refreshed = replace(
            session,
            status=status,
            progress=progress,
            poll_count=poll_count,
            updated_at=time.time(),
            result={
                **session.result,
                **({"observation": observation_result} if observation_result else {}),
            },
        )
        self.sessions[monitor_id] = refreshed
        return refreshed

    def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        monitor_id = str(payload.get("monitor_id") or "")
        session = self.sessions.get(monitor_id)
        if session is not None:
            self.sessions[monitor_id] = replace(
                session,
                status=MONITOR_STATUS_FAIL,
                message="monitor stopped",
                updated_at=time.time(),
            )
        return {"stopped": True, "monitor_id": monitor_id}

    def health(self) -> dict[str, Any]:
        return {
            "status": MONITOR_STATUS_RUNNING,
            "provider": "deterministic",
            "sessions": len(self.sessions),
            "auto_success_after_polls": self.auto_success_after_polls,
            "observation_client": self.observation_client.health() if self.observation_client else None,
        }


class RobotRuntimeObservationClient:
    """Pulls latest binary JPEG frames from Robot Runtime."""

    def __init__(
        self,
        *,
        runtime_url: str,
        cameras: list[str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.runtime_url = runtime_url.rstrip("/")
        self.cameras = cameras or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.timeout = timeout

    def fetch(self) -> dict[str, Any]:
        metadata = self._get_json("/observations/latest/metadata")
        endpoints = metadata.get("binary_endpoints") if isinstance(metadata, dict) else None
        if not isinstance(endpoints, dict):
            raise RuntimeError("robot runtime observation metadata did not include binary_endpoints")
        images: dict[str, dict[str, Any]] = {}
        for camera in self.cameras:
            path = endpoints.get(camera) or f"/observations/latest/{camera}.jpg"
            data, headers = self._get_bytes(str(path))
            images[camera] = {
                "bytes": len(data),
                "content_type": headers.get("Content-Type") or headers.get("content-type"),
                "frame_id": headers.get("X-Frame-Id") or headers.get("x-frame-id"),
                "timestamp": headers.get("X-Timestamp") or headers.get("x-timestamp"),
            }
        return {
            "runtime_url": self.runtime_url,
            "frame_id": metadata.get("frame_id"),
            "timestamp": metadata.get("timestamp"),
            "cameras": list(images),
            "images": images,
        }

    def health(self) -> dict[str, Any]:
        return {
            "runtime_url": self.runtime_url,
            "cameras": list(self.cameras),
            "transport": "http_binary_jpeg",
        }

    def _get_json(self, path: str) -> dict[str, Any]:
        data, _headers = self._get_bytes(path)
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("robot runtime returned non-object JSON")

    def _get_bytes(self, path: str) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(self.runtime_url + _safe_path(path), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                headers = {key: value for key, value in response.headers.items()}
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"failed to fetch robot observation: {exc}") from exc
        return data, headers


def create_app(backend: DeterministicMonitorBackend | None = None) -> "FastAPI":
    from fastapi import FastAPI

    backend = backend or DeterministicMonitorBackend()
    app = FastAPI(title="Robo-Dopamine Monitor Service")

    @app.get("/health")
    async def health():
        return _ok(backend.health())

    @app.post("/monitors/start")
    async def monitors_start(body: dict[str, Any]):
        try:
            session = backend.start(body)
        except ValueError as exc:
            return _fail(str(exc), status=400)
        return _ok(session.to_dict())

    @app.post("/monitors/status")
    async def monitors_status(body: dict[str, Any]):
        try:
            session = backend.status(body)
        except KeyError as exc:
            return _fail(str(exc), status=404)
        except ValueError as exc:
            return _fail(str(exc), status=409)
        return _ok(session.to_dict())

    @app.post("/monitors/stop")
    async def monitors_stop(body: dict[str, Any]):
        return _ok(backend.stop(body))

    return app


def _ok(data: object = None) -> "JSONResponse":
    from fastapi.responses import JSONResponse

    return JSONResponse({"success": True, "data": data, "message": "ok"})


def _fail(message: str, *, status: int) -> "JSONResponse":
    from fastapi.responses import JSONResponse

    return JSONResponse({"success": False, "data": None, "message": message}, status_code=status)


def _safe_path(path: str) -> str:
    if not path.startswith("/") or "://" in path or ".." in path.split("/"):
        raise ValueError(f"unsafe robot runtime path: {path!r}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Robo-Dopamine monitor service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--auto-success-after-polls", type=int, default=0)
    parser.add_argument(
        "--robot-runtime-url",
        default=None,
        help="Optional Robot Runtime URL used to fetch binary JPEG observations.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera to fetch from Robot Runtime; repeatable. Defaults to all dual-Franka cameras.",
    )
    parser.add_argument("--observation-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    import uvicorn

    observation_client = None
    if args.robot_runtime_url:
        observation_client = RobotRuntimeObservationClient(
            runtime_url=args.robot_runtime_url,
            cameras=args.camera or None,
            timeout=args.observation_timeout,
        )

    uvicorn.run(
        create_app(
            DeterministicMonitorBackend(
                auto_success_after_polls=args.auto_success_after_polls,
                observation_client=observation_client,
            )
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
