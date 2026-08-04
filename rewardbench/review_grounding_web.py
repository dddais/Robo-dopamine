"""Local HTTP UI for reviewing instruction-conditioned grounding boxes.

Run on the remote machine, then forward the loopback-only port over SSH:

    python rewardbench/review_grounding_web.py --run-dir <grounding_dino_dir>
    ssh -L 8765:127.0.0.1:8765 user@remote-host

Open http://127.0.0.1:8765 locally.  Images and annotations never leave the
remote machine except through the user's SSH tunnel.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewardbench.io import append_jsonl, read_jsonl


LABELS = {"correct", "incorrect", "uncertain"}


@dataclass
class ReviewStore:
    run_dir: Path
    reviewer_id: str

    def __post_init__(self) -> None:
        template_path = self.run_dir / "audit_template.jsonl"
        if not template_path.is_file():
            raise FileNotFoundError(
                f"Missing {template_path}; run grounding audit once before starting the web reviewer."
            )
        self.rows = list(read_jsonl(template_path))
        if not self.rows:
            raise ValueError("Audit template has no reviewable rows")
        targets_path = self.run_dir.parent / "targets.jsonl"
        self.targets = (
            {row["example_id"]: row for row in read_jsonl(targets_path)}
            if targets_path.is_file()
            else {}
        )
        self.by_id = {row["example_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("Duplicate example IDs in audit template")
        self.index_by_id = {
            row["example_id"]: index for index, row in enumerate(self.rows)
        }
        self.output = self.run_dir / f"{self.reviewer_id}.jsonl"
        self._lock = Lock()

    def completed(self) -> dict[str, dict[str, Any]]:
        if not self.output.exists():
            return {}
        latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.output):
            if (
                row.get("example_id") in self.by_id
                and row.get("first_label") in LABELS
                and row.get("last_label") in LABELS
            ):
                latest[row["example_id"]] = row
        return latest

    def state(
        self,
        example_id: str | None = None,
        position: int | None = None,
    ) -> dict[str, Any]:
        completed = self.completed()
        if example_id is not None and position is not None:
            raise ValueError("Specify either example_id or position, not both")
        if position is not None:
            if not 1 <= position <= len(self.rows):
                raise ValueError(f"position must be in 1..{len(self.rows)}")
            current = self.rows[position - 1]
        elif example_id is not None:
            if example_id not in self.by_id:
                raise KeyError("Unknown review example")
            current = self.by_id[example_id]
        else:
            current = next(
                (row for row in self.rows if row["example_id"] not in completed),
                None,
            )
        current_index = (
            self.index_by_id[current["example_id"]] if current is not None else None
        )
        previous_example_id = (
            self.rows[current_index - 1]["example_id"]
            if current_index is not None and current_index > 0
            else self.rows[-1]["example_id"] if current is None and completed else None
        )
        next_example_id = (
            self.rows[current_index + 1]["example_id"]
            if current_index is not None and current_index + 1 < len(self.rows)
            else None
        )
        saved = completed.get(current["example_id"]) if current is not None else None
        return {
            "total": len(self.rows),
            "completed": len(completed),
            "current": self.public_row(current) if current else None,
            "position": current_index + 1 if current_index is not None else None,
            "current_label": saved.get("first_label") if saved else None,
            "previous_example_id": previous_example_id,
            "next_example_id": next_example_id,
            "done": current is None,
        }

    def public_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        target = self.targets.get(row["example_id"], {})
        return {
            "example_id": row["example_id"],
            "data_number": row.get("data_number"),
            "visualization_number": row.get("visualization_number"),
            "instruction": row.get("instruction", ""),
            "target_phrase": target.get("target_phrase", "(target unavailable)"),
            "entity_type": target.get("entity_type", "unknown"),
            "first_image": f"/api/image?example_id={row['example_id']}&frame=first",
            "last_image": f"/api/image?example_id={row['example_id']}&frame=last",
        }

    def image_path(self, example_id: str, frame: str) -> Path:
        if frame not in {"first", "last"} or example_id not in self.by_id:
            raise KeyError("Unknown review image")
        raw = self.by_id[example_id]["endpoints"][frame]["visualization_path"]
        path = Path(raw).resolve()
        root = self.run_dir.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Visualization path escapes run directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def submit(self, example_id: str, label: str) -> dict[str, Any]:
        if label not in LABELS:
            raise ValueError("label must be correct, incorrect, or uncertain")
        if example_id not in self.by_id:
            raise KeyError("Unknown example ID")
        with self._lock:
            row = self.by_id[example_id]
            append_jsonl(
                self.output,
                {
                    "data_number": row.get("data_number"),
                    "visualization_number": row.get("visualization_number"),
                    "example_id": example_id,
                    "grounding_fingerprint": row["grounding_fingerprint"],
                    "reviewer_id": self.reviewer_id,
                    "first_label": label,
                    "last_label": label,
                    "error_categories": [],
                    "reason": "",
                },
            )
        return self.state()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounding review</title><style>
body{font-family:system-ui,sans-serif;margin:22px;background:#f6f7f9;color:#17181a}#meta{color:#555}
#task{font-size:1.15rem;font-weight:650;margin:8px 0}#target{margin:6px 0 16px;color:#333}
#images{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:white;padding:10px;border-radius:8px;box-shadow:0 1px 3px #0002}
img{width:100%;height:auto;display:block}.label{font-weight:650;margin:0 0 8px}button{font-size:1.2rem;padding:10px 25px;margin:18px 10px 0 0;border:0;border-radius:7px;cursor:pointer}
#yes{background:#14883d;color:white}#no{background:#c33333;color:white}#uncertain{background:#d99c15;color:#111}
#back,#next,#go{background:#5f6874;color:white}button:disabled{cursor:not-allowed;opacity:.45}
#jump-controls{display:inline-flex;align-items:center;gap:7px;margin:18px 0 0 8px}#jump{font-size:1.1rem;width:6rem;padding:9px;border:1px solid #aab0b8;border-radius:7px}#go{margin:0;padding:10px 18px}
#saved{margin:10px 0;color:#394150;font-weight:600}#done{font-size:1.3rem;margin-top:45px}
@media(max-width:800px){#images{grid-template-columns:1fr}}
</style></head><body><h1>Grounding review</h1><div id="app">Loading…</div>
<script>
let current=null,previousExampleId=null,nextExampleId=null,busy=false;
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function jumpControls(state){return `<span id="jump-controls"><label for="jump">Page</label><input id="jump" type="number" min="1" max="${state.total}" value="${state.position ?? state.total}"><span>/ ${state.total}</span><button id="go" onclick="goToPosition()">Go</button></span>`;}
async function load(exampleId=null,position=null){if(busy)return;busy=true;const suffix=exampleId?`?example_id=${encodeURIComponent(exampleId)}`:position!==null?`?position=${encodeURIComponent(position)}`:'';const response=await fetch(`/api/state${suffix}`);const state=await response.json();const app=document.querySelector('#app');previousExampleId=state.previous_example_id;nextExampleId=state.next_example_id;
if(state.done){current=null;app.innerHTML=`<div id="done">Completed ${state.completed}/${state.total}. You may close this page.</div><button id="back" onclick="goBack()" ${previousExampleId?'':'disabled'}>← Previous (←)</button>${jumpControls(state)}`;busy=false;return;}
current=state.current;const saved=state.current_label?`<div id="saved">Saved label: ${esc(state.current_label)} · choose again to correct it</div>`:'';
app.innerHTML=`<div id="meta">Item ${state.position}/${state.total} · Completed ${state.completed}/${state.total} · data #${current.data_number ?? '?'} · visual #${current.visualization_number ?? '?'}</div><div id="task">${esc(current.instruction)}</div><div id="target">Target object: <b>${esc(current.target_phrase)}</b> <small>[${esc(current.entity_type)}]</small></div>${saved}<div id="images"><div class="panel"><div class="label">FIRST endpoint</div><img src="${current.first_image}" alt="first endpoint"></div><div class="panel"><div class="label">LAST endpoint</div><img src="${current.last_image}" alt="last endpoint"></div></div><button id="yes" onclick="submit('correct')">Yes (Y)</button><button id="no" onclick="submit('incorrect')">No (N)</button><button id="uncertain" onclick="submit('uncertain')">Uncertain (U)</button><button id="back" onclick="goBack()" ${previousExampleId?'':'disabled'}>← Previous (←)</button><button id="next" onclick="goNext()" ${nextExampleId?'':'disabled'}>Next (→) →</button>${jumpControls(state)}`;busy=false;}
async function submit(label){if(!current||busy)return;const nextAfterSubmit=nextExampleId;busy=true;document.querySelectorAll('button').forEach(x=>x.disabled=true);await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({example_id:current.example_id,label})});busy=false;if(nextAfterSubmit){await load(nextAfterSubmit);}else{await load();}}
async function goBack(){if(!previousExampleId||busy)return;await load(previousExampleId);}
async function goNext(){if(!nextExampleId||busy)return;await load(nextExampleId);}
async function goToPosition(){if(busy)return;const input=document.querySelector('#jump');const position=Number(input?.value);if(!Number.isInteger(position)||position<1||position>Number(input?.max)){input?.focus();return;}await load(null,position);}
window.addEventListener('keydown',e=>{if(e.target?.id==='jump'){if(e.key==='Enter')goToPosition();return;}if(busy)return;if(e.key==='y'||e.key==='Y')submit('correct');if(e.key==='n'||e.key==='N')submit('incorrect');if(e.key==='u'||e.key==='U')submit('uncertain');if(e.key==='ArrowLeft')goBack();if(e.key==='ArrowRight')goNext();});load();
</script></body></html>"""


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                example_id = query.get("example_id", [None])[0]
                try:
                    raw_position = query.get("position", [None])[0]
                    position = int(raw_position) if raw_position is not None else None
                    state = store.state(example_id, position)
                except (KeyError, ValueError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(state)
                return
            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                try:
                    image = store.image_path(query.get("example_id", [""])[0], query.get("frame", [""])[0])
                except (KeyError, ValueError, FileNotFoundError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                    return
                content = image.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(image.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/decision":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = store.submit(str(payload["example_id"]), str(payload["label"]))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(result)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[review-web] {self.address_string()} {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback-only HTTP grounding reviewer")
    parser.add_argument("--run-dir", required=True, help="Grounding backend output directory")
    parser.add_argument("--reviewer", default="reviewer1", choices=("reviewer1", "reviewer2"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    store = ReviewStore(Path(args.run_dir).resolve(), args.reviewer)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Review server: http://127.0.0.1:{args.port}")
    print(f"Output JSONL: {store.output}")
    print("Bind address is loopback only. Forward the port with SSH or your IDE.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
