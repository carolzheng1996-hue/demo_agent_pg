from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

try:
    from .config import BASE_DIR, OUTPUT_DIR, ensure_directories
    from .orchestrator import DGOrchestrator
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
except ImportError:
    from config import BASE_DIR, OUTPUT_DIR, ensure_directories
    from orchestrator import DGOrchestrator
    from state import DGGlobalState
    from task_manager import DGTaskManager


WEB_DIR = BASE_DIR / "web"
DEFAULT_DATASET_PATH = (BASE_DIR.parent / "data" / "ETTh1.csv").resolve()
JOB_PAYLOAD_FILE = "job_payload.json"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _summary_without_predictions(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return payload
    summary = dict(payload)
    if "predictions" in summary:
        summary["prediction_points"] = len(summary.get("predictions") or [])
        summary.pop("predictions", None)
    return summary


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _step_detail(step_dir: Path, root: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"step_name": step_dir.name, "files": []}
    result_json = step_dir / "result.json"
    if result_json.exists():
        payload["result"] = _read_json(result_json)

    markdown_files = sorted(step_dir.glob("*.md"))
    if markdown_files:
        payload["markdown"] = _read_text(markdown_files[0])

    for file_path in sorted(step_dir.iterdir()):
        if file_path.is_file():
            payload["files"].append({"name": file_path.name, "relative_path": _safe_relpath(file_path, root)})
    return payload


def load_task_detail(task_id: str) -> Dict[str, Any]:
    task_dir = OUTPUT_DIR / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task not found: {task_id}")

    iteration_dirs = sorted([path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("iteration_")])
    iterations: List[Dict[str, Any]] = []
    latest_dataset_profile: Dict[str, Any] = {}
    task_type = "unknown"
    requires_modeling = False

    for iteration_dir in iteration_dirs:
        plan_payload = _read_json(iteration_dir / "plan.json") or {}
        task_type = plan_payload.get("task_type") or task_type
        requires_modeling = requires_modeling or bool(
            plan_payload.get("task_type") == "forecast_modeling"
            or any(step in (plan_payload.get("plan") or []) for step in ["model_selection", "model_training", "model_integration", "evaluator"])
        )
        steps = []
        for step_dir in sorted([path for path in iteration_dir.iterdir() if path.is_dir()]):
            detail = _step_detail(step_dir, task_dir)
            result = detail.get("result") or {}
            latest_dataset_profile = result.get("dataset_profile") or latest_dataset_profile
            steps.append(detail)

        iterations.append(
            {
                "iteration_id": iteration_dir.name,
                "plan": plan_payload,
                "summary_markdown": _read_text(iteration_dir / "summary.md"),
                "steps": steps,
            }
        )

    final_summary = _read_text(task_dir / "final_summary.md")
    iteration_history = _read_json(task_dir / "iteration_history.json") or {}
    cross_iteration_ensemble = _summary_without_predictions(_read_json(task_dir / "cross_iteration_ensemble.json"))
    status = "completed" if final_summary else "running"
    return {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "status": status,
        "task_type": task_type,
        "requires_modeling": requires_modeling,
        "dataset_profile": latest_dataset_profile,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "pending_approval": None,
        "final_summary": final_summary,
        "iteration_history": iteration_history,
        "cross_iteration_ensemble": cross_iteration_ensemble,
    }


def list_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    if not OUTPUT_DIR.exists():
        return []

    task_dirs = sorted([path for path in OUTPUT_DIR.iterdir() if path.is_dir()], reverse=True)
    items: List[Dict[str, Any]] = []
    for task_dir in task_dirs[:limit]:
        try:
            detail = load_task_detail(task_dir.name)
        except Exception:
            continue
        items.append(
            {
                "task_id": detail["task_id"],
                "status": detail["status"],
                "iteration_count": detail["iteration_count"],
                "dataset_path": detail.get("dataset_profile", {}).get("dataset_path"),
                "target_column": detail.get("dataset_profile", {}).get("target_column"),
                "target_columns": detail.get("dataset_profile", {}).get("target_columns", []),
                "is_directory": bool(detail.get("dataset_profile", {}).get("is_directory", False)),
                "available_file_count": int(detail.get("dataset_profile", {}).get("available_file_count") or 0),
                "updated_at": datetime.fromtimestamp(task_dir.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return items


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, payload: Dict[str, Any]) -> str:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "payload": payload,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._jobs.get(job_id)
            return dict(payload) if payload else None


class DGWebApplication:
    def __init__(self) -> None:
        self.jobs = JobStore()

    @staticmethod
    def default_form_values() -> Dict[str, Any]:
        return {
            "query": "针对当前数据集构建一个时序预测模型",
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "dataset_name": "sample_dataset",
            "target_col": "",
            "input_feature_cols": "",
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "input_length": 96,
            "output_length": 24,
            "time_increment": 1,
            "normalization_policy": "off",
            "use_system_random": True,
            "max_iterations": 3,
        }

    def submit_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = self.jobs.create(payload)
        thread = threading.Thread(target=self._run_job, args=(job_id, payload), daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "queued"}

    @staticmethod
    def _task_dir(task_id: str) -> Path:
        return OUTPUT_DIR / task_id

    def _persist_task_metadata(self, task_id: Optional[str], payload: Dict[str, Any]) -> None:
        if not task_id:
            return
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / JOB_PAYLOAD_FILE, payload)

    def _run_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.jobs.update(job_id, status="running")
        stop_event = threading.Event()
        try:
            ensure_directories()
            initial_state = {
                "target_col": str(payload.get("target_col") or "").strip(),
                "input_feature_cols": str(payload.get("input_feature_cols") or "").strip(),
                "train_ratio": payload.get("train_ratio"),
                "val_ratio": payload.get("val_ratio"),
                "test_ratio": payload.get("test_ratio"),
                "input_length": payload.get("input_length"),
                "output_length": payload.get("output_length"),
                "time_increment": payload.get("time_increment"),
                "normalization_policy": str(payload.get("normalization_policy") or "off"),
                "use_system_random": bool(payload.get("use_system_random", True)),
                "max_iterations": int(payload.get("max_iterations", 3)),
            }
            state = DGGlobalState(load_existing=False, initial=initial_state)
            task_manager = DGTaskManager()
            orchestrator = DGOrchestrator(state=state, task_manager=task_manager)
            monitor = threading.Thread(target=self._monitor_job_progress, args=(job_id, state, stop_event), daemon=True)
            monitor.start()
            plan, _ = orchestrator.run(
                user_query=str(payload["query"]),
                dataset_path=str(payload["dataset_path"]),
                dataset_name=str(payload.get("dataset_name") or "custom"),
            )
            stop_event.set()
            snapshot = state.snapshot()
            task_id = snapshot.get("task_id")
            self._persist_task_metadata(task_id, payload)
            task_detail = load_task_detail(task_id) if task_id else None
            self.jobs.update(
                job_id,
                status="completed",
                result={
                    "plan": plan,
                    "task_id": task_id,
                    "summary_text": snapshot.get("summary_text"),
                    "task_detail": task_detail,
                },
            )
        except Exception as exc:
            stop_event.set()
            self.jobs.update(
                job_id,
                status="failed",
                error={"message": str(exc), "traceback": traceback.format_exc()},
            )

    def _monitor_job_progress(self, job_id: str, state: DGGlobalState, stop_event: threading.Event) -> None:
        while not stop_event.wait(0.8):
            snapshot = state.snapshot()
            task_id = snapshot.get("task_id")
            task_detail = None
            if task_id:
                try:
                    job_payload = (self.jobs.get(job_id) or {}).get("payload") or {}
                    self._persist_task_metadata(task_id, job_payload)
                    task_detail = load_task_detail(task_id)
                except FileNotFoundError:
                    task_detail = None
            self.jobs.update(
                job_id,
                result={
                    "plan": snapshot.get("current_plan") or snapshot.get("plan") or [],
                    "task_id": task_id,
                    "summary_text": snapshot.get("summary_text"),
                    "task_detail": task_detail,
                    "tasks": snapshot.get("tasks") or [],
                },
            )


class DGRequestHandler(BaseHTTPRequestHandler):
    server_version = "DGWebServer/1.0"

    @property
    def app(self) -> DGWebApplication:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/config":
                self._send_json({"defaults": self.app.default_form_values(), "recent_tasks": list_tasks()})
                return

            if parsed.path == "/api/tasks":
                limit = int(parse_qs(parsed.query).get("limit", ["20"])[0])
                self._send_json({"tasks": list_tasks(limit=limit)})
                return

            if parsed.path.startswith("/api/tasks/"):
                task_id = parsed.path.rsplit("/", 1)[-1]
                self._send_json(load_task_detail(task_id))
                return

            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                job = self.app.jobs.get(job_id)
                if not job:
                    self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                    return
                self._send_json(job)
                return

            self._serve_static(parsed.path)
        except FileNotFoundError as exc:
            self._send_json({"message": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"message": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"message": str(exc), "traceback": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                payload = self._read_json_body()
                normalized = self._normalize_payload(payload)
                result = self.app.submit_job(normalized)
                self._send_json(result, status=HTTPStatus.ACCEPTED)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except json.JSONDecodeError:
            self._send_json({"message": "Request body must be valid JSON"}, status=HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._send_json({"message": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"message": str(exc), "traceback": traceback.format_exc()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ratios_required = any(payload.get(name) not in (None, "") for name in ("train_ratio", "val_ratio", "test_ratio"))
        normalized = {
            "query": str(payload.get("query", "")).strip(),
            "dataset_path": str(payload.get("dataset_path", "")).strip(),
            "dataset_name": str(payload.get("dataset_name", "custom")).strip() or "custom",
            "target_col": str(payload.get("target_col", "")).strip(),
            "input_feature_cols": str(payload.get("input_feature_cols", "")).strip(),
            "train_ratio": self._coerce_float(payload.get("train_ratio")) if ratios_required else None,
            "val_ratio": self._coerce_float(payload.get("val_ratio")) if ratios_required else None,
            "test_ratio": self._coerce_float(payload.get("test_ratio")) if ratios_required else None,
            "input_length": self._coerce_int(payload.get("input_length")),
            "output_length": self._coerce_int(payload.get("output_length")),
            "time_increment": self._coerce_int(payload.get("time_increment")),
            "normalization_policy": str(payload.get("normalization_policy", "off")).strip() or "off",
            "use_system_random": self._coerce_bool(payload.get("use_system_random"), default=True),
            "max_iterations": int(payload.get("max_iterations") or 3),
        }
        if not normalized["query"]:
            raise ValueError("query is required")
        if not normalized["dataset_path"]:
            raise ValueError("dataset_path is required")
        return normalized

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _serve_static(self, request_path: str) -> None:
        path = request_path if request_path not in ("", "/") else "/index.html"
        target = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR not in target.parents and target != WEB_DIR / "index.html":
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class DGHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: DGWebApplication):
        super().__init__(server_address, DGRequestHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pipline web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    app = DGWebApplication()
    server = DGHTTPServer((args.host, args.port), app)
    print(f"Serving pipline UI on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
