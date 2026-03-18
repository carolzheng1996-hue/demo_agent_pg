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
    from .config import BASE_DIR, CONFIG_ALL_FILE, OUTPUT_DIR, build_runtime_state, ensure_directories, runtime_defaults
    from .orchestrator import DGOrchestrator
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
except ImportError:
    from config import BASE_DIR, CONFIG_ALL_FILE, OUTPUT_DIR, build_runtime_state, ensure_directories, runtime_defaults
    from orchestrator import DGOrchestrator
    from state import DGGlobalState
    from task_manager import DGTaskManager


WEB_DIR = BASE_DIR / "web"
DEFAULT_DATASET_PATH = ""
JOB_PAYLOAD_FILE = "job_payload.json"
TASK_PLAN_FILE = "task_plan.json"


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


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _list_artifacts(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {
            "name": path.name,
            "relative_path": _safe_relpath(path, root.parent),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def load_task_detail(task_id: str) -> Dict[str, Any]:
    task_dir = OUTPUT_DIR / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task not found: {task_id}")

    iteration_dirs = sorted([path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("iteration_")])
    iterations: List[Dict[str, Any]] = []
    latest_dataset_profile: Dict[str, Any] = {}
    task_type = "unknown"
    requires_modeling = False
    job_payload = _read_json(task_dir / JOB_PAYLOAD_FILE) or {}
    task_plan = _read_json(task_dir / TASK_PLAN_FILE) or {}
    plan_meta = task_plan.get("plan_meta") if isinstance(task_plan.get("plan_meta"), dict) else {}
    task_type = str(plan_meta.get("task_type") or "unknown")
    requires_modeling = bool(plan_meta.get("requires_modeling", False))
    plan = task_plan.get("plan") if isinstance(task_plan.get("plan"), list) else []

    for iteration_dir in iteration_dirs:
        iterations.append(
            {
                "iteration_id": iteration_dir.name,
                "plan": {"task_type": task_type, "plan": plan},
                "summary_markdown": _read_text(iteration_dir / "summary.md"),
                "steps": [],
                "artifacts": _list_artifacts(iteration_dir),
            }
        )

    final_summary = _read_text(task_dir / "final_summary.md")
    latest_dataset_profile = {
        "dataset_path": job_payload.get("dataset_path") or task_plan.get("dataset_path"),
        "target_column": job_payload.get("target_col"),
        "target_columns": [item.strip() for item in str(job_payload.get("target_col", "")).split(",") if item.strip()],
        "available_file_count": len([item.strip() for item in str(job_payload.get("unit", "")).split(",") if item.strip()]) or 0,
        "selected_units": [item.strip() for item in str(job_payload.get("unit", "")).split(",") if item.strip()],
        "start_stage": job_payload.get("start_stage") or plan_meta.get("start_stage"),
        "end_stage": job_payload.get("end_stage") or plan_meta.get("end_stage"),
        "skip_split": job_payload.get("skip_split"),
        "data_artifacts": _list_artifacts(task_dir / "data"),
    }
    iteration_history: Dict[str, Any] = {}
    cross_iteration_ensemble = None
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
        "plan": {"task_type": task_type, "plan": plan, "plan_meta": plan_meta},
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
            "config_all": str(CONFIG_ALL_FILE),
            **runtime_defaults(CONFIG_ALL_FILE),
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
            config_path = Path(str(payload.get("config_all") or CONFIG_ALL_FILE)).expanduser()
            initial_state = build_runtime_state(payload, config_path=config_path)
            initial_state.update(
                {
                    "runtime_config_path": str(config_path),
                    "runtime_config": runtime_defaults(config_path),
                }
            )
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
        config_path = Path(str(payload.get("config_all") or CONFIG_ALL_FILE)).expanduser()
        defaults = runtime_defaults(config_path)
        ratios_required = any(payload.get(name) not in (None, "") for name in ("train_ratio", "val_ratio", "test_ratio"))
        normalized = {
            "config_all": str(config_path),
            "query": str(payload.get("query", "")).strip(),
            "dataset_path": str(payload.get("dataset_path", "")).strip(),
            "dataset_name": str(payload.get("dataset_name", defaults["dataset_name"])).strip() or defaults["dataset_name"],
            "unit": str(payload.get("unit", defaults["unit"])).strip(),
            "formatter_unit": str(payload.get("formatter_unit", defaults["formatter_unit"])).strip(),
            "start_stage": str(payload.get("start_stage", defaults["start_stage"])).strip() or defaults["start_stage"],
            "end_stage": str(payload.get("end_stage", defaults["end_stage"])).strip() or defaults["end_stage"],
            "skip_split": self._coerce_bool(payload.get("skip_split"), default=bool(defaults["skip_split"])),
            "target_col": str(payload.get("target_col", defaults["target_col"])).strip(),
            "input_feature_cols": str(payload.get("input_feature_cols", defaults["input_feature_cols"])).strip(),
            "split_method": str(payload.get("split_method", defaults["split_method"])).strip() or defaults["split_method"],
            "split_cutoff_date": str(payload.get("split_cutoff_date", defaults["split_cutoff_date"])).strip(),
            "split_test_units": str(payload.get("split_test_units", defaults["split_test_units"])).strip(),
            "train_ratio": self._coerce_float(payload.get("train_ratio")) if ratios_required else float(defaults["train_ratio"]),
            "val_ratio": self._coerce_float(payload.get("val_ratio")) if ratios_required else float(defaults["val_ratio"]),
            "test_ratio": self._coerce_float(payload.get("test_ratio")) if ratios_required else float(defaults["test_ratio"]),
            "input_length": self._coerce_int(payload.get("input_length")) if payload.get("input_length") not in (None, "") else int(defaults["input_length"]),
            "output_length": self._coerce_int(payload.get("output_length")) if payload.get("output_length") not in (None, "") else int(defaults["output_length"]),
            "points_per_day": self._coerce_int(payload.get("points_per_day")) if payload.get("points_per_day") not in (None, "") else int(defaults["points_per_day"]),
            "time_increment": self._coerce_int(payload.get("time_increment")) if payload.get("time_increment") not in (None, "") else int(defaults["time_increment"]),
            "normalization_policy": str(payload.get("normalization_policy", defaults["normalization_policy"])).strip() or defaults["normalization_policy"],
            "use_system_random": self._coerce_bool(payload.get("use_system_random"), default=bool(defaults["use_system_random"])),
            "max_iterations": int(payload.get("max_iterations") or defaults["max_iterations"]),
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
