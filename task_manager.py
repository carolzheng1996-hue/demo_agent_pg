from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import TASKS_FILE


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TaskManager:
    def __init__(self, persist_path: Path = TASKS_FILE):
        self._persist_path = persist_path
        self._tasks: List[Dict[str, str]] = []
        self.load()

    def load(self) -> None:
        if self._persist_path.exists():
            self._tasks = json.loads(self._persist_path.read_text(encoding="utf-8"))
        else:
            self._tasks = []

    def persist(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(self._tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_plan(self, plan: List[str]) -> None:
        self._tasks = [
            {
                "name": step,
                "status": "pending",
                "detail": "",
                "started_at": "",
                "finished_at": "",
            }
            for step in plan
        ]
        self.persist()

    def start(self, task_name: str) -> None:
        for item in self._tasks:
            if item["name"] == task_name:
                item["status"] = "in_progress"
                item["started_at"] = _now_iso()
        self.persist()

    def complete(self, task_name: str, detail: str = "ok") -> None:
        for item in self._tasks:
            if item["name"] == task_name:
                item["status"] = "done"
                item["detail"] = detail
                item["finished_at"] = _now_iso()
        self.persist()

    def fail(self, task_name: str, detail: str) -> None:
        for item in self._tasks:
            if item["name"] == task_name:
                item["status"] = "failed"
                item["detail"] = detail
                item["finished_at"] = _now_iso()
        self.persist()

    def list_tasks(self) -> List[Dict[str, str]]:
        return list(self._tasks)
