"""
Utility helpers: output persistence, session history, terminal ops.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path

import config

# ── Terminal clear ─────────────────────────────────────────────────────────────

def clear_terminal() -> None:
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


# ── Output persistence ────────────────────────────────────────────────────────

def save_task_output(status) -> Path:
    """
    Persist task stdout/stderr/metadata to OUTPUT_DIR/<task_id>/.
    Returns the directory path.
    """
    out_dir = config.OUTPUT_DIR / status.task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "task_id":    status.task_id,
        "status":     status.status,
        "success":    status.success,
        "iterations": status.iterations,
        "iteration":  status.iteration,
        "last_error": status.last_error,
        "saved_at":   datetime.utcnow().isoformat() + "Z",
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if status.stdout:
        (out_dir / "stdout.txt").write_text(status.stdout, encoding="utf-8")
    if status.stderr:
        (out_dir / "stderr.txt").write_text(status.stderr, encoding="utf-8")
    if status.last_error:
        (out_dir / "error.txt").write_text(status.last_error, encoding="utf-8")

    return out_dir


# ── Session history ───────────────────────────────────────────────────────────

class SessionHistory:
    """In-memory list of goals submitted this session + optional file persistence."""

    def __init__(self) -> None:
        self._entries: list[str] = []
        self._load_from_file()

    def _load_from_file(self) -> None:
        path = config.HISTORY_FILE
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                self._entries = [line for line in lines if line.strip()][-config.MAX_HISTORY:]
            except OSError:
                pass

    def add(self, goal: str) -> None:
        self._entries.append(goal)
        if len(self._entries) > config.MAX_HISTORY:
            self._entries = self._entries[-config.MAX_HISTORY:]
        self._flush()

    def _flush(self) -> None:
        try:
            config.HISTORY_FILE.write_text(
                "\n".join(self._entries) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def all(self) -> list[str]:
        return list(self._entries)

    def last(self, n: int = 20) -> list[str]:
        return self._entries[-n:]

    def clear(self) -> None:
        self._entries.clear()
        try:
            config.HISTORY_FILE.unlink(missing_ok=True)
        except OSError:
            pass
