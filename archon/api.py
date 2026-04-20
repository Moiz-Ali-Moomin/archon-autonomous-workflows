"""
HTTP client for the Archon / Gemma-VPS backend.
All functions raise ArchonAPIError on unrecoverable failures so the
caller never has to inspect raw HTTP exceptions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

import config


class ArchonAPIError(Exception):
    """Raised when the backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TaskStatus:
    task_id: str
    status: str
    iteration: Optional[int] = None
    last_error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    success: Optional[bool] = None
    iterations: Optional[int] = None


def _headers() -> dict[str, str]:
    if not config.API_KEY:
        raise ArchonAPIError(
            "API key not set. Export ARCHON_API_KEY or AGENT_API_KEY, "
            "or add it to your .env file."
        )
    return {"X-API-Key": config.API_KEY, "Content-Type": "application/json"}


def _url(path: str) -> str:
    return f"{config.API_URL}{path}"


def run_task(goal: str) -> str:
    """Submit a goal to the backend. Returns task_id."""
    try:
        resp = requests.post(
            _url("/run"),
            json={"goal": goal},
            headers=_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
    except ConnectionError:
        raise ArchonAPIError(f"Cannot reach backend at {config.API_URL}. Is the server running?")
    except Timeout:
        raise ArchonAPIError(f"Request timed out after {config.REQUEST_TIMEOUT}s.")
    except RequestException as exc:
        raise ArchonAPIError(f"Network error: {exc}")

    if resp.status_code == 401:
        raise ArchonAPIError("Authentication failed – check your API key.", 401)
    if resp.status_code == 422:
        detail = resp.json().get("detail", resp.text)
        raise ArchonAPIError(f"Validation error: {detail}", 422)
    if resp.status_code == 429:
        raise ArchonAPIError("Rate limit exceeded. Wait a moment and try again.", 429)
    if not resp.ok:
        raise ArchonAPIError(f"Backend error {resp.status_code}: {resp.text}", resp.status_code)

    return resp.json()["task_id"]


def get_status(task_id: str) -> TaskStatus:
    """Fetch the current status of a task."""
    try:
        resp = requests.get(
            _url(f"/status/{task_id}"),
            headers=_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
    except ConnectionError:
        raise ArchonAPIError(f"Lost connection to backend at {config.API_URL}.")
    except Timeout:
        raise ArchonAPIError(f"Status request timed out after {config.REQUEST_TIMEOUT}s.")
    except RequestException as exc:
        raise ArchonAPIError(f"Network error: {exc}")

    if resp.status_code == 404:
        raise ArchonAPIError(f"Task '{task_id}' not found.", 404)
    if resp.status_code == 401:
        raise ArchonAPIError("Authentication failed – check your API key.", 401)
    if not resp.ok:
        raise ArchonAPIError(f"Backend error {resp.status_code}: {resp.text}", resp.status_code)

    data = resp.json()
    return TaskStatus(
        task_id=data["task_id"],
        status=data.get("status", "unknown"),
        iteration=data.get("iteration"),
        last_error=data.get("last_error"),
        stdout=data.get("stdout"),
        stderr=data.get("stderr"),
        success=data.get("success"),
        iterations=data.get("iterations"),
    )


def health_check() -> dict:
    """Call /health and return the raw dict. Raises ArchonAPIError on failure."""
    try:
        resp = requests.get(
            _url("/health"),
            headers=_headers(),
            timeout=config.REQUEST_TIMEOUT,
        )
    except ConnectionError:
        raise ArchonAPIError(f"Cannot reach backend at {config.API_URL}.")
    except Timeout:
        raise ArchonAPIError("Health check timed out.")
    except RequestException as exc:
        raise ArchonAPIError(f"Network error: {exc}")

    if not resp.ok:
        raise ArchonAPIError(f"Health endpoint returned {resp.status_code}.", resp.status_code)

    return resp.json()


def poll_until_done(
    task_id: str,
    on_update,
    interval: float = config.POLL_INTERVAL,
    max_seconds: int = config.MAX_POLL_SECONDS,
) -> TaskStatus:
    """
    Poll GET /status/{task_id} every `interval` seconds until the task
    reaches a terminal state (success / failure).

    `on_update(status: TaskStatus)` is called after every successful poll
    so the caller can refresh the UI.

    Raises ArchonAPIError if polling times out or an unrecoverable error occurs.
    """
    terminal = {"success", "failure", "error"}
    deadline = time.monotonic() + max_seconds
    last_status: TaskStatus | None = None

    while time.monotonic() < deadline:
        status = get_status(task_id)
        on_update(status)
        last_status = status
        if status.status in terminal:
            return status
        time.sleep(interval)

    raise ArchonAPIError(
        f"Task '{task_id}' did not finish within {max_seconds}s. "
        "Use /status <task_id> to check manually."
    )
