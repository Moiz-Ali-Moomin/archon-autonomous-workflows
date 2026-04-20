"""
All rendering logic: banner, prompt, spinners, status panels, output formatting.
Nothing in here makes HTTP calls – it only receives data and displays it.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box

import config

console = Console()

# ── Theme helper ─────────────────────────────────────────────────────────────

def _t(key: str) -> str:
    """Return the colour/style for the active theme + key."""
    theme = config.THEMES.get(config.ACTIVE_THEME, config.THEMES["default"])
    return theme.get(key, "white")


def set_theme(name: str) -> bool:
    if name not in config.THEMES:
        return False
    config.ACTIVE_THEME = name
    return True


def list_themes() -> list[str]:
    return list(config.THEMES.keys())


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
   ___              __
  / _ |  __________/ /  ___  ___
 / __ | / __/ __/ _ \/ _ \/ _ \
/_/ |_|/_/  \__/_//_/\___/_//_/
"""

def print_banner() -> None:
    term_width = shutil.get_terminal_size((80, 20)).columns
    text = Text(BANNER, style=_t("banner"), justify="left")
    console.print(text)
    console.print(
        f"  [bold]⚡ Archon AI v1.0[/bold] – [dim]Autonomous Coding Agent[/dim]",
        style=_t("banner"),
    )
    console.print(
        f"  Connected to [underline]{config.API_URL}[/underline]\n",
        style=_t("info"),
    )
    print_separator()


def print_separator(char: str = "─") -> None:
    width = shutil.get_terminal_size((80, 20)).columns
    console.print(char * width, style=_t("separator"))


# ── Prompt ────────────────────────────────────────────────────────────────────

def get_prompt() -> str:
    """Return the styled prompt string (used by prompt_toolkit / input())."""
    return "archon > "


# ── Status helpers ────────────────────────────────────────────────────────────

_STATUS_ICONS = {
    "queued":  ("⏳", "queued"),
    "running": ("⚙ ", "running"),
    "success": ("✓ ", "success"),
    "failure": ("✗ ", "failure"),
    "error":   ("✗ ", "failure"),
    "unknown": ("?  ", "info"),
}


def _status_style(status: str) -> tuple[str, str]:
    icon, theme_key = _STATUS_ICONS.get(status.lower(), ("•", "info"))
    return icon, _t(theme_key)


def status_text(status: str) -> Text:
    icon, style = _status_style(status)
    t = Text()
    t.append(f"{icon} {status.upper()}", style=style)
    return t


# ── Spinner / live polling ────────────────────────────────────────────────────

def make_spinner(task_id: str) -> Progress:
    return Progress(
        SpinnerColumn(style=_t("running")),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ── Task result panel ─────────────────────────────────────────────────────────

def print_task_result(status) -> None:
    """Render a rich Panel with the final task result."""
    print_separator()

    terminal_success = status.status == "success" or status.success is True
    border_style = _t("success") if terminal_success else _t("failure")
    icon = "✓" if terminal_success else "✗"
    title = f"{icon} Task {status.task_id[:8]}… – {status.status.upper()}"

    rows: list[tuple[str, str]] = []

    if status.iterations is not None:
        rows.append(("Iterations", str(status.iterations)))
    if status.iteration is not None and status.iterations is None:
        rows.append(("Last iteration", str(status.iteration)))

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style=_t("info"), no_wrap=True)
    table.add_column("Value", style=_t("output"))

    for k, v in rows:
        table.add_row(k, v)

    content_parts = []
    if rows:
        content_parts.append(table)

    if status.stdout:
        content_parts.append(_render_output_block("stdout", status.stdout))

    if status.stderr:
        content_parts.append(_render_output_block("stderr", status.stderr, error=True))

    if status.last_error:
        content_parts.append(_render_output_block("error", status.last_error, error=True))

    from rich.console import Group

    panel_content = Group(*content_parts) if content_parts else Text(
        "No output captured.", style=_t("info")
    )

    console.print(
        Panel(
            panel_content,
            title=title,
            border_style=border_style,
            expand=True,
            padding=(1, 2),
        )
    )
    print_separator()


def _render_output_block(label: str, text: str, error: bool = False) -> Panel:
    # Try to detect JSON and pretty-print it
    stripped = text.strip()
    rendered = None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            rendered = Syntax(
                json.dumps(parsed, indent=2),
                "json",
                theme="monokai",
                word_wrap=True,
            )
        except json.JSONDecodeError:
            pass

    if rendered is None:
        style = _t("failure") if error else _t("output")
        rendered = Text(stripped, style=style)

    return Panel(
        rendered,
        title=f"[{_t('info')}]{label}[/{_t('info')}]",
        border_style=_t("failure") if error else _t("separator"),
        expand=True,
    )


# ── Help table ────────────────────────────────────────────────────────────────

def print_help() -> None:
    print_separator()
    table = Table(
        title="[bold]Available Commands[/bold]",
        box=box.ROUNDED,
        border_style=_t("separator"),
        show_header=True,
        header_style=_t("banner"),
    )
    table.add_column("Command", style=_t("cmd"), no_wrap=True, min_width=26)
    table.add_column("Description", style=_t("output"))

    commands = [
        ("/help",               "Show this help message"),
        ("/exit  [dim]or /quit[/dim]", "Exit Archon"),
        ("/clear",              "Clear the terminal"),
        ("/status <task_id>",   "Check the status of a task manually"),
        ("/history",            "Show session command history"),
        ("/health",             "Ping the backend health endpoint"),
        ("/theme <name>",       f"Switch theme  ({', '.join(list_themes())})"),
        ("  <any text>",        "Submit a natural language goal to the agent"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    print_separator()


# ── Inline status update (used during polling) ────────────────────────────────

def print_status_update(status) -> None:
    """Single-line status update printed inside the spinner loop."""
    icon, style = _status_style(status.status)
    parts = [f"[{style}]{icon} {status.status.upper()}[/{style}]"]
    if status.iteration is not None:
        parts.append(f"[{_t('info')}] iteration {status.iteration}[/{_t('info')}]")
    if status.last_error:
        short = status.last_error[:80].replace("\n", " ")
        parts.append(f"  [{_t('failure')}]⚠ {short}[/{_t('failure')}]")
    console.print("  " + "".join(parts))


# ── Simple message helpers ────────────────────────────────────────────────────

def info(msg: str) -> None:
    console.print(f"  [{_t('info')}]{msg}[/{_t('info')}]")


def success(msg: str) -> None:
    console.print(f"  [{_t('success')}]✓ {msg}[/{_t('success')}]")


def error(msg: str) -> None:
    console.print(f"  [{_t('failure')}]✗ {msg}[/{_t('failure')}]")


def warn(msg: str) -> None:
    console.print(f"  [{_t('running')}]⚠ {msg}[/{_t('running')}]")


def print_task_id(task_id: str) -> None:
    console.print(
        f"\n  [{_t('info')}]Task queued →[/{_t('info')}] "
        f"[bold]{task_id}[/bold]\n"
    )


# ── Health display ────────────────────────────────────────────────────────────

def print_health(data: dict) -> None:
    print_separator()
    overall = data.get("status", "unknown")
    icon = "✓" if overall == "ok" else "⚠"
    style = _t("success") if overall == "ok" else _t("failure")

    table = Table(box=box.ROUNDED, border_style=_t("separator"), show_header=True,
                  header_style=_t("banner"), title=f"[{style}]{icon} Backend Health[/{style}]")
    table.add_column("Service", style=_t("cmd"), no_wrap=True)
    table.add_column("Status", style=_t("output"))

    for service, status_val in data.get("checks", {}).items():
        s = _t("success") if status_val == "ok" else _t("failure")
        table.add_row(service, f"[{s}]{status_val}[/{s}]")

    console.print(table)
    print_separator()


# ── History display ───────────────────────────────────────────────────────────

def print_history(entries: list[str]) -> None:
    print_separator()
    if not entries:
        info("No history yet.")
        return
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("#", style=_t("info"), justify="right", no_wrap=True)
    table.add_column("Goal", style=_t("output"))
    for i, entry in enumerate(entries, 1):
        table.add_row(str(i), entry)
    console.print(table)
    print_separator()
