"""
Archon CLI – main entry point.

Run directly:   python cli.py
Or via package: python -m archon
"""

from __future__ import annotations

import os
import sys

# ── ensure archon/ is importable when run directly ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config
import api as backend
import ui
import utils

# ── prompt_toolkit for readline-style input (optional) ───────────────────────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style as PTStyle

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


# ── REPL ──────────────────────────────────────────────────────────────────────


class ArchonREPL:
    def __init__(self) -> None:
        self.history = utils.SessionHistory()
        self._build_prompt_session()

    # ── Input layer ──────────────────────────────────────────────────────────

    def _build_prompt_session(self) -> None:
        if not _HAS_PROMPT_TOOLKIT:
            self._session = None
            return
        pt_style = PTStyle.from_dict({"": "#00ff00 bold"})
        self._session = PromptSession(
            history=FileHistory(str(config.HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=pt_style,
        )

    def _read_input(self) -> str | None:
        prompt_str = ui.get_prompt()
        try:
            if self._session:
                return self._session.prompt(prompt_str)
            else:
                return input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            return None

    # ── Command dispatch ─────────────────────────────────────────────────────

    def _handle_command(self, raw: str) -> bool:
        """
        Handle /commands. Returns True if the REPL should keep running,
        False to exit.
        """
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            ui.info("Goodbye. 👋")
            return False

        if cmd == "/help":
            ui.print_help()
            return True

        if cmd == "/clear":
            utils.clear_terminal()
            ui.print_banner()
            return True

        if cmd == "/history":
            ui.print_history(self.history.last(20))
            return True

        if cmd == "/health":
            self._do_health()
            return True

        if cmd == "/status":
            if not arg:
                ui.error("Usage: /status <task_id>")
            else:
                self._do_status(arg)
            return True

        if cmd == "/theme":
            if not arg:
                current = config.ACTIVE_THEME
                ui.info(
                    f"Current theme: [bold]{current}[/bold]  "
                    f"Available: {', '.join(ui.list_themes())}"
                )
            else:
                if ui.set_theme(arg):
                    ui.success(f"Theme set to '{arg}'.")
                else:
                    ui.error(f"Unknown theme '{arg}'. Available: {', '.join(ui.list_themes())}")
            return True

        # Unknown /command
        ui.warn(f"Unknown command '{cmd}'. Type /help for a list of commands.")
        return True

    # ── Goal execution ────────────────────────────────────────────────────────

    def _do_goal(self, goal: str) -> None:
        self.history.add(goal)

        # 1. Submit
        try:
            task_id = backend.run_task(goal)
        except backend.ArchonAPIError as exc:
            ui.error(str(exc))
            return

        ui.print_task_id(task_id)

        # 2. Poll with spinner
        last_seen_status: str | None = None

        def on_update(status) -> None:
            nonlocal last_seen_status
            if status.status != last_seen_status:
                last_seen_status = status.status
                task_desc = f"[bold]{status.status.upper()}[/bold]" + (
                    f"  iteration {status.iteration}" if status.iteration else ""
                )
                progress.update(spinner_task, description=task_desc)
                ui.print_status_update(status)

        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(style=ui._t("running")),
            TextColumn("[progress.description]{task.description}", style=ui._t("info")),
            TimeElapsedColumn(),
            console=ui.console,
            transient=False,
        ) as progress:
            spinner_task = progress.add_task("Waiting for agent…", total=None)

            try:
                final = backend.poll_until_done(
                    task_id,
                    on_update=on_update,
                )
            except backend.ArchonAPIError as exc:
                ui.error(str(exc))
                return

        # 3. Display result
        ui.print_task_result(final)

        output_text = final.stdout or final.stderr or "No output"
        if len(output_text) > 1000:
            output_text = output_text[:1000] + "\n... [truncated]"

        print(f"\nFinal Output:\n{output_text}\n")

        # 4. Save output
        try:
            out_dir = utils.save_task_output(final)
            ui.info(f"Output saved → {out_dir}")
        except OSError as exc:
            ui.warn(f"Could not save output: {exc}")

    # ── Single-task status lookup ─────────────────────────────────────────────

    def _do_status(self, task_id: str) -> None:
        try:
            status = backend.get_status(task_id)
        except backend.ArchonAPIError as exc:
            ui.error(str(exc))
            return
        ui.print_task_result(status)

    # ── Health check ─────────────────────────────────────────────────────────

    def _do_health(self) -> None:
        try:
            data = backend.health_check()
        except backend.ArchonAPIError as exc:
            ui.error(str(exc))
            return
        ui.print_health(data)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        utils.clear_terminal()
        ui.print_banner()

        # Warn if API key is missing
        if not config.API_KEY:
            ui.warn(
                "No API key found. Set ARCHON_API_KEY or AGENT_API_KEY "
                "in your environment or .env file."
            )

        while True:
            try:
                raw = self._read_input()
            except KeyboardInterrupt:
                print()
                ui.info("Use /exit to quit.")
                continue

            if raw is None:
                # EOF (Ctrl-D)
                ui.info("Goodbye. 👋")
                break

            line = raw.strip()
            if not line:
                continue

            if line.lower() == "clear":
                utils.clear_terminal()
                ui.print_banner()
            elif line.startswith("/"):
                keep_running = self._handle_command(line)
                if not keep_running:
                    break
            else:
                self._do_goal(line)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    try:
        ArchonREPL().run()
    except KeyboardInterrupt:
        ui.info("\nInterrupted. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
