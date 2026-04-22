"""
Archon CLI – minimal clean version (no rich panels)
"""

from __future__ import annotations

import os
import sys
import time

# ensure imports work

sys.path.insert(0, os.path.dirname(__file__))

import config
import api as backend
import ui
import utils

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style as PTStyle

    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    _HAS_PROMPT_TOOLKIT = False


class ArchonREPL:
    def __init__(self) -> None:
        self.history = utils.SessionHistory()
        self._build_prompt_session()

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
        try:
            if self._session:
                return self._session.prompt("archon > ")
            return input("archon > ")
        except (EOFError, KeyboardInterrupt):
            return None

    def _do_goal(self, goal: str) -> None:
        self.history.add(goal)

        try:
            task_id = backend.run_task(goal)
        except backend.ArchonAPIError as exc:
            print(f"Error: {exc}")
            return

        print("Running...")

        start = time.time()

        try:
            final = backend.poll_until_done(task_id)
        except backend.ArchonAPIError as exc:
            print(f"Error: {exc}")
            return

        elapsed = time.time() - start

        print(f"\n⏱ {elapsed:.2f}s\n")

        output = final.stdout or final.stderr or ""

        if output:
            print(output.strip())
        else:
            print("No output")

        print()

        # optional save
        try:
            out_dir = utils.save_task_output(final)
            print(f"Saved → {out_dir}")
        except Exception:
            pass

    def run(self) -> None:
        print("⚡ Archon CLI (minimal mode)\n")

        while True:
            raw = self._read_input()

            if raw is None:
                print("Goodbye 👋")
                break

            line = raw.strip()
            if not line:
                continue

            if line in ("/exit", "/quit"):
                print("Goodbye 👋")
                break

            self._do_goal(line)


def main() -> None:
    ArchonREPL().run()


if __name__ == "__main__":
    main()
