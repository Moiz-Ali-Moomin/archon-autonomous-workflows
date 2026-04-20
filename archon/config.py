"""
Runtime configuration – loaded once at import time.
All values can be overridden via environment variables or a local .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from archon/ until we find a .env file (project root)
_here = Path(__file__).parent
for _candidate in [_here, _here.parent]:
    _env_file = _candidate / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        break

# ── Connection ──────────────────────────────────────────────────────────────
API_URL: str = os.environ.get("ARCHON_API_URL", "http://localhost:8000").rstrip("/")
API_KEY: str = os.environ.get("ARCHON_API_KEY") or os.environ.get("AGENT_API_KEY", "")

# ── Polling behaviour ────────────────────────────────────────────────────────
POLL_INTERVAL: float = float(os.environ.get("ARCHON_POLL_INTERVAL", "2"))
REQUEST_TIMEOUT: int = int(os.environ.get("ARCHON_REQUEST_TIMEOUT", "15"))
MAX_POLL_SECONDS: int = int(os.environ.get("ARCHON_MAX_POLL_SECONDS", "300"))

# ── Local output ─────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(os.environ.get("ARCHON_OUTPUT_DIR", "output"))

# ── Session history ───────────────────────────────────────────────────────────
HISTORY_FILE: Path = Path.home() / ".archon_history"
MAX_HISTORY: int = 500

# ── Themes ───────────────────────────────────────────────────────────────────
THEMES: dict[str, dict] = {
    "default": {
        "banner": "bold cyan",
        "prompt": "bold green",
        "success": "bold green",
        "failure": "bold red",
        "running": "bold yellow",
        "queued": "bold blue",
        "info": "dim white",
        "separator": "dim cyan",
        "output": "white",
        "cmd": "bold magenta",
    },
    "dracula": {
        "banner": "bold #ff79c6",
        "prompt": "bold #50fa7b",
        "success": "bold #50fa7b",
        "failure": "bold #ff5555",
        "running": "bold #f1fa8c",
        "queued": "bold #6272a4",
        "info": "#6272a4",
        "separator": "#44475a",
        "output": "#f8f8f2",
        "cmd": "bold #bd93f9",
    },
    "monokai": {
        "banner": "bold #a6e22e",
        "prompt": "bold #e6db74",
        "success": "bold #a6e22e",
        "failure": "bold #f92672",
        "running": "bold #fd971f",
        "queued": "bold #66d9ef",
        "info": "#75715e",
        "separator": "#3e3d32",
        "output": "#f8f8f2",
        "cmd": "bold #ae81ff",
    },
    "light": {
        "banner": "bold blue",
        "prompt": "bold dark_green",
        "success": "bold dark_green",
        "failure": "bold dark_red",
        "running": "bold dark_orange",
        "queued": "bold navy_blue",
        "info": "grey50",
        "separator": "grey70",
        "output": "black",
        "cmd": "bold purple",
    },
}

ACTIVE_THEME: str = os.environ.get("ARCHON_THEME", "default")
