import logging
import os
import subprocess
import sys
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

log = logging.getLogger("tools")


def _get_task_dir(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("task_dir", "/app/output")


@tool
def web_search(query: str) -> str:
    """Search the web for information relevant to the task."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results
        )
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return f"Search failed: {e}"


@tool
def write_file(
    filename: str,
    content: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Write content to a file in the task directory."""
    task_dir = _get_task_dir(config)
    os.makedirs(task_dir, exist_ok=True)
    safe_name = os.path.basename(filename)
    path = os.path.join(task_dir, safe_name)
    try:
        with open(path, "w") as f:
            f.write(content)
        log.info("wrote %s", path)
        return f"Written: {safe_name} ({len(content)} chars)"
    except OSError as e:
        return f"Error writing {safe_name}: {e}"


@tool
def read_file(
    filename: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Read a file from the task directory."""
    task_dir = _get_task_dir(config)
    safe_name = os.path.basename(filename)
    path = os.path.join(task_dir, safe_name)
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        return f"Error reading {safe_name}: {e}"


@tool
def list_files(config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """List all files in the task directory."""
    task_dir = _get_task_dir(config)
    try:
        files = os.listdir(task_dir)
        return "\n".join(files) if files else "(empty)"
    except OSError as e:
        return f"Error listing files: {e}"


@tool
def run_python(
    filename: str = "main.py",
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Execute a Python file in the task directory. Returns stdout, stderr, and exit_code."""
    task_dir = _get_task_dir(config)
    safe_name = os.path.basename(filename)
    path = os.path.join(task_dir, safe_name)
    if not os.path.exists(path):
        return f"File not found: {safe_name}\nexit_code: 1"
    try:
        result = subprocess.run(
            [sys.executable, safe_name],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = []
        if result.stdout.strip():
            output.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            output.append(f"stderr:\n{result.stderr.strip()}")
        output.append(f"exit_code: {result.returncode}")
        log.info(f"run_python stdout:\n{result.stdout.strip()}")
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return "stderr:\nexecution timed out after 30s\nexit_code: 1"
    except Exception as e:
        return f"stderr:\n{e}\nexit_code: 1"


@tool
def install_package(package: str) -> str:
    """Install a Python package using pip."""
    safe = package.strip().split()[0]  # one package, no shell injection
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", safe],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return f"Installed: {safe}"
        return f"pip error:\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return f"Timed out installing {safe}"
    except Exception as e:
        return f"Error: {e}"


ALL_TOOLS = [web_search, write_file, read_file, list_files, run_python, install_package]
