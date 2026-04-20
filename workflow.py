import json
import logging
import os
import threading
import uuid
from typing import Annotated, Literal, TypedDict

import psycopg2
import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from tools import ALL_TOOLS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("workflow")

# ── Config ─────────────────────────────────────────────────────────────────────
_OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_URL = f"{_OLLAMA_BASE}/api/embeddings"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "agent"),
    "user": os.getenv("DB_USER", "agent_user"),
    "password": os.getenv("DB_PASSWORD", "agent_pass"),
    "connect_timeout": 5,
}

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))


def _pg_conn_string() -> str:
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )


# ── State ──────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    goal: str
    task_id: str
    task_dir: str
    messages: Annotated[list, add_messages]
    memory_rows: list
    success: bool
    last_error: str | None
    iteration: int
    files_written: list


# ── DB ─────────────────────────────────────────────────────────────────────────


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS memory (
        id SERIAL PRIMARY KEY,
        goal TEXT,
        result TEXT,
        success BOOLEAN,
        embedding vector(768)
    );
    """)
    conn.commit()
    conn.close()
    log.info("DB schema ready")


# ── Embedding ──────────────────────────────────────────────────────────────────


def embed(text):
    try:
        res = requests.post(
            EMBED_URL,
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30,
        )
        return res.json()["embedding"]
    except Exception as e:
        log.warning("embed failed: %s", e)
        return [0.0] * 768


def embed_vector_str(text):
    return "[" + ",".join(map(str, embed(text))) + "]"


# ── Memory ─────────────────────────────────────────────────────────────────────


def save_memory(goal, result, success):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memory (goal, result, success, embedding) VALUES (%s, %s, %s, %s::vector)",
            (goal, json.dumps(result), success, embed_vector_str(goal)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("save_memory failed: %s", e)


def get_memory(goal):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT goal, result, success FROM memory ORDER BY embedding <=> %s::vector LIMIT 3",
            (embed_vector_str(goal),),
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        log.warning("get_memory failed: %s", e)
        return []


# ── Gemma chat (used by /ask endpoint) ────────────────────────────────────────


def ask_gemma(question):
    try:
        ollama_url = f"{_OLLAMA_BASE}/api/generate"
        r = requests.post(
            ollama_url,
            json={"model": "gemma:2b", "prompt": question, "stream": False},
            timeout=120,
        )
        return r.json().get("response", "")
    except Exception as e:
        log.error("ask_gemma error: %s", e)
        return ""


# ── LLM with tools ─────────────────────────────────────────────────────────────

_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = ChatAnthropic(
                    model=ANTHROPIC_MODEL,
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    max_tokens=4096,
                ).bind_tools(ALL_TOOLS)
    return _llm


# ── Success detection ──────────────────────────────────────────────────────────


def _last_run_python_result(messages: list) -> tuple[bool | None, str | None]:
    """Parse the last run_python ToolMessage for exit_code. Returns (success, error)."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "run_python":
            continue
        content = msg.content or ""
        if "exit_code: 0" in content:
            return True, None
        lines = content.splitlines()
        stderr_lines: list[str] = []
        in_stderr = False
        for line in lines:
            if line.startswith("stderr:"):
                in_stderr = True
                continue
            if in_stderr and line.startswith("exit_code:"):
                break
            if in_stderr:
                stderr_lines.append(line)
        error = "\n".join(stderr_lines).strip()[:200] if stderr_lines else content[:200]
        return False, error
    return None, None


# ── LangGraph nodes ────────────────────────────────────────────────────────────


def node_fetch_memory(state: AgentState) -> dict:
    rows = get_memory(state["goal"])
    log.info("task=%s memory_hits=%d", state["task_id"], len(rows))
    return {"memory_rows": rows}


def node_agent(state: AgentState) -> Command[Literal["tools", "save"]]:
    iteration = state["iteration"] + 1

    memory_section = ""
    if state.get("memory_rows"):
        lines = [
            f"- Goal: {g} | Status: {'succeeded' if s else 'failed'}"
            for g, _, s in state["memory_rows"]
        ]
        memory_section = "\n\nPast similar tasks:\n" + "\n".join(lines)

    system_content = (
        "You are an autonomous coding agent. "
        "Your task directory is available via tools — use write_file, run_python, "
        "web_search, read_file, list_files, and install_package to accomplish the goal. "
        "Always write code to main.py, then run it. "
        "If there is an error, fix it and run again. "
        "When the goal is complete and the code runs successfully, stop calling tools."
        + memory_section
    )

    # Always rebuild with a fresh SystemMessage; preserve all non-system history.
    prior = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    if not prior:
        prior = [HumanMessage(content=f"Goal: {state['goal']}")]
    messages = [SystemMessage(content=system_content)] + prior

    log.info("task=%s agent iteration=%d", state["task_id"], iteration)
    response = get_llm().invoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    goto: Literal["tools", "save"] = (
        "tools" if has_tool_calls and iteration < MAX_ITERATIONS else "save"
    )

    return Command(update={"messages": [response], "iteration": iteration}, goto=goto)


def node_save(state: AgentState) -> dict:
    messages = state.get("messages", [])

    ran_success, run_error = _last_run_python_result(messages)

    if ran_success is not None:
        success = ran_success
        last_error = run_error
    else:
        success = False
        last_error = "Agent did not execute any code"

    result = {
        "success": success,
        "stdout": "",
        "stderr": last_error or "",
        "iteration": state["iteration"],
    }
    save_memory(state["goal"], result, success)

    if success:
        log.info("task=%s done after %d agent step(s)", state["task_id"], state["iteration"])
    else:
        log.warning(
            "task=%s finished unsuccessfully after %d step(s): %s",
            state["task_id"],
            state["iteration"],
            last_error,
        )

    return {"success": success, "last_error": last_error}


# ── Graph assembly ─────────────────────────────────────────────────────────────


def _build_graph(checkpointer):
    g = StateGraph(AgentState)

    g.add_node("fetch_memory", node_fetch_memory)
    g.add_node("agent", node_agent)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("save", node_save)

    g.add_edge(START, "fetch_memory")
    g.add_edge("fetch_memory", "agent")
    # agent returns Command(goto=...) — no conditional edge needed
    g.add_edge("tools", "agent")
    g.add_edge("save", END)

    return g.compile(checkpointer=checkpointer)


_graph = None
_checkpointer = None
_graph_lock = threading.Lock()


def _get_graph():
    global _graph, _checkpointer
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _checkpointer = PostgresSaver.from_conn_string(_pg_conn_string())
                _checkpointer.setup()
                log.info("checkpoint store ready")
                _graph = _build_graph(_checkpointer)
    return _graph


# ── Public entry point ─────────────────────────────────────────────────────────


def run_workflow(goal: str, task_id: str = None, on_iteration=None) -> dict:
    task_id = task_id or str(uuid.uuid4())
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    log.info("starting task=%s goal=%r", task_id, goal)

    initial: AgentState = {
        "goal": goal,
        "task_id": task_id,
        "task_dir": task_dir,
        "messages": [],
        "memory_rows": [],
        "success": False,
        "last_error": None,
        "iteration": 0,
        "files_written": [],
    }

    # thread_id scopes the checkpoint to this task run
    run_config = {
        "configurable": {
            "thread_id": task_id,
            "task_dir": task_dir,
        }
    }

    graph = _get_graph()
    last_iteration = 0

    # Stream node-level updates — call on_iteration as each agent step completes
    for event in graph.stream(initial, config=run_config, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "agent":
                new_iter = updates.get("iteration", last_iteration)
                if on_iteration and new_iter > last_iteration:
                    on_iteration(new_iter, updates.get("last_error"))
                last_iteration = new_iter

    # Read final state from checkpointer (complete, not partial)
    final = graph.get_state(run_config).values

    files_written = []
    try:
        files_written = os.listdir(task_dir)
    except OSError:
        pass

    return {
        "success": final.get("success", False),
        "execution": {
            "success": final.get("success", False),
            "stdout": "",
            "stderr": final.get("last_error") or "",
        },
        "iterations": final.get("iteration", 0),
        "last_error": final.get("last_error"),
        "files_written": files_written,
        "task_dir": task_dir,
    }


# ── Dev entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    run_workflow("Create a Python CLI app that reads a file and prints line count")
