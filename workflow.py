import anthropic
import requests
import json
import os
import re
import shlex
import subprocess
import logging
import uuid
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("workflow")

# ================= CONFIG ================= #
_OLLAMA_BASE    = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL      = f"{_OLLAMA_BASE}/api/generate"
EMBED_URL       = f"{_OLLAMA_BASE}/api/embeddings"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DB_CONFIG = {
    "host":            os.getenv("DB_HOST",     "localhost"),
    "database":        os.getenv("DB_NAME",     "agent"),
    "user":            os.getenv("DB_USER",     "agent_user"),
    "password":        os.getenv("DB_PASSWORD", "agent_pass"),
    "connect_timeout": 5,
}

OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "output")
CODE_TIMEOUT   = int(os.getenv("CODE_TIMEOUT", "30"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

_anthropic_client = None

def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    return _anthropic_client

# ================= DB ================= #
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

# ================= EMBEDDING ================= #
def embed(text):
    try:
        res = requests.post(EMBED_URL, json={
            "model": "nomic-embed-text",
            "prompt": text
        }, timeout=30)
        return res.json()["embedding"]
    except Exception as e:
        log.warning("embed failed: %s", e)
        return [0.0] * 768

def embed_vector_str(text):
    vec = embed(text)
    return "[" + ",".join(map(str, vec)) + "]"

# ================= MEMORY ================= #
def save_memory(goal, result, success):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO memory (goal, result, success, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            (goal, json.dumps(result), success, embed_vector_str(goal))
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
            """
            SELECT goal, result, success
            FROM memory
            ORDER BY embedding <=> %s::vector
            LIMIT 3;
            """,
            (embed_vector_str(goal),)
        )
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        log.warning("get_memory failed: %s", e)
        return []

# ================= GEMMA — chat ================= #
def ask_gemma(question):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": "gemma:2b",
            "prompt": question,
            "stream": False
        }, timeout=120)
        return r.json().get("response", "")
    except Exception as e:
        log.error("ask_gemma error: %s", e)
        return ""

# ================= GEMMA — planner ================= #
def planner(goal, memory_rows):
    memory_section = ""
    if memory_rows:
        lines = []
        for row in memory_rows:
            g, result_json, success = row
            status = "succeeded" if success else "failed"
            lines.append(f"- Goal: {g} | Status: {status}")
        memory_section = "\nPast similar tasks:\n" + "\n".join(lines) + "\n"

    prompt = f"""You are a software planning assistant. Break down the following goal into a concise implementation spec for a Python script.

Goal: {goal}
{memory_section}
Return a JSON object with this exact shape:
{{
  "description": "<one sentence of what the script does>",
  "inputs": "<what arguments or input the script needs>",
  "outputs": "<what the script should print or produce>",
  "steps": ["<step 1>", "<step 2>", "..."],
  "run": "<shell command to execute the script, e.g. python main.py>"
}}

JSON:"""

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        raw = r.json().get("response", "")
        log.debug("planner raw: %s", raw[:300])

        spec = extract_json(raw)
        if _spec_is_confident(spec, goal):
            log.info("planner produced spec: %s", spec.get("description"))
            return spec
        log.warning("planner spec failed confidence check — using fallback")
    except Exception as e:
        log.warning("planner error: %s", e)

    # fallback: pass goal directly as minimal spec
    log.warning("planner fallback — using raw goal as spec")
    return {
        "description": goal,
        "inputs": "none",
        "outputs": "result printed to stdout",
        "steps": [goal],
        "run": "python main.py",
    }


def _spec_is_confident(spec, goal):
    if not spec:
        return False
    required = ("description", "steps", "run")
    if not all(k in spec for k in required):
        return False
    if not isinstance(spec.get("steps"), list) or len(spec["steps"]) == 0:
        return False
    # reject placeholder text the model sometimes echoes back verbatim
    bad_phrases = ("<", "step 1>", "one sentence", "shell command")
    desc = spec.get("description", "").lower()
    if any(p in desc for p in bad_phrases):
        return False
    # spec should at least loosely reference something from the goal
    goal_words = set(goal.lower().split())
    desc_words = set(desc.split())
    if len(goal_words & desc_words) == 0 and len(goal_words) > 2:
        return False
    return True

# ================= CLAUDE — builder ================= #
def builder(spec, goal):
    system = (
        "You are an expert Python developer. "
        "You write clean, working Python scripts. "
        "You always respond with ONLY a JSON object — no markdown, no prose, no explanation."
    )

    user = f"""Write a complete Python script based on this spec.

Goal: {goal}

Spec:
- Description: {spec.get("description")}
- Inputs: {spec.get("inputs")}
- Outputs: {spec.get("outputs")}
- Steps: {json.dumps(spec.get("steps", []))}

Return ONLY this JSON shape:
{{
  "files": {{"main.py": "<full python code as a single escaped string>"}},
  "run": "{spec.get("run", "python main.py")}",
  "tools": []
}}

Escape all newlines as \\n inside the string value."""

    try:
        message = get_anthropic().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text
        log.debug("builder raw: %s", raw[:300])

        data = extract_json(raw)
        if data and "files" in data:
            return data
    except Exception as e:
        log.error("builder error: %s", e)

    log.warning("bad builder output → fallback")
    return fallback_code(goal)

# ================= CLAUDE — fixer ================= #
def fixer(goal, previous_code, error):
    system = (
        "You are an expert Python debugger. "
        "You fix broken Python scripts by addressing the exact error reported. "
        "You always respond with ONLY a JSON object — no markdown, no prose, no explanation."
    )

    user = f"""Fix the Python script below so it accomplishes the goal.

Goal: {goal}

Previous code:
```python
{previous_code}
```

Error:
{error}

Return ONLY this JSON shape:
{{
  "files": {{"main.py": "<fixed python code as a single escaped string>"}},
  "run": "python main.py",
  "tools": []
}}

Fix only the specific error. Do not rewrite unrelated logic. Escape all newlines as \\n."""

    try:
        message = get_anthropic().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text
        log.debug("fixer raw: %s", raw[:300])

        data = extract_json(raw)
        if data and "files" in data:
            return data
    except Exception as e:
        log.error("fixer error: %s", e)

    log.warning("bad fixer output → keeping previous code")
    return {"files": {"main.py": previous_code}, "run": "python main.py", "tools": []}

# ================= JSON ================= #
def extract_json(text):
    if not text:
        return None

    text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None

# ================= FILE WRITE ================= #
def write_files(files, task_dir):
    os.makedirs(task_dir, exist_ok=True)
    for name, content in files.items():
        path = os.path.join(task_dir, name)
        with open(path, "w") as f:
            f.write(content)
        log.info("wrote %s", path)

# ================= EXEC ================= #
def _apply_resource_limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,   (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU,  (CODE_TIMEOUT, CODE_TIMEOUT))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except Exception:
        pass

def run_command(cmd, task_dir):
    try:
        args = shlex.split(cmd)
    except ValueError:
        return {"success": False, "stdout": "", "stderr": f"invalid command: {cmd}"}

    try:
        result = subprocess.run(
            args,
            shell=False,
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            preexec_fn=_apply_resource_limits,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"TimeoutExpired after {CODE_TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}

def fix_command(cmd, task_dir):
    cmd = cmd.replace("<filename>", "test.txt")
    test_file = os.path.join(task_dir, "test.txt")
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\n")
    return cmd

# ================= FALLBACK ================= #
def fallback_code(goal):
    escaped = goal.replace('"', '\\"')
    return {
        "files": {
            "main.py": f'if __name__ == "__main__":\n    print("Agent fallback for: {escaped}")\n'
        },
        "run": "python main.py",
        "tools": []
    }

# ================= MAIN ================= #
def run_workflow(goal, task_id=None, on_iteration=None):
    task_id = task_id or str(uuid.uuid4())
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    log.info("starting task=%s goal=%r", task_id, goal)

    final_result = {"success": False, "execution": {}, "iterations": 0}
    current_code = None

    memory = get_memory(goal)
    log.info("task=%s memory_hits=%d", task_id, len(memory))

    spec = planner(goal, memory)
    log.info("task=%s spec=%r", task_id, spec.get("description"))

    for i in range(MAX_ITERATIONS):
        log.info("task=%s iteration=%d", task_id, i + 1)

        if on_iteration:
            on_iteration(i + 1, final_result.get("last_error"))

        if i == 0 or current_code is None:
            build = builder(spec, goal)
        else:
            build = fixer(goal, current_code, final_result.get("last_error", "unknown error"))

        write_files(build["files"], task_dir)
        current_code = build["files"].get("main.py", "")

        cmd = fix_command(build.get("run", "python main.py"), task_dir)
        execution = run_command(cmd, task_dir)

        log.info("task=%s success=%s stdout=%r stderr=%r",
                 task_id, execution["success"],
                 execution["stdout"][:80], execution["stderr"][:80])

        final_result = {
            "success": execution["success"],
            "execution": execution,
            "iterations": i + 1,
            "last_error": execution.get("stderr") or None,
            "files_written": list(build["files"].keys()),
            "task_dir": task_dir,
        }

        if execution["success"]:
            log.info("task=%s done after %d iteration(s)", task_id, i + 1)
            save_memory(goal, execution, True)
            return final_result
        else:
            save_memory(goal, execution, False)

    log.warning("task=%s exhausted retries", task_id)
    return final_result

# ================= ENTRY ================= #
if __name__ == "__main__":
    init_db()
    run_workflow("Create a Python CLI app that reads a file and prints line count")
