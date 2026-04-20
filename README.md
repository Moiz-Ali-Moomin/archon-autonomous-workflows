# Gemma Agent Platform

A distributed autonomous coding agent. Send it a goal over HTTP, it writes Python code to solve it, executes it, and self-corrects on failure — all asynchronously.

## Architecture

```
HTTP Client
    │
    ▼
┌─────────┐    enqueue     ┌──────────────┐
│  FastAPI │ ─────────────▶│  Redis       │
│  :8000   │               │  (broker +   │
└─────────┘                │   task state)│
    │                      └──────┬───────┘
    │ GET /status                 │ dequeue
    ▼                             ▼
┌─────────┐              ┌──────────────────┐
│  Redis  │◀─ status ────│  Celery Worker   │
└─────────┘              │                  │
                         │  workflow.py     │
                         │  ┌────────────┐  │
                         │  │ builder()  │  │──▶ Ollama (gemma:2b)
                         │  │ fixer()    │  │──▶ Ollama (gemma:2b)
                         │  │ run_code() │  │
                         │  └────────────┘  │
                         └──────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌────────────┐         ┌────────────┐
             │ PostgreSQL │         │   Neo4j    │
             │ (pgvector) │         │ (GraphRAG) │
             │ long-term  │         │ Goal→File  │
             │ memory     │         │ Error→Fix  │
             └────────────┘         └────────────┘
```

## Services

| Service    | Port  | Purpose                              |
|------------|-------|--------------------------------------|
| API        | 8000  | REST interface                       |
| Worker     | —     | Celery task executor                 |
| Flower     | 5555  | Celery task monitoring dashboard     |
| Redis      | 6379  | Message broker + short-term state    |
| PostgreSQL | 5432  | Long-term memory (pgvector)          |
| Neo4j      | 7687  | Graph memory (Goal/File/Error nodes) |
| Neo4j UI   | 7474  | Browser-based graph explorer         |
| Ollama     | 11434 | Local LLM inference                  |

## Prerequisites

- Docker + Docker Compose
- 4 GB RAM minimum (8 GB recommended)
- The VPS this was built on: Debian, 4 GB RAM, Hetzner FSN1

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo>
cd gemma-agent

# 2. Create your env file
cp .env.example .env
# Edit .env if your passwords or ports differ

# 3. Build and start all services
docker compose up --build -d

# 4. Pull required models (one-time, ~2 GB)
docker compose exec ollama ollama pull gemma:2b
docker compose exec ollama ollama pull nomic-embed-text

# 5. Verify everything is healthy
docker compose ps
curl http://localhost:8000/health
```

## Running without Docker (VPS direct)

```bash
# Requires: PostgreSQL with pgvector, Redis, Neo4j, Ollama already running
cp .env.example .env   # fill in localhost credentials

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Init DB and run a single task directly
python workflow.py

# Start the API
uvicorn main:app --host 0.0.0.0 --port 8000

# Start a worker (separate terminal)
celery -A tasks.celery_app worker --loglevel=info
```

## API

### Submit a task

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "write a function that returns the nth fibonacci number"}'
```

Response:
```json
{"task_id": "3f2a1c...", "status": "queued"}
```

### Poll for status

```bash
curl http://localhost:8000/status/3f2a1c...
```

Response:
```json
{
  "task_id": "3f2a1c...",
  "status": "success",
  "iterations": 2,
  "stdout": "55",
  "stderr": null,
  "success": true
}
```

Possible `status` values: `queued` → `running` → `success` / `failed` / `timeout` / `error`

### Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "checks": {"redis": "ok", "postgres": "ok"}}
```

## Monitoring

Open **http://localhost:5555** for the Flower dashboard — live view of queued, active, and completed tasks.

Open **http://localhost:7474** for the Neo4j browser — explore the graph of goals, generated files, and errors.

## Environment Variables

| Variable          | Required | Default                        | Description                                      |
|-------------------|----------|--------------------------------|--------------------------------------------------|
| `OLLAMA_URL`      | Yes      | `http://localhost:11434`       | Ollama base URL — paths are appended automatically |
| `DB_HOST`         | Yes      | `localhost`                    | PostgreSQL host                                  |
| `DB_NAME`         | Yes      | `agent`                        | PostgreSQL database name                         |
| `DB_USER`         | Yes      | `agent_user`                   | PostgreSQL user                                  |
| `DB_PASSWORD`     | Yes      | `agent_pass`                   | PostgreSQL password                              |
| `REDIS_URL`       | Yes      | `redis://localhost:6379/0`     | Redis connection URL                             |
| `NEO4J_URI`       | Yes      | `bolt://localhost:7687`        | Neo4j bolt URI                                   |
| `NEO4J_USER`      | Yes      | `neo4j`                        | Neo4j username                                   |
| `NEO4J_PASSWORD`  | Yes      | `agentpassword`                | Neo4j password                                   |
| `MAX_ITERATIONS`  | No       | `3`                            | Max generate→fix cycles per task                 |
| `CODE_TIMEOUT`    | No       | `30`                           | Seconds before generated code is killed          |
| `OUTPUT_DIR`      | No       | `output`                       | Root dir for per-task generated files            |
| `AGENT_API_KEY`       | **Yes**  | —                              | Secret key for `X-API-Key` header on all requests — generate with `openssl rand -hex 32` |
| `ANTHROPIC_API_KEY`   | **Yes**  | —                              | Claude API key — used by builder and fixer       |
| `ANTHROPIC_MODEL`     | No       | `claude-sonnet-4-6`            | Claude model used for code generation            |

## Project Structure

```
.
├── workflow.py       # Core agent loop: build → run → fix → retry
├── main.py           # FastAPI: POST /run, GET /status/{id}, GET /health
├── tasks.py          # Celery worker wrapping workflow.py
├── redis_client.py   # Task state read/write (short-term memory)
├── graph.py          # Neo4j: Goal→File, Goal→Error relationships
├── db.py             # PostgreSQL connection helper + health check
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── output/           # Per-task generated files (gitignored)
    └── <task_id>/
        └── main.py
```

## How the agent works

1. `POST /run` queues a Celery task and returns a `task_id` immediately
2. Worker calls `run_workflow(goal, task_id)`
3. **Iteration 1** — `builder()` prompts Gemma to generate code as JSON
4. Code is written to `output/<task_id>/main.py` and executed in a sandboxed subprocess
5. On success — result saved to PostgreSQL (pgvector), graph updated in Neo4j
6. On failure — `fixer()` sends the exact previous code + stderr back to Gemma and asks it to patch the specific error
7. Repeats up to `MAX_ITERATIONS` times
8. Status is written to Redis after every iteration — `GET /status` reads from there live
