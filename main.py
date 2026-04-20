import uuid
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from redis_client import get_task_state, set_task_state, get_redis
from tasks import run_agent_task
from db import check_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("api")

MAX_GOAL_LENGTH = 500
API_KEY         = os.environ["AGENT_API_KEY"]
api_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)
limiter         = Limiter(key_func=get_remote_address)


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API starting up")
    yield
    log.info("API shutting down")


app = FastAPI(title="Gemma Agent API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


class RunRequest(BaseModel):
    goal: str

    @field_validator("goal")
    @classmethod
    def goal_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("goal must not be empty")
        if len(v) > MAX_GOAL_LENGTH:
            raise ValueError(f"goal must be under {MAX_GOAL_LENGTH} characters")
        return v


class RunResponse(BaseModel):
    task_id: str
    status: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    iteration: int | None = None
    last_error: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    success: bool | None = None
    iterations: int | None = None


@app.post("/run", response_model=RunResponse, status_code=202)
@limiter.limit("10/minute")
async def run_agent(request: Request, body: RunRequest, _key: str = Security(verify_api_key)):
    task_id = str(uuid.uuid4())
    set_task_state(task_id, {"status": "queued", "goal": body.goal})
    run_agent_task.apply_async(args=[task_id, body.goal], task_id=task_id)
    log.info("queued task_id=%s", task_id)
    return RunResponse(task_id=task_id, status="queued")


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str, _key: str = Security(verify_api_key)):
    state = get_task_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="task not found")
    return StatusResponse(
        task_id=task_id,
        status=state.get("status", "unknown"),
        iteration=state.get("iteration"),
        last_error=state.get("last_error"),
        stdout=state.get("stdout"),
        stderr=state.get("stderr"),
        success=state.get("success"),
        iterations=state.get("iterations"),
    )


@app.get("/health")
async def health():
    checks = {}
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    try:
        check_db()
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
