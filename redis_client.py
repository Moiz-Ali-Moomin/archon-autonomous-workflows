import os
import json
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_client = None


def get_redis():
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def set_task_state(task_id: str, state: dict, ttl: int = 86400):
    get_redis().setex(f"task:{task_id}", ttl, json.dumps(state))


def get_task_state(task_id: str):
    raw = get_redis().get(f"task:{task_id}")
    return json.loads(raw) if raw else None


def update_task_state(task_id: str, updates: dict, ttl: int = 86400):
    state = get_task_state(task_id) or {}
    state.update(updates)
    set_task_state(task_id, state, ttl)
