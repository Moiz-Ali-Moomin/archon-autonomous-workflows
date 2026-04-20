import logging
import os

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from graph import store_task_graph
from redis_client import update_task_state
from workflow import init_db, run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "agent_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
)


@celery_app.task(
    bind=True,
    name="tasks.run_agent_task",
    time_limit=600,
    soft_time_limit=540,
)
def run_agent_task(self, task_id: str, goal: str):
    log.info("starting task_id=%s", task_id)
    update_task_state(task_id, {"status": "running", "iteration": 0, "last_error": None})

    def on_iteration(iteration: int, last_error):
        update_task_state(
            task_id,
            {"status": "running", "iteration": iteration, "last_error": last_error},
        )

    try:
        init_db()
        result = run_workflow(goal, task_id=task_id, on_iteration=on_iteration)

        status = "success" if result["success"] else "failed"
        execution = result.get("execution", {})

        update_task_state(
            task_id,
            {
                "status": status,
                "success": result["success"],
                "iterations": result.get("iterations", 0),
                "stdout": execution.get("stdout", "")[:1000],
                "stderr": execution.get("stderr", "")[:500],
            },
        )

        store_task_graph(task_id, goal, result)
        log.info("completed task_id=%s status=%s", task_id, status)
        return result

    except SoftTimeLimitExceeded:
        log.error("task_id=%s hit soft time limit", task_id)
        update_task_state(task_id, {"status": "timeout", "last_error": "task exceeded time limit"})
        raise

    except Exception as exc:
        log.exception("task_id=%s failed: %s", task_id, exc)
        update_task_state(task_id, {"status": "error", "last_error": str(exc)})
        raise self.retry(exc=exc, countdown=10, max_retries=1) from exc
