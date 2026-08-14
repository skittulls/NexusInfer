"""
NexusInfer — Celery Tasks

Defines the background tasks that Celery workers execute.

Task Lifecycle:
    1. API creates job record (PENDING) in PostgreSQL
    2. API dispatches `run_inference_task.delay(job_id, model_type, input_text)`
    3. Worker picks up the task from the 'inference' queue
    4. Worker opens a DB session and transitions job → PROCESSING
    5. Worker runs the real HuggingFace inference engine
    6. Worker transitions job → COMPLETED (with result) or FAILED (with error)
    7. Worker commits and closes the DB session
"""

import time
import logging

from celery import Task
from app.workers.celery_app import celery_app
from app.schemas.job import JobStatus, ModelType

logger = logging.getLogger(__name__)


class InferenceTask(Task):
    """
    Custom Celery Task base class for ML inference.

    Provides structured lifecycle logging. DB sessions are opened per-task
    via get_db_session() — each task gets its own session and commits
    independently, matching Celery's concurrency model.
    """

    name = "app.workers.tasks.run_inference_task"

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called by Celery when the task raises an unhandled exception."""
        job_id = args[0] if args else "unknown"
        logger.error(f"Task failed | job={job_id} | error={exc}")

        # Attempt to mark job as failed in DB (best effort)
        try:
            from app.core.database import get_db_session
            from app.services.job_service import JobService
            with get_db_session() as db:
                JobService(db).set_failed(job_id, str(exc))
        except Exception as e:
            logger.error(f"Could not persist failure state for job {job_id}: {e}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else "unknown"
        logger.warning(f"Task retrying | job={job_id} | error={exc}")

    def on_success(self, retval, task_id, args, kwargs):
        job_id = args[0] if args else "unknown"
        logger.info(f"Task succeeded | job={job_id}")


@celery_app.task(
    base=InferenceTask,
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
)
def run_inference_task(self, job_id: str, model_type: str, input_text: str):
    """
    Execute ML inference as a background Celery task.

    Opens its own DB session to update job state, runs the
    HuggingFace model (pre-loaded in worker memory), and persists
    the result back to PostgreSQL.

    Args:
        job_id: UUID of the job to process (persisted in PostgreSQL).
        model_type: The ML model to use (sentiment, summarization, ner).
        input_text: The text input for inference.

    Returns:
        dict with job_id, status, and processing_time_ms.
    """
    from app.core.database import get_db_session
    from app.services.job_service import JobService
    from app.services.inference import run_inference

    logger.info(
        f"Worker picked up job | job={job_id} | model={model_type} | "
        f"input_len={len(input_text)} | attempt={self.request.retries + 1}"
    )

    # Each state transition is its own short-lived session to minimize
    # the window a transaction is held open during long inference runs.

    # ── 1. Transition to PROCESSING ──
    try:
        with get_db_session() as db:
            JobService(db).update_status(job_id, JobStatus.PROCESSING)
    except Exception as e:
        logger.error(f"Failed to set PROCESSING state for job {job_id}: {e}")
        raise self.retry(exc=e)

    # ── 2. Run inference ──
    try:
        start = time.time()
        model_enum = ModelType(model_type)
        result = run_inference(model_enum, input_text)
        elapsed_ms = (time.time() - start) * 1000

    except Exception as exc:
        logger.exception(f"Inference failed for job {job_id}")

        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 2})")
            raise self.retry(exc=exc)

        # Final failure — persist error state
        with get_db_session() as db:
            JobService(db).set_failed(job_id, str(exc))
        return {"job_id": job_id, "status": "failed", "error": str(exc)}

    # ── 3. Persist result ──
    with get_db_session() as db:
        JobService(db).set_result(job_id, result, elapsed_ms)

    logger.info(
        f"Job complete | job={job_id} | model={model_type} | "
        f"time={elapsed_ms:.1f}ms"
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "processing_time_ms": round(elapsed_ms, 2),
    }
