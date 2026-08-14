"""
NexusInfer — Celery Tasks

Defines the background tasks that Celery workers execute.
Each task is a unit of work consumed from the Redis broker.

Task Lifecycle:
    1. API receives POST /jobs/submit
    2. API creates job record (PENDING) in the job store
    3. API dispatches `run_inference_task.delay(job_id, model_type, input_text)`
    4. Celery worker picks up the task from the 'inference' queue
    5. Worker transitions job → PROCESSING
    6. Worker runs the real HuggingFace inference engine
    7. Worker transitions job → COMPLETED (with result) or FAILED (with error)
"""

import time
import logging

from celery import Task
from app.workers.celery_app import celery_app
from app.schemas.job import JobStatus, ModelType
from app.services.job_service import job_store

logger = logging.getLogger(__name__)


class InferenceTask(Task):
    """
    Custom Celery Task base class for inference.

    Provides lifecycle hooks for logging and error handling.
    The model_manager is accessed via the singleton in inference.py,
    which was pre-loaded during worker_process_init.
    """

    name = "app.workers.tasks.run_inference_task"

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when the task raises an exception."""
        job_id = args[0] if args else "unknown"
        logger.error(f"Task failed for job {job_id}: {exc}")
        job_store.set_failed(job_id, str(exc))

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when the task is retried."""
        job_id = args[0] if args else "unknown"
        logger.warning(f"Task retrying for job {job_id}: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when the task completes successfully."""
        job_id = args[0] if args else "unknown"
        logger.info(f"Task completed for job {job_id}")


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

    Uses real HuggingFace transformer models via the inference engine.
    Models are pre-loaded in worker memory (no cold-start per request).

    Args:
        job_id: UUID of the job to process.
        model_type: The ML model to use (sentiment, summarization, ner).
        input_text: The text input for inference.

    Returns:
        dict with job_id, status, and processing_time_ms.
    """
    from app.services.inference import run_inference

    logger.info(
        f"Worker picked up job {job_id} | model={model_type} | "
        f"input_len={len(input_text)} | attempt={self.request.retries + 1}"
    )

    # ── Transition to PROCESSING ──
    job_store.update_status(job_id, JobStatus.PROCESSING)

    try:
        # ── Run real ML inference ──
        start = time.time()
        model_enum = ModelType(model_type)
        result = run_inference(model_enum, input_text)
        elapsed_ms = (time.time() - start) * 1000

        # ── Store result ──
        job_store.set_result(job_id, result, elapsed_ms)

        logger.info(
            f"Job {job_id} completed | model={model_type} | "
            f"time={elapsed_ms:.1f}ms"
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "processing_time_ms": round(elapsed_ms, 2),
        }

    except Exception as exc:
        logger.exception(f"Inference failed for job {job_id}")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 2})")
            raise self.retry(exc=exc)

        # Final failure
        job_store.set_failed(job_id, str(exc))
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(exc),
        }
