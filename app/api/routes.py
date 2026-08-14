"""
NexusInfer — API Route Handlers

Defines all REST endpoints for the inference API.
Routes are thin — they validate input, delegate to the job service,
and format the response. No business logic lives here.

Day 4: Job service now backed by PostgreSQL via SQLAlchemy.
       Sessions are injected per-request via FastAPI's Depends().
"""

import time
import logging

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from app.schemas.job import (
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
    HealthResponse,
)
from app.core.database import get_db
from app.services.job_service import JobService
from app.services.inference import run_inference
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = time.time()


def _check_redis_connection() -> bool:
    """Check if Redis is reachable for Celery broker."""
    try:
        import redis
        settings = get_settings()
        r = redis.Redis.from_url(settings.CELERY_BROKER_URL, socket_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


# ──────────────────────────── Health Check ────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns API health status, Redis connectivity, loaded models, and queue depth.",
)
async def health_check(db: Session = Depends(get_db)):
    settings = get_settings()
    redis_ok = _check_redis_connection()

    from app.services.model_manager import model_manager
    service = JobService(db)

    return HealthResponse(
        status="healthy" if redis_ok else "degraded (Redis unavailable, sync mode)",
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _start_time, 2),
        jobs_in_queue=service.pending_count,
        redis_connected=redis_ok,
        models_loaded=model_manager.loaded_models,
    )


# ──────────────────────────── Submit Job ────────────────────────────


@router.post(
    "/jobs/submit",
    response_model=JobSubmitResponse,
    status_code=202,
    tags=["Jobs"],
    summary="Submit an inference job",
    description=(
        "Submits a new ML inference job. "
        "When Redis is available, the job is dispatched to a Celery worker asynchronously. "
        "Falls back to synchronous in-request inference if Redis is unavailable."
    ),
)
async def submit_job(request: JobSubmitRequest, db: Session = Depends(get_db)):
    service = JobService(db)

    # 1. Persist the job (PENDING state)
    response = service.create_job(request)
    job_id = response.job_id

    # 2. Attempt async dispatch via Celery
    if _check_redis_connection():
        try:
            from app.workers.tasks import run_inference_task

            run_inference_task.apply_async(
                args=[job_id, request.model_type.value, request.input_text],
                queue="inference",
                priority=request.priority,
            )

            response.message = (
                f"Job queued for async processing. "
                f"Model: {request.model_type.value}. "
                f"Poll GET /api/v1/jobs/{job_id} for results."
            )
            logger.info(f"Job {job_id} dispatched to Celery")
            return response

        except Exception as e:
            logger.warning(f"Celery dispatch failed for {job_id}, falling back to sync: {e}")

    # 3. Sync fallback — run inference in-request
    logger.info(f"Job {job_id} executing synchronously (Redis unavailable)")
    try:
        service.update_status(job_id, JobStatus.PROCESSING)

        start = time.time()
        result = run_inference(request.model_type, request.input_text)
        elapsed_ms = (time.time() - start) * 1000

        service.set_result(job_id, result, elapsed_ms)

        response.status = JobStatus.COMPLETED
        response.message = (
            f"Job completed synchronously in {elapsed_ms:.1f}ms. "
            f"(Start Redis + Celery worker for async processing)"
        )

    except Exception as e:
        service.set_failed(job_id, str(e))
        response.status = JobStatus.FAILED
        response.message = f"Job failed: {str(e)}"
        logger.exception(f"Inference failed for job {job_id}")

    return response


# ──────────────────────────── Get Job Status ────────────────────────────


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Get job status and result",
    description="Retrieve the current status and result of a specific inference job.",
)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    service = JobService(db)
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )
    return job


# ──────────────────────────── List Jobs ────────────────────────────


@router.get(
    "/jobs",
    response_model=JobListResponse,
    tags=["Jobs"],
    summary="List all jobs",
    description="Returns a paginated list of all inference jobs, newest first.",
)
async def list_jobs(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status: JobStatus | None = Query(default=None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    service = JobService(db)
    return service.list_jobs(
        page=page,
        page_size=page_size,
        status_filter=status,
    )
