"""
NexusInfer — API Route Handlers

Defines all REST endpoints for the inference API.
Routes are thin — they validate input, delegate to the job service,
and format the response. No business logic lives here.

Day 2: Jobs are now dispatched to Celery workers via Redis.
       Falls back to synchronous execution if Redis is unavailable.
"""

import time
import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.job import (
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
    HealthResponse,
)
from app.services.job_service import job_store
from app.services.inference import run_mock_inference
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track server start time for uptime calculation
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
    description="Returns the API health status, version, and queue depth.",
)
async def health_check():
    settings = get_settings()
    redis_ok = _check_redis_connection()

    return HealthResponse(
        status="healthy" if redis_ok else "degraded (Redis unavailable, sync mode)",
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _start_time, 2),
        jobs_in_queue=job_store.pending_count,
    )


# ──────────────────────────── Submit Job ────────────────────────────


@router.post(
    "/jobs/submit",
    response_model=JobSubmitResponse,
    status_code=202,
    tags=["Jobs"],
    summary="Submit an inference job",
    description=(
        "Submits a new ML inference job to the processing queue. "
        "Returns a job ID that can be used to poll for results. "
        "Jobs are dispatched to Celery workers via Redis. "
        "Falls back to synchronous execution if Redis is unavailable."
    ),
)
async def submit_job(request: JobSubmitRequest):
    # 1. Create the job record (PENDING)
    response = job_store.create_job(request)
    job_id = response.job_id

    # 2. Try async dispatch via Celery
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
            logger.info(f"Job {job_id} dispatched to Celery (async)")
            return response

        except Exception as e:
            logger.warning(
                f"Celery dispatch failed for job {job_id}, "
                f"falling back to sync: {e}"
            )

    # 3. Fallback: synchronous execution (no Redis / Celery unavailable)
    logger.info(f"Job {job_id} executing synchronously (Redis unavailable)")
    try:
        job_store.update_status(job_id, JobStatus.PROCESSING)

        start = time.time()
        result = run_mock_inference(request.model_type, request.input_text)
        elapsed_ms = (time.time() - start) * 1000

        job_store.set_result(job_id, result, elapsed_ms)

        response.status = JobStatus.COMPLETED
        response.message = (
            f"Job completed synchronously in {elapsed_ms:.1f}ms. "
            f"(Start Redis + Celery worker for async processing)"
        )

    except Exception as e:
        job_store.set_failed(job_id, str(e))
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
async def get_job_status(job_id: str):
    job = job_store.get_job(job_id)
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
):
    return job_store.list_jobs(
        page=page,
        page_size=page_size,
        status_filter=status,
    )
