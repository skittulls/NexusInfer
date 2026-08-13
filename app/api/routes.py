"""
NexusInfer — API Route Handlers

Defines all REST endpoints for the inference API.
Routes are thin — they validate input, delegate to the job service,
and format the response. No business logic lives here.
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
    return HealthResponse(
        status="healthy",
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
        "In Day 1, inference runs synchronously; from Day 2 onwards, "
        "jobs are dispatched to Celery workers via Redis."
    ),
)
async def submit_job(request: JobSubmitRequest):
    # 1. Create the job record (PENDING)
    response = job_store.create_job(request)
    job_id = response.job_id

    # 2. Day 1: Run inference synchronously (blocking).
    #    Day 2+: This block is replaced with a Celery task dispatch:
    #            celery_app.send_task("run_inference", args=[job_id])
    try:
        job_store.update_status(job_id, JobStatus.PROCESSING)

        start = time.time()
        result = run_mock_inference(request.model_type, request.input_text)
        elapsed_ms = (time.time() - start) * 1000

        job_store.set_result(job_id, result, elapsed_ms)

        # Update the response to reflect completion
        response.status = JobStatus.COMPLETED
        response.message = (
            f"Job completed synchronously in {elapsed_ms:.1f}ms. "
            f"(Async dispatch available from Day 2)"
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
