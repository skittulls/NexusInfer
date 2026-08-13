"""
NexusInfer — Job Service

Business logic layer for job management.
Implements an in-memory job store for Day 1 (replaced with PostgreSQL on Day 4).

This service is the single source of truth for job state transitions.
All job mutations go through this layer, making it easy to swap the
backing store without touching the API or worker code.
"""

import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Optional

from app.schemas.job import (
    ModelType,
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
)

logger = logging.getLogger(__name__)


class JobStore:
    """
    In-memory job store.

    Thread-safe for Day 1 usage with uvicorn's single-process mode.
    Will be replaced with SQLAlchemy + PostgreSQL on Day 4.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}

    def create_job(self, request: JobSubmitRequest) -> JobSubmitResponse:
        """
        Create a new job entry and return its ID.
        The job starts in PENDING state.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        job_record = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "model_type": request.model_type,
            "input_text": request.input_text,
            "priority": request.priority,
            "result": None,
            "error": None,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "processing_time_ms": None,
        }

        self._jobs[job_id] = job_record
        logger.info(f"Job created: {job_id} | model={request.model_type.value}")

        return JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Job submitted successfully. Model: {request.model_type.value}",
            created_at=now,
        )

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """Retrieve a job by its ID. Returns None if not found."""
        record = self._jobs.get(job_id)
        if record is None:
            return None

        return JobStatusResponse(**record)

    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[JobStatus] = None,
    ) -> JobListResponse:
        """List all jobs with pagination and optional status filtering."""
        all_jobs = list(self._jobs.values())

        # Apply status filter
        if status_filter:
            all_jobs = [j for j in all_jobs if j["status"] == status_filter]

        # Sort by creation time (newest first)
        all_jobs.sort(key=lambda j: j["created_at"], reverse=True)

        total = len(all_jobs)
        start = (page - 1) * page_size
        end = start + page_size
        page_jobs = all_jobs[start:end]

        return JobListResponse(
            jobs=[JobStatusResponse(**j) for j in page_jobs],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_status(self, job_id: str, status: JobStatus) -> bool:
        """Transition a job to a new status."""
        if job_id not in self._jobs:
            return False

        self._jobs[job_id]["status"] = status

        if status == JobStatus.PROCESSING:
            self._jobs[job_id]["started_at"] = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} → {status.value}")
        return True

    def set_result(self, job_id: str, result: dict, processing_time_ms: float) -> bool:
        """Mark a job as completed with its inference result."""
        if job_id not in self._jobs:
            return False

        now = datetime.now(timezone.utc)
        self._jobs[job_id].update({
            "status": JobStatus.COMPLETED,
            "result": result,
            "completed_at": now,
            "processing_time_ms": processing_time_ms,
        })

        logger.info(
            f"Job {job_id} completed in {processing_time_ms:.1f}ms"
        )
        return True

    def set_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed with an error message."""
        if job_id not in self._jobs:
            return False

        self._jobs[job_id].update({
            "status": JobStatus.FAILED,
            "error": error,
            "completed_at": datetime.now(timezone.utc),
        })

        logger.error(f"Job {job_id} failed: {error}")
        return True

    @property
    def pending_count(self) -> int:
        """Number of jobs currently in PENDING state."""
        return sum(
            1 for j in self._jobs.values() if j["status"] == JobStatus.PENDING
        )

    @property
    def total_count(self) -> int:
        """Total number of jobs in the store."""
        return len(self._jobs)


# ──────────────────── Singleton Instance ────────────────────
# Replaced with dependency injection + DB session on Day 4.
job_store = JobStore()
