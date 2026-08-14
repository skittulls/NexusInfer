"""
NexusInfer — Job Service (PostgreSQL / SQLAlchemy backed)

Business logic layer for job management.
Replaced the in-memory dict (Day 1) with a persistent SQLAlchemy session.

This service is the single source of truth for all job state transitions.
It is storage-agnostic from the caller's perspective — routes and workers
interact through the same interface regardless of the backing store.

Usage in FastAPI routes (dependency injection):
    def submit_job(request: JobSubmitRequest, db: Session = Depends(get_db)):
        service = JobService(db)
        return service.create_job(request)

Usage in Celery workers (context manager):
    with get_db_session() as db:
        service = JobService(db)
        service.update_status(job_id, JobStatus.PROCESSING)
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.job import Job
from app.schemas.job import (
    ModelType,
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
)

logger = logging.getLogger(__name__)


def _to_response(job: Job) -> JobStatusResponse:
    """Convert a SQLAlchemy Job ORM object to a Pydantic response model."""
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        model_type=job.model_type,
        input_text=job.input_text,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        processing_time_ms=job.processing_time_ms,
    )


class JobService:
    """
    Persistent job store backed by SQLAlchemy.

    Each instance is bound to a single database session.
    Sessions are managed by the caller (FastAPI dependency or Celery context manager).
    """

    def __init__(self, db: Session):
        self.db = db

    # ──────────────────────────── Create ────────────────────────────

    def create_job(self, request: JobSubmitRequest) -> JobSubmitResponse:
        """
        Persist a new job in PENDING state and return its metadata.
        """
        job = Job(
            id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            model_type=request.model_type,
            input_text=request.input_text,
            priority=request.priority,
        )

        self.db.add(job)
        self.db.flush()     # Write to DB, generate created_at via server_default

        logger.info(f"Job created: {job.id} | model={request.model_type.value}")

        return JobSubmitResponse(
            job_id=job.id,
            status=JobStatus.PENDING,
            message=f"Job submitted successfully. Model: {request.model_type.value}",
            created_at=job.created_at or datetime.now(timezone.utc),
        )

    # ──────────────────────────── Read ────────────────────────────

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """Retrieve a job by ID. Returns None if not found."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return None
        return _to_response(job)

    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[JobStatus] = None,
    ) -> JobListResponse:
        """List all jobs with pagination and optional status filtering."""
        query = self.db.query(Job)

        if status_filter:
            query = query.filter(Job.status == status_filter)

        total = query.count()

        jobs = (
            query
            .order_by(Job.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return JobListResponse(
            jobs=[_to_response(j) for j in jobs],
            total=total,
            page=page,
            page_size=page_size,
        )

    # ──────────────────────────── Update ────────────────────────────

    def update_status(self, job_id: str, status: JobStatus) -> bool:
        """Transition a job to a new status."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return False

        job.status = status

        if status == JobStatus.PROCESSING:
            job.started_at = datetime.now(timezone.utc)

        self.db.flush()
        logger.info(f"Job {job_id} → {status.value}")
        return True

    def set_result(
        self, job_id: str, result: dict, processing_time_ms: float
    ) -> bool:
        """Mark a job as completed with its inference result."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return False

        job.status = JobStatus.COMPLETED
        job.result = result
        job.completed_at = datetime.now(timezone.utc)
        job.processing_time_ms = processing_time_ms

        self.db.flush()
        logger.info(f"Job {job_id} completed in {processing_time_ms:.1f}ms")
        return True

    def set_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed with an error message."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return False

        job.status = JobStatus.FAILED
        job.error = error
        job.completed_at = datetime.now(timezone.utc)

        self.db.flush()
        logger.error(f"Job {job_id} failed: {error}")
        return True

    # ──────────────────────────── Aggregates ────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of jobs currently in PENDING state."""
        return (
            self.db.query(func.count(Job.id))
            .filter(Job.status == JobStatus.PENDING)
            .scalar() or 0
        )

    @property
    def total_count(self) -> int:
        """Total number of jobs in the database."""
        return self.db.query(func.count(Job.id)).scalar() or 0
