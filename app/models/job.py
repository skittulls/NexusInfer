"""
NexusInfer — SQLAlchemy ORM Job Model

Defines the `jobs` table schema.
Maps 1:1 to the JobStatusResponse Pydantic schema — the DB is the
authoritative source of truth, replacing the in-memory dict from Day 1.
"""

import uuid
from sqlalchemy import (
    Column, String, Text, Integer, Float,
    DateTime, Enum as SAEnum, JSON
)
from sqlalchemy.sql import func

from app.core.database import Base
from app.schemas.job import JobStatus, ModelType


class Job(Base):
    """
    ORM model representing a single inference job.

    State machine enforced at the service layer:
        PENDING → PROCESSING → COMPLETED
                             → FAILED
    """

    __tablename__ = "jobs"

    # ── Primary Key ──
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key",
    )

    # ── Job Metadata ──
    status = Column(
        SAEnum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,                     # Index for status-filtered queries
        comment="Current lifecycle state",
    )
    model_type = Column(
        SAEnum(ModelType),
        nullable=False,
        comment="Which HuggingFace model to use",
    )
    input_text = Column(
        Text,
        nullable=False,
        comment="Raw input text submitted by the client",
    )
    priority = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Queue priority (0-10)",
    )

    # ── Result / Error ──
    result = Column(
        JSON,
        nullable=True,
        comment="Inference output (populated on COMPLETED)",
    )
    error = Column(
        Text,
        nullable=True,
        comment="Error message (populated on FAILED)",
    )

    # ── Timestamps ──
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,                     # Index for recency-sorted listing
        comment="When the job was created",
    )
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When a worker picked up the job",
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the job reached a terminal state",
    )

    # ── Performance ──
    processing_time_ms = Column(
        Float,
        nullable=True,
        comment="Inference wall-clock time in milliseconds",
    )

    def __repr__(self) -> str:
        return (
            f"<Job id={self.id[:8]}... "
            f"model={self.model_type.value} "
            f"status={self.status.value}>"
        )
