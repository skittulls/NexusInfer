"""
NexusInfer — Pydantic Schemas

Defines the request/response contracts for the API.
Pydantic models handle validation, serialization, and OpenAPI doc generation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
from datetime import datetime


class ModelType(str, Enum):
    """Supported ML model types for inference."""
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"  # Named Entity Recognition


class JobStatus(str, Enum):
    """
    Job lifecycle states.

    State machine:
        PENDING → PROCESSING → COMPLETED
                             → FAILED
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ──────────────────────────── Request Schemas ────────────────────────────


class JobSubmitRequest(BaseModel):
    """Schema for submitting a new inference job."""
    model_type: ModelType = Field(
        default=ModelType.SENTIMENT,
        description="The ML model to use for inference.",
    )
    input_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The input text to run inference on.",
        examples=["NexusInfer makes ML inference incredibly fast and scalable!"],
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Job priority (0 = lowest, 10 = highest).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "sentiment",
                    "input_text": "This product is absolutely amazing!",
                    "priority": 0,
                }
            ]
        }
    }


# ──────────────────────────── Response Schemas ────────────────────────────


class JobSubmitResponse(BaseModel):
    """Response returned after successfully submitting a job."""
    job_id: str = Field(description="Unique identifier for the submitted job.")
    status: JobStatus = Field(description="Current status of the job.")
    message: str = Field(description="Human-readable status message.")
    created_at: datetime = Field(description="Timestamp when the job was created.")


class JobStatusResponse(BaseModel):
    """Response for querying the status/result of a specific job."""
    job_id: str
    status: JobStatus
    model_type: ModelType
    input_text: str
    result: Optional[Any] = Field(
        default=None,
        description="Inference result. Populated when status is COMPLETED.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message. Populated when status is FAILED.",
    )
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Total inference processing time in milliseconds.",
    )


class JobListResponse(BaseModel):
    """Paginated list of jobs."""
    jobs: list[JobStatusResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    uptime_seconds: float
    jobs_in_queue: int = 0
