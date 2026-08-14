"""
NexusInfer — Celery Application

Configures the Celery distributed task queue backed by Redis.
This module is the entry point for Celery workers:

    celery -A app.workers.celery_app worker --loglevel=info

Architecture:
    FastAPI (producer) → Redis (broker) → Celery Worker (consumer)
                                        ↓
                                  Redis (result backend)

The broker and result backend are both Redis, but use different
database numbers (0 and 1) to keep concerns separated.
"""

import logging
from celery import Celery
from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ──────────────────────────── Celery Instance ────────────────────────────

celery_app = Celery(
    "nexusinfer",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# ──────────────────────────── Configuration ────────────────────────────

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task behavior
    task_track_started=True,           # Track PROCESSING state
    task_acks_late=True,               # Ack after completion (fault tolerance)
    worker_prefetch_multiplier=1,      # Fair scheduling: one task at a time per worker

    # Result expiry
    result_expires=3600,               # Results expire after 1 hour

    # Retry policy for broker connection
    broker_connection_retry_on_startup=True,

    # Task time limits
    task_soft_time_limit=settings.TASK_TIMEOUT,
    task_time_limit=settings.TASK_TIMEOUT + 30,

    # Task routes (organize tasks into queues)
    task_routes={
        "app.workers.tasks.run_inference_task": {"queue": "inference"},
    },

    # Default queue
    task_default_queue="default",
)

# ──────────────────────────── Auto-discover Tasks ────────────────────────────

celery_app.autodiscover_tasks(["app.workers"])

logger.info(
    f"Celery configured | broker={settings.CELERY_BROKER_URL} | "
    f"backend={settings.CELERY_RESULT_BACKEND}"
)
