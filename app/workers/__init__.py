"""
Workers package — Celery task queue integration.

Contains the Celery app instance and all background tasks.
"""

from app.workers.celery_app import celery_app

__all__ = ["celery_app"]
