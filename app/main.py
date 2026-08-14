"""
NexusInfer — Application Entry Point

Creates and configures the FastAPI application instance.
This is the file uvicorn points to: `uvicorn app.main:app`
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

# ──────────────────────────── Logging ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nexusinfer")


# ──────────────────────────── Lifespan ────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: Log configuration, verify external services.
    Shutdown: Graceful cleanup.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"  Debug Mode: {settings.DEBUG}")
    logger.info(f"  Default Model: {settings.DEFAULT_MODEL}")
    logger.info("=" * 60)

    # ── Create database tables ──
    try:
        from app.core.database import create_tables
        create_tables()
        logger.info("  Database: READY")
    except Exception as e:
        logger.error(f"  Database: FAILED ({e})")

    # ── Check Redis connectivity ──
    try:
        import redis
        r = redis.Redis.from_url(settings.CELERY_BROKER_URL, socket_timeout=2)
        r.ping()
        logger.info(f"  Redis: CONNECTED ({settings.CELERY_BROKER_URL})")
        logger.info("  Mode: ASYNC (Celery workers)")
    except Exception as e:
        logger.warning(f"  Redis: UNAVAILABLE ({e})")
        logger.warning("  Mode: SYNC FALLBACK (jobs processed in-request)")
        logger.warning("  → Start Redis: brew services start redis")
        logger.warning("  → Start Worker: celery -A app.workers.celery_app worker "
                       "--loglevel=info -Q inference")

    logger.info("=" * 60)
    logger.info("Starting up...")

    yield  # Application runs here

    logger.info("Shutting down...")


# ──────────────────────────── App Factory ────────────────────────────


def create_app() -> FastAPI:
    """
    Application factory.

    Creates a configured FastAPI instance with middleware, routes, and
    OpenAPI documentation.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        description=(
            "A production-grade, distributed AI inference API. "
            "Submit ML inference jobs via REST, which are dispatched to "
            "background workers through a Redis-backed Celery task queue. "
            "Supports sentiment analysis, text summarization, and NER."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS Middleware ──
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register Routes ──
    application.include_router(router, prefix="/api/v1")

    return application


# ── Create the application instance ──
app = create_app()
