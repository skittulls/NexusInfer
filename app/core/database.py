"""
NexusInfer — Database Engine & Session Management

Provides:
  - SQLAlchemy engine (configured from DATABASE_URL env var)
  - Declarative Base (all ORM models inherit from this)
  - `get_db()` — FastAPI dependency for per-request sessions
  - `get_db_session()` — context manager for Celery workers

Design decisions:
  - Single engine singleton (connection pooling managed by SQLAlchemy)
  - Separate session per request in FastAPI (no shared state across requests)
  - Separate session per task in Celery (workers are synchronous)
  - Local dev: SQLite (DATABASE_URL=sqlite:///./nexusinfer.db)
  - Production/Docker: PostgreSQL (DATABASE_URL=postgresql://...)
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ──────────────────────────── Declarative Base ────────────────────────────

class Base(DeclarativeBase):
    """All ORM models inherit from this base class."""
    pass


# ──────────────────────────── Engine ────────────────────────────

def _build_engine():
    """
    Create the SQLAlchemy engine from config.

    SQLite: uses check_same_thread=False for single-file dev setup.
    PostgreSQL: uses connection pooling (pool_size=5, max_overflow=10).
    """
    db_url = settings.DATABASE_URL

    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=settings.DEBUG,
        )
        # Enable WAL mode for SQLite (better concurrent read performance)
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    else:
        engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,         # Verify connection health before use
            echo=settings.DEBUG,
        )

    logger.info(f"Database engine created: {db_url.split('@')[-1]}")
    return engine


engine = _build_engine()

# ──────────────────────────── Session Factory ────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,     # Don't expire objects after commit (safe for workers)
)


# ──────────────────────────── FastAPI Dependency ────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage:
        @router.get("/jobs")
        async def list_jobs(db: Session = Depends(get_db)):
            ...

    The session is automatically committed on success and closed on exit.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ──────────────────────────── Celery Context Manager ────────────────────────────

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager providing a database session for Celery tasks.

    Usage:
        with get_db_session() as db:
            service = JobService(db)
            service.update_status(job_id, JobStatus.PROCESSING)

    The session is committed on clean exit and rolled back on exception.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ──────────────────────────── Schema Init ────────────────────────────

def create_tables():
    """
    Create all tables defined in ORM models.

    Called at application startup. Safe to call multiple times —
    SQLAlchemy only creates tables that don't already exist.

    For production migrations use Alembic instead.
    """
    from app.models import job  # noqa: F401 — import to register model

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")
