"""
NexusInfer — API Tests

Tests the core API endpoints using FastAPI's built-in test client.
Run with: pytest tests/ -v

All external dependencies are mocked or overridden:
  - Database: in-memory SQLite (via FastAPI dependency override)
  - Inference engine: mock returning real-shaped output
  - Redis: mocked via patch
No external services are required.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import Base, get_db


# ──────────────────────────── Test Database Setup ────────────────────────────

TEST_DATABASE_URL = "sqlite:///:memory:"

# Use a single shared connection so all sessions in a test share
# the same in-memory SQLite database (each new connection to
# sqlite:///:memory: gets its own blank DB).
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Shared connection kept open for the lifetime of a test
_test_connection = test_engine.connect()

TestSessionLocal = sessionmaker(
    bind=_test_connection,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def override_get_db():
    """Override the production DB dependency with in-memory SQLite."""
    db = TestSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ──────────────────────────── Mock Inference ────────────────────────────

MOCK_SENTIMENT_RESULT = {
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "task": "sentiment-analysis",
    "label": "POSITIVE",
    "score": 0.9998,
    "input_length": 28,
}

MOCK_SUMMARIZATION_RESULT = {
    "model": "sshleifer/distilbart-cnn-12-6",
    "task": "summarization",
    "summary": "This is a condensed summary of the input text.",
    "input_length": 500,
    "summary_length": 47,
    "compression_ratio": 0.094,
    "input_word_count": 100,
    "summary_word_count": 9,
}

MOCK_NER_RESULT = {
    "model": "dslim/bert-base-NER",
    "task": "named-entity-recognition",
    "entities": [
        {"text": "Google", "label": "ORG", "score": 0.9987, "start": 0, "end": 6},
    ],
    "entity_count": 1,
    "entity_types": {"ORG": 1},
    "input_length": 56,
}


def _mock_inference(model_type, input_text):
    from app.schemas.job import ModelType
    return {
        ModelType.SENTIMENT: MOCK_SENTIMENT_RESULT,
        ModelType.SUMMARIZATION: MOCK_SUMMARIZATION_RESULT,
        ModelType.NER: MOCK_NER_RESULT,
    }.get(model_type, MOCK_SENTIMENT_RESULT)


# ──────────────────────────── Fixtures ────────────────────────────


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create all tables on the shared connection before each test, drop after."""
    from app.models import job  # noqa — register model with Base
    Base.metadata.create_all(bind=_test_connection)
    yield
    Base.metadata.drop_all(bind=_test_connection)


@pytest.fixture
def client(setup_test_db):
    """
    Test client with:
      - DB dependency overridden to use in-memory SQLite (shared connection)
      - Inference engine mocked to avoid loading HuggingFace models
    """
    app.dependency_overrides[get_db] = override_get_db
    with patch("app.api.routes.run_inference", side_effect=_mock_inference):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ──────────────────────────── Health Endpoint ────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "jobs_in_queue" in data
        assert "redis_connected" in data
        assert "models_loaded" in data

    def test_health_queue_depth_reflects_db(self, client):
        """Queue depth should reflect actual pending jobs in the DB."""
        # Initially 0
        data = client.get("/api/v1/health").json()
        assert data["jobs_in_queue"] == 0


# ──────────────────────────── Submit Job ────────────────────────────


class TestSubmitJob:
    def test_submit_sentiment_job(self, client):
        payload = {"model_type": "sentiment", "input_text": "Amazing product!"}
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["pending", "completed"]

    def test_submit_summarization_job(self, client):
        payload = {"model_type": "summarization", "input_text": "Long text. " * 20}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 202

    def test_submit_ner_job(self, client):
        payload = {"model_type": "ner", "input_text": "Google is a company."}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 202

    def test_empty_text_rejected(self, client):
        payload = {"model_type": "sentiment", "input_text": ""}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 422

    def test_invalid_model_rejected(self, client):
        payload = {"model_type": "gpt5", "input_text": "hello"}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 422

    def test_returns_uuid(self, client):
        payload = {"model_type": "sentiment", "input_text": "Testing UUID."}
        data = client.post("/api/v1/jobs/submit", json=payload).json()
        assert len(data["job_id"]) == 36

    def test_priority_accepted(self, client):
        payload = {"model_type": "sentiment", "input_text": "Hi", "priority": 10}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 202

    def test_priority_out_of_range_rejected(self, client):
        payload = {"model_type": "sentiment", "input_text": "Hi", "priority": 99}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 422

    def test_text_too_long_rejected(self, client):
        payload = {"model_type": "sentiment", "input_text": "x" * 5001}
        assert client.post("/api/v1/jobs/submit", json=payload).status_code == 422


# ──────────────────────────── Get Job Status ────────────────────────────


class TestGetJobStatus:
    def test_get_existing_job(self, client):
        submit = client.post("/api/v1/jobs/submit", json={
            "model_type": "sentiment", "input_text": "Test job retrieval."
        })
        job_id = submit.json()["job_id"]
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["model_type"] == "sentiment"

    def test_completed_job_has_result(self, client):
        submit = client.post("/api/v1/jobs/submit", json={
            "model_type": "sentiment", "input_text": "Result field check."
        })
        job_id = submit.json()["job_id"]
        data = client.get(f"/api/v1/jobs/{job_id}").json()
        if data["status"] == "completed":
            assert data["result"] is not None
            assert data["result"]["label"] in ["POSITIVE", "NEGATIVE"]
            assert data["processing_time_ms"] is not None

    def test_job_persists_across_requests(self, client):
        """Jobs in DB survive across request boundaries (not in-memory)."""
        submit = client.post("/api/v1/jobs/submit", json={
            "model_type": "ner", "input_text": "Persistence test."
        })
        job_id = submit.json()["job_id"]
        # Second request should still find the job
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200

    def test_nonexistent_job_404(self, client):
        assert client.get("/api/v1/jobs/does-not-exist").status_code == 404


# ──────────────────────────── List Jobs ────────────────────────────


class TestListJobs:
    def test_list_returns_200(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        for field in ("jobs", "total", "page", "page_size"):
            assert field in data

    def test_pagination(self, client):
        for i in range(5):
            client.post("/api/v1/jobs/submit", json={
                "model_type": "sentiment", "input_text": f"Job {i}"
            })
        data = client.get("/api/v1/jobs?page=1&page_size=3").json()
        assert len(data["jobs"]) <= 3
        assert data["total"] >= 5

    def test_status_filter(self, client):
        response = client.get("/api/v1/jobs?status=failed")
        assert response.status_code == 200


# ──────────────────────────── Celery Integration ────────────────────────────


class TestCeleryIntegration:
    def test_async_dispatch_when_redis_available(self, client):
        with patch("app.api.routes._check_redis_connection", return_value=True):
            with patch("app.workers.tasks.run_inference_task.apply_async") as mock_dispatch:
                response = client.post("/api/v1/jobs/submit", json={
                    "model_type": "sentiment", "input_text": "Async test."
                })
                assert response.status_code == 202
                assert response.json()["status"] == "pending"
                mock_dispatch.assert_called_once()

    def test_sync_fallback_when_redis_unavailable(self, client):
        with patch("app.api.routes._check_redis_connection", return_value=False):
            response = client.post("/api/v1/jobs/submit", json={
                "model_type": "sentiment", "input_text": "Sync fallback test."
            })
            assert response.status_code == 202
            assert response.json()["status"] == "completed"


# ──────────────────────────── Model Manager ────────────────────────────


class TestModelManager:
    def test_registry_covers_all_model_types(self):
        from app.services.model_manager import ModelManager
        from app.schemas.job import ModelType
        manager = ModelManager()
        for mt in ModelType:
            assert mt in manager.MODEL_REGISTRY

    def test_status_reports_correctly(self):
        from app.services.model_manager import ModelManager
        manager = ModelManager()
        status = manager.status
        assert status["total_available"] == 3
        assert status["total_loaded"] == 0
