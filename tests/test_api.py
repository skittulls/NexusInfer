"""
NexusInfer — API Tests

Tests the core API endpoints using FastAPI's built-in test client.
Run with: pytest tests/ -v

Tests run without Redis or ML models by mocking external dependencies.
This ensures tests are fast, deterministic, and CI-friendly.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app


# ──────────────────────────── Mock Inference ────────────────────────────
# Mock the real inference engine so tests don't require downloading
# HuggingFace models (2GB+). The mock returns structured output
# matching the real model output format.

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
        {"text": "Microsoft", "label": "ORG", "score": 0.9945, "start": 11, "end": 20},
    ],
    "entity_count": 2,
    "entity_types": {"ORG": 2},
    "input_length": 56,
}


def _mock_inference(model_type, input_text):
    """Return mock results matching real model output format."""
    from app.schemas.job import ModelType
    results = {
        ModelType.SENTIMENT: MOCK_SENTIMENT_RESULT,
        ModelType.SUMMARIZATION: MOCK_SUMMARIZATION_RESULT,
        ModelType.NER: MOCK_NER_RESULT,
    }
    return results.get(model_type, MOCK_SENTIMENT_RESULT)


@pytest.fixture
def client():
    """Create a fresh test client with mocked inference."""
    with patch("app.api.routes.run_inference", side_effect=_mock_inference):
        yield TestClient(app)


# ──────────────────────────── Health Endpoint ────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /api/v1/health"""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "jobs_in_queue" in data
        assert "redis_connected" in data
        assert "models_loaded" in data

    def test_health_reports_redis_status(self, client):
        """Health should report whether Redis is connected."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert isinstance(data["redis_connected"], bool)


# ──────────────────────────── Submit Job ────────────────────────────


class TestSubmitJob:
    """Tests for POST /api/v1/jobs/submit"""

    def test_submit_sentiment_job(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "This is an amazing product!",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202

        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["pending", "completed"]

    def test_submit_summarization_job(self, client):
        payload = {
            "model_type": "summarization",
            "input_text": "This is a long text that needs to be summarized. " * 10,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202

    def test_submit_ner_job(self, client):
        payload = {
            "model_type": "ner",
            "input_text": "Google and Microsoft are tech companies based in the US.",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202

    def test_submit_empty_text_fails(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422

    def test_submit_invalid_model_fails(self, client):
        payload = {
            "model_type": "nonexistent_model",
            "input_text": "Hello world",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422

    def test_submit_returns_uuid(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "Testing job ID generation.",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        data = response.json()
        assert len(data["job_id"]) == 36  # UUID format

    def test_submit_with_priority(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "High priority inference.",
            "priority": 10,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202

    def test_submit_invalid_priority_fails(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "Invalid priority.",
            "priority": 99,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422

    def test_submit_text_too_long_fails(self, client):
        """Input text exceeding max_length should be rejected."""
        payload = {
            "model_type": "sentiment",
            "input_text": "x" * 5001,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422


# ──────────────────────────── Get Job Status ────────────────────────────


class TestGetJobStatus:
    """Tests for GET /api/v1/jobs/{job_id}"""

    def test_get_existing_job(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "Test input for status check.",
        }
        submit_resp = client.post("/api/v1/jobs/submit", json=payload)
        job_id = submit_resp.json()["job_id"]

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert data["model_type"] == "sentiment"

    def test_get_completed_job_has_result(self, client):
        """Completed jobs should have structured result and timing."""
        payload = {
            "model_type": "sentiment",
            "input_text": "Check result field.",
        }
        submit_resp = client.post("/api/v1/jobs/submit", json=payload)
        job_id = submit_resp.json()["job_id"]

        response = client.get(f"/api/v1/jobs/{job_id}")
        data = response.json()

        if data["status"] == "completed":
            assert data["result"] is not None
            assert data["result"]["model"] == "distilbert-base-uncased-finetuned-sst-2-english"
            assert data["result"]["label"] in ["POSITIVE", "NEGATIVE"]
            assert data["processing_time_ms"] is not None

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/v1/jobs/nonexistent-uuid")
        assert response.status_code == 404


# ──────────────────────────── List Jobs ────────────────────────────


class TestListJobs:
    """Tests for GET /api/v1/jobs"""

    def test_list_jobs_returns_200(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data

    def test_list_jobs_pagination(self, client):
        for i in range(3):
            client.post("/api/v1/jobs/submit", json={
                "model_type": "sentiment",
                "input_text": f"Test input number {i}",
            })

        response = client.get("/api/v1/jobs?page=1&page_size=2")
        data = response.json()
        assert len(data["jobs"]) <= 2
        assert data["page"] == 1

    def test_list_jobs_status_filter(self, client):
        response = client.get("/api/v1/jobs?status=failed")
        assert response.status_code == 200


# ──────────────────────────── Celery Integration ────────────────────────────


class TestCeleryIntegration:
    """Tests for async Celery dispatch (mocked)."""

    def test_async_dispatch_when_redis_available(self, client):
        """Jobs are dispatched to Celery when Redis is available."""
        with patch("app.api.routes._check_redis_connection", return_value=True):
            with patch("app.workers.tasks.run_inference_task.apply_async") as mock_dispatch:
                payload = {
                    "model_type": "sentiment",
                    "input_text": "Async dispatch test.",
                }
                response = client.post("/api/v1/jobs/submit", json=payload)

                assert response.status_code == 202
                data = response.json()
                assert data["status"] == "pending"
                mock_dispatch.assert_called_once()

    def test_fallback_to_sync_when_redis_unavailable(self, client):
        """Jobs run synchronously when Redis is down."""
        with patch("app.api.routes._check_redis_connection", return_value=False):
            payload = {
                "model_type": "sentiment",
                "input_text": "Sync fallback test.",
            }
            response = client.post("/api/v1/jobs/submit", json=payload)

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "completed"


# ──────────────────────────── Model Manager ────────────────────────────


class TestModelManager:
    """Tests for the ModelManager singleton."""

    def test_model_registry_has_all_types(self):
        from app.services.model_manager import ModelManager
        from app.schemas.job import ModelType

        manager = ModelManager()
        for mt in ModelType:
            assert mt in manager.MODEL_REGISTRY

    def test_status_reports_available_models(self):
        from app.services.model_manager import ModelManager

        manager = ModelManager()
        status = manager.status
        assert status["total_available"] == 3
        assert status["total_loaded"] == 0  # Nothing loaded yet in test
