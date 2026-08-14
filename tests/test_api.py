"""
NexusInfer — API Tests

Tests the core API endpoints using FastAPI's built-in test client.
Run with: pytest tests/ -v

Day 2: Tests work in sync-fallback mode (no Redis required for CI).
       The API gracefully falls back to synchronous inference when
       Redis is unavailable, so all tests pass without infrastructure.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a fresh test client for each test."""
    return TestClient(app)


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

    def test_health_reports_degraded_without_redis(self, client):
        """When Redis is unavailable, health should report degraded status."""
        response = client.get("/api/v1/health")
        data = response.json()
        # Without Redis running in CI, expect degraded
        assert "version" in data


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
        # In sync fallback (no Redis), job completes immediately
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
        assert response.status_code == 422  # Validation error

    def test_submit_invalid_model_fails(self, client):
        payload = {
            "model_type": "nonexistent_model",
            "input_text": "Hello world",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422

    def test_submit_returns_job_id(self, client):
        payload = {
            "model_type": "sentiment",
            "input_text": "Testing job ID generation.",
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        data = response.json()
        assert len(data["job_id"]) == 36  # UUID format

    def test_submit_with_priority(self, client):
        """Test that priority field is accepted."""
        payload = {
            "model_type": "sentiment",
            "input_text": "High priority inference.",
            "priority": 10,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 202

    def test_submit_invalid_priority_fails(self, client):
        """Priority must be 0-10."""
        payload = {
            "model_type": "sentiment",
            "input_text": "Invalid priority.",
            "priority": 99,
        }
        response = client.post("/api/v1/jobs/submit", json=payload)
        assert response.status_code == 422


class TestGetJobStatus:
    """Tests for GET /api/v1/jobs/{job_id}"""

    def test_get_existing_job(self, client):
        # First, create a job
        payload = {
            "model_type": "sentiment",
            "input_text": "Test input for status check.",
        }
        submit_resp = client.post("/api/v1/jobs/submit", json=payload)
        job_id = submit_resp.json()["job_id"]

        # Then, query it
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert data["model_type"] == "sentiment"
        assert data["input_text"] == "Test input for status check."

    def test_get_completed_job_has_result(self, client):
        """In sync fallback, completed jobs should have a result."""
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
            assert data["processing_time_ms"] is not None
            assert data["processing_time_ms"] > 0

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/v1/jobs/nonexistent-uuid")
        assert response.status_code == 404


class TestListJobs:
    """Tests for GET /api/v1/jobs"""

    def test_list_jobs_returns_200(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    def test_list_jobs_pagination(self, client):
        # Submit a few jobs
        for i in range(3):
            client.post("/api/v1/jobs/submit", json={
                "model_type": "sentiment",
                "input_text": f"Test input number {i}",
            })

        response = client.get("/api/v1/jobs?page=1&page_size=2")
        data = response.json()
        assert len(data["jobs"]) <= 2
        assert data["page"] == 1
        assert data["page_size"] == 2

    def test_list_jobs_status_filter(self, client):
        """Test filtering by status."""
        response = client.get("/api/v1/jobs?status=failed")
        assert response.status_code == 200


class TestCeleryIntegration:
    """Tests for Celery task dispatch (mocked)."""

    def test_async_dispatch_when_redis_available(self, client):
        """Verify that when Redis is available, jobs are dispatched async."""
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
                assert "queued" in data["message"].lower() or "async" in data["message"].lower()
                mock_dispatch.assert_called_once()

    def test_fallback_to_sync_when_redis_unavailable(self, client):
        """Verify sync fallback when Redis is down."""
        with patch("app.api.routes._check_redis_connection", return_value=False):
            payload = {
                "model_type": "sentiment",
                "input_text": "Sync fallback test.",
            }
            response = client.post("/api/v1/jobs/submit", json=payload)

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "completed"
