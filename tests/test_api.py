"""
NexusInfer — API Tests (Day 1)

Tests the core API endpoints using FastAPI's built-in test client.
Run with: pytest tests/ -v
"""

import pytest
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
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "jobs_in_queue" in data


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
        assert data["status"] == "completed"  # Sync in Day 1

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
        assert data["status"] == "completed"
        assert data["result"] is not None

    def test_get_nonexistent_job_returns_404(self, client):
        response = client.get("/api/v1/jobs/nonexistent-uuid")
        assert response.status_code == 404


class TestListJobs:
    """Tests for GET /api/v1/jobs"""

    def test_list_jobs_empty(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data

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
