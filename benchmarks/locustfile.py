"""
NexusInfer — Locust Load Testing

Simulates high-concurrency traffic to benchmark the system.
Run this script while the Docker stack is running:
    locust -f benchmarks/locustfile.py --headless -u 100 -r 10 --run-time 1m --host http://localhost:8000
"""

import time
import random
from locust import HttpUser, task, between, events


class InferenceUser(HttpUser):
    """Simulates a client submitting jobs and polling for results."""
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1.0, 3.0)

    @task(3)
    def submit_sentiment_job(self):
        """Submit a sentiment analysis job and poll until complete."""
        payload = {
            "model_type": "sentiment",
            "input_text": "Locust load testing is incredibly useful for benchmarking performance! " * random.randint(1, 5)
        }
        self._submit_and_poll(payload, name="/jobs/submit (sentiment)")

    @task(1)
    def submit_ner_job(self):
        """Submit a Named Entity Recognition job and poll until complete."""
        payload = {
            "model_type": "ner",
            "input_text": "Satya Nadella is the CEO of Microsoft, located in Redmond, Washington. " * random.randint(1, 3)
        }
        self._submit_and_poll(payload, name="/jobs/submit (ner)")
        
    @task(2)
    def check_health(self):
        """Check API health status."""
        self.client.get("/api/v1/health", name="/health")

    def _submit_and_poll(self, payload: dict, name: str):
        """Helper to submit a job and poll until completion."""
        # 1. Submit the job
        with self.client.post("/api/v1/jobs/submit", json=payload, name=name, catch_response=True) as response:
            if response.status_code != 202:
                response.failure(f"Expected 202 Accepted, got {response.status_code}")
                return
            
            data = response.json()
            job_id = data.get("job_id")
            
            if not job_id:
                response.failure("No job_id returned in response")
                return

        # 2. Poll for results
        start_time = time.time()
        max_poll_time = 30.0  # Give up after 30 seconds
        
        while time.time() - start_time < max_poll_time:
            # Sleep between 0.5s and 1.5s before polling
            time.sleep(random.uniform(0.5, 1.5))
            
            with self.client.get(f"/api/v1/jobs/{job_id}", name="/jobs/{id} (poll)", catch_response=True) as poll_resp:
                if poll_resp.status_code != 200:
                    poll_resp.failure(f"Poll failed with {poll_resp.status_code}")
                    return
                
                poll_data = poll_resp.json()
                status = poll_data.get("status")
                
                if status == "completed":
                    poll_resp.success()
                    return
                elif status == "failed":
                    poll_resp.failure(f"Job failed: {poll_data.get('error')}")
                    return
                # If pending/processing, loop continues
                
        # If we exit the loop, we timed out
        events.request.fire(
            request_type="GET",
            name="/jobs/{id} (timeout)",
            response_time=(time.time() - start_time) * 1000,
            response_length=0,
            exception=TimeoutError("Polling timed out after 30 seconds")
        )
