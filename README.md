# NexusInfer

**High-Performance Distributed AI Inference API & Task Queue**

NexusInfer is a production-grade, asynchronous inference platform that decouples machine learning model serving from HTTP request handling. Built on a Redis-backed distributed task queue and a PostgreSQL state machine, it allows clients to submit inference jobs via REST and retrieve results without blocking the API gateway.

The platform is designed for horizontal scalability, fault tolerance, and low-latency inference, utilizing singleton caching to serve open-source NLP models from HuggingFace efficiently.

---

## System Architecture

```
                          ┌──────────────────────────────────────┐
                          │          NexusInfer System           │
                          └──────────────────────────────────────┘

┌──────────┐    HTTP     ┌──────────────────┐    ENQUEUE    ┌──────────────┐
│          │ ─────────►  │   FastAPI API    │ ───────────►  │    Redis     │
│  Client  │             │     Gateway      │               │   (Broker)   │
│          │ ◄─────────  │   /api/v1/...    │               │              │
└──────────┘   JSON      └────────┬─────────┘               └──────┬───────┘
                                  │                                │
                                  │ READ                   CONSUME │
                                  ▼                                ▼
                          ┌──────────────────┐          ┌───────────────────┐
                          │   PostgreSQL     │ ◄──────  │  Celery Workers   │
                          │   (Job Store)    │  WRITE   │  ┌─────────────┐  │
                          │                  │          │  │ HuggingFace │  │
                          └──────────────────┘          │  │ Transformers│  │
                                                        │  └─────────────┘  │
                                                        └───────────────────┘

  Lifecycle:  PENDING ──► PROCESSING ──► COMPLETED (or FAILED with retry)
```

## Core Features

- **Asynchronous Job Processing** — Non-blocking REST architecture for submitting jobs and polling results.
- **Distributed Task Queue** — Redis-backed Celery workers handle heavy ML inference, enabling independent scaling of the API and worker pools.
- **Fault-Tolerant Dispatch** — Implements late-acknowledgment, automated retries with exponential backoff, and graceful degradation to synchronous processing during broker outages.
- **Optimized Model Execution** — Utilizes lazy-loaded singleton caching to load HuggingFace models into RAM once per process, eliminating cold-start latency.
- **Persistent State Machine** — Storage-agnostic job lifecycle management backed by PostgreSQL and SQLAlchemy ORM.
- **Containerized Deployment** — A fully containerized multi-service stack deployable via Docker Compose.

## Supported Models

| Task | HuggingFace Model Architecture |
|------|--------------------------------|
| **Sentiment Analysis** | `distilbert-base-uncased-finetuned-sst-2-english` |
| **Text Summarization** | `sshleifer/distilbart-cnn-12-6` |
| **Named Entity Recognition** | `dslim/bert-base-NER` |

## Quick Start

### Prerequisites
- Docker and Docker Compose

### Deployment

To spin up the entire multi-container stack (API, Celery Workers, Redis, and PostgreSQL):

```bash
# Clone the repository
git clone https://github.com/skittulls/NexusInfer.git
cd NexusInfer

# Build and start the stack in detached mode
make docker-up-detached
```

The API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

### Running Benchmarks
To evaluate system throughput and latency, run the included Locust load tests:
```bash
make benchmark
```

## API Reference

### Submit an Inference Job
```bash
curl -X POST http://localhost:8000/api/v1/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sentiment",
    "input_text": "NexusInfer ensures high availability and horizontal scaling.",
    "priority": 5
  }'
```
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "pending",
  "message": "Job queued for async processing. Model: sentiment. Poll GET /api/v1/jobs/9b1deb4d-... for results.",
  "created_at": "2024-08-16T10:30:00Z"
}
```

### Check Job Status
```bash
curl http://localhost:8000/api/v1/jobs/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```
```json
{
  "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "completed",
  "model_type": "sentiment",
  "result": {
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "task": "sentiment-analysis",
    "label": "POSITIVE",
    "score": 0.9998
  },
  "processing_time_ms": 42.7,
  "created_at": "2024-08-16T10:30:00Z",
  "started_at": "2024-08-16T10:30:00.015Z",
  "completed_at": "2024-08-16T10:30:00.057Z"
}
```

## Tech Stack

| Component | Technologies Used |
|-----------|------------------|
| **API Gateway** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Task Queue** | Celery, Redis |
| **ML Inference** | PyTorch, HuggingFace Transformers |
| **Database** | PostgreSQL, SQLAlchemy ORM |
| **Infrastructure** | Docker, Docker Compose |
| **Testing** | pytest, httpx, Locust |

## License

MIT License
