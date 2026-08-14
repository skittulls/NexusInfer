# NexusInfer

**High-Performance Distributed AI Inference API & Task Queue**

A production-grade, asynchronous inference platform that decouples model serving from request handling through a Redis-backed distributed task queue. Submit ML inference jobs via REST — they're dispatched to background workers running real HuggingFace transformer models, with results retrievable by polling.

Built with **FastAPI**, **Celery**, **Redis**, **HuggingFace Transformers**, **PostgreSQL**, and **Docker**.

---

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │          NexusInfer System           │
                          └──────────────────────────────────────┘

┌──────────┐    HTTP     ┌──────────────────┐    ENQUEUE    ┌──────────────┐
│          │ ─────────►  │   FastAPI API     │ ───────────►  │    Redis     │
│  Client  │             │   Gateway        │               │   (Broker)   │
│          │ ◄─────────  │   /api/v1/...    │               │              │
└──────────┘   JSON      └────────┬─────────┘               └──────┬───────┘
                                  │                                 │
                                  │ READ                      CONSUME│
                                  ▼                                 ▼
                          ┌──────────────────┐          ┌───────────────────┐
                          │   PostgreSQL     │ ◄──────  │  Celery Workers   │
                          │   (Job Store)    │  WRITE   │  ┌─────────────┐  │
                          │                  │          │  │ HuggingFace │  │
                          └──────────────────┘          │  │ Transformers│  │
                                                        │  └─────────────┘  │
                                                        └───────────────────┘

  Job Flow:  SUBMIT ──► PENDING ──► PROCESSING ──► COMPLETED
                                                └──► FAILED (with retry)
```

## Supported ML Models

| Model | Task | HuggingFace Model ID |
|-------|------|---------------------|
| Sentiment Analysis | Text classification (POSITIVE/NEGATIVE) | `distilbert-base-uncased-finetuned-sst-2-english` |
| Text Summarization | Abstractive summarization | `sshleifer/distilbart-cnn-12-6` |
| Named Entity Recognition | Entity extraction (PER, ORG, LOC) | `dslim/bert-base-NER` |

Models are loaded **once per worker process** on boot and cached in memory — zero cold-start on inference requests.

## Features

- **Asynchronous Job Processing** — Submit inference requests and poll for results. No blocking.
- **Distributed Task Queue** — Redis-backed Celery workers consume jobs independently, enabling horizontal scaling.
- **Real ML Models** — HuggingFace Transformers pipelines for sentiment, summarization, and NER.
- **Fault Tolerance** — Automatic retry (max 2), late acknowledgment, graceful degradation to synchronous mode when Redis is unavailable.
- **Job State Machine** — Full lifecycle tracking: `PENDING → PROCESSING → COMPLETED/FAILED`.
- **Priority Queues** — Jobs accept priority levels (0–10) for queue ordering.
- **Containerized Deployment** — Single `docker-compose up` spins up the full stack.
- **22 Automated Tests** — Full test coverage via pytest, CI-friendly (no GPU required).

## Quick Start

### Prerequisites
- Python 3.10+
- Redis (`brew install redis && brew services start redis`)

### Local Development

```bash
# Clone
git clone https://github.com/skittulls/NexusInfer.git
cd NexusInfer

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the API server
make dev
# → http://localhost:8000/docs (Swagger UI)

# In a separate terminal — start the Celery worker
make worker
```

### Docker (Full Stack)
```bash
docker-compose up --build
```

## API Reference

### Submit an Inference Job
```bash
curl -X POST http://localhost:8000/api/v1/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sentiment",
    "input_text": "NexusInfer makes ML inference incredibly fast and scalable!",
    "priority": 5
  }'
```
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Job queued for async processing. Poll GET /api/v1/jobs/a1b2c3d4-... for results.",
  "created_at": "2024-08-14T10:30:00Z"
}
```

### Check Job Status
```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "model_type": "sentiment",
  "result": {
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "task": "sentiment-analysis",
    "label": "POSITIVE",
    "score": 0.9998
  },
  "processing_time_ms": 42.7
}
```

### All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs/submit` | Submit a new inference job |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and result |
| `GET` | `/api/v1/jobs` | List all jobs (paginated, filterable) |
| `GET` | `/api/v1/health` | System health (Redis, models, queue depth) |
| `GET` | `/docs` | Interactive Swagger UI |

## Project Structure

```
NexusInfer/
├── app/
│   ├── main.py                 # FastAPI app factory, lifespan, CORS
│   ├── api/
│   │   └── routes.py           # REST endpoint handlers
│   ├── core/
│   │   └── config.py           # 12-factor config (pydantic-settings)
│   ├── schemas/
│   │   └── job.py              # Pydantic request/response models
│   ├── services/
│   │   ├── inference.py        # HuggingFace inference engine
│   │   ├── model_manager.py    # Singleton model lifecycle manager
│   │   └── job_service.py      # Job store + state machine
│   ├── models/                 # SQLAlchemy ORM models
│   └── workers/
│       ├── celery_app.py       # Celery configuration + model preloading
│       └── tasks.py            # Background inference tasks
├── tests/
│   └── test_api.py             # 22 tests (pytest)
├── Makefile                    # Dev workflow commands
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile.api
├── Dockerfile.worker
└── requirements.txt
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Gateway | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| Task Queue | Celery, Redis |
| ML Inference | PyTorch, HuggingFace Transformers |
| Database | PostgreSQL, SQLAlchemy |
| Containerization | Docker, Docker Compose |
| Testing | pytest, httpx |

## Roadmap

- [x] FastAPI async REST API with Pydantic validation
- [x] Redis-backed Celery distributed task queue
- [x] HuggingFace Transformers inference (sentiment, summarization, NER)
- [x] Singleton model caching + lazy loading
- [x] Fault-tolerant task dispatch (late-ack, retries, graceful degradation)
- [x] Containerized with Docker + Docker Compose
- [ ] **Persistent Job Store** — PostgreSQL + SQLAlchemy ORM with Alembic migrations
- [ ] **Locust Benchmarks** — Load testing with throughput and p99 latency reports
- [ ] **Dynamic Batching** — Group concurrent requests to maximize GPU/CPU throughput

## License

MIT
