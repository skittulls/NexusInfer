# NexusInfer

**High-Performance Distributed AI Inference API & Task Queue**

A production-grade, asynchronous AI inference platform built with FastAPI, Celery, Redis, and PostgreSQL. NexusInfer decouples model serving from request handling through a distributed task queue architecture, enabling horizontal scaling of ML inference workers independently of the API layer.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Client     │────▶│  FastAPI      │────▶│   Redis      │────▶│  Celery      │
│  (REST API)  │◀────│  Gateway      │     │  (Broker)    │     │  Workers     │
└─────────────┘     └──────┬───────┘     └─────────────┘     └──────┬───────┘
                           │                                        │
                           ▼                                        ▼
                    ┌──────────────┐                         ┌──────────────┐
                    │ PostgreSQL   │◀────────────────────────│  ML Models   │
                    │ (Job Store)  │                         │ (HuggingFace)│
                    └──────────────┘                         └──────────────┘
```

## Features

- **Asynchronous Job Processing**: Submit inference requests and poll for results — no blocking.
- **Distributed Task Queue**: Redis-backed Celery workers consume jobs independently, enabling horizontal scaling.
- **Multi-Model Support**: Sentiment analysis, text summarization, and named entity recognition via HuggingFace Transformers.
- **Persistent Job Store**: PostgreSQL-backed job tracking with full state machine (PENDING → PROCESSING → COMPLETED/FAILED).
- **Dynamic Batching** *(Coming Soon)*: Groups concurrent requests to maximize GPU/CPU throughput.
- **Containerized Deployment**: Single `docker-compose up` spins up the full stack.
- **Benchmarked**: Load-tested with Locust to validate throughput under concurrent load.

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| API Gateway    | Python, FastAPI, Uvicorn          |
| Task Queue     | Celery, Redis                     |
| ML Inference   | PyTorch, HuggingFace Transformers |
| Database       | PostgreSQL, SQLAlchemy            |
| Containerization | Docker, Docker Compose          |
| Benchmarking   | Locust                            |

## Quick Start

### Prerequisites
- Python 3.10+
- Redis (or Docker)
- PostgreSQL (or Docker)

### Local Development

```bash
# Clone the repo
git clone https://github.com/yourusername/NexusInfer.git
cd NexusInfer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --reload --port 8000

# In a separate terminal, start the Celery worker
celery -A app.workers.celery_app worker --loglevel=info
```

### Docker (Full Stack)
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| POST   | `/jobs/submit`     | Submit a new inference job           |
| GET    | `/jobs/{job_id}`   | Get status and result of a job       |
| GET    | `/jobs`            | List all jobs (with pagination)      |
| GET    | `/health`          | Health check                         |

### Example: Submit a Job
```bash
curl -X POST http://localhost:8000/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sentiment",
    "input_text": "NexusInfer makes ML inference incredibly fast and scalable!"
  }'
```

### Example: Check Job Status
```bash
curl http://localhost:8000/jobs/{job_id}
```

## Project Structure

```
NexusInfer/
├── app/
│   ├── api/            # API route handlers
│   ├── core/           # Configuration, constants
│   ├── models/         # SQLAlchemy ORM models
│   ├── schemas/        # Pydantic request/response schemas
│   ├── services/       # Business logic (job management)
│   └── workers/        # Celery tasks & ML inference logic
├── tests/              # Unit and integration tests
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── requirements.txt
└── README.md
```

## License

MIT
