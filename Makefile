.PHONY: dev worker test redis-start redis-stop docker-up docker-down docker-scale clean help

# ──────────────────────────── Local Development ────────────────────────────

## Start the FastAPI dev server (requires local Redis)
dev:
	uvicorn app.main:app --reload --port 8000

## Start a Celery worker consuming from the inference queue
worker:
	celery -A app.workers.celery_app worker --loglevel=info -Q inference --concurrency=4

## Start Celery worker with verbose task monitoring
worker-verbose:
	celery -A app.workers.celery_app worker --loglevel=debug -Q inference --concurrency=2

# ──────────────────────────── Redis ────────────────────────────

## Start Redis via brew
redis-start:
	brew services start redis

## Stop Redis
redis-stop:
	brew services stop redis

## Check Redis status
redis-status:
	redis-cli ping

# ──────────────────────────── Docker ────────────────────────────

## Build and start the full stack (API + Worker + Redis) via Docker Compose
docker-up:
	docker-compose up --build

## Start the full stack in the background
docker-up-detached:
	docker-compose up --build -d

## Stop all containers
docker-down:
	docker-compose down

## Stop containers and remove volumes
docker-clean:
	docker-compose down -v

## Scale worker replicas (e.g. make docker-scale WORKERS=3)
WORKERS ?= 2
docker-scale:
	docker-compose up -d --scale worker=$(WORKERS)

## View live logs from all containers
docker-logs:
	docker-compose logs -f

## View logs from the worker only
worker-logs:
	docker-compose logs -f worker

# ──────────────────────────── Testing ────────────────────────────

## Run all tests
test:
	python -m pytest tests/ -v

## Run tests with coverage report
test-cov:
	python -m pytest tests/ -v --cov=app --cov-report=term-missing

# ──────────────────────────── Utilities ────────────────────────────

## Submit a sample sentiment job
sample-job:
	@echo "Submitting sample sentiment analysis job..."
	curl -s -X POST http://localhost:8000/api/v1/jobs/submit \
		-H "Content-Type: application/json" \
		-d '{"model_type": "sentiment", "input_text": "NexusInfer is incredibly fast and scalable!"}' | python -m json.tool

## Submit a sample NER job
sample-ner:
	@echo "Submitting sample NER job..."
	curl -s -X POST http://localhost:8000/api/v1/jobs/submit \
		-H "Content-Type: application/json" \
		-d '{"model_type": "ner", "input_text": "Apple and Microsoft are leading tech companies based in the United States."}' | python -m json.tool

## Check API health
health:
	curl -s http://localhost:8000/api/v1/health | python -m json.tool

## List all jobs
list-jobs:
	curl -s http://localhost:8000/api/v1/jobs | python -m json.tool

## Remove Python caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .mypy_cache

## Show all available commands
help:
	@echo ""
	@echo "  NexusInfer — Development Commands"
	@echo "  ──────────────────────────────────"
	@echo ""
	@echo "  Local Dev:"
	@echo "    make dev              Start FastAPI dev server"
	@echo "    make worker           Start Celery inference worker"
	@echo "    make redis-start      Start Redis (brew)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-up        Build + start full stack"
	@echo "    make docker-down      Stop all containers"
	@echo "    make docker-scale     Scale workers (WORKERS=3)"
	@echo "    make docker-logs      Live log stream"
	@echo ""
	@echo "  Testing:"
	@echo "    make test             Run test suite"
	@echo "    make test-cov         Run with coverage report"
	@echo ""
	@echo "  Utilities:"
	@echo "    make sample-job       Submit a test sentiment job"
	@echo "    make health           Check API health"
	@echo "    make list-jobs        List all jobs"
	@echo ""
