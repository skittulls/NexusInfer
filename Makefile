.PHONY: dev worker test redis-start redis-stop all clean

# ──────────────────────────── Development ────────────────────────────

## Start the FastAPI dev server
dev:
	uvicorn app.main:app --reload --port 8000

## Start a Celery worker consuming from the inference queue
worker:
	celery -A app.workers.celery_app worker --loglevel=info -Q inference --concurrency=4

## Start both API and worker (requires Redis running)
all: redis-start
	@echo "Starting API server and Celery worker..."
	@uvicorn app.main:app --reload --port 8000 &
	@celery -A app.workers.celery_app worker --loglevel=info -Q inference --concurrency=4

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

# ──────────────────────────── Testing ────────────────────────────

## Run all tests
test:
	python -m pytest tests/ -v

## Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=app --cov-report=term-missing

# ──────────────────────────── Utilities ────────────────────────────

## Submit a sample job via curl
sample-job:
	@echo "Submitting sample sentiment analysis job..."
	curl -s -X POST http://localhost:8000/api/v1/jobs/submit \
		-H "Content-Type: application/json" \
		-d '{"model_type": "sentiment", "input_text": "NexusInfer is incredibly fast!"}' | python -m json.tool

## Check API health
health:
	curl -s http://localhost:8000/api/v1/health | python -m json.tool

## Clean up caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .mypy_cache

## Show help
help:
	@echo "NexusInfer — Development Commands"
	@echo ""
	@echo "  make dev          Start FastAPI dev server"
	@echo "  make worker       Start Celery worker"
	@echo "  make all          Start everything (Redis + API + Worker)"
	@echo "  make test         Run tests"
	@echo "  make sample-job   Submit a test inference job"
	@echo "  make health       Check API health"
	@echo "  make redis-start  Start Redis"
	@echo "  make redis-stop   Stop Redis"
	@echo "  make clean        Remove caches"
