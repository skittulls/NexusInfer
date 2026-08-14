"""
NexusInfer — ML Model Manager

Manages the lifecycle of HuggingFace transformer models.
Models are loaded once per process and reused across all inference
requests to avoid the overhead of reloading on every call.

Architecture:
    - Singleton ModelManager holds all loaded pipelines in memory
    - Lazy loading: models are loaded on first use, not at import time
    - Thread-safe: HuggingFace pipelines are safe for concurrent reads
    - Worker integration: Celery workers call load_all() on boot via
      the worker_process_init signal
"""

import logging
import time
from typing import Optional

from app.schemas.job import ModelType

logger = logging.getLogger(__name__)

# Pipeline type hints (avoid importing transformers at module level
# so the module can be imported even without transformers installed)
Pipeline = object


class ModelManager:
    """
    Singleton manager for HuggingFace transformer pipelines.

    Loads models lazily on first inference request and caches them
    in memory for the lifetime of the process. In a Celery worker,
    models are pre-loaded on boot via load_all().

    Supported models:
        - sentiment: distilbert-base-uncased-finetuned-sst-2-english
        - summarization: sshleifer/distilbart-cnn-12-6
        - ner: dslim/bert-base-NER
    """

    # Model registry: maps ModelType → (task_name, model_id)
    MODEL_REGISTRY = {
        ModelType.SENTIMENT: (
            "sentiment-analysis",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ),
        ModelType.SUMMARIZATION: (
            "summarization",
            "sshleifer/distilbart-cnn-12-6",
        ),
        ModelType.NER: (
            "ner",
            "dslim/bert-base-NER",
        ),
    }

    def __init__(self):
        self._pipelines: dict[ModelType, Pipeline] = {}
        self._load_times: dict[ModelType, float] = {}

    def _load_model(self, model_type: ModelType) -> Pipeline:
        """Load a single HuggingFace pipeline into memory."""
        from transformers import pipeline

        task_name, model_id = self.MODEL_REGISTRY[model_type]

        logger.info(f"Loading model: {model_type.value} ({model_id})...")
        start = time.time()

        pipe = pipeline(
            task=task_name,
            model=model_id,
            device=-1,  # CPU (use 0 for GPU)
        )

        elapsed = time.time() - start
        self._load_times[model_type] = elapsed
        logger.info(
            f"Model loaded: {model_type.value} | "
            f"time={elapsed:.2f}s | model={model_id}"
        )

        return pipe

    def get_pipeline(self, model_type: ModelType) -> Pipeline:
        """
        Get a loaded pipeline, loading it on first access.

        This is the primary interface for the inference engine.
        Thread-safe: dict reads/writes are atomic in CPython.
        """
        if model_type not in self._pipelines:
            self._pipelines[model_type] = self._load_model(model_type)
        return self._pipelines[model_type]

    def load_all(self):
        """
        Pre-load all registered models into memory.

        Called by Celery workers on boot to eliminate cold-start
        latency on the first inference request.
        """
        logger.info("Pre-loading all models...")
        total_start = time.time()

        for model_type in self.MODEL_REGISTRY:
            self.get_pipeline(model_type)

        total_elapsed = time.time() - total_start
        logger.info(
            f"All models loaded | count={len(self._pipelines)} | "
            f"total_time={total_elapsed:.2f}s"
        )

    def is_loaded(self, model_type: ModelType) -> bool:
        """Check if a specific model is loaded."""
        return model_type in self._pipelines

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded model names."""
        return [mt.value for mt in self._pipelines]

    @property
    def status(self) -> dict:
        """Status report of all models."""
        return {
            "loaded": self.loaded_models,
            "load_times": {
                mt.value: round(t, 2)
                for mt, t in self._load_times.items()
            },
            "total_loaded": len(self._pipelines),
            "total_available": len(self.MODEL_REGISTRY),
        }


# ──────────────────── Singleton Instance ────────────────────
model_manager = ModelManager()
