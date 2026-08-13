"""
NexusInfer — Mock Inference Engine

Simulates ML model inference for Day 1 development.
On Day 3, this module is replaced with real HuggingFace model loading.

The mock engine introduces realistic latency and returns structured
results that match the real model output format, so the rest of the
system (API, job service, schemas) doesn't need to change later.
"""

import time
import random
import logging

from app.schemas.job import ModelType

logger = logging.getLogger(__name__)


def run_mock_inference(model_type: ModelType, input_text: str) -> dict:
    """
    Simulate model inference with realistic latency and structured output.

    Args:
        model_type: Which model pipeline to simulate.
        input_text: The text input to "analyze."

    Returns:
        A dictionary matching the real model output schema.
    """

    # Simulate variable processing time (100ms – 800ms)
    latency = random.uniform(0.1, 0.8)
    time.sleep(latency)

    logger.info(
        f"Mock inference | model={model_type.value} | "
        f"input_len={len(input_text)} | latency={latency:.3f}s"
    )

    if model_type == ModelType.SENTIMENT:
        return _mock_sentiment(input_text)
    elif model_type == ModelType.SUMMARIZATION:
        return _mock_summarization(input_text)
    elif model_type == ModelType.NER:
        return _mock_ner(input_text)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _mock_sentiment(text: str) -> dict:
    """Mock sentiment analysis — returns label + confidence score."""
    labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    label = random.choice(labels)
    score = round(random.uniform(0.70, 0.99), 4)

    return {
        "model": "mock-sentiment-v1",
        "label": label,
        "score": score,
        "details": [
            {"label": "POSITIVE", "score": round(random.uniform(0.0, 1.0), 4)},
            {"label": "NEGATIVE", "score": round(random.uniform(0.0, 1.0), 4)},
            {"label": "NEUTRAL", "score": round(random.uniform(0.0, 1.0), 4)},
        ],
    }


def _mock_summarization(text: str) -> dict:
    """Mock text summarization — returns a truncated version."""
    words = text.split()
    summary_len = max(5, len(words) // 3)
    summary = " ".join(words[:summary_len]) + "..."

    return {
        "model": "mock-summarization-v1",
        "summary": summary,
        "input_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": round(len(summary) / max(len(text), 1), 4),
    }


def _mock_ner(text: str) -> dict:
    """Mock Named Entity Recognition — returns fake entities."""
    mock_entities = [
        {"text": "NexusInfer", "label": "ORG", "start": 0, "end": 10, "score": 0.95},
        {"text": "FastAPI", "label": "TECH", "start": 15, "end": 22, "score": 0.88},
    ]

    return {
        "model": "mock-ner-v1",
        "entities": mock_entities[:min(3, len(text.split()))],
        "entity_count": min(3, len(text.split())),
    }
