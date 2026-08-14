"""
NexusInfer — Inference Engine

Executes ML inference using real HuggingFace transformer models.
Each model type has a dedicated post-processing function that
normalizes the raw pipeline output into a clean, structured response.

The engine delegates model loading to the ModelManager singleton,
which handles caching and lifecycle management.
"""

import time
import logging
from typing import Any

from app.schemas.job import ModelType
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)


def run_inference(model_type: ModelType, input_text: str) -> dict[str, Any]:
    """
    Run ML inference using a real HuggingFace pipeline.

    Args:
        model_type: Which model pipeline to use.
        input_text: The text input to analyze.

    Returns:
        A structured dictionary with model name, results, and metadata.

    Raises:
        ValueError: If the model type is unsupported.
        RuntimeError: If the model fails to produce output.
    """
    logger.info(
        f"Inference request | model={model_type.value} | "
        f"input_len={len(input_text)}"
    )

    # Get the cached pipeline (loads on first use)
    pipeline = model_manager.get_pipeline(model_type)

    # Route to the appropriate handler
    if model_type == ModelType.SENTIMENT:
        return _run_sentiment(pipeline, input_text)
    elif model_type == ModelType.SUMMARIZATION:
        return _run_summarization(pipeline, input_text)
    elif model_type == ModelType.NER:
        return _run_ner(pipeline, input_text)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ──────────────────────────── Sentiment Analysis ────────────────────────────


def _run_sentiment(pipeline, text: str) -> dict:
    """
    Run sentiment analysis.

    Model: distilbert-base-uncased-finetuned-sst-2-english
    Output: label (POSITIVE/NEGATIVE) + confidence score
    """
    # Pipeline returns: [{"label": "POSITIVE", "score": 0.9998}]
    results = pipeline(text, truncation=True, max_length=512)

    if not results:
        raise RuntimeError("Sentiment model returned empty output")

    top_result = results[0]

    return {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment-analysis",
        "label": top_result["label"],
        "score": round(top_result["score"], 4),
        "input_length": len(text),
    }


# ──────────────────────────── Text Summarization ────────────────────────────


def _run_summarization(pipeline, text: str) -> dict:
    """
    Run abstractive text summarization.

    Model: sshleifer/distilbart-cnn-12-6
    Output: condensed summary of the input text
    """
    # Summarization needs enough text to work with
    min_input_length = 30
    input_word_count = len(text.split())

    # Set dynamic length bounds based on input
    max_length = min(150, max(30, input_word_count // 2))
    min_length = min(15, max(5, input_word_count // 4))

    results = pipeline(
        text,
        max_length=max_length,
        min_length=min_length,
        truncation=True,
        do_sample=False,
    )

    if not results:
        raise RuntimeError("Summarization model returned empty output")

    summary = results[0]["summary_text"]

    return {
        "model": "sshleifer/distilbart-cnn-12-6",
        "task": "summarization",
        "summary": summary,
        "input_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": round(len(summary) / max(len(text), 1), 4),
        "input_word_count": input_word_count,
        "summary_word_count": len(summary.split()),
    }


# ──────────────────────────── Named Entity Recognition ────────────────────────────


def _run_ner(pipeline, text: str) -> dict:
    """
    Run Named Entity Recognition.

    Model: dslim/bert-base-NER
    Output: list of detected entities with type, position, and confidence
    Labels: PER (person), ORG (organization), LOC (location), MISC (miscellaneous)
    """
    raw_entities = pipeline(text, aggregation_strategy="simple")

    # Normalize entity output
    entities = []
    for ent in raw_entities:
        entities.append({
            "text": ent["word"],
            "label": ent["entity_group"],
            "score": round(ent["score"], 4),
            "start": ent["start"],
            "end": ent["end"],
        })

    # Group entity counts by type
    entity_counts = {}
    for ent in entities:
        label = ent["label"]
        entity_counts[label] = entity_counts.get(label, 0) + 1

    return {
        "model": "dslim/bert-base-NER",
        "task": "named-entity-recognition",
        "entities": entities,
        "entity_count": len(entities),
        "entity_types": entity_counts,
        "input_length": len(text),
    }
