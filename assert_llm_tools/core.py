from typing import Dict, Union, List, Optional
from .metrics.rouge import calculate_rouge
from .metrics.bleu import calculate_bleu
from .metrics.bert_score import calculate_bert_score
from .metrics.faithfulness import calculate_faithfulness
from .metrics.topic_preservation import calculate_topic_preservation
from .llm.config import LLMConfig

# Define available metrics
AVAILABLE_METRICS = [
    "rouge",
    "bleu",
    "bert_score",
    "faithfulness",
    "topic_preservation",
]


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    remove_stopwords: bool = False,
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate a summary using specified metrics.

    Args:
        full_text: Original text
        summary: Generated summary to evaluate
        metrics: List of metrics to calculate. Defaults to all available metrics.
        remove_stopwords: Whether to remove stopwords before evaluation
        llm_config: Configuration for LLM-based metrics (e.g., faithfulness, topic_preservation)

    Returns:
        Dictionary containing scores for each metric
    """
    # Default to all metrics if none specified
    if metrics is None:
        metrics = AVAILABLE_METRICS

    # Validate metrics
    valid_metrics = set(AVAILABLE_METRICS)
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")

    # Initialize results dictionary
    results = {}

    # Calculate requested metrics
    if "rouge" in metrics:
        results.update(calculate_rouge(full_text, summary))

    if "bleu" in metrics:
        results["bleu"] = calculate_bleu(full_text, summary)

    if "bert_score" in metrics:
        results.update(calculate_bert_score(full_text, summary))

    if "faithfulness" in metrics:
        results.update(calculate_faithfulness(full_text, summary, llm_config))

    if "topic_preservation" in metrics:
        results.update(calculate_topic_preservation(full_text, summary, llm_config))

    return results
