from typing import Dict, Union, List, Optional
from .metrics.rouge import calculate_rouge
from .metrics.bleu import calculate_bleu
from .metrics.bert_score import calculate_bert_score

# Define available metrics
AVAILABLE_METRICS = ["rouge", "bleu", "bert_score"]


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    remove_stopwords: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a summary against the original text using selected metrics.

    Args:
        full_text (str): The original full text
        summary (str): The summary to evaluate
        metrics (List[str], optional): List of metrics to calculate.
            Options: ["rouge", "bleu", "bert_score"]. Defaults to all metrics.
        remove_stopwords (bool, optional): Whether to remove stopwords before evaluation.
            Defaults to False.

    Returns:
        Dict[str, float]: Dictionary containing requested evaluation metrics

    Raises:
        ValueError: If an invalid metric is specified
    """
    # If no metrics specified, use all available metrics
    if metrics is None:
        metrics = AVAILABLE_METRICS

    # Validate metrics
    invalid_metrics = [m for m in metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics: {invalid_metrics}. "
            f"Available metrics are: {AVAILABLE_METRICS}"
        )

    if remove_stopwords:
        from .utils import remove_stopwords as remove_sw

        full_text = remove_sw(full_text)
        summary = remove_sw(summary)

    # Initialize results dictionary
    results = {}

    # Calculate requested metrics
    if "rouge" in metrics:
        results.update(calculate_rouge(full_text, summary))

    if "bleu" in metrics:
        results["bleu"] = calculate_bleu(full_text, summary)

    if "bert_score" in metrics:
        results.update(calculate_bert_score(full_text, summary))

    return results
