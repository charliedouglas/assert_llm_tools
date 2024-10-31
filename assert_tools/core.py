from typing import Dict, Union, List
from .metrics.rouge import calculate_rouge
from .metrics.bleu import calculate_bleu


def evaluate_summary(full_text: str, summary: str) -> Dict[str, float]:
    """
    Evaluate a summary against the original text using ROUGE and BLEU scores.

    Args:
        full_text (str): The original full text
        summary (str): The summary to evaluate

    Returns:
        Dict[str, float]: Dictionary containing various evaluation metrics
    """
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(full_text, summary)

    # Calculate BLEU score
    bleu_score = calculate_bleu(full_text, summary)

    # Combine all metrics
    metrics = {**rouge_scores, "bleu": bleu_score}

    return metrics
