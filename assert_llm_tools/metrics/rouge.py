from rouge_score import rouge_scorer
from typing import Dict


def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores for a candidate summary against a reference text.

    Returns detailed metrics including precision, recall, and f-measure for each ROUGE type.

    Returns:
        Dict with keys in format:
        - rouge{n}_precision
        - rouge{n}_recall
        - rouge{n}_fmeasure
        where n is 1, 2, or L
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    detailed_scores = {}
    for rouge_type, score in scores.items():
        detailed_scores[f"{rouge_type}_precision"] = score.precision
        detailed_scores[f"{rouge_type}_recall"] = score.recall
        detailed_scores[f"{rouge_type}_fmeasure"] = score.fmeasure

    return detailed_scores
