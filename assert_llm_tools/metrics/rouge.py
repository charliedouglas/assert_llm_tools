from rouge_score import rouge_scorer


def calculate_rouge(reference: str, candidate: str) -> dict[str, float]:
    """
    Calculate ROUGE scores for a candidate summary against a reference text.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
