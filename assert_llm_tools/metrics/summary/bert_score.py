from typing import Dict, Literal
import bert_score
import torch
import warnings

# Define valid model options using Literal type
ModelType = Literal[
    "microsoft/deberta-base-mnli",
    "microsoft/deberta-xlarge-mnli",
]


def calculate_bert_score(
    reference: str,
    candidate: str,
    model_type: ModelType = "microsoft/deberta-base-mnli",
) -> Dict[str, float]:
    """
    Calculate BERTScore for a candidate summary against a reference text.

    Args:
        reference (str): The reference text
        candidate (str): The candidate summary to evaluate
        model_type (ModelType): The model to use for BERTScore calculation. Options are:
            - "microsoft/deberta-base-mnli" (~86M parameters)
            - "microsoft/deberta-xlarge-mnli" (~750M parameters)

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 scores
    """
    references = [reference]
    candidates = [candidate]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score.score(
        cands=candidates,
        refs=references,
        lang="en",
        device=device,
        model_type=model_type,
        verbose=True,
    )

    scores = {
        "bert_score_precision": P.item(),
        "bert_score_recall": R.item(),
        "bert_score_f1": F1.item(),
    }

    return scores
