from typing import Dict
import bert_score
import torch


def calculate_bert_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate BERTScore for a candidate summary against a reference text.

    BERTScore leverages the pre-trained contextual embeddings from BERT and matches words
    in candidate and reference sentences by cosine similarity.

    Args:
        reference (str): The reference text
        candidate (str): The candidate summary to evaluate

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1 scores
    """
    # Convert single strings to lists as bert_score expects lists
    references = [reference]
    candidates = [candidate]

    # Use cuda if available for faster computation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate BERTScore
    P, R, F1 = bert_score.score(
        cands=candidates, refs=references, lang="en", device=device, verbose=False
    )

    # Convert tensor values to float
    scores = {
        "bert_score_precision": P.item(),
        "bert_score_recall": R.item(),
        "bert_score_f1": F1.item(),
    }

    return scores
