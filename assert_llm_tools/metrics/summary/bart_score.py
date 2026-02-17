from typing import Dict
import torch
from transformers import BartTokenizer, BartForConditionalGeneration


class BARTScorer:
    def __init__(self, device=None, checkpoint="facebook/bart-large-cnn"):
        """Initialize BARTScore"""
        # Set up device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

    def score(self, src: str, tgt: str) -> float:
        """Calculate BARTScore from source to target text"""
        with torch.no_grad():
            inputs = self.tokenizer(
                [src], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            targets = self.tokenizer(
                [tgt], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids,
            )

            log_likelihood = outputs.loss * targets.attention_mask.sum(1)
            normalized_score = (-log_likelihood / targets.attention_mask.sum(1)).item()

            return normalized_score


def calculate_bart_score(
    reference: str,
    candidate: str,
    checkpoint: str = "facebook/bart-large-cnn",
) -> Dict[str, float]:
    """
    Calculate BARTScore for a candidate summary against a reference text.

    Args:
        reference (str): The reference text
        candidate (str): The candidate summary to evaluate
        checkpoint (str): The BART model checkpoint to use (default: facebook/bart-large-cnn)

    Returns:
        Dict[str, float]: Dictionary containing the BARTScore
    """
    scorer = BARTScorer(checkpoint=checkpoint)

    # Calculate score in both directions (as per the paper)
    score_src2tgt = scorer.score(reference, candidate)  # reference -> candidate
    score_tgt2src = scorer.score(candidate, reference)  # candidate -> reference

    # Average bidirectional scores
    avg_score = (score_src2tgt + score_tgt2src) / 2

    return {"bart_score": avg_score}
