from typing import Dict, List, Optional, Tuple
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
from nltk.tokenize import sent_tokenize


class RedundancyCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating redundancy in text.

    Uses LLM-based semantic analysis to identify redundant information.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        custom_instruction: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize redundancy calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed redundant pair analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

    def _identify_redundant_segments(self, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Identify redundant sentence pairs using LLM-based semantic analysis.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of tuples (index1, index2) for redundant pairs
        """
        if len(sentences) <= 1:
            return []

        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))

        prompt = f"""System: You are a redundancy detection assistant. Identify pairs of sentences that convey the same or highly overlapping information (paraphrases, restatements, or near-duplicates).

Sentences:
{numbered}

List each redundant pair as two indices separated by a comma, one pair per line.
If no redundant pairs exist, respond with "NONE".
Only output index pairs (e.g. "0,3") or "NONE". No other text."""

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

        response = self.llm.generate(prompt, max_tokens=500).strip()

        if response.upper() == "NONE":
            return []

        pairs = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if "," in line:
                parts = line.split(",")
                try:
                    i, j = int(parts[0].strip()), int(parts[1].strip())
                    if 0 <= i < len(sentences) and 0 <= j < len(sentences) and i != j:
                        pairs.append((min(i, j), max(i, j)))
                except (ValueError, IndexError):
                    continue

        # Deduplicate
        return list(set(pairs))

    def calculate_score(self, text: str) -> Dict[str, any]:
        """
        Calculate redundancy score using LLM-based semantic analysis.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with redundancy score and redundant segment pairs
        """
        # Split text into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            return {
                "redundancy_score": 1.0,  # Single sentence cannot be redundant
                "redundant_pair_count": 0,
                "total_sentences": len(sentences),
                "redundant_sentences_count": 0,
                "redundant_pairs": [],
            }

        # Identify redundant sentence pairs
        redundant_pairs = self._identify_redundant_segments(sentences)

        # Calculate redundancy score
        # Count unique sentences involved in redundancy
        redundant_sentence_indices = set()
        for i, j in redundant_pairs:
            redundant_sentence_indices.add(i)
            redundant_sentence_indices.add(j)

        # Calculate what percentage of sentences are involved in redundancy
        redundancy_ratio = len(redundant_sentence_indices) / len(sentences)

        # Invert the score so 1 means no redundancy (better) and 0 means highly redundant (worse)
        redundancy_score = 1.0 - redundancy_ratio

        result = {
            "redundancy_score": redundancy_score,
            "redundant_pair_count": len(redundant_pairs),
            "total_sentences": len(sentences),
            "redundant_sentences_count": len(redundant_sentence_indices),
            # Always include redundant_pairs to preserve API compatibility
            "redundant_pairs": [
                {
                    "sentence_1_index": i,
                    "sentence_2_index": j,
                    "sentence_1": sentences[i],
                    "sentence_2": sentences[j],
                }
                for i, j in redundant_pairs
            ],
        }

        # Include full sentence list when verbose is enabled
        if self.verbose:
            result["sentences"] = sentences

        return result


def calculate_redundancy(
    text: str,
    llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None,
    similarity_threshold: float = 0.85,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Calculate redundancy score using LLM-based semantic analysis.

    This method identifies redundant sentences by asking the LLM to find semantically
    similar or overlapping sentence pairs. This approach catches paraphrased redundancy
    effectively.

    Args:
        text (str): The text to analyze for redundancy
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction for evaluation
        similarity_threshold (float): Kept for API compatibility (unused)
        verbose (bool): If True, include detailed redundant pair analysis showing each pair of
            redundant sentences

    Returns:
        Dict[str, any]: Dictionary containing:
            - redundancy_score: float between 0 and 1
              (1 = no redundancy/best, 0 = highly redundant/worst)
            - redundant_pair_count: Number of redundant sentence pairs found
            - total_sentences: Total number of sentences in text
            - redundant_sentences_count: Number of unique sentences involved in redundancy
            - redundant_pairs (only if verbose=True): List of dicts with redundant sentence pairs
            - sentences (only if verbose=True): List of all sentences in the text
    """
    calculator = RedundancyCalculator(
        llm_config,
        custom_instruction=custom_instruction,
        verbose=verbose
    )
    return calculator.calculate_score(text)
