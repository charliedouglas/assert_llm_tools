from typing import Dict, List, Optional, Tuple
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine


class RedundancyCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating redundancy in text.

    Uses semantic similarity to identify redundant information, providing more
    robust detection than string-based methods.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        custom_instruction: Optional[str] = None,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
        verbose: bool = False
    ):
        """
        Initialize redundancy calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            similarity_threshold: Cosine similarity threshold for identifying redundancy (default: 0.85)
            embedding_model: Name of sentence transformer model for embeddings
            verbose: Whether to include detailed redundant pair analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer(embedding_model)
        self.verbose = verbose

    def _identify_redundant_segments_semantic(self, sentences: List[str]) -> List[Tuple[int, int, float]]:
        """
        Identify redundant sentence pairs using semantic similarity.

        Args:
            sentences: List of sentences to analyze

        Returns:
            List of tuples (index1, index2, similarity_score) for redundant pairs
        """
        if len(sentences) <= 1:
            return []

        # Generate embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Find pairs with high similarity
        redundant_pairs = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # Calculate cosine similarity
                similarity = 1 - cosine(embeddings[i], embeddings[j])

                # If similarity exceeds threshold, consider it redundant
                if similarity >= self.similarity_threshold:
                    redundant_pairs.append((i, j, float(similarity)))

        return redundant_pairs

    def calculate_score(self, text: str) -> Dict[str, any]:
        """
        Calculate redundancy score using semantic similarity.

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
                "redundant_pairs": [],
                "redundant_pair_count": 0,
            }

        # Identify redundant sentence pairs
        redundant_pairs = self._identify_redundant_segments_semantic(sentences)

        # Calculate redundancy score
        # Count unique sentences involved in redundancy
        redundant_sentence_indices = set()
        for i, j, _ in redundant_pairs:
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
        }

        # Include detailed redundant pair analysis when verbose is enabled
        if self.verbose:
            result["redundant_pairs"] = [
                {
                    "sentence_1_index": i,
                    "sentence_2_index": j,
                    "sentence_1": sentences[i],
                    "sentence_2": sentences[j],
                    "similarity": similarity
                }
                for i, j, similarity in redundant_pairs
            ]
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
    Calculate redundancy score using semantic similarity detection.

    This method identifies redundant sentences by comparing their semantic similarity using
    sentence embeddings. Sentences with cosine similarity above the threshold are considered
    redundant. This approach is more robust than string-based matching as it catches
    paraphrased redundancy.

    Args:
        text (str): The text to analyze for redundancy
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use (maintained for compatibility)
        custom_instruction (Optional[str]): Custom instruction (maintained for compatibility)
        similarity_threshold (float): Cosine similarity threshold for redundancy detection (default: 0.85)
        verbose (bool): If True, include detailed redundant pair analysis showing each pair of
            redundant sentences with their text and similarity scores

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
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
    return calculator.calculate_score(text)
