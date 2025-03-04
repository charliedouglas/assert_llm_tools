from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class CoherenceCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating coherence of text.

    Measures logical flow, transitions, and overall text cohesion.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize coherence calculator.

        Args:
            llm_config: Configuration for LLM
            embedding_model: Name of sentence transformer model for embeddings
        """
        super().__init__(llm_config)
        # Initialize embedding model for semantic analysis
        self.embedding_model = SentenceTransformer(embedding_model)

    def _calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """
        Calculate average cosine similarity between consecutive sentences.

        Args:
            sentences: List of sentences to analyze

        Returns:
            Average similarity score between consecutive sentences
        """
        if len(sentences) <= 1:
            return 1.0  # If only one sentence, it's coherent by default

        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        # Return average similarity
        return float(np.mean(similarities))

    def _evaluate_discourse_coherence(self, text: str) -> float:
        """
        Use LLM to evaluate discourse-level coherence.

        Args:
            text: Text to evaluate

        Returns:
            Discourse coherence score between 0.0 and 1.0
        """
        prompt = f"""Evaluate the coherence of the following text. Focus on:
1. Logical flow between sentences and paragraphs
2. Appropriate use of transition words and phrases
3. Consistent referencing (pronouns, definite articles)
4. Natural topic progression
5. Absence of abrupt topic shifts

Text to evaluate:
"{text}"

Rate the coherence on a scale of 0 to 1, where:
0.0: Completely incoherent - sentences appear random and disconnected
0.5: Partially coherent - some logical connections but with gaps or inconsistencies
1.0: Highly coherent - smooth and logical progression throughout

Important: Your response must be only a numerical score between 0.0 and 1.0.
"""

        # Get response from LLM and extract score
        response = self.llm.generate(prompt).strip()
        return self._extract_float_from_response(response)

    def calculate_score(self, text: str) -> float:
        """
        Calculate overall coherence score.

        Args:
            text: Text to evaluate

        Returns:
            Coherence score between 0.0 and 1.0
        """
        # Split text into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            return 1.0  # Single sentence is coherent by default

        # Get similarity-based coherence
        similarity_score = self._calculate_sentence_similarity(sentences)

        # Get discourse-based coherence
        discourse_score = self._evaluate_discourse_coherence(text)

        # Combine scores (weighted more toward discourse evaluation)
        final_score = 0.3 * similarity_score + 0.7 * discourse_score

        return final_score


def calculate_coherence(
    summary: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, float]:
    """
    Evaluate coherence of a summary.

    Args:
        summary (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for LLM-based evaluation

    Returns:
        Dict[str, float]: Dictionary containing the coherence score
    """
    calculator = CoherenceCalculator(llm_config)
    score = calculator.calculate_score(summary)

    return {"coherence": score}
