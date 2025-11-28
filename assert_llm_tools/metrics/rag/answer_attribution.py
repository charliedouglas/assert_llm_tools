from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator
from sentence_transformers import SentenceTransformer
import numpy as np


class AnswerAttributionCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating how much of an answer is derived from context.

    Uses embedding similarity, n-gram overlap, and LLM evaluation.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        """
        Initialize answer attribution calculator.

        Args:
            llm_config: Configuration for LLM
            embedding_model: Name of sentence transformer model for embeddings
            verbose: Whether to include detailed score breakdown in the output
        """
        super().__init__(llm_config)
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.verbose = verbose

    def _calculate_embedding_similarity(self, answer: str, context: str) -> float:
        """
        Calculate cosine similarity between answer and context embeddings.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get embeddings
        answer_embedding = self.embedding_model.encode(answer)
        context_embedding = self.embedding_model.encode(context)

        # Calculate cosine similarity
        similarity = np.dot(answer_embedding, context_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)
        )

        return float(similarity)

    def _calculate_ngram_overlap(self, answer: str, context: str, n: int = 3) -> float:
        """
        Calculate n-gram overlap between answer and context.

        Args:
            answer: Generated answer
            context: Retrieved context
            n: Size of n-grams

        Returns:
            Overlap score between 0.0 and 1.0
        """
        answer_words = answer.lower().split()
        context_words = context.lower().split()

        if len(answer_words) < n:
            return 0.0

        answer_ngrams = set(
            " ".join(answer_words[i : i + n]) for i in range(len(answer_words) - n + 1)
        )
        context_ngrams = set(
            " ".join(context_words[i : i + n])
            for i in range(len(context_words) - n + 1)
        )

        if not answer_ngrams:
            return 0.0

        overlap = len(answer_ngrams.intersection(context_ngrams)) / len(answer_ngrams)
        return overlap

    def _calculate_llm_score(self, answer: str, context: str) -> float:
        """
        Use LLM to evaluate if the answer is derived from context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Attribution score between 0.0 and 1.0
        """
        prompt = f"""You are an expert evaluator. Assess whether the given answer appears to be derived from the provided context.

Context: {context}
Answer: {answer}

Rate on a scale of 0 to 1, where:
0.0: Answer shows no evidence of using the context
0.5: Answer partially uses the context but includes external information
1.0: Answer is completely derived from the context

Important: Your response must start with just the numerical score (0.00 to 1.00).
You may provide explanation after the score on a new line.

Score:"""

        response = self.llm.generate(prompt).strip()
        return self._extract_float_from_response(response)

    def calculate_score(self, answer: str, context: Union[str, List[str]]) -> Dict[str, any]:
        """
        Calculate overall answer attribution score.

        Args:
            answer: Generated answer
            context: Retrieved context(s)

        Returns:
            Dictionary with attribution score and optionally detailed breakdown
        """
        # Normalize context if it's a list
        context_text = self._normalize_context(context)

        # Calculate individual scores
        embedding_score = self._calculate_embedding_similarity(answer, context_text)
        ngram_score = self._calculate_ngram_overlap(answer, context_text)
        llm_score = self._calculate_llm_score(answer, context_text)

        # Combine scores with weights
        weights = {"embedding": 0.3, "ngram": 0.3, "llm": 0.4}

        final_score = (
            weights["embedding"] * embedding_score
            + weights["ngram"] * ngram_score
            + weights["llm"] * llm_score
        )

        final_score = max(0.0, min(1.0, final_score))

        result = {"answer_attribution": final_score}

        # Include detailed score breakdown when verbose is enabled
        if self.verbose:
            result["embedding_score"] = embedding_score
            result["ngram_score"] = ngram_score
            result["llm_score"] = llm_score
            result["answer"] = answer
            result["context"] = context

        return result


def calculate_answer_attribution(
    answer: str,
    context: Union[str, List[str]],
    llm_config: Optional[LLMConfig] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate how much of the answer appears to be derived from the provided context.

    Args:
        answer: The generated answer to evaluate
        context: Retrieved context(s). Can be a single string or list of strings.
        llm_config: Configuration for LLM-based evaluation
        verbose: If True, include detailed score breakdown (embedding, ngram, LLM scores)

    Returns:
        Dictionary containing:
            - answer_attribution: Combined attribution score (0-1)
            - embedding_score (only if verbose=True): Embedding similarity score
            - ngram_score (only if verbose=True): N-gram overlap score
            - llm_score (only if verbose=True): LLM-based attribution score
            - answer (only if verbose=True): The evaluated answer
            - context (only if verbose=True): The evaluated context
    """
    calculator = AnswerAttributionCalculator(llm_config, verbose=verbose)
    return calculator.calculate_score(answer, context)
