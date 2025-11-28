from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class ContextRelevanceCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating relevance of retrieved context to a question.

    Measures how well retrieved context relates to the original question.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, verbose: bool = False):
        """
        Initialize context relevance calculator.

        Args:
            llm_config: Configuration for LLM
            verbose: Whether to include input echo in the output
        """
        super().__init__(llm_config)
        self.verbose = verbose

    def calculate_score(self, question: str, context: Union[str, List[str]]) -> Dict[str, any]:
        """
        Calculate relevance score for retrieved context.

        Args:
            question: The original question
            context: Retrieved context as string or list of strings

        Returns:
            Dictionary with relevance score and optionally input echo
        """
        # Normalize context if it's a list
        context_text = self._normalize_context(context)

        prompt = f"""You are an expert evaluator. Assess how relevant the retrieved context is to the given question.

Question: {question}
Retrieved Context: {context_text}

Rate the relevance on a scale of 0 to 1, where:
0.0: Completely irrelevant - The context has no connection to the question
0.5: Partially relevant - The context contains some relevant information but includes unnecessary content or misses key aspects
1.0: Highly relevant - The context contains precisely the information needed to answer the question

Important: Your response must start with just the numerical score between 0.00 to 1.00.

Score:"""

        # Get response from LLM and extract score
        response = self.llm.generate(prompt).strip()
        score = self._extract_float_from_response(response)

        result = {"context_relevance": score}

        if self.verbose:
            result["question"] = question
            result["context"] = context

        return result


def calculate_context_relevance(
    question: str,
    context: Union[str, List[str]],
    llm_config: Optional[LLMConfig] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate how relevant the retrieved context is to the given question.

    Args:
        question: The input question
        context: Retrieved context(s). Can be a single string or list of strings.
        llm_config: Configuration for LLM-based evaluation
        verbose: If True, include the question and context in the output

    Returns:
        Dictionary containing:
            - context_relevance: Score from 0-1
            - question (only if verbose=True): The input question
            - context (only if verbose=True): The evaluated context
    """
    calculator = ContextRelevanceCalculator(llm_config, verbose=verbose)
    return calculator.calculate_score(question, context)
