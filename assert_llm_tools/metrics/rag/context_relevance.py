from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class ContextRelevanceCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating relevance of retrieved context to a question.

    Measures how well retrieved context relates to the original question.
    """

    def calculate_score(self, question: str, context: Union[str, List[str]]) -> float:
        """
        Calculate relevance score for retrieved context.

        Args:
            question: The original question
            context: Retrieved context as string or list of strings

        Returns:
            Relevance score between 0.0 and 1.0
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
        return self._extract_float_from_response(response)


def calculate_context_relevance(
    question: str,
    context: Union[str, List[str]],
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate how relevant the retrieved context is to the given question.

    Args:
        question: The input question
        context: Retrieved context(s). Can be a single string or list of strings.
        llm_config: Configuration for LLM-based evaluation

    Returns:
        Dictionary containing the context_relevance score
    """
    calculator = ContextRelevanceCalculator(llm_config)
    score = calculator.calculate_score(question, context)

    return {"context_relevance": score}
