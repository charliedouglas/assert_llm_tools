from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class AnswerRelevanceCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating relevance of an answer to a question.

    Measures how well an answer addresses the original question.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, verbose: bool = False):
        """
        Initialize answer relevance calculator.

        Args:
            llm_config: Configuration for LLM
            verbose: Whether to include input echo in the output
        """
        super().__init__(llm_config)
        self.verbose = verbose

    def calculate_score(self, question: str, answer: str) -> Dict[str, any]:
        """
        Calculate relevance score for an answer.

        Args:
            question: The original question
            answer: The generated answer to evaluate

        Returns:
            Dictionary with relevance score and optionally input echo
        """
        prompt = f"""You are an expert evaluator. Assess how relevant the following answer is to the given question.

Question: {question}
Answer: {answer}

Rate the relevance on a scale of 0 to 1, where:
0.0: Completely irrelevant - The answer has no connection to the question
0.5: Partially relevant - The answer addresses some aspects but misses key points or includes irrelevant information
1.0: Highly relevant - The answer directly addresses the question

Important: Your response must start with just the numerical score (0.0 to 1.0).
You may provide explanation after the score on a new line.

Score:"""

        # Get response from LLM and extract score
        response = self.llm.generate(prompt).strip()
        score = self._extract_float_from_response(response)

        result = {"answer_relevance": score}

        if self.verbose:
            result["question"] = question
            result["answer"] = answer

        return result


# Wrapper functions to maintain the existing API
def calculate_answer_relevance(
    question: str,
    answer: str,
    llm_config: Optional[LLMConfig] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate how relevant the answer is to the given question.

    Args:
        question: The input question
        answer: The generated answer to evaluate
        llm_config: Configuration for LLM-based evaluation
        verbose: If True, include the question and answer in the output

    Returns:
        Dictionary containing:
            - answer_relevance: Score from 0-1
            - question (only if verbose=True): The input question
            - answer (only if verbose=True): The evaluated answer
    """
    calculator = AnswerRelevanceCalculator(llm_config, verbose=verbose)
    return calculator.calculate_score(question, answer)
