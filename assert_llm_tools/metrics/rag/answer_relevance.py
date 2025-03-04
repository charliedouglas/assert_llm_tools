from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class AnswerRelevanceCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating relevance of an answer to a question.

    Measures how well an answer addresses the original question.
    """

    def calculate_score(self, question: str, answer: str) -> float:
        """
        Calculate relevance score for an answer.

        Args:
            question: The original question
            answer: The generated answer to evaluate

        Returns:
            Relevance score between 0.0 and 1.0
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
        return self._extract_float_from_response(response)


# Wrapper functions to maintain the existing API
def calculate_answer_relevance(
    question: str,
    answer: str,
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate how relevant the answer is to the given question.

    Args:
        question: The input question
        answer: The generated answer to evaluate
        llm_config: Configuration for LLM-based evaluation

    Returns:
        Dictionary containing the answer_relevance score
    """
    calculator = AnswerRelevanceCalculator(llm_config)
    score = calculator.calculate_score(question, answer)

    return {"answer_relevance": score}
