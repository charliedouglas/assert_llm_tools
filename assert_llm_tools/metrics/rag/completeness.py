from typing import Dict, List, Union
import re
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class CompletenessCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating completeness of RAG answers.

    Measures how well an answer addresses all aspects of a question.
    """

    def __init__(self, llm_config: LLMConfig = None, verbose: bool = False):
        """
        Initialize completeness calculator.

        Args:
            llm_config: Configuration for LLM
            verbose: Whether to include detailed point-level analysis in the output
        """
        super().__init__(llm_config)
        self.verbose = verbose

    def _extract_required_points(self, question: str) -> List[str]:
        """
        Extract key points that a complete answer should address.

        Args:
            question: The original question

        Returns:
            List of required points for a complete answer
        """
        prompt = f"""
        System: You are a helpful assistant that analyzes questions to determine what points need to be addressed for a complete answer. List only the key points, one per line, without any preamble or numbering.

        Human: What key points must be covered to completely answer this question: {question}

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=500)
        # Clean up the response by removing empty lines and any numbered prefixes
        points = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Remove numbering if present (e.g., "1.", "1)", "(1)")
            line = re.sub(r"^\s*[\d\.)\-]+\s*", "", line)
            if line:
                points.append(line)
        return points

    def _verify_points_coverage(self, points: List[str], answer: str) -> List[bool]:
        """
        Verify which required points are addressed in the answer.

        Args:
            points: List of required points
            answer: The generated answer

        Returns:
            List of boolean values indicating if each point is covered
        """
        points_text = "\n".join(points)
        prompt = f"""
        System: You are a helpful assistant that verifies if an answer addresses required points. Respond ONLY with 'true' or 'false' for each point, one per line.

        Answer to analyze:
        ---
        {answer}
        ---

        For each point below, respond with ONLY 'true' or 'false' to indicate if the answer adequately addresses it:
        ---
        {points_text}
        ---

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=300)
        results = []
        for line in response.strip().split("\n"):
            line = line.strip().lower()
            if line in ["true", "false"]:
                results.append(line == "true")

        # Ensure we have a result for each point
        if len(results) != len(points):
            results.extend([False] * (len(points) - len(results)))

        return results[: len(points)]

    def calculate_score(
        self, question: str, answer: str
    ) -> Dict[str, Union[float, List[str], int]]:
        """
        Calculate completeness score for a RAG answer.

        Args:
            question: The original question
            answer: The generated answer

        Returns:
            Dictionary with completeness score and point analysis
        """
        # Extract required points and verify coverage
        required_points = self._extract_required_points(question)
        points_coverage = self._verify_points_coverage(required_points, answer)
        covered_points_count = sum(points_coverage)

        # Identify missing points
        points_not_covered = [
            point
            for point, is_covered in zip(required_points, points_coverage)
            if not is_covered
        ]

        # Calculate completeness score
        completeness_score = (
            (covered_points_count / len(required_points)) if required_points else 1.0
        )

        result = {
            "completeness": completeness_score,
            "required_points_count": len(required_points),
            "covered_points_count": covered_points_count,
        }

        # Include detailed point-level analysis when verbose is enabled
        if self.verbose:
            result["points_analysis"] = [
                {"point": point, "is_covered": is_covered}
                for point, is_covered in zip(required_points, points_coverage)
            ]
            result["points_not_covered"] = points_not_covered

        return result


def calculate_completeness(
    question: str, answer: str, llm_config: LLMConfig, verbose: bool = False
) -> Dict[str, Union[float, List[str], int]]:
    """
    Calculate completeness score by analyzing how well the answer addresses all required points from the question.

    Args:
        question (str): The original question asked
        answer (str): The generated answer to evaluate
        llm_config (LLMConfig): Configuration for the LLM to use
        verbose (bool): If True, include detailed point-level analysis

    Returns:
        Dict containing:
            - completeness: Score from 0-1
            - required_points_count: Number of required points extracted
            - covered_points_count: Number of points covered by the answer
            - points_analysis (only if verbose=True): List of points with coverage status
            - points_not_covered (only if verbose=True): List of uncovered point strings
    """
    calculator = CompletenessCalculator(llm_config, verbose=verbose)
    return calculator.calculate_score(question, answer)
