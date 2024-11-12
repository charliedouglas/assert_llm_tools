from typing import Dict, List, Union
import re
from assert_llm_tools.llm.config import LLMConfig
from assert_llm_tools.llm.bedrock import BedrockLLM
from assert_llm_tools.llm.openai import OpenAILLM


class RAGCompletenessCalculator:
    def __init__(self, llm_config: LLMConfig):
        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _extract_required_points(self, question: str) -> List[str]:
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


def calculate_completeness(
    question: str, answer: str, llm_config: LLMConfig
) -> Dict[str, Union[float, List[str], int]]:
    """
    Calculate completeness score by analyzing how well the answer addresses all required points from the question.

    Args:
        question (str): The original question asked
        answer (str): The generated answer to evaluate
        llm_config (LLMConfig): Configuration for the LLM to use

    Returns:
        Dict containing completeness scores and point analysis
    """
    calculator = RAGCompletenessCalculator(llm_config)

    # Extract required points and verify coverage
    required_points = calculator._extract_required_points(question)
    points_coverage = calculator._verify_points_coverage(required_points, answer)
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

    return {
        "completeness": completeness_score,
        "required_points_count": len(required_points),
        "covered_points_count": covered_points_count,
        "required_points": required_points,
        "points_not_covered": points_not_covered,
    }
