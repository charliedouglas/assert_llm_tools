from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class TopicPreservationCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating topic preservation in summaries.

    Measures how well a summary preserves the main topics from the original text.
    """

    def _check_topics_in_summary(self, topics: List[str], summary: str) -> List[bool]:
        """
        Check if topics from the original text are present in the summary.

        Args:
            topics: List of topics to check
            summary: Summary text to analyze

        Returns:
            List of boolean values indicating if each topic is present
        """
        topics_str = "\n".join([f"- {topic}" for topic in topics])
        prompt = f"""
        System: You are a topic coverage analysis assistant. Your task is to check if specific topics are present in a summary.

        For each topic listed below, respond with ONLY "yes" or "no" to indicate if the topic is covered in the summary.
        Respond with one answer per line, nothing else.

        Summary: {summary}

        Topics to check:
        {topics_str}

        Answer with yes/no for each topic:"""

        response = self.llm.generate(prompt, max_tokens=500)
        results = [
            line.strip().lower()
            for line in response.strip().split("\n")
            if line.strip()
        ]
        return ["yes" in result for result in results]

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, any]:
        """
        Calculate topic preservation score.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with topic preservation score and analysis
        """
        # First, extract topics from reference text
        reference_topics = self._extract_topics(reference)

        # Then check which topics are present in the summary
        topic_present = self._check_topics_in_summary(reference_topics, candidate)

        # Separate preserved and missing topics
        preserved_topics = [
            topic for topic, present in zip(reference_topics, topic_present) if present
        ]
        missing_topics = [
            topic
            for topic, present in zip(reference_topics, topic_present)
            if not present
        ]

        # Calculate preservation score
        topic_preservation_score = (
            len(preserved_topics) / len(reference_topics) if reference_topics else 0.0
        )

        return {
            "topic_preservation": topic_preservation_score,
            "reference_topics": reference_topics,
            "preserved_topics": preserved_topics,
            "missing_topics": missing_topics,
        }


def calculate_topic_preservation(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, any]:
    """
    Evaluate how well a summary preserves the main topics from the original text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use

    Returns:
        Dict[str, any]: Dictionary containing topic preservation score and analysis
    """
    calculator = TopicPreservationCalculator(llm_config)
    return calculator.calculate_score(reference, candidate)
