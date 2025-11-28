from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class TopicPreservationCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating topic preservation in summaries.

    Measures how well a summary preserves the main topics from the original text.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize topic preservation calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed topic-level analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

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

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

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

        result = {
            "topic_preservation": topic_preservation_score,
            "reference_topics_count": len(reference_topics),
            "preserved_topics_count": len(preserved_topics),
            "missing_topics_count": len(missing_topics),
        }

        # Include detailed topic-level analysis when verbose is enabled
        if self.verbose:
            result["topics_analysis"] = [
                {"topic": topic, "is_preserved": present}
                for topic, present in zip(reference_topics, topic_present)
            ]
            result["preserved_topics"] = preserved_topics
            result["missing_topics"] = missing_topics

        return result


def calculate_topic_preservation(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, any]:
    """
    Evaluate how well a summary preserves the main topics from the original text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed topic-level analysis showing each extracted
            topic and whether it was preserved in the summary

    Returns:
        Dict[str, any]: Dictionary containing:
            - topic_preservation: Score from 0-1
            - reference_topics_count: Total topics extracted from reference
            - preserved_topics_count: Number of topics preserved in summary
            - missing_topics_count: Number of topics missing from summary
            - topics_analysis (only if verbose=True): List of dicts with topic text and preservation status
            - preserved_topics (only if verbose=True): List of preserved topic strings
            - missing_topics (only if verbose=True): List of missing topic strings
    """
    calculator = TopicPreservationCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(reference, candidate)
