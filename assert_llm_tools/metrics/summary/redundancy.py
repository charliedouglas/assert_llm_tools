from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class RedundancyCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating redundancy in text.

    Identifies redundant information and calculates a redundancy score.
    """

    def _identify_redundant_segments(self, text: str) -> List[Dict[str, str]]:
        """
        Identify redundant segments in the text using LLM.

        Args:
            text: The text to analyze

        Returns:
            List of dictionaries containing original and repeated text
        """
        prompt = f"""
        System: You are a helpful assistant that identifies redundant information in text. 
        Find segments of text that express the same information in different ways or repeat information unnecessarily.
        For each redundant segment, provide the original text and its repetition.

        Human: Analyze this text for redundant information:
        {text}

        Format your response as follows:
        Original: [first occurrence of information]
        Repeated: [where the information is repeated]
        ---
        (Use --- to separate multiple instances)

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=500)
        segments = []

        if "---" in response:
            pairs = response.strip().split("---")
            for pair in pairs:
                if "Original:" in pair and "Repeated:" in pair:
                    original = pair.split("Original:")[1].split("Repeated:")[0].strip()
                    repeated = pair.split("Repeated:")[1].strip()
                    segments.append({"original": original, "repeated": repeated})

        return segments

    def calculate_score(self, text: str) -> Dict[str, any]:
        """
        Calculate redundancy score and identify redundant segments.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with redundancy score and segments
        """
        redundant_segments = self._identify_redundant_segments(text)

        total_length = len(text)
        redundant_length = sum(
            len(segment["repeated"]) for segment in redundant_segments
        )

        # Calculate raw redundancy (higher means more redundant)
        raw_redundancy = redundant_length / total_length if total_length > 0 else 0.0

        # Invert the score so 1 means no redundancy (better) and 0 means highly redundant (worse)
        redundancy_score = 1.0 - min(1.0, raw_redundancy)

        return {
            "redundancy_score": redundancy_score,
            "redundant_segments": redundant_segments,
            "segment_count": len(redundant_segments),
        }


def calculate_redundancy(
    text: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, any]:
    """
    Calculate redundancy score and identify redundant segments in the text.

    Args:
        text (str): The text to analyze for redundancy
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use

    Returns:
        Dict[str, any]: Dictionary containing:
            - redundancy_score: float between 0 and 1
              (1 = no redundancy/best, 0 = highly redundant/worst)
            - redundant_segments: List of dictionaries containing original and repeated text
            - segment_count: Number of redundant segments found
    """
    calculator = RedundancyCalculator(llm_config)
    return calculator.calculate_score(text)
