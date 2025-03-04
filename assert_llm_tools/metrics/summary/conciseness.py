from typing import Dict, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
from nltk.tokenize import sent_tokenize, word_tokenize


class ConcisenessCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating conciseness of summaries.

    Measures information density and brevity of expression.
    """

    def _get_llm_conciseness_evaluation(self, summary: str) -> float:
        """
        Use LLM to evaluate the conciseness of the summary.

        Args:
            summary: Summary to evaluate

        Returns:
            Conciseness score between 0.0 and 1.0
        """
        prompt = f"""
        Evaluate the conciseness of the following summary. Consider:
        1. Are there unnecessary words or phrases?
        2. Could the same information be expressed more briefly?
        3. Is there any redundant information?
        
        Summary: {summary}
        
        Return a single float score between 0 and 1, where:
        - 1.0 means perfectly concise with no unnecessary words
        - 0.0 means extremely verbose with significant redundancy
        
        Just return the number, nothing else.
        """

        response = self.llm.generate(prompt)
        return self._extract_float_from_response(response)

    def _calculate_statistical_score(self, source_text: str, summary: str) -> float:
        """
        Calculate conciseness score based on statistical text features.

        Args:
            source_text: Original full text
            summary: Summary to evaluate

        Returns:
            Statistical conciseness score between 0.0 and 1.0
        """
        # Get basic text metrics
        source_words = word_tokenize(source_text)
        summary_words = word_tokenize(summary)
        source_sents = sent_tokenize(source_text)
        summary_sents = sent_tokenize(summary)

        # Calculate compression ratio (optimal range: 0.2 - 0.4)
        compression_ratio = len(summary_words) / len(source_words)
        compression_score = 1.0 - abs(0.3 - compression_ratio) / 0.3
        compression_score = max(0, min(1, compression_score))

        # Calculate average words per sentence (penalize very long sentences)
        avg_words_per_sent = len(summary_words) / max(1, len(summary_sents))
        sentence_length_score = 1.0 - min(1, max(0, (avg_words_per_sent - 20) / 20))

        # Combine scores
        return 0.6 * compression_score + 0.4 * sentence_length_score

    def calculate_score(self, source_text: str, summary: str) -> float:
        """
        Calculate overall conciseness score.

        Args:
            source_text: Original full text
            summary: Summary to evaluate

        Returns:
            Conciseness score between 0.0 and 1.0
        """
        # Calculate base statistical metrics
        statistical_score = self._calculate_statistical_score(source_text, summary)

        # Get LLM-based evaluation
        llm_score = self._get_llm_conciseness_evaluation(summary)

        # Combine scores with more weight on statistical analysis
        return 0.7 * statistical_score + 0.3 * llm_score


def calculate_conciseness_score(
    source_text: str, summary: str, llm_config: Optional[LLMConfig] = None
) -> float:
    """
    Calculate a conciseness score based on multiple factors:
    1. Compression ratio (how well the text was condensed)
    2. Information density (average word length and sentence complexity)
    3. LLM evaluation of unnecessary verbosity

    Args:
        source_text (str): The original full text
        summary (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for LLM-based evaluation

    Returns:
        float: Conciseness score between 0 and 1, where 1 indicates optimal conciseness
    """
    calculator = ConcisenessCalculator(llm_config)
    return calculator.calculate_score(source_text, summary)
