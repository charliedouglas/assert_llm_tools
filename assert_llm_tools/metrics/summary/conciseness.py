from typing import Dict, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
from nltk.tokenize import sent_tokenize, word_tokenize


class ConcisenessCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating conciseness of summaries.

    Measures information density and brevity of expression.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize conciseness calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed score breakdown in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

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

        Just return the number, nothing else."""

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

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

    def calculate_score(self, source_text: str, summary: str) -> Dict[str, any]:
        """
        Calculate overall conciseness score.

        Args:
            source_text: Original full text
            summary: Summary to evaluate

        Returns:
            Dictionary with conciseness score and optionally detailed breakdown
        """
        # Get basic text metrics for verbose output
        source_words = word_tokenize(source_text)
        summary_words = word_tokenize(summary)
        summary_sents = sent_tokenize(summary)

        # Calculate base statistical metrics
        statistical_score = self._calculate_statistical_score(source_text, summary)

        # Get LLM-based evaluation
        llm_score = self._get_llm_conciseness_evaluation(summary)

        # Combine scores with more weight on statistical analysis
        final_score = 0.7 * statistical_score + 0.3 * llm_score

        result = {"conciseness": final_score}

        # Include detailed breakdown when verbose is enabled
        if self.verbose:
            compression_ratio = len(summary_words) / len(source_words)
            avg_words_per_sent = len(summary_words) / max(1, len(summary_sents))
            result["statistical_score"] = statistical_score
            result["llm_score"] = llm_score
            result["compression_ratio"] = compression_ratio
            result["source_word_count"] = len(source_words)
            result["summary_word_count"] = len(summary_words)
            result["summary_sentence_count"] = len(summary_sents)
            result["avg_words_per_sentence"] = avg_words_per_sent

        return result


def calculate_conciseness_score(
    source_text: str, summary: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, any]:
    """
    Calculate a conciseness score based on multiple factors:
    1. Compression ratio (how well the text was condensed)
    2. Information density (average word length and sentence complexity)
    3. LLM evaluation of unnecessary verbosity

    Args:
        source_text (str): The original full text
        summary (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for LLM-based evaluation
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed score breakdown showing statistical score,
            LLM score, compression ratio, and word/sentence counts

    Returns:
        Dict[str, any]: Dictionary containing:
            - conciseness: Combined conciseness score (0-1)
            - statistical_score (only if verbose=True): Statistical analysis score
            - llm_score (only if verbose=True): LLM-based evaluation score
            - compression_ratio (only if verbose=True): Summary length / source length
            - source_word_count (only if verbose=True): Word count in source
            - summary_word_count (only if verbose=True): Word count in summary
            - summary_sentence_count (only if verbose=True): Sentence count in summary
            - avg_words_per_sentence (only if verbose=True): Average words per sentence
    """
    calculator = ConcisenessCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(source_text, summary)
