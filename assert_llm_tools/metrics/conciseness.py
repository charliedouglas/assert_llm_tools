from typing import Dict, Optional
from assert_llm_tools.llm.config import LLMConfig
from assert_llm_tools.llm.bedrock import BedrockLLM
from assert_llm_tools.llm.openai import OpenAILLM
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


class ConcisenessCalculator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            # Default to Bedrock with Claude
            llm_config = LLMConfig(
                provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
            )

        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _get_llm_conciseness_evaluation(self, summary: str) -> float:
        """
        Uses an LLM to evaluate the conciseness of the summary.
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

        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except ValueError:
            return 0.5  # fallback score if LLM response isn't valid


def _calculate_statistical_score(source_text: str, summary: str) -> float:
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
    avg_words_per_sent = len(summary_words) / len(summary_sents)
    sentence_length_score = 1.0 - min(1, max(0, (avg_words_per_sent - 20) / 20))

    # Combine scores
    return 0.6 * compression_score + 0.4 * sentence_length_score


def calculate_conciseness_score(
    source_text: str, summary: str, llm_config: Optional[LLMConfig] = None
) -> float:
    """
    Calculates a conciseness score based on multiple factors:
    1. Compression ratio (how well the text was condensed)
    2. Information density (average word length and sentence complexity)
    3. Optional LLM evaluation of unnecessary verbosity

    Returns a score between 0 and 1, where 1 indicates optimal conciseness.
    """
    # Calculate base statistical metrics
    statistical_score = _calculate_statistical_score(source_text, summary)

    if llm_config:
        calculator = ConcisenessCalculator(llm_config)
        llm_score = calculator._get_llm_conciseness_evaluation(summary)
        # Combine scores with more weight on statistical analysis
        return 0.7 * statistical_score + 0.3 * llm_score

    return statistical_score
