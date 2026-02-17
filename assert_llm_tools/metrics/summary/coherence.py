from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator
from nltk.tokenize import sent_tokenize


class CoherenceCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating coherence of text.

    Measures logical flow, transitions, and overall text cohesion using LLM evaluation.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        custom_instruction: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize coherence calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed score breakdown in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

    def _evaluate_discourse_coherence(self, text: str) -> float:
        """
        Use LLM to evaluate discourse-level coherence.

        Args:
            text: Text to evaluate

        Returns:
            Discourse coherence score between 0.0 and 1.0
        """
        prompt = f"""Evaluate the coherence of the following text. Focus on:
1. Logical flow between sentences and paragraphs
2. Appropriate use of transition words and phrases
3. Consistent referencing (pronouns, definite articles)
4. Natural topic progression
5. Absence of abrupt topic shifts

Text to evaluate:
"{text}"

Rate the coherence on a scale of 0 to 1, where:
0.0: Completely incoherent - sentences appear random and disconnected
0.5: Partially coherent - some logical connections but with gaps or inconsistencies
1.0: Highly coherent - smooth and logical progression throughout

Important: Your response must be only a numerical score between 0.0 and 1.0."""

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

        # Get response from LLM and extract score
        response = self.llm.generate(prompt).strip()
        return self._extract_float_from_response(response)

    def calculate_score(self, text: str) -> Dict[str, any]:
        """
        Calculate overall coherence score.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with coherence score and optionally detailed breakdown
        """
        # Split text into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            result = {"coherence": 1.0, "sentence_count": len(sentences)}
            if self.verbose:
                result["discourse_score"] = 1.0
                result["sentences"] = sentences
            return result

        # Get discourse-based coherence from LLM
        discourse_score = self._evaluate_discourse_coherence(text)

        result = {"coherence": discourse_score, "sentence_count": len(sentences)}

        # Include detailed breakdown when verbose is enabled
        if self.verbose:
            result["discourse_score"] = discourse_score
            result["sentences"] = sentences

        return result


def calculate_coherence(
    summary: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate coherence of a summary.

    Args:
        summary (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for LLM-based evaluation
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed score breakdown showing discourse score
            and individual sentences

    Returns:
        Dict[str, float]: Dictionary containing:
            - coherence: Coherence score (0-1)
            - sentence_count: Number of sentences in the text
            - discourse_score (only if verbose=True): LLM-based discourse coherence score
            - sentences (only if verbose=True): List of sentences in the text

    Migration note (v0.9.0): `similarity_score` is no longer returned. The sentence-transformer
    cosine similarity approach was replaced with LLM-based discourse evaluation. Update any
    callers that reference `result["similarity_score"]`.
    """
    calculator = CoherenceCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(summary)
