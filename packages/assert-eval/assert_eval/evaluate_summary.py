import logging
from typing import Dict, List, Optional

from assert_core.llm.config import LLMConfig

from .metrics.coverage import calculate_coverage
from .metrics.factual_alignment import calculate_factual_alignment
from .metrics.factual_consistency import calculate_factual_consistency
from .metrics.topic_preservation import calculate_topic_preservation

logger = logging.getLogger(__name__)

AVAILABLE_SUMMARY_METRICS = [
    "coverage",
    "factual_consistency",
    "factual_alignment",
    "topic_preservation",
]

LLM_REQUIRED_SUMMARY_METRICS = [
    "coverage",
    "factual_consistency",
    "factual_alignment",
    "topic_preservation",
]


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    llm_config: Optional[LLMConfig] = None,
    custom_prompt_instructions: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a summary using the specified metrics.

    Args:
        full_text: Original source text
        summary: Generated summary to evaluate
        metrics: List of metrics to calculate. Defaults to all available metrics.
        llm_config: Configuration for LLM-based metrics (required for all metrics)
        custom_prompt_instructions: Optional dict mapping metric names to custom prompt
            instructions. Example: {"coverage": "Apply strict scientific standards"}
        verbose: If True, include detailed analysis for LLM-based metrics showing
            individual claims/topics and their verification status

    Returns:
        Dictionary containing scores for each requested metric
    """
    if metrics is None:
        metrics = AVAILABLE_SUMMARY_METRICS

    invalid = set(metrics) - set(AVAILABLE_SUMMARY_METRICS)
    if invalid:
        raise ValueError(f"Invalid metrics: {invalid}")

    llm_metrics = set(metrics) & set(LLM_REQUIRED_SUMMARY_METRICS)
    if llm_metrics and llm_config is None:
        raise ValueError(f"LLM configuration required for metrics: {llm_metrics}")

    results = {}

    for metric in metrics:
        if metric == "coverage":
            if "coverage" not in results:
                instruction = custom_prompt_instructions.get("coverage") if custom_prompt_instructions else None
                results.update(calculate_coverage(full_text, summary, llm_config, instruction, verbose=verbose))

        elif metric == "factual_consistency":
            if "factual_consistency" not in results:
                instruction = custom_prompt_instructions.get("factual_consistency") if custom_prompt_instructions else None
                results.update(calculate_factual_consistency(full_text, summary, llm_config, instruction, verbose=verbose))

        elif metric == "factual_alignment":
            instruction = custom_prompt_instructions.get("factual_alignment") if custom_prompt_instructions else None
            results.update(
                calculate_factual_alignment(
                    full_text,
                    summary,
                    llm_config,
                    instruction,
                    verbose=verbose,
                    _precomputed_coverage=results if "coverage" in results else None,
                    _precomputed_consistency=results if "factual_consistency" in results else None,
                )
            )

        elif metric == "topic_preservation":
            instruction = custom_prompt_instructions.get("topic_preservation") if custom_prompt_instructions else None
            results.update(calculate_topic_preservation(full_text, summary, llm_config, instruction, verbose=verbose))

    return results
