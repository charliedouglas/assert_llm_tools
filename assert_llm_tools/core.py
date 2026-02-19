from typing import Dict, Union, List, Optional, Any

# Import base calculator classes
from .metrics.base import BaseCalculator, SummaryMetricCalculator

# Import note evaluation
from .metrics.note.evaluate_note import evaluate_note  # noqa: F401

# Import summary metrics
from .metrics.summary.coverage import calculate_coverage
from .metrics.summary.factual_consistency import calculate_factual_consistency
from .metrics.summary.factual_alignment import calculate_factual_alignment
from .metrics.summary.topic_preservation import calculate_topic_preservation
from .metrics.summary.redundancy import calculate_redundancy
from .metrics.summary.conciseness import calculate_conciseness_score
from .metrics.summary.coherence import calculate_coherence
# Old metric names (deprecated) - kept for backwards compatibility
from .metrics.summary.faithfulness import calculate_faithfulness
from .metrics.summary.hallucination import calculate_hallucination

from .llm.config import LLMConfig
import logging

# Configure logging
logger = logging.getLogger(__name__)


# Define available metrics
AVAILABLE_SUMMARY_METRICS = [
    "coverage",
    "factual_consistency",
    "factual_alignment",
    "topic_preservation",
    "redundancy",
    "conciseness",
    "coherence",
    # Deprecated metric names (kept for backwards compatibility)
    "faithfulness",  # Use 'coverage' instead
    "hallucination",  # Use 'factual_consistency' instead
]

# Define which metrics require LLM
LLM_REQUIRED_SUMMARY_METRICS = [
    "coverage",
    "factual_consistency",
    "factual_alignment",
    "topic_preservation",
    "conciseness",
    "coherence",
    # Deprecated metric names
    "faithfulness",
    "hallucination",
]


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    remove_stopwords: bool = False,  # no-op since v0.9.0; kept for API compatibility
    llm_config: Optional[LLMConfig] = None,
    show_progress: bool = True,  # no-op since v0.9.0 (tqdm removed); kept for API compatibility
    custom_prompt_instructions: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    **kwargs,  # Accept additional kwargs
) -> Dict[str, float]:
    """
    Evaluate a summary using specified metrics.

    Args:
        full_text: Original text
        summary: Generated summary to evaluate
        metrics: List of metrics to calculate. Defaults to all available metrics.
        remove_stopwords: Whether to remove stopwords before evaluation
        llm_config: Configuration for LLM-based metrics (e.g., coverage, factual_consistency)
        show_progress: No-op since v0.9.0 (tqdm removed). Kept for backwards compatibility.
        custom_prompt_instructions: Optional dictionary mapping metric names to custom prompt instructions.
            For LLM-based metrics (coverage, factual_consistency, factual_alignment, topic_preservation,
            redundancy, conciseness, coherence), you can provide additional instructions to customize the
            evaluation criteria.
            Example: {"coverage": "Apply strict scientific standards", "coherence": "Focus on narrative flow"}
            Note: Old metric names (faithfulness, hallucination) are deprecated but still supported.
        verbose: If True, include detailed analysis for LLM-based metrics showing individual claims,
            topics, and their verification status. Useful for debugging and understanding metric results.
        **kwargs: Additional keyword arguments for specific metrics

    Returns:
        Dictionary containing scores for each metric
    """
    # Default to all metrics if none specified
    if metrics is None:
        metrics = AVAILABLE_SUMMARY_METRICS

    # Validate metrics
    valid_metrics = set(AVAILABLE_SUMMARY_METRICS)
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")

    # Check for deprecated metric names and issue warnings
    deprecated_metrics = {
        "faithfulness": "coverage",
        "hallucination": "factual_consistency"
    }
    for metric in metrics:
        if metric in deprecated_metrics:
            import warnings
            warnings.warn(
                f"Metric '{metric}' is deprecated and will be removed in a future version. "
                f"Use '{deprecated_metrics[metric]}' instead.",
                DeprecationWarning,
                stacklevel=2
            )

    # Validate LLM config for metrics that require it
    llm_metrics = set(metrics) & set(LLM_REQUIRED_SUMMARY_METRICS)
    if llm_metrics and llm_config is None:
        raise ValueError(f"LLM configuration required for metrics: {llm_metrics}")

    # Initialize results dictionary
    results = {}

    # Calculate requested metrics
    for metric in metrics:
        if metric == "coverage":
            # Skip if already computed (e.g., by factual_alignment)
            if "coverage" not in results:
                custom_instruction = custom_prompt_instructions.get("coverage") if custom_prompt_instructions else None
                results.update(calculate_coverage(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "factual_consistency":
            # Skip if already computed (e.g., by factual_alignment)
            if "factual_consistency" not in results:
                custom_instruction = custom_prompt_instructions.get("factual_consistency") if custom_prompt_instructions else None
                results.update(calculate_factual_consistency(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "factual_alignment":
            custom_instruction = custom_prompt_instructions.get("factual_alignment") if custom_prompt_instructions else None
            # Pass precomputed results if available to avoid redundant LLM calls
            results.update(calculate_factual_alignment(
                full_text, summary, llm_config, custom_instruction, verbose=verbose,
                _precomputed_coverage=results if "coverage" in results else None,
                _precomputed_consistency=results if "factual_consistency" in results else None
            ))

        elif metric == "topic_preservation":
            custom_instruction = custom_prompt_instructions.get("topic_preservation") if custom_prompt_instructions else None
            results.update(calculate_topic_preservation(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "redundancy":
            custom_instruction = custom_prompt_instructions.get("redundancy") if custom_prompt_instructions else None
            results.update(calculate_redundancy(summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "conciseness":
            custom_instruction = custom_prompt_instructions.get("conciseness") if custom_prompt_instructions else None
            results.update(calculate_conciseness_score(
                full_text, summary, llm_config, custom_instruction, verbose=verbose
            ))

        elif metric == "coherence":
            custom_instruction = custom_prompt_instructions.get("coherence") if custom_prompt_instructions else None
            results.update(calculate_coherence(summary, llm_config, custom_instruction, verbose=verbose))

        # Deprecated metrics (backwards compatibility)
        elif metric == "faithfulness":
            custom_instruction = custom_prompt_instructions.get("faithfulness") if custom_prompt_instructions else None
            results.update(calculate_faithfulness(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "hallucination":
            custom_instruction = custom_prompt_instructions.get("hallucination") if custom_prompt_instructions else None
            results.update(calculate_hallucination(full_text, summary, llm_config, custom_instruction, verbose=verbose))

    return results
