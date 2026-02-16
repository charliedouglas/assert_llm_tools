from typing import Dict, Union, List, Optional, Tuple, Any

# Import base calculator classes
from .metrics.base import BaseCalculator, SummaryMetricCalculator

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
from .utils import detect_and_mask_pii, remove_stopwords
from tqdm import tqdm
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
    remove_stopwords: bool = False,
    llm_config: Optional[LLMConfig] = None,
    show_progress: bool = True,
    mask_pii: bool = False,
    mask_pii_char: str = "*",
    mask_pii_preserve_partial: bool = False,
    mask_pii_entity_types: Optional[List[str]] = None,
    return_pii_info: bool = False,
    custom_prompt_instructions: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    **kwargs,  # Accept additional kwargs
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Evaluate a summary using specified metrics.

    Args:
        full_text: Original text
        summary: Generated summary to evaluate
        metrics: List of metrics to calculate. Defaults to all available metrics.
        remove_stopwords: Whether to remove stopwords before evaluation
        llm_config: Configuration for LLM-based metrics (e.g., faithfulness, topic_preservation)
        show_progress: Whether to show progress bar (default: True)
        mask_pii: Whether to mask personally identifiable information (PII) before evaluation (default: False)
        mask_pii_char: Character to use for masking PII (default: "*")
        mask_pii_preserve_partial: Whether to preserve part of the PII (e.g., for phone numbers: 123-***-***) (default: False)
        mask_pii_entity_types: List of PII entity types to detect and mask. If None, all supported types are used.
        return_pii_info: Whether to return information about detected PII (default: False)
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
        If return_pii_info is False:
            Dictionary containing scores for each metric
        If return_pii_info is True:
            Tuple containing:
                - Dictionary containing scores for each metric
                - Dictionary containing PII detection information
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

    # Handle PII masking if enabled
    pii_info = {}
    if mask_pii:
        logger.info("Masking PII in text and summary...")
        try:
            masked_full_text, full_text_pii = detect_and_mask_pii(
                full_text,
                entity_types=mask_pii_entity_types,
                mask_char=mask_pii_char,
                preserve_partial=mask_pii_preserve_partial
            )
            
            masked_summary, summary_pii = detect_and_mask_pii(
                summary,
                entity_types=mask_pii_entity_types,
                mask_char=mask_pii_char,
                preserve_partial=mask_pii_preserve_partial
            )
            
            # Store PII information if requested
            if return_pii_info:
                pii_info = {
                    "full_text_pii": full_text_pii,
                    "summary_pii": summary_pii,
                    "full_text_masked": masked_full_text != full_text,
                    "summary_masked": masked_summary != summary
                }
            
            # Update the texts with masked versions
            full_text = masked_full_text
            summary = masked_summary
            
            logger.info("PII masking complete.")
            
        except Exception as e:
            logger.error(f"Error during PII masking: {e}. Continuing with original text.")
            # Continue with original text in case of errors

    # Validate LLM config for metrics that require it
    llm_metrics = set(metrics) & set(LLM_REQUIRED_SUMMARY_METRICS)
    if llm_metrics and llm_config is None:
        raise ValueError(f"LLM configuration required for metrics: {llm_metrics}")

    # Initialize results dictionary
    results = {}

    # Calculate requested metrics
    metric_iterator = tqdm(
        metrics, disable=not show_progress, desc="Calculating metrics"
    )
        
    for metric in metric_iterator:
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

    # Return results with or without PII info
    if return_pii_info and mask_pii:
        return results, pii_info
    else:
        return results
