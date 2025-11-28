from typing import Dict, Optional
from ...llm.config import LLMConfig
from .coverage import calculate_coverage
from .factual_consistency import calculate_factual_consistency


def calculate_factual_alignment(
    reference: str,
    candidate: str,
    llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate factual alignment score as the F1 score combining coverage and factual consistency.

    This metric provides a balanced measure of summary quality by combining:
    - Coverage (recall): What percentage of source claims appear in the summary
    - Factual Consistency (precision): What percentage of summary claims are supported by the source

    The F1 score is the harmonic mean of these two metrics, providing a single score that
    balances completeness and accuracy. This is useful when you want to ensure summaries
    are both comprehensive and factually grounded.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed claim-level analysis from both coverage
            and factual_consistency metrics

    Returns:
        Dict[str, float]: Dictionary containing:
            - factual_alignment: F1 score combining coverage and factual_consistency (0-1)
            - coverage: Recall score (how much of source is in summary)
            - factual_consistency: Precision score (how much of summary is supported)
            - reference_claims_count: Total claims in reference
            - summary_claims_count: Total claims in summary
            - claims_in_summary_count: Source claims found in summary
            - supported_claims_count: Summary claims supported by source
            - coverage_claims_analysis (only if verbose=True): Detailed coverage claim analysis
            - consistency_claims_analysis (only if verbose=True): Detailed consistency claim analysis
    """
    # Calculate coverage (recall)
    coverage_results = calculate_coverage(reference, candidate, llm_config, custom_instruction, verbose=verbose)
    coverage_score = coverage_results['coverage']

    # Calculate factual consistency (precision)
    consistency_results = calculate_factual_consistency(reference, candidate, llm_config, custom_instruction, verbose=verbose)
    consistency_score = consistency_results['factual_consistency']

    # Calculate F1 score (harmonic mean)
    if coverage_score + consistency_score > 0:
        factual_alignment_score = 2 * (coverage_score * consistency_score) / (coverage_score + consistency_score)
    else:
        factual_alignment_score = 0.0

    # Combine all results
    result = {
        "factual_alignment": factual_alignment_score,
        "coverage": coverage_score,
        "factual_consistency": consistency_score,
        "reference_claims_count": coverage_results['reference_claims_count'],
        "claims_in_summary_count": coverage_results['claims_in_summary_count'],
        "summary_claims_count": consistency_results['summary_claims_count'],
        "supported_claims_count": consistency_results['supported_claims_count'],
        "unsupported_claims_count": consistency_results['unsupported_claims_count'],
    }

    # Include detailed claim-level analysis when verbose is enabled
    if verbose:
        if 'claims_analysis' in coverage_results:
            result['coverage_claims_analysis'] = coverage_results['claims_analysis']
        if 'claims_analysis' in consistency_results:
            result['consistency_claims_analysis'] = consistency_results['claims_analysis']

    return result
