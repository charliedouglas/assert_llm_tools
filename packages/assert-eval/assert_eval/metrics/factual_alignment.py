from typing import Dict, Optional

from assert_core.llm.config import LLMConfig

from .coverage import calculate_coverage
from .factual_consistency import calculate_factual_consistency


def calculate_factual_alignment(
    reference: str,
    candidate: str,
    llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None,
    verbose: bool = False,
    _precomputed_coverage: Optional[Dict[str, float]] = None,
    _precomputed_consistency: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Calculate factual alignment score as the F1 score combining coverage and factual consistency.

    This metric provides a balanced measure of summary quality by combining:
    - Coverage (recall): What percentage of source claims appear in the summary
    - Factual Consistency (precision): What percentage of summary claims are supported by the source

    The F1 score is the harmonic mean of these two metrics, providing a single score that
    balances completeness and accuracy.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt
        verbose (bool): If True, include detailed claim-level analysis from both metrics
        _precomputed_coverage (Optional[Dict[str, float]]): Internal parameter to pass
            precomputed coverage results and avoid redundant LLM calls
        _precomputed_consistency (Optional[Dict[str, float]]): Internal parameter to pass
            precomputed consistency results and avoid redundant LLM calls

    Returns:
        Dict[str, float]: Dictionary containing:
            - factual_alignment: F1 score combining coverage and factual_consistency (0-1)
            - coverage: Recall score
            - factual_consistency: Precision score
            - reference_claims_count: Total claims in reference
            - summary_claims_count: Total claims in summary
            - claims_in_summary_count: Source claims found in summary
            - supported_claims_count: Summary claims supported by source
            - unsupported_claims_count: Summary claims not supported by source
            - coverage_claims_analysis (only if verbose=True)
            - consistency_claims_analysis (only if verbose=True)
    """
    if _precomputed_coverage is not None and "coverage" in _precomputed_coverage:
        coverage_results = _precomputed_coverage
        coverage_score = coverage_results["coverage"]
    else:
        coverage_results = calculate_coverage(reference, candidate, llm_config, custom_instruction, verbose=verbose)
        coverage_score = coverage_results["coverage"]

    if _precomputed_consistency is not None and "factual_consistency" in _precomputed_consistency:
        consistency_results = _precomputed_consistency
        consistency_score = consistency_results["factual_consistency"]
    else:
        consistency_results = calculate_factual_consistency(
            reference, candidate, llm_config, custom_instruction, verbose=verbose
        )
        consistency_score = consistency_results["factual_consistency"]

    if coverage_score + consistency_score > 0:
        factual_alignment_score = 2 * (coverage_score * consistency_score) / (coverage_score + consistency_score)
    else:
        factual_alignment_score = 0.0

    result = {
        "factual_alignment": factual_alignment_score,
        "coverage": coverage_score,
        "factual_consistency": consistency_score,
        "reference_claims_count": coverage_results["reference_claims_count"],
        "claims_in_summary_count": coverage_results["claims_in_summary_count"],
        "summary_claims_count": consistency_results["summary_claims_count"],
        "supported_claims_count": consistency_results["supported_claims_count"],
        "unsupported_claims_count": consistency_results["unsupported_claims_count"],
    }

    if verbose:
        if "claims_analysis" in coverage_results:
            result["coverage_claims_analysis"] = coverage_results["claims_analysis"]
        if "claims_analysis" in consistency_results:
            result["consistency_claims_analysis"] = consistency_results["claims_analysis"]

    return result
