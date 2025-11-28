from typing import Dict, Union, List, Optional, Tuple, Any

# Import base calculator classes
from .metrics.base import BaseCalculator, SummaryMetricCalculator, RAGMetricCalculator

# Import summary metrics
from .metrics.summary.rouge import calculate_rouge
from .metrics.summary.bleu import calculate_bleu
from .metrics.summary.bert_score import calculate_bert_score, ModelType
from .metrics.summary.coverage import calculate_coverage
from .metrics.summary.factual_consistency import calculate_factual_consistency
from .metrics.summary.factual_alignment import calculate_factual_alignment
from .metrics.summary.topic_preservation import calculate_topic_preservation
from .metrics.summary.redundancy import calculate_redundancy
from .metrics.summary.conciseness import calculate_conciseness_score
from .metrics.summary.bart_score import calculate_bart_score
from .metrics.summary.coherence import calculate_coherence
# Old metric names (deprecated) - kept for backwards compatibility
from .metrics.summary.faithfulness import calculate_faithfulness
from .metrics.summary.hallucination import calculate_hallucination

# Import RAG metrics
from .metrics.rag.answer_relevance import calculate_answer_relevance
from .metrics.rag.context_relevance import calculate_context_relevance
from .metrics.rag.answer_attribution import calculate_answer_attribution
from .metrics.rag.faithfulness import calculate_rag_faithfulness
from .metrics.rag.completeness import calculate_completeness

from .llm.config import LLMConfig
from .utils import detect_and_mask_pii, remove_stopwords, initialize_nltk
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)


# Define available metrics
AVAILABLE_SUMMARY_METRICS = [
    "rouge",
    "bleu",
    "bert_score",
    "bart_score",
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

# Define available metrics for RAG evaluation
AVAILABLE_RAG_METRICS = [
    "answer_relevance",
    "context_relevance",
    "faithfulness",
    "coherence",
    "completeness",
    "answer_attribution",
]

# All RAG metrics require LLM
LLM_REQUIRED_RAG_METRICS = AVAILABLE_RAG_METRICS


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    remove_stopwords: bool = False,
    llm_config: Optional[LLMConfig] = None,
    bert_model: Optional[ModelType] = "microsoft/deberta-base-mnli",
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
        bert_model: Model to use for BERTScore calculation. Options are:
            - "microsoft/deberta-base-mnli" (~86M parameters)
            - "microsoft/deberta-xlarge-mnli" (~750M parameters) (default)
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
    
    # Initialize NLTK only if BLEU metric is requested
    if "bleu" in metrics:
        initialize_nltk()
        
    for metric in metric_iterator:
        if metric == "rouge":
            results.update(calculate_rouge(full_text, summary))

        elif metric == "bleu":
            results["bleu"] = calculate_bleu(full_text, summary)

        elif metric == "bert_score":
            results.update(
                calculate_bert_score(full_text, summary, model_type=bert_model)
            )

        elif metric == "coverage":
            custom_instruction = custom_prompt_instructions.get("coverage") if custom_prompt_instructions else None
            results.update(calculate_coverage(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "factual_consistency":
            custom_instruction = custom_prompt_instructions.get("factual_consistency") if custom_prompt_instructions else None
            results.update(calculate_factual_consistency(full_text, summary, llm_config, custom_instruction, verbose=verbose))

        elif metric == "factual_alignment":
            custom_instruction = custom_prompt_instructions.get("factual_alignment") if custom_prompt_instructions else None
            results.update(calculate_factual_alignment(full_text, summary, llm_config, custom_instruction, verbose=verbose))

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

        elif metric == "bart_score":
            results.update(calculate_bart_score(full_text, summary))

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


def evaluate_rag(
    question: str,
    answer: str,
    context: Union[str, List[str]],
    llm_config: LLMConfig,
    metrics: Optional[List[str]] = None,
    show_progress: bool = True,
    mask_pii: bool = False,
    mask_pii_char: str = "*",
    mask_pii_preserve_partial: bool = False,
    mask_pii_entity_types: Optional[List[str]] = None,
    return_pii_info: bool = False,
    verbose: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Evaluate a RAG (Retrieval-Augmented Generation) system's output using specified metrics.

    Args:
        question: The input question
        answer: The generated answer to evaluate
        context: Retrieved context(s) used to generate the answer. Can be a single string or list of strings.
        llm_config: Configuration for LLM-based metrics
        metrics: List of metrics to calculate. Defaults to all available metrics.
        show_progress: Whether to show progress bar (default: True)
        mask_pii: Whether to mask personally identifiable information (PII) before evaluation (default: False)
        mask_pii_char: Character to use for masking PII (default: "*")
        mask_pii_preserve_partial: Whether to preserve part of the PII (e.g., for phone numbers: 123-***-***) (default: False)
        mask_pii_entity_types: List of PII entity types to detect and mask. If None, all supported types are used.
        return_pii_info: Whether to return information about detected PII (default: False)
        verbose: If True, include detailed analysis for metrics showing individual claims,
            topics, and their verification status. Useful for debugging and understanding metric results.

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
        metrics = AVAILABLE_RAG_METRICS

    # Validate metrics
    valid_metrics = set(AVAILABLE_RAG_METRICS)
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")

    # Initialize results dictionary
    results = {}
    
    # Handle PII masking if enabled
    pii_info = {}
    if mask_pii:
        logger.info("Masking PII in question, answer, and context...")
        try:
            # Mask question
            masked_question, question_pii = detect_and_mask_pii(
                question,
                entity_types=mask_pii_entity_types,
                mask_char=mask_pii_char,
                preserve_partial=mask_pii_preserve_partial
            )
            
            # Mask answer
            masked_answer, answer_pii = detect_and_mask_pii(
                answer,
                entity_types=mask_pii_entity_types,
                mask_char=mask_pii_char,
                preserve_partial=mask_pii_preserve_partial
            )
            
            # Mask context (could be string or list of strings)
            context_pii = {}
            if isinstance(context, str):
                masked_context, context_pii = detect_and_mask_pii(
                    context,
                    entity_types=mask_pii_entity_types,
                    mask_char=mask_pii_char,
                    preserve_partial=mask_pii_preserve_partial
                )
            else:  # List of strings
                masked_context = []
                context_pii = {"contexts": []}
                for i, ctx in enumerate(context):
                    masked_ctx, ctx_pii = detect_and_mask_pii(
                        ctx,
                        entity_types=mask_pii_entity_types,
                        mask_char=mask_pii_char,
                        preserve_partial=mask_pii_preserve_partial
                    )
                    masked_context.append(masked_ctx)
                    context_pii["contexts"].append({"index": i, "pii": ctx_pii})
            
            # Store PII information if requested
            if return_pii_info:
                pii_info = {
                    "question_pii": question_pii,
                    "answer_pii": answer_pii,
                    "context_pii": context_pii,
                    "question_masked": masked_question != question,
                    "answer_masked": masked_answer != answer,
                    "context_masked": masked_context != context
                }
            
            # Update with masked versions
            question = masked_question
            answer = masked_answer
            context = masked_context
            
            logger.info("PII masking complete.")
            
        except Exception as e:
            logger.error(f"Error during PII masking: {e}. Continuing with original text.")
            # Continue with original text in case of errors

    # Calculate requested metrics
    metric_iterator = tqdm(
        metrics, disable=not show_progress, desc="Calculating RAG metrics"
    )
    for metric in metric_iterator:
        if metric == "answer_relevance":
            results.update(calculate_answer_relevance(question, answer, llm_config, verbose=verbose))
        elif metric == "context_relevance":
            results.update(calculate_context_relevance(question, context, llm_config, verbose=verbose))
        elif metric == "answer_attribution":
            results.update(calculate_answer_attribution(answer, context, llm_config, verbose=verbose))
        elif metric == "faithfulness":
            results.update(calculate_rag_faithfulness(answer, context, llm_config, verbose=verbose))
        elif metric == "completeness":
            results.update(calculate_completeness(question, answer, llm_config, verbose=verbose))
        # Note: RAG coherence not yet implemented but could be added here

    # Return results with or without PII info
    if return_pii_info and mask_pii:
        return results, pii_info
    else:
        return results
