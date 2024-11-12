from typing import Dict, Union, List, Optional
from .metrics.summary.rouge import calculate_rouge
from .metrics.summary.bleu import calculate_bleu
from .metrics.summary.bert_score import calculate_bert_score, ModelType
from .metrics.summary.faithfulness import calculate_faithfulness
from .metrics.summary.topic_preservation import calculate_topic_preservation
from .metrics.summary.redundancy import calculate_redundancy
from .metrics.summary.conciseness import calculate_conciseness_score
from .metrics.summary.bart_score import calculate_bart_score
from .llm.config import LLMConfig
from typing import Dict, Union, List, Optional
from tqdm import tqdm
from .metrics.rag.answer_relevance import calculate_answer_relevance
from .metrics.rag.context_relevance import calculate_context_relevance
from .metrics.rag.answer_attribution import calculate_answer_attribution
from .metrics.rag.faithfulness import calculate_faithfulness
from .metrics.rag.completeness import calculate_completeness

# Define available metrics
AVAILABLE_SUMMARY_METRICS = [
    "rouge",
    "bleu",
    "bert_score",
    "bart_score",
    "faithfulness",
    "topic_preservation",
    "redundancy",
    "conciseness",
]

# Define which metrics require LLM
LLM_REQUIRED_SUMMARY_METRICS = [
    "faithfulness",
    "topic_preservation",
    "redundancy",
    "conciseness",
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
) -> Dict[str, float]:
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
        if metric == "rouge":
            results.update(calculate_rouge(full_text, summary))

        elif metric == "bleu":
            results["bleu"] = calculate_bleu(full_text, summary)

        elif metric == "bert_score":
            results.update(
                calculate_bert_score(full_text, summary, model_type=bert_model)
            )

        elif metric == "faithfulness":
            results.update(calculate_faithfulness(full_text, summary, llm_config))

        elif metric == "topic_preservation":
            results.update(calculate_topic_preservation(full_text, summary, llm_config))

        elif metric == "redundancy":
            results.update(calculate_redundancy(summary, llm_config))

        elif metric == "conciseness":
            results["conciseness"] = calculate_conciseness_score(
                full_text, summary, llm_config
            )

        elif metric == "bart_score":
            results.update(calculate_bart_score(full_text, summary))

    return results


def evaluate_rag(
    question: str,
    answer: str,
    context: Union[str, List[str]],
    llm_config: LLMConfig,
    metrics: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a RAG (Retrieval-Augmented Generation) system's output using specified metrics.

    Args:
        question: The input question
        answer: The generated answer to evaluate
        context: Retrieved context(s) used to generate the answer. Can be a single string or list of strings.
        llm_config: Configuration for LLM-based metrics
        metrics: List of metrics to calculate. Defaults to all available metrics.
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Dictionary containing scores for each metric
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

    # Calculate requested metrics
    metric_iterator = tqdm(
        metrics, disable=not show_progress, desc="Calculating RAG metrics"
    )
    for metric in metric_iterator:
        if metric == "answer_relevance":
            results.update(calculate_answer_relevance(question, answer, llm_config))
        elif metric == "context_relevance":
            results.update(calculate_context_relevance(question, context, llm_config))
        elif metric == "answer_attribution":
            results.update(calculate_answer_attribution(answer, context, llm_config))
        elif metric == "faithfulness":
            results.update(calculate_faithfulness(answer, context, llm_config))
        elif metric == "completeness":
            results.update(calculate_completeness(question, answer, llm_config))
        # ... other metrics to be implemented ...

    return results
