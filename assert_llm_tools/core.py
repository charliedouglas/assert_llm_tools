from typing import Dict, Union, List, Optional
from .metrics.rouge import calculate_rouge
from .metrics.bleu import calculate_bleu
from .metrics.bert_score import calculate_bert_score
from .metrics.faithfulness import calculate_faithfulness
from .metrics.topic_preservation import calculate_topic_preservation
from .metrics.redundancy import calculate_redundancy
from .metrics.conciseness import calculate_conciseness_score
from .llm.config import LLMConfig
from typing import Dict, Union, List, Optional
from tqdm import tqdm

# Define available metrics
AVAILABLE_METRICS = [
    "rouge",
    "bleu",
    "bert_score",
    "faithfulness",
    "topic_preservation",
    "redundancy",
    "conciseness",
]

# Define which metrics require LLM
LLM_REQUIRED_METRICS = [
    "faithfulness",
    "topic_preservation",
    "redundancy",
    "conciseness",
]


def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    remove_stopwords: bool = False,
    llm_config: Optional[LLMConfig] = None,
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
        show_progress: Whether to show progress bar (default: True)

    Returns:
        Dictionary containing scores for each metric
    """
    # Default to all metrics if none specified
    if metrics is None:
        metrics = AVAILABLE_METRICS

    # Validate metrics
    valid_metrics = set(AVAILABLE_METRICS)
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")

    # Validate LLM config for metrics that require it
    llm_metrics = set(metrics) & set(LLM_REQUIRED_METRICS)
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
            results.update(calculate_bert_score(full_text, summary))

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

    return results
