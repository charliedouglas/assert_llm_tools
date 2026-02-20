"""Smoke tests — assert-eval installs and exports correctly (END-91)."""

from unittest.mock import MagicMock, patch

import pytest

from assert_eval import AVAILABLE_SUMMARY_METRICS, LLMConfig, evaluate_summary
from assert_eval.metrics import (
    calculate_coverage,
    calculate_factual_alignment,
    calculate_factual_consistency,
    calculate_topic_preservation,
)

_REFERENCE = "The company reported $5.2 billion in revenue for Q3 2024, a 15% increase year-over-year."
_SUMMARY = "The company had $5.2 billion in Q3 2024 revenue, up 15% year-over-year."
_LLM_CFG = LLMConfig(provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1")


# ---------------------------------------------------------------------------
# Import / API surface
# ---------------------------------------------------------------------------


def test_public_api_importable():
    assert evaluate_summary is not None
    assert AVAILABLE_SUMMARY_METRICS is not None
    assert LLMConfig is not None


def test_available_metrics_list():
    assert set(AVAILABLE_SUMMARY_METRICS) == {
        "coverage",
        "factual_consistency",
        "factual_alignment",
        "topic_preservation",
    }


def test_individual_calculators_importable():
    assert calculate_coverage is not None
    assert calculate_factual_consistency is not None
    assert calculate_factual_alignment is not None
    assert calculate_topic_preservation is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_llm_config_raises():
    with pytest.raises(ValueError, match="LLM configuration required"):
        evaluate_summary(_REFERENCE, _SUMMARY, metrics=["coverage"], llm_config=None)


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="Invalid metrics"):
        evaluate_summary(_REFERENCE, _SUMMARY, metrics=["nonexistent"], llm_config=_LLM_CFG)


# ---------------------------------------------------------------------------
# Metric execution (mocked LLM)
# ---------------------------------------------------------------------------


def _make_mock_generate(responses):
    """Return a side_effect function that cycles through a list of responses."""
    call_count = {"n": 0}

    def _generate(prompt, **kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    return _generate


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_coverage(mock_generate, mock_init):
    # First call: extract claims from reference → 2 claims
    # Second call: check which claims are in summary → both supported
    mock_generate.side_effect = _make_mock_generate(
        [
            "The company reported $5.2 billion in revenue\nRevenue increased 15% year-over-year",
            "supported\nsupported",
        ]
    )
    result = calculate_coverage(_REFERENCE, _SUMMARY, _LLM_CFG)
    assert "coverage" in result
    assert 0.0 <= result["coverage"] <= 1.0
    assert "reference_claims_count" in result
    assert "claims_in_summary_count" in result


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_factual_consistency(mock_generate, mock_init):
    mock_generate.side_effect = _make_mock_generate(
        [
            "The company had $5.2 billion in Q3 2024 revenue\nRevenue was up 15% year-over-year",
            "supported\nsupported",
        ]
    )
    result = calculate_factual_consistency(_REFERENCE, _SUMMARY, _LLM_CFG)
    assert "factual_consistency" in result
    assert 0.0 <= result["factual_consistency"] <= 1.0
    assert "summary_claims_count" in result
    assert "supported_claims_count" in result
    assert "unsupported_claims_count" in result


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_topic_preservation(mock_generate, mock_init):
    mock_generate.side_effect = _make_mock_generate(
        [
            "revenue\nearnings growth",  # topics extracted
            "yes\nyes",  # both preserved
        ]
    )
    result = calculate_topic_preservation(_REFERENCE, _SUMMARY, _LLM_CFG)
    assert "topic_preservation" in result
    assert 0.0 <= result["topic_preservation"] <= 1.0
    assert "reference_topics_count" in result
    assert "preserved_topics_count" in result
    assert "missing_topics_count" in result


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_factual_alignment_uses_precomputed(mock_generate, mock_init):
    """factual_alignment reuses precomputed coverage/consistency results."""
    mock_generate.side_effect = _make_mock_generate(
        [
            # coverage — extract reference claims
            "claim one\nclaim two",
            # coverage — check claims in summary
            "supported\nsupported",
            # factual_consistency — extract summary claims
            "claim a\nclaim b",
            # factual_consistency — verify claims
            "supported\nsupported",
        ]
    )
    result = evaluate_summary(
        _REFERENCE,
        _SUMMARY,
        metrics=["coverage", "factual_consistency", "factual_alignment"],
        llm_config=_LLM_CFG,
    )
    assert "coverage" in result
    assert "factual_consistency" in result
    assert "factual_alignment" in result
    # factual_alignment should not trigger extra LLM calls (only 4 calls total)
    assert mock_generate.call_count == 4


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_evaluate_summary_single_metric(mock_generate, mock_init):
    mock_generate.side_effect = _make_mock_generate(
        ["claim one\nclaim two", "supported\nunsupported"]
    )
    result = evaluate_summary(_REFERENCE, _SUMMARY, metrics=["coverage"], llm_config=_LLM_CFG)
    assert list(result.keys()) == ["coverage", "reference_claims_count", "claims_in_summary_count"]


@patch("assert_core.llm.bedrock.BedrockLLM._initialize")
@patch("assert_core.llm.bedrock.BedrockLLM.generate")
def test_verbose_mode_includes_analysis(mock_generate, mock_init):
    mock_generate.side_effect = _make_mock_generate(
        ["claim one\nclaim two", "supported\nsupported"]
    )
    result = calculate_coverage(_REFERENCE, _SUMMARY, _LLM_CFG, verbose=True)
    assert "claims_analysis" in result
    assert isinstance(result["claims_analysis"], list)
