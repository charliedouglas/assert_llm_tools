"""Smoke tests — assert-core installs and exports correctly (END-89)."""

from assert_core import LLMConfig, BaseLLM, BedrockLLM, OpenAILLM, detect_and_mask_pii
from assert_core.metrics import BaseCalculator


def test_llmconfig_importable():
    cfg = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-test")
    cfg.validate()  # must not raise


def test_llmconfig_azure_model_id_passes():
    """Azure deployment names must pass after END-88 fix."""
    cfg = LLMConfig(provider="openai", model_id="my-azure-deployment", api_key="sk-test")
    cfg.validate()


def test_llmconfig_bedrock_passes():
    cfg = LLMConfig(provider="bedrock", model_id="anthropic.claude-3-sonnet-20240229-v1:0", region="us-east-1")
    cfg.validate()


def test_bedrock_llm_importable():
    assert BedrockLLM is not None


def test_openai_llm_importable():
    assert OpenAILLM is not None


def test_base_llm_importable():
    assert BaseLLM is not None


def test_basecalculator_importable():
    assert BaseCalculator is not None


def test_detect_and_mask_pii_importable():
    assert detect_and_mask_pii is not None


def test_detect_and_mask_pii_masks_email():
    result = detect_and_mask_pii("Contact alice@example.com for details.")
    assert "[EMAIL]" in result
    assert "alice@example.com" not in result


def test_detect_and_mask_pii_masks_phone():
    result = detect_and_mask_pii("Call us at 555-867-5309.")
    assert "[PHONE]" in result
    assert "555-867-5309" not in result


def test_detect_and_mask_pii_masks_ssn():
    result = detect_and_mask_pii("SSN: 123-45-6789")
    assert "[SSN]" in result
    assert "123-45-6789" not in result


def test_detect_and_mask_pii_empty_string():
    assert detect_and_mask_pii("") == ""


def test_detect_and_mask_pii_no_pii():
    text = "The quick brown fox jumps over the lazy dog."
    assert detect_and_mask_pii(text) == text
