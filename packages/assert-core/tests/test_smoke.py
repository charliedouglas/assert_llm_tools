"""Smoke tests — assert-core installs and exports correctly (END-89)."""

from assert_core import LLMConfig, BaseLLM, BedrockLLM, OpenAILLM
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
