"""
Unit tests for LLMConfig.validate() — END-88.

Validates that:
  - validate() does not inspect model_id contents for OpenAI
  - Azure OpenAI model IDs (gpt-4o, gpt-4-turbo, deployment names) pass
  - Future model IDs (o1, o3, etc.) pass
  - Bedrock validation unchanged (region required)
  - OpenAI validation unchanged (api_key required)
  - Empty model_id still raises
  - Unknown provider still raises
"""

import pytest
from assert_llm_tools.llm.config import LLMConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai(model_id: str, api_key: str = "sk-test") -> LLMConfig:
    return LLMConfig(provider="openai", model_id=model_id, api_key=api_key)


def _bedrock(model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", region: str = "us-east-1") -> LLMConfig:
    return LLMConfig(provider="bedrock", model_id=model_id, region=region)


# ---------------------------------------------------------------------------
# OpenAI — model IDs that were previously rejected must now pass
# ---------------------------------------------------------------------------

class TestOpenAIModelIdValidation:

    def test_gpt4o_passes(self):
        """Azure OpenAI gpt-4o must not raise."""
        _openai("gpt-4o").validate()

    def test_gpt4_turbo_passes(self):
        """Standard gpt-4-turbo must pass."""
        _openai("gpt-4-turbo").validate()

    def test_gpt4_passes(self):
        """Classic gpt-4 must still pass."""
        _openai("gpt-4").validate()

    def test_gpt35_turbo_passes(self):
        """gpt-3.5-turbo must still pass."""
        _openai("gpt-3.5-turbo").validate()

    def test_o1_passes(self):
        """o1 model must pass (no gpt-4/gpt-3.5 substring)."""
        _openai("o1").validate()

    def test_o3_passes(self):
        """o3 model must pass."""
        _openai("o3").validate()

    def test_azure_deployment_name_passes(self):
        """Azure deployment names (arbitrary strings) must pass."""
        _openai("my-azure-deployment").validate()

    def test_arbitrary_model_id_passes(self):
        """Any non-empty string must pass — provider validates model at call time."""
        _openai("some-future-model-v99").validate()


# ---------------------------------------------------------------------------
# OpenAI — required field checks unchanged
# ---------------------------------------------------------------------------

class TestOpenAIRequiredFields:

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            LLMConfig(provider="openai", model_id="gpt-4o").validate()

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            LLMConfig(provider="openai", model_id="", api_key="sk-test").validate()


# ---------------------------------------------------------------------------
# Bedrock — unchanged behaviour
# ---------------------------------------------------------------------------

class TestBedrockValidation:

    def test_valid_bedrock_passes(self):
        _bedrock().validate()

    def test_bedrock_missing_region_raises(self):
        with pytest.raises(ValueError, match="region"):
            LLMConfig(provider="bedrock", model_id="anthropic.claude-3-sonnet-20240229-v1:0").validate()

    def test_bedrock_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            LLMConfig(provider="bedrock", model_id="", region="us-east-1").validate()

    def test_bedrock_any_model_id_passes(self):
        """Bedrock model IDs are not string-matched either."""
        _bedrock(model_id="amazon.titan-text-express-v1").validate()
        _bedrock(model_id="meta.llama3-8b-instruct-v1:0").validate()


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

class TestUnknownProvider:

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMConfig(provider="azure", model_id="gpt-4o", api_key="sk-test").validate()

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMConfig(provider="", model_id="gpt-4o").validate()
