"""
test_llm_config_validate.py — Unit tests for LLMConfig.validate() (END-88)

Validates that:
- Unsupported providers are rejected
- Required fields per provider are enforced
- Model ID validation has been removed (no substring matching on 'gpt-4'/'gpt-3.5')
- Azure OpenAI deployment names work correctly with the openai provider
- Empty model_id is rejected
"""
import pytest
from assert_llm_tools.llm.config import LLMConfig


class TestLLMConfigValidateProvider:

    def test_unsupported_provider_raises(self):
        config = LLMConfig(provider="anthropic", model_id="claude-3")
        with pytest.raises(ValueError, match="Unsupported provider"):
            config.validate()

    def test_bedrock_provider_accepted(self):
        config = LLMConfig(provider="bedrock", model_id="anthropic.claude-3", region="us-east-1")
        config.validate()  # must not raise

    def test_openai_provider_accepted(self):
        config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-test")
        config.validate()  # must not raise


class TestLLMConfigValidateRequiredFields:

    def test_bedrock_without_region_raises(self):
        config = LLMConfig(provider="bedrock", model_id="anthropic.claude-3")
        with pytest.raises(ValueError, match="region"):
            config.validate()

    def test_bedrock_with_region_passes(self):
        config = LLMConfig(provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="eu-west-1")
        config.validate()  # must not raise

    def test_openai_without_api_key_raises(self):
        config = LLMConfig(provider="openai", model_id="gpt-4o")
        with pytest.raises(ValueError, match="API key"):
            config.validate()

    def test_openai_with_api_key_passes(self):
        config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-abc123")
        config.validate()  # must not raise

    def test_empty_model_id_raises(self):
        config = LLMConfig(provider="openai", model_id="", api_key="sk-test")
        with pytest.raises(ValueError, match="model_id"):
            config.validate()


class TestLLMConfigAzureOpenAI:
    """
    Azure OpenAI uses deployment names (not model IDs like 'gpt-4' or 'gpt-3.5').
    The old validate() broke these. After END-88, any non-empty model_id is accepted.
    """

    def test_azure_deployment_name_accepted(self):
        """Azure deployment names like 'my-gpt4-deployment' must not raise (END-88)."""
        config = LLMConfig(
            provider="openai",
            model_id="my-gpt4-deployment",  # Azure deployment name — not a raw model ID
            api_key="sk-abc",
        )
        config.validate()  # must not raise

    def test_azure_custom_deployment_name_accepted(self):
        """Arbitrary Azure deployment name accepted."""
        config = LLMConfig(
            provider="openai",
            model_id="barclays-compliance-gpt4-turbo",
            api_key="sk-abc",
        )
        config.validate()  # must not raise

    def test_future_openai_model_accepted(self):
        """Future OpenAI model names (e.g. 'gpt-5') must not break validation."""
        config = LLMConfig(
            provider="openai",
            model_id="gpt-5",
            api_key="sk-abc",
        )
        config.validate()  # must not raise

    def test_o1_model_accepted(self):
        """OpenAI o1-series models were also broken by the old validation."""
        config = LLMConfig(
            provider="openai",
            model_id="o1-preview",
            api_key="sk-abc",
        )
        config.validate()  # must not raise
