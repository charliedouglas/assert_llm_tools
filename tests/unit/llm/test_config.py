import pytest
from assert_llm_tools.llm.config import LLMConfig


def test_llm_config_initialization():
    """Test that LLMConfig initializes correctly with valid parameters."""
    # Test OpenAI configuration
    openai_config = LLMConfig(provider="openai", model_id="gpt-4", api_key="test_key")
    assert openai_config.provider == "openai"
    assert openai_config.model_id == "gpt-4"
    assert openai_config.api_key == "test_key"

    # Test Bedrock configuration
    bedrock_config = LLMConfig(
        provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
    )
    assert bedrock_config.provider == "bedrock"
    assert bedrock_config.model_id == "anthropic.claude-v2"
    assert bedrock_config.region == "us-east-1"


def test_llm_config_validation():
    """Test that LLMConfig validation works correctly."""
    # Test invalid provider
    with pytest.raises(ValueError, match="Unsupported provider"):
        config = LLMConfig(provider="invalid", model_id="model")
        config.validate()

    # Test missing region for Bedrock
    with pytest.raises(ValueError, match="AWS region is required"):
        config = LLMConfig(provider="bedrock", model_id="anthropic.claude-v2")
        config.validate()

    # Test missing API key for OpenAI
    with pytest.raises(ValueError, match="API key is required"):
        config = LLMConfig(provider="openai", model_id="gpt-4")
        config.validate()

    # Test invalid OpenAI model ID
    with pytest.raises(ValueError, match="Invalid OpenAI model ID"):
        config = LLMConfig(provider="openai", model_id="invalid-model", api_key="test")
        config.validate()

    # Valid configurations should not raise exceptions
    valid_openai = LLMConfig(provider="openai", model_id="gpt-4", api_key="test")
    valid_openai.validate()  # Should not raise

    valid_bedrock = LLMConfig(
        provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
    )
    valid_bedrock.validate()  # Should not raise
