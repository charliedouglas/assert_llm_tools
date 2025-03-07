import pytest
from assert_llm_tools.llm.config import LLMConfig


def test_llm_config_creation():
    """Test that an LLMConfig can be created with required parameters"""
    # Bedrock config
    bedrock_config = LLMConfig(
        provider="bedrock",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
    )
    assert bedrock_config.provider == "bedrock"
    assert bedrock_config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert bedrock_config.region == "us-east-1"
    
    # OpenAI config
    openai_config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key",
    )
    assert openai_config.provider == "openai"
    assert openai_config.model_id == "gpt-4"
    assert openai_config.api_key == "test_key"


def test_bedrock_config_validation():
    """Test validation for Bedrock config"""
    # Valid config
    valid_config = LLMConfig(
        provider="bedrock",
        model_id="anthropic.claude-v2",
        region="us-east-1",
    )
    valid_config.validate()  # Should not raise
    
    # Missing region
    invalid_config = LLMConfig(
        provider="bedrock",
        model_id="anthropic.claude-v2",
    )
    with pytest.raises(ValueError, match="AWS region is required for Bedrock"):
        invalid_config.validate()


def test_openai_config_validation():
    """Test validation for OpenAI config"""
    # Valid config
    valid_config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key",
    )
    valid_config.validate()  # Should not raise
    
    # Missing API key
    invalid_config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
    )
    with pytest.raises(ValueError, match="API key is required for OpenAI"):
        invalid_config.validate()
    
    # Invalid model ID
    invalid_model_config = LLMConfig(
        provider="openai",
        model_id="not-supported-model",
        api_key="test_key",
    )
    with pytest.raises(ValueError, match="Invalid OpenAI model ID"):
        invalid_model_config.validate()


def test_unsupported_provider():
    """Test validation for unsupported provider"""
    config = LLMConfig(
        provider="unsupported",
        model_id="some-model",
    )
    with pytest.raises(ValueError, match="Unsupported provider"):
        config.validate()


def test_proxy_configuration():
    """Test proxy configuration setting"""
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key",
        http_proxy="http://proxy.example.com:8080",
        https_proxy="https://proxy.example.com:8443",
    )
    
    assert config.http_proxy == "http://proxy.example.com:8080"
    assert config.https_proxy == "https://proxy.example.com:8443"

    # Test with general proxy_url
    config2 = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key",
        proxy_url="http://proxy.example.com:8080",
    )
    
    assert config2.proxy_url == "http://proxy.example.com:8080"