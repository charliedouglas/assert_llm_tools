import pytest
from unittest.mock import MagicMock, patch
from assert_llm_tools.llm.base import BaseLLM
from assert_llm_tools.llm.config import LLMConfig


class TestLLM(BaseLLM):
    """Test implementation of BaseLLM for testing"""
    
    def _initialize(self):
        """Mock initialization"""
        self.client = MagicMock()
    
    def generate(self, prompt, **kwargs):
        """Mock generation that returns the prompt reversed"""
        return prompt[::-1]


def test_base_llm_initialization():
    """Test that BaseLLM initialization validates config"""
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key"
    )
    
    # With valid config, should initialize without error
    llm = TestLLM(config)
    assert llm.config == config
    
    # Check that invalid config raises ValueError
    with patch.object(LLMConfig, 'validate', side_effect=ValueError("Test error")):
        with pytest.raises(ValueError, match="Test error"):
            TestLLM(config)


def test_llm_generate():
    """Test that concrete implementation works"""
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key"
    )
    
    llm = TestLLM(config)
    
    # Test generate method (our implementation reverses the prompt)
    prompt = "Hello, world!"
    response = llm.generate(prompt)
    assert response == prompt[::-1]


def test_base_llm_abstract():
    """Test that BaseLLM cannot be instantiated directly"""
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key"
    )
    
    # Should not be able to instantiate abstract class
    with pytest.raises(TypeError):
        BaseLLM(config)