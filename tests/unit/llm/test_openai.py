import pytest
from unittest.mock import patch, MagicMock
from assert_llm_tools.llm.openai import OpenAILLM
from assert_llm_tools.llm.config import LLMConfig


def test_openai_initialization(mock_openai_llm):
    """Test that OpenAILLM initializes correctly."""
    config = LLMConfig(provider="openai", model_id="gpt-4", api_key="test_key")

    llm = OpenAILLM(config)

    # Check that the OpenAI client was initialized
    mock_openai_llm.assert_called_once_with(api_key="test_key")


def test_openai_generate(mock_openai_llm):
    """Test that OpenAILLM.generate works correctly."""
    config = LLMConfig(provider="openai", model_id="gpt-3.5-turbo", api_key="test_key")

    # Configure the mock to return a specific response
    instance = mock_openai_llm.return_value
    mock_message = MagicMock()
    mock_message.content = "This is a test response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    instance.chat.completions.create.return_value.choices = [mock_choice]

    # Initialize the LLM and call generate
    llm = OpenAILLM(config)
    response = llm.generate("Test prompt")

    # Check that the response is as expected
    assert response == "This is a test response"

    # Check that the chat.completions.create was called with the expected arguments
    instance.chat.completions.create.assert_called_once()
    call_args = instance.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-3.5-turbo"
    assert call_args["temperature"] == 0
    assert call_args["max_tokens"] == 500
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]


def test_openai_generate_with_parameters(mock_openai_llm):
    """Test that OpenAILLM.generate handles additional parameters correctly."""
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test_key",
        additional_params={"presence_penalty": 0.5},
    )

    # Configure the mock
    instance = mock_openai_llm.return_value
    mock_message = MagicMock()
    mock_message.content = "This is a test response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    instance.chat.completions.create.return_value.choices = [mock_choice]

    # Initialize the LLM and call generate with custom parameters
    llm = OpenAILLM(config)
    response = llm.generate("Test prompt", temperature=0.7, max_tokens=1000)

    # Check the call arguments
    call_args = instance.chat.completions.create.call_args[1]
    assert call_args["temperature"] == 0.7
    assert call_args["max_tokens"] == 1000
    assert call_args["presence_penalty"] == 0.5  # from additional_params


def test_openai_dependencies():
    """Test that OpenAILLM checks for dependencies."""
    # Mock the import check to fail
    with patch("assert_llm_tools.llm.openai._check_dependencies") as mock_check:
        mock_check.side_effect = ImportError(
            "OpenAI support requires additional dependencies"
        )

        config = LLMConfig(provider="openai", model_id="gpt-4", api_key="test_key")

        # Should raise ImportError
        with pytest.raises(
            ImportError, match="OpenAI support requires additional dependencies"
        ):
            OpenAILLM(config)
