import pytest
from unittest.mock import MagicMock, patch
import json
import os
from assert_llm_tools.llm.config import LLMConfig


# Fixture for sample texts
@pytest.fixture
def sample_texts():
    return {
        "source_text": "Climate change is a pressing global issue. Rising temperatures are causing sea levels to rise. Extreme weather events are becoming more frequent. Carbon emissions from human activities are the main contributor. Governments worldwide are working to reduce emissions and transition to renewable energy sources.",
        "summary": "Climate change is causing rising sea levels and extreme weather. Human carbon emissions are the primary cause, and governments are working on solutions.",
        "question": "What are the main effects of climate change?",
        "answer": "The main effects of climate change include rising sea levels and more frequent extreme weather events.",
        "context": "Climate change is leading to rising global temperatures. This causes polar ice to melt, resulting in rising sea levels. It also causes more frequent and intense extreme weather events like hurricanes, floods, and droughts.",
    }


# Fixture for mocking OpenAI LLM
@pytest.fixture
def mock_openai_llm():
    with patch("assert_llm_tools.llm.openai.OpenAI") as mock_client:
        # Configure the mock to return a specific response
        instance = mock_client.return_value
        instance.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="This is a mock LLM response"))
        ]
        yield mock_client


# Fixture for mocking Bedrock LLM
@pytest.fixture
def mock_bedrock_llm():
    with patch("boto3.Session") as mock_session:
        # Configure mock for Bedrock responses
        mock_client = MagicMock()
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "This is a mock Bedrock response"}]}
        )
        mock_client.invoke_model.return_value = mock_response
        mock_session.return_value.client.return_value = mock_client
        yield mock_session


# Fixture for LLM configuration
@pytest.fixture
def llm_config():
    return LLMConfig(provider="openai", model_id="gpt-3.5-turbo", api_key="test_key")


# Fixture for Bedrock LLM configuration
@pytest.fixture
def bedrock_config():
    return LLMConfig(
        provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
    )


# Fixture to temporarily set environment variables for tests
@pytest.fixture
def env_setup():
    original_env = os.environ.copy()
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["AWS_REGION"] = "us-east-1"
    yield
    os.environ.clear()
    os.environ.update(original_env)
