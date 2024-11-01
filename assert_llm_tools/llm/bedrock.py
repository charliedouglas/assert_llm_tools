import boto3
import json
from typing import Dict, Any
from .base import BaseLLM
from .config import LLMConfig


def _check_dependencies():
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "Bedrock support requires additional dependencies. "
            "Install them with: pip install assert_llm_tools[bedrock]"
        )


class BedrockLLM(BaseLLM):
    def _initialize(self) -> None:
        _check_dependencies()
        session_kwargs = {}
        if self.config.api_key and self.config.api_secret:
            session_kwargs.update(
                {
                    "aws_access_key_id": self.config.api_key,
                    "aws_secret_access_key": self.config.api_secret,
                }
            )

        session = boto3.Session(region_name=self.config.region, **session_kwargs)
        self.client = session.client("bedrock-runtime")

    def generate(self, prompt: str, **kwargs) -> str:
        default_params = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0),
            "messages": [{"role": "user", "content": prompt}],
        }

        # Merge with additional parameters from config
        if self.config.additional_params:
            default_params.update(self.config.additional_params)

        response = self.client.invoke_model(
            modelId=self.config.model_id, body=json.dumps(default_params)
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
