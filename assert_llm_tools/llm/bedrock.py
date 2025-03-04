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
            if self.config.aws_session_token:
                session_kwargs["aws_session_token"] = self.config.aws_session_token

        # Add proxy configuration if provided
        proxies = {}
        # Using a single proxy_url for both protocols if specified
        if hasattr(self.config, "proxy_url") and self.config.proxy_url:
            proxies["http"] = self.config.proxy_url
            proxies["https"] = self.config.proxy_url

        # Using protocol-specific proxies if specified
        if hasattr(self.config, "http_proxy") and self.config.http_proxy:
            proxies["http"] = self.config.http_proxy
        if hasattr(self.config, "https_proxy") and self.config.https_proxy:
            proxies["https"] = self.config.https_proxy

        # Apply proxies if any are defined
        if proxies:
            session_kwargs["proxies"] = proxies

        session = boto3.Session(region_name=self.config.region, **session_kwargs)

        # Configure client with proxy if needed
        client_kwargs = {}
        if proxies:
            client_kwargs["config"] = boto3.config.Config(proxies=proxies)

        self.client = session.client("bedrock-runtime", **client_kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        # Determine if it's a Nova model
        is_nova = "nova" in self.config.model_id.lower()

        if is_nova:
            default_params = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": "You should respond to all messages in english"}],
                "inferenceConfig": {
                    "max_new_tokens": kwargs.get("max_tokens", 500),
                    "temperature": kwargs.get("temperature", 0),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 20),
                },
            }
        else:
            # Anthropic model format
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

        # Parse response based on model type
        if is_nova:
            return response_body["output"]["message"]["content"][0]["text"]
        else:
            return response_body["content"][0]["text"]
