from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """
    Configuration class for LLM services.

    This class holds all necessary configuration parameters for connecting to
    and using various LLM providers (Bedrock, OpenAI, etc.).

    Attributes:
        provider (str): The LLM provider name ('bedrock', 'openai').
        model_id (str): The specific model identifier to use.
        region (str, optional): AWS region for Bedrock models.
        api_key (str, optional): API key for authentication.
        api_secret (str, optional): API secret for authentication.
        aws_session_token (str, optional): AWS session token for temporary credentials.
        proxy_url (str, optional): General proxy URL for both HTTP and HTTPS.
        http_proxy (str, optional): HTTP-specific proxy URL.
        https_proxy (str, optional): HTTPS-specific proxy URL.
        additional_params (Dict[str, Any], optional): Additional provider-specific parameters.
    """

    provider: str  # 'bedrock', 'openai'
    model_id: str
    region: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    aws_session_token: Optional[str] = None
    proxy_url: Optional[str] = None
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        """
        Validate the configuration parameters.

        Ensures all required fields are set for the specified provider
        and that values meet provider-specific requirements.

        Raises:
            ValueError: If configuration is invalid for the specified provider.
        """
        if self.provider not in ["bedrock", "openai"]:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if self.provider == "bedrock" and not self.region:
            raise ValueError("AWS region is required for Bedrock")

        if self.provider == "openai" and not self.api_key:
            raise ValueError("API key is required for OpenAI")

        # Model ID validation
        if self.provider == "openai" and not any(
            model in self.model_id for model in ["gpt-4", "gpt-3.5"]
        ):
            raise ValueError(
                "Invalid OpenAI model ID. Must be GPT-4 or GPT-3.5 variant"
            )