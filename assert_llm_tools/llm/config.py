from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    provider: str  # 'bedrock', 'openai'
    model_id: str
    region: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        """Validate the configuration"""
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
