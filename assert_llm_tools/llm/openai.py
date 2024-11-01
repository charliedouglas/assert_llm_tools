from openai import OpenAI
from .base import BaseLLM
from .config import LLMConfig


def _check_dependencies():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI support requires additional dependencies. "
            "Install them with: pip install assert_llm_tools[openai]"
        )


class OpenAILLM(BaseLLM):
    def _initialize(self) -> None:
        _check_dependencies()
        self.client = OpenAI(api_key=self.config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        default_params = {
            "model": self.config.model_id,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 500),
            "messages": [{"role": "user", "content": prompt}],
        }

        # Merge with additional parameters from config
        if self.config.additional_params:
            default_params.update(self.config.additional_params)

        response = self.client.chat.completions.create(**default_params)
        return response.choices[0].message.content
