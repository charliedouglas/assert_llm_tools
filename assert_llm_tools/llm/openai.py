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
    """
    Implementation of BaseLLM for OpenAI API.

    This class handles communication with OpenAI API to run inference
    using models like GPT-4 and GPT-3.5.

    Attributes:
        client: OpenAI client instance.
        config (LLMConfig): Configuration for the OpenAI LLM.
    """

    def _initialize(self) -> None:
        _check_dependencies()
        self.client = OpenAI(api_key=self.config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI models.

        Formats the request for OpenAI chat completions API, sends the request,
        and extracts the response content.

        Args:
            prompt (str): The input prompt to send to the model.
            **kwargs: Additional parameters for text generation:
                - temperature (float): Controls randomness (0-1).
                - max_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated text response from the model.
        """
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
