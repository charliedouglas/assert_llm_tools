from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .config import LLMConfig


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.

    This class defines the interface that all specific LLM implementations must follow.
    It handles validation of the configuration and provides abstract methods that
    concrete subclasses must implement.

    Attributes:
        config (LLMConfig): Configuration parameters for the LLM.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the BaseLLM with configuration.

        Args:
            config (LLMConfig): Configuration parameters for the LLM.
        """
        self.config = config
        self.config.validate()
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the LLM client.

        This method should be implemented by subclasses to set up
        any necessary clients, connections, or resources needed for
        the specific LLM service.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt using the LLM.

        Args:
            prompt (str): The input prompt to send to the LLM.
            **kwargs: Additional parameters to customize generation
                      (temperature, max_tokens, etc.)

        Returns:
            str: The generated text response from the LLM.
        """
        pass
