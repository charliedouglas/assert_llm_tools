from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .config import LLMConfig


class BaseLLM(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.config.validate()
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the LLM client"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
