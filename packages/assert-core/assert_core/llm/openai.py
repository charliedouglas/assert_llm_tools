from openai import OpenAI
import os
from typing import Dict, Optional
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
        """
        Initialize the OpenAI client with API key and proxy configuration.
        
        Raises:
            ImportError: If OpenAI dependency is not installed.
            ValueError: If configuration is invalid.
        """
        _check_dependencies()
        
        # Basic client configuration
        client_args = {"api_key": self.config.api_key}
        
        # Get proxy configuration
        proxies = self._get_proxy_config()
        
        if proxies:
            # OpenAI client uses a http_client parameter for proxy configuration
            from httpx import HTTPTransport
            client_args["http_client"] = HTTPTransport(proxy=proxies)
            print(f"Using proxy configuration for OpenAI client: {proxies}")
            
            # Test proxy connectivity
            self._test_proxy_connectivity(proxies)
        
        # Initialize the client
        self.client = OpenAI(**client_args)

    def _get_proxy_config(self) -> Optional[Dict[str, str]]:
        """
        Get proxy configuration from config object or environment variables.
        
        Returns:
            Dict containing proxy URLs or None if no proxy is configured
        """
        proxies = {}
        
        # First check for proxy settings in the config object
        if hasattr(self.config, "proxy_url") and self.config.proxy_url:
            proxies["http://"] = self.config.proxy_url
            proxies["https://"] = self.config.proxy_url
            
        if hasattr(self.config, "http_proxy") and self.config.http_proxy:
            proxies["http://"] = self.config.http_proxy
            
        if hasattr(self.config, "https_proxy") and self.config.https_proxy:
            proxies["https://"] = self.config.https_proxy
            
        # If no proxies in config, check environment variables
        if not proxies:
            if "HTTP_PROXY" in os.environ:
                proxies["http://"] = os.environ["HTTP_PROXY"]
            if "HTTPS_PROXY" in os.environ:
                proxies["https://"] = os.environ["HTTPS_PROXY"]
                
        return proxies if proxies else None
    
    def _test_proxy_connectivity(self, proxies: Dict[str, str]) -> None:
        """
        Test connectivity through the proxy before making API calls.
        
        Args:
            proxies: Dict with proxy URLs
            
        Raises:
            Warning: If proxy connectivity test fails (just a warning, not an exception)
        """
        import socket
        from urllib.parse import urlparse
        
        # Only test if we have HTTPS proxy (most common for API calls)
        https_proxy = proxies.get("https://") or proxies.get("https://")
        
        if https_proxy:
            parsed = urlparse(https_proxy)
            
            try:
                # Try to connect to the proxy host/port
                with socket.create_connection(
                    (parsed.hostname, parsed.port or 443), 
                    timeout=5
                ):
                    print(f"Successfully connected to proxy at {parsed.hostname}:{parsed.port or 443}")
            except (socket.timeout, socket.error) as e:
                print(f"Warning: Could not connect to proxy: {e}")
                # Don't raise here as the proxy might still work
                # Just warn the user

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

        try:
            response = self.client.chat.completions.create(**default_params)
            return response.choices[0].message.content
        except Exception as e:
            # Provide more helpful error message for proxy issues
            error_message = str(e)
            if "proxy" in error_message.lower() or "connect" in error_message.lower():
                raise ConnectionError(
                    f"Error connecting through proxy: {error_message}. "
                    "Please check your proxy configuration and connectivity."
                )
            raise