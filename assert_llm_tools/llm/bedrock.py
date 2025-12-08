import boto3
import json
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from .base import BaseLLM
from .config import LLMConfig
from botocore.config import Config


def _check_dependencies():
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "Bedrock support requires additional dependencies. "
            "Install them with: pip install assert_llm_tools[bedrock]"
        )


class BedrockLLM(BaseLLM):
    """
    Implementation of BaseLLM for AWS Bedrock service.

    This class handles communication with AWS Bedrock API to run inference
    using various foundation models including:
    - Amazon Nova (amazon.nova-*, us.amazon.nova-*)
    - Anthropic Claude (anthropic.claude-*)
    - Meta Llama (meta.llama*, us.meta.llama*)
    - Mistral AI (mistral.mistral-*)
    - Cohere Command (cohere.command-*)
    - AI21 Labs Jamba (ai21.jamba-*)

    Attributes:
        client: Boto3 client for Bedrock Runtime service.
        config (LLMConfig): Configuration for the Bedrock LLM.
    """

    # Model family identifiers for request/response format detection
    MODEL_FAMILIES = {
        "nova": ["amazon.nova", "us.amazon.nova"],
        "anthropic": ["anthropic.claude"],
        "llama": ["meta.llama", "us.meta.llama"],
        "mistral": ["mistral.mistral"],
        "cohere": ["cohere.command"],
        "ai21": ["ai21.jamba", "ai21.j2"],
    }

    def _initialize(self) -> None:
        """
        Initialize the AWS Bedrock client.

        Sets up the Boto3 client with appropriate authentication and configuration
        including region, API credentials, and proxy settings if specified.

        Raises:
            ImportError: If boto3 dependency is not installed.
            ValueError: If configuration is invalid.
        """
        _check_dependencies()
        
        # Setup session kwargs (credentials)
        session_kwargs = {}
        if self.config.api_key and self.config.api_secret:
            session_kwargs.update({
                "aws_access_key_id": self.config.api_key,
                "aws_secret_access_key": self.config.api_secret,
            })
            if self.config.aws_session_token:
                session_kwargs["aws_session_token"] = self.config.aws_session_token

        # Create the session
        session = boto3.Session(region_name=self.config.region, **session_kwargs)
        
        # Setup proxy configuration for the client
        client_config = None
        proxies = self._get_proxy_config()

        if proxies:
            client_config = Config(proxies=proxies)  # Use proxies directly
            # Create a copy of proxies with masked passwords for printing
            masked_proxies = self._mask_proxy_passwords(proxies.copy())
            print(f"Using proxy configuration: {masked_proxies}")
            self._test_proxy_connectivity(proxies)

        # Create the client with proxy config if available
        client_kwargs = {}
        if client_config:
            client_kwargs["config"] = client_config  # Use the already created client_config

        self.client = session.client("bedrock-runtime", **client_kwargs)

    def _detect_model_family(self) -> str:
        """
        Detect the model family based on model_id.

        Returns:
            str: The model family name (nova, anthropic, llama, mistral, cohere, ai21)
                 or 'unknown' if not recognized.
        """
        model_id_lower = self.config.model_id.lower()
        for family, prefixes in self.MODEL_FAMILIES.items():
            for prefix in prefixes:
                if prefix in (model_id_lower.):
                    return family
        return "unknown"

    def _get_proxy_config(self) -> Dict[str, str]:
        """
        Get proxy configuration from config object or environment variables.
        
        Returns:
            Dict containing proxy URLs for http and https
        """
        proxies = {}
        
        # First check for proxy settings in the config object
        if hasattr(self.config, "proxy_url") and self.config.proxy_url:
            proxies["http"] = self.config.proxy_url
            proxies["https"] = self.config.proxy_url
            
        if hasattr(self.config, "http_proxy") and self.config.http_proxy:
            proxies["http"] = self.config.http_proxy
            
        if hasattr(self.config, "https_proxy") and self.config.https_proxy:
            proxies["https"] = self.config.https_proxy
            
        # If no proxies in config, check environment variables
        if not proxies:
            if "HTTP_PROXY" in os.environ:
                proxies["http"] = os.environ["HTTP_PROXY"]
            if "HTTPS_PROXY" in os.environ:
                proxies["https"] = os.environ["HTTPS_PROXY"]
                
        return proxies
        
    def _mask_proxy_passwords(self, proxies: Dict[str, str]) -> Dict[str, str]:
        """
        Mask passwords in proxy URLs for secure logging.
        
        Args:
            proxies: Dict with 'http' and 'https' keys containing proxy URLs
            
        Returns:
            Dict with the same keys but passwords replaced with '*****'
        """
        for protocol in proxies:
            proxy_url = proxies[protocol]
            if proxy_url and '@' in proxy_url:
                # URL format: protocol://user:pass@host:port
                parts = proxy_url.split('@', 1)
                auth_part = parts[0]
                host_part = parts[1]
                
                if ':' in auth_part:
                    # Extract username and mask password
                    protocol_and_user, _ = auth_part.rsplit(':', 1)
                    # Replace with masked password
                    proxies[protocol] = f"{protocol_and_user}:*****@{host_part}"
                
        return proxies

    
    def _test_proxy_connectivity(self, proxies: Dict[str, str]) -> None:
        """
        Test connectivity through the proxy before making API calls.
        
        Args:
            proxies: Dict with 'http' and 'https' keys
            
        Raises:
            ConnectionError: If proxy connectivity test fails
        """
        import socket
        import urllib.error
        import urllib.request
        
        # Only test if we have HTTPS proxy (most common for API calls)
        if "https" in proxies and proxies["https"]:
            proxy_url = proxies["https"]
            parsed = urlparse(proxy_url)
            
            try:
                # Try to connect to the proxy host/port
                with socket.create_connection(
                    (parsed.hostname, parsed.port or 80), 
                    timeout=5
                ):
                    # Use masked host for secure logging
                    print(f"Successfully connected to proxy at {parsed.hostname}:{parsed.port or 80}")
            except (socket.timeout, socket.error) as e:
                print(f"Warning: Could not connect to proxy: {e}")
                # Don't raise here as the proxy might still work with boto3
                # Just warn the user

    def _build_request_params(self, prompt: str, model_family: str, **kwargs) -> Dict[str, Any]:
        """
        Build request parameters based on model family.

        Args:
            prompt: The input prompt
            model_family: The detected model family
            **kwargs: Additional generation parameters

        Returns:
            Dict containing the request parameters for the specific model family
        """
        max_tokens = kwargs.get("max_tokens", 500)
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 20)

        if model_family == "nova":
            return {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": "You should respond to all messages in english"}],
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
            }

        elif model_family == "anthropic":
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

        elif model_family == "llama":
            # Meta Llama models use a simpler prompt-based format
            return {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        elif model_family == "mistral":
            # Mistral uses a messages-based format
            return {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }

        elif model_family == "cohere":
            # Cohere Command models
            return {
                "message": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "p": top_p,
                "k": top_k,
            }

        elif model_family == "ai21":
            # AI21 Jamba models use messages format
            return {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        else:
            # Default/unknown models - try generic messages format
            # This provides a fallback for new models not yet explicitly supported
            return {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

    def _parse_response(self, response_body: Dict[str, Any], model_family: str) -> str:
        """
        Parse the response based on model family.

        Args:
            response_body: The parsed JSON response from Bedrock
            model_family: The detected model family

        Returns:
            str: The extracted text response
        """
        if model_family == "nova":
            return response_body["output"]["message"]["content"][0]["text"]

        elif model_family == "anthropic":
            return response_body["content"][0]["text"]

        elif model_family == "llama":
            return response_body["generation"]

        elif model_family == "mistral":
            return response_body["outputs"][0]["text"]

        elif model_family == "cohere":
            return response_body["text"]

        elif model_family == "ai21":
            return response_body["choices"][0]["message"]["content"]

        else:
            # Try common response formats for unknown models
            if "content" in response_body:
                content = response_body["content"]
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "text" in content[0]:
                        return content[0]["text"]
                    return str(content[0])
                return str(content)
            elif "generation" in response_body:
                return response_body["generation"]
            elif "text" in response_body:
                return response_body["text"]
            elif "outputs" in response_body:
                return response_body["outputs"][0]["text"]
            elif "choices" in response_body:
                choice = response_body["choices"][0]
                if "message" in choice:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
            # Last resort - return the whole response as string
            raise ValueError(
                f"Unable to parse response from unknown model family. "
                f"Response structure: {list(response_body.keys())}"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using AWS Bedrock models.

        Formats the request appropriately based on the model family,
        sends the request to Bedrock, and parses the response.

        Supported model families:
        - Amazon Nova (amazon.nova-*, us.amazon.nova-*)
        - Anthropic Claude (anthropic.claude-*)
        - Meta Llama (meta.llama*, us.meta.llama*)
        - Mistral AI (mistral.mistral-*)
        - Cohere Command (cohere.command-*)
        - AI21 Labs Jamba (ai21.jamba-*)

        Args:
            prompt (str): The input prompt to send to the model.
            **kwargs: Additional parameters for text generation:
                - max_tokens (int): Maximum number of tokens to generate (default: 500).
                - temperature (float): Controls randomness, 0-1 (default: 0).
                - top_p (float): Controls diversity via nucleus sampling (default: 0.9).
                - top_k (int): Controls diversity via top-k sampling (default: 20).

        Returns:
            str: The generated text response from the model.

        Raises:
            ConnectionError: If there are proxy connectivity issues.
            ValueError: If response parsing fails for unknown model types.
        """
        model_family = self._detect_model_family()

        # Build request parameters for the specific model family
        default_params = self._build_request_params(prompt, model_family, **kwargs)

        # Merge with additional parameters from config
        if self.config.additional_params:
            default_params.update(self.config.additional_params)

        try:
            response = self.client.invoke_model(
                modelId=self.config.model_id, body=json.dumps(default_params)
            )

            response_body = json.loads(response["body"].read())

            # Parse response based on model family
            return self._parse_response(response_body, model_family)

        except Exception as e:
            # Provide more helpful error message for proxy issues
            error_message = str(e)
            if "proxy" in error_message.lower() or "connect" in error_message.lower():
                raise ConnectionError(
                    f"Error connecting through proxy: {error_message}. "
                    "Please check your proxy configuration and connectivity."
                )
            raise