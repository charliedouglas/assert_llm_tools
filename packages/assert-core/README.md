# assert-core

Shared LLM provider layer for [assert-review](../assert-review) and [assert-eval](../assert-eval).

Not intended for direct installation by end users — install `assert-review` or `assert-eval` instead.

## What's in here

- `LLMConfig` — provider configuration dataclass (supports OpenAI, Azure OpenAI, AWS Bedrock)
- `BaseLLM`, `BedrockLLM`, `OpenAILLM` — provider implementations
- `BaseCalculator` — base class for all metric calculators

## Install (internal use)

```bash
pip install assert-core
```

## Usage

```python
from assert_core import LLMConfig, OpenAILLM

config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-...")
llm = OpenAILLM(config)
```
