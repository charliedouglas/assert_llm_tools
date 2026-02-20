# assert-eval

LLM-based summary quality evaluation.

Scores a summary against source text for coverage, factual accuracy, alignment, and topic preservation. No PyTorch, no BERT, no heavy dependencies.

## Installation

```bash
pip install assert-eval
```

## Quick Start

```python
from assert_eval import evaluate_summary, LLMConfig

config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

results = evaluate_summary(
    full_text="Original long text goes here...",
    summary="Summary to evaluate goes here...",
    metrics=["coverage", "factual_consistency", "factual_alignment", "topic_preservation"],
    llm_config=config,
)

print(results)
# {'coverage': 0.85, 'factual_consistency': 0.92, 'factual_alignment': 0.88, 'topic_preservation': 0.90}
```

## Available Metrics

| Metric | Description |
|--------|-------------|
| `coverage` | What % of source document claims appear in the summary (recall/completeness) |
| `factual_consistency` | What % of summary claims are supported by the source (precision/accuracy) |
| `factual_alignment` | F1 score combining coverage and factual_consistency |
| `topic_preservation` | How well the main topics from the source are preserved in the summary |

## Custom Evaluation Instructions

Tailor the LLM's evaluation criteria for your domain:

```python
results = evaluate_summary(
    full_text=text,
    summary=summary,
    metrics=["coverage", "factual_consistency"],
    llm_config=config,
    custom_prompt_instructions={
        "coverage": "Apply strict standards. Only mark a claim as covered if it is clearly and explicitly represented.",
        "factual_consistency": "Flag any claim that adds detail not present in the original text.",
    },
)
```

## Verbose Output

Pass `verbose=True` to include per-claim LLM reasoning in the results:

```python
results = evaluate_summary(
    full_text=text,
    summary=summary,
    metrics=["coverage", "factual_consistency"],
    llm_config=config,
    verbose=True,
)
```

## PII Masking

Pass `mask_pii=True` to detect and mask personally identifiable information before any text is sent to the LLM:

```python
results = evaluate_summary(
    full_text=text,
    summary=summary,
    metrics=["coverage"],
    llm_config=config,
    mask_pii=True,
)
```

`mask_pii=False` is the default. For production use with real client data, set `mask_pii=True`.

## LLM Configuration

```python
from assert_eval import LLMConfig

# AWS Bedrock (uses ~/.aws credentials by default)
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

# AWS Bedrock with explicit credentials
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
    api_key="your-aws-access-key-id",
    api_secret="your-aws-secret-access-key",
    aws_session_token="your-session-token",  # optional
)

# OpenAI
config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="your-openai-api-key",
)
```

### Supported Bedrock Model Families

| Model Family | Example Model IDs |
|---|---|
| Amazon Nova | `us.amazon.nova-pro-v1:0`, `amazon.nova-lite-v1:0` |
| Anthropic Claude | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Meta Llama | `meta.llama3-70b-instruct-v1:0` |
| Mistral AI | `mistral.mistral-large-2402-v1:0` |
| Cohere Command | `cohere.command-r-plus-v1:0` |
| AI21 Labs | `ai21.jamba-1-5-large-v1:0` |

## Proxy Configuration

```python
# Single proxy
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    proxy_url="http://proxy.example.com:8080",
)

# Protocol-specific proxies
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    http_proxy="http://proxy.example.com:8080",
    https_proxy="http://proxy.example.com:8443",
)

# Authenticated proxy
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    proxy_url="http://username:password@proxy.example.com:8080",
)
```

Standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables are also respected.

## Dependencies

- [assert-core](https://pypi.org/p/assert-core) — shared LLM provider layer (AWS Bedrock, OpenAI)

## Migrating from assert_llm_tools

`assert-eval` replaces the summary evaluation functionality of `assert_llm_tools`, which is now deprecated. The API is largely the same — swap the import:

```python
# Before
from assert_llm_tools import evaluate_summary, LLMConfig

# After
from assert_eval import evaluate_summary, LLMConfig
```

## License

MIT
