# assert-eval

LLM-based summary quality evaluation.

Scores a summary against source text across four metrics: coverage, factual consistency, factual alignment, and topic preservation. No PyTorch, no BERT, no heavy dependencies.

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
| `coverage` | What % of source document claims appear in the summary (recall) |
| `factual_consistency` | What % of summary claims are supported by the source (precision) |
| `factual_alignment` | F1 score combining coverage and factual_consistency |
| `topic_preservation` | How well the main topics from the source are preserved |

## Custom Evaluation Instructions

Tailor LLM evaluation criteria for your domain:

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

## LLM Configuration

```python
from assert_eval import LLMConfig

# AWS Bedrock
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

# OpenAI
config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="your-openai-api-key",
)
```

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
