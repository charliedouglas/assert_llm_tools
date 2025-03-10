# ASSERT LLM Tools

**A**utomated **S**ummary **S**coring & **E**valuation of **R**etained **T**ext

ASSERT LLM Tools is a Python library for evaluating summaries and RAG (Retrieval-Augmented Generation) outputs using various metrics, both traditional (ROUGE, BLEU, BERTScore) and LLM-based.

## Features

- **Summary Evaluation**: Measure summary quality with metrics like faithfulness, topic preservation, coherence, and more
- **RAG Evaluation**: Evaluate RAG systems with metrics for answer relevance, context relevance, and faithfulness
- **Multiple LLM Providers**: Support for OpenAI and AWS Bedrock APIs
- **PII Detection & Masking**: Automatically detect and mask personally identifiable information before evaluation
- **Proxy Support**: Comprehensive proxy configuration for corporate environments
- **Extensible Architecture**: Easy to add new metrics or LLM providers

## Installation

```bash
pip install assert_llm_tools
```

For additional features, install optional dependencies:

```bash
# For AWS Bedrock support
pip install assert_llm_tools[bedrock]

# For OpenAI support
pip install assert_llm_tools[openai]

# For all optional dependencies
pip install assert_llm_tools[all]
```

## Quick Start

### Summary Evaluation

Evaluate a summary against original text:

```python
from assert_llm_tools import evaluate_summary, LLMConfig

# Configure LLM for evaluation
config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    api_key="your-openai-api-key"
)

# Evaluate the summary
results = evaluate_summary(
    full_text="Original long text goes here...",
    summary="Summary to evaluate goes here...",
    metrics=["rouge", "faithfulness", "hallucination", "coherence"],
    llm_config=config
)

print(results)
```

### RAG Evaluation

Evaluate a RAG system output:

```python
from assert_llm_tools import evaluate_rag, LLMConfig

# Configure LLM for evaluation
config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-v2",
    region="us-east-1"
)

# Evaluate the RAG output
results = evaluate_rag(
    question="What are the main effects of climate change?",
    answer="Climate change leads to rising sea levels, increased temperatures...",
    context="Climate change refers to long-term shifts in temperatures...",
    llm_config=config,
    metrics=["answer_relevance", "faithfulness"]
)

print(results)
```

## Proxy Configuration

ASSERT LLM Tools supports various proxy configurations for environments that require proxies to access external APIs.

### Using a General Proxy

```python
from assert_llm_tools import LLMConfig

# Configure with a single proxy for both HTTP and HTTPS
config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    api_key="your-openai-api-key",
    proxy_url="http://proxy.example.com:8080"
)
```

### Using Protocol-Specific Proxies

```python
from assert_llm_tools import LLMConfig

# Configure with separate proxies for HTTP and HTTPS
config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-v2",
    region="us-east-1",
    http_proxy="http://http-proxy.example.com:8080",
    https_proxy="http://https-proxy.example.com:8443"
)
```

### Using Environment Variables

The library also respects standard environment variables for proxy configuration:

```bash
# Set environment variables
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8443"
```

Then create configuration without explicit proxy settings:

```python
# No proxy settings in code - will use environment variables
config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    api_key="your-openai-api-key"
)
```

### Proxy Authentication

For proxies that require authentication, include the username and password in the URL:

```python
config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-v2",
    region="us-east-1",
    proxy_url="http://username:password@proxy.example.com:8080"
)
```

## Available Metrics

### Summary Metrics

- `rouge`: ROUGE-1, ROUGE-2, and ROUGE-L scores
- `bleu`: BLEU score
- `bert_score`: BERTScore precision, recall, and F1
- `bart_score`: BARTScore
- `faithfulness`: Measures factual consistency with the source text
- `hallucination`: Detects claims in the summary not supported by the source text (returns hallucination_score)
- `topic_preservation`: How well the summary preserves main topics
- `redundancy`: Measures repetitive content
- `conciseness`: Evaluates information density and brevity
- `coherence`: Measures logical flow and readability

### RAG Metrics

- `answer_relevance`: How well the answer addresses the question
- `context_relevance`: How relevant the retrieved context is to the question
- `faithfulness`: Factual consistency between answer and context
- `answer_attribution`: How much of the answer is derived from the context
- `completeness`: Whether the answer addresses all aspects of the question

## Advanced Configuration

### PII Detection and Masking

For privacy-sensitive applications, you can automatically detect and mask personally identifiable information (PII) before evaluation:

```python
# Basic PII masking
results = evaluate_summary(
    full_text="John Smith (john.smith@example.com) lives in New York.",
    summary="John's contact is john.smith@example.com.",
    metrics=["rouge", "faithfulness"],
    llm_config=config,
    mask_pii=True  # Enable PII masking
)

# Advanced PII masking with more options
results, pii_info = evaluate_summary(
    full_text=text_with_pii,
    summary=summary_with_pii,
    metrics=["rouge", "faithfulness"],
    llm_config=config,
    mask_pii=True,
    mask_pii_char="#",  # Custom masking character
    mask_pii_preserve_partial=True,  # Preserve parts of emails, phone numbers, etc.
    mask_pii_entity_types=["PERSON", "EMAIL_ADDRESS", "LOCATION"],  # Only mask specific entities
    return_pii_info=True  # Return information about detected PII
)

# Access PII detection results
print(f"PII in original text: {pii_info['full_text_pii']}")
print(f"PII in summary: {pii_info['summary_pii']}")
```

The same PII masking options are available for RAG evaluation:

```python
results = evaluate_rag(
    question="Who is John Smith and what is his email?",
    answer="John Smith's email is john.smith@example.com.",
    context="John Smith (john.smith@example.com) is our company's CEO.",
    llm_config=config,
    metrics=["answer_relevance", "faithfulness"],
    mask_pii=True
)
```

### Custom Model Selection

For BERTScore calculation, you can specify the model to use:

```python
results = evaluate_summary(
    full_text=source,
    summary=summary,
    metrics=["bert_score"],
    bert_model="microsoft/deberta-xlarge-mnli"  # More accurate but slower
)
```

### AWS Credentials for Bedrock

```python
config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-v2",
    region="us-east-1",
    api_key="YOUR_AWS_ACCESS_KEY",
    api_secret="YOUR_AWS_SECRET_KEY",
    aws_session_token="YOUR_SESSION_TOKEN"  # Optional
)
```

### Additional Model Parameters

```python
config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    api_key="your-openai-api-key",
    additional_params={
        "response_format": {"type": "json_object"},
        "seed": 42
    }
)
```

## License

MIT