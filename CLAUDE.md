# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASSERT LLM Tools is a Python library for evaluating summaries and RAG (Retrieval-Augmented Generation) outputs using various metrics. The library supports both traditional metrics (ROUGE, BLEU, BERTScore) and LLM-based metrics (faithfulness, hallucination detection, topic preservation).

## Development Commands

### Setup
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies (includes Bedrock and OpenAI)
pip install -e ".[all]"
```

### Testing
```bash
# Run all tests
pytest

# Run a specific test file
pytest test_pii_masking.py

# Run with verbose output
pytest -v
```

### Building and Publishing
```bash
# Build the package
python -m build

# Check distribution before upload
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

### Code Quality
```bash
# Format code with black
black assert_llm_tools/

# Sort imports
isort assert_llm_tools/

# Lint with flake8
flake8 assert_llm_tools/
```

## Architecture

### Core Components

**Entry Points** ([core.py](assert_llm_tools/core.py)):
- `evaluate_summary()`: Main function for summary evaluation
- `evaluate_rag()`: Main function for RAG evaluation
- Handles PII masking, metric orchestration, and progress tracking

**LLM Abstraction Layer** ([llm/](assert_llm_tools/llm/)):
- `LLMConfig`: Unified configuration for all LLM providers
- `BedrockLLM`: AWS Bedrock integration with proxy support
- `OpenAILLM`: OpenAI API integration with proxy support
- All providers implement the same base interface for consistent metric evaluation

**Metric System** ([metrics/](assert_llm_tools/metrics/)):
- **Base Classes** ([base.py](assert_llm_tools/metrics/base.py)):
  - `BaseCalculator`: Common LLM initialization and response parsing
  - `SummaryMetricCalculator`: Summary-specific utilities (topic/claim extraction)
  - `RAGMetricCalculator`: RAG-specific utilities (context normalization)

- **Summary Metrics** ([metrics/summary/](assert_llm_tools/metrics/summary/)):
  - Traditional: ROUGE, BLEU, BERTScore, BARTScore
  - LLM-based: faithfulness, hallucination, topic_preservation, redundancy, conciseness, coherence

- **RAG Metrics** ([metrics/rag/](assert_llm_tools/metrics/rag/)):
  - All LLM-based: answer_relevance, context_relevance, faithfulness, answer_attribution, completeness

**Utilities** ([utils.py](assert_llm_tools/utils.py)):
- PII detection and masking using Presidio
- NLTK initialization (lazy loaded only when BLEU is used)
- Stopword management

### Key Design Patterns

1. **Lazy Initialization**: NLTK data is only downloaded when BLEU metric is requested to avoid unnecessary dependencies
2. **Proxy Support**: All LLM providers support proxy configuration via `proxy_url`, `http_proxy`, or `https_proxy`
3. **Provider Abstraction**: Metrics never directly call provider APIs; they use the LLM abstraction layer
4. **PII Safety**: Optional PII masking with configurable entity types and masking strategies (full or partial)

### Metric Categories

- **LLM-required summary metrics**: faithfulness, topic_preservation, redundancy, conciseness, coherence, hallucination
- **Traditional summary metrics**: rouge, bleu, bert_score, bart_score
- **All RAG metrics require LLM**: answer_relevance, context_relevance, faithfulness, answer_attribution, completeness

## Common Patterns

### Adding a New Summary Metric

1. Create metric file in [assert_llm_tools/metrics/summary/](assert_llm_tools/metrics/summary/)
2. Implement `calculate_<metric_name>()` function that returns `Dict[str, float]`
3. For LLM-based metrics, use `SummaryMetricCalculator` base class
4. Import in [core.py](assert_llm_tools/core.py) and add to `AVAILABLE_SUMMARY_METRICS`
5. If requires LLM, add to `LLM_REQUIRED_SUMMARY_METRICS`

### Adding a New RAG Metric

1. Create metric file in [assert_llm_tools/metrics/rag/](assert_llm_tools/metrics/rag/)
2. Implement `calculate_<metric_name>()` function
3. Use `RAGMetricCalculator` base class
4. Import in [core.py](assert_llm_tools/core.py) and add to `AVAILABLE_RAG_METRICS`

### Adding a New LLM Provider

1. Create provider file in [assert_llm_tools/llm/](assert_llm_tools/llm/)
2. Extend `BaseLLM` class
3. Implement `generate()` method
4. Add initialization logic in `BaseCalculator.__init__()`
5. Update [llm/config.py](assert_llm_tools/llm/config.py) validation

## Testing Notes

- Test files (test*.py) in root demonstrate usage patterns
- PII masking tests are in [test_pii_masking.py](test_pii_masking.py)
- RAG evaluation tests in [test_rag.py](test_rag.py)
- Documentation tests in [test_docs.py](test_docs.py)
