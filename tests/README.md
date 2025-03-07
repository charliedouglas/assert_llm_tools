# ASSERT LLM Tools Test Suite

This directory contains tests for the ASSERT LLM Tools package.

## Test Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests that test multiple components together
- `conftest.py`: Shared pytest fixtures and configuration

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test files:

```bash
pytest tests/unit/test_config.py
```

To run tests with verbose output:

```bash
pytest -v
```

To run tests with coverage report:

```bash
pytest --cov=assert_llm_tools
```

## Test Categories

### Unit Tests

These test individual components in isolation:

- `test_config.py`: Tests for LLM configuration
- `test_utils.py`: Tests for utility functions
- `test_rouge.py`, `test_bleu.py`: Tests for individual metrics
- `test_base_llm.py`: Tests for the LLM base classes

### Integration Tests

These test how components work together:

- `test_summary_core.py`: Tests for the summary evaluation pipeline
- `test_rag_core.py`: Tests for the RAG evaluation pipeline

## Writing New Tests

- Follow the existing patterns for unit and integration tests
- Use pytest fixtures from `conftest.py` where applicable
- Mock external API calls in LLM-based tests
- Keep test functions focused and specific

## Conventions

- Test functions should be named with a `test_` prefix
- Use descriptive test function names that explain what's being tested
- Include docstrings in test functions explaining the test purpose
- Group related assertions in the same test function