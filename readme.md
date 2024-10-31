# ASSERT LLM TOOLS

This repository contains tools for evaluating the quality of summaries generated by LLMs.

## Metrics

### Currently Supported

- **ROUGE Score**: Measures overlap of n-grams between the reference text and generated summary
- **BLEU Score**: Evaluates translation quality by comparing n-gram matches, with custom weights emphasizing unigrams and bigrams

### Planned Features

- **BERT Score**: Will leverage contextual embeddings to better capture semantic similarity
- **Truthfulness Assessment**: Will evaluate factual consistency between summary and source text

## Features

- **Remove Common Stopwords**: Allows for adding custom stopwords to the evaluation process
    - This is useful for removing common words that are often included in summaries but do not contribute to the overall meaning
    - evaluate_summary(full_text, summary, remove_stopwords=True)
- **Custom Stopwords**: Allows for adding custom stopwords to the evaluation process
    - Usage: from assert_llm_tools.utils import add_custom_stopwords
    - Example: add_custom_stopwords(["your", "custom", "stopwords", "here"])

## Installation

```bash
pip install assert_llm_tools
```

## Usage

```python
# test_assert.py
from assert_llm_tools.core import evaluate_summary
from assert_llm_tools.utils import add_custom_stopwords


# Add custom stopwords
add_custom_stopwords(["this", "artificial", "intelligence"])


# Example text from an article
full_text = """
Artificial intelligence is rapidly transforming the world economy. Companies 
are investing billions in AI research and development, leading to breakthroughs 
in automation, data analysis, and decision-making processes. While this 
technology offers immense benefits, it also raises concerns about job 
displacement and ethical considerations.
"""

# Example summary
summary = """
AI is transforming the economy through major investments, bringing advances in 
automation and analytics while raising job and ethical concerns.
"""

# Get evaluation metrics
metrics = evaluate_summary(full_text, summary, remove_stopwords=False)

# Print results
print("\nOriginal Text:")
print(full_text)
print("\nSummary:")
print(summary)
print("\nEvaluation Metrics:")
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")


```

## LICENSE

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

- [ROUGE](https://github.com/google-research/google-research/tree/master/rouge)
- [NLTK](https://www.nltk.org/)