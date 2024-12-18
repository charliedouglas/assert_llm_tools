# ASSERT LLM TOOLS

Automated Summary Scoring & Evaluation of Retained Text

This repository contains tools for evaluating the quality of summaries generated by LLMs.

## Demo

View a live demo of the library [here](https://www.getassert.io/demo)


## Metrics

### Summary Evaluation Metrics

#### Currently Supported Non-LLM Metrics

- **ROUGE Score**: Measures overlap of n-grams between the reference text and generated summary
- **BLEU Score**: Evaluates translation quality by comparing n-gram matches, with custom weights emphasizing unigrams and bigrams
- **BERT Score**: Leverages contextual embeddings to better capture semantic similarity
- **BART Score**: Uses BART's sequence-to-sequence model to evaluate semantic similarity and generation quality

#### Currently Supported LLM Metrics

- **Faithfulness**: Measures factual consistency between summary and source text (requires an LLM provider)
- **Topic Preservation**: Will verify that the most important topics from the source are retained in the summary (requires an LLM provider)
- **Redundancy Detection**: Will identify and flag repeated information within summaries (requires an LLM provider)
- **Conciseness Assessment**: Will evaluate if the summary effectively condenses information without unnecessary verbosity (requires an LLM provider)

### RAG Evaluation Metrics

#### Currently Supported Metrics

- **Context Relevance**: Evaluates how well the retrieved context matches the query
- **Answer Accuracy**: Measures the factual correctness of the generated answer based on the provided context
- **Context Utilization**: Assesses how effectively the model uses the provided context in generating the answer


### Planned Features


- **Coherence Evaluation**: Will assess the logical flow and readability of the generated summary
- **Style Consistency**: Will evaluate if the summary maintains a consistent writing style and tone
- **Information Density**: Will measure the ratio of meaningful content to length in summaries



## Features

- **Remove Common Stopwords**: Allows for adding custom stopwords to the evaluation process
  - This is useful for removing common words that are often included in summaries but do not contribute to the overall meaning
  - evaluate_summary(full_text, summary, remove_stopwords=True)
- **Custom Stopwords**: Allows for adding custom stopwords to the evaluation process
  - Usage: from assert_llm_tools.utils import add_custom_stopwords
  - Example: add_custom_stopwords(["your", "custom", "stopwords", "here"])
  - remove_stopwords=True must be enabled 
- **Select Metrics**: Allows for selecting which metrics to calculate
  - Usage: evaluate_summary(full_text, summary, metrics=["rouge", "bleu"])
  - Defaults to all metrics if not included
  - Available metrics: ["rouge", "bleu", "bert_score", "bart_score", "faithfulness", "topic_preservation", "redundancy", "conciseness"]
- **LLM Provider**: Allows for specifying the LLM provider and model to use for the faithfulness metric
  - Usage: evaluate_summary(full_text, summary, llm_config=LLMConfig(provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1", api_key="your-api-key", api_secret="your-api-secret"))
  - Available providers: ["bedrock", "openai"]
- **Show Progress**: Allows for showing a progress bar during metric calculation
  - Usage: evaluate_summary(full_text, summary, show_progress=True)
  - Defaults to showing progress bar if not included.

## Understanding Scores

### Summary Evaluation Scores

All metrics are normalized to return scores between 0 and 1, where higher scores indicate better performance:

- ROUGE Score: Higher means better overlap with reference
- BLEU Score: Higher means better translation quality
- BERT Score: Higher means better semantic similarity
  - Note that running BERT score for the first time will require a download of the model weights, which may take a while.
  - Use the `bert_model` parameter to specify the model to use for BERTScore calculation.
  - Default model is "microsoft/deberta-base-mnli". (~500mb download on first use.)
  - Other options is "microsoft/deberta-xlarge-mnli". (~3gb download on first use.)
- BART Score: Higher means better semantic similarity and generation quality
  - Returns log-likelihood scores normalized to be interpretable, therefore results are likely to be negative. Closer to 0 is better.
  - Calculates bidirectional scores (reference→summary and summary→reference)
  - Uses the BART-large-CNN model by default (~1.6GB download on first use)
- Faithfulness: Higher means better factual consistency
- Topic Preservation: Higher means better retention of key topics
- Redundancy: Higher means less redundant content (1.0 = no redundancy)
- Conciseness: Higher means less verbose content (1.0 = optimal conciseness)

### RAG Evaluation Scores

All RAG metrics return scores between 0 and 1:

- Context Relevance: Higher means better match between query and retrieved context
- Answer Accuracy: Higher means better factual correctness based on context
- Context Utilization: Higher means better use of provided context
- Citation Accuracy: Higher means better support for claims from context

## Installation

Basic installation:
```bash
pip install assert_llm_tools
```

Optional Dependencies:

- For Amazon Bedrock support:
  ```bash
  pip install "assert_llm_tools[bedrock]"
  ```

- For OpenAI support:
  ```bash
  pip install "assert_llm_tools[openai]"
  ```

- To install all optional dependencies:
  ```bash
  pip install "assert_llm_tools[all]"
  ```

## Usage

```python
from assert_llm_tools.core import evaluate_summary, evaluate_rag
from assert_llm_tools.utils import add_custom_stopwords
from assert_llm_tools.llm.config import LLMConfig

# Add custom stopwords
add_custom_stopwords(["this", "artificial", "intelligence"])

metrics = ["rouge", "bleu", "bert_score"]

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

# Using OpenAI
config = LLMConfig(
    provider="openai",
    model_id="gpt-4",
    api_key="your-api-key"
)

# Summary Evaluation Example
metrics = evaluate_summary(full_text, summary, 
                         remove_stopwords=False, 
                         metrics=["rouge", "bleu", "bert_score"], 
                         llm_config=config)

# RAG Evaluation Example
rag_metrics = evaluate_rag(query="What is the capital of France?",
                          context="Paris is the capital and largest city of France.",
                          answer="The capital of France is Paris.",
                          metrics=["context_relevance", "answer_accuracy"],
                          llm_config=config)

# Print results
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
