# Summarization Metrics Assessment & Improvement Recommendations

## Executive Summary

This document provides a comprehensive assessment of the current summarization metrics in ASSERT LLM Tools and proposes concrete improvements. The analysis reveals several strengths in the hybrid approach (combining traditional and LLM-based metrics) but also identifies conceptual issues, missing metrics, and opportunities for enhancement.

---

## Current Metrics Overview

### Traditional Metrics
1. **ROUGE** (assert_llm_tools/metrics/summary/rouge.py) - N-gram overlap (recall-oriented)
2. **BLEU** - Precision-oriented n-gram matching
3. **BERTScore** - Contextual embedding similarity
4. **BARTScore** - Generation-based scoring using BART

### LLM-Based Metrics
1. **Faithfulness** - Claim coverage from source (recall-based)
2. **Hallucination** - Unsupported claims detection (precision-based)
3. **Topic Preservation** - Coverage of main topics
4. **Redundancy** - Repetitive information detection
5. **Conciseness** - Information density evaluation
6. **Coherence** - Logical flow and cohesion (hybrid: embeddings + LLM)

---

## Critical Issues Identified

### 🔴 Issue 1: Semantic Confusion Between Faithfulness and Hallucination

**Problem:**
- **Faithfulness** (faithfulness.py:61-96) measures **how many source claims appear in summary** (recall/completeness)
- **Hallucination** (hallucination.py:128-171) measures **how many summary claims are unsupported** (inverse precision)
- These are complementary but the naming is misleading

**Current Implementation:**
```python
# Faithfulness: reference_claims → check in summary → recall
faithfulness_score = claims_in_summary_count / len(reference_claims)

# Hallucination: summary_claims → check in reference → inverse precision
hallucination_free_score = 1.0 - (hallucinated_claims_count / len(summary_claims))
```

**Impact:**
- Users expect "faithfulness" to mean "factually accurate/supported"
- Current faithfulness is actually measuring **completeness/coverage**
- The two metrics measure opposite directions of the claim relationship

**Recommendation:**
Rename and clarify:
- **Faithfulness** → **Coverage** or **Completeness** (what % of source claims are in summary)
- **Hallucination** → **Faithfulness** or **Factual_Consistency** (what % of summary claims are supported)
- Add a combined F1-style metric: **Factual_Alignment** = harmonic mean of both

---

### 🟡 Issue 2: Topic Preservation Lacks Nuance

**Problem** (topic_preservation.py:24-58):
- Binary yes/no check per topic
- No importance weighting
- Doesn't measure depth or quality of coverage
- All topics treated equally

**Example Limitation:**
```
Source topics: ["Machine Learning", "Data Privacy", "Ethics", "Implementation"]
Summary: Briefly mentions ML and privacy, deeply covers ethics

Current score: 3/4 = 0.75
Reality: Ethics is most important but this isn't reflected
```

**Recommendation:**
Add weighted topic preservation:
1. Extract topics with importance scores (LLM-based)
2. Assess coverage depth (0-1 scale per topic)
3. Calculate weighted score: `Σ(topic_importance × coverage_depth) / Σ(topic_importance)`

---

### 🟡 Issue 3: Redundancy Detection is Fragile

**Problem** (redundancy.py:24-64):
- Relies on LLM to identify redundant text segments
- Calculates redundancy based on character length of "repeated" text
- Structured prompt parsing (`Original:` / `Repeated:` / `---`) is brittle
- Length-based scoring doesn't account for semantic redundancy

**Limitations:**
- Short paraphrases may have different lengths but same meaning
- LLM output format variance breaks parsing
- Doesn't detect subtle redundancy (e.g., "The company succeeded" + "The firm was successful")

**Recommendation:**
Implement semantic redundancy detection:
1. Split summary into sentences/segments
2. Compute pairwise semantic similarity using embeddings
3. Identify clusters of similar segments (threshold-based)
4. Calculate redundancy as: `1 - (unique_semantic_content / total_segments)`

---

### 🟡 Issue 4: Conciseness Scoring Has Arbitrary Thresholds

**Problem** (conciseness.py:72-74):
```python
# Optimal compression ratio hardcoded to 0.2-0.4
compression_score = 1.0 - abs(0.3 - compression_ratio) / 0.3
```

**Limitations:**
- Different domains require different compression ratios
  - News: 0.1-0.2 (very concise)
  - Research papers: 0.3-0.5 (detailed)
  - Legal documents: 0.4-0.6 (comprehensive)
- Sentence length penalty (20 words) is also arbitrary
- No adaptation to content type

**Recommendation:**
Make compression ratios configurable:
1. Add `target_compression_ratio` parameter (default: 0.3)
2. Add `acceptable_range` parameter (default: ±0.1)
3. Consider adaptive thresholds based on source document characteristics

---

### 🟡 Issue 5: Coherence Weighting Lacks Justification

**Problem** (coherence.py:119):
```python
# Why 30% similarity and 70% discourse?
final_score = 0.3 * similarity_score + 0.7 * discourse_score
```

**Issues:**
- No empirical justification for 0.3/0.7 split
- Fixed weights don't adapt to document characteristics
- Sentence similarity only looks at consecutive sentences (misses long-range coherence)

**Recommendation:**
1. Make weights configurable parameters
2. Add paragraph-level coherence analysis
3. Consider discourse markers (transition words) as additional signal
4. Add entity continuity tracking (coreference chains)

---

## Missing Metrics

### ⭐ High Priority Additions

#### 1. **Factual Alignment (F1-style)**
Combine coverage (current faithfulness) and precision (current hallucination):
```
factual_alignment = 2 × (coverage × precision) / (coverage + precision)
```
**Benefit:** Single metric capturing both completeness and accuracy

#### 2. **Information Density**
Measure content-to-length ratio:
```python
def calculate_information_density(summary, reference):
    # Extract key concepts (entities, key phrases)
    summary_concepts = extract_concepts(summary)
    reference_concepts = extract_concepts(reference)

    # Calculate overlap
    relevant_concepts = summary_concepts & reference_concepts

    # Density = relevant_concepts / summary_length
    return len(relevant_concepts) / len(summary.split())
```
**Benefit:** Identifies verbose vs. information-rich summaries

#### 3. **Extractiveness Score**
Measure how much summary directly copies vs. abstracts:
```python
def calculate_extractiveness(summary, reference):
    # Find longest common subsequences
    # Calculate percentage of summary that's direct extraction
    # Score: 0 = fully abstractive, 1 = fully extractive
```
**Benefit:** Understand summarization strategy, detect copy-paste summaries

#### 4. **Salient Information Coverage**
Not all source content is equally important:
```python
def calculate_salient_coverage(summary, reference, llm_config):
    # 1. LLM extracts "most important" claims from reference (with scores)
    # 2. Check which salient claims are in summary
    # 3. Weight by importance: high-importance claims matter more
```
**Benefit:** Better than current faithfulness which treats all claims equally

#### 5. **Temporal Consistency**
For documents with time-ordered information:
```python
def calculate_temporal_consistency(summary, reference):
    # Extract events with timestamps
    # Check if summary maintains chronological order
    # Penalize temporal inversions or confusions
```
**Benefit:** Catch summaries that mix up sequence of events

### ⭐ Medium Priority Additions

#### 6. **Named Entity Coverage**
Track preservation of key entities:
```python
def calculate_entity_coverage(summary, reference):
    ref_entities = extract_entities(reference, types=['PERSON', 'ORG', 'GPE', 'DATE'])
    sum_entities = extract_entities(summary, types=['PERSON', 'ORG', 'GPE', 'DATE'])

    # Calculate recall and precision for each entity type
    return {
        'entity_recall': len(sum_entities & ref_entities) / len(ref_entities),
        'entity_precision': len(sum_entities & ref_entities) / len(sum_entities)
    }
```
**Benefit:** Ensures key actors, organizations, locations are preserved

#### 7. **Semantic Coverage (Beyond Topics)**
Topic preservation is too coarse-grained:
```python
def calculate_semantic_coverage(summary, reference):
    # Split reference into semantic segments (paragraphs or key points)
    # For each segment, measure if its meaning is captured in summary
    # Use sentence embeddings + similarity threshold
    # Return fine-grained coverage map
```
**Benefit:** More granular than topic preservation, less strict than claim-level

#### 8. **Attribution Quality**
For summaries that include citations/attributions:
```python
def calculate_attribution_quality(summary, reference):
    # Extract attribution statements ("X said", "According to Y")
    # Verify each attribution is accurate
    # Check for missing attributions on controversial claims
```
**Benefit:** Critical for news/research summaries

---

## Improvements to Existing Metrics

### 1. Enhance Faithfulness (or rename to Coverage)

**Current limitations:**
- Extracts claims from source with generic prompt (base.py:103-125)
- Binary check if claim is "present" in summary
- No handling of partial coverage

**Improvements:**
```python
def calculate_coverage_enhanced(reference, summary, llm_config):
    # 1. Extract claims with importance scores
    claims = extract_claims_with_importance(reference, llm_config)

    # 2. For each claim, check coverage level (not just binary)
    #    0: Not mentioned
    #    0.5: Partially mentioned
    #    1: Fully mentioned
    coverage_scores = []
    for claim in claims:
        coverage = check_claim_coverage_level(claim, summary, llm_config)
        weighted_coverage = coverage * claim['importance']
        coverage_scores.append(weighted_coverage)

    # 3. Return weighted average
    return {
        'coverage_score': sum(coverage_scores) / sum(c['importance'] for c in claims),
        'critical_claims_covered': count_critical_claims_covered(claims, coverage_scores),
        'partial_coverage_count': len([c for c in coverage_scores if 0 < c < 1])
    }
```

### 2. Improve Hallucination Detection

**Current limitations:**
- Depends on accurate claim extraction from summary
- Binary hallucination/supported classification
- No severity levels

**Improvements:**
```python
def calculate_hallucination_enhanced(reference, summary, llm_config):
    summary_claims = extract_claims(summary, llm_config)

    hallucination_details = []
    for claim in summary_claims:
        # Multi-level verification
        verification = verify_claim_against_source(
            claim, reference, llm_config,
            return_evidence=True
        )

        # Classify severity
        # 0: Fully supported (evidence found)
        # 1: Plausibly inferred (logical but not explicit)
        # 2: Unsupported (no evidence)
        # 3: Contradicted (evidence against)

        hallucination_details.append({
            'claim': claim,
            'severity': verification['severity'],
            'evidence': verification['evidence_snippet']
        })

    return {
        'hallucination_score': calculate_weighted_score(hallucination_details),
        'severe_hallucinations': [h for h in hallucination_details if h['severity'] >= 2],
        'contradiction_count': len([h for h in hallucination_details if h['severity'] == 3])
    }
```

### 3. Enhance Topic Preservation

**See Issue #2 above** - Add importance weighting and depth scoring.

### 4. Improve Redundancy Detection

**See Issue #3 above** - Use semantic similarity instead of string matching.

### 5. Make Coherence More Comprehensive

**Additions:**
```python
def calculate_coherence_enhanced(summary, llm_config):
    # 1. Sentence-level similarity (existing)
    similarity_score = calculate_sentence_similarity(sentences)

    # 2. Discourse marker analysis (new)
    discourse_score = analyze_discourse_markers(summary)

    # 3. Entity continuity (new)
    entity_continuity = check_entity_coreference_chains(summary)

    # 4. LLM discourse evaluation (existing)
    llm_score = evaluate_discourse_coherence(summary)

    # 5. Combine with configurable weights
    return {
        'coherence': weighted_average([
            (similarity_score, 0.25),
            (discourse_score, 0.15),
            (entity_continuity, 0.10),
            (llm_score, 0.50)
        ]),
        'coherence_breakdown': {
            'sentence_similarity': similarity_score,
            'discourse_markers': discourse_score,
            'entity_continuity': entity_continuity,
            'discourse_quality': llm_score
        }
    }
```

---

## Implementation Priorities

### Phase 1: Critical Fixes (Week 1)
1. ✅ Rename faithfulness → coverage/completeness
2. ✅ Rename hallucination → faithfulness/factual_consistency
3. ✅ Add combined factual_alignment metric (F1-style)
4. ✅ Fix redundancy detection (use semantic similarity)

### Phase 2: High-Value Additions (Week 2-3)
1. ✅ Add information density metric
2. ✅ Add extractiveness score
3. ✅ Add salient information coverage
4. ✅ Enhance topic preservation with weighting

### Phase 3: Refinements (Week 4)
1. ✅ Add named entity coverage
2. ✅ Improve coherence with discourse markers
3. ✅ Make conciseness thresholds configurable
4. ✅ Add temporal consistency for time-ordered content

### Phase 4: Advanced Features (Future)
1. ⏳ Add attribution quality metric
2. ⏳ Add sentence fusion quality
3. ⏳ Add domain-specific metrics (news, scientific, legal)
4. ⏳ Add multi-document summarization metrics

---

## Testing Recommendations

### Unit Tests Needed
1. Test claim extraction consistency
2. Test edge cases (empty summaries, single-sentence summaries)
3. Test multi-lingual support (if applicable)
4. Test performance on various summary lengths

### Evaluation Datasets
Validate against standard benchmarks:
- **CNN/DailyMail** - news summarization
- **XSum** - extreme summarization
- **PubMed** - scientific abstracts
- **MultiNews** - multi-document

### Human Evaluation
Correlate metrics with human judgments:
- Collect human ratings on: relevance, coherence, factuality, conciseness
- Calculate Pearson/Spearman correlation with automated metrics
- Target: ρ > 0.6 for primary metrics

---

## API Design Recommendations

### Backward Compatibility
```python
# Maintain old names as aliases with deprecation warnings
def evaluate_summary(
    full_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    # ... existing params ...
):
    # Map old names to new
    metric_aliases = {
        'faithfulness': 'coverage',  # with deprecation warning
        'hallucination': 'factual_consistency'
    }

    # Emit deprecation warnings if old names used
    # Process with new implementations
```

### New Parameters
```python
def evaluate_summary(
    # ... existing params ...

    # New configuration options
    importance_weighting: bool = True,  # For topic/claim importance
    target_compression: float = 0.3,    # For conciseness
    domain: Optional[str] = None,       # For domain-specific tuning
    return_detailed_analysis: bool = False,  # Return claim-level details
):
    pass
```

### Output Format Enhancement
```python
# Current: flat dictionary
results = {'faithfulness': 0.85, 'coherence': 0.92, ...}

# Proposed: structured with details
results = {
    'scores': {
        'coverage': 0.85,
        'factual_consistency': 0.92,
        'factual_alignment': 0.88,  # F1 of above
        ...
    },
    'details': {
        'coverage': {
            'reference_claims_count': 15,
            'claims_in_summary_count': 13,
            'critical_claims_covered': True,
            'missing_claims': ['claim 1', 'claim 2']
        },
        'factual_consistency': {
            'summary_claims_count': 12,
            'hallucinated_claims_count': 1,
            'severe_hallucinations': [],
            'contradictions': []
        }
    },
    'summary': 'Overall: High quality summary with good coverage and factual accuracy.'
}
```

---

## Metric Selection Guidance

### Use Case Matrix

| Use Case | Recommended Metrics | Priority Order |
|----------|-------------------|----------------|
| **News Summarization** | Factual Alignment, Entity Coverage, Temporal Consistency, Conciseness | 1. Factual Alignment<br>2. Entity Coverage<br>3. Temporal Consistency |
| **Scientific Papers** | Coverage, Salient Coverage, Technical Coherence, Attribution | 1. Salient Coverage<br>2. Coverage<br>3. Attribution Quality |
| **Meeting Notes** | Topic Preservation, Redundancy, Coherence, Extractiveness | 1. Topic Preservation<br>2. Coherence<br>3. Redundancy |
| **Legal Documents** | Factual Alignment, Coverage, Entity Coverage (names, dates) | 1. Factual Alignment<br>2. Coverage<br>3. Entity Coverage |
| **Social Media** | Conciseness, Coherence, Hallucination Detection | 1. Hallucination Detection<br>2. Conciseness<br>3. Coherence |

---

## Performance Considerations

### Current Bottlenecks
1. **Multiple LLM calls per metric** - Claim extraction, verification, scoring all separate
2. **Sequential processing** - Metrics calculated one at a time
3. **No caching** - Re-extracting claims for multiple metrics

### Optimization Strategies

#### 1. Batch LLM Operations
```python
# Instead of:
claims = extract_claims(text)  # LLM call 1
for claim in claims:
    verify(claim)  # LLM call 2, 3, 4...

# Do:
claims_and_verification = extract_and_verify_claims_batch(text)  # Single LLM call
```

#### 2. Shared Preprocessing
```python
class MetricContext:
    """Shared context to avoid redundant computations"""
    def __init__(self, reference, summary, llm_config):
        self.reference = reference
        self.summary = summary

        # Compute once, use many times
        self._reference_claims = None
        self._summary_claims = None
        self._reference_topics = None
        self._reference_entities = None

    @cached_property
    def reference_claims(self):
        if self._reference_claims is None:
            self._reference_claims = extract_claims(self.reference, self.llm_config)
        return self._reference_claims
```

#### 3. Parallel Metric Calculation
```python
# Run independent metrics in parallel
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(calculate_rouge, ref, summ): 'rouge',
        executor.submit(calculate_bert_score, ref, summ): 'bert_score',
        executor.submit(calculate_coherence, summ): 'coherence',
    }
    results = {name: future.result() for future, name in futures.items()}
```

---

## Conclusion

The current summarization metrics provide a solid foundation with good coverage of traditional and LLM-based approaches. However, key improvements are needed:

### Must-Fix Issues
1. **Naming confusion** between faithfulness and hallucination metrics
2. **Fragile parsing** in redundancy detection
3. **Lack of importance weighting** in topic preservation

### High-Impact Additions
1. **Information density** - Identify truly concise vs. verbose summaries
2. **Extractiveness** - Understand copy vs. abstraction
3. **Salient coverage** - Focus on what matters most
4. **Factual alignment** - Combined precision/recall metric

### Long-Term Vision
- Domain-specific metric bundles
- Multi-lingual support
- Adaptive thresholds based on document type
- Integration with human feedback loops

**Estimated Effort:**
- Phase 1 (Critical): 1 week
- Phase 2 (High-value): 2-3 weeks
- Phase 3 (Refinements): 1 week
- Total: 4-5 weeks for comprehensive improvement

**Expected Impact:**
- More intuitive metric naming → Better user experience
- Importance weighting → Better correlation with human judgments
- Additional metrics → Cover more summarization quality aspects
- Performance optimizations → 2-3x faster evaluation

