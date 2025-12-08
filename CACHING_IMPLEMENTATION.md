# Metric Caching Implementation

## Problem
When users requested `factual_alignment` along with `coverage` and/or `factual_consistency`, redundant LLM API calls were being made because `factual_alignment` internally calculates both metrics to compute its F1 score.

**Example:** With `metrics=["coverage", "factual_consistency", "factual_alignment"]`:
- `coverage` runs (1 LLM call)
- `factual_consistency` runs (1 LLM call)
- `factual_alignment` runs and internally:
  - Calls `calculate_coverage()` again (duplicate LLM call!)
  - Calls `calculate_factual_consistency()` again (duplicate LLM call!)

**Result:** 4 LLM calls instead of 2 = wasted time and API costs.

## Solution
Implemented a caching mechanism that:
1. Allows `factual_alignment` to accept precomputed results
2. Passes already-computed results from the main evaluation loop
3. Skips redundant calculations when metrics are already in results

## Changes Made

### 1. [factual_alignment.py](assert_llm_tools/metrics/summary/factual_alignment.py)
Added internal parameters to accept precomputed values:

```python
def calculate_factual_alignment(
    reference: str,
    candidate: str,
    llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None,
    verbose: bool = False,
    _precomputed_coverage: Optional[Dict[str, float]] = None,      # NEW
    _precomputed_consistency: Optional[Dict[str, float]] = None    # NEW
) -> Dict[str, float]:
```

Logic now checks for precomputed values before calculating:
```python
# Use precomputed if available, otherwise calculate
if _precomputed_coverage is not None and 'coverage' in _precomputed_coverage:
    coverage_results = _precomputed_coverage
    coverage_score = coverage_results['coverage']
else:
    coverage_results = calculate_coverage(...)
```

### 2. [core.py](assert_llm_tools/core.py)
Updated the metric evaluation loop in `evaluate_summary()`:

**a) Skip already-computed metrics:**
```python
elif metric == "coverage":
    if "coverage" not in results:  # NEW: Check before computing
        results.update(calculate_coverage(...))

elif metric == "factual_consistency":
    if "factual_consistency" not in results:  # NEW: Check before computing
        results.update(calculate_factual_consistency(...))
```

**b) Pass precomputed results to factual_alignment:**
```python
elif metric == "factual_alignment":
    results.update(calculate_factual_alignment(
        full_text, summary, llm_config, custom_instruction, verbose=verbose,
        _precomputed_coverage=results if "coverage" in results else None,      # NEW
        _precomputed_consistency=results if "factual_consistency" in results else None  # NEW
    ))
```

## How It Works

### Scenario 1: User requests all three metrics in order
`metrics=["coverage", "factual_consistency", "factual_alignment"]`

1. Loop iteration 1: Calculate `coverage` → add to `results`
2. Loop iteration 2: Calculate `factual_consistency` → add to `results`
3. Loop iteration 3: Calculate `factual_alignment`:
   - Receives both precomputed values
   - Uses cached results instead of recalculating
   - **Result: 2 LLM calls instead of 4 ✓**

### Scenario 2: User requests alignment first
`metrics=["factual_alignment", "coverage", "factual_consistency"]`

1. Loop iteration 1: Calculate `factual_alignment`:
   - No precomputed values available
   - Calculates both coverage and consistency
   - Adds all three scores to `results`
2. Loop iteration 2: Process `coverage`:
   - Checks: `"coverage" not in results` → False
   - **Skips calculation** ✓
3. Loop iteration 3: Process `factual_consistency`:
   - Checks: `"factual_consistency" not in results` → False
   - **Skips calculation** ✓
   - **Result: 2 LLM calls instead of 4 ✓**

### Scenario 3: User requests only alignment
`metrics=["factual_alignment"]`

1. Loop iteration 1: Calculate `factual_alignment`:
   - No precomputed values available
   - Calculates both coverage and consistency
   - Returns all three scores
   - **Result: 2 LLM calls (same as before) ✓**

## Benefits

1. **Automatic optimization** - No user action required
2. **Order-independent** - Works regardless of metric order
3. **Backward compatible** - Existing code works identically
4. **Cost savings** - Eliminates redundant LLM API calls
5. **Time savings** - Faster evaluation when using multiple metrics
6. **Clean API** - Internal parameters (prefixed with `_`) don't clutter public interface

## Testing

Created comprehensive logic tests in `test_caching_logic.py` that verify:
- ✓ No precomputed values: both metrics calculated
- ✓ Precomputed coverage only: consistency still calculated
- ✓ Precomputed consistency only: coverage still calculated
- ✓ Both precomputed: neither recalculated
- ✓ Metric order 1: coverage → consistency → alignment (uses cache)
- ✓ Metric order 2: alignment → coverage → consistency (skips duplicates)

All tests pass successfully.

## Files Modified

1. `assert_llm_tools/metrics/summary/factual_alignment.py` - Added caching parameters
2. `assert_llm_tools/core.py` - Added result checking and precomputed value passing
3. `assert_llm_tools/llm/bedrock.py` - Fixed syntax error (line 104)

## Files Created

1. `test_caching_logic.py` - Logic verification tests
2. `test_metric_caching.py` - Integration test suite (requires full installation)
3. `test_metric_caching_unit.py` - Unit test approach
4. `CACHING_IMPLEMENTATION.md` - This documentation
