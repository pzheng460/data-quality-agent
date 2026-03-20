# Task: Refactor Layer 2 from Absolute Scoring to Rule-Based Binary Classification

## Context
Current LLM scoring uses absolute 1-6 scales (complexity, quality, educational value, writing quality). This is unreliable because:
- LLMs default to safe middle scores (3-4)
- No calibration — "3" has no clear meaning
- Cross-dataset comparison is meaningless (different domains)

## Goal
Replace absolute scoring with **rule-based binary classification**: each sample is judged HIGH or LOW quality based on specific, verifiable rules. Each rule produces a binary pass/fail + brief reason.

## New Architecture

### 1. Create `src/dq/sft/llm_judge.py` — SFT Quality Judge

Binary classification for SFT (instruction-response) data. Check these 5 rules:

| Rule | PASS (HIGH) | FAIL (LOW) |
|------|------------|------------|
| `instruction_following` | Response directly addresses the instruction | Off-topic, ignores key requirements |
| `factuality` | No obvious factual errors | Contains fabricated facts, wrong information |
| `completeness` | Covers what the instruction asks for | Missing critical steps, incomplete answer |
| `format_compliance` | Matches requested format (if any) | Asked for JSON but gave prose, etc. |
| `harmlessness` | No harmful/dangerous content | Contains dangerous instructions, discrimination |

**Prompt design**: Send ONE prompt with all 5 rules, ask LLM to return structured JSON:
```json
{
  "instruction_following": {"pass": true, "reason": ""},
  "factuality": {"pass": false, "reason": "Claims Python was created in 2005"},
  "completeness": {"pass": true, "reason": ""},
  "format_compliance": {"pass": true, "reason": ""},
  "harmlessness": {"pass": true, "reason": ""}
}
```

Final verdict: ALL rules pass → HIGH quality. ANY rule fails → LOW quality.

Class interface:
```python
class SFTQualityJudge:
    def judge_one(self, instruction: str, output: str) -> dict:
        """Returns {"quality": "high"|"low", "rules": {...}, "failed_rules": [...]}"""
    
    def judge_batch(self, docs: list[dict]) -> list[dict]:
        """Judge multiple documents."""
```

### 2. Create `src/dq/model_filters/llm_quality_judge.py` — Pre-training Quality Judge

Binary classification for pre-training (plain text) data. Check these 3 rules:

| Rule | PASS (HIGH) | FAIL (LOW) |
|------|------------|------------|
| `information_density` | Contains substantive, informative content | Ads, navigation, boilerplate, cookie notices |
| `coherence` | Well-structured, complete paragraphs | Truncated, fragmented, garbled text |
| `originality` | Appears to be original content | SEO spam, template-generated, machine-generated filler |

Same structured JSON output format. ALL pass → HIGH, ANY fail → LOW.

Class interface:
```python
class PretrainingQualityJudge:
    def judge_one(self, text: str) -> dict:
        """Returns {"quality": "high"|"low", "rules": {...}, "failed_rules": [...]}"""
    
    def judge_batch(self, docs: list[dict]) -> list[dict]:
        """Judge multiple documents."""
```

### 3. Keep old scorers but mark as deprecated

Don't delete `complexity.py`, `quality.py`, `educational.py`, `writing_quality.py`. Just add a deprecation notice at the top of each file:
```python
# DEPRECATED: Use SFTQualityJudge (llm_judge.py) for binary quality classification.
# This absolute scoring approach (1-6) is unreliable. Kept for backward compatibility.
```

### 4. Add tests

Create `tests/test_llm_judge.py`:
- Mock the LLM API calls
- Test that well-formed instruction-response pairs get HIGH
- Test that bad examples (off-topic, factual errors, incomplete) get LOW  
- Test that pre-training judge correctly identifies ads/boilerplate as LOW
- Test JSON parsing edge cases (malformed LLM response → graceful fallback)
- Test that all rules are checked (no early return)

### 5. Update benchmark integration

In the benchmark script (`/tmp/run_bench.py` or `benchmark.py`), Layer 2 should now show:
```
LAYER 2: LLM Quality Judge (N samples per dataset)

--- SFT Quality ---
Dataset          HIGH%   LOW%   Top Fail Reasons
Alpaca Orig      62%     38%    factuality(25%), completeness(18%)
Alpaca Clean     85%     15%    factuality(10%), completeness(8%)
Dolly            78%     22%    completeness(15%), format(5%)
WizardLM         72%     28%    harmlessness(12%), factuality(10%)

--- Pre-training Quality ---
Dataset          HIGH%   LOW%   Top Fail Reasons  
C4               65%     35%    info_density(20%), coherence(12%)
OpenWebText      82%     18%    info_density(10%), originality(5%)
FineWeb          75%     25%    info_density(18%), coherence(8%)
```

## Implementation Notes
- Use `dq.llm_client.get_client()` and `get_default_model()` for API access
- Use `temperature=0.0` for deterministic results
- Parse JSON from LLM response robustly — try `json.loads()` first, then regex fallback for extracting pass/fail
- Truncate input to 4000 chars to avoid token limits
- Add retry logic (same pattern as existing scorers)
- `max_tokens=500` should be enough for the structured JSON response

## Files to create/modify
1. **CREATE**: `src/dq/sft/llm_judge.py`
2. **CREATE**: `src/dq/model_filters/llm_quality_judge.py`
3. **CREATE**: `tests/test_llm_judge.py`
4. **MODIFY**: `src/dq/sft/complexity.py` — add deprecation notice
5. **MODIFY**: `src/dq/sft/quality.py` — add deprecation notice
6. **MODIFY**: `src/dq/sft/educational.py` — add deprecation notice
7. **MODIFY**: `src/dq/sft/writing_quality.py` — add deprecation notice

## After implementation
Run `uv run pytest` to verify all tests pass, then `git add -A && git commit -m "refactor: replace absolute LLM scoring with rule-based binary classification" && git push`.
