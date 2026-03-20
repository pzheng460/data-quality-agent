# Task: Implement Phase 3 — Contamination Detection

## Context
Phase 1 (heuristic filters) and Phase 2 (model-based filters) are complete. 94 tests passing.

Your job: implement Phase 3 contamination detection in `/home/pzheng46/data-quality-agent`.

Read existing code first: `src/dq/contamination/` has stubs for ngram.py, min_k_prob.py, ts_guessing.py.

## Phase 3 Modules

### 1. N-gram Overlap Detection (`src/dq/contamination/ngram.py`)
**Paper**: GPT-3, Llama, various decontamination papers
**Method**: Check if training data overlaps with benchmark test sets using n-gram matching.

**Implementation**:
- `NgramContaminationDetector(n: int = 13, threshold: float = 0.8)`
- `load_benchmark(name: str, split: str = "test") -> list[str]` — load known benchmarks:
  - Support loading from: MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, HumanEval
  - Also support loading custom benchmark from a text/jsonl file
- `build_index(benchmark_texts: list[str])` — build n-gram set from benchmark
- `check_contamination(doc: str) -> ContaminationResult` with fields:
  - `is_contaminated: bool`
  - `overlap_ratio: float` (fraction of doc n-grams found in benchmark)
  - `matched_ngrams: int`
  - `total_ngrams: int`
  - `matched_benchmark: str` (which benchmark matched)
- `scan_dataset(docs: list[dict], benchmarks: dict[str, list[str]]) -> ContaminationReport`
  - Scan full dataset against multiple benchmarks
  - Report: per-benchmark contamination rate, sample contaminated docs
- Use efficient set lookup (frozenset of n-gram tuples)
- Normalize text: lowercase, strip punctuation, collapse whitespace

### 2. Min-K% Prob Detection (`src/dq/contamination/min_k_prob.py`)
**Paper**: "Detecting Pretraining Data from Large Language Models" (Shi et al., 2023)
**Method**: Black-box detection — if a model assigns high probability to ALL tokens in a text, that text was likely in training data. Min-K% looks at the K% lowest-probability tokens.

**Implementation**:
- `MinKProbDetector(model_name: str = "Qwen/Qwen2-0.5B", k_percent: float = 20.0, threshold: float = -0.5, device: str = "auto")`
- `compute_min_k_prob(text: str) -> float` — compute Min-K% Prob score
  - Tokenize text, get log probabilities for each token
  - Take the bottom K% of log probs
  - Average them → score
  - Higher score (less negative) = more likely contaminated
- `check_contamination(doc: str) -> ContaminationResult`
- Batch support for efficiency
- Lazy model loading, GPU/CPU auto-detection
- Graceful skip if transformers not installed

### 3. TS-Guessing for MCQ (`src/dq/contamination/ts_guessing.py`)
**Paper**: "Time Travel in LLMs: Tracing Data Contamination in Large Language Models"
**Method**: For multiple-choice questions, check if a model can guess the correct answer given only the choices (without the question). If yes → likely contaminated.

**Implementation**:
- `TSGuessingDetector(api_url: str | None, api_key: str | None, model: str = "gpt-4o-mini")`
- `check_mcq_contamination(question: str, choices: list[str], correct_idx: int) -> TSGuessingResult`
  - Present only the choices to the model
  - Ask it to guess the correct answer
  - If it guesses correctly at rate >> 1/num_choices → contaminated
- `scan_mcq_dataset(items: list[dict]) -> ContaminationReport`
  - Each item: {question, choices, correct_idx}
  - Report: overall guess accuracy vs random baseline
  - Statistical test (binomial) to determine significance
- Rate limiting + retry for API calls
- Graceful skip if no API configured

### 4. Contamination Report (`src/dq/contamination/report.py`) — NEW FILE
- `ContaminationReport` dataclass with:
  - `dataset_name: str`
  - `total_docs: int`
  - `contaminated_docs: int`
  - `contamination_rate: float`
  - `per_benchmark: dict[str, BenchmarkContamination]`
  - `sample_contaminated: list[dict]` (top 10 examples)
- Rich table output
- Markdown report generation
- JSON export

### 5. CLI Integration
Add to `src/dq/cli.py`:
```bash
# Check contamination against common benchmarks
uv run dq contamination <input.jsonl> --benchmarks mmlu,hellaswag,arc

# Check against custom benchmark file
uv run dq contamination <input.jsonl> --benchmark-file my_test_set.jsonl

# N-gram only (fast, no model needed)
uv run dq contamination <input.jsonl> --method ngram

# Min-K% Prob (needs GPU/model)
uv run dq contamination <input.jsonl> --method min-k-prob

# Full report
uv run dq contamination <input.jsonl> -o contamination_report/
```

### 6. Tests
- `tests/test_contamination.py`
  - Test n-gram overlap with known overlapping/non-overlapping texts
  - Test Min-K% Prob with mocked model
  - Test TS-Guessing with mocked API
  - Test report generation
  - Test CLI command
  - All model/API tests use mocks

### 7. Update benchmark
- Add contamination check to `dq bench` with `--check-contamination` flag
- Check Alpaca original vs cleaned against MMLU/HellaSwag (n-gram only, fast)
- Include results in benchmark report

### 8. After implementation
- Run `uv run pytest -q` — all tests must pass
- Git commit with descriptive messages
- Update README.md with Phase 3 usage

## Key Points
- N-gram detection is the PRIMARY method (fast, no model needed, well-validated)
- Min-K% Prob and TS-Guessing are secondary (need models/API)
- All model-dependent code must gracefully degrade
- N-gram matching must be FAST (use set operations, not loops)
- Normalize text consistently before n-gram extraction

When completely finished, run: openclaw system event --text "Done: Phase 3 contamination detection (n-gram, Min-K%, TS-Guessing)" --mode now
