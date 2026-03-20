# Architecture Refactor Task

You are working on `/home/pzheng46/data-quality-agent`. Read `CLAUDE.md` first for project rules.

## Tasks (do ALL of them)

### 1. Delete `model_filters/llm_scorer.py` (dead code)
- This is the FineWeb-Edu transformer classifier (1-5 scale scoring)
- Dependencies (torch, transformers) are NOT installed and never will be
- It's registered as `llm_quality` filter — remove that registration too
- Delete the file, fix any imports, ensure tests pass

### 2. Merge `_score_sft_docs` and `_score_pretrain_docs` in `benchmark.py`
- These two functions are structurally identical — same pattern: create judge → iterate docs → count high/low
- Merge into a single `_score_docs()` function that takes data type as parameter
- Use the unified `LLMJudge` from `src/dq/judge.py` (it has `judge_sft()` and `judge_text()`)
- Update all callers

### 3. Split `benchmark.py` (652 lines) into a package
- Current `benchmark.py` does 4 things: dataset loading, type detection, Layer 1 running, Layer 2 running
- Refactor into `benchmark/` package:
  - `benchmark/__init__.py` — re-exports for backward compat
  - `benchmark/datasets.py` — all `load_*` functions, `_merge_alpaca_fields`
  - `benchmark/runner.py` — `run_benchmark`, `run_llm_scoring`, `_score_docs`
  - `benchmark/types.py` — dataclasses (SFTScores, PretrainScores, RuleStats, FilterResult, DatasetResult, BenchmarkReport)
  - `benchmark/utils.py` — `detect_data_type`, `_extract_sft_fields`, `SFT_FIELDS` import
- Keep ALL existing public API accessible from `from dq.benchmark import X`

### 4. Clarify contamination module positioning
- `contamination/` has 4 files but is NOT integrated into the main pipeline filter chain
- Add a clear docstring to `contamination/__init__.py` explaining it's a **standalone analysis tool**, not a pipeline filter
- Make sure CLI `dq contamination` commands work independently

### 5. Update README.md and CLAUDE.md
- Reflect ALL recent changes:
  - Unified LLM Judge (`src/dq/judge.py`) with data-driven rules
  - DEITA fully removed (no more DiversityFilter, no more 1-6 scorers)
  - Auto-scan filter registration (no manual `__init__.py` imports needed)
  - `filter()` delegates to `filter_detailed()` in sft_rules
  - `SFT_DETECT_FIELDS` single source of truth
  - `benchmark/` is now a package (if you split it)
  - Architecture: Layer 1 (Rule-based) + Layer 2 (LLM Binary Judge with unified judge)
- Update the "Adding New Rules" section to show the data-driven approach

## Rules
- Run `uv run pytest tests/ -q` after EACH change and ensure ALL tests pass before moving on
- `git add -A && git commit` after each logical change with a descriptive message
- `git push` after all commits
- Do NOT delete test files unless the feature they test is also deleted
- Do NOT modify tests in ways that reduce coverage — only update mocks/assertions to match new API
