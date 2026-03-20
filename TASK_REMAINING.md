# Remaining Refactor Tasks

Tasks 1-2 are DONE. Do tasks 3, 4, 5 below.

## Task 3: Split `benchmark.py` (currently ~570 lines) into a package

Current `benchmark.py` does 4 things: dataset loading, type detection, Layer 1 running, Layer 2 running.

Refactor into `benchmark/` package:
- `src/dq/benchmark/__init__.py` — re-exports ALL current public names for backward compat
- `src/dq/benchmark/datasets.py` — `load_fineweb_sample`, `load_alpaca_sample`, `load_alpaca_original`, `load_alpaca_cleaned`, `_merge_alpaca_fields`, `_ensure_datasets`, constants (FINEWEB_DATASET, etc.)
- `src/dq/benchmark/runner.py` — `run_benchmark`, `run_llm_scoring`, `_score_docs`
- `src/dq/benchmark/types.py` — dataclasses: SFTScores, PretrainScores, RuleStats, FilterResult, DatasetResult, BenchmarkReport
- `src/dq/benchmark/utils.py` — `detect_data_type`, `_extract_sft_fields`, SFT_FIELDS import

**CRITICAL**: 
- Delete the old `src/dq/benchmark.py` file AFTER creating the package
- Keep ALL existing imports working: `from dq.benchmark import run_benchmark` etc.
- The old `benchmark.py` and new `benchmark/` directory CANNOT coexist — Python will be confused
- Run tests to verify nothing breaks

## Task 4: Clarify contamination module positioning

`src/dq/contamination/` has 4 files (ngram.py, min_k_prob.py, ts_guessing.py, report.py) but is NOT in the pipeline filter chain.

- Update `contamination/__init__.py` with a clear docstring: it's a standalone analysis tool, NOT a pipeline filter
- Verify `dq contamination` CLI commands still work by checking the CLI code references

## Task 5: Update README.md and CLAUDE.md

Reflect ALL recent changes:
- Unified LLM Judge (`src/dq/judge.py`) with data-driven rules (add rule = append to RULES list)
- DEITA fully removed (no DiversityFilter, no 1-6 scorers)  
- Auto-scan filter registration via `ensure_registered()` — no manual __init__.py imports
- `filter()` delegates to `filter_detailed()` in sft_rules
- `SFT_DETECT_FIELDS` single source of truth from sft_rules.py
- `benchmark/` is now a package
- Dead code removed: llm_scorer.py, model_filters test
- Architecture: Layer 1 (Rule-based) + Layer 2 (LLM Binary Judge)
- "Adding New Rules" section showing data-driven approach

## Rules
- Run `uv run pytest tests/ -q` after EACH change — all 254 tests must pass
- `git add -A && git commit` with descriptive messages
- `git push` after all commits
- Do NOT reduce test count
