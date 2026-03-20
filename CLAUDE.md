# CLAUDE.md — dq (Training Data Quality Agent)

## What is this?
A Python CLI + library for detecting and filtering low-quality LLM training data. Implements methods from Gopher, C4, FineWeb, DCLM, DEITA, and contamination detection papers.

## ⚠️ Mandatory Rules
1. **Every significant change must be committed and pushed** — `git add -A && git commit -m "..." && git push`
2. **Update both README.md AND CLAUDE.md** when changing architecture, adding features, or modifying filters/configs
3. **Run `uv run pytest` before every commit** — never push broken tests
4. **Read the Permanent Memory section below** before starting work — it contains critical lessons

## 🧠 Permanent Memory

### Architectural Decisions
- **Filter Registry pattern**: Filters self-register via `@register_filter("name")`. New filters MUST use this decorator.
- **Graceful degradation**: Model/API-dependent code catches ImportError → warns → passes through. NEVER crash on missing optional deps.
- **Config-driven thresholds**: Every numeric threshold must be YAML-configurable. No hardcoded magic numbers in filter logic.
- **Chinese support is first-class**: All text processing must handle CJK. Use `stats.py` utilities (CJK-aware word count, Chinese punctuation).

### Known Issues & Lessons Learned
- **`char_repetition_ratio` is a length proxy, not a quality signal**: Investigation (2026-03-20) revealed the root cause:
  - Longer text → more char 10-gram collisions (birthday paradox) → higher score
  - Cleaned Alpaca (avg 727 chars, mean score 0.625) scores HIGHER than Original (avg 353 chars, mean 0.510)
  - It penalizes detailed, high-quality responses and passes short/empty ones
  - Algorithm: `sum((count-1)*n for repeated n-grams) / len(text)` — not properly normalized for text length
  - **For SFT data**: disable entirely (sft.yaml sets threshold=1.0)
  - **For pre-training data**: still marginally useful for catching true copy-paste boilerplate in long docs, but threshold 0.40 is barely above normal English text (~0.39)
- **Heuristic filters don't work for SFT quality**: Gopher/C4/FineWeb are designed for pre-training web data. Alpaca Original vs Cleaned benchmark showed no meaningful discrimination on SFT data. SFT quality differences (hallucinations, errors) require Phase 2 model-based evaluation.
- **`tatsu-lab/stanford_alpaca` removed from HuggingFace**: Must download from GitHub raw URL. Cached at `~/.cache/dq/alpaca_original.json`.
- **`trust_remote_code=True` deprecated in newer HuggingFace datasets lib**: Remove this parameter from all `load_dataset()` calls.
- **`gopher_repetition` direction reverses on SFT data**: Cleaned Alpaca has LOWER pass rate than original because cleaned responses are longer with more natural phrase repetition. This is expected behavior, not a bug.

### Redundancy Analysis (2026-03-20, revised)
**No truly redundant steps.** Every filter serves an independent role:
- `length`: universal baseline pre-filter (no paper affiliation). Needed when running C4/FineWeb without Gopher.
- `char_repetition` vs `n-gram ratio`: character-level vs word-level, complementary
- `fineweb.dup_line_ratio` vs `gopher_repetition.dup_line_ratio`: same metric from independent papers (Gopher 2021 vs FineWeb 2024), kept for method-level independence
- `max_symbol_ratio` vs `min_alpha_ratio`: different measurements (symbol words vs alphabetic chars)
- Top 2/3/4-gram: multi-scale analysis at different granularities

**Principle**: Each paper's filter suite (Gopher, C4, FineWeb) should remain self-contained. Users may run any subset. Don't break method independence. `length` is the universal safety net.

### Benchmark Reference (2026-03-20, default config, 1000 samples)
| Filter | Alpaca Original | Alpaca Cleaned | Δ |
|--------|----------------|----------------|---|
| gopher_quality | 44.7% | 59.6% | +14.9% ✅ |
| gopher_repetition | 32.2% | 19.3% | -12.9% ⚠️ reversed |
| c4 | 93.1% | 94.8% | +1.7% — |
| fineweb | 97.0% | 96.3% | -0.7% — |
| pii | 100.0% | 100.0% | 0% — |
| contamination (ngram) | 0% | — | clean ✅ |

## Quick Commands
```bash
uv sync                              # Install deps
uv sync --extra model                # + model-based filters (torch, transformers)
uv sync --extra bench                # + benchmark datasets (HuggingFace datasets)
uv run pytest                        # Run all tests (140 tests, ~0.7s)
uv run dq run input.jsonl -o out.jsonl       # Run pipeline
uv run dq bench --no-dedup -n 1000           # Benchmark: Alpaca Original vs Cleaned
uv run dq contamination input.jsonl --benchmarks mmlu,hellaswag  # Contamination check
```

## Architecture
```
src/dq/
├── cli.py              # Click CLI: run, stats, report, dedup, bench, score, contamination
├── pipeline.py         # Pipeline orchestrator — chains filters, tracks per-filter stats
├── config.py           # PipelineConfig loaded from YAML (configs/*.yaml)
├── report.py           # JSON + Markdown report generator
├── benchmark.py        # Alpaca Original vs Cleaned comparison framework
├── benchmark_report.py # Rich table + MD/JSON benchmark output
├── filters/
│   ├── base.py         # BaseFilter ABC: filter(doc) -> (keep: bool, info: dict)
│   ├── gopher.py       # GopherQualityFilter + GopherRepetitionFilter
│   ├── c4.py           # C4Filter
│   ├── fineweb.py      # FineWebFilter
│   ├── pii.py          # PIIFilter (detect/redact mode, CN phone/ID/bank)
│   └── length.py       # LengthFilter
├── dedup/
│   ├── exact.py        # SHA256 exact dedup
│   └── minhash.py      # MinHash LSH (datasketch, FineWeb params: 5-gram/112h/14×8)
├── model_filters/      # Phase 2 — require `uv sync --extra model`
│   ├── fasttext_quality.py  # DCLM-style fastText classifier
│   ├── perplexity.py        # Small LM perplexity filter (Qwen2-0.5B)
│   └── llm_scorer.py        # FineWeb-Edu quality scorer (0-5 scale)
├── sft/                # Phase 2 — SFT-specific (DEITA paper)
│   ├── complexity.py   # Instruction complexity scorer (LLM API)
│   ├── quality.py      # Response quality scorer (LLM API)
│   └── diversity.py    # Embedding diversity filter (batch, cosine sim)
├── contamination/      # Phase 3
│   ├── ngram.py        # 13-gram overlap against benchmark test sets (primary method)
│   ├── min_k_prob.py   # Min-K% Prob black-box detection (needs model)
│   ├── ts_guessing.py  # TS-Guessing for MCQ benchmarks (needs API)
│   └── report.py       # ContaminationReport dataclass + rich output
└── utils/
    ├── io.py           # read_documents / write_documents (jsonl/parquet/csv)
    ├── stats.py        # word_count, avg_word_len, char_repetition_ratio, etc. (CJK-aware)
    └── tokenizer.py    # Simple tokenizer utilities
```

## Key Design Patterns

### Filter Registry
Filters self-register via `@register_filter("name")` decorator in `filters/base.py`. Pipeline looks up filters by name from YAML config.

### BaseFilter Interface
```python
class BaseFilter(ABC):
    def filter(self, doc: dict) -> tuple[bool, dict]:
        """Returns (keep, info_dict). info_dict has filter name + reason on drop."""
```

### Pipeline Flow
```
Input docs → [filter1] → [filter2] → ... → [dedup] → Output docs
                ↓              ↓
           stats tracked   stats tracked
```
Each filter sees only docs that passed previous filters. Stats are per-filter.

### Graceful Degradation
Model filters (Phase 2/3) catch ImportError on missing deps → log warning → pass all docs through. Never crashes on missing optional packages.

### Config-Driven
Everything is threshold-configurable via YAML. Three presets: `default.yaml` (pre-training EN), `pretrain_zh.yaml` (pre-training CN), `sft.yaml` (SFT/instruction data).

## Important Thresholds & Params
- **char_repetition**: default 0.40 (was 0.20, raised because normal English text reaches ~0.39)
- **MinHash LSH**: 5-gram, 112 perms, 14 bands × 8 rows → Jaccard ~0.75 (from FineWeb paper)
- **N-gram contamination**: 13-gram, overlap threshold 0.8
- **Gopher min_words**: 50 for pre-training, 10 for SFT
- **SFT char_repetition**: disabled (1.0) — char n-gram method unsuited for short SFT data

## Benchmark Findings
Alpaca Original vs yahma/alpaca-cleaned (1000 samples, default config):
- `gopher_quality`: Original 44.7% vs Cleaned 59.6% (+14.9%) ✅ — catches short/malformed text
- `gopher_repetition`: reversed direction for SFT data — char n-gram not suitable
- `c4/fineweb/pii`: no signal on SFT data (expected)
- **Key insight**: Heuristic filters work for pre-training web data; SFT quality differences (hallucinations) need model-based evaluation (Phase 2)

## Data Flow Notes
- `tatsu-lab/stanford_alpaca` removed from HuggingFace — downloaded from GitHub raw URL, cached at `~/.cache/dq/alpaca_original.json`
- `yahma/alpaca-cleaned` loaded via HuggingFace datasets
- Alpaca fields merged: `instruction + "\n" + input + "\n" + output → text`

## Testing
```bash
uv run pytest -q          # 140 tests, all mocked (no real models/APIs/downloads)
uv run pytest -v -k gopher   # Run specific filter tests
```
Tests use fixtures in `tests/fixtures/` (sample_good.jsonl, sample_bad.jsonl, sample_sft.jsonl). Model/API tests fully mocked with unittest.mock.

## Common Tasks

### Add a new filter
1. Create `src/dq/filters/my_filter.py`
2. Subclass `BaseFilter`, implement `filter(self, doc) -> (bool, dict)`
3. Decorate with `@register_filter("my_filter")`
4. Import in `src/dq/filters/__init__.py`
5. Add to config YAML under `pipeline.filters`
6. Add tests in `tests/`

### Change filter thresholds
Edit `configs/*.yaml` — every numeric threshold is configurable. CLI: `dq run -c my_config.yaml`.

### Run on Chinese data
Use `configs/pretrain_zh.yaml` — CJK-aware word counting, Chinese punctuation, CN-specific PII patterns.

## Dependencies
- **Core**: click, pyyaml, datasketch, xxhash, regex, tqdm, rich
- **Model** (optional): fasttext-wheel, transformers, sentence-transformers, torch, openai
- **Bench** (optional): datasets (HuggingFace)
- **Dev**: pytest, pyarrow
chuang