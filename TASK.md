# Task: Build a Training Data Quality Detection Agent

## Overview
Build a Python CLI tool + library for detecting and filtering low-quality training data for LLM pre-training and SFT fine-tuning. The tool should implement a pipeline of filters based on validated methods from top-tier papers (FineWeb, DCLM, Gopher, DEITA, etc.).

## Architecture

```
data-quality-agent/
├── pyproject.toml          # uv project, Python 3.10+
├── src/
│   └── dq/                 # Main package
│       ├── __init__.py
│       ├── cli.py           # CLI entry point (click/typer)
│       ├── pipeline.py      # Pipeline orchestrator
│       ├── config.py        # YAML-based pipeline config
│       ├── report.py        # Quality report generation (JSON + Markdown)
│       ├── filters/
│       │   ├── __init__.py
│       │   ├── base.py      # BaseFilter ABC
│       │   ├── gopher.py    # Gopher Quality + Repetition filters
│       │   ├── c4.py        # C4 filters (terminal punct, js, lorem ipsum, etc.)
│       │   ├── fineweb.py   # FineWeb custom filters (list docs, dup lines, bad line breaks)
│       │   ├── language.py  # Language detection (fastText lid.176.bin)
│       │   ├── pii.py       # PII detection/redaction (email, IP, phone, ID card)
│       │   ├── kl_div.py    # KL divergence filter (Llama 3 style)
│       │   └── length.py    # Simple length-based filters
│       ├── dedup/
│       │   ├── __init__.py
│       │   ├── exact.py     # SHA256 exact dedup
│       │   ├── minhash.py   # MinHash LSH near-dedup (FineWeb params: 5-gram, 112 hashes, 14×8 bands)
│       │   └── semantic.py  # SemDeDup (embedding + clustering, optional)
│       ├── model_filters/
│       │   ├── __init__.py
│       │   ├── fasttext_quality.py  # fastText quality classifier (DCLM style)
│       │   ├── perplexity.py        # Perplexity filter (using small LM)
│       │   └── llm_scorer.py        # LLM scoring + distillation (FineWeb-Edu style)
│       ├── sft/
│       │   ├── __init__.py
│       │   ├── complexity.py   # DEITA Evol-Complexity scorer
│       │   ├── quality.py      # DEITA Evol-Quality scorer
│       │   └── diversity.py    # DEITA Repr Filter (embedding diversity)
│       ├── contamination/
│       │   ├── __init__.py
│       │   ├── ngram.py        # 13-gram overlap detection
│       │   ├── min_k_prob.py   # Min-K% Prob black-box detection
│       │   └── ts_guessing.py  # TS-Guessing for MCQ benchmarks
│       └── utils/
│           ├── __init__.py
│           ├── io.py           # Read/write jsonl, parquet, csv
│           ├── stats.py        # Document statistics (word count, avg word len, etc.)
│           └── tokenizer.py    # Simple tokenizer utilities
├── configs/
│   ├── default.yaml         # Default pipeline config
│   ├── pretrain_zh.yaml     # Chinese pre-training data config
│   └── sft.yaml             # SFT data config
├── tests/
│   ├── test_gopher.py
│   ├── test_c4.py
│   ├── test_dedup.py
│   ├── test_pipeline.py
│   └── fixtures/            # Small test data files
│       ├── sample_good.jsonl
│       ├── sample_bad.jsonl
│       └── sample_sft.jsonl
└── README.md
```

## Phase 1 Implementation (MUST complete)

### 1. Project Setup
- Use `uv` for project management
- pyproject.toml with dependencies: click, pyyaml, datasketch, xxhash, regex, tqdm, rich
- Optional deps group: `[model]` for fasttext, transformers, sentence-transformers

### 2. Core Pipeline Framework
- `BaseFilter` ABC with `filter(doc: dict) -> tuple[bool, dict]` (keep/drop + reason)
- `Pipeline` class that chains filters, tracks stats per filter
- YAML config to define which filters to run with what params
- Support for jsonl/parquet/csv input/output

### 3. Gopher Quality Filters (gopher.py)
Implement ALL rules from the Gopher paper:
- Word count: 50 < words < 100,000
- Avg word length: 3 < avg_word_len < 10
- Symbol/word ratio: < 0.1
- Lines ending with punctuation: > 0.1
- Stopword count: >= 2
- Alpha ratio: > 0.8

### 4. Gopher Repetition Filters (gopher.py)
- Top 2-gram ratio < 0.20
- Top 3-gram ratio < 0.18
- Top 4-gram ratio < 0.16
- Duplicate line ratio < 0.30
- Duplicate paragraph ratio < 0.30
- Character-level repetition < 0.20

### 5. C4 Filters (c4.py)
- Remove lines without terminal punctuation
- Remove lines with "javascript"
- Remove lines with "terms of use" / "cookie policy"
- Remove docs with "lorem ipsum"
- Remove docs with "{" (configurable, aggressive)
- Remove docs with < 3 sentences

### 6. FineWeb Custom Filters (fineweb.py)
- List document detection (>90% lines start with bullet/number)
- Duplicate line document (>30% identical lines)
- Bad line break detection (avg line length < 30 chars and > 10 lines)

### 7. PII Detection (pii.py)
- Email regex → redact to email@example.com
- Public IP regex → redact to 0.0.0.0
- Chinese phone: 1[3-9]\d{9}
- Chinese ID card: full regex
- Bank card: \d{16,19}
- Mode: detect (report only) or redact (replace)

### 8. Exact Dedup (exact.py)
- SHA256 hash of normalized text
- Track duplicate count per hash

### 9. MinHash LSH Dedup (minhash.py)
- Use datasketch library
- FineWeb params: 5-gram shingling, 112 hash functions, 14 bands × 8 rows
- Jaccard threshold ~0.75
- Output: cluster IDs, keep one per cluster

### 10. Report Generation (report.py)
- Per-filter stats: docs_in, docs_out, docs_dropped, drop_rate
- Overall pipeline stats
- Sample dropped documents with reasons
- Output: JSON + Markdown summary

### 11. CLI (cli.py)
Commands:
- `dq run <input> -c <config.yaml> -o <output>` — Run full pipeline
- `dq stats <input>` — Show dataset statistics without filtering  
- `dq report <input> -c <config.yaml>` — Dry-run, generate report only
- `dq dedup <input> -o <output>` — Dedup only

### 12. Tests
- Test each filter with known good/bad samples
- Test pipeline end-to-end
- Test IO for jsonl/parquet/csv
- Create fixture files with edge cases

## Phase 2 (Stub out, don't implement fully)

Create stub classes with docstrings + `raise NotImplementedError`:
- fasttext_quality.py — fastText quality classifier
- perplexity.py — Perplexity filter
- llm_scorer.py — LLM scoring
- complexity.py / quality.py / diversity.py — DEITA SFT evaluation
- ngram.py / min_k_prob.py / ts_guessing.py — Contamination detection
- semantic.py — SemDeDup

## Key Design Decisions

1. **Chinese support**: All text processing must handle Chinese correctly (no space-based word splitting for Chinese — use character-level or jieba)
2. **Streaming**: Process documents one-at-a-time for filters (no need to load full dataset). Dedup requires two passes.
3. **Configurable thresholds**: Every threshold should be configurable via YAML, with sensible defaults from the papers
4. **Progress bars**: Use tqdm/rich for progress indication
5. **Parallel processing**: Use multiprocessing for CPU-bound filters (optional, use `--workers` flag)

## Example Usage

```bash
# Install
uv sync

# Run default pipeline on a jsonl file
uv run dq run data/train.jsonl -o data/train_clean.jsonl

# Run with custom config
uv run dq run data/train.jsonl -c configs/pretrain_zh.yaml -o data/train_clean.jsonl

# Stats only
uv run dq stats data/train.jsonl

# Dedup only
uv run dq dedup data/train.jsonl -o data/train_deduped.jsonl

# Dry-run report
uv run dq report data/train.jsonl -c configs/default.yaml
```

## Example Config (default.yaml)

```yaml
pipeline:
  text_field: "text"  # field name in jsonl containing the text
  
  filters:
    - name: length
      params:
        min_words: 50
        max_words: 100000
    
    - name: gopher_quality
      params:
        min_avg_word_len: 3
        max_avg_word_len: 10
        max_symbol_ratio: 0.1
        min_lines_end_punct: 0.1
        min_stopwords: 2
        min_alpha_ratio: 0.8
    
    - name: gopher_repetition
      params:
        max_top_2gram: 0.20
        max_top_3gram: 0.18
        max_top_4gram: 0.16
        max_dup_line_ratio: 0.30
        max_dup_para_ratio: 0.30
        max_char_repetition: 0.20
    
    - name: c4
      params:
        remove_no_terminal_punct: true
        remove_javascript_lines: true
        remove_policy_lines: true
        remove_lorem_ipsum: true
        remove_curly_brace: false  # aggressive, off by default
        min_sentences: 3
    
    - name: fineweb
      params:
        max_list_line_ratio: 0.90
        max_dup_line_ratio: 0.30
        min_avg_line_len: 30
        max_short_line_count: 10
    
    - name: pii
      mode: redact  # detect | redact
  
  dedup:
    exact: true
    minhash:
      enabled: true
      num_perm: 112
      bands: 14
      rows: 8
      ngram_size: 5
```

## Important Notes
- Use `uv` for everything (uv init, uv add, uv run)
- All code should have type hints
- Docstrings on all public classes/methods
- Run tests with `uv run pytest`
- Git commit after each major milestone
- The text field in JSONL defaults to "text" but is configurable
- Handle both English AND Chinese text properly

When completely finished, run this command to notify me:
openclaw system event --text "Done: Built data-quality-agent with full Phase 1 pipeline (Gopher/C4/FineWeb filters, dedup, PII, CLI, tests)" --mode now
