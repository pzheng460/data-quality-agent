# dq — Training Data Quality Agent

A Python CLI tool and library for detecting and filtering low-quality training data for LLM pre-training and SFT fine-tuning. Implements validated methods from Gopher, C4, FineWeb, DCLM, and DEITA papers.

## Installation

```bash
uv sync
```

For model-based filters (Phase 2):
```bash
uv sync --extra model
```

## Quick Start

```bash
# Run default pipeline on a JSONL file
uv run dq run data/train.jsonl -o data/train_clean.jsonl

# Run with custom config
uv run dq run data/train.jsonl -c configs/pretrain_zh.yaml -o data/train_clean.jsonl

# Show dataset statistics
uv run dq stats data/train.jsonl

# Dry-run report (no output file)
uv run dq report data/train.jsonl -c configs/default.yaml

# Dedup only
uv run dq dedup data/train.jsonl -o data/train_deduped.jsonl
```

## Pipeline Filters

### Phase 1 (Implemented)

| Filter | Description | Source |
|--------|-------------|--------|
| `length` | Min/max word count | — |
| `gopher_quality` | Word count, avg word length, symbol ratio, punctuation, stopwords, alpha ratio | Gopher (Rae et al., 2021) |
| `gopher_repetition` | Top n-gram ratios, duplicate lines/paragraphs, char repetition | Gopher |
| `c4` | Terminal punctuation, javascript/policy removal, lorem ipsum, sentence count | C4 (Raffel et al., 2020) |
| `fineweb` | List document detection, duplicate lines, bad line breaks | FineWeb (Penedo et al., 2024) |
| `pii` | Email, IP, CN phone, CN ID card, bank card detection/redaction | — |

### Deduplication

| Method | Description |
|--------|-------------|
| Exact (SHA256) | Hash-based exact duplicate removal |
| MinHash LSH | Near-duplicate detection (5-gram, 112 perms, 14×8 bands, ~0.75 Jaccard) |

### Phase 2 (Stubbed)

- Language detection (fastText)
- KL divergence filter
- fastText quality classifier (DCLM)
- Perplexity filter
- LLM scorer (FineWeb-Edu)
- DEITA SFT evaluation (complexity, quality, diversity)
- Contamination detection (13-gram, Min-K% Prob, TS-Guessing)
- Semantic dedup (SemDeDup)

## Configuration

Pipeline behavior is controlled via YAML config files. See `configs/` for examples:

- `configs/default.yaml` — Standard English pre-training pipeline
- `configs/pretrain_zh.yaml` — Chinese pre-training (CJK-aware thresholds)
- `configs/sft.yaml` — SFT data (relaxed thresholds, code-friendly)

Every threshold is configurable. The `text_field` setting controls which JSON field contains the document text.

## Supported Formats

- JSONL (`.jsonl`) — default
- Parquet (`.parquet`)
- CSV (`.csv`)

## Chinese Text Support

All text processing handles Chinese correctly:
- CJK characters are treated as individual words for word counting
- Chinese punctuation (`。！？；`) is recognized as terminal punctuation
- PII patterns include Chinese phone numbers, ID cards, and bank cards
- The `pretrain_zh.yaml` config provides tuned thresholds

## Development

```bash
# Run tests
uv run pytest

# Run with verbose output
uv run pytest -v
```

## Project Structure

```
src/dq/
├── cli.py           # CLI entry point (click)
├── pipeline.py      # Pipeline orchestrator
├── config.py        # YAML-based config
├── report.py        # JSON + Markdown report generation
├── filters/         # Quality filters (gopher, c4, fineweb, pii, length)
├── dedup/           # Deduplication (exact, minhash, semantic*)
├── model_filters/   # Model-based filters (Phase 2 stubs)
├── sft/             # SFT evaluation (Phase 2 stubs)
├── contamination/   # Contamination detection (Phase 2 stubs)
└── utils/           # IO, stats, tokenizer utilities
```

## License

MIT
