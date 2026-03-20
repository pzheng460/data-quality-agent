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

# Run with model-based filters enabled (requires `uv sync --extra model`)
uv run dq run data/train.jsonl -c configs/default.yaml -o clean.jsonl --with-model-filters

# Score documents without filtering (adds _scores field)
uv run dq score data/train.jsonl -o scored.jsonl

# Check contamination against common benchmarks (requires `uv sync --extra bench`)
uv run dq contamination data/train.jsonl --benchmarks mmlu,hellaswag,arc

# Check against a custom benchmark file (fast, no extra deps)
uv run dq contamination data/train.jsonl --benchmark-file my_test_set.jsonl

# N-gram only (default, fast, no model needed)
uv run dq contamination data/train.jsonl --benchmark-file test.jsonl --method ngram

# Save contamination report
uv run dq contamination data/train.jsonl --benchmark-file test.jsonl -o contamination_report/

# Run benchmark with contamination check
uv run dq bench --check-contamination
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

### Phase 2: Model-Based Filters

| Filter | Description | Source |
|--------|-------------|--------|
| `fasttext_quality` | Binary quality classifier using trained fastText model | DCLM (Li et al., 2024) |
| `perplexity` | Small LM perplexity — drops gibberish (high) and boilerplate (low) | CCNet / Llama 3 |
| `llm_quality` | Educational quality scorer (0-5 scale) | FineWeb-Edu (Lozhkov et al., 2024) |

### Phase 2: SFT Scoring (DEITA)

| Module | Description | Source |
|--------|-------------|--------|
| `ComplexityScorer` | Instruction complexity scoring (1-6) via LLM API | DEITA (Liu et al., 2024) |
| `QualityScorer` | Response quality scoring (1-6) via LLM API | DEITA |
| `DiversityFilter` | Embedding-based near-duplicate removal (batch) | DEITA Repr Filter |

### Phase 3: Contamination Detection

| Method | Description | Source |
|--------|-------------|--------|
| `NgramContaminationDetector` | 13-gram overlap detection against benchmark test sets (fast, no model) | GPT-3 / Llama decontamination |
| `MinKProbDetector` | Min-K% Prob — flags memorized text via token log-prob analysis | Shi et al., 2023 |
| `TSGuessingDetector` | MCQ contamination — tests if model can guess answers from choices alone | Time Travel in LLMs |

### Not Yet Implemented

- Language detection (fastText)
- KL divergence filter
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

## Phase 2: Model-Based Filters

Model filters require optional dependencies. Install with:

```bash
uv sync --extra model
```

All model filters **gracefully skip** if dependencies aren't installed — they log a warning and pass all documents through.

### Python API

```python
# Model filters (auto-registered, same BaseFilter interface)
from dq.model_filters import FastTextQualityFilter, PerplexityFilter, LLMQualityScorer

# Use in pipeline via config:
# - name: perplexity
#   params: { model_name: "Qwen/Qwen2-0.5B", max_perplexity: 1000 }

# SFT scoring (not pipeline filters — standalone scorers)
from dq.sft import ComplexityScorer, QualityScorer, DiversityFilter

scorer = ComplexityScorer(model="gpt-4o-mini")
doc = scorer.score({"instruction": "Explain quantum computing"})
# doc["complexity_score"] = 4.0

# Batch diversity filtering
diversity = DiversityFilter(threshold=0.95)
filtered = diversity.filter_batch(docs)
```

## Phase 3: Contamination Detection

Detect whether training data overlaps with benchmark test sets. The primary method (N-gram) is fast and needs no model; secondary methods need optional deps.

```python
from dq.contamination import NgramContaminationDetector, load_benchmark

# Build detector and index
detector = NgramContaminationDetector(n=13, threshold=0.8)
benchmark_texts = load_benchmark("path/to/benchmark.jsonl")
detector.build_index(benchmark_texts, benchmark_name="my_benchmark")

# Check a single document
result = detector.check_contamination("some training document text...")
print(result.is_contaminated, result.overlap_ratio)

# Scan full dataset against multiple benchmarks
report = detector.scan_dataset(
    docs=[{"text": "..."}],
    benchmarks={"mmlu": mmlu_texts, "hellaswag": hellaswag_texts},
    dataset_name="my_dataset",
)
report.print_rich()       # Rich table output
report.to_json("out.json")
report.to_markdown("out.md")

# Min-K% Prob (requires transformers + torch)
from dq.contamination import MinKProbDetector
mk = MinKProbDetector(model_name="Qwen/Qwen2-0.5B", k_percent=20.0)
result = mk.check_contamination("possibly memorized text")

# TS-Guessing for MCQ (requires openai)
from dq.contamination import TSGuessingDetector
ts = TSGuessingDetector(model="gpt-4o-mini")
report = ts.scan_mcq_dataset([
    {"question": "Q", "choices": ["A", "B", "C", "D"], "correct_idx": 1},
])
```

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
├── model_filters/   # Model-based filters (fastText, perplexity, LLM scorer)
├── sft/             # SFT scoring (DEITA complexity, quality, diversity)
├── contamination/   # Contamination detection (n-gram, Min-K%, TS-Guessing)
└── utils/           # IO, stats, tokenizer utilities
```

## License

MIT
