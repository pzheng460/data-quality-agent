# dq — Training Data Quality Agent

A Python CLI tool and library for detecting and filtering low-quality training data for LLM pre-training and SFT fine-tuning.

## Architecture

**Two-layer quality evaluation:**

- **Layer 1: Rule-based Filters** — deterministic, free, millisecond-level
  - Pre-training rules: Gopher, C4, FineWeb heuristics + PII detection
  - SFT rules: empty output, output too short (with closed-form task awareness), instruction copy, AI refusal (hard/soft distinction), language mismatch
- **Layer 2: LLM Binary Judge** — semantic quality assessment via LLM API
  - SFT Judge: instruction_following, factuality, completeness, format_compliance, harmlessness
  - Pretrain Judge: information_density, coherence, originality

## Installation

```bash
uv sync
```

For model-based filters (optional):
```bash
uv sync --extra model
```

For benchmark datasets (optional):
```bash
uv sync --extra bench
```

## Quick Start

```bash
# Run default pipeline (pre-training data)
uv run dq run data/train.jsonl -o data/train_clean.jsonl

# Run SFT pipeline
uv run dq run data/sft.jsonl -c configs/sft.yaml -o data/sft_clean.jsonl

# Chinese pre-training
uv run dq run data/zh.jsonl -c configs/pretrain_zh.yaml -o data/zh_clean.jsonl

# Show dataset statistics
uv run dq stats data/train.jsonl

# Dry-run report (no output file)
uv run dq report data/train.jsonl

# Dedup only
uv run dq dedup data/train.jsonl -o data/train_deduped.jsonl

# Random sample
uv run dq sample data/huge.jsonl -n 5000 -o data/sample.jsonl

# Score documents without filtering
uv run dq score data/train.jsonl -o scored.jsonl

# Check contamination against common benchmarks
uv run dq contamination data/train.jsonl --benchmarks mmlu,hellaswag,arc

# Run benchmark
uv run dq bench
uv run dq bench --with-llm-scoring --llm-samples 50
```

## Layer 1: Rule-based Filters

### Pre-training Rules

| Filter | Description | Source |
|--------|-------------|--------|
| `length` | Min/max word count | — |
| `gopher_quality` | Word count, avg word length, symbol ratio, punctuation, stopwords, alpha ratio | Gopher (Rae et al., 2021) |
| `gopher_repetition` | Word-level duplicate n-gram fractions (5-10 gram), duplicate lines/paragraphs, top n-gram ratios | Gopher |
| `c4` | Line-level cleaning (javascript/policy/no-punct removal), then sentence count check | C4 (Raffel et al., 2020) |
| `fineweb` | List document detection, duplicate lines, bad line breaks | FineWeb (Penedo et al., 2024) |
| `pii` | Email, IP, CN phone, CN ID card, bank card detection/redaction | — |

### SFT Rules

| Rule | Description |
|------|-------------|
| `missing_sft_fields` | Rejects data without instruction/output structure |
| `empty_output` | Empty response |
| `output_too_short` | Response too short for instruction length (with closed-form task awareness per InsTag) |
| `instruction_copy` | Response copies the instruction |
| `ai_refusal` | Hard refusal ("I cannot") always rejects; soft refusal ("As an AI...") only if < 50 words |
| `language_mismatch` | Instruction and response in different languages |

SFT filter auto-detects common field names: `instruction`/`output`, `prompt`/`response`, `question`/`answer`, `conversations` (ShareGPT format).

### Deduplication

| Method | Description |
|--------|-------------|
| Exact (SHA256) | Hash-based exact duplicate removal |
| MinHash LSH | Near-duplicate detection (5-gram, 112 perms, 14×8 bands, ~0.75 Jaccard) |

## Layer 2: LLM Binary Judge

Uses rule-based binary classification (HIGH/LOW) instead of absolute 1-6 scoring. Each document is evaluated against specific rules via LLM API.

| Judge | Rules | Use Case |
|-------|-------|----------|
| SFT Judge | instruction_following, factuality, completeness, format_compliance, harmlessness | SFT data |
| Pretrain Judge | information_density, coherence, originality | Pre-training data |

```python
from dq.sft.llm_judge import SFTQualityJudge
from dq.model_filters.llm_quality_judge import PretrainingQualityJudge

# SFT
judge = SFTQualityJudge()
result = judge.judge_one("Explain quantum computing", "Quantum computing uses qubits...")
# {"quality": "high", "rules": {...}, "failed_rules": []}

# Pretrain
judge = PretrainingQualityJudge()
result = judge.judge_one("Article about physics...")
# {"quality": "low", "rules": {...}, "failed_rules": ["originality"]}
```

## Contamination Detection

| Method | Description | Source |
|--------|-------------|--------|
| N-gram | 13-gram overlap detection (fast, no model needed) | GPT-3 / Llama decontamination |
| Min-K% Prob | Token log-prob analysis for memorization detection | Shi et al., 2023 |
| TS-Guessing | MCQ contamination via choice-only guessing | Time Travel in LLMs |

## Model-Based Filters (Optional)

Require `uv sync --extra model`. Gracefully skip if dependencies missing.

| Filter | Description | Source |
|--------|-------------|--------|
| `fasttext_quality` | Binary quality classifier | DCLM (Li et al., 2024) |
| `perplexity` | Small LM perplexity filter | CCNet / Llama 3 |

## Configuration

YAML config files in `configs/`:

- `configs/default.yaml` — Standard English pre-training
- `configs/pretrain_zh.yaml` — Chinese pre-training (CJK-aware)
- `configs/sft.yaml` — SFT data (SFT rules + PII only, no pre-training heuristics)

## Benchmark Results

### Layer 1: Pre-training Rules (1000 samples each)

| Dataset | PASS Rate |
|---------|:---------:|
| TinyStories | 99.4% |
| OpenWebText | 94.1% |
| FineWeb | 93.6% |
| CC-News | 91.3% |
| Wikipedia | 89.8% |
| C4 | 86.6% |
| Wikitext-103 | 47.4% |

### Layer 1: SFT Rules (1000 samples each)

| Dataset | PASS Rate |
|---------|:---------:|
| Dolly | 99.8% |
| Alpaca GPT-4 | 99.8% |
| No Robots | 99.6% |
| Alpaca Orig | 99.6% |
| UltraChat | 99.3% |
| GPT4All | 99.7% |
| WizardLM | 98.3% |
| OpenOrca | 95.0% |

Pre-training data gets 0% on SFT rules (`missing_sft_fields`), confirming correct rejection.

## Supported Formats

- JSONL (`.jsonl`) — default
- Parquet (`.parquet`)
- CSV (`.csv`)

## Chinese Text Support

- CJK characters treated as individual words
- Chinese punctuation recognized (`。！？；`)
- PII: CN phone numbers, ID cards, bank cards
- Tuned thresholds in `pretrain_zh.yaml`

## Development

```bash
uv run pytest        # 283 tests
uv run pytest -v     # verbose
```

## Project Structure

```
src/dq/
├── cli.py                  # CLI entry point (click)
├── pipeline.py             # Pipeline orchestrator + filter registry
├── config.py               # YAML-based config
├── benchmark.py            # Multi-dataset benchmark runner
├── benchmark_report.py     # Rich/Markdown/JSON report output
├── llm_client.py           # Shared OpenAI-compatible LLM client
├── report.py               # JSON + Markdown report generation
├── filters/                # Layer 1: Rule-based filters
│   ├── gopher.py           #   Gopher quality + repetition
│   ├── c4.py               #   C4 filters
│   ├── fineweb.py          #   FineWeb filters
│   ├── sft_rules.py        #   SFT-specific rules
│   ├── pii.py              #   PII detection/redaction
│   └── length.py           #   Length filter
├── dedup/                  # Deduplication
│   ├── exact.py            #   SHA256 exact dedup
│   ├── minhash.py          #   MinHash LSH near-dedup
│   └── semantic.py         #   Semantic dedup (stub)
├── model_filters/          # Model-based filters
│   ├── llm_quality_judge.py  # Layer 2: Pretrain Binary Judge
│   ├── fasttext_quality.py   # FastText quality classifier
│   └── perplexity.py         # Perplexity filter
├── sft/                    # SFT scoring
│   ├── llm_judge.py        #   Layer 2: SFT Binary Judge
│   ├── diversity.py        #   Embedding diversity filter
│   ├── complexity.py       #   (deprecated) 1-6 complexity scorer
│   ├── quality.py          #   (deprecated) 1-6 quality scorer
│   ├── educational.py      #   (deprecated) educational value scorer
│   └── writing_quality.py  #   (deprecated) writing quality scorer
├── contamination/          # Contamination detection
│   ├── ngram.py            #   N-gram overlap
│   ├── min_k_prob.py       #   Min-K% Prob
│   └── ts_guessing.py      #   TS-Guessing for MCQ
└── utils/                  # Utilities
    ├── io.py               #   File I/O (JSONL/Parquet/CSV)
    ├── stats.py            #   Text statistics
    └── tokenizer.py        #   Tokenization
```

## References

- Gopher (Rae et al., 2021) — Pre-training quality/repetition heuristics
- C4 (Raffel et al., 2020) — Line-level cleaning + sentence filtering
- FineWeb (Penedo et al., 2024) — Web-scale dedup and filtering
- DEITA (Liu et al., 2024) — Automatic data selection for instruction tuning
- AlpaGasus (Chen et al., 2023) — LLM-based SFT data filtering
- InsTag (Lu et al., 2023) — Instruction tagging (open-ended vs closed-form)
- FineWeb-Edu (Lozhkov et al., 2024) — Educational quality classification

## License

MIT
