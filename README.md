# dq — Training Data Quality Agent

A Python CLI for detecting low-quality LLM training data. Single command `dq bench` covers dataset statistics, rule-based quality filters, dedup detection, contamination check, and LLM-based quality assessment.

## Architecture

- **Layer 1: Rule-based Filters** — deterministic, free, millisecond-level
  - Pre-training: Gopher quality/repetition, C4, FineWeb, PII
  - SFT: empty output, output too short, instruction copy, AI refusal, language mismatch
  - Dedup: exact duplicate detection (SHA256)
- **Layer 2: LLM Binary Judge** — semantic quality assessment via LLM API
  - SFT: instruction_following, factuality, completeness, format_compliance, harmlessness
  - Pretrain: information_density, coherence, originality
- **Contamination Detection** — n-gram overlap against benchmarks

## Installation

```bash
uv sync                    # Core
uv sync --extra bench      # + HuggingFace datasets
```

## Usage

```bash
# Local file
dq bench data.jsonl -n 1000

# HuggingFace dataset (streaming, no full download)
dq bench allenai/dolma3_mix-6T -n 10000

# With contamination check (built-in benchmarks)
dq bench data.jsonl --check-contamination mmlu,hellaswag

# With contamination check (all built-in benchmarks)
dq bench data.jsonl --check-contamination all

# With contamination check (HuggingFace dataset as benchmark)
dq bench data.jsonl --check-contamination cais/mmlu

# With contamination check (local file as benchmark)
dq bench data.jsonl --check-contamination /path/to/benchmark.jsonl

# With Layer 2 LLM judge
dq bench data.jsonl --with-llm-scoring --llm-samples 50
```

Reports are saved to `reports/` by default (JSON + Markdown). Override with `-o`.

## Layer 1: Rule-based Filters

### Pre-training Rules

| Filter | Description | Source |
|--------|-------------|--------|
| `gopher_quality` | Word count, avg word length, symbol ratio, punctuation, stopwords, alpha ratio | Gopher (Rae et al., 2021) |
| `gopher_repetition` | Duplicate n-gram fractions (5-10 gram), duplicate lines/paragraphs, top n-gram ratios | Gopher |
| `c4` | Line-level cleaning (javascript/policy/no-punct removal), sentence count check | C4 (Raffel et al., 2020) |
| `fineweb` | List document detection, duplicate lines, bad line breaks | FineWeb (Penedo et al., 2024) |
| `pii` | Email, IP, CN phone, CN ID card, bank card detection/redaction | — |

### SFT Rules

| Rule | Description |
|------|-------------|
| `missing_sft_fields` | Rejects data without instruction/output structure |
| `empty_output` | Empty response |
| `output_too_short` | Response too short (with closed-form task awareness per InsTag) |
| `instruction_copy` | Response copies the instruction |
| `ai_refusal` | Hard refusal always rejects; soft refusal only if < 50 words |
| `language_mismatch` | Instruction and response in different languages |

Auto-detects SFT field names: `instruction`/`output`, `prompt`/`response`, `question`/`answer`, `conversations` (ShareGPT).

## Layer 2: LLM Binary Judge

Data-driven binary classification (HIGH/LOW) via LLM API. Auto-detects data type and applies appropriate rules.

```python
from dq.judge import LLMJudge

judge = LLMJudge()
result = judge.judge_sft("Explain quantum computing", "Quantum computing uses qubits...")
# {"quality": "high", "rules": {...}, "failed_rules": []}
```

Requires env vars: `DQ_API_BASE_URL`, `DQ_API_KEY`, `DQ_MODEL`.

## Contamination Detection

N-gram overlap detection against benchmark datasets. Supports:
- Built-in benchmarks: mmlu, hellaswag, arc, truthfulqa, gsm8k, humaneval
- Any HuggingFace dataset ID (e.g. `cais/mmlu`)
- Local files (text/jsonl)

## Configuration

YAML config files in `configs/`:

- `configs/default.yaml` — English pre-training
- `configs/pretrain_zh.yaml` — Chinese pre-training (CJK-aware)
- `configs/sft.yaml` — SFT data (SFT rules + PII only)

## Benchmark Results (Layer 1, 1000 samples)

**Pre-training:**

| Dataset | Pass Rate |
|---------|:---------:|
| TinyStories | 99.4% |
| OpenWebText | 94.1% |
| FineWeb | 93.6% |
| CC-News | 91.3% |
| Wikipedia | 89.8% |
| C4 | 86.6% |
| Wikitext-103 | 47.4% |

**SFT:**

| Dataset | Pass Rate |
|---------|:---------:|
| Dolly | 99.8% |
| Alpaca GPT-4 | 99.8% |
| No Robots | 99.6% |
| GPT4All | 99.7% |
| WizardLM | 98.3% |
| OpenOrca | 95.0% |

## Supported Formats

JSONL (`.jsonl`), Parquet (`.parquet`), CSV (`.csv`)

## Development

```bash
uv run pytest          # Run all tests
```

## Project Structure

```
src/dq/
├── cli.py                  # CLI — single `dq bench` command
├── pipeline.py             # Pipeline orchestrator + filter registry
├── config.py               # YAML config loader
├── judge.py                # LLM Binary Judge (Layer 2)
├── llm_client.py           # Shared OpenAI-compatible client
├── benchmark/              # Benchmark runner
│   ├── runner.py           #   run_benchmark, run_llm_scoring
│   ├── datasets.py         #   Dataset loading (local + HuggingFace)
│   ├── types.py            #   BenchmarkReport, DatasetResult, etc.
│   └── utils.py            #   detect_data_type, SFT field extraction
├── benchmark_report.py     # Report output (Rich/Markdown/JSON)
├── filters/                # Layer 1: Rule-based filters
│   ├── gopher.py           #   Gopher quality + repetition
│   ├── c4.py               #   C4 filters
│   ├── fineweb.py          #   FineWeb filters
│   ├── sft_rules.py        #   SFT rules
│   └── pii.py              #   PII detection/redaction
├── dedup/                  # Deduplication (used in bench stats)
│   ├── exact.py            #   SHA256 exact dedup
│   └── minhash.py          #   MinHash LSH near-dedup
├── contamination/          # Contamination detection
│   ├── ngram.py            #   N-gram overlap (built-in + HF + local)
│   └── report.py           #   Reports
└── utils/                  # Utilities
    ├── io.py               #   File I/O
    ├── stats.py            #   Text statistics
    └── tokenizer.py        #   Tokenization
```

## References

- Gopher (Rae et al., 2021) — Pre-training quality/repetition heuristics
- C4 (Raffel et al., 2020) — Line-level cleaning + sentence filtering
- FineWeb (Penedo et al., 2024) — Web-scale dedup and filtering
- InsTag (Lu et al., 2023) — Instruction tagging (open-ended vs closed-form)
- AlpaGasus (Chen et al., 2023) — LLM-based SFT data filtering

## License

MIT
