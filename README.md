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

### `gopher_quality` — Basic Quality Heuristics

Source: Gopher (Rae et al., 2021)

| Rule | What It Checks | Default Threshold | Value |
|------|----------------|-------------------|-------|
| `min_words` | Document too short | 50 words | Word count |
| `max_words` | Document too long | 100,000 words | Word count |
| `min_avg_word_len` | Average word length too short | 3.0 chars | Avg word length |
| `max_avg_word_len` | Average word length too long | 10.0 chars | Avg word length |
| `symbol_ratio` | Too many symbol tokens (`#`, `...`, `…`) | 10% | Symbol-to-word ratio |
| `lines_end_punct` | Too few lines ending with `.!?。！？;；` | 10% | Fraction of lines with terminal punct |
| `stopwords` | Too few English stopwords (non-natural-language) | 2 | Stopword count |
| `alpha_ratio` | Too few alphabetic/CJK characters | 80% | Alpha char ratio |

### `gopher_repetition` — Repetition Detection

Source: Gopher (Rae et al., 2021)

| Rule | What It Checks | Default Threshold | Value |
|------|----------------|-------------------|-------|
| `top_2gram` | Most frequent word 2-gram covers too much text | 20% | Character coverage ratio |
| `top_3gram` | Most frequent word 3-gram covers too much text | 18% | Character coverage ratio |
| `top_4gram` | Most frequent word 4-gram covers too much text | 16% | Character coverage ratio |
| `dup_line_ratio` | Too many duplicate lines | 30% | Duplicate line fraction |
| `dup_para_ratio` | Too many duplicate paragraphs | 30% | Duplicate paragraph fraction |
| `dup_5gram_frac` | Text covered by duplicate 5-grams | 15% | Character fraction |
| `dup_6gram_frac` | Text covered by duplicate 6-grams | 14% | Character fraction |
| `dup_7gram_frac` | Text covered by duplicate 7-grams | 13% | Character fraction |
| `dup_8gram_frac` | Text covered by duplicate 8-grams | 12% | Character fraction |
| `dup_9gram_frac` | Text covered by duplicate 9-grams | 11% | Character fraction |
| `dup_10gram_frac` | Text covered by duplicate 10-grams | 10% | Character fraction |

### `c4` — Line-level Cleaning + Document Check

Source: C4 (Raffel et al., 2020)

C4 first removes problematic lines, then checks if the remaining document is valid.

**Line removal (not rejection — lines are cleaned from document):**
- Lines containing `javascript` (case-insensitive)
- Lines with policy/cookie language (`terms of use`, `privacy policy`, `cookie policy`, etc.)
- Lines without terminal punctuation (`.!?。！？;；`)

| Rule | What It Checks | Default | Value |
|------|----------------|---------|-------|
| `empty_after_line_filter` | Document becomes empty after line cleaning | N/A | Detail of removals: `javascript(N)`, `policy(N)`, `no_terminal_punct(N)` |
| `min_sentences` | Too few sentences after cleaning | 3 | Sentence count |
| `lorem_ipsum` | Contains "lorem ipsum" text | Enabled | Boolean |
| `curly_brace` | Contains curly braces `{` | Disabled | Boolean |

### `fineweb` — Web Document Quality

Source: FineWeb (Penedo et al., 2024). Aligned with datatrove's `FineWebQualityFilter`.

| Rule | What It Checks | Default Threshold | Value |
|------|----------------|-------------------|-------|
| `empty_doc` | Document has no content | N/A | 0 |
| `line_punct_ratio` | Too few lines end with terminal punctuation | 12% min | Line punct fraction |
| `short_line_ratio` | Too many short lines (≤ 30 chars) | 67% max | Short line fraction |
| `char_dup_ratio` | Too many duplicate lines by character coverage | 1% max | Char dup fraction |
| `list_ratio` | Too many newlines relative to words (list-like) | 0.3 max | Newline / word count |

### `pii` — Personal Identifiable Information

Default mode: `redact` (replaces PII with placeholders, does not reject documents).

| Rule | What It Detects | Replacement |
|------|-----------------|-------------|
| `email` | Email addresses | `email@example.com` |
| `ip` | Public IPv4 addresses (excludes private ranges) | `0.0.0.0` |
| `cn_phone` | Chinese phone numbers (1[3-9]XXXXXXXXX) | `1XXXXXXXXXX` |
| `cn_id` | Chinese ID card numbers (18 digits) | `XXXXXXXXXXXXXXXXXX` |
| `bank_card` | Bank card numbers (16-19 digits) | `XXXXXXXXXXXXXXXX` |

### `sft_rules` — SFT Data Quality

| Rule | What It Checks | Default Threshold | Value |
|------|----------------|-------------------|-------|
| `missing_sft_fields` | No instruction/output fields found | N/A | Document keys |
| `empty_output` | Output is empty or whitespace-only | N/A | 0 |
| `output_too_short` | Output too short relative to instruction (≥20 words) | 5 words (1 for closed-form tasks) | Output word count |
| `instruction_copy` | Output too similar to instruction (char 3-gram Jaccard) | 80% similarity | Similarity score |
| `ai_refusal` | Output starts with refusal pattern | Hard: always reject; Soft: reject if < 50 words | Matched pattern |
| `language_mismatch` | CJK ratio differs between instruction and output | 30% difference | CJK ratio difference |

**Closed-form task detection:** Instructions matching patterns like `classify`, `categorize`, `yes or no`, `true or false`, `extract`, `name the`, `in one word` etc. get a reduced `output_too_short` threshold (5 → 1 word).

**AI refusal distinction:**
- Hard refusal (always reject): "I cannot", "I'm sorry, but I cannot", "I'm not able to"
- Soft refusal (reject only if < 50 words): "As an AI language model", "As an AI assistant", "I apologize"

**SFT field auto-detection:** `instruction`/`output`, `prompt`/`response`, `question`/`answer`, `query`/`reply`, `human`/`assistant`, `conversations` (ShareGPT).

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
