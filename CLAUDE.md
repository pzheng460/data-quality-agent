# CLAUDE.md — dq (Training Data Quality Agent)

## What is this?
A Python CLI for detecting low-quality LLM training data. Single command `dq bench` covers all quality detection: dataset stats, Layer 1 rule-based filters, dedup detection, contamination check, and optional Layer 2 LLM judge.

## ⚠️ Mandatory Rules
1. **Every significant change must be committed and pushed** — `git add -A && git commit -m "..." && git push`
2. **Update both README.md, README_CN.md AND CLAUDE.md** when changing architecture, adding features, or modifying filters/configs
3. **Run `uv run pytest` before every commit** — never push broken tests
4. **Read the Permanent Memory section below** before starting work

## 🧠 Permanent Memory

### Architecture (current)
- **CLI**: Single command `dq bench` — accepts local files or HuggingFace dataset IDs
- **Layer 1: Rule-based Filters** — deterministic, free, millisecond-level
  - Pre-training: Gopher quality/repetition, C4, FineWeb, PII
  - SFT: `SFTRulesFilter` — empty_output, output_too_short (closed-form aware), instruction_copy, ai_refusal (hard/soft), language_mismatch, missing_sft_fields
  - Dedup detection: exact duplicate rate (SHA256)
- **Layer 2: Unified LLM Binary Judge** — `src/dq/judge.py` with data-driven rules (HIGH/LOW classification)
  - SFT rules: instruction_following, factuality, completeness, format_compliance, harmlessness
  - Pretrain rules: information_density, coherence, originality
  - **Adding New Rules**: append to RULES list in judge.py — no code changes needed
- **Contamination Detection** — n-gram overlap against benchmarks (mmlu, hellaswag, etc.), HF datasets, or local files

### Filter Registry
Filters self-register via `@register_filter("name")`. Auto-scan filter registration via `ensure_registered()` — no manual `__init__.py` imports needed. New filters MUST use this decorator.

### SFT Field Detection
`sft_rules.py` auto-detects SFT fields: instruction/output, prompt/response, question/answer, query/reply, human/assistant, conversations (ShareGPT). If no SFT fields found → rejects with `missing_sft_fields`.

`SFT_DETECT_FIELDS` single source of truth from `sft_rules.py` — imported by benchmark package. `filter()` delegates to `filter_detailed()` in sft_rules.

### Closed-form Task Awareness
`output_too_short` uses regex to detect classification/extraction/factoid QA instructions (per InsTag taxonomy). Closed-form tasks have min_output_words=1 instead of 5.

### AI Refusal Distinction
- **Hard refusal** ("I cannot", "I'm sorry but I cannot"): always reject
- **Soft refusal** ("As an AI assistant", "I apologize"): only reject if output < 50 words

### Shared LLM Client
`src/dq/llm_client.py` — singleton OpenAI-compatible client. Env vars: `DQ_API_BASE_URL`, `DQ_API_KEY`, `DQ_MODEL`. All LLM-using modules (judges) use this.

### Known Issues & Lessons Learned
- **`char_repetition_ratio` is a length proxy**: birthday paradox causes longer text → higher score. SFT should disable (threshold=1.0).
- **Heuristic filters don't work for SFT quality**: Gopher/C4/FineWeb designed for web data. SFT quality (hallucinations) requires Layer 2.
- **C4 `no_terminal_punct_lines` is line-cleaning, not rejection**: `filter_detailed()` must NOT report it as a failure.
- **SFT pipeline must NOT use pre-training heuristics**: `sft.yaml` only has `sft_rules` + `pii`.
- **Pre-training data gets 0% on SFT rules**: correct behavior — `missing_sft_fields` rejection.

### Benchmark Reference (Layer 1, 1000 samples each)
Pre-training: TinyStories 99.4%, OpenWebText 94.1%, FineWeb 93.6%, CC-News 91.3%, Wikipedia 89.8%, C4 86.6%, Wikitext-103 47.4%
SFT: Dolly 99.8%, Alpaca GPT-4 99.8%, No Robots 99.6%, GPT4All 99.7%, WizardLM 98.3%, OpenOrca 95.0%

## Quick Commands
```bash
uv sync                              # Install deps
uv sync --extra bench                # + HuggingFace datasets
uv run pytest                        # Run all tests
uv run dq bench data.jsonl -n 1000
uv run dq bench allenai/dolma3_mix-6T -n 1000
uv run dq bench data.jsonl --check-contamination mmlu,hellaswag
uv run dq bench data.jsonl --with-llm-scoring --llm-samples 50
```

## Key Files
```
src/dq/
├── cli.py                      # CLI — single `dq bench` command
├── filters/sft_rules.py        # SFT rules (Layer 1)
├── judge.py                    # Unified LLM Binary Judge (Layer 2)
├── llm_client.py               # Shared LLM client
├── benchmark/                  # Benchmark runner
│   ├── runner.py               #   run_benchmark, run_llm_scoring
│   ├── datasets.py             #   Dataset loading (local + HuggingFace)
│   ├── types.py                #   BenchmarkReport, DatasetResult, etc.
│   └── utils.py                #   detect_data_type, SFT_DETECT_FIELDS
├── benchmark_report.py         # Report output (Rich/Markdown/JSON)
├── contamination/ngram.py      # N-gram contamination detection
├── pipeline.py                 # Pipeline orchestrator + filter registry
└── config.py                   # YAML config loader
```

## Dependencies
- **Core**: click, pyyaml, datasketch, xxhash, regex, tqdm, rich, openai
- **Bench** (optional): datasets (HuggingFace)
