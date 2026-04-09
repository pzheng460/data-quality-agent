# CLAUDE.md — dq (Training Data Quality Agent)

## What is this?
A Python CLI for detecting and cleaning low-quality LLM training data. Two main commands: `dq bench` for quality analysis and `dq run` for production data cleaning with a web dashboard. Layer 1 rule-based filters are 100% aligned with datatrove's reference implementation.

## ⚠️ Mandatory Rules
1. **Every significant change must be committed and pushed** — `git add -A && git commit -m "..." && git push`
2. **Update both README.md, README_CN.md AND CLAUDE.md** when changing architecture, adding features, or modifying filters/configs
3. **Run `uv run pytest` before every commit** — never push broken tests
4. **Run `uv run python scripts/align_with_datatrove.py -n 1000`** after modifying any pre-training filter to verify 100% alignment
5. **Read the Permanent Memory section below** before starting work

## 🧠 Permanent Memory

### Architecture (current)
- **CLI**: `dq bench` for analysis, `dq run` for production cleaning — accepts local files, directories, or HuggingFace dataset IDs
- **Layer 1: Rule-based Filters** — deterministic, free, aligned with datatrove
  - Pre-training: Gopher quality/repetition, C4, FineWeb, Language ID (fasttext), Bad Words (LDNOOBW), PII (all aligned with datatrove)
  - Arxiv: `ArxivFilter` — clean-then-judge pattern: strips LaTeX residuals (\cite, \textbf, \ref, etc.) while preserving math ($...$, $$...$$), then rejects if residual fraction still too high or missing structure
  - SFT: `SFTRulesFilter` — empty_output, output_too_short (closed-form aware), instruction_copy, ai_refusal (hard/soft), language_mismatch, missing_sft_fields
  - Dedup detection: exact (SHA256) + MinHash LSH (near-duplicate)
- **Layer 2: Unified LLM Binary Judge** — `src/dq/judge.py` with data-driven rules (HIGH/LOW classification)
  - SFT rules: instruction_following, factuality, completeness, format_compliance, harmlessness
  - Pretrain rules: information_density, coherence, originality
  - **Adding New Rules**: append to RULES list in judge.py — no code changes needed
- **Contamination Detection** — n-gram overlap against benchmarks (mmlu, hellaswag, etc.), HF datasets, or local files
- **Multiprocessing** — benchmark runner uses `multiprocessing.Pool` with per-chunk evaluation. `OMP_NUM_THREADS=1` prevents spacy thread contention.
- **Rejected Doc Export** — `--save-rejected <path>` exports all filtered docs with full text, all triggered filter/rule/value/threshold to JSONL. Fields: `__dq_rejections` (list of rejections), `__dq_dataset` (source dataset name).
- **Production Runner** (`dq run`) — Multi-phase pipeline: parse → filter → dedup → contamination → package. Each phase writes to `stage{N}_{name}/{kept,rejected}/shard-*.jsonl.zst`. Supports `--resume` (skips completed phases via `_SUCCESS` markers), `--phase N` (run single phase), `--dry-run`.
- **Web Dashboard** — FastAPI backend (`src/dq/server/app.py`) + Vite/React frontend (`dashboard/`). Controls pipeline execution, shows real-time progress, and provides side-by-side original vs cleaned document comparison with KaTeX math rendering.
- **Shard I/O** — `ShardWriter` writes zstd-compressed JSONL with auto-rotation by size. `read_shards()` reads all shards from a directory. `write_manifest()` generates `manifest.json` with SHA256 checksums.

### Datatrove Alignment
Pre-training filters (Gopher, C4, FineWeb) are 100% aligned with datatrove's reference implementation (verified on 4000/4000 C4 samples). Key implementation details:
- **Word tokenization**: spacy via datatrove's `load_word_tokenizer` (NOT `str.split()`)
- **alpha_ratio**: word-level (frac of words with ≥1 alpha char), not char-level
- **symbol_word_ratio**: checks `#` and `...` separately via `text.count() / n_words`
- **word_count**: filters pure-punctuation tokens (non_symbol_words)
- **stopwords**: set intersection with 8-word Gopher set `{"the","be","to","of","and","that","have","with"}`
- **top_ngram_ratio**: char coverage / len(text), not frequency ratio
- **duplicate_line_ratio**: `\n+` split (merges empty lines), counts subsequent duplicates only
- **dup_line_char_frac / dup_para_char_frac**: sum of len(element) for duplicate elements / len(text)
- **dup_ngram_char_frac**: denominator is `len(text)`, not `sum(word_lens)`
- **C4 terminal punct**: `endswith(('.','?','!','"',"'"))` excluding `...`
- **C4 sentences**: spacy sentence splitter, not regex
- **FineWeb**: line_punct_ratio, short_line_ratio, char_dup_ratio, list_ratio (newline/word)
- **Alignment test**: `scripts/align_with_datatrove.py` — run after any filter changes

### Performance
- Stats functions accept pre-tokenized `words` kwarg to avoid redundant spacy calls
- `tokenize_words()` has `@lru_cache(8)` for cross-filter cache hits
- Benchmark runner uses `multiprocessing.Pool` with `spawn` context
- `OMP_NUM_THREADS=1` set before spawn to prevent CPU contention
- Dataset stats (word_count, avg_word_length) computed inside workers alongside filtering

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
- **spacy tokenizer is the performance bottleneck**: ~800x slower than `str.split()`. Mitigated by tokenize-once + LRU cache + multiprocessing.

## Quick Commands
```bash
uv sync                              # Install deps
uv sync --extra bench                # + HuggingFace datasets
uv run pytest                        # Run all tests
uv run dq bench data.jsonl -n 1000
uv run dq bench data.jsonl -n 1000 -w 16           # 16 parallel workers
uv run dq bench allenai/dolma3_mix-6T -n 1000
uv run dq bench data.jsonl --check-contamination mmlu,hellaswag
uv run dq bench data.jsonl --with-llm-scoring --llm-samples 50
uv run dq bench data.jsonl -n 1000 --save-rejected rejected.jsonl  # Export all rejected docs
uv run dq run INPUT -o OUTPUT -c configs/arxiv.yaml    # Production cleaning pipeline
uv run dq run INPUT -o OUTPUT -c configs/arxiv.yaml --phase 2  # Run single phase
uv run dq run INPUT -o OUTPUT -c configs/arxiv.yaml --resume   # Resume from checkpoint
uv run uvicorn dq.server.app:app --port 8001           # Start API backend
cd dashboard && npm run dev                             # Start web dashboard
uv run python scripts/align_with_datatrove.py -n 1000  # Verify datatrove alignment
```

## Key Files
```
src/dq/
├── cli.py                      # CLI — `dq bench` + `dq run`
├── filters/
│   ├── gopher.py               #   Gopher quality + repetition (datatrove-aligned)
│   ├── c4.py                   #   C4 filter (datatrove-aligned)
│   ├── fineweb.py              #   FineWeb filter (datatrove-aligned)
│   ├── arxiv.py                #   Arxiv clean-then-judge (LaTeX cleanup)
│   ├── language.py             #   Language ID (fasttext/glotlid)
│   ├── badwords.py             #   Bad words (LDNOOBW)
│   ├── sft_rules.py            #   SFT rules
│   └── pii.py                  #   PII detection/redaction
├── runner/                     # Production pipeline
│   ├── engine.py               #   PhaseEngine (phase orchestration + resume)
│   ├── phases.py               #   Phase 1-5 implementations
│   ├── shard.py                #   ShardWriter / read_shards (zstd JSONL)
│   └── stats.py                #   PhaseStats + overview
├── server/                     # Web dashboard backend
│   └── app.py                  #   FastAPI API
├── benchmark/                  # Benchmark runner
│   ├── runner.py               #   run_benchmark (multiprocessing)
│   ├── datasets.py             #   Dataset loading (local + HuggingFace)
│   └── types.py                #   BenchmarkReport, DatasetResult
├── contamination/ngram.py      # N-gram contamination detection
├── dedup/                      # exact (SHA256) + MinHash LSH
├── pipeline.py                 # Filter registry + Pipeline orchestrator
├── config.py                   # YAML config loader
├── judge.py                    # LLM Binary Judge (Layer 2)
└── utils/io.py                 # JSONL/Parquet/CSV/zstd read/write
configs/
├── default.yaml                # Pre-training config
├── arxiv.yaml                  # Arxiv cleaning config
└── sft.yaml                    # SFT config
scripts/
├── align_with_datatrove.py     # Alignment regression test
├── sample_arxiv.py             # Sampling (random/stratified/rule-hit)
└── build_dashboard.py          # Pipeline output → dashboard JSON
dashboard/                      # Vite + React + TypeScript
├── src/pages/                  #   PipelineControl, Overview, SampleBrowser, etc.
└── src/components/             #   Layout, shared UI
```

## Dependencies
- **Core**: click, pyyaml, datasketch, xxhash, regex, tqdm, rich, openai, spacy, datatrove, zstandard, fastapi, uvicorn
- **Bench** (optional): datasets (HuggingFace)
