# CLAUDE.md — dq (Training Data Quality Agent)

## What is this?
A Python CLI + web dashboard for detecting and cleaning low-quality LLM training data. Two main commands: `dq bench` for quality analysis and `dq run` for production data cleaning. Layer 1 rule-based filters are 100% aligned with datatrove's reference implementation.

## ⚠️ Mandatory Rules
1. **Run `uv run pytest` before every commit** — full test suite as gate, never push broken tests
2. **Every significant change must be committed and pushed** — `git add -A && git commit -m "..." && git push`
3. **Update CLAUDE.md, README.md, README_CN.md** when changing architecture, adding features, or modifying filters/configs
4. **Run `uv run python scripts/align_with_datatrove.py -n 1000`** after modifying any pre-training filter to verify 100% alignment
5. **CLI + Dashboard must stay in sync** — every new feature must work in both `dq` CLI commands and the web dashboard
6. **Read the Permanent Memory section below** before starting work

## 🧠 Permanent Memory

### Architecture: 4-Stage Pipeline
```
stages/
  ingestion/        → Fetch raw data (LaTeX, HTML, JSONL) via @register_source plugins
  extraction/       → Convert format to text (LaTeXML, HTML→text) via @register_extractor plugins
  curation/         → Data cleaning
    filters/        →   Heuristic filters (@register_filter: gopher, c4, fineweb, arxiv, pii, etc.)
    dedup/          →   Exact (SHA256) + MinHash LSH dedup
    contamination/  →   N-gram overlap against benchmarks
  packaging/        → Sort, shard, manifest
shared/
  shard.py          → ShardWriter / read_shards (zstd JSONL)
  stats.py          → PhaseStats + overview
runner/
  engine.py         → PhaseEngine: 4-stage orchestration with _SUCCESS resume markers
  stages.py         → stage_ingest, stage_extract, stage_curate, stage_package
  phases.py         → Deprecated 6-phase compat shims
server/
  app.py            → FastAPI API for dashboard
benchmark/
  runner.py         → run_benchmark (multiprocessing)
```

### Plugin Registry Pattern (3 registries)
All plugins self-register via decorators with auto-discovery:
- **Sources**: `@register_source("name")` in `stages/ingestion/` — `ensure_sources_registered()`
- **Extractors**: `@register_extractor("name")` in `stages/extraction/` — `ensure_extractors_registered()`
- **Filters**: `@register_filter("name")` in `stages/curation/filters/` — `ensure_registered()`

Adding a new source/extractor/filter = one file + decorator. No API or UI changes needed.

### Data Sources (stages/ingestion/)
Sources declare `domain`, `priority`, `output_format`, and `params_schema()`:

| Source | Domain | Priority | Output Format | Description |
|--------|--------|----------|---------------|-------------|
| `arxiv_hf_bulk` | arxiv | 100 | text | HF dataset (marin-community/ar5iv-no-problem-markdown) |
| `arxiv_ar5iv` | arxiv | 200 | html | Fetch HTML from ar5iv.labs.arxiv.org |
| `arxiv_latexml` | arxiv | 300 | latex | Download LaTeX source, local LaTeXML conversion |
| `local_file` | local | 100 | text | Read local JSONL/Parquet/CSV |

### Extractors (stages/extraction/)
| Extractor | Input Format | Method |
|-----------|-------------|--------|
| `passthrough` | text | No-op (data already clean) |
| `html` | html | BeautifulSoup: math→$LaTeX$, tables→pipe, headings→## |
| `latex` | latex | LaTeXML subprocess → HTML → html extractor |

### Ingest vs Extract vs Filter boundary
- **Ingest**: ONLY fetches raw data. No format conversion. No cleaning.
- **Extract**: ONLY converts format (HTML→text, LaTeX→text). No quality decisions.
- **Filter**: ALL cleaning decisions (citations, residual LaTeX, section numbering, quality checks).

### Unified API
- `GET /api/sources` → available sources grouped by domain
- `POST /api/ingest` → `{source, params, output_path, limit}`
- `POST /api/run` → run full 4-stage pipeline
- `POST /api/bench` → run benchmark (background, poll `/api/bench/status`)
- `GET /api/bench/status` → benchmark progress and results

### Web Dashboard
5 pages: Pipeline Control, Overview, Phase Details, Sample Browser, Quality Benchmark.
Frontend dynamically renders data source selector from `/api/sources` (domain tabs → source picker → param form).

### CLI Commands
```bash
dq bench INPUT -n 1000                           # Quality analysis
dq bench INPUT -n 1000 -w 16                     # Parallel
dq bench INPUT --with-llm-scoring --llm-samples 50
dq bench INPUT --check-contamination mmlu,hellaswag
dq bench INPUT --save-rejected rejected.jsonl

dq run INPUT -o OUTPUT -c configs/arxiv.yaml     # 4-stage pipeline
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --stage 3   # Single stage
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --resume
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --dry-run
```

### Datatrove Alignment
Pre-training filters (Gopher, C4, FineWeb) are 100% aligned with datatrove (verified 4000/4000 C4 samples). Key details:
- **Word tokenization**: spacy via datatrove's `load_word_tokenizer`
- **Alignment test**: `scripts/align_with_datatrove.py` — run after any filter changes

### Layer 2: LLM Judge
`src/dq/judge.py` — binary HIGH/LOW classification with data-driven rules.
- SFT rules: instruction_following, factuality, completeness, format_compliance, harmlessness
- Pretrain rules: information_density, coherence, originality
- Shared client: `src/dq/llm_client.py` (env: `DQ_API_BASE_URL`, `DQ_API_KEY`, `DQ_MODEL`)

### Known Design Decisions
- **spacy is intentionally slow** — aligned with datatrove. Mitigated by LRU cache + multiprocessing.
- **ArxivFilter does ALL cleaning** — ingest/extraction only do format conversion.
- **Curation stage consolidates** old phases 2-5 (filter + quality_score + dedup + contamination) into one logical stage with sub-steps.
- **`--phase` CLI flag is deprecated** — use `--stage` (1=ingest, 2=extract, 3=curate, 4=package).

## Key Files
```
src/dq/
├── cli.py                              # CLI entry points
├── stages/
│   ├── ingestion/                      # @register_source data sources
│   │   ├── arxiv_source.py             #   LaTeX download (raw)
│   │   ├── arxiv_ar5iv.py              #   ar5iv HTML fetch
│   │   ├── arxiv_hf_bulk.py            #   HF dataset streaming
│   │   └── local_file.py               #   Local JSONL/Parquet
│   ├── extraction/                     # @register_extractor format converters
│   │   ├── latex.py                    #   LaTeXML pipeline
│   │   ├── html.py                     #   BeautifulSoup HTML→text
│   │   └── passthrough.py              #   No-op for clean text
│   └── curation/
│       ├── filters/                    # @register_filter quality filters
│       │   ├── gopher.py, c4.py, fineweb.py  # datatrove-aligned
│       │   ├── arxiv.py                #   LaTeX residual cleaning
│       │   ├── language.py, badwords.py, pii.py, sft_rules.py
│       ├── dedup/                      # exact + MinHash
│       └── contamination/              # n-gram detection
├── shared/
│   ├── shard.py                        # Shard I/O (zstd JSONL)
│   └── stats.py                        # PhaseStats
├── runner/
│   ├── engine.py                       # PhaseEngine (4-stage)
│   └── stages.py                       # Stage implementations
├── server/app.py                       # FastAPI backend
├── benchmark/                          # dq bench
├── pipeline.py                         # Filter registry
├── config.py                           # YAML config
└── judge.py                            # LLM judge
dashboard/src/pages/                    # React frontend
  PipelineControl.tsx                   #   Ingest + pipeline control
  QualityCheck.tsx                      #   Benchmark UI
  Overview.tsx, PhaseDetails.tsx, SampleBrowser.tsx
configs/
  arxiv.yaml, default.yaml, sft.yaml
```

## Dependencies
- **Core**: click, pyyaml, datasketch, xxhash, regex, tqdm, rich, openai, spacy, datatrove, zstandard, fastapi, uvicorn, beautifulsoup4, lxml
- **Bench** (optional): datasets (HuggingFace)
- **System**: latexml (for arxiv_latexml source)
