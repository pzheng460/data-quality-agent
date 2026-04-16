# dq — Training Data Quality Agent

A Python CLI + web dashboard for detecting and cleaning low-quality LLM training data, with a focus on arXiv scientific papers.

## Architecture

### 4-Stage Pipeline

```
Stage 1: Ingestion    → Fetch raw data (LaTeX, HTML, JSONL)
Stage 2: Extraction   → Convert to clean text (LaTeXML + pre-processing)
Stage 3: Curation     → Filter + dedup + contamination check
Stage 4: Packaging    → Sort, shard, manifest
```

### LaTeX Extraction Pipeline (Industry-leading)

```
raw .tex
   ↓
preprocess.py     ← Extract algorithm/tikz/align*/mdframed, replace with placeholders
   ↓              ← Expand user-defined macros (\bx → \mathbf{x})
cleaned .tex
   ↓
LaTeXML → HTML    ← Only processes content it handles well
   ↓
html_to_text      ← Table extraction (multirow/colspan/span-based)
   ↓              ← Math cleanup (alttext → KaTeX-compatible)
   ↓              ← Algorithm pseudocode (algorithm2e + algorithmic)
katex_compat      ← Fix non-KaTeX commands (\mathbbm → \mathbb, etc.)
   ↓
restore placeholders
   ↓
clean output      ← Ready for rendering or pre-training
```

### Quality Filters

| Filter | Source | What it checks |
|--------|--------|----------------|
| `gopher_quality` | Gopher (2021) | Word count, avg length, symbol ratio, alpha ratio, stopwords |
| `gopher_repetition` | Gopher (2021) | Duplicate paragraphs/lines, top n-gram coverage |
| `c4` | C4 (2020) | Lorem ipsum, JS/policy lines, min sentences |
| `fineweb` | FineWeb (2024) | Line punctuation, short lines, char dedup, newline ratio |
| `arxiv` | Custom | LaTeX residual fraction, section count, text cleaning |
| `language` | fastText | Language detection |
| `badwords` | Custom | Profanity/NSFW word lists (28 languages) |
| `pii` | Custom | Email, phone, IP, ID card, bank card → redact |
| `sft_rules` | Custom | Empty output, instruction copy, AI refusal |

### Deduplication

- **Exact**: SHA256 hash
- **Near-duplicate**: MinHash LSH (configurable threshold)
- **Version**: Keep latest arXiv version only

### Contamination Detection

N-gram overlap against benchmarks: MMLU, GSM8K, HellaSwag, ARC, HumanEval, TruthfulQA.

## Installation

```bash
uv sync                          # Core
uv sync --extra bench            # + HuggingFace datasets
# System dependency for LaTeX sources:
# apt install latexml
```

## Usage

### CLI — Full Pipeline

```bash
# 4-stage pipeline on arXiv papers
dq run INPUT -o OUTPUT -c configs/arxiv.yaml

# Single stage
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --stage 2

# Resume from last checkpoint
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --resume
```

### CLI — Quality Benchmark

```bash
dq bench data.jsonl -n 1000
dq bench data.jsonl -n 1000 -w 16                    # Parallel
dq bench data.jsonl --with-llm-scoring --llm-samples 50
dq bench data.jsonl --check-contamination mmlu,hellaswag
```

### Web Dashboard

```bash
# Start backend
uv run uvicorn src.dq.server.app:app --port 8001 --host 0.0.0.0 --reload

# Start frontend
cd dashboard && npm run dev
```

5 pages: Pipeline Control, Stats, Samples Browser, Quality Benchmark, and integrated markdown/KaTeX rendering.

## Data Sources

| Source | Domain | Method | Format |
|--------|--------|--------|--------|
| `arxiv_latexml` | arXiv | LaTeXML conversion | LaTeX → text |
| `arxiv_ar5iv` | arXiv | ar5iv HTML fetch | HTML → text |
| `arxiv_hf_bulk` | arXiv | HuggingFace dataset | Pre-converted text |
| `local_file` | Any | Local JSONL/Parquet/CSV | text |

## Key Features

- **LaTeX Pre-processor**: Extracts algorithm/tikz/align* environments before LaTeXML, preventing garbage leaks
- **Algorithm Parsing**: Supports both `algorithm2e` and `algorithmic` packages with proper indentation
- **Macro Expansion**: Extracts `\newcommand`/`\def` from preamble and expands in all math regions
- **KaTeX Compatibility Layer**: Unified mapping of non-KaTeX commands to supported equivalents
- **Table Extraction**: Handles rowspan, colspan, multi-row headers, span-based tables, `\makecell`
- **Auto-benchmark**: Pipeline automatically benchmarks final output quality

## Configuration

See `configs/arxiv.yaml` for arXiv-specific settings. Key parameters:

```yaml
filters:
  - name: arxiv
    params: { max_latex_residual: 0.05, min_sections: 2 }
  - name: gopher_quality
    params: { min_alpha_ratio: 0.4 }  # Relaxed for academic text
  - name: c4
    params: { remove_no_terminal_punct: false }  # Headings don't end with punct

dedup:
  exact: true
  minhash: { enabled: true, num_perm: 112, bands: 14, rows: 8 }
```

## Web Dashboard

```bash
# Backend (port 8001) + Frontend (port 5173)
tmux new-session -d -s dq
tmux send-keys 'uv run uvicorn src.dq.server.app:app --port 8001 --host 0.0.0.0 --reload' Enter
tmux new-window
tmux send-keys 'cd dashboard && npm run dev' Enter
```

## Development

```bash
uv run pytest              # Run all tests (340+)
uv run pytest tests/test_extraction_golden.py -v  # Golden tests only
```

## Project Structure

```
src/dq/
  stages/
    ingestion/          → @register_source data sources
    extraction/         → @register_extractor format converters
      preprocess.py     →   LaTeX pre-processor (neutralize bad envs)
      algorithm.py      →   algorithm2e + algorithmic parser
      table.py          →   LaTeX table parser
      katex_compat.py   →   KaTeX compatibility layer
      html.py           →   HTML → text (BeautifulSoup)
      latex.py          →   LaTeXML pipeline orchestrator
    curation/
      filters/          → @register_filter quality filters
      dedup/            →   Exact (SHA256) + MinHash LSH
      contamination/    →   N-gram overlap detection
  runner/
    engine.py           → 4-stage orchestration with resume
    stages.py           → Stage implementations
  server/app.py         → FastAPI backend
  benchmark/            → Quality benchmark runner
dashboard/              → React + shadcn/ui frontend
configs/                → YAML configurations
tests/                  → 340+ tests including golden regression
```

## License

MIT
