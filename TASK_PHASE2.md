# Task: Implement Phase 2 — Model-Based Filtering & SFT Evaluation

## Context
Phase 1 is complete with heuristic filters (Gopher/C4/FineWeb), dedup, PII, CLI, and benchmarks. 
Phase 1 benchmark showed heuristic filters can't distinguish Alpaca original vs cleaned — **semantic quality differences require model-based approaches**.

Your job: implement Phase 2 in the existing codebase at `/home/pzheng46/data-quality-agent`.

Read the existing code first: `src/dq/` structure, `pipeline.py`, `config.py`, `filters/base.py`.

## Phase 2 Modules to Implement

### 1. fastText Quality Classifier (`src/dq/model_filters/fasttext_quality.py`)
**Paper**: DCLM (DataComp-LM)
**Method**: 
- Train a binary fastText classifier: positive = high-quality text (e.g., Wikipedia, OpenHermes), negative = random web text
- Score each document, keep those above threshold
- We won't train a model — instead, **use an existing pre-trained model or provide a training script**

**Implementation**:
- `FastTextQualityFilter(model_path: str, threshold: float = 0.5, label: str = "__label__hq")`
- Inherits from `BaseFilter`
- Downloads fastText model if not present (or user provides path)
- `filter(doc)` → returns (keep, {"score": float, "label": str})
- Also provide `train_fasttext_classifier(positive_file, negative_file, output_path)` utility function
- For benchmark: if no model available, skip gracefully with warning

### 2. Perplexity Filter (`src/dq/model_filters/perplexity.py`)
**Paper**: CCNet / Llama 3
**Method**: Use a small language model to compute perplexity. Very high perplexity = gibberish/low quality. Very low perplexity = repetitive/boilerplate.

**Implementation**:
- `PerplexityFilter(model_name: str = "Qwen/Qwen2-0.5B", max_perplexity: float = 1000, min_perplexity: float = 5, batch_size: int = 8, device: str = "auto")`
- Inherits from `BaseFilter`
- Uses HuggingFace transformers to load model
- Compute token-level perplexity with sliding window (stride = model max_length // 2)
- `filter(doc)` → returns (keep, {"perplexity": float})
- Handle GPU/CPU gracefully
- Cache model loading (singleton pattern)

### 3. LLM Quality Scorer (`src/dq/model_filters/llm_scorer.py`)
**Paper**: FineWeb-Edu
**Method**: Use a small trained classifier to predict educational/quality score (0-5).
- FineWeb-Edu approach: Llama-3-70B scores 460K samples → train Snowflake-arctic-embed + regression
- We implement the **inference** side: load a scoring model, predict scores

**Implementation**:
- `LLMQualityScorer(model_name: str = "HuggingFaceFW/fineweb-edu-classifier", threshold: float = 3.0, batch_size: int = 16, device: str = "auto")`
- Inherits from `BaseFilter`
- Load the actual FineWeb-Edu classifier from HuggingFace
- `filter(doc)` → returns (keep, {"edu_score": float})
- Score >= threshold → keep
- Also provide a standalone `score_documents(docs, model_name) -> list[float]` function

### 4. DEITA Complexity Scorer (`src/dq/sft/complexity.py`)
**Paper**: DEITA (Data-Efficient Instruction Tuning)
**Method**: Score instruction complexity using an LLM. More complex instructions → more valuable training data.

**Implementation**:
- `ComplexityScorer(api_url: str | None = None, api_key: str | None = None, model: str = "gpt-4o-mini", batch_size: int = 10)`
- NOT a filter (doesn't drop docs) — it's a **scorer** that adds a `complexity_score` field
- Uses OpenAI-compatible API to score instruction complexity (1-6 scale)
- Prompt from DEITA paper: "Score the complexity of the following instruction on a scale of 1 to 6..."
- Configurable: use local model via vLLM endpoint or OpenAI API
- Rate limiting + retry logic
- If no API configured, skip with warning

### 5. DEITA Quality Scorer (`src/dq/sft/quality.py`)
**Paper**: DEITA
**Method**: Score response quality given an instruction. Higher quality → better training signal.

**Implementation**:
- `QualityScorer(api_url, api_key, model, batch_size)`
- Similar to ComplexityScorer but evaluates response quality
- Prompt: "Score the quality of the response given the instruction on a scale of 1 to 6..."
- Adds `quality_score` field to doc
- Requires doc to have `instruction` and `output` fields (SFT format)

### 6. DEITA Diversity Filter (`src/dq/sft/diversity.py`)
**Paper**: DEITA Repr Filter
**Method**: Use embeddings to ensure diversity. Remove near-duplicate instructions by embedding clustering.

**Implementation**:
- `DiversityFilter(model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.95, batch_size: int = 32)`
- Compute embeddings for all documents
- Use cosine similarity to find near-duplicates
- Keep the highest-scoring doc from each cluster (based on complexity * quality)
- This is a **batch filter** (needs all docs at once, not streaming)
- Add a `BatchFilter` base class if needed

### 7. Integration

#### Register all new filters in `src/dq/model_filters/__init__.py` and `src/dq/sft/__init__.py`
- Use the same registry pattern as Phase 1 filters
- Model filters should be registered with a `model_` prefix or under a separate registry

#### Update Pipeline (`src/dq/pipeline.py`)
- Support optional model filters in the pipeline
- Model filters should be skippable (if dependencies not installed, warn and skip)
- Add `--model-filters` flag to CLI

#### Update configs
- Add model filter configs to `configs/default.yaml` (commented out by default)
- Add DEITA configs to `configs/sft.yaml`

#### Update CLI
- `dq run --with-model-filters` — enable model filters
- `dq score <input> -o <output>` — new command that only adds scores (doesn't filter)
- Update `dq bench` to optionally include model filters

### 8. Tests
- `tests/test_model_filters.py` — test with mocks (don't require actual models in CI)
- `tests/test_sft_scorers.py` — test DEITA scorers with mock API
- Mock the model loading / API calls
- Test that filters gracefully handle missing dependencies

### 9. After implementation
- Run `uv run pytest -q` to verify all tests pass
- Run the benchmark with model filters if GPU/API available
- Git commit with descriptive message
- Update README.md with Phase 2 usage

## Dependencies
Add to `[project.optional-dependencies]` model group:
```
fasttext-wheel>=0.9
transformers>=4.30
sentence-transformers>=2.2
torch>=2.0
openai>=1.0  # for DEITA scorers
```

## Key Design Principles
1. **Graceful degradation**: If a model/API isn't available, warn and skip — never crash
2. **Lazy loading**: Don't load models until first `filter()` call
3. **Batch support**: Model filters should support batching for efficiency
4. **Device auto-detection**: auto → cuda if available, else cpu
5. **Caching**: Cache model instances, don't reload per document

When completely finished, run: openclaw system event --text "Done: Phase 2 model filters implemented (fastText, perplexity, LLM scorer, DEITA)" --mode now
