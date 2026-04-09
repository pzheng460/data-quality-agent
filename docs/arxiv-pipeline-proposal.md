# Arxiv 预训练数据清洗方案 (基于 dq 框架)

## 文档信息

- **项目**: PanguProMoE 预训练数据 — Arxiv 子集清洗
- **负责人**: peizhen
- **预估工期**: 2-3 周
- **基础框架**: `dq` (本仓库)

---

## 一、项目目标

从 RedPajama-v2 arxiv 子集中清洗出高质量 Markdown 格式 JSONL,用于 PanguProMoE 预训练。

**核心要求**:
1. 保留公式（LaTeX 源码）和章节结构
2. 文档级去重（exact + MinHash）+ 评测集污染检测
3. 每篇论文的清洗决策可追溯（trace）
4. 静态可视化 Dashboard 展示清洗过程与结果
5. Golden set 回归测试
6. 版本化管理，支持多版本对比

---

## 一、现有能力盘点（dq 已有）

| 能力 | 模块 | 说明 |
|---|---|---|
| Filter 注册 & 链式执行 | `pipeline.py` + `@register_filter` | 新 filter 只需装饰器，无需改 `__init__.py` |
| Gopher 质量/重复度 | `filters/gopher.py` | 与 datatrove 100% 对齐 |
| C4 规则 | `filters/c4.py` | terminal_punct, javascript, lorem, min_sentences |
| FineWeb 规则 | `filters/fineweb.py` | line_punct, short_line, char_dup, newline_ratio |
| Language ID | `filters/language.py` | fasttext ft176 / glotlid |
| Bad Words | `filters/badwords.py` | LDNOOBW 词表 |
| PII 检测 | `filters/pii.py` | email, IP, phone, ID card, bank card |
| 精确去重 | `dedup/exact.py` | SHA256 |
| 近似去重 | `dedup/minhash.py` | MinHash LSH (5-gram, 112 perm, 14×8 bands) |
| 污染检测 | `contamination/ngram.py` | 13-gram overlap, 支持内置/HF/本地评测集 |
| LLM Judge | `judge.py` | 二分类质量打分 (Layer 2) |
| YAML 配置 | `config.py` | `PipelineConfig`, `FilterConfig`, `DedupConfig` |
| 并行处理 | `benchmark/runner.py` | `multiprocessing.Pool` + spawn, chunk 分发 |
| 被拒文档导出 | `cli.py` `--save-rejected` | JSONL, 含 `__dq_rejections` + `__dq_dataset` |
| 报告生成 | `benchmark_report.py` | Rich / Markdown / JSON |
| 数据加载 | `benchmark/datasets.py` + `utils/io.py` | JSONL/CSV/Parquet + HuggingFace streaming |

**结论**: dq 已覆盖质量过滤、去重、污染检测、并行处理、报告生成的核心链路。Arxiv 清洗需要在此基础上新增：`dq run` 生产命令、Arxiv 专属 filter、多阶段落盘、Dashboard 可视化。

---

## 二、需新增的能力

### 2.1 `dq run` 命令 — 生产清洗流水线

`dq bench` 是只读分析（dry-run 统计），不落盘。生产清洗需要新命令 `dq run`：

```
dq run INPUT_PATH -o OUTPUT_DIR -c configs/arxiv.yaml [options]
```

**核心区别**:

| | `dq bench` | `dq run` |
|---|---|---|
| 目的 | 质量分析 + 报告 | 实际清洗 + 落盘 |
| 输出 | Rich 表格 + JSON/MD 报告 | 分阶段 JSONL.zst shard |
| 去重 | 只统计不删 | 实际去重 |
| 文档 | 全部保留 | kept/ + rejected/ 分流 |
| 阶段 | 单次跑完 | Phase 1-5 可单独重跑 |

**实现方案**: 在 `src/dq/cli.py` 新增 `dq run` 子命令（此前已 revert 的功能重新设计），在 `src/dq/runner/` 下新增 production runner 模块。

### 2.2 Arxiv 专属 Filter

新增 `src/dq/filters/arxiv.py`，通过 `@register_filter("arxiv")` 注册：

```python
@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv-specific quality rules."""

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        failures = []
        text = doc.get(self.text_field, "")
        metadata = doc.get("metadata", {})

        # 1. LaTeX 残留率：LaTeX 命令占比 > threshold → reject
        latex_cmd_frac = _count_latex_commands(text) / max(len(text), 1)
        if latex_cmd_frac > self.params.get("max_latex_residual", 0.05):
            failures.append({"filter": "arxiv", "rule": "latex_residual",
                             "value": latex_cmd_frac, "threshold": 0.05})

        # 2. 结构完整性：缺少 title/abstract → reject
        if not _has_abstract(text):
            failures.append({"filter": "arxiv", "rule": "missing_abstract"})

        # 3. 版本去重：同一 arxiv_id 只保留最新 version
        #    (在 pipeline 层面处理，这里只打标)

        return (len(failures) == 0, failures)
```

### 2.3 新增的阶段式 Runner

区别于 `benchmark/runner.py` 的 bench-only 模式，production runner 支持多阶段落盘：

```
Phase 1: 解析 + Schema 转换  → stage1_parsed/{kept,rejected}/
Phase 2: 质量过滤             → stage2_filtered/{kept,rejected}/
Phase 3: 去重                 → stage3_dedup/{kept,rejected}/
Phase 4: 污染检测             → stage4_contamination/{kept,rejected}/
Phase 5: 最终打包             → stage5_final/shard-*.jsonl.zst
```

每个阶段复用 dq 已有模块：
- Phase 2 → `Pipeline` + filter chain（gopher, c4, fineweb, language, arxiv 等）
- Phase 3 → `ExactDedup` + `MinHashDedup`（从 `dedup/` 导入）
- Phase 4 → `NgramContaminationDetector`（从 `contamination/ngram.py` 导入）

### 2.4 Dashboard (Phase 6)

Vite + React + TypeScript SPA，读取 `public/data/*.json` 产物，build 后为纯静态文件可直接部署。

---

## 三、技术选型

| 维度 | 选择 | 理由 |
|---|---|---|
| 数据源 | RedPajama-v2 arxiv subset | 已解析好 LaTeX → Markdown |
| 下载 | `hf-mirror.com` + `huggingface_hub` | 内网标准方案 |
| 存储 | `jsonl.zst` | dq 已支持（`utils/io.py`） |
| 并行 | `multiprocessing.Pool` (spawn) | dq 已有成熟实现 |
| 去重 | `datasketch` MinHash LSH | dq 已集成 |
| 可视化 | Vite + React + TypeScript | 组件化，HMR 开发体验好，build 后纯静态 |
| 测试 | `pytest` | 与 dq 一致 |

---

## 四、数据 Schema

### 4.1 文档 Schema

复用 dq 的 `dict` 文档格式（与 `text_field` 配置对齐），增加 arxiv 专属字段：

```python
{
    # dq 标准字段
    "text": "# Title\n\n## Abstract\n...",       # Markdown 正文

    # arxiv 元数据
    "id": "arxiv_2301.00001",
    "source": "arxiv",
    "metadata": {
        "arxiv_id": "2301.00001",
        "version": "v2",
        "categories": ["cs.CL", "cs.AI"],
        "primary_category": "cs.CL",
        "created": "2023-01-01",
        "title": "...",
        "license": "...",
    },

    # 质量信号（由 filter 填充）
    "quality_signals": {
        "lang": "en",
        "lang_score": 0.98,
        "n_chars": 15000,
        "n_words": 3200,
        "frac_latex_commands": 0.03,
        # ... gopher/c4/fineweb 信号自动计算
    },

    # 结构检查
    "structural_checks": {
        "has_title": True,
        "has_abstract": True,
        "has_sections": True,
        "num_sections": 7,
        "num_equations_inline": 42,
        "num_equations_display": 12,
        "num_code_blocks": 2,
    },

    # 清洗追踪
    "trace": {
        "phase1_parse": {"status": "ok"},
        "phase2_filter": {"status": "ok", "triggered_rules": []},
        "phase3_dedup": {"status": "ok"},
        "phase4_contamination": {"status": "ok", "hits": []},
    },

    # Preview（用于 Dashboard）
    "__raw_preview_head": "...(前 500 字符)",
    "__raw_preview_tail": "...(后 500 字符)",
    "__parsed_preview_head": "...",
    "__parsed_preview_tail": "...",
}
```

### 4.2 被拒文档

与保留文档用相同 schema。被拒信息写入 `__dq_rejections`（与现有 `--save-rejected` 格式一致）：

```python
{
    "__dq_rejections": [
        {"filter": "arxiv", "rule": "latex_residual", "value": 0.12, "threshold": 0.05},
        {"filter": "gopher_quality", "rule": "min_stopwords", "value": 0.04, "threshold": 0.06},
    ]
}
```

---

## 五、目录结构

```
data-quality-agent/                     # 本仓库（代码）
├── src/dq/
│   ├── filters/
│   │   └── arxiv.py                    # [新增] Arxiv 专属 filter
│   ├── runner/                         # [新增] Production runner
│   │   ├── __init__.py
│   │   ├── engine.py                   #   多阶段 pipeline 引擎
│   │   ├── phases.py                   #   Phase 1-5 实现
│   │   └── shard.py                    #   Shard 切分 + manifest 生成
│   ├── cli.py                          # [修改] 新增 `dq run` 子命令
│   └── ...                             # 现有模块不动
├── configs/
│   ├── default.yaml                    # 现有
│   └── arxiv.yaml                      # [新增] Arxiv 清洗配置
├── scripts/
│   ├── align_with_datatrove.py         # 现有
│   ├── sample.py                       # [新增] 抽样脚本
│   └── build_dashboard.py             # [新增] Dashboard 数据生成
├── dashboard/                          # [新增] Vite + React 项目
│   ├── package.json
│   ├── vite.config.ts
│   ├── public/data/                    #   build_dashboard.py 输出的 JSON
│   └── src/
│       ├── App.tsx
│       ├── pages/                      #   7 个页面组件
│       └── components/                 #   可复用 UI 组件
├── golden/                             # [新增] Golden set 回归测试
│   ├── 2301.00001/
│   │   ├── input.jsonl
│   │   ├── expected.json
│   │   └── expected_phase1.json
│   └── ...
└── tests/
    ├── test_arxiv_filter.py            # [新增]
    └── test_runner.py                  # [新增]

/data/arxiv/                            # 数据目录（不在仓库内）
├── raw/                                # RedPajama 原始下载
├── stage1_parsed/{kept,rejected}/
├── stage2_filtered/{kept,rejected}/
├── stage3_dedup/{kept,rejected}/
├── stage4_contamination/{kept,rejected}/
├── stage5_final/
│   ├── shard-00000-of-00256.jsonl.zst
│   ├── manifest.json
│   └── dataset_card.md
├── stats/v2025-04/
├── samples/v2025-04/
└── dashboard/v2025-04/                 # build 产物 (dist/)
```

---

## 六、配置文件 (`configs/arxiv.yaml`)

直接复用 dq 的 `PipelineConfig` schema，无需新增配置类：

```yaml
pipeline:
  text_field: "text"

  filters:
    # ── Arxiv 专属 ──
    - name: arxiv
      params:
        max_latex_residual_frac: 0.05    # LaTeX 命令残留率上限
        require_abstract: true
        require_title: true

    # ── 复用现有 filter（参数按 arxiv 特性微调） ──
    - name: gopher_quality
      params:
        min_words: 200                   # 论文通常 >200 词
        max_words: 500000
        min_avg_word_len: 3
        max_avg_word_len: 10
        max_symbol_ratio: 0.1
        max_ellipsis_lines_ratio: 0.3
        min_stopwords: 2                 # 学术文本 stopword 占比偏低，放宽
        min_alpha_ratio: 0.7             # 含公式，放宽

    - name: gopher_repetition
      params:
        dup_line_frac: 0.30
        dup_para_frac: 0.30
        top_2gram_frac: 0.20
        top_3gram_frac: 0.18
        top_4gram_frac: 0.16
        dup_5gram_frac: 0.15
        dup_10gram_frac: 0.10

    - name: c4
      params:
        min_sentences: 3                 # 论文可能有短 section

    - name: language
      enabled: true
      params:
        languages: ["en"]
        language_threshold: 0.5
        backend: "ft176"

    - name: badwords
      enabled: true
      params:
        default_language: "en"

    - name: pii
      mode: redact

  dedup:
    exact: true
    minhash:
      enabled: true
      num_perm: 112
      bands: 14
      rows: 8
      ngram_size: 5

# ── Arxiv 专属配置（扩展字段，由 runner 读取） ──
arxiv:
  version: v2025-04
  data_source: "togethercomputer/RedPajama-v2"
  hf_endpoint: "https://hf-mirror.com"
  subset: "arxiv"

  phase1:
    min_text_length: 200
    max_latex_residual: 0.20       # 解析失败阈值
    preview_length: 500

  phase3:
    version_dedup: true             # 同一 arxiv_id 只保留最新 version
    # minhash 参数从 pipeline.dedup.minhash 继承

  phase4:
    ngram_size: 13
    threshold: 0.8
    benchmarks:
      - MMLU
      - GSM8K
      - MATH
      - ARC-C
      - HellaSwag
      - HumanEval

  phase5:
    shard_target_bytes: 1073741824  # 1GB
    num_shards_power_of_two: true
    sort_by: "id"

  parallelism:
    num_workers: null               # null = auto (cpu_count // 4)
    chunk_size: 1000
```

---

## 四、清洗流水线设计

### Phase 0: 准备

| 任务 | 说明 |
|---|---|
| 目录结构 | 创建 `/data/arxiv/{raw,stage1,...,stage5,stats,samples,dashboards,golden,logs}` |
| 配置 | 编写 `configs/arxiv.yaml` |
| 新 filter | 实现 `src/dq/filters/arxiv.py`（`@register_filter("arxiv")`） |
| Schema | 用 dq 标准 dict 格式，无需 pydantic（与框架一致） |
| Golden set | 准备 10 篇论文（5 cs + 3 math + 2 physics），手工标注期望输出 |

**产出**: `configs/arxiv.yaml`, `src/dq/filters/arxiv.py`, `golden/` 目录

---

### Phase 1: 下载 + 解析 + Schema 映射

**输入**: RedPajama-v2 arxiv subset (via `huggingface_hub`)
**输出**: `stage1_parsed/{kept,rejected}/shard-*.jsonl.zst`

**复用**:
- `src/dq/benchmark/datasets.py` 中的 HuggingFace streaming 加载逻辑
- `src/dq/utils/io.py` 中的 JSONL 读写

**新增**:
- `src/dq/runner/phases.py::phase1_parse()` — RedPajama schema → dq schema 转换
  - 填充 `metadata.*`, `structural_checks.*`, `__raw_preview_*`, `__parsed_preview_*`
  - LaTeX 残留率 > 20% → reject (`trace.phase1.status = "rejected"`)
  - text 为空 → reject

**统计产出**: `stats/v2025-04/phase1_stats.json`
```json
{
  "total_raw": 2000000,
  "parsed_ok": 1950000,
  "rejected": 50000,
  "reject_reasons": {
    "empty_text": 10000,
    "latex_residual_high": 30000,
    "parse_error": 10000
  },
  "category_distribution": {"cs.LG": 120000, "math.AG": 50000, ...}
}
```

**校验**: Golden test Phase 1 通过，随机抽 50 篇人工 review。

---

### Phase 2: 质量过滤

**输入**: `stage1_parsed/kept/`
**输出**: `stage2_filtered/{kept,rejected}/shard-*.jsonl.zst`

**完全复用 dq 现有 filter chain**，通过 `configs/arxiv.yaml` 配置：

```
Pipeline(PipelineConfig.from_yaml("configs/arxiv.yaml"))
```

Filter 执行顺序（由 YAML 定义）：
1. `arxiv` — **先正则清理 LaTeX 残留（就地修改 `doc["text"]`），再检查结构**，仅修不好的才 reject
2. `gopher_quality` — 词长、符号率、停用词、alpha 比例（**在已清理的文本上执行**）
3. `gopher_repetition` — n-gram 重复、行/段重复
4. `c4` — terminal punct, javascript/policy line, min sentences（**也会就地修改 `doc["text"]`**）
5. `fineweb` — line punct, short line, char dup, newline ratio
6. `language` — fasttext 语言检测（en, score > 0.5）
7. `badwords` — LDNOOBW 词表
8. `pii` — PII redact 模式（**也会就地修改 `doc["text"]`**）

**关键设计：三个 filter 会修改文本**（clean-then-judge 模式）：
- `arxiv` filter（第 1 个）：清理 LaTeX 残留 → 后续 filter 在干净文本上计算指标
- `c4` filter（第 4 个）：删 javascript/policy 行 → 行级清理
- `pii` filter（最后）：redact PII → 最终落盘文本是完全清理后的版本

因此 **`arxiv` 必须排第一**，否则 Gopher/C4 的统计指标会被 LaTeX 噪声污染。

**关键复用点**:
- `_eval_chunk()` 已支持 `collect_rejected=True`，rejected 文档自带 `__dq_rejections`
- `filter_detailed()` 返回所有规则命中，不只是第一条
- `multiprocessing.Pool` + spawn 并行，已处理好 `OMP_NUM_THREADS=1`

**Arxiv 阈值微调（全在 YAML，不改 filter 代码）**:
- Gopher: `min_stopwords: 2`, `min_alpha_ratio: 0.7`（学术文本 stopword 占比天然偏低）
- C4: `min_sentences: 3`（论文可能有短 section）

**统计产出**: `stats/v2025-04/phase2_stats.json` + `signals_histograms.json`

### Phase 3: 去重

**输入**: `stage2_filtered/kept/`
**输出**: `stage3_dedup/{kept,rejected}/shard-*.jsonl.zst`

**复用**:
- `dq.dedup.exact.ExactDedup` — SHA256 精确去重
- `dq.dedup.minhash.MinHashDedup` — MinHash LSH 近似去重

**新增**: arxiv_id 版本去重
- 同一 `metadata.arxiv_id` 多版本 → 只保留最新
- 实现位置：`src/dq/runner/phases.py::phase3_dedup()`
- 先做版本去重 → 再精确去重 → 再 MinHash

**统计产出**: `stats/v2025-04/phase3_stats.json`
```json
{
  "input": 2000000,
  "after_version_dedup": 1850000,
  "after_exact_dedup": 1820000,
  "after_minhash_dedup": 1750000,
  "top_clusters": [...]
}
```

**已知限制**: 当前 MinHash 需要全量文档在内存（`datasketch.MinHashLSH`）。~150 万篇论文的 MinHash signature 约需 20-30GB 内存。如果 OOM，切分为两步：per-shard 算 signature（并行），全局 LSH 查询（串行）。

**Caveat**: 此阶段只做 arxiv 内部去重。跨数据源全局去重需等其他子集就绪后另行处理。

---

### Phase 4: 污染检测

**输入**: `stage3_dedup/kept/`
**输出**: `stage4_contamination/{kept,rejected}/shard-*.jsonl.zst`

**完全复用** `dq.contamination.ngram.NgramContaminationDetector`：

```python
from dq.contamination.ngram import NgramContaminationDetector

detector = NgramContaminationDetector(n=13, threshold=0.8)
report = detector.check(docs, benchmarks={"MMLU": mmlu_texts, "MATH": math_texts, ...})
```

已内置的评测集：`mmlu`, `hellaswag`, `arc`, `gsm8k`, `humaneval`, `truthfulqa`

**新增评测集**（数学相关）:
- MATH → 本地 JSONL 加载（`NgramContaminationDetector` 已支持本地文件）
- ProofNet, MiniF2F → 同上

**统计产出**: `stats/v2025-04/phase4_stats.json` + `contamination.json`（复用 `ContaminationReport` 格式）

---

### Phase 5: 最终打包

**输入**: `stage4_contamination/kept/`
**输出**: `stage5_final/shard-{00000..00255}-of-00256.jsonl.zst`

**新增** `src/dq/runner/shard.py`:
- 按 `id` 排序（确保可复现）
- 切分为目标大小的 shard（1GB 压缩后）
- shard 数对齐 2 的幂
- 生成 `manifest.json`（含每个 shard 的 SHA256、文档数、token 估计）
- 生成 `dataset_card.md`

**统计产出**: `stats/v2025-04/overview.json`
```json
{
    "version": "arxiv-v2025-04",
    "phases": {
        "raw": 2000000,
        "phase1_parsed": 1950000,
        "phase2_filtered": 1650000,
        "phase3_dedup": 1500000,
        "phase4_clean": 1480000,
        "phase5_final": 1480000
    },
    "estimated_tokens": 25000000000,
    "config_sha256": "abc123..."
}
```

---

## 七、`dq run` CLI 设计

```python
# src/dq/cli.py 新增

@click.command()
@click.argument("input_path")
@click.option("-o", "--output-dir", required=True, help="Output base directory")
@click.option("-c", "--config", "config_path", required=True, help="Pipeline config YAML")
@click.option("--phase", default=None, help="Run specific phase (1-5), default: all")
@click.option("--resume", is_flag=True, help="Resume from last completed phase")
@click.option("-w", "--workers", default=None, type=int, help="Parallel workers")
@click.option("-n", "--num-samples", default=0, type=int, help="Limit docs (0=all, for testing)")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def run(input_path, output_dir, config_path, phase, resume, workers, num_samples, dry_run):
    """Run production data cleaning pipeline.

    \b
    Examples:
      dq run /data/arxiv/raw -o /data/arxiv -c configs/arxiv.yaml
      dq run /data/arxiv/raw -o /data/arxiv -c configs/arxiv.yaml --phase 2
      dq run /data/arxiv/raw -o /data/arxiv -c configs/arxiv.yaml --resume
      dq run /data/arxiv/raw -o /data/arxiv -c configs/arxiv.yaml -n 1000 --dry-run
    """
```

**可恢复性**: 每个 Phase 完成后写 `_SUCCESS` marker。`--resume` 检查 marker 跳过已完成的阶段。

---

## 八、Production Runner 架构

```
src/dq/runner/
├── __init__.py
├── engine.py       # PhaseEngine: 调度各 phase, 管理 _SUCCESS marker
├── phases.py       # phase1_parse, phase2_filter, phase3_dedup, phase4_contamination, phase5_package
├── shard.py        # ShardWriter: zstd 压缩写入, auto-rotate by size
└── stats.py        # 各 phase 统计收集 + JSON 输出
```

**`engine.py` 核心逻辑**:

```python
class ProductionRunner:
    def __init__(self, config_path: str, input_path: str, output_dir: str, workers: int):
        self.config = PipelineConfig.from_yaml(config_path)
        self.arxiv_config = self._load_arxiv_config(config_path)  # 读 arxiv: 段
        ...

    def run_all(self):
        for phase_fn in [phase1_parse, phase2_filter, phase3_dedup, phase4_contamination, phase5_package]:
            phase_name = phase_fn.__name__
            if self._is_done(phase_name):
                console.print(f"[dim]Skipping {phase_name} (already done)[/dim]")
                continue
            phase_fn(self)
            self._mark_done(phase_name)
```

**`phases.py` 中 Phase 2 的实现**（展示如何复用 Pipeline）:

```python
def phase2_filter(runner: ProductionRunner):
    """质量过滤 — 直接复用 dq Pipeline。"""
    from dq.pipeline import Pipeline

    pipeline = Pipeline(runner.config)
    kept_writer = ShardWriter(runner.output_dir / "stage2_filtered/kept")
    rejected_writer = ShardWriter(runner.output_dir / "stage2_filtered/rejected")

    docs = read_shards(runner.output_dir / "stage1_parsed/kept")

    # 复用现有并行架构
    if runner.workers > 1:
        chunk_results = _run_parallel(
            docs, runner.filter_configs, runner.config.text_field,
            workers=runner.workers, collect_rejected=True,
        )
    # 分流写入 kept/rejected
    ...
```

---

## 九、Arxiv Filter 详细设计 — Clean-then-Judge 模式

设计原则：**先修后判**。与 PII filter（`pii.py` redact 模式就地替换）和 C4 filter（`c4.py:133` 行级清理后更新 `doc[text_field]`）一样，先就地清理文本，再检查清理后的质量，只有清理后仍不达标才 reject。

### 处理流程

```
原始文本
  │
  ├─ Step 1: 正则清理（就地更新 doc["text"]）
  │   ├─ 剥离 \cite{...}, \ref{...}, \label{...}, \eqref{...}
  │   ├─ 剥离 \begin{...}, \end{...} 环境标记（保留内容）
  │   ├─ 展开 \textbf{content} → content（保留格式命令内容）
  │   ├─ 删除 \vspace{...}, \noindent 等排版指令
  │   ├─ 不动 $...$, $$...$$ 公式区域
  │   └─ 合并 3+ 连续空行 → 2 空行
  │
  ├─ Step 2: 清理后检查残留率
  │   └─ residual > threshold → reject（修不好）
  │
  └─ Step 3: 结构检查
      ├─ missing_abstract → reject
      └─ too_few_sections → reject
```

### 完整代码

```python
# src/dq/filters/arxiv.py

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

# ── Step 1: 清理用正则 ──

# 引用 / 交叉引用 → 删除整个命令
_CITE_REF_RE = re.compile(r'\\(?:cite|ref|eqref|label|autoref|cref|Cref)\{[^}]*\}')

# 格式命令 → 保留花括号内容：\textbf{important} → important
_FORMAT_CMD_RE = re.compile(r'\\(?:textbf|textit|emph|underline|texttt|textrm|textsf)\{([^}]*)\}')

# 环境标记 → 删除（保留环境内文本）
_ENV_MARKER_RE = re.compile(r'\\(?:begin|end)\{[^}]*\}')

# 排版指令 → 删除
_LAYOUT_CMD_RE = re.compile(
    r'\\(?:noindent|maketitle|centering|raggedright|raggedleft'
    r'|newline|linebreak|pagebreak|newpage'
    r'|smallskip|medskip|bigskip|vfill|hfill'
    r'|tableofcontents|bibliographystyle)\b'
    r'|\\(?:vspace|hspace)\*?\{[^}]*\}'
)

# 公式区域（保护，清理时跳过）
_DISPLAY_MATH_RE = re.compile(r'\$\$.*?\$\$', re.DOTALL)
_INLINE_MATH_RE = re.compile(r'(?<!\$)\$(?!\$)[^$]+?\$(?!\$)')

# 残留检测
_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+(?:\{[^}]*\})*')

# 结构检测
_ABSTRACT_RE = re.compile(r'(?:^|\n)#+\s*abstract', re.IGNORECASE)
_SECTION_RE = re.compile(r'^#+\s+', re.MULTILINE)

# 空行合并
_MULTI_BLANK_RE = re.compile(r'\n{3,}')


def _clean_latex(text: str) -> str:
    """清理公式区域外的 LaTeX 命令残留，就地返回清理后文本。"""
    # 保护公式区域
    protected: list[tuple[str, str]] = []
    counter = 0

    def _protect(m: re.Match) -> str:
        nonlocal counter
        key = f"\x00M{counter}\x00"
        protected.append((key, m.group()))
        counter += 1
        return key

    text = _DISPLAY_MATH_RE.sub(_protect, text)
    text = _INLINE_MATH_RE.sub(_protect, text)

    # 清理（顺序：引用 → 格式展开 → 环境标记 → 排版）
    text = _CITE_REF_RE.sub('', text)
    text = re.sub(r'\\(?:textbf|textit|emph|underline|texttt|textrm|textsf)\{([^}]*)\}', r'\1', text)
    text = _ENV_MARKER_RE.sub('', text)
    text = _LAYOUT_CMD_RE.sub('', text)

    # 恢复公式
    for key, val in protected:
        text = text.replace(key, val)

    # 合并多余空行
    text = _MULTI_BLANK_RE.sub('\n\n', text)
    return text.strip()


def _residual_frac(text: str) -> float:
    """清理后文本中残留 LaTeX 命令的字符占比（排除公式区域）。"""
    no_math = _DISPLAY_MATH_RE.sub('', text)
    no_math = _INLINE_MATH_RE.sub('', no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(len(m.group()) for m in _LATEX_CMD_RE.finditer(no_math))
    return latex_chars / len(no_math)


@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv clean + reject filter.

    Step 1: Clean — strip LaTeX commands outside math regions (in-place, like C4/PII).
    Step 2: Reject — if residual LaTeX fraction still exceeds threshold.
    Step 3: Reject — if structural checks fail.
    """

    def __init__(self, text_field: str = "text", **kwargs: Any) -> None:
        super().__init__(text_field=text_field, **kwargs)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)

        # Step 1: Clean（就地更新，同 c4.py:133, pii.py:82）
        cleaned = _clean_latex(text)
        doc[self.text_field] = cleaned

        # Step 2: Residual check
        max_residual = self.params.get("max_latex_residual", 0.05)
        frac = _residual_frac(cleaned)
        if frac > max_residual:
            return False, {"filter": self.name, "rule": "latex_residual",
                           "value": round(frac, 4), "threshold": max_residual}

        # Step 3: Structural checks
        if self.params.get("require_abstract", True) and not _ABSTRACT_RE.search(cleaned):
            return False, {"filter": self.name, "rule": "missing_abstract"}

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_SECTION_RE.findall(cleaned))
        if num_sections < min_sections:
            return False, {"filter": self.name, "rule": "too_few_sections",
                           "value": num_sections, "threshold": min_sections}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        """Run all rules (for benchmark stats collection)."""
        text = self.get_text(doc)
        failures: list[dict] = []

        # Step 1: Clean
        cleaned = _clean_latex(text)
        doc[self.text_field] = cleaned

        # Step 2: Residual
        max_residual = self.params.get("max_latex_residual", 0.05)
        frac = _residual_frac(cleaned)
        if frac > max_residual:
            failures.append({"filter": self.name, "rule": "latex_residual",
                             "value": round(frac, 4), "threshold": max_residual})

        # Step 3: Structure
        if self.params.get("require_abstract", True) and not _ABSTRACT_RE.search(cleaned):
            failures.append({"filter": self.name, "rule": "missing_abstract",
                             "value": False, "threshold": True})

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_SECTION_RE.findall(cleaned))
        if num_sections < min_sections:
            failures.append({"filter": self.name, "rule": "too_few_sections",
                             "value": num_sections, "threshold": min_sections})

        return len(failures) == 0, failures
```

### 清理规则速查

| LaTeX 残留 | 动作 | 示例 |
|---|---|---|
| `\cite{ref}` | 删除 | `as shown in \cite{vaswani2017}` → `as shown in` |
| `\ref{fig1}`, `\label{eq1}` | 删除 | `See Figure \ref{fig1}` → `See Figure` |
| `\textbf{word}` | 保留内容 | `\textbf{important}` → `important` |
| `\begin{theorem}` | 删标记 | 环境内文本保留 |
| `\vspace{1em}`, `\noindent` | 删除 | 排版指令无语义 |
| `$E=mc^2$`, `$$\int_0^1$$` | **不动** | 公式是核心内容 |

### 与 Phase 1 的分工

```
Phase 1 (解析阶段):  residual > 20%  →  reject（解析彻底失败，不可修复）
Phase 2 (arxiv filter): 先清理 → 清理后 residual > 5%  →  reject（清理后仍太脏）
                         清理后 residual ≤ 5%  →  keep（修复成功）
```

---

## 十、Golden Test 设计

### 选择标准

| 编号 | arxiv_id | 类型 | 选择理由 |
|---|---|---|---|
| 1-3 | cs.LG 论文 | 标准 ML | 正常 case |
| 4-5 | cs.CL 论文 | NLP | 含大量文本示例 |
| 6-7 | math.AG 论文 | 纯数学 | 密集公式 |
| 8 | physics.hep 论文 | 物理 | 大量 table/figure |
| 9 | cs.SE 论文 | 含代码 | 多 code block |
| 10 | 低质量论文 | 边界 | LaTeX 残留严重，应被过滤 |

### 目录结构

```
golden/
├── arxiv_2301.00001/
│   ├── input.jsonl              # RedPajama 原始格式
│   ├── expected_phase1.json     # Phase 1 期望输出
│   ├── expected_final.json      # Phase 5 期望输出
│   └── notes.md                 # 选择理由
└── ...
```

### 测试 Runner

```bash
uv run pytest tests/test_golden.py -v
```

每篇论文跑完整 pipeline，diff 实际 vs 期望的：
- `trace` 各阶段 status
- `__dq_rejections` 命中规则
- `quality_signals` 关键数值（允许 ±1% 浮点误差）

---

## 七、抽样脚本

复用 `dq.utils.io.reservoir_sample()` 已有的水塘抽样：

```bash
# 均匀抽样
python scripts/sample_arxiv.py --strategy random --n 100 --input stage1/kept --seed 42

# 分层抽样（按 category）
python scripts/sample_arxiv.py --strategy stratified --key metadata.primary_category --per-stratum 20

# 按规则命中
python scripts/sample_arxiv.py --strategy rule-hit --rule latex_residual --n 50

# 按信号分桶
python scripts/sample_arxiv.py --strategy buckets --signal quality_signals.frac_latex_commands --bins 5 --per-bin 10
```

---

## 八、Dashboard（Vite + React）

### 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 构建 | Vite | 快速 HMR，零配置 TypeScript |
| UI 框架 | React 18 + TypeScript | 组件化，状态管理清晰 |
| 样式 | Tailwind CSS | utility-first，与 Vite 集成好 |
| 路由 | React Router | Tab 切换 → URL hash 路由 |
| 图表 | Recharts | React 原生图表库，比 Chart.js 集成更自然 |
| 公式渲染 | KaTeX + react-katex | LaTeX 公式渲染 |
| Markdown | react-markdown + remark-math | Markdown 渲染 + 数学公式支持 |
| Diff | react-diff-viewer-continued | Golden test / dedup cluster 文本对比 |
| 数据 | 静态 JSON（fetch from `public/data/`） | 无后端，build 后可直接部署 |

### 项目结构

```
dashboard/                              # 独立 Vite 项目
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── public/
│   └── data/                           # build_dashboard.py 输出到这里
│       ├── overview.json
│       ├── phase_stats.json
│       ├── signals_histograms.json
│       ├── clusters.json
│       ├── contamination.json
│       ├── golden_results.json
│       └── samples/
│           ├── index.json
│           ├── phase1_random_100.json
│           └── ...
├── src/
│   ├── main.tsx
│   ├── App.tsx                         # 路由 + layout
│   ├── components/
│   │   ├── Layout.tsx                  # 侧边栏 / Tab 导航
│   │   ├── KPICard.tsx                 # 数字指标卡片
│   │   ├── FunnelChart.tsx             # Pipeline 漏斗图
│   │   ├── HistogramChart.tsx          # Signal 直方图 + 阈值线
│   │   ├── MarkdownViewer.tsx          # react-markdown + KaTeX
│   │   ├── DocumentDetail.tsx          # 文档详情面板（raw/parsed/json 切换）
│   │   ├── DiffViewer.tsx              # 文本 diff 对比
│   │   └── TagButton.tsx               # 好/坏/可疑 标记按钮
│   ├── pages/
│   │   ├── Overview.tsx                # 漏斗 + KPI + category 饼图
│   │   ├── PhaseDetails.tsx            # 各阶段统计 + 规则命中柱状图
│   │   ├── QualitySignals.tsx          # 每个 signal 的直方图网格
│   │   ├── SampleBrowser.tsx           # 三栏：文件列表 / 文档列表 / 详情
│   │   ├── DedupClusters.tsx           # Cluster 列表 + side-by-side diff
│   │   ├── Contamination.tsx           # 评测集分组 + 13-gram 高亮
│   │   └── GoldenTests.tsx             # Pass/fail 状态 + diff
│   ├── hooks/
│   │   ├── useData.ts                  # fetch + cache JSON 数据
│   │   └── useLocalTags.ts             # localStorage 标记管理
│   └── types/
│       └── index.ts                    # 数据 TypeScript 类型定义
└── tailwind.config.js
```

### 页面设计

**Overview** — 顶部 4 个 KPI 卡片（原始文档数、最终文档数、总 tokens、保留率）+ Pipeline 漏斗图（Recharts BarChart）+ Category 饼图

**Sample Browser**（最核心页面）：
- 左栏：sample 文件列表（`useData` hook 从 `samples/index.json` 加载）
- 中栏：当前 sample 的文档列表（id, category, 关键 signal 预览）
- 右栏：选中文档详情
  - Tab 切换：Raw Preview / Parsed Markdown（`react-markdown` + `react-katex`）/ Full JSON
  - Trace 时间线（phase1 → phase5 状态）
  - Quality signals 详细值表格
  - 标记按钮（好/坏/可疑 → `localStorage`）+ 导出标记

**Quality Signals** — CSS Grid 布局，每个 signal 一个 `HistogramChart` 卡片，红色竖线标注当前阈值，hover 显示 bin 详情

**Dedup Clusters** — 左侧 cluster 列表（按 size 排序），右侧 `react-diff-viewer` 对比两篇文档

### 开发 & 部署

```bash
cd dashboard
npm install
npm run dev                # 开发：http://localhost:5173

# 生产构建 → 纯静态文件
npm run build              # 输出到 dashboard/dist/
# 把 dist/ 复制到任意 HTTP server 即可访问
```

`scripts/build_dashboard.py` 负责把 `stats/` + `samples/` 整合成 `dashboard/public/data/*.json`，然后 Vite build 会把它们打入 `dist/`。

### Tab 路由

| 路径 | 页面 | 对应组件 |
|---|---|---|
| `/` | Overview | `Overview.tsx` |
| `/phases` | Phase Details | `PhaseDetails.tsx` |
| `/signals` | Quality Signals | `QualitySignals.tsx` |
| `/samples` | Sample Browser | `SampleBrowser.tsx` |
| `/dedup` | Dedup Clusters | `DedupClusters.tsx` |
| `/contamination` | Contamination | `Contamination.tsx` |
| `/golden` | Golden Tests | `GoldenTests.tsx` |

---

## 九、CLI 接口汇总

```bash
# 现有（不变）
dq bench data.jsonl -n 1000                                    # 质量检测报告
dq bench data.jsonl --save-rejected rejected.jsonl             # 导出被拒文档
dq bench data.jsonl --check-contamination mmlu,hellaswag       # 污染检测

# 新增
dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml         # 全量清洗
dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml --phase 2  # 只跑 Phase 2
dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml --resume   # 断点续跑
dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml -n 1000    # 少量测试
```

---

## 十、实施计划

### 第 1 周：地基

| 天 | 任务 | 产出 |
|---|---|---|
| D1 | `configs/arxiv.yaml` + `src/dq/filters/arxiv.py` | 新 filter 可注册 |
| D1 | `uv run pytest tests/test_arxiv_filter.py` 通过 | 单元测试 |
| D2 | `src/dq/runner/engine.py` + `shard.py` 框架 | PhaseEngine 可初始化 |
| D3 | Phase 1 实现（RedPajama 下载 + schema 转换） | `phase1_parse()` 可运行 |
| D3 | `dq run` CLI 子命令 | `dq run --help` 可用 |
| D4-5 | Phase 2 实现（复用 Pipeline + 并行 worker） | 小数据集端到端通过 |

### 第 2 周：核心 Pipeline

| 天 | 任务 | 产出 |
|---|---|---|
| D6 | Phase 3（版本去重 + exact + MinHash） | 去重可运行 |
| D7 | Phase 4（污染检测，复用 `NgramContaminationDetector`） | 污染检查通过 |
| D8 | Phase 5（shard 打包 + manifest） | `stage5_final/` 完整产出 |
| D9 | Golden test（10 篇端到端） | `uv run pytest tests/test_golden.py` 全绿 |
| D10 | 抽样脚本 | `scripts/sample_arxiv.py` 可用 |

### 第 3 周：可视化 + 全量运行

| 天 | 任务 | 产出 |
|---|---|---|
| D11-12 | Dashboard React 项目（Vite + 7 个页面组件） | `npm run dev` 可用 |
| D13 | `build_dashboard.py` + `npm run build` | 静态产物可部署 |
| D14 | 全量运行（~200 万文档） | `stage5_final/` 产出 |
| D15 | 人工 QA + 调参 + dataset card | 最终交付 |

---

## 十一、代码改动清单

### 新增文件

| 文件 | 行数估计 | 说明 |
|---|---|---|
| `src/dq/filters/arxiv.py` | ~100 | Arxiv 专属 filter |
| `src/dq/runner/__init__.py` | ~10 | Production runner 包 |
| `src/dq/runner/engine.py` | ~150 | 多阶段引擎 + 断点续跑 |
| `src/dq/runner/phases.py` | ~300 | 5 个 phase 实现 |
| `src/dq/runner/shard.py` | ~80 | Shard 切分 + manifest |
| `src/dq/runner/stats.py` | ~100 | 统计收集 |
| `configs/arxiv.yaml` | ~80 | Arxiv 配置 |
| `scripts/sample_arxiv.py` | ~120 | 抽样脚本 |
| `scripts/build_dashboard.py` | ~150 | Dashboard 数据构建 |
| `dashboard/` (Vite + React) | ~1500 | 7 个页面 + 组件 + hooks |
| `tests/test_arxiv_filter.py` | ~60 | Filter 单测 |
| `tests/test_runner.py` | ~80 | Runner 单测 |
| `tests/test_golden.py` | ~50 | Golden set 回归 |

### 修改文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `src/dq/cli.py` | +30 行 | 新增 `dq run` 子命令 |
| `CLAUDE.md` | 更新 | 新增 runner 相关文档 |
| `README.md` / `README_CN.md` | 更新 | 新增 `dq run` 文档 |

### 不改动的文件

所有现有 filter（`gopher.py`, `c4.py`, `fineweb.py`, `language.py`, `badwords.py`, `pii.py`）、`pipeline.py`、`dedup/`、`contamination/`、`benchmark/` — **零改动**，仅通过配置和 import 复用。

---

## 十二、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Gopher/C4 阈值对学术文本过严 | 高 | 过滤率偏高 | 先跑 `dq bench` 看分布，按直方图调参 |
| MinHash 内存不足（150 万篇） | 中 | OOM | 分批算 signature，单进程跑 LSH |
| RedPajama LaTeX 解析质量差 | 中 | 噪声入库 | Phase 1 严格过滤 + Golden test 兜底 |
| 去重只在 arxiv 内部 | 已知 | 跨源重复 | dataset_card 标注 caveat，后续全局去重 |
| 新评测集需重跑 Phase 4 | 低 | 重复工作 | 阶段独立可重跑（`--phase 4`） |

---

## 十三、验收标准

1. `uv run pytest` 全绿（含新增测试）
2. `uv run dq bench` 现有功能不受影响（无回归）
3. `uv run dq run` 在 1000 篇样本上端到端通过
4. Golden set 10 篇全部通过
5. `align_with_datatrove.py` 仍 100% 对齐（未改动现有 filter）
6. Dashboard `npm run dev` 可运行，`npm run build` 产出可部署，公式用 KaTeX 正确渲染
7. 全量清洗率在 20-40% 区间（即保留 60-80%）
8. `stage5_final/manifest.json` 数字交叉验证一致

---

## 十四、已知风险 & Caveat

1. **去重仅限 arxiv 内部** — 跨数据源去重需等所有子集就绪后二次处理
2. **LaTeX 质量依赖 RedPajama** — 无法控制上游解析精度
3. **数学论文的 stopword 率天然低** — Gopher `min_stopwords` 需特别调参
4. **公式密集文本会拉高 symbol_ratio** — 考虑排除 `$...$` 区域后再计算（在 `arxiv.py` 中预处理）
5. **License 过滤未做** — arxiv 论文 license 复杂,需律师确认后再加规则
