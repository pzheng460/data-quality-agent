# dq — 训练数据质量智能体

Python CLI + Web 控制台，用于检测和清洗低质量 LLM 训练数据，重点支持 arXiv 学术论文。

## 架构

### 4 阶段 Pipeline

```
阶段1: 采集 (Ingestion)    → 获取原始数据 (LaTeX/HTML/JSONL)
阶段2: 提取 (Extraction)   → 转为纯文本 (LaTeXML + 预处理)
阶段3: 策展 (Curation)     → 过滤 + 去重 + 污染检测
阶段4: 打包 (Packaging)    → 排序、分片、清单
```

### LaTeX 提取流程

```
raw .tex
   ↓
预处理器 (preprocess.py)
   ├── 提取 algorithm → 解析为缩进伪代码
   ├── 删除 tikz/pgfplots (图片代码，无文本价值)
   ├── 提取 align*/equation* → 展开宏 → $$...$$ display math
   └── 剥离 mdframed/tcolorbox 框架参数
   ↓
LaTeXML → HTML → html_to_text
   ↓
KaTeX 兼容层 (katex_compat.py)
   ├── \mathbbm → \mathbb, \bm → \boldsymbol, \textsc → \text ...
   ├── 去除 \label, \tag, 尺寸命令
   └── 清理 \v 前缀合并命令
   ↓
恢复占位符 → 最终输出
```

### 质量过滤器

| 过滤器 | 来源 | 检查内容 |
|--------|------|---------|
| `gopher_quality` | Gopher (2021) | 词数、平均词长、符号比例、字母比例、停用词 |
| `gopher_repetition` | Gopher (2021) | n-gram 重复率、重复段落/行 |
| `c4` | C4 (2020) | Lorem ipsum、JS/政策行、最少句数 |
| `fineweb` | FineWeb (2024) | 行标点、短行比例、字符去重 |
| `arxiv` | 自研 | LaTeX 残留率、章节数、引用/tikz/mdframed 清理 |
| `pii` | 自研 | 邮箱、电话、IP、身份证、银行卡 → 脱敏 |
| `sft_rules` | 自研 | 空输出、指令复制、AI 拒答、语言不匹配 |

### 去重

- **精确去重**: SHA256
- **近似去重**: MinHash LSH (112 permutations, 14 bands)
- **版本去重**: 同一 arXiv 论文只保留最新版

### 污染检测

N-gram 重叠检测：MMLU, GSM8K, HellaSwag, ARC, HumanEval, TruthfulQA

## 安装

```bash
uv sync                          # 核心
uv sync --extra bench            # + HuggingFace datasets
# 系统依赖: apt install latexml
```

## 使用

### 完整 Pipeline

```bash
dq run INPUT -o OUTPUT -c configs/arxiv.yaml
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --stage 2    # 单阶段
dq run INPUT -o OUTPUT -c configs/arxiv.yaml --resume     # 断点续跑
```

### 质量评测

```bash
dq bench data.jsonl -n 1000
dq bench data.jsonl -n 1000 -w 16
dq bench data.jsonl --with-llm-scoring --llm-samples 50
dq bench data.jsonl --check-contamination mmlu,hellaswag
```

### Web 控制台

```bash
uv run uvicorn src.dq.server.app:app --port 8001 --host 0.0.0.0 --reload
cd dashboard && npm run dev
```

## 数据源

| 数据源 | 输出格式 | 说明 |
|--------|----------|------|
| `arxiv_latexml` | latex → text | LaTeXML 转换，质量最高 |
| `arxiv_ar5iv` | html → text | ar5iv HTML，速度快 |
| `arxiv_hf_bulk` | text | HF 预转换，按 shard 快速查找 |
| `local_file` | text | 本地 JSONL/Parquet/CSV |

## 核心特性

- **LaTeX 预处理器**: 在 LaTeXML 之前提取有问题的环境，避免渲染垃圾泄露
- **算法解析**: 支持 `algorithm2e`（\ForEach/\If）和 `algorithmic`（\STATE/\FOR）
- **宏展开**: 从 preamble 提取 \newcommand/\def，在所有数学区域展开
- **KaTeX 兼容层**: 统一修复 50+ 种不兼容 LaTeX 命令
- **表格提取**: rowspan/colspan、多行表头合并、span-based 表格、makecell
- **自动评测**: Pipeline 完成后自动 benchmark 最终输出

## 开发

```bash
uv run pytest                    # 全量测试 (340+)
uv run pytest tests/test_extraction_golden.py -v  # Golden 回归测试
```

## 项目结构

```
src/dq/
  stages/
    ingestion/           → @register_source 数据源
    extraction/          → @register_extractor 格式转换
      preprocess.py      →   LaTeX 预处理器
      algorithm.py       →   算法伪代码解析器
      table.py           →   LaTeX 表格解析器
      katex_compat.py    →   KaTeX 兼容层
      html.py            →   HTML → text
      latex.py           →   LaTeXML 编排
    curation/
      filters/           → @register_filter 过滤器
      dedup/             →   精确 + MinHash 去重
      contamination/     →   N-gram 污染检测
  runner/engine.py       → 4 阶段引擎
  server/app.py          → FastAPI 后端
  benchmark/             → 质量评测
dashboard/               → React + shadcn/ui 前端
configs/                 → YAML 配置
tests/                   → 340+ 测试
```

## License

MIT
