# dq — 训练数据质量检测工具

一个用于检测低质量 LLM 训练数据的 Python CLI 工具。单一命令 `dq bench` 覆盖数据集统计、规则过滤、去重检测、污染检查和基于 LLM 的质量评估。

## 架构

- **第一层：规则过滤器** — 确定性、零成本、毫秒级
  - 预训练：Gopher 质量/重复检测、C4、FineWeb、PII
  - SFT：空输出、输出过短、指令复制、AI 拒答、语言不匹配
  - 去重：精确去重（SHA256）
- **第二层：LLM 二分类评判** — 通过 LLM API 进行语义质量评估
  - SFT：指令遵循、事实性、完整性、格式合规、无害性
  - 预训练：信息密度、连贯性、原创性
- **污染检测** — 基于 n-gram 的基准数据集重叠检测

## 安装

```bash
uv sync                    # 核心依赖
uv sync --extra bench      # + HuggingFace datasets
```

## 使用方法

```bash
# 本地文件
dq bench data.jsonl -n 1000

# 多进程并行（16 个工作进程）
dq bench data.jsonl -n 1000 -w 16

# HuggingFace 数据集（流式加载，无需完整下载）
dq bench allenai/dolma3_mix-6T -n 10000

# 污染检查（内置基准）
dq bench data.jsonl --check-contamination mmlu,hellaswap

# 污染检查（所有内置基准）
dq bench data.jsonl --check-contamination all

# 污染检查（HuggingFace 数据集作为基准）
dq bench data.jsonl --check-contamination cais/mmlu

# 污染检查（本地文件作为基准）
dq bench data.jsonl --check-contamination /path/to/benchmark.jsonl

# 使用第二层 LLM 评判
dq bench data.jsonl --with-llm-scoring --llm-samples 50
```

报告默认保存到 `reports/` 目录（JSON + Markdown 格式），可通过 `-o` 自定义输出路径。

## 第一层：规则过滤器

### `gopher_quality` — 基础质量启发式检查

来源：Gopher（Rae et al., 2021）

| 规则 | 检查内容 | 默认阈值 | 检测值 |
|------|---------|----------|--------|
| `min_words` | 文档过短 | 50 词 | 词数 |
| `max_words` | 文档过长 | 100,000 词 | 词数 |
| `min_avg_word_len` | 平均词长过短 | 3.0 字符 | 平均词长 |
| `max_avg_word_len` | 平均词长过长 | 10.0 字符 | 平均词长 |
| `symbol_ratio` | 符号标记过多（`#`、`...`、`…`） | 10% | 符号词比 |
| `lines_end_punct` | 以标点结尾的行过少（`.!?。！？;；`） | 10% | 终端标点行比例 |
| `stopwords` | 英语停用词过少（非自然语言） | 2 | 停用词数 |
| `alpha_ratio` | 字母/中日韩字符过少 | 80% | 字母字符比 |

### `gopher_repetition` — 重复检测

来源：Gopher（Rae et al., 2021）

| 规则 | 检查内容 | 默认阈值 | 检测值 |
|------|---------|----------|--------|
| `top_2gram` | 最频繁 2-gram 覆盖过多文本 | 20% | 字符覆盖率 |
| `top_3gram` | 最频繁 3-gram 覆盖过多文本 | 18% | 字符覆盖率 |
| `top_4gram` | 最频繁 4-gram 覆盖过多文本 | 16% | 字符覆盖率 |
| `dup_line_ratio` | 重复行过多 | 30% | 重复行比例 |
| `dup_para_ratio` | 重复段落过多 | 30% | 重复段落比例 |
| `dup_5gram_frac` | 重复 5-gram 覆盖的文本 | 15% | 字符比例 |
| `dup_6gram_frac` | 重复 6-gram 覆盖的文本 | 14% | 字符比例 |
| `dup_7gram_frac` | 重复 7-gram 覆盖的文本 | 13% | 字符比例 |
| `dup_8gram_frac` | 重复 8-gram 覆盖的文本 | 12% | 字符比例 |
| `dup_9gram_frac` | 重复 9-gram 覆盖的文本 | 11% | 字符比例 |
| `dup_10gram_frac` | 重复 10-gram 覆盖的文本 | 10% | 字符比例 |

### `c4` — 行级清洗 + 文档检查

来源：C4（Raffel et al., 2020）

C4 先移除有问题的行，再检查剩余文档是否有效。

**行移除（非拒绝——从文档中清洗）：**
- 包含 `javascript`（不区分大小写）的行
- 包含政策/Cookie 语言的行（`terms of use`、`privacy policy`、`cookie policy` 等）
- 没有终端标点（`.!?。！？;；`）的行

| 规则 | 检查内容 | 默认值 | 检测值 |
|------|---------|--------|--------|
| `empty_after_line_filter` | 行清洗后文档为空 | N/A | 移除详情：`javascript(N)`、`policy(N)`、`no_terminal_punct(N)` |
| `min_sentences` | 清洗后句子过少 | 3 | 句子数 |
| `lorem_ipsum` | 包含 "lorem ipsum" 文本 | 启用 | 布尔值 |
| `curly_brace` | 包含花括号 `{` | 禁用 | 布尔值 |

### `fineweb` — 网页文档质量

来源：FineWeb（Penedo et al., 2024）。已与 datatrove 的 `FineWebQualityFilter` 对齐。

| 规则 | 检查内容 | 默认阈值 | 检测值 |
|------|---------|----------|--------|
| `empty_doc` | 文档无内容 | N/A | 0 |
| `line_punct_ratio` | 以标点结尾的行过少 | 最低 12% | 行标点比例 |
| `short_line_ratio` | 短行（≤ 30 字符）过多 | 最高 67% | 短行比例 |
| `char_dup_ratio` | 重复行字符覆盖率过高 | 最高 1% | 字符重复比例 |
| `list_ratio` | 换行符相对词数过多（列表特征） | 最高 0.3 | 换行/词数比 |

### `pii` — 个人身份信息检测

默认模式：`redact`（替换 PII 占位符，不拒绝文档）。

| 规则 | 检测内容 | 替换值 |
|------|---------|--------|
| `email` | 邮箱地址 | `email@example.com` |
| `ip` | 公网 IPv4 地址（排除私有地址段） | `0.0.0.0` |
| `cn_phone` | 中国手机号（1[3-9]XXXXXXXXX） | `1XXXXXXXXXX` |
| `cn_id` | 中国身份证号（18 位） | `XXXXXXXXXXXXXXXXXX` |
| `bank_card` | 银行卡号（16-19 位） | `XXXXXXXXXXXXXXXX` |

### `sft_rules` — SFT 数据质量

| 规则 | 检查内容 | 默认阈值 | 检测值 |
|------|---------|----------|--------|
| `missing_sft_fields` | 未找到指令/输出字段 | N/A | 文档字段 |
| `empty_output` | 输出为空或仅含空白 | N/A | 0 |
| `output_too_short` | 输出相对指令（≥20 词）过短 | 5 词（封闭式任务为 1） | 输出词数 |
| `instruction_copy` | 输出与指令过于相似（字符 3-gram Jaccard） | 80% 相似度 | 相似度分数 |
| `ai_refusal` | 输出以拒答模式开头 | 强拒答：总是拒绝；弱拒答：< 50 词时拒绝 | 匹配模式 |
| `language_mismatch` | 指令与输出的中日韩字符比差异 | 30% 差异 | CJK 比例差 |

**封闭式任务检测：** 匹配 `classify`、`categorize`、`yes or no`、`true or false`、`extract`、`name the`、`in one word` 等模式的指令，`output_too_short` 阈值降低（5 → 1 词）。

**AI 拒答区分：**
- 强拒答（总是拒绝）："I cannot"、"I'm sorry, but I cannot"、"I'm not able to"
- 弱拒答（仅 < 50 词时拒绝）："As an AI language model"、"As an AI assistant"、"I apologize"

**SFT 字段自动检测：** `instruction`/`output`、`prompt`/`response`、`question`/`answer`、`query`/`reply`、`human`/`assistant`、`conversations`（ShareGPT 格式）。

## 第二层：LLM 二分类评判

通过 LLM API 进行数据驱动的二分类（HIGH/LOW）。自动检测数据类型并应用相应规则。

```python
from dq.judge import LLMJudge

judge = LLMJudge()
result = judge.judge_sft("解释量子计算", "量子计算使用量子比特...")
# {"quality": "high", "rules": {...}, "failed_rules": []}
```

### LLM 配置

创建 `configs/llm.yaml` 填入 API 凭证（此文件已被 gitignore）：

```bash
cp configs/llm.yaml.example configs/llm.yaml
# 编辑 configs/llm.yaml 填入凭证
```

```yaml
# configs/llm.yaml
api_url: "https://api.openai.com/v1"
api_key: "sk-your-api-key"
model: "gpt-4o-mini"
samples: 50
```

配置优先级：CLI 参数 > `configs/llm.yaml` > 环境变量（`DQ_API_BASE_URL`、`DQ_API_KEY`、`DQ_MODEL`）。

## 污染检测

基于 n-gram 重叠的基准数据集污染检测。支持：
- 内置基准：mmlu、hellaswap、arc、truthfulqa、gsm8k、humaneval
- 任意 HuggingFace 数据集 ID（如 `cais/mmlu`）
- 本地文件（text/jsonl）

## 配置

YAML 配置文件位于 `configs/`：

- `configs/default.yaml` — 英文预训练
- `configs/pretrain_zh.yaml` — 中文预训练（CJK 感知）
- `configs/sft.yaml` — SFT 数据（仅 SFT 规则 + PII）

## 基准测试结果（第一层，每组 1000 条样本）

**预训练：**

| 数据集 | 通过率 |
|--------|:------:|
| TinyStories | 99.4% |
| OpenWebText | 94.1% |
| FineWeb | 93.6% |
| CC-News | 91.3% |
| Wikipedia | 89.8% |
| C4 | 86.6% |
| Wikitext-103 | 47.4% |

**SFT：**

| 数据集 | 通过率 |
|--------|:------:|
| Dolly | 99.8% |
| Alpaca GPT-4 | 99.8% |
| No Robots | 99.6% |
| GPT4All | 99.7% |
| WizardLM | 98.3% |
| OpenOrca | 95.0% |

## 支持格式

JSONL（`.jsonl`）、Parquet（`.parquet`）、CSV（`.csv`）

## 开发

```bash
uv run pytest          # 运行所有测试
```

## 项目结构

```
src/dq/
├── cli.py                  # CLI — 单一 `dq bench` 命令
├── pipeline.py             # 流水线编排 + 过滤器注册
├── config.py               # YAML 配置加载
├── judge.py                # LLM 二分类评判（第二层）
├── llm_client.py           # 共享 OpenAI 兼容客户端
├── benchmark/              # 基准测试运行器
│   ├── runner.py           #   run_benchmark, run_llm_scoring
│   ├── datasets.py         #   数据集加载（本地 + HuggingFace）
│   ├── types.py            #   BenchmarkReport, DatasetResult 等
│   └── utils.py            #   detect_data_type, SFT 字段提取
├── benchmark_report.py     # 报告输出（Rich/Markdown/JSON）
├── filters/                # 第一层：规则过滤器
│   ├── gopher.py           #   Gopher 质量 + 重复检测
│   ├── c4.py               #   C4 过滤器
│   ├── fineweb.py          #   FineWeb 过滤器
│   ├── sft_rules.py        #   SFT 规则
│   └── pii.py              #   PII 检测/脱敏
├── dedup/                  # 去重（用于基准统计）
│   ├── exact.py            #   SHA256 精确去重
│   └── minhash.py          #   MinHash LSH 近似去重
├── contamination/          # 污染检测
│   ├── ngram.py            #   N-gram 重叠（内置 + HF + 本地）
│   └── report.py           #   报告
└── utils/                  # 工具函数
    ├── io.py               #   文件 I/O
    ├── stats.py            #   文本统计
    └── tokenizer.py        #   分词
```

## 参考文献

- Gopher（Rae et al., 2021）— 预训练质量/重复启发式
- C4（Raffel et al., 2020）— 行级清洗 + 句子过滤
- FineWeb（Penedo et al., 2024）— 网络规模去重与过滤
- InsTag（Lu et al., 2023）— 指令标注（开放式 vs 封闭式）
- AlpaGasus（Chen et al., 2023）— 基于 LLM 的 SFT 数据过滤

## 许可证

MIT
