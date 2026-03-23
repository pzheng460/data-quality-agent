# Data Quality Benchmark: dolma3_sample

**Samples per dataset**: 1000

| Filter | dolma3_sample | Δ | Verdict |
|--------|--------|--------|--------|
| **gopher_quality** | 85.8% | +0.0% | — No signal |
|   ├─ min_words | 92.2% | +0.0% | — No signal |
|   ├─ min_avg_word_len | 98.9% | +0.0% | — No signal |
|   ├─ lines_end_punct | 86.6% | +0.0% | — No signal |
|   ├─ stopwords | 93.2% | +0.0% | — No signal |
|   ├─ alpha_ratio | 95.6% | +0.0% | — No signal |
|   └─ max_avg_word_len | 99.6% | +0.0% | — No signal |
| **gopher_repetition** | 99.7% | +0.0% | — No signal |
|   ├─ top_2gram | 95.4% | +0.0% | — No signal |
|   ├─ top_3gram | 96.7% | +0.0% | — No signal |
|   ├─ top_4gram | 97.2% | +0.0% | — No signal |
|   ├─ dup_line_ratio | 99.7% | +0.0% | — No signal |
|   ├─ dup_para_ratio | 99.8% | +0.0% | — No signal |
|   ├─ dup_5gram_frac | 99.9% | +0.0% | — No signal |
|   ├─ dup_6gram_frac | 99.9% | +0.0% | — No signal |
|   ├─ dup_8gram_frac | 99.9% | +0.0% | — No signal |
|   ├─ dup_9gram_frac | 99.9% | +0.0% | — No signal |
|   └─ dup_10gram_frac | 99.9% | +0.0% | — No signal |
| **c4** | 99.2% | +0.0% | — No signal |
|   ├─ empty_after_line_filter | 91.4% | +0.0% | — No signal |
|   └─ min_sentences | 98.3% | +0.0% | — No signal |
| **fineweb** | 99.2% | +0.0% | — No signal |
|   ├─ bad_line_breaks | 99.4% | +0.0% | — No signal |
|   ├─ dup_line_ratio | 99.7% | +0.0% | — No signal |
|   └─ list_line_ratio | 99.5% | +0.0% | — No signal |
| **pii** | 100.0% | +0.0% | — No signal |
|   ├─ email | 98.2% | +0.0% | — No signal |
|   └─ ip | 99.9% | +0.0% | — No signal |
| **Overall pipeline** | **84.1%** | **+0.0%** | **— No signal** |

### Legend

- ✅ = Cleaned version passes significantly more (>5%) → filter catches real issues
- ⚠️ = Small difference (2-5%) → filter has weak signal for this data type
- — = No meaningful difference (<2%) → filter not relevant for SFT data
