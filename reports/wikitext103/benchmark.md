# Data Quality Report: wikitext

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | pretrain |
| Avg word count | 88.0 |
| Min word count | 2 |
| Max word count | 542 |
| Avg word length | 3.90 |
| Fields | text |
| Exact duplicates | 15 (1.5%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 474 |
| Failed | 526 |
| Average pass rate (per filter) | 74.0% |
| Overall pass rate (pass all) | 47.4% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 50.9% | 491 | 1000 |
| **gopher_repetition** | 67.3% | 327 | 1000 |
| **c4** | 51.7% | 483 | 1000 |
| **fineweb** | 100.0% | 0 | 1000 |
| **pii** | 100.0% | 0 | 1000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| min_words | 445 | 1000 | 44.50% |
| lines_end_punct | 421 | 1000 | 42.10% |
| stopwords | 310 | 1000 | 31.00% |
| alpha_ratio | 234 | 1000 | 23.40% |
| min_avg_word_len | 177 | 1000 | 17.70% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| top_4gram | 288 | 1000 | 28.80% |
| top_2gram | 266 | 1000 | 26.60% |
| top_3gram | 266 | 1000 | 26.60% |
| dup_9gram_frac | 1 | 1000 | 0.10% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 386 | 1000 | 38.60% |
| min_sentences | 97 | 1000 | 9.70% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `min_words` (value: 5)
>  = Valkyria Chronicles III =  ...

**Sample 2**: `min_words` (value: 5)
>  = = Gameplay = =  ...

**Sample 3**: `min_words` (value: 5)
>  = = Plot = =  ...

### gopher_repetition

**Sample 1**: `top_2gram` (value: 0.25)
>  = Valkyria Chronicles III =  ...

**Sample 2**: `top_2gram` (value: 0.5)
>  = = Gameplay = =  ...

**Sample 3**: `top_2gram` (value: 0.5)
>  = = Plot = =  ...

### c4

**Sample 1**: `empty_after_line_filter` (value: no_terminal_punct(1))
>  = Valkyria Chronicles III =  ...

**Sample 2**: `empty_after_line_filter` (value: no_terminal_punct(1))
>  = = Gameplay = =  ...

**Sample 3**: `empty_after_line_filter` (value: no_terminal_punct(1))
>  = = Plot = =  ...
