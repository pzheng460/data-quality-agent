# Data Quality Report: test_input

**Samples**: 5

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 5 |
| Data type | pretrain |
| Avg word count | 19.2 |
| Min word count | 0 |
| Max word count | 76 |
| Avg word length | 3.26 |
| Fields | text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 5 |
| Passed (all filters) | 1 |
| Failed | 4 |
| Average pass rate (per filter) | 56.0% |
| Overall pass rate (pass all) | 20.0% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 20.0% | 4 | 5 |
| **gopher_repetition** | 100.0% | 0 | 5 |
| **c4** | 20.0% | 4 | 5 |
| **fineweb** | 40.0% | 3 | 5 |
| **pii** | 100.0% | 0 | 5 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| min_words | 4 | 5 | 80.00% |
| lines_end_punct | 3 | 5 | 60.00% |
| stopwords | 3 | 5 | 60.00% |
| min_avg_word_len | 2 | 5 | 40.00% |
| alpha_ratio | 1 | 5 | 20.00% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 3 | 5 | 60.00% |
| min_sentences | 1 | 5 | 20.00% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| line_punct_ratio | 2 | 5 | 40.00% |
| short_line_ratio | 2 | 5 | 40.00% |
| empty_doc | 1 | 5 | 20.00% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `min_words` (value: 1)
> Hello...

**Sample 2**: `min_words` (value: 18)
> This is a normal sentence with enough words to pass most filters and quality checks in the pipeline....

**Sample 3**: `min_words` (value: 0)
> ...

### c4

**Sample 1**: `empty_after_line_filter` (value: no_terminal_punct(1))
> Hello...

**Sample 2**: `min_sentences` (value: sentences=1)
> This is a normal sentence with enough words to pass most filters and quality checks in the pipeline....

**Sample 3**: `empty_after_line_filter` (value: all lines empty)
> ...

### fineweb

**Sample 1**: `line_punct_ratio` (value: 0.0)
> Hello...

**Sample 2**: `empty_doc` (value: 0)
> ...

**Sample 3**: `line_punct_ratio` (value: 0.0)
> x...
