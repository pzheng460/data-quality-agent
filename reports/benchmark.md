# Data Quality Report: shard-00000.jsonl

**Samples**: 2
**Config**: `configs/arxiv.yaml`

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 2 |
| Data type | pretrain |
| Avg word count | 10395.0 |
| Min word count | 7063 |
| Max word count | 13727 |
| Avg word length | 5.37 |
| Fields | id, text, source, metadata |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 2 |
| Passed (all filters) | 2 |
| Failed | 0 |
| Average pass rate (per filter) | 100.0% |
| Overall pass rate (pass all) | 100.0% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **arxiv** | 100.0% | 0 | 2 |
| **gopher_quality** | 100.0% | 0 | 2 |
| **c4** | 100.0% | 0 | 2 |
| **pii** | 100.0% | 0 | 2 |

## Per-Rule Breakdown

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 2 | 2 | 100.00% |
