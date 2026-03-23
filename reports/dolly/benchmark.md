# Data Quality Report: databricks-dolly-15k

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | sft |
| Avg word count | 130.1 |
| Min word count | 7 |
| Max word count | 1971 |
| Avg word length | 5.14 |
| Fields | instruction, context, response, category, text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 997 |
| Failed | 3 |
| Average pass rate (per filter) | 99.8% |
| Overall pass rate (pass all) | 99.7% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **sft_rules** | 99.8% | 2 | 1000 |
| **pii** | 99.9% | 1 | 1000 |

## Per-Rule Breakdown

### sft_rules

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| instruction_copy | 2 | 1000 | 0.20% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 1 | 1000 | 0.10% |

## Sample Dropped Documents

### sft_rules

**Sample 1**: `instruction_copy` (value: 0.805)
> Classify each of the following as either titles by J.K. Rowling or Suzanne Collins: Harry Potter and...

**Sample 2**: `instruction_copy` (value: 0.881)
> Which films contain Tom Cruise and which do not contain Tom Cruise: "Trading Places", "Risky Busines...

### pii

**Sample 1**: `email` (value: 1)
> What is the format of an email? Emails contain a username, the @ symbol, and a website domain. The w...
