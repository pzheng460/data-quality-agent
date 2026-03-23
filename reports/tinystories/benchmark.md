# Data Quality Report: TinyStories

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | pretrain |
| Avg word count | 183.8 |
| Min word count | 61 |
| Max word count | 837 |
| Avg word length | 4.12 |
| Fields | text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 994 |
| Failed | 6 |
| Average pass rate (per filter) | 99.9% |
| Overall pass rate (pass all) | 99.4% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 99.6% | 4 | 1000 |
| **gopher_repetition** | 99.8% | 2 | 1000 |
| **c4** | 100.0% | 0 | 1000 |
| **fineweb** | 100.0% | 0 | 1000 |
| **pii** | 100.0% | 0 | 1000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 4 | 1000 | 0.40% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_5gram_frac | 2 | 1000 | 0.20% |
| dup_6gram_frac | 1 | 1000 | 0.10% |
| dup_10gram_frac | 1 | 1000 | 0.10% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `lines_end_punct` (value: 0.0)
> Once upon a time there was a mom and a child who liked to go for walks together. One day, when they ...

**Sample 2**: `lines_end_punct` (value: 0.0)
> Once upon a time, there was a little girl named Lily. She wanted to go on a trip with her family, so...

**Sample 3**: `lines_end_punct` (value: 0.0)
> Once upon a time, there was a little girl named Lily. She loved to bake cookies with her mom. One da...

### gopher_repetition

**Sample 1**: `dup_5gram_frac` (value: 0.1605)
> Sara was a smart girl who liked to read books and learn new things. She had many books in her room, ...

**Sample 2**: `dup_5gram_frac` (value: 0.2239)
> Tom and Sue are twins. They like to play with their toys and talk to each other. Sometimes they repe...
