# Data Quality Report: c4

**Samples**: 100

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 100 |
| Data type | pretrain |
| Avg word count | 342.8 |
| Min word count | 20 |
| Max word count | 4014 |
| Avg word length | 5.26 |
| Fields | text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 100 |
| Passed (all filters) | 83 |
| Failed | 17 |
| Average pass rate (per filter) | 96.0% |
| Overall pass rate (pass all) | 83.0% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 86.0% | 14 | 100 |
| **gopher_repetition** | 96.0% | 4 | 100 |
| **c4** | 100.0% | 0 | 100 |
| **fineweb** | 99.0% | 1 | 100 |
| **pii** | 99.0% | 1 | 100 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| min_words | 12 | 100 | 12.00% |
| alpha_ratio | 3 | 100 | 3.00% |
| stopwords | 3 | 100 | 3.00% |
| max_avg_word_len | 1 | 100 | 1.00% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_5gram_frac | 3 | 100 | 3.00% |
| top_3gram | 2 | 100 | 2.00% |
| dup_6gram_frac | 2 | 100 | 2.00% |
| top_4gram | 1 | 100 | 1.00% |
| top_2gram | 1 | 100 | 1.00% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| short_line_ratio | 1 | 100 | 1.00% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 1 | 100 | 1.00% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `min_words` (value: 29)
> Foil plaid lycra and spandex shortall with metallic slinky insets. Attached metallic elastic belt wi...

**Sample 2**: `min_words` (value: 40)
> I thought I was going to finish the 3rd season of the Wire tonight. But there was a commentary on ep...

**Sample 3**: `min_words` (value: 44)
> Pencarian FILM Untuk "Peace Breaker 2017" yuk mampir ke channel say.. Edges East provides the l.. A ...

### gopher_repetition

**Sample 1**: `top_3gram` (value: 0.2422)
> Embrace world class facilities at East Bourne Resort & Spa Shimla. Facilities at East Bourne Resort ...

**Sample 2**: `top_2gram` (value: 0.2102)
> Movers & Moving Companies in Randallsville, New York for every moving service .: Movers MAX :. Mover...

**Sample 3**: `dup_5gram_frac` (value: 0.2038)
> Pre-Owned, AWD Titanium 4dr Crossover, Ebony interior, Gas, 20(city)/27(highwa­y) mpg, Auto 6 speed,...

### fineweb

**Sample 1**: `short_line_ratio` (value: 0.8889)
> Pencarian FILM Untuk "Peace Breaker 2017" yuk mampir ke channel say.. Edges East provides the l.. A ...

### pii

**Sample 1**: `email` (value: 1)
> Farmington, CT, August 30, 2016 -- Many Americans realize they need long-term care insurance, but ba...
