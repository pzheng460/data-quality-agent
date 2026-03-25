# Data Quality Report: dolma3_mix-6T

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | pretrain |
| Avg word count | 1675.2 |
| Min word count | 1 |
| Max word count | 32325 |
| Avg word length | 4.42 |
| Fields | text |
| Exact duplicates | 1 (0.1%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 690 |
| Failed | 310 |
| Average pass rate (per filter) | 87.2% |
| Overall pass rate (pass all) | 69.0% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 80.4% | 196 | 1000 |
| **gopher_repetition** | 94.0% | 60 | 1000 |
| **c4** | 89.2% | 108 | 1000 |
| **fineweb** | 75.5% | 245 | 1000 |
| **pii** | 96.9% | 31 | 1000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 134 | 1000 | 13.40% |
| alpha_ratio | 114 | 1000 | 11.40% |
| stopwords | 70 | 1000 | 7.00% |
| min_words | 69 | 1000 | 6.90% |
| min_avg_word_len | 9 | 1000 | 0.90% |
| max_avg_word_len | 2 | 1000 | 0.20% |
| hash_ratio | 1 | 1000 | 0.10% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| top_2gram | 52 | 1000 | 5.20% |
| top_3gram | 47 | 1000 | 4.70% |
| top_4gram | 33 | 1000 | 3.30% |
| dup_para_ratio | 1 | 1000 | 0.10% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 88 | 1000 | 8.80% |
| min_sentences | 20 | 1000 | 2.00% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| line_punct_ratio | 146 | 1000 | 14.60% |
| short_line_ratio | 92 | 1000 | 9.20% |
| char_dup_ratio | 83 | 1000 | 8.30% |
| list_ratio | 13 | 1000 | 1.30% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 29 | 1000 | 2.90% |
| ip | 2 | 1000 | 0.20% |
| bank_card | 1 | 1000 | 0.10% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `min_words` (value: 1)
> 5...

**Sample 2**: `alpha_ratio` (value: 0.7918)
> tagNovels and NovellasPleasant Street Ch. 07 Pt. 02-03  Pleasant Street Ch. 07 Pt. 02-03  byD.C. Roi...

**Sample 3**: `lines_end_punct` (value: 0.0)
> Rating: 4 out of 5 by 49 visitors  Yayoi inami  Delicious Yayoi Inami scream from enjoyment during o...

### gopher_repetition

**Sample 1**: `top_2gram` (value: 0.9333)
> Karen Kamensek ...

**Sample 2**: `top_2gram` (value: 1.0)
> withholding order...

**Sample 3**: `top_2gram` (value: 0.5455)
> four + 13 =...

### c4

**Sample 1**: `empty_after_line_filter` (value: no_terminal_punct(1))
> 5...

**Sample 2**: `empty_after_line_filter` (value: no_terminal_punct(6))
> Rating: 4 out of 5 by 49 visitors  Yayoi inami  Delicious Yayoi Inami scream from enjoyment during o...

**Sample 3**: `min_sentences` (value: sentences=1 after removing no_terminal_punct_removed(3))
> Nice to Nasty  Nice to Naughty  “I know some people might not believe this coz they’re about to watc...

### fineweb

**Sample 1**: `line_punct_ratio` (value: 0.0)
> 5...

**Sample 2**: `line_punct_ratio` (value: 0.0)
> Rating: 4 out of 5 by 49 visitors  Yayoi inami  Delicious Yayoi Inami scream from enjoyment during o...

**Sample 3**: `line_punct_ratio` (value: 0.0)
> Karen Kamensek ...

### pii

**Sample 1**: `email` (value: 1)
>  Submitted by: Chastity Novice's Keyholder  Recently you received a report from my husband under the...

**Sample 2**: `email` (value: 1)
> Aneros Progasm Prostate Massager  Aneros Progasm Prostate Massager  Precio habitual $59.99   • En st...

**Sample 3**: `email` (value: 1)
> Top Definition A town in Gippsland, Victoria, Australia. Where the water is horrible and lots of der...
