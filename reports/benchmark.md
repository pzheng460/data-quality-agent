# Data Quality Report: dolma3_mix-6T

**Samples**: 10000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 10000 |
| Data type | pretrain |
| Avg word count | 1641.5 |
| Min word count | 0 |
| Max word count | 49794 |
| Avg word length | 4.41 |
| Fields | text |
| Exact duplicates | 78 (0.8%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 10000 |
| Passed (all filters) | 7167 |
| Failed | 2833 |
| Average pass rate (per filter) | 88.4% |
| Overall pass rate (pass all) | 71.7% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 82.0% | 1798 | 10000 |
| **gopher_repetition** | 94.7% | 534 | 10000 |
| **c4** | 89.9% | 1014 | 10000 |
| **fineweb** | 78.5% | 2149 | 10000 |
| **pii** | 97.0% | 301 | 10000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 1248 | 10000 | 12.48% |
| alpha_ratio | 952 | 10000 | 9.52% |
| stopwords | 597 | 10000 | 5.97% |
| min_words | 595 | 10000 | 5.95% |
| min_avg_word_len | 72 | 10000 | 0.72% |
| max_avg_word_len | 12 | 10000 | 0.12% |
| hash_ratio | 4 | 10000 | 0.04% |
| ellipsis_ratio | 3 | 10000 | 0.03% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| top_2gram | 439 | 10000 | 4.39% |
| top_3gram | 420 | 10000 | 4.20% |
| top_4gram | 311 | 10000 | 3.11% |
| dup_para_ratio | 3 | 10000 | 0.03% |
| dup_5gram_frac | 3 | 10000 | 0.03% |
| dup_6gram_frac | 3 | 10000 | 0.03% |
| dup_line_ratio | 3 | 10000 | 0.03% |
| dup_7gram_frac | 2 | 10000 | 0.02% |
| dup_8gram_frac | 2 | 10000 | 0.02% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 856 | 10000 | 8.56% |
| min_sentences | 158 | 10000 | 1.58% |
| lorem_ipsum | 3 | 10000 | 0.03% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| line_punct_ratio | 1348 | 10000 | 13.48% |
| short_line_ratio | 779 | 10000 | 7.79% |
| char_dup_ratio | 677 | 10000 | 6.77% |
| list_ratio | 101 | 10000 | 1.01% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 275 | 10000 | 2.75% |
| ip | 12 | 10000 | 0.12% |
| bank_card | 12 | 10000 | 0.12% |
| cn_phone | 4 | 10000 | 0.04% |

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
