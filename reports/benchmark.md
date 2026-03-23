# Data Quality Report: dolma3_mix-6T

**Samples**: 10000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 10000 |
| Data type | pretrain |
| Avg word count | 1612.5 |
| Min word count | 1 |
| Max word count | 48556 |
| Avg word length | 4.64 |
| Fields | text |
| Exact duplicates | 78 (0.8%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 10000 |
| Passed | 8557 |
| Failed | 1443 |
| Overall pass rate | 85.6% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 86.8% | 1321 | 10000 |
| **gopher_repetition** | 99.7% | 30 | 8679 |
| **c4** | 99.4% | 50 | 8649 |
| **fineweb** | 99.5% | 42 | 8599 |
| **pii** | 100.0% | 0 | 8557 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 1248 | 10000 | 12.48% |
| min_words | 691 | 10000 | 6.91% |
| stopwords | 555 | 10000 | 5.55% |
| alpha_ratio | 294 | 10000 | 2.94% |
| min_avg_word_len | 71 | 10000 | 0.71% |
| max_avg_word_len | 24 | 10000 | 0.24% |
| symbol_ratio | 1 | 10000 | 0.01% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| top_2gram | 383 | 10000 | 3.83% |
| top_3gram | 321 | 10000 | 3.21% |
| top_4gram | 216 | 10000 | 2.16% |
| dup_line_ratio | 38 | 10000 | 0.38% |
| dup_para_ratio | 25 | 10000 | 0.25% |
| dup_6gram_frac | 9 | 10000 | 0.09% |
| dup_5gram_frac | 7 | 10000 | 0.07% |
| dup_10gram_frac | 5 | 10000 | 0.05% |
| dup_8gram_frac | 3 | 10000 | 0.03% |
| dup_9gram_frac | 3 | 10000 | 0.03% |
| dup_7gram_frac | 3 | 10000 | 0.03% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 836 | 10000 | 8.36% |
| min_sentences | 129 | 10000 | 1.29% |
| lorem_ipsum | 3 | 10000 | 0.03% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_line_ratio | 38 | 10000 | 0.38% |
| bad_line_breaks | 33 | 10000 | 0.33% |
| list_line_ratio | 31 | 10000 | 0.31% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 174 | 10000 | 1.74% |
| ip | 6 | 10000 | 0.06% |
| bank_card | 3 | 10000 | 0.03% |
| cn_phone | 3 | 10000 | 0.03% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `too_few_words` (value: 1)
> 5...

**Sample 2**: `low_terminal_punct` (value: 0.0)
> Rating: 4 out of 5 by 49 visitors  Yayoi inami  Delicious Yayoi Inami scream from enjoyment during o...

**Sample 3**: `too_few_words` (value: 2)
> Karen Kamensek ...

### gopher_repetition

**Sample 1**: `high_dup_line_ratio` (value: 0.5)
> Sex Tape 2014  Sex Tape 2014  Action , Comedy ,When Jay (Jason Segel) and Annie (Cameron Diaz) first...

**Sample 2**: `high_dup_line_ratio` (value: 0.3333)
> I had an amazing and intensive time with Bianca. Thank you. starstarstar starstar_border Amazing ser...

**Sample 3**: `high_dup_line_ratio` (value: 0.4)
> Chat gratuit with Ninasa Pictures lj webcam Ninasa Pictures lj webcam Ninasa.  Hello! I'm Nina from ...

### c4

**Sample 1**: `too_few_sentences` (value: 2)
> Nice to Nasty  Nice to Naughty  “I know some people might not believe this coz they’re about to watc...

**Sample 2**: `too_few_sentences` (value: 2)
>  PASSIONAL Boutique no longer sells sexuality products in the store or on our website. As of Septemb...

**Sample 3**: `too_few_sentences` (value: 1)
> Piss in his white panties  Click to this video! Gay guy sits half naked on the floor in front of the...

### fineweb

**Sample 1**: `bad_line_breaks` (value: 28.0)
> 03:39 Gays Jake and Scott .. 04:07 CBT Monster Dicked S.. 05:02 Gay muscled jock dic.. 05:13 Hunky j...

**Sample 2**: `bad_line_breaks` (value: 28.9286)
> tagErotic PoetryJack and Jill!  Jack and Jill!   But not to fetch no water! To do what they thought ...

**Sample 3**: `list_document` (value: 1.0)
> 1. ensure you’re both for a passing fancy page. if your wanting to also think about hooking up, make...
