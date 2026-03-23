# Data Quality Report

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total documents in | 1000 |
| Total documents out | 841 |
| Total dropped | 159 |
| Overall drop rate | 15.90% |

## Per-Filter Statistics

| Filter | Docs In | Docs Out | Dropped | Drop Rate |
|--------|---------|----------|---------|-----------|
| gopher_quality | 1000 | 858 | 142 | 14.20% |
| gopher_repetition | 858 | 855 | 3 | 0.35% |
| c4 | 855 | 848 | 7 | 0.82% |
| fineweb | 848 | 841 | 7 | 0.83% |
| pii | 841 | 841 | 0 | 0.00% |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 134 | 1000 | 13.40% |
| min_words | 78 | 1000 | 7.80% |
| stopwords | 68 | 1000 | 6.80% |
| alpha_ratio | 44 | 1000 | 4.40% |
| min_avg_word_len | 11 | 1000 | 1.10% |
| max_avg_word_len | 4 | 1000 | 0.40% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| top_2gram | 46 | 1000 | 4.60% |
| top_3gram | 33 | 1000 | 3.30% |
| top_4gram | 28 | 1000 | 2.80% |
| dup_line_ratio | 3 | 1000 | 0.30% |
| dup_para_ratio | 2 | 1000 | 0.20% |
| dup_5gram_frac | 1 | 1000 | 0.10% |
| dup_6gram_frac | 1 | 1000 | 0.10% |
| dup_8gram_frac | 1 | 1000 | 0.10% |
| dup_9gram_frac | 1 | 1000 | 0.10% |
| dup_10gram_frac | 1 | 1000 | 0.10% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 86 | 1000 | 8.60% |
| min_sentences | 17 | 1000 | 1.70% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| bad_line_breaks | 6 | 1000 | 0.60% |
| list_line_ratio | 5 | 1000 | 0.50% |
| dup_line_ratio | 3 | 1000 | 0.30% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 18 | 1000 | 1.80% |
| ip | 1 | 1000 | 0.10% |

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
