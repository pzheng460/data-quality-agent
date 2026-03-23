# Data Quality Report: openwebtext

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | pretrain |
| Avg word count | 822.2 |
| Min word count | 128 |
| Max word count | 14787 |
| Avg word length | 4.98 |
| Fields | text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 907 |
| Failed | 93 |
| Average pass rate (per filter) | 97.6% |
| Overall pass rate (pass all) | 90.7% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 97.9% | 21 | 1000 |
| **gopher_repetition** | 97.7% | 23 | 1000 |
| **c4** | 97.6% | 24 | 1000 |
| **fineweb** | 99.5% | 5 | 1000 |
| **pii** | 95.4% | 46 | 1000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| lines_end_punct | 19 | 1000 | 1.90% |
| alpha_ratio | 3 | 1000 | 0.30% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_5gram_frac | 17 | 1000 | 1.70% |
| dup_6gram_frac | 17 | 1000 | 1.70% |
| dup_9gram_frac | 16 | 1000 | 1.60% |
| dup_10gram_frac | 16 | 1000 | 1.60% |
| dup_7gram_frac | 15 | 1000 | 1.50% |
| dup_8gram_frac | 15 | 1000 | 1.50% |
| dup_line_ratio | 5 | 1000 | 0.50% |
| dup_para_ratio | 5 | 1000 | 0.50% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| empty_after_line_filter | 17 | 1000 | 1.70% |
| min_sentences | 7 | 1000 | 0.70% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_line_ratio | 5 | 1000 | 0.50% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 45 | 1000 | 4.50% |
| bank_card | 1 | 1000 | 0.10% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `lines_end_punct` (value: 0.0769)
> Get cool in-game extras with amiibo accessories! Just tap to score new characters, game modes, or ot...

**Sample 2**: `lines_end_punct` (value: 0.0)
> No. 1 - Retired for Billy Martin Earle Combs 1929-1935 George Selkirk 1934 Roy Johnson 1936 Frank Cr...

**Sample 3**: `lines_end_punct` (value: 0.0588)
> At the sacred convocation of Concordia Lutheran Seminary, Edmonton (25 May), and the Call Service of...

### gopher_repetition

**Sample 1**: `dup_5gram_frac` (value: 0.175)
> Today, Toyota announced changes in executives’ areas of responsibility, as well as personnel changes...

**Sample 2**: `dup_5gram_frac` (value: 0.4216)
> EU politics: renewing the wedding vows 19/01/2015  Follow @eureferendum  Jean-Claude Juncker says he...

**Sample 3**: `dup_line_ratio` (value: 1.0)
> If you live abroad and are requesting an ITIN for a foreign child who has been adopted or legally pl...

### c4

**Sample 1**: `empty_after_line_filter` (value: no_terminal_punct(2))
> No. 1 - Retired for Billy Martin Earle Combs 1929-1935 George Selkirk 1934 Roy Johnson 1936 Frank Cr...

**Sample 2**: `empty_after_line_filter` (value: javascript(1), no_terminal_punct(1))
> Caution: JavaScript execution is disabled in your browser or for this website. You may not be able t...

**Sample 3**: `min_sentences` (value: sentences=2 after removing no_terminal_punct_removed(4))
> Lea Michele broke the news on Monday that she would be appearing on the seventh and final season of ...

### fineweb

**Sample 1**: `dup_line_ratio` (value: 1.0)
> If you live abroad and are requesting an ITIN for a foreign child who has been adopted or legally pl...

**Sample 2**: `dup_line_ratio` (value: 0.36)
> It was Dec. 1, 2013. With the score at 7-7 in the second quarter, the Philadelphia Eagles marched to...

**Sample 3**: `dup_line_ratio` (value: 0.4442)
> Document number: N2271=07-0131 Date: 2007-04-27 Reply to: Paul Pedriana  Electronic Arts  ppedriana ...

### pii

**Sample 1**: `email` (value: 1)
> He was a few blocks from home, waiting for a bus in the cold, checking emails on his phone, when Cou...

**Sample 2**: `email` (value: 1)
> It’s a well-kept secret, but 95% of the climate models we are told prove the link between human CO₂ ...

**Sample 3**: `email` (value: 1)
> Two of the three investigations into the actions of Salt Lake City Police officers Detective Jeff Pa...
