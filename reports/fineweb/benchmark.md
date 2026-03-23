# Data Quality Report: fineweb

**Samples**: 1000

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Data type | pretrain |
| Avg word count | 510.9 |
| Min word count | 31 |
| Max word count | 19419 |
| Avg word length | 4.94 |
| Fields | text |
| Exact duplicates | 0 (0.0%) |

## Quality Summary

| Metric | Value |
|--------|-------|
| Total documents | 1000 |
| Passed (all filters) | 908 |
| Failed | 92 |
| Average pass rate (per filter) | 98.0% |
| Overall pass rate (pass all) | 90.8% |

## Per-Filter Statistics

| Filter | Pass Rate | Failed | Total |
|--------|-----------|--------|-------|
| **gopher_quality** | 98.5% | 15 | 1000 |
| **gopher_repetition** | 99.2% | 8 | 1000 |
| **c4** | 97.1% | 29 | 1000 |
| **fineweb** | 99.6% | 4 | 1000 |
| **pii** | 95.4% | 46 | 1000 |

## Per-Rule Breakdown

### gopher_quality

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| min_words | 9 | 1000 | 0.90% |
| lines_end_punct | 4 | 1000 | 0.40% |
| alpha_ratio | 2 | 1000 | 0.20% |

### gopher_repetition

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| dup_5gram_frac | 3 | 1000 | 0.30% |
| dup_10gram_frac | 2 | 1000 | 0.20% |
| dup_6gram_frac | 2 | 1000 | 0.20% |
| dup_7gram_frac | 2 | 1000 | 0.20% |
| dup_9gram_frac | 1 | 1000 | 0.10% |
| dup_line_ratio | 1 | 1000 | 0.10% |

### c4

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| min_sentences | 29 | 1000 | 2.90% |

### fineweb

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| list_line_ratio | 2 | 1000 | 0.20% |
| bad_line_breaks | 1 | 1000 | 0.10% |
| dup_line_ratio | 1 | 1000 | 0.10% |

### pii

| Rule | Failed | Total | Fail Rate |
|------|--------|-------|-----------|
| email | 45 | 1000 | 4.50% |
| ip | 1 | 1000 | 0.10% |

## Sample Dropped Documents

### gopher_quality

**Sample 1**: `min_words` (value: 49)
> Subject: Binary updates all installed pkgs? To: None <tech-pkg@NetBSD.org> From: Mark Weinem <firstn...

**Sample 2**: `lines_end_punct` (value: 0.0909)
> [amsat-bb] Re: Dish surface 'flatness' antonio at qualcomm.com Fri Jul 25 21:35:16 PDT 2008 At 03:00...

**Sample 3**: `lines_end_punct` (value: 0.0)
> SCOTTSDALE -- The Arizona Fall League has long been recognized as Major League Baseball's "graduate"...

### gopher_repetition

**Sample 1**: `dup_9gram_frac` (value: 0.1157)
> Reprinted from the renowned Bach-Gesellschaft edition, this work features the complete Sonatas and P...

**Sample 2**: `dup_5gram_frac` (value: 0.1642)
> Anaconda iSCSI Cleanup Improve anaconda's iSCSI support. This is a complete iscsi rewrite that also ...

**Sample 3**: `dup_7gram_frac` (value: 0.1381)
> Post Number: 5136 |Posted on Saturday, February 16, 2008 - 3:42 pm: | It might be hard to tell on yo...

### c4

**Sample 1**: `min_sentences` (value: sentences=2 after removing no_terminal_punct_removed(1))
> The Net Neutrality repeal vote is coming. Tell these Dems to vote Yes. The House of Representatives ...

**Sample 2**: `min_sentences` (value: sentences=2 after removing no_terminal_punct_removed(9))
> Under Armour® Camo Back Cap - Embroidered UA logo up front - UA wordmark on cap bill - Embroidered U...

**Sample 3**: `min_sentences` (value: sentences=2 after removing no_terminal_punct_removed(5))
> Welcome to my official blog. I am a professional photographer based in London / Cape Town specialisi...

### fineweb

**Sample 1**: `bad_line_breaks` (value: 28.8571)
> You must be a registered member to view this page.| If you are already a member, sign in now. To reg...

**Sample 2**: `list_line_ratio` (value: 1.0)
> 33. Letter from Wash Swimmer, acting chief, to National Council, regarding Dawes Commission correspo...

**Sample 3**: `list_line_ratio` (value: 0.95)
> I love the buttery taste and the crunchiness of the coarse sugar and cornmeal in this tart. - 1/2 cu...

### pii

**Sample 1**: `email` (value: 1)
> Exhibition will provide a meeting place for businesses interested in capitalizing on India's waste m...

**Sample 2**: `email` (value: 2)
> Subject: Binary updates all installed pkgs? To: None <tech-pkg@NetBSD.org> From: Mark Weinem <firstn...

**Sample 3**: `email` (value: 3)
> |general shipping policy||shipping to a different address||international shipping||using the shoppin...
