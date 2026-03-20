# Data Quality Benchmark: Alpaca Original vs Alpaca Cleaned

**Samples per dataset**: 1000

| Filter | Alpaca Original | Alpaca Cleaned | Δ | Verdict |
|--------|--------|--------|--------|--------|
| gopher_quality | 44.7% | 59.6% | +14.9% | ✅ Discriminates |
| gopher_repetition | 32.2% | 19.3% | +12.9% | ✅ Discriminates |
| c4 | 93.1% | 94.8% | +1.7% | — No signal |
| fineweb | 97.0% | 96.3% | +0.7% | — No signal |
| pii | 100.0% | 100.0% | +0.0% | — No signal |
| **Overall pipeline** | **13.0%** | **10.5%** | **+2.5%** | **⚠️ Weak signal** |

### Legend

- ✅ = Cleaned version passes significantly more (>5%) → filter catches real issues
- ⚠️ = Small difference (2-5%) → filter has weak signal for this data type
- — = No meaningful difference (<2%) → filter not relevant for SFT data

### Sample failures from Alpaca Original

Documents that **fail** in the original dataset (issues the cleaning addressed):

#### gopher_quality — failed 553/1000

1. `Classify the following phrase "I am so cold": I am so cold This phrase is an expression of discomfort....`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=19`
2. `Make a list of 3 things to do to preserve the environment. - Reduce water and energy usage - Recycle waste - Plant trees and other plants to build gre...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=31`
3. `Transform the sentence so that it uses direct quotations. The speaker said that education was important. The speaker said, “Education is important.”...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=22`

#### gopher_repetition — failed 303/447

1. `Examine the differences between an LLC and a C-corporation. An LLC and a C-corporation are two different legal structures used for businesses.   LLCs ...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
2. `Generate a single sentence that summarizes the effects of the given policy. Policy: Tax credits for businesses investing in renewable energy Tax credi...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
3. `How can cities become more eco-friendly? Cities can become more eco-friendly by encouraging public transportation, promoting green building standards ...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=0.6451612903225806`
