# Data Quality Benchmark: Alpaca Original vs Alpaca Cleaned

**Samples per dataset**: all
**Config**: `configs/sft.yaml`

| Filter | Alpaca Original | Alpaca Cleaned | Δ | Verdict |
|--------|--------|--------|--------|--------|
| length | 98.7% | 99.6% | +0.9% | — No signal |
| gopher_quality | 47.7% | 66.2% | +18.4% | ✅ Discriminates |
| gopher_repetition | 22.7% | 13.5% | +9.2% | ✅ Discriminates |
| fineweb | 99.7% | 98.8% | +0.9% | — No signal |
| pii | 100.0% | 100.0% | +0.0% | — No signal |
| **Overall pipeline** | **10.7%** | **8.8%** | **+1.9%** | **— No signal** |

### Legend

- ✅ = Cleaned version passes significantly more (>5%) → filter catches real issues
- ⚠️ = Small difference (2-5%) → filter has weak signal for this data type
- — = No meaningful difference (<2%) → filter not relevant for SFT data

### Sample failures from Alpaca Original

Documents that **fail** in the original dataset (issues the cleaning addressed):

#### gopher_quality — failed 26822/51328

1. `Classify the following phrase "I am so cold": I am so cold This phrase is an expression of discomfort....`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=19`
2. `Make a list of 3 things to do to preserve the environment. - Reduce water and energy usage - Recycle waste - Plant trees and other plants to build gre...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=31`
3. `Transform the sentence so that it uses direct quotations. The speaker said that education was important. The speaker said, “Education is important.”...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=22`

#### gopher_repetition — failed 18942/24506

1. `Describe an example of a time you used influence in a positive way I recently had a team project at work where I had to influence my team members to c...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=0.32934131736526945`
2. `Examine the differences between an LLC and a C-corporation. An LLC and a C-corporation are two different legal structures used for businesses.   LLCs ...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
3. `Generate a single sentence that summarizes the effects of the given policy. Policy: Tax credits for businesses investing in renewable energy Tax credi...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
