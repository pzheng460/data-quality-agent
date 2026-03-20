# Data Quality Benchmark: FineWeb vs Alpaca 52K

**Samples per dataset**: 1000

| Filter | FineWeb | Alpaca 52K | Δ | Verdict |
|--------|--------|--------|--------|--------|
| gopher_quality | 98.5% | 45.8% | +52.7% | ✅ Discriminates |
| gopher_repetition | 20.0% | 14.6% | +5.4% | ✅ Discriminates |
| c4 | 90.9% | 92.5% | +1.7% | — No signal |
| fineweb | 100.0% | 100.0% | +0.0% | — No signal |
| pii | 100.0% | 100.0% | +0.0% | — No signal |
| **Overall pipeline** | **17.9%** | **6.2%** | **+11.7%** | **✅ Validated** |

### Legend

- ✅ = Cleaned version passes significantly more (>5%) → filter catches real issues
- ⚠️ = Small difference (2-5%) → filter has weak signal for this data type
- — = No meaningful difference (<2%) → filter not relevant for SFT data

### Sample failures from FineWeb

Documents that **fail** in the original dataset (issues the cleaning addressed):

#### gopher_quality — failed 15/1000

1. `Abby, Sadie (E, F, G) Two little girls, one big wry streak. Saturday, January 22, 2011 Mom's shoes are hard to fill. Maybe I should get out the Play-D...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=48`
2. `by Shelley Woodward (Dunsborough, Western Australia.) York City versus, Aston Villa, first ever match in Division two 17th August 1974. Perfect condit...`
   - Reason: `filter=gopher_quality`, `reason=too_few_words`, `value=49`
3. `International Monetary Fund (21 - 30 of 58 items) Developing Countries: Challenges Confronting Debt Relief and IMF Lending to Poor Countries GAO-01-74...`
   - Reason: `filter=gopher_quality`, `reason=low_alpha_ratio`, `value=0.7960812772133526`

#### gopher_repetition — failed 788/985

1. `Radisson Blu Dubai Deira Creek Unveils New Renovations The new designs bring a fresh look to the heritage property, which opened in 1974 as the first ...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
2. `It's the final week of our fabulous Coach giveaways for PopSugar Daily. This week baby needs a new pair of shoes and now is your chance to enter to wi...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=0.5021520803443329`
3. `From the soil From the air It is a kind of plant. It is the food that the plant gets. It is a process by which plants reproduce. It is a process by wh...`
   - Reason: `filter=gopher_repetition`, `reason=high_char_repetition`, `value=1.0`
