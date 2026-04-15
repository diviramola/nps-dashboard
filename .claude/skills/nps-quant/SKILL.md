---
name: nps-quant
description: "Quantitative NPS analysis for Wiom. Use this skill whenever the user asks about NPS scores, NPS trends, NPS by segment (tenure, city, device, channel, sprint), score distributions, promoter/detractor ratios, statistical significance of NPS changes, sample sizes, confidence intervals, or any numerical NPS question. Also trigger when they say things like 'what's the NPS', 'compare sprints', 'NPS by channel', 'is this drop significant', 'show me the numbers', 'how are we trending', 'which segment moved', 'sprint over sprint', 'monthly NPS', 'quarterly trend', or any request involving NPS data slicing, filtering, or aggregation. Even if the user just says 'NPS' or 'scores' in the context of Wiom data, use this skill."
---

# Wiom NPS Quantitative Analysis

This skill answers numerical NPS questions by reading sprint CSVs from `data/sprints/` in the wiom-nps-analysis project.

## Data Location

Sprint CSVs live in `data/sprints/` (17 files: sprint_1.csv through sprint_14.csv, sprint_rsp1.csv through sprint_rsp3.csv). Each has 12 columns: respondent_id, score, feedback, nps_reason_primary, nps_reason_secondary, primary_category, plan_type, city, tenure_days, source, sprint_id, sprint_start.

Enrichment sidecars (if available) live in `data/enrichment/` as JSON files containing city, device, and tenure data from Metabase.

## How to answer NPS questions

### Step 1: Load the data

Read the relevant sprint CSV(s) using Python (pandas) or by parsing directly. Handle:
- UTF-8 BOM encoding (files may start with `\uFEFF`)
- Score column is required; skip rows where score is non-numeric
- Clamp scores to [0, 10]
- Strip ".0" suffix from respondent_id (Excel artifact)
- Treat city values "#REF!" and "unknown" as empty

### Step 2: Classify and compute

**Score bands:**
- Promoter: score 9–10
- Passive: score 7–8
- Detractor: score 0–6

**NPS formula:**
```
NPS = ((promoters - detractors) / total) * 100
```
Round to 1 decimal place. Range: [-100, +100].

**Tenure bands** (3-tier, from install date as day 1):
- Early (1–2 mo): tenure_days 1–60
- Mid (3–6 mo): tenure_days 61–180
- Long (6+ mo): tenure_days 181+
- Unknown: tenure_days = 0 or missing (Sprints 13–14 and RSP1–3 have this gap; note it when relevant)

**Source normalization** (case-insensitive):
- WA, In - WA, G- form in WA → WhatsApp
- CT, CLEVERTAP → CleverTap
- CALL → Call

### Step 3: Always include confidence

Every NPS number must come with sample size and a 95% confidence interval. This is non-negotiable — a number without context is misleading.

**Multinomial variance of NPS estimator:**
```
Var(NPS) = [P_pro(1 - P_pro) + P_det(1 - P_det) + 2 * P_pro * P_det] / n
SE = sqrt(Var) * 100
CI = NPS ± 1.96 * SE
```

**Confidence level** (based on CI width and sample size):

| Condition | Level | What to say |
|-----------|-------|-------------|
| n ≥ 100 AND CI width < 15 | High | Report normally |
| n ≥ 30 AND CI width < 25 | Medium | Note "medium confidence" |
| Otherwise | Low | Caveat: "Directional only — small sample (n=X)" |

### Step 4: For comparisons, always run a significance test

Never claim a "drop" or "improvement" between sprints without a p-value. Use the two-proportion z-test:

```
var1 = [P_pro1(1-P_pro1) + P_det1(1-P_det1) + 2*P_pro1*P_det1] / n1
var2 = [P_pro2(1-P_pro2) + P_det2(1-P_det2) + 2*P_pro2*P_det2] / n2
SE_diff = sqrt(var1 + var2) * 100
z = abs(delta) / SE_diff
p-value (two-tailed) = 2 * (1 - Phi(z))
```

At 95% confidence (p < 0.05):
- Significant → state direction ("improved by X pts, p=Y")
- Not significant → "changed by X pts (p=Y) — not statistically significant"

**Direction classification** for editorial language:
- Delta > +2: "improving"
- Delta < -2: "declining"
- ±2: "stable"

## Sprint Timeline

Use these labels consistently:

| Sprint | Month Label | Month | Quarter |
|--------|------------|-------|---------|
| Sprint 1 | Jul-1H '25 | Jul '25 | Q1 FY26 (Jul-Sep '25) |
| Sprint 2 | Jul-2H '25 | Jul '25 | Q1 FY26 |
| Sprint 3 | Aug-1H '25 | Aug '25 | Q1 FY26 |
| Sprint 4 | Aug-2H '25 | Aug '25 | Q1 FY26 |
| Sprint 5 | Sep-1H '25 | Sep '25 | Q1 FY26 |
| Sprint 6 | Sep-2H '25 | Sep '25 | Q1 FY26 |
| Sprint 7 | Oct-1H '25 | Oct '25 | Q2 FY26 (Oct-Dec '25) |
| Sprint 8 | Oct-2H '25 | Oct '25 | Q2 FY26 |
| Sprint 9 | Nov-1H '25 | Nov '25 | Q2 FY26 |
| Sprint 10 | Nov-2H '25 | Nov '25 | Q2 FY26 |
| Sprint 11 | Dec-1H '25 | Dec '25 | Q2 FY26 |
| Sprint 12 | Dec-2H '25 | Dec '25 | Q2 FY26 |
| Sprint 13 | Jan-1H '26 | Jan '26 | Q3 FY26 (Jan-Mar '26) |
| Sprint 14 | Jan-2H '26 | Jan '26 | Q3 FY26 |
| RSP1 | Feb-1H '26 | Feb '26 | Q3 FY26 |
| RSP2 | Feb-2H '26 | Feb '26 | Q3 FY26 |
| RSP3 | Mar-1H '26 | Mar '26 | Q3 FY26 |

**Sprint ordering**: numbered sprints (1–14) sort by number, then RSP sprints (RSP1–3) by their number.

**Time aggregation**: When the user asks for monthly or quarterly views, group sprints accordingly (two sprints per month, six per quarter) and compute NPS on the combined records.

## Available Segments

These are the filter dimensions the user can ask about:

- **Channel/Source**: WhatsApp, CleverTap, Call
- **Score band**: promoter, passive, detractor
- **Tenure**: Early (1–2 mo), Mid (3–6 mo), Long (6+ mo) — note: unavailable for Sprints 13+ without Metabase enrichment
- **Sprint**: Sprint 1–14, RSP1–RSP3
- **City**: Delhi, UP, Mumbai, Unknown — note: requires Metabase enrichment (empty in raw CSVs)
- **Device**: Not in CSVs — requires Metabase enrichment
- **Time period**: Sprint (biweekly), Month, Quarter

When a segment dimension has no data for the selected sprint(s), say so explicitly rather than silently omitting it.

## Output Format

Structure your answer clearly:

1. **Headline number** with CI and sample size
2. **Breakdown table** if multiple segments
3. **Significance test** if comparing periods
4. **Caveats** — always flag: small samples (n < 30), missing data (tenure/city gaps), right-censoring for recent sprints

Example format for a sprint comparison:
```
Sprint RSP3 (Mar-1H '26): NPS +2.3 [CI: -4.1, +8.7] (n=904, medium confidence)
Sprint RSP2 (Feb-2H '26): NPS -1.8 [CI: -8.5, +4.9] (n=777, medium confidence)
Change: +4.1 pts (p=0.312) — not statistically significant
```

## Data Quality Notes

Keep these in mind when interpreting results:
- Score 0 accounts for ~20% of responses (unusually high — likely a survey UX issue where 0 is the default)
- Overall NPS hovers near 0.0 (46% promoters, 46% detractors, 8% passive)
- Sprint 1 has no feedback or reason data at all (quantitative scores only)
- Churn data is not in the sprint CSVs (lives in the consolidated analysis dataset)
- ~42% of responses have open-text feedback; detractors comment 1.3x more than promoters
