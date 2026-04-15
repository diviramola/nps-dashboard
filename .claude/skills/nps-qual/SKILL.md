---
name: nps-qual
description: "Qualitative NPS analysis for Wiom — themes, verbatims, and sentiment from open-text responses. Use this skill whenever the user asks about NPS themes, what customers are saying, verbatim quotes, sentiment analysis, why detractors are unhappy, what promoters like, theme trends, theme comparisons, complaint patterns, or any qualitative NPS question. Also trigger for: 'what are people saying', 'show me quotes', 'why is NPS dropping', 'top complaints', 'theme analysis', 'customer feedback', 'open text', 'reasons for NPS', 'Hindi feedback', 'what changed in themes', 'emerging issues', 'verbatims', or any request to understand the why behind NPS scores. If the user asks why NPS moved or what's driving scores, use this skill alongside nps-quant."
---

# Wiom NPS Qualitative Analysis

This skill answers qualitative NPS questions — themes, verbatims, sentiment, and the "why" behind scores — using sprint CSV data from `data/sprints/` in the wiom-nps-analysis project.

## Data Sources

Two qualitative signals exist in the sprint CSVs:

1. **Pre-coded reasons** (`nps_reason_primary` column) — Human-tagged during survey. Available for most sprints (not Sprint 1). Reliable but coarse-grained.
2. **Open-text feedback** (`feedback` column) — Free-text in Hindi, Hinglish, or English. ~42% fill rate. Richer but needs interpretation.

Both are useful. Pre-coded reasons give you structured theme counts; open-text gives you the voice of the customer.

## 22-Theme Taxonomy

Use this keyword-based taxonomy for classifying open-text feedback. It was developed through bottom-up analysis of 5,488 comments across 13 sprints.

### Theme Definitions

| Theme Key | Label | What It Covers |
|-----------|-------|----------------|
| disconnection_frequency | Disconnection Frequency | Repeated drops, unstable connection, "baar baar band" |
| slow_speed | Slow Speed | Buffering, low throughput, "speed bahut slow" |
| good_speed | Good Speed | Fast, smooth, "tez hai" |
| internet_down_outage | Internet Down/Outage | Complete outage, server down, "net nahi chal raha" |
| range_coverage | Range/Coverage | Weak signal, dead zones, router distance issues |
| complaint_resolution_bad | Bad Complaint Resolution | Unresolved tickets, slow response, "complaint ka koi fayda nahi" |
| complaint_resolution_good | Good Complaint Resolution | Fast fix, responsive support |
| call_center_bad | Bad Call Centre | Can't reach support, unhelpful agents |
| call_center_good | Good Call Centre | Helpful, responsive agents |
| technician_partner_bad | Bad Technician/Partner | Late, rude, didn't show up |
| technician_partner_good | Good Technician/Partner | Quick install, polite, helpful |
| pricing_expensive | Pricing — Expensive | "Mehnga hai", rate too high |
| pricing_affordable | Pricing — Affordable | Good value, "budget mein hai" |
| billing_28day | 28-Day Billing | The 28-day vs calendar month issue — a known pain point |
| rdni_recharge_no_internet | RDNI | Recharge done but no internet — a specific failure mode |
| general_positive | General Positive | "Good", "badhiya", non-specific praise |
| general_negative | General Negative | "Bekar", "worst", non-specific complaints |
| app_tech_issues | App/Tech Issues | App crashes, login problems |
| ott_content | OTT/Content | Streaming requests, TV complaints |
| shifting_relocation | Shifting/Relocation | Can't shift connection to new address |
| router_device | Router/Device | Hardware issues, router restarts |
| autopay_payment_mode | Autopay/Payment | UPI, autopay, payment mode issues |
| competitor_comparison | Competitor Comparison | Mentions Jio, Airtel, BSNL |

### Pre-coded Reason → Theme Mapping

The `nps_reason_primary` column values map to themes as follows:

| nps_reason_primary (from CSV) | Theme |
|-------------------------------|-------|
| Slow Speed | speed_quality |
| Good Speed | speed_quality |
| Frequent Disconnections | reliability |
| Internet Down | reliability |
| RDNI (Recharge Done, No Internet) | reliability |
| Affordable | pricing_value |
| Expensive | pricing_value |
| 28-Day Plan Issue | pricing_value |
| No/Late Complaint Resolution | customer_support |
| No/ Late Complaint Resolution | customer_support |
| Fast Complaint Resolution | customer_support |
| Bad Customer Support - Call Centre | customer_support |
| Good Customer Support - Call Centre | customer_support |
| Bad Customer Support - Technician | technician_service |
| Range Issue | network_coverage |
| General Good Service | general_satisfaction |
| General Bad Service | general_satisfaction |
| Bad Application | content_usage |
| OTT Request, No OTT complaint | content_usage |
| Shifting not Feasible | installation |

Use this mapping to convert pre-coded reasons to themes for analysis.

## How to Answer Theme Questions

### Step 1: Load and classify

Read the relevant sprint CSV(s). For each record with a non-empty `nps_reason_primary`, map it to a theme using the table above. For records without a pre-coded reason but with `feedback` text, you can use keyword matching against the 22-theme taxonomy.

### Step 2: Compute theme distributions

Report themes as both absolute count and percentage of total responses (not just percentage of themed responses — the denominator matters).

```
Theme: Speed & Quality — 143 mentions (19.2% of 745 responses)
  Of which: 89 detractors (62%), 12 passive (8%), 42 promoters (29%)
  Avg NPS score for this theme: 4.2
```

### Step 3: Always cite the source

Every theme claim must include:
- Which sprint(s) the data comes from
- Sample size (n)
- Whether the classification is from pre-coded reasons (reliable, human-tagged) or keyword matching (approximate)

## Selecting Verbatims

When the user asks for quotes, customer voice, or "what are people saying," select verbatims using this evocativeness scoring:

**Evocativeness score** = min(text_length, 300) + |score − 5| × 20

This prioritizes longer, more detailed responses AND extreme scores (0–2 or 9–10) where the signal is strongest. A long, passionate detractor rant is more useful than a one-word "bad."

### Verbatim Selection Rules

1. **Never fabricate or paraphrase** — use the exact text from the `feedback` column
2. **Preserve original language** — Hindi/Hinglish is the authentic voice; add a brief English translation in parentheses only if needed for comprehension
3. **Show both sides** of a theme — if a theme has positive and negative instances (e.g., speed_quality has both "Good Speed" and "Slow Speed"), show both
4. **Include metadata** with each verbatim: NPS score, score band, theme, sprint, channel
5. **Aim for 3–5 per theme** when doing a theme deep-dive; 1–2 per theme when doing an overview

### Verbatim Format

```
[NPS 0 | Detractor | Reliability | Sprint RSP3 | WhatsApp]
"Roj net ruk kar chalta h or kabhi nhi chalta bhut bekar h"
(Internet stutters daily and sometimes doesn't work at all — very bad)

[NPS 10 | Promoter | Speed & Quality | Sprint 14 | CleverTap]
"Mere budget me hai. Mere liye to sabse best hai ye... Good work..."
(It's within my budget. For me, this is the best... Good work...)
```

## Theme Comparison Between Sprints

When comparing themes across sprints, compute the percentage-point change and classify:

| Condition | Label |
|-----------|-------|
| Theme in current but not previous | **Emerging** — new issue |
| Theme in previous but not current | **Resolved** — went away |
| Current count > previous × 1.2 | **Growing** — getting worse |
| Current count < previous × 0.8 | **Shrinking** — improving |
| Otherwise | **Stable** |

Report as: "Reliability complaints grew from 12.3% to 17.8% (+5.5pp) — this is a growing issue."

## Language Handling

The feedback data is multilingual:
- ~60% Hinglish (romanized Hindi mixed with English)
- ~25% English
- ~10% Hindi (Devanagari script)
- ~5% Mixed

When presenting verbatims or analyzing text patterns, preserve the original language. The authenticity matters — "bekar hai" hits differently than "it's useless."

### Common Hindi/Hinglish Terms in Feedback

| Term | Meaning | Context |
|------|---------|---------|
| bekar | useless/terrible | Strong negative |
| badhiya | excellent | Strong positive |
| dikkat | problem/issue | Neutral complaint marker |
| mehnga/mahanga | expensive | Pricing complaint |
| sasta | cheap/affordable | Pricing praise |
| kharab | bad/broken | Quality complaint |
| accha | good | Mild positive |
| tez | fast | Speed praise |
| band | stopped/off | Outage indicator |
| baar baar | again and again | Frequency indicator |

## Data Quality Notes

Keep these in mind when interpreting qualitative data:

- **Sprint 1 has zero feedback/reasons** — only scores. Say "No qualitative data available for Sprint 1."
- **42% feedback fill rate overall** — theme distributions represent the vocal subset, not all customers
- **Detractors comment 1.3x more** than promoters — theme distributions naturally skew negative
- **32.9% of comments are unclassifiable** by keyword matching (generic statements like "ok" or "theek hai") — report as "Unclassified" rather than hiding them
- **NPS-sentiment mismatch** exists in ~5–10% of records (promoter text coded negative or vice versa) — this is a data quality artifact, not something to override
- **Comment quality varies** — single-word responses ("good", "bad") are low-signal; prioritize longer, more specific verbatims

## What This Skill Does NOT Cover

- Quantitative NPS scores, trends, significance tests → use **nps-quant**
- Churn prediction or statistical modeling → not in sprint CSVs (lives in the consolidated analysis)
- LLM-based deep sentiment analysis → requires the K.3 agent pipeline in user-insights-agents
