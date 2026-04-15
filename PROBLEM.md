# Wiom NPS Analysis

## Problem Statement

Wiom runs biweekly NPS (Net Promoter Score) surveys across its WiFi customer base. Each sprint collects 600–2,700 responses via WhatsApp, CleverTap, and outbound calls. The raw data includes NPS scores (0–10), open-text feedback (Hindi/Hinglish/English), pre-coded reason categories, tenure, and channel source.

Today, sprint-over-sprint NPS analysis is manual: CSVs are downloaded, scores are computed in spreadsheets, and comparisons are eyeballed. There is no consistent way to slice data by segment (tenure, city, device, channel) or toggle between time granularities (sprint, month, quarter). This makes it hard to spot trends, identify which segments are moving, and communicate findings to stakeholders.

## Objective

Build an interactive NPS dashboard that:

1. **Ingests sprint CSVs** — upload new sprint data and have it immediately reflected in all views
2. **Computes NPS with confidence** — every number shown with sample size and 95% CI
3. **Supports segment comparison** — filter/toggle by Tenure, City, Device, Source, and Sprint to compare any cut side-by-side
4. **Toggles time granularity** — view trends at sprint-level (biweekly), monthly, or quarterly roll-ups
5. **Surfaces themes** — show pre-coded reason distributions (22-theme taxonomy) and open-text verbatims by segment
6. **Enriches from Metabase** — pulls city, device, and tenure data for respondents using phone number lookup against metabase.wiom.in
7. **Runs locally first** — `npm run dev` on localhost, then deploy to GitHub Pages

---

## Data Schema

Each sprint CSV has 12 columns:

| Column | Type | Notes |
|--------|------|-------|
| respondent_id | string | 10-digit phone number |
| score | int | 0–10 NPS rating. Clamp to [0,10]; skip row if non-numeric. |
| feedback | string | Open-text (Hindi/Hinglish/English). 42% fill rate overall. Detractors comment 1.3x more than promoters. |
| nps_reason_primary | string | Pre-coded reason category (e.g., "Slow Speed", "Affordable", "General Bad Service"). Maps to 22-theme taxonomy. |
| nps_reason_secondary | string | Secondary reason (often empty) |
| primary_category | string | Reserved (always empty) |
| plan_type | string | Plan name (e.g., "Verma") |
| city | string | Empty in CSVs — must be enriched from Metabase |
| tenure_days | int | Days since install. Populated for Sprints 1–12, zero for Sprints 13–14 and RSP1–3 — must be enriched from Metabase for those. |
| source | string | Collection channel raw code |
| sprint_id | string | e.g., "Sprint 14", "Sprint RSP3" |
| sprint_start | date | YYYY-MM-DD |

---

## Guardrails & Rules

### NPS Calculation

- **Promoter**: score 9–10
- **Passive**: score 7–8
- **Detractor**: score 0–6
- **NPS** = (Promoters − Detractors) / Total × 100, rounded to 1 decimal
- **95% CI** (multinomial variance): `Var(NPS) = [P_pro(1−P_pro) + P_det(1−P_det) + 2·P_pro·P_det] / n`; CI = NPS ± 1.96 × sqrt(Var) × 100

### Confidence Levels

Every NPS number must show sample size and confidence:

| Condition | Level |
|-----------|-------|
| n ≥ 100 AND CI width < 15 | **High** |
| n ≥ 30 AND CI width < 25 | **Medium** |
| Otherwise | **Low** — label as "Directional only — small sample" |

### Significance Testing

Sprint-over-sprint changes must use a two-proportion z-test before claiming "improvement" or "decline":

- Combined SE = sqrt(Var1 + Var2) × 100
- z = abs(delta) / SE
- p-value (two-tailed) < 0.05 → significant
- **Never claim a "drop" or "improvement" without a p-value.**

### Direction Classification

| NPS delta | Label |
|-----------|-------|
| > +2 | Improving |
| < −2 | Declining |
| ±2 | Stable |

---

## Segment Definitions

### Tenure Bands (3-tier, from install date = day 1)

| Days | Label |
|------|-------|
| 1–60 | Early (1–2 mo) |
| 61–180 | Mid (3–6 mo) |
| 181+ | Long (6+ mo) |
| Missing/0 | Unknown — must enrich from Metabase |

### Source/Channel Normalization

| Raw Input | Normalized |
|-----------|-----------|
| WA, In - WA, G- form in WA | WhatsApp |
| CT, CLEVERTAP | CleverTap |
| CALL, Call | Call |
| Other | Preserved as-is |

### City (from Metabase enrichment)

Known cities: Delhi (79%), UP (9%), Mumbai (6%), Unknown (6%). In CSVs: treat "#REF!" and "unknown" as empty.

### Device (from Metabase enrichment)

Not present in CSV. Must be pulled via phone number lookup from Metabase. Categories TBD from data.

---

## Sprint Timeline & Aggregation

### Sprint-to-Month Mapping

| Sprint | Month | Quarter |
|--------|-------|---------|
| Sprint 1, Sprint 2 | Jul '25 | Q1 FY26 (Jul-Sep '25) |
| Sprint 3, Sprint 4 | Aug '25 | Q1 FY26 |
| Sprint 5, Sprint 6 | Sep '25 | Q1 FY26 |
| Sprint 7, Sprint 8 | Oct '25 | Q2 FY26 (Oct-Dec '25) |
| Sprint 9, Sprint 10 | Nov '25 | Q2 FY26 |
| Sprint 11, Sprint 12 | Dec '25 | Q2 FY26 |
| Sprint 13, Sprint 14 | Jan '26 | Q3 FY26 (Jan-Mar '26) |
| RSP1, RSP2 | Feb '26 | Q3 FY26 |
| RSP3 | Mar '26 | Q3 FY26 |

### Sprint Ordering

Two-bucket sort: numbered sprints (1–14) sort by number first, then RSP sprints (RSP1, RSP2, RSP3) by their number. Combined/rollup files are excluded.

---

## Theme Taxonomy (22 themes)

Use the 22-theme keyword-based taxonomy from wiom-nps-analysis. Primary keywords score 2 points, secondary 1 point. Highest score wins. Score 0 = "unclassified" (32.9% of comments).

### Pre-coded Reason → Theme Mapping

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

---

## Edge Cases & Data Quality

### Known Issues

| Issue | Handling |
|-------|---------|
| Score 0 = 20.4% of responses (atypically high) | Treat as valid; possible response bias or survey UX issue |
| tenure_days = 0 for Sprints 13–14, RSP1–3 | Enrich from Metabase; show "Unknown" until enriched |
| City empty in all CSVs | Enrich from Metabase via phone number |
| Device not in CSV | Enrich from Metabase via phone number |
| Feedback contains Hindi (Devanagari), Hinglish, and English | Preserve original text in verbatim views; detect language for display |
| Phone numbers may have ".0" suffix (Excel artifact) | Strip `.0$` before using as key |
| City value "#REF!" | Treat as empty/Unknown |
| Negative tenure_days | Treat as data error → Unknown |
| Comments with newlines, emoji, Unicode artifacts | Sanitize for display but preserve for analysis |
| 32.9% of comments unclassifiable by theme keywords | Show as "Unclassified" — don't hide |
| Sprint 1 has 0 feedback/reason values | Theme/verbatim views show "No qualitative data for this sprint" |
| NPS-sentiment mismatch (~5–10%): promoter text coded negative or vice versa | Flag but don't override — let the data speak |

### Score Validation

- Score must be numeric integer; non-numeric rows are skipped silently
- Scores clamped to [0, 10]: `max(0, min(10, score))`
- Missing scores → skip the row (score is required)

### Column Matching (flexible CSV headers)

Accept alternative header names (case-insensitive): score/nps_score/nps/rating, feedback/comment/verbatim/open_text, respondent_id/id/phone/mobile, etc.

---

## Metabase Enrichment

Connect to `metabase.wiom.in` using API key from `C:\credentials\.env`:

```
METABASE_API_KEY=mb_xxx
METABASE_URL=https://metabase.wiom.in
METABASE_DATABASE_ID=1
```

### Enrichment Fields (via phone number lookup)

- **city**: from user/subscriber table
- **device**: from device or connection table
- **tenure_days**: computed from install_date where missing in CSV
- **install_date**: for accurate tenure computation

### Enrichment Script

A Python script (`scripts/enrich_from_metabase.py`) runs before dashboard launch to:
1. Read all sprint CSVs from `data/sprints/`
2. Collect unique phone numbers with missing city/device/tenure
3. Query Metabase in batches
4. Write enriched CSVs back (or a separate enrichment JSON sidecar)

---

## Relationship to User Insights Agents

This repo is the **NPS analysis UI and data pipeline**. It is separated from `user-insights-agents`, which is the broader qualitative research automation system (16 agents covering the full research lifecycle from problem statement to insight cards).

### What lives where

| Concern | This repo (wiom-nps-analysis) | user-insights-agents |
|---------|-------------------------------|---------------------|
| NPS dashboard UI | React app | Streamlit prototype (to be deprecated) |
| Sprint CSV ingestion | Client-side parsing | K.1 agent (server-side) |
| NPS calculation + CI | JavaScript (ported from nps_utils.py) | nps_utils.py (Python) |
| Theme taxonomy | 22-theme keyword taxonomy | taxonomy.json (15-theme canonical) |
| Metabase enrichment | enrich_from_metabase.py | agent_r_recruit.py (for qual research) |
| LLM-based sentiment | Not included (use K.3 output if available) | K.3 agent |
| Trend comparison | Dashboard view | K.4 agent |
| Stakeholder report | Dashboard export | K.5 agent |

### Ported Logic

The following was ported from `user-insights-agents/dashboard/nps_utils.py` to JavaScript:
- `calc_nps()` — NPS with multinomial CI
- `nps_significance_test()` — two-proportion z-test
- `sprint_sort_key()` — chronological sprint ordering
- `assign_tenure_cut()` — 3-tier tenure bucketing
- `REASON_TO_THEME` — pre-coded reason → theme mapping
- `SPRINT_TO_MONTH`, `SPRINT_TO_QUARTER` — time aggregation mappings
- `get_verbatims()` — evocativeness-scored verbatim selection
