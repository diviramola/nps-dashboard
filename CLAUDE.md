# Wiom NPS Analysis

## What This Project Is

Interactive NPS (Net Promoter Score) dashboard and analysis pipeline for Wiom, an Indian ISP serving Tier 2-3 cities. Ingests biweekly sprint survey data (600-2,700 responses per sprint via WhatsApp, CleverTap, and outbound calls), computes NPS with confidence intervals, and surfaces themes and verbatims by segment.

**Owner:** Divi Ramola, Head of User Insights, Wiom

## Project Structure

```
wiom-nps-analysis/
  PROBLEM.md                    Full problem statement, data schema, guardrails
  CLAUDE.md                     This file
  nps_promoter_report.js        Word doc report generator (uses docx library)
  package.json                  Node deps (docx ^9.5.0)

  dashboard/                    React + Vite interactive dashboard
    src/
      App.jsx                   Main app
      main.jsx                  Entry point
      FileUpload.jsx            CSV intake
      DataDownload.jsx          Export
      Overview.jsx              Dashboard overview
      ScoreTrends.jsx           Time-series trends
      SegmentComparison.jsx     Segment comparison
      ThemesVerbatims.jsx       Theme + verbatim display
      NPSCard.jsx               Score card component
      FirstTimeUsers.jsx        New user cohort analysis
      styles/index.css
      utils/
        csvParser.js            CSV parsing with flexible column matching
        npsCalculator.js        NPS with multinomial CI + z-tests
        constants.js            Sprint mappings, REASON_TO_THEME, theme config

  data/
    sprints/                    Raw sprint CSVs (19 sprints: Sprint 1-14, RSP1-5)
    config/
      nps_tags_taxonomy.json    Canonical NPS tag taxonomy (source of truth)
    enrichment/                 Metabase enrichment data

  scripts/                      Python analysis pipeline (6 phases)
    phase0_parse_nps.py         Raw CSV parsing
    phase1_*.py                 Theme discovery + Hindi NLP
    phase3_*.py                 Feature engineering
    phase4_*.py                 Statistical modeling (churn prediction)
    phase5_triangulation.py     Cross-method validation
    phase6_ceo_synthesis.py     Executive synthesis
    enrich_from_metabase.py     City/device/tenure enrichment

  output/                       Analysis reports and processed data

  .claude/
    skills/nps-quant/           Quantitative analysis skill + evals
    skills/nps-qual/            Qualitative analysis skill + evals
```

## NPS Tag Taxonomy

The tagging system for NPS responses uses a canonical taxonomy defined in `data/config/nps_tags_taxonomy.json`. This is the single source of truth.

### Where Tags Come From

Tags are manually applied by call center agents when reviewing survey responses. Each response gets up to 3 tags: `nps_reason_primary`, `nps_reason_secondary`, and `nps_reason_tertiary`. The canonical tag list lives in the Google Sheet "Tags" tab and has been codified into `nps_tags_taxonomy.json`.

### Canonical Tags (22 tags, 9 categories)

**Bad Internet Experience (Detractor):**
- General Bad Service — generic negative remarks about service
- Slow Speed — speed complaints (slow, buffering, low speed)
- Frequent Disconnections — intermittent outages (bar-bar band, disconnects)

**Bad Cx Support (Detractor):**
- No/Late Complaint Resolution — unresolved complaints
- Bad Customer Support - Call Centre — call center issues
- Bad Customer Support - Technician — technician issues

**Offering/Plan Dissatisfaction (Detractor):**
- Expensive — cost complaints
- 28-Day Plan Issue — plan validity concerns
- Shifting not Feasible — connection shifting issues
- OTT Request, No OTT complaint — OTT service requests

**Application/Tech (Detractor):**
- Bad Application — app crashes, missing features

**Bad Service (Detractor):**
- RDNI (Recharge Done, No Internet) — paid but no service
- Range Issue — weak WiFi range
- Internet Down — prolonged outage

**NA:**
- N/A — irrelevant/filler responses

**Good Price (Promoter):**
- Affordable — low price appreciation

**Good Internet Experience (Promoter):**
- General Good Service — generic positive remarks
- Good Speed — explicit speed praise
- Quick Installation — fast setup

**Good Cx Support (Promoter):**
- Good customer Support - call centre
- Good Partner/ Techinician Support
- Fast Complaint Resolution

### Tag Normalization

The taxonomy includes a normalization map that corrects known typo variants in historical data. Examples: "General Bad Servie" → "General Bad Service", "Frequent Disconnetions" → "Frequent Disconnections", "No/ Late Complaint Resolution" → "No/Late Complaint Resolution". All 18 sprint CSVs have been normalized against this map.

### Tag → Theme Mapping

Tags map to 12 dashboard themes via `REASON_TO_THEME` in `dashboard/src/utils/constants.js`. The mapping is also stored in `nps_tags_taxonomy.json` under `reason_to_theme`. When adding new tags, update both files.

## Sprint Data

19 sprints covering Jul 2025 – Apr 2026:

| Sprint | Period | Quarter |
|--------|--------|---------|
| Sprint 1-2 | Jul '25 | Q1 FY26 |
| Sprint 3-4 | Aug '25 | Q1 FY26 |
| Sprint 5-6 | Sep '25 | Q1 FY26 |
| Sprint 7-8 | Oct '25 | Q2 FY26 |
| Sprint 9-10 | Nov '25 | Q2 FY26 |
| Sprint 11-12 | Dec '25 | Q2 FY26 |
| Sprint 13-14 | Jan '26 | Q3 FY26 |
| RSP1-2 | Feb '26 | Q3 FY26 |
| RSP3-4 | Mar '26 | Q3 FY26 |
| RSP5 | Apr '26 | Q1 FY27 |

Total: ~18,200 responses across all sprints.

## Data Schema (Sprint CSVs)

Each sprint CSV has these columns: respondent_id, score (0-10), feedback (Hindi/Hinglish/English open text), nps_reason_primary, nps_reason_secondary, primary_category, plan_type, city, tenure_days, source, sprint_id, sprint_start.

NPS classification: Promoter (9-10), Passive (7-8), Detractor (0-6).

## Credentials & Metabase Connection

All API keys live in `C:\credentials\.env` (never committed to version control). To connect to Metabase:

1. **Read the API key** from `C:\credentials\.env` — the key is `METABASE_API_KEY` (starts with `mb_`)
2. **Base URL**: `https://metabase.wiom.in`
3. **Database ID**: `1` (Snowflake / Postgres RDS)
4. **Auth header**: `X-API-Key: <your METABASE_API_KEY value>`

### Metabase API Usage

Query endpoint: `POST https://metabase.wiom.in/api/dataset`

```json
{
  "database": 1,
  "type": "native",
  "native": {
    "query": "SELECT PHONE_NUMBER, CITY, DEVICE_TYPE, INSTALL_DATE, DATEDIFF('day', INSTALL_DATE, CURRENT_DATE()) AS TENURE_DAYS FROM POSTGRES_RDS_PUBLIC.ACTIVE_BASE WHERE PHONE_NUMBER IN ('9876543210', '9123456789')"
  }
}
```

### Enrichment Script

Run from **local machine** (Metabase is not reachable from sandboxed environments):

```bash
cd "C:\Users\divir\claude code\wiom-nps-analysis"
python scripts/enrich_from_metabase.py              # all sprints
python scripts/enrich_from_metabase.py --sprint sprint_rsp5.csv  # one sprint
python scripts/enrich_from_metabase.py --dry-run     # preview only
```

The script reads phone numbers from sprint CSVs, queries Metabase in batches of 500, and fills in `city`, `tenure_days`, and `device` where missing. It also writes enrichment sidecar JSONs to `data/enrichment/`.

### Network Note

Metabase is only reachable from the local network or VPN. Cowork/sandbox environments cannot reach `metabase.wiom.in` due to egress restrictions. Always run enrichment scripts locally.

## Relationship to user-insights-agents

This repo is the NPS analysis UI and data pipeline. The broader qualitative research system lives in `user-insights-agents/`. Key differences:

- **Taxonomy**: This repo uses a 22-tag human-applied taxonomy from the Google Sheet Tags tab. `user-insights-agents` has a 15-theme AI taxonomy in `taxonomy.json`. They are related but distinct — the tag taxonomy here feeds into the theme taxonomy there via REASON_TO_THEME mapping.
- **Dashboard**: This repo has the production React dashboard. `user-insights-agents` has a deprecated Streamlit prototype.
- **LLM sentiment**: Not in this repo. Use K.3 agent output from `user-insights-agents` if available.

## When Adding a New Sprint

1. Export the sprint CSV from the Google Sheet (Data tab only)
2. Save as `data/sprints/sprint_{id}.csv` using the standard schema
3. Run the retag script to normalize any tag variants: `python scripts/retag.py` (or use the taxonomy normalization map manually)
4. Update `SPRINT_MONTH_MAP`, `SPRINT_SHORT_MAP`, `SPRINT_TO_MONTH`, and `SPRINT_TO_QUARTER` in `dashboard/src/utils/constants.js`
5. If new tags appear in the Tags tab, update `data/config/nps_tags_taxonomy.json` and `REASON_TO_THEME` in constants.js

## Known Issues

- Score 0 = ~20% of responses (possible response bias or survey UX issue) — treated as valid
- tenure_days = 0 for Sprints 13-14, RSP1-5 — needs Metabase enrichment (run `python scripts/enrich_from_metabase.py` from local machine where Metabase is reachable)
- City empty in CSVs — needs Metabase enrichment via phone number
- ~33% of comments unclassifiable by theme keywords — shown as "Unclassified"
- Sprint 1 has 0 feedback/reason values
- Sprint RSP4 has only 25/906 rows tagged (rest were #REF! broken formulas in the source sheet)
- Feedback contains Hindi (Devanagari), Hinglish, and English — preserved as-is
