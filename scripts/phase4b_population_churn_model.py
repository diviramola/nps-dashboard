"""
Phase 4B: Population Ops→Churn Model
======================================
Train the churn model on the FULL Wiom customer population (not just NPS respondents).

Pipeline:
  1. Pull 30K random customers from Snowflake (with churn labels)
  2. Compute same ops features used in v5 model
  3. Normalize tenure-biased features
  4. Train Model B (Ops→Churn) on population
  5. Compare with NPS-respondent-trained model

Snowflake tables used:
  - t_router_user_mapping: customer list, churn labels, tenure
  - t_wg_customer: device_id, partner_lng_id mapping
  - PARTNER_INFLUX_SUMMARY: partner-level uptime
  - NETWORK_SCORECARD: per-customer speed, rxpower, optical
  - SERVICE_TICKET_MODEL: tickets, resolution, SLA
  - tata_ivr_events: IVR call metrics (inbound, missed, answered)
  - taskvanilla_audit: install TAT

Output: output/phase4b_population_churn_model.txt
"""

import sys, io, os, warnings, time, json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import requests

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv(r'C:\credentials\.env')
METABASE_API_KEY = os.getenv('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

BASE_DIR = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(BASE_DIR, "output")

report = []
def rpt(line=""):
    report.append(line)
    print(line, flush=True)


def run_query(sql, timeout=300):
    """Execute Snowflake query via Metabase API. Handles 2000 row limit."""
    url = "https://metabase.wiom.in/api/dataset"
    headers = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}
    payload = {"database": 113, "type": "native", "native": {"query": sql}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        rpt(f"  QUERY ERROR: {data['error'][:200]}")
        return pd.DataFrame()
    rows = data["data"]["rows"]
    cols = [c["name"] for c in data["data"]["cols"]]
    df = pd.DataFrame(rows, columns=cols)
    if len(df) == 2000:
        rpt(f"  WARNING: Hit 2000-row Metabase limit — results truncated")
    return df


def batch_query(phone_list, sql_template, batch_size=500, id_col='MOBILE', timeout=300):
    """Run batched queries for large phone lists. Stops early on repeated errors."""
    results = []
    total = len(phone_list)
    consecutive_errors = 0
    for i in range(0, total, batch_size):
        batch = phone_list[i:i+batch_size]
        phone_str = ",".join(f"'{p}'" for p in batch)
        sql = sql_template.replace('{{PHONE_LIST}}', phone_str)
        try:
            df_batch = run_query(sql, timeout=timeout)
            if len(df_batch) > 0:
                results.append(df_batch)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
            rpt(f"    Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}: {len(df_batch)} rows")
        except Exception as e:
            consecutive_errors += 1
            rpt(f"    Batch {i//batch_size + 1} FAILED: {str(e)[:100]}")
        if consecutive_errors >= 3:
            rpt(f"    Stopping early after {consecutive_errors} consecutive failures")
            break
        time.sleep(0.5)  # Rate limiting
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


rpt("=" * 100)
rpt("PHASE 4B: POPULATION OPS→CHURN MODEL")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SPRINT-BASED COHORT SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 1: SPRINT-BASED COHORT SAMPLING (13 sprints, ~1200/sprint)")
rpt("=" * 100)

# Sprint-based cohort design:
# For each of 13 NPS sprints, sample customers who were ACTIVE at that sprint date.
# Active = recharged within 28 days before sprint date (plan still active or just expired).
# Churn = measured as of CURRENT_DATE (same as NPS data — no artificial forward window).
# Churned = DATEDIFF(last_recharge_ever, CURRENT_DATE) >= 44 (28-day plan + 16-day grace).
SPRINTS = [
    (1,  '2025-07-07'),
    (2,  '2025-07-21'),
    (3,  '2025-08-04'),
    (4,  '2025-08-18'),
    (5,  '2025-09-01'),
    (6,  '2025-09-15'),
    (7,  '2025-09-29'),
    (8,  '2025-10-13'),
    (9,  '2025-10-27'),
    (10, '2025-11-10'),
    (11, '2025-11-24'),
    (12, '2025-12-08'),
    (13, '2026-01-05'),
]
SAMPLE_PER_SPRINT = 1200  # ~15.6K total across 13 sprints
LOOKBACK_DAYS = 28        # Plan cycle — active = recharged within 28 days
CHURN_THRESHOLD_DAYS = 44  # 28-day plan + 16-day grace = 44 days since last recharge

rpt(f"  Design: {len(SPRINTS)} sprints x {SAMPLE_PER_SPRINT}/sprint = {len(SPRINTS)*SAMPLE_PER_SPRINT} target")
rpt(f"  Active = recharged within {LOOKBACK_DAYS} days before sprint date")
rpt(f"  Churn = DATEDIFF(last_recharge_ever, CURRENT_DATE) >= {CHURN_THRESHOLD_DAYS} (no forward window)")


def build_sprint_query(sprint_num, sprint_date, sample_size, lookback_days, churn_threshold):
    """Build SQL to sample active customers at a sprint date. Churn measured as of CURRENT_DATE."""
    lookback_date = (pd.to_datetime(sprint_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    return f"""
    WITH deduped_recharges AS (
        SELECT mobile,
               TO_DATE(DATEADD(minute, 330, created_on)) AS recharge_date,
               ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id) AS rn1,
               ROW_NUMBER() OVER (PARTITION BY mobile, router_nas_id, charges,
                   TO_DATE(DATEADD(minute, 330, created_on)) ORDER BY id) AS rn2
        FROM prod_db.public.t_router_user_mapping
        WHERE device_limit = '10'
          AND otp = 'DONE'
          AND mobile > '5999999999'
          AND store_group_id = 0
          AND TO_DATE(DATEADD(minute, 330, created_on)) >= '2025-01-01'
    ),
    valid_recharges AS (
        SELECT mobile, recharge_date
        FROM deduped_recharges
        WHERE rn1 = 1 AND rn2 = 1
    ),
    -- Customers who were ACTIVE at this sprint date (recharged within {lookback_days} days)
    active_at_sprint AS (
        SELECT mobile, MAX(recharge_date) AS last_recharge_before
        FROM valid_recharges
        WHERE recharge_date BETWEEN '{lookback_date}' AND '{sprint_date}'
        GROUP BY mobile
    ),
    -- First ever recharge for tenure calculation
    first_ever AS (
        SELECT mobile, MIN(recharge_date) AS first_recharge_ever
        FROM valid_recharges
        WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    ),
    -- Last recharge EVER (no forward window — churn measured as of today)
    last_recharge_ever AS (
        SELECT mobile, MAX(recharge_date) AS last_recharge_date
        FROM valid_recharges
        WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    ),
    -- Total lifetime recharges
    recharge_stats AS (
        SELECT mobile, COUNT(*) AS recharge_count
        FROM valid_recharges
        WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    )
    SELECT
        a.mobile AS MOBILE,
        {sprint_num} AS SPRINT_NUM,
        r.recharge_count AS RECHARGE_COUNT,
        fe.first_recharge_ever AS FIRST_RECHARGE_EVER,
        lre.last_recharge_date AS LAST_RECHARGE_DATE,
        DATEDIFF(DAY, fe.first_recharge_ever, '{sprint_date}') AS TENURE_DAYS,
        DATEDIFF(DAY, lre.last_recharge_date, CURRENT_DATE()) AS DAYS_SINCE_LAST,
        CASE WHEN DATEDIFF(DAY, lre.last_recharge_date, CURRENT_DATE()) >= {churn_threshold}
             THEN 1 ELSE 0 END AS IS_CHURNED
    FROM active_at_sprint a
    JOIN first_ever fe ON a.mobile = fe.mobile
    JOIN last_recharge_ever lre ON a.mobile = lre.mobile
    JOIN recharge_stats r ON a.mobile = r.mobile
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """

# Pull cohorts sprint-by-sprint
pop_frames = []
seen_phones = set()

for sprint_num, sprint_date in SPRINTS:
    sql = build_sprint_query(sprint_num, sprint_date, SAMPLE_PER_SPRINT, LOOKBACK_DAYS, CHURN_THRESHOLD_DAYS)
    try:
        df_batch = run_query(sql, timeout=600)
        if len(df_batch) > 0:
            # Deduplicate across sprints (each customer appears once)
            new = df_batch[~df_batch['MOBILE'].isin(seen_phones)]
            seen_phones.update(new['MOBILE'].tolist())
            pop_frames.append(new)
            churn_pct = pd.to_numeric(new['IS_CHURNED'], errors='coerce').mean() * 100
            rpt(f"    Sprint {sprint_num:2d} ({sprint_date}): "
                f"{len(new):5d} new, churn={churn_pct:.1f}%")
        else:
            rpt(f"    Sprint {sprint_num:2d} ({sprint_date}): 0 rows")
    except Exception as e:
        rpt(f"    Sprint {sprint_num:2d} ({sprint_date}): ERROR - {str(e)[:150]}")
    time.sleep(1)

df_pop = pd.concat(pop_frames, ignore_index=True) if pop_frames else pd.DataFrame()
rpt(f"\n  Total: {len(df_pop)} customers from {len(pop_frames)} sprints")

if len(df_pop) == 0:
    rpt("FATAL: No customers returned. Check Snowflake query.")
    sys.exit(1)

# Clean types
for c in ['RECHARGE_COUNT', 'TENURE_DAYS', 'DAYS_SINCE_LAST', 'IS_CHURNED', 'SPRINT_NUM']:
    if c in df_pop.columns:
        df_pop[c] = pd.to_numeric(df_pop[c], errors='coerce')

rpt(f"  Overall churn rate: {df_pop['IS_CHURNED'].mean()*100:.1f}%")
rpt(f"  Avg tenure: {df_pop['TENURE_DAYS'].mean():.0f} days")
rpt(f"  Median recharges: {df_pop['RECHARGE_COUNT'].median():.0f}")
rpt(f"  Sprint distribution:")
for s in sorted(df_pop['SPRINT_NUM'].unique()):
    sub = df_pop[df_pop['SPRINT_NUM'] == s]
    rpt(f"    Sprint {int(s):2d}: n={len(sub):5d}, churn={sub['IS_CHURNED'].mean()*100:.1f}%")

# Check overlap with NPS respondents
df_nps = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False,
                       usecols=['phone_number'])
nps_phones = set(df_nps['phone_number'].astype(str).str.strip())
pop_phones = set(df_pop['MOBILE'].astype(str).str.strip())
overlap = nps_phones & pop_phones
rpt(f"  Overlap with NPS respondents: {len(overlap)} ({len(overlap)/max(len(pop_phones),1)*100:.1f}%)")

phone_list = df_pop['MOBILE'].astype(str).str.strip().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PARTNER MAPPING + DEVICE ID
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 2: PARTNER MAPPING (t_wg_customer → partner_lng_id, device_id)")
rpt("=" * 100)

sql_partner = """
WITH customer_partner AS (
    SELECT
        twc.mobile,
        twc.device_id,
        twc.shard,
        twc.lco_account_id,
        prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS partner_lng_id,
        ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({{PHONE_LIST}})
      AND twc._FIVETRAN_DELETED = false
)
SELECT
    cp.mobile AS MOBILE,
    cp.device_id AS DEVICE_ID,
    cp.partner_lng_id AS PARTNER_LNG_ID,
    hb.MIS_CITY AS CITY,
    hb.CLUSTER AS CLUSTER
FROM customer_partner cp
LEFT JOIN prod_db.public.HIERARCHY_BASE hb
    ON cp.partner_lng_id = hb.PARTNER_ACCOUNT_ID
    AND hb.DEDUP_FLAG = 1
WHERE cp.rn = 1
"""

rpt("  Querying partner mapping...")
try:
    df_partner = batch_query(phone_list, sql_partner, batch_size=500)
    rpt(f"  Partner mapping: {len(df_partner)} rows")

    if len(df_partner) > 0:
        df_pop = df_pop.merge(df_partner, on='MOBILE', how='left')
        rpt(f"  Partner match rate: {df_pop['PARTNER_LNG_ID'].notna().mean()*100:.1f}%")
        rpt(f"  Cities: {df_pop['CITY'].value_counts().head(5).to_dict()}")
    else:
        rpt("  WARNING: Partner mapping returned 0 rows — continuing without partner data")
except Exception as e:
    rpt(f"  ERROR in partner mapping: {str(e)[:200]} — continuing without partner data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PARTNER UPTIME (PARTNER_INFLUX_SUMMARY)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 3: PARTNER UPTIME (90-day avg from PARTNER_INFLUX_SUMMARY)")
rpt("=" * 100)

# Get unique partner_lng_ids
try:
    partner_ids = df_pop['PARTNER_LNG_ID'].dropna().unique().tolist() if 'PARTNER_LNG_ID' in df_pop.columns else []
except Exception:
    partner_ids = []
rpt(f"  Unique partners: {len(partner_ids)}")

if len(partner_ids) > 0:
    # Query uptime per partner (single global query — partners are much fewer than customers)
    partner_str = ",".join(f"'{p}'" for p in partner_ids[:5000])  # Cap at 5K
    sql_uptime = f"""
    SELECT
        partner_id AS PARTNER_LNG_ID,
        AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) * 100 AS avg_uptime_pct,
        STDDEV(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) * 100 AS stddev_uptime,
        AVG(CASE WHEN HOUR(DATEADD(minute, 330, appended_date)) BETWEEN 19 AND 23
            THEN TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0) END) * 100 AS peak_uptime_pct,
        AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) * 100 AS overall_uptime_pct,
        COUNT(DISTINCT TO_DATE(appended_date)) AS uptime_data_days
    FROM prod_db.public.PARTNER_INFLUX_SUMMARY
    WHERE partner_id IN ({partner_str})
      AND DATEADD(day, -1, appended_date) >= DATEADD(DAY, -90, CURRENT_DATE())
    GROUP BY partner_id
    """

    try:
        rpt("  Querying partner uptime...")
        df_uptime = run_query(sql_uptime, timeout=300)
        rpt(f"  Uptime data: {len(df_uptime)} partners")

        if len(df_uptime) > 0:
            for c in ['AVG_UPTIME_PCT', 'STDDEV_UPTIME', 'PEAK_UPTIME_PCT', 'OVERALL_UPTIME_PCT']:
                if c in df_uptime.columns:
                    df_uptime[c] = pd.to_numeric(df_uptime[c], errors='coerce')
            df_pop = df_pop.merge(df_uptime, on='PARTNER_LNG_ID', how='left')
            rpt(f"  Uptime match rate: {df_pop['AVG_UPTIME_PCT'].notna().mean()*100:.1f}%")
            rpt(f"  Mean uptime: {df_pop['AVG_UPTIME_PCT'].mean():.1f}%")
    except Exception as e:
        rpt(f"  ERROR in uptime query: {str(e)[:200]} — continuing without uptime data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: NETWORK SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 4: NETWORK SCORECARD (speed, rxpower, optical)")
rpt("=" * 100)

sql_scorecard = """
SELECT
    MOBILE,
    AVG(PLAN_SPEED) AS SC_AVG_PLAN_SPEED,
    AVG(LATEST_SPEED::FLOAT) AS SC_AVG_LATEST_SPEED,
    AVG(CASE WHEN PLAN_SPEED > 0 AND LATEST_SPEED IS NOT NULL
        THEN (PLAN_SPEED - LATEST_SPEED::FLOAT) / PLAN_SPEED END) AS SC_SPEED_GAP_PCT,
    AVG(SPEED_IN_RANGE) AS SC_AVG_SPEED_IN_RANGE,
    AVG(RXPOWER::FLOAT) AS SC_AVG_RXPOWER,
    AVG(RXPOWER_IN_RANGE) AS SC_AVG_RXPOWER_IN_RANGE,
    AVG(OPTICALPOWER_IN_RANGE) AS SC_AVG_OPTICALPOWER_IN_RANGE,
    AVG(DATA_USED_GB::FLOAT) AS SC_AVG_WEEKLY_DATA_GB,
    COUNT(*) AS SCORECARD_WEEKS
FROM prod_db.public.NETWORK_SCORECARD
WHERE MOBILE IN ({{PHONE_LIST}})
  AND WEEK_START >= '2025-06-01'
GROUP BY MOBILE
"""

rpt("  Querying network scorecard...")
try:
    df_scorecard = batch_query(phone_list, sql_scorecard, batch_size=500)
    rpt(f"  Scorecard data: {len(df_scorecard)} customers")

    if len(df_scorecard) > 0:
        for c in df_scorecard.columns:
            if c != 'MOBILE':
                df_scorecard[c] = pd.to_numeric(df_scorecard[c], errors='coerce')
        df_pop = df_pop.merge(df_scorecard, on='MOBILE', how='left')
        rpt(f"  Scorecard match rate: {df_pop['SC_AVG_RXPOWER_IN_RANGE'].notna().mean()*100:.1f}%")
except Exception as e:
    rpt(f"  ERROR in scorecard query: {str(e)[:200]} — continuing without scorecard data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SERVICE TICKETS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 5: SERVICE TICKETS (resolution, SLA, severity)")
rpt("=" * 100)

# Need device_ids for ticket query
device_ids = df_pop.loc[df_pop['DEVICE_ID'].notna(), 'DEVICE_ID'].unique().tolist() if 'DEVICE_ID' in df_pop.columns else []
rpt(f"  Device IDs available: {len(device_ids)}")

if len(device_ids) > 0:
    sql_tickets = """
    SELECT
        DEVICE_ID,
        COUNT(*) AS TOTAL_TICKETS,
        COUNT(CASE WHEN CX_PX = 'Cx' THEN 1 END) AS CX_TICKETS,
        COUNT(CASE WHEN CX_PX = 'Px' THEN 1 END) AS PX_TICKETS,
        AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) AS AVG_RESOLUTION_MINS,
        COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END) * 1.0 / NULLIF(COUNT(*), 0) AS SLA_COMPLIANCE_PCT,
        SUM(CASE WHEN TIMES_REOPENED > 0 THEN 1 ELSE 0 END) AS TICKETS_REOPENED_ONCE,
        MAX(TIMES_REOPENED) AS MAX_TIMES_REOPENED,
        COUNT(DISTINCT FIRST_TITLE) AS DISTINCT_ISSUE_TYPES,
        AVG(NO_TIMES_CUSTOMER_CALLED) AS AVG_CUSTOMER_CALLS,
        COUNT(CASE WHEN IS_RESOLVED = true THEN 1 END) AS TICKETS_RESOLVED,
        AVG(CASE WHEN RATING_SCORE_BY_CUSTOMER > 0 THEN RATING_SCORE_BY_CUSTOMER ELSE NULL END) AS AVG_TICKET_RATING
    FROM prod_db.public.SERVICE_TICKET_MODEL
    WHERE DEVICE_ID IN ({{PHONE_LIST}})
      AND LOWER(FIRST_TITLE) NOT LIKE '%shifting%'
      AND CX_PX <> 'CC'
      AND TO_DATE(DATEADD(minute, 330, TICKET_ADDED_TIME)) >= '2025-06-01'
    GROUP BY DEVICE_ID
    """

    try:
        rpt("  Querying service tickets...")
        df_tickets = batch_query(device_ids, sql_tickets, batch_size=300, id_col='DEVICE_ID')
        rpt(f"  Ticket data: {len(df_tickets)} devices")

        if len(df_tickets) > 0:
            for c in df_tickets.columns:
                if c != 'DEVICE_ID':
                    df_tickets[c] = pd.to_numeric(df_tickets[c], errors='coerce')
            df_pop = df_pop.merge(df_tickets, on='DEVICE_ID', how='left')

            # Derived ticket features
            df_pop['has_tickets'] = (df_pop['TOTAL_TICKETS'].fillna(0) > 0).astype(int)
            df_pop['avg_resolution_hours'] = df_pop['AVG_RESOLUTION_MINS'] / 60.0
            # Winsorized version
            p99_val = df_pop['avg_resolution_hours'].quantile(0.99)
            df_pop['avg_resolution_hours_w'] = df_pop['avg_resolution_hours'].clip(upper=p99_val)
            # SLA compliance
            df_pop['tk_sla_compliance'] = df_pop['SLA_COMPLIANCE_PCT']
            # Ticket severity
            total_t = df_pop['TOTAL_TICKETS'].fillna(0)
            reopens = df_pop['TICKETS_REOPENED_ONCE'].fillna(0)
            df_pop['ticket_severity'] = np.where(total_t > 0,
                np.log1p(total_t) * 0.3 + reopens * 0.3 +
                (1 - df_pop['SLA_COMPLIANCE_PCT'].fillna(0.5)) * 0.4,
                0)
            # Resolution rate
            resolved = df_pop['TICKETS_RESOLVED'].fillna(0)
            df_pop['resolution_rate'] = np.where(total_t > 0, resolved / total_t, np.nan)

            rpt(f"  Ticket match rate: {df_pop['TOTAL_TICKETS'].notna().mean()*100:.1f}%")
    except Exception as e:
        rpt(f"  ERROR in ticket query: {str(e)[:200]} — continuing without ticket data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: IVR CALLS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 6: IVR CALL METRICS")
rpt("=" * 100)

sql_ivr = """
SELECT
    RIGHT(CLIENT_NUMBER, 10) AS MOBILE,
    COUNT(*) AS TOTAL_IVR_CALLS,
    COUNT(CASE WHEN STATUS = 'answered' THEN 1 END) AS ANSWERED_CALLS,
    COUNT(CASE WHEN STATUS = 'missed' THEN 1 END) AS MISSED_CALLS,
    COUNT(CASE WHEN STATUS = 'dropped' THEN 1 END) AS DROPPED_CALLS,
    COUNT(CASE WHEN DIRECTION = 'inbound' THEN 1 END) AS INBOUND_CALLS,
    COUNT(CASE WHEN DIRECTION = 'inbound' AND STATUS = 'answered' THEN 1 END) AS INBOUND_ANSWERED,
    COUNT(CASE WHEN DIRECTION = 'inbound' AND STATUS != 'answered' THEN 1 END) AS INBOUND_UNANSWERED,
    AVG(CASE WHEN STATUS = 'answered' THEN ANSWERED_SECONDS END) AS AVG_ANSWERED_SECONDS,
    AVG(CALL_DURATION) AS AVG_CALL_DURATION
FROM prod_db.public.tata_ivr_events
WHERE RIGHT(CLIENT_NUMBER, 10) IN ({{PHONE_LIST}})
  AND LENGTH(CLIENT_NUMBER) >= 10
  AND TO_DATE(DATEADD(minute, 330, TIMESTAMP)) >= '2025-06-01'
GROUP BY RIGHT(CLIENT_NUMBER, 10)
"""

rpt("  Querying IVR calls...")
try:
    df_ivr = batch_query(phone_list, sql_ivr, batch_size=500)
    rpt(f"  IVR data: {len(df_ivr)} customers")

    if len(df_ivr) > 0:
        for c in df_ivr.columns:
            if c != 'MOBILE':
                df_ivr[c] = pd.to_numeric(df_ivr[c], errors='coerce')
        df_pop = df_pop.merge(df_ivr, on='MOBILE', how='left')

        # Derived: missed_call_ratio
        tot = df_pop['TOTAL_IVR_CALLS'].fillna(0)
        miss = df_pop['MISSED_CALLS'].fillna(0)
        df_pop['missed_call_ratio'] = np.where(tot > 0, miss / tot, 0)

        rpt(f"  IVR match rate: {df_pop['TOTAL_IVR_CALLS'].notna().mean()*100:.1f}%")
except Exception as e:
    rpt(f"  ERROR in IVR query: {str(e)[:200]} — continuing without IVR data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: INSTALL TAT
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 7: INSTALL TAT")
rpt("=" * 100)

sql_install = """
SELECT
    mobile AS MOBILE,
    MIN(DATEADD(minute, 330, added_time)) AS FIRST_INSTALL_TS,
    COUNT(*) AS INSTALL_ATTEMPTS
FROM prod_db.public.taskvanilla_audit
WHERE mobile IN ({{PHONE_LIST}})
  AND event_name = 'OTP_VERIFIED'
  AND mobile > '5999999999'
GROUP BY mobile
"""

rpt("  Querying install TAT...")
try:
    df_install = batch_query(phone_list, sql_install, batch_size=500)
    rpt(f"  Install data: {len(df_install)} customers")

    if len(df_install) > 0:
        for c in df_install.columns:
            if c != 'MOBILE' and c != 'FIRST_INSTALL_TS':
                df_install[c] = pd.to_numeric(df_install[c], errors='coerce')
        df_pop = df_pop.merge(df_install, on='MOBILE', how='left')
        rpt(f"  Install match rate: {df_pop['INSTALL_ATTEMPTS'].notna().mean()*100:.1f}%")
except Exception as e:
    rpt(f"  ERROR in install query: {str(e)[:200]} — continuing without install data")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: NORMALIZE TENURE-BIASED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 8: FEATURE ENGINEERING + NORMALIZATION")
rpt("=" * 100)

df_pop['TENURE_DAYS'] = pd.to_numeric(df_pop['TENURE_DAYS'], errors='coerce').fillna(30)
tenure_months = np.maximum(df_pop['TENURE_DAYS'].values / 30.0, 1.0)

# Normalize absolute counts to per-month rates
NORMALIZE_MAP = {
    'TOTAL_IVR_CALLS':       'ivr_calls_per_month',
    'MISSED_CALLS':          'missed_calls_per_month',
    'INBOUND_CALLS':         'inbound_calls_per_month',
    'ANSWERED_CALLS':        'answered_calls_per_month',
    'INBOUND_ANSWERED':      'inbound_answered_per_month',
    'INBOUND_UNANSWERED':    'inbound_unanswered_per_month',
    'TOTAL_TICKETS':         'tickets_per_month',
    'CX_TICKETS':            'cx_tickets_per_month',
    'PX_TICKETS':            'px_tickets_per_month',
    'DISTINCT_ISSUE_TYPES':  'distinct_issues_per_month',
    'TICKETS_REOPENED_ONCE': 'reopened_once_per_month',
    'MAX_TIMES_REOPENED':    'max_reopened_per_month',
    'INSTALL_ATTEMPTS':      'install_attempts_per_month',
}

for orig, normed in NORMALIZE_MAP.items():
    if orig in df_pop.columns:
        raw = pd.to_numeric(df_pop[orig], errors='coerce').fillna(0).values
        df_pop[normed] = raw / tenure_months
        rpt(f"  {orig:30s} → {normed}")

# Uptime-derived features
if 'AVG_UPTIME_PCT' in df_pop.columns and 'PEAK_UPTIME_PCT' in df_pop.columns:
    df_pop['PEAK_VS_OVERALL_GAP'] = (df_pop['PEAK_UPTIME_PCT'].fillna(0) -
                                      df_pop['OVERALL_UPTIME_PCT'].fillna(0))

# autopay_ratio placeholder (would need wiomBillingWifi, skip for now — fill 0)
if 'autopay_ratio' not in df_pop.columns:
    df_pop['autopay_ratio'] = 0  # Will be filled by payment query if available

# Fill NaN for numeric features
rpt(f"\n  Final population dataset: {len(df_pop)} rows × {len(df_pop.columns)} cols")
rpt(f"  Churn rate: {df_pop['IS_CHURNED'].mean()*100:.1f}%")

# Save population dataset
pop_file = os.path.join(DATA, "population_ops_features.csv")
df_pop.to_csv(pop_file, index=False)
rpt(f"  SAVED: {pop_file}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: BUILD MODEL B ON POPULATION
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 9: TRAIN MODEL B (Ops→Churn) ON POPULATION")
rpt("=" * 100)

# Feature list matching v5 (using population column names — may be uppercase)
POP_FEATURES = [
    # Network quality (from scorecard + uptime)
    'AVG_UPTIME_PCT', 'OVERALL_UPTIME_PCT', 'PEAK_UPTIME_PCT', 'PEAK_VS_OVERALL_GAP',
    'STDDEV_UPTIME',
    'SC_AVG_RXPOWER_IN_RANGE', 'SC_AVG_RXPOWER', 'SC_AVG_OPTICALPOWER_IN_RANGE',
    'SC_AVG_LATEST_SPEED', 'SC_AVG_SPEED_IN_RANGE', 'SC_SPEED_GAP_PCT', 'SC_AVG_PLAN_SPEED',
    'SC_AVG_WEEKLY_DATA_GB',
    # Service/support
    'avg_resolution_hours', 'avg_resolution_hours_w', 'SLA_COMPLIANCE_PCT',
    'tk_sla_compliance', 'AVG_ANSWERED_SECONDS', 'AVG_CUSTOMER_CALLS',
    'DROPPED_CALLS', 'missed_call_ratio', 'AVG_TICKET_RATING',
    'has_tickets', 'ticket_severity', 'resolution_rate',
    # Payment
    'autopay_ratio',
    # Normalized (per-month rates)
    'ivr_calls_per_month', 'missed_calls_per_month', 'inbound_calls_per_month',
    'answered_calls_per_month', 'inbound_answered_per_month', 'inbound_unanswered_per_month',
    'tickets_per_month', 'cx_tickets_per_month', 'px_tickets_per_month',
    'distinct_issues_per_month', 'reopened_once_per_month', 'max_reopened_per_month',
    'install_attempts_per_month',
]

# Also try lowercase versions (Metabase sometimes returns lowercase)
pop_features_avail = []
for f in POP_FEATURES:
    if f in df_pop.columns:
        pop_features_avail.append(f)
    elif f.lower() in df_pop.columns:
        pop_features_avail.append(f.lower())
    elif f.upper() in df_pop.columns:
        pop_features_avail.append(f.upper())

rpt(f"  Available features: {len(pop_features_avail)}/{len(POP_FEATURES)}")
missing = [f for f in POP_FEATURES if f not in pop_features_avail and f.lower() not in [x.lower() for x in pop_features_avail]]
if missing:
    rpt(f"  Missing: {missing}")

# Build matrix
X_pop = df_pop[pop_features_avail].copy()
for c in X_pop.columns:
    X_pop[c] = pd.to_numeric(X_pop[c], errors='coerce')

# Drop all-NaN columns
all_nan = X_pop.columns[X_pop.isna().all()]
if len(all_nan) > 0:
    rpt(f"  Dropping {len(all_nan)} all-NaN columns: {list(all_nan)}")
    X_pop = X_pop.drop(columns=all_nan)

# Impute
imputer = SimpleImputer(strategy='median')
X_pop_imp = pd.DataFrame(imputer.fit_transform(X_pop), columns=X_pop.columns, index=X_pop.index)
y_pop = df_pop['IS_CHURNED'].values.astype(int)

rpt(f"  Training matrix: {X_pop_imp.shape[0]} rows × {X_pop_imp.shape[1]} features")
rpt(f"  Churn: {y_pop.sum()}/{len(y_pop)} ({y_pop.mean()*100:.1f}%)")

# 3-fold CV
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 min_samples_leaf=20, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)

rpt("\n  3-Fold CV on population data...")
gb_cv = cross_val_score(gb, X_pop_imp, y_pop, cv=cv, scoring='roc_auc')
rf_cv = cross_val_score(rf, X_pop_imp, y_pop, cv=cv, scoring='roc_auc')
rpt(f"    GB CV AUC: {gb_cv.mean():.4f} (+/- {gb_cv.std():.4f})")
rpt(f"    RF CV AUC: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# Fit on full population
gb.fit(X_pop_imp, y_pop)
rf.fit(X_pop_imp, y_pop)

# Feature importance
imp = pd.Series(gb.feature_importances_, index=X_pop_imp.columns).sort_values(ascending=False)
rpt(f"\n  Top 20 Population Churn Drivers:")
for i, (f, v) in enumerate(imp.head(20).items()):
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: CROSS-VALIDATE ON NPS RESPONDENTS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 10: VALIDATE POPULATION MODEL ON NPS RESPONDENTS")
rpt("=" * 100)

# Load NPS data and score with population-trained model
df_nps_full = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df_nps_full['churn_binary'] = df_nps_full['is_churned'].astype(int)

# Map v5 feature names to population model features
# The NPS dataset has lowercase names, population may have uppercase
# Build a mapping from population feature names to NPS equivalents
nps_feature_map = {}
for f in X_pop_imp.columns:
    if f in df_nps_full.columns:
        nps_feature_map[f] = f
    elif f.lower() in df_nps_full.columns:
        nps_feature_map[f] = f.lower()
    elif f.upper() in df_nps_full.columns:
        nps_feature_map[f] = f.upper()

rpt(f"  Mapped {len(nps_feature_map)}/{len(X_pop_imp.columns)} features to NPS dataset")
unmapped = [f for f in X_pop_imp.columns if f not in nps_feature_map]
if unmapped:
    rpt(f"  Unmapped: {unmapped}")

# Build NPS X matrix with matching features
if len(nps_feature_map) > 0:
    X_nps_val = pd.DataFrame(index=df_nps_full.index)
    for pop_name, nps_name in nps_feature_map.items():
        X_nps_val[pop_name] = pd.to_numeric(df_nps_full[nps_name], errors='coerce')

    # Add missing columns as NaN
    for f in X_pop_imp.columns:
        if f not in X_nps_val.columns:
            X_nps_val[f] = np.nan
    X_nps_val = X_nps_val[X_pop_imp.columns]

    # Impute with population imputer
    X_nps_imp = pd.DataFrame(imputer.transform(X_nps_val), columns=X_nps_val.columns, index=X_nps_val.index)
    y_nps = df_nps_full['churn_binary'].values

    # Score NPS respondents with population model
    gb_proba_nps = gb.predict_proba(X_nps_imp)[:, 1]
    rf_proba_nps = rf.predict_proba(X_nps_imp)[:, 1]
    ens_proba_nps = (gb_proba_nps + rf_proba_nps) / 2

    auc_gb_nps = roc_auc_score(y_nps, gb_proba_nps)
    auc_rf_nps = roc_auc_score(y_nps, rf_proba_nps)
    auc_ens_nps = roc_auc_score(y_nps, ens_proba_nps)

    rpt(f"\n  Population-trained model scored on NPS respondents (n={len(y_nps)}):")
    rpt(f"    GB AUC:       {auc_gb_nps:.4f}")
    rpt(f"    RF AUC:       {auc_rf_nps:.4f}")
    rpt(f"    Ensemble AUC: {auc_ens_nps:.4f}")

    # Also do OOT split on NPS data
    sprint_num = df_nps_full['sprint_num'].values
    oot_mask = (sprint_num >= 8) & (sprint_num <= 11)
    if oot_mask.sum() > 100:
        auc_oot = roc_auc_score(y_nps[oot_mask], ens_proba_nps[oot_mask])
        rpt(f"    OOT (Sprints 8-11) AUC: {auc_oot:.4f} (n={oot_mask.sum()})")

    rpt(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    rpt(f"  ║  COMPARISON: Population vs NPS-Trained Models                    ║")
    rpt(f"  ╠══════════════════════════════════════════════════════════════════╣")
    rpt(f"  ║  v5 (NPS-trained, OOT):    AUC = 0.8824                        ║")
    rpt(f"  ║  Population (CV):           AUC = {gb_cv.mean():.4f}                        ║")
    rpt(f"  ║  Population → NPS scored:   AUC = {auc_ens_nps:.4f}                        ║")
    rpt(f"  ╚══════════════════════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output_file = os.path.join(OUTPUT, "phase4b_population_churn_model.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n{'='*100}")
rpt(f"SAVED: {output_file}")
rpt(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*100}")
