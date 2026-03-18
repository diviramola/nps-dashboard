"""
Phase 3D v2: Enriched Feature Engineering — Fixed Join Paths
=============================================================
Queries Snowflake tables to add customer-experience features.

FIXES FROM v1:
- NETWORK_SCORECARD: Use ::FLOAT cast instead of TRY_TO_DOUBLE()
- CUSTOMER_INFLUX: Join via LNG_NAS from NETWORK_SCORECARD (not idmaker)
- DAILY_USAGE_L1: Join via long-form NAS_ID from NETWORK_SCORECARD
- SERVICE_TICKETS: Reduced batch size (200), simplified query

JOIN STRATEGY:
1. NETWORK_SCORECARD via MOBILE (phone) — also extracts LNG_NAS per customer
2. CUSTOMER_INFLUX_SUMMARY via LNG_NAS (from scorecard, not idmaker)
3. DAILY_USAGE_L1 via LNG_NAS (same as NASID in that table)
4. SERVICE_TICKET_MODEL via DEVICE_ID (from t_wg_customer) — batch 200

Output: data/nps_enriched_v2.csv + output/phase3d_enrichment.txt
"""

import sys, io, os, time
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

BASE_DIR = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT, exist_ok=True)

BATCH_DELAY = 2
RETRY_DELAY = 10

report_lines = []
def rpt(line=""):
    report_lines.append(line)
    print(line)

def run_query(sql, description="query", timeout=180):
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if resp.status_code in (200, 202):
            data = resp.json()
            if 'data' in data and 'rows' in data['data']:
                cols = [c['name'].lower() for c in data['data']['cols']]
                rows = data['data']['rows']
                return pd.DataFrame(rows, columns=cols)
            elif 'error' in data:
                print(f"    [ERROR] {description}: {str(data['error'])[:300]}")
                return pd.DataFrame()
            else:
                print(f"    [ERROR] {description}: unexpected shape")
                return pd.DataFrame()
        else:
            print(f"    [ERROR] {description}: HTTP {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    [ERROR] {description}: {str(e)[:200]}")
        return pd.DataFrame()


def run_batched(id_list, sql_template, feature_name, dedup_col=None,
                batch_size=500, placeholder="{id_list}"):
    all_dfs = []
    total_batches = (len(id_list) + batch_size - 1) // batch_size
    failed = []

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        ids_str = ",".join([f"'{p}'" for p in batch])
        sql = sql_template.replace(placeholder, ids_str)
        bn = i // batch_size + 1

        print(f"  [{bn}/{total_batches}] {feature_name}...", end="", flush=True)
        df = run_query(sql, f"{feature_name} {bn}/{total_batches}")
        if len(df) == 0:
            print(f" retry...", end="", flush=True)
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"{feature_name} {bn}/{total_batches} retry")
            if len(df) == 0:
                print(f" SKIP")
                failed.append(bn)
                time.sleep(BATCH_DELAY)
                continue
        print(f" {len(df)} rows")
        all_dfs.append(df)
        if bn < total_batches:
            time.sleep(BATCH_DELAY)

    if failed:
        print(f"  WARNING: {len(failed)} batches failed: {failed}")

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        if dedup_col and dedup_col in result.columns:
            result = result.drop_duplicates(subset=dedup_col, keep='first')
        return result
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
rpt("=" * 70)
rpt("PHASE 3D v2: ENRICHED FEATURE ENGINEERING (FIXED)")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

# ── Load data ──
rpt("\n[LOAD] Reading existing dataset...")
nps = pd.read_csv(os.path.join(DATA, "nps_with_risk_scores.csv"), low_memory=False)
nps['phone_number'] = nps['phone_number'].astype(str).str.strip()
phones = nps['phone_number'].unique().tolist()
rpt(f"  Respondents: {len(nps)}, Unique phones: {len(phones)}, Columns: {len(nps.columns)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 0: DEVICE_ID MAPPING (for service tickets)
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[STEP 0] DEVICE_ID MAPPING (t_wg_customer)")
rpt("=" * 70)

MAPPING_SQL = """
SELECT mobile, device_id
FROM (
    SELECT mobile, device_id,
           ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer
    WHERE mobile IN ({id_list})
      AND mobile > '5999999999'
      AND _FIVETRAN_DELETED = false
) t WHERE rn = 1
"""

device_map_df = run_batched(phones, MAPPING_SQL, "Device Map", dedup_col="mobile")
if len(device_map_df) > 0:
    device_map_df['mobile'] = device_map_df['mobile'].astype(str)
    device_map_df['device_id'] = device_map_df['device_id'].astype(str)
    rpt(f"  Mapped: {len(device_map_df)} phones -> device_id ({len(device_map_df)/len(phones)*100:.1f}%)")
else:
    rpt("  ERROR: No mapping data")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# FEATURE 1: NETWORK SCORECARD (via MOBILE — direct)
# + Extract LNG_NAS for influx/usage joins
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[1/4] NETWORK SCORECARD — Speed Gap, Optical Power, Data Usage")
rpt("  + Extracts LNG_NAS per customer for downstream joins")
rpt("=" * 70)

# Use ::FLOAT cast for TEXT columns (NOT TRY_TO_DOUBLE which fails via Metabase)
SCORECARD_SQL = """
SELECT
    MOBILE,
    -- LNG_NAS for downstream joins (take most recent)
    MAX(LNG_NAS) AS lng_nas,
    -- Speed metrics
    AVG(PLAN_SPEED) AS avg_plan_speed,
    AVG(LATEST_SPEED::FLOAT) AS avg_latest_speed,
    AVG(CASE WHEN PLAN_SPEED > 0 AND LATEST_SPEED IS NOT NULL
        THEN (PLAN_SPEED - LATEST_SPEED::FLOAT) / PLAN_SPEED
        ELSE NULL END) AS speed_gap_pct,
    AVG(SPEED_IN_RANGE) AS avg_speed_in_range,
    -- Optical power
    AVG(RXPOWER::FLOAT) AS avg_rxpower,
    AVG(RXPOWER_IN_RANGE) AS avg_rxpower_in_range,
    AVG(OPTICALPOWER_IN_RANGE) AS avg_opticalpower_in_range,
    -- Data usage
    AVG(DATA_USED_GB::FLOAT) AS avg_weekly_data_gb,
    -- Ticket count from scorecard
    SUM(TICKET_COUNT::INT) AS scorecard_ticket_count,
    -- Coverage & plan
    COUNT(*) AS scorecard_weeks,
    AVG(IS_PLAN_ACTIVE) AS plan_active_ratio
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE IN ({id_list})
  AND WEEK_START >= '2025-06-01'
  AND WEEK_START < '2026-02-01'
GROUP BY MOBILE
"""

scorecard_df = run_batched(phones, SCORECARD_SQL, "Scorecard", dedup_col="mobile")
if len(scorecard_df) > 0:
    scorecard_df['mobile'] = scorecard_df['mobile'].astype(str)
    # Clean lng_nas (remove .0 if present)
    scorecard_df['lng_nas'] = scorecard_df['lng_nas'].astype(str).str.replace(r'\.0$', '', regex=True)
    num_cols = [c for c in scorecard_df.columns if c not in ('mobile', 'lng_nas')]
    for c in num_cols:
        scorecard_df[c] = pd.to_numeric(scorecard_df[c], errors='coerce')
    rpt(f"  Matched: {len(scorecard_df)} phones ({len(scorecard_df)/len(phones)*100:.1f}%)")
    rpt(f"  Unique LNG_NAS extracted: {scorecard_df['lng_nas'].nunique()}")
    rpt(f"  Avg plan speed: {scorecard_df['avg_plan_speed'].mean():.1f} Mbps")
    rpt(f"  Avg actual speed: {scorecard_df['avg_latest_speed'].mean():.1f} Mbps")
    rpt(f"  Avg speed gap: {scorecard_df['speed_gap_pct'].mean()*100:.1f}%")
    rpt(f"  Avg speed in range: {scorecard_df['avg_speed_in_range'].mean():.2f}")
    rpt(f"  Avg weekly data GB: {scorecard_df['avg_weekly_data_gb'].mean():.1f}")
    rpt(f"  Avg rxpower: {scorecard_df['avg_rxpower'].mean():.2f} dBm")
else:
    rpt("  WARNING: No scorecard data")


# ══════════════════════════════════════════════════════════════════════
# FEATURE 2: CUSTOMER INFLUX SUMMARY (via LNG_NAS from scorecard)
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[2/4] CUSTOMER INFLUX — Customer-Level Uptime, Peak Interruption")
rpt("  Join: LNG_NAS from Network Scorecard (not idmaker)")
rpt("=" * 70)

if len(scorecard_df) > 0:
    lng_nas_list = scorecard_df['lng_nas'].dropna().unique().tolist()
    lng_nas_list = [n for n in lng_nas_list if n not in ('', 'None', 'nan', 'None')]
    rpt(f"  LNG_NAS IDs to query: {len(lng_nas_list)}")

    # LNG_NAS in influx is FLOAT — need to handle type matching
    INFLUX_SQL = """
    SELECT
        CAST(LNG_NAS AS VARCHAR) AS lng_nas,
        AVG(CASE WHEN EXPECTED_PINGS_SO_FAR > 0
            THEN PINGS_RECEIVED_TODAY * 1.0 / EXPECTED_PINGS_SO_FAR
            ELSE NULL END) AS avg_customer_uptime_pct,
        AVG(CASE WHEN PINGS_RECEIVED_TODAY > 0
            THEN STABLE_PINGS_COUNT * 1.0 / PINGS_RECEIVED_TODAY
            ELSE NULL END) AS avg_stable_ping_ratio,
        AVG(CASE WHEN EXPECTED_PEAK_PINGS > 0
            THEN PINGS_PEAK_HOURS * 1.0 / EXPECTED_PEAK_PINGS
            ELSE NULL END) AS avg_peak_uptime_pct,
        AVG(HAD_PEAK_INTERRUPTION) AS peak_interruption_rate,
        AVG(IS_ACTIVE_TODAY) AS active_day_ratio,
        COUNT(*) AS influx_data_days
    FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY
    WHERE CAST(LNG_NAS AS VARCHAR) IN ({id_list})
      AND APPENDED_DATE >= '2025-06-01'
      AND APPENDED_DATE < '2026-02-01'
    GROUP BY CAST(LNG_NAS AS VARCHAR)
    """

    influx_df = run_batched(lng_nas_list, INFLUX_SQL, "Influx", dedup_col="lng_nas",
                             batch_size=300)
    if len(influx_df) > 0:
        influx_df['lng_nas'] = influx_df['lng_nas'].astype(str).str.replace(r'\.0$', '', regex=True)
        num_cols = [c for c in influx_df.columns if c != 'lng_nas']
        for c in num_cols:
            influx_df[c] = pd.to_numeric(influx_df[c], errors='coerce')
        rpt(f"  Matched: {len(influx_df)} LNG_NAS ({len(influx_df)/len(lng_nas_list)*100:.1f}%)")
        rpt(f"  Avg customer uptime: {influx_df['avg_customer_uptime_pct'].mean()*100:.1f}%")
        rpt(f"  Avg peak uptime: {influx_df['avg_peak_uptime_pct'].mean()*100:.1f}%")
        rpt(f"  Peak interruption rate: {influx_df['peak_interruption_rate'].mean()*100:.1f}%")
        rpt(f"  Active day ratio: {influx_df['active_day_ratio'].mean()*100:.1f}%")
    else:
        rpt("  WARNING: No influx data")
        influx_df = pd.DataFrame()
else:
    rpt("  SKIP: No scorecard data (no LNG_NAS available)")
    influx_df = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# FEATURE 3: DAILY USAGE L1 (via LNG_NAS from scorecard as NASID)
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[3/4] DAILY USAGE L1 — Sessions, Devices, Peak-Hour, Volatility")
rpt("  Join: LNG_NAS from Scorecard → NASID in DAILY_USAGE_L1")
rpt("=" * 70)

if len(scorecard_df) > 0 and len(lng_nas_list) > 0:
    USAGE_SQL = """
    SELECT
        NASID AS lng_nas,
        AVG(SESSIONS_COUNT_DAILY) AS avg_daily_sessions,
        AVG(TOTAL_CONNECTED_DEVICES_DAILY) AS avg_daily_devices,
        AVG(ACTIVE_HOURS_COUNT_DAILY) AS avg_active_hours,
        AVG(TOTAL_DATA_BYTES_DAILY / 1073741824.0) AS avg_daily_data_gb,
        AVG(TOTAL_SESSION_DURATION_SEC_DAILY / 3600.0) AS avg_session_hours,
        AVG(CASE WHEN TOTAL_DATA_BYTES_DAILY > 0
            THEN PEAK_HOUR_TOTAL_DATA_BYTES_DAILY * 1.0 / TOTAL_DATA_BYTES_DAILY
            ELSE NULL END) AS avg_peak_hour_ratio,
        AVG(CASE WHEN DAY_USAGE_BYTES_DAILY > 0
            THEN NIGHT_USAGE_BYTES_DAILY * 1.0 / DAY_USAGE_BYTES_DAILY
            ELSE NULL END) AS avg_night_day_ratio,
        AVG(USAGE_VOLATILITY_INDEX_DAILY) AS avg_usage_volatility,
        AVG(CASE WHEN MANY_SHORT_SESSIONS_FLAG_DAILY = 1 THEN 1 ELSE 0 END) AS pct_short_sessions,
        AVG(DISTINCT_FRAMEDIP_COUNT_DAILY) AS avg_distinct_ips,
        COUNT(*) AS usage_data_days
    FROM PROD_DB.PUBLIC.DAILY_USAGE_L1
    WHERE NASID IN ({id_list})
      AND DATE_IST >= '2025-06-01'
      AND DATE_IST < '2026-02-01'
    GROUP BY NASID
    """

    usage_df = run_batched(lng_nas_list, USAGE_SQL, "Usage L1", dedup_col="lng_nas",
                            batch_size=300)
    if len(usage_df) > 0:
        usage_df['lng_nas'] = usage_df['lng_nas'].astype(str).str.replace(r'\.0$', '', regex=True)
        num_cols = [c for c in usage_df.columns if c != 'lng_nas']
        for c in num_cols:
            usage_df[c] = pd.to_numeric(usage_df[c], errors='coerce')
        rpt(f"  Matched: {len(usage_df)} NASIDs ({len(usage_df)/len(lng_nas_list)*100:.1f}%)")
        rpt(f"  Avg daily sessions: {usage_df['avg_daily_sessions'].mean():.1f}")
        rpt(f"  Avg daily devices: {usage_df['avg_daily_devices'].mean():.1f}")
        rpt(f"  Avg daily data GB: {usage_df['avg_daily_data_gb'].mean():.1f}")
        rpt(f"  Avg peak-hour ratio: {usage_df['avg_peak_hour_ratio'].mean():.3f}")
        rpt(f"  Avg usage volatility: {usage_df['avg_usage_volatility'].mean():.1f}")
    else:
        rpt("  WARNING: No usage data")
        usage_df = pd.DataFrame()
else:
    rpt("  SKIP: No LNG_NAS available")
    usage_df = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# FEATURE 4: ENRICHED SERVICE TICKETS (via DEVICE_ID, batch 200)
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[4/4] ENRICHED SERVICE TICKETS — Categories, Reopens, Calls")
rpt("  Join: DEVICE_ID from t_wg_customer (batch size 200)")
rpt("=" * 70)

device_ids = device_map_df['device_id'].dropna().unique().tolist()
device_ids = [d for d in device_ids if d not in ('', 'None', 'nan')]
rpt(f"  Device IDs to query: {len(device_ids)}")

TICKET_SQL = """
SELECT
    DEVICE_ID,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN CX_PX = 'Cx' THEN 1 END) AS cx_tickets,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%internet supply down%' THEN 1 END) AS tickets_internet_down,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%disconnect%' OR LOWER(FIRST_TITLE) LIKE '%discontinue%' THEN 1 END) AS tickets_disconnect,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%recharge done but%' THEN 1 END) AS tickets_recharge_issue,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%frequent disconnection%' THEN 1 END) AS tickets_freq_disconnect,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%slow speed%' OR LOWER(FIRST_TITLE) LIKE '%range issue%' THEN 1 END) AS tickets_slow_speed,
    AVG(TIMES_REOPENED) AS avg_reopens,
    AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) AS avg_resolution_mins,
    COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END) AS tickets_within_tat,
    COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'beyond 2*TAT' THEN 1 END) AS tickets_beyond_2x_tat,
    AVG(NO_TIMES_CUSTOMER_CALLED) AS avg_customer_calls,
    AVG(NO_TIMES_PARTNER_CALLED) AS avg_partner_calls,
    AVG(CASE WHEN RATING_SCORE_BY_CUSTOMER > 0 THEN RATING_SCORE_BY_CUSTOMER ELSE NULL END) AS avg_ticket_rating,
    COUNT(CASE WHEN IS_RESOLVED = true THEN 1 END) AS tickets_resolved
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE DEVICE_ID IN ({id_list})
  AND TICKET_ADDED_TIME >= '2025-06-01'
  AND TICKET_ADDED_TIME < '2026-02-01'
GROUP BY DEVICE_ID
"""

ticket_df = run_batched(device_ids, TICKET_SQL, "Tickets", dedup_col="device_id",
                         batch_size=200)
if len(ticket_df) > 0:
    ticket_df['device_id'] = ticket_df['device_id'].astype(str)
    num_cols = [c for c in ticket_df.columns if c != 'device_id']
    for c in num_cols:
        ticket_df[c] = pd.to_numeric(ticket_df[c], errors='coerce')
    rpt(f"  Matched: {len(ticket_df)} devices ({len(ticket_df)/len(device_ids)*100:.1f}%)")
    rpt(f"  Avg total tickets: {ticket_df['total_tickets'].mean():.1f}")
    rpt(f"  Avg internet down tickets: {ticket_df['tickets_internet_down'].mean():.1f}")
    rpt(f"  Avg reopens: {ticket_df['avg_reopens'].mean():.3f}")
    rpt(f"  Avg resolution mins: {ticket_df['avg_resolution_mins'].mean():.0f}")
    rpt(f"  Avg customer calls: {ticket_df['avg_customer_calls'].mean():.2f}")
else:
    rpt("  WARNING: No ticket data")


# ══════════════════════════════════════════════════════════════════════
# MERGE ALL NEW FEATURES
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("MERGING ALL NEW FEATURES")
rpt("=" * 70)

merged = nps.copy()
orig_cols = set(merged.columns)

# Build phone → lng_nas lookup from scorecard
if len(scorecard_df) > 0:
    phone_lng_nas = scorecard_df[['mobile', 'lng_nas']].rename(
        columns={'mobile': 'phone_number'}).drop_duplicates(subset='phone_number')
    merged = merged.merge(phone_lng_nas, on='phone_number', how='left')
    rpt(f"  + LNG_NAS mapping: {merged['lng_nas'].notna().sum()}/{len(merged)} phones")

# Build phone → device_id lookup
phone_device = device_map_df[['mobile', 'device_id']].rename(
    columns={'mobile': 'phone_number'}).drop_duplicates(subset='phone_number')
merged = merged.merge(phone_device, on='phone_number', how='left', suffixes=('', '_mapped'))
device_col = 'device_id_mapped' if 'device_id_mapped' in merged.columns else 'device_id'
rpt(f"  + Device mapping: {merged[device_col].notna().sum()}/{len(merged)} phones")

# 1. SCORECARD — via phone_number
if len(scorecard_df) > 0:
    sc_features = scorecard_df.drop(columns=['lng_nas']).rename(columns={'mobile': 'phone_number'})
    sc_features = sc_features.add_prefix('sc_')
    sc_features = sc_features.rename(columns={'sc_phone_number': 'phone_number'})
    merged = merged.merge(sc_features, on='phone_number', how='left')
    n = merged['sc_avg_plan_speed'].notna().sum()
    rpt(f"  + Scorecard: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")

# 2. INFLUX — via lng_nas
if len(influx_df) > 0 and 'lng_nas' in merged.columns:
    inf_features = influx_df.add_prefix('cis_')
    inf_features = inf_features.rename(columns={'cis_lng_nas': 'lng_nas'})
    merged = merged.merge(inf_features, on='lng_nas', how='left')
    n = merged['cis_avg_customer_uptime_pct'].notna().sum()
    rpt(f"  + Influx: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")

# 3. USAGE — via lng_nas
if len(usage_df) > 0 and 'lng_nas' in merged.columns:
    us_features = usage_df.add_prefix('ul1_')
    us_features = us_features.rename(columns={'ul1_lng_nas': 'lng_nas'})
    # Avoid duplicate lng_nas merge
    merged = merged.merge(us_features, on='lng_nas', how='left')
    n = merged['ul1_avg_daily_sessions'].notna().sum()
    rpt(f"  + Usage L1: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")

# 4. TICKETS — via device_id
if len(ticket_df) > 0:
    tk_features = ticket_df.add_prefix('tk_')
    tk_features = tk_features.rename(columns={'tk_device_id': device_col})
    merged = merged.merge(tk_features, on=device_col, how='left')
    n = merged['tk_total_tickets'].notna().sum()
    rpt(f"  + Tickets: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("COMPUTING DERIVED FEATURES")
rpt("=" * 70)

# Speed gap severity
if 'sc_speed_gap_pct' in merged.columns:
    merged['sc_speed_gap_pct'] = pd.to_numeric(merged['sc_speed_gap_pct'], errors='coerce')
    merged['speed_gap_severity'] = pd.cut(
        merged['sc_speed_gap_pct'],
        bins=[-999, 0, 0.2, 0.4, 0.6, 999],
        labels=['overdelivering', 'acceptable', 'moderate_gap', 'large_gap', 'severe_gap']
    )
    rpt(f"  Speed gap severity: {merged['speed_gap_severity'].value_counts().to_dict()}")

# Connection instability composite
if 'ul1_pct_short_sessions' in merged.columns and 'ul1_avg_usage_volatility' in merged.columns:
    mss = pd.to_numeric(merged['ul1_pct_short_sessions'], errors='coerce')
    vol = pd.to_numeric(merged['ul1_avg_usage_volatility'], errors='coerce')
    q95_mss = mss.quantile(0.95)
    q95_vol = vol.quantile(0.95)
    mss_norm = mss / max(q95_mss, 0.001)
    vol_norm = vol / max(q95_vol, 0.001)
    merged['connection_instability'] = (mss_norm * 0.6 + vol_norm * 0.4).clip(0, 1)
    rpt(f"  Connection instability: {merged['connection_instability'].notna().sum()} computed")

# Customer uptime tier
if 'cis_avg_customer_uptime_pct' in merged.columns:
    merged['cis_avg_customer_uptime_pct'] = pd.to_numeric(merged['cis_avg_customer_uptime_pct'], errors='coerce')
    merged['customer_uptime_tier'] = pd.cut(
        merged['cis_avg_customer_uptime_pct'],
        bins=[-999, 0.5, 0.7, 0.85, 0.95, 999],
        labels=['critical', 'poor', 'fair', 'good', 'excellent']
    )
    rpt(f"  Uptime tier: {merged['customer_uptime_tier'].value_counts().to_dict()}")

# Peak vs overall uptime gap
if 'cis_avg_peak_uptime_pct' in merged.columns and 'cis_avg_customer_uptime_pct' in merged.columns:
    peak = pd.to_numeric(merged['cis_avg_peak_uptime_pct'], errors='coerce')
    overall = pd.to_numeric(merged['cis_avg_customer_uptime_pct'], errors='coerce')
    merged['peak_uptime_gap'] = overall - peak
    rpt(f"  Peak uptime gap: {merged['peak_uptime_gap'].notna().sum()} computed")

# Ticket severity composite
if 'tk_total_tickets' in merged.columns:
    tk = pd.to_numeric(merged['tk_total_tickets'], errors='coerce').fillna(0)
    reop = pd.to_numeric(merged.get('tk_avg_reopens', pd.Series(dtype=float)), errors='coerce').fillna(0)
    beyond = pd.to_numeric(merged.get('tk_tickets_beyond_2x_tat', pd.Series(dtype=float)), errors='coerce').fillna(0)
    merged['ticket_severity'] = np.log1p(tk) * 0.3 + reop * 0.3 + (beyond / tk.clip(lower=1)) * 0.4
    rpt(f"  Ticket severity: {merged['ticket_severity'].notna().sum()} computed")

# SLA compliance ratio
if 'tk_tickets_within_tat' in merged.columns and 'tk_total_tickets' in merged.columns:
    within = pd.to_numeric(merged['tk_tickets_within_tat'], errors='coerce').fillna(0)
    total = pd.to_numeric(merged['tk_total_tickets'], errors='coerce').fillna(0)
    merged['tk_sla_compliance'] = within / total.clip(lower=1)
    rpt(f"  SLA compliance: {merged['tk_sla_compliance'].notna().sum()} computed")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("SAVING")
rpt("=" * 70)

new_cols = [c for c in merged.columns if c not in orig_cols]
output_csv = os.path.join(DATA, "nps_enriched_v2.csv")
merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
rpt(f"  File: {output_csv}")
rpt(f"  Rows: {len(merged)}, Columns: {len(merged.columns)} (+{len(new_cols)} new)")

rpt(f"\n  New columns ({len(new_cols)}):")
for c in sorted(new_cols):
    fill = merged[c].notna().sum()
    rpt(f"    {c:45s}: {fill:6d}/{len(merged)} ({fill/len(merged)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# VALIDATION: FEATURES BY NPS GROUP & CHURN
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("VALIDATION: NEW FEATURES BY NPS GROUP (Promoter vs Detractor)")
rpt("=" * 70)

nps_col = None
for c in merged.columns:
    if c.lower() == 'nps_group':
        nps_col = c; break

key_features = [
    ('sc_avg_plan_speed', 'Plan Speed (Mbps)'),
    ('sc_avg_latest_speed', 'Actual Speed (Mbps)'),
    ('sc_speed_gap_pct', 'Speed Gap (%)'),
    ('sc_avg_speed_in_range', 'Speed In Range'),
    ('sc_avg_rxpower', 'Optical Power (dBm)'),
    ('sc_avg_weekly_data_gb', 'Weekly Data (GB)'),
    ('cis_avg_customer_uptime_pct', 'Customer Uptime'),
    ('cis_avg_peak_uptime_pct', 'Peak Uptime'),
    ('cis_peak_interruption_rate', 'Peak Interruption'),
    ('cis_active_day_ratio', 'Active Day Ratio'),
    ('ul1_avg_daily_sessions', 'Daily Sessions'),
    ('ul1_avg_daily_devices', 'Daily Devices'),
    ('ul1_avg_daily_data_gb', 'Daily Data GB'),
    ('ul1_avg_peak_hour_ratio', 'Peak Hour Ratio'),
    ('tk_total_tickets', 'Total Tickets'),
    ('tk_avg_reopens', 'Avg Reopens'),
    ('tk_avg_resolution_mins', 'Avg Resolution Mins'),
    ('tk_avg_customer_calls', 'Avg Customer Calls'),
    ('tk_avg_ticket_rating', 'Avg Ticket Rating'),
    ('tk_sla_compliance', 'SLA Compliance'),
    ('connection_instability', 'Connection Instability'),
    ('ticket_severity', 'Ticket Severity'),
    ('peak_uptime_gap', 'Peak Uptime Gap'),
]

if nps_col:
    rpt(f"\n  {'Feature':35s} | {'Promoter':>10s} | {'Detractor':>10s} | {'D-P Gap':>10s} | {'Signal':>8s}")
    rpt(f"  {'-'*35} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
    for col, label in key_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            grp = merged.groupby(nps_col)[col].mean()
            p = grp.get('Promoter', float('nan'))
            d = grp.get('Detractor', float('nan'))
            if pd.notna(p) and pd.notna(d) and p != 0:
                gap = d - p
                gap_pct = abs(gap / p * 100) if p != 0 else 0
                sig = "STRONG" if gap_pct > 10 else "moderate" if gap_pct > 5 else "weak"
                rpt(f"  {label:35s} | {p:10.3f} | {d:10.3f} | {gap:+10.3f} | {sig:>8s}")

rpt("\n" + "=" * 70)
rpt("VALIDATION: NEW FEATURES BY CHURN STATUS")
rpt("=" * 70)

churn_col = None
for c in merged.columns:
    if 'churn' in c.lower() and 'overall' in c.lower():
        churn_col = c; break
if not churn_col:
    for c in merged.columns:
        if c.lower() == 'is_churned':
            churn_col = c; break

if churn_col:
    rpt(f"  Churn column: {churn_col}")
    rpt(f"\n  {'Feature':35s} | {'Active':>10s} | {'Churned':>10s} | {'Gap':>10s} | {'Signal':>8s}")
    rpt(f"  {'-'*35} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")
    for col, label in key_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            grp = merged.groupby(churn_col)[col].mean()
            active_v = churn_v = None
            for k in grp.index:
                ks = str(k).lower()
                if ks in ('active', '0', '0.0', 'false'):
                    active_v = grp[k]
                elif ks in ('churn', 'churned', '1', '1.0', 'true'):
                    churn_v = grp[k]
            if active_v is not None and churn_v is not None and active_v != 0:
                gap = churn_v - active_v
                gap_pct = abs(gap / active_v * 100)
                sig = "STRONG" if gap_pct > 10 else "moderate" if gap_pct > 5 else "weak"
                rpt(f"  {label:35s} | {active_v:10.3f} | {churn_v:10.3f} | {gap:+10.3f} | {sig:>8s}")


# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt(f"PHASE 3D v2 COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

report_path = os.path.join(OUTPUT, "phase3d_enrichment.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport: {report_path}")
print(f"Dataset: {output_csv}")
