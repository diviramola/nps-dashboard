"""
Phase 3D: Enriched Feature Engineering from New Snowflake Tables
================================================================
Queries newly discovered Snowflake tables to add customer-experience features
that were missing from the original Phase 3.

NEW FEATURES:
1. NETWORK_SCORECARD  — speed gap (plan vs actual), optical power, speed consistency
2. DAILY_USAGE_L1     — session counts, device counts, peak-hour usage, volatility
3. CUSTOMER_INFLUX_SUMMARY — customer-level uptime, peak interruption
4. Enriched SERVICE_TICKET_MODEL — ticket categories, reopens, customer/partner calls

JOINS:
- NETWORK_SCORECARD: directly via MOBILE (phone number)
- DAILY_USAGE_L1: via NASID from t_wg_customer mapping
- CUSTOMER_INFLUX_SUMMARY: via LNG_NAS (computed via idmaker from t_wg_customer)
- SERVICE_TICKET_MODEL: via DEVICE_ID from t_wg_customer mapping

Output:
- data/nps_enriched_v2.csv
- output/phase3d_enrichment.txt
"""

import sys, io, os, json, time, traceback
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# -- Credentials --
from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

BATCH_SIZE = 500
BATCH_DELAY = 2
RETRY_DELAY = 10

report_lines = []
def rpt(line=""):
    report_lines.append(line)
    print(line)


# ── Query runner ──
def run_query(sql, description="query", timeout=180):
    """Execute a Snowflake query via Metabase API. Returns DataFrame."""
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if resp.status_code in (200, 202):
            data = resp.json()
            if 'data' in data and 'rows' in data['data']:
                cols = [c['name'].lower() for c in data['data']['cols']]
                rows = data['data']['rows']
                df = pd.DataFrame(rows, columns=cols)
                return df
            elif 'error' in data:
                print(f"    [ERROR] {description}: {str(data['error'])[:300]}")
                return pd.DataFrame()
            else:
                print(f"    [ERROR] {description}: unexpected response shape")
                return pd.DataFrame()
        else:
            print(f"    [ERROR] {description}: HTTP {resp.status_code} -- {resp.text[:200]}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    [ERROR] {description}: {str(e)[:200]}")
        return pd.DataFrame()


def run_batched_query(id_list, sql_template, feature_name, id_placeholder="{id_list}",
                      dedup_col=None, batch_size=BATCH_SIZE):
    """Run a query in batches. sql_template must have the id_placeholder placeholder."""
    all_dfs = []
    total_batches = (len(id_list) + batch_size - 1) // batch_size
    failed_batches = []

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        formatted_ids = ",".join([f"'{p}'" for p in batch])
        sql = sql_template.replace(id_placeholder, formatted_ids)
        batch_num = i // batch_size + 1

        desc = f"{feature_name} batch {batch_num}/{total_batches}"
        print(f"  [{batch_num}/{total_batches}] {feature_name} ({len(batch)} IDs)...", end="", flush=True)

        df = run_query(sql, desc)
        if len(df) == 0:
            print(f" RETRY...", end="", flush=True)
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"{desc} (retry)")
            if len(df) == 0:
                print(f" SKIPPED")
                failed_batches.append(batch_num)
                time.sleep(BATCH_DELAY)
                continue

        print(f" {len(df)} rows")
        all_dfs.append(df)

        if batch_num < total_batches:
            time.sleep(BATCH_DELAY)

    if failed_batches:
        print(f"  WARNING: {len(failed_batches)} batches failed: {failed_batches}")

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        if dedup_col and dedup_col in result.columns:
            result = result.drop_duplicates(subset=dedup_col, keep='first')
        return result
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
rpt("=" * 70)
rpt("PHASE 3D: ENRICHED FEATURE ENGINEERING")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

# ── Load existing data ──
rpt("\n[LOAD] Reading existing modeling dataset...")
nps = pd.read_csv(os.path.join(DATA, "nps_with_risk_scores.csv"), low_memory=False)
nps['phone_number'] = nps['phone_number'].astype(str).str.strip()
phones = nps['phone_number'].unique().tolist()
rpt(f"  Total respondents: {len(nps)}")
rpt(f"  Unique phones: {len(phones)}")
rpt(f"  Existing columns: {len(nps.columns)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 0: GET CUSTOMER -> DEVICE/NAS MAPPING
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[STEP 0] CUSTOMER -> DEVICE/NAS MAPPING (t_wg_customer)")
rpt("=" * 70)

MAPPING_SQL = """
SELECT
    twc.mobile,
    twc.nasid,
    twc.device_id,
    twc.shard,
    twc.lco_account_id,
    prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS lng_nas_id
FROM (
    SELECT mobile, nasid, device_id, shard, lco_account_id,
           ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer
    WHERE mobile IN ({id_list})
      AND mobile > '5999999999'
      AND _FIVETRAN_DELETED = false
) twc
WHERE twc.rn = 1
"""

mapping_df = run_batched_query(phones, MAPPING_SQL, "Customer Mapping", dedup_col="mobile")
if len(mapping_df) > 0:
    mapping_df['mobile'] = mapping_df['mobile'].astype(str)
    rpt(f"  Mapped: {len(mapping_df)} phones -> device/NAS")
    rpt(f"  Unique NASIDs: {mapping_df['nasid'].nunique()}")
    rpt(f"  Unique devices: {mapping_df['device_id'].nunique()}")
else:
    rpt("  ERROR: No mapping data returned. Cannot proceed with NAS/device-level queries.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# FEATURE SET 1: NETWORK SCORECARD (via MOBILE — direct join)
# Speed gap, optical power, speed consistency
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[1/4] NETWORK SCORECARD — Speed, Optical Power, Data Usage")
rpt("  Source: PROD_DB.PUBLIC.NETWORK_SCORECARD (6.2M rows)")
rpt("  Join key: MOBILE (phone number)")
rpt("  Date filter: 2025-06-01 to 2026-02-01 (covers NPS survey period)")
rpt("=" * 70)

SCORECARD_SQL = """
SELECT
    MOBILE,
    AVG(TRY_TO_DOUBLE(PLAN_SPEED)) AS avg_plan_speed,
    AVG(TRY_TO_DOUBLE(LATEST_SPEED)) AS avg_latest_speed,
    -- Speed gap: how much actual speed falls short of plan speed
    AVG(CASE WHEN TRY_TO_DOUBLE(PLAN_SPEED) > 0 AND TRY_TO_DOUBLE(LATEST_SPEED) IS NOT NULL
        THEN (TRY_TO_DOUBLE(PLAN_SPEED) - TRY_TO_DOUBLE(LATEST_SPEED)) / TRY_TO_DOUBLE(PLAN_SPEED)
        ELSE NULL END) AS speed_gap_pct,
    -- Speed in range: % of weeks where speed meets threshold
    AVG(TRY_TO_DOUBLE(SPEED_IN_RANGE)) AS avg_speed_in_range,
    -- Optical power (signal quality)
    AVG(TRY_TO_DOUBLE(RXPOWER)) AS avg_rxpower,
    AVG(TRY_TO_DOUBLE(RXPOWER_IN_RANGE)) AS avg_rxpower_in_range,
    AVG(TRY_TO_DOUBLE(OPTICALPOWER_IN_RANGE)) AS avg_opticalpower_in_range,
    -- Data usage
    AVG(TRY_TO_DOUBLE(DATA_USED_GB)) AS avg_weekly_data_gb,
    AVG(TRY_TO_DOUBLE(AVG_DATA_USED)) AS avg_data_used_metric,
    -- Ticket count from scorecard
    SUM(TRY_TO_NUMBER(TICKET_COUNT)) AS scorecard_ticket_count,
    -- Coverage
    COUNT(*) AS scorecard_weeks,
    -- Plan active ratio
    AVG(CASE WHEN IS_PLAN_ACTIVE = 'true' OR IS_PLAN_ACTIVE = '1' THEN 1 ELSE 0 END) AS plan_active_ratio
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE IN ({id_list})
  AND WEEK_START >= '2025-06-01'
  AND WEEK_START < '2026-02-01'
GROUP BY MOBILE
"""

scorecard_df = run_batched_query(phones, SCORECARD_SQL, "Network Scorecard", dedup_col="mobile")
if len(scorecard_df) > 0:
    scorecard_df['mobile'] = scorecard_df['mobile'].astype(str)
    rpt(f"  Matched: {len(scorecard_df)} phones ({len(scorecard_df)/len(phones)*100:.1f}%)")
    # Convert numeric columns
    num_cols = [c for c in scorecard_df.columns if c != 'mobile']
    for c in num_cols:
        scorecard_df[c] = pd.to_numeric(scorecard_df[c], errors='coerce')
    # Preview
    rpt(f"  Avg plan speed: {scorecard_df['avg_plan_speed'].mean():.1f} Mbps")
    rpt(f"  Avg latest speed: {scorecard_df['avg_latest_speed'].mean():.1f} Mbps")
    rpt(f"  Avg speed gap: {scorecard_df['speed_gap_pct'].mean()*100:.1f}%")
    rpt(f"  Avg speed in range: {scorecard_df['avg_speed_in_range'].mean():.2f}")
    rpt(f"  Avg weekly data GB: {scorecard_df['avg_weekly_data_gb'].mean():.1f}")
    rpt(f"  Avg rxpower: {scorecard_df['avg_rxpower'].mean():.2f}")
else:
    rpt("  WARNING: No scorecard data returned")


# ══════════════════════════════════════════════════════════════════════
# FEATURE SET 2: DAILY USAGE L1 (via NASID — needs mapping)
# Sessions, devices, peak-hour, volatility
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[2/4] DAILY USAGE L1 — Sessions, Devices, Peak-Hour, Volatility")
rpt("  Source: PROD_DB.PUBLIC.DAILY_USAGE_L1 (48.8M rows)")
rpt("  Join key: NASID (from customer mapping)")
rpt("  Date filter: 2025-06-01 to 2026-02-01")
rpt("=" * 70)

# Get unique NASIDs from mapping
nasids = mapping_df['nasid'].dropna().unique().tolist()
nasids = [str(n) for n in nasids if str(n) not in ('', 'None', 'nan')]
rpt(f"  Unique NASIDs to query: {len(nasids)}")

USAGE_SQL = """
SELECT
    NASID,
    AVG(SESSIONS_COUNT_DAILY) AS avg_daily_sessions,
    AVG(TOTAL_CONNECTED_DEVICES_DAILY) AS avg_daily_devices,
    AVG(ACTIVE_HOURS_COUNT_DAILY) AS avg_active_hours,
    AVG(TOTAL_DATA_BYTES_DAILY / 1073741824.0) AS avg_daily_data_gb,
    AVG(TOTAL_SESSION_DURATION_SEC_DAILY / 3600.0) AS avg_session_duration_hrs,
    AVG(AVG_SESSION_DURATION_SEC_DAILY) AS avg_avg_session_duration_sec,
    -- Peak hour usage ratio
    AVG(CASE WHEN TOTAL_DATA_BYTES_DAILY > 0
        THEN PEAK_HOUR_TOTAL_DATA_BYTES_DAILY * 1.0 / TOTAL_DATA_BYTES_DAILY
        ELSE NULL END) AS avg_peak_hour_data_ratio,
    -- Night vs day ratio
    AVG(CASE WHEN DAY_USAGE_BYTES_DAILY > 0
        THEN NIGHT_USAGE_BYTES_DAILY * 1.0 / DAY_USAGE_BYTES_DAILY
        ELSE NULL END) AS avg_night_day_ratio,
    -- Usage volatility
    AVG(USAGE_VOLATILITY_INDEX_DAILY) AS avg_usage_volatility,
    AVG(DATA_SPREAD_FACTOR_DAILY) AS avg_data_spread_factor,
    -- Connection stability proxy: % of days with many short sessions flag
    AVG(CASE WHEN MANY_SHORT_SESSIONS_FLAG_DAILY = 1 THEN 1 ELSE 0 END) AS pct_days_many_short_sessions,
    -- Distinct IPs (device diversity proxy)
    AVG(DISTINCT_FRAMEDIP_COUNT_DAILY) AS avg_distinct_ips,
    -- Coverage
    COUNT(*) AS usage_data_days
FROM PROD_DB.PUBLIC.DAILY_USAGE_L1
WHERE NASID IN ({id_list})
  AND DATE_IST >= '2025-06-01'
  AND DATE_IST < '2026-02-01'
GROUP BY NASID
"""

usage_df = run_batched_query(nasids, USAGE_SQL, "Daily Usage L1", dedup_col="nasid")
if len(usage_df) > 0:
    usage_df['nasid'] = usage_df['nasid'].astype(str)
    rpt(f"  Matched: {len(usage_df)} NASIDs ({len(usage_df)/len(nasids)*100:.1f}%)")
    # Convert numeric
    num_cols = [c for c in usage_df.columns if c != 'nasid']
    for c in num_cols:
        usage_df[c] = pd.to_numeric(usage_df[c], errors='coerce')
    # Preview
    rpt(f"  Avg daily sessions: {usage_df['avg_daily_sessions'].mean():.1f}")
    rpt(f"  Avg daily devices: {usage_df['avg_daily_devices'].mean():.1f}")
    rpt(f"  Avg daily data GB: {usage_df['avg_daily_data_gb'].mean():.1f}")
    rpt(f"  Avg peak-hour ratio: {usage_df['avg_peak_hour_data_ratio'].mean():.2f}")
    rpt(f"  Avg volatility index: {usage_df['avg_usage_volatility'].mean():.2f}")
    rpt(f"  Pct days with many short sessions: {usage_df['pct_days_many_short_sessions'].mean()*100:.1f}%")
else:
    rpt("  WARNING: No usage data returned")


# ══════════════════════════════════════════════════════════════════════
# FEATURE SET 3: CUSTOMER INFLUX SUMMARY (via LNG_NAS — needs mapping)
# Customer-level uptime, peak interruption
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[3/4] CUSTOMER INFLUX SUMMARY — Customer Uptime, Peak Interruption")
rpt("  Source: PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY (17M rows)")
rpt("  Join key: LNG_NAS (from customer mapping)")
rpt("  Date filter: 2025-06-01 to 2026-02-01")
rpt("=" * 70)

# Get unique LNG_NAS IDs from mapping
lng_nas_ids = mapping_df['lng_nas_id'].dropna().unique().tolist()
lng_nas_ids = [str(n) for n in lng_nas_ids if str(n) not in ('', 'None', 'nan')]
rpt(f"  Unique LNG_NAS IDs to query: {len(lng_nas_ids)}")

INFLUX_SQL = """
SELECT
    LNG_NAS,
    -- Customer uptime: pings received / expected
    AVG(CASE WHEN EXPECTED_PINGS_SO_FAR > 0
        THEN PINGS_RECEIVED_TODAY * 1.0 / EXPECTED_PINGS_SO_FAR
        ELSE NULL END) AS avg_customer_uptime_pct,
    -- Stable ping ratio
    AVG(CASE WHEN PINGS_RECEIVED_TODAY > 0
        THEN STABLE_PINGS_COUNT * 1.0 / PINGS_RECEIVED_TODAY
        ELSE NULL END) AS avg_stable_ping_ratio,
    -- Peak hour uptime
    AVG(CASE WHEN EXPECTED_PEAK_PINGS > 0
        THEN PINGS_PEAK_HOURS * 1.0 / EXPECTED_PEAK_PINGS
        ELSE NULL END) AS avg_peak_uptime_pct,
    -- Peak interruption rate
    AVG(CASE WHEN HAD_PEAK_INTERRUPTION = true THEN 1 ELSE 0 END) AS peak_interruption_rate,
    -- Days active
    SUM(CASE WHEN IS_ACTIVE_TODAY = true THEN 1 ELSE 0 END) AS days_active,
    COUNT(*) AS influx_data_days,
    -- Active ratio
    AVG(CASE WHEN IS_ACTIVE_TODAY = true THEN 1 ELSE 0 END) AS active_day_ratio
FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY
WHERE LNG_NAS IN ({id_list})
  AND APPENDED_DATE >= '2025-06-01'
  AND APPENDED_DATE < '2026-02-01'
GROUP BY LNG_NAS
"""

influx_df = run_batched_query(lng_nas_ids, INFLUX_SQL, "Customer Influx", dedup_col="lng_nas")
if len(influx_df) > 0:
    influx_df['lng_nas'] = influx_df['lng_nas'].astype(str)
    rpt(f"  Matched: {len(influx_df)} LNG_NAS IDs ({len(influx_df)/len(lng_nas_ids)*100:.1f}%)")
    num_cols = [c for c in influx_df.columns if c != 'lng_nas']
    for c in num_cols:
        influx_df[c] = pd.to_numeric(influx_df[c], errors='coerce')
    rpt(f"  Avg customer uptime: {influx_df['avg_customer_uptime_pct'].mean()*100:.1f}%")
    rpt(f"  Avg peak uptime: {influx_df['avg_peak_uptime_pct'].mean()*100:.1f}%")
    rpt(f"  Peak interruption rate: {influx_df['peak_interruption_rate'].mean()*100:.1f}%")
    rpt(f"  Active day ratio: {influx_df['active_day_ratio'].mean()*100:.1f}%")
else:
    rpt("  WARNING: No influx data returned")


# ══════════════════════════════════════════════════════════════════════
# FEATURE SET 4: ENRICHED SERVICE TICKET MODEL (via DEVICE_ID)
# Richer ticket metrics: categories, reopens, calls, ratings
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("[4/4] ENRICHED SERVICE TICKETS — Categories, Reopens, Calls, Ratings")
rpt("  Source: PROD_DB.PUBLIC.SERVICE_TICKET_MODEL (2.3M rows)")
rpt("  Join key: DEVICE_ID (from customer mapping)")
rpt("  Date filter: 2025-06-01 to 2026-02-01")
rpt("=" * 70)

# Get unique device IDs from mapping
device_ids = mapping_df['device_id'].dropna().unique().tolist()
device_ids = [str(d) for d in device_ids if str(d) not in ('', 'None', 'nan')]
rpt(f"  Unique device IDs to query: {len(device_ids)}")

TICKET_ENRICHED_SQL = """
SELECT
    DEVICE_ID,
    -- Volume
    COUNT(*) AS enriched_total_tickets,
    COUNT(CASE WHEN CX_PX = 'Cx' THEN 1 END) AS cx_ticket_count,
    COUNT(CASE WHEN CX_PX = 'Px' THEN 1 END) AS px_ticket_count,
    -- Categories (top ticket types)
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%internet supply down%' THEN 1 END) AS tickets_internet_down,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%disconnect%' OR LOWER(FIRST_TITLE) LIKE '%discontinue%' THEN 1 END) AS tickets_disconnect,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%recharge done but%' THEN 1 END) AS tickets_recharge_not_working,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%frequent disconnection%' THEN 1 END) AS tickets_frequent_disconnection,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%slow speed%' OR LOWER(FIRST_TITLE) LIKE '%range issue%' THEN 1 END) AS tickets_slow_speed,
    -- Resolution quality
    AVG(TIMES_REOPENED) AS avg_times_reopened,
    AVG(TIMES_REOPENED_POSTCLOSURE) AS avg_reopened_postclosure,
    SUM(CASE WHEN TIMES_REOPENED > 0 THEN 1 ELSE 0 END) AS tickets_with_reopens,
    -- TAT metrics
    AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) AS avg_resolution_mins_calendar,
    AVG(RESOLUTION_PERIOD_MINS_WORKINGHRS) AS avg_resolution_mins_working,
    AVG(ALLOCATIONTAT_MINS_CALENDARHRS) AS avg_allocation_tat_mins,
    -- SLA compliance
    COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END) AS tickets_within_tat,
    COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'beyond 2*TAT' THEN 1 END) AS tickets_beyond_2x_tat,
    -- Customer/Partner interactions
    AVG(NO_TIMES_CUSTOMER_CALLED) AS avg_customer_calls_per_ticket,
    AVG(NO_TIMES_PARTNER_CALLED) AS avg_partner_calls_per_ticket,
    SUM(NO_TIMES_CUSTOMER_CALLED) AS total_customer_calls,
    SUM(NO_TIMES_PARTNER_CALLED) AS total_partner_calls,
    -- Customer satisfaction
    AVG(CASE WHEN RATING_SCORE_BY_CUSTOMER IS NOT NULL AND RATING_SCORE_BY_CUSTOMER > 0
        THEN RATING_SCORE_BY_CUSTOMER ELSE NULL END) AS avg_ticket_rating,
    -- Resolution type
    COUNT(CASE WHEN IS_RESOLVED = true OR IS_RESOLVED = 'true' THEN 1 END) AS tickets_resolved,
    COUNT(CASE WHEN ISPINGRESOLVED = true OR ISPINGRESOLVED = 'true' THEN 1 END) AS tickets_ping_resolved,
    -- Assignment
    COUNT(CASE WHEN IS_PARTNERASSIGNED = true OR IS_PARTNERASSIGNED = 'true' THEN 1 END) AS tickets_partner_assigned,
    COUNT(CASE WHEN IS_WIOMASSIGNED = true OR IS_WIOMASSIGNED = 'true' THEN 1 END) AS tickets_wiom_assigned
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE DEVICE_ID IN ({id_list})
  AND TICKET_ADDED_TIME >= '2025-06-01'
  AND TICKET_ADDED_TIME < '2026-02-01'
GROUP BY DEVICE_ID
"""

ticket_df = run_batched_query(device_ids, TICKET_ENRICHED_SQL, "Enriched Tickets", dedup_col="device_id")
if len(ticket_df) > 0:
    ticket_df['device_id'] = ticket_df['device_id'].astype(str)
    rpt(f"  Matched: {len(ticket_df)} devices ({len(ticket_df)/len(device_ids)*100:.1f}%)")
    num_cols = [c for c in ticket_df.columns if c != 'device_id']
    for c in num_cols:
        ticket_df[c] = pd.to_numeric(ticket_df[c], errors='coerce')
    rpt(f"  Avg total tickets: {ticket_df['enriched_total_tickets'].mean():.1f}")
    rpt(f"  Avg internet down tickets: {ticket_df['tickets_internet_down'].mean():.1f}")
    rpt(f"  Avg times reopened: {ticket_df['avg_times_reopened'].mean():.2f}")
    rpt(f"  Avg resolution mins (calendar): {ticket_df['avg_resolution_mins_calendar'].mean():.0f}")
    rpt(f"  Avg customer calls/ticket: {ticket_df['avg_customer_calls_per_ticket'].mean():.2f}")
    rpt(f"  Avg ticket rating: {ticket_df['avg_ticket_rating'].mean():.2f}")
else:
    rpt("  WARNING: No enriched ticket data returned")


# ══════════════════════════════════════════════════════════════════════
# MERGE ALL NEW FEATURES INTO EXISTING DATASET
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("MERGING ALL NEW FEATURES")
rpt("=" * 70)

merged = nps.copy()
original_col_count = len(merged.columns)

# Create mapping lookup: phone -> nasid, device_id, lng_nas_id
mapping_df['mobile'] = mapping_df['mobile'].astype(str)
mapping_df['nasid'] = mapping_df['nasid'].astype(str)
mapping_df['device_id'] = mapping_df['device_id'].astype(str)
mapping_df['lng_nas_id'] = mapping_df['lng_nas_id'].astype(str)

# Add mapping columns if not already present
if 'device_id_mapped' not in merged.columns:
    map_cols = mapping_df[['mobile', 'nasid', 'device_id', 'lng_nas_id']].rename(columns={
        'mobile': 'phone_number',
        'nasid': 'nasid_mapped',
        'device_id': 'device_id_mapped',
        'lng_nas_id': 'lng_nas_id_mapped'
    })
    merged = merged.merge(map_cols, on='phone_number', how='left')
    n = merged['device_id_mapped'].notna().sum()
    rpt(f"  + Mapping: {n}/{len(merged)} phones mapped ({n/len(merged)*100:.1f}%)")

# 1. NETWORK SCORECARD — merge via phone (MOBILE)
if len(scorecard_df) > 0:
    scorecard_df_renamed = scorecard_df.rename(columns={'mobile': 'phone_number'})
    sc_cols = [c for c in scorecard_df_renamed.columns if c != 'phone_number']
    # Prefix to avoid collisions
    scorecard_df_renamed = scorecard_df_renamed.rename(
        columns={c: f'sc_{c}' if not c.startswith('sc_') else c for c in sc_cols}
    )
    merged = merged.merge(scorecard_df_renamed, on='phone_number', how='left')
    n = merged['sc_avg_plan_speed'].notna().sum()
    rpt(f"  + Network Scorecard: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    rpt("  - Network Scorecard: skipped (no data)")

# 2. DAILY USAGE L1 — merge via nasid
if len(usage_df) > 0:
    usage_df_renamed = usage_df.rename(columns={'nasid': 'nasid_mapped'})
    us_cols = [c for c in usage_df_renamed.columns if c != 'nasid_mapped']
    usage_df_renamed = usage_df_renamed.rename(
        columns={c: f'ul1_{c}' if not c.startswith('ul1_') else c for c in us_cols}
    )
    merged = merged.merge(usage_df_renamed, on='nasid_mapped', how='left')
    n = merged['ul1_avg_daily_sessions'].notna().sum()
    rpt(f"  + Daily Usage L1: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    rpt("  - Daily Usage L1: skipped (no data)")

# 3. CUSTOMER INFLUX SUMMARY — merge via lng_nas
if len(influx_df) > 0:
    influx_df_renamed = influx_df.rename(columns={'lng_nas': 'lng_nas_id_mapped'})
    inf_cols = [c for c in influx_df_renamed.columns if c != 'lng_nas_id_mapped']
    influx_df_renamed = influx_df_renamed.rename(
        columns={c: f'cis_{c}' if not c.startswith('cis_') else c for c in inf_cols}
    )
    merged = merged.merge(influx_df_renamed, on='lng_nas_id_mapped', how='left')
    n = merged['cis_avg_customer_uptime_pct'].notna().sum()
    rpt(f"  + Customer Influx: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    rpt("  - Customer Influx: skipped (no data)")

# 4. ENRICHED TICKETS — merge via device_id
if len(ticket_df) > 0:
    ticket_df_renamed = ticket_df.rename(columns={'device_id': 'device_id_mapped'})
    tk_cols = [c for c in ticket_df_renamed.columns if c != 'device_id_mapped']
    ticket_df_renamed = ticket_df_renamed.rename(
        columns={c: f'tk_{c}' if not c.startswith('tk_') else c for c in tk_cols}
    )
    merged = merged.merge(ticket_df_renamed, on='device_id_mapped', how='left')
    n = merged['tk_enriched_total_tickets'].notna().sum()
    rpt(f"  + Enriched Tickets: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    rpt("  - Enriched Tickets: skipped (no data)")


# ══════════════════════════════════════════════════════════════════════
# DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("COMPUTING DERIVED FEATURES")
rpt("=" * 70)

# Speed gap severity buckets
if 'sc_speed_gap_pct' in merged.columns:
    merged['sc_speed_gap_pct'] = pd.to_numeric(merged['sc_speed_gap_pct'], errors='coerce')
    merged['speed_gap_severity'] = pd.cut(
        merged['sc_speed_gap_pct'],
        bins=[-999, 0, 0.2, 0.4, 0.6, 999],
        labels=['overdelivering', 'acceptable', 'moderate_gap', 'large_gap', 'severe_gap']
    )
    rpt(f"  Speed gap severity: {merged['speed_gap_severity'].value_counts().to_dict()}")

# Connection stability score (composite)
if 'ul1_pct_days_many_short_sessions' in merged.columns and 'ul1_avg_usage_volatility' in merged.columns:
    mss = pd.to_numeric(merged['ul1_pct_days_many_short_sessions'], errors='coerce')
    vol = pd.to_numeric(merged['ul1_avg_usage_volatility'], errors='coerce')
    # Higher = more unstable (normalize both to 0-1 scale)
    mss_norm = mss / mss.quantile(0.95).clip(lower=0.001)
    vol_norm = vol / vol.quantile(0.95).clip(lower=0.001)
    merged['connection_instability_score'] = (mss_norm * 0.6 + vol_norm * 0.4).clip(0, 1)
    n = merged['connection_instability_score'].notna().sum()
    rpt(f"  Connection instability score: {n} computed (0=stable, 1=very unstable)")

# Customer uptime severity
if 'cis_avg_customer_uptime_pct' in merged.columns:
    merged['cis_avg_customer_uptime_pct'] = pd.to_numeric(merged['cis_avg_customer_uptime_pct'], errors='coerce')
    merged['customer_uptime_tier'] = pd.cut(
        merged['cis_avg_customer_uptime_pct'],
        bins=[-999, 0.5, 0.7, 0.85, 0.95, 999],
        labels=['critical', 'poor', 'fair', 'good', 'excellent']
    )
    rpt(f"  Customer uptime tier: {merged['customer_uptime_tier'].value_counts().to_dict()}")

# Ticket severity composite
if 'tk_enriched_total_tickets' in merged.columns:
    merged['tk_enriched_total_tickets'] = pd.to_numeric(merged['tk_enriched_total_tickets'], errors='coerce')
    merged['tk_avg_times_reopened'] = pd.to_numeric(merged.get('tk_avg_times_reopened', pd.Series(dtype=float)), errors='coerce')
    merged['tk_tickets_beyond_2x_tat'] = pd.to_numeric(merged.get('tk_tickets_beyond_2x_tat', pd.Series(dtype=float)), errors='coerce')

    total_tk = merged['tk_enriched_total_tickets'].fillna(0)
    reopens = merged['tk_avg_times_reopened'].fillna(0)
    beyond_tat = merged['tk_tickets_beyond_2x_tat'].fillna(0) / total_tk.clip(lower=1)

    merged['ticket_severity_score'] = (
        np.log1p(total_tk) * 0.3 +
        reopens * 0.3 +
        beyond_tat * 0.4
    )
    n = merged['ticket_severity_score'].notna().sum()
    rpt(f"  Ticket severity score: {n} computed")

# Peak-hour penalty (industry expert hypothesis: peak-hour degradation matters more)
if 'cis_avg_peak_uptime_pct' in merged.columns and 'cis_avg_customer_uptime_pct' in merged.columns:
    peak = pd.to_numeric(merged['cis_avg_peak_uptime_pct'], errors='coerce')
    overall = pd.to_numeric(merged['cis_avg_customer_uptime_pct'], errors='coerce')
    merged['peak_vs_overall_uptime_gap'] = overall - peak  # positive = peak is worse
    n = merged['peak_vs_overall_uptime_gap'].notna().sum()
    rpt(f"  Peak vs overall uptime gap: {n} computed")


# ══════════════════════════════════════════════════════════════════════
# SAVE ENRICHED DATASET
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("SAVING ENRICHED DATASET")
rpt("=" * 70)

new_col_count = len(merged.columns)
new_cols = [c for c in merged.columns if c not in nps.columns]

output_csv = os.path.join(DATA, "nps_enriched_v2.csv")
merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
rpt(f"  Saved: {output_csv}")
rpt(f"  Rows: {len(merged)}")
rpt(f"  Total columns: {new_col_count} (was {original_col_count}, added {len(new_cols)})")

rpt(f"\n  New columns added ({len(new_cols)}):")
for c in sorted(new_cols):
    fill = merged[c].notna().sum()
    rpt(f"    {c:45s}: {fill:6d}/{len(merged)} ({fill/len(merged)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# QUICK ANALYSIS: NEW FEATURES BY NPS GROUP
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("NEW FEATURES BY NPS GROUP (Promoter vs Detractor)")
rpt("=" * 70)

# Find NPS group column
nps_grp_col = None
for c in merged.columns:
    if c.lower() == 'nps_group':
        nps_grp_col = c
        break

if nps_grp_col:
    key_features = [
        ('sc_avg_plan_speed', 'Plan Speed (Mbps)'),
        ('sc_avg_latest_speed', 'Actual Speed (Mbps)'),
        ('sc_speed_gap_pct', 'Speed Gap (%)'),
        ('sc_avg_speed_in_range', 'Speed In Range'),
        ('sc_avg_rxpower', 'Optical Power (dBm)'),
        ('sc_avg_weekly_data_gb', 'Weekly Data (GB)'),
        ('ul1_avg_daily_sessions', 'Daily Sessions'),
        ('ul1_avg_daily_devices', 'Daily Devices'),
        ('ul1_avg_daily_data_gb', 'Daily Data (GB)'),
        ('ul1_avg_peak_hour_data_ratio', 'Peak Hour Ratio'),
        ('ul1_pct_days_many_short_sessions', 'Pct Days Short Sessions'),
        ('ul1_avg_usage_volatility', 'Usage Volatility'),
        ('cis_avg_customer_uptime_pct', 'Customer Uptime (%)'),
        ('cis_avg_peak_uptime_pct', 'Peak Uptime (%)'),
        ('cis_peak_interruption_rate', 'Peak Interruption Rate'),
        ('tk_enriched_total_tickets', 'Total Tickets'),
        ('tk_avg_times_reopened', 'Avg Reopens'),
        ('tk_avg_resolution_mins_calendar', 'Avg Resolution Mins'),
        ('tk_avg_customer_calls_per_ticket', 'Avg Customer Calls/Ticket'),
        ('tk_avg_ticket_rating', 'Avg Ticket Rating'),
        ('connection_instability_score', 'Connection Instability'),
        ('ticket_severity_score', 'Ticket Severity'),
        ('peak_vs_overall_uptime_gap', 'Peak vs Overall Gap'),
    ]

    rpt(f"\n  {'Feature':40s} | {'Promoter':>10s} | {'Detractor':>10s} | {'Gap':>10s} | {'Signal':>8s}")
    rpt(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

    for col, label in key_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            grp = merged.groupby(nps_grp_col)[col].mean()
            prom = grp.get('Promoter', float('nan'))
            det = grp.get('Detractor', float('nan'))
            if pd.notna(prom) and pd.notna(det) and prom != 0:
                gap = det - prom
                gap_pct = abs(gap / prom * 100)
                signal = "STRONG" if gap_pct > 10 else "moderate" if gap_pct > 5 else "weak"
                rpt(f"  {label:40s} | {prom:10.3f} | {det:10.3f} | {gap:+10.3f} | {signal:>8s}")

# Also by churn
rpt("\n" + "=" * 70)
rpt("NEW FEATURES BY CHURN STATUS")
rpt("=" * 70)

churn_col = None
for c in merged.columns:
    if 'churn' in c.lower() and 'overall' in c.lower():
        churn_col = c
        break
if not churn_col:
    for c in merged.columns:
        if c.lower() == 'is_churned':
            churn_col = c
            break

if churn_col:
    rpt(f"  Using churn column: {churn_col}")
    rpt(f"\n  {'Feature':40s} | {'Active':>10s} | {'Churned':>10s} | {'Gap':>10s} | {'Signal':>8s}")
    rpt(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

    for col, label in key_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            grp = merged.groupby(churn_col)[col].mean()
            # Try different churn labels
            active_val = None
            churn_val = None
            for k in grp.index:
                k_str = str(k).lower()
                if k_str in ('active', '0', '0.0', 'false'):
                    active_val = grp[k]
                elif k_str in ('churn', 'churned', '1', '1.0', 'true'):
                    churn_val = grp[k]
            if active_val is not None and churn_val is not None and active_val != 0:
                gap = churn_val - active_val
                gap_pct = abs(gap / active_val * 100)
                signal = "STRONG" if gap_pct > 10 else "moderate" if gap_pct > 5 else "weak"
                rpt(f"  {label:40s} | {active_val:10.3f} | {churn_val:10.3f} | {gap:+10.3f} | {signal:>8s}")
else:
    rpt("  Could not find churn column")


# ══════════════════════════════════════════════════════════════════════
# SAVE REPORT
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt(f"PHASE 3D COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

report_path = os.path.join(OUTPUT, "phase3d_enrichment.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport saved: {report_path}")
print(f"Dataset saved: {output_csv}")
