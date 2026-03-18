"""
Phase 3B: Industry Expert Features
====================================
Queries Snowflake via Metabase API to compute the 10 industry-expert-recommended
NPS features for all NPS respondents. Processes in batches of 500 phones.

INDUSTRY EXPERT FEATURES:
1. Disconnection FREQUENCY (outage events per device)
2. Peak-hour quality (7-11 PM uptime at partner level)
3. First-Call Resolution Rate (tickets resolved without reopening)
4. Repeat complaints (multiple tickets for same issue)
5. Outage exposure at DEVICE level (not just partner level)
6. Recharge attempt failures (tried to pay but failed)
7. Partner dispatch decline rate (willingness to respond)
8. Time-to-value post-install (how quickly customer starts usage)
9. IVR call volume (calls before/after NPS survey)
10. Data usage patterns (proxy for speed experience)

Output:
- data/industry_expert_features.csv
- output/phase3b_industry_expert_features.txt
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

# -- Query runner --
def run_query(sql, description="query", timeout=180):
    """Execute a Snowflake query via Metabase API. Returns DataFrame."""
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if resp.status_code in (200, 202):
            data = resp.json()
            if 'data' in data and 'rows' in data['data']:
                cols = [c['name'].upper() for c in data['data']['cols']]
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


def run_batched_query(phones, sql_template, feature_name, timeout=180):
    """Run a query in batches of BATCH_SIZE. sql_template must have {phone_list} placeholder."""
    all_dfs = []
    total_batches = (len(phones) + BATCH_SIZE - 1) // BATCH_SIZE
    failed_batches = []

    for i in range(0, len(phones), BATCH_SIZE):
        batch = phones[i:i + BATCH_SIZE]
        phone_list = ",".join([f"'{p}'" for p in batch])
        sql = sql_template.format(phone_list=phone_list)
        batch_num = i // BATCH_SIZE + 1

        desc = f"{feature_name} batch {batch_num}/{total_batches}"
        print(f"  [{batch_num}/{total_batches}] {feature_name} (phones {i+1}-{i+len(batch)})...", end="", flush=True)

        df = run_query(sql, desc, timeout=timeout)
        if len(df) == 0:
            print(f" RETRY...", end="", flush=True)
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"{desc} (retry)", timeout=timeout)
            if len(df) == 0:
                print(f" SKIPPED (0 rows after retry)")
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
        if 'MOBILE' in result.columns:
            result = result.drop_duplicates(subset='MOBILE', keep='first')
        return result
    return pd.DataFrame()


# ======================================================================
print("=" * 70)
print("PHASE 3B: INDUSTRY EXPERT FEATURES")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# -- Load NPS base --
print("\n[LOAD] Reading NPS base data...")
nps = pd.read_csv(os.path.join(DATA, "nps_clean_base.csv"), low_memory=False)
nps['phone_number'] = nps['phone_number'].astype(str).str.strip()
phones = nps['phone_number'].unique().tolist()
print(f"  Total respondents: {len(nps)}")
print(f"  Unique phones: {len(phones)}")
print(f"  Batches needed: {(len(phones) + BATCH_SIZE - 1) // BATCH_SIZE} (batch size={BATCH_SIZE})")

# We need sprint dates for time-scoping
nps['Sprint Start Date'] = pd.to_datetime(nps['Sprint Start Date'], errors='coerce')
nps['Sprint End Date'] = pd.to_datetime(nps['Sprint End Date'], errors='coerce')

# ======================================================================
# STEP 0: GET CUSTOMER -> DEVICE MAPPING (needed for features 1,5,10)
# ======================================================================
print("\n" + "-" * 70)
print("[0/10] CUSTOMER-DEVICE MAPPING (t_wg_customer)")
print("-" * 70)

DEVICE_MAP_SQL = """
SELECT MOBILE, DEVICE_ID, NASID,
       prod_db.public.idmaker(SHARD, 4, LCO_ACCOUNT_ID) AS PARTNER_LNG_ID,
       ROW_NUMBER() OVER (PARTITION BY MOBILE ORDER BY ADDED_TIME DESC) AS RN
FROM prod_db.public.t_wg_customer
WHERE MOBILE IN ({phone_list})
  AND MOBILE > '5999999999'
  AND _FIVETRAN_DELETED = false
"""

device_map_df = run_batched_query(phones, DEVICE_MAP_SQL, "Device Map")
if len(device_map_df) > 0:
    # Keep latest device per phone
    device_map_df = device_map_df[device_map_df['RN'].astype(int) == 1].copy()
    device_map_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    device_map_df['phone_number'] = device_map_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(device_map_df)} phones with device mapping ({len(device_map_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No device mapping returned")

# ======================================================================
# FEATURE 1 & 5: DISCONNECTION FREQUENCY + DEVICE-LEVEL OUTAGE EXPOSURE
# ======================================================================
print("\n" + "-" * 70)
print("[1,5/10] DISCONNECTION FREQUENCY + DEVICE-LEVEL OUTAGE EXPOSURE")
print("         (IMPACTED_DEVICES via device_id)")
print("-" * 70)
print("  NOTE: Outage data available Dec 20, 2025 - Jan 8, 2026 (~20 days)")

outage_df = pd.DataFrame()
if len(device_map_df) > 0:
    device_ids = device_map_df['DEVICE_ID'].dropna().unique().tolist()
    print(f"  Unique device_ids to check: {len(device_ids)}")

    # Query in batches of device_ids
    all_outage_dfs = []
    total_dev_batches = (len(device_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(device_ids), BATCH_SIZE):
        batch = device_ids[i:i + BATCH_SIZE]
        dev_list = ",".join([f"'{d}'" for d in batch])
        batch_num = i // BATCH_SIZE + 1
        print(f"  [{batch_num}/{total_dev_batches}] Outage devices ({i+1}-{i+len(batch)})...", end="", flush=True)

        sql = f"""
        SELECT
            DEVICE_ID,
            COUNT(*) AS outage_events,
            COUNT(DISTINCT ALERT_ID) AS distinct_outages,
            MIN(WINDOW_DATETIME) AS first_outage,
            MAX(WINDOW_DATETIME) AS last_outage,
            COUNT(CASE WHEN DEVICE_RECOVERY_STATE = 'RECOVERED' THEN 1 END) AS recovered_events,
            AVG(CASE WHEN RECOVERY_DATETIME IS NOT NULL
                THEN DATEDIFF(minute, WINDOW_DATETIME, RECOVERY_DATETIME) END) AS avg_recovery_mins
        FROM PROD_DB.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC.IMPACTED_DEVICES
        WHERE DEVICE_ID IN ({dev_list})
          AND _FIVETRAN_DELETED = false
        GROUP BY DEVICE_ID
        """
        df = run_query(sql, f"Outage batch {batch_num}", timeout=180)
        if len(df) == 0:
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"Outage batch {batch_num} (retry)", timeout=180)
        if len(df) > 0:
            print(f" {len(df)} devices")
            all_outage_dfs.append(df)
        else:
            print(f" 0 devices")
        time.sleep(BATCH_DELAY)

    if all_outage_dfs:
        outage_df = pd.concat(all_outage_dfs, ignore_index=True)
        outage_df = outage_df.drop_duplicates(subset='DEVICE_ID', keep='first')
        print(f"  TOTAL: {len(outage_df)} devices with outage data")
    else:
        print("  WARNING: No outage data found for NPS respondent devices")
else:
    print("  SKIPPED: No device mapping available")


# ======================================================================
# FEATURE 2: PEAK-HOUR QUALITY (partner-level)
# ======================================================================
print("\n" + "-" * 70)
print("[2/10] PEAK-HOUR QUALITY (PARTNER_INFLUX_SUMMARY peak hour columns)")
print("-" * 70)

PEAK_HOUR_SQL = """
SELECT
    PARTNER_ID,
    AVG(PEAK_HOUR_PINGS_RECEIVED * 1.0 / NULLIF(PEAK_HOUR_EXPECTED_PINGS, 0)) AS peak_uptime_pct,
    AVG(PEAK_HOUR_STABLE_PINGS * 1.0 / NULLIF(PEAK_HOUR_EXPECTED_PINGS, 0)) AS peak_stable_pct,
    AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) AS overall_uptime_pct,
    AVG(CUSTOMERS_WITH_PEAK_INTERRUPTIONS) AS avg_peak_interruptions,
    AVG(CASE WHEN PEAK_HOUR_EXPECTED_PINGS > 0
        THEN (TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0))
             - (PEAK_HOUR_PINGS_RECEIVED * 1.0 / NULLIF(PEAK_HOUR_EXPECTED_PINGS, 0))
        END) AS peak_vs_overall_gap,
    COUNT(*) AS uptime_data_days
FROM prod_db.public.PARTNER_INFLUX_SUMMARY
WHERE DATEADD(day, -1, appended_date) >= DATEADD(day, -90, CURRENT_DATE())
GROUP BY PARTNER_ID
HAVING COUNT(*) >= 10
"""

print("  Running global partner peak-hour uptime query...", end="", flush=True)
peak_hour_df = run_query(PEAK_HOUR_SQL, "Peak hour uptime (global)", timeout=180)
if len(peak_hour_df) > 0:
    print(f" {len(peak_hour_df)} partners with peak hour data")
    for col in ['PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT', 'OVERALL_UPTIME_PCT',
                'AVG_PEAK_INTERRUPTIONS', 'PEAK_VS_OVERALL_GAP']:
        if col in peak_hour_df.columns:
            peak_hour_df[col] = pd.to_numeric(peak_hour_df[col], errors='coerce')
    mean_peak = peak_hour_df['PEAK_UPTIME_PCT'].mean() if 'PEAK_UPTIME_PCT' in peak_hour_df.columns else 0
    mean_overall = peak_hour_df['OVERALL_UPTIME_PCT'].mean() if 'OVERALL_UPTIME_PCT' in peak_hour_df.columns else 0
    print(f"  Mean peak uptime: {mean_peak:.4f}")
    print(f"  Mean overall uptime: {mean_overall:.4f}")
    print(f"  Peak-vs-overall gap: {mean_overall - mean_peak:.4f}")
else:
    print(" WARNING: No peak hour data returned")


# ======================================================================
# FEATURE 3: FIRST-CALL RESOLUTION RATE
# ======================================================================
print("\n" + "-" * 70)
print("[3/10] FIRST-CALL RESOLUTION RATE (service_ticket_model)")
print("-" * 70)

FCR_SQL = """
WITH customer_devices AS (
    SELECT DISTINCT MOBILE, DEVICE_ID
    FROM prod_db.public.t_wg_customer
    WHERE MOBILE IN ({phone_list})
      AND MOBILE > '5999999999'
      AND _FIVETRAN_DELETED = false
),
tickets AS (
    SELECT
        cd.MOBILE,
        stm.TICKET_ID,
        stm.TIMES_REOPENED,
        stm.TIMES_REOPENED_POSTCLOSURE,
        stm.IS_RESOLVED,
        stm.NO_TIMES_CUSTOMER_CALLED,
        stm.NO_TIMES_PARTNER_CALLED,
        stm.RATING_SCORE_BY_CUSTOMER,
        stm.RESOLUTION_TAT_BUCKET,
        stm.CX_PX
    FROM customer_devices cd
    JOIN prod_db.public.service_ticket_model stm ON cd.DEVICE_ID = stm.DEVICE_ID
    WHERE stm.CX_PX != 'CC'
)
SELECT
    MOBILE,
    COUNT(*) AS total_tickets_fcr,
    COUNT(CASE WHEN TIMES_REOPENED = 0 AND IS_RESOLVED = 1 THEN 1 END) AS fcr_tickets,
    COUNT(CASE WHEN TIMES_REOPENED = 0 AND IS_RESOLVED = 1 THEN 1 END) * 100.0
        / NULLIF(COUNT(CASE WHEN IS_RESOLVED = 1 THEN 1 END), 0) AS fcr_rate,
    AVG(TIMES_REOPENED) AS avg_times_reopened,
    MAX(TIMES_REOPENED) AS max_times_reopened,
    AVG(NO_TIMES_CUSTOMER_CALLED) AS avg_customer_calls_per_ticket,
    SUM(NO_TIMES_CUSTOMER_CALLED) AS total_customer_calls,
    AVG(NO_TIMES_PARTNER_CALLED) AS avg_partner_calls_per_ticket,
    AVG(RATING_SCORE_BY_CUSTOMER) AS avg_ticket_rating,
    COUNT(CASE WHEN TIMES_REOPENED >= 1 THEN 1 END) AS tickets_reopened_once,
    COUNT(CASE WHEN TIMES_REOPENED >= 3 THEN 1 END) AS tickets_reopened_3plus
FROM tickets
GROUP BY MOBILE
"""

fcr_df = run_batched_query(phones, FCR_SQL, "FCR")
if len(fcr_df) > 0:
    fcr_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    fcr_df['phone_number'] = fcr_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(fcr_df)} phones with FCR data ({len(fcr_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No FCR data returned")


# ======================================================================
# FEATURE 4: REPEAT COMPLAINTS
# ======================================================================
print("\n" + "-" * 70)
print("[4/10] REPEAT COMPLAINTS (multiple tickets same title/device)")
print("-" * 70)

REPEAT_SQL = """
WITH customer_devices AS (
    SELECT DISTINCT MOBILE, DEVICE_ID
    FROM prod_db.public.t_wg_customer
    WHERE MOBILE IN ({phone_list})
      AND MOBILE > '5999999999'
      AND _FIVETRAN_DELETED = false
),
tickets AS (
    SELECT
        cd.MOBILE,
        stm.TICKET_ID,
        stm.FIRST_TITLE,
        stm.DEVICE_ID,
        stm.TICKET_ADDED_TIME,
        stm.CX_PX,
        stm.IS_RESOLVED
    FROM customer_devices cd
    JOIN prod_db.public.service_ticket_model stm ON cd.DEVICE_ID = stm.DEVICE_ID
    WHERE stm.CX_PX != 'CC'
),
title_counts AS (
    SELECT
        MOBILE,
        FIRST_TITLE,
        COUNT(*) AS tickets_with_same_title
    FROM tickets
    GROUP BY MOBILE, FIRST_TITLE
)
SELECT
    t.MOBILE,
    COUNT(DISTINCT t.TICKET_ID) AS total_tickets_repeat,
    COUNT(DISTINCT t.FIRST_TITLE) AS distinct_issue_types,
    MAX(tc.tickets_with_same_title) AS max_tickets_same_issue,
    AVG(tc.tickets_with_same_title) AS avg_tickets_per_issue,
    COUNT(DISTINCT CASE WHEN tc.tickets_with_same_title >= 3 THEN t.FIRST_TITLE END) AS issues_with_3plus_tickets,
    CASE WHEN MAX(tc.tickets_with_same_title) >= 3 THEN 1 ELSE 0 END AS has_repeat_complaint
FROM tickets t
JOIN title_counts tc ON t.MOBILE = tc.MOBILE AND t.FIRST_TITLE = tc.FIRST_TITLE
GROUP BY t.MOBILE
"""

repeat_df = run_batched_query(phones, REPEAT_SQL, "Repeat Complaints")
if len(repeat_df) > 0:
    repeat_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    repeat_df['phone_number'] = repeat_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(repeat_df)} phones with repeat complaint data ({len(repeat_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No repeat complaint data returned")


# ======================================================================
# FEATURE 6: RECHARGE ATTEMPT FAILURES
# ======================================================================
print("\n" + "-" * 70)
print("[6/10] RECHARGE ATTEMPT FAILURES (payment_logs)")
print("-" * 70)

PAYMENT_FAIL_SQL = """
SELECT
    MOBILE,
    COUNT(CASE WHEN EVENT_NAME = 'order_failed' THEN 1 END) AS payment_failures,
    COUNT(CASE WHEN EVENT_NAME = 'order_succeeded' THEN 1 END) AS payment_successes,
    COUNT(CASE WHEN EVENT_NAME = 'order_created' THEN 1 END) AS payment_attempts,
    COUNT(*) AS total_payment_events,
    COUNT(CASE WHEN EVENT_NAME = 'order_failed' THEN 1 END) * 100.0
        / NULLIF(COUNT(CASE WHEN EVENT_NAME IN ('order_failed', 'order_succeeded') THEN 1 END), 0)
        AS failure_rate_pct,
    MIN(DATEADD(minute, 330, ADDED_TIME)) AS first_payment_event,
    MAX(DATEADD(minute, 330, ADDED_TIME)) AS last_payment_event
FROM prod_db.public.payment_logs
WHERE MOBILE IN ({phone_list})
  AND MOBILE > '5999999999'
  AND EVENT_NAME IN ('order_failed', 'order_succeeded', 'order_created')
GROUP BY MOBILE
"""

payment_fail_df = run_batched_query(phones, PAYMENT_FAIL_SQL, "Payment Failures")
if len(payment_fail_df) > 0:
    payment_fail_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    payment_fail_df['phone_number'] = payment_fail_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(payment_fail_df)} phones with payment failure data ({len(payment_fail_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No payment failure data returned")


# ======================================================================
# FEATURE 7: PARTNER DISPATCH DECLINE RATE
# ======================================================================
print("\n" + "-" * 70)
print("[7/10] PARTNER DISPATCH DECLINE RATE (profile_lead_model)")
print("-" * 70)

DISPATCH_SQL = """
SELECT
    MOBILE,
    COUNT(*) AS total_leads,
    COUNT(CASE WHEN ALL_PARTNERS_DECLINED = 1 THEN 1 END) AS leads_all_declined,
    COUNT(LEAD_FIRST_ACCEPTANCE_TIME) AS leads_accepted,
    COUNT(LEAD_INSTALLATION_TIME) AS leads_installed,
    AVG(LEAD_INSTALLATION_TAT) AS avg_install_tat_mins,
    MIN(DATEADD(minute, 330, LEAD_INSTALLATION_TIME)) AS first_install_time,
    MAX(DATEADD(minute, 330, LEAD_INSTALLATION_TIME)) AS last_install_time,
    AVG(INSTALLATION_RATING_CUSTOMER_TO_PARTNER) AS avg_install_rating,
    COUNT(CASE WHEN ALL_PARTNERS_DECLINED = 1 THEN 1 END) * 100.0
        / NULLIF(COUNT(*), 0) AS dispatch_decline_rate_pct
FROM prod_db.public.profile_lead_model
WHERE MOBILE IN ({phone_list})
  AND MOBILE > '5999999999'
GROUP BY MOBILE
"""

dispatch_df = run_batched_query(phones, DISPATCH_SQL, "Dispatch Decline")
if len(dispatch_df) > 0:
    dispatch_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    dispatch_df['phone_number'] = dispatch_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(dispatch_df)} phones with dispatch data ({len(dispatch_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No dispatch data returned")


# ======================================================================
# FEATURE 8: TIME-TO-VALUE POST-INSTALL
# ======================================================================
print("\n" + "-" * 70)
print("[8/10] TIME-TO-VALUE POST-INSTALL (profile_lead_model + t_router_user_mapping)")
print("-" * 70)

TTV_SQL = """
WITH install_time AS (
    SELECT MOBILE,
           MIN(DATEADD(minute, 330, LEAD_INSTALLATION_TIME)) AS install_ts,
           MIN(FIRST_PING_POST_INSTALLATION) AS first_ping_str
    FROM prod_db.public.profile_lead_model
    WHERE MOBILE IN ({phone_list})
      AND MOBILE > '5999999999'
      AND LEAD_INSTALLATION_TIME IS NOT NULL
    GROUP BY MOBILE
),
first_recharge AS (
    SELECT MOBILE,
           MIN(DATEADD(minute, 330, CREATED_ON)) AS first_recharge_ts
    FROM prod_db.public.t_router_user_mapping
    WHERE MOBILE IN ({phone_list})
      AND MOBILE > '5999999999'
      AND DEVICE_LIMIT = '10'
      AND OTP = 'DONE'
      AND STORE_GROUP_ID = 0
    GROUP BY MOBILE
)
SELECT
    it.MOBILE,
    it.install_ts,
    it.first_ping_str,
    fr.first_recharge_ts,
    DATEDIFF(hour, it.install_ts, fr.first_recharge_ts) AS hours_to_first_recharge,
    DATEDIFF(day, it.install_ts, fr.first_recharge_ts) AS days_to_first_recharge
FROM install_time it
LEFT JOIN first_recharge fr ON it.MOBILE = fr.MOBILE
"""

ttv_df = run_batched_query(phones, TTV_SQL, "Time-to-Value")
if len(ttv_df) > 0:
    ttv_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    ttv_df['phone_number'] = ttv_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(ttv_df)} phones with TTV data ({len(ttv_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No TTV data returned")


# ======================================================================
# FEATURE 9: IVR CALL VOLUME
# ======================================================================
print("\n" + "-" * 70)
print("[9/10] IVR CALL VOLUME (tata_ivr_events)")
print("       CLIENT_NUMBER format: +91XXXXXXXXXX -- stripping prefix")
print("-" * 70)

# IVR has +91 prefix, our phones are 10-digit. We need to match using RIGHT(CLIENT_NUMBER, 10)
IVR_SQL = """
SELECT
    RIGHT(CLIENT_NUMBER, 10) AS MOBILE,
    COUNT(*) AS total_ivr_calls,
    COUNT(CASE WHEN DIRECTION = 'inbound' THEN 1 END) AS inbound_calls,
    COUNT(CASE WHEN DIRECTION = 'outbound' THEN 1 END) AS outbound_calls,
    COUNT(CASE WHEN STATUS = 'answered' THEN 1 END) AS answered_calls,
    COUNT(CASE WHEN STATUS = 'missed' THEN 1 END) AS missed_calls,
    COUNT(CASE WHEN STATUS = 'dropped' THEN 1 END) AS dropped_calls,
    AVG(CASE WHEN STATUS = 'answered' THEN ANSWERED_SECONDS END) AS avg_answered_seconds,
    AVG(CALL_DURATION) AS avg_call_duration,
    MIN(DATEADD(minute, 330, TIMESTAMP)) AS first_call_time,
    MAX(DATEADD(minute, 330, TIMESTAMP)) AS last_call_time,
    COUNT(CASE WHEN DIRECTION = 'inbound' AND STATUS = 'answered' THEN 1 END) AS inbound_answered,
    COUNT(CASE WHEN DIRECTION = 'inbound' AND STATUS != 'answered' THEN 1 END) AS inbound_unanswered
FROM prod_db.public.tata_ivr_events
WHERE RIGHT(CLIENT_NUMBER, 10) IN ({phone_list})
  AND LENGTH(CLIENT_NUMBER) >= 10
GROUP BY RIGHT(CLIENT_NUMBER, 10)
"""

ivr_df = run_batched_query(phones, IVR_SQL, "IVR Calls", timeout=300)
if len(ivr_df) > 0:
    ivr_df.rename(columns={'MOBILE': 'phone_number'}, inplace=True)
    ivr_df['phone_number'] = ivr_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(ivr_df)} phones with IVR data ({len(ivr_df)/len(phones)*100:.1f}%)")
else:
    print("  WARNING: No IVR data returned")


# ======================================================================
# FEATURE 10: DATA USAGE PATTERNS (NAS-level, via t_wg_customer.nasid)
# ======================================================================
print("\n" + "-" * 70)
print("[10/10] DATA USAGE PATTERNS (data_usage_okr via NAS)")
print("        Note: NAS-level data (shared router), not per-customer")
print("-" * 70)

data_usage_df = pd.DataFrame()
if len(device_map_df) > 0:
    nas_ids = device_map_df['NASID'].dropna().unique().tolist()
    # Convert to string for matching
    nas_ids = [str(int(float(n))) for n in nas_ids if pd.notna(n)]
    print(f"  Unique NAS IDs to check: {len(nas_ids)}")

    all_usage_dfs = []
    total_nas_batches = (len(nas_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(nas_ids), BATCH_SIZE):
        batch = nas_ids[i:i + BATCH_SIZE]
        nas_list = ",".join([f"'{n}'" for n in batch])
        batch_num = i // BATCH_SIZE + 1
        print(f"  [{batch_num}/{total_nas_batches}] Data usage NAS ({i+1}-{i+len(batch)})...", end="", flush=True)

        sql = f"""
        SELECT
            NASID,
            COUNT(*) AS usage_data_days,
            AVG(TOTAL_DATA_USED) AS avg_daily_data_gb,
            MEDIAN(TOTAL_DATA_USED) AS median_daily_data_gb,
            STDDEV(TOTAL_DATA_USED) AS stddev_daily_data_gb,
            MIN(TOTAL_DATA_USED) AS min_daily_data_gb,
            MAX(TOTAL_DATA_USED) AS max_daily_data_gb,
            MIN(DT) AS first_usage_date,
            MAX(DT) AS last_usage_date,
            COUNT(CASE WHEN TOTAL_DATA_USED < 1 THEN 1 END) AS low_usage_days,
            COUNT(CASE WHEN TOTAL_DATA_USED > 50 THEN 1 END) AS high_usage_days
        FROM prod_db.public.data_usage_okr
        WHERE NASID IN ({nas_list})
        GROUP BY NASID
        """
        df = run_query(sql, f"Data usage batch {batch_num}", timeout=180)
        if len(df) == 0:
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"Data usage batch {batch_num} (retry)", timeout=180)
        if len(df) > 0:
            print(f" {len(df)} NAS")
            all_usage_dfs.append(df)
        else:
            print(f" 0 NAS")
        time.sleep(BATCH_DELAY)

    if all_usage_dfs:
        data_usage_df = pd.concat(all_usage_dfs, ignore_index=True)
        data_usage_df = data_usage_df.drop_duplicates(subset='NASID', keep='first')
        print(f"  TOTAL: {len(data_usage_df)} NAS IDs with usage data")
    else:
        print("  WARNING: No data usage found")
else:
    print("  SKIPPED: No device mapping available")


# ======================================================================
# MERGE ALL FEATURES
# ======================================================================
print("\n" + "=" * 70)
print("MERGING ALL INDUSTRY EXPERT FEATURES")
print("=" * 70)

merged = nps[['phone_number', 'nps_score', 'nps_group', 'Sprint Start Date', 'Sprint End Date']].copy()
merged['phone_number'] = merged['phone_number'].astype(str)

feature_match_rates = {}

# Feature 1 & 5: Outage exposure (via device_id)
if len(outage_df) > 0 and len(device_map_df) > 0:
    outage_merge = device_map_df[['phone_number', 'DEVICE_ID']].merge(
        outage_df, on='DEVICE_ID', how='left'
    )[['phone_number', 'OUTAGE_EVENTS', 'DISTINCT_OUTAGES', 'RECOVERED_EVENTS', 'AVG_RECOVERY_MINS']]
    merged = merged.merge(outage_merge, on='phone_number', how='left')
    n = merged['OUTAGE_EVENTS'].notna().sum()
    feature_match_rates['1. Disconnection Frequency'] = n
    feature_match_rates['5. Device-Level Outage Exposure'] = n
    print(f"  + Outage (Feat 1,5): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Outage (Feat 1,5): skipped")
    feature_match_rates['1. Disconnection Frequency'] = 0
    feature_match_rates['5. Device-Level Outage Exposure'] = 0

# Feature 2: Peak-hour quality (via partner)
if len(peak_hour_df) > 0 and len(device_map_df) > 0:
    peak_merge = device_map_df[['phone_number', 'PARTNER_LNG_ID']].merge(
        peak_hour_df, left_on='PARTNER_LNG_ID', right_on='PARTNER_ID', how='left'
    )
    peak_cols = ['phone_number']
    for c in ['PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT', 'OVERALL_UPTIME_PCT',
              'AVG_PEAK_INTERRUPTIONS', 'PEAK_VS_OVERALL_GAP', 'UPTIME_DATA_DAYS']:
        if c in peak_merge.columns:
            peak_cols.append(c)
    merged = merged.merge(peak_merge[peak_cols], on='phone_number', how='left')
    n = merged['PEAK_UPTIME_PCT'].notna().sum()
    feature_match_rates['2. Peak-Hour Quality'] = n
    print(f"  + Peak-Hour (Feat 2): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Peak-Hour (Feat 2): skipped")
    feature_match_rates['2. Peak-Hour Quality'] = 0

# Feature 3: FCR
if len(fcr_df) > 0:
    fcr_cols = ['phone_number']
    for c in ['TOTAL_TICKETS_FCR', 'FCR_TICKETS', 'FCR_RATE', 'AVG_TIMES_REOPENED',
              'MAX_TIMES_REOPENED', 'AVG_CUSTOMER_CALLS_PER_TICKET', 'TOTAL_CUSTOMER_CALLS',
              'AVG_PARTNER_CALLS_PER_TICKET', 'AVG_TICKET_RATING',
              'TICKETS_REOPENED_ONCE', 'TICKETS_REOPENED_3PLUS']:
        if c in fcr_df.columns:
            fcr_cols.append(c)
    merged = merged.merge(fcr_df[fcr_cols], on='phone_number', how='left')
    n = merged['FCR_RATE'].notna().sum()
    feature_match_rates['3. First-Call Resolution Rate'] = n
    print(f"  + FCR (Feat 3): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - FCR (Feat 3): skipped")
    feature_match_rates['3. First-Call Resolution Rate'] = 0

# Feature 4: Repeat complaints
if len(repeat_df) > 0:
    repeat_cols = ['phone_number']
    for c in ['TOTAL_TICKETS_REPEAT', 'DISTINCT_ISSUE_TYPES', 'MAX_TICKETS_SAME_ISSUE',
              'AVG_TICKETS_PER_ISSUE', 'ISSUES_WITH_3PLUS_TICKETS', 'HAS_REPEAT_COMPLAINT']:
        if c in repeat_df.columns:
            repeat_cols.append(c)
    merged = merged.merge(repeat_df[repeat_cols], on='phone_number', how='left')
    n = merged['HAS_REPEAT_COMPLAINT'].notna().sum()
    feature_match_rates['4. Repeat Complaints'] = n
    print(f"  + Repeat (Feat 4): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Repeat (Feat 4): skipped")
    feature_match_rates['4. Repeat Complaints'] = 0

# Feature 6: Payment failures
if len(payment_fail_df) > 0:
    pf_cols = ['phone_number']
    for c in ['PAYMENT_FAILURES', 'PAYMENT_SUCCESSES', 'PAYMENT_ATTEMPTS',
              'TOTAL_PAYMENT_EVENTS', 'FAILURE_RATE_PCT']:
        if c in payment_fail_df.columns:
            pf_cols.append(c)
    merged = merged.merge(payment_fail_df[pf_cols], on='phone_number', how='left')
    n = merged['PAYMENT_FAILURES'].notna().sum()
    feature_match_rates['6. Recharge Attempt Failures'] = n
    print(f"  + Payment Failures (Feat 6): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Payment Failures (Feat 6): skipped")
    feature_match_rates['6. Recharge Attempt Failures'] = 0

# Feature 7: Partner dispatch decline
if len(dispatch_df) > 0:
    disp_cols = ['phone_number']
    for c in ['TOTAL_LEADS', 'LEADS_ALL_DECLINED', 'LEADS_ACCEPTED', 'LEADS_INSTALLED',
              'AVG_INSTALL_TAT_MINS', 'AVG_INSTALL_RATING', 'DISPATCH_DECLINE_RATE_PCT']:
        if c in dispatch_df.columns:
            disp_cols.append(c)
    merged = merged.merge(dispatch_df[disp_cols], on='phone_number', how='left')
    n = merged['DISPATCH_DECLINE_RATE_PCT'].notna().sum()
    feature_match_rates['7. Partner Dispatch Decline Rate'] = n
    print(f"  + Dispatch (Feat 7): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Dispatch (Feat 7): skipped")
    feature_match_rates['7. Partner Dispatch Decline Rate'] = 0

# Feature 8: Time-to-value
if len(ttv_df) > 0:
    ttv_cols = ['phone_number']
    for c in ['HOURS_TO_FIRST_RECHARGE', 'DAYS_TO_FIRST_RECHARGE']:
        if c in ttv_df.columns:
            ttv_cols.append(c)
    merged = merged.merge(ttv_df[ttv_cols], on='phone_number', how='left')
    n = merged['DAYS_TO_FIRST_RECHARGE'].notna().sum()
    feature_match_rates['8. Time-to-Value Post-Install'] = n
    print(f"  + TTV (Feat 8): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - TTV (Feat 8): skipped")
    feature_match_rates['8. Time-to-Value Post-Install'] = 0

# Feature 9: IVR call volume
if len(ivr_df) > 0:
    ivr_cols = ['phone_number']
    for c in ['TOTAL_IVR_CALLS', 'INBOUND_CALLS', 'OUTBOUND_CALLS',
              'ANSWERED_CALLS', 'MISSED_CALLS', 'DROPPED_CALLS',
              'AVG_ANSWERED_SECONDS', 'AVG_CALL_DURATION',
              'INBOUND_ANSWERED', 'INBOUND_UNANSWERED']:
        if c in ivr_df.columns:
            ivr_cols.append(c)
    merged = merged.merge(ivr_df[ivr_cols], on='phone_number', how='left')
    n = merged['TOTAL_IVR_CALLS'].notna().sum()
    feature_match_rates['9. IVR Call Volume'] = n
    print(f"  + IVR (Feat 9): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - IVR (Feat 9): skipped")
    feature_match_rates['9. IVR Call Volume'] = 0

# Feature 10: Data usage (via NAS)
if len(data_usage_df) > 0 and len(device_map_df) > 0:
    # Map NASID to string for joining
    device_map_df['NASID_STR'] = device_map_df['NASID'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else None
    )
    usage_merge = device_map_df[['phone_number', 'NASID_STR']].merge(
        data_usage_df, left_on='NASID_STR', right_on='NASID', how='left'
    )
    usage_cols = ['phone_number']
    for c in ['AVG_DAILY_DATA_GB', 'MEDIAN_DAILY_DATA_GB', 'STDDEV_DAILY_DATA_GB',
              'MIN_DAILY_DATA_GB', 'MAX_DAILY_DATA_GB', 'USAGE_DATA_DAYS',
              'LOW_USAGE_DAYS', 'HIGH_USAGE_DAYS']:
        if c in usage_merge.columns:
            usage_cols.append(c)
    merged = merged.merge(usage_merge[usage_cols], on='phone_number', how='left')
    n = merged['AVG_DAILY_DATA_GB'].notna().sum()
    feature_match_rates['10. Data Usage Patterns'] = n
    print(f"  + Data Usage (Feat 10): {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
else:
    print("  - Data Usage (Feat 10): skipped")
    feature_match_rates['10. Data Usage Patterns'] = 0

# Convert numeric columns
numeric_cols = [c for c in merged.columns if c not in ['phone_number', 'nps_group', 'Sprint Start Date', 'Sprint End Date']]
for col in numeric_cols:
    if col != 'nps_score':
        merged[col] = pd.to_numeric(merged[col], errors='coerce')


# ======================================================================
# SAVE
# ======================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUT")
print("=" * 70)

output_csv = os.path.join(DATA, "industry_expert_features.csv")
merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"  CSV: {output_csv}")
print(f"  Rows: {len(merged)}, Columns: {len(merged.columns)}")


# ======================================================================
# GENERATE REPORT
# ======================================================================
print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

r_lines = []
r = r_lines.append

r("=" * 70)
r("PHASE 3B: INDUSTRY EXPERT FEATURES REPORT")
r(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r("=" * 70)
r("")
r(f"NPS respondents: {len(merged)}")
r(f"Unique phones: {len(phones)}")
r(f"Total columns: {len(merged.columns)}")
r("")

# ── Feasibility Assessment ──
r("=" * 70)
r("FEASIBILITY ASSESSMENT: WHICH FEATURES CAN WE COMPUTE?")
r("=" * 70)

feasibility = [
    {
        'num': 1,
        'name': 'Disconnection FREQUENCY',
        'can_compute': True,
        'tables': 'IMPACTED_DEVICES + t_wg_customer',
        'description': 'Count of outage events per device, distinct outage alerts, avg recovery time',
        'caveats': 'Only ~20 days of outage data available (Dec 20 2025 - Jan 8 2026). Not all NPS respondents fall in this window. Device-level, not customer-perception level.',
        'columns_used': 'OUTAGE_EVENTS, DISTINCT_OUTAGES, AVG_RECOVERY_MINS',
    },
    {
        'num': 2,
        'name': 'Peak-hour quality (7-11 PM)',
        'can_compute': True,
        'tables': 'PARTNER_INFLUX_SUMMARY',
        'description': 'Peak-hour uptime %, stable ping %, peak-vs-overall gap, peak interruptions count',
        'caveats': 'Partner-level aggregation (not per-device or per-customer). Peak hours defined by Wiom system, likely 7-11 PM. Last 90 days rolling.',
        'columns_used': 'PEAK_UPTIME_PCT, PEAK_STABLE_PCT, PEAK_VS_OVERALL_GAP, AVG_PEAK_INTERRUPTIONS',
    },
    {
        'num': 3,
        'name': 'First-Call Resolution Rate',
        'can_compute': True,
        'tables': 'service_ticket_model + t_wg_customer',
        'description': 'FCR rate = resolved tickets with zero reopenings / total resolved. Also: avg reopenings, customer call counts per ticket.',
        'caveats': 'FCR proxy uses TIMES_REOPENED=0 which may miss cases where issue was not truly resolved but ticket was not reopened. Excludes CC-type tickets.',
        'columns_used': 'FCR_RATE, FCR_TICKETS, AVG_TIMES_REOPENED, AVG_CUSTOMER_CALLS_PER_TICKET',
    },
    {
        'num': 4,
        'name': 'Repeat complaints (3+ calls same issue)',
        'can_compute': True,
        'tables': 'service_ticket_model + t_wg_customer',
        'description': 'Multiple tickets with same FIRST_TITLE per customer. Flag for 3+ tickets on same issue.',
        'caveats': 'Uses FIRST_TITLE as issue proxy. Same title != same root cause always. Does not capture repeat calls that did not result in tickets.',
        'columns_used': 'HAS_REPEAT_COMPLAINT, MAX_TICKETS_SAME_ISSUE, ISSUES_WITH_3PLUS_TICKETS',
    },
    {
        'num': 5,
        'name': 'Outage exposure at DEVICE level',
        'can_compute': True,
        'tables': 'IMPACTED_DEVICES + t_wg_customer',
        'description': 'Same as Feature 1 but emphasizing device-level granularity vs partner-level. Count of outage events affecting specific customer device.',
        'caveats': 'Same data window limitation as Feature 1. Shared with disconnection frequency metric.',
        'columns_used': 'OUTAGE_EVENTS, DISTINCT_OUTAGES, RECOVERED_EVENTS',
    },
    {
        'num': 6,
        'name': 'Recharge attempt failures',
        'can_compute': True,
        'tables': 'payment_logs',
        'description': 'Count of order_failed vs order_succeeded events. Failure rate percentage.',
        'caveats': 'payment_logs tracks gateway-level events. Some failures may be user error (insufficient balance, wrong card). Does not distinguish payment method.',
        'columns_used': 'PAYMENT_FAILURES, PAYMENT_SUCCESSES, FAILURE_RATE_PCT',
    },
    {
        'num': 7,
        'name': 'Partner dispatch decline rate',
        'can_compute': True,
        'tables': 'profile_lead_model',
        'description': 'Percentage of leads where ALL partners declined. Also: leads accepted, installed, avg install TAT, install rating.',
        'caveats': 'ALL_PARTNERS_DECLINED is a binary flag at lead level. Does not capture partial declines or slow responses. Some leads may have multiple entries.',
        'columns_used': 'DISPATCH_DECLINE_RATE_PCT, LEADS_ALL_DECLINED, AVG_INSTALL_RATING',
    },
    {
        'num': 8,
        'name': 'Time-to-value post-install',
        'can_compute': True,
        'tables': 'profile_lead_model + t_router_user_mapping',
        'description': 'Hours/days from installation to first recharge (proxy for when customer starts meaningful paid usage).',
        'caveats': 'First recharge may not be first usage (free trial period). FIRST_PING_POST_INSTALLATION available but as TEXT, harder to parse. Negative values possible if recharge precedes install record.',
        'columns_used': 'HOURS_TO_FIRST_RECHARGE, DAYS_TO_FIRST_RECHARGE',
    },
    {
        'num': 9,
        'name': 'IVR call volume',
        'can_compute': True,
        'tables': 'tata_ivr_events',
        'description': 'Total calls, inbound/outbound split, answered/missed/dropped, avg call duration.',
        'caveats': 'CLIENT_NUMBER uses +91 prefix, matched via RIGHT(,10). Not time-scoped to NPS survey window in this version. Includes all historical calls.',
        'columns_used': 'TOTAL_IVR_CALLS, INBOUND_CALLS, OUTBOUND_CALLS, ANSWERED_CALLS, AVG_ANSWERED_SECONDS',
    },
    {
        'num': 10,
        'name': 'Data usage patterns',
        'can_compute': True,
        'tables': 'data_usage_okr + t_wg_customer',
        'description': 'Daily data usage at NAS (router) level. Avg/median/stddev GB per day. Low-usage and high-usage day counts.',
        'caveats': 'NAS-level aggregation (shared among all users on router), NOT per-customer. A busy router will show high usage regardless of individual experience. Best interpreted as a "neighborhood quality" proxy.',
        'columns_used': 'AVG_DAILY_DATA_GB, MEDIAN_DAILY_DATA_GB, LOW_USAGE_DAYS, HIGH_USAGE_DAYS',
    },
]

for f in feasibility:
    status = "CAN COMPUTE" if f['can_compute'] else "CANNOT COMPUTE"
    r(f"\n  Feature {f['num']}: {f['name']}")
    r(f"  Status: {status}")
    r(f"  Tables: {f['tables']}")
    r(f"  Description: {f['description']}")
    r(f"  Caveats: {f['caveats']}")
    r(f"  Columns: {f['columns_used']}")
    # Look up match rate by trying different key formats
    match_n = 0
    for k, v in feature_match_rates.items():
        if str(f['num']) in k.split('.')[0]:
            match_n = v
            break
    r(f"  Match rate: {match_n}/{len(merged)} ({match_n/len(merged)*100:.1f}%)")

# ── Match Rate Summary ──
r("")
r("=" * 70)
r("MATCH RATE SUMMARY")
r("=" * 70)
r(f"\n  {'Feature':45s} | {'Matched':>7s} | {'Total':>7s} | {'Rate':>6s}")
r(f"  {'-'*45} | {'-'*7} | {'-'*7} | {'-'*6}")
for name, n in sorted(feature_match_rates.items()):
    pct = n / len(merged) * 100
    r(f"  {name:45s} | {n:7d} | {len(merged):7d} | {pct:5.1f}%")

# ── Feature Statistics ──
r("")
r("=" * 70)
r("FEATURE STATISTICS")
r("=" * 70)

stat_features = {
    '1. Disconnection Frequency': ['OUTAGE_EVENTS', 'DISTINCT_OUTAGES', 'AVG_RECOVERY_MINS'],
    '2. Peak-Hour Quality': ['PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT', 'OVERALL_UPTIME_PCT', 'PEAK_VS_OVERALL_GAP', 'AVG_PEAK_INTERRUPTIONS'],
    '3. First-Call Resolution': ['FCR_RATE', 'AVG_TIMES_REOPENED', 'AVG_CUSTOMER_CALLS_PER_TICKET', 'TOTAL_CUSTOMER_CALLS', 'AVG_TICKET_RATING'],
    '4. Repeat Complaints': ['MAX_TICKETS_SAME_ISSUE', 'AVG_TICKETS_PER_ISSUE', 'ISSUES_WITH_3PLUS_TICKETS', 'HAS_REPEAT_COMPLAINT'],
    '6. Payment Failures': ['PAYMENT_FAILURES', 'PAYMENT_SUCCESSES', 'FAILURE_RATE_PCT'],
    '7. Partner Dispatch': ['DISPATCH_DECLINE_RATE_PCT', 'LEADS_ALL_DECLINED', 'AVG_INSTALL_RATING', 'AVG_INSTALL_TAT_MINS'],
    '8. Time-to-Value': ['HOURS_TO_FIRST_RECHARGE', 'DAYS_TO_FIRST_RECHARGE'],
    '9. IVR Call Volume': ['TOTAL_IVR_CALLS', 'INBOUND_CALLS', 'OUTBOUND_CALLS', 'ANSWERED_CALLS', 'MISSED_CALLS', 'AVG_ANSWERED_SECONDS'],
    '10. Data Usage': ['AVG_DAILY_DATA_GB', 'MEDIAN_DAILY_DATA_GB', 'STDDEV_DAILY_DATA_GB', 'LOW_USAGE_DAYS', 'HIGH_USAGE_DAYS'],
}

for group_name, cols in stat_features.items():
    r(f"\n  --- {group_name} ---")
    for col in cols:
        if col in merged.columns:
            s = merged[col].dropna()
            if len(s) > 0:
                r(f"    {col} (n={len(s)}):")
                r(f"      Mean={s.mean():.2f}  Median={s.median():.2f}  Std={s.std():.2f}  Min={s.min():.2f}  Max={s.max():.2f}")
            else:
                r(f"    {col}: no data")
        else:
            r(f"    {col}: NOT IN DATASET")

# ── Features by NPS group ──
r("")
r("=" * 70)
r("KEY FEATURES BY NPS GROUP")
r("=" * 70)

key_features_for_groups = [
    'OUTAGE_EVENTS', 'PEAK_UPTIME_PCT', 'PEAK_VS_OVERALL_GAP',
    'FCR_RATE', 'AVG_TIMES_REOPENED', 'AVG_CUSTOMER_CALLS_PER_TICKET',
    'HAS_REPEAT_COMPLAINT', 'MAX_TICKETS_SAME_ISSUE',
    'PAYMENT_FAILURES', 'FAILURE_RATE_PCT',
    'DISPATCH_DECLINE_RATE_PCT', 'AVG_INSTALL_RATING',
    'DAYS_TO_FIRST_RECHARGE',
    'TOTAL_IVR_CALLS', 'INBOUND_CALLS',
    'AVG_DAILY_DATA_GB', 'LOW_USAGE_DAYS',
]

available_key_features = [c for c in key_features_for_groups if c in merged.columns and merged[c].notna().any()]
nps_groups = ['Promoter', 'Passive', 'Detractor']

if available_key_features:
    for col in available_key_features:
        r(f"\n  {col}:")
        r(f"    {'Group':12s} | {'Mean':>10s} | {'Median':>10s} | {'Count':>6s}")
        r(f"    {'-'*12} | {'-'*10} | {'-'*10} | {'-'*6}")
        for grp in nps_groups:
            grp_data = merged[merged['nps_group'] == grp][col].dropna()
            if len(grp_data) > 0:
                r(f"    {grp:12s} | {grp_data.mean():10.2f} | {grp_data.median():10.2f} | {len(grp_data):6d}")
            else:
                r(f"    {grp:12s} | {'N/A':>10s} | {'N/A':>10s} | {0:6d}")

# ── Key Insights ──
r("")
r("=" * 70)
r("PRELIMINARY INSIGHTS")
r("=" * 70)

# Check if detractors have worse metrics
for col in ['PEAK_UPTIME_PCT', 'FCR_RATE', 'FAILURE_RATE_PCT', 'AVG_TIMES_REOPENED',
            'TOTAL_IVR_CALLS', 'OUTAGE_EVENTS', 'HAS_REPEAT_COMPLAINT']:
    if col in merged.columns:
        det = merged[merged['nps_group'] == 'Detractor'][col].dropna()
        pro = merged[merged['nps_group'] == 'Promoter'][col].dropna()
        if len(det) > 10 and len(pro) > 10:
            det_mean = det.mean()
            pro_mean = pro.mean()
            diff_pct = ((det_mean - pro_mean) / abs(pro_mean) * 100) if pro_mean != 0 else 0
            direction = "HIGHER" if det_mean > pro_mean else "LOWER"
            r(f"\n  {col}:")
            r(f"    Detractor mean: {det_mean:.3f}")
            r(f"    Promoter mean:  {pro_mean:.3f}")
            r(f"    Detractors are {abs(diff_pct):.1f}% {direction} than Promoters")


# Save report
report_path = os.path.join(OUTPUT, "phase3b_industry_expert_features.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(r_lines))
print(f"\n  Report saved: {report_path}")

# Print report to console
print("\n" + "\n".join(r_lines))

print("\n" + "=" * 70)
print(f"PHASE 3B COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Features CSV: {output_csv}")
print(f"  Report: {report_path}")
print("=" * 70)
