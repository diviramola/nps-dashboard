"""
Phase 3: Snowflake Feature Engineering (Full Run)
==================================================
Queries Snowflake via Metabase API to compute operational features
for ALL 13,045 NPS respondents. Processes in batches of 500 phones.

Features computed:
1. Recharge History (triple-dedup on t_router_user_mapping)
2. Service Tickets (t_wg_customer -> service_ticket_model)
3. Partner Mapping + Geography (t_wg_customer -> HIERARCHY_BASE)
4. Partner Uptime (PARTNER_INFLUX_SUMMARY - single global query)
5. Install TAT (taskvanilla_audit + booking_logs)
6. Payment Mode (wiomBillingWifi with >=99 threshold)

Output:
- data/nps_analytical_base.csv
- output/phase3_feature_engineering.txt
"""

import sys, io, os, json, time, traceback
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Credentials ──
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
BATCH_DELAY = 2       # seconds between batches
RETRY_DELAY = 10      # seconds before retry on failure

# ── Query runner ──
def run_query(sql, description="query", timeout=120):
    """Execute a Snowflake query via Metabase API. Returns DataFrame."""
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        # Metabase returns 202 with data in body for Snowflake queries
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
            print(f"    [ERROR] {description}: HTTP {resp.status_code} — {resp.text[:200]}")
            return pd.DataFrame()
    except Exception as e:
        print(f"    [ERROR] {description}: {str(e)[:200]}")
        return pd.DataFrame()


def run_batched_query(phones, sql_template, feature_name):
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

        df = run_query(sql, desc)
        if len(df) == 0:
            # Retry once after RETRY_DELAY
            print(f" RETRY...", end="", flush=True)
            time.sleep(RETRY_DELAY)
            df = run_query(sql, f"{desc} (retry)")
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
        # Deduplicate by mobile in case of overlapping results
        if 'mobile' in result.columns:
            result = result.drop_duplicates(subset='mobile', keep='first')
        return result
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 3: SNOWFLAKE FEATURE ENGINEERING (FULL RUN)")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── Load NPS base ──
print("\n[LOAD] Reading NPS base data...")
nps = pd.read_csv(os.path.join(DATA, "nps_clean_base.csv"), low_memory=False)
nps['phone_number'] = nps['phone_number'].astype(str).str.strip()
phones = nps['phone_number'].unique().tolist()
print(f"  Total respondents: {len(nps)}")
print(f"  Unique phones: {len(phones)}")
print(f"  Batches needed: {(len(phones) + BATCH_SIZE - 1) // BATCH_SIZE} (batch size={BATCH_SIZE})")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 1: RECHARGE HISTORY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[1/6] RECHARGE HISTORY (triple-dedup on t_router_user_mapping)")
print("─" * 70)

RECHARGE_SQL = """
WITH deduped AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id) AS rn1,
        ROW_NUMBER() OVER (PARTITION BY mobile, router_nas_id, charges,
            TO_DATE(DATEADD(minute, 330, created_on)) ORDER BY id) AS rn2
    FROM prod_db.public.t_router_user_mapping
    WHERE mobile IN ({phone_list})
      AND device_limit = '10'
      AND otp = 'DONE'
      AND mobile > '5999999999'
      AND store_group_id = 0
)
SELECT
    mobile,
    COUNT(*) AS total_recharges,
    MIN(TO_DATE(DATEADD(minute, 330, created_on))) AS first_recharge,
    MAX(TO_DATE(DATEADD(minute, 330, created_on))) AS last_recharge,
    AVG(COALESCE(TRY_TO_NUMBER(PARSE_JSON(extra_data):totalPaid::STRING), charges)) AS avg_recharge_amount
FROM deduped
WHERE rn1 = 1 AND rn2 = 1
GROUP BY mobile
"""

recharge_df = run_batched_query(phones, RECHARGE_SQL, "Recharge")
if len(recharge_df) > 0:
    recharge_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    recharge_df['phone_number'] = recharge_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(recharge_df)} phones with recharge data ({len(recharge_df)/len(phones)*100:.1f}% match)")
else:
    print("  WARNING: No recharge data returned")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 2: SERVICE TICKETS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[2/6] SERVICE TICKETS (t_wg_customer -> service_ticket_model)")
print("─" * 70)

TICKET_SQL = """
WITH customer_devices AS (
    SELECT DISTINCT mobile, device_id
    FROM prod_db.public.t_wg_customer
    WHERE mobile IN ({phone_list})
      AND mobile > '5999999999'
      AND _FIVETRAN_DELETED = false
),
tickets AS (
    SELECT
        cd.mobile,
        stm.ticket_id,
        stm.cx_px,
        stm.resolution_tat_bucket,
        DATEDIFF(hour, stm.ticket_added_time, stm.final_resolved_time) AS resolution_hours
    FROM customer_devices cd
    JOIN prod_db.public.service_ticket_model stm ON cd.device_id = stm.device_id
)
SELECT
    mobile,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN cx_px = 'Cx' THEN 1 END) AS cx_tickets,
    COUNT(CASE WHEN cx_px = 'Px' THEN 1 END) AS px_tickets,
    AVG(resolution_hours) AS avg_resolution_hours,
    COUNT(CASE WHEN resolution_tat_bucket = 'within TAT' THEN 1 END) * 100.0 /
        NULLIF(COUNT(*), 0) AS sla_compliance_pct
FROM tickets
GROUP BY mobile
"""

ticket_df = run_batched_query(phones, TICKET_SQL, "Tickets")
if len(ticket_df) > 0:
    ticket_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    ticket_df['phone_number'] = ticket_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(ticket_df)} phones with ticket data ({len(ticket_df)/len(phones)*100:.1f}% match)")
else:
    print("  WARNING: No ticket data returned")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 3: PARTNER MAPPING + GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[3/6] PARTNER MAPPING + GEOGRAPHY (t_wg_customer -> HIERARCHY_BASE)")
print("─" * 70)

PARTNER_SQL = """
WITH customer_partner AS (
    SELECT
        twc.mobile,
        twc.nasid,
        twc.shard,
        twc.lco_account_id,
        prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS partner_lng_id,
        ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({phone_list})
      AND twc.mobile > '5999999999'
      AND twc._FIVETRAN_DELETED = false
)
SELECT
    cp.mobile,
    cp.partner_lng_id,
    hb.cluster,
    hb.MIS_CITY,
    hb.ZONE,
    hb.PARTNER_NAME,
    hb.PARTNER_STATUS
FROM customer_partner cp
LEFT JOIN prod_db.public.HIERARCHY_BASE hb
    ON cp.partner_lng_id = hb.PARTNER_ACCOUNT_ID
    AND hb.DEDUP_FLAG = 1
WHERE cp.rn = 1
"""

partner_df = run_batched_query(phones, PARTNER_SQL, "Partner/Geo")
if len(partner_df) > 0:
    partner_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    partner_df['phone_number'] = partner_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(partner_df)} phones with partner data ({len(partner_df)/len(phones)*100:.1f}% match)")
    print(f"  Unique partners: {partner_df['partner_lng_id'].nunique()}")
    if 'mis_city' in partner_df.columns:
        print(f"  Unique cities: {partner_df['mis_city'].nunique()}")
    if 'zone' in partner_df.columns:
        print(f"  Unique zones: {partner_df['zone'].nunique()}")
else:
    print("  WARNING: No partner data returned")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 4: PARTNER UPTIME (SINGLE GLOBAL QUERY)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[4/6] PARTNER UPTIME (PARTNER_INFLUX_SUMMARY — single query, last 90 days)")
print("─" * 70)

UPTIME_SQL = """
SELECT
    partner_id,
    AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) AS avg_uptime_pct,
    STDDEV(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) AS stddev_uptime,
    MIN(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) AS min_uptime,
    COUNT(*) AS uptime_data_days
FROM prod_db.public.PARTNER_INFLUX_SUMMARY
WHERE DATEADD(day, -1, appended_date) >= DATEADD(day, -90, CURRENT_DATE())
GROUP BY partner_id
HAVING COUNT(*) >= 10
"""

print("  Running global partner uptime query...", end="", flush=True)
uptime_df = run_query(UPTIME_SQL, "Partner uptime (global)", timeout=180)
if len(uptime_df) > 0:
    print(f" {len(uptime_df)} partners with uptime data")
    # Preview stats
    if 'avg_uptime_pct' in uptime_df.columns:
        uptime_df['avg_uptime_pct'] = pd.to_numeric(uptime_df['avg_uptime_pct'], errors='coerce')
        print(f"  Mean uptime across partners: {uptime_df['avg_uptime_pct'].mean():.4f}")
        print(f"  Median uptime: {uptime_df['avg_uptime_pct'].median():.4f}")
else:
    print(" WARNING: No uptime data returned")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 5: INSTALL TAT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[5/6] INSTALL TAT (taskvanilla_audit + booking_logs)")
print("─" * 70)

INSTALL_SQL = """
WITH install_events AS (
    SELECT
        mobile,
        MIN(DATEADD(minute, 330, added_time)) AS install_time,
        COUNT(*) AS install_attempts
    FROM prod_db.public.taskvanilla_audit
    WHERE mobile IN ({phone_list})
      AND event_name = 'OTP_VERIFIED'
      AND mobile > '5999999999'
    GROUP BY mobile
),
booking_events AS (
    SELECT
        mobile,
        MIN(DATEADD(minute, 330, added_time)) AS booking_time
    FROM prod_db.public.booking_logs
    WHERE mobile IN ({phone_list})
      AND event_name = 'booking_fee_captured'
    GROUP BY mobile
)
SELECT
    ie.mobile,
    ie.install_time,
    ie.install_attempts,
    be.booking_time,
    DATEDIFF(hour, be.booking_time, ie.install_time) AS install_tat_hours
FROM install_events ie
LEFT JOIN booking_events be ON ie.mobile = be.mobile
"""

install_df = run_batched_query(phones, INSTALL_SQL, "Install TAT")
if len(install_df) > 0:
    install_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    install_df['phone_number'] = install_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(install_df)} phones with install data ({len(install_df)/len(phones)*100:.1f}% match)")
    if 'install_tat_hours' in install_df.columns:
        install_df['install_tat_hours'] = pd.to_numeric(install_df['install_tat_hours'], errors='coerce')
        valid_tat = install_df[install_df['install_tat_hours'].notna() & (install_df['install_tat_hours'] > 0)]
        if len(valid_tat) > 0:
            print(f"  Valid TAT values: {len(valid_tat)}")
            print(f"  Median install TAT: {valid_tat['install_tat_hours'].median():.0f} hours")
            print(f"  Mean install TAT: {valid_tat['install_tat_hours'].mean():.0f} hours")
else:
    print("  WARNING: No install TAT data returned")

# ══════════════════════════════════════════════════════════════════════
# FEATURE 6: PAYMENT MODE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[6/6] PAYMENT MODE (wiomBillingWifi, threshold >= 99)")
print("─" * 70)

PAYMENT_SQL = """
SELECT
    mobile,
    COUNT(*) AS total_payments,
    COUNT(CASE WHEN LOWER(transaction_id) LIKE '%wgsubs%'
                 OR LOWER(transaction_id) LIKE '%custwgsubs%'
                 OR LOWER(transaction_id) LIKE '%wgmand%'
                 OR LOWER(transaction_id) LIKE '%cussubs%'
               THEN 1 END) AS autopay_payments,
    COUNT(CASE WHEN transaction_id LIKE 'BILL_PAID%'
               THEN 1 END) AS cash_payments,
    AVG(total_price) AS avg_payment_amount
FROM prod_db.public.wiomBillingWifi
WHERE mobile IN ({phone_list})
  AND total_price >= 99
  AND paymentStatus = 1
  AND mobile > '5999999999'
  AND transaction_id NOT LIKE 'mr%'
  AND payment_type <> 2
  AND (refund_status <> 1 OR refund_status IS NULL)
GROUP BY mobile
"""

payment_df = run_batched_query(phones, PAYMENT_SQL, "Payments")
if len(payment_df) > 0:
    payment_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    payment_df['phone_number'] = payment_df['phone_number'].astype(str)
    print(f"  TOTAL: {len(payment_df)} phones with payment data ({len(payment_df)/len(phones)*100:.1f}% match)")
else:
    print("  WARNING: No payment data returned")


# ══════════════════════════════════════════════════════════════════════
# MERGE ALL FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MERGING ALL FEATURES INTO ANALYTICAL BASE")
print("=" * 70)

merged = nps.copy()
merged['phone_number'] = merged['phone_number'].astype(str)

# 1. Recharge features
if len(recharge_df) > 0:
    merged = merged.merge(
        recharge_df[['phone_number', 'total_recharges', 'first_recharge', 'last_recharge', 'avg_recharge_amount']],
        on='phone_number', how='left'
    )
    n = merged['total_recharges'].notna().sum()
    print(f"  + Recharge: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    print("  - Recharge: skipped (no data)")

# 2. Ticket features
if len(ticket_df) > 0:
    merged = merged.merge(
        ticket_df[['phone_number', 'total_tickets', 'cx_tickets', 'px_tickets',
                    'avg_resolution_hours', 'sla_compliance_pct']],
        on='phone_number', how='left'
    )
    n = merged['total_tickets'].notna().sum()
    print(f"  + Tickets: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    print("  - Tickets: skipped (no data)")

# 3. Partner + Geography
if len(partner_df) > 0:
    partner_cols = ['phone_number', 'partner_lng_id']
    for c in ['cluster', 'mis_city', 'zone', 'partner_name', 'partner_status']:
        if c in partner_df.columns:
            partner_cols.append(c)
    merged = merged.merge(partner_df[partner_cols], on='phone_number', how='left')
    n = merged['partner_lng_id'].notna().sum()
    print(f"  + Partner/Geo: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    print("  - Partner/Geo: skipped (no data)")

# 4. Partner Uptime — join via partner_lng_id
if len(uptime_df) > 0 and 'partner_lng_id' in merged.columns:
    uptime_cols = ['partner_id']
    for c in ['avg_uptime_pct', 'stddev_uptime', 'min_uptime', 'uptime_data_days']:
        if c in uptime_df.columns:
            uptime_cols.append(c)
    merged = merged.merge(
        uptime_df[uptime_cols],
        left_on='partner_lng_id', right_on='partner_id', how='left'
    )
    if 'partner_id' in merged.columns:
        merged.drop(columns=['partner_id'], inplace=True)
    n = merged['avg_uptime_pct'].notna().sum()
    print(f"  + Uptime: {n}/{len(merged)} matched via partner ({n/len(merged)*100:.1f}%)")
else:
    print("  - Uptime: skipped (no data or no partner_lng_id)")

# 5. Install TAT
if len(install_df) > 0:
    install_merge_cols = ['phone_number']
    for c in ['install_time', 'install_attempts', 'booking_time', 'install_tat_hours']:
        if c in install_df.columns:
            install_merge_cols.append(c)
    merged = merged.merge(install_df[install_merge_cols], on='phone_number', how='left')
    n = merged['install_tat_hours'].notna().sum() if 'install_tat_hours' in merged.columns else 0
    print(f"  + Install TAT: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    print("  - Install TAT: skipped (no data)")

# 6. Payment mode
if len(payment_df) > 0:
    merged = merged.merge(
        payment_df[['phone_number', 'total_payments', 'autopay_payments', 'cash_payments', 'avg_payment_amount']],
        on='phone_number', how='left'
    )
    n = merged['total_payments'].notna().sum()
    print(f"  + Payments: {n}/{len(merged)} matched ({n/len(merged)*100:.1f}%)")
else:
    print("  - Payments: skipped (no data)")


# ══════════════════════════════════════════════════════════════════════
# DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("COMPUTING DERIVED FEATURES")
print("─" * 70)

# Payment mode classification
if 'autopay_payments' in merged.columns and 'cash_payments' in merged.columns:
    merged['total_payments'] = pd.to_numeric(merged['total_payments'], errors='coerce')
    merged['autopay_payments'] = pd.to_numeric(merged['autopay_payments'], errors='coerce')
    merged['cash_payments'] = pd.to_numeric(merged['cash_payments'], errors='coerce')

    def classify_payment(row):
        tp = row.get('total_payments')
        if pd.isna(tp) or tp == 0:
            return 'unknown'
        autopay_pct = row.get('autopay_payments', 0) / tp
        cash_pct = row.get('cash_payments', 0) / tp
        if autopay_pct > 0.5:
            return 'autopay'
        elif cash_pct > 0.5:
            return 'cash'
        else:
            return 'online'

    merged['payment_mode'] = merged.apply(classify_payment, axis=1)
    pm_dist = merged['payment_mode'].value_counts()
    print(f"  payment_mode: {dict(pm_dist)}")

# has_tickets flag
if 'total_tickets' in merged.columns:
    merged['total_tickets'] = pd.to_numeric(merged['total_tickets'], errors='coerce').fillna(0)
    merged['has_tickets'] = (merged['total_tickets'] > 0).astype(int)
    print(f"  has_tickets: {merged['has_tickets'].sum()} with tickets, {(merged['has_tickets']==0).sum()} without")

# days_since_last_recharge
if 'last_recharge' in merged.columns and 'Sprint End Date' in merged.columns:
    merged['last_recharge'] = pd.to_datetime(merged['last_recharge'], errors='coerce', utc=True)
    merged['last_recharge'] = merged['last_recharge'].dt.tz_localize(None)  # strip tz
    merged['sprint_end_dt'] = pd.to_datetime(merged['Sprint End Date'], errors='coerce')
    merged['days_since_last_recharge'] = (merged['sprint_end_dt'] - merged['last_recharge']).dt.days
    n = merged['days_since_last_recharge'].notna().sum()
    print(f"  days_since_last_recharge: {n} computed")
    # Drop helper column
    merged.drop(columns=['sprint_end_dt'], inplace=True)

# recharge_regularity = (total_recharges * avg_plan_days) / tenure_days
# We don't have avg_plan_days from simplified recharge query, so approximate:
# regularity = total_recharges / (tenure_days / 28) — how many recharges vs expected monthly recharges
if 'total_recharges' in merged.columns and 'tenure_days' in merged.columns:
    merged['total_recharges'] = pd.to_numeric(merged['total_recharges'], errors='coerce')
    merged['tenure_days'] = pd.to_numeric(merged['tenure_days'], errors='coerce')
    expected_recharges = merged['tenure_days'] / 28.0
    merged['recharge_regularity'] = merged['total_recharges'] / expected_recharges.replace(0, np.nan)
    n = merged['recharge_regularity'].notna().sum()
    print(f"  recharge_regularity: {n} computed (1.0 = perfectly regular monthly)")

# Convert numeric columns
for col in ['avg_recharge_amount', 'avg_resolution_hours', 'sla_compliance_pct',
            'avg_payment_amount', 'install_tat_hours', 'install_attempts',
            'avg_uptime_pct', 'stddev_uptime', 'min_uptime', 'uptime_data_days',
            'cx_tickets', 'px_tickets']:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING OUTPUT")
print("=" * 70)

output_csv = os.path.join(DATA, "nps_analytical_base.csv")
merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"  CSV: {output_csv}")
print(f"  Rows: {len(merged)}, Columns: {len(merged.columns)}")

# List new columns added
original_cols = set(nps.columns)
new_cols = [c for c in merged.columns if c not in original_cols]
print(f"  New features added: {len(new_cols)}")
for c in new_cols:
    fill = merged[c].notna().sum()
    print(f"    {c}: {fill}/{len(merged)} ({fill/len(merged)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

r_lines = []
r = r_lines.append

r("=" * 70)
r("PHASE 3: FEATURE ENGINEERING REPORT")
r(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r("=" * 70)
r("")
r(f"NPS respondents: {len(merged)}")
r(f"Unique phones: {len(phones)}")
r(f"Total columns: {len(merged.columns)}")
r(f"New features added: {len(new_cols)}")
r("")

# Match rates
r("=" * 70)
r("MATCH RATES PER FEATURE")
r("=" * 70)
feature_groups = {
    'Recharge History': ['total_recharges', 'first_recharge', 'last_recharge', 'avg_recharge_amount'],
    'Service Tickets': ['total_tickets', 'cx_tickets', 'px_tickets', 'avg_resolution_hours', 'sla_compliance_pct'],
    'Partner/Geography': ['partner_lng_id', 'cluster', 'mis_city', 'zone', 'partner_name', 'partner_status'],
    'Partner Uptime': ['avg_uptime_pct', 'stddev_uptime', 'min_uptime', 'uptime_data_days'],
    'Install TAT': ['install_time', 'install_attempts', 'booking_time', 'install_tat_hours'],
    'Payment Mode': ['total_payments', 'autopay_payments', 'cash_payments', 'avg_payment_amount'],
    'Derived Features': ['payment_mode', 'has_tickets', 'days_since_last_recharge', 'recharge_regularity'],
}

for group_name, cols in feature_groups.items():
    r(f"\n  {group_name}:")
    for col in cols:
        if col in merged.columns:
            fill = merged[col].notna().sum()
            pct = fill / len(merged) * 100
            r(f"    {col:35s}: {fill:6d} / {len(merged)} ({pct:5.1f}%)")
        else:
            r(f"    {col:35s}: NOT AVAILABLE")

# Feature statistics
r("")
r("=" * 70)
r("FEATURE STATISTICS (mean / median / std)")
r("=" * 70)

stat_cols = ['total_recharges', 'avg_recharge_amount', 'total_tickets', 'cx_tickets', 'px_tickets',
             'avg_resolution_hours', 'sla_compliance_pct', 'total_payments', 'autopay_payments',
             'cash_payments', 'avg_payment_amount', 'install_tat_hours', 'install_attempts',
             'avg_uptime_pct', 'stddev_uptime', 'min_uptime',
             'days_since_last_recharge', 'recharge_regularity']

for col in stat_cols:
    if col in merged.columns:
        s = merged[col].dropna()
        if len(s) > 0:
            r(f"\n  {col} (n={len(s)}):")
            r(f"    Mean:   {s.mean():.2f}")
            r(f"    Median: {s.median():.2f}")
            r(f"    Std:    {s.std():.2f}")
            r(f"    Min:    {s.min():.2f}")
            r(f"    Max:    {s.max():.2f}")

# Features by NPS group
r("")
r("=" * 70)
r("FEATURES BY NPS GROUP")
r("=" * 70)

nps_groups = ['Promoter', 'Passive', 'Detractor']
numeric_features_for_groups = [c for c in stat_cols if c in merged.columns and merged[c].notna().any()]

if numeric_features_for_groups:
    group_stats = merged.groupby('nps_group')[numeric_features_for_groups].agg(['mean', 'median', 'count'])

    for col in numeric_features_for_groups:
        r(f"\n  {col}:")
        r(f"    {'Group':12s} | {'Mean':>10s} | {'Median':>10s} | {'Count':>6s}")
        r(f"    {'-'*12} | {'-'*10} | {'-'*10} | {'-'*6}")
        for grp in nps_groups:
            if grp in group_stats.index:
                mean_val = group_stats.at[grp, (col, 'mean')]
                med_val = group_stats.at[grp, (col, 'median')]
                cnt_val = group_stats.at[grp, (col, 'count')]
                if pd.notna(mean_val):
                    r(f"    {grp:12s} | {mean_val:10.2f} | {med_val:10.2f} | {int(cnt_val):6d}")
                else:
                    r(f"    {grp:12s} | {'N/A':>10s} | {'N/A':>10s} | {0:6d}")

# Payment mode distribution
if 'payment_mode' in merged.columns:
    r("")
    r("=" * 70)
    r("PAYMENT MODE DISTRIBUTION")
    r("=" * 70)

    pm_dist = merged['payment_mode'].value_counts()
    r(f"\n  Overall:")
    for mode, cnt in pm_dist.items():
        r(f"    {mode:15s}: {cnt:6d} ({cnt/len(merged)*100:.1f}%)")

    r(f"\n  By NPS Group (row %):")
    pm_cross = pd.crosstab(merged['payment_mode'], merged['nps_group'], normalize='index') * 100
    header_cols = [g for g in nps_groups if g in pm_cross.columns]
    r(f"    {'Mode':15s} | " + " | ".join([f"{g:>10s}" for g in header_cols]))
    r(f"    {'-'*15} | " + " | ".join([f"{'-'*10}" for _ in header_cols]))
    for mode in pm_cross.index:
        vals = " | ".join([f"{pm_cross.at[mode, g]:9.1f}%" if g in pm_cross.columns else f"{'N/A':>10s}" for g in header_cols])
        r(f"    {mode:15s} | {vals}")

    r(f"\n  By NPS Group (column %):")
    pm_cross2 = pd.crosstab(merged['payment_mode'], merged['nps_group'], normalize='columns') * 100
    r(f"    {'Mode':15s} | " + " | ".join([f"{g:>10s}" for g in header_cols]))
    r(f"    {'-'*15} | " + " | ".join([f"{'-'*10}" for _ in header_cols]))
    for mode in pm_cross2.index:
        vals = " | ".join([f"{pm_cross2.at[mode, g]:9.1f}%" if g in pm_cross2.columns else f"{'N/A':>10s}" for g in header_cols])
        r(f"    {mode:15s} | {vals}")

# Geography distribution
if 'zone' in merged.columns:
    r("")
    r("=" * 70)
    r("GEOGRAPHY DISTRIBUTION")
    r("=" * 70)
    zone_dist = merged['zone'].value_counts().head(15)
    r(f"\n  Top zones:")
    for zone, cnt in zone_dist.items():
        r(f"    {str(zone):25s}: {cnt:6d} ({cnt/len(merged)*100:.1f}%)")

if 'mis_city' in merged.columns:
    city_dist = merged['mis_city'].value_counts().head(15)
    r(f"\n  Top MIS cities:")
    for city, cnt in city_dist.items():
        r(f"    {str(city):25s}: {cnt:6d} ({cnt/len(merged)*100:.1f}%)")

# Partner status distribution
if 'partner_status' in merged.columns:
    r(f"\n  Partner status:")
    ps_dist = merged['partner_status'].value_counts()
    for status, cnt in ps_dist.items():
        r(f"    {str(status):25s}: {cnt:6d} ({cnt/len(merged)*100:.1f}%)")

# Save report
report_path = os.path.join(OUTPUT, "phase3_feature_engineering.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(r_lines))
print(f"  Report saved: {report_path}")

# Also print report to console
print("\n" + "\n".join(r_lines))

print("\n" + "=" * 70)
print(f"PHASE 3 COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Analytical base: {output_csv}")
print(f"  Report: {report_path}")
print("=" * 70)
