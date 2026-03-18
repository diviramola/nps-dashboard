"""
Phase 3: Feature Engineering from Snowflake
============================================
Queries Snowflake via Metabase API to compute operational features
for each NPS respondent. Joins NPS Excel data to Snowflake by phone number.

Features computed:
A. Customer Lifecycle (recharge count, plan type, tenure)
B. Network Quality (uptime proxy via partner uptime)
C. Service & Support (ticket count, resolution TAT, SLA)
D. Payment & Value (avg recharge amount, payment mode)
E. Installation Experience (install TAT)
F. Geography & Partner (city, partner quality proxies)
"""

import sys, io, os, json, time
import pandas as pd
import requests
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load credentials
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

def run_query(sql, description="", max_retries=3):
    """Execute a Snowflake query via Metabase API."""
    print(f"  Running: {description}...")
    payload = {
        "database": DB_ID,
        "type": "native",
        "native": {"query": sql}
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and 'rows' in data['data']:
                    cols = [c['name'] for c in data['data']['cols']]
                    rows = data['data']['rows']
                    df = pd.DataFrame(rows, columns=cols)
                    print(f"    -> {len(df)} rows returned")
                    return df
                elif 'error' in data:
                    print(f"    -> Query error: {str(data['error'])[:200]}")
                    return pd.DataFrame()
                else:
                    print(f"    -> Unexpected response format")
                    return pd.DataFrame()
            elif resp.status_code == 202:
                # Query still running, wait and retry
                print(f"    -> Query still running, waiting 10s...")
                time.sleep(10)
                continue
            else:
                print(f"    -> HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return pd.DataFrame()
        except Exception as e:
            print(f"    -> Error: {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return pd.DataFrame()
    return pd.DataFrame()

print("=" * 70)
print("PHASE 3: FEATURE ENGINEERING FROM SNOWFLAKE")
print("=" * 70)

# ── Load NPS base ──
print("\n[1/8] Loading NPS base data...")
nps = pd.read_csv(os.path.join(DATA, "nps_clean_base.csv"), low_memory=False)
print(f"  NPS respondents: {len(nps)}")

# Get unique phone numbers for queries
phones = nps['phone_number'].astype(str).unique().tolist()
print(f"  Unique phone numbers: {len(phones)}")

# Sprint date ranges (from Phase 0 data)
sprint_dates = nps.groupby('Sprint ID').agg({
    'Sprint Start Date': 'first',
    'Sprint End Date': 'first'
}).to_dict()

# ── 2. Join validation: How many NPS phones exist in Snowflake? ──
print("\n[2/8] Join validation — checking phone match rate in Snowflake...")

# Sample first to validate approach (use first 500 phones)
sample_phones = phones[:500]
phone_list = ",".join([f"'{p}'" for p in sample_phones])

join_check_sql = f"""
SELECT
    mobile,
    COUNT(DISTINCT transaction_id) as recharge_count,
    MIN(TO_DATE(DATEADD(minute, 330, created_on))) as first_recharge_date,
    MAX(TO_DATE(DATEADD(minute, 330, created_on))) as last_recharge_date
FROM prod_db.public.t_router_user_mapping
WHERE mobile IN ({phone_list})
  AND device_limit = '10'
  AND otp = 'DONE'
  AND mobile > '5999999999'
  AND store_group_id = 0
GROUP BY mobile
"""

join_df = run_query(join_check_sql, "Join validation (sample 500)")
if len(join_df) > 0:
    match_rate = len(join_df) / len(sample_phones) * 100
    print(f"  Match rate (sample): {len(join_df)}/{len(sample_phones)} = {match_rate:.1f}%")
else:
    print("  WARNING: No matches found. Check phone number format.")

# ── 3. Recharge history features (batched) ──
print("\n[3/8] Computing recharge history features...")

# Process in batches of 1000 phones
BATCH_SIZE = 1000
all_recharge_features = []

for i in range(0, len(phones), BATCH_SIZE):
    batch = phones[i:i+BATCH_SIZE]
    phone_list = ",".join([f"'{p}'" for p in batch])
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(phones) + BATCH_SIZE - 1) // BATCH_SIZE

    recharge_sql = f"""
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
    ),
    clean_recharges AS (
        SELECT
            mobile,
            charges,
            selected_plan_id,
            TO_DATE(DATEADD(minute, 330, created_on)) AS recharge_date,
            TRY_TO_NUMBER(PARSE_JSON(extra_data):totalPaid::STRING) AS total_paid
        FROM deduped
        WHERE rn1 = 1 AND rn2 = 1
    ),
    plan_info AS (
        SELECT id, ROUND(time_limit / 60 / 60 / 24) AS plan_days,
               speed_limit_mbps, NAME
        FROM prod_db.public.t_plan_configuration
    )
    SELECT
        cr.mobile,
        COUNT(*) AS total_recharges,
        MIN(cr.recharge_date) AS first_recharge,
        MAX(cr.recharge_date) AS last_recharge,
        AVG(COALESCE(cr.total_paid, cr.charges)) AS avg_recharge_amount,
        STDDEV(COALESCE(cr.total_paid, cr.charges)) AS stddev_recharge_amount,
        MAX(CASE WHEN pi.plan_days >= 28 THEN 1 ELSE 0 END) AS has_monthly_plan,
        MAX(pi.speed_limit_mbps) AS max_speed_plan,
        AVG(pi.plan_days) AS avg_plan_days
    FROM clean_recharges cr
    LEFT JOIN plan_info pi ON cr.selected_plan_id = pi.id
    GROUP BY cr.mobile
    """

    batch_df = run_query(recharge_sql, f"Recharge features batch {batch_num}/{total_batches}")
    if len(batch_df) > 0:
        all_recharge_features.append(batch_df)

if all_recharge_features:
    recharge_features = pd.concat(all_recharge_features, ignore_index=True)
    recharge_features.rename(columns={'MOBILE': 'phone_number', 'mobile': 'phone_number'}, inplace=True)
    # Normalize column names to lowercase
    recharge_features.columns = [c.lower() if c != 'phone_number' else c for c in recharge_features.columns]
    if 'phone_number' not in recharge_features.columns:
        # Try to find the mobile column
        for col in recharge_features.columns:
            if 'mobile' in col.lower():
                recharge_features.rename(columns={col: 'phone_number'}, inplace=True)
                break
    print(f"  Total recharge features: {len(recharge_features)} phones matched")
else:
    recharge_features = pd.DataFrame()
    print("  WARNING: No recharge features computed")

# ── 4. Service ticket features (batched) ──
print("\n[4/8] Computing service ticket features...")

all_ticket_features = []

for i in range(0, len(phones), BATCH_SIZE):
    batch = phones[i:i+BATCH_SIZE]
    phone_list = ",".join([f"'{p}'" for p in batch])
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(phones) + BATCH_SIZE - 1) // BATCH_SIZE

    # Join via t_wg_customer to get device_id, then to service_ticket_model
    ticket_sql = f"""
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
            stm.ticket_type_final,
            stm.cx_px,
            stm.ticket_added_time,
            stm.final_resolved_time,
            stm.resolution_tat_bucket,
            DATEDIFF(hour, stm.ticket_added_time, stm.final_resolved_time) AS resolution_hours
        FROM customer_devices cd
        JOIN prod_db.public.service_ticket_model stm ON cd.device_id = stm.device_id
        WHERE LOWER(stm.last_title) NOT ILIKE '%shifting%'
          AND LOWER(stm.cx_px) <> 'cc'
    )
    SELECT
        mobile,
        COUNT(*) AS total_tickets,
        COUNT(CASE WHEN cx_px = 'Cx' THEN 1 END) AS cx_tickets,
        COUNT(CASE WHEN cx_px = 'Px' THEN 1 END) AS px_tickets,
        AVG(resolution_hours) AS avg_resolution_hours,
        MEDIAN(resolution_hours) AS median_resolution_hours,
        COUNT(CASE WHEN resolution_tat_bucket = 'within TAT' THEN 1 END) AS tickets_within_sla,
        COUNT(CASE WHEN resolution_tat_bucket = 'within TAT' THEN 1 END) * 100.0 /
            NULLIF(COUNT(*), 0) AS sla_compliance_pct,
        MIN(DATEADD(minute, 330, ticket_added_time)) AS first_ticket_date,
        MAX(DATEADD(minute, 330, ticket_added_time)) AS last_ticket_date
    FROM tickets
    GROUP BY mobile
    """

    batch_df = run_query(ticket_sql, f"Ticket features batch {batch_num}/{total_batches}")
    if len(batch_df) > 0:
        all_ticket_features.append(batch_df)

if all_ticket_features:
    ticket_features = pd.concat(all_ticket_features, ignore_index=True)
    ticket_features.columns = [c.lower() for c in ticket_features.columns]
    ticket_features.rename(columns={'mobile': 'phone_number'}, inplace=True)
    print(f"  Total ticket features: {len(ticket_features)} phones matched")
else:
    ticket_features = pd.DataFrame()
    print("  WARNING: No ticket features computed")

# ── 5. Partner quality proxy features ──
print("\n[5/8] Computing partner quality proxy features...")

# First get partner mapping for our NPS phones
sample_phones_500 = phones[:500]
phone_list_500 = ",".join([f"'{p}'" for p in sample_phones_500])

partner_mapping_sql = f"""
WITH customer_partner AS (
    SELECT
        twc.mobile,
        twc.nasid,
        twc.shard,
        twc.lco_account_id,
        prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS partner_lng_id,
        ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({phone_list_500})
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

partner_map_df = run_query(partner_mapping_sql, "Partner mapping (sample 500)")
if len(partner_map_df) > 0:
    print(f"  Partner mapping: {len(partner_map_df)} phones matched to partners")
    # Check how many unique partners
    partner_map_df.columns = [c.lower() for c in partner_map_df.columns]
    n_partners = partner_map_df['partner_lng_id'].nunique() if 'partner_lng_id' in partner_map_df.columns else 0
    print(f"  Unique partners: {n_partners}")

# ── 6. Partner uptime proxy ──
print("\n[6/8] Computing partner uptime proxy...")

# Get partner uptime from PARTNER_INFLUX_SUMMARY for the last 90 days
partner_uptime_sql = """
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

partner_uptime_df = run_query(partner_uptime_sql, "Partner uptime (90-day)")
if len(partner_uptime_df) > 0:
    partner_uptime_df.columns = [c.lower() for c in partner_uptime_df.columns]
    print(f"  Partners with uptime data: {len(partner_uptime_df)}")

# ── 7. Install TAT features ──
print("\n[7/8] Computing install TAT features (sample)...")

sample_phones_1000 = phones[:1000]
phone_list_1k = ",".join([f"'{p}'" for p in sample_phones_1000])

install_tat_sql = f"""
WITH install_events AS (
    SELECT
        mobile,
        MIN(DATEADD(minute, 330, added_time)) AS install_time,
        COUNT(*) AS install_attempts
    FROM prod_db.public.taskvanilla_audit
    WHERE mobile IN ({phone_list_1k})
      AND event_name = 'OTP_VERIFIED'
      AND mobile > '5999999999'
    GROUP BY mobile
),
booking_events AS (
    SELECT
        mobile,
        MIN(DATEADD(minute, 330, added_time)) AS booking_time
    FROM prod_db.public.booking_logs
    WHERE mobile IN ({phone_list_1k})
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

install_df = run_query(install_tat_sql, "Install TAT (sample 1000)")
if len(install_df) > 0:
    install_df.columns = [c.lower() for c in install_df.columns]
    install_df.rename(columns={'mobile': 'phone_number'}, inplace=True)
    print(f"  Install TAT data: {len(install_df)} phones")
    if 'install_tat_hours' in install_df.columns:
        valid_tat = install_df[install_df['install_tat_hours'].notna() & (install_df['install_tat_hours'] > 0)]
        if len(valid_tat) > 0:
            print(f"  Median Install TAT: {valid_tat['install_tat_hours'].median():.0f} hours")
            print(f"  Mean Install TAT: {valid_tat['install_tat_hours'].mean():.0f} hours")

# ── 8. Payment mode features ──
print("\n[8/8] Computing payment mode features...")

all_payment_features = []

for i in range(0, len(phones), BATCH_SIZE):
    batch = phones[i:i+BATCH_SIZE]
    phone_list = ",".join([f"'{p}'" for p in batch])
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(phones) + BATCH_SIZE - 1) // BATCH_SIZE

    payment_sql = f"""
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
        AVG(total_price) AS avg_payment_amount,
        MAX(TO_DATE(DATEADD(minute, 330, createDate))) AS last_payment_date
    FROM prod_db.public.wiomBillingWifi
    WHERE mobile IN ({phone_list})
      AND total_price >= 299
      AND paymentStatus = 1
      AND mobile > '5999999999'
      AND transaction_id NOT LIKE 'mr%'
      AND payment_type <> 2
      AND (refund_status <> 1 OR refund_status IS NULL)
    GROUP BY mobile
    """

    batch_df = run_query(payment_sql, f"Payment features batch {batch_num}/{total_batches}")
    if len(batch_df) > 0:
        all_payment_features.append(batch_df)

if all_payment_features:
    payment_features = pd.concat(all_payment_features, ignore_index=True)
    payment_features.columns = [c.lower() for c in payment_features.columns]
    payment_features.rename(columns={'mobile': 'phone_number'}, inplace=True)
    print(f"  Total payment features: {len(payment_features)} phones matched")
else:
    payment_features = pd.DataFrame()
    print("  WARNING: No payment features computed")

# ── MERGE all features into nps_analytical_base ──
print("\n" + "=" * 70)
print("MERGING ALL FEATURES")
print("=" * 70)

nps['phone_number'] = nps['phone_number'].astype(str)

# Merge recharge features
if len(recharge_features) > 0:
    recharge_features['phone_number'] = recharge_features['phone_number'].astype(str)
    nps = nps.merge(recharge_features, on='phone_number', how='left', suffixes=('', '_rech'))
    print(f"  After recharge merge: {nps['total_recharges'].notna().sum()} matched")

# Merge ticket features
if len(ticket_features) > 0:
    ticket_features['phone_number'] = ticket_features['phone_number'].astype(str)
    nps = nps.merge(ticket_features, on='phone_number', how='left', suffixes=('', '_tkt'))
    print(f"  After ticket merge: {nps['total_tickets'].notna().sum()} matched")

# Merge payment features
if len(payment_features) > 0:
    payment_features['phone_number'] = payment_features['phone_number'].astype(str)
    nps = nps.merge(payment_features, on='phone_number', how='left', suffixes=('', '_pay'))
    print(f"  After payment merge: {nps['total_payments'].notna().sum()} matched")

# Merge install TAT (only for matched phones)
if len(install_df) > 0:
    install_df['phone_number'] = install_df['phone_number'].astype(str)
    nps = nps.merge(install_df[['phone_number', 'install_tat_hours', 'install_attempts']],
                    on='phone_number', how='left', suffixes=('', '_inst'))
    print(f"  After install merge: {nps['install_tat_hours'].notna().sum()} matched")

# ── Compute derived features ──
print("\nComputing derived features...")

# Payment mode (primary)
if 'autopay_payments' in nps.columns and 'cash_payments' in nps.columns:
    def classify_payment_mode(row):
        if pd.isna(row.get('total_payments')) or row.get('total_payments', 0) == 0:
            return 'unknown'
        autopay_pct = row.get('autopay_payments', 0) / row['total_payments']
        cash_pct = row.get('cash_payments', 0) / row['total_payments']
        if autopay_pct > 0.5:
            return 'autopay'
        elif cash_pct > 0.5:
            return 'cash'
        else:
            return 'online'
    nps['payment_mode'] = nps.apply(classify_payment_mode, axis=1)

# Has tickets flag
if 'total_tickets' in nps.columns:
    nps['has_tickets'] = (nps['total_tickets'] > 0).astype(int)
    nps['total_tickets'] = nps['total_tickets'].fillna(0)

# Fill NAs for numeric features
numeric_fill_cols = ['total_recharges', 'avg_recharge_amount', 'total_tickets',
                     'cx_tickets', 'px_tickets', 'avg_resolution_hours',
                     'sla_compliance_pct', 'total_payments', 'autopay_payments',
                     'cash_payments', 'install_tat_hours']
for col in numeric_fill_cols:
    if col in nps.columns:
        # Don't fill with 0 — keep NaN for unmatched to distinguish from actual zeros
        pass

# ── Save analytical base ──
print("\nSaving analytical base...")
output_path = os.path.join(DATA, "nps_analytical_base.csv")
nps.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved: {output_path}")
print(f"  Rows: {len(nps)}")
print(f"  Columns: {len(nps.columns)}")

# ── Generate feature summary report ──
report_lines = []
r = report_lines.append
r("=" * 70)
r("PHASE 3 — FEATURE ENGINEERING REPORT")
r("=" * 70)
r("")
r(f"Total NPS respondents: {len(nps)}")
r(f"Total features: {len(nps.columns)}")
r("")

r("--- FEATURE COVERAGE ---")
for col in nps.columns:
    fill = nps[col].notna().sum()
    pct = fill / len(nps) * 100
    if col not in ['phone_number', 'oe_raw', 'Sprint Start Date', 'Sprint End Date', 'Install Date']:
        r(f"  {col:45s}: {fill:6d} / {len(nps)} ({pct:5.1f}%)")

r("")
r("--- SNOWFLAKE MATCH RATES ---")
if len(recharge_features) > 0:
    r(f"  Recharge data match: {nps['total_recharges'].notna().sum()} / {len(nps)} ({nps['total_recharges'].notna().sum()/len(nps)*100:.1f}%)")
if 'total_tickets' in nps.columns:
    r(f"  Ticket data match: {(nps['total_tickets'] > 0).sum()} / {len(nps)} ({(nps['total_tickets'] > 0).sum()/len(nps)*100:.1f}%)")
if len(payment_features) > 0:
    r(f"  Payment data match: {nps['total_payments'].notna().sum()} / {len(nps)} ({nps['total_payments'].notna().sum()/len(nps)*100:.1f}%)")

r("")
r("--- KEY FEATURE STATISTICS ---")
stat_cols = ['total_recharges', 'avg_recharge_amount', 'total_tickets',
             'avg_resolution_hours', 'sla_compliance_pct', 'total_payments',
             'install_tat_hours']
for col in stat_cols:
    if col in nps.columns and nps[col].notna().any():
        r(f"\n  {col}:")
        r(f"    Mean:   {nps[col].mean():.1f}")
        r(f"    Median: {nps[col].median():.1f}")
        r(f"    Std:    {nps[col].std():.1f}")
        r(f"    Min:    {nps[col].min():.1f}")
        r(f"    Max:    {nps[col].max():.1f}")

# Feature by NPS group
r("")
r("--- MEAN FEATURES BY NPS GROUP ---")
group_cols = [c for c in stat_cols if c in nps.columns and nps[c].notna().any()]
if group_cols:
    group_means = nps.groupby('nps_group')[group_cols].mean()
    for col in group_cols:
        r(f"\n  {col}:")
        for grp in ['Promoter', 'Passive', 'Detractor']:
            if grp in group_means.index:
                r(f"    {grp:12s}: {group_means.at[grp, col]:.2f}")

# Payment mode distribution
if 'payment_mode' in nps.columns:
    r("")
    r("--- PAYMENT MODE DISTRIBUTION ---")
    pm_dist = nps['payment_mode'].value_counts()
    for mode, cnt in pm_dist.items():
        r(f"  {mode:15s}: {cnt:6d} ({cnt/len(nps)*100:.1f}%)")

    r("")
    r("--- PAYMENT MODE x NPS GROUP ---")
    pm_nps = pd.crosstab(nps['payment_mode'], nps['nps_group'], normalize='index') * 100
    r(f"  {'Mode':15s} | {'Promoter':>10s} | {'Passive':>10s} | {'Detractor':>10s}")
    for mode in pm_nps.index:
        prom = pm_nps.at[mode, 'Promoter'] if 'Promoter' in pm_nps.columns else 0
        pas = pm_nps.at[mode, 'Passive'] if 'Passive' in pm_nps.columns else 0
        det = pm_nps.at[mode, 'Detractor'] if 'Detractor' in pm_nps.columns else 0
        r(f"  {mode:15s} | {prom:9.1f}% | {pas:9.1f}% | {det:9.1f}%")

report_path = os.path.join(OUTPUT, "phase3_feature_engineering.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n  Report saved to: {report_path}")
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)
