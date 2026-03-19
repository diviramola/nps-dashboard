"""
Score Wiom Customers Due for Recharge (Next 7 Days) with Churn Risk
====================================================================
1. Retrain population churn model from saved training data
2. Pull customers due for recharge in next 7 days (last recharge 21-28 days ago)
3. Compute operational features (same pipeline as population model)
4. Score each customer with churn probability
5. Identify top 3 risk drivers per customer
6. Export to Google Sheets (+ local CSV backup)

Output: Google Sheet URL + output/churn_risk_scores.csv
Runtime: ~15-20 minutes (smaller customer set)
"""

import sys, io, os, warnings, time, json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
os.makedirs(OUTPUT, exist_ok=True)

RISK_THRESHOLD = 0.5  # Default threshold — customers above this are "at risk"

# Persistent Google Sheet — daily runs add a new tab (named by date) to this sheet.
# Model Summary and Definitions tabs live here permanently.
GSHEET_ID = '1ewGLfhlizAGVURMjYSE52EsWLGZFysf2L_aAP5Is2Nk'

# Human-readable labels for features
# Only features actually used in the model (no volume-biased, broken, or duplicates)
FEATURE_LABELS = {
    'OVERALL_UPTIME_PCT': 'Overall Uptime (%)',
    'STDDEV_UPTIME': 'Uptime Variability',
    'SC_AVG_RXPOWER_IN_RANGE': 'Optical Signal Quality (%)',
    'SC_AVG_RXPOWER': 'Optical Power (dBm)',
    'SC_AVG_OPTICALPOWER_IN_RANGE': 'Optical Power In Range (%)',
    'SC_AVG_LATEST_SPEED': 'Avg Actual Speed (Mbps)',
    'SC_AVG_SPEED_IN_RANGE': 'Speed In Range (%)',
    'SC_SPEED_GAP_PCT': 'Speed Gap vs Plan (%)',
    'SC_AVG_PLAN_SPEED': 'Plan Speed (Mbps)',
    'SC_AVG_WEEKLY_DATA_GB': 'Weekly Data Usage (GB)',
    'avg_resolution_hours': 'Avg Resolution Time (hrs)',
    'SLA_COMPLIANCE_PCT': 'SLA Compliance (%)',
    'AVG_ANSWERED_SECONDS': 'Avg Call Duration (sec)',
    'missed_call_ratio': 'Missed Call Ratio',
    'resolution_rate': 'Ticket Resolution Rate',
    'autopay_ratio': 'Autopay Ratio',
}

report = []
def rpt(line=""):
    report.append(line)
    print(line, flush=True)


def run_query(sql, timeout=300):
    """Execute Snowflake query via Metabase API."""
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
        rpt(f"  WARNING: Hit 2000-row Metabase limit")
    return df


def batch_query(id_list, sql_template, batch_size=500, id_col='MOBILE', timeout=300, label=""):
    """Run batched queries for large ID lists."""
    results = []
    total = len(id_list)
    n_batches = (total - 1) // batch_size + 1
    consecutive_errors = 0
    rows_total = 0
    for i in range(0, total, batch_size):
        batch = id_list[i:i+batch_size]
        id_str = ",".join(f"'{p}'" for p in batch)
        sql = sql_template.replace('{{PHONE_LIST}}', id_str)
        batch_num = i // batch_size + 1
        try:
            df_batch = run_query(sql, timeout=timeout)
            if len(df_batch) > 0:
                results.append(df_batch)
                rows_total += len(df_batch)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
            # Progress every 10 batches
            if batch_num % 10 == 0 or batch_num == n_batches:
                rpt(f"    {label} batch {batch_num}/{n_batches}: {rows_total} rows so far")
        except Exception as e:
            consecutive_errors += 1
            if batch_num % 10 == 0:
                rpt(f"    {label} batch {batch_num}/{n_batches}: ERROR - {str(e)[:80]}")
        if consecutive_errors >= 5:
            rpt(f"    Stopping early after {consecutive_errors} consecutive failures")
            break
        time.sleep(0.3)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


rpt("=" * 100)
rpt("WIOM CHURN RISK SCORING — CUSTOMERS DUE FOR RECHARGE (NEXT 7 DAYS)")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: RETRAIN MODEL FROM SAVED POPULATION DATA
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 1: RETRAIN POPULATION CHURN MODEL")
rpt("=" * 100)

pop_file = os.path.join(DATA, "population_ops_features_50k.csv")
if not os.path.exists(pop_file):
    rpt(f"FATAL: Training data not found at {pop_file}")
    rpt("Run phase4b_population_churn_model.py first.")
    sys.exit(1)

df_train = pd.read_csv(pop_file, low_memory=False)
rpt(f"  Loaded training data: {len(df_train)} rows × {len(df_train.columns)} cols")

# Clean feature list — removed:
#   BROKEN: PEAK_UPTIME_PCT (all-NaN — PARTNER_INFLUX_SUMMARY is daily, no hourly data),
#           PEAK_VS_OVERALL_GAP (derived from broken PEAK_UPTIME_PCT)
#   VOLUME-BIASED CALL METRICS (r>0.7 correlated, missed_call_ratio already captures quality):
#           ivr_calls_per_month, missed_calls_per_month, answered_calls_per_month,
#           inbound_calls_per_month, inbound_answered_per_month, inbound_unanswered_per_month,
#           DROPPED_CALLS
#   VOLUME-BIASED TICKET METRICS (r>0.7 correlated):
#           tickets_per_month, cx_tickets_per_month, px_tickets_per_month,
#           distinct_issues_per_month, reopened_once_per_month, max_reopened_per_month,
#           has_tickets, install_attempts_per_month, ticket_severity
#   DUPLICATES: avg_resolution_hours_w (dup of avg_resolution_hours),
#               tk_sla_compliance (r=1.0 with SLA_COMPLIANCE_PCT),
#               AVG_CUSTOMER_CALLS (volume-biased)
POP_FEATURES = [
    # Uptime (from PARTNER_INFLUX_SUMMARY — daily level)
    'OVERALL_UPTIME_PCT', 'STDDEV_UPTIME',
    # Network quality (from NETWORK_SCORECARD)
    'SC_AVG_RXPOWER_IN_RANGE', 'SC_AVG_RXPOWER', 'SC_AVG_OPTICALPOWER_IN_RANGE',
    'SC_AVG_LATEST_SPEED', 'SC_AVG_SPEED_IN_RANGE', 'SC_SPEED_GAP_PCT', 'SC_AVG_PLAN_SPEED',
    'SC_AVG_WEEKLY_DATA_GB',
    # Service quality (quality ratios, not raw counts)
    'avg_resolution_hours', 'SLA_COMPLIANCE_PCT',
    'AVG_ANSWERED_SECONDS', 'missed_call_ratio',
    'resolution_rate',
    # Behavior
    'autopay_ratio',
]

# Find available features (handle case variations)
feature_cols = []
for f in POP_FEATURES:
    if f in df_train.columns:
        feature_cols.append(f)
    elif f.lower() in df_train.columns:
        feature_cols.append(f.lower())
    elif f.upper() in df_train.columns:
        feature_cols.append(f.upper())

rpt(f"  Features available: {len(feature_cols)}/{len(POP_FEATURES)}")

X_train = df_train[feature_cols].copy()
for c in X_train.columns:
    X_train[c] = pd.to_numeric(X_train[c], errors='coerce')

# Drop all-NaN columns
all_nan = X_train.columns[X_train.isna().all()]
if len(all_nan) > 0:
    rpt(f"  Dropping {len(all_nan)} all-NaN columns: {list(all_nan)}")
    X_train = X_train.drop(columns=all_nan)
    feature_cols = [f for f in feature_cols if f not in all_nan]

# Imputer
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
y_train = df_train['IS_CHURNED'].astype(int).values

rpt(f"  Training matrix: {X_train_imp.shape[0]} rows × {X_train_imp.shape[1]} features")
rpt(f"  Training churn rate: {y_train.mean()*100:.1f}%")

# Train models (same hyperparams as population script)
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_samples_leaf=20, random_state=42
)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
gb.fit(X_train_imp, y_train)
rf.fit(X_train_imp, y_train)
rpt("  Models trained (GB + RF)")

# Feature importances
feat_importance = pd.Series(gb.feature_importances_, index=X_train_imp.columns).sort_values(ascending=False)

# Compute healthy baselines for driver explanation
healthy_mask = y_train == 0
healthy_medians = X_train_imp[healthy_mask].median()
healthy_stds = X_train_imp[healthy_mask].std()
# Churn direction: positive correlation = higher is riskier
churn_corr = X_train_imp.corrwith(pd.Series(y_train, index=X_train_imp.index))
churn_direction = np.sign(churn_corr)

rpt(f"  Top 5 drivers: {', '.join(feat_importance.head(5).index.tolist())}")
rpt("  Healthy baselines computed for driver explanation")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PULL ALL ACTIVE CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 2: PULL CUSTOMERS DUE FOR RECHARGE (plan expires within 7 days)")
rpt("=" * 100)

# Count customers whose current plan expires within the next 7 days
# This captures both M+ (28-day) and PayG (<28-day) plans via otp_expiry_time
count_sql = """
WITH deduped_recharges AS (
    SELECT mobile,
           TO_DATE(DATEADD(minute, 330, created_on)) AS recharge_date,
           TO_DATE(DATEADD(minute, 330, otp_expiry_time)) AS plan_expiry_date,
           ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id) AS rn1,
           ROW_NUMBER() OVER (PARTITION BY mobile, router_nas_id, charges,
               TO_DATE(DATEADD(minute, 330, created_on)) ORDER BY id) AS rn2
    FROM prod_db.public.t_router_user_mapping
    WHERE device_limit = '10' AND otp = 'DONE' AND mobile > '5999999999' AND store_group_id = 0
      AND TO_DATE(DATEADD(minute, 330, created_on)) >= '2025-01-01'
),
valid_recharges AS (
    SELECT mobile, recharge_date, plan_expiry_date FROM deduped_recharges WHERE rn1 = 1 AND rn2 = 1
),
latest_per_customer AS (
    SELECT mobile, MAX(plan_expiry_date) AS plan_expiry
    FROM valid_recharges
    GROUP BY mobile
)
SELECT COUNT(*) AS due_count
FROM latest_per_customer
WHERE plan_expiry BETWEEN CURRENT_DATE() AND DATEADD(DAY, 7, CURRENT_DATE())
"""

rpt("  Counting customers due for recharge (plan expiry within 7 days)...")
df_count = run_query(count_sql, timeout=600)
active_count = int(df_count.iloc[0, 0]) if len(df_count) > 0 else 0
rpt(f"  Customers due for recharge: {active_count:,}")

if active_count == 0:
    rpt("FATAL: No active customers found")
    sys.exit(1)

# Determine partition count (keep each partition under 1500 rows for safety)
N_PARTITIONS = max(int(np.ceil(active_count / 1500)), 10)
rpt(f"  Using {N_PARTITIONS} hash partitions (~{active_count // N_PARTITIONS} per partition)")

# Pull active customers partition by partition
active_frames = []
for p in range(N_PARTITIONS):
    sql = f"""
    WITH deduped_recharges AS (
        SELECT mobile,
               TO_DATE(DATEADD(minute, 330, created_on)) AS recharge_date,
               TO_DATE(DATEADD(minute, 330, otp_expiry_time)) AS plan_expiry_date,
               ROUND(DATEDIFF(hour, otp_issued_time, otp_expiry_time) / 24.0) AS plan_days,
               ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id) AS rn1,
               ROW_NUMBER() OVER (PARTITION BY mobile, router_nas_id, charges,
                   TO_DATE(DATEADD(minute, 330, created_on)) ORDER BY id) AS rn2
        FROM prod_db.public.t_router_user_mapping
        WHERE device_limit = '10' AND otp = 'DONE' AND mobile > '5999999999' AND store_group_id = 0
          AND TO_DATE(DATEADD(minute, 330, created_on)) >= '2025-01-01'
    ),
    valid_recharges AS (
        SELECT mobile, recharge_date, plan_expiry_date, plan_days
        FROM deduped_recharges WHERE rn1 = 1 AND rn2 = 1
    ),
    latest_recharge AS (
        SELECT mobile, recharge_date, plan_expiry_date, plan_days,
               ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY recharge_date DESC) AS rn
        FROM valid_recharges
    ),
    due_customers AS (
        SELECT mobile, recharge_date AS last_recharge_date, plan_expiry_date AS plan_expiry,
               plan_days,
               CASE WHEN plan_days >= 28 THEN 'M+' ELSE 'PayG' END AS plan_type
        FROM latest_recharge
        WHERE rn = 1
          AND plan_expiry_date BETWEEN CURRENT_DATE() AND DATEADD(DAY, 7, CURRENT_DATE())
    ),
    first_ever AS (
        SELECT mobile, MIN(recharge_date) AS first_recharge_ever
        FROM valid_recharges WHERE mobile IN (SELECT mobile FROM due_customers)
        GROUP BY mobile
    ),
    recharge_stats AS (
        SELECT mobile, COUNT(*) AS recharge_count
        FROM valid_recharges WHERE mobile IN (SELECT mobile FROM due_customers)
        GROUP BY mobile
    )
    SELECT d.mobile AS MOBILE, r.recharge_count AS RECHARGE_COUNT,
           fe.first_recharge_ever AS FIRST_RECHARGE_EVER,
           d.last_recharge_date AS LAST_RECHARGE_DATE,
           d.plan_expiry AS PLAN_EXPIRY,
           d.plan_type AS PLAN_TYPE,
           d.plan_days AS PLAN_DAYS,
           DATEDIFF(DAY, fe.first_recharge_ever, CURRENT_DATE()) AS TENURE_DAYS,
           DATEDIFF(DAY, d.last_recharge_date, CURRENT_DATE()) AS DAYS_SINCE_LAST
    FROM due_customers d
    JOIN first_ever fe ON d.mobile = fe.mobile
    JOIN recharge_stats r ON d.mobile = r.mobile
    WHERE ABS(HASH(d.mobile)) % {N_PARTITIONS} = {p}
    """
    try:
        df_p = run_query(sql, timeout=600)
        if len(df_p) > 0:
            active_frames.append(df_p)
    except Exception as e:
        rpt(f"    Partition {p} ERROR: {str(e)[:100]}")
    if (p + 1) % 20 == 0 or p == N_PARTITIONS - 1:
        total_so_far = sum(len(f) for f in active_frames)
        rpt(f"    Partition {p+1}/{N_PARTITIONS}: {total_so_far:,} customers so far")
    time.sleep(0.3)

df_active = pd.concat(active_frames, ignore_index=True) if active_frames else pd.DataFrame()
# Deduplicate (shouldn't have dupes but safety)
df_active = df_active.drop_duplicates(subset='MOBILE')
rpt(f"\n  Total active customers pulled: {len(df_active):,}")

if len(df_active) == 0:
    rpt("FATAL: No active customers pulled")
    sys.exit(1)

for c in ['RECHARGE_COUNT', 'TENURE_DAYS', 'DAYS_SINCE_LAST', 'PLAN_DAYS']:
    if c in df_active.columns:
        df_active[c] = pd.to_numeric(df_active[c], errors='coerce')

rpt(f"  Avg tenure: {df_active['TENURE_DAYS'].mean():.0f} days")
rpt(f"  Median recharges: {df_active['RECHARGE_COUNT'].median():.0f}")
rpt(f"  Avg days since last: {df_active['DAYS_SINCE_LAST'].mean():.1f}")
if 'PLAN_TYPE' in df_active.columns:
    rpt(f"  Plan type split: {df_active['PLAN_TYPE'].value_counts().to_dict()}")

phone_list = df_active['MOBILE'].astype(str).str.strip().tolist()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: COMPUTE FEATURES (same pipeline as population model)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 3: COMPUTE OPERATIONAL FEATURES")
rpt("=" * 100)

# --- 3a: Partner Mapping ---
rpt("\n  3a. Partner Mapping...")
sql_partner = """
WITH customer_partner AS (
    SELECT twc.mobile, twc.device_id, twc.shard, twc.lco_account_id,
           prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS partner_lng_id,
           ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({{PHONE_LIST}}) AND twc._FIVETRAN_DELETED = false
)
SELECT cp.mobile AS MOBILE, cp.device_id AS DEVICE_ID, cp.partner_lng_id AS PARTNER_LNG_ID,
       hb.MIS_CITY AS CITY, hb.CLUSTER AS CLUSTER
FROM customer_partner cp
LEFT JOIN prod_db.public.HIERARCHY_BASE hb
    ON cp.partner_lng_id = hb.PARTNER_ACCOUNT_ID AND hb.DEDUP_FLAG = 1
WHERE cp.rn = 1
"""
df_partner = batch_query(phone_list, sql_partner, batch_size=1000, label="Partner")
if len(df_partner) > 0:
    df_active = df_active.merge(df_partner, on='MOBILE', how='left')
    rpt(f"  Partner match: {df_active['PARTNER_LNG_ID'].notna().mean()*100:.1f}%")

# --- 3b: Partner Uptime ---
rpt("\n  3b. Partner Uptime...")
partner_ids = df_active['PARTNER_LNG_ID'].dropna().unique().tolist() if 'PARTNER_LNG_ID' in df_active.columns else []
rpt(f"  Unique partners: {len(partner_ids)}")

if len(partner_ids) > 0:
    # Batch partners if too many for one query (>2000 unique)
    uptime_frames = []
    for i in range(0, len(partner_ids), 1500):
        batch_partners = partner_ids[i:i+1500]
        partner_str = ",".join(f"'{p}'" for p in batch_partners)
        sql_uptime = f"""
        SELECT partner_id AS PARTNER_LNG_ID,
               AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) * 100 AS OVERALL_UPTIME_PCT,
               STDDEV(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) * 100 AS STDDEV_UPTIME
        FROM prod_db.public.PARTNER_INFLUX_SUMMARY
        WHERE partner_id IN ({partner_str})
          AND DATEADD(day, -1, appended_date) >= DATEADD(DAY, -90, CURRENT_DATE())
        GROUP BY partner_id
        """
        try:
            df_u = run_query(sql_uptime, timeout=300)
            if len(df_u) > 0:
                uptime_frames.append(df_u)
        except Exception as e:
            rpt(f"    Uptime batch error: {str(e)[:100]}")
        time.sleep(0.3)

    if uptime_frames:
        df_uptime = pd.concat(uptime_frames, ignore_index=True)
        for c in ['OVERALL_UPTIME_PCT', 'STDDEV_UPTIME']:
            if c in df_uptime.columns:
                df_uptime[c] = pd.to_numeric(df_uptime[c], errors='coerce')
        df_active = df_active.merge(df_uptime, on='PARTNER_LNG_ID', how='left')
        rpt(f"  Uptime match: {df_active['OVERALL_UPTIME_PCT'].notna().mean()*100:.1f}%")

# --- 3c: Network Scorecard ---
rpt("\n  3c. Network Scorecard...")
sql_scorecard = """
SELECT MOBILE,
       AVG(PLAN_SPEED) AS SC_AVG_PLAN_SPEED,
       AVG(LATEST_SPEED::FLOAT) AS SC_AVG_LATEST_SPEED,
       AVG(CASE WHEN PLAN_SPEED > 0 AND LATEST_SPEED IS NOT NULL
           THEN (PLAN_SPEED - LATEST_SPEED::FLOAT) / PLAN_SPEED END) AS SC_SPEED_GAP_PCT,
       AVG(SPEED_IN_RANGE) AS SC_AVG_SPEED_IN_RANGE,
       AVG(RXPOWER::FLOAT) AS SC_AVG_RXPOWER,
       AVG(RXPOWER_IN_RANGE) AS SC_AVG_RXPOWER_IN_RANGE,
       AVG(OPTICALPOWER_IN_RANGE) AS SC_AVG_OPTICALPOWER_IN_RANGE,
       AVG(DATA_USED_GB::FLOAT) AS SC_AVG_WEEKLY_DATA_GB
FROM prod_db.public.NETWORK_SCORECARD
WHERE MOBILE IN ({{PHONE_LIST}}) AND WEEK_START >= '2025-06-01'
GROUP BY MOBILE
"""
df_scorecard = batch_query(phone_list, sql_scorecard, batch_size=1000, label="Scorecard")
if len(df_scorecard) > 0:
    for c in df_scorecard.columns:
        if c != 'MOBILE':
            df_scorecard[c] = pd.to_numeric(df_scorecard[c], errors='coerce')
    df_active = df_active.merge(df_scorecard, on='MOBILE', how='left')
    rpt(f"  Scorecard match: {df_active['SC_AVG_RXPOWER_IN_RANGE'].notna().mean()*100:.1f}%")

# --- 3d: Service Tickets ---
rpt("\n  3d. Service Tickets...")
device_ids = df_active.loc[df_active['DEVICE_ID'].notna(), 'DEVICE_ID'].unique().tolist() if 'DEVICE_ID' in df_active.columns else []
rpt(f"  Device IDs available: {len(device_ids)}")

if len(device_ids) > 0:
    sql_tickets = """
    SELECT DEVICE_ID,
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
    df_tickets = batch_query(device_ids, sql_tickets, batch_size=500, id_col='DEVICE_ID', label="Tickets")
    if len(df_tickets) > 0:
        for c in df_tickets.columns:
            if c != 'DEVICE_ID':
                df_tickets[c] = pd.to_numeric(df_tickets[c], errors='coerce')
        df_active = df_active.merge(df_tickets, on='DEVICE_ID', how='left')

        # Derived ticket features
        df_active['has_tickets'] = (df_active['TOTAL_TICKETS'].fillna(0) > 0).astype(int)
        df_active['avg_resolution_hours'] = df_active['AVG_RESOLUTION_MINS'] / 60.0
        p99_val = df_active['avg_resolution_hours'].quantile(0.99)
        df_active['avg_resolution_hours_w'] = df_active['avg_resolution_hours'].clip(upper=p99_val)
        df_active['tk_sla_compliance'] = df_active['SLA_COMPLIANCE_PCT']
        total_t = df_active['TOTAL_TICKETS'].fillna(0)
        reopens = df_active['TICKETS_REOPENED_ONCE'].fillna(0)
        df_active['ticket_severity'] = np.where(total_t > 0,
            np.log1p(total_t) * 0.3 + reopens * 0.3 +
            (1 - df_active['SLA_COMPLIANCE_PCT'].fillna(0.5)) * 0.4, 0)
        resolved = df_active['TICKETS_RESOLVED'].fillna(0)
        df_active['resolution_rate'] = np.where(total_t > 0, resolved / total_t, np.nan)
        rpt(f"  Ticket match: {df_active['TOTAL_TICKETS'].notna().mean()*100:.1f}%")

# --- 3e: IVR Calls ---
rpt("\n  3e. IVR Calls...")
sql_ivr = """
SELECT RIGHT(CLIENT_NUMBER, 10) AS MOBILE,
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
df_ivr = batch_query(phone_list, sql_ivr, batch_size=1000, label="IVR")
if len(df_ivr) > 0:
    for c in df_ivr.columns:
        if c != 'MOBILE':
            df_ivr[c] = pd.to_numeric(df_ivr[c], errors='coerce')
    df_active = df_active.merge(df_ivr, on='MOBILE', how='left')
    tot = df_active['TOTAL_IVR_CALLS'].fillna(0)
    miss = df_active['MISSED_CALLS'].fillna(0)
    df_active['missed_call_ratio'] = np.where(tot > 0, miss / tot, 0)
    rpt(f"  IVR match: {df_active['TOTAL_IVR_CALLS'].notna().mean()*100:.1f}%")

# --- 3f: Install TAT ---
rpt("\n  3f. Install TAT...")
sql_install = """
SELECT mobile AS MOBILE, MIN(DATEADD(minute, 330, added_time)) AS FIRST_INSTALL_TS,
       COUNT(*) AS INSTALL_ATTEMPTS
FROM prod_db.public.taskvanilla_audit
WHERE mobile IN ({{PHONE_LIST}}) AND event_name = 'OTP_VERIFIED' AND mobile > '5999999999'
GROUP BY mobile
"""
df_install = batch_query(phone_list, sql_install, batch_size=1000, label="Install")
if len(df_install) > 0:
    for c in df_install.columns:
        if c != 'MOBILE' and c != 'FIRST_INSTALL_TS':
            df_install[c] = pd.to_numeric(df_install[c], errors='coerce')
    df_active = df_active.merge(df_install, on='MOBILE', how='left')
    rpt(f"  Install match: {df_active['INSTALL_ATTEMPTS'].notna().mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: NORMALIZE + ENGINEER FEATURES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 4: NORMALIZE + ENGINEER FEATURES")
rpt("=" * 100)

df_active['TENURE_DAYS'] = pd.to_numeric(df_active['TENURE_DAYS'], errors='coerce').fillna(30)
tenure_months = np.maximum(df_active['TENURE_DAYS'].values / 30.0, 1.0)

# No per-month normalization needed — volume-biased count features removed from model.
# Quality ratios (missed_call_ratio, resolution_rate, SLA_COMPLIANCE_PCT) are already normalized.
NORMALIZE_MAP = {}

for orig, normed in NORMALIZE_MAP.items():
    if orig in df_active.columns:
        raw = pd.to_numeric(df_active[orig], errors='coerce').fillna(0).values
        df_active[normed] = raw / tenure_months

# autopay_ratio placeholder
if 'autopay_ratio' not in df_active.columns:
    df_active['autopay_ratio'] = 0

rpt(f"  Feature engineering complete: {len(df_active)} rows × {len(df_active.columns)} cols")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SCORE ALL CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 5: SCORE ALL CUSTOMERS")
rpt("=" * 100)

# Align features to training columns
X_score = pd.DataFrame(index=df_active.index)
for f in X_train_imp.columns:
    if f in df_active.columns:
        X_score[f] = pd.to_numeric(df_active[f], errors='coerce')
    elif f.lower() in df_active.columns:
        X_score[f] = pd.to_numeric(df_active[f.lower()], errors='coerce')
    elif f.upper() in df_active.columns:
        X_score[f] = pd.to_numeric(df_active[f.upper()], errors='coerce')
    else:
        X_score[f] = np.nan

# Impute with training medians
X_score_imp = pd.DataFrame(
    imputer.transform(X_score), columns=X_score.columns, index=X_score.index
)

# Score
gb_proba = gb.predict_proba(X_score_imp)[:, 1]
rf_proba = rf.predict_proba(X_score_imp)[:, 1]
ensemble_proba = (gb_proba + rf_proba) / 2

df_active['RISK_SCORE'] = ensemble_proba
df_active['RISK_TIER'] = pd.cut(
    ensemble_proba, bins=[0, 0.4, 0.6, 1.01],
    labels=['Low', 'Medium', 'High'], include_lowest=True
).astype(str)
df_active['AT_RISK'] = np.where(ensemble_proba >= RISK_THRESHOLD, 'YES', 'NO')

rpt(f"  Scored {len(df_active):,} customers")
rpt(f"  Score distribution:")
for pct in [10, 25, 50, 75, 90, 95]:
    rpt(f"    P{pct}: {np.percentile(ensemble_proba, pct):.3f}")
rpt(f"  Mean risk score: {ensemble_proba.mean():.3f}")
rpt(f"  At risk (>={RISK_THRESHOLD}): {(ensemble_proba >= RISK_THRESHOLD).sum():,} "
    f"({(ensemble_proba >= RISK_THRESHOLD).mean()*100:.1f}%)")
rpt(f"  Risk tiers: High={sum(df_active['RISK_TIER']=='High'):,}, "
    f"Medium={sum(df_active['RISK_TIER']=='Medium'):,}, "
    f"Low={sum(df_active['RISK_TIER']=='Low'):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: PER-CUSTOMER TOP 3 DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 6: IDENTIFY TOP 3 RISK DRIVERS PER CUSTOMER")
rpt("=" * 100)

def get_top_drivers(row_features, n=3):
    """For a single customer, return top N risk drivers with explanations."""
    drivers = []
    for feat in X_train_imp.columns:
        val = row_features[feat]
        importance = feat_importance.get(feat, 0)
        h_med = healthy_medians[feat]
        h_std = max(healthy_stds[feat], 0.001)
        direction = churn_direction.get(feat, 0)

        # Deviation in risky direction
        deviation = (val - h_med) * direction
        if deviation > 0:  # Only count if deviating in risky direction
            risk_contrib = importance * (deviation / h_std)
            label = FEATURE_LABELS.get(feat, feat)
            drivers.append((risk_contrib, label, val, h_med))

    drivers.sort(reverse=True)
    return drivers[:n]

# Compute drivers for all customers (vectorized where possible)
rpt("  Computing drivers for all customers...")
driver_cols = {f'DRIVER_{i+1}': [] for i in range(3)}

feature_matrix = X_score_imp.values
feat_names = X_score_imp.columns.tolist()
imp_vals = feat_importance.reindex(feat_names).fillna(0).values
h_med_vals = healthy_medians.reindex(feat_names).fillna(0).values
h_std_vals = np.maximum(healthy_stds.reindex(feat_names).fillna(0.001).values, 0.001)
dir_vals = churn_direction.reindex(feat_names).fillna(0).values

for idx in range(len(feature_matrix)):
    row = feature_matrix[idx]
    deviations = (row - h_med_vals) * dir_vals
    risk_contribs = imp_vals * np.maximum(deviations / h_std_vals, 0)

    # Top 3 indices
    top_idx = np.argsort(risk_contribs)[::-1][:3]

    for j in range(3):
        if j < len(top_idx) and risk_contribs[top_idx[j]] > 0:
            fi = top_idx[j]
            label = FEATURE_LABELS.get(feat_names[fi], feat_names[fi])
            val = row[fi]
            h_med = h_med_vals[fi]
            driver_cols[f'DRIVER_{j+1}'].append(f"{label}: {val:.2f} (healthy: {h_med:.2f})")
        else:
            driver_cols[f'DRIVER_{j+1}'].append("")

    if (idx + 1) % 50000 == 0:
        rpt(f"    Processed {idx+1:,}/{len(feature_matrix):,}")

for col_name, values in driver_cols.items():
    df_active[col_name] = values

rpt(f"  Drivers computed for {len(df_active):,} customers")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 7: EXPORT TO GOOGLE SHEETS + CSV")
rpt("=" * 100)

# Prepare output dataframe
# Compute "Recharge Due In" = days until plan expiry
if 'PLAN_EXPIRY' in df_active.columns:
    plan_exp = pd.to_datetime(df_active['PLAN_EXPIRY'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    today = pd.Timestamp.now().normalize()
    df_active['RECHARGE_DUE_IN'] = (plan_exp - today).dt.days

output_cols = [
    'MOBILE', 'RISK_SCORE', 'RISK_TIER', 'AT_RISK',
    'DRIVER_1', 'DRIVER_2', 'DRIVER_3',
    'PLAN_TYPE', 'PLAN_EXPIRY', 'RECHARGE_DUE_IN', 'CITY', 'TENURE_DAYS',
    'DAYS_SINCE_LAST', 'RECHARGE_COUNT', 'LAST_RECHARGE_DATE',
]
# Only include columns that exist
output_cols = [c for c in output_cols if c in df_active.columns]
df_output = df_active[output_cols].sort_values('RISK_SCORE', ascending=False).reset_index(drop=True)

# Round risk score
df_output['RISK_SCORE'] = df_output['RISK_SCORE'].round(4)

# Rename for readability
col_rename = {
    'MOBILE': 'Phone',
    'RISK_SCORE': 'Risk Score',
    'RISK_TIER': 'Risk Tier',
    'AT_RISK': 'At Risk',
    'DRIVER_1': 'Driver 1',
    'DRIVER_2': 'Driver 2',
    'DRIVER_3': 'Driver 3',
    'PLAN_TYPE': 'Plan Type',
    'PLAN_EXPIRY': 'Plan Expiry',
    'RECHARGE_DUE_IN': 'Recharge Due In (Days)',
    'CITY': 'City',
    'TENURE_DAYS': 'Tenure (Days)',
    'DAYS_SINCE_LAST': 'Days Since Last Recharge',
    'RECHARGE_COUNT': 'Recharge Count',
    'LAST_RECHARGE_DATE': 'Last Recharge Date',
}
df_output = df_output.rename(columns=col_rename)

# Save local CSV
csv_path = os.path.join(OUTPUT, "churn_risk_scores.csv")
df_output.to_csv(csv_path, index=False)
rpt(f"  CSV saved: {csv_path}")
rpt(f"  Rows: {len(df_output):,}")

# --- Google Sheets Export ---
GSPREAD_CREDS = r'C:\Users\nikhi\.config\gspread\credentials.json'
GSPREAD_TOKEN = r'C:\Users\nikhi\.config\gspread\authorized_user.json'

try:
    import gspread
    if not os.path.exists(GSPREAD_CREDS):
        rpt(f"\n  Google Sheets: credentials.json not found at {GSPREAD_CREDS}")
        rpt("  Skipping Google Sheets — CSV exported successfully.")
    else:
        rpt("\n  Connecting to Google Sheets...")
        gc = gspread.oauth(
            credentials_filename=GSPREAD_CREDS,
            authorized_user_filename=GSPREAD_TOKEN
        )

        # Open the persistent sheet (Model Summary + Definitions tabs live here)
        sh = gc.open_by_key(GSHEET_ID)
        rpt(f"  Opened sheet: {sh.title}")

        # Tab name = today's date
        tab_name = datetime.now().strftime('%Y-%m-%d')

        # If a tab with today's date already exists, clear and reuse it
        try:
            ws = sh.worksheet(tab_name)
            ws.clear()
            rpt(f"  Reusing existing tab: {tab_name}")
        except gspread.exceptions.WorksheetNotFound:
            # Add new tab at position 0 (leftmost, before Model Summary/Definitions)
            ws = sh.add_worksheet(title=tab_name, rows=len(df_output) + 1, cols=len(df_output.columns))
            # Move to first position so latest run is the first tab
            sh.reorder_worksheets([ws] + [s for s in sh.worksheets() if s.id != ws.id])
            rpt(f"  Created new tab: {tab_name}")

        # Resize to fit data
        ws.resize(rows=len(df_output) + 1, cols=len(df_output.columns))

        # Write header
        headers = df_output.columns.tolist()
        ws.update(range_name='A1', values=[headers])

        # Write data in batches (gspread limit: ~50K cells per batch)
        BATCH_ROWS = 5000
        total_rows = len(df_output)
        for start in range(0, total_rows, BATCH_ROWS):
            end = min(start + BATCH_ROWS, total_rows)
            chunk = df_output.iloc[start:end].fillna('').values.tolist()
            chunk = [[str(v) if not isinstance(v, (int, float, str)) else v for v in row] for row in chunk]
            cell_range = f'A{start + 2}'
            ws.update(range_name=cell_range, values=chunk)
            rpt(f"    Uploaded rows {start+1:,}-{end:,}/{total_rows:,}")
            time.sleep(1)

        # Bold header row
        ws.format(f'A1:{chr(64 + len(headers))}1', {'textFormat': {'bold': True}})

        sheet_url = sh.url
        rpt(f"\n  Google Sheet URL: {sheet_url}")
        rpt(f"  Tab: {tab_name}")

except ImportError:
    rpt("  gspread not installed — skipping Google Sheets export")
except Exception as e:
    rpt(f"  Google Sheets export failed: {str(e)[:200]}")
    rpt("  CSV export succeeded — use that as fallback")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("SUMMARY")
rpt("=" * 100)
rpt(f"  Total active customers scored: {len(df_output):,}")
rpt(f"  At risk (score >= {RISK_THRESHOLD}): {sum(df_output['At Risk'] == 'YES'):,}")
rpt(f"  High risk (>= 0.6): {sum(df_output['Risk Tier'] == 'High'):,}")
rpt(f"  Medium risk (0.4-0.6): {sum(df_output['Risk Tier'] == 'Medium'):,}")
rpt(f"  Low risk (< 0.4): {sum(df_output['Risk Tier'] == 'Low'):,}")
rpt(f"  Mean risk score: {df_output['Risk Score'].mean():.3f}")
rpt(f"  CSV: {csv_path}")

# Save report
report_file = os.path.join(OUTPUT, "scoring_report.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n{'='*100}")
rpt(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*100}")
