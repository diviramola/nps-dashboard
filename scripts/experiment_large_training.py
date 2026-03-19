"""
Experiment: Train churn model on ~50K customers (vs current 15K)
================================================================
Compare feature importances, model parameters, and performance metrics
between the 15K baseline model and a 50K expanded model.

Approach: Same sprint-based cohort design, but use hash partitioning to
pull ~4000 customers per sprint (2 partitions × 2000 rows each).
13 sprints × ~4000 = ~52K target, minus dedup = ~45-50K.
"""

import sys, io, os, warnings, time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    return pd.DataFrame(rows, columns=cols)


def batch_query(id_list, sql_template, batch_size=500, id_col='MOBILE', timeout=300, label=""):
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
            if batch_num % 20 == 0 or batch_num == n_batches:
                rpt(f"    {label} batch {batch_num}/{n_batches}: {rows_total} rows so far")
        except Exception as e:
            consecutive_errors += 1
            if batch_num % 20 == 0:
                rpt(f"    {label} batch {batch_num}/{n_batches}: ERROR - {str(e)[:80]}")
        if consecutive_errors >= 5:
            rpt(f"    Stopping early after {consecutive_errors} consecutive failures")
            break
        time.sleep(0.3)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


rpt("=" * 100)
rpt("EXPERIMENT: LARGE TRAINING SET (50K) vs BASELINE (15K)")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 100)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SAMPLE ~50K CUSTOMERS (hash-partitioned sprint sampling)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 1: SAMPLE ~50K CUSTOMERS")
rpt("=" * 100)

SPRINTS = [
    (1,  '2025-07-07'), (2,  '2025-07-21'), (3,  '2025-08-04'),
    (4,  '2025-08-18'), (5,  '2025-09-01'), (6,  '2025-09-15'),
    (7,  '2025-09-29'), (8,  '2025-10-13'), (9,  '2025-10-27'),
    (10, '2025-11-10'), (11, '2025-11-24'), (12, '2025-12-08'),
    (13, '2026-01-05'),
]
LOOKBACK_DAYS = 28
CHURN_THRESHOLD_DAYS = 44
# Use 2 hash partitions per sprint, each returning up to 2000 rows = ~4000/sprint
NUM_PARTITIONS = 2


def build_sprint_query_partitioned(sprint_num, sprint_date, partition_id, num_partitions):
    lookback_date = (pd.to_datetime(sprint_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    return f"""
    WITH deduped_recharges AS (
        SELECT mobile,
               TO_DATE(DATEADD(minute, 330, created_on)) AS recharge_date,
               ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY id) AS rn1,
               ROW_NUMBER() OVER (PARTITION BY mobile, router_nas_id, charges,
                   TO_DATE(DATEADD(minute, 330, created_on)) ORDER BY id) AS rn2
        FROM prod_db.public.t_router_user_mapping
        WHERE device_limit = '10' AND otp = 'DONE' AND mobile > '5999999999' AND store_group_id = 0
          AND TO_DATE(DATEADD(minute, 330, created_on)) >= '2025-01-01'
    ),
    valid_recharges AS (
        SELECT mobile, recharge_date FROM deduped_recharges WHERE rn1 = 1 AND rn2 = 1
    ),
    active_at_sprint AS (
        SELECT mobile, MAX(recharge_date) AS last_recharge_before
        FROM valid_recharges
        WHERE recharge_date BETWEEN '{lookback_date}' AND '{sprint_date}'
        GROUP BY mobile
        HAVING ABS(HASH(mobile)) % {num_partitions} = {partition_id}
    ),
    first_ever AS (
        SELECT mobile, MIN(recharge_date) AS first_recharge_ever
        FROM valid_recharges WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    ),
    last_recharge_ever AS (
        SELECT mobile, MAX(recharge_date) AS last_recharge_date
        FROM valid_recharges WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    ),
    recharge_stats AS (
        SELECT mobile, COUNT(*) AS recharge_count
        FROM valid_recharges WHERE mobile IN (SELECT mobile FROM active_at_sprint)
        GROUP BY mobile
    )
    SELECT a.mobile AS MOBILE, {sprint_num} AS SPRINT_NUM,
           r.recharge_count AS RECHARGE_COUNT,
           fe.first_recharge_ever AS FIRST_RECHARGE_EVER,
           lre.last_recharge_date AS LAST_RECHARGE_DATE,
           DATEDIFF(DAY, fe.first_recharge_ever, '{sprint_date}') AS TENURE_DAYS,
           DATEDIFF(DAY, lre.last_recharge_date, CURRENT_DATE()) AS DAYS_SINCE_LAST,
           CASE WHEN DATEDIFF(DAY, lre.last_recharge_date, CURRENT_DATE()) >= {CHURN_THRESHOLD_DAYS}
                THEN 1 ELSE 0 END AS IS_CHURNED
    FROM active_at_sprint a
    JOIN first_ever fe ON a.mobile = fe.mobile
    JOIN last_recharge_ever lre ON a.mobile = lre.mobile
    JOIN recharge_stats r ON a.mobile = r.mobile
    """


pop_frames = []
seen_phones = set()

for sprint_num, sprint_date in SPRINTS:
    sprint_total = 0
    for part_id in range(NUM_PARTITIONS):
        sql = build_sprint_query_partitioned(sprint_num, sprint_date, part_id, NUM_PARTITIONS)
        try:
            df_batch = run_query(sql, timeout=600)
            if len(df_batch) > 0:
                new = df_batch[~df_batch['MOBILE'].isin(seen_phones)]
                seen_phones.update(new['MOBILE'].tolist())
                pop_frames.append(new)
                sprint_total += len(new)
        except Exception as e:
            rpt(f"    Sprint {sprint_num} part {part_id}: ERROR - {str(e)[:100]}")
        time.sleep(0.5)
    churn_pct = pd.to_numeric(pd.concat(pop_frames[-NUM_PARTITIONS:])['IS_CHURNED'], errors='coerce').mean() * 100 if pop_frames else 0
    rpt(f"  Sprint {sprint_num:2d} ({sprint_date}): {sprint_total:5d} new customers, churn={churn_pct:.1f}%")

df_pop = pd.concat(pop_frames, ignore_index=True) if pop_frames else pd.DataFrame()
df_pop['IS_CHURNED'] = pd.to_numeric(df_pop['IS_CHURNED'], errors='coerce')
df_pop['TENURE_DAYS'] = pd.to_numeric(df_pop['TENURE_DAYS'], errors='coerce')
rpt(f"\n  Total: {len(df_pop):,} customers from {len(SPRINTS)} sprints")
rpt(f"  Churn rate: {df_pop['IS_CHURNED'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE FEATURES (same pipeline as scoring script)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 2: COMPUTE FEATURES")
rpt("=" * 100)

phone_list = df_pop['MOBILE'].tolist()

# 2a: Partner mapping
rpt("  2a. Partner Mapping...")
sql_partner = """
WITH customer_partner AS (
    SELECT twc.mobile, twc.device_id, twc.shard, twc.lco_account_id,
           prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS partner_lng_id,
           ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({{PHONE_LIST}}) AND twc._FIVETRAN_DELETED = false
)
SELECT cp.mobile AS MOBILE, cp.device_id AS DEVICE_ID, cp.partner_lng_id AS PARTNER_LNG_ID,
       hb.MIS_CITY AS CITY
FROM customer_partner cp
LEFT JOIN prod_db.public.HIERARCHY_BASE hb
    ON cp.partner_lng_id = hb.PARTNER_ACCOUNT_ID AND hb.DEDUP_FLAG = 1
WHERE cp.rn = 1
"""
df_partner = batch_query(phone_list, sql_partner, batch_size=1000, label="Partner")
if len(df_partner) > 0:
    df_pop = df_pop.merge(df_partner, on='MOBILE', how='left')
    rpt(f"  Partner match: {df_pop['PARTNER_LNG_ID'].notna().mean()*100:.1f}%")

# 2b: Partner uptime
rpt("\n  2b. Partner Uptime...")
partner_ids = df_pop['PARTNER_LNG_ID'].dropna().unique().tolist() if 'PARTNER_LNG_ID' in df_pop.columns else []
rpt(f"  Unique partners: {len(partner_ids)}")

if len(partner_ids) > 0:
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
        df_pop = df_pop.merge(df_uptime, on='PARTNER_LNG_ID', how='left')
        rpt(f"  Uptime match: {df_pop['OVERALL_UPTIME_PCT'].notna().mean()*100:.1f}%")

# 2c: Network Scorecard
rpt("\n  2c. Network Scorecard...")
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
    df_pop = df_pop.merge(df_scorecard, on='MOBILE', how='left')
    rpt(f"  Scorecard match: {df_pop['SC_AVG_PLAN_SPEED'].notna().mean()*100:.1f}%")

# 2d: Service Tickets (via device_id — already have DEVICE_ID from partner mapping)
rpt("\n  2d. Service Tickets...")
if 'DEVICE_ID' not in df_pop.columns:
    rpt("    No DEVICE_ID column — fetching via t_wg_customer...")
    sql_device = """
    SELECT twc.mobile AS MOBILE, twc.device_id AS DEVICE_ID
    FROM prod_db.public.t_wg_customer twc
    WHERE twc.mobile IN ({{PHONE_LIST}}) AND twc._FIVETRAN_DELETED = false
    QUALIFY ROW_NUMBER() OVER (PARTITION BY twc.mobile ORDER BY twc.added_time DESC) = 1
    """
    df_device = batch_query(phone_list, sql_device, batch_size=1000, label="DeviceID")
    if len(df_device) > 0:
        df_pop = df_pop.merge(df_device, on='MOBILE', how='left')

device_ids = df_pop['DEVICE_ID'].dropna().unique().tolist() if 'DEVICE_ID' in df_pop.columns else []
rpt(f"  Device IDs: {len(device_ids)}")

if len(device_ids) > 0:
    sql_tickets = """
    SELECT DEVICE_ID,
           COUNT(*) AS TOTAL_TICKETS,
           AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) AS AVG_RESOLUTION_MINS,
           COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END) * 1.0 / NULLIF(COUNT(*), 0) AS SLA_COMPLIANCE_PCT,
           COUNT(CASE WHEN IS_RESOLVED = true THEN 1 END) AS TICKETS_RESOLVED
    FROM prod_db.public.SERVICE_TICKET_MODEL
    WHERE DEVICE_ID IN ({{PHONE_LIST}})
      AND LOWER(FIRST_TITLE) NOT LIKE '%shifting%' AND CX_PX <> 'CC'
      AND TO_DATE(DATEADD(minute, 330, TICKET_ADDED_TIME)) >= '2025-06-01'
    GROUP BY DEVICE_ID
    """
    df_tickets = batch_query(device_ids, sql_tickets, batch_size=300, id_col='DEVICE_ID', label="Tickets")
    if len(df_tickets) > 0:
        for c in df_tickets.columns:
            if c != 'DEVICE_ID':
                df_tickets[c] = pd.to_numeric(df_tickets[c], errors='coerce')
        df_tickets['avg_resolution_hours'] = df_tickets['AVG_RESOLUTION_MINS'] / 60.0
        df_tickets['resolution_rate'] = df_tickets['TICKETS_RESOLVED'] / df_tickets['TOTAL_TICKETS'].replace(0, np.nan)
        df_pop = df_pop.merge(df_tickets, on='DEVICE_ID', how='left')
        rpt(f"  Ticket match: {df_pop['TOTAL_TICKETS'].notna().mean()*100:.1f}%")

# 2e: IVR Calls
rpt("\n  2e. IVR Calls...")
sql_ivr = """
SELECT RIGHT(CLIENT_NUMBER, 10) AS MOBILE,
       COUNT(*) AS TOTAL_IVR_CALLS,
       COUNT(CASE WHEN STATUS = 'missed' THEN 1 END) AS MISSED_CALLS,
       AVG(CASE WHEN STATUS = 'answered' THEN ANSWERED_SECONDS END) AS AVG_ANSWERED_SECONDS
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
    df_pop = df_pop.merge(df_ivr, on='MOBILE', how='left')
    tot = df_pop['TOTAL_IVR_CALLS'].fillna(0)
    miss = df_pop['MISSED_CALLS'].fillna(0)
    df_pop['missed_call_ratio'] = np.where(tot > 0, miss / tot, 0)
    rpt(f"  IVR match: {df_pop['TOTAL_IVR_CALLS'].notna().mean()*100:.1f}%")

# autopay_ratio placeholder
if 'autopay_ratio' not in df_pop.columns:
    df_pop['autopay_ratio'] = 0

rpt(f"\n  Feature engineering complete: {len(df_pop)} rows x {len(df_pop.columns)} cols")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN MODEL ON 50K AND COMPARE WITH 15K BASELINE
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 3: TRAIN MODEL AND COMPARE")
rpt("=" * 100)

POP_FEATURES = [
    'OVERALL_UPTIME_PCT', 'STDDEV_UPTIME',
    'SC_AVG_RXPOWER_IN_RANGE', 'SC_AVG_RXPOWER', 'SC_AVG_OPTICALPOWER_IN_RANGE',
    'SC_AVG_LATEST_SPEED', 'SC_AVG_SPEED_IN_RANGE', 'SC_SPEED_GAP_PCT', 'SC_AVG_PLAN_SPEED',
    'SC_AVG_WEEKLY_DATA_GB',
    'avg_resolution_hours', 'SLA_COMPLIANCE_PCT',
    'AVG_ANSWERED_SECONDS', 'missed_call_ratio',
    'resolution_rate',
    'autopay_ratio',
]

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

# --- Load 15K baseline ---
rpt("\n  Loading 15K baseline training data...")
df_baseline = pd.read_csv(os.path.join(DATA, "population_ops_features.csv"), low_memory=False)
rpt(f"  Baseline: {len(df_baseline):,} rows, churn rate: {df_baseline['IS_CHURNED'].mean()*100:.1f}%")

def train_and_evaluate(df, label, features=POP_FEATURES):
    """Train GB+RF on given dataset, return metrics and importances."""
    feature_cols = [f for f in features if f in df.columns]
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # Drop all-NaN columns
    all_nan = X.columns[X.isna().all()]
    if len(all_nan) > 0:
        X = X.drop(columns=all_nan)
        feature_cols = [f for f in feature_cols if f not in all_nan]

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    y = df['IS_CHURNED'].astype(int).values

    rpt(f"\n  [{label}] Training on {len(X_imp):,} x {len(feature_cols)} features, churn={y.mean()*100:.1f}%")

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gb_scores = cross_val_score(gb, X_imp, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    rf_scores = cross_val_score(rf, X_imp, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    rpt(f"  [{label}] GB 5-fold AUC: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
    rpt(f"  [{label}] RF 5-fold AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # Final fit for importances
    gb.fit(X_imp, y)
    rf.fit(X_imp, y)

    gb_imp = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Healthy medians
    healthy_mask = y == 0
    healthy_medians = X_imp[healthy_mask].median()

    return {
        'label': label,
        'n_rows': len(X_imp),
        'n_features': len(feature_cols),
        'churn_rate': y.mean(),
        'gb_auc_mean': gb_scores.mean(),
        'gb_auc_std': gb_scores.std(),
        'rf_auc_mean': rf_scores.mean(),
        'rf_auc_std': rf_scores.std(),
        'gb_importance': gb_imp,
        'rf_importance': rf_imp,
        'healthy_medians': healthy_medians,
        'feature_cols': feature_cols,
    }


# Train both models
rpt("\n  Training baseline (15K) model...")
baseline_results = train_and_evaluate(df_baseline, "15K Baseline")

rpt("\n  Training expanded (50K) model...")
expanded_results = train_and_evaluate(df_pop, "50K Expanded")

# Save expanded training data
expanded_file = os.path.join(DATA, "population_ops_features_50k.csv")
df_pop.to_csv(expanded_file, index=False)
rpt(f"\n  Saved 50K training data: {expanded_file}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPARATIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("STEP 4: COMPARATIVE REPORT")
rpt("=" * 100)

rpt("\n  " + "-" * 80)
rpt("  MODEL PERFORMANCE")
rpt("  " + "-" * 80)
rpt(f"  {'Metric':<30s} {'15K Baseline':>15s} {'50K Expanded':>15s} {'Delta':>10s}")
rpt(f"  {'-'*30:<30s} {'-'*15:>15s} {'-'*15:>15s} {'-'*10:>10s}")

metrics = [
    ('Training rows', f"{baseline_results['n_rows']:,}", f"{expanded_results['n_rows']:,}", ""),
    ('Churn rate', f"{baseline_results['churn_rate']*100:.1f}%", f"{expanded_results['churn_rate']*100:.1f}%",
     f"{(expanded_results['churn_rate'] - baseline_results['churn_rate'])*100:+.1f}pp"),
    ('GB CV AUC', f"{baseline_results['gb_auc_mean']:.4f}", f"{expanded_results['gb_auc_mean']:.4f}",
     f"{expanded_results['gb_auc_mean'] - baseline_results['gb_auc_mean']:+.4f}"),
    ('RF CV AUC', f"{baseline_results['rf_auc_mean']:.4f}", f"{expanded_results['rf_auc_mean']:.4f}",
     f"{expanded_results['rf_auc_mean'] - baseline_results['rf_auc_mean']:+.4f}"),
]
for name, base, exp, delta in metrics:
    rpt(f"  {name:<30s} {base:>15s} {exp:>15s} {delta:>10s}")

rpt("\n  " + "-" * 80)
rpt("  FEATURE IMPORTANCE COMPARISON (Gradient Boosting)")
rpt("  " + "-" * 80)
rpt(f"  {'Feature':<30s} {'15K':>8s} {'50K':>8s} {'Delta':>8s} {'Direction':>12s}")
rpt(f"  {'-'*30:<30s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*12:>12s}")

# Align features
all_features = baseline_results['gb_importance'].index.tolist()
for f in all_features:
    base_val = baseline_results['gb_importance'].get(f, 0) * 100
    exp_val = expanded_results['gb_importance'].get(f, 0) * 100
    delta = exp_val - base_val
    direction = "UP" if delta > 0.5 else ("DOWN" if delta < -0.5 else "~same")
    label = FEATURE_LABELS.get(f, f)
    rpt(f"  {label:<30s} {base_val:>7.1f}% {exp_val:>7.1f}% {delta:>+7.1f}% {direction:>12s}")

rpt("\n  " + "-" * 80)
rpt("  HEALTHY MEDIANS COMPARISON")
rpt("  " + "-" * 80)
rpt(f"  {'Feature':<30s} {'15K Median':>12s} {'50K Median':>12s} {'Delta':>10s}")
rpt(f"  {'-'*30:<30s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s}")

for f in all_features:
    base_med = baseline_results['healthy_medians'].get(f, 0)
    exp_med = expanded_results['healthy_medians'].get(f, 0)
    delta = exp_med - base_med
    label = FEATURE_LABELS.get(f, f)
    rpt(f"  {label:<30s} {base_med:>12.2f} {exp_med:>12.2f} {delta:>+10.2f}")


rpt("\n" + "=" * 100)
rpt("KEY FINDINGS")
rpt("=" * 100)

# Auto-detect significant shifts
rpt("\n  Feature importance shifts > 2pp:")
for f in all_features:
    base_val = baseline_results['gb_importance'].get(f, 0) * 100
    exp_val = expanded_results['gb_importance'].get(f, 0) * 100
    delta = exp_val - base_val
    if abs(delta) > 2:
        label = FEATURE_LABELS.get(f, f)
        rpt(f"    {label}: {base_val:.1f}% -> {exp_val:.1f}% ({delta:+.1f}pp)")

rpt("\n  Healthy median shifts > 10%:")
for f in all_features:
    base_med = baseline_results['healthy_medians'].get(f, 0)
    exp_med = expanded_results['healthy_medians'].get(f, 0)
    if base_med != 0:
        pct_shift = (exp_med - base_med) / abs(base_med) * 100
        if abs(pct_shift) > 10:
            label = FEATURE_LABELS.get(f, f)
            rpt(f"    {label}: {base_med:.2f} -> {exp_med:.2f} ({pct_shift:+.1f}%)")

auc_delta = expanded_results['gb_auc_mean'] - baseline_results['gb_auc_mean']
if abs(auc_delta) < 0.005:
    rpt(f"\n  AUC is stable (delta={auc_delta:+.4f}) — model generalizes well to larger population.")
elif auc_delta > 0:
    rpt(f"\n  AUC improved by {auc_delta:+.4f} — larger training set helps model discrimination.")
else:
    rpt(f"\n  AUC decreased by {auc_delta:+.4f} — possible distribution shift in larger sample.")


# Save report
report_file = os.path.join(OUTPUT, "experiment_50k_comparison.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n  Report saved: {report_file}")
rpt(f"\n{'='*100}")
rpt(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*100}")
