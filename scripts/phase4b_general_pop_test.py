"""
Phase 4B — General Population Validation
==========================================
Tests the churn model on 100 / 1000 / 10000 random Wiom customers
(NOT just NPS respondents) from the same Jul 2025 - Jan 2026 window.

Excludes: call features, tickets_post_sprint
Keeps: FAILURE_RATE_PCT (experience quality, not longevity proxy)
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from dotenv import load_dotenv
import requests

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv(r'C:\credentials\.env')
METABASE_API_KEY = os.getenv('METABASE_API_KEY')

BASE_DIR = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE_DIR, "data")

def run_query(sql, timeout=180):
    url = "https://metabase.wiom.in/api/dataset"
    headers = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}
    payload = {"database": 113, "type": "native", "native": {"query": sql}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    rows = data["data"]["rows"]
    cols = [c["name"] for c in data["data"]["cols"]]
    return pd.DataFrame(rows, columns=cols)


print("=" * 80)
print("GENERAL POPULATION CHURN MODEL TEST")
print(f"Testing on 100 / 1,000 / 10,000 random Wiom customers")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: TRAIN MODEL ON NPS DATA (clean version)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Training model on NPS respondent data...")

df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df['churn_binary'] = df['is_churned'].astype(int)

churn_col = 'is_churned'
EXCLUDE_ALWAYS = {
    'churn_binary', churn_col, 'phone_number', 'response_id', 'user_id',
    'timestamp', 'profile_all_identities', 'alt_mobile', 'lng_nas',
    'device_id', 'device_id_mapped',
    'churn_risk_score', 'churn_risk_pct', 'churn_label', 'risk_category',
    'partner_churn_rate', 'partner_at_risk', 'partner_risk_level',
    'PAYMENT_ATTEMPTS', 'PAYMENT_FAILURES', 'PAYMENT_SUCCESSES',
    'TOTAL_PAYMENT_EVENTS',
    'total_payments', 'autopay_payments', 'cash_payments',
    'avg_payment_amount', 'avg_recharge_amount',
    'total_recharges', 'recharge_done', 'recharges_before_sprint',
    'recharge_regularity', 'recharge_same_day', 'days_since_last_recharge',
    'DAYS_TO_FIRST_RECHARGE', 'HOURS_TO_FIRST_RECHARGE',
    'first_recharge', 'last_recharge', 'last_recharge_date',
    'plan_expiry_date', 'plan_expiry_window',
    'is_cash_payment', 'payment_successes',
    'partner_status', 'partner_status_at_survey', 'call_status',
    'sc_scorecard_weeks', 'cis_influx_data_days', 'ul1_usage_data_days',
    'sc_plan_active_ratio', 'cis_active_day_ratio', 'sc_scorecard_ticket_count',
    # NEW: call features (biased — reactive to risk)
    'OUTBOUND_CALLS', 'INBOUND_CALLS', 'MISSED_CALLS', 'TOTAL_IVR_CALLS',
    'AVG_ANSWERED_SECONDS', 'missed_call_ratio',
    # NEW: temporal leakage
    'tickets_post_sprint',
}
EXCLUDE_SUBSTRINGS = ['churn_risk', 'risk_category', 'churn_label', 'partner_risk']
NAN_PATTERNS = ['_nan', '_None', '_missing']

nps_and_theme_cols = {'nps_score', 'nps_group',
    'primary_theme', 'secondary_theme', 'NPS Reason - Primary', 'Primary Category',
    'sentiment_polarity', 'sentiment_intensity', 'comment_quality_flag'}

def is_excluded(col_name):
    if col_name in EXCLUDE_ALWAYS:
        return True
    for sub in EXCLUDE_SUBSTRINGS:
        if sub in col_name.lower():
            return True
    return False

# Classify safe features
safe_feats = []
for c in df.columns:
    if is_excluded(c) or c in nps_and_theme_cols:
        continue
    if df[c].notna().sum() / len(df) <= 0.05:
        continue
    if df[c].dtype == 'object' and df[c].nunique() > 30:
        continue
    active_fill = df.loc[df['churn_binary']==0, c].notna().mean()
    churn_fill = df.loc[df['churn_binary']==1, c].notna().mean()
    if abs(active_fill - churn_fill) * 100 < 10:
        safe_feats.append(c)

print(f"  Safe features (no calls, no tickets_post_sprint): {len(safe_feats)}")

# Build training matrix
def build_matrix(df_in, feature_list):
    num_feats, cat_feats = [], []
    for f in feature_list:
        if f not in df_in.columns:
            continue
        if any(pat in f for pat in NAN_PATTERNS):
            continue
        if df_in[f].dtype in ('object', 'category', 'bool'):
            if df_in[f].nunique() <= 20:
                cat_feats.append(f)
        else:
            num_feats.append(f)
    frames = []
    if num_feats:
        num_df = df_in[num_feats].copy()
        for c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors='coerce')
        frames.append(num_df)
    if cat_feats:
        for c in cat_feats:
            dummies = pd.get_dummies(df_in[c].astype(str), prefix=c, drop_first=True)
            nan_cols = [col for col in dummies.columns if any(p in col.lower() for p in NAN_PATTERNS)]
            dummies = dummies.drop(columns=nan_cols, errors='ignore')
            if len(dummies.columns) > 10:
                dummies = dummies[dummies.sum().nlargest(10).index]
            frames.append(dummies)
    if frames:
        X = pd.concat(frames, axis=1)
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imp, imputer
    return pd.DataFrame(), None

X_train, train_imputer = build_matrix(df, safe_feats)
y_train = df['churn_binary'].values
train_columns = list(X_train.columns)

print(f"  Training matrix: {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"  Training churn rate: {y_train.mean()*100:.1f}%")

# Quick CV to confirm model quality
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 min_samples_leaf=20, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)

gb_cv = cross_val_score(gb, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"  CV AUC (GB): {gb_cv.mean():.4f}")

# Fit on full training set
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Identify which features we can get from Snowflake
# Top model features (GB importance)
gb_imp = pd.Series(gb.feature_importances_, index=train_columns).sort_values(ascending=False)
print(f"\n  Top 10 model features:")
for i, (f, v) in enumerate(gb_imp.head(10).items()):
    print(f"    {i+1}. {f}: {v:.4f} ({v*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: QUERY SNOWFLAKE FOR GENERAL POPULATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[STEP 2] Querying Snowflake for general customer population...")
print("=" * 80)

# Get 10000 random customers who had at least one recharge in Jul 2025 - Jan 2026
# and check their churn status (16+ days since last recharge = churned)
print("\n  Pulling 10,000 random customers active during Jul 2025 - Jan 2026...")

sql_customers = """
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
      AND TO_DATE(DATEADD(minute, 330, created_on)) >= '2025-06-01'
),
valid_recharges AS (
    SELECT mobile, recharge_date
    FROM deduped_recharges
    WHERE rn1 = 1 AND rn2 = 1
),
active_customers AS (
    SELECT
        mobile AS MOBILE,
        COUNT(*) AS recharge_count,
        MIN(recharge_date) AS first_recharge,
        MAX(recharge_date) AS last_recharge_in_window
    FROM valid_recharges
    WHERE recharge_date BETWEEN '2025-07-01' AND '2026-01-31'
    GROUP BY mobile
    HAVING COUNT(*) >= 2
),
latest_recharge AS (
    SELECT
        mobile AS MOBILE,
        MAX(recharge_date) AS last_recharge_ever
    FROM valid_recharges
    WHERE mobile IN (SELECT MOBILE FROM active_customers)
    GROUP BY mobile
),
churned AS (
    SELECT
        ac.MOBILE,
        ac.recharge_count,
        lr.last_recharge_ever,
        DATEDIFF(DAY, lr.last_recharge_ever, CURRENT_DATE()) AS days_since_last_recharge,
        CASE WHEN DATEDIFF(DAY, lr.last_recharge_ever, CURRENT_DATE()) >= 16 THEN 1 ELSE 0 END AS is_churned
    FROM active_customers ac
    JOIN latest_recharge lr ON ac.MOBILE = lr.MOBILE
)
SELECT * FROM churned
ORDER BY RANDOM()
LIMIT 10000
"""

df_pop = run_query(sql_customers)
print(f"  Got {len(df_pop)} customers")
print(f"  Churn rate: {df_pop['IS_CHURNED'].astype(int).mean()*100:.1f}%")
print(f"  Avg days since last recharge: {df_pop['DAYS_SINCE_LAST_RECHARGE'].astype(float).mean():.0f}")

# Check overlap with NPS respondents
nps_phones = set(df['phone_number'].astype(str).str.strip())
pop_phones = set(df_pop['MOBILE'].astype(str).str.strip())
overlap = nps_phones & pop_phones
print(f"  Overlap with NPS respondents: {len(overlap)} ({len(overlap)/len(pop_phones)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PULL OPERATIONAL FEATURES FOR THESE CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[STEP 3] Pulling operational features from Snowflake...")
print("=" * 80)

mobiles = df_pop['MOBILE'].astype(str).tolist()

# We'll query in batches to avoid query size limits
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

all_features = []
batch_size = 2000

for batch_num, mobile_batch in enumerate(chunked(mobiles, batch_size)):
    print(f"  Batch {batch_num+1}/{(len(mobiles)-1)//batch_size+1} ({len(mobile_batch)} customers)...")

    values_clause = ",".join([f"('{m}')" for m in mobile_batch])

    sql_features = f"""
    WITH customers AS (
        SELECT COLUMN1 AS MOBILE FROM VALUES {values_clause}
    ),
    -- Scorecard: join directly on MOBILE (matches enrichment script pattern)
    scorecard AS (
        SELECT
            ns.MOBILE,
            MAX(ns.LNG_NAS) AS lng_nas,
            AVG(ns.CUSTOMER_UPTIME::FLOAT) AS avg_uptime_pct,
            STDDEV(ns.CUSTOMER_UPTIME::FLOAT) AS stddev_uptime,
            MIN(ns.CUSTOMER_UPTIME::FLOAT) AS min_uptime,
            AVG(ns.LATEST_SPEED::FLOAT) AS sc_avg_latest_speed,
            AVG(ns.PLAN_SPEED) AS sc_avg_plan_speed,
            AVG(ns.RXPOWER_IN_RANGE) AS sc_avg_rxpower_in_range,
            AVG(ns.RXPOWER::FLOAT) AS sc_avg_rxpower,
            AVG(ns.SPEED_IN_RANGE) AS sc_avg_speed_in_range,
            AVG(ns.DATA_USED_GB::FLOAT) AS sc_avg_weekly_data_gb,
            AVG(CASE WHEN ns.PLAN_SPEED > 0
                THEN (ns.PLAN_SPEED - ns.LATEST_SPEED::FLOAT) / ns.PLAN_SPEED
                ELSE NULL END) AS sc_speed_gap_pct
        FROM PROD_DB.PUBLIC.NETWORK_SCORECARD ns
        WHERE ns.MOBILE IN (SELECT MOBILE FROM customers)
          AND ns.WEEK_START >= '2025-06-01'
          AND ns.WEEK_START < '2026-02-01'
        GROUP BY ns.MOBILE
    ),
    -- Influx: join via LNG_NAS from scorecard (matches enrichment pattern)
    influx AS (
        SELECT
            sc.MOBILE,
            AVG(CASE WHEN cis.EXPECTED_PINGS_SO_FAR > 0
                THEN cis.PINGS_RECEIVED_TODAY * 1.0 / cis.EXPECTED_PINGS_SO_FAR
                ELSE NULL END) AS OVERALL_UPTIME_PCT,
            AVG(CASE WHEN cis.EXPECTED_PEAK_PINGS > 0
                THEN cis.PINGS_PEAK_HOURS * 1.0 / cis.EXPECTED_PEAK_PINGS
                ELSE NULL END) AS PEAK_UPTIME_PCT,
            AVG(CASE WHEN cis.PINGS_RECEIVED_TODAY > 0
                THEN cis.STABLE_PINGS_COUNT * 1.0 / cis.PINGS_RECEIVED_TODAY
                ELSE NULL END) AS PEAK_STABLE_PCT,
            AVG(cis.HAD_PEAK_INTERRUPTION) AS AVG_PEAK_INTERRUPTIONS
        FROM scorecard sc
        JOIN PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY cis
            ON CAST(cis.LNG_NAS AS VARCHAR) = sc.lng_nas
        WHERE cis.APPENDED_DATE >= '2025-06-01'
          AND cis.APPENDED_DATE < '2026-02-01'
        GROUP BY sc.MOBILE
    ),
    -- Device mapping for ticket join
    device_map AS (
        SELECT mobile, device_id
        FROM (
            SELECT mobile, device_id,
                   ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY added_time DESC) AS rn
            FROM prod_db.public.t_wg_customer
            WHERE mobile IN (SELECT MOBILE FROM customers)
              AND mobile > '5999999999'
              AND _FIVETRAN_DELETED = false
        ) t WHERE rn = 1
    ),
    -- Ticket features (via device_id, matching enrichment pattern)
    tickets AS (
        SELECT
            dm.mobile AS MOBILE,
            COUNT(*) AS total_tickets,
            CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END AS has_tickets,
            AVG(st.RESOLUTION_PERIOD_MINS_CALENDARHRS) / 60.0 AS avg_resolution_hours,
            COUNT(CASE WHEN st.RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END)::FLOAT /
                NULLIF(COUNT(*), 0) AS sla_compliance_pct,
            COUNT(CASE WHEN st.RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END)::FLOAT /
                NULLIF(COUNT(*), 0) AS tk_sla_compliance,
            AVG(st.RESOLUTION_PERIOD_MINS_CALENDARHRS) AS tk_avg_resolution_mins
        FROM device_map dm
        JOIN PROD_DB.PUBLIC.SERVICE_TICKET_MODEL st
            ON dm.device_id = st.DEVICE_ID
        WHERE st.TICKET_ADDED_TIME >= '2025-06-01'
          AND st.TICKET_ADDED_TIME < '2026-02-01'
        GROUP BY dm.mobile
    ),
    -- Payment failure rate (keep FAILURE_RATE_PCT only)
    payments AS (
        SELECT
            c.MOBILE,
            CASE WHEN COUNT(*) > 0 THEN
                COUNT(CASE WHEN pl.STATUS = 'order_failed' THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100
            ELSE NULL END AS FAILURE_RATE_PCT
        FROM customers c
        JOIN PROD_DB.PUBLIC.PAYMENT_LOGS pl
            ON c.MOBILE = pl.MOBILE
        WHERE pl.CREATED_AT BETWEEN '2025-07-01' AND '2026-01-31'
        GROUP BY c.MOBILE
    )
    SELECT
        c.MOBILE,
        sc.avg_uptime_pct,
        sc.stddev_uptime,
        sc.min_uptime,
        sc.sc_avg_latest_speed,
        sc.sc_avg_plan_speed,
        sc.sc_avg_rxpower_in_range,
        sc.sc_avg_rxpower,
        sc.sc_avg_speed_in_range,
        sc.sc_avg_weekly_data_gb,
        sc.sc_speed_gap_pct,
        tk.total_tickets,
        tk.has_tickets,
        tk.avg_resolution_hours,
        tk.sla_compliance_pct,
        tk.tk_sla_compliance,
        tk.tk_avg_resolution_mins,
        inf.OVERALL_UPTIME_PCT,
        inf.PEAK_UPTIME_PCT,
        inf.PEAK_STABLE_PCT,
        inf.AVG_PEAK_INTERRUPTIONS,
        p.FAILURE_RATE_PCT,
        NULL AS ANSWERED_CALLS,
        NULL AS AVG_CALL_DURATION
    FROM customers c
    LEFT JOIN scorecard sc ON c.MOBILE = sc.MOBILE
    LEFT JOIN tickets tk ON c.MOBILE = tk.MOBILE
    LEFT JOIN influx inf ON c.MOBILE = inf.MOBILE
    LEFT JOIN payments p ON c.MOBILE = p.MOBILE
    """

    try:
        batch_df = run_query(sql_features, timeout=300)
        all_features.append(batch_df)
        print(f"    Got {len(batch_df)} rows")
    except Exception as e:
        print(f"    ERROR: {str(e)[:200]}")
        # Try a simpler query with just scorecard data
        try:
            sql_simple = f"""
            WITH customers AS (
                SELECT COLUMN1 AS MOBILE FROM VALUES {values_clause}
            )
            SELECT
                ns.MOBILE,
                AVG(ns.CUSTOMER_UPTIME::FLOAT) AS avg_uptime_pct,
                STDDEV(ns.CUSTOMER_UPTIME::FLOAT) AS stddev_uptime,
                MIN(ns.CUSTOMER_UPTIME::FLOAT) AS min_uptime,
                AVG(ns.LATEST_SPEED::FLOAT) AS sc_avg_latest_speed,
                AVG(ns.PLAN_SPEED) AS sc_avg_plan_speed,
                AVG(ns.RXPOWER_IN_RANGE) AS sc_avg_rxpower_in_range,
                AVG(ns.SPEED_IN_RANGE) AS sc_avg_speed_in_range,
                AVG(ns.DATA_USED_GB::FLOAT) AS sc_avg_weekly_data_gb
            FROM PROD_DB.PUBLIC.NETWORK_SCORECARD ns
            WHERE ns.MOBILE IN (SELECT MOBILE FROM customers)
              AND ns.WEEK_START >= '2025-06-01'
              AND ns.WEEK_START < '2026-02-01'
            GROUP BY ns.MOBILE
            """
            batch_df = run_query(sql_simple, timeout=300)
            all_features.append(batch_df)
            print(f"    FALLBACK (scorecard only): Got {len(batch_df)} rows")
        except Exception as e2:
            print(f"    FALLBACK ALSO FAILED: {str(e2)[:200]}")

if not all_features:
    print("  FATAL: No feature data retrieved. Cannot proceed.")
    sys.exit(1)

df_features = pd.concat(all_features, ignore_index=True)
print(f"\n  Total features retrieved: {len(df_features)} rows")

# Merge with churn labels
df_pop['MOBILE'] = df_pop['MOBILE'].astype(str).str.strip()
df_features['MOBILE'] = df_features['MOBILE'].astype(str).str.strip()
df_test_pop = df_pop.merge(df_features, on='MOBILE', how='left')

# Convert numeric columns
for c in df_test_pop.columns:
    if c not in ('MOBILE',):
        df_test_pop[c] = pd.to_numeric(df_test_pop[c], errors='coerce')

print(f"  Merged dataset: {len(df_test_pop)} rows")

# Check feature fill rates
print(f"\n  Feature fill rates in general population:")
key_features = ['avg_uptime_pct', 'sc_avg_rxpower_in_range', 'tk_sla_compliance',
                'FAILURE_RATE_PCT', 'total_tickets', 'avg_resolution_hours',
                'sla_compliance_pct', 'OVERALL_UPTIME_PCT', 'ANSWERED_CALLS']
for f in key_features:
    if f in df_test_pop.columns:
        fill = df_test_pop[f].notna().mean() * 100
        print(f"    {f:35s}: {fill:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD TEST MATRICES AND PREDICT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[STEP 4] Building feature matrices and predicting...")
print("=" * 80)

# Map Snowflake column names to model feature names (case-sensitive matching)
# The model was trained with specific column names from the enriched CSV
col_map = {}
for tc in train_columns:
    # Direct match
    if tc in df_test_pop.columns:
        col_map[tc] = tc
    # Case-insensitive match
    elif tc.upper() in df_test_pop.columns:
        col_map[tc] = tc.upper()
    elif tc.lower() in df_test_pop.columns:
        col_map[tc] = tc.lower()

print(f"  Model features matched to Snowflake: {len(col_map)}/{len(train_columns)}")
print(f"  Unmatched features (will use training median): {len(train_columns) - len(col_map)}")

# Build test feature matrix aligned to training columns
X_pop = pd.DataFrame(index=df_test_pop.index)
for tc in train_columns:
    if tc in col_map:
        X_pop[tc] = pd.to_numeric(df_test_pop[col_map[tc]], errors='coerce')
    else:
        X_pop[tc] = np.nan  # Will be imputed with training median

# Impute using training imputer
X_pop_imp = pd.DataFrame(
    train_imputer.transform(X_pop),
    columns=train_columns,
    index=X_pop.index
)

y_pop = df_test_pop['IS_CHURNED'].astype(int).values

# Feature fill summary
real_data_pct = (X_pop.notna().sum(axis=1).mean() / len(train_columns)) * 100
print(f"  Avg features with real data per customer: {X_pop.notna().sum(axis=1).mean():.0f}/{len(train_columns)} ({real_data_pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: EVALUATE AT 100 / 1000 / 10000
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("[STEP 5] Model performance at different sample sizes")
print("=" * 80)

# Predict on full set
gb_proba_all = gb.predict_proba(X_pop_imp)[:, 1]
rf_proba_all = rf.predict_proba(X_pop_imp)[:, 1]
ens_proba_all = (gb_proba_all + rf_proba_all) / 2

# Evaluate at different sample sizes
np.random.seed(42)
full_indices = np.arange(len(df_test_pop))
np.random.shuffle(full_indices)

print(f"\n  {'Sample':>8s} | {'N':>6s} | {'Churn%':>7s} | {'GB AUC':>8s} | {'RF AUC':>8s} | {'Ens AUC':>8s} | {'Acc':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s}")
print(f"  {'-'*8} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")

# Also NPS baseline for comparison
nps_gb_cv = cross_val_score(gb, X_train, y_train, cv=cv, scoring='roc_auc')
# Temporal holdout baseline
train_mask = df['sprint_num'].between(1, 9)
test_mask = df['sprint_num'].between(10, 11)

# Baseline: NPS respondent CV
print(f"  {'NPS CV':>8s} | {len(df):6d} | {df['churn_binary'].mean()*100:6.1f}% | {nps_gb_cv.mean():8.4f} |      N/A |      N/A |   N/A |   N/A |   N/A |   N/A")

for sample_size in [100, 1000, 10000]:
    n = min(sample_size, len(full_indices))
    idx = full_indices[:n]

    y_s = y_pop[idx]
    gb_s = gb_proba_all[idx]
    rf_s = rf_proba_all[idx]
    ens_s = ens_proba_all[idx]

    # Need both classes for AUC
    if y_s.sum() == 0 or y_s.sum() == len(y_s):
        print(f"  {sample_size:>8d} | {n:6d} | {y_s.mean()*100:6.1f}% | Only one class — cannot compute AUC")
        continue

    auc_gb = roc_auc_score(y_s, gb_s)
    auc_rf = roc_auc_score(y_s, rf_s)
    auc_ens = roc_auc_score(y_s, ens_s)

    pred = (ens_s >= 0.5).astype(int)
    cm = confusion_matrix(y_s, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    print(f"  {sample_size:>8d} | {n:6d} | {y_s.mean()*100:6.1f}% | {auc_gb:8.4f} | {auc_rf:8.4f} | {auc_ens:8.4f} | {acc*100:5.1f}% | {prec*100:5.1f}% | {rec*100:5.1f}% | {f1*100:5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SHOW 30 SAMPLE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS — 30 Random General-Population Customers")
print("=" * 80)

# Pick a balanced sample for display
churned_pop = np.where(y_pop == 1)[0]
active_pop = np.where(y_pop == 0)[0]
n_show_churn = min(15, len(churned_pop))
n_show_active = min(15, len(active_pop))

show_idx = np.concatenate([
    np.random.choice(churned_pop, n_show_churn, replace=False) if n_show_churn > 0 else np.array([], dtype=int),
    np.random.choice(active_pop, n_show_active, replace=False) if n_show_active > 0 else np.array([], dtype=int),
])
np.random.shuffle(show_idx)

def mask_phone(phone):
    phone = str(phone)
    if len(phone) >= 10:
        return phone[:3] + '***' + phone[-3:]
    return '***'

print(f"\n  {'#':>3s} | {'Phone':>10s} | {'DaysSinceRch':>12s} | {'Actual':>8s} | {'Pred%':>6s} | {'Verdict':>10s} | {'Uptime':>7s} | {'SLA':>5s} | {'Tickets':>7s}")
print(f"  {'-'*3} | {'-'*10} | {'-'*12} | {'-'*8} | {'-'*6} | {'-'*10} | {'-'*7} | {'-'*5} | {'-'*7}")

correct = 0
for i, idx in enumerate(show_idx):
    row = df_test_pop.iloc[idx]
    phone = mask_phone(row['MOBILE'])
    days = int(row['DAYS_SINCE_LAST_RECHARGE']) if pd.notna(row['DAYS_SINCE_LAST_RECHARGE']) else -1
    actual = 'CHURNED' if int(row['IS_CHURNED']) == 1 else 'ACTIVE'
    prob = ens_proba_all[idx]
    pred = 'CHURN' if prob >= 0.5 else 'ACTIVE'

    if (actual == 'CHURNED' and pred == 'CHURN') or (actual == 'ACTIVE' and pred == 'ACTIVE'):
        verdict = 'CORRECT'
        correct += 1
    elif actual == 'CHURNED' and pred == 'ACTIVE':
        verdict = 'MISSED'
    else:
        verdict = 'FALSE_ALARM'

    uptime = row.get('avg_uptime_pct', None)
    uptime_s = f"{uptime:.0f}%" if pd.notna(uptime) else "N/A"
    sla = row.get('tk_sla_compliance', None)
    sla_s = f"{sla:.0f}%" if pd.notna(sla) else "N/A"
    tix = row.get('total_tickets', None)
    tix_s = str(int(tix)) if pd.notna(tix) else "0"

    print(f"  {i+1:3d} | {phone:>10s} | {days:12d} | {actual:>8s} | {prob*100:5.1f}% | {verdict:>10s} | {uptime_s:>7s} | {sla_s:>5s} | {tix_s:>7s}")

n_shown = len(show_idx)
print(f"\n  SAMPLE ACCURACY: {correct}/{n_shown} = {correct/n_shown*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
  MODEL: Leakage-free v3 (no calls, no tickets_post_sprint, keeps FAILURE_RATE_PCT)
  TRAINED ON: 13,045 NPS respondents (Jul 2025 - Jan 2026)
  TESTED ON: {len(df_test_pop)} random Wiom customers from same period

  Feature coverage: {len(col_map)}/{len(train_columns)} features matched from Snowflake
  ({len(train_columns) - len(col_map)} features imputed with training median)

  Key insight: Compare NPS-respondent CV AUC vs general-population AUC to assess
  whether the model generalizes beyond survey takers.
""")

print(f"{'='*80}")
print(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
