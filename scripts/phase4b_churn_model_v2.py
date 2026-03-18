"""
Phase 4B: Churn Prediction Model v2 — Correct Framing
======================================================
OBJECTIVE (per user): Use NPS scores, comments, and experience/service
parameters as inputs to model churn/retention.

KEY CHANGES FROM v1:
1. Correct framing: NPS + themes + experience → churn (not ops → NPS)
2. Exclude self-fulfilling features (total_payments, PAYMENT_SUCCESSES)
3. Include new enriched features from Phase 3D (speed gap, uptime, sessions, tickets)
4. Tenure-stratified analysis
5. Incremental model comparison (ops only → ops+NPS → ops+NPS+themes)

Output:
- output/phase4b_churn_model_v2.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(BASE_DIR, "output")

report = []
def rpt(line=""):
    report.append(line)
    print(line)


rpt("=" * 70)
rpt("PHASE 4B: CHURN PREDICTION MODEL v2")
rpt(f"Framing: NPS + Comments + Experience → Churn")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

# ── Load enriched data ──
enriched_path = os.path.join(DATA, "nps_enriched_v2.csv")
if not os.path.exists(enriched_path):
    rpt(f"ERROR: {enriched_path} not found. Run phase3d_enrichment_v2.py first.")
    sys.exit(1)

df = pd.read_csv(enriched_path, low_memory=False)
rpt(f"\n[LOAD] {enriched_path}")
rpt(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ── Identify target variable (churn) ──
churn_col = None
for c in df.columns:
    if 'churn' in c.lower() and 'overall' in c.lower():
        churn_col = c; break
if not churn_col:
    for c in df.columns:
        if c.lower() == 'is_churned':
            churn_col = c; break
if not churn_col:
    for c in df.columns:
        if 'active/churn' in c.lower():
            churn_col = c; break

if not churn_col:
    rpt("ERROR: No churn column found")
    sys.exit(1)

rpt(f"\n  Churn column: {churn_col}")
rpt(f"  Values: {df[churn_col].value_counts().to_dict()}")

# Convert to binary
df['churn_binary'] = 0
for idx, val in df[churn_col].items():
    s = str(val).lower().strip()
    if s in ('churn', 'churned', '1', '1.0', 'true'):
        df.at[idx, 'churn_binary'] = 1

rpt(f"  Binary churn: {df['churn_binary'].value_counts().to_dict()}")
churn_rate = df['churn_binary'].mean()
rpt(f"  Churn rate: {churn_rate*100:.1f}%")

# ── Identify NPS score column ──
nps_score_col = None
for c in df.columns:
    if c.lower() == 'nps':
        nps_score_col = c; break
if not nps_score_col:
    for c in df.columns:
        if c.lower() == 'nps_score':
            nps_score_col = c; break

rpt(f"  NPS score column: {nps_score_col}")

# ── Identify NPS group column ──
nps_grp_col = None
for c in df.columns:
    if c.lower() == 'nps_group':
        nps_grp_col = c; break

rpt(f"  NPS group column: {nps_grp_col}")

# ── Find tenure column ──
tenure_col = None
for c in df.columns:
    if c.lower() == 'tenure_days':
        tenure_col = c; break
if not tenure_col:
    for c in df.columns:
        if c.lower() == 'tenure':
            tenure_col = c; break

rpt(f"  Tenure column: {tenure_col}")

# ── Define feature groups ──

# SELF-FULFILLING FEATURES TO EXCLUDE (correlation with churn is mechanical)
# Also DATA-EXISTENCE PROXY features that leak churn status
EXCLUDE_FEATURES = [
    # Payment features (mechanically correlated with churn)
    'total_payments', 'autopay_payments', 'cash_payments',
    'total_recharges', 'recharge_regularity', 'days_since_last_recharge',
    'payment_successes', 'PAYMENT_SUCCESSES',
    # DATA-EXISTENCE PROXIES — these measure "how long was customer active"
    # not "how good was their experience". Churned customers have fewer
    # data points, so these features ARE the churn label in disguise.
    'sc_scorecard_weeks',           # weeks of scorecard data = time active
    'cis_influx_data_days',         # days of influx data = time active
    'ul1_usage_data_days',          # days of usage data = time active
    'sc_plan_active_ratio',         # AVG(IS_PLAN_ACTIVE) = churn itself
    'cis_active_day_ratio',         # AVG(IS_ACTIVE_TODAY) = churn itself
    'sc_scorecard_ticket_count',    # total ticket count corr with time active
    # Target variable and identifiers
    'churn_binary', churn_col, 'phone_number', 'response_id', 'user_id',
    'timestamp', 'profile_all_identities', 'alt_mobile',
    'lng_nas', 'device_id', 'device_id_mapped',
]

# Additionally exclude any one-hot dummies of NaN categories (leaks missingness = churn)
EXCLUDE_PATTERNS = ['_nan', '_None', '_missing']

# NPS-derived features
NPS_FEATURES = []
if nps_score_col:
    NPS_FEATURES.append(nps_score_col)
if nps_grp_col:
    NPS_FEATURES.append(nps_grp_col)

# Theme features (from NPS comment analysis)
THEME_CANDIDATES = [
    'primary_theme', 'secondary_theme', 'primary_category',
    'nps_reason_primary', 'NPS Reason - Primary', 'Primary Category',
    'sentiment_polarity', 'sentiment_intensity', 'comment_quality_flag',
]

# Operational/experience features — categorized
SCORECARD_FEATURES = [c for c in df.columns if c.startswith('sc_')]
INFLUX_FEATURES = [c for c in df.columns if c.startswith('cis_')]
USAGE_FEATURES = [c for c in df.columns if c.startswith('ul1_')]
TICKET_FEATURES = [c for c in df.columns if c.startswith('tk_')]

# Derived features from enrichment
DERIVED_FEATURES = [
    'speed_gap_severity', 'connection_instability', 'customer_uptime_tier',
    'peak_uptime_gap', 'ticket_severity', 'tk_sla_compliance',
]

# Pre-existing operational features
PRE_EXISTING_OPS = [
    'avg_uptime_pct', 'stddev_uptime', 'min_uptime',
    'install_tat_hours', 'install_attempts',
    'avg_recharge_amount', 'avg_payment_amount',
    'sla_compliance_pct', 'avg_resolution_hours',
    'total_tickets', 'cx_tickets', 'px_tickets',
    'has_tickets', 'payment_mode',
]

# Partner/geo features
GEO_FEATURES = ['cluster', 'zone', 'mis_city', 'partner_status', 'city', 'City core']

rpt(f"\n  Feature counts:")
rpt(f"    Scorecard (speed/power): {len(SCORECARD_FEATURES)}")
rpt(f"    Influx (uptime): {len(INFLUX_FEATURES)}")
rpt(f"    Usage L1 (sessions): {len(USAGE_FEATURES)}")
rpt(f"    Ticket (enriched): {len(TICKET_FEATURES)}")
rpt(f"    Derived: {len([f for f in DERIVED_FEATURES if f in df.columns])}")


# ══════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("FEATURE PREPARATION")
rpt("=" * 70)

# Collect all valid features
all_features = set()
exclude_lower = set(x.lower() for x in EXCLUDE_FEATURES)

# Add experience/operational features
for feat_list in [SCORECARD_FEATURES, INFLUX_FEATURES, USAGE_FEATURES,
                  TICKET_FEATURES, DERIVED_FEATURES, PRE_EXISTING_OPS]:
    for f in feat_list:
        if f in df.columns and f.lower() not in exclude_lower:
            all_features.add(f)

# Filter to columns with >5% fill rate
valid_features = []
for f in sorted(all_features):
    if f in df.columns:
        # Skip NaN-indicator dummy features (leak missingness = churn)
        if any(pat in f for pat in EXCLUDE_PATTERNS):
            rpt(f"  Dropped (NaN indicator): {f}")
            continue
        fill = df[f].notna().sum() / len(df)
        if fill > 0.05:
            valid_features.append(f)
        else:
            rpt(f"  Dropped (low fill {fill*100:.1f}%): {f}")

rpt(f"\n  Excluded as leaky/self-fulfilling:")
for f in sorted(EXCLUDE_FEATURES):
    if f in df.columns:
        rpt(f"    - {f}")
rpt(f"\n  Valid experience features: {len(valid_features)}")

# Categorize features as numeric vs categorical
numeric_features = []
categorical_features = []

for f in valid_features:
    col = df[f]
    if col.dtype in ['object', 'category', 'bool']:
        nunique = col.nunique()
        if nunique <= 20:
            categorical_features.append(f)
        else:
            rpt(f"  Dropped (too many categories {nunique}): {f}")
    else:
        numeric_features.append(f)

rpt(f"  Numeric: {len(numeric_features)}")
rpt(f"  Categorical: {len(categorical_features)}")

# Prepare NPS features
nps_numeric = []
nps_categorical = []
for f in NPS_FEATURES:
    if f in df.columns:
        if df[f].dtype in ['object', 'category']:
            nps_categorical.append(f)
        else:
            nps_numeric.append(f)

# Prepare theme features
theme_actual = []
for f in THEME_CANDIDATES:
    for c in df.columns:
        if c.lower() == f.lower() and c not in theme_actual:
            theme_actual.append(c)

rpt(f"  NPS features: {nps_numeric + nps_categorical}")
rpt(f"  Theme features found: {theme_actual}")


# ══════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRICES FOR 3 MODEL TIERS
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("BUILDING 3-TIER FEATURE MATRICES")
rpt("=" * 70)

def build_feature_matrix(df, num_feats, cat_feats, label=""):
    """Build X matrix from numeric + one-hot-encoded categorical features."""
    frames = []

    # Numeric features
    if num_feats:
        num_df = df[num_feats].copy()
        for c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors='coerce')
        frames.append(num_df)

    # Categorical features (one-hot encode)
    if cat_feats:
        for c in cat_feats:
            if c in df.columns:
                dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
                # Remove NaN/None dummies (leak missingness = churn)
                nan_cols = [col for col in dummies.columns
                            if any(pat in col.lower() for pat in ['_nan', '_none', '_missing'])]
                dummies = dummies.drop(columns=nan_cols, errors='ignore')
                # Limit to top categories to avoid explosion
                if len(dummies.columns) > 10:
                    top_cols = dummies.sum().nlargest(10).index
                    dummies = dummies[top_cols]
                frames.append(dummies)

    if frames:
        X = pd.concat(frames, axis=1)
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imp
    return pd.DataFrame()


# Tier A: Experience/operational features ONLY
X_ops = build_feature_matrix(df, numeric_features, categorical_features, "ops")
rpt(f"  Tier A (Ops only): {X_ops.shape[1]} features")

# Tier B: Ops + NPS score/group
X_nps = build_feature_matrix(df,
    numeric_features + nps_numeric,
    categorical_features + nps_categorical, "ops+nps")
rpt(f"  Tier B (Ops + NPS): {X_nps.shape[1]} features")

# Tier C: Ops + NPS + themes
theme_num = [f for f in theme_actual if f in df.columns and df[f].dtype not in ['object', 'category']]
theme_cat = [f for f in theme_actual if f in df.columns and df[f].dtype in ['object', 'category']]
X_full = build_feature_matrix(df,
    numeric_features + nps_numeric + theme_num,
    categorical_features + nps_categorical + theme_cat, "ops+nps+themes")
rpt(f"  Tier C (Ops + NPS + Themes): {X_full.shape[1]} features")

y = df['churn_binary'].values

# Drop rows where target is NaN
valid_mask = ~pd.isna(y) & (df[churn_col].notna())
rpt(f"\n  Valid samples for modeling: {valid_mask.sum()}/{len(df)}")


# ══════════════════════════════════════════════════════════════════════
# MODEL TRAINING — 3-TIER COMPARISON
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("MODEL TRAINING — 5-FOLD CV, 3 TIERS")
rpt("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_eval(X, y, model_name, tier_label):
    """Train model with cross-validation, return AUC and feature importances."""
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask].astype(int)

    if len(X_clean) < 100:
        rpt(f"  {tier_label}: Too few samples ({len(X_clean)})")
        return None, None

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf_aucs = cross_val_score(rf, X_clean, y_clean, cv=cv, scoring='roc_auc')

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
    gb_aucs = cross_val_score(gb, X_clean, y_clean, cv=cv, scoring='roc_auc')

    # Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_aucs = cross_val_score(lr, X_scaled, y_clean, cv=cv, scoring='roc_auc')

    rpt(f"\n  {tier_label}:")
    rpt(f"    Random Forest AUC:     {rf_aucs.mean():.4f} (+/- {rf_aucs.std():.4f})")
    rpt(f"    Gradient Boosting AUC: {gb_aucs.mean():.4f} (+/- {gb_aucs.std():.4f})")
    rpt(f"    Logistic Regression:   {lr_aucs.mean():.4f} (+/- {lr_aucs.std():.4f})")

    # Best model for feature importance
    best_auc = max(rf_aucs.mean(), gb_aucs.mean(), lr_aucs.mean())
    best_model_name = 'RF' if rf_aucs.mean() == best_auc else 'GB' if gb_aucs.mean() == best_auc else 'LR'
    rpt(f"    Best: {best_model_name} (AUC {best_auc:.4f})")

    # Get feature importance from RF (fit on full data for importance)
    rf.fit(X_clean, y_clean)
    importances = pd.Series(rf.feature_importances_, index=X_clean.columns)
    importances = importances.sort_values(ascending=False)

    return {
        'rf_auc': rf_aucs.mean(), 'rf_std': rf_aucs.std(),
        'gb_auc': gb_aucs.mean(), 'gb_std': gb_aucs.std(),
        'lr_auc': lr_aucs.mean(), 'lr_std': lr_aucs.std(),
        'best_auc': best_auc, 'best_model': best_model_name,
        'n_features': X_clean.shape[1], 'n_samples': len(X_clean),
    }, importances


# Train all 3 tiers
results = {}
importances_dict = {}

for tier_label, X_tier in [
    ('Tier A: Experience Only', X_ops),
    ('Tier B: Experience + NPS', X_nps),
    ('Tier C: Experience + NPS + Themes', X_full),
]:
    res, imp = train_and_eval(X_tier, y, 'models', tier_label)
    results[tier_label] = res
    importances_dict[tier_label] = imp


# ══════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("3-TIER MODEL COMPARISON")
rpt("=" * 70)

rpt(f"\n  {'Tier':40s} | {'RF AUC':>10s} | {'GB AUC':>10s} | {'LR AUC':>10s} | {'Best':>10s} | {'Lift':>8s}")
rpt(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

base_auc = None
for tier_label, res in results.items():
    if res:
        if base_auc is None:
            base_auc = res['best_auc']
            lift = "baseline"
        else:
            lift_val = (res['best_auc'] - base_auc) * 100
            lift = f"+{lift_val:.2f}pp" if lift_val >= 0 else f"{lift_val:.2f}pp"
        rpt(f"  {tier_label:40s} | {res['rf_auc']:10.4f} | {res['gb_auc']:10.4f} | {res['lr_auc']:10.4f} | {res['best_auc']:10.4f} | {lift:>8s}")


# ══════════════════════════════════════════════════════════════════════
# TOP FEATURE IMPORTANCES (from best tier)
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("TOP 30 FEATURE IMPORTANCES (per tier)")
rpt("=" * 70)

for tier_label, imp in importances_dict.items():
    if imp is not None:
        rpt(f"\n  {tier_label}:")
        rpt(f"  {'Rank':>4s}  {'Feature':45s}  {'Importance':>10s}  {'Cumul':>8s}")
        rpt(f"  {'-'*4}  {'-'*45}  {'-'*10}  {'-'*8}")
        cumul = 0
        for i, (feat, imp_val) in enumerate(imp.head(30).items()):
            cumul += imp_val
            rpt(f"  {i+1:4d}  {feat:45s}  {imp_val:10.4f}  {cumul:7.1%}")


# ══════════════════════════════════════════════════════════════════════
# TENURE-STRATIFIED ANALYSIS
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("TENURE-STRATIFIED CHURN MODEL")
rpt("=" * 70)

# Smart tenure column selection: use tenure_days if mostly populated,
# else fall back to coarse 'Tenure' column (1-2/3-6/6+)
use_coarse_tenure = True
if tenure_col and tenure_col in df.columns:
    df['tenure_val'] = pd.to_numeric(df[tenure_col], errors='coerce')
    n_valid = df['tenure_val'].notna().sum()
    rpt(f"  tenure_days populated: {n_valid}/{len(df)} ({n_valid/len(df)*100:.1f}%)")
    if n_valid > len(df) * 0.5:
        use_coarse_tenure = False

if not use_coarse_tenure:
    # Fine-grained tenure buckets
    tenure_buckets = [
        ('Onboarding (0-30d)', 0, 30),
        ('Early Life (31-90d)', 31, 90),
        ('Establishing (91-180d)', 91, 180),
        ('Mature (180d+)', 181, 9999),
    ]
    rpt(f"\n  Using fine-grained tenure_days for stratification:")
    rpt(f"  {'Tenure Bucket':25s} | {'N':>6s} | {'Churn%':>7s} | {'AUC':>8s} | {'Top 5 Features'}")
    rpt(f"  {'-'*25} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*60}")

    for bucket_name, low, high in tenure_buckets:
        mask = (df['tenure_val'] >= low) & (df['tenure_val'] <= high)
        subset = df[mask]
        if len(subset) < 100:
            rpt(f"  {bucket_name:25s} | {len(subset):6d} | N/A    | N/A      | Too few samples")
            continue
        X_sub = X_full.loc[mask]
        y_sub = subset['churn_binary'].values
        if y_sub.sum() < 10 or (len(y_sub) - y_sub.sum()) < 10:
            rpt(f"  {bucket_name:25s} | {len(subset):6d} | {subset['churn_binary'].mean()*100:6.1f}% | N/A      | Imbalanced classes")
            continue
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=15,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            cv_sub = StratifiedKFold(n_splits=min(5, max(2, int(y_sub.sum() / 5))),
                                      shuffle=True, random_state=42)
            aucs = cross_val_score(rf, X_sub, y_sub.astype(int), cv=cv_sub, scoring='roc_auc')
            rf.fit(X_sub, y_sub.astype(int))
            imp_sub = pd.Series(rf.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
            top5 = ', '.join([f"{f}({v:.3f})" for f, v in imp_sub.head(5).items()])
            rpt(f"  {bucket_name:25s} | {len(subset):6d} | {subset['churn_binary'].mean()*100:6.1f}% | {aucs.mean():8.4f} | {top5}")
        except Exception as e:
            rpt(f"  {bucket_name:25s} | {len(subset):6d} | {subset['churn_binary'].mean()*100:6.1f}% | ERROR    | {str(e)[:50]}")
else:
    # Use coarse tenure (1-2 / 3-6 / 6+ months)
    tenure_coarse = None
    for c in df.columns:
        if c.lower() == 'tenure' and df[c].dtype == 'object':
            tenure_coarse = c; break

    if tenure_coarse:
        rpt(f"\n  Using coarse tenure column: '{tenure_coarse}'")
        rpt(f"  Values: {df[tenure_coarse].value_counts().to_dict()}")
        rpt(f"\n  {'Tenure Bucket':25s} | {'N':>6s} | {'Churn%':>7s} | {'AUC':>8s} | {'Top 5 Features'}")
        rpt(f"  {'-'*25} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*60}")

        for bucket in sorted(df[tenure_coarse].dropna().unique()):
            mask = df[tenure_coarse] == bucket
            subset = df[mask]
            if len(subset) < 100:
                continue
            X_sub = X_full.loc[mask]
            y_sub = subset['churn_binary'].values
            if y_sub.sum() < 10 or (len(y_sub) - y_sub.sum()) < 10:
                rpt(f"  {str(bucket):25s} | {len(subset):6d} | {subset['churn_binary'].mean()*100:6.1f}% | N/A      | Imbalanced")
                continue
            try:
                rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=15,
                                             class_weight='balanced', random_state=42, n_jobs=-1)
                aucs = cross_val_score(rf, X_sub, y_sub.astype(int), cv=3, scoring='roc_auc')
                rf.fit(X_sub, y_sub.astype(int))
                imp_sub = pd.Series(rf.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
                top5 = ', '.join([f"{f}({v:.3f})" for f, v in imp_sub.head(5).items()])
                rpt(f"  {str(bucket):25s} | {len(subset):6d} | {subset['churn_binary'].mean()*100:6.1f}% | {aucs.mean():8.4f} | {top5}")
            except Exception as e:
                rpt(f"  {str(bucket):25s} | ERROR: {str(e)[:60]}")
    else:
        rpt("  No usable tenure column found — skipping stratification")


# ══════════════════════════════════════════════════════════════════════
# BUSINESS INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt("BUSINESS INSIGHTS SUMMARY")
rpt("=" * 70)

# Check which new features made it to top-20
if importances_dict.get('Tier C: Experience + NPS + Themes') is not None:
    imp_c = importances_dict['Tier C: Experience + NPS + Themes']
    top20 = imp_c.head(20)

    new_feature_prefixes = ['sc_', 'cis_', 'ul1_', 'tk_', 'speed_gap', 'connection_',
                            'customer_uptime', 'peak_uptime', 'ticket_severity', 'tk_sla']

    rpt("\n  NEW FEATURES IN TOP-20 CHURN PREDICTORS:")
    for feat, val in top20.items():
        is_new = any(feat.startswith(p) or feat.startswith(p) for p in new_feature_prefixes)
        marker = " *** NEW ***" if is_new else ""
        rpt(f"    {feat:45s}: {val:.4f}{marker}")

    rpt("\n  INTERPRETATION:")
    # Speed gap
    if any('speed_gap' in f for f in top20.index):
        rpt("  - Speed gap (plan vs actual) is a top predictor — customers notice when they don't get advertised speed")
    # Uptime
    if any('uptime' in f for f in top20.index):
        rpt("  - Customer-level uptime predicts churn — infrastructure issues directly drive attrition")
    # Tickets
    if any('tk_' in f for f in top20.index):
        rpt("  - Enriched ticket metrics (reopens, TAT, customer calls) predict churn beyond simple ticket count")
    # NPS
    if any(f == nps_score_col for f in top20.index):
        rpt("  - NPS score itself is a churn predictor — survey captures genuine satisfaction signal")

    # Tier comparison insight
    tier_a = results.get('Tier A: Experience Only', {})
    tier_b = results.get('Tier B: Experience + NPS', {})
    tier_c = results.get('Tier C: Experience + NPS + Themes', {})

    if tier_a and tier_b:
        lift_nps = (tier_b.get('best_auc', 0) - tier_a.get('best_auc', 0)) * 100
        rpt(f"\n  NPS LIFT: Adding NPS score gives {lift_nps:+.2f} percentage points AUC improvement")
        if lift_nps > 1:
            rpt("    → NPS captures ADDITIONAL churn signal beyond operational metrics")
        elif lift_nps > 0:
            rpt("    → NPS adds marginal value on top of operational metrics")
        else:
            rpt("    → NPS adds NO additional predictive power — ops metrics already capture churn risk")

    if tier_b and tier_c:
        lift_themes = (tier_c.get('best_auc', 0) - tier_b.get('best_auc', 0)) * 100
        rpt(f"  THEME LIFT: Adding comment themes gives {lift_themes:+.2f} percentage points AUC improvement")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 70)
rpt(f"PHASE 4B COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 70)

report_path = os.path.join(OUTPUT, "phase4b_churn_model_v2.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"\nReport saved: {report_path}")
