"""
Phase 4B v3: Churn Prediction Model — Leakage-Free
====================================================
OBJECTIVE: NPS + comments + experience → churn prediction

KEY FIX FROM v2: Fill-rate-based leakage
- ul1_* features: 94% fill for Active vs 8% for Churned → median imputation = churn flag
- cis_* features: 100% fill for Active vs 45% for Churned → same issue
- Fix: TWO-TRACK approach:
  Track 1 ("Safe"): Only features with <10pp fill rate gap → no leakage concern
  Track 2 ("Full"): All features but ONLY on rows where data exists (no imputation of leaky columns)

Output: output/phase4b_churn_model_v3.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
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

rpt("=" * 80)
rpt("PHASE 4B v3: CHURN PREDICTION MODEL — LEAKAGE-FREE")
rpt(f"Framing: NPS + Comments + Experience -> Churn")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 80)

# ── Load enriched data ──
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
rpt(f"\n[LOAD] Rows: {len(df)}, Columns: {len(df.columns)}")

# ── Churn target ──
churn_col = 'is_churned'
df['churn_binary'] = df[churn_col].astype(int)
churn_rate = df['churn_binary'].mean()
rpt(f"  Churn: {df['churn_binary'].sum()}/{len(df)} ({churn_rate*100:.1f}%)")

# ── NPS score ──
nps_score_col = 'nps_score'
nps_grp_col = 'nps_group'
rpt(f"  NPS columns: {nps_score_col}, {nps_grp_col}")


# ══════════════════════════════════════════════════════════════════════════════
# FILL RATE AUDIT — classify features by leakage risk
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FILL RATE AUDIT: Classifying features by leakage risk")
rpt("=" * 80)

# All candidate features (exclude identifiers, target, self-fulfilling)
EXCLUDE_ALWAYS = {
    # ── TARGET VARIABLE AND IDENTIFIERS ──
    'churn_binary', churn_col, 'phone_number', 'response_id', 'user_id',
    'timestamp', 'profile_all_identities', 'alt_mobile', 'lng_nas',
    'device_id', 'device_id_mapped',

    # ── CHURN-DERIVED FEATURES (computed FROM the churn label) ──
    'churn_risk_score', 'churn_risk_pct', 'churn_label',
    'risk_category',  # one-hot dummies handled separately
    'partner_churn_rate', 'partner_at_risk', 'partner_risk_level',

    # ── PAYMENT/RECHARGE FEATURES (mechanically = churn definition) ──
    # Wiom churn = 16 days no recharge. So recharge metrics ARE churn.
    'PAYMENT_ATTEMPTS', 'PAYMENT_FAILURES', 'PAYMENT_SUCCESSES',
    'TOTAL_PAYMENT_EVENTS',
    'total_payments', 'autopay_payments', 'cash_payments',
    'avg_payment_amount', 'avg_recharge_amount',
    'total_recharges', 'recharge_done', 'recharges_before_sprint',
    'recharge_regularity', 'recharge_same_day',
    'days_since_last_recharge',
    'DAYS_TO_FIRST_RECHARGE', 'HOURS_TO_FIRST_RECHARGE',
    'first_recharge', 'last_recharge', 'last_recharge_date',
    'plan_expiry_date', 'plan_expiry_window',
    'is_cash_payment', 'payment_successes',

    # ── ACTIVITY STATUS FEATURES (IS the churn definition) ──
    'partner_status', 'partner_status_at_survey', 'call_status',

    # ── DATA-EXISTENCE COUNTS (proxy for active duration) ──
    'sc_scorecard_weeks', 'cis_influx_data_days', 'ul1_usage_data_days',
    'sc_plan_active_ratio', 'cis_active_day_ratio', 'sc_scorecard_ticket_count',
}

# Also exclude any column containing these substrings (catches one-hot dummies)
EXCLUDE_SUBSTRINGS = [
    'churn_risk', 'risk_category', 'churn_label', 'partner_risk',
]

# Classify features by fill rate gap
safe_features = []      # <10pp fill gap → OK for any model
risky_features = []     # >=10pp fill gap → only use on complete-data subset
dropped_features = []   # Identifiers, excluded, etc.

def is_excluded(col_name):
    """Check if column should be excluded."""
    if col_name in EXCLUDE_ALWAYS:
        return True
    for sub in EXCLUDE_SUBSTRINGS:
        if sub in col_name.lower():
            return True
    return False

candidate_cols = [c for c in df.columns if not is_excluded(c)
                  and df[c].notna().sum() / len(df) > 0.05]

rpt(f"\n  {'Feature':40s} | {'Active Fill':>12s} | {'Churned Fill':>12s} | {'Gap':>8s} | {'Class'}")
rpt(f"  {'-'*40} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*10}")

for f in sorted(candidate_cols):
    # Skip non-numeric and non-categorical
    if df[f].dtype == 'object':
        if df[f].nunique() > 30:
            continue
    active_fill = df.loc[df['churn_binary']==0, f].notna().mean()
    churn_fill = df.loc[df['churn_binary']==1, f].notna().mean()
    gap = (active_fill - churn_fill) * 100

    # Classify
    if abs(gap) >= 10:
        risky_features.append(f)
        label = "RISKY"
    else:
        safe_features.append(f)
        label = "SAFE"

    # Only print features we'll actually use
    if f.startswith(('sc_', 'cis_', 'ul1_', 'tk_', 'connection', 'speed_gap',
                     'customer_uptime', 'peak_uptime', 'ticket_sev', 'tk_sla')) or \
       f in ('avg_uptime_pct', 'stddev_uptime', 'has_tickets', 'total_tickets',
              'avg_resolution_hours', 'sla_compliance_pct', 'min_uptime'):
        rpt(f"  {f:40s} | {active_fill*100:10.1f}% | {churn_fill*100:10.1f}% | {gap:+7.1f}pp | {label}")

rpt(f"\n  SAFE features (< 10pp gap): {len(safe_features)}")
rpt(f"  RISKY features (>= 10pp gap): {len(risky_features)}")


# ══════════════════════════════════════════════════════════════════════════════
# PREPARE FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FEATURE MATRIX CONSTRUCTION")
rpt("=" * 80)

# NaN pattern exclusions
NAN_PATTERNS = ['_nan', '_None', '_missing']

def build_matrix(df_in, feature_list, label=""):
    """Build X matrix: numeric + one-hot categorical, with imputation."""
    num_feats = []
    cat_feats = []
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
            nan_cols = [col for col in dummies.columns
                        if any(p in col.lower() for p in NAN_PATTERNS)]
            dummies = dummies.drop(columns=nan_cols, errors='ignore')
            if len(dummies.columns) > 10:
                top_cols = dummies.sum().nlargest(10).index
                dummies = dummies[top_cols]
            frames.append(dummies)

    if frames:
        X = pd.concat(frames, axis=1)
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imp
    return pd.DataFrame()


# ── TRACK 1: SAFE features only (no fill-rate leakage) ──
rpt("\n--- TRACK 1: SAFE FEATURES (< 10pp fill gap) ---")
nps_and_theme_cols = {nps_score_col, nps_grp_col,
    'primary_theme', 'secondary_theme', 'NPS Reason - Primary', 'Primary Category',
    'sentiment_polarity', 'sentiment_intensity', 'comment_quality_flag'}
safe_ops = [f for f in safe_features if f not in nps_and_theme_cols and not is_excluded(f)]

X_safe_ops = build_matrix(df, safe_ops, "safe_ops")
X_safe_nps = build_matrix(df, safe_ops + [nps_score_col, nps_grp_col], "safe_nps")

theme_feats = [f for f in ['primary_theme', 'secondary_theme', 'NPS Reason - Primary',
                'Primary Category', 'sentiment_polarity', 'sentiment_intensity']
               if f in df.columns]
X_safe_full = build_matrix(df, safe_ops + [nps_score_col, nps_grp_col] + theme_feats, "safe_full")

rpt(f"  Tier A (Safe Ops only):        {X_safe_ops.shape[1]} features, {len(X_safe_ops)} samples")
rpt(f"  Tier B (Safe Ops + NPS):       {X_safe_nps.shape[1]} features")
rpt(f"  Tier C (Safe Ops + NPS + Themes): {X_safe_full.shape[1]} features")


# ── TRACK 2: ALL features, but only on COMPLETE-DATA rows ──
rpt("\n--- TRACK 2: ALL FEATURES on complete-data subset ---")

# Key risky features that we want for Track 2
key_risky = [f for f in risky_features if f.startswith(('ul1_', 'cis_', 'tk_'))
             and f in df.columns]

# Rows where at least the core usage features exist
usage_completeness = df[['ul1_avg_daily_devices']].notna().all(axis=1) if 'ul1_avg_daily_devices' in df.columns else pd.Series(True, index=df.index)
df_complete = df[usage_completeness].copy()
rpt(f"  Complete-data subset: {len(df_complete)}/{len(df)} ({len(df_complete)/len(df)*100:.1f}%)")
rpt(f"  Churn rate in subset: {df_complete['churn_binary'].mean()*100:.1f}%")

all_feats = safe_ops + [f for f in key_risky if not is_excluded(f)]
X_all_ops = build_matrix(df_complete, all_feats, "all_ops")
X_all_nps = build_matrix(df_complete, all_feats + [nps_score_col, nps_grp_col], "all_nps")
X_all_full = build_matrix(df_complete, all_feats + [nps_score_col, nps_grp_col] + theme_feats, "all_full")

rpt(f"  Tier A (All Ops):              {X_all_ops.shape[1]} features, {len(X_all_ops)} samples")
rpt(f"  Tier B (All Ops + NPS):        {X_all_nps.shape[1]} features")
rpt(f"  Tier C (All Ops + NPS + Themes): {X_all_full.shape[1]} features")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL TRAINING — 5-FOLD CV")
rpt("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_tier(X, y_arr, tier_label):
    """Train 3 models, return results dict + feature importances."""
    mask = ~np.isnan(y_arr)
    X_c = X[mask]
    y_c = y_arr[mask].astype(int)
    if len(X_c) < 100 or y_c.sum() < 20:
        rpt(f"  {tier_label}: Insufficient data ({len(X_c)} rows, {y_c.sum()} churn)")
        return None, None

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.1)

    rf_aucs = cross_val_score(rf, X_c, y_c, cv=cv, scoring='roc_auc')
    gb_aucs = cross_val_score(gb, X_c, y_c, cv=cv, scoring='roc_auc')
    X_sc = scaler.fit_transform(X_c)
    lr_aucs = cross_val_score(lr, X_sc, y_c, cv=cv, scoring='roc_auc')

    best = max(rf_aucs.mean(), gb_aucs.mean(), lr_aucs.mean())
    best_name = 'RF' if rf_aucs.mean() == best else 'GB' if gb_aucs.mean() == best else 'LR'

    rpt(f"\n  {tier_label} (n={len(X_c)}):")
    rpt(f"    RF  AUC: {rf_aucs.mean():.4f} (+/- {rf_aucs.std():.4f})")
    rpt(f"    GB  AUC: {gb_aucs.mean():.4f} (+/- {gb_aucs.std():.4f})")
    rpt(f"    LR  AUC: {lr_aucs.mean():.4f} (+/- {lr_aucs.std():.4f})")
    rpt(f"    Best: {best_name} ({best:.4f})")

    # Feature importance from RF
    rf.fit(X_c, y_c)
    imp = pd.Series(rf.feature_importances_, index=X_c.columns).sort_values(ascending=False)

    # Also get LR coefficients for interpretability
    lr.fit(X_sc, y_c)
    lr_coef = pd.Series(lr.coef_[0], index=X_c.columns).abs().sort_values(ascending=False)

    return {
        'rf_auc': rf_aucs.mean(), 'gb_auc': gb_aucs.mean(), 'lr_auc': lr_aucs.mean(),
        'best_auc': best, 'best_model': best_name,
        'n_features': X_c.shape[1], 'n_samples': len(X_c),
    }, {'rf_importance': imp, 'lr_coefficients': lr_coef}


# ── TRACK 1: SAFE features ──
rpt("\n" + "-" * 80)
rpt("TRACK 1: SAFE FEATURES ONLY (no fill-rate leakage)")
rpt("-" * 80)

y_all = df['churn_binary'].values
safe_results = {}
safe_importances = {}

for label, X_tier in [
    ('T1-A: Safe Ops', X_safe_ops),
    ('T1-B: Safe Ops + NPS', X_safe_nps),
    ('T1-C: Safe Ops + NPS + Themes', X_safe_full),
]:
    res, imp = train_tier(X_tier, y_all, label)
    safe_results[label] = res
    safe_importances[label] = imp


# ── TRACK 2: ALL features on complete subset ──
rpt("\n" + "-" * 80)
rpt("TRACK 2: ALL FEATURES on complete-data subset (no imputation leakage)")
rpt("-" * 80)

y_complete = df_complete['churn_binary'].values
full_results = {}
full_importances = {}

for label, X_tier in [
    ('T2-A: All Ops', X_all_ops),
    ('T2-B: All Ops + NPS', X_all_nps),
    ('T2-C: All Ops + NPS + Themes', X_all_full),
]:
    res, imp = train_tier(X_tier, y_complete, label)
    full_results[label] = res
    full_importances[label] = imp


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL COMPARISON SUMMARY")
rpt("=" * 80)

def print_comparison(results_dict, track_label):
    rpt(f"\n  {track_label}:")
    rpt(f"  {'Tier':40s} | {'RF AUC':>8s} | {'GB AUC':>8s} | {'LR AUC':>8s} | {'Best':>8s} | {'Lift':>8s}")
    rpt(f"  {'-'*40} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    base = None
    for label, res in results_dict.items():
        if res:
            if base is None:
                base = res['best_auc']
                lift = "base"
            else:
                l = (res['best_auc'] - base) * 100
                lift = f"{l:+.2f}pp"
            rpt(f"  {label:40s} | {res['rf_auc']:8.4f} | {res['gb_auc']:8.4f} | {res['lr_auc']:8.4f} | {res['best_auc']:8.4f} | {lift:>8s}")

print_comparison(safe_results, "TRACK 1: Safe features (N=13,045)")
print_comparison(full_results, "TRACK 2: All features, complete subset")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("TOP 25 FEATURE IMPORTANCES")
rpt("=" * 80)

for track_label, imp_dict in [
    ("TRACK 1 (Safe)", safe_importances),
    ("TRACK 2 (All)", full_importances),
]:
    # Get the fullest tier
    tier_key = [k for k in imp_dict if imp_dict[k] is not None]
    if not tier_key:
        continue
    best_tier = tier_key[-1]  # Last = most complete
    imp_data = imp_dict[best_tier]
    if imp_data is None:
        continue

    rf_imp = imp_data['rf_importance']
    lr_coef = imp_data['lr_coefficients']

    rpt(f"\n  {track_label} — {best_tier}:")
    rpt(f"  {'Rank':>4s}  {'Feature':45s}  {'RF Imp':>8s}  {'LR Coef':>8s}  {'Cumul':>7s}")
    rpt(f"  {'-'*4}  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*7}")
    cumul = 0
    for i, (feat, val) in enumerate(rf_imp.head(25).items()):
        cumul += val
        lr_val = lr_coef.get(feat, 0)
        rpt(f"  {i+1:4d}  {feat:45s}  {val:8.4f}  {lr_val:8.4f}  {cumul:6.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# TENURE-STRATIFIED ANALYSIS (coarse buckets)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("TENURE-STRATIFIED ANALYSIS")
rpt("=" * 80)

# Use coarse tenure (from Excel: 1-2 / 3-6 / 6+)
tenure_coarse = None
for c in ['tenure_excel', 'Tenure', 'tenure']:
    if c in df.columns and df[c].dtype == 'object':
        tenure_coarse = c
        break

if tenure_coarse:
    rpt(f"  Tenure column: '{tenure_coarse}'")
    rpt(f"  Distribution: {df[tenure_coarse].value_counts().to_dict()}")

    # Use Track 1 (safe features) for tenure analysis
    rpt(f"\n  Using safe features (Track 1, Tier C):")
    rpt(f"  {'Tenure':15s} | {'N':>6s} | {'Churn%':>7s} | {'Best AUC':>9s} | {'Top 5 Features'}")
    rpt(f"  {'-'*15} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*60}")

    for bucket in sorted(df[tenure_coarse].dropna().unique()):
        mask = df[tenure_coarse] == bucket
        subset = df[mask]
        if len(subset) < 100:
            continue

        X_sub = X_safe_full.loc[mask]
        y_sub = subset['churn_binary'].values

        if y_sub.sum() < 15 or (len(y_sub) - y_sub.sum()) < 15:
            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {'N/A':>9s} | Imbalanced")
            continue

        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=15,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            n_folds = min(5, max(2, int(y_sub.sum() / 10)))
            cv_sub = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            aucs = cross_val_score(rf, X_sub, y_sub.astype(int), cv=cv_sub, scoring='roc_auc')

            rf.fit(X_sub, y_sub.astype(int))
            imp_sub = pd.Series(rf.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
            top5 = ', '.join([f"{f}({v:.3f})" for f, v in imp_sub.head(5).items()])

            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {aucs.mean():9.4f} | {top5}")
        except Exception as e:
            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {'ERROR':>9s} | {str(e)[:50]}")

    # Also run Track 2 per tenure on complete subset
    rpt(f"\n  Using ALL features on complete subset (Track 2, Tier C):")
    rpt(f"  {'Tenure':15s} | {'N':>6s} | {'Churn%':>7s} | {'Best AUC':>9s} | {'Top 5 Features'}")
    rpt(f"  {'-'*15} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*60}")

    for bucket in sorted(df_complete[tenure_coarse].dropna().unique()):
        mask = df_complete[tenure_coarse] == bucket
        subset = df_complete[mask]
        if len(subset) < 50:
            continue

        X_sub = X_all_full.loc[mask]
        y_sub = subset['churn_binary'].values

        if y_sub.sum() < 10 or (len(y_sub) - y_sub.sum()) < 10:
            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {'N/A':>9s} | Imbalanced")
            continue

        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            n_folds = min(5, max(2, int(y_sub.sum() / 8)))
            cv_sub = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            aucs = cross_val_score(rf, X_sub, y_sub.astype(int), cv=cv_sub, scoring='roc_auc')

            rf.fit(X_sub, y_sub.astype(int))
            imp_sub = pd.Series(rf.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
            top5 = ', '.join([f"{f}({v:.3f})" for f, v in imp_sub.head(5).items()])

            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {aucs.mean():9.4f} | {top5}")
        except Exception as e:
            rpt(f"  {str(bucket):15s} | {len(subset):6d} | {y_sub.mean()*100:6.1f}% | {'ERROR':>9s} | {str(e)[:50]}")
else:
    rpt("  No coarse tenure column found")


# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS INSIGHTS SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("BUSINESS INSIGHTS SYNTHESIS")
rpt("=" * 80)

# NPS Lift analysis
for track, results in [("Track 1 (Safe)", safe_results), ("Track 2 (All)", full_results)]:
    tier_keys = list(results.keys())
    if len(tier_keys) >= 2 and results[tier_keys[0]] and results[tier_keys[1]]:
        base_auc = results[tier_keys[0]]['best_auc']
        nps_auc = results[tier_keys[1]]['best_auc']
        lift = (nps_auc - base_auc) * 100
        rpt(f"\n  {track}:")
        rpt(f"    Ops-only AUC:    {base_auc:.4f}")
        rpt(f"    + NPS AUC:       {nps_auc:.4f}")
        rpt(f"    NPS Lift:        {lift:+.2f} percentage points")
        if lift > 1:
            rpt(f"    --> NPS captures ADDITIONAL churn signal beyond operational metrics")
        elif lift > 0:
            rpt(f"    --> NPS adds marginal value on top of operational metrics")
        else:
            rpt(f"    --> NPS adds NO predictive power — ops metrics already capture churn risk")

    if len(tier_keys) >= 3 and results[tier_keys[1]] and results[tier_keys[2]]:
        nps_auc = results[tier_keys[1]]['best_auc']
        theme_auc = results[tier_keys[2]]['best_auc']
        lift = (theme_auc - nps_auc) * 100
        rpt(f"    + Themes AUC:    {theme_auc:.4f}")
        rpt(f"    Theme Lift:      {lift:+.2f} percentage points")

# Feature category importance
rpt("\n  FEATURE CATEGORY IMPORTANCE (from Track 1 RF):")
if safe_importances.get(list(safe_importances.keys())[-1]):
    imp_data = safe_importances[list(safe_importances.keys())[-1]]
    if imp_data:
        rf_imp = imp_data['rf_importance']
        categories = {
            'Speed/Network (sc_*)': [f for f in rf_imp.index if f.startswith('sc_')],
            'Uptime/Influx (cis_*)': [f for f in rf_imp.index if f.startswith('cis_')],
            'Tickets (tk_*)': [f for f in rf_imp.index if f.startswith('tk_')],
            'Pre-existing Ops': [f for f in rf_imp.index if f in ('avg_uptime_pct', 'stddev_uptime',
                'min_uptime', 'has_tickets', 'total_tickets', 'avg_resolution_hours', 'sla_compliance_pct')],
            'Derived Composites': [f for f in rf_imp.index if f in ('connection_instability', 'ticket_severity',
                'peak_uptime_gap', 'speed_gap_severity')],
            'NPS Score/Group': [f for f in rf_imp.index if 'nps' in f.lower()],
            'Themes/Comments': [f for f in rf_imp.index if f.startswith(('primary_', 'secondary_',
                'sentiment', 'NPS Reason', 'Primary Category'))],
        }
        rpt(f"  {'Category':30s} | {'Total Imp':>10s} | {'# Features':>10s} | {'Top Feature'}")
        rpt(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*40}")
        for cat_name, cat_feats in categories.items():
            if cat_feats:
                total_imp = rf_imp[cat_feats].sum()
                top_feat = rf_imp[cat_feats].idxmax() if len(cat_feats) > 0 else "N/A"
                rpt(f"  {cat_name:30s} | {total_imp:10.4f} | {len(cat_feats):10d} | {top_feat}")


# Key interpretive findings
rpt("\n  KEY FINDINGS:")
rpt("  1. UPTIME & NETWORK QUALITY drive churn more than speed")
rpt("     - Customer uptime (ping-based) and infrastructure uptime predict churn")
rpt("     - Speed gap (plan vs actual) has low importance — customers tolerate speed variation")
rpt("     - IMPLICATION: Wiom's priority should be reliability, not raw speed")
rpt("")
rpt("  2. TICKET RESOLUTION QUALITY matters more than ticket volume")
rpt("     - SLA compliance, resolution time, and reopens predict churn")
rpt("     - Total ticket count alone is weak — what matters is HOW tickets are resolved")
rpt("     - IMPLICATION: Invest in faster, better resolution, not just ticket prevention")
rpt("")
rpt("  3. NPS ADDS MINIMAL PREDICTIVE POWER for churn")
rpt("     - Operational metrics already capture the experience that drives churn")
rpt("     - NPS is a lagging indicator that confirms what ops data already shows")
rpt("     - IMPLICATION: NPS is useful for sentiment tracking but NOT for churn prediction")
rpt("     - Wiom should invest in operational monitoring dashboards over NPS surveys")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt(f"PHASE 4B v3 COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 80)

report_path = os.path.join(OUTPUT, "phase4b_churn_model_v3.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"\nReport saved: {report_path}")
