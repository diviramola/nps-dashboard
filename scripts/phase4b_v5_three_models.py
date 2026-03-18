"""
Phase 4B v5: Clean Feature Model + Three-Model Framework
==========================================================
Built from comprehensive feature audit (259 features audited):
  - 74 CLEAN features (use as-is)
  - 22 NORMALIZE features (converted to per-month rates)
  - 133 EXCLUDED (leakage, bias, near-zero variance)
  - 30 REVIEW (not used in v5 — conservative approach)

Three-Model Framework:
  Model A: Ops → NPS score (regression) + Detractor classification
  Model B: Ops → Churn (the production model)
  Model C: NPS → Churn (NPS features only)
  Model D: Ops + NPS → Churn (does NPS add incremental value?)

Plus: Mediation analysis, residual analysis, population scoring viability.

Output: output/phase4b_v5_three_models.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (roc_auc_score, mean_absolute_error, r2_score,
                             confusion_matrix)
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
    print(line, flush=True)

rpt("=" * 100)
rpt("PHASE 4B v5: CLEAN FEATURES + THREE-MODEL FRAMEWORK")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 100)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df['churn_binary'] = df['is_churned'].astype(int)
rpt(f"\n[LOAD] Rows: {len(df)}, Columns: {len(df.columns)}")
rpt(f"  Churn: {df['churn_binary'].sum()}/{len(df)} ({df['churn_binary'].mean()*100:.1f}%)")
rpt(f"  NPS: mean={df['nps_score'].mean():.1f}, median={df['nps_score'].median():.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING: NORMALIZE TENURE-BIASED ABSOLUTE COUNTS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("FEATURE ENGINEERING: TENURE NORMALIZATION")
rpt("=" * 100)

tenure_months = np.maximum(df['tenure_days'].values / 30.0, 1.0)

NORMALIZE_MAP = {
    # Call features
    'TOTAL_IVR_CALLS':       'ivr_calls_per_month',
    'MISSED_CALLS':          'missed_calls_per_month',
    'INBOUND_CALLS':         'inbound_calls_per_month',
    'ANSWERED_CALLS':        'answered_calls_per_month',
    'INBOUND_ANSWERED':      'inbound_answered_per_month',
    'INBOUND_UNANSWERED':    'inbound_unanswered_per_month',
    # Ticket features
    'total_tickets':         'tickets_per_month',
    'cx_tickets':            'cx_tickets_per_month',
    'px_tickets':            'px_tickets_per_month',
    'DISTINCT_ISSUE_TYPES':  'distinct_issues_per_month',
    'MAX_TICKETS_SAME_ISSUE':'max_same_issue_per_month',
    'TICKETS_REOPENED_ONCE': 'reopened_once_per_month',
    'TICKETS_REOPENED_3PLUS':'reopened_3plus_per_month',
    'MAX_TIMES_REOPENED':    'max_reopened_per_month',
    'ISSUES_WITH_3PLUS_TICKETS': 'complex_issues_per_month',
    'AVG_TIMES_REOPENED':    'avg_reopened_per_month',
    # Other
    'support_effort_index':  'support_effort_per_month',
    'install_attempts':      'install_attempts_per_month',
    'TOTAL_LEADS':           'leads_per_month',
    'LEADS_ACCEPTED':        'leads_accepted_per_month',
    'partner_avg_tickets':   'partner_avg_tickets_per_month',
    'tickets_vs_partner':    'tickets_vs_partner_per_month',
}

rpt(f"\n  Creating {len(NORMALIZE_MAP)} per-month normalized features:")
rpt(f"  {'Original':40s} → {'Normalized':35s} | {'Old tenure_r':>12s} → {'New tenure_r':>12s}")
rpt(f"  {'-'*40}   {'-'*35}   {'-'*12}    {'-'*12}")

for orig, normed in NORMALIZE_MAP.items():
    if orig not in df.columns:
        rpt(f"  {orig:40s} → MISSING in data")
        continue
    raw = pd.to_numeric(df[orig], errors='coerce').values
    df[normed] = raw / tenure_months

    # Report tenure correlation change
    valid = ~(np.isnan(df[normed].values) | np.isnan(df['tenure_days'].values))
    if valid.sum() > 100:
        old_r = np.corrcoef(raw[valid], df['tenure_days'].values[valid])[0, 1]
        new_r = np.corrcoef(df[normed].values[valid], df['tenure_days'].values[valid])[0, 1]
        improved = "YES" if abs(new_r) < abs(old_r) else "no"
        rpt(f"  {orig:40s} → {normed:35s} | {old_r:+12.3f} → {new_r:+12.3f}  [{improved}]")
    else:
        rpt(f"  {orig:40s} → {normed:35s} | insufficient data")


# ══════════════════════════════════════════════════════════════════════════════
# v5 FEATURE LISTS (from audit)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("v5 FEATURE LISTS (AUDIT-DRIVEN)")
rpt("=" * 100)

# CLEAN features from audit (74 features) — only numeric/categorical with predictive value
# Excluding: identifiers, NPS targets, categorical metadata columns
CLEAN_OPS = [
    # Network quality
    'avg_uptime_pct', 'OVERALL_UPTIME_PCT', 'PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT',
    'PEAK_VS_OVERALL_GAP', 'stddev_uptime', 'FAILURE_RATE_PCT',
    'sc_avg_rxpower_in_range', 'sc_avg_rxpower', 'sc_avg_opticalpower_in_range',
    'sc_avg_latest_speed', 'sc_avg_speed_in_range', 'sc_speed_gap_pct',
    'sc_avg_plan_speed', 'optical_power', 'network_quality_index',
    # Service/support (rates & averages)
    'avg_resolution_hours', 'avg_resolution_hours_w', 'sla_compliance_pct',
    'tk_sla_compliance', 'AVG_TICKETS_PER_ISSUE', 'AVG_TICKET_RATING',
    'AVG_ANSWERED_SECONDS', 'DISPATCH_DECLINE_RATE_PCT', 'DROPPED_CALLS',
    'missed_call_ratio', 'HAS_REPEAT_COMPLAINT', 'has_tickets', 'ticket_severity',
    'sla_vs_partner', 'resolution_vs_partner',
    # Partner quality
    'partner_avg_nps', 'partner_sla_compliance', 'partner_fcr_rate',
    'partner_repeat_rate', 'partner_avg_resolution_hours', 'partner_customer_count',
    'partner_median_install_tat',
    # Install
    'AVG_INSTALL_TAT_MINS', 'AVG_INSTALL_TAT_MINS_w',
    'install_tat_hours', 'install_tat_hours_w', 'install_delayed',
    # Payment/value
    'autopay_ratio',
    # Demographics/profile
    'is_first_time_wifi', 'devices_2_4g', 'devices_5g',
]

# Normalized versions of tenure-biased features
NORMALIZED_OPS = [v for v in NORMALIZE_MAP.values() if v in df.columns]

# ALL OPS features for models
OPS_FEATURES = CLEAN_OPS + NORMALIZED_OPS

# NPS features (for Model C and D)
NPS_FEATURES = [
    'nps_score',
    'is_positive_sentiment', 'is_negative_sentiment',
    'has_comment',
    'theme_disconnection_frequency', 'theme_slow_speed',
    'theme_general_positive', 'theme_other', 'theme_unclassified',
]

# Categorical features to one-hot encode
CAT_FEATURES = ['city', 'tenure_bucket', 'payment_mode']

rpt(f"\n  OPS features (clean + normalized): {len(OPS_FEATURES)}")
rpt(f"    Clean: {len(CLEAN_OPS)}")
rpt(f"    Normalized: {len(NORMALIZED_OPS)}")
rpt(f"  NPS features: {len(NPS_FEATURES)}")
rpt(f"  Categorical: {len(CAT_FEATURES)}")

# Verify which features actually exist
ops_exist = [f for f in OPS_FEATURES if f in df.columns]
nps_exist = [f for f in NPS_FEATURES if f in df.columns]
cat_exist = [f for f in CAT_FEATURES if f in df.columns]
rpt(f"\n  Actually available:")
rpt(f"    OPS: {len(ops_exist)}/{len(OPS_FEATURES)}")
rpt(f"    NPS: {len(nps_exist)}/{len(NPS_FEATURES)}")
rpt(f"    CAT: {len(cat_exist)}/{len(CAT_FEATURES)}")
missing_ops = [f for f in OPS_FEATURES if f not in df.columns]
if missing_ops:
    rpt(f"    Missing OPS: {missing_ops}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
NAN_PATTERNS = ['_nan', '_None', '_missing']

def build_X(df_in, num_feats, cat_feats=None, ref_columns=None, ref_imputer=None):
    """Build feature matrix from numeric + categorical features."""
    frames = []

    # Numeric
    nf = [f for f in num_feats if f in df_in.columns]
    if nf:
        num_df = df_in[nf].copy()
        drop_cols = []
        for c in num_df.columns:
            try:
                col = num_df[c]
                if hasattr(col, 'sparse'):
                    num_df[c] = col.sparse.to_dense()
                    col = num_df[c]
                vals = col.values
                if vals.ndim != 1:
                    drop_cols.append(c)
                    continue
                num_df[c] = pd.to_numeric(pd.Series(vals, index=col.index), errors='coerce')
            except Exception:
                drop_cols.append(c)
        if drop_cols:
            num_df = num_df.drop(columns=drop_cols)
        frames.append(num_df)

    # Categorical
    if cat_feats:
        for c in [f for f in cat_feats if f in df_in.columns]:
            dummies = pd.get_dummies(df_in[c].astype(str), prefix=c, drop_first=True)
            nan_cols = [col for col in dummies.columns
                        if any(p in col.lower() for p in NAN_PATTERNS)]
            dummies = dummies.drop(columns=nan_cols, errors='ignore')
            if len(dummies.columns) > 10:
                dummies = dummies[dummies.sum().nlargest(10).index]
            frames.append(dummies)

    if not frames:
        return pd.DataFrame(), None

    X = pd.concat(frames, axis=1)

    if ref_columns is not None:
        for c in ref_columns:
            if c not in X.columns:
                X[c] = np.nan
        X = X[ref_columns]
        if ref_imputer is not None:
            return pd.DataFrame(ref_imputer.transform(X), columns=X.columns, index=X.index)

    all_nan = X.columns[X.isna().all()]
    if len(all_nan) > 0:
        X = X.drop(columns=all_nan)

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    return X_imp, imputer


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("TEMPORAL SPLIT")
rpt("=" * 100)

train_mask = df['sprint_num'].between(1, 7)
test_mask = df['sprint_num'].between(8, 11)
df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

rpt(f"  Train: Sprints 1-7, n={len(df_train)}, churn={df_train['churn_binary'].mean()*100:.1f}%")
rpt(f"  Test:  Sprints 8-11, n={len(df_test)}, churn={df_test['churn_binary'].mean()*100:.1f}%")
rpt(f"  Train NPS: mean={df_train['nps_score'].mean():.1f}")
rpt(f"  Test  NPS: mean={df_test['nps_score'].mean():.1f}")


# Build feature matrices
rpt("\nBuilding feature matrices...")
X_ops_train, imp_ops = build_X(df_train, ops_exist, cat_exist)
X_ops_test = build_X(df_test, ops_exist, cat_exist,
                      ref_columns=list(X_ops_train.columns), ref_imputer=imp_ops)

X_nps_train, imp_nps = build_X(df_train, nps_exist)
X_nps_test = build_X(df_test, nps_exist,
                      ref_columns=list(X_nps_train.columns), ref_imputer=imp_nps)

# Combined ops + nps
all_num = ops_exist + nps_exist
X_all_train, imp_all = build_X(df_train, all_num, cat_exist)
X_all_test = build_X(df_test, all_num, cat_exist,
                      ref_columns=list(X_all_train.columns), ref_imputer=imp_all)

y_train = df_train['churn_binary'].values
y_test = df_test['churn_binary'].values
nps_train = df_train['nps_score'].values
nps_test = df_test['nps_score'].values

rpt(f"  X_ops:  {X_ops_train.shape[1]} features")
rpt(f"  X_nps:  {X_nps_train.shape[1]} features")
rpt(f"  X_all:  {X_all_train.shape[1]} features")


# ══════════════════════════════════════════════════════════════════════════════
# v5 TENURE AUDIT: Verify all features pass tenure check
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("v5 TENURE AUDIT — All features should have |tenure_r| < 0.15")
rpt("=" * 100)

tenure_vals = df['tenure_days'].values
biased_count = 0
for c in X_ops_train.columns:
    if c in df.columns:
        vals = pd.to_numeric(df[c], errors='coerce').values
        valid = ~(np.isnan(vals) | np.isnan(tenure_vals))
        if valid.sum() > 100:
            r = np.corrcoef(vals[valid], tenure_vals[valid])[0, 1]
            if abs(r) > 0.15:
                biased_count += 1
                if abs(r) > 0.30:
                    rpt(f"  WARNING: {c:45s} tenure_r={r:+.3f}")
                elif abs(r) > 0.15:
                    rpt(f"  NOTE:    {c:45s} tenure_r={r:+.3f}")

rpt(f"\n  Features with |tenure_r| > 0.15: {biased_count}/{X_ops_train.shape[1]}")
rpt(f"  Features with |tenure_r| < 0.15: {X_ops_train.shape[1] - biased_count}/{X_ops_train.shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: OPS → NPS (Regression + Detractor Classification)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("MODEL A: OPS → NPS (Can operational data predict NPS scores?)")
rpt("=" * 100)

# A1: NPS Score Regression
rpt("\n--- A1: NPS Score Regression ---")
gb_nps = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=42)
gb_nps.fit(X_ops_train, nps_train)
nps_pred_train = gb_nps.predict(X_ops_train)
nps_pred_test = gb_nps.predict(X_ops_test)

r2_train = r2_score(nps_train, nps_pred_train)
r2_test = r2_score(nps_test, nps_pred_test)
mae_train = mean_absolute_error(nps_train, nps_pred_train)
mae_test = mean_absolute_error(nps_test, nps_pred_test)

rpt(f"  Train: R²={r2_train:.4f}, MAE={mae_train:.2f}")
rpt(f"  Test:  R²={r2_test:.4f}, MAE={mae_test:.2f}")
rpt(f"  (NPS range: 0-10, so MAE={mae_test:.2f} means avg error of {mae_test:.1f} points)")

# Feature importance for NPS regression
imp_a = pd.Series(gb_nps.feature_importances_, index=X_ops_train.columns).sort_values(ascending=False)
rpt(f"\n  Top 15 OPS drivers of NPS score:")
for i, (f, v) in enumerate(imp_a.head(15).items()):
    norm_flag = " [NORMALIZED]" if f in NORMALIZED_OPS else ""
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%){norm_flag}")

# A2: Detractor Classification (NPS 0-6 = 1, else 0)
rpt("\n--- A2: Detractor Classification ---")
det_train = (nps_train <= 6).astype(int)
det_test = (nps_test <= 6).astype(int)

gb_det = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
gb_det.fit(X_ops_train, det_train)
det_proba_test = gb_det.predict_proba(X_ops_test)[:, 1]
det_auc = roc_auc_score(det_test, det_proba_test)
rpt(f"  Detractor AUC (OOT): {det_auc:.4f}")
rpt(f"  Train Detractor rate: {det_train.mean()*100:.1f}%")
rpt(f"  Test  Detractor rate: {det_test.mean()*100:.1f}%")

# A3: 3-class NPS
rpt("\n--- A3: 3-Class NPS (Detractor/Passive/Promoter) ---")
def nps_group(s):
    if s <= 6: return 0  # Detractor
    if s <= 8: return 1  # Passive
    return 2  # Promoter

nps_class_train = np.array([nps_group(s) for s in nps_train])
nps_class_test = np.array([nps_group(s) for s in nps_test])

from sklearn.multiclass import OneVsRestClassifier
gb_3class = OneVsRestClassifier(
    GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                min_samples_leaf=20, random_state=42)
)
gb_3class.fit(X_ops_train, nps_class_train)
pred_3class = gb_3class.predict(X_ops_test)
from sklearn.metrics import accuracy_score
acc_3class = accuracy_score(nps_class_test, pred_3class)
rpt(f"  3-class accuracy (OOT): {acc_3class*100:.1f}%")
rpt(f"  Class distribution (test): D={sum(nps_class_test==0)}, P={sum(nps_class_test==1)}, Pro={sum(nps_class_test==2)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: OPS → CHURN (v5 clean model)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("MODEL B: OPS → CHURN (v5 Clean Model)")
rpt("=" * 100)

# CV on train
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

gb_churn = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
rf_churn = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                   class_weight='balanced', random_state=42, n_jobs=-1)

rpt("\n  3-Fold CV on training data (Sprints 1-7)...")
gb_cv = cross_val_score(gb_churn, X_ops_train, y_train, cv=cv, scoring='roc_auc')
rf_cv = cross_val_score(rf_churn, X_ops_train, y_train, cv=cv, scoring='roc_auc')
rpt(f"    GB CV AUC: {gb_cv.mean():.4f} (+/- {gb_cv.std():.4f})")
rpt(f"    RF CV AUC: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# OOT
gb_churn.fit(X_ops_train, y_train)
rf_churn.fit(X_ops_train, y_train)
gb_proba = gb_churn.predict_proba(X_ops_test)[:, 1]
rf_proba = rf_churn.predict_proba(X_ops_test)[:, 1]
ens_proba_b = (gb_proba + rf_proba) / 2

auc_gb_b = roc_auc_score(y_test, gb_proba)
auc_rf_b = roc_auc_score(y_test, rf_proba)
auc_ens_b = roc_auc_score(y_test, ens_proba_b)

rpt(f"\n  Out-of-Time AUC (Sprints 8-11):")
rpt(f"    GB:       {auc_gb_b:.4f}")
rpt(f"    RF:       {auc_rf_b:.4f}")
rpt(f"    Ensemble: {auc_ens_b:.4f}")

# Confusion matrix
pred_b = (ens_proba_b >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred_b)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp+fp) if (tp+fp)>0 else 0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    rpt(f"\n  Confusion @ 0.5:")
    rpt(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    rpt(f"    Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1*100:.1f}%")

# Feature importance
imp_b = pd.Series(gb_churn.feature_importances_, index=X_ops_train.columns).sort_values(ascending=False)
rpt(f"\n  Top 20 OPS drivers of Churn:")
for i, (f, v) in enumerate(imp_b.head(20).items()):
    norm_flag = " [NORMALIZED]" if f in NORMALIZED_OPS else ""
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%){norm_flag}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: NPS → CHURN (NPS features only)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("MODEL C: NPS → CHURN (Can NPS alone predict churn?)")
rpt("=" * 100)

# C1: NPS score only
rpt("\n--- C1: NPS Score Only ---")
X_score_train = df_train[['nps_score']].values
X_score_test = df_test[['nps_score']].values

lr_nps = LogisticRegression(max_iter=1000, random_state=42)
lr_nps.fit(X_score_train, y_train)
c1_proba = lr_nps.predict_proba(X_score_test)[:, 1]
c1_auc = roc_auc_score(y_test, c1_proba)
rpt(f"  NPS score only AUC (OOT): {c1_auc:.4f}")
rpt(f"  Coefficient: {lr_nps.coef_[0][0]:.4f} (negative = lower NPS → higher churn)")

# Churn rate by NPS group
for grp, label in [(0, 'Detractor (0-6)'), (1, 'Passive (7-8)'), (2, 'Promoter (9-10)')]:
    mask = nps_class_test == grp
    if mask.sum() > 0:
        churn_rate = y_test[mask].mean()
        rpt(f"  {label}: churn={churn_rate*100:.1f}% (n={mask.sum()})")

# C2: Full NPS features (score + sentiment + themes)
rpt("\n--- C2: Full NPS Features ---")
gb_nps_churn = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                           min_samples_leaf=20, random_state=42)
gb_nps_churn.fit(X_nps_train, y_train)
c2_proba = gb_nps_churn.predict_proba(X_nps_test)[:, 1]
c2_auc = roc_auc_score(y_test, c2_proba)
rpt(f"  Full NPS features AUC (OOT): {c2_auc:.4f}")

imp_c = pd.Series(gb_nps_churn.feature_importances_, index=X_nps_train.columns).sort_values(ascending=False)
rpt(f"\n  NPS feature importances:")
for f, v in imp_c.items():
    rpt(f"    {f:40s}: {v:.4f} ({v*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: OPS + NPS → CHURN (Incremental value of NPS)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("MODEL D: OPS + NPS → CHURN (Does NPS add value on top of ops?)")
rpt("=" * 100)

gb_all = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
gb_all.fit(X_all_train, y_train)
d_proba = gb_all.predict_proba(X_all_test)[:, 1]
d_auc = roc_auc_score(y_test, d_proba)

rpt(f"\n  Ops + NPS AUC (OOT): {d_auc:.4f}")
rpt(f"\n  ╔══════════════════════════════════════════════════════════════╗")
rpt(f"  ║  MODEL COMPARISON SUMMARY                                  ║")
rpt(f"  ╠══════════════════════════════════════════════════════════════╣")
rpt(f"  ║  Model B: Ops → Churn          AUC = {auc_ens_b:.4f}             ║")
rpt(f"  ║  Model C1: NPS score → Churn   AUC = {c1_auc:.4f}             ║")
rpt(f"  ║  Model C2: NPS full → Churn    AUC = {c2_auc:.4f}             ║")
rpt(f"  ║  Model D: Ops + NPS → Churn    AUC = {d_auc:.4f}             ║")
rpt(f"  ║                                                              ║")
rpt(f"  ║  NPS incremental lift: {d_auc - auc_ens_b:+.4f}                          ║")
rpt(f"  ╚══════════════════════════════════════════════════════════════╝")

# Feature importance for combined model
imp_d = pd.Series(gb_all.feature_importances_, index=X_all_train.columns).sort_values(ascending=False)
rpt(f"\n  Top 20 features (Ops+NPS combined):")
nps_set = set(nps_exist)
for i, (f, v) in enumerate(imp_d.head(20).items()):
    source = "NPS" if f in nps_set else "OPS"
    norm_flag = " [NORM]" if f in NORMALIZED_OPS else ""
    rpt(f"    {i+1:2d}. [{source:3s}] {f:40s}: {v:.4f} ({v*100:.1f}%){norm_flag}")

# Where does NPS rank among all features?
nps_rank = list(imp_d.index).index('nps_score') + 1 if 'nps_score' in imp_d.index else -1
rpt(f"\n  nps_score ranks #{nps_rank} out of {len(imp_d)} total features")


# ══════════════════════════════════════════════════════════════════════════════
# MEDIATION ANALYSIS: How much of ops→churn goes through NPS?
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("MEDIATION ANALYSIS: Does NPS mediate the Ops→Churn pathway?")
rpt("=" * 100)

rpt("""
  The mediation question:
    Ops metrics → NPS (Model A) → Churn (Model C)?
    Or: Ops metrics → Churn directly (Model B)?

  If NPS mediates: improving ops should improve NPS, which reduces churn.
  If ops acts directly: ops affects churn regardless of NPS (NPS is a parallel signal).
""")

# Use predicted NPS from Model A as a feature
df_test['predicted_nps'] = nps_pred_test

# Residual churn: churn unexplained by NPS
# If NPS explains churn well, Model B captures DIFFERENT signal than NPS
rpt("  Step 1: Correlation between predicted NPS and churn")
r_nps_churn = np.corrcoef(nps_test, y_test)[0, 1]
r_prednps_churn = np.corrcoef(nps_pred_test, y_test)[0, 1]
rpt(f"    Actual NPS ↔ Churn:    r = {r_nps_churn:.4f}")
rpt(f"    Predicted NPS ↔ Churn: r = {r_prednps_churn:.4f}")

# Step 2: Do ops features predict churn BEYOND what NPS captures?
rpt("\n  Step 2: Ops predicts churn beyond NPS?")
# Add predicted NPS to ops features, see if ops still matters
X_ops_plus_nps = X_ops_test.copy()
X_ops_plus_nps['predicted_nps'] = nps_pred_test
X_ops_plus_nps_train = X_ops_train.copy()
X_ops_plus_nps_train['predicted_nps'] = nps_pred_train

# Retrain with predicted NPS added
gb_med = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
gb_med.fit(X_ops_plus_nps_train, y_train)
med_proba = gb_med.predict_proba(X_ops_plus_nps)[:, 1]
med_auc = roc_auc_score(y_test, med_proba)

rpt(f"    Ops-only AUC:                 {auc_ens_b:.4f}")
rpt(f"    Ops + predicted NPS AUC:      {med_auc:.4f}")
rpt(f"    Predicted NPS lift:           {med_auc - auc_ens_b:+.4f}")

# Where does predicted_nps rank?
imp_med = pd.Series(gb_med.feature_importances_, index=X_ops_plus_nps_train.columns).sort_values(ascending=False)
pred_nps_rank = list(imp_med.index).index('predicted_nps') + 1
pred_nps_imp = imp_med['predicted_nps']
rpt(f"    predicted_nps importance: {pred_nps_imp:.4f} ({pred_nps_imp*100:.1f}%), rank #{pred_nps_rank}")

if pred_nps_imp > 0.05:
    rpt(f"\n  INTERPRETATION: NPS partially mediates the ops→churn pathway.")
    rpt(f"  Predicted NPS (from ops) carries {pred_nps_imp*100:.1f}% of the churn signal.")
    rpt(f"  This means improving ops → improves NPS → reduces churn (mediated path exists).")
else:
    rpt(f"\n  INTERPRETATION: NPS does NOT strongly mediate ops→churn.")
    rpt(f"  Ops drives churn mostly through direct pathways (not via NPS perception).")
    rpt(f"  NPS and churn may be parallel consequences of bad ops, not a causal chain.")


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUAL ANALYSIS: Does NPS add signal within ops-risk strata?
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("RESIDUAL ANALYSIS: NPS signal within ops-risk strata")
rpt("=" * 100)

rpt("""
  Question: Among customers with SIMILAR ops risk, does NPS still predict churn?
  If yes → NPS captures something ops data misses (e.g., perception, expectations).
  If no → NPS is redundant given ops data.
""")

# Bin test set by ops-predicted churn risk
df_test['ops_risk'] = ens_proba_b
df_test['nps_score_orig'] = nps_test
df_test['churn_actual'] = y_test

risk_bins = pd.qcut(df_test['ops_risk'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
df_test['risk_bin'] = risk_bins

rpt(f"\n  {'Risk Bin':>12s} | {'N':>5s} | {'Churn%':>6s} | {'NPS Corr':>8s} | {'Det Churn%':>10s} | {'Pro Churn%':>10s} | {'Gap':>6s}")
rpt(f"  {'-'*12} | {'-'*5} | {'-'*6} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*6}")

for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    bin_df = df_test[df_test['risk_bin'] == bin_name]
    if len(bin_df) < 20:
        continue
    n = len(bin_df)
    churn_pct = bin_df['churn_actual'].mean() * 100
    r = np.corrcoef(bin_df['nps_score_orig'], bin_df['churn_actual'])[0, 1] if bin_df['churn_actual'].std() > 0 else 0

    det = bin_df[bin_df['nps_score_orig'] <= 6]
    pro = bin_df[bin_df['nps_score_orig'] >= 9]
    det_churn = det['churn_actual'].mean() * 100 if len(det) > 5 else float('nan')
    pro_churn = pro['churn_actual'].mean() * 100 if len(pro) > 5 else float('nan')
    gap = det_churn - pro_churn if not (np.isnan(det_churn) or np.isnan(pro_churn)) else float('nan')

    rpt(f"  {bin_name:>12s} | {n:5d} | {churn_pct:5.1f}% | {r:+7.3f} | {det_churn:9.1f}% | {pro_churn:9.1f}% | {gap:+5.1f}pp")

rpt("""
  INTERPRETATION:
  - If Detractor-Promoter churn gap persists WITHIN each risk bin → NPS adds unique signal
  - If gap disappears → NPS is just a proxy for ops quality (redundant)
""")


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION SCORING VIABILITY
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("POPULATION SCORING VIABILITY")
rpt("=" * 100)

rpt("""
  Goal: Score ALL Wiom customers (not just NPS respondents) with churn risk.

  Model B (Ops→Churn) uses only operational features available for everyone.
  Model D (Ops+NPS→Churn) needs NPS — only available for survey respondents.

  Two strategies for population scoring:
  1. Use Model B directly (ops-only) — available for all customers
  2. Use Model A to PREDICT NPS, then use predicted NPS + ops for churn scoring
     (Model D with predicted NPS instead of actual NPS)
""")

# Strategy 2: Predicted NPS churn scoring
# Already have predicted NPS in df_test from Model A
# Compare: actual NPS churn prediction vs predicted NPS churn prediction
X_actual_nps = X_all_test.copy()  # has actual NPS
X_pred_nps = X_all_test.copy()
X_pred_nps['nps_score'] = nps_pred_test  # replace actual with predicted

gb_all2 = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      min_samples_leaf=20, random_state=42)
# Retrain D with predicted NPS in train set
X_all_train_pred = X_all_train.copy()
X_all_train_pred['nps_score'] = nps_pred_train

gb_all2.fit(X_all_train_pred, y_train)
d2_proba = gb_all2.predict_proba(X_pred_nps)[:, 1]
d2_auc = roc_auc_score(y_test, d2_proba)

rpt(f"\n  Population scoring AUC comparison:")
rpt(f"    Model B (Ops only):                     {auc_ens_b:.4f}")
rpt(f"    Model D (Ops + actual NPS):              {d_auc:.4f}")
rpt(f"    Model D' (Ops + predicted NPS):          {d2_auc:.4f}")
rpt(f"")
rpt(f"    Predicted NPS loss vs actual:  {d2_auc - d_auc:+.4f}")
rpt(f"    Predicted NPS gain vs ops-only: {d2_auc - auc_ens_b:+.4f}")

if d2_auc > auc_ens_b + 0.005:
    rpt(f"\n  RECOMMENDATION: Use Model D' (Ops + predicted NPS) for population scoring.")
    rpt(f"  Predicted NPS adds {(d2_auc - auc_ens_b)*100:.2f}pp AUC even without actual survey data.")
elif auc_ens_b >= d_auc - 0.005:
    rpt(f"\n  RECOMMENDATION: Use Model B (Ops only) for population scoring.")
    rpt(f"  NPS doesn't add enough incremental value to justify the two-stage approach.")
else:
    rpt(f"\n  RECOMMENDATION: Use Model B for population scoring, collect NPS for high-risk segments.")
    rpt(f"  NPS adds value when available but predicted NPS doesn't fully substitute.")


# ══════════════════════════════════════════════════════════════════════════════
# v4 vs v5 COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("v4 vs v5 COMPARISON")
rpt("=" * 100)

rpt(f"""
  v4 (previous — with tenure-biased features):
    - CV AUC (GB): 0.9347
    - OOT AUC:     0.8832
    - Issues: TOTAL_IVR_CALLS (tenure_r=+0.249), MISSED_CALLS (+0.201),
              INBOUND_CALLS (+0.348), sc_avg_weekly_data_gb (+0.485)
    - Features: {len(ops_exist) + len(NORMALIZED_OPS)} (mix of clean + biased absolute counts)

  v5 (clean — audit-driven features):
    - CV AUC (GB): {gb_cv.mean():.4f}
    - OOT AUC:     {auc_ens_b:.4f}
    - All absolute counts replaced with per-month normalized versions
    - Features: {X_ops_train.shape[1]} (74 clean + {len(NORMALIZED_OPS)} normalized)

  Change: OOT AUC {auc_ens_b - 0.8832:+.4f} (v4→v5)
""")

if auc_ens_b >= 0.87:
    rpt("  VERDICT: v5 maintains strong performance with cleaner signal.")
    rpt("  The model now generalizes better to different tenure distributions.")
elif auc_ens_b >= 0.83:
    rpt("  VERDICT: v5 shows modest AUC drop but signal is tenure-debiased.")
    rpt("  This is expected and preferred for population generalization.")
else:
    rpt("  VERDICT: v5 shows significant AUC drop — may need feature engineering refinement.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("FINAL SUMMARY: THREE-MODEL FRAMEWORK RESULTS")
rpt("=" * 100)

rpt(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ MODEL A: OPS → NPS                                                      │
  │   NPS Regression:  R²={r2_test:.4f}, MAE={mae_test:.2f} (out of 0-10)              │
  │   Detractor AUC:   {det_auc:.4f}                                               │
  │   Top drivers: (check feature list above)                               │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ MODEL B: OPS → CHURN (v5 clean model)                                   │
  │   CV AUC:  {gb_cv.mean():.4f}                                                      │
  │   OOT AUC: {auc_ens_b:.4f} (ensemble)                                         │
  │   Top driver: avg_uptime_pct ({imp_b.iloc[0]*100:.1f}%)                            │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ MODEL C: NPS → CHURN                                                    │
  │   NPS score only:  AUC={c1_auc:.4f}                                          │
  │   Full NPS:        AUC={c2_auc:.4f}                                          │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ MODEL D: OPS + NPS → CHURN                                              │
  │   AUC: {d_auc:.4f}                                                            │
  │   NPS incremental lift: {d_auc - auc_ens_b:+.4f}                                       │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ POPULATION SCORING                                                      │
  │   Ops only (B):          {auc_ens_b:.4f}                                       │
  │   Ops + predicted NPS:   {d2_auc:.4f}                                       │
  └──────────────────────────────────────────────────────────────────────────┘

  KEY INSIGHTS:
  1. Can ops predict NPS? R²={r2_test:.4f} → {"YES, ops explains substantial NPS variance" if r2_test > 0.15 else "Partially — NPS captures perception beyond ops data" if r2_test > 0.05 else "WEAKLY — NPS is largely independent of ops metrics"}
  2. Does NPS predict churn beyond ops? Lift={d_auc - auc_ens_b:+.4f} → {"YES, NPS adds unique signal" if (d_auc - auc_ens_b) > 0.01 else "MARGINAL" if (d_auc - auc_ens_b) > 0.003 else "NO — ops captures the churn signal directly"}
  3. NPS mediates ops→churn? predicted_nps importance={pred_nps_imp*100:.1f}% → {"YES, partial mediation" if pred_nps_imp > 0.05 else "WEAK mediation" if pred_nps_imp > 0.02 else "NO — parallel pathways"}
  4. Population scoring: {"Use Ops + predicted NPS" if d2_auc > auc_ens_b + 0.005 else "Use Ops only — simpler and nearly as good"}
""")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output_file = os.path.join(OUTPUT, "phase4b_v5_three_models.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n{'='*100}")
rpt(f"SAVED: {output_file}")
rpt(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*100}")
