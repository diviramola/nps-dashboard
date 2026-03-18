"""
Phase 4B v4: Churn Prediction Model — Clean & Comprehensive
=============================================================
CHANGES FROM v3:
1. REMOVE sprint_num (temporal artifact from right-censoring, not a predictive signal)
2. REMOVE OUTBOUND_CALLS only (biased — reactive to risk); keep inbound/response metrics
3. REMOVE tickets_post_sprint (temporal leakage — post-survey behavior)
4. ADD unexplored features: DISPATCH_DECLINE_RATE_PCT, DROPPED_CALLS,
   MAX_TICKETS_SAME_ISSUE, HAS_REPEAT_COMPLAINT, TICKETS_REOPENED_ONCE,
   TICKETS_REOPENED_3PLUS, MAX_TIMES_REOPENED
5. ADD BACK cis_* features via TWO strategies:
   - Strategy A: All sprints, cis features imputed only where data exists (complete-case)
   - Strategy B: Late sprints only (9-13) where cis gap < 2pp
6. KEEP FAILURE_RATE_PCT (experience quality metric, not longevity proxy)

Output: output/phase4b_churn_model_v4.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
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
rpt("PHASE 4B v4: CHURN PREDICTION MODEL -- CLEAN & COMPREHENSIVE")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 80)

# Load data
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
churn_col = 'is_churned'
df['churn_binary'] = df[churn_col].astype(int)
rpt(f"\n[LOAD] Rows: {len(df)}, Columns: {len(df.columns)}")
rpt(f"  Churn: {df['churn_binary'].sum()}/{len(df)} ({df['churn_binary'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# EXCLUSION LIST — Expanded from v3
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FEATURE EXCLUSION LIST (v4)")
rpt("=" * 80)

EXCLUDE_ALWAYS = {
    # -- TARGET & IDENTIFIERS --
    'churn_binary', churn_col, 'phone_number', 'response_id', 'user_id',
    'timestamp', 'profile_all_identities', 'alt_mobile', 'lng_nas',
    'device_id', 'device_id_mapped',

    # -- CHURN-DERIVED --
    'churn_risk_score', 'churn_risk_pct', 'churn_label', 'risk_category',
    'partner_churn_rate', 'partner_at_risk', 'partner_risk_level',

    # -- PAYMENT/RECHARGE (mechanically = churn definition) --
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

    # -- ACTIVITY STATUS (IS the churn definition) --
    'partner_status', 'partner_status_at_survey', 'call_status',

    # -- DATA-EXISTENCE COUNTS (proxy for active duration) --
    'sc_scorecard_weeks', 'cis_influx_data_days', 'ul1_usage_data_days',
    'sc_plan_active_ratio', 'cis_active_day_ratio', 'sc_scorecard_ticket_count',

    # -- OUTBOUND CALLS ONLY (v3 finding: biased by reverse causality) --
    # Keep inbound/answered/duration/dropped as legitimate CX signals
    'OUTBOUND_CALLS',
    # INBOUND_CALLS kept (legitimate inbound experience)
    # MISSED_CALLS kept (could indicate poor reachability)
    # TOTAL_IVR_CALLS kept (engagement signal)
    # AVG_ANSWERED_SECONDS kept (response time — key CX metric)
    # missed_call_ratio kept (service quality)

    # -- TEMPORAL LEAKAGE --
    'tickets_post_sprint',

    # -- NEW in v4: SPRINT_NUM (right-censoring artifact) --
    'sprint_num',
    # Sprint 1 has 34.7% churn, Sprint 13 has 0% — this is observation window
    # bias, not a predictive signal. Model should learn from ops metrics only.

    # -- SPRINT METADATA (not predictive features) --
    'Sprint ID', 'Sprint Start Date', 'Sprint End Date', 'Cycle ID',
    'snap_date',
}

EXCLUDE_SUBSTRINGS = [
    'churn_risk', 'risk_category', 'churn_label', 'partner_risk',
]

NAN_PATTERNS = ['_nan', '_None', '_missing']

def is_excluded(col_name):
    if col_name in EXCLUDE_ALWAYS:
        return True
    for sub in EXCLUDE_SUBSTRINGS:
        if sub in col_name.lower():
            return True
    return False

rpt(f"  Excluded features: {sum(1 for c in df.columns if is_excluded(c))}")
rpt(f"  Key v4 changes:")
rpt(f"    - REMOVED: sprint_num (right-censoring artifact, was #3 in v3 at 12.3%)")
rpt(f"    - REMOVED: OUTBOUND_CALLS only (reverse causality); kept inbound/response CX signals")
rpt(f"    - REMOVED: tickets_post_sprint (temporal leakage)")
rpt(f"    - KEPT: FAILURE_RATE_PCT (experience quality metric)")


# ══════════════════════════════════════════════════════════════════════════════
# FILL RATE AUDIT
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FILL RATE AUDIT")
rpt("=" * 80)

nps_and_theme_cols = {'nps_score', 'nps_group',
    'primary_theme', 'secondary_theme', 'NPS Reason - Primary', 'Primary Category',
    'sentiment_polarity', 'sentiment_intensity', 'comment_quality_flag'}

safe_features = []
risky_features = []
recoverable_features = []  # NEW: gap shrinks in late sprints

candidate_cols = [c for c in df.columns if not is_excluded(c)
                  and c not in nps_and_theme_cols
                  and df[c].notna().sum() / len(df) > 0.05]

for f in candidate_cols:
    if df[f].dtype == 'object' and df[f].nunique() > 30:
        continue
    active_fill = df.loc[df['churn_binary']==0, f].notna().mean()
    churn_fill = df.loc[df['churn_binary']==1, f].notna().mean()
    gap = (active_fill - churn_fill) * 100

    if abs(gap) >= 10:
        # Check if gap shrinks in late sprints (9-13)
        late = df[df['sprint_num'].between(9, 13)]
        if len(late) > 100 and f in late.columns:
            late_af = late.loc[late['churn_binary']==0, f].notna().mean()
            late_cf = late.loc[late['churn_binary']==1, f].notna().mean()
            late_gap = (late_af - late_cf) * 100
            if abs(late_gap) < 10:
                recoverable_features.append(f)
            else:
                risky_features.append(f)
        else:
            risky_features.append(f)
    else:
        safe_features.append(f)

rpt(f"  SAFE features (< 10pp gap):        {len(safe_features)}")
rpt(f"  RECOVERABLE features (gap shrinks): {len(recoverable_features)}")
rpt(f"  RISKY features (persistent gap):    {len(risky_features)}")
rpt(f"\n  Recoverable features (new in v4):")
for f in sorted(recoverable_features):
    af = df.loc[df['churn_binary']==0, f].notna().mean()*100
    cf = df.loc[df['churn_binary']==1, f].notna().mean()*100
    late = df[df['sprint_num'].between(9, 13)]
    late_af = late.loc[late['churn_binary']==0, f].notna().mean()*100
    late_cf = late.loc[late['churn_binary']==1, f].notna().mean()*100
    rpt(f"    {f:40s} Overall: {af:.0f}%/{cf:.0f}% ({af-cf:+.0f}pp) | Late: {late_af:.0f}%/{late_cf:.0f}% ({late_af-late_cf:+.0f}pp)")


# ══════════════════════════════════════════════════════════════════════════════
# NEW FEATURES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("NEW FEATURE CANDIDATES (v4 additions)")
rpt("=" * 80)

new_candidates = [
    'DISPATCH_DECLINE_RATE_PCT',
    'DROPPED_CALLS',
    'MAX_TICKETS_SAME_ISSUE',
    'HAS_REPEAT_COMPLAINT',
    'TICKETS_REOPENED_ONCE',
    'TICKETS_REOPENED_3PLUS',
    'MAX_TIMES_REOPENED',
    'DISTINCT_ISSUE_TYPES',
    'FCR_RATE',
    'INBOUND_UNANSWERED',
]

rpt(f"\n  {'Feature':35s} | {'Active':>10s} | {'Churned':>10s} | {'Diff%':>8s} | {'Fill':>6s}")
rpt(f"  {'-'*35} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*6}")
new_features_used = []
for f in new_candidates:
    if f not in df.columns:
        rpt(f"  {f:35s} | NOT FOUND")
        continue
    fill = df[f].notna().mean()*100
    active_fill = df.loc[df['churn_binary']==0, f].notna().mean()*100
    churn_fill = df.loc[df['churn_binary']==1, f].notna().mean()*100
    gap = active_fill - churn_fill

    active_mean = df.loc[df['churn_binary']==0, f].astype(float).mean()
    churn_mean = df.loc[df['churn_binary']==1, f].astype(float).mean()
    diff_pct = ((churn_mean - active_mean) / active_mean * 100) if active_mean != 0 else 0

    rpt(f"  {f:35s} | {active_mean:10.3f} | {churn_mean:10.3f} | {diff_pct:+7.0f}% | {fill:.0f}%")

    # Only add if fill gap is <10pp (SAFE) and has signal
    if abs(gap) < 10 and fill > 50:
        new_features_used.append(f)

rpt(f"\n  Features added to v4 model: {len(new_features_used)}")
for f in new_features_used:
    rpt(f"    + {f}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FEATURE MATRIX CONSTRUCTION")
rpt("=" * 80)

def build_matrix(df_in, feature_list, label="", ref_columns=None, ref_imputer=None):
    """Build X matrix. If ref_columns/ref_imputer provided, align to training schema."""
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
        drop_cols = []
        for c in num_df.columns:
            try:
                col = num_df[c]
                # Handle sparse arrays
                if hasattr(col, 'sparse'):
                    num_df[c] = col.sparse.to_dense()
                    col = num_df[c]
                # Force to a flat numpy-backed series
                vals = col.values
                if vals.ndim != 1:
                    drop_cols.append(c)
                    continue
                num_df[c] = pd.to_numeric(pd.Series(vals, index=col.index), errors='coerce')
            except Exception as e:
                # Last resort: drop the column
                rpt(f"    WARNING: dropping {c} (type={type(num_df[c].iloc[0])}, err={e})")
                drop_cols.append(c)
        if drop_cols:
            num_df = num_df.drop(columns=drop_cols)
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

        if ref_columns is not None:
            # Align to reference columns (for test sets)
            for c in ref_columns:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[ref_columns]
            if ref_imputer is not None:
                X_imp = pd.DataFrame(ref_imputer.transform(X), columns=X.columns, index=X.index)
                return X_imp

        # Drop all-NaN columns before imputing (SimpleImputer drops them, causing shape mismatch)
        all_nan_cols = X.columns[X.isna().all()]
        if len(all_nan_cols) > 0:
            rpt(f"    Dropping {len(all_nan_cols)} all-NaN columns: {list(all_nan_cols[:5])}{'...' if len(all_nan_cols)>5 else ''}")
            X = X.drop(columns=all_nan_cols)
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imp, imputer
    return pd.DataFrame(), None


# ── TRACK 1: SAFE + NEW features (all sprints) ──
rpt("\n--- TRACK 1: SAFE + NEW FEATURES (all sprints, no fill-rate leakage) ---")
track1_feats = safe_features + new_features_used
track1_feats = [f for f in track1_feats if not is_excluded(f)]
X_t1, imp_t1 = build_matrix(df, track1_feats, "track1")
rpt(f"  Features: {X_t1.shape[1]}, Samples: {X_t1.shape[0]}")

# ── TRACK 2: SAFE + NEW + RECOVERABLE (late sprints 7-13 where gap is small) ──
rpt("\n--- TRACK 2: SAFE + RECOVERABLE on LATE SPRINTS (7-13) ---")
df_late = df[df['sprint_num'].between(7, 13)].copy()
track2_feats = safe_features + new_features_used + recoverable_features
track2_feats = [f for f in track2_feats if not is_excluded(f)]
X_t2, imp_t2 = build_matrix(df_late, track2_feats, "track2")
rpt(f"  Features: {X_t2.shape[1]}, Samples: {X_t2.shape[0]} (late sprints only)")
rpt(f"  Churn rate: {df_late['churn_binary'].mean()*100:.1f}%")

# ── TRACK 3: SAFE + NEW + RECOVERABLE on COMPLETE-CASE rows (all sprints) ──
rpt("\n--- TRACK 3: ALL FEATURES on COMPLETE-DATA subset ---")
# Key risky features that have persistent gaps (tk_*, ul1_*)
key_risky = [f for f in risky_features if f.startswith(('tk_'))
             and f in df.columns and not is_excluded(f)]
track3_feats = safe_features + new_features_used + recoverable_features + key_risky
track3_feats = [f for f in track3_feats if not is_excluded(f)]

# Complete-case: rows where tk_ data exists (proxy for engagement)
tk_check = [c for c in ['tk_total_tickets'] if c in df.columns]
if tk_check:
    tk_complete_mask = df[tk_check[0]].notna()
    df_complete = df[tk_complete_mask].copy()
else:
    df_complete = df.copy()

X_t3, imp_t3 = build_matrix(df_complete, track3_feats, "track3")
rpt(f"  Features: {X_t3.shape[1]}, Samples: {X_t3.shape[0]}")
rpt(f"  Churn rate: {df_complete['churn_binary'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING — 5-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL TRAINING -- 5-FOLD CV")
rpt("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate(X, y, label):
    """Train RF+GB+LR, return results and feature importances."""
    mask = ~np.isnan(y)
    X_c = X[mask]
    y_c = y[mask].astype(int)

    if len(X_c) < 100 or y_c.sum() < 20:
        rpt(f"  {label}: Insufficient data ({len(X_c)} rows, {y_c.sum()} churn)")
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

    best_auc = max(rf_aucs.mean(), gb_aucs.mean(), lr_aucs.mean())
    best_name = 'RF' if rf_aucs.mean() == best_auc else 'GB' if gb_aucs.mean() == best_auc else 'LR'

    rpt(f"\n  {label} (n={len(X_c)}, churn={y_c.sum()}/{len(y_c)} = {y_c.mean()*100:.1f}%):")
    rpt(f"    RF  AUC: {rf_aucs.mean():.4f} (+/- {rf_aucs.std():.4f})")
    rpt(f"    GB  AUC: {gb_aucs.mean():.4f} (+/- {gb_aucs.std():.4f})")
    rpt(f"    LR  AUC: {lr_aucs.mean():.4f} (+/- {lr_aucs.std():.4f})")
    rpt(f"    Best: {best_name} ({best_auc:.4f})")

    # Fit RF for feature importance
    rf.fit(X_c, y_c)
    gb.fit(X_c, y_c)
    imp = pd.Series(gb.feature_importances_, index=X_c.columns).sort_values(ascending=False)

    return {
        'rf_auc': rf_aucs.mean(), 'gb_auc': gb_aucs.mean(), 'lr_auc': lr_aucs.mean(),
        'best_auc': best_auc, 'best_model': best_name,
        'n_features': X_c.shape[1], 'n_samples': len(X_c),
    }, imp


# Train all tracks
y_all = df['churn_binary'].values
y_late = df_late['churn_binary'].values
y_complete = df_complete['churn_binary'].values

rpt("\n--- TRACK 1: Safe + New (all sprints, no sprint_num) ---")
t1_res, t1_imp = train_and_evaluate(X_t1, y_all, "Track 1")

rpt("\n--- TRACK 2: Safe + Recoverable (late sprints 7-13) ---")
t2_res, t2_imp = train_and_evaluate(X_t2, y_late, "Track 2")

rpt("\n--- TRACK 3: All features on complete-data subset ---")
t3_res, t3_imp = train_and_evaluate(X_t3, y_complete, "Track 3")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("TOP FEATURE IMPORTANCES (GB)")
rpt("=" * 80)

for label, imp in [("Track 1", t1_imp), ("Track 2", t2_imp), ("Track 3", t3_imp)]:
    if imp is not None:
        rpt(f"\n  {label} — Top 20:")
        for i, (f, v) in enumerate(imp.head(20).items()):
            new_flag = " [NEW in v4]" if f in new_features_used or f in recoverable_features else ""
            rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%){new_flag}")


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL HOLDOUT VALIDATION (Train: Sprints 1-7, Test: Sprints 8-11)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("TEMPORAL HOLDOUT VALIDATION")
rpt("(Train: Sprints 1-7, Test: Sprints 8-11, Exclude: 12-13 right-censored)")
rpt("=" * 80)

# Use sprint_num for splitting (even though excluded from features)
raw_sprint = df['sprint_num']
train_mask = raw_sprint.between(1, 7)
test_mask = raw_sprint.between(8, 11)

df_train_oot = df[train_mask].copy()
df_test_oot = df[test_mask].copy()

rpt(f"\n  Train: Sprints 1-7, n={len(df_train_oot)}, churn={df_train_oot['churn_binary'].mean()*100:.1f}%")
rpt(f"  Test:  Sprints 8-11, n={len(df_test_oot)}, churn={df_test_oot['churn_binary'].mean()*100:.1f}%")

# Build matrices
X_train_oot, imp_train = build_matrix(df_train_oot, track1_feats, "oot_train")
train_cols = list(X_train_oot.columns)

X_test_oot = build_matrix(df_test_oot, track1_feats, "oot_test",
                            ref_columns=train_cols, ref_imputer=imp_train)

y_train_oot = df_train_oot['churn_binary'].values
y_test_oot = df_test_oot['churn_binary'].values

# Train
gb_oot = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
rf_oot = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)

gb_oot.fit(X_train_oot, y_train_oot)
rf_oot.fit(X_train_oot, y_train_oot)

gb_proba = gb_oot.predict_proba(X_test_oot)[:, 1]
rf_proba = rf_oot.predict_proba(X_test_oot)[:, 1]
ens_proba = (gb_proba + rf_proba) / 2

auc_gb = roc_auc_score(y_test_oot, gb_proba)
auc_rf = roc_auc_score(y_test_oot, rf_proba)
auc_ens = roc_auc_score(y_test_oot, ens_proba)

rpt(f"\n  Out-of-Time AUC:")
rpt(f"    GB:       {auc_gb:.4f}")
rpt(f"    RF:       {auc_rf:.4f}")
rpt(f"    Ensemble: {auc_ens:.4f}")

# Confusion matrix at 0.5 threshold
pred_ens = (ens_proba >= 0.5).astype(int)
cm = confusion_matrix(y_test_oot, pred_ens)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    rpt(f"\n  Confusion Matrix (threshold=0.5):")
    rpt(f"    Predicted:    Active    Churn")
    rpt(f"    Actual Active: {tn:5d}    {fp:5d}")
    rpt(f"    Actual Churn:  {fn:5d}    {tp:5d}")
    rpt(f"\n    Accuracy:  {acc*100:.1f}%")
    rpt(f"    Precision: {prec*100:.1f}%")
    rpt(f"    Recall:    {rec*100:.1f}%")
    rpt(f"    F1:        {f1*100:.1f}%")

# Multiple thresholds
rpt(f"\n  Threshold Sensitivity:")
rpt(f"    {'Thresh':>7s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | {'Acc':>6s}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred_t = (ens_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test_oot, pred_t)
    if cm_t.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) > 0 else 0
        a_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
        rpt(f"    {t:7.1f} | {p_t*100:5.1f}% | {r_t*100:5.1f}% | {f_t*100:5.1f}% | {a_t*100:5.1f}%")


# OOT Feature importance
rpt(f"\n  Out-of-Time Feature Importances (GB, top 15):")
oot_imp = pd.Series(gb_oot.feature_importances_, index=train_cols).sort_values(ascending=False)
for i, (f, v) in enumerate(oot_imp.head(15).items()):
    new_flag = " [NEW]" if f in new_features_used else ""
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%){new_flag}")


# ══════════════════════════════════════════════════════════════════════════════
# v3 vs v4 COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("v3 vs v4 COMPARISON")
rpt("=" * 80)

v4_gb = t1_res['gb_auc'] if t1_res else 0
rpt(f"""
  v3 (leakage-free):
    - CV AUC (GB): 0.9339
    - Top features: avg_uptime_pct (31.5%), sc_avg_rxpower_in_range (14.6%), sprint_num (12.3%)
    - Included sprint_num (right-censoring artifact, 12.3% importance)
    - Included OUTBOUND_CALLS, all call features
    - OOT Ensemble AUC: 0.8634

  v4 (clean & comprehensive):
    - CV AUC (GB): {v4_gb:.4f}
    - Removed sprint_num (right-censoring artifact)
    - Removed OUTBOUND_CALLS only (reverse causality); kept inbound CX signals
    - Added: {len(new_features_used)} new features ({', '.join(new_features_used[:5])})
    - OOT Ensemble AUC: {auc_ens:.4f}

  KEY QUESTION: Did removing sprint_num + cleaning call features hurt or help?
  - v3 OOT AUC (with all calls, with sprint_num): 0.8634
  - v4 OOT AUC (no OUTBOUND, no sprint_num):      {auc_ens:.4f}
""")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
output_file = os.path.join(OUTPUT, "phase4b_churn_model_v4.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n{'='*80}")
rpt(f"SAVED: {output_file}")
rpt(f"COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*80}")
