"""
Phase 4B v3 — Comparison: With vs Without OUTBOUND_CALLS
=========================================================
Tests whether removing the potentially biased outbound calls feature
changes model performance and which features rise to fill the gap.
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE_DIR, "data")

print("=" * 80)
print("OUTBOUND CALLS BIAS TEST")
print(f"Comparing: Full model vs Model WITHOUT call features")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ── Load ──
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df['churn_binary'] = df['is_churned'].astype(int)

# ── Same exclusions as v3 ──
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
}
EXCLUDE_SUBSTRINGS = ['churn_risk', 'risk_category', 'churn_label', 'partner_risk']
NAN_PATTERNS = ['_nan', '_None', '_missing']

# Additional exclusions for the "no outbound" variant
CALL_FEATURES = {
    'OUTBOUND_CALLS', 'INBOUND_CALLS', 'MISSED_CALLS', 'TOTAL_IVR_CALLS',
    'AVG_ANSWERED_SECONDS', 'missed_call_ratio',
}

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

def classify_features(df_in):
    safe = []
    for c in df_in.columns:
        if is_excluded(c):
            continue
        if df_in[c].notna().sum() / len(df_in) <= 0.05:
            continue
        if df_in[c].dtype == 'object' and df_in[c].nunique() > 30:
            continue
        active_fill = df_in.loc[df_in['churn_binary']==0, c].notna().mean()
        churn_fill = df_in.loc[df_in['churn_binary']==1, c].notna().mean()
        if abs(active_fill - churn_fill) * 100 < 10:
            safe.append(c)
    return safe

def build_matrix(df_in, feature_list, ref_columns=None, ref_imputer=None):
    """Build X matrix. If ref_columns/ref_imputer given, align to training schema."""
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
        if ref_columns is not None:
            # Align to training columns
            for mc in set(ref_columns) - set(X.columns):
                X[mc] = 0
            X = X[[c for c in ref_columns if c in X.columns]]
            # Fill any still-missing columns
            for mc in ref_columns:
                if mc not in X.columns:
                    X[mc] = 0
            X = X[ref_columns]
        if ref_imputer is not None:
            X_imp = pd.DataFrame(ref_imputer.transform(X), columns=X.columns, index=X.index)
        else:
            imputer = SimpleImputer(strategy='median')
            X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
            return X_imp, imputer, list(X.columns)
        return X_imp, ref_imputer, list(X.columns)
    return pd.DataFrame(), None, []


# ── Split: Train Sprints 1-9, Test Sprints 10-11 ──
train_mask = df['sprint_num'].between(1, 9)
test_mask = df['sprint_num'].between(10, 11)
df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

print(f"\n  Train: {len(df_train)} rows ({df_train['churn_binary'].mean()*100:.1f}% churn)")
print(f"  Test:  {len(df_test)} rows ({df_test['churn_binary'].mean()*100:.1f}% churn)")

# ── Classify features ──
safe_feats = classify_features(df_train)
safe_ops = [f for f in safe_feats if f not in nps_and_theme_cols and not is_excluded(f)]

# ── Variant A: WITH call features (original v3) ──
feats_with = safe_ops
# ── Variant B: WITHOUT call features ──
feats_without = [f for f in safe_ops if f not in CALL_FEATURES]

print(f"\n  Variant A (with calls):    {len(feats_with)} features")
print(f"  Variant B (without calls): {len(feats_without)} features")
print(f"  Removed: {set(feats_with) - set(feats_without)}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE BOTH VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

results = {}

for label, feat_list in [("WITH calls", feats_with), ("WITHOUT calls", feats_without)]:
    print(f"\n{'─'*70}")
    print(f"  VARIANT: {label}")
    print(f"{'─'*70}")

    # Build training matrix
    X_tr, imp_tr, train_cols = build_matrix(df_train, feat_list)
    y_tr = df_train['churn_binary'].values

    # Build test matrix aligned to training schema
    X_te, _, _ = build_matrix(df_test, feat_list, ref_columns=train_cols, ref_imputer=imp_tr)
    y_te = df_test['churn_binary'].values

    # 5-fold CV on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)

    gb_cv = cross_val_score(gb, X_tr, y_tr, cv=cv, scoring='roc_auc')
    rf_cv = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring='roc_auc')

    print(f"  CV AUC (GB): {gb_cv.mean():.4f} +/- {gb_cv.std():.4f}")
    print(f"  CV AUC (RF): {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")

    # Fit on full train, predict on test
    gb.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    gb_proba = gb.predict_proba(X_te)[:, 1]
    rf_proba = rf.predict_proba(X_te)[:, 1]
    ens_proba = (gb_proba + rf_proba) / 2

    auc_gb = roc_auc_score(y_te, gb_proba)
    auc_rf = roc_auc_score(y_te, rf_proba)
    auc_ens = roc_auc_score(y_te, ens_proba)

    print(f"\n  Out-of-time AUC (GB):       {auc_gb:.4f}")
    print(f"  Out-of-time AUC (RF):       {auc_rf:.4f}")
    print(f"  Out-of-time AUC (Ensemble): {auc_ens:.4f}")

    # Confusion at 0.5
    pred = (ens_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

    print(f"\n  Confusion (thresh=0.5): TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Accuracy: {acc*100:.1f}%, Precision: {prec*100:.1f}%, Recall: {rec*100:.1f}%, F1: {f1*100:.1f}%")

    # Feature importances (GB)
    gb_imp = pd.Series(gb.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print(f"\n  TOP 15 FEATURE IMPORTANCES (GB):")
    print(f"  {'Rank':>4s}  {'Feature':40s}  {'Importance':>10s}  {'Cumul':>7s}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*10}  {'-'*7}")
    cumul = 0
    for rank, (feat, imp) in enumerate(gb_imp.head(15).items()):
        cumul += imp
        print(f"  {rank+1:4d}  {feat:40s}  {imp:10.4f}  {cumul:6.1%}")

    results[label] = {
        'cv_gb': gb_cv.mean(), 'cv_rf': rf_cv.mean(),
        'oot_gb': auc_gb, 'oot_rf': auc_rf, 'oot_ens': auc_ens,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'top10': list(gb_imp.head(10).items()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 80)

a = results["WITH calls"]
b = results["WITHOUT calls"]

print(f"\n  {'Metric':30s} | {'WITH Calls':>12s} | {'WITHOUT Calls':>14s} | {'Difference':>12s}")
print(f"  {'-'*30} | {'-'*12} | {'-'*14} | {'-'*12}")
print(f"  {'CV AUC (GB)':30s} | {a['cv_gb']:12.4f} | {b['cv_gb']:14.4f} | {(b['cv_gb']-a['cv_gb'])*100:+11.2f}pp")
print(f"  {'CV AUC (RF)':30s} | {a['cv_rf']:12.4f} | {b['cv_rf']:14.4f} | {(b['cv_rf']-a['cv_rf'])*100:+11.2f}pp")
print(f"  {'Out-of-Time AUC (GB)':30s} | {a['oot_gb']:12.4f} | {b['oot_gb']:14.4f} | {(b['oot_gb']-a['oot_gb'])*100:+11.2f}pp")
print(f"  {'Out-of-Time AUC (RF)':30s} | {a['oot_rf']:12.4f} | {b['oot_rf']:14.4f} | {(b['oot_rf']-a['oot_rf'])*100:+11.2f}pp")
print(f"  {'Out-of-Time AUC (Ensemble)':30s} | {a['oot_ens']:12.4f} | {b['oot_ens']:14.4f} | {(b['oot_ens']-a['oot_ens'])*100:+11.2f}pp")
print(f"  {'Accuracy':30s} | {a['acc']*100:11.1f}% | {b['acc']*100:13.1f}% | {(b['acc']-a['acc'])*100:+11.2f}pp")
print(f"  {'Precision':30s} | {a['prec']*100:11.1f}% | {b['prec']*100:13.1f}% | {(b['prec']-a['prec'])*100:+11.2f}pp")
print(f"  {'Recall':30s} | {a['rec']*100:11.1f}% | {b['rec']*100:13.1f}% | {(b['rec']-a['rec'])*100:+11.2f}pp")
print(f"  {'F1':30s} | {a['f1']*100:11.1f}% | {b['f1']*100:13.1f}% | {(b['f1']-a['f1'])*100:+11.2f}pp")

print(f"\n  FEATURE RANKING COMPARISON (GB, top 10):")
print(f"  {'Rank':>4s} | {'WITH Calls':40s} | {'WITHOUT Calls':40s}")
print(f"  {'-'*4} | {'-'*40} | {'-'*40}")
for i in range(10):
    f_a, v_a = a['top10'][i]
    f_b, v_b = b['top10'][i]
    print(f"  {i+1:4d} | {f_a:30s} ({v_a:.4f}) | {f_b:30s} ({v_b:.4f})")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
oot_diff = (b['oot_ens'] - a['oot_ens']) * 100
if abs(oot_diff) < 1:
    print(f"\n  AUC difference: {oot_diff:+.2f}pp — NEGLIGIBLE.")
    print(f"  Removing call features has minimal impact on model performance.")
    print(f"  The genuine operational signals (uptime, SLA, resolution time)")
    print(f"  carry the predictive power regardless of call features.")
    print(f"\n  CONCLUSION: Outbound calls were providing signal that is")
    print(f"  already captured by other operational metrics. The model")
    print(f"  without calls is CLEANER and equally effective.")
elif oot_diff < -1:
    print(f"\n  AUC dropped by {abs(oot_diff):.2f}pp when removing call features.")
    print(f"  Call features do contain SOME unique predictive signal,")
    print(f"  but the question is whether it's causal or just a proxy")
    print(f"  for 'customer was flagged as at-risk'.")
else:
    print(f"\n  AUC INCREASED by {oot_diff:.2f}pp without call features!")
    print(f"  This suggests call features were adding noise or overfitting.")

print(f"\n{'='*80}")
print(f"COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
