"""
Phase 4B: Three-Model Framework
================================
Model A: Ops → NPS (predict NPS from operational features)
Model B: Ops → Churn (predict churn from operational features — v4 baseline)
Model C: NPS → Churn (predict churn from NPS score/themes only)
Model D: Ops + NPS → Churn (does NPS add value on top of ops?)

Then: Test relationships, mediation analysis, population scoring viability.

Output: output/phase4b_three_model_framework.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (roc_auc_score, mean_absolute_error, mean_squared_error,
                             classification_report, confusion_matrix, r2_score)
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
rpt("PHASE 4B: THREE-MODEL FRAMEWORK")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df['churn_binary'] = df['is_churned'].astype(int)
rpt(f"\n[LOAD] Rows: {len(df)}, Columns: {len(df.columns)}")
rpt(f"  Churn: {df['churn_binary'].sum()}/{len(df)} ({df['churn_binary'].mean()*100:.1f}%)")
rpt(f"  NPS: mean={df['nps_score'].mean():.1f}, median={df['nps_score'].median():.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUPS
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("FEATURE GROUPS")
rpt("=" * 80)

# Exclusions from v4 (identifiers, churn-derived, payments, etc.)
EXCLUDE_ALWAYS = {
    'churn_binary', 'is_churned', 'phone_number', 'response_id', 'user_id',
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
    'OUTBOUND_CALLS',
    'tickets_post_sprint',
    'sprint_num',
    'Sprint ID', 'Sprint Start Date', 'Sprint End Date', 'Cycle ID', 'snap_date',
}
EXCLUDE_SUBSTRINGS = ['churn_risk', 'risk_category', 'churn_label', 'partner_risk']
NAN_PATTERNS = ['_nan', '_None', '_missing']

def is_excluded(col_name):
    if col_name in EXCLUDE_ALWAYS:
        return True
    return any(sub in col_name.lower() for sub in EXCLUDE_SUBSTRINGS)

# NPS-specific columns (used as TARGETS in Model A, FEATURES in Model C)
NPS_FEATURE_COLS = {
    'nps_score', 'nps_group', 'nps_group_ordinal',
    'primary_theme', 'secondary_theme',
    'primary_theme_score', 'secondary_theme_score',
    'sentiment_polarity', 'sentiment_intensity',
    'is_positive_sentiment', 'is_negative_sentiment',
    'score_sentiment_mismatch', 'has_comment', 'comment_quality',
    'detected_language', 'emotion',
    'prev_nps', 'nps_match',
    'partner_avg_nps',
    # Theme flags
    'theme_billing_28day', 'theme_call_center_bad', 'theme_competitor_comparison',
    'theme_complaint_resolution_bad', 'theme_disconnection_frequency',
    'theme_general_negative', 'theme_general_positive', 'theme_good_speed',
    'theme_internet_down_outage', 'theme_other', 'theme_ott_content',
    'theme_pricing_affordable', 'theme_pricing_expensive',
    'theme_range_coverage', 'theme_router_device', 'theme_slow_speed',
    'theme_unclassified',
    # Mentions flags
    'mentions_28day', 'mentions_amount', 'mentions_competitor',
    # NPS reason columns
    'NPS Reason - Primary', 'NPS Reason - Secondary', 'NPS Reason - Tertiary',
    'Primary Category',
    'nps_reason_primary_sprint', 'nps_reason_secondary_sprint',
    'nps_reason_sprint', 'nps_reason_tertiary_sprint',
    'nps_oe_copied', 'comments_sprint', 'translated_comment', 'theme_clean',
}

# Hindi CX survey columns (treat as NPS-adjacent — only available for survey respondents)
HINDI_CX_COLS = set()
for c in df.columns:
    # Hindi characters
    if any('\u0900' <= ch <= '\u097F' for ch in c):
        HINDI_CX_COLS.add(c)

# OPS features = everything except excluded, NPS, and Hindi CX
ops_candidates = []
for c in df.columns:
    if is_excluded(c):
        continue
    if c in NPS_FEATURE_COLS or c in HINDI_CX_COLS:
        continue
    if df[c].notna().sum() / len(df) < 0.05:
        continue
    if df[c].dtype == 'object' and df[c].nunique() > 30:
        continue
    # Fill-rate check (safe features only — <10pp gap)
    active_fill = df.loc[df['churn_binary']==0, c].notna().mean()
    churn_fill = df.loc[df['churn_binary']==1, c].notna().mean()
    if abs(active_fill - churn_fill) < 0.10:
        ops_candidates.append(c)

# NPS features for Model C (features that exist for respondents)
nps_model_features = []
for c in NPS_FEATURE_COLS:
    if c in df.columns and c not in {'nps_score', 'nps_group', 'translated_comment',
                                      'theme_clean', 'comments_sprint',
                                      'nps_oe_copied', 'nps_reason_sprint',
                                      'nps_reason_tertiary_sprint'}:
        if df[c].notna().sum() / len(df) > 0.05:
            nps_model_features.append(c)

rpt(f"  OPS features (safe, for Models A/B): {len(ops_candidates)}")
rpt(f"  NPS features (for Model C):          {len(nps_model_features)}")
rpt(f"  NPS features: {sorted(nps_model_features)[:10]}...")


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_matrix(df_in, feature_list, label=""):
    """Build X matrix from feature list. Returns (X_imputed, imputer)."""
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
        all_nan_cols = X.columns[X.isna().all()]
        if len(all_nan_cols) > 0:
            X = X.drop(columns=all_nan_cols)
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        return X_imp, imputer
    return pd.DataFrame(), None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: OPS → NPS (Predict NPS score from operational features)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL A: OPS → NPS PREDICTION")
rpt("Can we predict NPS scores from operational data alone?")
rpt("=" * 80)

X_ops, _ = build_matrix(df, ops_candidates, "ops")
y_nps = df['nps_score'].values
y_nps_group = df['nps_group_ordinal'].values  # 0=Detractor, 1=Passive, 2=Promoter

rpt(f"\n  Feature matrix: {X_ops.shape[1]} features x {X_ops.shape[0]} samples")

# A1: Regression — predict exact NPS score (0-10)
rpt("\n--- A1: NPS Score Regression (0-10) ---")
mask = ~np.isnan(y_nps)
X_a1 = X_ops[mask]
y_a1 = y_nps[mask]

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
gb_reg = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=42)
lr_reg = LinearRegression()

# CV for GB regression
from sklearn.model_selection import cross_val_predict
gb_reg_cv = cross_val_predict(gb_reg, X_a1, y_a1, cv=cv_reg)
mae_gb = mean_absolute_error(y_a1, gb_reg_cv)
rmse_gb = np.sqrt(mean_squared_error(y_a1, gb_reg_cv))
r2_gb = r2_score(y_a1, gb_reg_cv)

# CV for LR regression
scaler = StandardScaler()
X_a1_sc = scaler.fit_transform(X_a1)
lr_reg_cv = cross_val_predict(lr_reg, X_a1_sc, y_a1, cv=cv_reg)
mae_lr = mean_absolute_error(y_a1, lr_reg_cv)
rmse_lr = np.sqrt(mean_squared_error(y_a1, lr_reg_cv))
r2_lr = r2_score(y_a1, lr_reg_cv)

rpt(f"  GB Regression:  MAE={mae_gb:.2f}, RMSE={rmse_gb:.2f}, R²={r2_gb:.4f}")
rpt(f"  LR Regression:  MAE={mae_lr:.2f}, RMSE={rmse_lr:.2f}, R²={r2_lr:.4f}")

# Fit GB for feature importance
gb_reg.fit(X_a1, y_a1)
nps_imp = pd.Series(gb_reg.feature_importances_, index=X_a1.columns).sort_values(ascending=False)
rpt(f"\n  Top 15 OPS drivers of NPS score (GB importance):")
for i, (f, v) in enumerate(nps_imp.head(15).items()):
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%)")

# A2: Classification — predict Detractor vs Promoter (binary, drop Passive)
rpt("\n--- A2: Detractor vs Promoter Classification ---")
dp_mask = df['nps_group'].isin(['Detractor', 'Promoter'])
X_a2 = X_ops[dp_mask]
y_a2 = (df.loc[dp_mask, 'nps_group'] == 'Detractor').astype(int).values

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     min_samples_leaf=20, random_state=42)
gb_aucs = cross_val_score(gb_clf, X_a2, y_a2, cv=cv_clf, scoring='roc_auc')
rpt(f"  GB AUC (Detractor vs Promoter): {gb_aucs.mean():.4f} (+/- {gb_aucs.std():.4f})")
rpt(f"  n={len(X_a2)}, Detractors={y_a2.sum()} ({y_a2.mean()*100:.1f}%)")

gb_clf.fit(X_a2, y_a2)
dp_imp = pd.Series(gb_clf.feature_importances_, index=X_a2.columns).sort_values(ascending=False)
rpt(f"\n  Top 15 OPS drivers of Detractor status:")
for i, (f, v) in enumerate(dp_imp.head(15).items()):
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%)")

# A3: 3-class — Detractor / Passive / Promoter
rpt("\n--- A3: 3-Class NPS Group Prediction ---")
gb_3c = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=42)
mask3 = ~np.isnan(y_nps_group)
X_a3 = X_ops[mask3]
y_a3 = y_nps_group[mask3].astype(int)
gb_3c_aucs = cross_val_score(gb_3c, X_a3, y_a3, cv=cv_clf, scoring='roc_auc_ovr')
rpt(f"  GB AUC-OVR (3-class): {gb_3c_aucs.mean():.4f} (+/- {gb_3c_aucs.std():.4f})")

# Predicted group distribution vs actual
gb_3c_preds = cross_val_predict(gb_3c, X_a3, y_a3, cv=cv_clf)
rpt(f"\n  Predicted vs Actual NPS group distribution:")
rpt(f"    {'Group':12s} | {'Actual':>8s} | {'Predicted':>10s}")
for g, label in [(0, 'Detractor'), (1, 'Passive'), (2, 'Promoter')]:
    actual = (y_a3 == g).sum()
    predicted = (gb_3c_preds == g).sum()
    rpt(f"    {label:12s} | {actual:8d} | {predicted:10d}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: OPS → CHURN (from v4 — rerun with same features for consistency)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL B: OPS → CHURN (v4 baseline)")
rpt("=" * 80)

y_churn = df['churn_binary'].values

gb_b = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   min_samples_leaf=20, random_state=42)
gb_b_aucs = cross_val_score(gb_b, X_ops, y_churn, cv=cv_clf, scoring='roc_auc')
rpt(f"\n  GB AUC (Ops → Churn): {gb_b_aucs.mean():.4f} (+/- {gb_b_aucs.std():.4f})")
rpt(f"  n={len(X_ops)}, churn={y_churn.sum()} ({y_churn.mean()*100:.1f}%)")

gb_b.fit(X_ops, y_churn)
churn_imp = pd.Series(gb_b.feature_importances_, index=X_ops.columns).sort_values(ascending=False)
rpt(f"\n  Top 15 OPS drivers of Churn:")
for i, (f, v) in enumerate(churn_imp.head(15).items()):
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: NPS → CHURN (predict churn from NPS features only)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL C: NPS → CHURN")
rpt("Can NPS score and themes alone predict churn?")
rpt("=" * 80)

X_nps, _ = build_matrix(df, nps_model_features, "nps")
rpt(f"\n  NPS feature matrix: {X_nps.shape[1]} features x {X_nps.shape[0]} samples")

# C1: NPS score only (single feature)
rpt("\n--- C1: NPS Score Only → Churn ---")
X_c1 = df[['nps_score']].copy()
X_c1 = X_c1.fillna(X_c1.median())
gb_c1 = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1,
                                    random_state=42)
gb_c1_aucs = cross_val_score(gb_c1, X_c1, y_churn, cv=cv_clf, scoring='roc_auc')
rpt(f"  GB AUC (NPS score only → Churn): {gb_c1_aucs.mean():.4f} (+/- {gb_c1_aucs.std():.4f})")

# C2: NPS group ordinal only
rpt("\n--- C2: NPS Group → Churn ---")
X_c2 = df[['nps_group_ordinal']].copy()
X_c2 = X_c2.fillna(1)  # default to Passive
gb_c2 = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1,
                                    random_state=42)
gb_c2_aucs = cross_val_score(gb_c2, X_c2, y_churn, cv=cv_clf, scoring='roc_auc')
rpt(f"  GB AUC (NPS group only → Churn): {gb_c2_aucs.mean():.4f} (+/- {gb_c2_aucs.std():.4f})")

# C3: All NPS features (score + themes + sentiment)
rpt("\n--- C3: Full NPS Feature Set → Churn ---")
gb_c3 = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=42)
gb_c3_aucs = cross_val_score(gb_c3, X_nps, y_churn, cv=cv_clf, scoring='roc_auc')
rpt(f"  GB AUC (Full NPS → Churn): {gb_c3_aucs.mean():.4f} (+/- {gb_c3_aucs.std():.4f})")
rpt(f"  Features used: {X_nps.shape[1]}")

gb_c3.fit(X_nps, y_churn)
nps_churn_imp = pd.Series(gb_c3.feature_importances_, index=X_nps.columns).sort_values(ascending=False)
rpt(f"\n  Top 15 NPS drivers of Churn:")
for i, (f, v) in enumerate(nps_churn_imp.head(15).items()):
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: OPS + NPS → CHURN (incremental NPS value)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL D: OPS + NPS → CHURN")
rpt("Does adding NPS to ops features improve churn prediction?")
rpt("=" * 80)

X_combined = pd.concat([X_ops, X_nps], axis=1)
# Remove any duplicate columns
X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
rpt(f"\n  Combined matrix: {X_combined.shape[1]} features")

gb_d = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   min_samples_leaf=20, random_state=42)
gb_d_aucs = cross_val_score(gb_d, X_combined, y_churn, cv=cv_clf, scoring='roc_auc')
rpt(f"  GB AUC (Ops+NPS → Churn): {gb_d_aucs.mean():.4f} (+/- {gb_d_aucs.std():.4f})")

gb_d.fit(X_combined, y_churn)
comb_imp = pd.Series(gb_d.feature_importances_, index=X_combined.columns).sort_values(ascending=False)
rpt(f"\n  Top 20 Combined drivers of Churn:")
for i, (f, v) in enumerate(comb_imp.head(20).items()):
    nps_flag = " [NPS]" if f in [c for c in X_nps.columns] else ""
    rpt(f"    {i+1:2d}. {f:40s}: {v:.4f} ({v*100:.1f}%){nps_flag}")

# How many NPS features appear in top 20?
top20_nps = [f for f in comb_imp.head(20).index if f in X_nps.columns]
rpt(f"\n  NPS features in top 20: {len(top20_nps)}")
for f in top20_nps:
    rpt(f"    - {f}: {comb_imp[f]:.4f} ({comb_imp[f]*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MODEL COMPARISON SUMMARY")
rpt("=" * 80)

rpt(f"""
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Model                              │ AUC     │ Features │ Signal    │
  ├───────────────────────────────────────────────────────────────────────┤
  │ A1: Ops → NPS Score (regression)   │ R²={r2_gb:.3f}│ {X_a1.shape[1]:8d} │ NPS       │
  │ A2: Ops → Detractor (binary)       │ {gb_aucs.mean():.4f}  │ {X_a2.shape[1]:8d} │ NPS       │
  │ B:  Ops → Churn                    │ {gb_b_aucs.mean():.4f}  │ {X_ops.shape[1]:8d} │ Churn     │
  │ C1: NPS Score → Churn              │ {gb_c1_aucs.mean():.4f}  │        1 │ Churn     │
  │ C2: NPS Group → Churn              │ {gb_c2_aucs.mean():.4f}  │        1 │ Churn     │
  │ C3: Full NPS → Churn               │ {gb_c3_aucs.mean():.4f}  │ {X_nps.shape[1]:8d} │ Churn     │
  │ D:  Ops + NPS → Churn              │ {gb_d_aucs.mean():.4f}  │ {X_combined.shape[1]:8d} │ Churn     │
  └───────────────────────────────────────────────────────────────────────┘

  KEY INSIGHTS:
  1. NPS incremental lift:  D - B = {(gb_d_aucs.mean() - gb_b_aucs.mean())*10000:.0f} bps
     Adding NPS to ops features {'improves' if gb_d_aucs.mean() > gb_b_aucs.mean() + 0.005 else 'barely changes'} churn prediction.

  2. NPS alone (C3) vs Ops alone (B): {gb_c3_aucs.mean():.4f} vs {gb_b_aucs.mean():.4f}
     Ops features are {'much stronger' if gb_b_aucs.mean() - gb_c3_aucs.mean() > 0.10 else 'stronger' if gb_b_aucs.mean() > gb_c3_aucs.mean() else 'weaker'} than NPS for churn prediction.

  3. Ops → NPS predictability: R²={r2_gb:.3f}, AUC(D/P)={gb_aucs.mean():.4f}
     {'Ops features explain substantial NPS variance — population scoring is viable.' if r2_gb > 0.15 or gb_aucs.mean() > 0.70 else 'Ops features have limited NPS predictive power — NPS captures something ops data misses.'}

  4. NPS Score alone → Churn: AUC={gb_c1_aucs.mean():.4f}
     {'NPS score is a meaningful but weak churn signal.' if gb_c1_aucs.mean() < 0.65 else 'NPS score is a moderate churn predictor.' if gb_c1_aucs.mean() < 0.75 else 'NPS score is a strong churn predictor.'}
""")


# ══════════════════════════════════════════════════════════════════════════════
# MEDIATION ANALYSIS: Ops → NPS → Churn pathway
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("MEDIATION ANALYSIS: Does NPS mediate the Ops → Churn relationship?")
rpt("=" * 80)

# Path analysis:
# Total effect: Ops → Churn (Model B AUC)
# Direct effect: Ops → Churn controlling for NPS (partial correlation)
# Indirect effect: Ops → NPS → Churn (mediation)

# Simple approach: Compare feature importances across models
rpt("\n  Feature importance comparison (top ops features across models):")
rpt(f"  {'Feature':35s} | {'Ops→NPS':>8s} | {'Ops→Churn':>10s} | {'Combined':>9s} | Pattern")
rpt(f"  {'-'*35} | {'-'*8} | {'-'*10} | {'-'*9} | {'-'*25}")

for f in churn_imp.head(10).index:
    nps_v = nps_imp.get(f, 0)
    churn_v = churn_imp.get(f, 0)
    comb_v = comb_imp.get(f, 0)

    # Pattern detection
    if nps_v > 0.02 and churn_v > 0.02:
        if comb_v < churn_v * 0.7:
            pattern = "Mediated by NPS"
        else:
            pattern = "Direct + NPS pathway"
    elif nps_v < 0.01 and churn_v > 0.02:
        pattern = "Direct churn driver"
    elif nps_v > 0.02 and churn_v < 0.01:
        pattern = "NPS driver (not churn)"
    else:
        pattern = "Weak"

    rpt(f"  {f:35s} | {nps_v*100:7.1f}% | {churn_v*100:9.1f}% | {comb_v*100:8.1f}% | {pattern}")


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUAL ANALYSIS: What does NPS capture that ops misses?
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("RESIDUAL ANALYSIS: What does NPS capture beyond ops?")
rpt("=" * 80)

# Predict churn with ops, then check if NPS residual adds signal
gb_b_full = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        min_samples_leaf=20, random_state=42)
ops_churn_proba = cross_val_predict(gb_b_full, X_ops, y_churn, cv=cv_clf, method='predict_proba')[:, 1]

# Bin customers by ops-predicted churn risk
risk_bins = pd.qcut(ops_churn_proba, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
df['ops_risk_bin'] = risk_bins.values

rpt(f"\n  Within each ops-risk bin, does NPS group still predict churn?")
rpt(f"  {'Risk Bin':12s} | {'n':>6s} | {'Promoter':>10s} | {'Passive':>10s} | {'Detractor':>10s} | {'D-P gap':>8s}")
rpt(f"  {'-'*12} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    bin_df = df[df['ops_risk_bin'] == bin_name]
    n = len(bin_df)
    p_churn = bin_df.loc[bin_df['nps_group']=='Promoter', 'churn_binary'].mean() * 100
    pa_churn = bin_df.loc[bin_df['nps_group']=='Passive', 'churn_binary'].mean() * 100
    d_churn = bin_df.loc[bin_df['nps_group']=='Detractor', 'churn_binary'].mean() * 100
    gap = d_churn - p_churn
    rpt(f"  {bin_name:12s} | {n:6d} | {p_churn:9.1f}% | {pa_churn:9.1f}% | {d_churn:9.1f}% | {gap:+7.1f}pp")

rpt(f"""
  INTERPRETATION:
  If NPS group still predicts churn WITHIN ops-risk bins, NPS captures
  something ops data doesn't (e.g., subjective experience, intent to leave,
  competitor awareness). This would justify the two-stage approach.

  If the D-P gap disappears within bins, NPS is fully explained by ops data
  and the simpler Ops→Churn model suffices.
""")


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION SCORING VIABILITY
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("POPULATION SCORING VIABILITY")
rpt("=" * 80)

# Model A tells us: can we predict NPS for non-respondents?
# If R² is decent (>0.15) or AUC(D/P) is decent (>0.70):
#   → Score entire population with predicted NPS
#   → Use predicted NPS in churn model for everyone

rpt(f"""
  Model A Performance:
  - NPS Score Regression:     R²={r2_gb:.3f}, MAE={mae_gb:.2f}
  - Detractor Classification: AUC={gb_aucs.mean():.4f}
  - 3-Class Classification:   AUC-OVR={gb_3c_aucs.mean():.4f}

  Population Scoring Strategy:
""")

if gb_aucs.mean() > 0.70:
    rpt(f"  ✓ VIABLE: Ops features can predict Detractor status (AUC={gb_aucs.mean():.4f})")
    rpt(f"    → Score entire Wiom population with predicted Detractor probability")
    rpt(f"    → Use as a proxy NPS risk score for non-respondents")
    rpt(f"    → Combine with actual NPS for respondents (hybrid approach)")
else:
    rpt(f"  ✗ LIMITED: Ops features predict Detractor status poorly (AUC={gb_aucs.mean():.4f})")
    rpt(f"    → NPS captures subjective factors not in ops data")
    rpt(f"    → Population scoring would be noisy")
    rpt(f"    → Better to increase survey coverage than predict NPS")

nps_lift = (gb_d_aucs.mean() - gb_b_aucs.mean()) * 10000
if nps_lift > 50:
    rpt(f"\n  ✓ NPS ADDS VALUE: +{nps_lift:.0f} bps lift in churn model when NPS is included")
    rpt(f"    → Worth investing in NPS prediction for population scoring")
elif nps_lift > 10:
    rpt(f"\n  ~ MARGINAL NPS VALUE: +{nps_lift:.0f} bps lift — small but present")
    rpt(f"    → NPS adds incremental signal; hybrid model may be worthwhile")
else:
    rpt(f"\n  ✗ NO INCREMENTAL NPS VALUE: +{nps_lift:.0f} bps lift")
    rpt(f"    → Ops features already capture what NPS measures")
    rpt(f"    → Simpler Ops→Churn model is sufficient for population scoring")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDED APPROACH
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("RECOMMENDED APPROACH")
rpt("=" * 80)

rpt(f"""
  Based on the three-model analysis:

  MODEL PERFORMANCE HIERARCHY:
    Ops → Churn (B):      AUC = {gb_b_aucs.mean():.4f}
    Ops+NPS → Churn (D):  AUC = {gb_d_aucs.mean():.4f}  (lift = {nps_lift:+.0f} bps)
    NPS → Churn (C3):     AUC = {gb_c3_aucs.mean():.4f}
    NPS Score → Churn:    AUC = {gb_c1_aucs.mean():.4f}

  OPS → NPS PREDICTION:
    Detractor/Promoter:   AUC = {gb_aucs.mean():.4f}
    NPS Score:            R² = {r2_gb:.3f}
""")

# Clean up temp column
df.drop(columns=['ops_risk_bin'], inplace=True, errors='ignore')


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL HOLDOUT (all models)
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 80)
rpt("TEMPORAL HOLDOUT (Train: Sprints 1-7, Test: 8-11)")
rpt("=" * 80)

raw_sprint = df['sprint_num']
train_mask = raw_sprint.between(1, 7)
test_mask = raw_sprint.between(8, 11)

df_train = df[train_mask]
df_test = df[test_mask]

rpt(f"  Train: n={len(df_train)}, churn={df_train['churn_binary'].mean()*100:.1f}%")
rpt(f"  Test:  n={len(df_test)}, churn={df_test['churn_binary'].mean()*100:.1f}%")

# Build matrices for train/test
X_ops_train, imp_ops = build_matrix(df_train, ops_candidates, "ops_train")
ops_train_cols = list(X_ops_train.columns)

# Build test ops matrix aligned to train columns
X_ops_test_raw, _ = build_matrix(df_test, ops_candidates, "ops_test")
for c in ops_train_cols:
    if c not in X_ops_test_raw.columns:
        X_ops_test_raw[c] = np.nan
X_ops_test = X_ops_test_raw[ops_train_cols]
# Impute with training imputer
X_ops_test = pd.DataFrame(imp_ops.transform(X_ops_test), columns=ops_train_cols, index=X_ops_test.index)

y_train = df_train['churn_binary'].values
y_test = df_test['churn_binary'].values

# Model B OOT
gb_b_oot = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
gb_b_oot.fit(X_ops_train, y_train)
b_proba = gb_b_oot.predict_proba(X_ops_test)[:, 1]
b_auc = roc_auc_score(y_test, b_proba)

# Model C OOT (NPS features)
X_nps_train, imp_nps = build_matrix(df_train, nps_model_features, "nps_train")
nps_train_cols = list(X_nps_train.columns)
X_nps_test_raw, _ = build_matrix(df_test, nps_model_features, "nps_test")
for c in nps_train_cols:
    if c not in X_nps_test_raw.columns:
        X_nps_test_raw[c] = np.nan
X_nps_test = X_nps_test_raw[nps_train_cols]
X_nps_test = pd.DataFrame(imp_nps.transform(X_nps_test), columns=nps_train_cols, index=X_nps_test.index)

gb_c_oot = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
gb_c_oot.fit(X_nps_train, y_train)
c_proba = gb_c_oot.predict_proba(X_nps_test)[:, 1]
c_auc = roc_auc_score(y_test, c_proba)

# Model D OOT (Ops + NPS)
X_comb_train = pd.concat([X_ops_train, X_nps_train], axis=1)
X_comb_train = X_comb_train.loc[:, ~X_comb_train.columns.duplicated()]
X_comb_test = pd.concat([X_ops_test, X_nps_test], axis=1)
X_comb_test = X_comb_test.loc[:, ~X_comb_test.columns.duplicated()]
# Align columns
for c in X_comb_train.columns:
    if c not in X_comb_test.columns:
        X_comb_test[c] = np.nan
X_comb_test = X_comb_test[X_comb_train.columns]

gb_d_oot = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
gb_d_oot.fit(X_comb_train, y_train)
d_proba = gb_d_oot.predict_proba(X_comb_test)[:, 1]
d_auc = roc_auc_score(y_test, d_proba)

# Model A OOT (Ops → Detractor)
dp_train = df_train[df_train['nps_group'].isin(['Detractor', 'Promoter'])]
dp_test = df_test[df_test['nps_group'].isin(['Detractor', 'Promoter'])]
X_a_train_dp, _ = build_matrix(dp_train, ops_candidates, "a_train")
a_train_cols = list(X_a_train_dp.columns)
X_a_test_raw, _ = build_matrix(dp_test, ops_candidates, "a_test")
for c in a_train_cols:
    if c not in X_a_test_raw.columns:
        X_a_test_raw[c] = np.nan
X_a_test_dp = X_a_test_raw[a_train_cols]

y_a_train = (dp_train['nps_group'] == 'Detractor').astype(int).values
y_a_test = (dp_test['nps_group'] == 'Detractor').astype(int).values

gb_a_oot = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, random_state=42)
gb_a_oot.fit(X_a_train_dp, y_a_train)
a_proba = gb_a_oot.predict_proba(X_a_test_dp)[:, 1]
a_auc = roc_auc_score(y_a_test, a_proba)

rpt(f"\n  OUT-OF-TIME AUC COMPARISON:")
rpt(f"    Model A (Ops → Detractor):  {a_auc:.4f}")
rpt(f"    Model B (Ops → Churn):      {b_auc:.4f}")
rpt(f"    Model C (NPS → Churn):      {c_auc:.4f}")
rpt(f"    Model D (Ops+NPS → Churn):  {d_auc:.4f}")
rpt(f"    NPS incremental lift (OOT): {(d_auc - b_auc)*10000:+.0f} bps")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output_file = os.path.join(OUTPUT, "phase4b_three_model_framework.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

rpt(f"\n{'='*80}")
rpt(f"SAVED: {output_file}")
rpt(f"COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*80}")
