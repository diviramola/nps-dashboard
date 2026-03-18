"""
Phase 4: Statistical Modeling — KDA, Segmentation, Churn Risk Scoring
=====================================================================
Uses nps_modeling_dataset.csv (unified dataset from Phase 3C).
Includes: Snowflake ops features, Industry Expert latent CX features,
partner-level aggregates, temporal partner status, NLP themes.

Key Driver Analysis using 4 methods, customer segmentation,
iterative churn risk scoring targeting AUC-ROC > 0.85.

Adaptive findings incorporated:
- PEAK_UPTIME_PCT > avg_uptime_pct (partner fleet vs customer metric)
- Temporal partner_status_at_survey (no data leakage)
- Support effort index as composite predictor
- Partner-level deviations as individual-vs-norm signals
- Principled outlier handling (winsorized features)

Outputs:
- output/phase4_statistical_modeling.txt
- data/nps_with_risk_scores.csv
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

report = []
def R(line=""):
    report.append(line)
    print(line)

def save_report():
    path = os.path.join(OUTPUT, "phase4_statistical_modeling.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n  Report saved: {path}")

# ══════════════════════════════════════════════════════════════════════
# STEP 0: Load unified modeling dataset
# ══════════════════════════════════════════════════════════════════════
R("=" * 70)
R("PHASE 4 -- STATISTICAL MODELING")
R("=" * 70)
R(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
R("")

print("\n[0] Loading unified modeling dataset...")

modeling_path = os.path.join(DATA, "nps_modeling_dataset.csv")
if not os.path.exists(modeling_path):
    print("  ERROR: nps_modeling_dataset.csv not found. Run phase3c_final_merge.py first.")
    sys.exit(1)

df_full = pd.read_csv(modeling_path, low_memory=False)
print(f"  Full dataset: {len(df_full)} rows x {len(df_full.columns)} cols")

# Model on CONSOLIDATED rows only (new sprint tab rows lack Snowflake features)
if '_source' in df_full.columns:
    df = df_full[df_full['_source'] == 'consolidated'].copy()
    R(f"  Modeling on consolidated rows: {len(df)} (excluded {len(df_full)-len(df)} new rows)")
else:
    df = df_full.copy()
    R(f"  No _source column; using all {len(df)} rows")

R(f"Final modeling dataset: {len(df)} rows x {len(df.columns)} columns")

# ── Standardize key columns ──
df['phone_number'] = df['phone_number'].astype(str)
if 'nps_score' in df.columns:
    df['nps_score'] = pd.to_numeric(df['nps_score'], errors='coerce')
if 'is_churned' in df.columns:
    df['is_churned'] = pd.to_numeric(df['is_churned'], errors='coerce').fillna(0).astype(int)
elif 'churn_label' in df.columns:
    df['is_churned'] = (df['churn_label'].str.lower() == 'churn').astype(int)

# ── Categorize features by source ──
# Phase 3: Snowflake operational features
snowflake_features = [c for c in df.columns if c in [
    'total_recharges', 'avg_recharge_amount',
    'total_tickets', 'cx_tickets', 'px_tickets', 'avg_resolution_hours',
    'avg_resolution_hours_w', 'sla_compliance_pct',
    'total_payments', 'autopay_payments', 'cash_payments', 'avg_payment_amount',
    'install_tat_hours', 'install_tat_hours_w', 'install_attempts', 'install_delayed',
    'avg_uptime_pct', 'stddev_uptime', 'min_uptime',
    'payment_mode', 'has_tickets',
    'days_since_last_recharge', 'recharge_regularity'
]]

# Phase 3b: Industry Expert latent CX features
industry_expert_features = [c for c in df.columns if c in [
    'OUTAGE_EVENTS', 'DISTINCT_OUTAGES', 'AVG_RECOVERY_MINS', 'RECOVERED_EVENTS',
    'PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT', 'OVERALL_UPTIME_PCT',
    'AVG_PEAK_INTERRUPTIONS', 'PEAK_VS_OVERALL_GAP',
    'FCR_RATE', 'AVG_TIMES_REOPENED', 'MAX_TIMES_REOPENED',
    'AVG_CUSTOMER_CALLS_PER_TICKET', 'TOTAL_CUSTOMER_CALLS', 'AVG_TICKET_RATING',
    'TICKETS_REOPENED_ONCE', 'TICKETS_REOPENED_3PLUS',
    'DISTINCT_ISSUE_TYPES', 'MAX_TICKETS_SAME_ISSUE', 'AVG_TICKETS_PER_ISSUE',
    'ISSUES_WITH_3PLUS_TICKETS', 'HAS_REPEAT_COMPLAINT',
    'PAYMENT_FAILURES', 'PAYMENT_SUCCESSES', 'FAILURE_RATE_PCT',
    'DISPATCH_DECLINE_RATE_PCT', 'LEADS_ALL_DECLINED',
    'AVG_INSTALL_TAT_MINS', 'AVG_INSTALL_TAT_MINS_w', 'AVG_INSTALL_RATING',
    'HOURS_TO_FIRST_RECHARGE', 'DAYS_TO_FIRST_RECHARGE', 'recharge_same_day',
    'TOTAL_IVR_CALLS', 'INBOUND_CALLS', 'OUTBOUND_CALLS',
    'ANSWERED_CALLS', 'MISSED_CALLS', 'DROPPED_CALLS', 'AVG_ANSWERED_SECONDS',
    'AVG_DAILY_DATA_GB', 'MEDIAN_DAILY_DATA_GB',
    'LOW_USAGE_DAYS', 'HIGH_USAGE_DAYS'
]]

# Derived features (from Phase 3C)
derived_features = [c for c in df.columns if c in [
    'support_effort_index', 'network_quality_index',
    'missed_call_ratio', 'autopay_ratio', 'ticket_severity',
    'partner_risk_level', 'partner_at_risk',
    'partner_avg_resolution_hours', 'partner_sla_compliance',
    'partner_avg_tickets', 'partner_churn_rate', 'partner_avg_nps',
    'partner_fcr_rate', 'partner_repeat_rate', 'partner_median_install_tat',
    'partner_customer_count',
    'resolution_vs_partner', 'sla_vs_partner', 'tickets_vs_partner'
]]

# Sprint tab features
sprint_tab_features = [c for c in df.columns if c in [
    'device_type', 'optical_power', 'devices_2_4g', 'devices_5g',
    'data_usage_amount', 'data_usage_days', 'data_usage_percentile',
    'data_usage_per_day', 'recharge_done', 'cash_online',
    'plan_expiry_window', 'tickets_last_3m_sprint'
]]

# Excel features
excel_features = [c for c in df.columns if c in [
    'tenure_days', 'sprint_num', 'recharges_before_sprint',
    'tickets_post_sprint', 'tickets_before_3m', 'first_time_wifi'
]]

R(f"\nFeature availability:")
R(f"  Snowflake ops features:     {len(snowflake_features)}")
R(f"  Industry Expert features:   {len(industry_expert_features)}")
R(f"  Derived composite features: {len(derived_features)}")
R(f"  Sprint tab features:        {len(sprint_tab_features)}")
R(f"  Excel features:             {len(excel_features)}")
R(f"  Theme features:             {'primary_theme' in df.columns}")
R(f"  Temporal partner status:    {'partner_status_at_survey' in df.columns}")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 1: EXPLORATORY DATA ANALYSIS")
R("=" * 70)

# NPS distribution
R("\n--- NPS Score Distribution ---")
nps_dist = df['nps_score'].value_counts().sort_index()
for score, cnt in nps_dist.items():
    bar = '#' * int(cnt / len(df) * 100)
    R(f"  {int(score):2d}: {cnt:5d} ({cnt/len(df)*100:5.1f}%) {bar}")

# NPS groups
R("\n--- NPS Group Distribution ---")
if 'nps_group' in df.columns:
    for grp in ['Promoter', 'Passive', 'Detractor']:
        cnt = (df['nps_group'] == grp).sum()
        R(f"  {grp:12s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")

# Overall churn rate
churn_rate = df['is_churned'].mean() * 100
R(f"\n--- Overall Churn Rate: {churn_rate:.1f}% ({df['is_churned'].sum()} / {len(df)}) ---")

# Churn by NPS group
R("\n--- Churn Rate by NPS Group ---")
if 'nps_group' in df.columns:
    churn_by_grp = df.groupby('nps_group')['is_churned'].agg(['mean', 'count'])
    for grp in ['Promoter', 'Passive', 'Detractor']:
        if grp in churn_by_grp.index:
            R(f"  {grp:12s}: {churn_by_grp.at[grp,'mean']*100:5.1f}% churn (n={int(churn_by_grp.at[grp,'count'])})")

# Churn by tenure bucket
R("\n--- Churn Rate by Tenure Bucket ---")
if 'tenure_bucket' in df.columns:
    tb_order = ['Onboarding', 'Early Life', 'Establishing', 'Steady State', 'Loyal']
    churn_tb = df.groupby('tenure_bucket')['is_churned'].agg(['mean', 'count'])
    for tb in tb_order:
        if tb in churn_tb.index:
            R(f"  {tb:15s}: {churn_tb.at[tb,'mean']*100:5.1f}% churn (n={int(churn_tb.at[tb,'count'])})")

# Churn by city
R("\n--- Churn Rate by City ---")
if 'city' in df.columns:
    churn_city = df.groupby('city')['is_churned'].agg(['mean', 'count']).sort_values('count', ascending=False)
    for city, row in churn_city.iterrows():
        if row['count'] >= 20:
            R(f"  {city:15s}: {row['mean']*100:5.1f}% churn (n={int(row['count'])})")

# Churn by sprint
R("\n--- Churn Rate by Sprint ---")
if 'sprint_num' in df.columns:
    churn_sprint = df.groupby('sprint_num')['is_churned'].agg(['mean', 'count']).sort_index()
    for sp, row in churn_sprint.iterrows():
        R(f"  Sprint {int(sp):2d}: {row['mean']*100:5.1f}% churn (n={int(row['count'])})")

# NPS by tenure bucket
R("\n--- Mean NPS by Tenure Bucket ---")
if 'tenure_bucket' in df.columns:
    nps_tb = df.groupby('tenure_bucket')['nps_score'].agg(['mean', 'median', 'count'])
    for tb in tb_order:
        if tb in nps_tb.index:
            R(f"  {tb:15s}: mean={nps_tb.at[tb,'mean']:.1f}, median={nps_tb.at[tb,'median']:.0f} (n={int(nps_tb.at[tb,'count'])})")

# Theme-score cross-tabs (if available)
if 'primary_theme' in df.columns:
    R("\n--- Mean NPS & Churn by Theme (classified comments only) ---")
    themed = df[df['primary_theme'].notna() & (df['primary_theme'] != 'unclassified')]
    if len(themed) > 0:
        theme_stats = themed.groupby('primary_theme').agg(
            count=('nps_score', 'count'),
            mean_nps=('nps_score', 'mean'),
            churn_rate=('is_churned', 'mean')
        ).sort_values('count', ascending=False)
        R(f"  {'Theme':35s} | {'Count':>6s} | {'Mean NPS':>8s} | {'Churn%':>7s}")
        R(f"  {'-'*35}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
        for theme, row in theme_stats.iterrows():
            R(f"  {theme:35s} | {int(row['count']):6d} | {row['mean_nps']:8.1f} | {row['churn_rate']*100:6.1f}%")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 2: FEATURE MATRIX CONSTRUCTION")
R("=" * 70)

# Identify all numeric features we can use
all_potential_features = []

# Excel-derived numeric features
excel_numeric = ['tenure_days', 'sprint_num', 'recharges_before_sprint',
                 'tickets_post_sprint', 'tickets_before_3m']
for col in excel_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        all_potential_features.append(col)

# Snowflake features (prefer winsorized versions)
for col in snowflake_features:
    if col in df.columns and col not in ['payment_mode']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        all_potential_features.append(col)

# Industry Expert features (Phase 3b)
for col in industry_expert_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        all_potential_features.append(col)

# Derived composite features
for col in derived_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        all_potential_features.append(col)

# Sprint tab features
for col in sprint_tab_features:
    if col in df.columns and col not in ['cash_online', 'device_type']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        all_potential_features.append(col)

# Encode categorical features as dummies
categorical_features = []

# City dummies
if 'city' in df.columns:
    city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True)
    for c in city_dummies.columns:
        df[c] = city_dummies[c]
        categorical_features.append(c)

# Tenure bucket dummies
if 'tenure_bucket' in df.columns:
    tb_dummies = pd.get_dummies(df['tenure_bucket'], prefix='tenure', drop_first=True)
    for c in tb_dummies.columns:
        df[c] = tb_dummies[c]
        categorical_features.append(c)

# NPS group ordinal
df['nps_group_ordinal'] = df['nps_group'].map({'Detractor': 0, 'Passive': 1, 'Promoter': 2})

# Channel dummies
if 'Channel' in df.columns:
    ch_dummies = pd.get_dummies(df['Channel'], prefix='channel', drop_first=True)
    for c in ch_dummies.columns:
        df[c] = ch_dummies[c]
        categorical_features.append(c)

# First-time WiFi
if 'first_time_wifi' in df.columns:
    df['is_first_time_wifi'] = df['first_time_wifi'].map({'Yes': 1, 'No': 0, 'हां': 1, 'नहीं': 0}).fillna(0).astype(int)
    all_potential_features.append('is_first_time_wifi')

# Payment mode dummies (from Snowflake)
if 'payment_mode' in df.columns:
    pm_dummies = pd.get_dummies(df['payment_mode'], prefix='paymode', drop_first=True)
    for c in pm_dummies.columns:
        df[c] = pm_dummies[c]
        categorical_features.append(c)

# Cash/online from sprint tabs
if 'cash_online' in df.columns:
    df['is_cash_payment'] = (df['cash_online'].str.upper() == 'CASH').astype(int)
    all_potential_features.append('is_cash_payment')

# Device type from sprint tabs
if 'device_type' in df.columns:
    dt_dummies = pd.get_dummies(df['device_type'], prefix='device', drop_first=True)
    for c in dt_dummies.columns:
        df[c] = dt_dummies[c]
        categorical_features.append(c)

# Theme dummies (for the subset with comments)
if 'primary_theme' in df.columns:
    # Collapse rare themes
    theme_counts = df['primary_theme'].value_counts()
    significant_themes = theme_counts[theme_counts >= 20].index.tolist()
    df['theme_clean'] = df['primary_theme'].where(df['primary_theme'].isin(significant_themes), 'other')
    theme_dummies = pd.get_dummies(df['theme_clean'], prefix='theme', drop_first=True)
    for c in theme_dummies.columns:
        df[c] = theme_dummies[c]
        categorical_features.append(c)

# Sentiment features
if 'sentiment_polarity' in df.columns:
    df['is_negative_sentiment'] = (df['sentiment_polarity'] == 'negative').astype(int)
    df['is_positive_sentiment'] = (df['sentiment_polarity'] == 'positive').astype(int)
    all_potential_features.extend(['is_negative_sentiment', 'is_positive_sentiment'])

if 'sentiment_intensity' in df.columns:
    df['sentiment_intensity'] = pd.to_numeric(df['sentiment_intensity'], errors='coerce')
    all_potential_features.append('sentiment_intensity')

# Has comment flag
if 'has_comment' in df.columns:
    df['has_comment'] = pd.to_numeric(df['has_comment'], errors='coerce').fillna(0).astype(int)
    all_potential_features.append('has_comment')

# PayG flag
if 'is_payg' in df.columns:
    all_potential_features.append('is_payg')
elif 'Install Date' in df.columns:
    df['install_date_parsed'] = pd.to_datetime(df['Install Date'], errors='coerce')
    df['is_payg'] = (df['install_date_parsed'] >= '2026-01-26').astype(int)
    all_potential_features.append('is_payg')

# Combine all features
all_features = list(set(all_potential_features + categorical_features))
# Remove any that are target or identifier columns
exclude = ['nps_score', 'nps_group', 'nps_group_ordinal', 'is_churned',
           'phone_number', 'churn_label', 'OE', 'phone_valid']
all_features = [f for f in all_features if f in df.columns and f not in exclude]

R(f"\nTotal candidate features: {len(all_features)}")
R(f"  Numeric (original): {len(all_potential_features)}")
R(f"  Categorical (encoded): {len(categorical_features)}")

# Check feature fill rates
R("\n--- Feature Fill Rates ---")
feature_fill = {}
for f in sorted(all_features):
    fill = df[f].notna().sum()
    pct = fill / len(df) * 100
    feature_fill[f] = pct
    if pct < 100:
        R(f"  {f:40s}: {fill:6d} / {len(df)} ({pct:5.1f}%)")

# Separate features into high-fill (>=50%) and low-fill
high_fill_features = [f for f in all_features if feature_fill.get(f, 0) >= 50]
R(f"\nHigh-fill features (>=50%): {len(high_fill_features)}")

# For modeling: use high-fill features, fill remaining NAs with median
model_features = high_fill_features.copy()
for f in model_features:
    if df[f].isna().any():
        if df[f].dtype in ['float64', 'int64', 'float32', 'int32']:
            df[f] = df[f].fillna(df[f].median())
        else:
            df[f] = df[f].fillna(0)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: KEY DRIVER ANALYSIS — NPS SCORE
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 3: KEY DRIVER ANALYSIS — What drives NPS score?")
R("=" * 70)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats

# Filter to rows with valid NPS scores and at least some features
valid_mask = df['nps_score'].notna()
for f in model_features[:3]:  # at least first 3 features should be non-NA
    valid_mask = valid_mask & df[f].notna()

df_model = df[valid_mask].copy()
R(f"\nModeling sample: {len(df_model)} rows (of {len(df)} total)")

X = df_model[model_features].values
y_nps = df_model['nps_score'].values

# Scale features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Method 1: Linear Regression (standardized coefficients) ──
R("\n--- Method 1: Linear Regression (Standardized Coefficients) ---")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

lr = Ridge(alpha=1.0)
lr.fit(X_scaled, y_nps)
r2 = r2_score(y_nps, lr.predict(X_scaled))
R(f"  R-squared: {r2:.3f}")

coef_df = pd.DataFrame({
    'feature': model_features,
    'coefficient': lr.coef_,
    'abs_coef': np.abs(lr.coef_)
}).sort_values('abs_coef', ascending=False)

R(f"\n  Top 15 drivers (standardized coefficients):")
R(f"  {'Feature':40s} | {'Coefficient':>12s} | {'Direction':>10s}")
R(f"  {'-'*40}-+-{'-'*12}-+-{'-'*10}")
for _, row in coef_df.head(15).iterrows():
    direction = "POSITIVE" if row['coefficient'] > 0 else "NEGATIVE"
    R(f"  {row['feature']:40s} | {row['coefficient']:+12.4f} | {direction:>10s}")

# ── Method 2: Random Forest Feature Importance ──
R("\n--- Method 2: Random Forest Feature Importance ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20,
                           random_state=42, n_jobs=-1)
rf.fit(X, y_nps)
rf_r2 = r2_score(y_nps, rf.predict(X))
R(f"  R-squared: {rf_r2:.3f}")

# Cross-validated score
cv_scores = cross_val_score(rf, X, y_nps, cv=5, scoring='r2')
R(f"  CV R-squared: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

rf_importance = pd.DataFrame({
    'feature': model_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

R(f"\n  Top 15 features by RF importance:")
R(f"  {'Feature':40s} | {'Importance':>12s} | {'Cum%':>6s}")
R(f"  {'-'*40}-+-{'-'*12}-+-{'-'*6}")
cum_imp = 0
for _, row in rf_importance.head(15).iterrows():
    cum_imp += row['importance']
    R(f"  {row['feature']:40s} | {row['importance']:12.4f} | {cum_imp*100:5.1f}%")

# ── Method 3: Partial Correlations ──
R("\n--- Method 3: Partial Correlations (NPS vs each feature, controlling others) ---")
# Simple approach: correlation of residuals
partial_corrs = []
for i, feat in enumerate(model_features):
    # Regress NPS on all OTHER features
    other_idx = [j for j in range(len(model_features)) if j != i]
    if len(other_idx) == 0:
        continue
    X_other = X_scaled[:, other_idx]

    # Residual of NPS after removing other features
    lr_y = LinearRegression().fit(X_other, y_nps)
    resid_y = y_nps - lr_y.predict(X_other)

    # Residual of feature after removing other features
    lr_x = LinearRegression().fit(X_other, X_scaled[:, i])
    resid_x = X_scaled[:, i] - lr_x.predict(X_other)

    # Partial correlation
    if np.std(resid_x) > 0 and np.std(resid_y) > 0:
        pcorr, pval = stats.pearsonr(resid_x, resid_y)
        partial_corrs.append({'feature': feat, 'partial_corr': pcorr, 'p_value': pval, 'abs_pcorr': abs(pcorr)})

pcorr_df = pd.DataFrame(partial_corrs).sort_values('abs_pcorr', ascending=False)
R(f"\n  Top 15 partial correlations with NPS:")
R(f"  {'Feature':40s} | {'Partial r':>10s} | {'p-value':>10s} | {'Sig':>4s}")
R(f"  {'-'*40}-+-{'-'*10}-+-{'-'*10}-+-{'-'*4}")
for _, row in pcorr_df.head(15).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    R(f"  {row['feature']:40s} | {row['partial_corr']:+10.4f} | {row['p_value']:10.6f} | {sig:>4s}")

# ── Method 4: Gradient Boosting + SHAP-like importance ──
R("\n--- Method 4: Gradient Boosting + Permutation Importance ---")
from sklearn.inspection import permutation_importance

gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                min_samples_leaf=20, random_state=42)
gb.fit(X, y_nps)
gb_r2 = r2_score(y_nps, gb.predict(X))
R(f"  R-squared: {gb_r2:.3f}")

# Permutation importance (more reliable than built-in for correlated features)
perm_imp = permutation_importance(gb, X, y_nps, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'feature': model_features,
    'importance_mean': perm_imp.importances_mean,
    'importance_std': perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)

R(f"\n  Top 15 by permutation importance:")
R(f"  {'Feature':40s} | {'Mean Imp':>10s} | {'Std':>8s}")
R(f"  {'-'*40}-+-{'-'*10}-+-{'-'*8}")
for _, row in perm_df.head(15).iterrows():
    R(f"  {row['feature']:40s} | {row['importance_mean']:10.4f} | {row['importance_std']:8.4f}")

# ── Consensus Ranking ──
R("\n--- CONSENSUS DRIVER RANKING (appears in top-10 across >=2 methods) ---")
top10_lr = set(coef_df.head(10)['feature'])
top10_rf = set(rf_importance.head(10)['feature'])
top10_pc = set(pcorr_df.head(10)['feature'])
top10_gb = set(perm_df.head(10)['feature'])

all_top10 = {}
for feat in set().union(top10_lr, top10_rf, top10_pc, top10_gb):
    methods = []
    if feat in top10_lr: methods.append('LR')
    if feat in top10_rf: methods.append('RF')
    if feat in top10_pc: methods.append('PC')
    if feat in top10_gb: methods.append('GB')
    all_top10[feat] = methods

consensus = {f: m for f, m in all_top10.items() if len(m) >= 2}
consensus_sorted = sorted(consensus.items(), key=lambda x: -len(x[1]))

R(f"\n  {'Feature':40s} | {'# Methods':>10s} | {'Methods'}")
R(f"  {'-'*40}-+-{'-'*10}-+-{'-'*20}")
for feat, methods in consensus_sorted:
    R(f"  {feat:40s} | {len(methods):10d} | {', '.join(methods)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: TENURE-STRATIFIED KEY DRIVER ANALYSIS
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 4: TENURE-STRATIFIED KEY DRIVER ANALYSIS")
R("=" * 70)

# Fine-grained tenure_bucket is 96.6% Loyal (270d+) — use Excel tenure instead
R("\n  NOTE: Fine-grained tenure_bucket is 96.6% Loyal. Using tenure_excel (1-2/3-6/6+).")

if 'tenure_excel' in df_model.columns:
    tenure_labels = {'1\x962': '1-2 months', '3\x966': '3-6 months', '6+': '6+ months'}
    # Try to decode the tenure column
    for raw_val in df_model['tenure_excel'].dropna().unique():
        if raw_val not in tenure_labels:
            tenure_labels[raw_val] = str(raw_val)

    for bucket_val in df_model['tenure_excel'].dropna().unique():
        bucket_mask = df_model['tenure_excel'] == bucket_val
        n_bucket = bucket_mask.sum()
        label = tenure_labels.get(bucket_val, str(bucket_val))

        if n_bucket < 50:
            R(f"\n  [{label}] Skipping — only {n_bucket} rows")
            continue

        R(f"\n--- [{label}] n={n_bucket}, mean NPS={df_model.loc[bucket_mask, 'nps_score'].mean():.1f}, churn={df_model.loc[bucket_mask, 'is_churned'].mean()*100:.1f}% ---")

        X_bucket = df_model.loc[bucket_mask, model_features].values
        y_bucket = df_model.loc[bucket_mask, 'nps_score'].values

        # Random Forest for this bucket
        rf_b = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                     random_state=42, n_jobs=-1)
        rf_b.fit(X_bucket, y_bucket)

        imp_b = pd.DataFrame({
            'feature': model_features,
            'importance': rf_b.feature_importances_
        }).sort_values('importance', ascending=False)

        R(f"  Top 5 drivers:")
        for _, row in imp_b.head(5).iterrows():
            R(f"    {row['feature']:40s}: {row['importance']:.4f}")
else:
    R("  No tenure_excel column available.")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 5: CUSTOMER SEGMENTATION")
R("=" * 70)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Select features for clustering (use high-importance features)
cluster_candidates = list(rf_importance.head(10)['feature'])
cluster_features = [f for f in cluster_candidates if f in df_model.columns]
R(f"\nClustering features: {cluster_features}")

X_cluster = df_model[cluster_features].fillna(0).values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Find optimal k
R("\n--- Silhouette Analysis ---")
best_k = 4
best_sil = -1
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    sil = silhouette_score(X_cluster_scaled, labels, sample_size=min(5000, len(X_cluster_scaled)))
    R(f"  k={k}: silhouette={sil:.3f}")
    if sil > best_sil:
        best_sil = sil
        best_k = k

R(f"\n  Best k={best_k} (silhouette={best_sil:.3f})")

# Final clustering
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_model['segment'] = km_final.fit_predict(X_cluster_scaled)

# Profile segments
R(f"\n--- Segment Profiles ---")
for seg in range(best_k):
    seg_mask = df_model['segment'] == seg
    seg_data = df_model[seg_mask]
    R(f"\n  SEGMENT {seg} (n={len(seg_data)}, {len(seg_data)/len(df_model)*100:.1f}%):")
    R(f"    Mean NPS:  {seg_data['nps_score'].mean():.1f}")
    R(f"    Churn%:    {seg_data['is_churned'].mean()*100:.1f}%")

    # NPS group distribution
    if 'nps_group' in seg_data.columns:
        for grp in ['Promoter', 'Passive', 'Detractor']:
            pct = (seg_data['nps_group'] == grp).mean() * 100
            R(f"    {grp}:  {pct:.1f}%")

    # Key feature values
    R(f"    Key features:")
    for feat in cluster_features[:5]:
        if feat in seg_data.columns:
            R(f"      {feat:35s}: {seg_data[feat].mean():.2f}")

    # Tenure distribution
    if 'tenure_bucket' in seg_data.columns:
        tb_dist = seg_data['tenure_bucket'].value_counts(normalize=True) * 100
        top_tb = tb_dist.head(2)
        R(f"    Top tenure: {', '.join([f'{k} ({v:.0f}%)' for k,v in top_tb.items()])}")

    # Theme distribution (if available)
    if 'primary_theme' in seg_data.columns:
        theme_dist = seg_data[seg_data['primary_theme'].notna()]['primary_theme'].value_counts(normalize=True) * 100
        top_themes = theme_dist.head(3)
        if len(top_themes) > 0:
            R(f"    Top themes: {', '.join([f'{k} ({v:.0f}%)' for k,v in top_themes.items()])}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: CHURN RISK SCORING MODEL
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 6: CHURN RISK SCORING MODEL")
R("=" * 70)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

# Prepare churn modeling data
# CRITICAL: Remove leaky features that encode churn status or observation time
# - days_since_last_recharge: median 239 days, directly encodes inactive status
# - sprint_num: encodes right-censoring (Sprint 13=0% churn due to short observation)
# - recharge_regularity: collinear with days_since_last_recharge
# - total_recharges: correlates with tenure/observation window
CHURN_LEAKAGE_FEATURES = {
    'days_since_last_recharge', 'sprint_num', 'recharge_regularity',
    'total_recharges', 'recharges_before_sprint',  # observation window proxies
}
R(f"\n  LEAKAGE GUARD: Excluding {len(CHURN_LEAKAGE_FEATURES)} features from churn model")
R(f"  Excluded: {sorted(CHURN_LEAKAGE_FEATURES)}")
R(f"  Reason: These features directly encode churn status or observation time")

churn_valid = df_model['is_churned'].notna()
df_churn = df_model[churn_valid].copy()
R(f"\nChurn modeling sample: {len(df_churn)} rows")
R(f"  Churn: {df_churn['is_churned'].sum()} ({df_churn['is_churned'].mean()*100:.1f}%)")
R(f"  Active: {(1-df_churn['is_churned']).sum():.0f} ({(1-df_churn['is_churned'].mean())*100:.1f}%)")

# Right-censoring check: Sprint 13 has 0% churn (too recent)
R(f"\n  RIGHT-CENSORING CHECK:")
if 'sprint_num' in df_churn.columns:
    for sp in sorted(df_churn['sprint_num'].dropna().unique()):
        sp_d = df_churn[df_churn['sprint_num'] == sp]
        R(f"    Sprint {int(sp):2d}: n={len(sp_d):5d}, churn={sp_d['is_churned'].mean()*100:5.1f}%")
    # Exclude sprints with insufficient observation time (< 5% churn likely means censored)
    censored_sprints = []
    for sp in sorted(df_churn['sprint_num'].dropna().unique()):
        sp_d = df_churn[df_churn['sprint_num'] == sp]
        if sp_d['is_churned'].mean() < 0.05:
            censored_sprints.append(sp)
    if censored_sprints:
        n_before = len(df_churn)
        df_churn = df_churn[~df_churn['sprint_num'].isin(censored_sprints)]
        R(f"  EXCLUDING right-censored sprints {[int(s) for s in censored_sprints]}: {n_before}->{len(df_churn)} rows")
        R(f"  Revised churn rate: {df_churn['is_churned'].mean()*100:.1f}%")

# Features for churn model - include NPS score as a feature
churn_features = [f for f in model_features if f not in CHURN_LEAKAGE_FEATURES]
if 'nps_score' not in churn_features:
    churn_features = ['nps_score'] + churn_features

# Also include NPS group ordinal
if 'nps_group_ordinal' in df_churn.columns:
    churn_features = ['nps_group_ordinal'] + churn_features

# Deduplicate
churn_features = list(dict.fromkeys(churn_features))
churn_features = [f for f in churn_features if f in df_churn.columns]

X_churn = df_churn[churn_features].fillna(0).values
y_churn = df_churn['is_churned'].values.astype(int)

R(f"  Features: {len(churn_features)}")

# ── Model A: Operational features only (no NPS) ──
R("\n--- Model A: Operational features only (no NPS) ---")
ops_features = [f for f in churn_features if f not in ['nps_score', 'nps_group_ordinal'] +
                [c for c in churn_features if c.startswith('theme_')]]
X_ops = df_churn[ops_features].fillna(0).values

gb_a = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   min_samples_leaf=20, subsample=0.8, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_a = cross_val_predict(gb_a, X_ops, y_churn, cv=cv, method='predict_proba')[:, 1]
auc_a = roc_auc_score(y_churn, y_pred_a)
ap_a = average_precision_score(y_churn, y_pred_a)
R(f"  AUC-ROC: {auc_a:.4f}")
R(f"  Avg Precision: {ap_a:.4f}")

# ── Model B: Operational + NPS score ──
R("\n--- Model B: Operational + NPS score ---")
nps_features = ops_features + ['nps_score']
if 'nps_group_ordinal' in df_churn.columns:
    nps_features.append('nps_group_ordinal')
nps_features = [f for f in nps_features if f in df_churn.columns]
X_nps = df_churn[nps_features].fillna(0).values

gb_b = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   min_samples_leaf=20, subsample=0.8, random_state=42)
y_pred_b = cross_val_predict(gb_b, X_nps, y_churn, cv=cv, method='predict_proba')[:, 1]
auc_b = roc_auc_score(y_churn, y_pred_b)
ap_b = average_precision_score(y_churn, y_pred_b)
R(f"  AUC-ROC: {auc_b:.4f}")
R(f"  Avg Precision: {ap_b:.4f}")
R(f"  Lift from NPS: +{(auc_b - auc_a):.4f} AUC")

# ── Model C: Operational + NPS + Themes ──
R("\n--- Model C: Operational + NPS + Themes ---")
X_full = df_churn[churn_features].fillna(0).values

gb_c = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   min_samples_leaf=20, subsample=0.8, random_state=42)
y_pred_c = cross_val_predict(gb_c, X_full, y_churn, cv=cv, method='predict_proba')[:, 1]
auc_c = roc_auc_score(y_churn, y_pred_c)
ap_c = average_precision_score(y_churn, y_pred_c)
R(f"  AUC-ROC: {auc_c:.4f}")
R(f"  Avg Precision: {ap_c:.4f}")
R(f"  Lift from themes: +{(auc_c - auc_b):.4f} AUC (over Model B)")
R(f"  Total lift: +{(auc_c - auc_a):.4f} AUC (over Model A)")

# ── Iterative Improvement: Hyperparameter tuning on best model ──
R("\n--- Iterative Tuning (best model) ---")
best_auc = max(auc_a, auc_b, auc_c)
best_model_name = 'A' if best_auc == auc_a else ('B' if best_auc == auc_b else 'C')
X_best = X_ops if best_model_name == 'A' else (X_nps if best_model_name == 'B' else X_full)
best_features = ops_features if best_model_name == 'A' else (nps_features if best_model_name == 'B' else churn_features)
R(f"  Best base model: Model {best_model_name} (AUC={best_auc:.4f})")

# Try different configurations
configs = [
    {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.03, 'min_samples_leaf': 15, 'subsample': 0.8},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'min_samples_leaf': 10, 'subsample': 0.7},
    {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.02, 'min_samples_leaf': 20, 'subsample': 0.8},
    {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'min_samples_leaf': 25, 'subsample': 0.9},
    {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.03, 'min_samples_leaf': 15, 'subsample': 0.75},
]

best_config = None
best_tuned_auc = best_auc
best_tuned_preds = y_pred_c if best_model_name == 'C' else (y_pred_b if best_model_name == 'B' else y_pred_a)

for i, cfg in enumerate(configs):
    gb_tune = GradientBoostingClassifier(**cfg, random_state=42)
    y_pred_tune = cross_val_predict(gb_tune, X_best, y_churn, cv=cv, method='predict_proba')[:, 1]
    auc_tune = roc_auc_score(y_churn, y_pred_tune)
    R(f"  Config {i+1}: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, trees={cfg['n_estimators']} -> AUC={auc_tune:.4f}")
    if auc_tune > best_tuned_auc:
        best_tuned_auc = auc_tune
        best_config = cfg
        best_tuned_preds = y_pred_tune

# Also try Random Forest
rf_churn = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=15,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
y_pred_rf = cross_val_predict(rf_churn, X_best, y_churn, cv=cv, method='predict_proba')[:, 1]
auc_rf = roc_auc_score(y_churn, y_pred_rf)
R(f"  RF balanced: AUC={auc_rf:.4f}")
if auc_rf > best_tuned_auc:
    best_tuned_auc = auc_rf
    best_tuned_preds = y_pred_rf

R(f"\n  BEST AUC-ROC achieved: {best_tuned_auc:.4f}")

# ── Lift Table ──
R("\n--- LIFT TABLE (Risk Deciles) ---")
df_churn['risk_score'] = best_tuned_preds
df_churn['risk_decile'] = pd.qcut(df_churn['risk_score'], 10, labels=False, duplicates='drop')
# Decile 9 = highest risk, 0 = lowest risk

lift_table = df_churn.groupby('risk_decile').agg(
    count=('is_churned', 'count'),
    churners=('is_churned', 'sum'),
    churn_rate=('is_churned', 'mean'),
    avg_risk=('risk_score', 'mean'),
    avg_nps=('nps_score', 'mean')
).sort_index(ascending=False)

total_churners = df_churn['is_churned'].sum()
lift_table['pct_churners'] = lift_table['churners'] / total_churners * 100
lift_table['cum_pct_churners'] = lift_table['pct_churners'].cumsum()
lift_table['lift'] = lift_table['churn_rate'] / df_churn['is_churned'].mean()

R(f"\n  {'Decile':>7s} | {'Count':>6s} | {'Churners':>8s} | {'Churn%':>7s} | {'Lift':>5s} | {'Cum%Churn':>10s} | {'Avg NPS':>8s}")
R(f"  {'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}")
for dec, row in lift_table.iterrows():
    R(f"  {dec:7d} | {int(row['count']):6d} | {int(row['churners']):8d} | {row['churn_rate']*100:6.1f}% | {row['lift']:5.2f} | {row['cum_pct_churners']:9.1f}% | {row['avg_nps']:8.1f}")

R(f"\n  Interpretation:")
R(f"    Top 10% risk captures {lift_table.iloc[0]['cum_pct_churners']:.0f}% of all churners (lift={lift_table.iloc[0]['lift']:.1f}x)")
R(f"    Top 20% risk captures {lift_table.iloc[:2]['pct_churners'].sum():.0f}% of all churners")
R(f"    Top 30% risk captures {lift_table.iloc[:3]['pct_churners'].sum():.0f}% of all churners")

# ── Feature Importance for Churn ──
R("\n--- Feature Importance for Churn Prediction ---")
# Retrain on full data with best config
if best_config:
    gb_final = GradientBoostingClassifier(**best_config, random_state=42)
else:
    gb_final = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          min_samples_leaf=20, subsample=0.8, random_state=42)
gb_final.fit(X_best, y_churn)

churn_imp = pd.DataFrame({
    'feature': best_features,
    'importance': gb_final.feature_importances_
}).sort_values('importance', ascending=False)

R(f"\n  Top 15 churn predictors:")
R(f"  {'Feature':40s} | {'Importance':>12s} | {'Cum%':>6s}")
R(f"  {'-'*40}-+-{'-'*12}-+-{'-'*6}")
cum = 0
for _, row in churn_imp.head(15).iterrows():
    cum += row['importance']
    R(f"  {row['feature']:40s} | {row['importance']:12.4f} | {cum*100:5.1f}%")

# ── Permutation importance for churn ──
perm_churn = permutation_importance(gb_final, X_best, y_churn, n_repeats=10, random_state=42,
                                     scoring='roc_auc', n_jobs=-1)
perm_churn_df = pd.DataFrame({
    'feature': best_features,
    'perm_importance': perm_churn.importances_mean
}).sort_values('perm_importance', ascending=False)

R(f"\n  Top 15 churn predictors (permutation importance):")
for _, row in perm_churn_df.head(15).iterrows():
    R(f"    {row['feature']:40s}: {row['perm_importance']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: TENURE-STRATIFIED CHURN MODELS
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 7: TENURE-STRATIFIED CHURN MODELS")
R("=" * 70)

# Use Excel tenure buckets (1-2/3-6/6+) for stratified churn analysis
if 'tenure_excel' in df_churn.columns:
    for bucket_val in df_churn['tenure_excel'].dropna().unique():
        bucket_data = df_churn[df_churn['tenure_excel'] == bucket_val]
        n = len(bucket_data)
        n_churn = int(bucket_data['is_churned'].sum())

        if n < 100 or n_churn < 10:
            R(f"\n  [{bucket_val}] Skipping — n={n}, churners={n_churn} (insufficient)")
            continue

        R(f"\n--- [{bucket_val}] n={n}, churners={n_churn} ({n_churn/n*100:.1f}%) ---")

        X_b = bucket_data[best_features].fillna(0).values
        y_b = bucket_data['is_churned'].values.astype(int)

        gb_tb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            min_samples_leaf=10, subsample=0.8, random_state=42)
        try:
            cv_folds = min(5, n_churn)
            if cv_folds >= 2:
                y_pred_tb = cross_val_predict(gb_tb, X_b, y_b, cv=cv_folds, method='predict_proba')[:, 1]
                auc_tb = roc_auc_score(y_b, y_pred_tb)
                R(f"  AUC-ROC: {auc_tb:.4f}")
            else:
                R(f"  CV skipped (too few churners for CV)")
        except Exception as e:
            R(f"  CV failed: {str(e)[:100]}")

        # Top drivers for this tenure
        gb_tb.fit(X_b, y_b)
        imp_tb = pd.DataFrame({
            'feature': best_features,
            'importance': gb_tb.feature_importances_
        }).sort_values('importance', ascending=False)

        R(f"  Top 5 churn drivers:")
        for _, row in imp_tb.head(5).iterrows():
            R(f"    {row['feature']:40s}: {row['importance']:.4f}")
else:
    R("  No tenure_excel column available.")

# ══════════════════════════════════════════════════════════════════════
# STEP 8: DOSE-RESPONSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 8: DOSE-RESPONSE ANALYSIS")
R("=" * 70)

# NPS score vs churn rate
R(f"\n--- NPS Score → Churn Rate (excluding right-censored sprints) ---")
nps_churn = df_churn.groupby('nps_score')['is_churned'].agg(['mean', 'count'])
R(f"  {'NPS':>4s} | {'Churn%':>7s} | {'Count':>6s}")
R(f"  {'-'*4}-+-{'-'*7}-+-{'-'*6}")
for score, row in nps_churn.iterrows():
    bar = '#' * int(row['mean'] * 50)
    R(f"  {int(score):4d} | {row['mean']*100:6.1f}% | {int(row['count']):6d} {bar}")

# Key continuous features dose-response
dose_features = ['tenure_days', 'recharges_before_sprint', 'tickets_before_3m']
# Add Snowflake/sprint/Industry Expert features
for f in ['total_recharges', 'avg_recharge_amount', 'sla_compliance_pct',
          'total_tickets', 'PEAK_UPTIME_PCT', 'TOTAL_IVR_CALLS', 'INBOUND_CALLS',
          'AVG_TIMES_REOPENED', 'FCR_RATE', 'MAX_TICKETS_SAME_ISSUE',
          'support_effort_index', 'network_quality_index',
          'partner_avg_nps', 'partner_churn_rate',
          'resolution_vs_partner', 'missed_call_ratio',
          'install_tat_hours_w', 'FAILURE_RATE_PCT']:
    if f in df_churn.columns and df_churn[f].notna().sum() > 100:
        dose_features.append(f)

for feat in dose_features:
    if feat not in df_churn.columns or df_churn[feat].notna().sum() < 100:
        continue
    R(f"\n--- {feat} → NPS & Churn ---")
    valid = df_churn[df_churn[feat].notna()]
    try:
        bins = pd.qcut(valid[feat], 5, duplicates='drop')
        grouped = valid.groupby(bins).agg(
            mean_nps=('nps_score', 'mean'),
            churn_rate=('is_churned', 'mean'),
            count=('nps_score', 'count')
        )
        R(f"  {'Range':>35s} | {'Mean NPS':>8s} | {'Churn%':>7s} | {'Count':>6s}")
        R(f"  {'-'*35}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")
        for rng, row in grouped.iterrows():
            R(f"  {str(rng):>35s} | {row['mean_nps']:8.1f} | {row['churn_rate']*100:6.1f}% | {int(row['count']):6d}")
    except Exception as e:
        R(f"  Could not bin: {str(e)[:100]}")

# ══════════════════════════════════════════════════════════════════════
# STEP 9: HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 9: HYPOTHESIS TESTING")
R("=" * 70)

# H1: Network uptime is #1 NPS driver
R("\n--- H1: Network uptime is #1 NPS driver ---")
R("  CAVEAT: avg_uptime_pct measures PARTNER FLEET health, not customer-experienced uptime")
R("  Testing with PEAK_UPTIME_PCT (peak-hour quality) as better proxy")
uptime_vars = ['PEAK_UPTIME_PCT', 'avg_uptime_pct', 'OUTAGE_EVENTS', 'network_quality_index']
for uv in uptime_vars:
    if uv in consensus:
        R(f"  {uv}: IN consensus top-10 across {len(consensus[uv])} methods")
    elif uv in all_top10:
        R(f"  {uv}: in top-10 of {all_top10[uv]} only")
    else:
        R(f"  {uv}: NOT in any top-10 list")
R("  NOTE: Support effort (tickets, IVR, reopenings) may dominate over network quality")

# H2: Install TAT is top-3 for first 30 days
R("\n--- H2: Install TAT is top-3 for customers in first 30 days ---")
if 'install_tat_hours' in df_churn.columns:
    early = df_churn[df_churn['tenure_days'] <= 30]
    if len(early) > 50:
        corr = early[['install_tat_hours', 'nps_score']].corr().iloc[0, 1]
        R(f"  Correlation (install_tat vs NPS, tenure<=30d): {corr:.3f}")
else:
    R(f"  CANNOT TEST: Install TAT data not available")

# H3: SLA compliance > ticket volume
R("\n--- H3: SLA compliance matters more than ticket volume ---")
if 'sla_compliance_pct' in df_churn.columns and 'total_tickets' in df_churn.columns:
    sla_corr = df_churn[['sla_compliance_pct', 'nps_score']].corr().iloc[0, 1]
    vol_corr = df_churn[['total_tickets', 'nps_score']].corr().iloc[0, 1]
    R(f"  SLA compliance ↔ NPS: r={sla_corr:.3f}")
    R(f"  Ticket volume ↔ NPS:  r={vol_corr:.3f}")
    R(f"  {'SUPPORTED' if abs(sla_corr) > abs(vol_corr) else 'NOT SUPPORTED'}: SLA {'>' if abs(sla_corr)>abs(vol_corr) else '<'} volume")
else:
    R(f"  CANNOT TEST: Need SLA compliance data")

# H6: Detractors with network complaints churn at 2x vs pricing complaints
R("\n--- H6: Network complaint Detractors churn at 2x vs pricing complaint Detractors ---")
if 'primary_theme' in df_churn.columns:
    det = df_churn[df_churn['nps_group'] == 'Detractor']
    net_themes = ['slow_speed', 'disconnection_frequency', 'internet_down_outage', 'range_coverage']
    price_themes = ['pricing_expensive', 'pricing_affordable', 'billing_28day']

    net_det = det[det['primary_theme'].isin(net_themes)]
    price_det = det[det['primary_theme'].isin(price_themes)]

    if len(net_det) > 10 and len(price_det) > 10:
        net_churn = net_det['is_churned'].mean()
        price_churn = price_det['is_churned'].mean()
        ratio = net_churn / price_churn if price_churn > 0 else float('inf')
        R(f"  Network complaint Detractors: {net_churn*100:.1f}% churn (n={len(net_det)})")
        R(f"  Pricing complaint Detractors: {price_churn*100:.1f}% churn (n={len(price_det)})")
        R(f"  Ratio: {ratio:.2f}x")
        R(f"  {'SUPPORTED' if ratio >= 1.5 else 'PARTIALLY SUPPORTED' if ratio >= 1.2 else 'NOT SUPPORTED'}")

# H7: Autopay users have higher NPS
R("\n--- H7: Autopay users have higher NPS than cash/online ---")
if 'payment_mode' in df_churn.columns:
    pm_nps = df_churn.groupby('payment_mode')['nps_score'].agg(['mean', 'count'])
    for mode, row in pm_nps.iterrows():
        R(f"  {mode:15s}: mean NPS={row['mean']:.1f} (n={int(row['count'])})")
elif 'cash_online' in df_churn.columns:
    co_nps = df_churn.groupby('cash_online')['nps_score'].agg(['mean', 'count'])
    for mode, row in co_nps.iterrows():
        R(f"  {mode:15s}: mean NPS={row['mean']:.1f} (n={int(row['count'])})")
else:
    R(f"  CANNOT TEST: Payment mode data not available")

# ══════════════════════════════════════════════════════════════════════
# STEP 9B: SPRINT-LEVEL TEMPORAL ANALYSIS (Time Dimension)
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 9B: SPRINT-LEVEL TEMPORAL ANALYSIS")
R("=" * 70)
R("\n  How do NPS drivers change OVER TIME across sprints?")

if 'sprint_num' in df_churn.columns:
    sprint_nums = sorted(df_churn['sprint_num'].dropna().unique())
    R(f"\n  Sprints available: {sprint_nums}")

    # Track how key metrics evolve by sprint
    sprint_metrics = []
    for sp in sprint_nums:
        sp_data = df_churn[df_churn['sprint_num'] == sp]
        if len(sp_data) < 30:
            continue
        row = {'sprint': int(sp), 'n': len(sp_data),
               'mean_nps': sp_data['nps_score'].mean(),
               'churn_rate': sp_data['is_churned'].mean() * 100}
        # Add key feature means
        for feat in ['TOTAL_IVR_CALLS', 'AVG_TIMES_REOPENED', 'sla_compliance_pct',
                     'PEAK_UPTIME_PCT', 'support_effort_index', 'total_tickets',
                     'partner_at_risk', 'FCR_RATE']:
            if feat in sp_data.columns:
                row[feat] = sp_data[feat].mean()
        sprint_metrics.append(row)

    if sprint_metrics:
        sm_df = pd.DataFrame(sprint_metrics)
        R(f"\n  {'Sprint':>7s} | {'N':>5s} | {'NPS':>5s} | {'Churn%':>6s} | {'IVR':>6s} | {'Reopen':>6s} | {'SLA%':>5s} | {'Risk%':>6s}")
        R(f"  {'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}")
        for _, row in sm_df.iterrows():
            ivr = f"{row.get('TOTAL_IVR_CALLS', 0):.1f}" if 'TOTAL_IVR_CALLS' in row else 'N/A'
            reopen = f"{row.get('AVG_TIMES_REOPENED', 0):.3f}" if 'AVG_TIMES_REOPENED' in row else 'N/A'
            sla = f"{row.get('sla_compliance_pct', 0):.1f}" if 'sla_compliance_pct' in row else 'N/A'
            risk = f"{row.get('partner_at_risk', 0)*100:.1f}" if 'partner_at_risk' in row else 'N/A'
            R(f"  {int(row['sprint']):7d} | {int(row['n']):5d} | {row['mean_nps']:5.1f} | {row['churn_rate']:5.1f}% | {ivr:>6s} | {reopen:>6s} | {sla:>5s} | {risk:>6s}")

    # Sprint-level KDA: run RF for each sprint to see if drivers are STABLE or SHIFTING
    R("\n--- Sprint-Level Top-3 Drivers (are drivers stable or shifting?) ---")
    sprint_drivers = {}
    for sp in sprint_nums:
        sp_data = df_churn[df_churn['sprint_num'] == sp]
        if len(sp_data) < 50:
            continue
        X_sp = sp_data[model_features].fillna(0).values
        y_sp = sp_data['nps_score'].values
        if len(np.unique(y_sp)) < 2:
            continue
        rf_sp = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                       random_state=42, n_jobs=-1)
        rf_sp.fit(X_sp, y_sp)
        imp_sp = pd.Series(rf_sp.feature_importances_, index=model_features).sort_values(ascending=False)
        top3 = imp_sp.head(3)
        sprint_drivers[int(sp)] = list(top3.index)
        R(f"  Sprint {int(sp)}: {', '.join([f'{f}({v:.3f})' for f, v in top3.items()])}")

    # Check driver stability
    if sprint_drivers:
        from collections import Counter
        all_driver_mentions = Counter()
        for sp, drivers in sprint_drivers.items():
            for d in drivers:
                all_driver_mentions[d] += 1
        R(f"\n  DRIVER STABILITY (appears in top-3 across sprints):")
        for feat, count in all_driver_mentions.most_common(10):
            pct = count / len(sprint_drivers) * 100
            label = "STABLE" if pct >= 60 else "MODERATE" if pct >= 30 else "SPORADIC"
            R(f"    {feat:40s}: {count}/{len(sprint_drivers)} sprints ({pct:.0f}%) [{label}]")

# ══════════════════════════════════════════════════════════════════════
# STEP 10: ASSIGN RISK SCORES TO ALL RESPONDENTS
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("STEP 10: RISK SCORE ASSIGNMENT")
R("=" * 70)

# Train final model on all data (using the non-censored sample)
gb_final.fit(X_best, y_churn)

# Calibrate the model for better probability estimates
from sklearn.calibration import CalibratedClassifierCV
gb_calib = CalibratedClassifierCV(gb_final, method='isotonic', cv=3)
gb_calib.fit(X_best, y_churn)

# Score ALL respondents (including those without churn labels)
# Use all model features (without leakage features)
X_all = df[best_features].fillna(0).values
risk_scores = gb_calib.predict_proba(X_all)[:, 1]
df['churn_risk_score'] = risk_scores
df['churn_risk_pct'] = (risk_scores * 100).round(1)

# Risk categories — thresholds based on actual churn rate distribution
base_churn = df_churn['is_churned'].mean()
df['risk_category'] = pd.cut(df['churn_risk_score'],
                              bins=[0, base_churn * 0.5, base_churn, base_churn * 1.5, 1.0],
                              labels=['Low', 'Medium', 'High', 'Critical'])

R("\n--- Risk Score Distribution ---")
risk_dist = df['risk_category'].value_counts()
for cat in ['Low', 'Medium', 'High', 'Critical']:
    if cat in risk_dist.index:
        cnt = risk_dist[cat]
        R(f"  {cat:10s}: {cnt:6d} ({cnt/len(df)*100:.1f}%)")

# Risk by NPS group
R("\n--- Risk Category by NPS Group ---")
risk_nps = pd.crosstab(df['risk_category'], df['nps_group'], normalize='columns') * 100
if 'Promoter' in risk_nps.columns and 'Detractor' in risk_nps.columns:
    R(f"  {'Category':10s} | {'Promoter':>10s} | {'Passive':>10s} | {'Detractor':>10s}")
    R(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for cat in ['Low', 'Medium', 'High', 'Critical']:
        if cat in risk_nps.index:
            prom = risk_nps.at[cat, 'Promoter'] if 'Promoter' in risk_nps.columns else 0
            pas = risk_nps.at[cat, 'Passive'] if 'Passive' in risk_nps.columns else 0
            det = risk_nps.at[cat, 'Detractor'] if 'Detractor' in risk_nps.columns else 0
            R(f"  {cat:10s} | {prom:9.1f}% | {pas:9.1f}% | {det:9.1f}%")

# Save final dataset with risk scores
output_path = os.path.join(DATA, "nps_with_risk_scores.csv")
df.to_csv(output_path, index=False, encoding='utf-8-sig')
R(f"\n  Saved: {output_path}")
R(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
R("\n" + "=" * 70)
R("PHASE 4 SUMMARY")
R("=" * 70)

R(f"\n  Dataset: {len(df)} respondents, {len(df.columns)} features")
R(f"  NPS KDA R-squared (RF): {rf_r2:.3f}")
R(f"  NPS KDA R-squared (GB): {gb_r2:.3f}")
R(f"  Churn AUC-ROC (Model A - ops only): {auc_a:.4f}")
R(f"  Churn AUC-ROC (Model B - ops+NPS): {auc_b:.4f}")
R(f"  Churn AUC-ROC (Model C - ops+NPS+themes): {auc_c:.4f}")
R(f"  Best tuned AUC-ROC: {best_tuned_auc:.4f}")
R(f"  Customer segments: {best_k}")
R(f"  Consensus NPS drivers (>=2 methods): {len(consensus)}")

R(f"\n  Top consensus drivers:")
for feat, methods in consensus_sorted[:10]:
    R(f"    {feat:40s} [{len(methods)} methods: {', '.join(methods)}]")

save_report()

print("\n" + "=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)
