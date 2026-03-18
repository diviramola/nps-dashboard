"""Check for data leakage: fill rates and means by churn status."""
import pandas as pd
import numpy as np
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv(r'C:\Users\nikhi\wiom-nps-analysis\data\nps_enriched_v2.csv', low_memory=False)

churn_col = 'is_churned'
key_feats = [
    'ul1_avg_daily_devices', 'ul1_avg_active_hours', 'ul1_avg_daily_data_gb',
    'ul1_avg_daily_sessions', 'ul1_avg_distinct_ips', 'ul1_avg_peak_hour_ratio',
    'ul1_avg_night_day_ratio', 'ul1_avg_session_hours', 'ul1_avg_usage_volatility',
    'connection_instability',
    'cis_avg_customer_uptime_pct', 'cis_avg_peak_uptime_pct', 'cis_peak_interruption_rate',
    'cis_avg_stable_ping_ratio',
    'sc_avg_latest_speed', 'sc_speed_gap_pct', 'sc_avg_rxpower',
    'tk_total_tickets', 'tk_sla_compliance', 'tk_avg_resolution_mins',
    'avg_uptime_pct', 'has_tickets', 'total_tickets',
]

print("=" * 90)
print("LEAKAGE DIAGNOSTIC: Fill rates by churn status")
print("  If churned customers have much lower fill → median imputation creates leakage")
print("=" * 90)
print(f"{'Feature':35s} | {'Active':>10s} | {'Churned':>10s} | {'Gap':>8s} | {'Verdict'}")
print("-" * 90)
for f in key_feats:
    if f in df.columns:
        active_fill = df.loc[df[churn_col]==0, f].notna().mean()
        churn_fill = df.loc[df[churn_col]==1, f].notna().mean()
        diff = (active_fill - churn_fill) * 100
        verdict = "LEAKAGE!" if abs(diff) > 10 else "CAUTION" if abs(diff) > 5 else "OK"
        print(f'{f:35s} | {active_fill*100:8.1f}% | {churn_fill*100:8.1f}% | {diff:+7.1f}pp | {verdict}')

print()
print("=" * 90)
print("LEAKAGE DIAGNOSTIC: Mean values by churn status (non-null only)")
print("  If ratio is extreme (>3x), the feature may proxy for churn itself")
print("=" * 90)
print(f"{'Feature':35s} | {'Active':>12s} | {'Churned':>12s} | {'A/C Ratio':>10s} | {'Verdict'}")
print("-" * 90)
for f in key_feats:
    if f in df.columns:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        active_mean = df.loc[df[churn_col]==0, f].mean()
        churn_mean = df.loc[df[churn_col]==1, f].mean()
        if churn_mean and churn_mean != 0:
            ratio = active_mean / churn_mean
        else:
            ratio = float('inf')
        verdict = "LEAKAGE!" if ratio > 5 or ratio < 0.2 else "CAUTION" if ratio > 3 or ratio < 0.33 else "OK"
        print(f'{f:35s} | {active_mean:12.3f} | {churn_mean:12.3f} | {ratio:10.2f}x | {verdict}')

# Also check: how do medians compare to NaN-imputed values?
print()
print("=" * 90)
print("IMPUTATION CHECK: What values do NaN churned customers get?")
print("=" * 90)
from sklearn.impute import SimpleImputer
for f in ['ul1_avg_daily_devices', 'ul1_avg_active_hours', 'cis_avg_customer_uptime_pct']:
    if f in df.columns:
        vals = pd.to_numeric(df[f], errors='coerce')
        overall_median = vals.median()
        active_median = vals[df[churn_col]==0].median()
        churn_median = vals[df[churn_col]==1].median()
        churn_nan_pct = vals[df[churn_col]==1].isna().mean() * 100
        print(f"{f}: overall_median={overall_median:.3f}, active_median={active_median:.3f}, churn_median={churn_median:.3f}, churn_NaN={churn_nan_pct:.1f}%")
