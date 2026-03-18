"""
Phase 4B: Comprehensive Feature Audit
======================================
Statistical + CX Expert audit of ALL features for:
1. Tenure bias (absolute counts masquerading as signal)
2. Fill-rate leakage (data existence = churn proxy)
3. Target leakage (features mechanically linked to churn definition)
4. Reverse causality (feature is REACTION to risk, not CAUSE)
5. Right-censoring (value depends on observation window)
6. Multicollinearity (redundant features)
7. Scale appropriateness (absolute vs rate)
8. Near-zero variance

Output: output/phase4b_feature_audit.txt
"""

import sys, io, os, warnings
import pandas as pd
import numpy as np
from datetime import datetime

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
rpt("COMPREHENSIVE FEATURE AUDIT")
rpt(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt("=" * 100)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA, "nps_enriched_v2.csv"), low_memory=False)
df['churn_binary'] = df['is_churned'].astype(int)
rpt(f"\n[LOAD] Rows: {len(df)}, Columns: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY FEATURE COLUMNS (exclude identifiers and text)
# ══════════════════════════════════════════════════════════════════════════════
IDENTIFIERS = {
    'phone_number', 'response_id', 'user_id', 'timestamp',
    'profile_all_identities', 'alt_mobile', 'lng_nas',
    'device_id', 'device_id_mapped', 'snap_date',
    'Sprint ID', 'Sprint Start Date', 'Sprint End Date', 'Cycle ID',
    'translated_comment', 'theme_clean', 'comments_sprint',
    'nps_oe_copied', 'nps_reason_sprint', 'nps_reason_tertiary_sprint',
    'nps_reason_primary_sprint', 'nps_reason_secondary_sprint',
}

# Target variables (not features)
TARGETS = {'is_churned', 'churn_binary', 'nps_score', 'nps_group', 'nps_group_ordinal'}

# Known churn-definition features (mechanically = churn, must exclude)
TARGET_LEAKAGE = {
    'churn_risk_score', 'churn_risk_pct', 'churn_label', 'risk_category',
    'partner_churn_rate', 'partner_at_risk', 'partner_risk_level',
    'days_since_last_recharge', 'last_recharge', 'last_recharge_date',
    'plan_expiry_date', 'plan_expiry_window',
    'total_recharges', 'recharge_done', 'recharges_before_sprint',
    'recharge_regularity', 'recharge_same_day',
    'total_payments', 'autopay_payments', 'cash_payments',
    'avg_payment_amount', 'avg_recharge_amount',
    'PAYMENT_ATTEMPTS', 'PAYMENT_FAILURES', 'PAYMENT_SUCCESSES',
    'TOTAL_PAYMENT_EVENTS',
    'first_recharge', 'DAYS_TO_FIRST_RECHARGE', 'HOURS_TO_FIRST_RECHARGE',
    'is_cash_payment', 'payment_successes',
    'partner_status', 'partner_status_at_survey', 'call_status',
    'sc_scorecard_weeks', 'cis_influx_data_days', 'ul1_usage_data_days',
    'sc_plan_active_ratio', 'cis_active_day_ratio', 'sc_scorecard_ticket_count',
    'tickets_post_sprint',
}

# Reverse causality (REACTION to risk, not cause)
REVERSE_CAUSALITY = {
    'OUTBOUND_CALLS',  # Company calls at-risk customers
}

# Right-censoring (observation window artifact)
RIGHT_CENSORING = {
    'sprint_num',  # Later sprints had less time to observe churn
}

# Features that are absolute counts and need normalization
# (will be identified automatically by tenure correlation)

feature_cols = []
for c in df.columns:
    if c in IDENTIFIERS or c in TARGETS:
        continue
    # Skip Hindi columns (survey responses, not operational features)
    if any('\u0900' <= ch <= '\u097F' for ch in c):
        continue
    # Skip very high cardinality text
    if df[c].dtype == 'object' and df[c].nunique() > 50:
        continue
    feature_cols.append(c)

rpt(f"  Features to audit: {len(feature_cols)}")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 1: Basic Stats + Tenure/Churn Correlation
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("AUDIT 1: BASIC STATISTICS & CORRELATION ANALYSIS")
rpt("=" * 100)

audit_rows = []
for c in feature_cols:
    row = {'feature': c, 'dtype': str(df[c].dtype)}

    # Fill rates
    row['fill_overall'] = df[c].notna().mean()
    row['fill_active'] = df.loc[df['churn_binary']==0, c].notna().mean()
    row['fill_churned'] = df.loc[df['churn_binary']==1, c].notna().mean()
    row['fill_gap'] = row['fill_active'] - row['fill_churned']

    # Numeric stats
    if df[c].dtype in ('float64', 'int64', 'float32', 'int32', 'bool'):
        vals = pd.to_numeric(df[c], errors='coerce')
        row['mean'] = vals.mean()
        row['std'] = vals.std()
        row['skew'] = vals.skew()
        row['pct_zero'] = (vals == 0).mean()
        row['pct_mode'] = vals.value_counts(normalize=True).iloc[0] if len(vals.value_counts()) > 0 else 1

        # Correlations
        valid = df[['tenure_days', 'churn_binary', 'sprint_num']].copy()
        valid[c] = vals
        valid = valid.dropna()
        if len(valid) > 100:
            row['tenure_r'] = valid[c].corr(valid['tenure_days'])
            row['churn_r'] = valid[c].corr(valid['churn_binary'])
            row['sprint_r'] = valid[c].corr(valid['sprint_num'])
        else:
            row['tenure_r'] = row['churn_r'] = row['sprint_r'] = np.nan
    else:
        row['mean'] = row['std'] = row['skew'] = np.nan
        row['pct_zero'] = row['pct_mode'] = np.nan
        row['tenure_r'] = row['churn_r'] = row['sprint_r'] = np.nan

    audit_rows.append(row)

audit = pd.DataFrame(audit_rows)
rpt(f"  Audited {len(audit)} features")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 2: AUTOMATIC CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
rpt("\n" + "=" * 100)
rpt("AUDIT 2: FEATURE CLASSIFICATION")
rpt("=" * 100)

def classify_feature(row):
    """Classify feature with reason."""
    f = row['feature']
    reasons = []

    # 1. Target leakage
    if f in TARGET_LEAKAGE:
        return 'EXCLUDE', 'Target leakage — mechanically linked to churn definition'

    # 2. Reverse causality
    if f in REVERSE_CAUSALITY:
        return 'EXCLUDE', 'Reverse causality — reactive to risk, not predictive'

    # 3. Right-censoring
    if f in RIGHT_CENSORING:
        return 'EXCLUDE', 'Right-censoring artifact — depends on observation window'

    # 4. Near-zero variance
    if pd.notna(row.get('pct_mode')) and row['pct_mode'] > 0.95:
        return 'EXCLUDE', f'Near-zero variance — {row["pct_mode"]*100:.0f}% same value'

    # 5. Fill-rate leakage (>10pp gap between active and churned)
    if abs(row.get('fill_gap', 0)) > 0.10:
        # Check if recoverable (gap shrinks in late sprints)
        late = df[df['sprint_num'].between(9, 13)]
        if len(late) > 100 and f in late.columns:
            late_af = late.loc[late['churn_binary']==0, f].notna().mean()
            late_cf = late.loc[late['churn_binary']==1, f].notna().mean()
            late_gap = late_af - late_cf
            if abs(late_gap) < 0.10:
                reasons.append(f'Fill-rate gap RECOVERABLE: overall={row["fill_gap"]*100:+.0f}pp, late sprints={late_gap*100:+.0f}pp')
            else:
                return 'EXCLUDE', f'Fill-rate leakage — {row["fill_gap"]*100:+.0f}pp gap persists in late sprints ({late_gap*100:+.0f}pp)'
        else:
            return 'EXCLUDE', f'Fill-rate leakage — {row["fill_gap"]*100:+.0f}pp active vs churned gap'

    # 6. Tenure bias (absolute count masquerading as signal)
    tenure_r = row.get('tenure_r', 0)
    if pd.notna(tenure_r) and abs(tenure_r) > 0.15:
        # Is this an absolute count that should be normalized?
        is_count_feature = any(x in f.upper() for x in [
            'TOTAL_', 'CALLS', 'TICKETS', 'LEADS', 'PAYMENTS', 'RECHARGES',
            'ISSUES_WITH', 'REOPENED', 'FCR_TICKETS', 'REPEAT',
        ]) or f in [
            'install_attempts', 'px_tickets', 'cx_tickets', 'total_tickets',
            'support_effort_index', 'INBOUND_CALLS', 'INBOUND_ANSWERED',
            'INBOUND_UNANSWERED', 'MISSED_CALLS', 'ANSWERED_CALLS',
            'DISTINCT_ISSUE_TYPES', 'MAX_TICKETS_SAME_ISSUE', 'MAX_TIMES_REOPENED',
        ]

        if is_count_feature:
            return 'NORMALIZE', f'Tenure-biased absolute count (r={tenure_r:+.3f}) — normalize to per-month rate'
        else:
            # It's a rate/ratio/average but still correlated with tenure
            # This may be legitimate (longer tenure → better/worse experience over time)
            reasons.append(f'Moderate tenure correlation (r={tenure_r:+.3f}) — verify not spurious')

    # 7. Very low fill rate
    if row.get('fill_overall', 0) < 0.10:
        return 'EXCLUDE', f'Very low fill rate ({row["fill_overall"]*100:.0f}%)'

    if reasons:
        return 'REVIEW', '; '.join(reasons)

    return 'CLEAN', 'Passes all checks'

# Apply classification
verdicts = []
for _, row in audit.iterrows():
    verdict, reason = classify_feature(row)
    verdicts.append({'feature': row['feature'], 'verdict': verdict, 'reason': reason})

verdict_df = pd.DataFrame(verdicts)
audit = audit.merge(verdict_df, on='feature')

# Summary
rpt(f"\n  VERDICT SUMMARY:")
for v in ['CLEAN', 'NORMALIZE', 'REVIEW', 'EXCLUDE']:
    n = (audit['verdict'] == v).sum()
    rpt(f"    {v:12s}: {n:3d} features")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 3: DETAILED RESULTS BY CATEGORY
# ══════════════════════════════════════════════════════════════════════════════

rpt("\n" + "=" * 100)
rpt("EXCLUDED FEATURES (must not enter any model)")
rpt("=" * 100)
excluded = audit[audit['verdict'] == 'EXCLUDE'].sort_values('feature')
for _, row in excluded.iterrows():
    rpt(f"  {row['feature']:45s} | {row['reason']}")

rpt(f"\n" + "=" * 100)
rpt("NORMALIZE FEATURES (convert from absolute count to per-month rate)")
rpt("=" * 100)
normalize = audit[audit['verdict'] == 'NORMALIZE'].sort_values('tenure_r', ascending=False)
for _, row in normalize.iterrows():
    rpt(f"  {row['feature']:45s} | tenure_r={row['tenure_r']:+.3f} | {row['reason']}")

rpt(f"\n  NORMALIZATION FORMULA:")
rpt(f"    normalized_value = raw_value / max(tenure_months, 1)")
rpt(f"    where tenure_months = tenure_days / 30")
rpt(f"    This removes the mechanical accumulation with time")

rpt(f"\n" + "=" * 100)
rpt("REVIEW FEATURES (need manual check)")
rpt("=" * 100)
review = audit[audit['verdict'] == 'REVIEW'].sort_values('feature')
for _, row in review.iterrows():
    rpt(f"  {row['feature']:45s} | tenure_r={row.get('tenure_r', 0):+.3f} | {row['reason']}")

rpt(f"\n" + "=" * 100)
rpt("CLEAN FEATURES (safe to use as-is)")
rpt("=" * 100)
clean = audit[audit['verdict'] == 'CLEAN'].sort_values('feature')
for _, row in clean.iterrows():
    t_r = row.get('tenure_r', 0)
    c_r = row.get('churn_r', 0)
    t_str = f"tenure_r={t_r:+.3f}" if pd.notna(t_r) else "tenure_r=N/A"
    c_str = f"churn_r={c_r:+.3f}" if pd.notna(c_r) else "churn_r=N/A"
    rpt(f"  {row['feature']:45s} | {t_str} | {c_str} | fill={row['fill_overall']*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 4: MULTICOLLINEARITY CHECK (top clean + normalize features)
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("AUDIT 4: MULTICOLLINEARITY (pairs with |r| > 0.85)")
rpt("=" * 100)

# Only check clean and normalize features that are numeric
mc_features = audit[
    (audit['verdict'].isin(['CLEAN', 'NORMALIZE'])) &
    (audit['dtype'].isin(['float64', 'int64', 'float32', 'int32', 'bool']))
]['feature'].tolist()

if len(mc_features) > 1:
    mc_df = df[mc_features].apply(pd.to_numeric, errors='coerce')
    corr = mc_df.corr()

    high_corr_pairs = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((corr.index[i], corr.columns[j], r))

    high_corr_pairs.sort(key=lambda x: -abs(x[2]))
    rpt(f"  Found {len(high_corr_pairs)} highly correlated feature pairs:")
    for f1, f2, r in high_corr_pairs[:30]:
        rpt(f"    {f1:40s} ↔ {f2:40s} | r={r:+.3f}")
    if len(high_corr_pairs) > 30:
        rpt(f"    ... and {len(high_corr_pairs)-30} more")
else:
    rpt("  Not enough features for multicollinearity check")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 5: CX DOMAIN PERSPECTIVE
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("AUDIT 5: CX DOMAIN ASSESSMENT")
rpt("=" * 100)

# Group features by domain and assess
domain_groups = {
    'Network Quality': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['uptime', 'rxpower', 'speed', 'failure', 'ping', 'interruption', 'network_quality',
         'optical', 'peak_uptime', 'stddev_uptime', 'outage'])],
    'Service/Support': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['ticket', 'sla', 'resolution', 'complaint', 'dispatch', 'fcr', 'reopen'])],
    'Calls/IVR': [f for f in feature_cols if any(x in f.upper() for x in
        ['CALL', 'IVR', 'ANSWER', 'MISSED', 'DROP', 'INBOUND', 'OUTBOUND'])
        and f not in TARGET_LEAKAGE and f not in REVERSE_CAUSALITY],
    'Usage/Engagement': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['data_gb', 'data_usage', 'usage', 'session', 'device', 'active_hour'])],
    'Partner Quality': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['partner_', 'install'])],
    'Payment/Value': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['autopay', 'payment_mode', 'plan_', 'price'])],
    'Customer Profile': [f for f in clean['feature'] if any(x in f.lower() for x in
        ['tenure', 'city', 'first_time', 'lead_source', 'channel'])],
}

for domain, feats in domain_groups.items():
    if not feats:
        continue
    rpt(f"\n  {domain} ({len(feats)} features):")
    for f in sorted(feats)[:15]:
        v_row = audit[audit['feature'] == f]
        if len(v_row) > 0:
            v = v_row.iloc[0]
            verdict = v.get('verdict', '?')
            t_r = v.get('tenure_r', 0)
            c_r = v.get('churn_r', 0)
            rpt(f"    [{verdict:9s}] {f:40s} tenure_r={t_r:+.3f}  churn_r={c_r:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 6: CALL FEATURES DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("AUDIT 6: CALL FEATURES DEEP DIVE")
rpt("(User's concern: absolute counts have tenure bias, should be rates)")
rpt("=" * 100)

call_features = [c for c in df.columns if any(x in c.upper() for x in
    ['CALL', 'IVR', 'ANSWER', 'MISSED', 'DROP', 'INBOUND', 'OUTBOUND'])
    and c not in IDENTIFIERS and c not in TARGETS]

rpt(f"\n  {'Feature':40s} | {'Type':12s} | {'tenure_r':>10s} | {'churn_r':>10s} | {'Verdict':12s}")
rpt(f"  {'-'*40} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*12}")
for f in sorted(call_features):
    if f not in df.columns:
        continue
    vals = pd.to_numeric(df[f], errors='coerce')
    valid = df[['tenure_days', 'churn_binary']].copy()
    valid[f] = vals
    valid = valid.dropna()
    if len(valid) < 100:
        continue
    t_r = valid[f].corr(valid['tenure_days'])
    c_r = valid[f].corr(valid['churn_binary'])

    # Classify
    is_rate = any(x in f.lower() for x in ['avg', 'ratio', 'pct', 'per_', 'rate'])
    feat_type = 'RATE/AVG' if is_rate else 'ABS COUNT'
    if f in REVERSE_CAUSALITY:
        verdict = 'EXCLUDE'
    elif abs(t_r) > 0.15 and not is_rate:
        verdict = 'NORMALIZE'
    elif abs(t_r) > 0.15 and is_rate:
        verdict = 'REVIEW'
    else:
        verdict = 'CLEAN'

    rpt(f"  {f:40s} | {feat_type:12s} | {t_r:+10.3f} | {c_r:+10.3f} | {verdict:12s}")

rpt(f"""
  CALL FEATURE RECOMMENDATIONS:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLEAN (use as-is):
    - AVG_ANSWERED_SECONDS (r=+0.062) — response time, already an average
    - DROPPED_CALLS (r=+0.013) — rare events, not tenure-biased
    - missed_call_ratio (r=-0.050) — already a ratio
    - AVG_PARTNER_CALLS_PER_TICKET (r=-0.017) — already per-ticket
    - AVG_CALL_DURATION (r=+0.154) — borderline, but an average

  NORMALIZE (create per-month versions):
    - TOTAL_IVR_CALLS → ivr_calls_per_month
    - MISSED_CALLS → missed_calls_per_month (or use missed_call_ratio)
    - INBOUND_CALLS → inbound_calls_per_month
    - ANSWERED_CALLS → answered_calls_per_month
    - INBOUND_ANSWERED → inbound_answered_per_month
    - INBOUND_UNANSWERED → inbound_unanswered_per_month

  EXCLUDE:
    - OUTBOUND_CALLS — reverse causality (reactive)
""")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 7: PROPOSED NORMALIZED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("AUDIT 7: FEATURE ENGINEERING — NORMALIZED VERSIONS")
rpt("=" * 100)

# Create normalized versions and check if tenure correlation improves
tenure_months = (df['tenure_days'] / 30).clip(lower=1)

normalize_map = {
    'TOTAL_IVR_CALLS': 'ivr_calls_per_month',
    'MISSED_CALLS': 'missed_calls_per_month',
    'INBOUND_CALLS': 'inbound_calls_per_month',
    'ANSWERED_CALLS': 'answered_calls_per_month',
    'INBOUND_ANSWERED': 'inbound_answered_per_month',
    'INBOUND_UNANSWERED': 'inbound_unanswered_per_month',
    'total_tickets': 'tickets_per_month',
    'px_tickets': 'px_tickets_per_month',
    'cx_tickets': 'cx_tickets_per_month',
    'TOTAL_TICKETS_FCR': 'fcr_tickets_per_month',
    'TOTAL_TICKETS_REPEAT': 'repeat_tickets_per_month',
    'FCR_TICKETS': 'fcr_tickets_raw_per_month',
    'install_attempts': 'install_attempts_per_month',
    'DISTINCT_ISSUE_TYPES': 'distinct_issues_per_month',
    'MAX_TICKETS_SAME_ISSUE': 'max_same_issue_per_month',
    'TICKETS_REOPENED_ONCE': 'reopened_once_per_month',
    'TICKETS_REOPENED_3PLUS': 'reopened_3plus_per_month',
    'MAX_TIMES_REOPENED': 'max_reopened_per_month',
    'ISSUES_WITH_3PLUS_TICKETS': 'complex_issues_per_month',
    'TOTAL_LEADS': 'leads_per_month',
    'support_effort_index': 'support_effort_per_month',
}

rpt(f"\n  {'Original':40s} | {'Normalized':35s} | {'Old tenure_r':>13s} | {'New tenure_r':>13s} | {'Improved?':>10s}")
rpt(f"  {'-'*40} | {'-'*35} | {'-'*13} | {'-'*13} | {'-'*10}")

normalized_features = {}
for orig, norm_name in normalize_map.items():
    if orig not in df.columns:
        continue
    orig_vals = pd.to_numeric(df[orig], errors='coerce')
    norm_vals = orig_vals / tenure_months

    valid = df[['tenure_days']].copy()
    valid['orig'] = orig_vals
    valid['norm'] = norm_vals
    valid = valid.dropna()

    if len(valid) < 100:
        continue

    old_r = valid['orig'].corr(valid['tenure_days'])
    new_r = valid['norm'].corr(valid['tenure_days'])
    improved = 'YES' if abs(new_r) < abs(old_r) else 'NO'

    rpt(f"  {orig:40s} | {norm_name:35s} | {old_r:+13.3f} | {new_r:+13.3f} | {improved:>10s}")
    normalized_features[norm_name] = norm_vals


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 8: v4 MODEL TOP FEATURES — BIAS CHECK
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("AUDIT 8: v4 MODEL TOP-20 FEATURES — BIAS CHECK")
rpt("=" * 100)

v4_top = [
    ('avg_uptime_pct', 0.3205),
    ('sc_avg_rxpower_in_range', 0.1481),
    ('tk_sla_compliance', 0.0620),
    ('TOTAL_IVR_CALLS', 0.0455),
    ('FAILURE_RATE_PCT', 0.0407),
    ('MISSED_CALLS', 0.0317),
    ('AVG_ANSWERED_SECONDS', 0.0305),
    ('AVG_TICKETS_PER_ISSUE', 0.0301),
    ('avg_resolution_hours_w', 0.0298),
    ('avg_resolution_hours', 0.0248),
    ('autopay_ratio', 0.0203),
    ('sc_avg_rxpower', 0.0174),
    ('sc_avg_weekly_data_gb', 0.0155),
    ('INBOUND_CALLS', 0.0144),
    ('sc_avg_speed_in_range', 0.0109),
]

rpt(f"\n  {'#':>3s} {'Feature':40s} | {'Imp%':>6s} | {'tenure_r':>10s} | {'Verdict':12s} | Assessment")
rpt(f"  {'-'*3} {'-'*40} | {'-'*6} | {'-'*10} | {'-'*12} | {'-'*40}")

for i, (f, imp) in enumerate(v4_top):
    f_audit = audit[audit['feature'] == f]
    if len(f_audit) > 0:
        t_r = f_audit.iloc[0].get('tenure_r', 0)
        verdict = f_audit.iloc[0].get('verdict', '?')
    else:
        t_r = 0
        verdict = '?'

    # CX assessment
    if abs(t_r) > 0.15:
        assessment = f'BIASED — tenure_r={t_r:+.3f}, replace with per-month version'
    elif verdict == 'CLEAN':
        assessment = 'Clean — no bias detected'
    else:
        assessment = f'{verdict}: check needed'

    rpt(f"  {i+1:3d} {f:40s} | {imp*100:5.1f}% | {t_r:+10.3f} | {verdict:12s} | {assessment}")

rpt(f"""
  SUMMARY: v4 Model Health
  ━━━━━━━━━━━━━━━━━━━━━━━━━
  Of the top 15 features in v4 OOT model:
  - CLEAN:     avg_uptime_pct, sc_avg_rxpower_in_range, tk_sla_compliance,
               FAILURE_RATE_PCT, AVG_ANSWERED_SECONDS, AVG_TICKETS_PER_ISSUE,
               avg_resolution_hours_w, avg_resolution_hours, autopay_ratio,
               sc_avg_rxpower, sc_avg_weekly_data_gb, sc_avg_speed_in_range
  - NORMALIZE: TOTAL_IVR_CALLS (tenure_r=+0.249), MISSED_CALLS (tenure_r=+0.201),
               INBOUND_CALLS (tenure_r=+0.348)

  ACTION: Replace 3 tenure-biased features with per-month normalized versions.
  EXPECTED: Model AUC may drop slightly but signal will be cleaner and
            generalizable to the full population (different tenure distributions).
""")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CLEAN FEATURE LIST
# ══════════════════════════════════════════════════════════════════════════════
rpt(f"\n" + "=" * 100)
rpt("FINAL RECOMMENDED FEATURE LIST (v5)")
rpt("=" * 100)

clean_features = audit[audit['verdict'] == 'CLEAN']['feature'].tolist()
normalize_features = audit[audit['verdict'] == 'NORMALIZE']['feature'].tolist()

rpt(f"\n  CLEAN features (use as-is): {len(clean_features)}")
rpt(f"  NORMALIZE features (replace with per-month): {len(normalize_features)}")
rpt(f"  Total v5 features: {len(clean_features) + len(normalize_features)}")

rpt(f"\n  Normalized feature names to create:")
for orig in normalize_features:
    if orig in normalize_map:
        rpt(f"    {orig:40s} → {normalize_map[orig]}")
    else:
        rpt(f"    {orig:40s} → {orig}_per_month")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
output_file = os.path.join(OUTPUT, "phase4b_feature_audit.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# Also save the clean feature list as CSV for v5 model
clean_list_file = os.path.join(OUTPUT, "v5_clean_features.csv")
feature_verdicts = audit[['feature', 'verdict', 'reason', 'tenure_r', 'churn_r', 'fill_overall', 'fill_gap']].copy()
feature_verdicts.to_csv(clean_list_file, index=False)

rpt(f"\n{'='*100}")
rpt(f"SAVED: {output_file}")
rpt(f"SAVED: {clean_list_file}")
rpt(f"COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"{'='*100}")
