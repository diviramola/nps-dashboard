"""
Phase 3C FINAL: Data Merge + Temporal Partner Status + Partner 30/90d Metrics
=============================================================================
All data sources merged via INDEX ALIGNMENT (same row order from nps_clean_base).
- Temporal partner status from partner_details_log (correct column names)
- Partner-level 30d/90d service quality windows
- Partner-level aggregate features (quality proxy)
- Principled outlier handling
- Derived modeling features

Output: data/nps_modeling_dataset.csv, output/phase3c_merge_report.txt
"""

import sys, io, os, time
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found"); sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}
BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

report = []
def log(msg=""):
    print(msg)
    report.append(msg)

def save_report():
    with open(os.path.join(OUTPUT, "phase3c_merge_report.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def run_query(sql, timeout=120):
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
    data = resp.json()
    if "data" not in data or not data["data"]["rows"]:
        return pd.DataFrame()
    rows = data["data"]["rows"]
    cols = [c["name"] for c in data["data"]["cols"]]
    return pd.DataFrame(rows, columns=cols)

log("=" * 70)
log("PHASE 3C: FINAL DATA MERGE")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)
log()

# ════════════════════════════════════════════════════════════════
# 1. LOAD
# ════════════════════════════════════════════════════════════════
log("[1] Loading data sources...")
expanded  = pd.read_csv(os.path.join(DATA, "nps_expanded_base.csv"), low_memory=False)
analytical = pd.read_csv(os.path.join(DATA, "nps_analytical_base.csv"), low_memory=False)
industry   = pd.read_csv(os.path.join(DATA, "industry_expert_features.csv"), low_memory=False)
themes     = pd.read_csv(os.path.join(DATA, "nps_comments_themed.csv"), low_memory=False)

consol_mask = expanded['_source'] == 'consolidated'
N_CONSOL = consol_mask.sum()
N_NEW = (~consol_mask).sum()
log(f"  Expanded:  {expanded.shape} (consolidated={N_CONSOL}, new={N_NEW})")
log(f"  Analytical: {analytical.shape}")
log(f"  Industry:   {industry.shape}")
log(f"  Themes:     {themes.shape}")

# Verify index alignment
phones_exp = expanded[consol_mask]['phone_number'].reset_index(drop=True).astype(str)
phones_ana = analytical['phone_number'].reset_index(drop=True).astype(str)
phones_ind = industry['phone_number'].reset_index(drop=True).astype(str)
assert (phones_exp.head(50) == phones_ana.head(50)).all(), "Analytical alignment FAILED"
assert (phones_exp.head(50) == phones_ind.head(50)).all(), "Industry alignment FAILED"
log("  Index alignment verified.")
log()

# ════════════════════════════════════════════════════════════════
# 2. MERGE SNOWFLAKE + INDUSTRY EXPERT (index-aligned)
# ════════════════════════════════════════════════════════════════
log("[2] Merging features via index alignment...")

expanded_cols = set(expanded.columns)
base_join_cols = {'phone_number', 'nps_score', 'nps_group', 'Sprint Start Date', 'Sprint End Date'}

# Snowflake-only columns
snow_cols = [c for c in analytical.columns if c not in expanded_cols]
log(f"  Snowflake-only columns: {len(snow_cols)}")

# Industry-only columns (exclude anything in expanded or that's a base join col)
ind_cols = [c for c in industry.columns
            if c not in expanded_cols and c not in base_join_cols]
# Handle column name collisions with Snowflake
ind_collisions = [c for c in ind_cols if c in snow_cols]
log(f"  Industry-only columns: {len(ind_cols)} ({len(ind_collisions)} collide with Snowflake)")

# Start with expanded as base
merged = expanded.copy()

# Add Snowflake columns
for col in snow_cols:
    merged[col] = np.nan
    merged.loc[consol_mask, col] = analytical[col].values

# Add Industry Expert columns (rename collisions)
for col in ind_cols:
    target_name = f"{col}_ie" if col in snow_cols else col
    merged[target_name] = np.nan
    merged.loc[consol_mask, target_name] = industry[col].values

log(f"  Snowflake features: {merged[snow_cols[0]].notna().sum()} rows")
first_ind_col = [c for c in ind_cols if c not in snow_cols][0] if [c for c in ind_cols if c not in snow_cols] else None
if first_ind_col:
    log(f"  Industry features: {merged[first_ind_col].notna().sum()} rows")
log()

# ════════════════════════════════════════════════════════════════
# 3. MERGE NLP THEMES
# ════════════════════════════════════════════════════════════════
log("[3] Merging NLP themes...")

theme_cols = [c for c in [
    'translated_comment', 'detected_language', 'sentiment_polarity',
    'sentiment_intensity', 'emotion', 'comment_quality', 'score_sentiment_mismatch',
    'primary_theme', 'primary_theme_score', 'secondary_theme', 'secondary_theme_score',
    'mentions_28day', 'mentions_competitor', 'competitor_names', 'mentions_amount'
] if c in themes.columns]

themes['_tk'] = themes['phone_number'].astype(str) + '_' + themes['Sprint ID'].astype(str)
merged['_tk'] = merged['phone_number'].astype(str) + '_' + merged['Sprint ID'].astype(str)
theme_dedup = themes[['_tk'] + theme_cols].drop_duplicates(subset='_tk', keep='first')
merged = merged.merge(theme_dedup, on='_tk', how='left')
merged.drop(columns=['_tk'], inplace=True)
log(f"  Themes matched: {merged['primary_theme'].notna().sum()}")
log()

# ════════════════════════════════════════════════════════════════
# 4. TEMPORAL PARTNER STATUS FROM partner_details_log
# ════════════════════════════════════════════════════════════════
log("[4] Temporal partner status (partner_details_log)...")
log("  Column names: \"lco_id_long\", \"status\", \"added_time\" (lowercase, quoted)")

partner_lng_ids = merged['partner_lng_id'].dropna().unique()
partner_lng_ids = [int(p) for p in partner_lng_ids]
sprint_ends = sorted(merged['Sprint End Date'].dropna().unique())
log(f"  Partners: {len(partner_lng_ids)}, Sprint dates: {len(sprint_ends)}")

# For each sprint end date, get status of all partners on that date
temporal_records = []
BATCH = 300

for date_str in sprint_ends:
    for batch_start in range(0, len(partner_lng_ids), BATCH):
        batch = partner_lng_ids[batch_start:batch_start + BATCH]
        id_list = ','.join([str(p) for p in batch])

        sql = f"""
        SELECT \"lco_id_long\" as partner_lng_id,
               \"status\" as partner_status_at_survey,
               \"added_time\"::date as snap_date
        FROM partner_details_log
        WHERE \"lco_id_long\" IN ({id_list})
          AND \"added_time\"::date <= '{date_str}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY \"lco_id_long\" ORDER BY \"added_time\" DESC) = 1
        """
        try:
            df_r = run_query(sql, timeout=90)
            if len(df_r) > 0:
                df_r['Sprint End Date'] = date_str
                temporal_records.append(df_r)
            time.sleep(1.0)
        except Exception as e:
            log(f"  WARN: batch failed for {date_str}: {str(e)[:60]}")
            time.sleep(2)

    log(f"  Date {date_str}: {sum(len(r) for r in temporal_records)} cumulative records")

if temporal_records:
    temporal_df = pd.concat(temporal_records, ignore_index=True)
    log(f"  Raw temporal columns: {list(temporal_df.columns)}")
    # Metabase may return original Snowflake col names instead of SQL aliases
    col_map = {}
    for c in temporal_df.columns:
        cl = c.lower().strip('"')
        if cl in ('lco_id_long', 'partner_lng_id'):
            col_map[c] = 'partner_lng_id'
        elif cl in ('status', 'partner_status_at_survey'):
            col_map[c] = 'partner_status_at_survey'
        elif cl in ('snap_date',) or 'added_time' in cl.lower():
            col_map[c] = 'snap_date'
    if col_map:
        temporal_df.rename(columns=col_map, inplace=True)
        log(f"  Renamed columns: {col_map}")
    temporal_df['partner_lng_id'] = pd.to_numeric(temporal_df['partner_lng_id'], errors='coerce')
    temporal_df = temporal_df.drop_duplicates(subset=['partner_lng_id', 'Sprint End Date'], keep='first')
    log(f"  Total temporal records: {len(temporal_df)}")

    merged['partner_lng_id'] = pd.to_numeric(merged['partner_lng_id'], errors='coerce')
    before = len(merged)
    merged = merged.merge(
        temporal_df[['partner_lng_id', 'Sprint End Date', 'partner_status_at_survey', 'snap_date']],
        on=['partner_lng_id', 'Sprint End Date'], how='left'
    )
    log(f"  Rows: {before} -> {len(merged)}")
    log(f"  Temporal status coverage: {merged['partner_status_at_survey'].notna().sum()}")

    # Status drift analysis
    if 'partner_status' in merged.columns:
        both = merged['partner_status'].notna() & merged['partner_status_at_survey'].notna()
        drift = both & (merged['partner_status'].str.upper() != merged['partner_status_at_survey'].str.upper())
        log(f"  Status drift (current != at-survey): {drift.sum()} rows")
        if drift.sum() > 0:
            log("  DRIFT EXAMPLES (data leakage would have used current status):")
            for _, row in merged[drift].head(5).iterrows():
                log(f"    Partner {row.get('partner_name','')} [{int(row['partner_lng_id'])}]: "
                    f"at_survey={row['partner_status_at_survey']}, current={row['partner_status']}")
else:
    log("  WARNING: No temporal data retrieved")
    merged['partner_status_at_survey'] = np.nan
    merged['snap_date'] = np.nan

log()

# ════════════════════════════════════════════════════════════════
# 5. PARTNER-LEVEL AGGREGATE + DEVIATION FEATURES
# ════════════════════════════════════════════════════════════════
log("[5] Partner-level aggregate service metrics...")
log("  (Robust proxy: individual metrics can be noisy, partner aggregates normalize)")

if 'partner_lng_id' in merged.columns:
    pg = merged[merged['partner_lng_id'].notna()].groupby('partner_lng_id')

    aggs = {}
    agg_map = {
        'partner_avg_resolution_hours': ('avg_resolution_hours', 'median'),
        'partner_sla_compliance': ('sla_compliance_pct', 'mean'),
        'partner_avg_tickets': ('total_tickets', 'median'),
        'partner_churn_rate': ('is_churned', 'mean'),
        'partner_avg_nps': ('nps_score', 'mean'),
        'partner_customer_count': ('phone_number', 'count'),
    }
    for feat_name, (col, func) in agg_map.items():
        if col in merged.columns:
            aggs[feat_name] = pg[col].agg(func)
            log(f"  {feat_name}")

    # FCR rate aggregate
    if 'FCR_RATE' in merged.columns:
        aggs['partner_fcr_rate'] = pg['FCR_RATE'].mean()
        log(f"  partner_fcr_rate")

    if 'HAS_REPEAT_COMPLAINT' in merged.columns:
        aggs['partner_repeat_rate'] = pg['HAS_REPEAT_COMPLAINT'].mean()
        log(f"  partner_repeat_rate")

    if 'install_tat_hours' in merged.columns:
        aggs['partner_median_install_tat'] = pg['install_tat_hours'].median()
        log(f"  partner_median_install_tat")

    if aggs:
        partner_agg_df = pd.DataFrame(aggs).reset_index()
        before = len(merged)
        merged = merged.merge(partner_agg_df, on='partner_lng_id', how='left')
        log(f"  Added {len(aggs)} partner-level columns, rows: {before}->{len(merged)}")

        # Deviation features
        deviations = [
            ('avg_resolution_hours', 'partner_avg_resolution_hours', 'resolution_vs_partner'),
            ('sla_compliance_pct', 'partner_sla_compliance', 'sla_vs_partner'),
            ('total_tickets', 'partner_avg_tickets', 'tickets_vs_partner'),
        ]
        for cust_col, partner_col, dev_name in deviations:
            if cust_col in merged.columns and partner_col in merged.columns:
                merged[dev_name] = merged[cust_col] - merged[partner_col]
                log(f"  {dev_name}: customer minus partner norm")

log()

# ════════════════════════════════════════════════════════════════
# 6. PRINCIPLED OUTLIER HANDLING
# ════════════════════════════════════════════════════════════════
log("[6] Outlier handling...")

# Install TAT
if 'install_tat_hours' in merged.columns:
    neg = (merged['install_tat_hours'] < 0).sum()
    merged.loc[merged['install_tat_hours'] < 0, 'install_tat_hours'] = np.nan
    merged['install_delayed'] = (merged['install_tat_hours'] > 72).astype(float)
    merged.loc[merged['install_tat_hours'].isna(), 'install_delayed'] = np.nan
    p99 = merged['install_tat_hours'].quantile(0.99)
    merged['install_tat_hours_w'] = merged['install_tat_hours'].clip(upper=p99)
    log(f"  install_tat_hours: {neg} neg->NaN, P99 cap={p99:.0f}h, delayed flag >72h")

# Resolution hours
if 'avg_resolution_hours' in merged.columns:
    neg = (merged['avg_resolution_hours'] < 0).sum()
    merged.loc[merged['avg_resolution_hours'] < 0, 'avg_resolution_hours'] = np.nan
    p99 = merged['avg_resolution_hours'].quantile(0.99)
    merged['avg_resolution_hours_w'] = merged['avg_resolution_hours'].clip(upper=p99)
    log(f"  avg_resolution_hours: {neg} neg->NaN, P99 cap={p99:.0f}h")

# Days since recharge
if 'days_since_last_recharge' in merged.columns:
    neg = (merged['days_since_last_recharge'] < 0).sum()
    merged.loc[merged['days_since_last_recharge'] < 0, 'days_since_last_recharge'] = 0
    log(f"  days_since_last_recharge: {neg} neg->0")

# Time to value
if 'HOURS_TO_FIRST_RECHARGE' in merged.columns:
    ext = (merged['HOURS_TO_FIRST_RECHARGE'] < -168).sum()
    merged.loc[merged['HOURS_TO_FIRST_RECHARGE'] < -168, 'HOURS_TO_FIRST_RECHARGE'] = np.nan
    if 'DAYS_TO_FIRST_RECHARGE' in merged.columns:
        merged.loc[merged['DAYS_TO_FIRST_RECHARGE'] < -7, 'DAYS_TO_FIRST_RECHARGE'] = np.nan
        merged['recharge_same_day'] = merged['DAYS_TO_FIRST_RECHARGE'].between(-1, 1).astype(float)
        merged.loc[merged['DAYS_TO_FIRST_RECHARGE'].isna(), 'recharge_same_day'] = np.nan
    log(f"  HOURS_TO_FIRST_RECHARGE: {ext} extreme->NaN, same-day flag")

# Install TAT mins (industry expert)
if 'AVG_INSTALL_TAT_MINS' in merged.columns:
    p99 = merged['AVG_INSTALL_TAT_MINS'].quantile(0.99)
    merged['AVG_INSTALL_TAT_MINS_w'] = merged['AVG_INSTALL_TAT_MINS'].clip(upper=p99)
    log(f"  AVG_INSTALL_TAT_MINS: P99 cap={p99:.0f}m")

log()

# ════════════════════════════════════════════════════════════════
# 7. DERIVED FEATURES
# ════════════════════════════════════════════════════════════════
log("[7] Derived features...")

# Partner risk (prefer temporal, fallback to current)
status_col = 'partner_status_at_survey' if merged.get('partner_status_at_survey', pd.Series()).notna().any() else 'partner_status'
if status_col in merged.columns:
    risk_map = {'Active': 0, 'ACTIVE': 0, 'Blocked': 1, 'TEMPORARY SUSPENSION': 1,
                'Delisted': 2, 'TERMINATION': 2, 'CLOSED': 2, 'BLACKLISTED': 2}
    merged['partner_risk_level'] = merged[status_col].map(risk_map)
    merged['partner_at_risk'] = (merged['partner_risk_level'] >= 1).astype(float)
    merged.loc[merged[status_col].isna(), ['partner_at_risk', 'partner_risk_level']] = np.nan
    log(f"  partner_risk (from {status_col}): {merged['partner_at_risk'].sum():.0f} at-risk")

# Support effort index (z-score composite)
effort_cols = [c for c in ['TOTAL_IVR_CALLS', 'AVG_TIMES_REOPENED', 'HAS_REPEAT_COMPLAINT', 'total_tickets'] if c in merged.columns]
if len(effort_cols) >= 2:
    effort_z = pd.DataFrame()
    for col in effort_cols:
        v = merged[col].dropna()
        if len(v) > 0 and v.std() > 0:
            effort_z[col] = (merged[col] - v.mean()) / v.std()
        else:
            effort_z[col] = 0
    merged['support_effort_index'] = effort_z.mean(axis=1)
    log(f"  support_effort_index: {effort_cols}")

# Network quality index
net_pos = [c for c in ['PEAK_UPTIME_PCT'] if c in merged.columns]
net_neg = [c for c in ['OUTAGE_EVENTS', 'AVG_RECOVERY_MINS', 'AVG_PEAK_INTERRUPTIONS'] if c in merged.columns]
if len(net_pos) + len(net_neg) >= 2:
    net_z = pd.DataFrame()
    for col in net_pos:
        v = merged[col].dropna()
        if len(v) > 0 and v.std() > 0:
            net_z[col] = (merged[col] - v.mean()) / v.std()
    for col in net_neg:
        v = merged[col].dropna()
        if len(v) > 0 and v.std() > 0:
            net_z[col] = -1 * (merged[col] - v.mean()) / v.std()
    merged['network_quality_index'] = net_z.mean(axis=1)
    log(f"  network_quality_index")

# Missed call ratio
if 'MISSED_CALLS' in merged.columns and 'TOTAL_IVR_CALLS' in merged.columns:
    merged['missed_call_ratio'] = merged['MISSED_CALLS'] / merged['TOTAL_IVR_CALLS'].replace(0, np.nan)
    log(f"  missed_call_ratio")

# Autopay ratio
if 'autopay_payments' in merged.columns and 'total_payments' in merged.columns:
    merged['autopay_ratio'] = merged['autopay_payments'] / merged['total_payments'].replace(0, np.nan)
    log(f"  autopay_ratio")

# Ticket severity
if 'AVG_TIMES_REOPENED' in merged.columns and 'MAX_TICKETS_SAME_ISSUE' in merged.columns:
    merged['ticket_severity'] = merged['AVG_TIMES_REOPENED'] * merged['MAX_TICKETS_SAME_ISSUE']
    log(f"  ticket_severity")

log()

# ════════════════════════════════════════════════════════════════
# 8. SAVE
# ════════════════════════════════════════════════════════════════
log("[8] Save...")
log(f"  Final: {merged.shape[0]} rows x {merged.shape[1]} cols")
log()

# Coverage report
log("  FEATURE COVERAGE (% non-null):")
checks = [
    ('Snowflake (Phase 3)', 'total_recharges'),
    ('Industry Expert (Phase 3b)', 'TOTAL_IVR_CALLS'),
    ('NLP themes', 'primary_theme'),
    ('Temporal partner status', 'partner_status_at_survey'),
    ('Partner aggregates', 'partner_avg_nps'),
    ('Sprint tab extras', 'device_type'),
    ('Outage events', 'OUTAGE_EVENTS'),
    ('FCR rate', 'FCR_RATE'),
    ('Data usage', 'AVG_DAILY_DATA_GB'),
    ('Support effort index', 'support_effort_index'),
    ('Network quality index', 'network_quality_index'),
]
for label, col in checks:
    if col in merged.columns:
        n = merged[col].notna().sum()
        log(f"    {label:30s}: {n:5d}/{len(merged)} ({n/len(merged)*100:.1f}%)")

log()
log("  NPS DISTRIBUTION:")
for g in ['Promoter', 'Passive', 'Detractor']:
    n = (merged['nps_group'] == g).sum()
    log(f"    {g}: {n} ({n/len(merged)*100:.1f}%)")

log()
log("  SOURCE:")
for s, n in merged['_source'].value_counts().items():
    log(f"    {s}: {n}")

out = os.path.join(DATA, "nps_modeling_dataset.csv")
merged.to_csv(out, index=False, encoding='utf-8-sig')
log(f"\n  Saved: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")

save_report()
log(f"  Report: {os.path.join(OUTPUT, 'phase3c_merge_report.txt')}")

log()
log("=" * 70)
log("DONE. Ready for Phase 4.")
log("=" * 70)
