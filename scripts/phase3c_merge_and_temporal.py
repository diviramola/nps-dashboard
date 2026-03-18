"""
Phase 3C: Data Merge + Temporal Feature Engineering
=====================================================
Merges ALL data sources into a single modeling-ready dataset:
1. nps_expanded_base.csv (16,604 rows - master base with sprint tab extras)
2. nps_analytical_base.csv (13,045 rows - Phase 3 Snowflake features)
3. industry_expert_features.csv (13,045 rows - Phase 3b Industry Expert features)
4. nps_comments_themed.csv (5,488 rows - Phase 1.3 NLP themes)
5. Temporal partner status from partner_details_log (Snowflake query)

Also applies principled outlier handling based on adaptive analysis.

Output:
- data/nps_modeling_dataset.csv — final unified dataset
- output/phase3c_merge_report.txt — merge report with diagnostics
"""

import sys, io, os, json, time, traceback
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Credentials ──
from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

def run_query(sql, timeout=120):
    """Execute SQL via Metabase API and return DataFrame."""
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
    data = resp.json()
    if "data" not in data:
        return pd.DataFrame()
    rows = data["data"]["rows"]
    cols = [c["name"] for c in data["data"]["cols"]]
    return pd.DataFrame(rows, columns=cols)

# ======================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ======================================================================
log("=" * 70)
log("PHASE 3C: DATA MERGE + TEMPORAL FEATURE ENGINEERING")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)
log()

log("STEP 1: Loading all data sources...")
log("-" * 40)

# 1a. Expanded base (master — 16,604 rows)
expanded = pd.read_csv(os.path.join(DATA, "nps_expanded_base.csv"))
log(f"  Expanded base: {expanded.shape[0]} rows x {expanded.shape[1]} cols")
log(f"    Consolidated: {(expanded['_source'] == 'consolidated').sum()}")
log(f"    Sprint tab new: {(expanded['_source'] == 'sprint_tab_new').sum()}")

# 1b. Analytical base (Phase 3 Snowflake features — 13,045 rows)
analytical = pd.read_csv(os.path.join(DATA, "nps_analytical_base.csv"))
log(f"  Analytical base: {analytical.shape[0]} rows x {analytical.shape[1]} cols")

# 1c. Industry expert features (Phase 3b — 13,045 rows)
industry = pd.read_csv(os.path.join(DATA, "industry_expert_features.csv"))
log(f"  Industry expert: {industry.shape[0]} rows x {industry.shape[1]} cols")

# 1d. Themed comments (Phase 1.3 — 5,488 rows)
themes = pd.read_csv(os.path.join(DATA, "nps_comments_themed.csv"))
log(f"  Themed comments: {themes.shape[0]} rows x {themes.shape[1]} cols")
log()

# ======================================================================
# STEP 2: IDENTIFY UNIQUE COLUMNS FROM EACH SOURCE
# ======================================================================
log("STEP 2: Identifying unique columns from each source...")
log("-" * 40)

# Common base columns (in all 3 main datasets)
base_cols = [
    'phone_number', 'phone_valid', 'nps_score', 'nps_group', 'OE', 'has_comment',
    'NPS Reason - Primary', 'NPS Reason - Secondary', 'NPS Reason - Tertiary',
    'Primary Category', 'Secondary Category', 'Tertiary Category',
    'Channel', 'first_time_wifi', 'Sprint ID', 'sprint_num', 'Cycle ID',
    'Sprint Start Date', 'Sprint End Date', 'Install Date',
    'tenure_days', 'tenure_bucket', 'tenure_excel', 'city',
    'churn_label', 'is_churned', 'recharges_before_sprint',
    'tickets_post_sprint', 'tickets_before_3m'
]

# Hindi survey columns (in both expanded and analytical)
hindi_cols = [c for c in expanded.columns if any(x in str(c) for x in ['Wi-Fi', 'dikkat', 'WiFi', 'samadhan', 'samasya'])]

# Snowflake feature columns (unique to analytical_base)
snowflake_cols = [
    'total_recharges', 'first_recharge', 'last_recharge', 'avg_recharge_amount',
    'total_tickets', 'cx_tickets', 'px_tickets', 'avg_resolution_hours',
    'sla_compliance_pct', 'partner_lng_id', 'cluster', 'mis_city',
    'partner_status', 'avg_uptime_pct', 'stddev_uptime', 'min_uptime',
    'uptime_data_days', 'install_time', 'install_attempts', 'booking_time',
    'install_tat_hours', 'total_payments', 'autopay_payments', 'cash_payments',
    'avg_payment_amount', 'payment_mode', 'has_tickets',
    'days_since_last_recharge', 'recharge_regularity'
]
# Filter to columns actually present
snowflake_cols = [c for c in snowflake_cols if c in analytical.columns]
log(f"  Snowflake features: {len(snowflake_cols)} columns")

# Industry expert columns (unique to industry_expert_features)
industry_base_cols = ['phone_number', 'nps_score', 'nps_group', 'Sprint Start Date', 'Sprint End Date']
industry_feature_cols = [c for c in industry.columns if c not in industry_base_cols]
log(f"  Industry expert features: {len(industry_feature_cols)} columns")

# Sprint tab extra columns (unique to expanded_base)
expanded_extra_cols = [c for c in expanded.columns
                       if c not in base_cols
                       and c not in hindi_cols
                       and c not in ['is_post_reset', 'is_payg', '_source']
                       and c not in snowflake_cols]
log(f"  Sprint tab extras: {len(expanded_extra_cols)} columns")

# Theme columns (unique to themed comments)
theme_feature_cols = [
    'translated_comment', 'detected_language', 'sentiment_polarity',
    'sentiment_intensity', 'emotion', 'comment_quality', 'score_sentiment_mismatch',
    'primary_theme', 'primary_theme_score', 'secondary_theme', 'secondary_theme_score',
    'mentions_28day', 'mentions_competitor', 'competitor_names', 'mentions_amount'
]
theme_feature_cols = [c for c in theme_feature_cols if c in themes.columns]
log(f"  Theme features: {len(theme_feature_cols)} columns")
log()

# ======================================================================
# STEP 3: BUILD MERGE KEYS
# ======================================================================
log("STEP 3: Building merge keys...")
log("-" * 40)

# Create unique row IDs for reliable merging
# Strategy: merge on (phone_number, Sprint Start Date) as composite key

# For expanded base, create merge key
expanded['_merge_key'] = expanded['phone_number'].astype(str) + '_' + expanded['Sprint Start Date'].astype(str)
log(f"  Expanded unique merge keys: {expanded['_merge_key'].nunique()} / {len(expanded)}")

# For analytical base
analytical['_merge_key'] = analytical['phone_number'].astype(str) + '_' + analytical['Sprint Start Date'].astype(str)
log(f"  Analytical unique merge keys: {analytical['_merge_key'].nunique()} / {len(analytical)}")

# For industry expert
industry['_merge_key'] = industry['phone_number'].astype(str) + '_' + industry['Sprint Start Date'].astype(str)
log(f"  Industry unique merge keys: {industry['_merge_key'].nunique()} / {len(industry)}")

# For themes — use phone_number + Sprint ID (themes have Sprint ID)
if 'Sprint ID' in themes.columns:
    themes['_merge_key'] = themes['phone_number'].astype(str) + '_' + themes['Sprint ID'].astype(str)
else:
    themes['_merge_key'] = themes['phone_number'].astype(str)

# Check for key overlaps
consol_keys = set(expanded[expanded['_source'] == 'consolidated']['_merge_key'])
analytical_keys = set(analytical['_merge_key'])
industry_keys = set(industry['_merge_key'])
log(f"  Consolidated-Analytical overlap: {len(consol_keys & analytical_keys)}")
log(f"  Consolidated-Industry overlap: {len(consol_keys & industry_keys)}")
log()

# ======================================================================
# STEP 4: MERGE SNOWFLAKE FEATURES INTO EXPANDED BASE
# ======================================================================
log("STEP 4: Merging Snowflake features...")
log("-" * 40)

# Extract only Snowflake feature columns + merge key from analytical
analytical_features = analytical[['_merge_key'] + snowflake_cols].copy()

# Handle column conflicts: zone and partner_name exist in both expanded and analytical
# Priority: analytical (Snowflake HIERARCHY_BASE) > expanded (sprint tab)
# Rename expanded's conflicting columns first
conflict_cols = ['zone', 'partner_name']
for col in conflict_cols:
    if col in expanded.columns and col in snowflake_cols:
        expanded.rename(columns={col: f'{col}_sprint_tab'}, inplace=True)

merged = expanded.merge(analytical_features, on='_merge_key', how='left')
matched_snowflake = merged[snowflake_cols[0]].notna().sum()
log(f"  Rows with Snowflake features: {matched_snowflake} / {len(merged)}")
log(f"  Rows WITHOUT Snowflake features: {len(merged) - matched_snowflake} (new sprint tab rows)")
log()

# ======================================================================
# STEP 5: MERGE INDUSTRY EXPERT FEATURES
# ======================================================================
log("STEP 5: Merging Industry Expert features...")
log("-" * 40)

# Handle UPTIME_DATA_DAYS collision (exists in both analytical and industry)
if 'UPTIME_DATA_DAYS' in industry_feature_cols and 'uptime_data_days' in merged.columns:
    industry_feature_cols = [c for c in industry_feature_cols if c != 'UPTIME_DATA_DAYS']
    industry.rename(columns={'UPTIME_DATA_DAYS': 'UPTIME_DATA_DAYS_IE'}, inplace=True)
    industry_feature_cols.append('UPTIME_DATA_DAYS_IE')

industry_features = industry[['_merge_key'] + industry_feature_cols].copy()

# Drop any columns that might conflict
for col in industry_feature_cols:
    if col in merged.columns:
        log(f"  WARNING: Column '{col}' already exists in merged — renaming to {col}_ie")
        industry_features.rename(columns={col: f'{col}_ie'}, inplace=True)
        industry_feature_cols = [f'{c}_ie' if c == col else c for c in industry_feature_cols]

merged = merged.merge(industry_features, on='_merge_key', how='left')
matched_industry = merged[industry_feature_cols[0]].notna().sum() if industry_feature_cols else 0
log(f"  Rows with Industry Expert features: {matched_industry} / {len(merged)}")
log()

# ======================================================================
# STEP 6: MERGE NLP THEMES
# ======================================================================
log("STEP 6: Merging NLP themes...")
log("-" * 40)

# For themes, we need a different merge key since themes use Sprint ID not Sprint Start Date
# Create Sprint ID-based merge key for expanded
expanded_sprint_key = expanded['phone_number'].astype(str) + '_' + expanded['Sprint ID'].astype(str)
merged['_theme_key'] = merged['phone_number'].astype(str) + '_' + merged['Sprint ID'].astype(str)

# Get only theme feature columns + merge key
theme_features = themes[['_merge_key'] + theme_feature_cols].copy()
theme_features.rename(columns={'_merge_key': '_theme_key'}, inplace=True)

# Some phones may have duplicate entries in themes — take first match
theme_features = theme_features.drop_duplicates(subset='_theme_key', keep='first')

merged = merged.merge(theme_features, on='_theme_key', how='left')
matched_themes = merged['primary_theme'].notna().sum()
log(f"  Rows with NLP themes: {matched_themes} / {len(merged)}")
log(f"  Rows with comments but no theme: {((merged['has_comment'] == True) & merged['primary_theme'].isna()).sum()}")
log()

# ======================================================================
# STEP 7: TEMPORAL PARTNER STATUS (Avoid Data Leakage)
# ======================================================================
log("STEP 7: Querying temporal partner status from Snowflake...")
log("-" * 40)
log("  (Using partner_details_log to get status AT SURVEY TIME, not current)")
log()

# Get unique partner_lng_id + Sprint End Date combinations
# We need to query partner_details_log for each partner at each survey date
partner_dates = merged[merged['partner_lng_id'].notna()][['partner_lng_id', 'Sprint End Date']].drop_duplicates()
partner_dates = partner_dates[partner_dates['partner_lng_id'].notna()]
log(f"  Unique (partner, date) combinations: {len(partner_dates)}")

# Batch query: for each Sprint End Date, get partner statuses
# Get unique sprint end dates
unique_dates = sorted(partner_dates['Sprint End Date'].dropna().unique())
log(f"  Unique survey dates: {len(unique_dates)}")

temporal_results = []
failed_dates = []

for i, end_date in enumerate(unique_dates):
    # Get partner IDs for this date
    partner_ids = partner_dates[partner_dates['Sprint End Date'] == end_date]['partner_lng_id'].unique()
    partner_list = ','.join([f"'{int(p)}'" if pd.notna(p) else '' for p in partner_ids if pd.notna(p)])

    if not partner_list:
        continue

    # Query partner_details_log for status on or before this date
    sql = f"""
    SELECT
        lng_id as partner_lng_id,
        partner_status as partner_status_at_survey,
        date as status_date
    FROM WIOM.WIOM_DW.partner_details_log
    WHERE lng_id IN ({partner_list})
      AND date <= '{end_date}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY lng_id ORDER BY date DESC) = 1
    """

    try:
        df_result = run_query(sql, timeout=60)
        if len(df_result) > 0:
            df_result['Sprint End Date'] = end_date
            temporal_results.append(df_result)
            if (i + 1) % 5 == 0 or i == len(unique_dates) - 1:
                log(f"  Processed {i+1}/{len(unique_dates)} dates, got {sum(len(r) for r in temporal_results)} records")
        time.sleep(1)  # Rate limit
    except Exception as e:
        failed_dates.append(end_date)
        log(f"  WARNING: Failed for date {end_date}: {str(e)[:80]}")
        time.sleep(2)

if temporal_results:
    temporal_df = pd.concat(temporal_results, ignore_index=True)
    log(f"  Total temporal partner records: {len(temporal_df)}")

    # Convert partner_lng_id to match merged dataset type
    temporal_df['partner_lng_id'] = pd.to_numeric(temporal_df['partner_lng_id'], errors='coerce')
    merged['partner_lng_id'] = pd.to_numeric(merged['partner_lng_id'], errors='coerce')

    # Merge temporal status into main dataset
    temporal_df = temporal_df.drop_duplicates(subset=['partner_lng_id', 'Sprint End Date'], keep='first')
    merged = merged.merge(
        temporal_df[['partner_lng_id', 'Sprint End Date', 'partner_status_at_survey', 'status_date']],
        on=['partner_lng_id', 'Sprint End Date'],
        how='left'
    )
    matched_temporal = merged['partner_status_at_survey'].notna().sum()
    log(f"  Rows with temporal partner status: {matched_temporal} / {len(merged)}")

    # Compare temporal vs current status
    if 'partner_status' in merged.columns:
        diff_mask = (merged['partner_status'].notna() &
                     merged['partner_status_at_survey'].notna() &
                     (merged['partner_status'] != merged['partner_status_at_survey']))
        log(f"  Rows where temporal != current status: {diff_mask.sum()}")
        if diff_mask.sum() > 0:
            log("  STATUS DRIFT EXAMPLES:")
            drift = merged[diff_mask][['partner_lng_id', 'partner_status', 'partner_status_at_survey', 'Sprint End Date']].head(5)
            for _, row in drift.iterrows():
                log(f"    Partner {row['partner_lng_id']}: current={row['partner_status']}, "
                    f"at_survey={row['partner_status_at_survey']}, survey_date={row['Sprint End Date']}")
else:
    log("  WARNING: No temporal partner data retrieved")
    merged['partner_status_at_survey'] = np.nan
    merged['status_date'] = np.nan

if failed_dates:
    log(f"  Failed dates: {len(failed_dates)}")
log()

# ======================================================================
# STEP 8: PRINCIPLED OUTLIER HANDLING
# ======================================================================
log("STEP 8: Principled outlier handling...")
log("-" * 40)
log()

# 8a. Install TAT: negative values are data errors, extreme positives need flagging
if 'install_tat_hours' in merged.columns:
    orig_count = merged['install_tat_hours'].notna().sum()
    neg_count = (merged['install_tat_hours'] < 0).sum()
    extreme_count = (merged['install_tat_hours'] > 720).sum()  # 30 days

    # Negative TAT = data error → set to NaN
    merged.loc[merged['install_tat_hours'] < 0, 'install_tat_hours'] = np.nan

    # Create binary flag for delayed installs (>72 hours = 3 days)
    merged['install_delayed'] = (merged['install_tat_hours'] > 72).astype(float)
    merged.loc[merged['install_tat_hours'].isna(), 'install_delayed'] = np.nan

    # Winsorize at 99th percentile for continuous use in models
    p99 = merged['install_tat_hours'].quantile(0.99)
    merged['install_tat_hours_winsorized'] = merged['install_tat_hours'].clip(upper=p99)

    log(f"  install_tat_hours:")
    log(f"    Negative values set to NaN: {neg_count}")
    log(f"    Extreme (>720h): {extreme_count} rows")
    log(f"    Winsorized at P99 ({p99:.0f}h): retains shape, caps extreme tail")
    log(f"    install_delayed flag created (>72h threshold)")
    log()

# 8b. avg_resolution_hours: negative values are data errors
if 'avg_resolution_hours' in merged.columns:
    neg_res = (merged['avg_resolution_hours'] < 0).sum()
    merged.loc[merged['avg_resolution_hours'] < 0, 'avg_resolution_hours'] = np.nan

    p99_res = merged['avg_resolution_hours'].quantile(0.99)
    merged['avg_resolution_hours_winsorized'] = merged['avg_resolution_hours'].clip(upper=p99_res)
    log(f"  avg_resolution_hours:")
    log(f"    Negative values set to NaN: {neg_res}")
    log(f"    Winsorized at P99 ({p99_res:.0f}h)")
    log()

# 8c. days_since_last_recharge: negative values = recharge after sprint end (timing issue)
if 'days_since_last_recharge' in merged.columns:
    neg_days = (merged['days_since_last_recharge'] < 0).sum()
    # Negative = recharge happened after sprint end → clamp to 0 (meaning "recent recharge")
    merged.loc[merged['days_since_last_recharge'] < 0, 'days_since_last_recharge'] = 0
    log(f"  days_since_last_recharge:")
    log(f"    Negative values clamped to 0: {neg_days}")
    log()

# 8d. Time-to-value: negative HOURS_TO_FIRST_RECHARGE means recharge before install (pre-paid setup)
if 'HOURS_TO_FIRST_RECHARGE' in merged.columns:
    neg_ttv = (merged['HOURS_TO_FIRST_RECHARGE'] < 0).sum()
    extreme_neg = (merged['HOURS_TO_FIRST_RECHARGE'] < -168).sum()  # More than 7 days before install

    # Extreme negatives (>7 days before install) are likely data errors
    merged.loc[merged['HOURS_TO_FIRST_RECHARGE'] < -168, 'HOURS_TO_FIRST_RECHARGE'] = np.nan
    merged.loc[merged['DAYS_TO_FIRST_RECHARGE'] < -7, 'DAYS_TO_FIRST_RECHARGE'] = np.nan

    # Create binary: was recharge same-day or not
    merged['recharge_same_day'] = (merged['DAYS_TO_FIRST_RECHARGE'].between(-1, 1)).astype(float)
    merged.loc[merged['DAYS_TO_FIRST_RECHARGE'].isna(), 'recharge_same_day'] = np.nan

    log(f"  HOURS_TO_FIRST_RECHARGE:")
    log(f"    Negative values (pre-install recharge): {neg_ttv}")
    log(f"    Extreme negatives (<-168h) set to NaN: {extreme_neg}")
    log(f"    recharge_same_day flag created")
    log()

# 8e. AVG_INSTALL_TAT_MINS: massive outliers
if 'AVG_INSTALL_TAT_MINS' in merged.columns:
    p99_mins = merged['AVG_INSTALL_TAT_MINS'].quantile(0.99)
    merged['AVG_INSTALL_TAT_MINS_winsorized'] = merged['AVG_INSTALL_TAT_MINS'].clip(upper=p99_mins)
    log(f"  AVG_INSTALL_TAT_MINS: Winsorized at P99 ({p99_mins:.0f} mins)")
    log()

# 8f. avg_uptime_pct: CAVEAT — this is partner fleet health, not customer uptime
# Keep but add flag about interpretation
log("  NOTE: avg_uptime_pct from Phase 3 is PARTNER FLEET HEALTH, not customer-experienced uptime.")
log("  Models should prioritize PEAK_UPTIME_PCT and OUTAGE_EVENTS from Industry Expert features.")
log()

# ======================================================================
# STEP 9: DERIVED FEATURES FOR MODELING
# ======================================================================
log("STEP 9: Creating derived features for modeling...")
log("-" * 40)

# 9a. Partner status risk flag (using TEMPORAL status)
status_col = 'partner_status_at_survey' if 'partner_status_at_survey' in merged.columns else 'partner_status'
if status_col in merged.columns:
    # Map statuses to risk levels
    # partner_details_log uses: Active, Blocked, Delisted
    # HIERARCHY_BASE uses: ACTIVE, TERMINATION, TEMPORARY SUSPENSION, CLOSED, BLACKLISTED
    risk_map = {
        'Active': 0, 'ACTIVE': 0,
        'Blocked': 1, 'TEMPORARY SUSPENSION': 1,
        'Delisted': 2, 'TERMINATION': 2, 'CLOSED': 2, 'BLACKLISTED': 2
    }
    merged['partner_risk_level'] = merged[status_col].map(risk_map)
    merged['partner_at_risk'] = (merged['partner_risk_level'] >= 1).astype(float)
    merged.loc[merged[status_col].isna(), 'partner_at_risk'] = np.nan
    merged.loc[merged[status_col].isna(), 'partner_risk_level'] = np.nan
    log(f"  partner_risk_level: 0=Active, 1=Blocked/Suspended, 2=Terminated/Delisted")
    log(f"  partner_at_risk: {merged['partner_at_risk'].sum():.0f} respondents with at-risk partners")

# 9b. Support effort index (composite of IVR calls + ticket reopenings + repeat complaints)
effort_cols = ['TOTAL_IVR_CALLS', 'AVG_TIMES_REOPENED', 'HAS_REPEAT_COMPLAINT', 'total_tickets']
available_effort = [c for c in effort_cols if c in merged.columns]
if len(available_effort) >= 2:
    # Z-score each and average for a composite
    for col in available_effort:
        col_valid = merged[col].dropna()
        if len(col_valid) > 0 and col_valid.std() > 0:
            merged[f'{col}_z'] = (merged[col] - col_valid.mean()) / col_valid.std()
        else:
            merged[f'{col}_z'] = 0

    z_cols = [f'{c}_z' for c in available_effort]
    merged['support_effort_index'] = merged[z_cols].mean(axis=1)
    # Drop z-score intermediates
    merged.drop(columns=z_cols, inplace=True)
    log(f"  support_effort_index: composite of {', '.join(available_effort)}")

# 9c. Network quality index (composite of peak uptime + outage events + recovery time)
net_cols_pos = ['PEAK_UPTIME_PCT']  # Higher = better
net_cols_neg = ['OUTAGE_EVENTS', 'AVG_RECOVERY_MINS', 'AVG_PEAK_INTERRUPTIONS']  # Higher = worse
net_available_pos = [c for c in net_cols_pos if c in merged.columns]
net_available_neg = [c for c in net_cols_neg if c in merged.columns]

if len(net_available_pos) + len(net_available_neg) >= 2:
    for col in net_available_pos:
        col_valid = merged[col].dropna()
        if len(col_valid) > 0 and col_valid.std() > 0:
            merged[f'{col}_z'] = (merged[col] - col_valid.mean()) / col_valid.std()
        else:
            merged[f'{col}_z'] = 0

    for col in net_available_neg:
        col_valid = merged[col].dropna()
        if len(col_valid) > 0 and col_valid.std() > 0:
            merged[f'{col}_z'] = -1 * (merged[col] - col_valid.mean()) / col_valid.std()  # Negate
        else:
            merged[f'{col}_z'] = 0

    all_net_z = [f'{c}_z' for c in net_available_pos + net_available_neg]
    merged['network_quality_index'] = merged[all_net_z].mean(axis=1)
    merged.drop(columns=all_net_z, inplace=True)
    log(f"  network_quality_index: composite of uptime, outages, recovery time")

# 9d. Missed call ratio (missed / total calls)
if 'MISSED_CALLS' in merged.columns and 'TOTAL_IVR_CALLS' in merged.columns:
    merged['missed_call_ratio'] = merged['MISSED_CALLS'] / merged['TOTAL_IVR_CALLS'].replace(0, np.nan)
    log(f"  missed_call_ratio: proportion of calls that went unanswered")

# 9e. Autopay adoption ratio
if 'autopay_payments' in merged.columns and 'total_payments' in merged.columns:
    merged['autopay_ratio'] = merged['autopay_payments'] / merged['total_payments'].replace(0, np.nan)
    log(f"  autopay_ratio: proportion of payments via autopay")

# 9f. Ticket severity (reopened + repeat = chronic issues)
if 'AVG_TIMES_REOPENED' in merged.columns and 'MAX_TICKETS_SAME_ISSUE' in merged.columns:
    merged['ticket_severity'] = merged['AVG_TIMES_REOPENED'] * merged['MAX_TICKETS_SAME_ISSUE']
    log(f"  ticket_severity: reopenings * repeat tickets (chronic issue indicator)")

log()

# ======================================================================
# STEP 10: CLEAN UP AND SAVE
# ======================================================================
log("STEP 10: Final cleanup and save...")
log("-" * 40)

# Drop internal merge keys
drop_cols = ['_merge_key', '_theme_key']
for col in drop_cols:
    if col in merged.columns:
        merged.drop(columns=[col], inplace=True)

# Final column count
log(f"  Final dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
log()

# Column inventory by source
log("  COLUMN INVENTORY:")
all_cols = set(merged.columns)
log(f"    Base/Excel columns: {len([c for c in base_cols if c in all_cols])}")
log(f"    Snowflake Phase 3 features: {len([c for c in snowflake_cols if c in all_cols])}")
log(f"    Industry Expert Phase 3b features: {len([c for c in industry_feature_cols if c in all_cols])}")
log(f"    NLP theme features: {len([c for c in theme_feature_cols if c in all_cols])}")
sprint_extras = [c for c in expanded_extra_cols if c in all_cols]
log(f"    Sprint tab extras: {len(sprint_extras)}")
derived = ['install_delayed', 'install_tat_hours_winsorized', 'avg_resolution_hours_winsorized',
           'recharge_same_day', 'AVG_INSTALL_TAT_MINS_winsorized', 'partner_risk_level',
           'partner_at_risk', 'support_effort_index', 'network_quality_index',
           'missed_call_ratio', 'autopay_ratio', 'ticket_severity',
           'partner_status_at_survey', 'status_date']
derived_present = [c for c in derived if c in all_cols]
log(f"    Derived/temporal features: {len(derived_present)}")
log(f"    Other/flags: {merged.shape[1] - len([c for c in base_cols if c in all_cols]) - len([c for c in snowflake_cols if c in all_cols]) - len([c for c in industry_feature_cols if c in all_cols]) - len([c for c in theme_feature_cols if c in all_cols]) - len(sprint_extras) - len(derived_present)}")
log()

# Coverage report
log("  FEATURE COVERAGE (% of rows with non-null):")
coverage_features = [
    ('Snowflake features', 'total_recharges'),
    ('Industry Expert features', 'TOTAL_IVR_CALLS' if 'TOTAL_IVR_CALLS' in merged.columns else None),
    ('NLP themes', 'primary_theme'),
    ('Temporal partner status', 'partner_status_at_survey'),
    ('Sprint tab extras', 'device_type' if 'device_type' in merged.columns else None),
    ('Outage data', 'OUTAGE_EVENTS' if 'OUTAGE_EVENTS' in merged.columns else None),
    ('Data usage', 'AVG_DAILY_DATA_GB' if 'AVG_DAILY_DATA_GB' in merged.columns else None),
]
for label, col in coverage_features:
    if col and col in merged.columns:
        pct = merged[col].notna().sum() / len(merged) * 100
        log(f"    {label}: {merged[col].notna().sum()}/{len(merged)} ({pct:.1f}%)")
    else:
        log(f"    {label}: N/A")
log()

# NPS group distribution
log("  NPS GROUP DISTRIBUTION:")
for group in ['Promoter', 'Passive', 'Detractor']:
    count = (merged['nps_group'] == group).sum()
    pct = count / len(merged) * 100
    log(f"    {group}: {count} ({pct:.1f}%)")
log()

# Source distribution
log("  DATA SOURCE DISTRIBUTION:")
for src in merged['_source'].value_counts().items():
    log(f"    {src[0]}: {src[1]} ({src[1]/len(merged)*100:.1f}%)")
log()

# Save
output_path = os.path.join(DATA, "nps_modeling_dataset.csv")
merged.to_csv(output_path, index=False, encoding='utf-8-sig')
log(f"  Saved: {output_path}")
log(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

# Save report
report_path = os.path.join(OUTPUT, "phase3c_merge_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"  Report: {report_path}")

log()
log("=" * 70)
log("PHASE 3C COMPLETE")
log("=" * 70)
log()
log("ADAPTIVE NOTES FOR PHASE 4 MODELING:")
log("-" * 40)
log("1. USE temporal partner_status_at_survey, NOT current partner_status (data leakage)")
log("2. PREFER PEAK_UPTIME_PCT and OUTAGE_EVENTS over avg_uptime_pct (partner fleet vs customer)")
log("3. STRONGEST signals: AVG_TIMES_REOPENED, INBOUND_CALLS, MAX_TICKETS_SAME_ISSUE, TOTAL_IVR_CALLS")
log("4. WEAKEST signals: payment failure rate, peak uptime gap, dispatch decline rate")
log("5. install_tat_hours_winsorized for continuous use; install_delayed for binary")
log("6. support_effort_index and network_quality_index as composite predictors")
log("7. partner_at_risk as binary flag for at-risk partner situations")
log("8. 3,559 new rows (Sprint 14 + RSP1-3) lack Snowflake features — model on consolidated rows")
