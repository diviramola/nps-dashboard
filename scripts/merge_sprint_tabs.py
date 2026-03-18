"""
merge_sprint_tabs.py
====================
Reads individual sprint tabs (Sprint 3 through Sprint RSP3) from the NPS Excel file,
merges them with the existing Consolidated-based nps_clean_base.csv to create an
expanded dataset with extra columns from sprint tabs.

Sprint 1 and Sprint 2 tabs contain only summary/pivot tables (no row-level data),
so they are skipped for merging. Their data is already in the Consolidated tab.

Outputs:
  data/nps_expanded_base.csv         — full expanded dataset
  output/merge_sprint_tabs_report.txt — summary report
"""

import sys
import io
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\nikhi\wiom-nps-analysis'
EXCEL_PATH = r'C:\Users\nikhi\Downloads\NPS Verma Parivar.xlsx'
CLEAN_BASE_PATH = os.path.join(BASE_DIR, 'data', 'nps_clean_base.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'nps_expanded_base.csv')
OUTPUT_REPORT = os.path.join(BASE_DIR, 'output', 'merge_sprint_tabs_report.txt')

# ── Sprint tab config ─────────────────────────────────────────────────────────
# Tab name -> (sprint_id for matching, sprint_num, is_post_reset)
# Sprint 1 and 2 tabs are summary-only; their data is already in Consolidated.
SPRINT_TABS = {
    'Sprint 3 Aug CT':     ('Sprint 3',    3,  False),
    'Sprint 4 Aug':        ('Sprint 4',    4,  False),
    'Sprint 5 Sep':        ('Sprint 5',    5,  False),
    'Sprint 6 Sep':        ('Sprint 6',    6,  False),
    'Sprint 7 Oct25':      ('Sprint 7',    7,  False),
    'Sprint 8 Oct25':      ('Sprint 8',    8,  False),
    'Sprint 9 Nov25':      ('Sprint 9',    9,  False),
    'Sprint 10 Nov25':     ('Sprint 10',  10,  False),
    'Sprint 11 Dec25':     ('Sprint 11',  11,  False),
    'Sprint 12 Dec25':     ('Sprint 12',  12,  False),
    'Sprint 13 Jan26':     ('Sprint 13',  13,  False),
    'Sprint 14 Jan26':     ('Sprint 14',  14,  False),
    'Sprint RSP1 Feb26':   ('Sprint RSP1', 15, True),
    'Sprint RSP2 Feb26':   ('Sprint RSP2', 16, True),
    'Sprint RSP3 Feb26':   ('Sprint RSP3', 17, True),
}

# ── Column mapping: sprint tab column names -> standardized names ─────────────
# These handle the fact that columns have slightly different names across sprints.
# We map to either existing Consolidated column names or new extra column names.

CORE_COL_MAP = {
    'USER_RATING':        'nps_score',
    'NPS_CLASSIFICATION': 'nps_group',
    'COMMENT':            'OE',
    'Mobile':             'phone_number',
    'City':               'city',
    'City ':              'city',       # trailing space variant
    'city ':              'city',       # lowercase variant
    'city':               'city',
    'Channel':            'Channel',
    'Channel ':           'Channel',
    'Tenure':             'tenure_excel',
    'USER_ID':            'user_id',
    'PROFILE_ALL_IDENTITIES': 'profile_all_identities',
    'TIMESTAMP':          'timestamp',
}

# Extra columns from sprint tabs NOT in Consolidated
EXTRA_COL_MAP = {
    # Device info
    'Device Type':        'device_type',
    'Device Type ':       'device_type',
    'Device type':        'device_type',
    'Device type ':       'device_type',
    'Device':             'device_model',

    # Network quality
    'Optical Power':      'optical_power',
    'Optical Power ':     'optical_power',
    'Optical Range':      'optical_range',
    'Optical Range ':     'optical_range',

    # Tickets
    '# of tickets in last 3 months':  'tickets_last_3m_sprint',
    '# of tickets in last 3 months ': 'tickets_last_3m_sprint',
    '# of tx in last 3 months':       'tickets_last_3m_sprint',
    '# of tx in last 3 months ':      'tickets_last_3m_sprint',
    'Payment_issue\nTickets Flag':     'payment_issue_tickets_flag',
    'Payment_issue\n Tickets Flag':    'payment_issue_tickets_flag',
    'Payment_issue Tickets Flag':      'payment_issue_tickets_flag',

    # WiFi devices
    'Devices on 2.4g':    'devices_2_4g',
    'Devices on 5g':      'devices_5g',

    # Data usage
    'Data Usage amount':  'data_usage_amount',
    'Data Usage amount ': 'data_usage_amount',
    'Data Usage Days':    'data_usage_days',
    'Data Usage Days ':   'data_usage_days',
    'Data Usage Percentile':  'data_usage_percentile',
    'Data Usage Percentile ': 'data_usage_percentile',
    'Data Usage / Day':   'data_usage_per_day',

    # Recharge / payment
    'Last recharge done - when?':     'last_recharge_date',
    'Last recharge done - when? ':    'last_recharge_date',
    'recharge done':      'recharge_done',
    'recharge done ':     'recharge_done',
    'cash/ online':       'cash_online',
    'cash/online':        'cash_online',

    # Plan
    'Plan Expiry Date':   'plan_expiry_date',
    'Plan Expiry Date ':  'plan_expiry_date',
    'Plan expiry date':   'plan_expiry_date',
    'Plan Expiry Window': 'plan_expiry_window',
    'Plan Expiry Window ': 'plan_expiry_window',

    # Partner / Zone
    'Partner':            'partner_name',
    'Partner ':           'partner_name',
    'Zone':               'zone',
    'Zone ':              'zone',
    'SLA':                'sla',
    'SLA ':               'sla',

    # Install date from sprint tab
    'Install Date':       'install_date_sprint',
    'Install Date ':      'install_date_sprint',
    'Installation Date':  'install_date_sprint',
    'Installation Date ': 'install_date_sprint',

    # First time user
    'First time user?':   'first_time_user',
    'First time user? ':  'first_time_user',

    # NPS Reason from sprint tabs
    'NPS Reason':              'nps_reason_sprint',
    'NPS Reason - primary':    'nps_reason_primary_sprint',
    'NPS Reason - secondary':  'nps_reason_secondary_sprint',
    'NPS reason tertiary':     'nps_reason_tertiary_sprint',

    # Calling/status columns
    'ALT mobile':         'alt_mobile',
    'ALT mobile ':        'alt_mobile',
    'Caller':             'caller',
    'Caller ':            'caller',
    'Status':             'call_status',
    'Response ID':        'response_id',
    'Respondent ID':      'response_id',

    # CX follow-up columns
    'Overall Service Experience':  'overall_service_experience',
    'Overall Service Experience ': 'overall_service_experience',
    'Comments':           'comments_sprint',
    'Comments ':          'comments_sprint',
    'NPS OE - Copied from comment': 'nps_oe_copied',

    # PayG
    'pay g?':             'pay_g',
    'pay g? ':            'pay_g',

    # Tenure from install
    'Tenure - install':    'tenure_from_install',
    'Tenure - install ':   'tenure_from_install',
    'Tenure // Install':   'tenure_from_install',
    'Tenure // Install ':  'tenure_from_install',

    # Previous NPS
    'PREV NPS':           'prev_nps',
    'MATCH?':             'nps_match',
}


def clean_phone(val):
    """Convert phone value to clean string, stripping .0 from floats."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove .0 suffix from float representation
    if s.endswith('.0'):
        s = s[:-2]
    # Remove non-digit characters
    s = ''.join(c for c in s if c.isdigit())
    if len(s) == 0:
        return None
    # 10-digit Indian mobile
    if len(s) == 12 and s.startswith('91'):
        s = s[2:]
    if len(s) != 10:
        return None
    return s


def clean_col_name(col):
    """Strip whitespace and newlines from column names for matching."""
    if isinstance(col, str):
        return col.strip().replace('\n', ' ').replace('\r', '')
    return str(col)


def compute_tenure_bucket(days):
    """Assign tenure bucket based on days."""
    if pd.isna(days) or days < 0:
        return None
    if days <= 15:
        return 'Onboarding (0-15d)'
    elif days <= 60:
        return 'Early Life (16-60d)'
    elif days <= 120:
        return 'Establishing (61-120d)'
    elif days <= 270:
        return 'Steady State (121-270d)'
    else:
        return 'Loyal (270+d)'


def read_sprint_tab(xlsx_path, tab_name, sprint_id, sprint_num, is_rsp):
    """
    Read a single sprint tab from the Excel file.
    Returns a DataFrame with standardized column names.
    """
    print(f'  Reading: {tab_name} ...', end=' ')
    df = pd.read_excel(xlsx_path, sheet_name=tab_name, header=13, engine='openpyxl')

    # Drop fully empty rows
    df = df.dropna(how='all').reset_index(drop=True)
    print(f'{len(df)} rows')

    # Build column mapping for this tab
    # Track which target names we've already seen to avoid duplicates
    col_rename = {}
    extra_cols_found = []
    seen_targets = set()

    for orig_col in df.columns:
        cleaned = clean_col_name(orig_col)
        target = None
        # Check core mapping
        if cleaned in CORE_COL_MAP:
            target = CORE_COL_MAP[cleaned]
        # Check extra mapping
        elif cleaned in EXTRA_COL_MAP:
            target = EXTRA_COL_MAP[cleaned]

        if target is not None:
            if target in seen_targets:
                # Duplicate target — skip this column (first one wins)
                continue
            col_rename[orig_col] = target
            seen_targets.add(target)
            if cleaned in EXTRA_COL_MAP:
                extra_cols_found.append(target)

    # Rename columns
    df = df.rename(columns=col_rename)

    # Keep only mapped columns (drop Unnamed and unmapped)
    keep_cols = list(col_rename.values())
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Clean phone numbers
    if 'phone_number' in df.columns:
        df['phone_number'] = df['phone_number'].apply(clean_phone)
        # Drop rows without phone
        df = df.dropna(subset=['phone_number']).reset_index(drop=True)

    # Clean NPS score - handle #REF! and non-numeric
    if 'nps_score' in df.columns:
        df['nps_score'] = pd.to_numeric(df['nps_score'], errors='coerce')

    # Add sprint metadata
    df['Sprint ID'] = sprint_id
    df['sprint_num'] = sprint_num
    df['is_post_reset'] = is_rsp
    df['source_tab'] = tab_name

    return df, extra_cols_found


def main():
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log('=' * 70)
    log('MERGE SPRINT TABS — NPS Expanded Base')
    log(f'Run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log('=' * 70)
    log('')

    # ── Step 1: Load existing clean base ──────────────────────────────────
    log('Step 1: Loading nps_clean_base.csv ...')
    base = pd.read_csv(CLEAN_BASE_PATH)
    base['phone_number'] = base['phone_number'].apply(clean_phone)
    log(f'  Loaded: {len(base)} rows, {len(base.columns)} columns')
    log(f'  Sprints in base: {sorted(base["Sprint ID"].unique())}')
    log('')

    # ── Step 2: Read all sprint tabs ──────────────────────────────────────
    log('Step 2: Reading sprint tabs from Excel ...')
    all_sprint_dfs = []
    all_extra_cols = set()

    for tab_name, (sprint_id, sprint_num, is_rsp) in SPRINT_TABS.items():
        df_tab, extra_found = read_sprint_tab(EXCEL_PATH, tab_name, sprint_id, sprint_num, is_rsp)
        all_sprint_dfs.append(df_tab)
        all_extra_cols.update(extra_found)
        log(f'    {tab_name}: {len(df_tab)} valid rows, {len(extra_found)} extra cols')

    sprint_all = pd.concat(all_sprint_dfs, ignore_index=True)
    log(f'\n  Total sprint tab rows: {len(sprint_all)}')
    log(f'  Extra columns found across tabs: {sorted(all_extra_cols)}')
    log('')

    # ── Step 3: Identify missing records ──────────────────────────────────
    log('Step 3: Identifying records NOT in Consolidated ...')

    # Create match key: phone_number + Sprint ID
    base['_match_key'] = base['phone_number'].astype(str) + '|' + base['Sprint ID'].astype(str)
    sprint_all['_match_key'] = sprint_all['phone_number'].astype(str) + '|' + sprint_all['Sprint ID'].astype(str)

    base_keys = set(base['_match_key'].unique())
    sprint_all['_in_base'] = sprint_all['_match_key'].isin(base_keys)

    new_records = sprint_all[~sprint_all['_in_base']].copy()
    existing_matches = sprint_all[sprint_all['_in_base']].copy()

    log(f'  Sprint tab records matching Consolidated: {len(existing_matches)}')
    log(f'  Sprint tab records NOT in Consolidated (new): {len(new_records)}')
    log('')

    # Breakdown by sprint
    log('  New records by sprint:')
    for sid in sorted(new_records['Sprint ID'].unique(), key=lambda x: SPRINT_TABS.get(
            next((k for k, v in SPRINT_TABS.items() if v[0] == x), ''), ('', 99, False))[1]):
        cnt = len(new_records[new_records['Sprint ID'] == sid])
        log(f'    {sid}: {cnt}')
    log('')

    # ── Step 4: Enrich existing Consolidated rows with extra columns ──────
    log('Step 4: Enriching Consolidated rows with extra columns ...')

    # For existing matches, keep only extra columns + match key (one row per key)
    extra_col_list = sorted(all_extra_cols)
    enrich_cols = ['_match_key'] + [c for c in extra_col_list if c in existing_matches.columns]

    if len(enrich_cols) > 1:
        # Deduplicate: if same phone+sprint appears multiple times in sprint tab, take first
        enrichment = existing_matches[enrich_cols].drop_duplicates(subset='_match_key', keep='first')
        log(f'  Enrichment rows available: {len(enrichment)}')

        # Merge extra columns onto base
        base_enriched = base.merge(enrichment, on='_match_key', how='left')
        # Count rows that have at least one extra column filled
        extra_in_enriched = [c for c in extra_col_list if c in base_enriched.columns]
        if extra_in_enriched:
            enriched_count = base_enriched[extra_in_enriched].notna().any(axis=1).sum()
        else:
            enriched_count = 0
        log(f'  Base rows enriched with at least one extra col: {enriched_count}')
    else:
        base_enriched = base.copy()
        log('  No extra columns to enrich.')
    log('')

    # ── Step 5: Prepare new records for appending ─────────────────────────
    log('Step 5: Preparing new records for appending ...')

    # Map new record columns to match base schema
    # Core columns that new records should have
    core_base_cols = [
        'phone_number', 'nps_score', 'nps_group', 'OE', 'Channel',
        'Sprint ID', 'sprint_num', 'city', 'tenure_excel'
    ]

    # Add has_comment
    if 'OE' in new_records.columns:
        new_records['has_comment'] = new_records['OE'].notna() & (new_records['OE'].astype(str).str.strip() != '')
    else:
        new_records['has_comment'] = False

    # phone_valid
    new_records['phone_valid'] = new_records['phone_number'].notna()

    # Map NPS reason columns if they exist
    reason_map = {
        'nps_reason_primary_sprint':   'NPS Reason - Primary',
        'nps_reason_secondary_sprint': 'NPS Reason - Secondary',
        'nps_reason_tertiary_sprint':  'NPS Reason - Tertiary',
    }
    for src, dst in reason_map.items():
        if src in new_records.columns:
            new_records[dst] = new_records[src]

    # Install Date from sprint tab
    if 'install_date_sprint' in new_records.columns:
        new_records['Install Date'] = pd.to_datetime(
            new_records['install_date_sprint'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')
    elif 'Install Date' not in new_records.columns:
        new_records['Install Date'] = None

    # Compute tenure_days for new records
    # Use Sprint End Date from sprint metadata or a reference date
    sprint_end_dates = {
        'Sprint 3':    '2025-08-31',
        'Sprint 4':    '2025-08-31',
        'Sprint 5':    '2025-09-30',
        'Sprint 6':    '2025-09-30',
        'Sprint 7':    '2025-10-31',
        'Sprint 8':    '2025-10-31',
        'Sprint 9':    '2025-11-30',
        'Sprint 10':   '2025-11-30',
        'Sprint 11':   '2025-12-31',
        'Sprint 12':   '2025-12-31',
        'Sprint 13':   '2026-01-31',
        'Sprint 14':   '2026-01-31',
        'Sprint RSP1': '2026-02-28',
        'Sprint RSP2': '2026-02-28',
        'Sprint RSP3': '2026-03-15',
    }

    def calc_tenure(row):
        install = row.get('Install Date')
        sprint_id = row.get('Sprint ID')
        if pd.isna(install) or install is None or str(install).strip() == '':
            return np.nan
        try:
            install_dt = pd.to_datetime(install)
            end_str = sprint_end_dates.get(sprint_id)
            if end_str:
                end_dt = pd.to_datetime(end_str)
            else:
                end_dt = pd.Timestamp('2026-03-15')
            return (end_dt - install_dt).days
        except:
            return np.nan

    new_records['tenure_days'] = new_records.apply(calc_tenure, axis=1)
    new_records['tenure_bucket'] = new_records['tenure_days'].apply(compute_tenure_bucket)

    # is_post_reset flag
    new_records['is_post_reset'] = new_records['Sprint ID'].isin(['Sprint RSP1', 'Sprint RSP2', 'Sprint RSP3'])

    new_row_count = len(new_records)
    log(f'  New records prepared: {new_row_count}')
    log('')

    # ── Step 6: Combine into expanded dataset ─────────────────────────────
    log('Step 6: Building expanded dataset ...')

    # Add is_post_reset to base_enriched
    base_enriched['is_post_reset'] = base_enriched['Sprint ID'].isin(['Sprint RSP1', 'Sprint RSP2', 'Sprint RSP3'])

    # Add source indicator
    base_enriched['_source'] = 'consolidated'
    new_records['_source'] = 'sprint_tab_new'

    # Get all columns from both
    all_cols = sorted(set(base_enriched.columns) | set(new_records.columns))
    # Remove internal columns
    all_cols = [c for c in all_cols if c not in ('_match_key', '_in_base', 'source_tab')]

    # Ensure both DataFrames have all columns
    for c in all_cols:
        if c not in base_enriched.columns:
            base_enriched[c] = np.nan
        if c not in new_records.columns:
            new_records[c] = np.nan

    # Concatenate
    expanded = pd.concat([base_enriched[all_cols], new_records[all_cols]], ignore_index=True)

    # ── Step 7: Add is_payg flag ──────────────────────────────────────────
    log('Step 7: Adding is_payg flag ...')

    def check_payg(row):
        install = row.get('Install Date')
        if pd.isna(install) or install is None or str(install).strip() == '':
            return False
        try:
            install_dt = pd.to_datetime(install)
            return install_dt >= pd.Timestamp('2026-01-26')
        except:
            return False

    expanded['is_payg'] = expanded.apply(check_payg, axis=1)

    payg_count = expanded['is_payg'].sum()
    log(f'  PayG customers (install >= 2026-01-26): {payg_count}')
    log('')

    # ── Step 8: Save outputs ──────────────────────────────────────────────
    log('Step 8: Saving outputs ...')

    # Ensure output directories exist
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'output'), exist_ok=True)

    # Reorder columns: core first, then extras
    core_order = [
        'phone_number', 'phone_valid', 'nps_score', 'nps_group', 'OE', 'has_comment',
        'NPS Reason - Primary', 'NPS Reason - Secondary', 'NPS Reason - Tertiary',
        'Primary Category', 'Secondary Category', 'Tertiary Category',
        'Channel', 'first_time_wifi', 'Sprint ID', 'sprint_num',
        'Cycle ID', 'Sprint Start Date', 'Sprint End Date',
        'Install Date', 'tenure_days', 'tenure_bucket', 'tenure_excel',
        'city', 'churn_label', 'is_churned',
        'recharges_before_sprint', 'tickets_post_sprint', 'tickets_before_3m',
        'is_post_reset', 'is_payg', '_source',
    ]

    # Hindi CX columns from base
    hindi_cols = [c for c in expanded.columns if c not in core_order and c not in sorted(all_extra_cols) and not c.startswith('_')]
    extra_ordered = sorted([c for c in all_extra_cols if c in expanded.columns])

    final_order = [c for c in core_order if c in expanded.columns] + hindi_cols + extra_ordered
    # Add any remaining columns
    remaining = [c for c in expanded.columns if c not in final_order and not c.startswith('_')]
    final_order += remaining

    expanded = expanded[final_order]

    expanded.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    log(f'  Saved: {OUTPUT_CSV}')
    log(f'  Shape: {expanded.shape}')
    log('')

    # ── Summary stats ─────────────────────────────────────────────────────
    log('=' * 70)
    log('SUMMARY')
    log('=' * 70)
    log(f'Original Consolidated rows:   {len(base)}')
    log(f'New rows from sprint tabs:    {new_row_count}')
    log(f'Total expanded rows:          {len(expanded)}')
    log(f'Total columns:                {len(expanded.columns)}')
    log(f'Extra columns added:          {len(extra_ordered)}')
    log('')

    log('Rows by Sprint ID:')
    sprint_counts = expanded['Sprint ID'].value_counts()
    for sid in sorted(sprint_counts.index, key=lambda x: (
        0 if x.startswith('Sprint RSP') else 1,
        int(''.join(c for c in x if c.isdigit()) or '0')
    )):
        src_breakdown = expanded[expanded['Sprint ID'] == sid]['_source'].value_counts().to_dict()
        log(f'  {sid:20s}: {sprint_counts[sid]:5d}  ({src_breakdown})')
    log('')

    log('Rows by source:')
    log(expanded['_source'].value_counts().to_string())
    log('')

    log('is_post_reset breakdown:')
    log(expanded['is_post_reset'].value_counts().to_string())
    log('')

    log('is_payg breakdown:')
    log(expanded['is_payg'].value_counts().to_string())
    log('')

    log('Extra columns and their fill rates:')
    for col in extra_ordered:
        non_null = expanded[col].notna().sum()
        pct = non_null / len(expanded) * 100
        log(f'  {col:40s}: {non_null:6d} / {len(expanded)} ({pct:5.1f}%)')
    log('')

    # Save report
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f'Report saved: {OUTPUT_REPORT}')
    log('Done.')


if __name__ == '__main__':
    main()
