"""
Phase 0: Parse NPS Excel Data & Build Clean Analytical Base
============================================================
Wiom NPS Driver Analysis
Owner: DB Expert

Reads the Consolidated sheet from NPS Excel, cleans data, computes tenure buckets,
and produces a clean CSV + data profile report.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys
import io
import os
from datetime import datetime

# Handle Unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
EXCEL_PATH = r'C:\Users\nikhi\Downloads\NPS Verma Parivar.xlsx'
OUTPUT_DIR = r'C:\Users\nikhi\wiom-nps-analysis\data'
REPORT_DIR = r'C:\Users\nikhi\wiom-nps-analysis\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 0: NPS Data Parsing & Cleaning")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 1. Read Consolidated sheet
# ──────────────────────────────────────────────────────────────────────
print("\n[1/7] Reading Consolidated sheet...")
df = pd.read_excel(EXCEL_PATH, sheet_name='Consolidated', engine='openpyxl')
print(f"  Raw shape: {df.shape}")

# Drop fully empty columns
df = df.dropna(axis=1, how='all')
# Drop unnamed columns that are just artifacts
unnamed_cols = [c for c in df.columns if str(c).startswith('Unnamed:')]
# Keep them for now but rename
for i, c in enumerate(unnamed_cols):
    df = df.rename(columns={c: f'_extra_{i}'})

print(f"  After cleanup: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ──────────────────────────────────────────────────────────────────────
# 2. Clean Phone Numbers
# ──────────────────────────────────────────────────────────────────────
print("\n[2/7] Cleaning phone numbers...")
df['phone_number'] = df['Phone Number'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
# Remove any non-digit characters
df['phone_number'] = df['phone_number'].str.replace(r'[^\d]', '', regex=True)
# Validate: should be 10 digits
df['phone_valid'] = df['phone_number'].str.match(r'^\d{10}$')
print(f"  Valid 10-digit phones: {df['phone_valid'].sum()} / {len(df)} ({df['phone_valid'].mean()*100:.1f}%)")
print(f"  Invalid phones (sample): {df[~df['phone_valid']]['phone_number'].head(5).tolist()}")

# ──────────────────────────────────────────────────────────────────────
# 3. Parse Dates
# ──────────────────────────────────────────────────────────────────────
print("\n[3/7] Parsing dates...")
for col in ['Sprint Start Date', 'Sprint End Date', 'Install Date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print(f"  Sprint Start Date: {df['Sprint Start Date'].notna().sum()} valid")
print(f"  Sprint End Date: {df['Sprint End Date'].notna().sum()} valid")
print(f"  Install Date: {df['Install Date'].notna().sum()} valid ({df['Install Date'].notna().mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 4. Standardize columns
# ──────────────────────────────────────────────────────────────────────
print("\n[4/7] Standardizing columns...")

# Use City core (more reliable than City which has #REF! errors)
df['city'] = df['City core'].fillna('Unknown').astype(str).str.strip()
# Standardize city names
city_map = {
    'Delhi': 'Delhi', 'delhi': 'Delhi',
    'Mumbai': 'Mumbai', 'mumbai': 'Mumbai',
    'UP': 'UP', 'up': 'UP',
    '#REF!': 'Unknown', 'nan': 'Unknown', '': 'Unknown'
}
df['city'] = df['city'].map(city_map).fillna(df['city'])

# Standardize First Time WiFi
ftw_col = 'First Time wifi?'
if ftw_col in df.columns:
    ftw_map = {
        'Yes - Wiom is first wifi': 'Yes',
        'Yes - wiom is first wifi': 'Yes',
        'हाँ – व्योम पहले वाई-फ़ाई है': 'Yes',
        'No - Had wifi before': 'No',
        'No - had wifi before': 'No',
        'नहीं पहले भी लगा था': 'No',
    }
    df['first_time_wifi'] = df[ftw_col].map(ftw_map).fillna('Unknown')
else:
    df['first_time_wifi'] = 'Unknown'

# Standardize NPS Group
df['nps_group'] = df['NPS Group'].astype(str).str.strip()
df['nps_score'] = pd.to_numeric(df['NPS'], errors='coerce')

# Standardize churn label
df['churn_label'] = df['Active/ Churn - overall'].astype(str).str.strip()
df['is_churned'] = (df['churn_label'] == 'Churn').astype(int)

# ──────────────────────────────────────────────────────────────────────
# 5. Compute Tenure Buckets
# ──────────────────────────────────────────────────────────────────────
print("\n[5/7] Computing tenure buckets...")

# Compute exact tenure in days from Install Date to Sprint End Date
df['tenure_days'] = np.nan
mask = df['Install Date'].notna() & df['Sprint End Date'].notna()
df.loc[mask, 'tenure_days'] = (df.loc[mask, 'Sprint End Date'] - df.loc[mask, 'Install Date']).dt.days

# Remove negative tenure (data errors)
df.loc[df['tenure_days'] < 0, 'tenure_days'] = np.nan

print(f"  Exact tenure computed for: {df['tenure_days'].notna().sum()} / {len(df)} ({df['tenure_days'].notna().mean()*100:.1f}%)")
print(f"  Tenure stats: min={df['tenure_days'].min():.0f}, median={df['tenure_days'].median():.0f}, "
      f"mean={df['tenure_days'].mean():.0f}, max={df['tenure_days'].max():.0f}")

# Assign tenure buckets
def assign_tenure_bucket(days):
    if pd.isna(days):
        return 'Unknown'
    if days <= 15:
        return 'Onboarding (0-15d)'
    elif days <= 60:
        return 'Early Life (16-60d)'
    elif days <= 120:
        return 'Establishing (61-120d)'
    elif days <= 270:
        return 'Steady State (121-270d)'
    else:
        return 'Loyal (270d+)'

df['tenure_bucket'] = df['tenure_days'].apply(assign_tenure_bucket)

# Also keep original Excel tenure for comparison
df['tenure_excel'] = df['Tenure'].astype(str).str.strip()

print(f"\n  Computed tenure bucket distribution:")
print(df['tenure_bucket'].value_counts().to_string())
print(f"\n  Original Excel tenure distribution:")
print(df['tenure_excel'].value_counts().to_string())

# Cross-validate: compare computed vs Excel buckets
print(f"\n  Cross-validation (computed vs Excel):")
ct = pd.crosstab(df['tenure_bucket'], df['tenure_excel'], margins=True)
print(ct.to_string())

# ──────────────────────────────────────────────────────────────────────
# 6. Additional computed columns
# ──────────────────────────────────────────────────────────────────────
print("\n[6/7] Computing additional features...")

# Recharges
df['recharges_before_sprint'] = pd.to_numeric(df.get('Number of recharges before end of sprint?'), errors='coerce')
df['tickets_post_sprint'] = pd.to_numeric(df.get('Number of tickets post sprint date?'), errors='coerce')
df['tickets_before_3m'] = pd.to_numeric(df.get('Number of tickets before 3M sprint date?'), errors='coerce')

# Has OE comment flag
df['has_comment'] = df['OE'].notna() & (df['OE'].astype(str).str.strip() != '')

# Sprint number (extract numeric)
df['sprint_num'] = df['Sprint ID'].astype(str).str.extract(r'(\d+)').astype(float)

print(f"  Recharges filled: {df['recharges_before_sprint'].notna().sum()}")
print(f"  Tickets post filled: {df['tickets_post_sprint'].notna().sum()}")
print(f"  Has OE comment: {df['has_comment'].sum()} ({df['has_comment'].mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 7. Save clean base + Tags codebook
# ──────────────────────────────────────────────────────────────────────
print("\n[7/7] Saving outputs...")

# Select and order key columns for clean base
clean_cols = [
    'phone_number', 'phone_valid', 'nps_score', 'nps_group',
    'OE', 'has_comment',
    'NPS Reason - Primary', 'NPS Reason - Secondary', 'NPS Reason - Tertiary',
    'Primary Category', 'Secondary Category', 'Tertiary Category',
    'Channel', 'first_time_wifi',
    'Sprint ID', 'sprint_num', 'Cycle ID',
    'Sprint Start Date', 'Sprint End Date', 'Install Date',
    'tenure_days', 'tenure_bucket', 'tenure_excel',
    'city', 'churn_label', 'is_churned',
    'recharges_before_sprint', 'tickets_post_sprint', 'tickets_before_3m',
]
# Add Hindi survey columns if they exist
hindi_cols = [c for c in df.columns if any(x in str(c) for x in ['Wi-Fi', 'स्पीड', 'समस्या', 'dikkat'])]
clean_cols.extend(hindi_cols)
# Only include columns that exist
clean_cols = [c for c in clean_cols if c in df.columns]

df_clean = df[clean_cols].copy()
df_clean.to_csv(os.path.join(OUTPUT_DIR, 'nps_clean_base.csv'), index=False, encoding='utf-8-sig')
print(f"  Saved nps_clean_base.csv ({len(df_clean)} rows, {len(df_clean.columns)} columns)")

# Read and save Tags codebook
print("  Reading Tags sheet...")
try:
    df_tags = pd.read_excel(EXCEL_PATH, sheet_name='Tags', engine='openpyxl')
    df_tags.to_csv(os.path.join(OUTPUT_DIR, 'tags_codebook.csv'), index=False, encoding='utf-8-sig')
    print(f"  Saved tags_codebook.csv ({len(df_tags)} rows)")
except Exception as e:
    print(f"  Warning: Could not read Tags sheet: {e}")

# ──────────────────────────────────────────────────────────────────────
# DATA PROFILE REPORT
# ──────────────────────────────────────────────────────────────────────
report_lines = []
def rpt(line=""):
    report_lines.append(line)
    print(line)

rpt("\n" + "=" * 70)
rpt("PHASE 0 — DATA PROFILE REPORT")
rpt("=" * 70)
rpt(f"\nSource: {EXCEL_PATH}")
rpt(f"Sheet: Consolidated")
rpt(f"Total rows: {len(df_clean)}")
rpt(f"Total columns: {len(df_clean.columns)}")

# Fill rates
rpt("\n--- COLUMN FILL RATES ---")
for col in clean_cols:
    if col in df_clean.columns:
        filled = df_clean[col].notna().sum()
        pct = filled / len(df_clean) * 100
        rpt(f"  {col:45s}: {filled:>6d} / {len(df_clean)} ({pct:5.1f}%)")

# NPS score distribution
rpt("\n--- NPS SCORE DISTRIBUTION ---")
for score in range(11):
    count = (df_clean['nps_score'] == score).sum()
    pct = count / len(df_clean) * 100
    bar = '#' * int(pct)
    rpt(f"  {score:>2d}: {count:>5d} ({pct:5.1f}%) {bar}")

# NPS Group
rpt("\n--- NPS GROUP ---")
for grp in ['Promoter', 'Passive', 'Detractor']:
    count = (df_clean['nps_group'] == grp).sum()
    pct = count / len(df_clean) * 100
    rpt(f"  {grp:12s}: {count:>5d} ({pct:5.1f}%)")

# Computed tenure bucket
rpt("\n--- COMPUTED TENURE BUCKETS ---")
bucket_order = ['Onboarding (0-15d)', 'Early Life (16-60d)', 'Establishing (61-120d)',
                'Steady State (121-270d)', 'Loyal (270d+)', 'Unknown']
for bucket in bucket_order:
    count = (df_clean['tenure_bucket'] == bucket).sum()
    pct = count / len(df_clean) * 100
    rpt(f"  {bucket:25s}: {count:>5d} ({pct:5.1f}%)")

# Cross-tab: NPS Group x Tenure Bucket
rpt("\n--- CROSS-TAB: NPS GROUP x TENURE BUCKET (counts) ---")
ct1 = pd.crosstab(df_clean['tenure_bucket'], df_clean['nps_group'])
ct1 = ct1.reindex(index=bucket_order, columns=['Promoter', 'Passive', 'Detractor'])
rpt(ct1.fillna(0).astype(int).to_string())

# Mean NPS by tenure bucket
rpt("\n--- MEAN NPS SCORE BY TENURE BUCKET ---")
means = df_clean.groupby('tenure_bucket')['nps_score'].agg(['mean', 'median', 'count'])
means = means.reindex(bucket_order)
rpt(means.to_string())

# Churn rate by NPS Group
rpt("\n--- CHURN RATE BY NPS GROUP ---")
churn_by_nps = df_clean.groupby('nps_group').agg(
    total=('is_churned', 'count'),
    churned=('is_churned', 'sum')
)
churn_by_nps['churn_rate'] = (churn_by_nps['churned'] / churn_by_nps['total'] * 100).round(1)
rpt(churn_by_nps.to_string())

# Churn rate by tenure bucket
rpt("\n--- CHURN RATE BY TENURE BUCKET ---")
churn_by_tenure = df_clean.groupby('tenure_bucket').agg(
    total=('is_churned', 'count'),
    churned=('is_churned', 'sum')
)
churn_by_tenure['churn_rate'] = (churn_by_tenure['churned'] / churn_by_tenure['total'] * 100).round(1)
churn_by_tenure = churn_by_tenure.reindex(bucket_order)
rpt(churn_by_tenure.to_string())

# Churn rate by City x NPS Group
rpt("\n--- CHURN RATE BY CITY x NPS GROUP ---")
for city in ['Delhi', 'Mumbai', 'UP']:
    mask = df_clean['city'] == city
    if mask.sum() == 0:
        continue
    rpt(f"\n  {city}:")
    sub = df_clean[mask].groupby('nps_group').agg(
        total=('is_churned', 'count'),
        churned=('is_churned', 'sum')
    )
    sub['churn_rate'] = (sub['churned'] / sub['total'] * 100).round(1)
    rpt(sub.to_string())

# Comment fill rate by NPS group and tenure
rpt("\n--- COMMENT (OE) FILL RATE BY NPS GROUP ---")
comment_by_nps = df_clean.groupby('nps_group')['has_comment'].mean() * 100
rpt(comment_by_nps.round(1).to_string())

rpt("\n--- COMMENT (OE) FILL RATE BY TENURE BUCKET ---")
comment_by_tenure = df_clean.groupby('tenure_bucket')['has_comment'].mean() * 100
comment_by_tenure = comment_by_tenure.reindex(bucket_order)
rpt(comment_by_tenure.round(1).to_string())

# Channel distribution
rpt("\n--- CHANNEL DISTRIBUTION ---")
rpt(df_clean['Channel'].value_counts().to_string())

# Primary Category
rpt("\n--- PRIMARY CATEGORY DISTRIBUTION ---")
rpt(df_clean['Primary Category'].value_counts().to_string())

# NPS Reason - Primary (top 20)
rpt("\n--- NPS REASON - PRIMARY (top 20) ---")
rpt(df_clean['NPS Reason - Primary'].value_counts().head(20).to_string())

# First Time WiFi
rpt("\n--- FIRST TIME WIFI ---")
rpt(df_clean['first_time_wifi'].value_counts().to_string())

# Sprint distribution
rpt("\n--- SPRINT DISTRIBUTION ---")
sprint_counts = df_clean.groupby('Sprint ID').agg(
    count=('nps_score', 'count'),
    mean_nps=('nps_score', 'mean'),
    churn_rate=('is_churned', 'mean')
)
sprint_counts['mean_nps'] = sprint_counts['mean_nps'].round(2)
sprint_counts['churn_rate'] = (sprint_counts['churn_rate'] * 100).round(1)
rpt(sprint_counts.to_string())

# Save report
report_path = os.path.join(REPORT_DIR, 'phase0_data_profile.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n  Report saved to {report_path}")

print("\n" + "=" * 70)
print("PHASE 0 COMPLETE")
print("=" * 70)
