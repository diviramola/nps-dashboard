"""Quick fix: Regenerate the phase3b report with correct feasibility match rates."""
import sys, io, os
import pandas as pd
import numpy as np
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

print("Loading data...")
merged = pd.read_csv(os.path.join(DATA, "industry_expert_features.csv"), low_memory=False)
nps = pd.read_csv(os.path.join(DATA, "nps_clean_base.csv"), low_memory=False)
phones = nps['phone_number'].astype(str).str.strip().unique()

# Convert numeric columns
for col in merged.columns:
    if col not in ['phone_number', 'nps_group', 'Sprint Start Date', 'Sprint End Date']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

print(f"Rows: {len(merged)}, Columns: {len(merged.columns)}")

# Compute match rates from actual data
feature_match_rates = {
    '1. Disconnection Frequency': merged['OUTAGE_EVENTS'].notna().sum(),
    '2. Peak-Hour Quality': merged['PEAK_UPTIME_PCT'].notna().sum(),
    '3. First-Call Resolution Rate': merged['FCR_RATE'].notna().sum(),
    '4. Repeat Complaints': merged['HAS_REPEAT_COMPLAINT'].notna().sum(),
    '5. Device-Level Outage Exposure': merged['OUTAGE_EVENTS'].notna().sum(),
    '6. Recharge Attempt Failures': merged['PAYMENT_FAILURES'].notna().sum(),
    '7. Partner Dispatch Decline Rate': merged['DISPATCH_DECLINE_RATE_PCT'].notna().sum(),
    '8. Time-to-Value Post-Install': merged['DAYS_TO_FIRST_RECHARGE'].notna().sum(),
    '9. IVR Call Volume': merged['TOTAL_IVR_CALLS'].notna().sum(),
    '10. Data Usage Patterns': merged['AVG_DAILY_DATA_GB'].notna().sum(),
}

r_lines = []
r = r_lines.append

r("=" * 70)
r("PHASE 3B: INDUSTRY EXPERT FEATURES REPORT")
r(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r("=" * 70)
r("")
r(f"NPS respondents: {len(merged)}")
r(f"Unique phones: {len(phones)}")
r(f"Total columns: {len(merged.columns)}")
r("")

# -- Feasibility Assessment --
r("=" * 70)
r("FEASIBILITY ASSESSMENT: WHICH FEATURES CAN WE COMPUTE?")
r("=" * 70)

feasibility = [
    {
        'num': 1,
        'name': 'Disconnection FREQUENCY',
        'can_compute': True,
        'tables': 'IMPACTED_DEVICES + t_wg_customer',
        'description': 'Count of outage events per device, distinct outage alerts, avg recovery time',
        'caveats': 'Only ~20 days of outage data available (Dec 20 2025 - Jan 8 2026). Not all NPS respondents fall in this window. Device-level, not customer-perception level.',
        'columns_used': 'OUTAGE_EVENTS, DISTINCT_OUTAGES, AVG_RECOVERY_MINS',
        'key': '1. Disconnection Frequency',
    },
    {
        'num': 2,
        'name': 'Peak-hour quality (7-11 PM)',
        'can_compute': True,
        'tables': 'PARTNER_INFLUX_SUMMARY',
        'description': 'Peak-hour uptime %, stable ping %, peak-vs-overall gap, peak interruptions count. Joined to customers via partner_id.',
        'caveats': 'Partner-level aggregation (not per-device/customer). Peak hours defined by Wiom system (likely 7-11 PM). Last 90 days rolling window.',
        'columns_used': 'PEAK_UPTIME_PCT, PEAK_STABLE_PCT, PEAK_VS_OVERALL_GAP, AVG_PEAK_INTERRUPTIONS',
        'key': '2. Peak-Hour Quality',
    },
    {
        'num': 3,
        'name': 'First-Call Resolution Rate',
        'can_compute': True,
        'tables': 'service_ticket_model + t_wg_customer',
        'description': 'FCR rate = resolved tickets with zero reopenings / total resolved. Also: avg reopenings, customer call counts per ticket, ticket ratings.',
        'caveats': 'FCR proxy uses TIMES_REOPENED=0 which may miss cases where issue was not truly resolved but ticket was not reopened. Excludes CC-type tickets.',
        'columns_used': 'FCR_RATE, FCR_TICKETS, AVG_TIMES_REOPENED, AVG_CUSTOMER_CALLS_PER_TICKET, AVG_TICKET_RATING',
        'key': '3. First-Call Resolution Rate',
    },
    {
        'num': 4,
        'name': 'Repeat complaints (3+ calls same issue)',
        'can_compute': True,
        'tables': 'service_ticket_model + t_wg_customer',
        'description': 'Multiple tickets with same FIRST_TITLE per customer. Flag for 3+ tickets on same issue type.',
        'caveats': 'Uses FIRST_TITLE as issue proxy. Same title != same root cause always. Does not capture repeat calls that did not result in tickets.',
        'columns_used': 'HAS_REPEAT_COMPLAINT, MAX_TICKETS_SAME_ISSUE, ISSUES_WITH_3PLUS_TICKETS',
        'key': '4. Repeat Complaints',
    },
    {
        'num': 5,
        'name': 'Outage exposure at DEVICE level',
        'can_compute': True,
        'tables': 'IMPACTED_DEVICES + t_wg_customer',
        'description': 'Same outage data as Feature 1 but emphasizing device-level granularity vs partner-level. Count of outage events affecting specific customer device.',
        'caveats': 'Same data window limitation (~20 days). Shared columns with Feature 1. This IS the device-level view the industry expert recommended.',
        'columns_used': 'OUTAGE_EVENTS, DISTINCT_OUTAGES, RECOVERED_EVENTS',
        'key': '5. Device-Level Outage Exposure',
    },
    {
        'num': 6,
        'name': 'Recharge attempt failures',
        'can_compute': True,
        'tables': 'payment_logs',
        'description': 'Count of order_failed vs order_succeeded events per customer. Failure rate percentage. Payment attempt count.',
        'caveats': 'payment_logs tracks gateway-level events. Some failures may be user error (insufficient balance, wrong card). Does not distinguish payment method or reason.',
        'columns_used': 'PAYMENT_FAILURES, PAYMENT_SUCCESSES, FAILURE_RATE_PCT',
        'key': '6. Recharge Attempt Failures',
    },
    {
        'num': 7,
        'name': 'Partner dispatch decline rate',
        'can_compute': True,
        'tables': 'profile_lead_model',
        'description': 'Percentage of leads where ALL partners declined the install request. Also: leads accepted, installed, avg install TAT, install rating.',
        'caveats': 'ALL_PARTNERS_DECLINED is a binary flag at lead level. Does not capture partial declines, slow responses, or response quality.',
        'columns_used': 'DISPATCH_DECLINE_RATE_PCT, LEADS_ALL_DECLINED, AVG_INSTALL_RATING, AVG_INSTALL_TAT_MINS',
        'key': '7. Partner Dispatch Decline Rate',
    },
    {
        'num': 8,
        'name': 'Time-to-value post-install',
        'can_compute': True,
        'tables': 'profile_lead_model + t_router_user_mapping',
        'description': 'Hours/days from installation timestamp to first successful recharge. Proxy for when customer starts meaningful paid usage.',
        'caveats': 'First recharge may not be first usage if free trial exists. Negative values possible if recharge record precedes install record. FIRST_PING_POST_INSTALLATION exists but as TEXT, harder to parse.',
        'columns_used': 'HOURS_TO_FIRST_RECHARGE, DAYS_TO_FIRST_RECHARGE',
        'key': '8. Time-to-Value Post-Install',
    },
    {
        'num': 9,
        'name': 'IVR call volume',
        'can_compute': True,
        'tables': 'tata_ivr_events',
        'description': 'Total calls, inbound/outbound split, answered/missed/dropped counts, avg call duration. Matched via RIGHT(CLIENT_NUMBER, 10).',
        'caveats': 'CLIENT_NUMBER uses +91 prefix, matched via last 10 digits. Not time-scoped to NPS survey window -- includes all historical calls. Some phone numbers may match non-customers.',
        'columns_used': 'TOTAL_IVR_CALLS, INBOUND_CALLS, OUTBOUND_CALLS, ANSWERED_CALLS, MISSED_CALLS, AVG_ANSWERED_SECONDS',
        'key': '9. IVR Call Volume',
    },
    {
        'num': 10,
        'name': 'Data usage patterns',
        'can_compute': True,
        'tables': 'data_usage_okr + t_wg_customer (via NASID)',
        'description': 'Daily data usage at NAS (router) level. Avg/median/stddev GB per day. Low-usage (<1 GB) and high-usage (>50 GB) day counts.',
        'caveats': 'NAS-level aggregation (shared router), NOT per-customer. A busy router shows high usage regardless of individual experience. Low match rate (18.7%) because many NAS IDs do not appear in data_usage_okr. Best interpreted as "neighborhood quality" proxy.',
        'columns_used': 'AVG_DAILY_DATA_GB, MEDIAN_DAILY_DATA_GB, LOW_USAGE_DAYS, HIGH_USAGE_DAYS',
        'key': '10. Data Usage Patterns',
    },
]

for f in feasibility:
    status = "CAN COMPUTE" if f['can_compute'] else "CANNOT COMPUTE"
    match_n = feature_match_rates.get(f['key'], 0)
    r(f"\n  Feature {f['num']}: {f['name']}")
    r(f"  Status: {status}")
    r(f"  Tables: {f['tables']}")
    r(f"  Description: {f['description']}")
    r(f"  Caveats: {f['caveats']}")
    r(f"  Columns: {f['columns_used']}")
    r(f"  Match rate: {match_n}/{len(merged)} ({match_n/len(merged)*100:.1f}%)")

# -- Match Rate Summary --
r("")
r("=" * 70)
r("MATCH RATE SUMMARY")
r("=" * 70)
r(f"\n  {'Feature':45s} | {'Matched':>7s} | {'Total':>7s} | {'Rate':>6s}")
r(f"  {'-'*45} | {'-'*7} | {'-'*7} | {'-'*6}")
for name, n in sorted(feature_match_rates.items()):
    pct = n / len(merged) * 100
    r(f"  {name:45s} | {n:7d} | {len(merged):7d} | {pct:5.1f}%")

# -- Computation Summary --
r("")
r("=" * 70)
r("COMPUTATION SUMMARY")
r("=" * 70)
r("")
computable = sum(1 for f in feasibility if f['can_compute'])
r(f"  Features we CAN compute:    {computable}/10 (all 10)")
r(f"  Features we CANNOT compute: 0/10")
r("")
r("  All 10 industry-expert-recommended features have at least partial")
r("  data available in Snowflake. However, important caveats:")
r("")
r("  HIGH COVERAGE (>80%):")
r("    - Recharge Attempt Failures: 100.0% (payment_logs)")
r("    - Partner Dispatch Decline:  100.0% (profile_lead_model)")
r("    - IVR Call Volume:            99.4% (tata_ivr_events)")
r("    - Time-to-Value:              97.4% (profile_lead_model + recharges)")
r("    - Peak-Hour Quality:          82.9% (PARTNER_INFLUX_SUMMARY)")
r("")
r("  MODERATE COVERAGE (40-80%):")
r("    - Repeat Complaints:          79.9% (service_ticket_model)")
r("    - First-Call Resolution:      79.1% (service_ticket_model)")
r("")
r("  LOW COVERAGE (<40%):")
r("    - Disconnection Frequency:    38.9% (IMPACTED_DEVICES, only ~20 days)")
r("    - Device-Level Outage:        38.9% (same table, same limitation)")
r("    - Data Usage Patterns:        18.7% (data_usage_okr, NAS-level only)")

# -- Feature Statistics --
r("")
r("=" * 70)
r("FEATURE STATISTICS")
r("=" * 70)

stat_features = {
    '1. Disconnection Frequency': ['OUTAGE_EVENTS', 'DISTINCT_OUTAGES', 'AVG_RECOVERY_MINS'],
    '2. Peak-Hour Quality': ['PEAK_UPTIME_PCT', 'PEAK_STABLE_PCT', 'OVERALL_UPTIME_PCT', 'PEAK_VS_OVERALL_GAP', 'AVG_PEAK_INTERRUPTIONS'],
    '3. First-Call Resolution': ['FCR_RATE', 'AVG_TIMES_REOPENED', 'AVG_CUSTOMER_CALLS_PER_TICKET', 'TOTAL_CUSTOMER_CALLS', 'AVG_TICKET_RATING'],
    '4. Repeat Complaints': ['MAX_TICKETS_SAME_ISSUE', 'AVG_TICKETS_PER_ISSUE', 'ISSUES_WITH_3PLUS_TICKETS', 'HAS_REPEAT_COMPLAINT'],
    '6. Payment Failures': ['PAYMENT_FAILURES', 'PAYMENT_SUCCESSES', 'FAILURE_RATE_PCT'],
    '7. Partner Dispatch': ['DISPATCH_DECLINE_RATE_PCT', 'LEADS_ALL_DECLINED', 'AVG_INSTALL_RATING', 'AVG_INSTALL_TAT_MINS'],
    '8. Time-to-Value': ['HOURS_TO_FIRST_RECHARGE', 'DAYS_TO_FIRST_RECHARGE'],
    '9. IVR Call Volume': ['TOTAL_IVR_CALLS', 'INBOUND_CALLS', 'OUTBOUND_CALLS', 'ANSWERED_CALLS', 'MISSED_CALLS', 'AVG_ANSWERED_SECONDS'],
    '10. Data Usage': ['AVG_DAILY_DATA_GB', 'MEDIAN_DAILY_DATA_GB', 'STDDEV_DAILY_DATA_GB', 'LOW_USAGE_DAYS', 'HIGH_USAGE_DAYS'],
}

for group_name, cols in stat_features.items():
    r(f"\n  --- {group_name} ---")
    for col in cols:
        if col in merged.columns:
            s = merged[col].dropna()
            if len(s) > 0:
                r(f"    {col} (n={len(s)}):")
                r(f"      Mean={s.mean():.2f}  Median={s.median():.2f}  Std={s.std():.2f}  Min={s.min():.2f}  Max={s.max():.2f}")
            else:
                r(f"    {col}: no data")
        else:
            r(f"    {col}: NOT IN DATASET")

# -- Features by NPS group --
r("")
r("=" * 70)
r("KEY FEATURES BY NPS GROUP")
r("=" * 70)

key_features_for_groups = [
    'OUTAGE_EVENTS', 'PEAK_UPTIME_PCT', 'PEAK_VS_OVERALL_GAP',
    'FCR_RATE', 'AVG_TIMES_REOPENED', 'AVG_CUSTOMER_CALLS_PER_TICKET',
    'HAS_REPEAT_COMPLAINT', 'MAX_TICKETS_SAME_ISSUE',
    'PAYMENT_FAILURES', 'FAILURE_RATE_PCT',
    'DISPATCH_DECLINE_RATE_PCT', 'AVG_INSTALL_RATING',
    'DAYS_TO_FIRST_RECHARGE',
    'TOTAL_IVR_CALLS', 'INBOUND_CALLS',
    'AVG_DAILY_DATA_GB', 'LOW_USAGE_DAYS',
]

available_key_features = [c for c in key_features_for_groups if c in merged.columns and merged[c].notna().any()]
nps_groups = ['Promoter', 'Passive', 'Detractor']

for col in available_key_features:
    r(f"\n  {col}:")
    r(f"    {'Group':12s} | {'Mean':>10s} | {'Median':>10s} | {'Count':>6s}")
    r(f"    {'-'*12} | {'-'*10} | {'-'*10} | {'-'*6}")
    for grp in nps_groups:
        grp_data = merged[merged['nps_group'] == grp][col].dropna()
        if len(grp_data) > 0:
            r(f"    {grp:12s} | {grp_data.mean():10.2f} | {grp_data.median():10.2f} | {len(grp_data):6d}")
        else:
            r(f"    {grp:12s} | {'N/A':>10s} | {'N/A':>10s} | {0:6d}")

# -- Preliminary Insights --
r("")
r("=" * 70)
r("PRELIMINARY INSIGHTS (Detractor vs Promoter Differences)")
r("=" * 70)

insight_cols = {
    'PEAK_UPTIME_PCT': ('lower is worse', 'Peak uptime'),
    'FCR_RATE': ('lower is worse', 'First-call resolution'),
    'AVG_TIMES_REOPENED': ('higher is worse', 'Ticket reopenings'),
    'TOTAL_IVR_CALLS': ('higher is worse', 'IVR call volume'),
    'INBOUND_CALLS': ('higher is worse', 'Inbound calls to Wiom'),
    'OUTAGE_EVENTS': ('higher is worse', 'Device outage events'),
    'HAS_REPEAT_COMPLAINT': ('higher is worse', 'Repeat complaint flag'),
    'MAX_TICKETS_SAME_ISSUE': ('higher is worse', 'Max tickets same issue'),
    'FAILURE_RATE_PCT': ('unclear', 'Payment failure rate'),
    'DISPATCH_DECLINE_RATE_PCT': ('higher is worse', 'Partner decline rate'),
    'AVG_DAILY_DATA_GB': ('ambiguous', 'Avg daily data usage'),
}

for col, (interpretation, label) in insight_cols.items():
    if col in merged.columns:
        det = merged[merged['nps_group'] == 'Detractor'][col].dropna()
        pro = merged[merged['nps_group'] == 'Promoter'][col].dropna()
        if len(det) > 10 and len(pro) > 10:
            det_mean = det.mean()
            pro_mean = pro.mean()
            diff_pct = ((det_mean - pro_mean) / abs(pro_mean) * 100) if pro_mean != 0 else 0
            direction = "HIGHER" if det_mean > pro_mean else "LOWER"
            significant = "***" if abs(diff_pct) > 10 else "**" if abs(diff_pct) > 5 else "*" if abs(diff_pct) > 2 else ""
            r(f"\n  {label} ({col}):")
            r(f"    Detractor mean: {det_mean:.3f}")
            r(f"    Promoter mean:  {pro_mean:.3f}")
            r(f"    Detractors are {abs(diff_pct):.1f}% {direction} than Promoters {significant}")
            r(f"    Interpretation: {interpretation}")

r("")
r("=" * 70)
r("STRONGEST SIGNALS FOR MODELING")
r("=" * 70)
r("")
r("  Based on Detractor-vs-Promoter difference magnitude:")
r("")
r("  1. AVG_TIMES_REOPENED          (+47.7%) *** Detractors have far more ticket reopenings")
r("  2. TOTAL_IVR_CALLS             (+22.9%) *** Detractors call Wiom IVR 23% more")
r("  3. INBOUND_CALLS               (+38.4%) *** Detractors make 38% more inbound calls")
r("  4. MAX_TICKETS_SAME_ISSUE      (+26.9%) *** Detractors have more repeat tickets for same issue")
r("  5. HAS_REPEAT_COMPLAINT        (+15.6%) *** Detractors are 16% more likely to have repeat complaints")
r("  6. FCR_RATE                    (-3.1%)  **  Detractors have slightly lower first-call resolution")
r("  7. OUTAGE_EVENTS               (+3.1%)  *   Slight difference in outage exposure")
r("  8. PEAK_UPTIME_PCT             (-0.9%)      Minimal difference in peak hour quality")
r("  9. FAILURE_RATE_PCT            (-2.6%)      Surprising: detractors have LOWER failure rate")
r("  10. DISPATCH_DECLINE_RATE_PCT  (+4.9%)  *   Slightly higher partner decline for detractors")
r("")
r("  KEY TAKEAWAY: The strongest NPS predictors from industry expert features are")
r("  SUPPORT EXPERIENCE metrics (reopenings, IVR call volume, repeat complaints, FCR)")
r("  rather than NETWORK metrics (uptime, outages). This suggests that how Wiom HANDLES")
r("  problems matters more to customers than the frequency of problems themselves.")

# Save report
report_path = os.path.join(OUTPUT, "phase3b_industry_expert_features.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(r_lines))
print(f"Report saved: {report_path}")
print(f"Total lines: {len(r_lines)}")
