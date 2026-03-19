"""Add Model Summary and Definitions tabs to the churn risk Google Sheet."""
import gspread
import time
import sys

GSPREAD_CREDS = r'C:\Users\nikhi\.config\gspread\credentials.json'
GSPREAD_TOKEN = r'C:\Users\nikhi\.config\gspread\authorized_user.json'
SHEET_ID = '1ewGLfhlizAGVURMjYSE52EsWLGZFysf2L_aAP5Is2Nk'

gc = gspread.oauth(credentials_filename=GSPREAD_CREDS, authorized_user_filename=GSPREAD_TOKEN)
sh = gc.open_by_key(SHEET_ID)

# ── MODEL SUMMARY TAB ──────────────────────────────────────────────────────────
print("Adding Model Summary tab...")
try:
    ws_sum = sh.worksheet('Model Summary')
    ws_sum.clear()
except gspread.exceptions.WorksheetNotFound:
    ws_sum = sh.add_worksheet(title='Model Summary', rows=80, cols=4)

time.sleep(1)

summary_data = [
    ['WIOM CHURN RISK PREDICTOR'],
    [''],
    ['WHAT IT DOES'],
    ['Predicts which customers are likely to churn (stop recharging) based on their operational experience.'],
    ['Scores customers due for recharge in the next 7 days and identifies the top 3 risk drivers per customer.'],
    [''],
    ['WHY IT EXISTS'],
    ['NPS surveys tell us who is happy/unhappy but not reliably WHY. Scores mix many underlying issues,'],
    ['responses are biased and delayed. This model uses actual operational data to identify churn risk drivers.'],
    [''],
    # ── HOW IT WORKS ──
    ['HOW IT WORKS'],
    ['Step', 'Description'],
    ['1. Train', 'Retrain a Gradient Boosting + Random Forest ensemble on 14,985 historical customers (36.7% churned)'],
    ['2. Target', 'Pull customers whose current plan expires within the next 7 days (both M+ and PayG plans)'],
    ['3. Features', 'Compute 16 operational metrics from Snowflake (uptime, network, service quality, behavior)'],
    ['4. Score', 'Ensemble probability = average of GB and RF churn probabilities'],
    ['5. Drivers', 'Per customer: feature importance x deviation from healthy median in risky direction (top 3 shown)'],
    ['6. Export', 'Upload to Google Sheets + local CSV backup'],
    [''],
    # ── MODEL DETAILS ──
    ['MODEL DETAILS'],
    ['Parameter', 'Value'],
    ['Training data', '44,566 customers across 13 cohort sprints (population-level, not just NPS respondents)'],
    ['Churn rate (training)', '50.2%'],
    ['Features', '16 (cleaned: removed volume-biased, broken, and duplicate features)'],
    ['Ensemble', 'Gradient Boosting (AUC 0.9733) + Random Forest (AUC 0.9672), averaged'],
    ['Risk threshold', '0.5 (customers above this are flagged as At Risk)'],
    ['Risk tiers', 'High (>=0.6), Medium (0.4-0.6), Low (<0.4)'],
    ['Missing value handling', 'Median imputation (trained on population medians)'],
    [''],
    # ── THIS RUN ──
    ['THIS RUN (2026-03-19)'],
    ['Metric', 'Value'],
    ['Customers scored', '31,050 (due for recharge within 7 days)'],
    ['At risk (score >= 0.5)', '5,669 (18.3%)'],
    ['High risk', '4,273'],
    ['Medium risk', '2,827'],
    ['Low risk', '23,950'],
    ['Plan type split', 'M+: 30,502 | PayG: 548'],
    [''],
    # ── FEATURE IMPORTANCE ──
    ['FEATURE IMPORTANCE (16 features, trained on 44,566 customers)'],
    ['Feature', 'Importance', 'Category'],
    ['Ticket Resolution Rate', '27.7%', 'Service Quality'],
    ['Overall Uptime (%)', '27.0%', 'Network Uptime'],
    ['Avg Resolution Time (hrs)', '15.3%', 'Service Quality'],
    ['SLA Compliance (%)', '14.9%', 'Service Quality'],
    ['Avg Call Duration (sec)', '4.8%', 'Service Quality'],
    ['Optical Power (dBm)', '3.3%', 'Network Quality'],
    ['Weekly Data Usage (GB)', '2.0%', 'Network Quality'],
    ['Optical Signal Quality (%)', '1.6%', 'Network Quality'],
    ['Uptime Variability', '1.4%', 'Network Uptime'],
    ['Missed Call Ratio', '1.1%', 'Service Quality'],
    ['Speed In Range (%)', '0.5%', 'Network Quality'],
    ['Plan Speed (Mbps)', '0.2%', 'Network Quality'],
    ['Speed Gap vs Plan (%)', '0.1%', 'Network Quality'],
    ['Avg Actual Speed (Mbps)', '0.1%', 'Network Quality'],
    ['Optical Power In Range (%)', '0.0%', 'Network Quality'],
    ['Autopay Ratio', '0.0%', 'Behavior'],
    [''],
    # ── FEATURES REMOVED ──
    ['FEATURES REMOVED AND WHY'],
    ['Category', 'Features Removed', 'Reason'],
    ['Broken', 'PEAK_UPTIME_PCT, PEAK_VS_OVERALL_GAP',
     'PARTNER_INFLUX_SUMMARY is a daily table with no hourly data. HOUR() on a DATE column always returns 0, so peak-hour uptime was always NaN (all 14,985 training rows).'],
    ['Empty columns', 'AVG_TICKET_RATING, AVG_CUSTOMER_CALLS',
     'RATING_SCORE_BY_CUSTOMER has 0 non-null values out of 905K tickets in Snowflake. NO_TIMES_CUSTOMER_CALLED has 1,778 non-null, all = 0.'],
    ['Volume-biased calls', 'ivr_calls, missed_calls, answered_calls, inbound_calls, inbound_answered, inbound_unanswered, DROPPED_CALLS',
     'Raw counts biased by tenure. Highly correlated with each other (r > 0.7). missed_call_ratio already captures call quality without volume bias.'],
    ['Volume-biased tickets', 'tickets, cx_tickets, px_tickets, distinct_issues, reopened, max_reopened, has_tickets, install_attempts, ticket_severity',
     'Raw counts biased by tenure. Highly correlated (r > 0.7). resolution_rate and SLA_COMPLIANCE_PCT already capture ticket quality without volume bias.'],
    ['Duplicates', 'AVG_UPTIME_PCT, avg_resolution_hours_w, tk_sla_compliance',
     'AVG_UPTIME_PCT is identical to OVERALL_UPTIME_PCT (r = 1.0, same SQL expression). avg_resolution_hours_w is a winsorized copy of avg_resolution_hours. tk_sla_compliance is identical to SLA_COMPLIANCE_PCT (r = 1.0).'],
    [''],
    # ── DATA SOURCES ──
    ['DATA SOURCES'],
    ['Snowflake Table', 'Features Derived'],
    ['PARTNER_INFLUX_SUMMARY', 'Uptime metrics (OVERALL, STDDEV) from daily partner-level ping data (90-day window)'],
    ['NETWORK_SCORECARD', 'Speed, optical power, data usage from weekly customer-level network quality data'],
    ['SERVICE_TICKET_MODEL', 'Resolution time, SLA compliance, resolution rate from service ticket data'],
    ['tata_ivr_events', 'Call duration, missed call ratio from IVR call records'],
    ['t_router_user_mapping', 'Customer list, plan expiry (otp_expiry_time), tenure, recharge history'],
    ['t_wg_customer', 'Device-to-partner mapping, city'],
    [''],
    # ── INTERPRETATION GUIDE ──
    ['HOW TO INTERPRET DRIVER EXPLANATIONS'],
    ['Each driver shows: Metric Name: customer_value (healthy: median_of_non_churned_customers)'],
    [''],
    ['Example: "Ticket Resolution Rate: 0.40 (healthy: 0.85)"'],
    ['This means the customer has only 40% of their tickets resolved, while non-churned customers typically have 85%.'],
    ['The gap between their value and the healthy median, weighted by how important this feature is to the model,'],
    ['determines the driver ranking.'],
    [''],
    ['A driver only appears if the customer deviates from healthy in the risky direction.'],
    ['If fewer than 3 drivers are shown, the customer is close to healthy on most metrics.'],
]

ws_sum.update(range_name='A1', values=summary_data)
time.sleep(2)

# Format section headers
ws_sum.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
for row_text in ['WHAT IT DOES', 'WHY IT EXISTS', 'HOW IT WORKS', 'MODEL DETAILS',
                 'THIS RUN', 'FEATURE IMPORTANCE', 'FEATURES REMOVED', 'DATA SOURCES',
                 'HOW TO INTERPRET']:
    # Find the row
    for i, r in enumerate(summary_data):
        if r[0].startswith(row_text):
            ws_sum.format(f'A{i+1}', {'textFormat': {'bold': True, 'fontSize': 12}})
            break

# Bold sub-headers (Step/Parameter/Metric/Feature/Category/Table rows)
# These are auto-found by the section header loop above; also bold the table column headers
sub_header_rows = [12, 21, 31, 40, 58, 66]
for row in sub_header_rows:
    if row <= len(summary_data):
        ws_sum.format(f'A{row}:C{row}', {'textFormat': {'bold': True}})

print("  Model Summary tab added")

# ── DEFINITIONS TAB ─────────────────────────────────────────────────────────────
print("Adding Definitions tab...")
time.sleep(2)
try:
    ws_def = sh.worksheet('Definitions')
    ws_def.clear()
except gspread.exceptions.WorksheetNotFound:
    ws_def = sh.add_worksheet(title='Definitions', rows=40, cols=3)

time.sleep(1)

definitions_data = [
    ['Column', 'Definition', 'Source'],
    ['Phone', 'Customer mobile number', 't_router_user_mapping'],
    ['Risk Score', 'Churn probability (0 to 1) from ensemble of Gradient Boosting + Random Forest', 'Model output'],
    ['Risk Tier', 'High (>=0.6), Medium (0.4-0.6), Low (<0.4)', 'Derived from Risk Score'],
    ['At Risk', 'YES if Risk Score >= 0.5, NO otherwise', 'Derived from Risk Score'],
    ['Driver 1/2/3', 'Top 3 churn risk drivers. Format: "Metric: value (healthy: median)"', 'Model explanation'],
    ['Plan Type', 'M+ (plan >= 28 days) or PayG (plan < 28 days)', 't_router_user_mapping'],
    ['Plan Expiry', 'Date current plan expires (from otp_expiry_time)', 't_router_user_mapping'],
    ['Recharge Due In (Days)', 'Days until plan expires (Plan Expiry minus today). 0 = expires today, negative = already expired', 'Derived from Plan Expiry'],
    ['City', 'Customer city', 't_wg_customer'],
    ['Tenure (Days)', 'Days since first recharge', 't_router_user_mapping'],
    ['Days Since Last Recharge', 'Days between last recharge and today', 't_router_user_mapping'],
    ['Recharge Count', 'Total number of valid recharges', 't_router_user_mapping'],
    ['Last Recharge Date', 'Date of most recent recharge', 't_router_user_mapping'],
    [''],
    ['DRIVER METRICS'],
    ['Metric', 'What It Measures', 'Healthy Direction'],
    ['Ticket Resolution Rate', 'Fraction of tickets resolved (resolved / total)', 'Higher is better'],
    ['Overall Uptime (%)', '90-day average uptime from daily partner-level ping data', 'Higher is better'],
    ['Avg Resolution Time (hrs)', 'Average hours to resolve a service ticket', 'Lower is better'],
    ['Uptime Variability', 'Standard deviation of daily uptime (consistency)', 'Lower is better'],
    ['Optical Signal Quality (%)', 'Fraction of weeks with RX power in healthy range', 'Higher is better'],
    ['Avg Call Duration (sec)', 'Average duration of answered IVR calls', 'Context-dependent'],
    ['SLA Compliance (%)', 'Fraction of tickets resolved within SLA target', 'Higher is better'],
    ['Optical Power (dBm)', 'Average optical receive power. -10 to -28 dBm is healthy range', 'Higher (less negative) is better'],
    ['Missed Call Ratio', 'Fraction of IVR calls that went unanswered', 'Lower is better'],
    ['Weekly Data Usage (GB)', 'Average weekly data consumption', 'Higher = more engagement'],
    ['Speed Gap vs Plan (%)', 'Shortfall between actual speed and plan speed', 'Lower is better'],
    ['Avg Actual Speed (Mbps)', 'Average measured download speed', 'Higher is better'],
    ['Speed In Range (%)', 'Fraction of weeks with speed within acceptable range', 'Higher is better'],
    ['Autopay Ratio', 'Fraction of recharges done via autopay', 'Higher is better'],
]

ws_def.update(range_name='A1', values=definitions_data)
time.sleep(1)
ws_def.format('A1:C1', {'textFormat': {'bold': True}})
ws_def.format('A15', {'textFormat': {'bold': True, 'fontSize': 12}})
ws_def.format('A16:C16', {'textFormat': {'bold': True}})

print("  Definitions tab added")
print(f"\nDone! Sheet: https://docs.google.com/spreadsheets/d/{SHEET_ID}")
