"""Add definitions tab to existing Google Sheet."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import gspread

gc = gspread.oauth(
    credentials_filename=r'C:\Users\nikhi\.config\gspread\credentials.json',
    authorized_user_filename=r'C:\Users\nikhi\.config\gspread\authorized_user.json'
)

# Open existing sheet
sh = gc.open_by_key('1OyzL40FVed1wW7Bdu76wep2PZF-nRDllr4kPgXvStEs')

# Add definitions tab
ws_def = sh.add_worksheet(title='Definitions', rows=60, cols=4)

definitions = [
    ['Column / Metric', 'Definition', 'Source', 'How to Interpret'],
    # Output columns
    ['--- OUTPUT COLUMNS ---', '', '', ''],
    ['Phone', 'Customer mobile number', 't_router_user_mapping', ''],
    ['Risk Score', 'Churn probability (0.0 to 1.0) from GB+RF ensemble model', 'Model output', 'Higher = more likely to churn. Model AUC 0.95 on 15K training customers.'],
    ['Risk Tier', 'High (>=0.6), Medium (0.4-0.6), Low (<0.4)', 'Derived from Risk Score', 'High = urgent intervention needed'],
    ['At Risk', 'YES if Risk Score >= 0.5, else NO', 'Derived from Risk Score', 'Binary flag for filtering'],
    ['Driver 1/2/3', 'Top 3 operational factors driving churn risk for this customer', 'Feature importance x deviation from healthy baseline', 'Format: Metric: customer_value (healthy: median_of_non_churned_customers)'],
    ['Plan Type', 'M+ (monthly, >=28 days) or PayG (pay-as-you-go, <28 days)', 'Plan duration from otp_expiry_time - otp_issued_time', 'M+ is the core product'],
    ['Plan Expiry', 'Date when current plan expires', 'otp_expiry_time (IST)', 'All shown customers have plan expiring within 7 days'],
    ['City', 'Delhi / Mumbai / Bharat', 'HIERARCHY_BASE.MIS_CITY via partner mapping', ''],
    ['Tenure (Days)', 'Days since first-ever Wiom recharge', 't_router_user_mapping', 'Longer = more established'],
    ['Days Since Last Recharge', 'Days between last recharge and today', 't_router_user_mapping', 'Higher = closer to plan expiry'],
    ['Recharge Count', 'Total lifetime recharges (triple-deduplicated)', 't_router_user_mapping', 'Higher = more loyal historically'],
    ['Last Recharge Date', 'Date of most recent recharge', 't_router_user_mapping', ''],
    ['', '', '', ''],
    # Driver metrics - clean ones that appear as top 3
    ['--- DRIVER METRICS (appear in Driver 1/2/3 columns) ---', '', '', ''],
    ['Network Uptime (%)', 'Overall partner uptime: avg(pings_received / expected_pings). Note: peak-hour data unavailable so this is overall uptime shown as negative gap.', 'PARTNER_INFLUX_SUMMARY (90 days)', 'Higher = better. Healthy ~71%. Most important feature (45% of model).'],
    ['Ticket Resolution Rate', 'Fraction of service tickets resolved: resolved / total', 'SERVICE_TICKET_MODEL', 'Higher = better. Healthy ~0.87. Second most important (35%).'],
    ['Avg Resolution Time (hrs)', 'Average hours to resolve service tickets', 'SERVICE_TICKET_MODEL', 'Lower = better. Healthy ~45 hrs.'],
    ['Optical Signal Quality (%)', 'Pct of time RX power is in acceptable range', 'NETWORK_SCORECARD.RXPOWER_IN_RANGE', 'Higher = better. Healthy ~74%. Measures fiber/ONT health.'],
    ['Avg Call Duration (sec)', 'Average duration of answered IVR calls. NOT hold/wait time.', 'tata_ivr_events (answered only)', '0 = never called. Healthy ~151 sec (~2.5 min per call).'],
    ['Uptime Variability', 'Std deviation of partner uptime across days', 'PARTNER_INFLUX_SUMMARY', 'Lower = more stable network. High = intermittent outages.'],
    ['SLA Compliance (%)', 'Fraction of tickets resolved within SLA target', 'SERVICE_TICKET_MODEL', 'Higher = better. 1.0 = all within SLA.'],
    ['Missed Call Ratio', 'Fraction of IVR calls missed: missed / total', 'tata_ivr_events', 'Lower = better. Ratio not volume. 0 = no missed calls.'],
    ['Speed Gap vs Plan (%)', '(plan_speed - actual_speed) / plan_speed', 'NETWORK_SCORECARD', 'Lower = better. 0 = full plan speed. 0.5 = half speed.'],
    ['Avg Actual Speed (Mbps)', 'Average measured download speed', 'NETWORK_SCORECARD.LATEST_SPEED', 'Higher = better. Compare against Plan Speed.'],
    ['Speed In Range (%)', 'Pct of time speed is within acceptable range', 'NETWORK_SCORECARD.SPEED_IN_RANGE', 'Higher = better.'],
    ['Plan Speed (Mbps)', 'Speed promised by the plan', 'NETWORK_SCORECARD.PLAN_SPEED', 'Reference value for speed gap calculation.'],
    ['Weekly Data Usage (GB)', 'Average weekly data consumption', 'NETWORK_SCORECARD.DATA_USED_GB', 'Very low may signal disengagement.'],
    ['Overall Uptime (%)', 'Partner-level uptime percentage', 'PARTNER_INFLUX_SUMMARY', 'Higher = better.'],
    ['Ticket Severity Score', 'Composite: log1p(tickets)*0.3 + reopened*0.3 + (1-SLA)*0.4', 'Derived', 'Higher = worse. 0 = no tickets ever. Non-zero for healthy is normal (long-tenure customers raise some tickets).'],
    ['Autopay Ratio', 'Placeholder - always 0 in current data', 'N/A', 'Not yet available.'],
    ['', '', '', ''],
    # Excluded metrics
    ['--- METRICS IN MODEL BUT EXCLUDED FROM TOP-3 DRIVERS ---', '', '', ''],
    ['(Reason: absolute volume metrics that double-count the same underlying signal. Ratios like Missed Call Ratio and Resolution Rate capture quality without volume bias.)', '', '', ''],
    ['Answered Calls/Month', 'Absolute count of answered calls per month', 'tata_ivr_events', 'Excluded: r=0.72-0.96 with other call volume metrics'],
    ['Inbound Answered/Month', 'Answered inbound calls per month', 'tata_ivr_events', 'Excluded: subset of total calls'],
    ['Inbound Unanswered/Month', 'Unanswered inbound calls per month', 'tata_ivr_events', 'Excluded: use Missed Call Ratio instead'],
    ['Support Calls/Month', 'Total IVR calls per month', 'tata_ivr_events', 'Excluded: raw volume not quality'],
    ['Missed Calls/Month', 'Absolute missed call count', 'tata_ivr_events', 'Excluded: use Missed Call Ratio instead'],
    ['Inbound Calls/Month', 'Total inbound calls per month', 'tata_ivr_events', 'Excluded: raw volume'],
    ['Dropped Calls', 'Total dropped calls', 'tata_ivr_events', 'Excluded: raw volume'],
    ['Tickets/Month', 'Total tickets per month', 'SERVICE_TICKET_MODEL', 'Excluded: Resolution Rate and SLA capture quality'],
    ['Customer Tickets/Month', 'Customer-raised tickets per month', 'SERVICE_TICKET_MODEL (Cx)', 'Excluded: subset of total tickets'],
    ['Partner Tickets/Month', 'Partner-raised tickets per month', 'SERVICE_TICKET_MODEL (Px)', 'Excluded: r=0.97 with total tickets'],
    ['Distinct Issues/Month', 'Unique issue types per month', 'SERVICE_TICKET_MODEL', 'Excluded: correlated with ticket volume'],
    ['Reopened Tickets/Month', 'Tickets reopened at least once', 'SERVICE_TICKET_MODEL', 'Excluded: correlated with volume'],
    ['Install Attempts/Month', 'Install attempts per month of tenure', 'taskvanilla_audit', 'Excluded: volume metric'],
]

ws_def.update(range_name='A1', values=definitions)
ws_def.format('A1:D1', {'textFormat': {'bold': True}})

# Bold section headers
for i, row in enumerate(definitions):
    if row[0].startswith('---'):
        ws_def.format(f'A{i+1}:D{i+1}', {'textFormat': {'bold': True, 'italic': True}})

print('Definitions tab added successfully')
print(f'Sheet URL: {sh.url}')
