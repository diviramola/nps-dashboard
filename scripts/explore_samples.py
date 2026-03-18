"""
Sample data exploration for Industry Expert features.
"""
import sys, io, os, json, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found")
    sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

def run_query(sql, desc="query", timeout=120):
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if resp.status_code in (200, 202):
            data = resp.json()
            if 'data' in data and 'rows' in data['data']:
                cols = [c['name'] for c in data['data']['cols']]
                rows = data['data']['rows']
                print(f"\n[{desc}] Cols: {cols}")
                for i, row in enumerate(rows[:10]):
                    print(f"  {row}")
                print(f"  Total rows: {len(rows)}")
                return cols, rows
            elif 'error' in data:
                print(f"\n[{desc}] ERROR: {str(data['error'])[:300]}")
        else:
            print(f"\n[{desc}] HTTP {resp.status_code}")
    except Exception as e:
        print(f"\n[{desc}] ERROR: {str(e)[:200]}")
    return None, None

# 1. IVR events - check structure, how CLIENT_NUMBER looks, direction values
print("=" * 60)
print("1. TATA_IVR_EVENTS sample")
print("=" * 60)
run_query("""
SELECT CLIENT_NUMBER, DIRECTION, STATUS, CALL_DURATION, ANSWERED_SECONDS,
       HANGUP_CAUSE, DTMF_INPUT, DID_NUMBER
FROM prod_db.public.tata_ivr_events
LIMIT 10
""", "IVR sample")
time.sleep(2)

# 1b. IVR distinct directions and statuses
print("\n" + "=" * 60)
print("1b. IVR distinct DIRECTION values")
print("=" * 60)
run_query("""
SELECT DIRECTION, COUNT(*) as cnt
FROM prod_db.public.tata_ivr_events
GROUP BY DIRECTION
""", "IVR directions")
time.sleep(2)

# 1c. IVR distinct statuses
print("\n" + "=" * 60)
print("1c. IVR distinct STATUS values")
print("=" * 60)
run_query("""
SELECT STATUS, COUNT(*) as cnt
FROM prod_db.public.tata_ivr_events
GROUP BY STATUS
ORDER BY cnt DESC
LIMIT 15
""", "IVR statuses")
time.sleep(2)

# 2. IMPACTED_DEVICES - check device_id and linking
print("\n" + "=" * 60)
print("2. IMPACTED_DEVICES sample")
print("=" * 60)
run_query("""
SELECT DEVICE_ID, ALERT_ID, PARTNER_ID, WINDOW_DATETIME,
       DEVICE_RECOVERY_STATE, RECOVERY_DATETIME
FROM PROD_DB.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC.IMPACTED_DEVICES
WHERE _FIVETRAN_DELETED = false
LIMIT 10
""", "Impacted devices")
time.sleep(2)

# 2b. Count of impacted devices
print("\n" + "=" * 60)
print("2b. IMPACTED_DEVICES counts")
print("=" * 60)
run_query("""
SELECT COUNT(*) as total_records,
       COUNT(DISTINCT DEVICE_ID) as unique_devices,
       COUNT(DISTINCT ALERT_ID) as unique_alerts,
       MIN(WINDOW_DATETIME) as earliest,
       MAX(WINDOW_DATETIME) as latest
FROM PROD_DB.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC.IMPACTED_DEVICES
WHERE _FIVETRAN_DELETED = false
""", "Impacted devices counts")
time.sleep(2)

# 3. PARTNER_OUTAGE_ALERTS sample
print("\n" + "=" * 60)
print("3. PARTNER_OUTAGE_ALERTS sample")
print("=" * 60)
run_query("""
SELECT PARTNER_ID, PORTFOLIO_TYPE, CONSECUTIVE_MISS_COUNT,
       PING_DROP_PERCENTAGE, ALERT_STATUS, CREATED_AT, UPDATED_AT
FROM PROD_DB.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC.PARTNER_OUTAGE_ALERTS
WHERE _FIVETRAN_DELETED = false
LIMIT 10
""", "Outage alerts")
time.sleep(2)

# 4. PAYMENT_LOGS - check event_name values
print("\n" + "=" * 60)
print("4. PAYMENT_LOGS distinct EVENT_NAME values")
print("=" * 60)
run_query("""
SELECT EVENT_NAME, COUNT(*) as cnt
FROM prod_db.public.payment_logs
GROUP BY EVENT_NAME
ORDER BY cnt DESC
LIMIT 20
""", "Payment logs events")
time.sleep(2)

# 4b. Payment logs sample with DATA column
print("\n" + "=" * 60)
print("4b. PAYMENT_LOGS sample (with DATA)")
print("=" * 60)
run_query("""
SELECT MOBILE, EVENT_NAME, SOURCE_FLAG, ADDED_TIME, LEFT(DATA, 200) as data_preview
FROM prod_db.public.payment_logs
WHERE EVENT_NAME LIKE '%fail%' OR EVENT_NAME LIKE '%FAIL%'
LIMIT 5
""", "Payment logs failures")
time.sleep(2)

# 5. DATA_USAGE_OKR sample
print("\n" + "=" * 60)
print("5. DATA_USAGE_OKR sample")
print("=" * 60)
run_query("""
SELECT NASID, DT, TOTAL_DATA_USED
FROM prod_db.public.data_usage_okr
LIMIT 10
""", "Data usage sample")
time.sleep(2)

# 5b. Data usage stats
print("\n" + "=" * 60)
print("5b. DATA_USAGE_OKR stats")
print("=" * 60)
run_query("""
SELECT COUNT(*) as total_records,
       COUNT(DISTINCT NASID) as unique_nas,
       MIN(DT) as earliest,
       MAX(DT) as latest,
       AVG(TOTAL_DATA_USED) as avg_usage
FROM prod_db.public.data_usage_okr
""", "Data usage stats")
time.sleep(2)

# 6. Service ticket - check TIMES_REOPENED, NO_TIMES_CUSTOMER_CALLED etc
print("\n" + "=" * 60)
print("6. SERVICE_TICKET_MODEL - reopening and call counts")
print("=" * 60)
run_query("""
SELECT
    COUNT(*) as total_tickets,
    AVG(TIMES_REOPENED) as avg_reopened,
    MAX(TIMES_REOPENED) as max_reopened,
    AVG(NO_TIMES_CUSTOMER_CALLED) as avg_cust_calls,
    AVG(NO_TIMES_PARTNER_CALLED) as avg_partner_calls,
    COUNT(CASE WHEN FIRST_TITLE = LAST_TITLE THEN 1 END) as same_title_count
FROM prod_db.public.service_ticket_model
WHERE CX_PX != 'CC'
""", "Ticket reopening stats")
time.sleep(2)

# 7. PARTNER_INFLUX_SUMMARY - check peak hour columns
print("\n" + "=" * 60)
print("7. PARTNER_INFLUX_SUMMARY peak hour check")
print("=" * 60)
run_query("""
SELECT
    COUNT(*) as total_records,
    AVG(PEAK_HOUR_PINGS_RECEIVED * 1.0 / NULLIF(PEAK_HOUR_EXPECTED_PINGS, 0)) as avg_peak_uptime,
    AVG(TOTAL_PINGS_RECEIVED * 1.0 / NULLIF(TOTAL_EXPECTED_PINGS, 0)) as avg_total_uptime,
    AVG(CUSTOMERS_WITH_PEAK_INTERRUPTIONS) as avg_peak_interruptions
FROM prod_db.public.PARTNER_INFLUX_SUMMARY
WHERE DATEADD(day, -1, appended_date) >= DATEADD(day, -90, CURRENT_DATE())
""", "Peak hour uptime")
time.sleep(2)

# 8. Profile lead model - partner dispatch
print("\n" + "=" * 60)
print("8. PROFILE_LEAD_MODEL - partner dispatch/decline")
print("=" * 60)
run_query("""
SELECT
    COUNT(*) as total_leads,
    COUNT(CASE WHEN ALL_PARTNERS_DECLINED = 1 THEN 1 END) as all_declined,
    COUNT(LEAD_FIRST_ACCEPTANCE_TIME) as first_accepted,
    AVG(LEAD_INSTALLATION_TAT) as avg_install_tat,
    COUNT(LEAD_INSTALLATION_TIME) as installed
FROM prod_db.public.profile_lead_model
WHERE MOBILE > '5999999999'
""", "Lead dispatch stats")
time.sleep(2)

# 9. Check if IVR has enough data to match phones
print("\n" + "=" * 60)
print("9. IVR CLIENT_NUMBER format check")
print("=" * 60)
run_query("""
SELECT CLIENT_NUMBER, LENGTH(CLIENT_NUMBER) as len, COUNT(*) as cnt
FROM prod_db.public.tata_ivr_events
GROUP BY CLIENT_NUMBER, LENGTH(CLIENT_NUMBER)
ORDER BY cnt DESC
LIMIT 15
""", "IVR phone format")
time.sleep(2)

# 10. Check t_wg_customer -> device_id -> impacted_devices linkability
print("\n" + "=" * 60)
print("10. Device-level outage linkability test")
print("=" * 60)
run_query("""
SELECT COUNT(DISTINCT twc.device_id) as customer_devices,
       COUNT(DISTINCT imp.device_id) as impacted_devices_matched
FROM prod_db.public.t_wg_customer twc
LEFT JOIN PROD_DB.BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC.IMPACTED_DEVICES imp
    ON twc.device_id = imp.device_id
    AND imp._FIVETRAN_DELETED = false
WHERE twc._FIVETRAN_DELETED = false
  AND twc.mobile > '5999999999'
""", "Device outage linkability", timeout=180)

print("\nDone!")
