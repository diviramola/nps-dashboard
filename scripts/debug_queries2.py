"""Debug round 2: Fix failing queries."""
import sys, os, requests
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')
API_KEY = os.getenv('METABASE_API_KEY')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = "https://metabase.wiom.in/api/dataset"
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}
DB_ID = 113

def query(sql, label=""):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        r = requests.post(BASE, headers=HEADERS, json=payload, timeout=180)
        data = r.json()
        if 'data' in data and 'rows' in data['data']:
            cols = [c['name'] for c in data['data']['cols']]
            rows = data['data']['rows']
            print(f"  OK — Columns: {cols}, Rows: {len(rows)}")
            for i, row in enumerate(rows[:10]):
                print(f"  [{i}] {row}")
            return cols, rows
        else:
            err = data.get('error', data.get('message', str(data)[:800]))
            print(f"  ERROR: {err}")
            return [], []
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return [], []

# ═══════════════════════════════════════════════════════════
# FIX 1: NETWORK_SCORECARD — try without TRY_TO_DOUBLE
# ═══════════════════════════════════════════════════════════

query("""
SELECT
    MOBILE,
    AVG(PLAN_SPEED) AS avg_plan_speed,
    AVG(LATEST_SPEED::FLOAT) AS avg_latest_speed,
    COUNT(*) AS weeks
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE IN ('6396712571','9897469561','9808716062')
  AND WEEK_START >= '2025-06-01'
GROUP BY MOBILE
""", "FIX1a: Scorecard — LATEST_SPEED as FLOAT cast")

query("""
SELECT
    MOBILE,
    AVG(PLAN_SPEED) AS avg_plan_speed,
    AVG(CAST(LATEST_SPEED AS FLOAT)) AS avg_latest_speed,
    COUNT(*) AS weeks
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE IN ('6396712571','9897469561','9808716062')
  AND WEEK_START >= '2025-06-01'
GROUP BY MOBILE
""", "FIX1b: Scorecard — explicit CAST to FLOAT")

# Check data types
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'NETWORK_SCORECARD' AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "FIX1c: NETWORK_SCORECARD column types")

# Simplest possible query
query("""
SELECT MOBILE, PLAN_SPEED, LATEST_SPEED, SPEED_IN_RANGE
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE = '6396712571'
  AND WEEK_START >= '2025-07-01'
LIMIT 5
""", "FIX1d: Scorecard — single mobile, no aggregation")

# ═══════════════════════════════════════════════════════════
# FIX 2: CUSTOMER_INFLUX_SUMMARY — find correct join path
# ═══════════════════════════════════════════════════════════

# Check if NETWORK_SCORECARD has LNG_NAS that matches influx
query("""
SELECT LNG_NAS, MOBILE, DEVICE_ID
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE MOBILE = '6396712571'
LIMIT 3
""", "FIX2a: NETWORK_SCORECARD LNG_NAS for a known mobile")

# Check if HOURLY_DEVICE_PING_SUMMARY has NAS_ID that matches
query("""
SELECT DEVICE_ID, NAS_ID, PARTNER_ID
FROM PROD_DB.PUBLIC.HOURLY_DEVICE_PING_SUMMARY
WHERE DEVICE_ID = 'SY053968'
  AND DATE_IST >= '2025-12-01'
LIMIT 3
""", "FIX2b: HOURLY_DEVICE_PING NAS_ID for device SY053968")

# Try joining influx via DEVICE_ID path
# First check if there's a device_id or similar in influx
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'CUSTOMER_INFLUX_SUMMARY' AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "FIX2c: CUSTOMER_INFLUX_SUMMARY full column list")

# Check what the LNG_NAS values look like vs what we compute
query("""
SELECT
    cis.LNG_NAS,
    ns.LNG_NAS as scorecard_lng_nas,
    ns.MOBILE,
    ns.DEVICE_ID
FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY cis
JOIN PROD_DB.PUBLIC.NETWORK_SCORECARD ns ON cis.LNG_NAS = ns.LNG_NAS
WHERE ns.WEEK_START >= '2025-12-01'
  AND cis.APPENDED_DATE >= '2025-12-01'
LIMIT 5
""", "FIX2d: Does influx LNG_NAS match scorecard LNG_NAS?")

# ═══════════════════════════════════════════════════════════
# FIX 3: SERVICE TICKETS — smaller batch, simpler query
# ═══════════════════════════════════════════════════════════

# Test devices from our mapping
query("""
SELECT
    DEVICE_ID,
    COUNT(*) AS total_tickets,
    COUNT(CASE WHEN CX_PX = 'Cx' THEN 1 END) AS cx_tickets,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%internet supply down%' THEN 1 END) AS tickets_internet_down,
    COUNT(CASE WHEN LOWER(FIRST_TITLE) LIKE '%slow speed%' OR LOWER(FIRST_TITLE) LIKE '%range issue%' THEN 1 END) AS tickets_slow_speed,
    AVG(TIMES_REOPENED) AS avg_reopens,
    AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) AS avg_resolution_mins,
    COUNT(CASE WHEN RESOLUTION_TAT_BUCKET = 'within TAT' THEN 1 END) AS within_tat,
    AVG(NO_TIMES_CUSTOMER_CALLED) AS avg_customer_calls,
    AVG(RATING_SCORE_BY_CUSTOMER) AS avg_rating
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE DEVICE_ID IN ('SY053968','GX89364','GX86950','SY085556','SY085146')
  AND TICKET_ADDED_TIME >= '2025-06-01'
GROUP BY DEVICE_ID
""", "FIX3: Ticket aggregation with 5 devices (moderate complexity)")

# ═══════════════════════════════════════════════════════════
# FIX 4: DAILY_USAGE_L1 — check NASID format match
# ═══════════════════════════════════════════════════════════

# Get a real NASID from DAILY_USAGE_L1
query("""
SELECT DISTINCT NASID FROM PROD_DB.PUBLIC.DAILY_USAGE_L1
WHERE DATE_IST >= '2025-12-01'
LIMIT 5
""", "FIX4a: Real NASIDs from DAILY_USAGE_L1")

# Check if our mapped nasid (481637) exists in DAILY_USAGE_L1
query("""
SELECT NASID, DATE_IST, SESSIONS_COUNT_DAILY, TOTAL_CONNECTED_DEVICES_DAILY
FROM PROD_DB.PUBLIC.DAILY_USAGE_L1
WHERE NASID = '481637'
  AND DATE_IST >= '2025-12-01'
LIMIT 3
""", "FIX4b: Does mapped nasid 481637 exist in DAILY_USAGE_L1?")

# Try as integer
query("""
SELECT NASID, DATE_IST, SESSIONS_COUNT_DAILY, TOTAL_CONNECTED_DEVICES_DAILY
FROM PROD_DB.PUBLIC.DAILY_USAGE_L1
WHERE NASID = 481637
  AND DATE_IST >= '2025-12-01'
LIMIT 3
""", "FIX4c: Same but NASID as integer")

print("\n\nDEBUG ROUND 2 COMPLETE")
