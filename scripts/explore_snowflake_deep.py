"""
Deep exploration of Snowflake tables for customer-side usage, device concurrency,
ticket detail, and install TAT — to validate Industry Expert hypotheses.
"""
import requests, json, os, sys
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')
API_KEY = os.getenv('METABASE_API_KEY')
if not API_KEY:
    sys.exit("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")

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
        r = requests.post(BASE, headers=HEADERS, json=payload, timeout=120)
        data = r.json()
        if 'data' in data and 'rows' in data['data']:
            cols = [c['name'] for c in data['data']['cols']]
            rows = data['data']['rows']
            print(f"  Columns: {cols}")
            print(f"  Rows returned: {len(rows)}")
            for i, row in enumerate(rows[:30]):
                print(f"  [{i}] {row}")
            if len(rows) > 30:
                print(f"  ... ({len(rows) - 30} more rows)")
            return cols, rows
        else:
            err = data.get('error', data.get('message', str(data)[:500]))
            print(f"  ERROR: {err}")
            return [], []
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return [], []

# ══════════════════════════════════════════════════════════════════
# A. CUSTOMER_DAILY_DATA_USAGE — 24.9M rows, customer-level!
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'CUSTOMER_DAILY_DATA_USAGE'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "A1. COLUMNS: CUSTOMER_DAILY_DATA_USAGE")

query("""
SELECT * FROM PROD_DB.PUBLIC.CUSTOMER_DAILY_DATA_USAGE LIMIT 5
""", "A2. SAMPLE: CUSTOMER_DAILY_DATA_USAGE")

query("""
SELECT
  COUNT(*) as total_rows,
  COUNT(DISTINCT CUSTOMER_ACCOUNT_ID) as distinct_customers,
  MIN(DATE) as min_date,
  MAX(DATE) as max_date
FROM PROD_DB.PUBLIC.CUSTOMER_DAILY_DATA_USAGE
""", "A3. COVERAGE: CUSTOMER_DAILY_DATA_USAGE")

# ══════════════════════════════════════════════════════════════════
# B. T_USAGE_DATA_USER_SESSIONS_AGGREGATED — 175M rows!
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'T_USAGE_DATA_USER_SESSIONS_AGGREGATED'
  AND TABLE_SCHEMA = 'DS_TABLES'
ORDER BY ORDINAL_POSITION
""", "B1. COLUMNS: T_USAGE_DATA_USER_SESSIONS_AGGREGATED")

query("""
SELECT * FROM PROD_DB.DS_TABLES.T_USAGE_DATA_USER_SESSIONS_AGGREGATED LIMIT 5
""", "B2. SAMPLE: T_USAGE_DATA_USER_SESSIONS_AGGREGATED")

# ══════════════════════════════════════════════════════════════════
# C. HOURLY_DEVICE_PING_SUMMARY — 174M rows, device-level ping!
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'HOURLY_DEVICE_PING_SUMMARY'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "C1. COLUMNS: HOURLY_DEVICE_PING_SUMMARY")

query("""
SELECT * FROM PROD_DB.PUBLIC.HOURLY_DEVICE_PING_SUMMARY LIMIT 5
""", "C2. SAMPLE: HOURLY_DEVICE_PING_SUMMARY")

query("""
SELECT
  MIN(HOUR) as min_hour,
  MAX(HOUR) as max_hour,
  COUNT(DISTINCT DEVICE_ID) as distinct_devices
FROM PROD_DB.PUBLIC.HOURLY_DEVICE_PING_SUMMARY
WHERE HOUR >= '2025-07-01'
""", "C3. COVERAGE: HOURLY_DEVICE_PING_SUMMARY (Jul25+)")

# ══════════════════════════════════════════════════════════════════
# D. CUSTOMER_INFLUX_SUMMARY — 17M rows, customer-level network
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'CUSTOMER_INFLUX_SUMMARY'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "D1. COLUMNS: CUSTOMER_INFLUX_SUMMARY")

query("""
SELECT * FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY LIMIT 5
""", "D2. SAMPLE: CUSTOMER_INFLUX_SUMMARY")

query("""
SELECT
  COUNT(*) as total_rows,
  COUNT(DISTINCT CUSTOMER_ACCOUNT_ID) as distinct_customers,
  MIN(DATE) as min_date,
  MAX(DATE) as max_date
FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY
""", "D3. COVERAGE: CUSTOMER_INFLUX_SUMMARY")

# ══════════════════════════════════════════════════════════════════
# E. SERVICE TICKET — ACTUAL COLUMNS DEEP DIVE
# ══════════════════════════════════════════════════════════════════
query("""
SELECT
  CURRENT_TICKET_STATUS, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY CURRENT_TICKET_STATUS
ORDER BY cnt DESC
""", "E1. TICKET STATUSES (actual column)")

query("""
SELECT
  FIRST_TITLE, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY FIRST_TITLE
ORDER BY cnt DESC
LIMIT 20
""", "E2. TICKET FIRST_TITLE (category proxy)")

query("""
SELECT
  LAST_TITLE, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY LAST_TITLE
ORDER BY cnt DESC
LIMIT 20
""", "E3. TICKET LAST_TITLE (sub-category)")

query("""
SELECT
  RESOLUTION_TAT_BUCKET, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY RESOLUTION_TAT_BUCKET
ORDER BY cnt DESC
""", "E4. RESOLUTION TAT BUCKETS")

query("""
SELECT
  COUNT(*) as total,
  AVG(TIMES_REOPENED) as avg_reopens,
  AVG(TIMES_REOPENED_POSTCLOSURE) as avg_reopens_postclosure,
  AVG(RESOLUTION_PERIOD_MINS_CALENDARHRS) as avg_resolution_mins_cal,
  AVG(RESOLUTION_PERIOD_MINS_WORKINGHRS) as avg_resolution_mins_work,
  AVG(ALLOCATIONTAT_MINS_CALENDARHRS) as avg_allocation_tat_cal,
  AVG(NO_TIMES_CUSTOMER_CALLED) as avg_customer_calls,
  AVG(NO_TIMES_PARTNER_CALLED) as avg_partner_calls,
  AVG(RATING_SCORE_BY_CUSTOMER) as avg_customer_rating
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE TICKET_ADDED_TIME >= '2025-07-01'
""", "E5. TICKET METRICS (Jul25+ period)")

query("""
SELECT
  CX_PX, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY CX_PX
ORDER BY cnt DESC
""", "E6. CX_PX FIELD (customer vs partner ticket?)")

query("""
SELECT
  CLOSED_BY, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY CLOSED_BY
ORDER BY cnt DESC
LIMIT 10
""", "E7. CLOSED_BY FIELD")

# ══════════════════════════════════════════════════════════════════
# F. INSTALL TAT — EO_INSTALL and related
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'EO_INSTALL'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "F1. COLUMNS: EO_INSTALL")

query("""
SELECT * FROM PROD_DB.PUBLIC.EO_INSTALL LIMIT 3
""", "F2. SAMPLE: EO_INSTALL")

# ══════════════════════════════════════════════════════════════════
# G. NETWORK_SCORECARD — 6.2M rows
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'NETWORK_SCORECARD'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "G1. COLUMNS: NETWORK_SCORECARD")

query("""
SELECT * FROM PROD_DB.PUBLIC.NETWORK_SCORECARD LIMIT 5
""", "G2. SAMPLE: NETWORK_SCORECARD")

# ══════════════════════════════════════════════════════════════════
# H. DAILY_USAGE_L1 — 48.7M rows
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'DAILY_USAGE_L1'
  AND TABLE_SCHEMA = 'PUBLIC'
ORDER BY ORDINAL_POSITION
""", "H1. COLUMNS: DAILY_USAGE_L1")

query("""
SELECT * FROM PROD_DB.PUBLIC.DAILY_USAGE_L1 LIMIT 5
""", "H2. SAMPLE: DAILY_USAGE_L1")

# ══════════════════════════════════════════════════════════════════
# I. CUSTOMER_METRICS — 50M rows
# ══════════════════════════════════════════════════════════════════
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'CUSTOMER_METRICS'
  AND TABLE_SCHEMA = 'CUSTOMER_DB_CUSTOMER_PROFILE_SERVICE_PUBLIC'
ORDER BY ORDINAL_POSITION
""", "I1. COLUMNS: CUSTOMER_METRICS")

query("""
SELECT * FROM PROD_DB.CUSTOMER_DB_CUSTOMER_PROFILE_SERVICE_PUBLIC.CUSTOMER_METRICS LIMIT 5
""", "I2. SAMPLE: CUSTOMER_METRICS")

print("\n\n" + "="*70)
print("  DEEP EXPLORATION COMPLETE")
print("="*70)
