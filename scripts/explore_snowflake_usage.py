"""
Explore Snowflake for customer-side usage, device, and service ticket data
to validate Industry Expert hypotheses and find missing features.
"""
import requests
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')
API_KEY = os.getenv('METABASE_API_KEY')
if not API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

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
        r = requests.post(BASE, headers=HEADERS, json=payload, timeout=60)
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
# 1. FIND ALL USAGE/DEVICE/SESSION/TRAFFIC TABLES
# ══════════════════════════════════════════════════════════════════
query("""
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM PROD_DB.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'PROD_DB'
AND (
  LOWER(TABLE_NAME) LIKE '%usage%'
  OR LOWER(TABLE_NAME) LIKE '%data%'
  OR LOWER(TABLE_NAME) LIKE '%device%'
  OR LOWER(TABLE_NAME) LIKE '%session%'
  OR LOWER(TABLE_NAME) LIKE '%bandwidth%'
  OR LOWER(TABLE_NAME) LIKE '%traffic%'
  OR LOWER(TABLE_NAME) LIKE '%nas%'
  OR LOWER(TABLE_NAME) LIKE '%radius%'
  OR LOWER(TABLE_NAME) LIKE '%concurrent%'
  OR LOWER(TABLE_NAME) LIKE '%connected%'
  OR LOWER(TABLE_NAME) LIKE '%throughput%'
  OR LOWER(TABLE_NAME) LIKE '%speed%'
  OR LOWER(TABLE_NAME) LIKE '%okr%'
  OR LOWER(TABLE_NAME) LIKE '%consumption%'
  OR LOWER(TABLE_NAME) LIKE '%byte%'
)
ORDER BY ROW_COUNT DESC NULLS LAST
""", "1. USAGE/DEVICE/SESSION TABLES IN PROD_DB")

# ══════════════════════════════════════════════════════════════════
# 2. FIND ALL SERVICE TICKET RELATED TABLES
# ══════════════════════════════════════════════════════════════════
query("""
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM PROD_DB.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'PROD_DB'
AND (
  LOWER(TABLE_NAME) LIKE '%ticket%'
  OR LOWER(TABLE_NAME) LIKE '%complaint%'
  OR LOWER(TABLE_NAME) LIKE '%service%'
  OR LOWER(TABLE_NAME) LIKE '%support%'
  OR LOWER(TABLE_NAME) LIKE '%ivr%'
  OR LOWER(TABLE_NAME) LIKE '%call%'
  OR LOWER(TABLE_NAME) LIKE '%escalat%'
  OR LOWER(TABLE_NAME) LIKE '%resolution%'
  OR LOWER(TABLE_NAME) LIKE '%sla%'
  OR LOWER(TABLE_NAME) LIKE '%dispatch%'
  OR LOWER(TABLE_NAME) LIKE '%install%'
  OR LOWER(TABLE_NAME) LIKE '%technician%'
  OR LOWER(TABLE_NAME) LIKE '%reopen%'
)
ORDER BY ROW_COUNT DESC NULLS LAST
""", "2. SERVICE TICKET / SUPPORT TABLES IN PROD_DB")

# ══════════════════════════════════════════════════════════════════
# 3. FIND NETWORK/OUTAGE/UPTIME TABLES
# ══════════════════════════════════════════════════════════════════
query("""
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM PROD_DB.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'PROD_DB'
AND (
  LOWER(TABLE_NAME) LIKE '%outage%'
  OR LOWER(TABLE_NAME) LIKE '%uptime%'
  OR LOWER(TABLE_NAME) LIKE '%downtime%'
  OR LOWER(TABLE_NAME) LIKE '%influx%'
  OR LOWER(TABLE_NAME) LIKE '%impact%'
  OR LOWER(TABLE_NAME) LIKE '%network%'
  OR LOWER(TABLE_NAME) LIKE '%router%'
  OR LOWER(TABLE_NAME) LIKE '%olt%'
  OR LOWER(TABLE_NAME) LIKE '%onu%'
  OR LOWER(TABLE_NAME) LIKE '%optical%'
  OR LOWER(TABLE_NAME) LIKE '%ping%'
  OR LOWER(TABLE_NAME) LIKE '%signal%'
)
ORDER BY ROW_COUNT DESC NULLS LAST
""", "3. NETWORK/OUTAGE/UPTIME TABLES IN PROD_DB")

# ══════════════════════════════════════════════════════════════════
# 4. GET COLUMNS FOR KEY TABLES WE ALREADY KNOW ABOUT
# ══════════════════════════════════════════════════════════════════

# data_usage_okr - the NAS-level usage table
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'DATA_USAGE_OKR'
ORDER BY ORDINAL_POSITION
""", "4a. COLUMNS: DATA_USAGE_OKR")

# service_ticket_model - the main ticket table
query("""
SELECT COLUMN_NAME, DATA_TYPE
FROM PROD_DB.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_CATALOG = 'PROD_DB' AND TABLE_NAME = 'SERVICE_TICKET_MODEL'
ORDER BY ORDINAL_POSITION
""", "4b. COLUMNS: SERVICE_TICKET_MODEL")

# Sample from service_ticket_model to see what fields we have
query("""
SELECT * FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL LIMIT 3
""", "4c. SAMPLE: SERVICE_TICKET_MODEL")

# ══════════════════════════════════════════════════════════════════
# 5. CHECK SERVICE TICKET DETAIL — CATEGORIES, SUBCATEGORIES, RESOLUTION
# ══════════════════════════════════════════════════════════════════
query("""
SELECT CATEGORY, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY CATEGORY
ORDER BY cnt DESC
LIMIT 20
""", "5a. TICKET CATEGORIES")

query("""
SELECT SUB_CATEGORY, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY SUB_CATEGORY
ORDER BY cnt DESC
LIMIT 30
""", "5b. TICKET SUB-CATEGORIES")

query("""
SELECT STATUS, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
GROUP BY STATUS
ORDER BY cnt DESC
""", "5c. TICKET STATUSES")

# ══════════════════════════════════════════════════════════════════
# 6. CHECK FOR RESOLUTION TIME, REOPEN, SLA FIELDS IN TICKETS
# ══════════════════════════════════════════════════════════════════
query("""
SELECT
  COUNT(*) as total_tickets,
  COUNT(CASE WHEN RESOLUTION_TIME IS NOT NULL THEN 1 END) as has_resolution_time,
  COUNT(CASE WHEN CLOSED_AT IS NOT NULL THEN 1 END) as has_closed_at,
  COUNT(CASE WHEN TIMES_REOPENED IS NOT NULL AND TIMES_REOPENED > 0 THEN 1 END) as has_reopens,
  AVG(TIMES_REOPENED) as avg_reopens,
  AVG(RESOLUTION_TIME) as avg_resolution_time
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
""", "6a. TICKET RESOLUTION & REOPEN STATS")

# ══════════════════════════════════════════════════════════════════
# 7. DATA USAGE OKR — SAMPLE AND COVERAGE
# ══════════════════════════════════════════════════════════════════
query("""
SELECT * FROM PROD_DB.PUBLIC.DATA_USAGE_OKR LIMIT 5
""", "7a. SAMPLE: DATA_USAGE_OKR")

query("""
SELECT
  COUNT(DISTINCT NAS_IP) as distinct_nas,
  COUNT(*) as total_rows,
  MIN(DATE) as min_date,
  MAX(DATE) as max_date,
  AVG(TOTAL_USAGE_GB) as avg_usage_gb
FROM PROD_DB.PUBLIC.DATA_USAGE_OKR
""", "7b. DATA_USAGE_OKR COVERAGE")

# ══════════════════════════════════════════════════════════════════
# 8. LOOK FOR DEVICE/USER CONCURRENCY DATA
# ══════════════════════════════════════════════════════════════════
query("""
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM PROD_DB.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'PROD_DB'
AND (
  LOWER(TABLE_NAME) LIKE '%user%'
  OR LOWER(TABLE_NAME) LIKE '%subscriber%'
  OR LOWER(TABLE_NAME) LIKE '%customer%'
  OR LOWER(TABLE_NAME) LIKE '%mapping%'
)
ORDER BY ROW_COUNT DESC NULLS LAST
LIMIT 20
""", "8. USER/SUBSCRIBER/MAPPING TABLES")

# ══════════════════════════════════════════════════════════════════
# 9. INSTALL TAT — CHECK DISPATCH/INSTALL DATA
# ══════════════════════════════════════════════════════════════════
query("""
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM PROD_DB.INFORMATION_SCHEMA.TABLES
WHERE TABLE_CATALOG = 'PROD_DB'
AND (
  LOWER(TABLE_NAME) LIKE '%dispatch%'
  OR LOWER(TABLE_NAME) LIKE '%install%'
  OR LOWER(TABLE_NAME) LIKE '%activation%'
  OR LOWER(TABLE_NAME) LIKE '%onboard%'
)
ORDER BY ROW_COUNT DESC NULLS LAST
""", "9. DISPATCH/INSTALL/ACTIVATION TABLES")

print("\n\n" + "="*70)
print("  EXPLORATION COMPLETE")
print("="*70)
