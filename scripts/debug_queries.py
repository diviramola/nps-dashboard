"""Debug failing Snowflake queries from Phase 3D — test with tiny batches."""
import sys, os, requests, json
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')
API_KEY = os.getenv('METABASE_API_KEY')
if not API_KEY:
    sys.exit("ERROR: METABASE_API_KEY not found")

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
            print(f"  Columns: {cols}")
            print(f"  Rows: {len(rows)}")
            for i, row in enumerate(rows[:10]):
                print(f"  [{i}] {row}")
            return cols, rows
        else:
            err = data.get('error', data.get('message', str(data)[:500]))
            print(f"  ERROR: {err}")
            return [], []
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return [], []

# ── Step 1: Get a few test phone numbers and their mappings ──
query("""
SELECT mobile, nasid, device_id, shard, lco_account_id,
       prod_db.public.idmaker(shard, 4, lco_account_id) AS lng_nas_id
FROM (
    SELECT mobile, nasid, device_id, shard, lco_account_id,
           ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer
    WHERE mobile IN ('9359518021','9891234567','9876543210','8888888888','7777777777')
      AND mobile > '5999999999'
      AND _FIVETRAN_DELETED = false
) t WHERE rn = 1
""", "TEST: Customer mapping sample")

# ── Step 2: Check NETWORK_SCORECARD MOBILE format ──
query("""
SELECT MOBILE, WEEK_START, PLAN_SPEED, LATEST_SPEED, SPEED_IN_RANGE, DATA_USED_GB
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
LIMIT 5
""", "TEST: NETWORK_SCORECARD sample (any 5 rows)")

query("""
SELECT DISTINCT SUBSTR(MOBILE, 1, 5) as mobile_prefix, COUNT(*) as cnt
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE WEEK_START >= '2025-07-01'
GROUP BY SUBSTR(MOBILE, 1, 5)
ORDER BY cnt DESC
LIMIT 10
""", "TEST: MOBILE field format in NETWORK_SCORECARD")

# ── Step 3: Check CUSTOMER_INFLUX_SUMMARY LNG_NAS format ──
query("""
SELECT LNG_NAS, PINGS_RECEIVED_TODAY, EXPECTED_PINGS_SO_FAR, IS_ACTIVE_TODAY
FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY
LIMIT 5
""", "TEST: CUSTOMER_INFLUX_SUMMARY sample (any 5 rows)")

query("""
SELECT
    COUNT(DISTINCT LNG_NAS) as distinct_lng_nas,
    MIN(LENGTH(CAST(LNG_NAS AS VARCHAR))) as min_len,
    MAX(LENGTH(CAST(LNG_NAS AS VARCHAR))) as max_len
FROM PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY
WHERE APPENDED_DATE >= '2025-07-01'
""", "TEST: LNG_NAS format in CUSTOMER_INFLUX_SUMMARY")

# ── Step 4: Check SERVICE_TICKET_MODEL with test device IDs ──
query("""
SELECT DEVICE_ID, TICKET_ID, CX_PX, FIRST_TITLE, TIMES_REOPENED
FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE TICKET_ADDED_TIME >= '2025-07-01'
LIMIT 5
""", "TEST: SERVICE_TICKET_MODEL sample")

# ── Step 5: Try NETWORK_SCORECARD with a KNOWN mobile ──
# First get a real mobile from the scorecard
cols, rows = query("""
SELECT MOBILE FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
WHERE WEEK_START >= '2025-07-01' LIMIT 3
""", "TEST: Get real MOBILE values from scorecard")

if rows:
    test_mobiles = [str(r[0]) for r in rows]
    print(f"\n  Test mobiles from scorecard: {test_mobiles}")

    mobile_list = ",".join([f"'{m}'" for m in test_mobiles])
    query(f"""
    SELECT
        MOBILE,
        AVG(TRY_TO_DOUBLE(PLAN_SPEED)) AS avg_plan_speed,
        AVG(TRY_TO_DOUBLE(LATEST_SPEED)) AS avg_latest_speed,
        COUNT(*) AS weeks
    FROM PROD_DB.PUBLIC.NETWORK_SCORECARD
    WHERE MOBILE IN ({mobile_list})
      AND WEEK_START >= '2025-06-01'
    GROUP BY MOBILE
    """, "TEST: Scorecard aggregation with KNOWN mobiles")

# ── Step 6: Check if NPS phone numbers exist in NETWORK_SCORECARD ──
query("""
SELECT COUNT(DISTINCT ns.MOBILE) as scorecard_phones
FROM PROD_DB.PUBLIC.NETWORK_SCORECARD ns
WHERE ns.MOBILE IN (
    SELECT DISTINCT mobile FROM prod_db.public.t_wg_customer
    WHERE mobile > '5999999999' AND _FIVETRAN_DELETED = false
)
AND ns.WEEK_START >= '2025-07-01'
""", "TEST: How many t_wg_customer mobiles exist in NETWORK_SCORECARD?")

# ── Step 7: Try SERVICE_TICKET_MODEL with real device IDs ──
cols2, rows2 = query("""
SELECT DISTINCT DEVICE_ID FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
WHERE TICKET_ADDED_TIME >= '2025-07-01' LIMIT 5
""", "TEST: Get real DEVICE_IDs from ticket model")

if rows2:
    test_devices = [str(r[0]) for r in rows2]
    device_list = ",".join([f"'{d}'" for d in test_devices])
    query(f"""
    SELECT
        DEVICE_ID,
        COUNT(*) AS total_tickets,
        AVG(TIMES_REOPENED) AS avg_reopens
    FROM PROD_DB.PUBLIC.SERVICE_TICKET_MODEL
    WHERE DEVICE_ID IN ({device_list})
      AND TICKET_ADDED_TIME >= '2025-06-01'
    GROUP BY DEVICE_ID
    """, "TEST: Ticket aggregation with KNOWN device IDs")

# ── Step 8: Check LNG_NAS matching ──
query("""
SELECT
    twc.mobile,
    prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) AS computed_lng_nas,
    cis.LNG_NAS AS influx_lng_nas,
    cis.PINGS_RECEIVED_TODAY
FROM (
    SELECT mobile, shard, lco_account_id,
           ROW_NUMBER() OVER (PARTITION BY mobile ORDER BY added_time DESC) AS rn
    FROM prod_db.public.t_wg_customer
    WHERE mobile > '5999999999' AND _FIVETRAN_DELETED = false
) twc
JOIN PROD_DB.PUBLIC.CUSTOMER_INFLUX_SUMMARY cis
    ON prod_db.public.idmaker(twc.shard, 4, twc.lco_account_id) = cis.LNG_NAS
WHERE twc.rn = 1
  AND cis.APPENDED_DATE >= '2025-12-01'
LIMIT 5
""", "TEST: LNG_NAS join validation (idmaker output matches influx?)")

print("\n\nDEBUG COMPLETE")
