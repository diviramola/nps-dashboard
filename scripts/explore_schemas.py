"""
Schema exploration for Industry Expert features.
Queries Snowflake table schemas via Metabase API.
"""
import sys, io, os, json, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(r'C:\credentials\.env')

METABASE_API_KEY = os.environ.get('METABASE_API_KEY')
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env")
    sys.exit(1)

METABASE_URL = "https://metabase.wiom.in/api/dataset"
DB_ID = 113
HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

def run_query(sql, description="query", timeout=120):
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    try:
        resp = requests.post(METABASE_URL, headers=HEADERS, json=payload, timeout=timeout)
        if resp.status_code in (200, 202):
            data = resp.json()
            if 'data' in data and 'rows' in data['data']:
                cols = [c['name'] for c in data['data']['cols']]
                rows = data['data']['rows']
                return cols, rows
            elif 'error' in data:
                print(f"  [ERROR] {description}: {str(data['error'])[:300]}")
                return None, None
        else:
            print(f"  [ERROR] {description}: HTTP {resp.status_code}")
            return None, None
    except Exception as e:
        print(f"  [ERROR] {description}: {str(e)[:200]}")
        return None, None

# Tables to explore
tables = [
    ("PUBLIC", "SERVICE_TICKET_MODEL"),
    ("PUBLIC", "TATA_IVR_EVENTS"),
    ("PUBLIC", "PARTNER_INFLUX_SUMMARY"),
    ("PUBLIC", "DATA_USAGE_OKR"),
    ("PUBLIC", "PAYMENT_LOGS"),
    ("PUBLIC", "T_ROUTER_USER_MAPPING"),
    ("PUBLIC", "T_WG_CUSTOMER"),
    ("PUBLIC", "TASKVANILLA_AUDIT"),
    ("PUBLIC", "BOOKING_LOGS"),
    ("PUBLIC", "PROFILE_LEAD_MODEL"),
    ("PUBLIC", "T_DEVICE_AUDIT"),
    ("PUBLIC", "T_PLAN_CONFIGURATION"),
]

# Also check outage tables
outage_tables = [
    ("BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC", "IMPACTED_DEVICES"),
    ("BUSINESS_EFFICIENCY_ROUTER_OUTAGE_DETECTION_PUBLIC", "PARTNER_OUTAGE_ALERTS"),
]

output_lines = []

for schema, table in tables + outage_tables:
    db_prefix = "PROD_DB" if schema != "PUBLIC" else "prod_db"
    print(f"\n{'='*60}")
    print(f"TABLE: {db_prefix}.{schema}.{table}")
    print(f"{'='*60}")
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"TABLE: {db_prefix}.{schema}.{table}")
    output_lines.append(f"{'='*60}")

    sql = f"""SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema='{schema}' AND table_name='{table}'
ORDER BY ordinal_position"""

    cols, rows = run_query(sql, f"Schema {table}")
    if rows:
        for row in rows:
            line = f"  {row[0]:45s}  {row[1]}"
            print(line)
            output_lines.append(line)
    else:
        print("  No schema returned")
        output_lines.append("  No schema returned")

    time.sleep(1)

# Save output
output_path = os.path.join(r"C:\Users\nikhi\wiom-nps-analysis\output", "schema_exploration.txt")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print(f"\nSaved to {output_path}")
