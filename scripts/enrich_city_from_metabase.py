"""
Enrich sprint CSVs with city data from Metabase.

Run locally:  python scripts/enrich_city_from_metabase.py

Reads phone numbers from data/sprints/*.csv, queries Metabase for their zone/city,
then updates both:
  - data/sprints/*.csv  (source files)
  - dashboard/public/data/sprints/*.csv  (hashed dashboard files)
"""

import os
import csv
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')
API_KEY = os.getenv('METABASE_API_KEY')
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}
BASE = "https://metabase.wiom.in"
DB_ID = 113

BASE_DIR = Path(r'C:\Users\divir\claude code\wiom-nps-analysis')
SOURCE_DIR = BASE_DIR / 'data' / 'sprints'
DASH_DIR = BASE_DIR / 'dashboard' / 'public' / 'data' / 'sprints'

ZONE_CITY_MAP = {
    'west delhi': 'Delhi', 'east delhi': 'Delhi', 'north delhi': 'Delhi',
    'south delhi': 'Delhi', 'central delhi': 'Delhi', 'delhi': 'Delhi',
    'noida': 'Noida', 'gurgaon': 'Gurgaon', 'gurugram': 'Gurgaon',
    'faridabad': 'Faridabad', 'ghaziabad': 'Ghaziabad',
    'mumbai': 'Mumbai', 'navi mumbai': 'Mumbai', 'vasai': 'Mumbai',
    'meerut': 'Meerut', 'agra': 'Agra', 'lucknow': 'Lucknow',
    'gorakhpur': 'Gorakhpur', 'bareilly': 'Bareilly', 'prayagraj': 'Prayagraj',
    'pune': 'Pune', 'bangalore': 'Bangalore', 'bengaluru': 'Bangalore',
}

def zone_to_city(zone):
    if not zone:
        return ''
    z = zone.lower()
    for key, city in ZONE_CITY_MAP.items():
        if key in z:
            return city
    return zone.split(',')[0].strip()


def query_metabase(sql):
    payload = {"database": DB_ID, "type": "native", "native": {"query": sql}}
    r = requests.post(f"{BASE}/api/dataset", headers=HEADERS, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json().get('data', {})
    cols = [c['name'] for c in data.get('cols', [])]
    rows = data.get('rows', [])
    return [dict(zip(cols, row)) for row in rows]


def get_city_for_phones(phone_list):
    """Query Snowflake via Metabase for city/zone by mobile number."""
    phones_sql = ', '.join(f"'{p}'" for p in phone_list if p)
    # Try CUSTOMER_DAILY_DATA_USAGE first (has zone/area manager data)
    sql = f"""
    SELECT DISTINCT
        MOBILE_NUMBER,
        CITY
    FROM PROD_DB.PUBLIC.CUSTOMER_DAILY_DATA_USAGE
    WHERE MOBILE_NUMBER IN ({phones_sql})
      AND CITY IS NOT NULL
      AND CITY != ''
    LIMIT 5000
    """
    try:
        rows = query_metabase(sql)
        return {str(r.get('MOBILE_NUMBER', '')): zone_to_city(r.get('CITY', ''))
                for r in rows if r.get('CITY')}
    except Exception as e:
        print(f"  CUSTOMER_DAILY_DATA_USAGE failed: {e}")

    # Fallback: MESSAGE_HISTORY might have city
    sql2 = f"""
    SELECT DISTINCT
        MOBILE_NO,
        CITY
    FROM PROD_DB.MESSAGE_ORCHESTRATOR_SERVICE_PUBLIC.MESSAGE_HISTORY
    WHERE MOBILE_NO IN ({phones_sql})
      AND CITY IS NOT NULL
    LIMIT 5000
    """
    try:
        rows = query_metabase(sql2)
        return {str(r.get('MOBILE_NO', '')): zone_to_city(r.get('CITY', ''))
                for r in rows if r.get('CITY')}
    except Exception as e:
        print(f"  MESSAGE_HISTORY failed: {e}")
        return {}


def enrich_sprint_file(source_csv, dash_csv):
    """Enrich one sprint CSV pair with city data."""
    sprint_name = source_csv.stem
    print(f"\nProcessing {sprint_name}...")

    # Load source file
    with open(source_csv, 'r', encoding='utf-8-sig') as f:
        source_rows = list(csv.DictReader(f))

    # Skip if already has city data
    has_city = sum(1 for r in source_rows if r.get('city', '').strip() not in ('', '#N/A'))
    if has_city > len(source_rows) * 0.5:
        print(f"  Already has city data ({has_city}/{len(source_rows)}), skipping")
        return

    phones = [r.get('respondent_id', '').strip() for r in source_rows]
    unique_phones = list(set(p for p in phones if p and len(p) >= 8))
    print(f"  Querying Metabase for {len(unique_phones)} unique phones...")

    # Query in chunks of 500
    city_map = {}
    for i in range(0, len(unique_phones), 500):
        chunk = unique_phones[i:i+500]
        result = get_city_for_phones(chunk)
        city_map.update(result)
        print(f"  Fetched {len(city_map)} cities so far...")

    if not city_map:
        print(f"  No city data returned, skipping")
        return

    filled = 0
    for r in source_rows:
        phone = r.get('respondent_id', '').strip()
        if not r.get('city', '').strip() and phone in city_map:
            r['city'] = city_map[phone]
            filled += 1

    print(f"  Filled {filled}/{len(source_rows)} rows with city data")

    # Write source file
    source_fieldnames = list(source_rows[0].keys())
    with open(source_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=source_fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    # Update dashboard file (uses hashed respondent_ids — match by row order)
    if dash_csv.exists():
        with open(dash_csv, 'r', encoding='utf-8') as f:
            dash_rows = list(csv.DictReader(f))
        dash_fieldnames = list(dash_rows[0].keys())

        for i, (src, dash) in enumerate(zip(source_rows, dash_rows)):
            if not dash.get('city', '').strip() and src.get('city', '').strip():
                dash['city'] = src['city']

        with open(dash_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dash_fieldnames)
            writer.writeheader()
            writer.writerows(dash_rows)
        print(f"  Dashboard CSV updated: {dash_csv.name}")


if __name__ == '__main__':
    sprint_files = sorted(SOURCE_DIR.glob('sprint_*.csv'))
    print(f"Found {len(sprint_files)} sprint files to process")

    for source_csv in sprint_files:
        dash_csv = DASH_DIR / source_csv.name
        try:
            enrich_sprint_file(source_csv, dash_csv)
        except Exception as e:
            print(f"  ERROR for {source_csv.name}: {e}")

    print("\nDone! Run 'npm run build' in dashboard/ to rebuild the app.")
