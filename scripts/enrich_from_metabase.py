"""
Metabase Enrichment Script for NPS Sprint CSVs.

Connects to metabase.wiom.in to fetch city, device, and tenure data
for NPS respondents using their phone number.

Usage:
    python scripts/enrich_from_metabase.py
    python scripts/enrich_from_metabase.py --sprint sprint_14.csv
    python scripts/enrich_from_metabase.py --dry-run

Credentials: Reads METABASE_API_KEY from C:\\credentials\\.env
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, date

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)


# ── Config ───────────────────────────────────────────────────────
CREDENTIALS_PATH = Path(r"C:\credentials\.env")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sprints"
ENRICHMENT_DIR = PROJECT_ROOT / "data" / "enrichment"

# Load credentials
if CREDENTIALS_PATH.exists():
    load_dotenv(str(CREDENTIALS_PATH))
else:
    print(f"WARNING: Credentials file not found at {CREDENTIALS_PATH}")
    print("Create it with: METABASE_API_KEY=mb_your-key-here")

METABASE_API_KEY = os.getenv("METABASE_API_KEY")
METABASE_BASE_URL = os.getenv("METABASE_URL", "https://metabase.wiom.in")
METABASE_DATABASE_ID = int(os.getenv("METABASE_DATABASE_ID", "1"))


def metabase_headers():
    if not METABASE_API_KEY:
        print("ERROR: METABASE_API_KEY not found in environment or credentials file.")
        print(f"Add it to {CREDENTIALS_PATH}: METABASE_API_KEY=mb_your-key-here")
        sys.exit(1)
    return {"X-API-Key": METABASE_API_KEY, "Content-Type": "application/json"}


def metabase_query(sql: str) -> list[dict]:
    """Run a native SQL query against Metabase/Snowflake and return rows as dicts."""
    url = f"{METABASE_BASE_URL}/api/dataset"
    body = {
        "database": METABASE_DATABASE_ID,
        "type": "native",
        "native": {"query": sql},
    }
    response = httpx.post(url, headers=metabase_headers(), json=body, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    cols = [c["name"] for c in data["data"]["cols"]]
    return [dict(zip(cols, row)) for row in data["data"]["rows"]]


def clean_phone(phone_str: str) -> str:
    """Clean phone number: strip .0 suffix, non-digits, ensure 10 digits."""
    s = str(phone_str).strip().replace(".0", "")
    digits = "".join(c for c in s if c.isdigit())
    # Take last 10 digits if longer (strip country code)
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def load_sprint_csv(filepath: Path) -> list[dict]:
    """Load a sprint CSV and return rows as dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_missing_enrichment(rows: list[dict]) -> set[str]:
    """Find phone numbers that need enrichment (missing city, device, or tenure)."""
    needs_enrichment = set()
    for row in rows:
        phone = clean_phone(row.get("respondent_id", ""))
        if not phone or len(phone) != 10:
            continue

        city = (row.get("city") or "").strip()
        tenure = (row.get("tenure_days") or "0").strip()
        # Need enrichment if city is empty/invalid or tenure is 0
        if city in ("", "#REF!", "unknown", "nan") or tenure == "0":
            needs_enrichment.add(phone)

    return needs_enrichment


def fetch_enrichment_batch(phones: list[str]) -> dict[str, dict]:
    """
    Fetch city, device, and install_date for a batch of phone numbers from Metabase.
    Returns: { phone: { city, device, install_date, tenure_days } }
    """
    if not phones:
        return {}

    # Build SQL with phone list
    phone_list = ", ".join(f"'{p}'" for p in phones)

    # Query user/subscriber table for city and install date
    # Adjust table/column names based on actual Metabase schema
    sql = f"""
    SELECT
        PHONE_NUMBER,
        CITY,
        DEVICE_TYPE,
        INSTALL_DATE,
        DATEDIFF('day', INSTALL_DATE, CURRENT_DATE()) AS TENURE_DAYS
    FROM POSTGRES_RDS_PUBLIC.ACTIVE_BASE
    WHERE PHONE_NUMBER IN ({phone_list})
    """

    try:
        results = metabase_query(sql)
        enrichment = {}
        for row in results:
            phone = clean_phone(str(row.get("PHONE_NUMBER", "")))
            if phone:
                tenure = row.get("TENURE_DAYS")
                enrichment[phone] = {
                    "city": (row.get("CITY") or "").strip(),
                    "device": (row.get("DEVICE_TYPE") or "").strip(),
                    "install_date": str(row.get("INSTALL_DATE") or ""),
                    "tenure_days": int(tenure) if tenure is not None else 0,
                }
        return enrichment
    except Exception as e:
        print(f"  WARNING: Metabase query failed: {e}")
        return {}


def enrich_and_save(filepath: Path, enrichment_data: dict[str, dict], dry_run: bool = False):
    """Enrich a sprint CSV with Metabase data and save."""
    rows = load_sprint_csv(filepath)
    enriched_count = 0

    for row in rows:
        phone = clean_phone(row.get("respondent_id", ""))
        if phone in enrichment_data:
            edata = enrichment_data[phone]

            # Only fill in missing fields
            city = (row.get("city") or "").strip()
            if city in ("", "#REF!", "unknown", "nan") and edata.get("city"):
                row["city"] = edata["city"]
                enriched_count += 1

            tenure = (row.get("tenure_days") or "0").strip()
            if tenure == "0" and edata.get("tenure_days", 0) > 0:
                row["tenure_days"] = str(edata["tenure_days"])

            # Device is a new column — always add if available
            if edata.get("device"):
                row["device"] = edata["device"]

    if dry_run:
        print(f"  DRY RUN: Would enrich {enriched_count} rows in {filepath.name}")
        return

    # Save enriched CSV (overwrite)
    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure 'device' column is included
        if "device" not in fieldnames:
            fieldnames.append("device")

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Enriched {enriched_count} rows in {filepath.name}")

    # Save enrichment sidecar JSON for the dashboard
    sidecar = ENRICHMENT_DIR / f"{filepath.stem}_enrichment.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({
            "enriched_at": datetime.now().isoformat(),
            "source_file": filepath.name,
            "phones_enriched": len(enrichment_data),
            "data": enrichment_data,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Enrich NPS sprint CSVs from Metabase")
    parser.add_argument("--sprint", help="Process a specific sprint CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--batch-size", type=int, default=500, help="Phone numbers per Metabase query")
    args = parser.parse_args()

    # Find CSV files
    if args.sprint:
        files = [DATA_DIR / args.sprint]
        if not files[0].exists():
            print(f"ERROR: File not found: {files[0]}")
            sys.exit(1)
    else:
        files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        print("Copy sprint CSVs to data/sprints/ first.")
        sys.exit(1)

    print(f"Enrichment source: {METABASE_BASE_URL}")
    print(f"Processing {len(files)} file(s)\n")

    for filepath in files:
        print(f"Processing: {filepath.name}")
        rows = load_sprint_csv(filepath)
        print(f"  Rows: {len(rows)}")

        needs = find_missing_enrichment(rows)
        if not needs:
            print("  No enrichment needed — all fields populated")
            continue

        print(f"  Phones needing enrichment: {len(needs)}")

        # Batch fetch from Metabase
        all_enrichment = {}
        phone_list = list(needs)
        for i in range(0, len(phone_list), args.batch_size):
            batch = phone_list[i:i + args.batch_size]
            print(f"  Fetching batch {i // args.batch_size + 1} ({len(batch)} phones)...")
            result = fetch_enrichment_batch(batch)
            all_enrichment.update(result)

        print(f"  Got enrichment for {len(all_enrichment)} / {len(needs)} phones")

        enrich_and_save(filepath, all_enrichment, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
