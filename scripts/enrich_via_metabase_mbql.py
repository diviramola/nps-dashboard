"""
Metabase Enrichment using MBQL (structured queries) instead of native SQL.
Use this when the API key doesn't have native query permissions.

Usage:
    python scripts/enrich_via_metabase_mbql.py
    python scripts/enrich_via_metabase_mbql.py --discover   # find table/field IDs first
    python scripts/enrich_via_metabase_mbql.py --sprint sprint_rsp5.csv
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

CREDENTIALS_PATH = Path(r"C:\credentials\.env")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sprints"
ENRICHMENT_DIR = PROJECT_ROOT / "data" / "enrichment"

if CREDENTIALS_PATH.exists():
    load_dotenv(str(CREDENTIALS_PATH))

METABASE_API_KEY = os.getenv("METABASE_API_KEY")
METABASE_BASE_URL = os.getenv("METABASE_URL", "https://metabase.wiom.in")
METABASE_DATABASE_ID = int(os.getenv("METABASE_DATABASE_ID", "1"))


def headers():
    if not METABASE_API_KEY:
        print("ERROR: METABASE_API_KEY not set")
        sys.exit(1)
    return {"X-API-Key": METABASE_API_KEY, "Content-Type": "application/json"}


def api_get(path):
    url = f"{METABASE_BASE_URL}{path}"
    r = httpx.get(url, headers=headers(), timeout=30.0)
    r.raise_for_status()
    return r.json()


def api_post(path, body):
    url = f"{METABASE_BASE_URL}{path}"
    r = httpx.post(url, headers=headers(), json=body, timeout=60.0)
    r.raise_for_status()
    return r.json()


def discover_schema():
    """Find the ACTIVE_BASE table and its field IDs."""
    print("Discovering tables in database {}...".format(METABASE_DATABASE_ID))

    # List all tables
    tables = api_get(f"/api/table")

    # Filter to our database
    db_tables = [t for t in tables if t.get("db_id") == METABASE_DATABASE_ID]
    print("Found {} tables in database {}".format(len(db_tables), METABASE_DATABASE_ID))

    # Find ACTIVE_BASE
    target = None
    for t in db_tables:
        name = t.get("name", "").upper()
        schema = (t.get("schema") or "").upper()
        if "ACTIVE_BASE" in name:
            target = t
            print("  MATCH: {} (schema: {}, id: {})".format(t["name"], t.get("schema"), t["id"]))

    if not target:
        print("\nNo ACTIVE_BASE table found. All tables:")
        for t in db_tables[:30]:
            print("  {} (schema: {}, id: {})".format(t["name"], t.get("schema"), t["id"]))
        return None

    # Get field metadata
    print("\nGetting fields for table {} (id={})...".format(target["name"], target["id"]))
    meta = api_get(f"/api/table/{target['id']}/query_metadata")

    fields = {}
    for f in meta.get("fields", []):
        fname = f["name"].upper()
        fields[fname] = {"id": f["id"], "name": f["name"], "type": f.get("base_type")}
        print("  Field: {} (id={}, type={})".format(f["name"], f["id"], f.get("base_type")))

    # Save discovery results
    discovery = {
        "table_id": target["id"],
        "table_name": target["name"],
        "schema": target.get("schema"),
        "fields": fields,
        "discovered_at": datetime.now().isoformat(),
    }

    disc_path = ENRICHMENT_DIR / "metabase_schema_discovery.json"
    disc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(disc_path, "w") as f:
        json.dump(discovery, f, indent=2)
    print("\nSaved discovery to {}".format(disc_path))

    return discovery


def load_discovery():
    disc_path = ENRICHMENT_DIR / "metabase_schema_discovery.json"
    if disc_path.exists():
        with open(disc_path) as f:
            return json.load(f)
    return None


def clean_phone(phone_str):
    s = str(phone_str).strip().replace(".0", "")
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def fetch_batch_mbql(phones, discovery):
    """Fetch enrichment data using MBQL structured query."""
    table_id = discovery["table_id"]
    fields = discovery["fields"]

    phone_field_id = None
    city_field_id = None
    device_field_id = None
    install_field_id = None

    for fname, finfo in fields.items():
        if "PHONE" in fname:
            phone_field_id = finfo["id"]
        if fname == "CITY":
            city_field_id = finfo["id"]
        if "DEVICE" in fname:
            device_field_id = finfo["id"]
        if "INSTALL" in fname and "DATE" in fname:
            install_field_id = finfo["id"]

    if not phone_field_id:
        print("  ERROR: Could not find phone field")
        return {}

    # Build MBQL query - filter by phone numbers
    # MBQL "=" with multiple values acts as IN
    filter_clause = ["=", ["field", phone_field_id, None]]
    for p in phones:
        filter_clause.append(p)

    # Select fields we want
    selected_fields = [["field", phone_field_id, None]]
    if city_field_id:
        selected_fields.append(["field", city_field_id, None])
    if device_field_id:
        selected_fields.append(["field", device_field_id, None])
    if install_field_id:
        selected_fields.append(["field", install_field_id, None])

    body = {
        "database": METABASE_DATABASE_ID,
        "type": "query",
        "query": {
            "source-table": table_id,
            "filter": filter_clause,
            "fields": selected_fields,
        },
    }

    try:
        data = api_post("/api/dataset", body)
        cols = [c["name"] for c in data["data"]["cols"]]
        results = {}
        for row in data["data"]["rows"]:
            row_dict = dict(zip(cols, row))
            phone_val = None
            for k, v in row_dict.items():
                if "PHONE" in k.upper():
                    phone_val = clean_phone(str(v))
                    break
            if phone_val:
                city = ""
                device = ""
                install_date = ""
                for k, v in row_dict.items():
                    ku = k.upper()
                    if ku == "CITY" and v:
                        city = str(v).strip()
                    if "DEVICE" in ku and v:
                        device = str(v).strip()
                    if "INSTALL" in ku and v:
                        install_date = str(v).strip()

                tenure = 0
                if install_date:
                    try:
                        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                                    "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y"]:
                            try:
                                dt = datetime.strptime(install_date[:10], fmt[:10] if "T" not in fmt else fmt)
                                tenure = (datetime.now() - dt).days
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                results[phone_val] = {
                    "city": city,
                    "device": device,
                    "install_date": install_date,
                    "tenure_days": max(0, tenure),
                }
        return results
    except httpx.HTTPStatusError as e:
        print("  WARNING: MBQL query failed ({}): {}".format(e.response.status_code, e.response.text[:200]))
        return {}
    except Exception as e:
        print("  WARNING: Query failed: {}".format(e))
        return {}


def enrich_sprint(filepath, discovery, dry_run=False):
    """Enrich a single sprint CSV."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Find phones needing enrichment
    needs = set()
    for row in rows:
        phone = clean_phone(row.get("respondent_id", ""))
        if not phone or len(phone) != 10:
            continue
        city = (row.get("city") or "").strip()
        tenure = (row.get("tenure_days") or "0").strip()
        if city in ("", "#REF!", "unknown", "nan") or tenure == "0":
            needs.add(phone)

    if not needs:
        print("  No enrichment needed")
        return

    print("  {} phones need enrichment".format(len(needs)))

    # Batch fetch (100 at a time for MBQL to avoid huge filter clauses)
    all_enrichment = {}
    phone_list = list(needs)
    batch_size = 100
    for i in range(0, len(phone_list), batch_size):
        batch = phone_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(phone_list) + batch_size - 1) // batch_size
        print("  Fetching batch {} of {} ({} phones)...".format(batch_num, total_batches, len(batch)))
        result = fetch_batch_mbql(batch, discovery)
        all_enrichment.update(result)

    print("  Got enrichment for {} / {} phones".format(len(all_enrichment), len(needs)))

    if dry_run:
        print("  DRY RUN: would enrich rows in {}".format(filepath.name))
        return

    # Apply enrichment
    enriched_count = 0
    for row in rows:
        phone = clean_phone(row.get("respondent_id", ""))
        if phone in all_enrichment:
            edata = all_enrichment[phone]
            city = (row.get("city") or "").strip()
            if city in ("", "#REF!", "unknown", "nan") and edata.get("city"):
                row["city"] = edata["city"]
                enriched_count += 1
            tenure = (row.get("tenure_days") or "0").strip()
            if tenure == "0" and edata.get("tenure_days", 0) > 0:
                row["tenure_days"] = str(edata["tenure_days"])

    # Write back
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("  Enriched {} rows in {}".format(enriched_count, filepath.name))

    # Save sidecar
    sidecar = ENRICHMENT_DIR / "{}_enrichment.json".format(filepath.stem)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w") as f:
        json.dump({
            "enriched_at": datetime.now().isoformat(),
            "source_file": filepath.name,
            "phones_enriched": len(all_enrichment),
            "data": all_enrichment,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true", help="Discover table/field IDs")
    parser.add_argument("--sprint", help="Process specific sprint CSV")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.discover:
        discover_schema()
        return

    # Load or run discovery
    discovery = load_discovery()
    if not discovery:
        print("No schema discovery found. Running discovery first...")
        discovery = discover_schema()
        if not discovery:
            print("ERROR: Could not discover schema. Run with --discover to debug.")
            sys.exit(1)

    print("Using table: {} (id={})".format(discovery["table_name"], discovery["table_id"]))

    # Find CSVs
    if args.sprint:
        files = [DATA_DIR / args.sprint]
    else:
        files = sorted(DATA_DIR.glob("*.csv"))

    for filepath in files:
        if not filepath.exists():
            print("File not found: {}".format(filepath))
            continue
        print("\nProcessing: {}".format(filepath.name))
        enrich_sprint(filepath, discovery, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
