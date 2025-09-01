#!/usr/bin/env python3
"""
Fetch QSO CSV (including launchangle) from Datasette, then update rm_toucans.db
with both f2m and launchangle for each id.
"""

import argparse
import requests
import csv
import sqlite3
import urllib.parse
import sys

SQL_TEMPLATE = """
WITH qso AS (
  SELECT *,
         midpoint_lat(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lat,
         midpoint_lng(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lng,
         haversine(tx_lat, tx_lng, rx_lat, rx_lng) as length
  FROM [rm_toucans_slice].rm_rnb_history_pres
  WHERE timestamp BETWEEN '{start}' AND '{end}'
    AND dB > 100
),
f2 AS (
  SELECT *
  FROM [glotec_slice].glotec
  WHERE timestamp BETWEEN '{start}' AND '{end}'
),
fmint AS (
  SELECT
    qso.id as id,
    qso.timestamp as qst,
    strftime('%Y%m%d', qso.timestamp) as date,
    strftime('%H%M', qso.timestamp) as time,
    qso.dB      as dB,
    qso.tx_rst  as tx_rst,
    qso.tx_lat  as tx_lat,
    qso.Spotter as Spotter,
    qso.tx_lng  as tx_lng,
    qso.rx_lat  as rx_lat,
    qso.rx_lng  as rx_lng,
    qso.ap_lat,
    qso.ap_lng,
    325 as ionosonde,
    -5  as elev_tx,
    "US-4578" as park,
    'KD0FNR'  as call,
    call,
    (
      SELECT max(f.hmF2)
      FROM f2 AS f
      WHERE f.latitude  BETWEEN qso.ap_lat - 1.5 AND qso.ap_lat + 1.5
        AND f.longitude BETWEEN qso.ap_lng - 3   AND qso.ap_lng + 3
        AND (ABS(strftime('%s', f.timestamp) - strftime('%s', qso.timestamp)) / 60) < 6
    ) AS f2m
  FROM qso
)
SELECT
  id,
  qst    AS timestamp,
  f2m,
  launchangle(tx_lat, tx_lng, ap_lat, ap_lng, f2m) AS launchangle
FROM fmint
ORDER BY timestamp ASC;
"""

def fetch_csv(start: str, end: str) -> list[tuple[str, str, str]]:
    """
    Fetch the CSV from the Datasette endpoint, returning list of (id, f2m, launchangle).
    """
    sql = SQL_TEMPLATE.format(start=start, end=end)
    url = (
        "http://127.0.0.1:8001/_memory.csv?"
        "sql=" + urllib.parse.quote_plus(sql) +
        "&_size=max"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    reader = csv.DictReader(lines)
    records = []
    for row in reader:
        records.append((row["id"], row["f2m"], row["launchangle"]))
    return records

def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str = "REAL"):
    """
    Make sure 'column' exists in 'table'; if not, ALTER TABLE to add it.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [info[1] for info in cur.fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

def update_database(db_path: str, records: list[tuple[str, str, str]], table: str = "rm_rnb_history_pres"):
    """
    Connect to SQLite DB at db_path, ensure f2m & launchangle columns exist,
    then update each row by id.
    """
    conn = sqlite3.connect(db_path)
    ensure_column(conn, table, "f2m", "REAL")
    ensure_column(conn, table, "launchangle", "REAL")

    for id_, f2m_str, launch_str in records:
        try:
            f2m_val = float(f2m_str) if f2m_str not in (None, "") else None
        except ValueError:
            f2m_val = None
        try:
            launch_val = float(launch_str) if launch_str not in (None, "") else None
        except ValueError:
            launch_val = None

        conn.execute(
            f"UPDATE {table} SET f2m = ?, launchangle = ? WHERE id = ?",
            (f2m_val, launch_val, id_)
        )

    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Fetch QSO f2m & launchangle and update rm_toucans.db")
    parser.add_argument("--start", required=True,
                        help="UTC start timestamp (e.g. 2025-04-25T21:25:00)")
    parser.add_argument("--end",   required=True,
                        help="UTC end   timestamp (e.g. 2025-04-26T00:15:00)")
    parser.add_argument("--db",    default="rm_toucans.db",
                        help="Path to the SQLite database file")
    args = parser.parse_args()

    try:
        records = fetch_csv(args.start, args.end)
    except Exception as e:
        print(f"Error fetching CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No records returned; nothing to update.")
        sys.exit(0)

    try:
        update_database(args.db, records)
        print(f"Updated {len(records)} rows with f2m and launchangle in {args.db}.")
    except Exception as e:
        print(f"Error updating database: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
