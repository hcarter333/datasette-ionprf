#!/usr/bin/env python3
"""
Fetch QSO CSV from Datasette, extract id and f2m, and update an SQLite database.
"""

import argparse
import requests
import csv
import sqlite3
import urllib.parse
import sys

def fetch_csv(start: str, end: str) -> list[tuple[str, str]]:
    """
    Fetch the CSV from the Datasette endpoint for the given time window.
    Returns a list of (id, f2m) tuples as strings.
    """
    sql_template = """
WITH qso AS (
  SELECT *,
         midpoint_lat(tx_lat, tx_lng, rx_lat, rx_lng)    AS ap_lat,
         midpoint_lng(tx_lat, tx_lng, rx_lat, rx_lng)   AS ap_lng,
         haversine(tx_lat, tx_lng, rx_lat, rx_lng)      AS length
  FROM [rm_toucans_slice].rm_rnb_history_pres
  WHERE timestamp BETWEEN '{start}' AND '{end}'
    AND dB > 100
),
f2 AS (
  SELECT *
  FROM [glotec_slice].glotec
  WHERE timestamp BETWEEN '{start}' AND '{end}'
)
SELECT
  qso.id,
  (
    SELECT max(f.hmF2)
    FROM f2 AS f
    WHERE f.latitude  BETWEEN qso.ap_lat - 1.75 AND qso.ap_lat + 1.75
      AND f.longitude BETWEEN qso.ap_lng - 2.6 AND qso.ap_lng + 2.6
      AND (ABS(strftime('%s', f.timestamp) - strftime('%s', qso.timestamp)) / 60) < 6
  ) AS f2m
FROM qso
ORDER BY timestamp ASC;
"""
    sql = sql_template.format(start=start, end=end)
    url = (
        "http://192.168.0.203/sam_datasette/_memory.csv?"
        "sql=" + urllib.parse.quote_plus(sql) +
        "&_size=max"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text.splitlines()
    reader = csv.DictReader(text)
    return [(row["id"], row["f2m"]) for row in reader]

def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str = "REAL"):
    """
    Ensure that 'column' exists in 'table'; if not, ALTER TABLE to add it.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [info[1] for info in cur.fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

def update_database(db_path: str, records: list[tuple[str, str]], table: str = "rm_rnb_history_pres"):
    """
    Update the given SQLite database file, setting f2m for each id.
    """
    conn = sqlite3.connect(db_path)
    ensure_column(conn, table, "f2m", "REAL")
    for id_, f2m_str in records:
        try:
            f2m_val = float(f2m_str) if f2m_str != "" else None
        except ValueError:
            f2m_val = None
        conn.execute(
            f"UPDATE {table} SET f2m = ? WHERE id = ?",
            (f2m_val, id_)
        )
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Fetch QSO f2m and update rm_toucans.db")
    parser.add_argument("--start", required=True,
                        help="UTC start timestamp (e.g. 2025-04-25T21:25:00)")
    parser.add_argument("--end", required=True,
                        help="UTC end   timestamp (e.g. 2025-04-26T00:15:00)")
    parser.add_argument("--db", default="rm_toucans.db",
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
        print(f"Updated f2m for {len(records)} rows in {args.db}.")
    except Exception as e:
        print(f"Error updating database: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
