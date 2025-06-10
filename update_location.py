#!/usr/bin/env python3
"""
Update tx_lng, tx_lat, and elev_tx for records in rm_rnb_history_pres
between two timestamps.
"""

import argparse
import sqlite3
import sys

def update_records(db_path: str, start_ts: str, end_ts: str, tx_lng: float, tx_lat: float, elev_tx: float):
    """
    Connect to the SQLite database at db_path, and for every row in
    rm_rnb_history_pres whose timestamp is between start_ts and end_ts
    (inclusive), set tx_lng, tx_lat, and elev_tx to the given values.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        sql = """
        UPDATE rm_rnb_history_pres
        SET tx_lng  = ?,
            tx_lat  = ?,
            elev_tx = ?
        WHERE timestamp BETWEEN ? AND ?
        """
        cursor.execute(sql, (tx_lng, tx_lat, elev_tx, start_ts, end_ts))
        conn.commit()
        print(f"{cursor.rowcount} rows updated.")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Update tx_lng, tx_lat, elev_tx for records in rm_rnb_history_pres "
                    "between two timestamps."
    )
    parser.add_argument(
        "--db", "-d",
        default="rm_toucans.db",
        help="Path to the SQLite database file (default: rm_toucans.db)"
    )
    parser.add_argument(
        "--start", "-s",
        required=True,
        help="Start timestamp (inclusive), e.g. 2025-05-01T00:00:00"
    )
    parser.add_argument(
        "--end", "-e",
        required=True,
        help="End timestamp (inclusive), e.g. 2025-05-02T00:00:00"
    )
    parser.add_argument(
        "--tx_lng",
        type=float,
        required=True,
        help="New transmitter longitude (decimal degrees)"
    )
    parser.add_argument(
        "--tx_lat",
        type=float,
        required=True,
        help="New transmitter latitude (decimal degrees)"
    )
    parser.add_argument(
        "--elev_tx",
        type=float,
        required=True,
        help="New transmitter elevation (meters)"
    )
    args = parser.parse_args()

    try:
        update_records(
            db_path=args.db,
            start_ts=args.start,
            end_ts=args.end,
            tx_lng=args.tx_lng,
            tx_lat=args.tx_lat,
            elev_tx=args.elev_tx
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
