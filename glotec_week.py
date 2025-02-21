#!/usr/bin/env python3
import sqlite3
import sys
from datetime import datetime, timedelta

def sanitize_timestamp(ts):
    """
    Replace dash (-), colon (:), and 'T' characters with underscores,
    and remove a trailing 'Z' if present.
    """
    ts = ts.replace('-', '_').replace(':', '_').replace('T', '_')
    if ts.endswith('Z'):
        ts = ts[:-1]
    return ts

def chunk_week(db_source="glotec.db", start_ts_str=None):
    if start_ts_str is None:
        print("Usage: {} <start_timestamp in ISO8601 format, e.g., 2025-02-18T11:55:00Z>".format(sys.argv[0]))
        sys.exit(1)
    try:
        start_dt = datetime.strptime(start_ts_str, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print("Error parsing timestamp:", e)
        sys.exit(1)
    
    # One week later
    end_dt = start_dt + timedelta(days=7)
    start_dt_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_dt_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print("Extracting records from {} to {}".format(start_dt_str, end_dt_str))
    
    # Open the source database.
    src_conn = sqlite3.connect(db_source)
    src_cur = src_conn.cursor()
    
    # Query for all rows with timestamp >= start and < end.
    query = """
      SELECT uid, timestamp, longitude, latitude, hmF2, NmF2, quality_flag
      FROM glotec
      WHERE timestamp >= ? AND timestamp < ?
    """
    src_cur.execute(query, (start_dt_str, end_dt_str))
    rows = src_cur.fetchall()
    src_conn.close()
    
    if not rows:
        print("No rows found in the specified date range.")
        sys.exit(0)
    
    print("Found {} rows.".format(len(rows)))
    
    # Sanitize the timestamp for the filename.
    sanitized = sanitize_timestamp(start_ts_str)
    new_db_filename = "glotec_" + sanitized + ".db"
    
    # Create the new database and table.
    dst_conn = sqlite3.connect(new_db_filename)
    dst_cur = dst_conn.cursor()
    dst_cur.execute("""
      CREATE TABLE IF NOT EXISTS glotec (
        uid INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        longitude REAL,
        latitude REAL,
        hmF2 INTEGER,
        NmF2 REAL,
        quality_flag TEXT
      )
    """)
    dst_conn.commit()
    
    # Insert the rows into the new database.
    insert_query = """
      INSERT INTO glotec (timestamp, longitude, latitude, hmF2, NmF2, quality_flag)
      VALUES (?, ?, ?, ?, ?, ?)
    """
    for row in rows:
        # row[0] is the original uid; ignore it so that the new database assigns its own uid.
        dst_cur.execute(insert_query, (row[1], row[2], row[3], row[4], row[5], row[6]))
    dst_conn.commit()
    dst_conn.close()
    
    print("New weekly chunk created: {}".format(new_db_filename))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} <start_timestamp in ISO8601 format, e.g., 2025-02-18T11:55:00Z>".format(sys.argv[0]))
        sys.exit(1)
    chunk_week(start_ts_str=sys.argv[1])
