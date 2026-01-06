#!/usr/bin/env python3
"""
Compute launch angle/f2m from local SQLite databases and update rm_toucans.db
and rm_toucans_slice.db with both f2m and launchangle for each id.
"""

import argparse
import math
import sqlite3
import sys

SQL_TEMPLATE = """
WITH qso AS (
  SELECT *,
         midpoint_lat(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lat,
         midpoint_lng(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lng,
         haversine(tx_lat, tx_lng, rx_lat, rx_lng) as length
  FROM [rm_toucans_slice].rm_rnb_history_pres
  WHERE timestamp BETWEEN '{start}' AND '{end}'
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

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi
EARTH_RADIUS_KM = 6357


def cartesian_x(latitude: float, longitude: float) -> float:
    return math.cos(latitude * DEG2RAD) * math.cos(longitude * DEG2RAD)


def cartesian_y(latitude: float, longitude: float) -> float:
    return math.cos(latitude * DEG2RAD) * math.sin(longitude * DEG2RAD)


def cartesian_z(latitude: float) -> float:
    return math.sin(latitude * DEG2RAD)


def spherical_lat(x: float, y: float, z: float) -> float:
    r = math.sqrt(x * x + y * y)
    return math.atan2(z, r) * RAD2DEG


def spherical_lng(x: float, y: float) -> float:
    return math.atan2(y, x) * RAD2DEG


def midpoint_lat(f0: float, l0: float, f1: float, l1: float) -> float:
    return partial_path_lat(f0, l0, f1, l1, 2)


def midpoint_lng(f0: float, l0: float, f1: float, l1: float) -> float:
    return partial_path_lng(f0, l0, f1, l1, 2)


def partial_path_lat(f0: float, l0: float, f1: float, l1: float, parts: float) -> float:
    x_0 = cartesian_x(f0, l0)
    y_0 = cartesian_y(f0, l0)
    z_0 = cartesian_z(f0)
    x_1 = cartesian_x(f1, l1)
    y_1 = cartesian_y(f1, l1)
    z_1 = cartesian_z(f1)
    x_mid = x_0 + ((x_1 - x_0) / parts)
    y_mid = y_0 + ((y_1 - y_0) / parts)
    z_mid = z_0 + ((z_1 - z_0) / parts)
    return spherical_lat(x_mid, y_mid, z_mid)


def partial_path_lng(f0: float, l0: float, f1: float, l1: float, parts: float) -> float:
    x_0 = cartesian_x(f0, l0)
    y_0 = cartesian_y(f0, l0)
    z_0 = cartesian_z(f0)
    x_1 = cartesian_x(f1, l1)
    y_1 = cartesian_y(f1, l1)
    z_1 = cartesian_z(f1)
    x_mid = x_0 + ((x_1 - x_0) / parts)
    y_mid = y_0 + ((y_1 - y_0) / parts)
    z_mid = z_0 + ((z_1 - z_0) / parts)
    return spherical_lng(x_mid, y_mid)


def swept_angle(f0: float, l0: float, f1: float, l1: float) -> float:
    tx_x = cartesian_x(f0, l0)
    tx_y = cartesian_y(f0, l0)
    tx_z = cartesian_z(f0)
    rx_x = cartesian_x(f1, l1)
    rx_y = cartesian_y(f1, l1)
    rx_z = cartesian_z(f1)
    cross_x = (tx_y * rx_z) - (tx_z * rx_y)
    cross_y = (tx_z * rx_x) - (tx_x * rx_z)
    cross_z = (tx_x * rx_y) - (tx_y * rx_x)
    g_mag = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    return math.asin(g_mag) * RAD2DEG


def law_cosines(re: float, fmax: float, swangl: float) -> float:
    return math.sqrt(re**2 + (re + fmax) ** 2 - 2 * re * (re + fmax) * math.cos(swangl * DEG2RAD))


def law_sines(re: float, c_side: float, swangle: float) -> float:
    return math.asin((re * math.sin(swangle * DEG2RAD)) / c_side) * RAD2DEG


def launchangle(f0: float, l0: float, f1: float, l1: float, f2m: float) -> float:
    sw = swept_angle(f0, l0, f1, l1)
    ts = law_cosines(EARTH_RADIUS_KM, f2m, sw)
    thdangle = law_sines(EARTH_RADIUS_KM, ts, sw)
    return (180 - sw - thdangle) - 90


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = lat1 * DEG2RAD
    lon1_r = lon1 * DEG2RAD
    lat2_r = lat2 * DEG2RAD
    lon2_r = lon2 * DEG2RAD
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371000 * c


def register_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("midpoint_lat", 4, midpoint_lat)
    conn.create_function("midpoint_lng", 4, midpoint_lng)
    conn.create_function("launchangle", 5, launchangle)
    conn.create_function("haversine", 4, haversine)


def fetch_records(
    start: str,
    end: str,
    rm_toucans_slice_db: str,
    glotec_slice_db: str,
) -> list[tuple[str, str, str]]:
    """
    Run the SQL query directly against SQLite databases, returning (id, f2m, launchangle).
    """
    conn = sqlite3.connect(":memory:")
    register_functions(conn)
    conn.execute("ATTACH DATABASE ? AS rm_toucans_slice", (rm_toucans_slice_db,))
    conn.execute("ATTACH DATABASE ? AS glotec_slice", (glotec_slice_db,))
    sql = SQL_TEMPLATE.format(start=start, end=end)
    rows = conn.execute(sql).fetchall()
    conn.close()
    records = []
    for row in rows:
        records.append((row[0], row[2], row[3]))
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

def update_databases(
    start: str,
    end: str,
    rm_db: str,
    rm_slice_db: str,
    glotec_slice_db: str,
) -> int:
    records = fetch_records(start, end, rm_slice_db, glotec_slice_db)
    if not records:
        return 0
    update_database(rm_db, records)
    update_database(rm_slice_db, records)
    return len(records)

def main():
    parser = argparse.ArgumentParser(
        description="Fetch QSO f2m & launchangle and update rm_toucans.db and rm_toucans_slice.db"
    )
    parser.add_argument("--start", required=True,
                        help="UTC start timestamp (e.g. 2025-04-25T21:25:00)")
    parser.add_argument("--end",   required=True,
                        help="UTC end   timestamp (e.g. 2025-04-26T00:15:00)")
    parser.add_argument("--rm-db", default="rm_toucans.db",
                        help="Path to the rm_toucans SQLite database file")
    parser.add_argument("--rm-slice-db", default="rm_toucans_slice.db",
                        help="Path to the rm_toucans_slice SQLite database file")
    parser.add_argument("--glotec-slice-db", default="glotec_slice.db",
                        help="Path to the glotec_slice SQLite database file")
    args = parser.parse_args()

    try:
        updated = update_databases(
            args.start,
            args.end,
            args.rm_db,
            args.rm_slice_db,
            args.glotec_slice_db,
        )
    except Exception as e:
        print(f"Error updating database: {e}", file=sys.stderr)
        sys.exit(1)

    if not updated:
        print("No records returned; nothing to update.")
        sys.exit(0)

    print(
        "Updated "
        f"{updated} rows with f2m and launchangle in {args.rm_db} and {args.rm_slice_db}."
    )

if __name__ == "__main__":
    main()
