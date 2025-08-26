import argparse
import csv
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import urllib.parse
import requests

CSV_URL = "https://raw.githubusercontent.com/hcarter333/rm-rbn-history/main/rm_rnb_history.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ts(ts: str) -> datetime:
    """Parse timestamps from the CSV (YYYY/MM/DD HH:MM:SS)."""
    return datetime.strptime(ts, "%Y/%m/%d %H:%M:%S")


def isoformat_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_qso_sql(min_lat: float, max_lat: float, min_lng: float, max_lng: float,
                  start_ts: str, end_ts: str, elev: float) -> str:
    """Return SQL string equivalent to buildQSOSQL in glotec_local3.html."""
    return f"""
WITH qso AS (
  SELECT *,
         midpoint_lat(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lat,
         midpoint_lng(tx_lat, tx_lng, rx_lat, rx_lng) as ap_lng,
         haversine(tx_lat, tx_lng, rx_lat, rx_lng) as length
  FROM [rm_toucans_slice].rm_rnb_history_pres
  WHERE timestamp BETWEEN '{start_ts}' AND '{end_ts}' AND dB > 100
),
f2 AS (
  SELECT *
  FROM [glotec_slice].glotec
  WHERE timestamp BETWEEN '{start_ts}' AND '{end_ts}'
)
SELECT
  qso.timestamp,
  strftime('%Y%m%d', qso.timestamp) as date,
  strftime('%H%M', qso.timestamp) as time,
  dB,
  tx_rst,
  qso.tx_lat,
  qso.Spotter,
  qso.tx_lng,
  qso.rx_lat,
  qso.rx_lng,
  325 as ionosonde,
  '{elev}' as elev_tx,
  0 as hmF2Map,
  "US-4578" as park,
  'KD0FNR' as call,
  call,
  (
    SELECT max(f.hmF2)
    FROM f2 AS f
    WHERE f.latitude BETWEEN qso.ap_lat - 1.75 AND qso.ap_lat + 1.75
      AND f.longitude BETWEEN qso.ap_lng - 2.6 AND qso.ap_lng + 2.6
      AND (ABS(strftime('%s', f.timestamp) - strftime('%s', qso.timestamp)) / 60) < 6
  ) AS f2m
FROM qso
ORDER BY timestamp ASC;"""

# ---------------------------------------------------------------------------
# Step 1: slice CSV into rm_toucans_slice.db
# ---------------------------------------------------------------------------

def slice_rm_history(start: datetime, end: datetime, out_db: Path) -> None:
    resp = requests.get(CSV_URL, timeout=30)
    resp.raise_for_status()
    rows = []
    reader = csv.DictReader(resp.text.splitlines())
    for row in reader:
        ts = parse_ts(row["timestamp"])
        if start <= ts <= end:
            rows.append({
                "id": int(row["id"]),
                "tx_lng": float(row["tx_lng"]),
                "tx_lat": float(row["tx_lat"]),
                "rx_lng": float(row["rx_lng"]),
                "rx_lat": float(row["rx_lat"]),
                "timestamp": isoformat_z(ts),
                "dB": float(row["dB"]),
                "frequency": float(row["frequency"]),
                "Spotter": row["Spotter"],
            })
    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS rm_rnb_history_pres")
    cur.execute(
        """CREATE TABLE rm_rnb_history_pres (
            id INTEGER,
            tx_lng REAL, tx_lat REAL,
            rx_lng REAL, rx_lat REAL,
            timestamp TEXT,
            dB REAL,
            frequency REAL,
            Spotter TEXT
        )"""
    )
    cur.executemany(
        "INSERT INTO rm_rnb_history_pres VALUES (:id,:tx_lng,:tx_lat,:rx_lng,:rx_lat,:timestamp,:dB,:frequency,:Spotter)",
        rows,
    )
    conn.commit(); conn.close()

# ---------------------------------------------------------------------------
# Step 2: populate glotec_slice.db via glotec.py
# ---------------------------------------------------------------------------

def populate_glotec(start: datetime, end: datetime) -> None:
    s = isoformat_z(start - timedelta(minutes=5))
    e = isoformat_z(end + timedelta(minutes=5))
    subprocess.check_call(["python", "glotec.py", "-nmpatch", s, e])
    # Rename database if needed
    if Path("glotec.db").exists():
        Path("glotec.db").rename("glotec_slice.db")

# ---------------------------------------------------------------------------
# Step 3: compute launch angle / f2m
# ---------------------------------------------------------------------------

def compute_f2m(start: datetime, end: datetime) -> None:
    subprocess.check_call([
        "python", "f2m_launch_once.py", "--start", isoformat_z(start), "--end", isoformat_z(end), "--db", "rm_toucans.db"
    ])

# ---------------------------------------------------------------------------
# Step 4: swap databases
# ---------------------------------------------------------------------------

def finalize_db(slice_db: Path) -> None:
    main = Path("rm_toucans.db")
    if not main.exists():
        return
    if main.stat().st_size >= slice_db.stat().st_size:
        slice_db.rename("rm_toucans_slice_bu.db")
        main.rename(slice_db)

# ---------------------------------------------------------------------------
# Step 5: start datasette
# ---------------------------------------------------------------------------

def start_datasette():
    cmd = [
        "python3", "-m", "datasette", "glotec_slice.db", "rm_toucans_slice.db",
        "--plugins-dir=plugins", "--template-dir", "plugins/templates", "--root",
        "--static", "assets:static-files/", "--setting", "sql_time_limit_ms", "17000",
        "--setting", "max_returned_rows", "100000", "--cors", "--crossdb"
    ]
    return subprocess.Popen(cmd)

# ---------------------------------------------------------------------------
# Step 6: fetch CZML
# ---------------------------------------------------------------------------

def fetch_czml(start_ts: datetime, end_ts: datetime, outfile: Path,
               base_url: str = "http://127.0.0.1:8001/") -> None:
    qsql = build_qso_sql(-90, 90, -180, 180, isoformat_z(start_ts), isoformat_z(end_ts), -5)
    url = base_url + "_memory.czml?sql=" + urllib.parse.quote(qsql, safe="")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    outfile.write_bytes(resp.content)

# ---------------------------------------------------------------------------
# Step 7: create directories for each row
# ---------------------------------------------------------------------------

def create_artifact_dirs(base_dir: Path, slice_db: Path) -> None:
    conn = sqlite3.connect(slice_db)
    cur = conn.cursor()
    cur.execute("SELECT Spotter, timestamp FROM rm_rnb_history_pres")
    rows = cur.fetchall(); conn.close()
    for spotter, ts in rows:
        date = datetime.fromisoformat(ts.replace("Z", "")).strftime("%Y_%m_%d")
        d = base_dir / f"{spotter}_{date}"
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run activation workflow")
    ap.add_argument("start", help="Start timestamp (UTC, e.g. 2025-08-19T00:00:00Z)")
    ap.add_argument("end", help="End timestamp   (UTC, e.g. 2025-08-21T00:00:00Z)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%SZ")
    end = datetime.strptime(args.end, "%Y-%m-%dT%H:%M:%SZ")

    slice_rm_history(start, end, Path("rm_toucans_slice.db"))
    populate_glotec(start, end)
    compute_f2m(start, end)
    finalize_db(Path("rm_toucans_slice.db"))

    ds_proc = start_datasette()

    czml_name = start.strftime("%Y_%m_%d_%H_%M.czml")
    fetch_czml(start, end, Path(czml_name))

    artifact_dir = Path(start.strftime("activation_artifact_%y_%m_%d_%H_%M"))
    artifact_dir.mkdir(exist_ok=True)
    create_artifact_dirs(artifact_dir, Path("rm_toucans_slice.db"))

    ds_proc.terminate()

if __name__ == "__main__":
    main()
