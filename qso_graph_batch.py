#!/usr/bin/env python3
"""
batch_qso_plots.py

Generate BOTH fof2 and hmf2 plots for every QSO fetched from <json_url>.

The endpoint may return either:
  * a plain JSON list of dicts
  * a Datasette object {"columns":[...], "rows":[...]}

Required columns/keys:
  tx_lat, tx_lng, rx_lat, rx_lng, timestamp
Optional:
  uid (or id)   → used in filename
  Spotter       → callsign in plot title

Usage:
  python batch_qso_plots.py <json_url> [output_dir]

Example:
  python batch_qso_plots.py \
      http://127.0.0.1:8000/my_qsos.json  plots
"""

import sys, json, urllib.request
from pathlib import Path
from glotec_path_plot import plot_qso
from glotec_path_plot import refr_json_qso
from glotec_path_plot import fetch_glotec_timestamps
import re 

def safe_station(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "station"
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return s.strip("_") or "station"

def safe_ts_for_filename(ts: str) -> str:
    # Windows disallows ":" and a few others; keep it simple.
    # Example: 2025-12-31T15:20:00Z -> 2025-12-31T152000Z
    return (ts or "").replace(":", "")
# ---------------------------------------------------------------------------

def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)

def ds_rows_to_dicts(obj):
    cols = obj["columns"]; out=[]
    for row in obj["rows"]:
        out.append({cols[i]: row[i] for i in range(len(cols))})
    return out

def normalize_records(data):
    """
    Yield dicts with at least:
      uid, tx_lat, tx_lng, rx_lat, rx_lng, timestamp, callsign
    """
    if isinstance(data, list):
        for i, rec in enumerate(data):
            rec.setdefault("uid", rec.get("id", i))
            rec.setdefault("callsign", rec.get("Spotter", ""))
            yield rec
    elif isinstance(data, dict) and "rows" in data and "columns" in data:
        for rec in ds_rows_to_dicts(data):
            rec.setdefault("uid", rec.get("id"))
            rec.setdefault("callsign", rec.get("Spotter", ""))
            yield rec
    else:
        raise ValueError("Unexpected JSON schema")


# ---------------------------------------------------------------------------

def main(argv):
    timeline = False
    if "-timeline" in argv:
        timeline = True
        argv = [a for a in argv if a != "-timeline"]
    if "--timeline" in argv:
        timeline = True
        argv = [a for a in argv if a != "--timeline"]


    if not (1 <= len(argv) <= 2):
        print(__doc__); return
    url      = argv[0]
    out_dir  = argv[1] if len(argv)==2 else "plots"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = fetch_json(url)
    records = list(normalize_records(data))

    # -------------------------
    # TIMELINE MODE
    # -------------------------
    if timeline:
        # (1) assume single station; use first record for endpoints + callsign
        rec0 = records[0]
        try:
            tx_lat, tx_lng = float(rec0["tx_lat"]), float(rec0["tx_lng"])
            rx_lat, rx_lng = float(rec0["rx_lat"]), float(rec0["rx_lng"])
        except (KeyError, ValueError):
            print("[SKIP] malformed first record:", rec0)
            return

        station_raw = rec0.get("callsign") or rec0.get("Spotter") or ""
        station = safe_station(station_raw)

        # (2) pull every timestamp available in glotec table
        ts_list = fetch_glotec_timestamps()  # uses default base in glotec_path_plot
        if not ts_list:
            print("[NO DATA] no timestamps found in glotec")
            return

        ok = fail = 0
        for ts in ts_list:
            tsf = safe_ts_for_filename(ts)

            # (3) create fof2 & hmf2 plots for each timestamp
            uid = f"{tsf}_{station}_tl"  # uid drives filename in your current plot_qso impl
            outs = plot_qso(
                tx_lng, tx_lat, rx_lng, rx_lat,
                ts,
                uid=uid,
                call_sign=station_raw,   # keep original label in titles
                out_dir=out_dir, fixedyaxis=True
            )

            # (optional but consistent with your current behavior)
            obj = refr_json_qso(
                tx_lng, tx_lat, rx_lng, rx_lat,
                ts,
                uid=uid,
                call_sign=station_raw,
                out_dir=out_dir
            )
            # If you don't want JSON spam in timeline mode, remove the next line.
            # print(json.dumps(obj, indent=2))

            if outs:
                for p in outs:
                    print("[OK]", p)
                    ok += 1
            else:
                fail += 1

        print(f"\nDone (timeline). {ok} PNG files written, {fail} skipped/no-data.")
        return

    # -------------------------
    # NORMAL MODE (existing)
    # -------------------------


    ok=fail=0
    for rec in records:
        try:
            uid   = rec["uid"]
            tx_lat, tx_lng = float(rec["tx_lat"]), float(rec["tx_lng"])
            rx_lat, rx_lng = float(rec["rx_lat"]), float(rec["rx_lng"])
            ts     = rec["timestamp"]
            call   = rec.get("callsign","")
        except (KeyError, ValueError):
            print("[SKIP] malformed:", rec); fail+=1; continue

        outs = plot_qso(tx_lng, tx_lat, rx_lng, rx_lat,
                        ts, uid=uid, call_sign=call, out_dir=out_dir)
        obj = refr_json_qso(tx_lng, tx_lat, rx_lng, rx_lat,
                        ts, uid=uid, call_sign=call, out_dir=out_dir)
        print(json.dumps(obj, indent=2))
        if outs:
            for p in outs: print("[OK]", p)
            ok += len(outs)
        else:
            print("[NO DATA] uid", uid); fail+=1

    print(f"\nDone. {ok} PNG files written, {fail} skipped/no-data.")

if __name__ == "__main__":
    main(sys.argv[1:])
