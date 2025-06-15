#!/usr/bin/env python3
"""
batch_qso_plots.py

Generate fof2 or hmf2 plots for many QSOs fetched from a JSON URL.

Supports:
  * plain JSON list of dicts
  * Datasette object {"columns":...,"rows":...}

Usage:
  python batch_qso_plots.py <json_url> [output_dir] [--var=fof2|hmf2]
"""

import sys, json, urllib.request, argparse
from pathlib import Path
from glotec_path_plot import plot_qso   # library above

# ---------------------------------------------------------------------------

def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)

def rows_to_dicts(ds):
    cols=ds["columns"]; out=[]
    for row in ds["rows"]:
        out.append({cols[i]:row[i] for i in range(len(cols))})
    return out

def normalize(data):
    if isinstance(data,list):
        for i,rec in enumerate(data):
            rec.setdefault("uid", rec.get("id", i))
            yield rec
    elif isinstance(data,dict) and "rows" in data and "columns" in data:
        for rec in rows_to_dicts(data):
            rec.setdefault("uid", rec.get("id"))
            yield rec
    else:
        raise ValueError("Unknown JSON schema")

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir", nargs="?", default="plots")
    ap.add_argument("--var", choices=["fof2","hmf2"], default="fof2",
                    help="parameter to plot (default fof2)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    data = fetch_json(args.url)
    records = list(normalize(data))

    ok=fail=0
    for rec in records:
        try:
            uid = rec["uid"]
            tx_lat, tx_lng = float(rec["tx_lat"]), float(rec["tx_lng"])
            rx_lat, rx_lng = float(rec["rx_lat"]), float(rec["rx_lng"])
            ts = rec["timestamp"]
        except (KeyError, ValueError):
            print("[SKIP] malformed record:", rec)
            fail+=1; continue

        out = plot_qso(tx_lng, tx_lat, rx_lng, rx_lat,
                       ts, args.outdir, var=args.var, uid=uid)
        if out:
            print(f"[OK] {out}")
            ok+=1
        else:
            fail+=1

    print(f"\nDone. {ok} plots, {fail} skipped/no-data.")

if __name__ == "__main__":
    main()
