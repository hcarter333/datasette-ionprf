#!/usr/bin/env python3
"""
qso_path_graph.py

Plot GloTEC fof2 along a great-circle path.

Usage:
  python qso_path_graph.py lon1 lat1 lon2 lat2 YYYY-MM-DDTHH:MM:SSZ

Example:
  python qso_path_graph.py \
      -114.17 38.9743 138.7917 35.2708 2025-05-26T13:11:00Z
"""

import sys
import math
import json
import urllib.parse
import urllib.request
from datetime import datetime
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) Timestamp → nearest GloTEC update (…05, 15, …, 55)
# ---------------------------------------------------------------------------

def round_to_gloTEC(ts_str: str) -> str:
    """
    Round to the nearest 10-minute boundary on the 5-minute mark.
    13:11 → 13:15, 13:17 → 13:15, 13:23 → 13:25, etc.
    """
    ts_str = ts_str.rstrip("Z")
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    minute = dt.minute
    candidates = [5 + 10 * k for k in range(6)]          # 5,15,25,35,45,55
    minute_rounded = min(candidates, key=lambda m: abs(m - minute))
    rounded = dt.replace(minute=minute_rounded, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 2) Fetch fof2 map for that timestamp
# ---------------------------------------------------------------------------

def fetch_fof2_map(ts_rounded: str) -> dict:
    sql = (
        "select uid, timestamp, longitude, latitude, hmF2, "
        "(sqrt(NmF2/.0124))/1000 as fof2 "
        "from glotec "
        f"where timestamp == '{ts_rounded}'"
    )
    url = "http://127.0.0.1:8001/glotec_slice.json?sql=" + \
          urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as resp:
        rows = json.load(resp)["rows"]

    fmap = {}
    hmap = {}
    for _uid, _ts, lon, lat, _hmf2, fof2 in rows:
        # GloTEC grid centres
        lon_cell = lon
        lat_cell = lat
        fmap[(lon_cell, lat_cell)] = fof2

    for _uid, _ts, lon, lat, hmf2, _fof2 in rows:
        # GloTEC grid centres
        lon_cell = lon
        lat_cell = lat
        hmap[(lon_cell, lat_cell)] = hmf2

    return fmap, hmap


# ---------------------------------------------------------------------------
# 3) Great-circle helpers
# ---------------------------------------------------------------------------

def sph2cart(lat_r, lon_r):
    return (
        math.cos(lat_r) * math.cos(lon_r),
        math.cos(lat_r) * math.sin(lon_r),
        math.sin(lat_r)
    )


def great_circle_samples(lat1, lon1, lat2, lon2, n=100):
    """Yield (lat, lon, frac) for frac ∈ [0,1] along the great circle."""
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    x1, y1, z1 = sph2cart(φ1, λ1)
    x2, y2, z2 = sph2cart(φ2, λ2)
    δ = math.acos(max(-1, min(1, x1*x2 + y1*y2 + z1*z2)))

    for i in range(n + 1):
        f = i / n
        if δ < 1e-12:
            xi, yi, zi = x1, y1, z1
        else:
            a = math.sin((1 - f) * δ) / math.sin(δ)
            b = math.sin(f * δ) / math.sin(δ)
            xi, yi, zi = (
                a * x1 + b * x2,
                a * y1 + b * y2,
                a * z1 + b * z2,
            )
        lat = math.degrees(math.atan2(zi, math.hypot(xi, yi)))
        lon = math.degrees(math.atan2(yi, xi))
        yield lat, lon, f


def collapse_to_cells(samples):
    """Yield first-encounter (lon_cell, lat_cell, frac) along the path."""
    last = None
    for lat, lon, f in samples:
        #find nearest lng
        lon_cell = 2.5 + 5.0 * round((lon - 2.5) / 5.0)
        #find nearest lat
        lat_cell = -88.75 + 2.5 * round((lat + 88.75) / 2.5)
        print("f " + str(f) + " lon " + str(lon_cell) + " lat " + str(lat_cell))
        if last != (lon_cell, lat_cell):
            last = (lon_cell, lat_cell)
            yield lon_cell, lat_cell, f


# ---------------------------------------------------------------------------
# 4) Main
# ---------------------------------------------------------------------------

def main(argv):
    if len(argv) != 5:            # lon1 lat1 lon2 lat2 timestamp
        print(__doc__)
        return

    lon1, lat1, lon2, lat2 = map(float, argv[:4])
    ts_raw = argv[4]

    ts_round = round_to_gloTEC(ts_raw)
    print("Rounded UTC:", ts_round)

    fof2_map, hmf2_map = fetch_fof2_map(ts_round)
    if not fof2_map:
        print("No fof2 GloTEC rows at that timestamp.")
        return
    print(f"{len(fof2_map)} grid cells fetched.")

    if not hmf2_map:
        print("No hmF2 GloTEC rows at that timestamp.")
        return
    print(f"{len(hmf2_map)} grid cells fetched.")

    samples = great_circle_samples(lat1, lon1, lat2, lon2, n=100)
    path = list(collapse_to_cells(samples))
    if not path:
        print("No unique cells along path.")
        return

    # total path angle in degrees
    p1 = sph2cart(math.radians(lat1), math.radians(lon1))
    p2 = sph2cart(math.radians(lat2), math.radians(lon2))
    dot = max(-1, min(1, sum(a * b for a, b in zip(p1, p2))))
    tot_deg = math.degrees(math.acos(dot))

    xs, ys, xfs, yfs = [], [], [], []
    for lon_c, lat_c, frac in path:
        fof2 = fof2_map.get((lon_c, lat_c))
        hmf2 = hmf2_map.get((lon_c, lat_c))
        print("f " + str(frac) + " deg " + str(frac*tot_deg) + " lng " + str(lon_c) + " lat " + str(lat_c) + " fof2 " + str(fof2) + " hmf2 " + str(hmf2))
        if fof2 is not None:
            xs.append(frac * tot_deg)
            ys.append(fof2)
        if hmf2 is not None:
            xfs.append(frac * tot_deg)
            yfs.append(hmf2)


    if not xs:
        print("No matching cells along path contained fof2 data.")
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, "o-")
    plt.xlabel("Angle along great circle (°)")
    plt.ylabel("fof2 (kHz)")
    plt.title(f"fof2 vs angle  ({ts_round})")
    plt.grid(True)
    plt.tight_layout()

    png_name = ts_round.replace("-", "").replace(":", "")
    plt.savefig(f"{png_name}fof2.png", dpi=150)
    print("Plot saved:", f"{png_name}fof2.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(xfs, yfs, "o-")
    plt.xlabel("Angle along great circle (°)")
    plt.ylabel("hmF2 (km)")
    plt.title(f"hmF2 vs angle  ({ts_round})")
    plt.grid(True)
    plt.tight_layout()

    png_name = ts_round.replace("-", "").replace(":", "")
    plt.savefig(f"{png_name}hmf2.png", dpi=150)
    print("Plot saved:", f"{png_name}hmf2.png")

if __name__ == "__main__":
    main(sys.argv[1:])
