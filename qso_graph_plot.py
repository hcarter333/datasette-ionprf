"""qso_graph_plot.py
Reusable helpers for plotting GloTEC parameters along a great-circle path.
"""

import math, json, urllib.parse, urllib.request
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def round_to_glotec(ts_str: str) -> str:
    ts_str = ts_str.rstrip("Z")
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    minute = dt.minute
    nearest = min((5 + 10*k for k in range(6)), key=lambda m: abs(m - minute))
    return dt.replace(minute=nearest, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def grid_lon(lon): return 2.5 + 5.0 * round((lon - 2.5) / 5.0)
def grid_lat(lat): return -88.75 + 1.5 * round((lat + 88.75) / 1.5)

# ---------------------------------------------------------------------------
# Remote data
# ---------------------------------------------------------------------------

def fetch_glotec_map(ts, base="http://127.0.0.1:8001"):
    sql = (
        "select longitude, latitude, hmF2, (sqrt(NmF2/.0124))/1000 as fof2 "
        "from glotec where timestamp == '{ts}'"
    ).format(ts=ts)
    url = f"{base}/glotec_slice.json?sql=" + urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as r:
        rows = json.load(r)["rows"]

    data = {}
    for lon, lat, hmf2, fof2 in rows:
        data[(grid_lon(lon), grid_lat(lat))] = {"hmf2": hmf2, "fof2": fof2}
    return data

# ---------------------------------------------------------------------------
# Great-circle helpers
# ---------------------------------------------------------------------------

def sph2cart(phi, lam):  # radians
    return math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi)

def great_circle(lat1, lon1, lat2, lon2, n=100):
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    x1,y1,z1 = sph2cart(φ1,λ1); x2,y2,z2 = sph2cart(φ2,λ2)
    δ = math.acos(max(-1,min(1, x1*x2 + y1*y2 + z1*z2)))
    for i in range(n+1):
        f = i/n
        if δ < 1e-12:
            xi, yi, zi = x1,y1,z1
        else:
            a,b = math.sin((1-f)*δ)/math.sin(δ), math.sin(f*δ)/math.sin(δ)
            xi, yi, zi = a*x1+b*x2, a*y1+b*y2, a*z1+b*z2
        lat = math.degrees(math.atan2(zi, math.hypot(xi, yi)))
        lon = math.degrees(math.atan2(yi, xi))
        yield lat, lon, f

def collapse_cells(samples):
    last=None
    for lat, lon, f in samples:
        cell=(grid_lon(lon), grid_lat(lat))
        if cell!=last:
            last=cell; yield *cell, f


# ---------------------------------------------------------------------------
# Lightweight public helpers
# ---------------------------------------------------------------------------

def great_circle_points(lat1, lon1, lat2, lon2, segments=100):
    """Yield (lat, lon) pairs along a great-circle path."""
    for lat, lon, _ in great_circle(lat1, lon1, lat2, lon2, n=segments):
        yield lat, lon


def filter_redundant(points):
    """Drop consecutive points that land in the same 5°×2.5° cell."""
    last_cell = None
    for lat, lon in points:
        cell = (math.floor(lat / 5.0), math.floor(lon / 2.5))
        if cell != last_cell:
            last_cell = cell
            yield (lat, lon)

# ---------------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------------

def plot_qso(lon1, lat1, lon2, lat2, ts_utc,
             out_dir=".", datasource="http://127.0.0.1:8001",
             var="fof2", uid=None):
    """
    Plot *var* ("fof2" or "hmf2") vs great-circle angle; return PNG path or None.
    """
    if var not in {"fof2","hmf2"}:
        raise ValueError("var must be 'fof2' or 'hmf2'")

    ts_round = round_to_glotec(ts_utc)
    gmap = fetch_glotec_map(ts_round, datasource)
    if not gmap:
        print(f"[WARN] No GloTEC data for {ts_round}")
        return None

    pts = list(collapse_cells(great_circle(lat1, lon1, lat2, lon2)))
    if not pts:
        print("[WARN] Path produced no cells")
        return None

    # path angle
    p1=sph2cart(math.radians(lat1), math.radians(lon1))
    p2=sph2cart(math.radians(lat2), math.radians(lon2))
    ang=math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(p1,p2))))))

    xs,ys=[],[]
    for lon_c,lat_c,f in pts:
        val=gmap.get((lon_c,lat_c),{}).get(var)
        if val is not None:
            xs.append(f*ang); ys.append(val)
    if not xs:
        print("[WARN] No grid cells with data along path")
        return None

    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    stem = f"{uid}_{var}_{ts_round.replace('-','').replace(':','')}"
    outfile = out/f"{stem}.png"

    plt.figure(figsize=(8,4.5))
    plt.plot(xs,ys,'o-')
    plt.xlabel("Angle along great circle (°)")
    plt.ylabel(f"{var.upper()} (km)" if var=="hmf2" else f"{var} (MHz)")
    plt.title(f"{var.upper()} vs angle  ({ts_round})")
    plt.grid(True); plt.tight_layout(); plt.savefig(outfile,dpi=150); plt.close()
    return str(outfile)
