"""
glotec_path_plot.py
Reusable helpers: plot GloTEC fof2 or hmf2 along a great-circle path.
"""

import math, json, urllib.parse, urllib.request
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def round_to_glotec(ts_str: str) -> str:
    """Nearest 10-minute boundary on the 5-minute mark."""
    ts_str = ts_str.rstrip("Z")
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    minute = dt.minute
    nearest = min((5 + 10*k for k in range(6)), key=lambda m: abs(m - minute))
    dt = dt.replace(minute=nearest, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_ts(ts: str) -> str:
    """Remove - and : for filenames."""
    return ts.replace("-", "").replace(":", "")

# ---------------------------------------------------------------------------
# Grid cell helpers
# ---------------------------------------------------------------------------

def grid_lon(lon): return 2.5 + 5.0 * round((lon - 2.5) / 5.0)
def grid_lat(lat): return -88.75 + 2.5 * round((lat + 88.75) / 2.5)
# ---------------------------------------------------------------------------
# Fixed-y axis support (computed over ALL glotec timestamps, per path)
# ---------------------------------------------------------------------------

_FIXED_Y_CACHE = {}  # (datasource, lon_tx, lat_tx, lon_rx, lat_rx) -> {"fof2":(lo,hi), "hmf2":(lo,hi)}

def _path_cells(lon_tx, lat_tx, lon_rx, lat_rx):
    samples = list(collapse_cells(great_circle(lat_tx, lon_tx, lat_rx, lon_rx)))
    return [(lon_c, lat_c) for lon_c, lat_c, _f in samples]

def _minmax_over_all_timestamps(var, lon_tx, lat_tx, lon_rx, lat_rx, datasource):
    """
    Compute min/max for `var` over ALL timestamps in glotec, restricted to
    grid cells along this one great-circle path.
    """
    cells = _path_cells(lon_tx, lat_tx, lon_rx, lat_rx)
    if not cells:
        return (None, None)

    ts_list = fetch_glotec_timestamps(datasource)
    if not ts_list:
        return (None, None)

    vmin = vmax = None
    seen = set()
    for ts_exact in ts_list:
        ts_round = round_to_glotec(ts_exact)
        if ts_round in seen:
            continue
        seen.add(ts_round)

        data = fetch_glotec_map(ts_round, datasource)
        if not data:
            continue

        for cell in cells:
            v = data.get(cell, {}).get(var)
            if v is None:
                continue
            if vmin is None or v < vmin:
                vmin = v
            if vmax is None or v > vmax:
                vmax = v

    if vmin is None or vmax is None:
        return (None, None)

    vmin = float(vmin); vmax = float(vmax)
    if var == "fof2":
        vmin = max(0.0, vmin)

    span = vmax - vmin
    pad = span * 0.05 if span > 0 else (0.2 if var == "fof2" else 1.0)
    return (vmin - pad, vmax + pad)

def _fixed_y_ranges_for_path(lon_tx, lat_tx, lon_rx, lat_rx, datasource):
    """
    Return {"fof2": (lo,hi), "hmf2": (lo,hi)} for this path, cached.
    """
    key = (datasource, float(lon_tx), float(lat_tx), float(lon_rx), float(lat_rx))
    if key in _FIXED_Y_CACHE:
        return _FIXED_Y_CACHE[key]

    y_fof2 = _minmax_over_all_timestamps("fof2", lon_tx, lat_tx, lon_rx, lat_rx, datasource)
    y_hmf2 = _minmax_over_all_timestamps("hmf2", lon_tx, lat_tx, lon_rx, lat_rx, datasource)

    ylims = {"fof2": y_fof2, "hmf2": y_hmf2}
    _FIXED_Y_CACHE[key] = ylims
    return ylims

# ---------------------------------------------------------------------------
# Fetch one time-slice of GloTEC
# ---------------------------------------------------------------------------

def fetch_glotec_map(ts, base="http://127.0.0.1:8001"):
    sql = (
        "select longitude, latitude, hmF2, (sqrt(NmF2/.0124))/1000 as fof2 "
        "from glotec "
        f"where timestamp == '{ts}' and NmF2 > 0"
    )
    url = f"{base}/glotec_slice.json?sql=" + urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as r:
        rows = json.load(r)["rows"]

    data = {}
    for lon, lat, hmf2, fof2 in rows:
        data[(grid_lon(lon), grid_lat(lat))] = {"hmf2": hmf2, "fof2": fof2}
    return data

def compute_timeline_ylims(lon_tx, lat_tx, lon_rx, lat_rx, ts_list,
                           datasource="http://127.0.0.1:8001",
                           pad_frac=0.05):
    """
    Compute y-axis limits for fof2 & hmf2 over a specific timestamp window,
    restricted to the grid-cells along this one great-circle path.

    Returns: {"fof2": (ymin, ymax), "hmf2": (ymin, ymax)}
    """
    samples = list(collapse_cells(great_circle(lat_tx, lon_tx, lat_rx, lon_rx)))
    if not samples:
        return {}

    # Only need lon/lat cells
    cells = [(lon_c, lat_c) for lon_c, lat_c, _f in samples]

    mins = {"fof2": None, "hmf2": None}
    maxs = {"fof2": None, "hmf2": None}

    # Use rounded timestamps (same as plotting)
    seen = set()
    for ts_exact in ts_list:
        ts_round = round_to_glotec(ts_exact)
        if ts_round in seen:
            continue
        seen.add(ts_round)

        data = fetch_glotec_map(ts_round, datasource)
        if not data:
            continue

        for var in ("fof2", "hmf2"):
            for cell in cells:
                v = data.get(cell, {}).get(var)
                if v is None:
                    continue
                if mins[var] is None or v < mins[var]:
                    mins[var] = v
                if maxs[var] is None or v > maxs[var]:
                    maxs[var] = v

    ylims = {}
    for var in ("fof2", "hmf2"):
        lo, hi = mins[var], maxs[var]
        if lo is None or hi is None:
            continue
        lo = float(lo); hi = float(hi)
        if var == "fof2":
            lo = max(0.0, lo)  # fof2 shouldn't be negative
        span = hi - lo
        pad = (span * pad_frac) if span > 0 else (1.0 if var == "hmf2" else 0.2)
        ylims[var] = (lo - pad, hi + pad)

    return ylims



# ---------------------------------------------------------------------------
# Optional: fixed y-axis ranges (global per datasource + var)
# ---------------------------------------------------------------------------

_YAXIS_CACHE = {}  # (datasource, var) -> (ymin, ymax)

def fetch_glotec_y_range(var: str, base="http://127.0.0.1:8001"):
    """
    Query the glotec table for global min/max for a variable.
    Cached so we only hit Datasette once per var per datasource.

    var: "fof2" or "hmf2"
    """
    key = (base, var)
    if key in _YAXIS_CACHE:
        return _YAXIS_CACHE[key]

    if var == "fof2":
        expr = "(sqrt(NmF2/.0124))/1000.0"
        sql = (
            f"select min({expr}) as ymin, max({expr}) as ymax "
            "from glotec where NmF2 > 0"
        )
    elif var == "hmf2":
        sql = (
            "select min(hmF2) as ymin, max(hmF2) as ymax "
            "from glotec where hmF2 is not null"
        )
    else:
        raise ValueError("var must be 'fof2' or 'hmf2'")

    url = f"{base}/glotec_slice.json?sql=" + urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as r:
        rows = json.load(r)["rows"]

    ymin, ymax = (rows[0][0], rows[0][1]) if rows and rows[0] else (None, None)

    # Defensive cleanup + small padding
    if ymin is None or ymax is None:
        ymin, ymax = (0.0, 1.0)
    else:
        ymin = float(ymin)
        ymax = float(ymax)
        if var == "fof2":
            ymin = max(0.0, ymin)  # fof2 shouldn't go negative
        pad = (ymax - ymin) * 0.05
        if pad <= 0:
            pad = 1.0
        ymin -= pad
        ymax += pad

    _YAXIS_CACHE[key] = (ymin, ymax)
    return ymin, ymax


# ---------------------------------------------------------------------------
# Great-circle helpers
# ---------------------------------------------------------------------------

def sph2cart(phi, lam):
    return math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi)

def great_circle(lat1, lon1, lat2, lon2, n=100):
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    x1,y1,z1 = sph2cart(φ1,λ1); x2,y2,z2 = sph2cart(φ2,λ2)
    δ = math.acos(max(-1,min(1, x1*x2 + y1*y2 + z1*z2)))
    for i in range(n+1):
        f = i/n
        if δ < 1e-12:
            xi, yi, zi = x1, y1, z1
        else:
            a = math.sin((1-f)*δ)/math.sin(δ)
            b = math.sin(f*δ)/math.sin(δ)
            xi, yi, zi = a*x1+b*x2, a*y1+b*y2, a*z1+b*z2
        lat = math.degrees(math.atan2(zi, math.hypot(xi, yi)))
        lon = math.degrees(math.atan2(yi, xi))
        yield lat, lon, f

def collapse_cells(samples):
    last=None
    for lat, lon, f in samples:
        cell = (grid_lon(lon), grid_lat(lat))
        if cell != last:
            last = cell
            yield *cell, f   # lon_cell, lat_cell, frac

# ---------------------------------------------------------------------------
# Core plotting routine
# ---------------------------------------------------------------------------

def plot_qso(lon_tx, lat_tx, lon_rx, lat_rx, ts_exact,
             uid, call_sign,
             out_dir="plots",
             datasource="http://127.0.0.1:8001",
             fixedyaxis: bool = False):
    """
    Produce BOTH fof2 and hmf2 plots.
    Saves  <exactTimestamp>_<callsign>.png  in *out_dir*.
    Returns list of generated file paths.
    """
    ts_round = round_to_glotec(ts_exact)
    data = fetch_glotec_map(ts_round, datasource)
    if not data:
        print(f"[WARN] No GloTEC for {ts_round}")
        return []

    samples = list(collapse_cells(great_circle(lat_tx, lon_tx, lat_rx, lon_rx)))
    if not samples:
        print("[WARN] zero cells on path")
        return []

    ylims = None
    if fixedyaxis:
        ylims = _fixed_y_ranges_for_path(lon_tx, lat_tx, lon_rx, lat_rx, datasource)


    # total great-circle angle
    p1 = sph2cart(math.radians(lat_tx), math.radians(lon_tx))
    p2 = sph2cart(math.radians(lat_rx), math.radians(lon_rx))
    tot_deg = math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(p1,p2))))))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for var in ("fof2", "hmf2"):
        xs, ys = [], []
        #print(call_sign)
        #print("angle," + var)
        for lon_c, lat_c, f in samples:
            v = data.get((lon_c, lat_c), {}).get(var)
            if v is not None:
                xs.append(f * tot_deg)
                ys.append(v)
                #print(str(f*tot_deg) + "," + str(v))
        if not xs:  # no data for this var along path
            continue

        fname = f"{safe_ts(ts_exact)}_{call_sign}_{var}.png"
        fpath = out_dir / fname

        plt.figure(figsize=(8,4.5))
        plt.plot(xs, ys, 'o-')
        plt.xlabel("Angle along great circle (°)")
        plt.ylabel(f"{var.upper()} (MHz)" if var=="fof2" else f"{var.upper()} (km)")
        plt.title(f"{var.upper()} vs angle • {call_sign} • {ts_exact}")
        plt.grid(True); 

        if fixedyaxis and ylims:
            lo, hi = ylims.get(var, (None, None))
            if lo is not None and hi is not None:
                plt.ylim(lo, hi)

        plt.tight_layout()
        plt.savefig(fpath, dpi=150)
        plt.close()
        generated.append(str(fpath))

    return generated

# ---------------------------------------------------------------------------
# Extra fetch: hmF2 + NmF2 at a single time slice
# ---------------------------------------------------------------------------

def fetch_glotec_Nm(ts, base="http://127.0.0.1:8001"):
    sql = (
        "select longitude, latitude, hmF2, NmF2 "
        "from glotec "
        f"where timestamp == '{ts}'"
    )
    url = f"{base}/glotec_slice.json?sql=" + urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as r:
        rows = json.load(r)["rows"]

    data = {}
    for lon, lat, hmf2, nmf2 in rows:
        # Keep your existing key style: lower-case 'hmf2', new 'nmf2'
        data[(grid_lon(lon), grid_lat(lat))] = {"hmf2": hmf2, "nmf2": nmf2}
    return data

# ---------------------------------------------------------------------------
# Get list of timestamps from glotec_slice
# ---------------------------------------------------------------------------
def fetch_glotec_timestamps(base="http://127.0.0.1:8001"):
    """
    Return sorted list of distinct timestamps from the glotec table.
    """
    sql = "select distinct timestamp from glotec order by timestamp"
    url = f"{base}/glotec_slice.json?sql=" + urllib.parse.quote(sql, safe="")
    with urllib.request.urlopen(url) as r:
        rows = json.load(r)["rows"]
    return [row[0] for row in rows if row and row[0]]

# ---------------------------------------------------------------------------
# Physics helpers for refractive index from NmF2
# ---------------------------------------------------------------------------

# Physical constants (SI)
_EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]
_ME   = 9.1093837015e-31   # electron mass [kg]
_E    = 1.602176634e-19    # elementary charge [C]
_PI   = math.pi
_EARTH_R_KM = 6371.0
_OPERATING_FREQ_HZ = 14_057_400.0  # 14.0574 MHz

def _plasma_freq_hz_from_N(N_m3: float) -> float:
    """Plasma frequency fp [Hz] from electron density N [m^-3]."""
    if N_m3 is None or N_m3 <= 0:
        return 0.0
    omega_p = math.sqrt(N_m3 * _E*_E / (_EPS0 * _ME))  # rad/s
    return omega_p / (2.0 * _PI)

def _refractive_index_from_N(N_m3: float, f_hz: float):
    """Return (n, X, evanescent) where X=(fp/f)^2."""
    if f_hz <= 0:
        return float("nan"), float("nan"), False
    fp = _plasma_freq_hz_from_N(N_m3)
    X = (fp / f_hz) ** 2
    if X >= 1.0:
        return 0.0, X, True  # imaginary n -> treat as reflective/evanescent
    return math.sqrt(max(0.0, 1.0 - X)), X, False

# ---------------------------------------------------------------------------
# JSON generator with the SAME signature as plot_qso
# ---------------------------------------------------------------------------

def refr_json_qso(lon_tx, lat_tx, lon_rx, lat_rx, ts_exact,
                  uid, call_sign,
                  out_dir="plots",
                  datasource="http://127.0.0.1:8001",
                  fixedyaxis: bool = False):
    """
    Build a JSON object describing refractive index n along the great-circle path
    at the F2 peak (hmF2) for the rounded GloTEC time slice.

    Signature is IDENTICAL to plot_qso; return value is a Python dict ready to
    json.dumps(...) and paste into the site's JSON editor.

    The JSON includes:
      - metadata (freq, timestamps, endpoints, call/uid)
      - an ordered 'points' list with:
          angle_deg, arc_km, lon_cell, lat_cell, hmF2_km, NmF2_m3,
          plasma_freq_hz, X_fp_over_f_sq, refractive_index_n, evanescent
    """
    ts_round = round_to_glotec(ts_exact)

    # Pull hmF2 + NmF2 for the rounded slice
    data = fetch_glotec_Nm(ts_round, datasource)
    if not data:
        return {
            "meta": {
                "warning": f"No GloTEC rows for {ts_round}",
                "timestamp_exact": ts_exact,
                "timestamp_glotec": ts_round,
                "uid": uid,
                "call_sign": call_sign,
                "frequency_hz": _OPERATING_FREQ_HZ
            },
            "points": []
        }

    # Sample along the same great-circle and collapse to grid cells
    samples = list(collapse_cells(great_circle(lat_tx, lon_tx, lat_rx, lon_rx)))
    if not samples:
        return {
            "meta": {
                "warning": "zero cells on path",
                "timestamp_exact": ts_exact,
                "timestamp_glotec": ts_round,
                "uid": uid,
                "call_sign": call_sign,
                "frequency_hz": _OPERATING_FREQ_HZ
            },
            "points": []
        }

    # Total great-circle angle (deg) for distance parameterization
    p1 = sph2cart(math.radians(lat_tx), math.radians(lon_tx))
    p2 = sph2cart(math.radians(lat_rx), math.radians(lon_rx))
    tot_deg = math.degrees(math.acos(max(-1, min(1, sum(a*b for a, b in zip(p1, p2))))))

    # Build JSON points, using the same "fraction along path" as plot_qso
    points = []
    prev_angle = 0.0
    cum_km = 0.0

    for lon_c, lat_c, f in samples:
        cell = (lon_c, lat_c)
        cell_data = data.get(cell)
        if not cell_data:
            continue

        nmf2 = cell_data.get("nmf2")
        hmf2 = cell_data.get("hmf2")

        # Compute n at hmF2 using collisionless plasma formula
        n, X, ev = _refractive_index_from_N(nmf2, _OPERATING_FREQ_HZ)
        fp = _plasma_freq_hz_from_N(nmf2)

        angle = f * tot_deg
        delta_angle = max(0.0, angle - prev_angle)
        delta_km = (_PI / 180.0) * _EARTH_R_KM * delta_angle
        cum_km += delta_km
        prev_angle = angle

        points.append({
            "angle_deg": angle,
            "delta_angle_deg": delta_angle,
            "arc_km": cum_km,
            "delta_km": delta_km,

            "lon_cell": lon_c,
            "lat_cell": lat_c,

            "hmF2_km": hmf2,
            "NmF2_m3": nmf2,

            "plasma_freq_hz": fp,
            "X_fp_over_f_sq": X,
            "refractive_index_n": n,
            "evanescent": ev
        })

    return {
        "meta": {
            "frequency_hz": _OPERATING_FREQ_HZ,
            "timestamp_exact": ts_exact,
            "timestamp_glotec": ts_round,
            "uid": uid,
            "call_sign": call_sign,
            "tx": {"lat": lat_tx, "lon": lon_tx},
            "rx": {"lat": lat_rx, "lon": lon_rx},
            "total_angle_deg": tot_deg
        },
        "points": points
    }
