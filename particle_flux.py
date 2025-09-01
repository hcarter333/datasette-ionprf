#!/usr/bin/env python3
"""
Plot GOES differential electron data for 2025-08-27 UTC:
1) Flux vs time (one line per energy band)
2) Daily median spectrum (median flux vs energy)
Works with the SWPC 7-day JSON or a local copy of it.
"""

import json, re, io, sys, math, datetime as dt
from urllib.request import urlopen, Request
from urllib.error import URLError
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-7-day.json"
TARGET_DATE = dt.date(2025, 8, 27)   # UTC day

# ---------------------------------------------------------------------------
# Energy parsing (returns a single float, in MeV)
# Handles: "271 keV", "31.6–45.7 MeV", "P6 (40.0-80.0 MeV)", "40300-73400 keV", "1.5 GeV"
# ---------------------------------------------------------------------------
_RANGE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(keV|MeV|GeV)?",
    re.I,
)
_SINGLE = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(keV|MeV|GeV)?", re.I)
_UNIT   = re.compile(r"(keV|MeV|GeV)", re.I)

def _num(s: str) -> float:
    return float(str(s).replace(",", ""))

def _to_mev(v: float, unit: str) -> float:
    u = (unit or "MeV").lower()
    if u == "kev": return v / 1000.0
    if u == "gev": return v * 1000.0
    return v  # MeV

def parse_energy(val):
    """
    Parse an energy label and return the numeric center **in MeV** (float).
    Returns None if it cannot be parsed.
    """
    if val is None:
        return None
    s = str(val)

    # Range → average the two values
    m = _RANGE.search(s)
    if m:
        lo, hi = _num(m.group(1)), _num(m.group(2))
        unit = (m.group(3) or (_UNIT.search(s).group(1) if _UNIT.search(s) else "MeV"))
        mid = 0.5 * (lo + hi)
        return _to_mev(mid, unit)

    # Single value
    m = _SINGLE.search(s)
    if m:
        v = _num(m.group(1))
        unit = (m.group(2) or (_UNIT.search(s).group(1) if _UNIT.search(s) else "MeV"))
        return _to_mev(v, unit)

    return None

# Optional back-compat alias (if other files still call this name)
parse_energy_to_mev = parse_energy

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def safe_get(url):
    try:
        req = Request(url, headers={"User-Agent":"python"})
        with urlopen(req, timeout=30) as r:
            return json.load(io.TextIOWrapper(r, encoding="utf-8"))
    except URLError:
        return None

def load_data():
    data = safe_get(URL)
    if data is None:
        # Fallback: try a local file named like the endpoint
        local = "differential-electrons-7-day.json"
        try:
            with open(local, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            sys.exit(
                "Couldn't fetch the URL from this environment.\n"
                "Save the file as 'differential-electrons-7-day.json' next to this script and rerun."
            )
    if not isinstance(data, list):
        # Some SWPC JSONs nest under keys; attempt to unwrap common shapes
        if "rows" in data and "columns" in data:
            cols = data["columns"]
            data = [ { cols[i]: row[i] for i in range(len(cols)) } for row in data["rows"] ]
        else:
            data = list(data)
    return data

def pick_key(rec, candidates, contains=None):
    for k in candidates:
        if k in rec:
            return k
    if contains:
        for k in rec.keys():
            if contains in k.lower():
                return k
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw = load_data()
    if not raw:
        sys.exit("No data loaded.")

    # Detect keys flexibly
    sample = raw[0]
    time_key = pick_key(sample, ["time_tag", "timeTag", "time", "timestamp"], contains="time")
    flux_key = pick_key(sample, ["flux", "differential_flux", "value"], contains="flux")
    energy_key = pick_key(sample, ["energy", "energy_bin", "chan", "channel"], contains="energy")

    if not time_key or not flux_key or not energy_key:
        sys.exit(f"Couldn't detect keys. Found keys: {list(sample.keys())}")

    # Build tidy rows
    rows = []
    for rec in raw:
        try:
            t = pd.to_datetime(rec[time_key], utc=True)
        except Exception:
            continue

        e = parse_energy(rec.get(energy_key))  # float (MeV) or None
        try:
            f = float(rec.get(flux_key))
        except (TypeError, ValueError):
            continue

        if e is None or math.isnan(e) or math.isnan(f):
            continue
        rows.append((t, e, f))

    if not rows:
        sys.exit("No usable rows after parsing.")

    df = pd.DataFrame(rows, columns=["time","energy_mev","flux"])
    df["date"] = df["time"].dt.date
    day = df[df["date"] == TARGET_DATE].drop(columns=["date"])
    if day.empty:
        sys.exit("No data for 2025-08-27 UTC in this file.")

    # 1) Flux vs time (one line per energy bin)
    pivot = day.pivot_table(index="time", columns="energy_mev", values="flux", aggfunc="mean")
    plt.figure(figsize=(11,4.5))
    pivot.sort_index(axis=1).plot(ax=plt.gca(), legend=True)
    plt.title("GOES Differential Electron Flux — 2025-08-27 (UTC)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·MeV$^{-1}$)")
    plt.legend(title="Energy (MeV)", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("diff_electrons_flux_2025-08-27.png", dpi=150)

    # 2) Daily median spectrum (median flux vs energy across the day)
    spec = day.groupby("energy_mev")["flux"].median().sort_index()
    plt.figure(figsize=(8,4.5))
    plt.plot(spec.index.values, spec.values, marker="o")
    plt.xscale("log"); plt.yscale("log")
    plt.title("GOES Differential Electron Daily Median Spectrum — 2025-08-27 (UTC)")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Median Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·MeV$^{-1}$)")
    plt.tight_layout()
    plt.savefig("diff_electrons_spectrum_2025-08-27.png", dpi=150)

    print("Saved:")
    print(" - diff_electrons_flux_2025-08-27.png")
    print(" - diff_electrons_spectrum_2025-08-27.png")

if __name__ == "__main__":
    import pandas as pd
    main()
