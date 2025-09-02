#!/usr/bin/env python3
"""
GOES electrons & protons: flux/time and spectrum over a UTC date range.

Produces FOUR PNGs:
  - electrons_flux_<start>_to_<end>.png
  - electrons_spectrum_<start>_to_<end>.png
  - protons_flux_<start>_to_<end>.png
  - protons_spectrum_<start>_to_<end>.png
"""

import argparse, io, json, math, re, sys
from pathlib import Path
from datetime import datetime, date
from urllib.request import urlopen, Request
from urllib.error import URLError

import pandas as pd
import matplotlib.pyplot as plt

ELECTRONS_URL = "https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-7-day.json"
PROTONS_URL   = "https://services.swpc.noaa.gov/json/goes/primary/differential-protons-7-day.json"
SECONDARY_ELECTRONS_URL = "https://services.swpc.noaa.gov/json/goes/secondary/differential-electrons-7-day.json"
SECONDARY_PROTONS_URL   = "https://services.swpc.noaa.gov/json/goes/secondary/differential-protons-7-day.json"

# -----------------------------------------------------------------------------
# Energy parsing → numeric center in MeV (float)
# Handles: "271 keV", "31.6–45.7 MeV", "40300-73400 keV", "P6 (40.0-80.0 MeV)", "1.5 GeV"
# -----------------------------------------------------------------------------
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
    """Return numeric energy center in MeV (float) or None."""
    if val is None:
        return None
    s = str(val)
    return s
    m = _RANGE.search(s)
    if m:
        lo, hi = _num(m.group(1)), _num(m.group(2))
        unit = (m.group(3) or (_UNIT.search(s).group(1) if _UNIT.search(s) else "MeV"))
        mid = 0.5 * (lo + hi)
        return _to_mev(mid, unit)

    m = _SINGLE.search(s)
    if m:
        v = _num(m.group(1))
        unit = (m.group(2) or (_UNIT.search(s).group(1) if _UNIT.search(s) else "MeV"))
        return _to_mev(v, unit)

    return None

# Back-compat alias if any old code imports this name
parse_energy_to_mev = parse_energy

# -----------------------------------------------------------------------------
# Fetch & normalize
# -----------------------------------------------------------------------------
def safe_get(url):
    try:
        req = Request(url, headers={"User-Agent": "python"})
        with urlopen(req, timeout=30) as r:
            return json.load(io.TextIOWrapper(r, encoding="utf-8"))
    except URLError:
        return None

def load_list(url, local_fallback=None):
    data = safe_get(url)
    if data is None and local_fallback:
        try:
            with open(local_fallback, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = None
    if data is None:
        sys.exit(f"Couldn't fetch {url} (and no local fallback).")

    if not isinstance(data, list):
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            cols = data["columns"]
            data = [{cols[i]: row[i] for i in range(len(cols))} for row in data["rows"]]
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

def build_df(url):
    """Return DataFrame with columns: time (datetime64[ns, UTC]), energy_mev (float), flux (float)"""
    raw = load_list(url)
    if not raw:
        sys.exit(f"No data from {url}")

    sample = raw[0]
    time_key   = pick_key(sample, ["time_tag", "timeTag", "time", "timestamp"], contains="time")
    flux_key   = pick_key(sample, ["flux", "differential_flux", "value"], contains="flux")
    energy_key = pick_key(sample, ["energy", "energy_bin", "chan", "channel"], contains="energy")

    if not time_key or not flux_key or not energy_key:
        sys.exit(f"Couldn't detect keys for {url}. Found keys: {list(sample.keys())}")

    rows = []
    for rec in raw:
        # time
        try:
            t = pd.to_datetime(rec[time_key], utc=True)
        except Exception:
            continue
        # energy
        e = parse_energy(rec.get(energy_key))
        #if e is None or math.isnan(e):
        #    continue
        # flux
        val = rec.get(flux_key)
        try:
            # Some feeds can provide nested like [value]; handle tuples/lists
            if isinstance(val, (list, tuple)):
                val = val[0]
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        rows.append((t, e, f))

    if not rows:
        sys.exit(f"No usable rows from {url}")

    df = pd.DataFrame(rows, columns=["time", "energy_mev", "flux"])
    return df.sort_values("time")

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_time_series(day_df, species, outpath):
    """
    Lines = energy bins across the selected time range.
    """
    if day_df.empty:
        print(f"[{species}] No data in range; skipping time-series plot.")
        return

    pivot = day_df.pivot_table(index="time", columns="energy_mev", values="flux", aggfunc="mean")
    ax = plt.figure(figsize=(11, 4.8)).gca()
    pivot.sort_index(axis=1).plot(ax=ax, legend=True)
    start = str(day_df["time"].dt.date.min())
    end   = str(day_df["time"].dt.date.max())
    plt.title(f"GOES {species.capitalize()} Differential Flux — {start} to {end} (UTC)")
    plt.xlabel("Time (UTC)")

    # Correct unit label per species (electrons per keV, protons per MeV)
    if species == "electrons":
        plt.ylabel("Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·keV$^{-1}$)")
        leg_title = "Energy (MeV center)"  # numeric centers are in MeV
    else:
        plt.ylabel("Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·MeV$^{-1}$)")
        leg_title = "Energy (MeV center)"

    lg = plt.legend(title=leg_title, ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

def plot_spectrum(day_df, species, outpath):
    """
    Median flux vs energy over the selected range (log-log).
    """
    if day_df.empty:
        print(f"[{species}] No data in range; skipping spectrum plot.")
        return

    spec = day_df.groupby("energy_mev")["flux"].median().sort_index()
    ax = plt.figure(figsize=(8.2, 4.8)).gca()
    ax.plot(spec.index.values, spec.values, marker="o")
    ax.set_xscale("log"); ax.set_yscale("log")

    start = str(day_df["time"].dt.date.min())
    end   = str(day_df["time"].dt.date.max())
    plt.title(f"GOES {species.capitalize()} Daily Median Spectrum — {start} to {end} (UTC)")
    plt.xlabel("Energy (MeV)")  # numeric centers plotted in MeV
    if species == "electrons":
        plt.ylabel("Median Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·keV$^{-1}$)")
    else:
        plt.ylabel("Median Flux (particles·cm$^{-2}$·s$^{-1}$·sr$^{-1}$·MeV$^{-1}$)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GOES differential electrons & protons over a UTC date range.")
    p.add_argument("start_day", help="UTC day (YYYY-MM-DD), inclusive")
    p.add_argument("end_day",   help="UTC day (YYYY-MM-DD), inclusive")
    p.add_argument("--outdir", default=".", help="Output directory for PNGs (default: current dir)")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        start_date = datetime.strptime(args.start_day, "%Y-%m-%d").date()
        end_date   = datetime.strptime(args.end_day,   "%Y-%m-%d").date()
    except ValueError:
        sys.exit("Dates must be in YYYY-MM-DD format.")

    if end_date < start_date:
        start_date, end_date = end_date, start_date  # swap

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build dataframes
    df_e = build_df(ELECTRONS_URL)
    df_p = build_df(PROTONS_URL)
    df_s_e = build_df(SECONDARY_ELECTRONS_URL)
    df_s_p = build_df(SECONDARY_PROTONS_URL)

    # Restrict to UTC date range (inclusive)
    def slice_range(df):
        d = df.copy()
        d["date"] = d["time"].dt.date
        return d[(d["date"] >= start_date) & (d["date"] <= end_date)].drop(columns=["date"])

    e_rng = slice_range(df_e)
    p_rng = slice_range(df_p)
    e_s_rng = slice_range(df_s_e)
    p_s_rng = slice_range(df_s_p)

    # Output paths
    suffix = f"{start_date}_to_{end_date}.png"
    e_flux_path = outdir / f"electrons_flux_{suffix}"
    e_spec_path = outdir / f"electrons_spectrum_{suffix}"
    p_flux_path = outdir / f"protons_flux_{suffix}"
    p_spec_path = outdir / f"protons_spectrum_{suffix}"
    suffix = f"{start_date}_to_{end_date}.png"
    e_s_flux_path = outdir / f"electrons_flux_s_{suffix}"
    e_s_spec_path = outdir / f"electrons_spectrum_s_{suffix}"
    p_s_flux_path = outdir / f"protons_flux_s_{suffix}"
    p_s_spec_path = outdir / f"protons_spectrum_s_{suffix}"

    # Plots (four total)
    plot_time_series(e_rng, "electrons", e_flux_path)
    plot_spectrum   (e_rng, "electrons", e_spec_path)
    plot_time_series(p_rng, "protons",   p_flux_path)
    plot_spectrum   (p_rng, "protons",   p_spec_path)

    plot_time_series(e_s_rng, "electrons", e_s_flux_path)
    plot_spectrum   (e_s_rng, "electrons", e_s_spec_path)
    plot_time_series(p_s_rng, "protons",   p_s_flux_path)
    plot_spectrum   (p_s_rng, "protons",   p_s_spec_path)

if __name__ == "__main__":
    main()
