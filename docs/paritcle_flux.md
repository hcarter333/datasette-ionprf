# `particle_flux.py` — GOES Differential Electrons & Protons (Time-Series + Spectrum)

This script fetches **GOES differential electron and proton flux** from SWPC’s 7-day JSON feeds, filters by a **UTC date range**, and produces **four plots**:

1) Electrons — flux vs time  
2) Electrons — median spectrum over the range  
3) Protons — flux vs time  
4) Protons — median spectrum over the range

---

## Requirements

- **Python** 3.9+
- Packages: `pandas`, `matplotlib`
  
Install (any OS):
```bash
python -m pip install pandas matplotlib
```

> The script uses Python’s built-in `urllib` for downloads—no extra HTTP libraries needed.

---

## Data Sources

- Electrons: `https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-7-day.json`  
- Protons:   `https://services.swpc.noaa.gov/json/goes/primary/differential-protons-7-day.json`

These endpoints provide rolling **7-day** windows. If your requested dates fall outside the last 7 days available at run time, the script will find no data.

---

## Usage

```bash
python particle_flux.py <START_DAY> <END_DAY> [--outdir OUTDIR]
```

**Arguments**
- `<START_DAY>` – UTC calendar date, `YYYY-MM-DD` (inclusive)
- `<END_DAY>` – UTC calendar date, `YYYY-MM-DD` (inclusive)
- `--outdir OUTDIR` – directory to write PNGs (default: current directory)

**Examples**
```bash
# Single day (UTC)
python particle_flux.py 2025-08-27 2025-08-27

# Multi-day range (UTC)
python particle_flux.py 2025-08-26 2025-08-28 --outdir plots
```

> On Windows PowerShell:
> ```powershell
> python .\particle_flux.py 2025-08-27 2025-08-27 --outdir .\plots
> ```

---

## Outputs

Four PNG files are written to `--outdir`:

- `electrons_flux_<start>_to_<end>.png`  
- `electrons_spectrum_<start>_to_<end>.png`  
- `protons_flux_<start>_to_<end>.png`  
- `protons_spectrum_<start>_to_<end>.png`

Where `<start>`/`<end>` are the UTC dates you provided.

**What they show**

- **Flux vs time (electrons/protons)**: one line per energy bin across the selected date range.
- **Median spectrum**: log–log plot of **median flux vs. bin center energy** aggregated over the entire date range.

---

## Units & Conventions (important for interpretation)

- **Electrons (differential):**  
  Flux units are **particles / (cm² · s · sr · keV)**.  
  The script:
  - parses energy labels (which may be in **keV**, **MeV**, or **GeV**, sometimes as ranges like `40300–73400 keV`),
  - converts **bin center energy to MeV** for sorting/plotting on the x-axis,  
  - **keeps the y-axis unit as “… per keV”** to match SWPC’s published electron differential units.

- **Protons (differential):**  
  Flux units are **particles / (cm² · s · sr · MeV)**.  
  The y-axis is labeled accordingly.

> Why electrons “per keV” but x-axis in MeV?  
> We sort/space energy bins numerically in MeV (consistent across electrons & protons) but preserve each species’ correct differential unit on the y-axis. If you’d prefer to relabel electron flux “per MeV,” multiply electron fluxes by **1/1000** before plotting (since 1 MeV = 1000 keV).

---

## How energy bins are parsed

- Accepts single values like `271 keV` or `1.5 GeV`.
- Accepts ranges like `31.6–45.7 MeV` or `40300-73400 keV`; **uses the average (midpoint)**.
- Detects units anywhere in the string; converts to a **numeric center in MeV** for plotting/sorting.

---

## Date handling

- The script converts timestamps to **UTC** and filters by the **UTC calendar dates** you supply.
- The range is **inclusive** of both start and end days.

---

## Troubleshooting

**“No data in range; skipping …”**  
- Your date range may not overlap the most recent **7 days** in the SWPC feeds. Choose dates within the current feed window.

**“Couldn’t fetch … (and no local fallback).”**  
- Check your internet connection and try again.  
- If needed for offline use, download the JSONs and modify the script to point to local files.

**NameError / TypeError around energy parsing**  
- This script’s `parse_energy()` returns a **float** (center energy in MeV). If you copy code into another module, ensure you’re not trying to treat the entire return value as a tuple, or calling an old function name (`parse_energy_to_mev`). In this script, those are already consistent.

---

## Notes & Extensions

- The time-series plots can be crowded if many bins are present. Consider sub-selecting bins or using a legend outside the axes for presentations.
- You can change the aggregation for the spectrum (e.g., mean, 25th/75th percentile) by editing the `groupby(...).median()` call.
- If you want **omnidirectional** estimates from directional flux assuming isotropy, multiply by **4π sr**—but document the assumption.

---

## License & Attribution

- Data: © NOAA/NWS Space Weather Prediction Center (SWPC).  
- Script: you may include this file in your repo with attribution to SWPC for the data sources.

---

## Quick checklist

- [ ] Python 3.9+  
- [ ] `pip install pandas matplotlib`  
- [ ] Run with UTC dates: `python particle_flux.py 2025-08-27 2025-08-27`  
- [ ] Find PNGs in `--outdir` (default `.`)  
