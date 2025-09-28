field_wedge.py was generated with the help of the GPT_5 LLM. Here's how the code turns a **CZML building polygon** into scatterers and gets their fields 
into the map/shells. 
The following flows the same order the code runs and ties each step to the physics it’s modelling.  
While this doc was started by the LLM, I'm double-checking and editing it as time allowws.

---

# 1) Parse the CZML geometry → local (x,y)

**What it reads**

* `polygon.positions.cartographicDegrees` = `[lon, lat, alt, …]` triplets forming a closed ring.
* `polygon.extrudedHeight` = the building height **H** (meters AGL).
* Optional `properties` (material, sampling, etc.) if you add them.

**What it does**

* Converts each `lon,lat` to **local ENU** coordinates `(x,y)` using your TX site as the origin:

[
(x,y)=\big(R_E ,\Delta\lambda \cos\phi_0, ; R_E,\Delta\phi\big)
]

* The result is `xy_ring` (meters), a loop of ground-plan vertices.

Why: we do all field math in a flat local frame (meters) around the site.

---

# 2) Sample the building into “Huygens sources”

From the ring we emit three kinds of point sets:

### (a) Wall planes → `extra_wall_xyz`

For each edge of the footprint:

* Sample **along** the edge every `ds` meters.
* Sample **up** the wall every `dz` meters, from `z=0…H`.

This gives a grid of points ((x_i,y_i,z_j)) on each vertical façade.

### (b) Top roof edge(s) → `extra_top_edge_xyz`

For each edge:

* Sample along the edge every `ds_top` (typically ≈ `ds/2`).
* Place points at `z = H` only.

### (c) Exterior corners → `extra_corner_xyz`

For each vertex:

* Make a vertical stick of points from `z=0…H` every `dz`.

> All of these are **equivalent secondary sources** (Huygens points). Coherent summation across many points creates directional lobes even though each point radiates with an isotropic kernel.

---

# 3) Drive each scatterer by the incident field

Before we can let the points radiate, we compute the **incident field** from your quarter-wave vertical at each scatterer:

```python
def tx_incident_to_pts(P):
    Ein = Σ_segments w_m * exp(-j k R_m)/R_m
```

* The radiator is discretized into ~15–21 z-segments with a sinusoidal current taper.
* (R_m) is the segment→point distance; we clamp (R\ge 0.1) m to avoid the Green’s-function singularity.

This yields complex `Ein_*` arrays (one per scatterer set).

---

# 4) Give walls a Fresnel reflection (Γ), edges/corners a scalar

### Walls (specular-like proxy)

For each wall point we estimate the **angle of incidence** (\theta_i):

* Compute the unit vector **from the wall point toward the TX** (approx. local arriving direction).
* Use its projection as a crude wall **normal** (points facing the TX).
* $\cos\theta_i = \hat{\mathbf{k}}_{\text{inc}}\cdot\hat{\mathbf{n}}$ (clamped to ([0,1])).

Then apply the **parallel-pol Fresnel coefficient** for a mildly lossy dielectric:

$\Gamma_\parallel(\theta_i, \tilde\varepsilon_r) ;=;\frac{\tilde\varepsilon_r\cos\theta_i - \sqrt{\tilde\varepsilon_r - \sin^2\theta_i}}{\tilde\varepsilon_r\cos\theta_i + \sqrt{\tilde\varepsilon_r - \sin^2\theta_i}}$

with $\tilde\varepsilon_r = \varepsilon'_r - j,\sigma/(\omega\varepsilon_0)$  
(defaults: $\varepsilon'_r!\approx!6,\ \sigma!\approx!0.03~\mathrm{S/m}$).

We then treat each wall point as a **re-radiating source** with amplitude $\Gamma \cdot E_{\text{inc}}$.

### Top edges & corners (diffraction / reradiation proxies)

* **Top edges**: each point uses `A_top_per * Ein_top` (incident-driven, scaled to keep the total edge contribution at a chosen level).
* **Corners**: each point uses `A_corner_per * Ein_corner` (same idea).
  For the **known North-building SE corner** we multiply the sum by a **UTD-inspired azimuthal weight** $W(\phi)$ using the wedge index $N=\pi/\beta=2/3$ for a 270° exterior (90° interior) corner.
  For **extra** corners (from CZML) we keep $W=1$ by default unless you specify the interior angle.

These scalar levels (`A_*_total` split “per point”) are the **only tunable gains** and are held fixed between runs so your comparisons remain apples-to-apples.

---

# 5) Radiate from all sources and sum coherently

At a batch of observation points (\mathbf{r}) (grid, ring, or shell):

```python
E_total(r) =
  Σ_TXsegments  w * e^{-jkR}/R
+ Σ_cornerPts   A_corner_per [⋅ W(φ)] * e^{-jkR}/R
+ Σ_topEdges    (A_top_per * Ein_top) * e^{-jkR}/R
+ Σ_balconies   (A_balc_per * Ein_balc) * e^{-jkR}/R
+ Σ_shortRoof   A_roof_per * e^{-jkR}/R
+ Σ(OP wall)    (Γ * Ein_OP) * e^{-jkR}/R
+ Σ(extra walls)(Γ * Ein_extra) * e^{-jkR}/R
+ Σ(extra tops) (A_top_per * Ein_extra) * e^{-jkR}/R
+ Σ(extra corners)(A_corner_per * Ein_extra) * e^{-jkR}/R
```

* Each term is the **free-space Green’s function** $e^{-jkR}/R$ from that source point to $\mathbf{r}$.
* We sum **complex** fields (amplitude + phase), so constructive/destructive interference arises naturally.
* The result is converted to **relative dB** (usually normalized to the median in the region) for plotting/colormaps.

---

# 6) How this maps to the outputs you’ve asked for

* **Plan overlays (250/500/1000 m)**: evaluate $(E(\mathbf{r}))$ on a horizontal grid at ~TX height; color by relative (|E|) in dB.
* **Polar lobes**: sample a horizontal ring; plot (|E|) vs azimuth (or vs bearing lines to AK/WA).
* **Zenith cuts / sky maps**: sample cones or spherical shells (your current “shells up to 90°”); color the samples by relative (|E|) in dB and export as CZML points.

---

## What’s approximated (and why it still works)

* We’re not ray-tracing exact specular paths; instead, façades are **dense Huygens sheets** with amplitude $\Gamma E_{\text{inc}}$. When you sum them, the **specular lobe** and its sidelobes naturally emerge from the phase progression across the wall.
* **Edge diffraction** is represented by line Huygens sources with a **UTD-like azimuthal weight** on the one corner we know well; others use a neutral weight unless you provide their wedge angles.
* The **short E–W roof** near the TX is kept as a weak, constant line source (a heuristic for strong near coupling); we can switch it to incident-driven if you prefer.

---

## How you can refine behavior via CZML `properties`

Per building (or per wall if you split faces), you can provide:

```json
"properties": {
  "epsilonR_real": 6.0,
  "sigma_S_per_m": 0.03,
  "sampling": { "ds_along_m": 3.0, "dz_m": 6.0 },
  "edgeModel": {
    "includeTopEdge": true,
    "includeCorners": true,
    "cornerInteriorDegDefault": 90
  }
}
```

* **Material** changes Γ and the wall strength/phase.
* **Sampling** controls lobe sharpness & runtime (denser ⇒ sharper but slower).
* **Corner angle** lets us apply UTD wedge weighting to those corners too.

---

### TL;DR

* **Walls**: grids of Huygens points whose amplitudes are **$Γ(θ) × incident field$**.
* **Edges/corners**: Huygens line sources driven by incident field; the known 90° corner uses a **UTD-like azimuth taper**.
* All contributions use the **free-space kernel** $e^{-jkR}/R$ and are **summed coherently**, producing the interference patterns you see in the overlays and shells.

If you want, I can flip on UTD wedge weighting for your new buildings’ corners (you can give me their **interior angles** or I’ll assume 90°), or assign different materials per façade to reflect glass vs. concrete dominance.
