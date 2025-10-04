# Build 3‑D field shells (CZML point clouds) up to 90° elevation for an azimuth sector 315°→360°→100°,
# at radii 250 m, 500 m, and 1000 m, using the current modeling geometry (TX, North building, Alcoa wall,
# short roof edge) plus any additional buildings supplied in /mnt/data/field_wedge_additional_buildings.czml.

import json, math, numpy as np, urllib.request
from pathlib import Path

INCLUDE = dict(north=False, alcoa=False, short_roof=False, bleachers=False, flat_ground=True)

# ---------- Recreate the solver context (compact but consistent with earlier runs) ----------
R_earth = 6371000.0
def ll_to_xy(lat0, lon0, lat, lon):
    lat0r = np.radians(lat0)
    dlat = np.radians(lat - lat0); dlon = np.radians(lon - lon0)
    x = R_earth * dlon * np.cos(lat0r); y = R_earth * dlat
    return np.array([x, y])
def xy_to_ll(lat0, lon0, x, y):
    lat0r = math.radians(lat0)
    lat = lat0 + math.degrees(y / R_earth)
    lon = lon0 + math.degrees(x / (R_earth * math.cos(lat0r)))
    return float(lat), float(lon)

#TX_lat = 37.79585318435206; TX_lon = -122.3999456223109 #Alcoa
TX_lat = 37.7251353308826
TX_lon = -122.45014631980361  #CCSF
c = 299_792_458.0; f = 14.0574e6; lam = c/f; k = 2*np.pi/lam
Lq = lam/4.0; h_roof = 0.0; h_base = h_roof + 0.3048; h_top = h_base + Lq

# --- Ground plane and material (horizontal plane) ---
ground_z = 0.0                            # plane z = constant (meters), your code already uses h_roof=0, h_base≈0.305
GROUND_EPS_R = 12.0                        # relative permittivity (typical soil 5–15)
GROUND_SIGMA = 0.01                        # conductivity in S/m (0.001–0.05 common)
eps0 = 8.854e-12
omega = 2*np.pi*f
eps_r_ground_complex = GROUND_EPS_R - 1j*GROUND_SIGMA/(omega*eps0)



def tx_segments(nseg=19):
    z = np.linspace(h_base, h_top, nseg)
    I = np.sin(k*(h_top - z)); I = I/np.sum(np.abs(I))
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1), I
TX_segs, TX_I = tx_segments(19)
TX_img_segs = TX_segs.copy()
TX_img_segs[:, 2] = 2.0*ground_z - TX_img_segs[:, 2]

def ll2xy(lat, lon): return ll_to_xy(TX_lat, TX_lon, lat, lon)


# North 67 m building
SW_south = (37.79599248586773, -122.40024848039079)
SE_south = (37.796028715764116, -122.3999764434289)
NE_east  = (37.79679074424414, -122.40013691478359)
H_north  = 67.0
SW_xy = ll2xy(*SW_south); SE_xy = ll2xy(*SE_south); NE_xy = ll2xy(*NE_east)
def sample_line_xy(P0, P1, n): t = np.linspace(0,1,n)[:,None]; return P0[None,:] + (P1-P0)[None,:]*t

south_line_xy = sample_line_xy(SW_xy, SE_xy, 31)
east_line_xy  = sample_line_xy(SE_xy, NE_xy, 31)

north_south_top_xyz = np.vstack([south_line_xy[:,0], south_line_xy[:,1], np.full(31, H_north)]).T
north_east_top_xyz  = np.vstack([east_line_xy[:,0],  east_line_xy[:,1],  np.full(31, H_north)]).T
z_vals_north = np.linspace(0.0, H_north, 13)
north_SE_corner_xyz = np.vstack([np.full(13, SE_xy[0]), np.full(13, SE_xy[1]), z_vals_north]).T
balc_z = np.arange(6.0, H_north-0.01, 6.0)
north_balc_south_xyz = np.vstack([np.vstack([south_line_xy[:,0], south_line_xy[:,1], np.full(31, z)]).T for z in balc_z])
north_balc_east_xyz  = np.vstack([np.vstack([east_line_xy[:,0],  east_line_xy[:,1],  np.full(31, z)]).T for z in balc_z])

# One Maritime (east wall)
east_wall_pts_ll = np.array([
    [37.79586161521426, -122.39948023378125],
    [37.79556064852436, -122.39942122901053],
    [37.79530631024343, -122.39935685962034],
])
east_xy = np.array([ll2xy(lat, lon) for lat,lon in east_wall_pts_ll])
def sample_line_xyN(P0,P1,N): t=np.linspace(0,1,N)[:,None]; return P0[None,:]+(P1-P0)[None,:]*t
east_line_xy_OP = sample_line_xyN(east_xy[0], east_xy[-1], 31)
z_vals_e = np.linspace(0.0, 121.0, 13)
OP_wall_xyz = np.vstack([np.vstack([east_line_xy_OP[:,0], east_line_xy_OP[:,1], np.full(31, z)]).T for z in z_vals_e])

# Short E–W roof near TX
y_edge = -5.16; x_east = +4.00; x_west = -17.56
xs_edge = np.linspace(x_west, x_east, 25)
h_roof_edge = h_roof + 4.45
short_roof_edge_xyz = np.vstack([xs_edge, np.full_like(xs_edge, y_edge), np.full_like(xs_edge, h_roof_edge)]).T

# Additional buildings (from supplied CZML)
czml_path = Path("ccsf_additional_buildings.czml")
extra_wall_xyz = np.zeros((0,3)); extra_top_edge_xyz = np.zeros((0,3)); extra_corner_xyz = np.zeros((0,3))
if czml_path.exists():
    doc = json.load(open(czml_path, "r"))
    entities = doc if isinstance(doc, list) else ([doc["elements"]] if isinstance(doc, dict) and "elements" in doc else [doc])
    def positions_to_xy_ring(cartographicDegrees):
        coords = np.array(cartographicDegrees, dtype=float).reshape(-1,3)  # lon,lat,alt
        xy = np.array([ll2xy(lat, lon) for lon, lat, _ in coords])
        return xy
    def sample_wall_from_ring(xy_ring, H, ds=8.0, dz=12.0):
        pts = []
        for i in range(len(xy_ring)-1):
            A = xy_ring[i]; B = xy_ring[i+1]
            L = float(np.linalg.norm(B-A)); 
            if L < 1e-3: continue
            n_along = max(2, int(np.ceil(L/ds)))
            along_xy = sample_line_xy(A, B, n_along)
            z = np.arange(0.0, H+1e-9, dz)
            grid = np.vstack([np.tile(along_xy[:,0], len(z)),
                              np.tile(along_xy[:,1], len(z)),
                              np.repeat(z, along_xy.shape[0])]).T
            pts.append(grid)
        return np.vstack(pts) if pts else np.zeros((0,3))
    def sample_top_edge_from_ring(xy_ring, H, ds=4.0):
        pts = []
        for i in range(len(xy_ring)-1):
            A = xy_ring[i]; B = xy_ring[i+1]
            L = float(np.linalg.norm(B-A)); 
            if L < 1e-3: continue
            n_along = max(2, int(np.ceil(L/ds)))
            along_xy = sample_line_xy(A, B, n_along)
            pts.append(np.vstack([along_xy[:,0], along_xy[:,1], np.full(n_along, H)]).T)
        return np.vstack(pts) if pts else np.zeros((0,3))
    def sample_corners_from_ring(xy_ring, H, dz=12.0):
        z = np.arange(0.0, H+1e-9, dz)
        pts = []
        for i in range(len(xy_ring)-1):
            vx, vy = xy_ring[i]
            pts.append(np.vstack([np.full_like(z, vx), np.full_like(z, vy), z]).T)
        return np.vstack(pts) if pts else np.zeros((0,3))

    for ent in entities:
        poly = ent.get("polygon")
        if not poly or "positions" not in poly: 
            continue
        pos = poly["positions"].get("cartographicDegrees")
        if not pos: continue
        xy_ring = positions_to_xy_ring(pos)
        H = float(poly.get("extrudedHeight", 0.0))
        if H <= 0: H = float(ent.get("properties", {}).get("extrudedHeight", 0.0))
        if H <= 0: continue
        extra_wall_xyz = np.vstack([extra_wall_xyz, sample_wall_from_ring(xy_ring, H)])
        extra_top_edge_xyz = np.vstack([extra_top_edge_xyz, sample_top_edge_from_ring(xy_ring, H)])
        extra_corner_xyz = np.vstack([extra_corner_xyz, sample_corners_from_ring(xy_ring, H)])

# EM: Fresnel coefficient for walls (parallel pol. proxy)
eps0 = 8.854e-12; omega = 2*np.pi*f
eps_r_complex = 6.0 - 1j*(0.03)/(omega*eps0)
def fresnel_gamma_parallel(theta_i, eps_r_complex):
    sin_t = np.sin(theta_i); cos_t = np.cos(theta_i) + 1e-9
    root = np.sqrt(eps_r_complex - sin_t**2)
    return (eps_r_complex*cos_t - root)/(eps_r_complex*cos_t + root)

def tx_incident_to_pts(pts_xyz):
    if pts_xyz.size == 0: return np.zeros(0, dtype=complex)
    Ein = np.zeros(pts_xyz.shape[0], dtype=complex)
    for (sx,sy,sz), w in zip(TX_segs, TX_I):
        R = np.sqrt(np.sum((pts_xyz - np.array([sx,sy,sz]))**2, axis=1)); R = np.maximum(R, 0.1)
        Ein += w * np.exp(-1j*k*R)/R
    return Ein

ez = np.array([0.0, 0.0, 1.0])  # antenna axis

def sin_theta_from(seg_xyz, obs):
    rv   = obs - seg_xyz                 # vector segment -> observer
    R    = np.linalg.norm(rv, axis=1)
    rhat = rv / (R[:,None] + 1e-12)
    cosT = np.abs(rhat @ ez)             # |cos(theta)| about the z-axis
    return np.sqrt(np.maximum(0.0, 1.0 - cosT**2)), np.maximum(R, 0.1)

# Perpendicular (⊥) and parallel (∥) Fresnel reflection coeffs
def fresnel_gamma_perp(theta_i, eps_r_complex):
    sin_t = np.sin(theta_i); cos_t = np.cos(theta_i) + 1e-12
    root = np.sqrt(eps_r_complex - sin_t**2)
    return (cos_t - root) / (cos_t + root)
    


if INCLUDE["bleachers"]:
    # ====== NEW: bring in elevation-driven bleacher & sloping-ground patches ======
    # Requirements: /mnt/data/czml_elev_fetch.py and /mnt/data/bleacher_patch_from_elev.py
    # The fetcher will:
    #   • read a CZML of polylines, skip ids that contain "f2"
    #   • sample the first maxFeet along each polyline
    #   • call USGS EPQS (feet) and return elevation packets
    #   • fit planes for (inside bleacher polygon) and (outside background ground)
    #   • return ENU point clouds wrt TX: x_east,y_north,z_up  [meters]
    
    
    # ---- 1) define inputs ----
    # (a) CZML with the polylines you want to sample for elevation
    paths_czml = "./qso_elev.czml"   # <<— set this to your file with non-"f2" polylines
    
    # (b) bleacher polygon (lat,lon) — order CW or CCW; four corners is fine
    BLEACHER_POLY_LATLON = [
        (37.725248486078605, -122.44917105079513),
        (37.72524742362892, -122.44933088645813),
        (37.72617867285204, -122.44933990648872),
        (37.72618127312993, -122.44917837709382),
    ]
    
    # (c) how far & how dense to sample along each path
    EPQS_MAX_FEET = 600.0
    EPQS_COUNT    = 100
    
    # ---- 2) fetch & build patches ----
    try:
        from czml_elev_fetch import build_patches_from_elev
    except Exception as e:
        raise RuntimeError("czml_elev_fetch.py must be on PYTHONPATH (or next to this script).") from e
    
    _bleach = build_patches_from_elev(
        czml_path=paths_czml,
        tx_lat=TX_lat, tx_lon=TX_lon,
        bleacher_corners_latlon=BLEACHER_POLY_LATLON,
        maxFeet=EPQS_MAX_FEET, count=EPQS_COUNT,
        out_prefix=None,         # set a string to also save .npy/.csv
        feet_to_m=True
    )
    
    bleacher_xyz = _bleach["meta"]["bleacher_xyz"]     # ENU meters (N×3)
    ground_xyz   = _bleach["meta"]["ground_xyz"]       # ENU meters (M×3)
    
    # Plane fits z = a x + b y + c (meters) for local surface normals
    bleacher_plane = _bleach["meta"]["bleacher_plane"]  # (a,b,c)
    ground_plane   = _bleach["meta"]["ground_plane"]    # (a,b,c)
    
    # ---- 3) incident field at those points (from your short dipole model) ----
    Ein_bleacher = tx_incident_to_pts(bleacher_xyz)
    Ein_ground   = tx_incident_to_pts(ground_xyz)
    
    # ---- 4) Fresnel reflection factors for sloped surfaces ----
    # For a plane z = a x + b y + c, an (unnormalized) upward normal is n0 = [-a,-b,1]
    def normal_from_plane(plane_abc):
        a,b,_ = plane_abc
        n = np.array([-a, -b, 1.0], dtype=float)
        return n / (np.linalg.norm(n) + 1e-12)
    
    n_bleacher = normal_from_plane(bleacher_plane)
    n_ground   = normal_from_plane(ground_plane)
    
    def fresnel_gamma_parallel(theta_i, eps_r_complex):
        sin_t = np.sin(theta_i); cos_t = np.cos(theta_i) + 1e-12
        root = np.sqrt(eps_r_complex - sin_t**2)
        return (eps_r_complex*cos_t - root) / (eps_r_complex*cos_t + root)
    
    # Compute incidence angles (TX → patch) vs each plane’s normal
    def incidence_angles_to_plane(pts_xyz, plane_normal):
        if pts_xyz.size == 0:
            return np.zeros(0)
        # direction from patch point to TX feedpoint (your h_base)
        v = np.array([0.0, 0.0, h_base])[None,:] - pts_xyz
        u = v / (np.linalg.norm(v, axis=1)[:,None] + 1e-12)
        cos_theta = np.abs(u @ plane_normal)  # |u·n|
        cos_theta = np.clip(cos_theta, 0, 1)
        return np.arccos(cos_theta)
    
    # Materials: metal bleachers (high-σ), soil for background ground
    eps0 = 8.854e-12; omega = 2*np.pi*f
    # “Metal” ≈ very good conductor → |Γ|≈1; keep a slight loss so it’s well-behaved
    epsr_bleacher = 1.0 - 1j*(3.0e6)/(omega*eps0)   # very high σ proxy
    epsr_ground   = 12.0 - 1j*(0.01)/(omega*eps0)   # soil
    
    theta_bleacher = incidence_angles_to_plane(bleacher_xyz, n_bleacher)
    theta_ground   = incidence_angles_to_plane(ground_xyz,   n_ground)
    
    # For vertical-dipole illumination on a (roughly) horizontal surface, use Γ⊥ as the better proxy.
    Gamma_bleacher = fresnel_gamma_perp(theta_bleacher, epsr_bleacher)
    Gamma_ground   = fresnel_gamma_perp(theta_ground,   epsr_ground)
    
    # Optional global scaling so dense clouds don't overwhelm the sum
    A_bleacher_total_dB = -12.0
    A_ground_total_dB   = -18.0
    def per_point_scale(total_dB, npts):
        return (10**(total_dB/20.0))/max(1, npts)
    
    A_bleacher_per = per_point_scale(A_bleacher_total_dB, bleacher_xyz.shape[0])
    A_ground_per   = per_point_scale(A_ground_total_dB,   ground_xyz.shape[0])
    
    # Pre-multiply like other “Ein-based” scatterers
    Ein_bleacher_scaled = A_bleacher_per * Gamma_bleacher * Ein_bleacher
    Ein_ground_scaled   = A_ground_per   * Gamma_ground   * Ein_ground
    # ====== end NEW block ======





def wall_reflection_terms(wall_xyz):
    if wall_xyz.size == 0: return np.zeros(0, dtype=complex)
    vec_to_tx = (np.array([0.0,0.0,h_base]) - wall_xyz)
    u_inc = vec_to_tx / (np.linalg.norm(vec_to_tx, axis=1)[:,None] + 1e-12)
    n_xy = u_inc[:,:2]; n_norm = np.linalg.norm(n_xy, axis=1) + 1e-12
    n_xy = (n_xy.T / n_norm).T
    wall_normal_3d = np.vstack([n_xy[:,0], n_xy[:,1], np.zeros_like(n_xy[:,0])]).T
    cos_theta = np.clip(np.abs(np.sum(u_inc * wall_normal_3d, axis=1)), 0, 1)
    theta_i = np.arccos(cos_theta)
    return fresnel_gamma_parallel(theta_i, eps_r_complex)

Ein_OP   = tx_incident_to_pts(OP_wall_xyz)
Ein_topS = tx_incident_to_pts(north_south_top_xyz)
Ein_topE = tx_incident_to_pts(north_east_top_xyz)
Ein_bS   = tx_incident_to_pts(north_balc_south_xyz)
Ein_bE   = tx_incident_to_pts(north_balc_east_xyz)
Ein_roof = tx_incident_to_pts(short_roof_edge_xyz)
Ein_extra_walls = tx_incident_to_pts(extra_wall_xyz)
Ein_extra_tops  = tx_incident_to_pts(extra_top_edge_xyz)
Ein_extra_corn  = tx_incident_to_pts(extra_corner_xyz)
Gamma_OP = wall_reflection_terms(OP_wall_xyz)
Gamma_extra = wall_reflection_terms(extra_wall_xyz)

# Amplitude scalars (same family as before)
A_corner_total = 10**(-23/20.0)
A_roof_total   = 10**(-9/20.0)
A_top_total    = 10**(-15/20.0)
A_balc_total   = 10**(-18/20.0)

def per_point_scale(total_amp, npts): return (total_amp / max(1, npts)) if npts>0 else 0.0
A_corner_per = per_point_scale(A_corner_total, north_SE_corner_xyz.shape[0])
A_roof_per   = per_point_scale(A_roof_total, short_roof_edge_xyz.shape[0])
A_topS_per   = per_point_scale(A_top_total, north_south_top_xyz.shape[0])
A_topE_per   = per_point_scale(A_top_total, north_east_top_xyz.shape[0])
A_bS_per     = per_point_scale(A_balc_total, north_balc_south_xyz.shape[0])
A_bE_per     = per_point_scale(A_balc_total, north_balc_east_xyz.shape[0])
A_top_extra_per  = per_point_scale(A_top_total, extra_top_edge_xyz.shape[0])
A_corner_extra_per = per_point_scale(A_corner_total, extra_corner_xyz.shape[0])

# Wedge weight for the SE corner (exterior 270°, N=2/3)
corner_xy = SE_xy.copy()
def wedge_weight_for_bearing_angles(obs_xy):
    angles = (np.degrees(np.mod(np.arctan2(obs_xy[:,0], obs_xy[:,1]), 2*np.pi)))
    Wb = np.empty_like(angles)
    v_inc = -corner_xy; phi_i = np.arctan2(v_inc[1], v_inc[0])
    Nw = 2.0/3.0; eps = 1e-6
    for idx, a in enumerate(angles):
        v_obs = np.array([np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))])
        phi_o = np.arctan2(v_obs[1], v_obs[0])
        dphi = (phi_o - phi_i + np.pi) % (2*np.pi) - np.pi
        term1 = 1.0/np.tan((np.pi + dphi)/(2*Nw) + eps)
        term2 = 1.0/np.tan((np.pi - dphi)/(2*Nw) + eps)
        Wb[idx] = min(50.0, abs(term1)+abs(term2))
    return Wb

def field_at_points_batched(obs_xyz, batch=10000):
    N = obs_xyz.shape[0]; Etot = np.zeros(N, dtype=complex)
    def add_scatterers(Et, obs, P, amp_or_Ein, is_Ein=False, wedge_weight=False):
        if P.size == 0: return Et
        R = np.sqrt(((obs[:,0,None]-P[None,:,0])**2 + (obs[:,1,None]-P[None,:,1])**2 + (obs[:,2,None]-P[None,:,2])**2)); R = np.maximum(R, 0.1)
        contrib = (amp_or_Ein[None,:] if is_Ein else (amp_or_Ein)) * np.exp(-1j*k*R)/R
        Et_add = np.sum(contrib, axis=1)
        if wedge_weight:
            Wb = wedge_weight_for_bearing_angles(obs[:,:2])
            Et_add = Et_add * Wb
        return Et + Et_add
    for i0 in range(0, N, batch):
        i1 = min(N, i0+batch); obs = obs_xyz[i0:i1]
        Et = np.zeros(obs.shape[0], dtype=complex)
        # Direct
        # ---- Direct field (with element pattern) ----
        for (sx,sy,sz), w in zip(TX_segs, TX_I):
            sinT, R = sin_theta_from(np.array([sx,sy,sz])[None,:], obs)
            Et += w * sinT * np.exp(-1j*k*R)/R
        
        # ---- Flat ground image (if enabled) ----
        if INCLUDE.get("flat_ground", False):
            n = np.array([0.0, 0.0, 1.0])                 # plane normal (horizontal ground)
            for (ix,iy,iz), w in zip(TX_img_segs, TX_I):
                rv   = obs - np.array([ix,iy,iz])
                Rimg = np.linalg.norm(rv, axis=1); Rimg = np.maximum(Rimg, 0.1)
                rhat = rv / (Rimg[:,None] + 1e-12)
        
                # element factor for the image (also vertical)
                sinT_img = np.sqrt(np.maximum(0.0, 1.0 - (np.abs(rhat @ ez))**2))
        
                # incidence angle for Fresnel Γ⊥ (angle from plane normal)
                cos_theta_i = np.clip(np.abs(rhat @ n), 0.0, 1.0)
                theta_i     = np.arccos(cos_theta_i)
                Gamma       = fresnel_gamma_perp(theta_i, eps_r_ground_complex)
        
                Et += w * Gamma * sinT_img * np.exp(-1j*k*Rimg)/Rimg


        # Scatterers
        if INCLUDE["north"]:
            Et = add_scatterers(Et, obs, north_SE_corner_xyz,  A_corner_per, is_Ein=False, wedge_weight=True)
            Et = add_scatterers(Et, obs, north_south_top_xyz, A_topS_per * Ein_topS, is_Ein=True)
            Et = add_scatterers(Et, obs, north_east_top_xyz,  A_topE_per * Ein_topE, is_Ein=True)
            Et = add_scatterers(Et, obs, north_balc_south_xyz, A_bS_per * Ein_bS, is_Ein=True)
            Et = add_scatterers(Et, obs, north_balc_east_xyz,  A_bE_per * Ein_bE, is_Ein=True)

        if INCLUDE["short_roof"]:
            Et = add_scatterers(Et, obs, short_roof_edge_xyz, A_roof_per, is_Ein=False)
        if INCLUDE["alcoa"]:
            Et = add_scatterers(Et, obs, OP_wall_xyz, Gamma_OP * Ein_OP, is_Ein=True)
        #Et = add_scatterers(Et, obs, extra_wall_xyz, Gamma_extra * Ein_extra_walls, is_Ein=True)
        #Et = add_scatterers(Et, obs, extra_top_edge_xyz, A_top_extra_per * Ein_extra_tops, is_Ein=True)
        #Et = add_scatterers(Et, obs, extra_corner_xyz, A_corner_extra_per * Ein_extra_corn, is_Ein=True)
        # --- NEW: bleacher and sloping ground patches (toggle with INCLUDE["bleachers"]) ---
        if INCLUDE["bleachers"]:
            print("Adding bleachers")
            Et = add_scatterers(Et, obs, bleacher_xyz, Ein_bleacher_scaled, is_Ein=True)
            Et = add_scatterers(Et, obs, ground_xyz,   Ein_ground_scaled,   is_Ein=True)
            print("Done with bleachers")

            print("\n[bleacher debug]")
            print("bleacher_xyz shape:", bleacher_xyz.shape, " ground_xyz shape:", ground_xyz.shape)
            print("inside/outside counts (if available):",
                getattr(_bleach, "get", lambda *_: {})("meta", {}).get("inside_count", "n/a"),
                getattr(_bleach, "get", lambda *_: {})("meta", {}).get("outside_count", "n/a"))
            print("A_bleacher_per =", A_bleacher_per, " |Gamma_bleacher| median =", np.median(np.abs(Gamma_bleacher)) if len(Gamma_bleacher) else "n/a")
            print("|Ein_bleacher| median:", np.median(np.abs(Ein_bleacher)) if len(Ein_bleacher) else "n/a")



        Etot[i0:i1] = Et
    return Etot

# ---------- Build the shells ----------
def build_shell_czml(radius_m, azi_start_deg=30.0, azi_end_deg=130.0, elev_max_deg=90.0,
                     d_azi=2.0, d_elev=3.0, pixel=6):
    # Azimuth set (wrap 315→360 and 0→100)
    #az1 = np.arange(azi_start_deg, 360.0, d_azi)
    az2 = np.arange(azi_start_deg, azi_end_deg + 1e-6, d_azi)
    az_all = np.concatenate([az2])
    el_all = np.arange(0.0, elev_max_deg + 1e-6, d_elev)

    # Build observation points on a spherical shell centered at TX
    b = np.deg2rad(az_all)     # bearing
    e = np.deg2rad(el_all)     # elevation
    B, E = np.meshgrid(b, e)   # shape (Ne, Na)

    # Direction cosines; note our convention: X=east, Y=north, Z=up; 0° az = +Y (north)
    ux = np.cos(E)*np.sin(B); uy = np.cos(E)*np.cos(B); uz = np.sin(E)
    X = radius_m * ux; Y = radius_m * uy; Z = h_base + radius_m * uz

    obs_xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    Efield = field_at_points_batched(obs_xyz, batch=12000).reshape(X.shape)

    # Relative level (dB) referenced to median across this shell
    Pr = np.abs(Efield)
    ref = np.median(Pr); Pr_dB = 20*np.log10(Pr/ref + 1e-12)
    vmin = float(np.percentile(Pr_dB, 5)); vmax = float(np.percentile(Pr_dB, 95))

    # Build CZML with points colored by dB
    def map_color_db(db):
        # viridis approx; map to 0..1
        t = (db - vmin) / max(1e-9, (vmax - vmin)); t = float(np.clip(t, 0.0, 1.0))
        # simple viridis subset
        import matplotlib.cm as cm
        r,g,b, a = cm.get_cmap("viridis")(t)
        return [int(r*255), int(g*255), int(b*255), 220]
    #shel_labeel = "m (az 315→100°, elev 0→{int(elev_max_deg)}°)"
    shell_label = "iso_gnd"
    czml = [{
        "id":"document",
        "name": f"3D shell {int(radius_m)} {shell_label}",
        "version":"1.0"
    },{
        "id":"TX",
        "name":"TX Site",
        "position":{"cartographicDegrees":[TX_lon, TX_lat, h_base]},
        "point":{"color":{"rgba":[255,255,255,255]},"pixelSize":10}
    }]

    # Populate shell points (thinned to keep file size reasonable)
    for ie in range(E.shape[0]):
        for ia in range(E.shape[1]):
            x = X[ie, ia]; y = Y[ie, ia]; z = Z[ie, ia]
            lat, lon = xy_to_ll(TX_lat, TX_lon, x, y)
            db = float(Pr_dB[ie, ia])
            czml.append({
                "id": f"rel_dB_{db}_{int(radius_m)}_e{ie:03d}_a{ia:03d}",
                "position": {"cartographicDegrees": [lon, lat, z]},
                "point": {"color":{"rgba": map_color_db(db)},"pixelSize": pixel}
            })
    #flabel = "m_az315to100_elev0to90.czml"
    flabel = "iso_gnd.czml"
    out_path = f"ccsf_field_shell_{int(radius_m)}{flabel}"
    with open(out_path, "w") as f: json.dump(czml, f)
    return out_path

paths = []
for R in (250.0, 500.0, 1000.0):
    paths.append(build_shell_czml(R, pixel=6 if R<=500 else 5))

paths
