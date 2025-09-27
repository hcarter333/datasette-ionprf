# Build 3‑D field shells (CZML point clouds) up to 90° elevation for an azimuth sector 315°→360°→100°,
# at radii 250 m, 500 m, and 1000 m, using the current modeling geometry (TX, North building, Alcoa wall,
# short roof edge) plus any additional buildings supplied in /mnt/data/field_wedge_additional_buildings.czml.

import json, math, numpy as np
from pathlib import Path

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

TX_lat = 37.79585318435206; TX_lon = -122.3999456223109
c = 299_792_458.0; f = 14.0574e6; lam = c/f; k = 2*np.pi/lam
Lq = lam/4.0; h_roof = 8.9; h_base = h_roof + 0.3048; h_top = h_base + Lq

def tx_segments(nseg=19):
    z = np.linspace(h_base, h_top, nseg)
    I = np.sin(k*(h_top - z)); I = I/np.sum(np.abs(I))
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1), I
TX_segs, TX_I = tx_segments(19)

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
czml_path = Path("field_wedge_additional_buildings.czml")
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
        for (sx,sy,sz), w in zip(TX_segs, TX_I):
            R = np.sqrt(np.sum((obs - np.array([sx,sy,sz]))**2, axis=1)); R = np.maximum(R, 0.1)
            Et += w * np.exp(-1j*k*R)/R
        # Scatterers
        Et = add_scatterers(Et, obs, north_SE_corner_xyz,  A_corner_per, is_Ein=False, wedge_weight=True)
        Et = add_scatterers(Et, obs, north_south_top_xyz, A_topS_per * Ein_topS, is_Ein=True)
        Et = add_scatterers(Et, obs, north_east_top_xyz,  A_topE_per * Ein_topE, is_Ein=True)
        Et = add_scatterers(Et, obs, north_balc_south_xyz, A_bS_per * Ein_bS, is_Ein=True)
        Et = add_scatterers(Et, obs, north_balc_east_xyz,  A_bE_per * Ein_bE, is_Ein=True)
        Et = add_scatterers(Et, obs, short_roof_edge_xyz, A_roof_per, is_Ein=False)
        Et = add_scatterers(Et, obs, OP_wall_xyz, Gamma_OP * Ein_OP, is_Ein=True)
        Et = add_scatterers(Et, obs, extra_wall_xyz, Gamma_extra * Ein_extra_walls, is_Ein=True)
        Et = add_scatterers(Et, obs, extra_top_edge_xyz, A_top_extra_per * Ein_extra_tops, is_Ein=True)
        Et = add_scatterers(Et, obs, extra_corner_xyz, A_corner_extra_per * Ein_extra_corn, is_Ein=True)
        Etot[i0:i1] = Et
    return Etot

# ---------- Build the shells ----------
def build_shell_czml(radius_m, azi_start_deg=315.0, azi_end_deg=100.0, elev_max_deg=90.0,
                     d_azi=2.0, d_elev=3.0, pixel=6):
    # Azimuth set (wrap 315→360 and 0→100)
    az1 = np.arange(azi_start_deg, 360.0, d_azi)
    az2 = np.arange(0.0, azi_end_deg + 1e-6, d_azi)
    az_all = np.concatenate([az1, az2])
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

    czml = [{
        "id":"document",
        "name": f"3D shell {int(radius_m)} m (az 315→100°, elev 0→{int(elev_max_deg)}°)",
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
                "id": f"rel_dB_{db}_{int(radius_m)}",
                "position": {"cartographicDegrees": [lon, lat, z]},
                "point": {"color":{"rgba": map_color_db(db)},"pixelSize": pixel}
            })

    out_path = f"field_shell_{int(radius_m)}m_az315to100_elev0to90.czml"
    with open(out_path, "w") as f: json.dump(czml, f)
    return out_path

paths = []
for R in (250.0, 500.0, 1000.0):
    paths.append(build_shell_czml(R, pixel=6 if R<=500 else 5))

paths