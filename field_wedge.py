# Re-run the modeled field with the corrected building geometry & wall sampling.
# Outputs:
#  1) Polar sky map (sector 270..360 ∪ 0..45, elevations 0..90) as PNG
#  2) CZML shell wedges at 250 m and 500 m radii, using the modeled field & same color scale
#
# Assumptions (as before):
#  - 14.0574 MHz quarter-wave vertical, sinusoidal current along the wire
#  - Short E–W roof edge near TX as local scatterer
#  - North building (67 m): SE corner vertical (wedge/diffraction proxy), top edges at 67 m,
#    balcony edges every 3 m on south & east façades as re-radiators with small amplitude
#  - East wall (One Maritime, 121 m): Fresnel reflection (parallel pol.), mild-loss dielectric
#
# NOTE: This reuses/updates the exact-field approach used in previous steps, now with corrected
# façade sampling (points lie ON the façade lines).

import json, math, numpy as np
import matplotlib.pyplot as plt

# ------------- Helpers -------------
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
    return lat, lon

def bearing_initial(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def unit_from_bearing_elev(bearing_deg, elev_deg):
    b = np.deg2rad(bearing_deg % 360.0); e = np.deg2rad(np.clip(elev_deg, 0, 90))
    return np.array([np.cos(e)*np.sin(b), np.cos(e)*np.cos(b), np.sin(e)])  # ENU

# Fresnel Γ∥
def fresnel_gamma_parallel(theta_i, eps_r_complex):
    sin_t = np.sin(theta_i); cos_t = np.cos(theta_i) + 1e-9
    root = np.sqrt(eps_r_complex - sin_t**2)
    return (eps_r_complex*cos_t - root)/(eps_r_complex*cos_t + root)

# ------------- TX & constants -------------
TX_lat = 37.79585318435206
TX_lon = -122.3999456223109
c = 299_792_458.0; f = 14.0574e6; lam = c/f; k = 2*np.pi/lam
Lq = lam/4.0
h_roof = 8.9; h_base = h_roof + 0.3048; h_top = h_base + Lq

def tx_segments(nseg=25):
    z = np.linspace(h_base, h_top, nseg)
    I = np.sin(k*(h_top - z)); I = I/np.sum(np.abs(I))
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1), I

TX_segs, TX_I = tx_segments(25)

# ------------- Geometry (corrected) -------------
# North 67 m building: given corners
SW_south = (37.79599248586773, -122.40024848039079)
SE_south = (37.796028715764116, -122.3999764434289)   # SE corner (closest to TX)
NE_east  = (37.79679074424414, -122.40013691478359)
H_north  = 67.0

def ll2xy(lat, lon): return ll_to_xy(TX_lat, TX_lon, lat, lon)
SW_xy = ll2xy(*SW_south); SE_xy = ll2xy(*SE_south); NE_xy = ll2xy(*NE_east)

num_len = 61; t = np.linspace(0.0, 1.0, num_len)
south_line_xy = SW_xy[None,:] + (SE_xy - SW_xy)[None,:] * t[:,None]
east_line_xy  = SE_xy[None,:] + (NE_xy - SE_xy)[None,:] * t[:,None]

z_vals_north = np.linspace(0.0, H_north, 23)
balc_z = np.arange(3.0, H_north-0.01, 3.0)

def line_z_grid(line_xy, z_vals):
    N = line_xy.shape[0]; M = z_vals.shape[0]
    XX = np.repeat(line_xy[:,0], M)
    YY = np.repeat(line_xy[:,1], M)
    ZZ = np.tile(z_vals, N)
    return np.stack([XX, YY, ZZ], axis=-1)

north_south_wall_xyz = line_z_grid(south_line_xy, z_vals_north)
north_east_wall_xyz  = line_z_grid(east_line_xy,  z_vals_north)

north_south_top_xyz = np.vstack([south_line_xy[:,0], south_line_xy[:,1], np.full(num_len, H_north)]).T
north_east_top_xyz  = np.vstack([east_line_xy[:,0],  east_line_xy[:,1],  np.full(num_len, H_north)]).T

north_SE_corner_xyz = np.vstack([np.full(z_vals_north.shape[0], SE_xy[0]),
                                 np.full(z_vals_north.shape[0], SE_xy[1]),
                                 z_vals_north]).T

def balcony_edges_from_line(line_xy, zz):
    edges = []
    for z in zz:
        edges.append(np.vstack([line_xy[:,0], line_xy[:,1], np.full(line_xy.shape[0], z)]).T)
    return np.vstack(edges) if edges else np.empty((0,3))

north_balc_south_xyz = balcony_edges_from_line(south_line_xy, balc_z)
north_balc_east_xyz  = balcony_edges_from_line(east_line_xy,  balc_z)

# East 121 m wall (One Maritime) from three points
east_wall_pts_ll = np.array([
    [37.79586161521426, -122.39948023378125],
    [37.79556064852436, -122.39942122901053],
    [37.79530631024343, -122.39935685962034],
])
east_xy = np.array([ll2xy(lat, lon) for lat,lon in east_wall_pts_ll])
p0, p2 = east_xy[0], east_xy[-1]
num_len_e = 41; t_e = np.linspace(0.0, 1.0, num_len_e)
east_line_xy_OP = p0[None,:] + (p2 - p0)[None,:] * t_e[:,None]
z_vals_e = np.linspace(0.0, 121.0, 21)

OP_wall_xyz = line_z_grid(east_line_xy_OP, z_vals_e)

# Seismic lattice sample points (approximate)
u_wall = (p2 - p0); u_wall = u_wall/np.linalg.norm(u_wall)
span_len = float(np.linalg.norm(p2 - p0))
diag_span_s = 15.24; diag_span_z = 17.5; num_bays = 5
diag_pts_list = []
bay_centers_s = np.linspace(0.1*span_len, 0.9*span_len, num_bays)
for i, sc in enumerate(bay_centers_s):
    center_xy = p0 + u_wall*sc
    start_xy = center_xy - u_wall*(diag_span_s/2.0)
    end_xy   = center_xy + u_wall*(diag_span_s/2.0)
    z0 = 10.0; z1 = z0 + (diag_span_z if (i%2==0) else -diag_span_z)
    xs = np.linspace(start_xy[0], end_xy[0], 25)
    ys = np.linspace(start_xy[1], end_xy[1], 25)
    zs = np.linspace(z0, z1, 25)
    diag_pts_list.append(np.vstack([xs, ys, zs]).T)
OP_lattice_xyz = np.vstack(diag_pts_list)

# Short E–W roof edge near TX
y_edge = -5.16; x_east = +4.00; x_west = -17.56
xs_edge = np.linspace(x_west, x_east, 41)
h_roof_edge = h_roof + 4.45
short_roof_edge_xyz = np.vstack([xs_edge, np.full_like(xs_edge, y_edge), np.full_like(xs_edge, h_roof_edge)]).T

# ------------- Amplitude weights -------------
A_corner_total = 10**(-23/20.0)
A_roof_total   = 10**(-9/20.0)
A_top_total    = 10**(-15/20.0)
A_balc_total   = 10**(-18/20.0)
A_lat_total    = 10**(-15/20.0)

# ------------- Incident fields & Γ -------------
def tx_incident_to_pts(pts_xyz):
    Ein = np.zeros(pts_xyz.shape[0], dtype=complex)
    for (sx,sy,sz), w in zip(TX_segs, TX_I):
        R = np.sqrt(np.sum((pts_xyz - np.array([sx,sy,sz]))**2, axis=1)); R = np.maximum(R, 0.1)
        Ein += w * np.exp(-1j*k*R)/R
    return Ein

Ein_OP   = tx_incident_to_pts(OP_wall_xyz)
Ein_topS = tx_incident_to_pts(north_south_top_xyz)
Ein_topE = tx_incident_to_pts(north_east_top_xyz)
Ein_bS   = tx_incident_to_pts(north_balc_south_xyz)
Ein_bE   = tx_incident_to_pts(north_balc_east_xyz)
Ein_roof = tx_incident_to_pts(short_roof_edge_xyz)
Ein_lat  = tx_incident_to_pts(OP_lattice_xyz)

# Fresnel Γ for OP wall (parallel pol) toward TX
eps0 = 8.854e-12; omega = 2*np.pi*f
eps_r_complex = 6.0 - 1j*(0.03)/(omega*eps0)
# Approx wall normal: horizontal, pointing toward the TX
# Use local tangent at mid-point to find the outward normal (rotate -90°)
t_wall = (east_line_xy_OP[-1] - east_line_xy_OP[0]); t_wall /= np.linalg.norm(t_wall)
n_wall2d = np.array([-t_wall[1], t_wall[0]])  # left-hand normal
# Ensure it faces the TX
mid_xy = 0.5*(east_line_xy_OP[0] + east_line_xy_OP[-1])
to_tx = -mid_xy
if np.dot(n_wall2d, to_tx) < 0: n_wall2d = -n_wall2d
wall_normal_3d = np.array([n_wall2d[0], n_wall2d[1], 0.0])

vec_tx_to_wall = OP_wall_xyz - np.array([0.0, 0.0, h_base])
u_inc = vec_tx_to_wall / (np.linalg.norm(vec_tx_to_wall, axis=1)[:,None] + 1e-12)
cos_theta = np.clip(np.abs(np.sum(u_inc * wall_normal_3d[None,:], axis=1)), 0, 1)
theta_i = np.arccos(cos_theta)
Gamma_OP = fresnel_gamma_parallel(theta_i, eps_r_complex)

# ------------- Wedge weighting for SE corner -------------
# Use the SE corner XY (closest to TX) as wedge location
corner_xy = SE_xy.copy()

def wedge_weight_for_bearing(bearing_deg):
    # Simple UTD-like azimuthal taper vs outgoing bearing
    v_inc = -corner_xy
    phi_i = np.arctan2(v_inc[1], v_inc[0])
    v_obs = np.array([np.sin(np.deg2rad(bearing_deg)), np.cos(np.deg2rad(bearing_deg))])
    phi_o = np.arctan2(v_obs[1], v_obs[0])
    dphi = (phi_o - phi_i + np.pi) % (2*np.pi) - np.pi
    Nw = 2.0/3.0; eps = 1e-6
    term1 = 1.0/np.tan((np.pi + dphi)/(2*Nw) + eps)
    term2 = 1.0/np.tan((np.pi - dphi)/(2*Nw) + eps)
    return np.clip(abs(term1) + abs(term2), 0, 50.0)

# Distribute corner amplitude per sample along its vertical line
corner_pts_xyz = np.vstack([np.full(23, corner_xy[0]), np.full(23, corner_xy[1]), np.linspace(0.0, H_north, 23)]).T
A_corner_per = A_corner_total / corner_pts_xyz.shape[0]
A_roof_per   = A_roof_total / short_roof_edge_xyz.shape[0]
A_topS_per   = A_top_total / north_south_top_xyz.shape[0]
A_topE_per   = A_top_total / north_east_top_xyz.shape[0]
A_bS_per     = A_balc_total / max(1, north_balc_south_xyz.shape[0])
A_bE_per     = A_balc_total / max(1, north_balc_east_xyz.shape[0])
A_lat_per    = A_lat_total / OP_lattice_xyz.shape[0]

# ------------- Field synthesis -------------
def field_at_points(obs_xyz):
    Etot = np.zeros(obs_xyz.shape[0], dtype=complex)
    # Direct
    for (sx,sy,sz), w in zip(TX_segs, TX_I):
        R = np.sqrt(np.sum((obs_xyz - np.array([sx,sy,sz]))**2, axis=1)); R = np.maximum(R, 0.1)
        Etot += w * np.exp(-1j*k*R)/R
    # Corner diffraction proxy
    angles = (np.degrees(np.mod(np.arctan2(obs_xyz[:,0], obs_xyz[:,1]), 2*np.pi)))
    Wb = np.array([wedge_weight_for_bearing(a) for a in angles])
    R = np.sqrt((obs_xyz[:,0,None]-corner_pts_xyz[None,:,0])**2 +
                (obs_xyz[:,1,None]-corner_pts_xyz[None,:,1])**2 +
                (obs_xyz[:,2,None]-corner_pts_xyz[None,:,2])**2); R = np.maximum(R, 0.1)
    Etot += np.sum(A_corner_per * np.exp(-1j*k*R)/R, axis=1) * Wb
    # Short roof edge
    R = np.sqrt((obs_xyz[:,0,None]-short_roof_edge_xyz[None,:,0])**2 +
                (obs_xyz[:,1,None]-short_roof_edge_xyz[None,:,1])**2 +
                (obs_xyz[:,2,None]-short_roof_edge_xyz[None,:,2])**2); R = np.maximum(R, 0.1)
    Etot += np.sum(A_roof_per * np.exp(-1j*k*R)/R, axis=1)
    # OP east wall reflection
    P = OP_wall_xyz; Ein = Ein_OP; Gam = Gamma_OP
    R = np.sqrt((obs_xyz[:,0,None]-P[None,:,0])**2 +
                (obs_xyz[:,1,None]-P[None,:,1])**2 +
                (obs_xyz[:,2,None]-P[None,:,2])**2); R = np.maximum(R, 0.1)
    Etot += np.sum(Gam[None,:] * Ein[None,:] * np.exp(-1j*k*R)/R, axis=1)
    # North building roof edges
    for pts, Aper, Ein_top in [(north_south_top_xyz, A_topS_per, Ein_topS),
                               (north_east_top_xyz,  A_topE_per, Ein_topE)]:
        R = np.sqrt((obs_xyz[:,0,None]-pts[None,:,0])**2 +
                    (obs_xyz[:,1,None]-pts[None,:,1])**2 +
                    (obs_xyz[:,2,None]-pts[None,:,2])**2); R = np.maximum(R, 0.1)
        Etot += np.sum(Aper * Ein_top[None,:] * np.exp(-1j*k*R)/R, axis=1)
    # Balconies
    for pts, Aper, Ein_b in [(north_balc_south_xyz, A_bS_per, Ein_bS),
                             (north_balc_east_xyz,  A_bE_per, Ein_bE)]:
        if pts.size > 0:
            R = np.sqrt((obs_xyz[:,0,None]-pts[None,:,0])**2 +
                        (obs_xyz[:,1,None]-pts[None,:,1])**2 +
                        (obs_xyz[:,2,None]-pts[None,:,2])**2); R = np.maximum(R, 0.1)
            Etot += np.sum(Aper * Ein_b[None,:] * np.exp(-1j*k*R)/R, axis=1)
    # Lattice (small)
    P = OP_lattice_xyz; Ein = Ein_lat
    R = np.sqrt((obs_xyz[:,0,None]-P[None,:,0])**2 +
                (obs_xyz[:,1,None]-P[None,:,1])**2 +
                (obs_xyz[:,2,None]-P[None,:,2])**2); R = np.maximum(R, 0.1)
    Etot += np.sum(A_lat_per * Ein[None,:] * np.exp(-1j*k*R)/R, axis=1)
    return Etot

# ------------- Sample sky wedge and build PNG + CZMLs -------------
def build_outputs(R_shell, png_path=None, czml_path=None, add_labels=True, keep_sector=(315, 200)):
    max_el_val = 90.0
    az_vals = np.concatenate([np.arange(keep_sector[0], 360+1e-6, 3.0), np.arange(0, keep_sector[1]+1e-6, 3.0)])
    el_vals = np.arange(0, max_el_val+1e-6, 3.0)
    obs = np.array([unit_from_bearing_elev(a,e)*R_shell for e in el_vals for a in az_vals])
    E = field_at_points(obs)
    amp = np.abs(E).reshape(len(el_vals), len(az_vals))
    ref = np.median(amp[amp>0]); rel_dB = 20*np.log10(amp/ref + 1e-12)

    # Color range from 5th to 95th percentile for visual robustness
    vmin = float(np.percentile(rel_dB, 5)); vmax = float(np.percentile(rel_dB, 95))
    cmap = plt.get_cmap("viridis")

    # Polar sky map (only for the 1000 m shell by default)
    if png_path is not None:
        edges_deg = np.concatenate([[az_vals[0]-1.5], az_vals+1.5])
        TH_edges = np.deg2rad(edges_deg)
        R_edges = np.linspace(max_el_val - el_vals.max(), max_el_val - el_vals.min(), len(el_vals)+1)
        T_edges, R_edges_mesh = np.meshgrid(TH_edges, R_edges)

        fig = plt.figure(figsize=(9.2,9.2)); ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        pcm = ax.pcolormesh(T_edges, R_edges_mesh, rel_dB, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        # Station headings
        targets = {
            "Alaska": (64.8333776, -147.4612627),
            "aa6c (WA)": (47.5706548, -122.2220673),
            "km7anm (WA)": (47.4958153, -122.1764987),
            "wq7o (WA)": (47.8143866, -117.3647516),
            "k0du (CO)": (39.1714443, -108.7239377),
            "k0uu (MN)": (44.9729495, -93.6225131),
        }
        bearings = {name: bearing_initial(TX_lat, TX_lon, lat, lon) for name,(lat,lon) in targets.items()}
        def in_kept(b): return (b >= keep_sector[0]) or (b <= keep_sector[1])
        for name,b in bearings.items():
            if in_kept(b):
                b_cont = b if b >= keep_sector[0] else b + 360.0
                th = np.deg2rad(b_cont)
                ax.plot([th, th], [0, 90], linestyle=':', linewidth=0.9, color='w')
                ax.text(th, 1.6, name, fontsize=9, ha='center', va='bottom', color='w')

        cb = fig.colorbar(pcm, ax=ax, pad=0.08, label="Relative field (dB)")
        ax.set_title(f"Sky map — polar (MODELED field, corrected geometry)\nSector: az {keep_sector[0]}–360 & 0–{keep_sector[1]}; el 0–{max_el_val}; R={R_shell:.0f} m", va='bottom')
        plt.tight_layout(); plt.savefig(png_path, dpi=240); plt.show()

    # CZML shell with points colored by rel_dB
    if czml_path is not None:
        def to_rgba(dB):
            t = (dB - vmin) / max(1e-9, (vmax - vmin))
            r,g,b,_ = cmap(np.clip(t,0,1))
            return [int(r*255), int(g*255), int(b*255), 220]

        czml = [{"id":"document","name":f"Shell wedge R={R_shell:.0f} m (MODELED field, corrected)","version":"1.0"}]
        czml.append({"id":"TX","name":"TX Site",
                     "position":{"cartographicDegrees":[TX_lon, TX_lat, 0.0]},
                     "point":{"color":{"rgba":[255,255,255,255]},"pixelSize":10}})

        for j, el in enumerate(el_vals):
            for i, az in enumerate(az_vals):
                u = unit_from_bearing_elev(az, el)
                x,y,z = (u*R_shell).tolist()
                plat, plon = xy_to_ll(TX_lat, TX_lon, x, y)
                czml.append({
                    "id": f"pt_{j}_{i}",
                    "name": f"az={az:.1f}, el={el:.1f}, dB={rel_dB[j,i]:.2f}",
                    "position": {"cartographicDegrees":[plon, plat, float(z)]},
                    "point": {"pixelSize": 6, "color": {"rgba": to_rgba(rel_dB[j, i])}}
                })

        # boundary radials & elevation arcs
        def polyline_from_samples(samples_xyz, rgba, width=2):
            positions = []
            for (x,y,z) in samples_xyz:
                plat, plon = xy_to_ll(TX_lat, TX_lon, x, y)
                positions += [plon, plat, float(z)]
            return {"polyline":{"width":width, "material":{"solidColor":{"color":{"rgba":rgba}}},
                                "positions":{"cartographicDegrees":positions}, "arcType":"NONE"}}

        for az in [keep_sector[0], keep_sector[1]]:
            seg = [unit_from_bearing_elev(az, e)*R_shell for e in np.linspace(0, 90, 31)]
            czml.append({"id":f"radial_{int(az)}", "name":f"az {az}°"} | polyline_from_samples(seg, [255,255,255,200], 3))

        for elev in [30.0, 60.0, 90.0]:
            arc_pts = []
            for az in np.concatenate([np.linspace(keep_sector[0], 360, 61), np.linspace(0, keep_sector[1], 16)]):
                arc_pts.append(unit_from_bearing_elev(az, elev)*R_shell)
            czml.append({"id":f"arc_el{int(elev)}", "name":f"el {int(elev)}°"} | polyline_from_samples(arc_pts, [200,200,200,180], 2))

        # Station headings
        targets = {
            "Alaska": (64.8333776, -147.4612627),
            "aa6c (WA)": (47.5706548, -122.2220673),
            "km7anm (WA)": (47.4958153, -122.1764987),
            "wq7o (WA)": (47.8143866, -117.3647516),
            "k0du (CO)": (39.1714443, -108.7239377),
            "k0uu (MN)": (44.9729495, -93.6225131),
        }
        bearings = {name: bearing_initial(TX_lat, TX_lon, lat, lon) for name,(lat,lon) in targets.items()}
        def in_kept(b): return (b >= keep_sector[0]) or (b <= keep_sector[1])
        for name,b in bearings.items():
            if in_kept(b):
                seg = [unit_from_bearing_elev(b, e)*R_shell for e in np.linspace(0, 90, 31)]
                czml.append({"id":f"hdg_{name}", "name":name} | polyline_from_samples(seg, [255,255,255,230], 2))

        with open(czml_path, "w") as f:
            json.dump(czml, f)

    return rel_dB, (vmin, vmax)

# Build sky map (1000 m) and shells (250 m, 500 m)
sky_png = "wedge_test/sky_polar_MODELED_corrected.png"
rel_dB_1000, vrng = build_outputs(1000.0, png_path=sky_png, czml_path="wedge_test/shell_wedge_MODELED_1000m.czml")
print("Starting 250 wedge")
rel_dB_250, _     = build_outputs( 250.0, png_path=None, czml_path="wedge_test/shell_wedge_MODELED_250m.czml")
print("Starting 500 wedge")
rel_dB_500, _     = build_outputs( 500.0, png_path=None, czml_path="wedge_test/shell_wedge_MODELED_500m.czml")
print("Done with wedges")
sky_png, "wedge_test/shell_wedge_MODELED_1000m.czml", "wedge_test/shell_wedge_MODELED_250m.czml", "wedge_test/shell_wedge_MODELED_500m.czml", vrng
