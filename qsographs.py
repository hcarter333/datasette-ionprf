#Generate evenly spaced lat/lon points along a great circle path
import math

def great_circle_points(lat1, lon1, lat2, lon2, segments=100):
    """
    Yield (lat, lon) pairs in degrees for i=0..segments along
    the great-circle from (lat1, lon1) to (lat2, lon2).
    """
    # 1) convert endpoints to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 2) convert spherical to Cartesian unit vectors
    def sph2cart(lat_r, lon_r):
        x = math.cos(lat_r) * math.cos(lon_r)
        y = math.cos(lat_r) * math.sin(lon_r)
        z = math.sin(lat_r)
        return x, y, z

    x1, y1, z1 = sph2cart(lat1_rad, lon1_rad)
    x2, y2, z2 = sph2cart(lat2_rad, lon2_rad)

    # 3) compute central angle between the two vectors
    dot = x1*x2 + y1*y2 + z1*z2
    dot = max(-1.0, min(1.0, dot))       # clamp for safety
    angle = math.acos(dot)

    # 4) interpolate in 'segments' equal steps
    for i in range(segments + 1):
        frac = i / segments
        if abs(angle) < 1e-12:
            xi, yi, zi = x1, y1, z1
        else:
            sin_ang = math.sin(angle)
            a = math.sin((1 - frac) * angle) / sin_ang
            b = math.sin(frac * angle)       / sin_ang
            xi = a*x1 + b*x2
            yi = a*y1 + b*y2
            zi = a*z1 + b*z2

        # 5) convert back to lat/lon in radians
        lat_rad_i = math.atan2(zi, math.hypot(xi, yi))
        lon_rad_i = math.atan2(yi, xi)

        # 6) to degrees
        yield math.degrees(lat_rad_i), math.degrees(lon_rad_i)


def filter_redundant(points):
    """
    Skip any point that falls in the same 5x2.5 cell
    as the last kept point.
    """
    last_cell = None
    for lat, lon in points:
        cell_lat = math.floor(lat / 5.0)
        cell_lon = math.floor(lon / 2.5)
        this_cell = (cell_lat, cell_lon)
        if this_cell != last_cell:
            yield lat, lon
            last_cell = this_cell


if __name__ == "__main__":
    # example endpoints
    start_lat, start_lon = 37.7749, -122.4194   # San Francisco
    end_lat,   end_lon   = 40.7128,  -74.0060   # New York

    raw_pts = great_circle_points(
        start_lat, start_lon,
        end_lat,   end_lon,
        segments=100
    )
    clean_pts = list(filter_redundant(raw_pts))

    for idx, (lat, lon) in enumerate(clean_pts, 1):
        print(
            "Point %3d: lat = %+8.4f deg, lon = %+8.4f deg"
            % (idx, lat, lon)
        )
