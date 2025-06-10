# test_sphere_path.py
import math
import pytest

from qsographs import great_circle_points, filter_redundant

# --- great_circle_points tests ---------------------------------------------

def test_same_point_yields_identical():
    """If start == end, every output point must equal that same lat/lon."""
    pts = list(great_circle_points(10.0, 20.0, 10.0, 20.0, segments=5))
    assert len(pts) == 6
    for lat, lon in pts:
        assert lat == pytest.approx(10.0, abs=1e-9)
        assert lon == pytest.approx(20.0, abs=1e-9)

def test_segment_count_matches_segments_arg():
    """segments=N should yield N+1 points."""
    for N in [1, 2, 10, 100]:
        pts = list(great_circle_points(0,0, 0,90, segments=N))
        assert len(pts) == N + 1

def test_equatorial_quadrant():
    """
    A path from (0,0) to (0,90) along the equator with 4 segments 
    should increment longitude by 22.5° each step.
    """
    pts = list(great_circle_points(0,0, 0,90, segments=4))
    expected_lons = [0.0, 22.5, 45.0, 67.5, 90.0]
    assert all(
        lat == pytest.approx(0.0, abs=1e-6) and
        lon == pytest.approx(exp_lon, abs=1e-6)
        for (lat, lon), exp_lon in zip(pts, expected_lons)
    )

def test_endpoints_exact_match():
    """First and last points must exactly equal the inputs."""
    a = (37.0, -122.0)
    b = (38.0, -121.0)
    pts = list(great_circle_points(*a, *b, segments=20))
    assert pts[0] == pytest.approx(a, abs=1e-9)
    assert pts[-1] == pytest.approx(b, abs=1e-9)

def test_small_angle_branch():
    """
    If the two points are extremely close, we hit the 'angle ~= 0' branch 
    and must still return sensible values.
    """
    lat, lon = 10.0, 20.0
    pts = list(great_circle_points(lat, lon, lat + 1e-12, lon + 1e-12, segments=3))
    # All points should be essentially the same.
    assert len(pts) == 4
    for p in pts:
        assert p[0] == pytest.approx(lat, abs=1e-9)
        assert p[1] == pytest.approx(lon, abs=1e-9)

# --- filter_redundant tests ------------------------------------------------

def test_no_redundancy_keeps_all():
    """If every point lies in a different cell, nothing is dropped."""
    pts = [
        (0.0, 0.0),
        (5.1, 2.6),
        (10.2, 5.2),
        (15.3, 7.8),
    ]
    filtered = list(filter_redundant(pts))
    assert filtered == pts

def test_basic_redundancy_removal():
    """
    Points (1,1) and (2,2) share the same 5°×2.5° cell → second is dropped.
    Then (6,2) is in a new lat-cell → kept.
    """
    pts = [(1,1), (2,2), (6,2), (6,3)]
    filtered = list(filter_redundant(pts))
    assert filtered == [(1,1), (6,2), (6,3)]

def test_boundary_crossing_inclusive():
    """
    A point exactly on the 5° boundary should count as the next cell.
    So (4.99, 2.49) → cell (0,0), (5.00, 2.49) → cell (1,0) → kept.
    """
    pts = [(4.99, 2.49), (5.00, 2.49)]
    filtered = list(filter_redundant(pts))
    assert filtered == pts

def test_negative_coordinates():
    """
    Negative lats/lons must floor correctly:
      e.g. lon = –1° → floor(–1/2.5)=–1; a subsequent lon=–0.1°→cell=–1 as well.
    The second should be dropped.
    """
    pts = [(-1.0, -1.0), (-0.1, -0.1), (4.9, -0.1)]
    filtered = list(filter_redundant(pts))
    assert filtered == [(-1.0, -1.0), (4.9, -0.1)]

def test_combined_pipeline():
    """
    Run great_circle_points through filter_redundant and check that
    no two consecutive outputs share the same cell.
    """
    raw = list(great_circle_points(0,0, 10,10, segments=50))
    clean = list(filter_redundant(raw))
    # verify cell changes between each pair
    last_cell = None
    for lat, lon in clean:
        cell = (math.floor(lat/5), math.floor(lon/2.5))
        assert cell != last_cell
        last_cell = cell
