#!/usr/bin/env python3
"""
qso_angle_indicators.py

Generate CZML angle indicators (horizontal line, angle arc, label) for QSO polylines.

Rules implemented:
1) A white horizontal line from the base (TX) along the great-circle toward RX.
2) An arc between the horizontal line and the ascending QSO polyline (TX side only).
3) A label showing the angle in decimal degrees (taken from polyline 'id' substring 'angle: N.N').
4) Skip any polyline whose TX altitude is 0.
5) Never add an arc to the receive side (we always use the first point as TX).

Usage:
    python qso_angle_indicators.py input_qsos.czml output_angles.czml \
        --horiz_km 20 --arc_radius_m 5000 --arc_samples 32 --label_up_m 500

Notes:
- Input polylines must provide positions as cartographicDegrees (lon,lat,height) in CZML.
- For each QSO polyline, we assume the first coordinate is the transmitter (TX) end
  and the last coordinate is the receiver (RX) end.
"""

import argparse
import json
import math
import re
from typing import List, Tuple, Optional, Dict, Any

# -------------------------
# Geometry helpers (spherical Earth)
# -------------------------

EARTH_RADIUS_M = 6371008.8  # mean radius

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def initial_bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """Initial great-circle bearing from point 1 to point 2 (degrees, [0,360))."""
    lat1 = deg2rad(lat1_deg)
    lat2 = deg2rad(lat2_deg)
    dlon = deg2rad(lon2_deg - lon1_deg)

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(x, y)
    brng_deg = (rad2deg(brng) + 360.0) % 360.0
    return brng_deg

def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    Destination lat/lon from start point, bearing, and distance on a sphere.
    Returns (lat_deg, lon_deg).
    """
    lat1 = deg2rad(lat_deg)
    lon1 = deg2rad(lon_deg)
    brng = deg2rad(bearing_deg)
    ang_dist = distance_m / EARTH_RADIUS_M

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_ad = math.sin(ang_dist)
    cos_ad = math.cos(ang_dist)

    sin_lat2 = sin_lat1 * cos_ad + cos_lat1 * sin_ad * math.cos(brng)
    lat2 = math.asin(sin_lat2)
    y = math.sin(brng) * sin_ad * cos_lat1
    x = cos_ad - sin_lat1 * sin_lat2
    lon2 = lon1 + math.atan2(y, x)

    # Normalize lon to [-180, 180]
    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi
    return (rad2deg(lat2), rad2deg(lon2))

# -------------------------
# CZML helpers
# -------------------------

def extract_cartographic_positions(packet: Dict[str, Any]) -> Optional[List[Tuple[float, float, float]]]:
    """
    Extract positions from a CZML polyline packet as list of (lon, lat, height).
    Supports 'positions': {'cartographicDegrees': [...]} only (common for Cesium).
    """
    poly = packet.get("polyline")
    if not poly:
        return None
    pos = poly.get("positions")
    if not pos:
        return None
    cardeg = pos.get("cartographicDegrees")
    if not isinstance(cardeg, list) or len(cardeg) < 6:
        return None

    triples = []
    it = iter(cardeg)
    try:
        while True:
            lon = float(next(it))
            lat = float(next(it))
            h = float(next(it))
            triples.append((lon, lat, h))
    except StopIteration:
        pass

    if len(triples) < 2:
        return None
    return triples

ANGLE_RE = re.compile(r"angle\s*:\s*([-+]?\d+(\.\d+)?)", re.IGNORECASE)

def parse_angle_from_id(packet_id: str) -> Optional[float]:
    """
    Parse 'angle: N.N' from the packet id. Returns float degrees or None.
    """
    if not isinstance(packet_id, str):
        return None
    m = ANGLE_RE.search(packet_id)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

# -------------------------
# Indicator builders
# -------------------------

def make_horizontal_line_packet(
    base_name: str,
    tx_lat: float,
    tx_lon: float,
    tx_alt_m: float,
    bearing_to_rx_deg: float,
    length_m: float
) -> Dict[str, Any]:
    """
    Create a CZML packet for the white horizontal line along the GC from TX toward RX.
    Clamped to ground, but we place height explicitly as 0 to honor 'horizontal'.
    """
    p1 = (tx_lon, tx_lat, 0.0)
    lat2, lon2 = destination_point(tx_lat, tx_lon, bearing_to_rx_deg, length_m)
    p2 = (lon2, lat2, 0.0)

    return {
        "id": f"{base_name}_horiz",
        "name": f"{base_name} horizontal",
        "polyline": {
            "positions": {"cartographicDegrees": [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]},
            "material": {"solidColor": {"color": {"rgba": [255, 255, 255, 255]}}},
            "width": 2,
            "clampToGround": True,
            "arcType": "GEODESIC"
        }
    }

def make_angle_arc_packet(
    base_name: str,
    tx_lat: float,
    tx_lon: float,
    bearing_to_rx_deg: float,
    angle_deg: float,
    arc_radius_m: float,
    arc_samples: int
) -> Dict[str, Any]:
    """
    Create a CZML packet for the curved angle arc in the TX vertical plane toward RX.
    We approximate a circular arc by sampling phi in [0, angle] where:
      horizontal_range = arc_radius_m * cos(phi)
      altitude         = arc_radius_m * sin(phi)
    Then we move along surface from TX by 'horizontal_range' at 'bearing_to_rx_deg'
    and assign the altitude for each sample.
    """
    angle_rad = deg2rad(angle_deg)
    positions = []
    for i in range(arc_samples + 1):
        phi = angle_rad * i / arc_samples
        horiz = arc_radius_m * math.cos(phi)
        alt = arc_radius_m * math.sin(phi)
        lat_i, lon_i = destination_point(tx_lat, tx_lon, bearing_to_rx_deg, horiz)
        positions.extend([lon_i, lat_i, alt])

    return {
        "id": f"{base_name}_arc",
        "name": f"{base_name} arc",
        "polyline": {
            "positions": {"cartographicDegrees": positions},
            "material": {"solidColor": {"color": {"rgba": [255, 165, 0, 255]}}},  # orange
            "width": 3,
            "clampToGround": False,
            "arcType": "NONE"  # We supply explicit points for the curve
        }
    }

def make_label_packet(
    base_name: str,
    tx_lat: float,
    tx_lon: float,
    bearing_to_rx_deg: float,
    angle_deg: float,
    arc_radius_m: float,
    label_up_m: float
) -> Dict[str, Any]:
    """
    Create label packet positioned close to the arc.
    Now it hugs the outer edge of the angle arc instead of being far away.
    """
    phi_mid = deg2rad(angle_deg) * 0.5
    # Move the label slightly beyond the midpoint of the arc for visibility
    horiz_mid = arc_radius_m * math.cos(phi_mid) * 0.95
    alt_mid = arc_radius_m * math.sin(phi_mid)  # right on the arc
    lat_m, lon_m = destination_point(tx_lat, tx_lon, bearing_to_rx_deg, horiz_mid)

    return {
        "id": f"{base_name}_label",
        "name": f"{base_name} label",
        "position": {"cartographicDegrees": [lon_m, lat_m, alt_mid + label_up_m]},
        "label": {
            "text": f"{angle_deg:.1f}°",
            "font": "16px sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "horizontalOrigin": "CENTER",
            "verticalOrigin": "CENTER",
            "pixelOffset": {"cartesian2": [0, 0]},
            "heightReference": "NONE",
            "disableDepthTestDistance": 10000000
        }
    }

# -------------------------
# Main processing
# -------------------------

def process_qso_czml(
    input_packets: List[Dict[str, Any]],
    horiz_km: float = 20.0,
    arc_radius_m: float = 5000.0,
    arc_samples: int = 32,
    label_up_m: float = 500.0
) -> List[Dict[str, Any]]:
    """Build the output CZML packets with angle indicators."""
    output_packets = [
        {
            "id": "document",
            "name": "QSO Angle Indicators",
            "version": "1.0"
        }
    ]

    for pkt in input_packets:
        if not isinstance(pkt, dict):
            continue
        if pkt.get("id") == "document":
            continue

        positions = extract_cartographic_positions(pkt)
        if not positions:
            continue

        pkt_id = pkt.get("id", "")
        angle_deg = parse_angle_from_id(pkt_id)
        if angle_deg is None:
            # No angle specifier, skip
            continue

        # Identify TX and RX ends (assume first is TX, last is RX)
        tx_lon, tx_lat, tx_alt = positions[0]
        rx_lon, rx_lat, rx_alt = positions[-1]

        # Rule 4: skip if TX altitude is 0 (no launch angle)
        if abs(tx_alt) <= 1e-6:
            continue

        # Compute initial bearing from TX to RX (for horizontal line direction)
        bearing = initial_bearing_deg(tx_lat, tx_lon, rx_lat, rx_lon)

        base_name = f"{pkt_id}_angle_indicator"

        # Horizontal line (on ground)
        horiz_len_m = max(1.0, horiz_km * 1000.0)
        horiz_packet = make_horizontal_line_packet(
            base_name, tx_lat, tx_lon, tx_alt, bearing, horiz_len_m
        )
        output_packets.append(horiz_packet)

        # Arc (vertical plane at TX, toward RX)
        arc_packet = make_angle_arc_packet(
            base_name, tx_lat, tx_lon, bearing, angle_deg, arc_radius_m, arc_samples
        )
        output_packets.append(arc_packet)

        # Label
        label_packet = make_label_packet(
            base_name, tx_lat, tx_lon, bearing, angle_deg, arc_radius_m, label_up_m
        )
        output_packets.append(label_packet)

    return output_packets

def main():
    ap = argparse.ArgumentParser(description="Generate CZML angle indicators for QSO polylines.")
    ap.add_argument("input_czml", help="Path to input CZML with QSO polylines (cartographicDegrees).")
    ap.add_argument("output_czml", help="Path to output CZML for angle indicators.")
    ap.add_argument("--horiz_km", type=float, default=20.0,
                    help="Horizontal line length from TX along great-circle (km). Default: 20.")
    ap.add_argument("--arc_radius_m", type=float, default=5000.0,
                    help="Radius of the angle arc (meters). Default: 5000.")
    ap.add_argument("--arc_samples", type=int, default=32,
                    help="Number of samples for the arc polyline. Default: 32.")
    ap.add_argument("--label_up_m", type=float, default=500.0,
                    help="Extra upward offset for the label (meters). Default: 500.")
    args = ap.parse_args()

    with open(args.input_czml, "r", encoding="utf-8") as f:
        input_packets = json.load(f)

    output_packets = process_qso_czml(
        input_packets,
        horiz_km=args.horiz_km,
        arc_radius_m=args.arc_radius_m,
        arc_samples=args.arc_samples,
        label_up_m=args.label_up_m
    )

    with open(args.output_czml, "w", encoding="utf-8") as f:
        json.dump(output_packets, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output_packets)-1} angle indicator elements to {args.output_czml}")

if __name__ == "__main__":
    main()
