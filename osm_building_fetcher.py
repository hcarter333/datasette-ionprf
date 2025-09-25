import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from urllib.parse import urlencode

REQUEST_HEADERS = {
    "User-Agent": "github.com/ionprf-osm-building-fetcher",
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class OverpassError(RuntimeError):
    """Raised when the Overpass API reports an error."""


AddressTags = Tuple[str, ...]
_ADDRESS_FIELDS: AddressTags = (
    "addr:housenumber",
    "addr:houseletter",
    "addr:street",
    "addr:suburb",
    "addr:city",
    "addr:state",
    "addr:postcode",
    "addr:country",
)


def _normalize_height(raw_height: Optional[str], raw_levels: Optional[str]) -> Optional[float]:
    """Return height in meters using tags from OSM."""

    def _clean(value: str) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", value)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    height = _clean(raw_height or "")
    if height is not None:
        return height

    height = _clean(raw_levels or "")
    if height is not None:
        # Convert levels to approximate height using a standard floor height
        return height * 3.0

    return None


def _format_address(tags: Dict[str, str]) -> Optional[str]:
    parts: List[str] = []
    for key in _ADDRESS_FIELDS:
        value = tags.get(key)
        if value:
            parts.append(value)
    return ", ".join(parts) if parts else None


def _ensure_closed_ring(nodes: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not nodes:
        return nodes
    if nodes[0] != nodes[-1]:
        nodes.append(nodes[0])
    return nodes


def _build_czml_document(way_id: int, coordinates: List[Dict[str, float]], height: Optional[float]) -> Dict:
    if height is None:
        height = 10.0

    # Cesium polygons expect a flat list of lon, lat, height values
    cartographic: List[float] = []
    for coord in coordinates:
        cartographic.extend([coord["lng"], coord["lat"], 0.0])

    return {
        "id": f"way-{way_id}",
        "name": f"OSM Way {way_id}",
        "version": "1.0",
        "clock": {
            "interval": "2020-01-01T00:00:00Z/2020-01-01T00:01:00Z",
            "currentTime": "2020-01-01T00:00:00Z",
            "multiplier": 1,
        },
        "entities": [
            {
                "id": f"building-{way_id}",
                "name": f"Building {way_id}",
                "polygon": {
                    "positions": {
                        "cartographicDegrees": cartographic,
                    },
                    "perPositionHeight": False,
                    "extrudedHeight": height,
                    "height": 0.0,
                    "material": {
                        "solidColor": {
                            "color": {
                                "rgba": [255, 165, 0, 160],
                            }
                        }
                    },
                    "outline": True,
                    "outlineColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                },
            }
        ],
    }


def fetch_osm_buildings(
    sw_corner: Tuple[float, float],
    ne_corner: Tuple[float, float],
    *,
    overpass_urls: Optional[Iterable[str]] = None,
    output_czml: bool = False,
    czml_directory: str = "czml_output",
    offline_payload_path: Optional[str] = None,
) -> Dict:
    """Fetch building footprints and metadata for a bounding box.

    Args:
        sw_corner: (latitude, longitude) of the south-west corner.
        ne_corner: (latitude, longitude) of the north-east corner.
        overpass_urls: Optional iterable of Overpass API endpoints to try in order.
        output_czml: If True, write a CZML file per building into ``czml_directory``.
        czml_directory: Destination directory for CZML files.

    Returns:
        A dictionary with the number of buildings and details for each building.
    """

    south = min(sw_corner[0], ne_corner[0])
    north = max(sw_corner[0], ne_corner[0])
    west = min(sw_corner[1], ne_corner[1])
    east = max(sw_corner[1], ne_corner[1])

    query = f"""
    [out:json][timeout:120];
    (
      way["building"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """

    candidate_urls = list(overpass_urls or [
        OVERPASS_URL,
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ])

    payload: Optional[Dict] = None
    errors: List[str] = []

    if offline_payload_path:
        offline_path = Path(offline_payload_path)
        if not offline_path.is_file():
            errors.append(f"Offline payload not found: {offline_path}")
        else:
            print(f"Loading offline Overpass payload from {offline_path}")
            with offline_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)

    if payload is None:
        import requests
        from requests import RequestException

        for overpass_url in candidate_urls:
            encoded_query = urlencode({"data": query.strip()})
            full_url = f"{overpass_url}?{encoded_query}"
            print(f"Requesting Overpass: {full_url}")
            try:
                response = requests.get(full_url, headers=REQUEST_HEADERS)
                response.raise_for_status()
            except RequestException as exc:
                errors.append(f"{overpass_url}: {exc}")
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                errors.append(f"{overpass_url}: invalid JSON response ({exc})")
                payload = None
                continue
            break

    if payload is None:
        detail = "; ".join(errors) if errors else "Overpass API returned no payload"
        raise OverpassError(detail)
    if "elements" not in payload:
        raise OverpassError("Invalid Overpass API response: missing 'elements'")

    nodes: Dict[int, Dict[str, float]] = {}
    buildings: Dict[int, Dict] = {}

    for element in payload["elements"]:
        el_type = element.get("type")
        if el_type == "node":
            nodes[element["id"]] = {"lat": element["lat"], "lng": element["lon"]}
        elif el_type == "way" and element.get("tags", {}).get("building"):
            buildings[element["id"]] = element

    results: List[Dict] = []
    czml_files: List[str] = []
    if output_czml:
        Path(czml_directory).mkdir(parents=True, exist_ok=True)

    for way_id, way in buildings.items():
        node_refs = way.get("nodes", [])
        coordinates: List[Dict[str, float]] = []
        for node_id in node_refs:
            node = nodes.get(node_id)
            if not node:
                continue
            coordinates.append({"lat": node["lat"], "lng": node["lng"]})

        coordinates = _ensure_closed_ring(coordinates)
        if len(coordinates) < 4:
            # Need at least 3 unique points + closing point
            continue

        tags = way.get("tags", {})
        height = _normalize_height(tags.get("height") or tags.get("building:height"), tags.get("building:levels"))
        address = _format_address(tags)

        building_info = {
            "way_id": way_id,
            "outline": coordinates,
            "height_m": height,
            "address": address,
        }
        results.append(building_info)

        if output_czml:
            czml_doc = _build_czml_document(way_id, coordinates, height)
            czml_path = os.path.join(czml_directory, f"way_{way_id}.czml")
            with open(czml_path, "w", encoding="utf-8") as fh:
                json.dump(czml_doc, fh, indent=2)
                fh.write("\n")
            czml_files.append(os.path.basename(czml_path))

    if output_czml and czml_files:
        manifest_path = os.path.join(czml_directory, "czml_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump({"files": czml_files}, manifest_file, indent=2)
            manifest_file.write("\n")

    return {
        "count": len(results),
        "buildings": results,
        "czml_files": czml_files if output_czml else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSM building data and optional CZML output.")
    parser.add_argument("--czml-directory", default="czml_output", help="Directory for CZML output files.")
    parser.add_argument(
        "--offline-payload",
        help="Path to a saved Overpass API JSON payload to use instead of performing network requests.",
    )
    parser.add_argument(
        "--overpass-url",
        action="append",
        dest="overpass_urls",
        help="Override default Overpass endpoints (can be supplied multiple times).",
    )
    parser.add_argument(
        "--sw",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=(37.794547743358315, -122.40069761028977),
        help="South-west corner latitude and longitude.",
    )
    parser.add_argument(
        "--ne",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=(37.79677364468366, -122.39509830705937),
        help="North-east corner latitude and longitude.",
    )
    parser.add_argument(
        "--no-czml",
        action="store_true",
        help="Disable CZML output even when running standalone.",
    )

    args = parser.parse_args()

    sw = (args.sw[0], args.sw[1])
    ne = (args.ne[0], args.ne[1])

    data = fetch_osm_buildings(
        sw,
        ne,
        overpass_urls=args.overpass_urls,
        output_czml=not args.no_czml,
        czml_directory=args.czml_directory,
        offline_payload_path=args.offline_payload,
    )
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
