import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osm_building_fetcher import fetch_osm_buildings


@pytest.fixture
def sample_rectangle():
    return (
        (37.794547743358315, -122.40069761028977),
        (37.79677364468366, -122.39509830705937),
    )


def test_fetch_osm_buildings_generates_expected_czml(tmp_path: Path, sample_rectangle):
    sw, ne = sample_rectangle
    czml_dir = tmp_path / "czml"

    result = fetch_osm_buildings(
        sw,
        ne,
        output_czml=True,
        czml_directory=str(czml_dir),
        offline_payload_path="tests/data/sample_overpass_response.json",
    )

    assert result["count"] == 3

    way_ids = {building["way_id"] for building in result["buildings"]}
    assert way_ids == {2001, 2002, 2003}

    for building in result["buildings"]:
        outline = building["outline"]
        assert isinstance(outline, list) and len(outline) >= 4
        for point in outline:
            assert set(point.keys()) == {"lat", "lng"}
        assert building["address"]

    czml_files = result["czml_files"]
    assert czml_files == [f"way_{way_id}.czml" for way_id in sorted(way_ids)]

    for filename in czml_files:
        path = czml_dir / filename
        assert path.is_file()
        content = json.loads(path.read_text())
        way_id = filename.split("_")[1].split(".")[0]
        assert content["id"] == f"way-{way_id}"

        expected_path = Path("czml_output") / filename
        assert expected_path.is_file()
        assert json.loads(expected_path.read_text()) == content

    manifest_path = czml_dir / "czml_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"] == czml_files

    expected_manifest = Path("czml_output") / "czml_manifest.json"
    assert expected_manifest.is_file()
    assert json.loads(expected_manifest.read_text())["files"] == czml_files
