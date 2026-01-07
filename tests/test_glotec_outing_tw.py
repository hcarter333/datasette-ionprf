import runpy
import sqlite3
import sys
import types
from pathlib import Path


def test_outing_glotec_tw_fetches_expected_files(monkeypatch, tmp_path):
    listing_url = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt.json"
    base_url = "https://services.swpc.noaa.gov/"
    start_ts = "2025-01-01T00:00:00Z"
    end_ts = "2025-01-01T01:00:00Z"

    listing = [
        {"time_tag": "2025-01-01T00:10:00Z", "url": "products/glotec/file1.geojson"},
        {"time_tag": "2025-01-01T00:40:00Z", "url": "products/glotec/file2.geojson"},
        {"time_tag": "2025-01-01T02:00:00Z", "url": "products/glotec/file3.geojson"},
    ]

    file_payloads = {
        f"{base_url}products/glotec/file1.geojson": {
            "features": [
                {
                    "properties": {"hmF2": 200, "NmF2": 1.2, "quality_flag": "good"},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                }
            ]
        },
        f"{base_url}products/glotec/file2.geojson": {
            "features": [
                {
                    "properties": {"hmF2": 210, "NmF2": 1.3, "quality_flag": "good"},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                }
            ]
        },
        f"{base_url}products/glotec/file3.geojson": {
            "features": [
                {
                    "properties": {"hmF2": 220, "NmF2": 1.4, "quality_flag": "good"},
                    "geometry": {"type": "Point", "coordinates": [2.0, 2.0]},
                }
            ]
        },
    }

    fetched_urls = []

    class DummyResponse:
        def __init__(self, json_data=None, text_data=None):
            self._json_data = json_data
            self.text = text_data

        def raise_for_status(self):
            return None

        def json(self):
            return self._json_data

    def fake_get(url, *args, **kwargs):
        if url == listing_url:
            return DummyResponse(json_data=listing)
        if url in file_payloads:
            fetched_urls.append(url)
            return DummyResponse(json_data=file_payloads[url])
        raise AssertionError(f"Unexpected URL fetched: {url}")

    requests_module = types.SimpleNamespace(get=fake_get)
    dummy_numpy = types.ModuleType("numpy")
    dummy_netcdf4 = types.ModuleType("netCDF4")
    dummy_netcdf4.Dataset = object
    dummy_netcdf4.num2date = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "numpy", dummy_numpy)
    monkeypatch.setitem(sys.modules, "netCDF4", dummy_netcdf4)
    monkeypatch.chdir(tmp_path)

    rm_db = tmp_path / "rm_toucans.db"
    with sqlite3.connect(rm_db) as conn:
        conn.execute(
            "CREATE TABLE rm_rnb_history_pres (timestamp TEXT)"
        )
        conn.commit()

    repo_root = Path(__file__).resolve().parents[1]
    glotec_path = repo_root / "glotec.py"

    monkeypatch.setattr(sys, "argv", ["glotec.py", "-outing_glotec_tw", start_ts, end_ts])

    try:
        runpy.run_path(str(glotec_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    assert len(fetched_urls) == 2
