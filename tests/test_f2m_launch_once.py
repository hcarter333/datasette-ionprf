import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f2m_launch_once import launchangle, midpoint_lat, midpoint_lng, update_databases


def create_rm_toucans_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE rm_rnb_history_pres (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            dB REAL,
            tx_rst TEXT,
            Spotter TEXT,
            tx_lat REAL,
            tx_lng REAL,
            rx_lat REAL,
            rx_lng REAL,
            call TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO rm_rnb_history_pres (
            id, timestamp, dB, tx_rst, Spotter, tx_lat, tx_lng, rx_lat, rx_lng, call
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def create_glotec_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE glotec (
            latitude REAL,
            longitude REAL,
            timestamp TEXT,
            hmF2 REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO glotec (latitude, longitude, timestamp, hmF2) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def fetch_values(path, row_id):
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT f2m, launchangle FROM rm_rnb_history_pres WHERE id = ?",
        (row_id,),
    )
    values = cur.fetchone()
    conn.close()
    return values


def test_updates_rm_toucans_and_slice(tmp_path):
    rm_db = tmp_path / "rm_toucans.db"
    rm_slice_db = tmp_path / "rm_toucans_slice.db"
    glotec_db = tmp_path / "glotec.db"
    glotec_slice_db = tmp_path / "glotec_slice.db"

    rm_rows = [
        (
            "row-1",
            "2025-02-18T00:03:00",
            120.0,
            "59",
            "spotter-1",
            0.0,
            0.0,
            0.0,
            2.0,
            "CALL1",
        ),
        (
            "row-2",
            "2025-02-18T00:04:00",
            110.0,
            "58",
            "spotter-2",
            10.0,
            10.0,
            20.0,
            20.0,
            "CALL2",
        ),
    ]
    create_rm_toucans_db(rm_db, rm_rows)
    create_rm_toucans_db(rm_slice_db, [rm_rows[0]])

    glotec_rows = [
        (0.0, 1.0, "2025-02-18T00:02:00", 250.0),
        (0.0, 1.0, "2025-02-18T00:04:00", 300.0),
    ]
    create_glotec_db(glotec_db, glotec_rows)
    create_glotec_db(glotec_slice_db, [glotec_rows[0], glotec_rows[1]])

    updated = update_databases(
        "2025-02-18T00:00:00",
        "2025-02-18T00:10:00",
        str(rm_db),
        str(rm_slice_db),
        str(glotec_slice_db),
    )
    assert updated == 1

    midpoint_latitude = midpoint_lat(0.0, 0.0, 0.0, 2.0)
    midpoint_longitude = midpoint_lng(0.0, 0.0, 0.0, 2.0)
    expected_f2m = 300.0
    expected_launch = launchangle(0.0, 0.0, midpoint_latitude, midpoint_longitude, expected_f2m)

    rm_values = fetch_values(rm_db, "row-1")
    slice_values = fetch_values(rm_slice_db, "row-1")

    assert rm_values == slice_values
    assert rm_values[0] == expected_f2m
    assert math.isclose(rm_values[1], expected_launch, rel_tol=1e-9, abs_tol=1e-9)
