import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import sqlite3
from datetime import datetime

import activation_workflow


def test_build_qso_sql_basic():
    sql = activation_workflow.build_qso_sql(-90, 90, -180, 180,
                                            "2025-08-19T00:00:00Z",
                                            "2025-08-19T01:00:00Z",
                                            -5)
    assert "WITH qso AS" in sql
    assert "FROM [rm_toucans_slice].rm_rnb_history_pres" in sql
    assert "ORDER BY timestamp ASC" in sql


def test_slice_rm_history_filters(tmp_path, monkeypatch):
    sample_csv = (
        "id,tx_lng,tx_lat,rx_lng,rx_lat,timestamp,dB,frequency,Spotter\n"
        "1,0,0,1,1,2025/08/19 00:05:00,120,14000,AA1ZZ\n"
        "2,0,0,1,1,2025/08/20 00:05:00,90,14000,BB1ZZ\n"
    )

    class Resp:
        text = sample_csv
        def raise_for_status(self):
            pass

    monkeypatch.setattr(activation_workflow.requests, "get", lambda url, timeout=30: Resp())

    start = datetime(2025, 8, 19, 0, 0, 0)
    end = datetime(2025, 8, 19, 23, 59, 59)
    db = tmp_path / "slice.db"
    activation_workflow.slice_rm_history(start, end, db)

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rm_rnb_history_pres")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_create_artifact_dirs(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE rm_rnb_history_pres (Spotter TEXT, timestamp TEXT)")
    cur.execute("INSERT INTO rm_rnb_history_pres VALUES ('AA1ZZ', '2025-08-19T00:05:00Z')")
    conn.commit(); conn.close()

    base = tmp_path / "activation"
    activation_workflow.create_artifact_dirs(base, db)

    assert (base / "AA1ZZ_2025_08_19").is_dir()
