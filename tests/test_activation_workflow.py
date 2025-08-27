import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import sqlite3
from datetime import datetime
import types

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
        "id,geometry,timestamp,dB,frequency,Spotter\n"
        "1,-122.0,37.0,-80.0,35.0,2025/08/19 00:05:00,120,14000,AA1ZZ\n"
        "2,-122.0,37.0,-80.0,35.0,2025/08/20 00:05:00,90,14000,BB1ZZ\n"
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


def test_main_no_data_errors(tmp_path, monkeypatch):
    sample_csv = (
        "id,geometry,timestamp,dB,frequency,Spotter\n"
        "1,-122.0,37.0,-80.0,35.0,2025/08/19 00:05:00,120,14000,AA1ZZ\n"
    )

    class Resp:
        text = sample_csv
        content = b"czml"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(activation_workflow.requests, "get", lambda url, timeout=30: Resp())
    monkeypatch.setattr(activation_workflow, "populate_glotec", lambda start, end: None)
    monkeypatch.setattr(activation_workflow, "compute_f2m", lambda start, end: None)
    monkeypatch.setattr(activation_workflow, "finalize_db", lambda db: None)
    monkeypatch.setattr(
        activation_workflow,
        "start_datasette",
        lambda: types.SimpleNamespace(terminate=lambda: None),
    )
    monkeypatch.setattr(activation_workflow, "fetch_czml", lambda s, e, outfile: outfile.write_text("czml"))
    monkeypatch.setattr(activation_workflow, "create_artifact_dirs", lambda base, db: None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "activation_workflow.py",
            "2025-08-19T00:00:00Z",
            "2025-08-19T00:10:00Z",
        ],
    )

    activation_workflow.main()

    conn = sqlite3.connect("rm_toucans_slice.db")
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


def test_compute_f2m_calls_fm2_once(monkeypatch):
    called = {}

    def fake_call(cmd):
        called['cmd'] = cmd

    monkeypatch.setattr(activation_workflow.subprocess, "check_call", fake_call)
    start = datetime(2025, 8, 19, 0, 0, 0)
    end = datetime(2025, 8, 19, 0, 5, 0)
    activation_workflow.compute_f2m(start, end)

    assert called['cmd'][1] == 'fm2_once.py'
    assert called['cmd'][3] == '2025-08-19T00:00:00'
    assert called['cmd'][5] == '2025-08-19T00:05:00'
