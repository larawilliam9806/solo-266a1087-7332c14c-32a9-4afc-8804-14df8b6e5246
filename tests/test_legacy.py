"""Tests for shared detail storage and the legacy-library organizer.

Covers:
* identical (event_id, utc_time, magnitude) triples share one detail across
  tasks while same-id/different-value events stay separate;
* the legacy 旧库整理 entry counts pending rows and shared details;
* migration is batched, interruptible and retryable, preserving all archived
  data and per-task progress;
* collection and backfill keep working while migration is open, and every row
  written in the meantime is absorbed before migration can finish;
* backfill replaces only the current task's day and a shared detail is
  reclaimed only when its last reference disappears;
* task pages and CSV downloads read the full task data from both stores.
"""

import datetime as dt
import io
import json

import pytest

from archive_app import create_app
from archive_app.catalog import (
    MIGRATION_DONE,
    MIGRATION_FAILED,
    MIGRATION_RUNNING,
    Event,
    MockCatalogSource,
    Store,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_catalog(tmp_path, events):
    path = tmp_path / "catalog.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return str(path)


def ev(eid, day, hh=0, mm=0, mag=3.0):
    return {"id": eid, "time": f"{day}T{hh:02d}:{mm:02d}:00Z", "magnitude": mag}


def make_client(tmp_path, events, **kwargs):
    path = write_catalog(tmp_path, events)
    app = create_app(
        database=str(tmp_path / "web.db"),
        source=MockCatalogSource(path),
        testing=True,
        **kwargs,
    )
    return app.test_client(), app


def make_store(tmp_path, name="archive.db"):
    store = Store(str(tmp_path / name))
    store.init()
    return store


def counts(store):
    with store.connect() as conn:
        return {
            "legacy": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "details": conn.execute("SELECT COUNT(*) FROM event_details").fetchone()[0],
            "refs": conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0],
        }


def simulate_legacy_database(tmp_path, db_name, rows):
    """Build a database that looks like the pre-upgrade application: legacy
    ``events`` rows and no migration state."""
    store = make_store(tmp_path, db_name)
    run_id = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    with store.connect() as conn:
        for day, event_id, utc_time, mag, source in rows:
            conn.execute(
                "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                "VALUES (1,?,?,?,?,?)",
                (day, event_id, utc_time, mag, source),
            )
        for day in {row[0] for row in rows}:
            count = sum(1 for row in rows if row[0] == day)
            conn.execute(
                "UPDATE day_status SET status='complete', event_count=? "
                "WHERE run_id=1 AND day=?",
                (count, day),
            )
        conn.execute("DELETE FROM migration_state")
        conn.commit()
    # Re-open: init() must discover the legacy rows and mark migration pending.
    reopened = Store(str(tmp_path / db_name))
    reopened.init()
    return reopened, run_id


def legacy_rows(prefix, n, day="2024-01-01"):
    return [
        (day, f"{prefix}{i:03d}", f"{day} {i % 24:02d}:{i:02d}:00", 1.0 + (i % 5), "api")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# shared storage semantics
# ---------------------------------------------------------------------------


def test_fresh_database_writes_straight_into_shared_storage(tmp_path):
    path = write_catalog(tmp_path, [ev("A", "2024-01-01"), ev("B", "2024-01-02")])
    store = make_store(tmp_path)
    assert store.legacy_status()["status"] == MIGRATION_DONE

    from archive_app.collector import collect

    run = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    collect(store, MockCatalogSource(path), run)

    c = counts(store)
    assert c == {"legacy": 0, "details": 2, "refs": 2}
    assert [r["event_id"] for r in store.events(run)] == ["A", "B"]
    # Per-task archive day and source are kept on the reference.
    with store.connect() as conn:
        ref = conn.execute("SELECT * FROM event_refs WHERE run_id=? AND day='2024-01-01'", (run,)).fetchone()
        assert ref["source"] == "api"


def test_identical_triples_share_one_detail_across_runs(tmp_path):
    events = [ev("A", "2024-01-01", mag=3.1), ev("B", "2024-01-01", mag=2.2)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    from archive_app.collector import collect

    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    collect(store, MockCatalogSource(path), r1)
    collect(store, MockCatalogSource(path), r2)

    c = counts(store)
    assert c == {"legacy": 0, "details": 2, "refs": 4}
    # Each task still reads its own full copy.
    assert {r["event_id"] for r in store.events(r1)} == {"A", "B"}
    assert {r["event_id"] for r in store.events(r2)} == {"A", "B"}


def test_same_id_different_value_is_separate_detail_per_task(tmp_path):
    path1 = write_catalog(tmp_path / "c1", [ev("E1", "2024-01-01", mag=3.0)])
    path2 = write_catalog(tmp_path / "c2", [ev("E1", "2024-01-01", mag=4.5)])
    store = make_store(tmp_path)
    from archive_app.collector import collect

    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    collect(store, MockCatalogSource(path1), r1)
    collect(store, MockCatalogSource(path2), r2)

    with store.connect() as conn:
        details = conn.execute(
            "SELECT event_id, magnitude FROM event_details ORDER BY magnitude"
        ).fetchall()
        assert [(d["event_id"], d["magnitude"]) for d in details] == [
            ("E1", 3.0),
            ("E1", 4.5),
        ]
        assert conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0] == 2
    # Each task reads its own value.
    assert store.events(r1)[0]["magnitude"] == 3.0
    assert store.events(r2)[0]["magnitude"] == 4.5


def test_same_id_conflict_within_one_task_is_rejected(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    with store.connect() as conn:
        with pytest.raises(ValueError, match="事件编号冲突"):
            store._insert_ref(conn, run_id, "2024-01-01", Event("E1", "2024-01-01 00:00:00", 3.0), "api")
            store._insert_ref(conn, run_id, "2024-01-01", Event("E1", "2024-01-01 01:00:00", 4.0), "api")
        conn.rollback()


# ---------------------------------------------------------------------------
# migration mechanics
# ---------------------------------------------------------------------------


def test_migration_moves_legacy_rows_and_keeps_task_data_readable(tmp_path):
    rows = legacy_rows("OLD", 25) + legacy_rows("D2", 7, day="2024-01-02")
    store, run_id = simulate_legacy_database(tmp_path, "up.db", rows)

    status = store.legacy_status()
    assert status["status"] != MIGRATION_DONE
    assert status["legacy_pending"] == 32
    assert status["shared_details"] == 0

    result = store.run_migration_batch()
    assert result["status"] == MIGRATION_DONE
    c = counts(store)
    assert c == {"legacy": 0, "details": 32, "refs": 32}

    # Task page / download data survives unchanged, including source and day.
    events = store.events(run_id)
    assert len(events) == 32
    assert {r["day"] for r in events} == {"2024-01-01", "2024-01-02"}
    assert {r["source"] for r in events} == {"api"}
    day = {d["day"]: d for d in store.days(run_id)}
    assert day["2024-01-01"]["event_count"] == 25
    assert day["2024-01-02"]["event_count"] == 7


def test_migration_deduplicates_shared_triples_from_legacy(tmp_path):
    # Two separate legacy runs archived the same event; a third distinct one.
    store = make_store(tmp_path, "dup.db")
    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
            "VALUES (1,'2024-01-01','S1','2024-01-01 00:00:00',3.0,'api')"
        )
        conn.execute(
            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
            "VALUES (2,'2024-01-01','S1','2024-01-01 00:00:00',3.0,'csv')"
        )
        conn.execute(
            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
            "VALUES (2,'2024-01-01','S2','2024-01-01 01:00:00',3.0,'api')"
        )
        conn.execute("UPDATE day_status SET status='complete'")
        conn.execute("DELETE FROM migration_state")
        conn.commit()
    store.init()

    store.run_migration_batch()
    c = counts(store)
    assert c["legacy"] == 0
    assert c["refs"] == 3
    assert c["details"] == 2  # the identical triple collapses to one detail
    # Both runs still read their own reference to the shared detail.
    assert [r["event_id"] for r in store.events(r1)] == ["S1"]
    assert {r["event_id"] for r in store.events(r2)} == {"S1", "S2"}


def test_migration_is_interruptible_and_retryable(tmp_path):
    rows = legacy_rows("OLD", 250)
    store, run_id = simulate_legacy_database(tmp_path, "fail.db", rows)

    with pytest.raises(OSError):
        store.run_migration_batch(budget=500, fail_after=150)
    partial = store.legacy_status()
    assert partial["status"] == MIGRATION_FAILED
    assert partial["legacy_pending"] == 150
    assert partial["moved_rows"] == 100  # first 100-row chunk committed
    assert "模拟写入失败" in (partial["last_error"] or "")

    # Archived data from committed chunks stays readable while migration is open.
    assert len(store.events(run_id)) == 250

    # Retry finishes the job; progress is not redone.
    result = store.run_migration_batch()
    assert result["status"] == MIGRATION_DONE
    c = counts(store)
    assert c == {"legacy": 0, "details": 250, "refs": 250}
    assert len(store.events(run_id)) == 250


def test_migration_progress_total_keeps_growing_across_retries(tmp_path):
    rows = legacy_rows("OLD", 250)
    store, _ = simulate_legacy_database(tmp_path, "progress.db", rows)
    with pytest.raises(OSError):
        store.run_migration_batch(fail_after=150)
    partial = store.legacy_status()
    assert partial["total_rows"] == 250
    assert partial["moved_rows"] == 100

    # New rows arrive while the migration is open: total must grow, not shrink.
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
            "VALUES (1,'2024-01-01','NEW1','2024-01-01 09:00:00',1.1,'api')"
        )
        conn.commit()
    again = store.run_migration_batch()
    assert again["total_rows"] == 251
    assert again["status"] == MIGRATION_DONE
    assert again["moved_rows"] == 251


def test_collection_during_migration_writes_legacy_and_is_absorbed(tmp_path):
    rows = legacy_rows("OLD", 250)
    store, _ = simulate_legacy_database(tmp_path, "mid.db", rows)

    # First batch moves part of the old library; migration stays open.
    first = store.run_migration_batch(budget=100)
    assert first["status"] == MIGRATION_RUNNING
    assert first["legacy_pending"] == 150

    # Meanwhile a new task is collected: writes must still succeed and land in
    # the legacy table while the migration is open.
    path = write_catalog(tmp_path, [ev("N1", "2024-02-01"), ev("N2", "2024-02-02")])
    from archive_app.collector import collect

    run2 = store.create_run(dt.date(2024, 2, 1), dt.date(2024, 2, 2))
    collect(store, MockCatalogSource(path), run2)
    mid = store.legacy_status()
    assert mid["status"] == MIGRATION_RUNNING
    assert mid["legacy_pending"] == 152
    assert len(store.events(run2)) == 2

    # Finishing the migration sweeps both old rows and the in-flight ingestion.
    done = store.run_migration_batch()
    assert done["status"] == MIGRATION_DONE
    c = counts(store)
    assert c == {"legacy": 0, "details": 252, "refs": 252}
    assert len(store.events(run2)) == 2


def test_reopened_finished_migration_absorbs_stray_legacy_rows(tmp_path):
    store, _ = simulate_legacy_database(tmp_path, "stray.db", legacy_rows("OLD", 10))
    store.run_migration_batch()
    assert store.legacy_status()["status"] == MIGRATION_DONE

    # Another process (e.g. an old worker) writes a legacy row after cut-over.
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
            "VALUES (1,'2024-01-01','STRAY','2024-01-01 09:00:00',1.1,'api')"
        )
        conn.commit()
    status = store.legacy_status()
    assert status["status"] != MIGRATION_DONE
    assert status["legacy_pending"] == 1

    store.run_migration_batch()
    assert counts(store)["legacy"] == 0


# ---------------------------------------------------------------------------
# backfill replacement and detail reclamation
# ---------------------------------------------------------------------------


def test_backfill_replaces_only_this_task_day_and_keeps_other_tasks(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-01", i % 24, i) for i in range(23)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    from archive_app.collector import collect

    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    collect(store, MockCatalogSource(path), r1)
    collect(store, MockCatalogSource(path), r2)
    assert counts(store) == {"legacy": 0, "details": 20, "refs": 40}

    from archive_app.csv_import import parse_csv

    csv_text = "event_id,utc_time,magnitude\n"
    for i in range(23):
        csv_text += f"X{i:02d},2024-01-01T{i % 24:02d}:{i:02d}:00Z,3.0\n"
    parsed = parse_csv(csv_text.encode(), "2024-01-01")
    store.backfill_day(r1, "2024-01-01", parsed.events)

    c = counts(store)
    assert c["refs"] == 43  # run1: 23, run2: 20 untouched
    assert c["details"] == 23
    # The other task's first-page snapshot is fully intact.
    assert len(store.events(r2)) == 20
    assert {e["source"] for e in store.events(r1)} == {"csv"}


def test_detail_reclaimed_only_when_last_reference_removed(tmp_path):
    store = make_store(tmp_path)
    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    with store.connect() as conn:
        now = Store._now()
        conn.execute(
            "INSERT INTO event_details(event_id, utc_time, magnitude, created_at) "
            "VALUES ('E1','2024-01-01 00:00:00',3.0,?)",
            (now,),
        )
        detail_id = conn.execute("SELECT id FROM event_details").fetchone()[0]
        conn.execute(
            "INSERT INTO event_refs(run_id, day, detail_id, source, created_at) "
            "VALUES (1,'2024-01-01',?,'api',?)",
            (detail_id, now),
        )
        conn.execute(
            "INSERT INTO event_refs(run_id, day, detail_id, source, created_at) "
            "VALUES (2,'2024-01-01',?,'api',?)",
            (detail_id, now),
        )
        conn.commit()

        # One of two references gone: detail stays.
        conn.execute("DELETE FROM event_refs WHERE run_id=1")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM event_details").fetchone()[0] == 1

        # Last reference gone: trigger reclaims the detail.
        conn.execute("DELETE FROM event_refs WHERE run_id=2")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM event_details").fetchone()[0] == 0


def test_deleting_run_cascades_and_reclaims_its_details(tmp_path):
    events = [ev("A", "2024-01-01", mag=3.0)]
    path = write_catalog(tmp_path, events)
    store = make_store(tmp_path)
    from archive_app.collector import collect

    r1 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    r2 = store.create_run(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    collect(store, MockCatalogSource(path), r1)
    collect(store, MockCatalogSource(path), r2)
    assert counts(store)["details"] == 1

    with store.connect() as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM runs WHERE id=?", (r1,))
        conn.commit()
    assert counts(store) == {"legacy": 0, "details": 1, "refs": 1}

    with store.connect() as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM runs WHERE id=?", (r2,))
        conn.commit()
    assert counts(store)["details"] == 0


# ---------------------------------------------------------------------------
# reads during migration and Flask flows
# ---------------------------------------------------------------------------


def test_download_reads_full_task_data_from_both_stores(tmp_path):
    rows = legacy_rows("OLD", 5)
    store, run_id = simulate_legacy_database(tmp_path, "mix.db", rows)
    with store.connect() as conn:
        conn.execute(
            "UPDATE day_status SET status='complete' WHERE run_id=? AND day='2024-01-02'",
            (run_id,),
        )
        conn.commit()
    # Move part into shared storage, leaving the task split across both stores.
    store.run_migration_batch(budget=3)
    assert counts(store)["legacy"] == 2 and counts(store)["refs"] == 3
    assert store.is_ready(run_id)
    assert len(store.events(run_id)) == 5

    # Reuse the same database through a fresh app pointing at it.
    app2 = create_app(
        database=str(tmp_path / "mix.db"),
        source=MockCatalogSource(write_catalog(tmp_path / "mc", [])),
        testing=True,
    )
    client2 = app2.test_client()
    resp = client2.get(f"/runs/{run_id}/download")
    assert resp.status_code == 200
    lines = resp.get_data(as_text=True).strip().splitlines()
    assert lines[0] == "event_id,utc_time,magnitude"
    assert len(lines) == 6


def test_web_legacy_page_shows_counts_and_organize_flow(tmp_path):
    rows = legacy_rows("OLD", 250)
    db_path = tmp_path / "weblegacy.db"
    simulate_legacy_database(tmp_path, "weblegacy.db", rows)

    app = create_app(
        database=str(db_path),
        source=MockCatalogSource(write_catalog(tmp_path / "wc", [])),
        testing=True,
        migrate_fail_after=150,
    )
    client = app.test_client()

    home = client.get("/").get_data(as_text=True)
    assert "旧库整理" in home
    assert "待整理记录数" in home
    assert "共享明细数" in home
    assert ">250<" in home

    page = client.get("/legacy").get_data(as_text=True)
    assert "250" in page and "待整理" in page

    resp = client.post("/legacy/organize", follow_redirects=True)
    text = resp.get_data(as_text=True)
    assert "整理中断" in text
    store = app.extensions["store"]
    assert store.legacy_status()["status"] == MIGRATION_FAILED
    assert counts(store)["refs"] == 100

    # A second request without the injected failure completes the migration.
    app.config["MIGRATE_FAIL_AFTER"] = None
    resp = client.post("/legacy/organize", follow_redirects=True)
    assert "旧库整理完成" in resp.get_data(as_text=True)
    status = store.legacy_status()
    assert status["status"] == MIGRATION_DONE
    assert status["legacy_pending"] == 0
    assert counts(store) == {"legacy": 0, "details": 250, "refs": 250}

    # Organizing again with nothing left is a harmless no-op.
    resp = client.post("/legacy/organize", follow_redirects=True)
    assert "暂无待整理记录" in resp.get_data(as_text=True)


def test_legacy_page_coexists_with_task_pages_and_collect(tmp_path):
    # Legacy rows still pending: normal task work must keep functioning.
    simulate_legacy_database(tmp_path, "lock.db", legacy_rows("OLD", 10))
    app = create_app(
        database=str(tmp_path / "lock.db"),
        source=MockCatalogSource(write_catalog(tmp_path / "c2dir", [ev("N1", "2024-02-01")])),
        testing=True,
    )
    client = app.test_client()
    assert client.get("/legacy").status_code == 200
    assert client.get("/").status_code == 200

    client.post("/runs", data={"start_date": "2024-02-01", "end_date": "2024-02-01"})
    resp = client.post("/runs/2/collect", follow_redirects=True)
    assert resp.status_code == 200
    store = app.extensions["store"]
    assert len(store.events(2)) == 1
    # New ingestion during an open migration stays in the legacy table...
    assert store.legacy_status()["legacy_pending"] == 11
    client.post("/legacy/organize", follow_redirects=True)
    # ...and the organizer sweeps it together with the old rows.
    assert counts(store) == {"legacy": 0, "details": 11, "refs": 11}
    assert len(store.events(2)) == 1


def test_backfill_during_open_migration_uses_legacy_then_migrates(tmp_path):
    # A pre-upgrade database keeps the migration open while a blocked day in a
    # new task is backfilled; the CSV replacement must land in the legacy table
    # and be swept into shared storage when the organizer finishes.
    simulate_legacy_database(tmp_path, "bf.db", legacy_rows("OLD", 5))
    dense = [ev(f"X{i:02d}", "2024-02-01", i % 24, i) for i in range(22)]
    app = create_app(
        database=str(tmp_path / "bf.db"),
        source=MockCatalogSource(write_catalog(tmp_path / "bfcat", dense)),
        testing=True,
    )
    client = app.test_client()
    store = app.extensions["store"]
    assert store.legacy_status()["status"] != MIGRATION_DONE

    client.post("/runs", data={"start_date": "2024-02-01", "end_date": "2024-02-01"})
    client.post("/runs/2/collect")
    assert {d["day"]: d["status"] for d in store.days(2)}["2024-02-01"] == "blocked"
    assert counts(store)["refs"] == 0  # migration still open -> legacy writes

    csv_text = "event_id,utc_time,magnitude\n"
    for i in range(22):
        csv_text += f"X{i:02d},2024-02-01T{i % 24:02d}:{i:02d}:00Z,3.0\n"
    resp = client.post(
        "/runs/2/days/2024-02-01/upload",
        data={"csv": (io.BytesIO(csv_text.encode()), "day.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "已补档" in resp.get_data(as_text=True)
    # Backfill replaced only this task's day; 5 old rows + 22 backfilled rows.
    assert store.legacy_status()["legacy_pending"] == 27
    assert all(e["source"] == "csv" for e in store.events(2))
    assert store.is_ready(2)
    assert client.get("/runs/2/download").status_code == 200

    result = store.run_migration_batch()
    assert result["status"] == MIGRATION_DONE
    c = counts(store)
    assert c == {"legacy": 0, "details": 27, "refs": 27}
    assert len(store.events(2)) == 22
    assert {e["source"] for e in store.events(2)} == {"csv"}


def test_backfill_in_shared_mode_end_to_end(tmp_path):
    events = [ev(f"X{i:02d}", "2024-01-01", i % 24, i) for i in range(22)]
    client, app = make_client(tmp_path, events)
    client.post("/runs", data={"start_date": "2024-01-01", "end_date": "2024-01-01"})
    client.post("/runs/1/collect")
    store = app.extensions["store"]
    assert counts(store)["legacy"] == 0  # fresh deployment: shared store in use

    csv_text = "event_id,utc_time,magnitude\n"
    for i in range(22):
        csv_text += f"X{i:02d},2024-01-01T{i % 24:02d}:{i:02d}:00Z,3.0\n"
    resp = client.post(
        "/runs/1/days/2024-01-01/upload",
        data={"csv": (io.BytesIO(csv_text.encode()), "day.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "已补档" in resp.get_data(as_text=True)
    assert len(store.events(1)) == 22
    assert client.get("/runs/1/download").status_code == 200
    assert counts(store)["legacy"] == 0
