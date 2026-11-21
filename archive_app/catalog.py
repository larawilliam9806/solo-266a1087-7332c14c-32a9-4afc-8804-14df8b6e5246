"""Local earthquake catalog archive service.

The application talks to a *catalog source* through a small, well-defined
interface.  The bundled ``MockCatalogSource`` simulates the regional network's
paged-by-date endpoint (max 20 events per response).  A real deployment can
swap in another source with the same two methods.
"""

from __future__ import annotations

import abc
import datetime as dt
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence


PAGE_SIZE = 20
MAX_SPAN_DAYS = 31
UTC = dt.timezone.utc


class SourceError(Exception):
    """Raised when the upstream catalog cannot be queried."""


@dataclass(frozen=True)
class Event:
    event_id: str
    utc_time: str  # canonical "YYYY-MM-DD HH:MM:SS"
    magnitude: float

    def as_tuple(self) -> tuple[str, str, float]:
        return (self.event_id, self.utc_time, self.magnitude)


def canonical_time(value: str) -> str:
    """Normalise a UTC issuance time to ``YYYY-MM-DD HH:MM:SS``."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析 UTC 时刻: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def parse_day(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"日期必须为 YYYY-MM-DD: {value!r}") from exc


class CatalogSource(abc.ABC):
    @abc.abstractmethod
    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        """Return the events with start <= UTC date <= end, capped at 20.

        The upstream has no paging: a full response means "there may be more
        inside this window", which the collector resolves by narrowing dates.
        """

    def page_size(self) -> int:
        return PAGE_SIZE


class MockCatalogSource(CatalogSource):
    """Reads a deterministic JSON catalog standing in for the live endpoint.

    File layout::

        {"events": [{"id": "...", "time": "2024-01-02T03:04:05Z",
                     "magnitude": 3.2}, ...]}

    The mock is intentionally strict: it only serves windows of at most 31
    days, returns at most 20 events, and reports an error for a configured list
    of dates so retry logic can be exercised.
    """

    def __init__(self, path: str | os.PathLike[str], fail_dates: Iterable[str] = ()):
        self.path = os.fspath(path)
        self.fail_dates = set(fail_dates)
        self._lock = threading.Lock()
        self._events: list[Event] | None = None
        self._mtime: float | None = None

    def _load(self) -> list[Event]:
        mtime = os.stat(self.path).st_mtime
        with self._lock:
            if self._events is None or mtime != self._mtime:
                with open(self.path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                events = []
                for raw in payload.get("events", []):
                    events.append(
                        Event(
                            event_id=str(raw["id"]).strip(),
                            utc_time=canonical_time(raw["time"]),
                            magnitude=float(raw["magnitude"]),
                        )
                    )
                events.sort(key=lambda e: (e.utc_time, e.event_id))
                self._events = events
                self._mtime = mtime
            return self._events

    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        if end < start:
            raise SourceError("日期区间无效")
        if (end - start).days + 1 > MAX_SPAN_DAYS:
            raise SourceError(f"日期区间不得超过 {MAX_SPAN_DAYS} 天")
        for day in self.fail_dates:
            d = dt.date.fromisoformat(day)
            if start <= d <= end:
                raise SourceError(f"模拟接口故障: {day}")
        events = [
            e
            for e in self._load()
            if start <= dt.date.fromisoformat(e.utc_time[:10]) <= end
        ]
        return events[:PAGE_SIZE]


def source_from_env() -> CatalogSource:
    path = os.environ.get("CATALOG_FILE", "")
    if not path:
        for candidate in ("data/catalog.json", "seed/catalog.json"):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            path = "data/catalog.json"
    if not os.path.exists(path):
        raise RuntimeError(
            f"找不到模拟目录数据文件 {path}；用 CATALOG_FILE 指定，"
            "或运行 scripts/generate_catalog.py 生成。"
        )
    fail = [d.strip() for d in os.environ.get("MOCK_FAIL_DATES", "").split(",") if d.strip()]
    return MockCatalogSource(path, fail_dates=fail)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS day_status (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    checked_at TEXT,
    last_error TEXT,
    PRIMARY KEY (run_id, day),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
-- 旧库表：共享存储上线前的归档数据，旧库整理期间仍可写入，整理后清空
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    event_id TEXT NOT NULL,
    utc_time TEXT NOT NULL,
    magnitude REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(run_id, day);

-- 共享明细：编号、UTC 发震时刻、震级完全相同的事件共用一份；
-- 同号异值（三元组不同）是不同明细，在不同任务中各存一份。
CREATE TABLE IF NOT EXISTS event_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    utc_time TEXT NOT NULL,
    magnitude REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (event_id, utc_time, magnitude)
);
-- 任务对明细的引用：归档日、来源、进度仍按任务分别保存
CREATE TABLE IF NOT EXISTS event_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    detail_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, detail_id),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
    FOREIGN KEY (detail_id) REFERENCES event_details(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_refs_run ON event_refs(run_id);
CREATE INDEX IF NOT EXISTS idx_refs_run_day ON event_refs(run_id, day);
-- 最后一个任务引用被移除（含任务删除、补档替换）时回收对应明细
CREATE TRIGGER IF NOT EXISTS trg_refs_after_delete AFTER DELETE ON event_refs
BEGIN
    DELETE FROM event_details
     WHERE id = OLD.detail_id
       AND NOT EXISTS (SELECT 1 FROM event_refs WHERE detail_id = OLD.detail_id);
END;
-- 统一读取视图：整理期间新旧两套存储并行，任务页与下载都读全本任务数据
CREATE VIEW IF NOT EXISTS run_events AS
SELECT run_id, day, event_id, utc_time, magnitude, source
  FROM events
UNION ALL
SELECT r.run_id, r.day, d.event_id, d.utc_time, d.magnitude, r.source
  FROM event_refs r
  JOIN event_details d ON d.id = r.detail_id;
-- 旧库整理进度（单行，id 固定为 1）
CREATE TABLE IF NOT EXISTS migration_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    status TEXT NOT NULL,
    total_rows INTEGER NOT NULL DEFAULT 0,
    moved_rows INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT
);
"""

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"   # day is fully archived
STATUS_BLOCKED = "blocked"     # day returned a full page; needs CSV backfill
STATUS_FAILED = "failed"       # request failed; retry later

# 旧库整理状态
MIGRATION_PENDING = "pending"    # 尚未开始或有待整理记录
MIGRATION_RUNNING = "running"    # 整理进行中（可中断、可重试）
MIGRATION_DONE = "done"          # 旧库已清空，全部进入共享存储
MIGRATION_FAILED = "failed"      # 上一批写入失败，进度保留、可重试

MIGRATION_CHUNK = 100  # 每提交一个事务的行数，保证中断后已提交进度不丢

ALLOWED_TRANSITIONS = {
    STATUS_PENDING: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_FAILED: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_BLOCKED: {STATUS_COMPLETE, STATUS_BLOCKED},
    STATUS_COMPLETE: set(),
}


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_migration_row(conn)
            conn.commit()

    # -- legacy migration state ---------------------------------------

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _legacy_count(self, conn: sqlite3.Connection) -> int:
        return int(
            conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )

    def _ensure_migration_row(self, conn: sqlite3.Connection) -> None:
        """Create the single migration row, deriving status from stored data.

        Fresh deployments (no legacy rows at all) start directly in the shared
        storage.  Databases carrying rows from the old schema start pending;
        if a previously-finished migration finds new legacy rows (for example
        rows written by another process during the cut-over), it reopens so the
        next organize call absorbs them.
        """
        row = conn.execute("SELECT status FROM migration_state WHERE id=1").fetchone()
        legacy = self._legacy_count(conn)
        if row is None:
            status = MIGRATION_PENDING if legacy else MIGRATION_DONE
            conn.execute(
                "INSERT INTO migration_state(id, status, total_rows, moved_rows, updated_at) "
                "VALUES (1,?,?,?,?)",
                (status, legacy, 0, self._now()),
            )
            return
        if row["status"] == MIGRATION_DONE and legacy:
            total = int(
                conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0]
            ) + legacy
            conn.execute(
                "UPDATE migration_state SET status=?, total_rows=?, last_error=NULL, updated_at=? "
                "WHERE id=1",
                (MIGRATION_PENDING, total, self._now()),
            )

    def legacy_status(self) -> dict[str, int | str | None]:
        """Counts for the 旧库整理 entry: pending legacy rows and shared details."""
        with self.connect() as conn:
            self._ensure_migration_row(conn)
            conn.commit()
            row = conn.execute(
                "SELECT * FROM migration_state WHERE id=1"
            ).fetchone()
            legacy = self._legacy_count(conn)
            shared = int(
                conn.execute("SELECT COUNT(*) FROM event_details").fetchone()[0]
            )
            refs = int(
                conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0]
            )
        return {
            "status": row["status"],
            "legacy_pending": legacy,
            "shared_details": shared,
            "shared_refs": refs,
            "total_rows": max(int(row["total_rows"]), legacy + refs),
            "moved_rows": refs,
            "last_error": row["last_error"],
            "updated_at": row["updated_at"],
        }

    def _use_shared(self, conn: sqlite3.Connection) -> bool:
        """Writes go shared once the legacy migration is fully done."""
        row = conn.execute("SELECT status FROM migration_state WHERE id=1").fetchone()
        return row is not None and row["status"] == MIGRATION_DONE

    def run_migration_batch(
        self, budget: int = 500, fail_after: int | None = None
    ) -> dict[str, int | str | None]:
        """Move one batch of legacy rows into the shared store.

        Each chunk commits in its own transaction, so a crash or write failure
        keeps everything committed so far: already-archived data and progress
        stay usable and the call can simply be retried.
        """
        with self._lock:
            with self.connect() as conn:
                self._ensure_migration_row(conn)
                conn.commit()
                state = conn.execute(
                    "SELECT * FROM migration_state WHERE id=1"
                ).fetchone()
                legacy = self._legacy_count(conn)
                if state["status"] == MIGRATION_DONE and not legacy:
                    return self.legacy_status()
                refs = int(
                    conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0]
                )
                total = max(int(state["total_rows"]), legacy + refs)
                conn.execute(
                    "UPDATE migration_state SET status=?, total_rows=?, last_error=NULL, updated_at=? "
                    "WHERE id=1",
                    (MIGRATION_RUNNING, total, self._now()),
                )
                conn.commit()

            processed = 0
            try:
                while processed < budget:
                    with self.connect() as conn:
                        rows = conn.execute(
                            "SELECT rowid, run_id, day, event_id, utc_time, magnitude, source "
                            "FROM events ORDER BY rowid LIMIT ?",
                            (min(MIGRATION_CHUNK, budget - processed),),
                        ).fetchall()
                        if not rows:
                            break
                        try:
                            conn.execute("BEGIN")
                            for row in rows:
                                self._migrate_row(conn, row)
                                processed += 1
                                if fail_after is not None and processed >= fail_after:
                                    raise OSError("模拟写入失败（MIGRATE_FAIL_AFTER）")
                            conn.commit()
                        except Exception:
                            conn.rollback()
                            raise
            except Exception as exc:
                with self.connect() as conn:
                    refs = int(
                        conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0]
                    )
                    legacy = self._legacy_count(conn)
                    conn.execute(
                        "UPDATE migration_state SET status=?, total_rows=?, moved_rows=?, "
                        "last_error=?, updated_at=? WHERE id=1",
                        (MIGRATION_FAILED, max(total, legacy + refs), refs, str(exc), self._now()),
                    )
                    conn.commit()
                raise

            with self.connect() as conn:
                refs = int(
                    conn.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0]
                )
                legacy = self._legacy_count(conn)
                new_total = max(total, legacy + refs)
                status = MIGRATION_DONE if not legacy else MIGRATION_RUNNING
                conn.execute(
                    "UPDATE migration_state SET status=?, total_rows=?, moved_rows=?, "
                    "last_error=NULL, updated_at=? WHERE id=1",
                    (status, new_total, refs, self._now()),
                )
                conn.commit()
        return self.legacy_status()

    def _migrate_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        detail_id = self._insert_detail(
            conn, row["event_id"], row["utc_time"], row["magnitude"]
        )
        conn.execute(
            "INSERT OR IGNORE INTO event_refs(run_id, day, detail_id, source, created_at) "
            "VALUES (?,?,?,?,?)",
            (row["run_id"], row["day"], detail_id, row["source"], self._now()),
        )
        conn.execute("DELETE FROM events WHERE rowid=?", (row["rowid"],))

    # -- runs ---------------------------------------------------------

    def create_run(self, start: dt.date, end: dt.date) -> int:
        with self._lock, self.connect() as conn:
            run_id = self._create_run(conn, start, end)
            conn.commit()
            return run_id

    @staticmethod
    def _create_run(conn: sqlite3.Connection, start: dt.date, end: dt.date) -> int:
        now = Store._now()
        cur = conn.execute(
            "INSERT INTO runs(start_date, end_date, created_at) VALUES (?,?,?)",
            (start.isoformat(), end.isoformat(), now),
        )
        run_id = int(cur.lastrowid)
        day = start
        while day <= end:
            conn.execute(
                "INSERT INTO day_status(run_id, day, status) VALUES (?,?,?)",
                (run_id, day.isoformat(), STATUS_PENDING),
            )
            day += dt.timedelta(days=1)
        return run_id

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def runs(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT r.*, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='complete') AS complete_days, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='blocked') AS blocked_days "
                "FROM runs r ORDER BY r.id DESC"
            ).fetchall()

    def days(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM day_status WHERE run_id=? ORDER BY day", (run_id,)
            ).fetchall()

    def events(self, run_id: int, day: str | None = None) -> list[sqlite3.Row]:
        """All events visible to a task, from whichever store currently holds them."""
        with self.connect() as conn:
            if day is None:
                return conn.execute(
                    "SELECT run_id, day, event_id, utc_time, magnitude, source "
                    "FROM run_events WHERE run_id=? ORDER BY utc_time, event_id",
                    (run_id,),
                ).fetchall()
            return conn.execute(
                "SELECT run_id, day, event_id, utc_time, magnitude, source "
                "FROM run_events WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                (run_id, day),
            ).fetchall()

    # -- progress -----------------------------------------------------

    def mark(self, run_id: int, day: str, status: str, last_error: str | None = None) -> None:
        if status not in {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED}:
            raise ValueError(status)
        with self._lock, self.connect() as conn:
            self._mark(conn, run_id, day, status, last_error)
            conn.commit()

    @staticmethod
    def _mark(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        status: str,
        last_error: str | None = None,
    ) -> None:
        row = conn.execute(
            "SELECT status FROM day_status WHERE run_id=? AND day=?", (run_id, day)
        ).fetchone()
        if row is None:
            raise KeyError(day)
        current = row["status"]
        if current not in ALLOWED_TRANSITIONS or status not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"状态不允许从 {current} 变为 {status}")
        now = Store._now()
        count = conn.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id=? AND day=?", (run_id, day)
        ).fetchone()[0]
        conn.execute(
            "UPDATE day_status SET status=?, event_count=?, checked_at=?, last_error=? "
            "WHERE run_id=? AND day=?",
            (status, count, now, last_error, run_id, day),
        )

    # -- shared detail helpers ----------------------------------------

    def _insert_detail(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        utc_time: str,
        magnitude: float,
    ) -> int:
        """Return the shared detail id for an event triple, creating it once."""
        conn.execute(
            "INSERT OR IGNORE INTO event_details(event_id, utc_time, magnitude, created_at) "
            "VALUES (?,?,?,?)",
            (event_id, utc_time, magnitude, self._now()),
        )
        row = conn.execute(
            "SELECT id FROM event_details "
            "WHERE event_id=? AND utc_time=? AND magnitude=?",
            (event_id, utc_time, magnitude),
        ).fetchone()
        return int(row["id"])

    def _insert_ref(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        event: Event,
        source: str,
    ) -> None:
        if event.utc_time[:10] != day:
            raise ValueError(f"事件 {event.event_id} 不在 {day}")
        # 同一任务内编号仍唯一：同号异值不能同时出现在一个任务里，
        # 但在不同任务中各存一份（作为不同的共享明细 + 各自的引用）。
        clash = conn.execute(
            "SELECT d.event_id FROM event_refs r JOIN event_details d ON d.id=r.detail_id "
            "WHERE r.run_id=? AND d.event_id=?",
            (run_id, event.event_id),
        ).fetchone()
        if clash is not None:
            raise ValueError(f"事件编号冲突: {event.event_id}")
        detail_id = self._insert_detail(conn, event.event_id, event.utc_time, event.magnitude)
        conn.execute(
            "INSERT INTO event_refs(run_id, day, detail_id, source, created_at) "
            "VALUES (?,?,?,?,?)",
            (run_id, day, detail_id, source, self._now()),
        )

    def _insert_legacy(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        event: Event,
        source: str,
    ) -> None:
        if event.utc_time[:10] != day:
            raise ValueError(f"事件 {event.event_id} 不在 {day}")
        try:
            conn.execute(
                "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, day, event.event_id, event.utc_time, event.magnitude, source),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"事件编号冲突: {event.event_id}") from exc

    def _store_event(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        event: Event,
        source: str,
    ) -> None:
        if self._use_shared(conn):
            self._insert_ref(conn, run_id, day, event, source)
        else:
            self._insert_legacy(conn, run_id, day, event, source)

    def complete_days(
        self,
        run_id: int,
        by_day: dict[str, Sequence[Event]],
        source: str = "api",
    ) -> None:
        """Persist a whole exhausted range and mark its days complete at once.

        The write is transactional: any conflict (for example the same event ID
        showing up on two days) rolls the range back so no day is half-recorded.
        Works against the legacy table while the migration is open and against
        the shared detail/ref store once it has finished.
        """
        with self._lock, self.connect() as conn:
            try:
                for day, events in by_day.items():
                    for e in events:
                        self._store_event(conn, run_id, day, e, source)
                    self._mark(conn, run_id, day, STATUS_COMPLETE)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def record_blocked_day(self, run_id: int, day: str, events: Sequence[Event]) -> None:
        """Store the first (full) page for a day that cannot be exhausted."""
        with self._lock, self.connect() as conn:
            try:
                for e in events:
                    self._store_event(conn, run_id, day, e, "api")
                self._mark(conn, run_id, day, STATUS_BLOCKED)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def _replace_shared(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        events: Sequence[Event],
        source: str,
    ) -> None:
        """Backfill in the shared store: replace only this task's day.

        Removing the refs drops the old details too once no other task
        references them (the trigger handles reclamation).
        """
        conn.execute(
            "DELETE FROM event_refs WHERE run_id=? AND day=?", (run_id, day)
        )
        for e in events:
            self._insert_ref(conn, run_id, day, e, source)

    @staticmethod
    def _replace_legacy(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        events: Sequence[Event],
        source: str,
    ) -> None:
        conn.execute("DELETE FROM events WHERE run_id=? AND day=?", (run_id, day))
        for e in events:
            try:
                conn.execute(
                    "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, day, e.event_id, e.utc_time, e.magnitude, source),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"事件编号已被其他日期使用: {e.event_id}") from exc

    def backfill_day(
        self,
        run_id: int,
        day: str,
        events: Sequence[Event],
        verify=None,
    ) -> None:
        """Atomically replace a blocked day's events and mark it complete.

        ``verify`` receives the already-collected events and may raise to abort;
        the whole operation rolls back in that case, preserving prior progress.
        Only this task's rows for that day are replaced; shared details used by
        other tasks are reclaimed only when their last reference disappears.
        """
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM day_status WHERE run_id=? AND day=?", (run_id, day)
            ).fetchone()
            if row is None:
                raise KeyError(day)
            if row["status"] != STATUS_BLOCKED:
                raise ValueError(f"{day} 不是待补档日期")
            existing = [
                Event(r["event_id"], r["utc_time"], r["magnitude"])
                for r in conn.execute(
                    "SELECT event_id, utc_time, magnitude FROM run_events "
                    "WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                    (run_id, day),
                )
            ]
            try:
                if verify is not None:
                    verify(existing)
                if self._use_shared(conn):
                    self._replace_shared(conn, run_id, day, events, "csv")
                else:
                    self._replace_legacy(conn, run_id, day, events, "csv")
                self._mark(conn, run_id, day, STATUS_COMPLETE)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def is_ready(self, run_id: int) -> bool:
        with self.connect() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=? AND status!='complete'",
                (run_id,),
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            return total > 0 and bad == 0
