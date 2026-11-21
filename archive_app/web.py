from __future__ import annotations

import csv
import io
import os
import threading

from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from .catalog import (
    MAX_SPAN_DAYS,
    MIGRATION_DONE,
    MIGRATION_FAILED,
    MIGRATION_PENDING,
    MIGRATION_RUNNING,
    STATUS_BLOCKED,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PENDING,
    Store,
    parse_day,
    source_from_env,
)
from .collector import collect
from .csv_import import CsvValidationError, parse_csv, verify_includes_collected

STATUS_LABELS = {
    STATUS_PENDING: "待查询",
    STATUS_COMPLETE: "已查全",
    STATUS_BLOCKED: "待补档",
    STATUS_FAILED: "待重试",
}

MIGRATION_LABELS = {
    MIGRATION_PENDING: "待整理",
    MIGRATION_RUNNING: "整理中",
    MIGRATION_FAILED: "上次失败，可重试",
    MIGRATION_DONE: "整理完成",
}

MIGRATION_BATCH_SIZE = 500


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def create_app(
    database: str | None = None,
    source=None,
    testing: bool = False,
    migrate_fail_after: int | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=database or os.environ.get("DATABASE", "/data/archive.db"),
        MAX_CONTENT_LENGTH=4 * 1024 * 1024,
        TESTING=testing,
        MIGRATE_FAIL_AFTER=migrate_fail_after
        if migrate_fail_after is not None
        else _int_env("MIGRATE_FAIL_AFTER"),
    )
    store = Store(app.config["DATABASE"])
    store.init()
    app.extensions["store"] = store
    app.extensions["source"] = source or source_from_env()
    work_lock = threading.Lock()

    def current_store() -> Store:
        return app.extensions["store"]

    def current_source():
        return app.extensions["source"]

    @app.template_filter("status_label")
    def status_label(value: str) -> str:
        return STATUS_LABELS.get(value, value)

    @app.template_filter("migration_label")
    def migration_label(value: str) -> str:
        return MIGRATION_LABELS.get(value, value)

    @app.get("/")
    def index():
        st: Store = current_store()
        return render_template(
            "index.html",
            runs=st.runs(),
            max_span=MAX_SPAN_DAYS,
            page_size=current_source().page_size(),
            legacy=st.legacy_status(),
        )

    @app.get("/legacy")
    def legacy_overview():
        st: Store = current_store()
        return render_template(
            "legacy.html",
            legacy=st.legacy_status(),
            labels=MIGRATION_LABELS,
        )

    @app.post("/legacy/organize")
    def legacy_organize():
        st: Store = current_store()
        if not work_lock.acquire(blocking=False):
            flash("已有采集、补档或整理操作在进行，请稍后重试", "warn")
            return redirect(url_for("legacy_overview"))
        try:
            before = st.legacy_status()
            status = st.run_migration_batch(
                budget=MIGRATION_BATCH_SIZE,
                fail_after=app.config["MIGRATE_FAIL_AFTER"],
            )
        except Exception as exc:
            # 中断/写入失败：已提交的批次和进度仍在，可直接重试。
            flash(f"整理中断（已保存的进度不受影响，可再次点击整理）：{exc}", "warn")
            return redirect(url_for("legacy_overview"))
        finally:
            work_lock.release()

        if status["status"] == MIGRATION_DONE and before["status"] != MIGRATION_DONE:
            flash(
                "旧库整理完成：旧记录与整理期间入库的记录均已纳入共享存储，"
                f"共 {status['moved_rows']} 条引用、{status['shared_details']} 份共享明细。",
                "info",
            )
        elif status["status"] == MIGRATION_DONE:
            flash("旧库已全部纳入共享存储，暂无待整理记录。", "info")
        else:
            flash(
                f"本批已整理 {status['moved_rows'] - before['moved_rows']} 条，"
                f"还剩 {status['legacy_pending']} 条待整理，可继续点击整理。",
                "info",
            )
        return redirect(url_for("legacy_overview"))

    @app.post("/runs")
    def create_run():
        st: Store = current_store()
        try:
            start = parse_day(request.form.get("start_date", ""))
            end = parse_day(request.form.get("end_date", ""))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        if end < start:
            flash("结束日期不能早于开始日期", "error")
            return redirect(url_for("index"))
        span = (end - start).days + 1
        if span > MAX_SPAN_DAYS:
            flash(f"一次最多选择 {MAX_SPAN_DAYS} 个连续日", "error")
            return redirect(url_for("index"))
        run_id = st.create_run(start, end)
        flash(f"已创建 {start} 至 {end} 的归档任务", "info")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<int:run_id>")
    def run_detail(run_id: int):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        days = st.days(run_id)
        counts = {status: 0 for status in STATUS_LABELS}
        for row in days:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return render_template(
            "run.html",
            run=run,
            days=days,
            counts=counts,
            labels=STATUS_LABELS,
            ready=st.is_ready(run_id),
            events=st.events(run_id),
        )

    @app.post("/runs/<int:run_id>/collect")
    def run_collect(run_id: int):
        st: Store = current_store()
        if st.get_run(run_id) is None:
            abort(404)
        if not work_lock.acquire(blocking=False):
            flash("已有采集或补档操作在进行，请稍后重试", "warn")
            return redirect(url_for("run_detail", run_id=run_id))
        try:
            summary = collect(st, current_source(), run_id)
            flash(
                f"本次采集：查全 {summary.completed} 天，待补档 {summary.blocked} 天，"
                f"待重试 {summary.failed} 天",
                "info" if not summary.failed else "warn",
            )
        finally:
            work_lock.release()
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<int:run_id>/days/<day>/upload")
    def upload_day(run_id: int, day: str):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        rows = [r for r in st.days(run_id) if r["day"] == day]
        if not rows:
            abort(404)
        if rows[0]["status"] != STATUS_BLOCKED:
            flash(f"{day} 不是待补档日期，无需上传", "warn")
            return redirect(url_for("run_detail", run_id=run_id))

        upload = request.files.get("csv")
        if upload is None or not upload.filename:
            flash("请选择 CSV 文件", "error")
            return redirect(url_for("run_detail", run_id=run_id))

        try:
            parsed = parse_csv(upload.read(), day)
            st.backfill_day(
                run_id,
                day,
                parsed.events,
                verify=lambda existing: verify_includes_collected(parsed, existing),
            )
        except CsvValidationError as exc:
            flash("补档文件校验失败，进度保持不变：" + "；".join(exc.errors), "error")
            return redirect(url_for("run_detail", run_id=run_id))
        except ValueError as exc:
            flash(f"补档失败，进度保持不变：{exc}", "error")
            return redirect(url_for("run_detail", run_id=run_id))

        flash(f"{day} 已补档，共 {len(parsed.events)} 条", "info")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<int:run_id>/download")
    def download_run(run_id: int):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        if not st.is_ready(run_id):
            flash("还有未查全或未补档的日期，暂不能下载完整目录", "error")
            return redirect(url_for("run_detail", run_id=run_id))
        events = st.events(run_id)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["event_id", "utc_time", "magnitude"])
        for row in events:
            writer.writerow([row["event_id"], row["utc_time"], f"{row['magnitude']:g}"])
        filename = f"catalog_{run['start_date']}_{run['end_date']}.csv"
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("404.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        flash("补档文件过大（上限 4 MB）", "error")
        return redirect(request.referrer or url_for("index"))

    return app
