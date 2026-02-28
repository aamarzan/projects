#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_bi_pipeline.py
Premium BI ETL: CSV -> Clean -> MySQL (upsert) -> KPI -> Excel report

Install:
  pip install pandas sqlalchemy pymysql openpyxl python-dateutil

Run:
  python etl_bi_pipeline.py --config config.json
  python etl_bi_pipeline.py --config config.json --full-refresh
  python etl_bi_pipeline.py --config config.json --demo-data

Input CSV expected (can be changed in config):
  requests.csv columns (minimum):
    request_id, user_id, partner_id, status, amount, created_time, approved_time, updated_at
  events.csv columns (minimum):
    event_id, user_id, device_id, event_type, ip, timestamp, updated_at

Config example (config.json):
{
  "mysql": {
    "host": "127.0.0.1",
    "port": 3306,
    "database": "bi_demo",
    "username": "root",
    "password": "YOUR_PASSWORD"
  },
  "paths": {
    "input_dir": "data_in",
    "requests_csv": "requests.csv",
    "events_csv": "events.csv",
    "state_file": "state.json",
    "output_dir": "outputs"
  },
  "etl": {
    "timezone": "UTC",
    "reprocess_days": 2,
    "max_rows_preview": 2000
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dtparser
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    BigInteger,
    Integer,
    String,
    DateTime,
    Numeric,
    Float,
    Index,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine


# -----------------------------
# Logging
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# Config + State
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_state(state_path: str, state: dict) -> None:
    ensure_dir(os.path.dirname(state_path) or ".")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def parse_dt(x) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()):
        return None
    if isinstance(x, datetime):
        return x
    try:
        return dtparser.parse(str(x))
    except Exception:
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------
# DB (MySQL) Setup
# -----------------------------
def make_engine(mysql_cfg: dict) -> Engine:
    user = mysql_cfg["username"]
    pwd = mysql_cfg["password"]
    host = mysql_cfg["host"]
    port = mysql_cfg.get("port", 3306)
    db = mysql_cfg["database"]
    # pymysql driver
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def create_tables(engine: Engine) -> None:
    """
    Creates minimal star-schema + KPI tables if they don't exist.
    """
    meta = MetaData()

    # Dimensions
    dim_user = Table(
        "dim_user",
        meta,
        Column("user_id", BigInteger, primary_key=True),
        Column("created_at", DateTime, nullable=True),
        Column("segment", String(50), nullable=True),
        mysql_charset="utf8mb4",
    )

    dim_partner = Table(
        "dim_partner",
        meta,
        Column("partner_id", BigInteger, primary_key=True),
        Column("partner_type", String(50), nullable=True),
        Column("region", String(80), nullable=True),
        mysql_charset="utf8mb4",
    )

    dim_device = Table(
        "dim_device",
        meta,
        Column("device_id", String(128), primary_key=True),
        Column("os", String(50), nullable=True),
        Column("model", String(80), nullable=True),
        mysql_charset="utf8mb4",
    )

    # Facts
    fact_request = Table(
        "fact_request",
        meta,
        Column("request_id", BigInteger, primary_key=True),
        Column("user_id", BigInteger, nullable=False),
        Column("partner_id", BigInteger, nullable=False),
        Column("status", String(30), nullable=False),
        Column("amount", Numeric(12, 2), nullable=True),
        Column("created_time", DateTime, nullable=True),
        Column("approved_time", DateTime, nullable=True),
        Column("time_to_approve_sec", Float, nullable=True),
        Column("updated_at", DateTime, nullable=False),
        mysql_charset="utf8mb4",
    )
    Index("ix_fact_request_updated_at", fact_request.c.updated_at)

    fact_event = Table(
        "fact_event",
        meta,
        Column("event_id", BigInteger, primary_key=True),
        Column("user_id", BigInteger, nullable=False),
        Column("device_id", String(128), nullable=True),
        Column("event_type", String(60), nullable=False),
        Column("ip", String(64), nullable=True),
        Column("timestamp", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
        mysql_charset="utf8mb4",
    )
    Index("ix_fact_event_timestamp", fact_event.c.timestamp)
    Index("ix_fact_event_updated_at", fact_event.c.updated_at)

    # KPI table (daily partner)
    kpi_daily_partner = Table(
        "kpi_daily_partner",
        meta,
        Column("date_key", DateTime, primary_key=True),  # store as midnight UTC
        Column("partner_id", BigInteger, primary_key=True),
        Column("total_requests", Integer, nullable=False),
        Column("approved", Integer, nullable=False),
        Column("rejected", Integer, nullable=False),
        Column("approval_rate", Float, nullable=False),
        Column("avg_time_to_approve_sec", Float, nullable=True),
        Column("suspicious_count", Integer, nullable=False, default=0),
        Column("updated_at", DateTime, nullable=False),
        mysql_charset="utf8mb4",
    )

    meta.create_all(engine)
    logging.info("Ensured MySQL tables exist.")


# -----------------------------
# Extraction
# -----------------------------
def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def filter_incremental(df: pd.DataFrame, last_ts: Optional[datetime], updated_col: str) -> pd.DataFrame:
    if last_ts is None:
        return df.copy()
    # parse updated_at if needed
    if df[updated_col].dtype != "datetime64[ns, UTC]" and df[updated_col].dtype != "datetime64[ns]":
        df[updated_col] = df[updated_col].apply(parse_dt)
    return df[df[updated_col].notna() & (df[updated_col] > last_ts)].copy()


# -----------------------------
# Transform + Quality
# -----------------------------
@dataclass
class DQResult:
    ok: bool
    issues: List[str]


def standardize_requests(df: pd.DataFrame) -> pd.DataFrame:
    required = ["request_id", "user_id", "partner_id", "status", "created_time", "updated_at"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"requests.csv missing required column: {c}")

    df = df.copy()

    # Parse datetimes
    for col in ["created_time", "approved_time", "updated_at"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_dt)

    # Numeric amount
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Clean status
    df["status"] = df["status"].astype(str).str.upper().str.strip()

    # Compute time_to_approve_sec
    if "approved_time" in df.columns:
        df["time_to_approve_sec"] = (df["approved_time"] - df["created_time"]).dt.total_seconds()
        df.loc[df["time_to_approve_sec"] < 0, "time_to_approve_sec"] = None
    else:
        df["time_to_approve_sec"] = None

    # Dedup: keep latest updated_at per request_id
    df = df.sort_values("updated_at").drop_duplicates(subset=["request_id"], keep="last")

    return df


def standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    required = ["event_id", "user_id", "event_type", "timestamp", "updated_at"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"events.csv missing required column: {c}")

    df = df.copy()
    for col in ["timestamp", "updated_at"]:
        df[col] = df[col].apply(parse_dt)

    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()

    # Dedup: keep latest updated_at per event_id
    df = df.sort_values("updated_at").drop_duplicates(subset=["event_id"], keep="last")
    return df


def dq_checks_requests(df: pd.DataFrame) -> DQResult:
    issues = []

    # Not null checks
    for c in ["request_id", "user_id", "partner_id", "status", "updated_at"]:
        if df[c].isna().any():
            issues.append(f"Nulls found in requests.{c}")

    # Unique PK
    if df["request_id"].duplicated().any():
        issues.append("Duplicate request_id found after dedup (unexpected)")

    # Range sanity example
    if "amount" in df.columns:
        bad_amt = df["amount"].notna() & (df["amount"] < 0)
        if bad_amt.any():
            issues.append("Negative amount found in requests")

    return DQResult(ok=(len(issues) == 0), issues=issues)


def dq_checks_events(df: pd.DataFrame) -> DQResult:
    issues = []
    for c in ["event_id", "user_id", "event_type", "timestamp", "updated_at"]:
        if df[c].isna().any():
            issues.append(f"Nulls found in events.{c}")
    if df["event_id"].duplicated().any():
        issues.append("Duplicate event_id found after dedup (unexpected)")
    return DQResult(ok=(len(issues) == 0), issues=issues)


# -----------------------------
# Load (UPSERT)
# -----------------------------
def upsert_df(engine: Engine, table_name: str, df: pd.DataFrame, pk_cols: List[str]) -> int:
    """
    Efficient MySQL UPSERT using INSERT ... ON DUPLICATE KEY UPDATE.
    Requires PK in table schema.
    """
    if df.empty:
        return 0

    meta = MetaData()
    meta.reflect(bind=engine, only=[table_name])
    table = meta.tables[table_name]

    # Convert NaNs to None for SQLAlchemy
    df2 = df.where(pd.notnull(df), None)

    rows = df2.to_dict(orient="records")
    stmt = mysql_insert(table).values(rows)

    # Update all cols except PK
    update_cols = {c.name: stmt.inserted[c.name] for c in table.columns if c.name not in pk_cols}
    stmt = stmt.on_duplicate_key_update(**update_cols)

    with engine.begin() as conn:
        res = conn.execute(stmt)
        # MySQL may report "affected rows" as 2 for updated rows; keep it as info only.
        return int(res.rowcount or 0)


# -----------------------------
# KPI Computation
# -----------------------------
def compute_kpi_daily_partner(
    requests_df: pd.DataFrame,
    events_df: pd.DataFrame,
    reprocess_days: int,
) -> pd.DataFrame:
    """
    Produces one row per day per partner:
      total_requests, approved, rejected, approval_rate, avg_time_to_approve_sec, suspicious_count
    Suspicious_count here is a simple, explainable rule-based signal:
      user attempts (events) > 5 in a day (per user) contributes to suspicious_count for that partner-day
    """
    if requests_df.empty:
        return pd.DataFrame(
            columns=[
                "date_key",
                "partner_id",
                "total_requests",
                "approved",
                "rejected",
                "approval_rate",
                "avg_time_to_approve_sec",
                "suspicious_count",
                "updated_at",
            ]
        )

    # Determine date range to recompute: last N days based on max(updated_at/created_time)
    max_dt = requests_df["updated_at"].max() or utc_now()
    start_day = (max_dt - timedelta(days=reprocess_days)).date()

    req = requests_df.copy()
    req["date_key"] = req["created_time"].dt.floor("D")
    req = req[req["date_key"].notna()]
    req = req[req["date_key"].dt.date >= start_day]

    # Basic KPI
    grp = req.groupby(["date_key", "partner_id"], as_index=False).agg(
        total_requests=("request_id", "count"),
        approved=("status", lambda s: int((s == "APPROVED").sum())),
        rejected=("status", lambda s: int((s == "REJECTED").sum())),
        avg_time_to_approve_sec=("time_to_approve_sec", "mean"),
    )
    grp["approval_rate"] = grp["approved"] / grp["total_requests"]
    grp["avg_time_to_approve_sec"] = grp["avg_time_to_approve_sec"].fillna(0).astype(float)

    # Suspicious signal (simple): users with >5 events/day
    suspicious = pd.DataFrame(columns=["date_key", "user_id", "suspicious_user"])
    if not events_df.empty:
        ev = events_df.copy()
        ev["date_key"] = ev["timestamp"].dt.floor("D")
        ev = ev[ev["date_key"].notna()]
        ev = ev[ev["date_key"].dt.date >= start_day]
        ev_u = ev.groupby(["date_key", "user_id"], as_index=False).agg(cnt=("event_id", "count"))
        ev_u["suspicious_user"] = (ev_u["cnt"] > 5).astype(int)
        suspicious = ev_u[["date_key", "user_id", "suspicious_user"]]

    # Map suspicious users to partner-day via requests (user-partner relationship per day)
    if not suspicious.empty:
        req_user_partner = req[["date_key", "partner_id", "user_id"]].drop_duplicates()
        s_map = req_user_partner.merge(suspicious, on=["date_key", "user_id"], how="left")
        s_map["suspicious_user"] = s_map["suspicious_user"].fillna(0).astype(int)
        s_grp = s_map.groupby(["date_key", "partner_id"], as_index=False).agg(
            suspicious_count=("suspicious_user", "sum")
        )
        grp = grp.merge(s_grp, on=["date_key", "partner_id"], how="left")
    else:
        grp["suspicious_count"] = 0

    grp["suspicious_count"] = grp["suspicious_count"].fillna(0).astype(int)
    grp["updated_at"] = utc_now().replace(tzinfo=None)  # store naive in MySQL

    # Ensure columns order
    grp = grp[
        [
            "date_key",
            "partner_id",
            "total_requests",
            "approved",
            "rejected",
            "approval_rate",
            "avg_time_to_approve_sec",
            "suspicious_count",
            "updated_at",
        ]
    ]
    return grp


# -----------------------------
# Demo Data (Optional)
# -----------------------------
def generate_demo_data(n_requests: int = 5000, n_events: int = 20000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np

    now = utc_now().replace(tzinfo=None)
    start = now - timedelta(days=7)

    rng = np.random.default_rng(42)

    user_ids = rng.integers(1000, 2000, size=800)
    partner_ids = rng.integers(10, 30, size=20)

    # Requests
    request_ids = rng.integers(10_000_000, 99_999_999, size=n_requests)
    users = rng.choice(user_ids, size=n_requests)
    partners = rng.choice(partner_ids, size=n_requests)
    created = [start + timedelta(seconds=int(rng.integers(0, 7 * 24 * 3600))) for _ in range(n_requests)]
    statuses = rng.choice(["APPROVED", "REJECTED", "PENDING"], size=n_requests, p=[0.6, 0.25, 0.15])
    amount = rng.normal(5000, 1500, size=n_requests).clip(0, None)

    approved_time = []
    for c, st in zip(created, statuses):
        if st == "APPROVED":
            approved_time.append(c + timedelta(minutes=int(rng.integers(5, 240))))
        else:
            approved_time.append(None)

    updated = []
    for c in created:
        updated.append(c + timedelta(minutes=int(rng.integers(1, 600))))

    requests = pd.DataFrame(
        {
            "request_id": request_ids,
            "user_id": users,
            "partner_id": partners,
            "status": statuses,
            "amount": amount.round(2),
            "created_time": created,
            "approved_time": approved_time,
            "updated_at": updated,
        }
    )

    # Events
    event_ids = rng.integers(100_000_000, 999_999_999, size=n_events)
    ev_users = rng.choice(user_ids, size=n_events)
    device_ids = [f"dev-{int(x)}" for x in rng.integers(1, 1500, size=n_events)]
    event_types = rng.choice(["login", "otp", "doc_upload", "retry", "view", "submit"], size=n_events)
    timestamps = [start + timedelta(seconds=int(rng.integers(0, 7 * 24 * 3600))) for _ in range(n_events)]
    ip = [f"192.168.{int(rng.integers(0, 255))}.{int(rng.integers(1, 255))}" for _ in range(n_events)]
    ev_updated = [t + timedelta(minutes=int(rng.integers(1, 180))) for t in timestamps]

    events = pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": ev_users,
            "device_id": device_ids,
            "event_type": event_types,
            "ip": ip,
            "timestamp": timestamps,
            "updated_at": ev_updated,
        }
    )

    return requests, events


# -----------------------------
# Excel Export
# -----------------------------
def export_excel(
    out_path: str,
    kpi_df: pd.DataFrame,
    dq_report: pd.DataFrame,
    requests_preview: pd.DataFrame,
    events_preview: pd.DataFrame,
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        kpi_df.to_excel(writer, index=False, sheet_name="kpi_daily_partner")
        dq_report.to_excel(writer, index=False, sheet_name="data_quality")
        requests_preview.to_excel(writer, index=False, sheet_name="requests_preview")
        events_preview.to_excel(writer, index=False, sheet_name="events_preview")


# -----------------------------
# Main ETL
# -----------------------------
def main() -> int:
    setup_logging()

    ap = argparse.ArgumentParser(description="Premium BI ETL: CSV -> MySQL -> KPI -> Excel")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--full-refresh", action="store_true", help="Ignore watermark and reload all")
    ap.add_argument("--demo-data", action="store_true", help="Generate demo data instead of reading CSVs")
    args = ap.parse_args()

    cfg = load_config(args.config)

    paths = cfg["paths"]
    etl_cfg = cfg.get("etl", {})
    reprocess_days = int(etl_cfg.get("reprocess_days", 2))
    max_preview = int(etl_cfg.get("max_rows_preview", 2000))

    input_dir = paths["input_dir"]
    state_file = paths.get("state_file", "state.json")
    output_dir = paths.get("output_dir", "outputs")

    ensure_dir(output_dir)

    # State path relative to config folder (clean behavior on Windows)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    state_path = os.path.join(cfg_dir, state_file)
    state = read_state(state_path)

    last_requests_ts = None if args.full_refresh else parse_dt(state.get("requests_last_updated_at"))
    last_events_ts = None if args.full_refresh else parse_dt(state.get("events_last_updated_at"))

    # Extract
    if args.demo_data:
        logging.info("Generating demo data...")
        req_raw, ev_raw = generate_demo_data()
    else:
        req_path = os.path.join(cfg_dir, input_dir, paths["requests_csv"])
        ev_path = os.path.join(cfg_dir, input_dir, paths["events_csv"])
        logging.info(f"Reading: {req_path}")
        logging.info(f"Reading: {ev_path}")
        req_raw = read_csv_safe(req_path)
        ev_raw = read_csv_safe(ev_path)

    # Ensure updated_at exists before incremental filter
    if "updated_at" not in req_raw.columns or "updated_at" not in ev_raw.columns:
        raise ValueError("Both requests.csv and events.csv must have an 'updated_at' column for incremental loads.")

    # Incremental filter (raw)
    req_inc = filter_incremental(req_raw, last_requests_ts, "updated_at") if not args.full_refresh else req_raw.copy()
    ev_inc = filter_incremental(ev_raw, last_events_ts, "updated_at") if not args.full_refresh else ev_raw.copy()

    logging.info(f"Extracted requests rows: {len(req_inc)} (incremental)")
    logging.info(f"Extracted events rows:   {len(ev_inc)} (incremental)")

    # Transform
    req_clean = standardize_requests(req_inc) if not req_inc.empty else pd.DataFrame(columns=req_raw.columns)
    ev_clean = standardize_events(ev_inc) if not ev_inc.empty else pd.DataFrame(columns=ev_raw.columns)

    # DQ checks
    dq_rows = []
    dq_req = dq_checks_requests(req_clean) if not req_clean.empty else DQResult(True, [])
    dq_evt = dq_checks_events(ev_clean) if not ev_clean.empty else DQResult(True, [])

    dq_rows.append({"dataset": "requests", "ok": dq_req.ok, "issues": "; ".join(dq_req.issues) if dq_req.issues else ""})
    dq_rows.append({"dataset": "events", "ok": dq_evt.ok, "issues": "; ".join(dq_evt.issues) if dq_evt.issues else ""})
    dq_report = pd.DataFrame(dq_rows)

    if not dq_req.ok or not dq_evt.ok:
        logging.error("Data quality checks failed. Aborting load to avoid informational risk.")
        logging.error(dq_report.to_string(index=False))
        # Still export DQ report to Excel for review
        out_xlsx = os.path.join(output_dir, f"etl_report_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        export_excel(
            out_xlsx,
            kpi_df=pd.DataFrame(),
            dq_report=dq_report,
            requests_preview=req_clean.head(min(max_preview, len(req_clean))),
            events_preview=ev_clean.head(min(max_preview, len(ev_clean))),
        )
        logging.info(f"Exported failure report: {out_xlsx}")
        return 2

    # Load to MySQL
    engine = make_engine(cfg["mysql"])
    create_tables(engine)

    # Build minimal dims from incoming data
    dim_user_df = pd.DataFrame({"user_id": pd.unique(pd.concat([req_clean.get("user_id", pd.Series(dtype="int")),
                                                                ev_clean.get("user_id", pd.Series(dtype="int"))], ignore_index=True))})
    dim_user_df = dim_user_df.dropna().astype({"user_id": "int64"})
    dim_user_df["created_at"] = None
    dim_user_df["segment"] = None

    dim_partner_df = pd.DataFrame({"partner_id": pd.unique(req_clean["partner_id"])}) if not req_clean.empty else pd.DataFrame({"partner_id": []})
    if not dim_partner_df.empty:
        dim_partner_df = dim_partner_df.dropna().astype({"partner_id": "int64"})
        dim_partner_df["partner_type"] = None
        dim_partner_df["region"] = None

    dim_device_df = pd.DataFrame({"device_id": pd.unique(ev_clean["device_id"])}) if ("device_id" in ev_clean.columns and not ev_clean.empty) else pd.DataFrame({"device_id": []})
    if not dim_device_df.empty:
        dim_device_df = dim_device_df.dropna().astype({"device_id": "string"})
        dim_device_df["os"] = None
        dim_device_df["model"] = None

    # Upsert dims
    upsert_df(engine, "dim_user", dim_user_df, pk_cols=["user_id"])
    if not dim_partner_df.empty:
        upsert_df(engine, "dim_partner", dim_partner_df, pk_cols=["partner_id"])
    if not dim_device_df.empty:
        upsert_df(engine, "dim_device", dim_device_df, pk_cols=["device_id"])

    # Upsert facts
    # Ensure MySQL-friendly naive datetimes
    for col in ["created_time", "approved_time", "updated_at"]:
        if col in req_clean.columns:
            req_clean[col] = req_clean[col].apply(lambda d: d.replace(tzinfo=None) if isinstance(d, datetime) else d)
    for col in ["timestamp", "updated_at"]:
        if col in ev_clean.columns:
            ev_clean[col] = ev_clean[col].apply(lambda d: d.replace(tzinfo=None) if isinstance(d, datetime) else d)

    loaded_req = upsert_df(engine, "fact_request", req_clean[
        ["request_id", "user_id", "partner_id", "status", "amount", "created_time", "approved_time", "time_to_approve_sec", "updated_at"]
    ] if not req_clean.empty else req_clean, pk_cols=["request_id"])

    loaded_evt = upsert_df(engine, "fact_event", ev_clean[
        ["event_id", "user_id", "device_id", "event_type", "ip", "timestamp", "updated_at"]
    ] if not ev_clean.empty else ev_clean, pk_cols=["event_id"])

    logging.info(f"MySQL upsert fact_request rowcount: {loaded_req}")
    logging.info(f"MySQL upsert fact_event   rowcount: {loaded_evt}")

    # KPI recomputation:
    # Pro approach: use incoming (incremental) + reprocess window from DB for correctness
    # For simplicity, we recompute KPI from current incremental batch; in production you would pull last N days from DB.
    kpi_df = compute_kpi_daily_partner(req_clean, ev_clean, reprocess_days=reprocess_days)

    # Upsert KPI (PK: date_key + partner_id)
    if not kpi_df.empty:
        kpi_df["date_key"] = pd.to_datetime(kpi_df["date_key"]).dt.tz_localize(None)
        upsert_df(engine, "kpi_daily_partner", kpi_df, pk_cols=["date_key", "partner_id"])
        logging.info(f"Computed KPI rows: {len(kpi_df)}")
    else:
        logging.info("No KPI rows computed (no incremental requests).")

    # Update watermark state
    if not req_clean.empty:
        state["requests_last_updated_at"] = str(req_clean["updated_at"].max())
    if not ev_clean.empty:
        state["events_last_updated_at"] = str(ev_clean["updated_at"].max())
    write_state(state_path, state)
    logging.info(f"Updated state: {state_path}")

    # Export Excel
    out_xlsx = os.path.join(output_dir, f"etl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    export_excel(
        out_xlsx,
        kpi_df=kpi_df,
        dq_report=dq_report,
        requests_preview=req_clean.head(min(max_preview, len(req_clean))),
        events_preview=ev_clean.head(min(max_preview, len(ev_clean))),
    )
    logging.info(f"Exported Excel report: {out_xlsx}")

    logging.info("ETL completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
