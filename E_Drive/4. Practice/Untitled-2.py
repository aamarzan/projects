#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
basic_etl_demo.py
Interview-ready ETL demo:
1) Read CSV (requests + events)
2) Clean + deduplicate
3) Load into MySQL (replace tables for simplicity)
4) Build daily KPI by partner
5) Export Excel report

Install:
  pip install pandas sqlalchemy pymysql openpyxl
Run:
  python basic_etl_demo.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine


# -----------------------------
# 1) CONFIG (edit these)
# -----------------------------
INPUT_DIR = "data_in"
REQUESTS_CSV = "requests.csv"
EVENTS_CSV = "events.csv"

MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_DB = "bi_demo"
MYSQL_USER = "root"
MYSQL_PASS = "YOUR_PASSWORD"

OUTPUT_DIR = "outputs"
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "etl_report.xlsx")


# -----------------------------
# 2) HELPERS
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_engine():
    url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


# -----------------------------
# 3) EXTRACT (read CSV)
# -----------------------------
def extract_requests() -> pd.DataFrame:
    path = os.path.join(INPUT_DIR, REQUESTS_CSV)
    return pd.read_csv(path)


def extract_events() -> pd.DataFrame:
    path = os.path.join(INPUT_DIR, EVENTS_CSV)
    return pd.read_csv(path)


# -----------------------------
# 4) TRANSFORM (clean)
# -----------------------------
def transform_requests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
      request_id, user_id, partner_id, status, amount, created_time, approved_time, updated_at
    """
    df = df.copy()

    # Parse datetimes
    for col in ["created_time", "approved_time", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalize status
    df["status"] = df["status"].astype(str).str.upper().str.strip()

    # Amount numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Derived field: time_to_approve_sec
    if "approved_time" in df.columns:
        df["time_to_approve_sec"] = (df["approved_time"] - df["created_time"]).dt.total_seconds()
        df.loc[df["time_to_approve_sec"] < 0, "time_to_approve_sec"] = None
    else:
        df["time_to_approve_sec"] = None

    # Dedup: keep latest updated row per request_id
    df = df.sort_values("updated_at").drop_duplicates(subset=["request_id"], keep="last")

    # Basic key checks (drop rows missing keys)
    df = df.dropna(subset=["request_id", "user_id", "partner_id", "status", "updated_at"])

    return df


def transform_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
      event_id, user_id, device_id, event_type, timestamp, updated_at
    """
    df = df.copy()

    for col in ["timestamp", "updated_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()

    # Dedup: keep latest updated row per event_id
    df = df.sort_values("updated_at").drop_duplicates(subset=["event_id"], keep="last")

    # Drop missing keys
    df = df.dropna(subset=["event_id", "user_id", "event_type", "timestamp", "updated_at"])

    return df


# -----------------------------
# 5) LOAD (MySQL)
# -----------------------------
def load_to_mysql(engine, requests_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    """
    Interview-simple approach:
    - replace tables each run (easy demo)
    Production note: use upserts + watermark incremental loading.
    """
    requests_df.to_sql("fact_request", engine, if_exists="replace", index=False)
    events_df.to_sql("fact_event", engine, if_exists="replace", index=False)


# -----------------------------
# 6) KPI (daily partner summary)
# -----------------------------
def build_kpi_daily_partner(requests_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per day per partner:
      total_requests, approved, rejected, approval_rate, avg_time_to_approve_sec
    """
    df = requests_df.copy()
    df["date_key"] = df["created_time"].dt.floor("D")

    kpi = (
        df.groupby(["date_key", "partner_id"], as_index=False)
        .agg(
            total_requests=("request_id", "count"),
            approved=("status", lambda s: int((s == "APPROVED").sum())),
            rejected=("status", lambda s: int((s == "REJECTED").sum())),
            avg_time_to_approve_sec=("time_to_approve_sec", "mean"),
        )
    )

    kpi["approval_rate"] = kpi["approved"] / kpi["total_requests"]
    return kpi


# -----------------------------
# 7) EXPORT (Excel)
# -----------------------------
def export_excel(out_path: str, kpi: pd.DataFrame, req: pd.DataFrame, ev: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        kpi.to_excel(writer, index=False, sheet_name="kpi_daily_partner")
        req.head(2000).to_excel(writer, index=False, sheet_name="requests_preview")
        ev.head(2000).to_excel(writer, index=False, sheet_name="events_preview")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Starting basic ETL...")

    # Extract
    req_raw = extract_requests()
    ev_raw = extract_events()

    # Transform
    req = transform_requests(req_raw)
    ev = transform_events(ev_raw)

    # Load
    engine = make_engine()
    load_to_mysql(engine, req, ev)

    # KPI
    kpi = build_kpi_daily_partner(req)

    # Export
    export_excel(OUTPUT_XLSX, kpi, req, ev)

    print("Done.")
    print(f"MySQL tables: fact_request, fact_event")
    print(f"Excel report: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
