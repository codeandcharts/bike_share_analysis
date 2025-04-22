#!/usr/bin/env python3
"""
Metro Bike Share ETL Pipeline
=============================

This script runs end‑to‑end:
  1. EXTRACT  – Unzip raw monthly .zip files into interim folders
  2. LOAD     – Read and concatenate CSVs plus station JSON
  3. TRANSFORM– Clean, impute, remove outliers, engineer features, drop unused cols
  4. LOAD     – Persist cleaned tables into DuckDB
  5. VALIDATE – Quick count check to ensure data loaded correctly
"""

import zipfile, shutil, logging, json, re
from pathlib import Path
from datetime import datetime

import pandas as pd
import duckdb
from geopy.distance import geodesic
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = Path(r"D:\portfolio_projects\bike_share_analysis\data\raw")
EXTRACT_DIR = Path(
    r"D:\portfolio_projects\bike_share_analysis\data\interim\extracted_csvs"
)
DUCKDB_PATH = Path(
    r"D:\portfolio_projects\bike_share_analysis\data\processed\metro_bike_share.duckdb"
)

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─── 1. EXTRACT ────────────────────────────────────────────────────────────────
def extract_all_zip_files():
    zips = [p for p in RAW_DATA_PATH.iterdir() if zipfile.is_zipfile(p)]
    logging.info(f"Found {len(zips)} ZIP files")
    for z in zips:
        base = z.stem.replace(".csv", "")
        m = re.search(r"(\d{4})[_-]?q([1-4])", base.lower())
        sub = f"{m.group(1)}_Q{m.group(2)}" if m else base
        dest = EXTRACT_DIR / sub
        dest.mkdir(parents=True, exist_ok=True)
        logging.info(f"Extracting {z.name} → {dest}")
        with zipfile.ZipFile(z, "r") as arch:
            arch.extractall(dest)


# ─── 2a. LOAD – Trips CSVs ────────────────────────────────────────────────────
def load_and_combine_csvs() -> pd.DataFrame:
    logging.info(f"Searching CSVs in {EXTRACT_DIR}")
    paths = [
        p
        for p in EXTRACT_DIR.rglob("*.csv")
        if p.is_file() and "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]
    logging.info(f"Found {len(paths)} CSV files")

    df_list = []
    for p in paths:
        try:
            tmp = pd.read_csv(p, low_memory=False)
            tmp.columns = tmp.columns.str.lower().str.strip()
            df_list.append(tmp)
            logging.info(f"Loaded {p.name} ({len(tmp):,} rows)")
        except Exception as e:
            logging.warning(f"Failed to load {p.name}: {e}")

    combined = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined trips: {len(combined):,} rows")

    desired = [
        "trip_id",
        "duration",
        "start_time",
        "end_time",
        "start_station",
        "start_lat",
        "start_lon",
        "end_station",
        "end_lat",
        "end_lon",
        "bike_id",
        "passholder_type",
        "trip_route_category",
        "bike_type",
        "plan_duration",
    ]
    available = [c for c in desired if c in combined.columns]
    missing = set(desired) - set(available)
    if missing:
        logging.warning(f"Missing expected columns: {missing}")

    return combined[available]


# ─── 2b. LOAD – Station JSON ──────────────────────────────────────────────────
def load_station_data() -> pd.DataFrame:
    path = RAW_DATA_PATH / "index.json"
    if not path.exists():
        logging.error("Station JSON not found!")
        return None

    logging.info(f"Loading stations from {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = []
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
        recs.append(
            {
                "station_id": p.get("kioskId"),
                "station_name": p.get("name"),
                "station_address": p.get("addressStreet"),
                "station_city": p.get("addressCity"),
                "station_state": p.get("addressState"),
                "station_zip": p.get("addressZipCode"),
                "station_lat": lat,
                "station_lon": lon,
                "total_docks": p.get("totalDocks"),
                "bikes_available": p.get("bikesAvailable"),
                "docks_available": p.get("docksAvailable"),
                "trikes_available": p.get("trikesAvailable"),
                "classic_bikes_available": p.get("classicBikesAvailable"),
                "smart_bikes_available": p.get("smartBikesAvailable"),
                "electric_bikes_available": p.get("electricBikesAvailable"),
                "is_event_based": p.get("isEventBased"),
                "is_virtual": p.get("isVirtual"),
                "is_visible": p.get("isVisible"),
                "is_archived": p.get("isArchived"),
                "kiosk_public_status": p.get("kioskPublicStatus"),
                "kiosk_status": p.get("kioskStatus"),
                "kiosk_connection_status": p.get("kioskConnectionStatus"),
                "kiosk_unresponsive_time": p.get("kioskUnresponsiveTime"),
                "kiosk_type": p.get("kioskType"),
                "open_time": p.get("openTime"),
                "close_time": p.get("closeTime"),
                "public_text": p.get("publicText"),
                "time_zone": p.get("timeZone"),
                "client_version": p.get("clientVersion"),
                "data_snapshot_time": datetime.now(),
            }
        )
    df = pd.DataFrame(recs)
    logging.info(f"Loaded {len(df)} stations")
    return df


# ─── 3a. TRANSFORM – Trips ─────────────────────────────────────────────────────
def transform_bike_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Transforming trip data…")

    # Convert to datetime
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    # Drop rows where start_time or end_time is missing
    df.dropna(subset=["start_time", "end_time"], inplace=True)

    num_cols = [
        "trip_id",
        "duration",
        "plan_duration",
        "start_station",
        "end_station",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop unused or deprecated columns
    df.drop(
        columns=["notes", "event_start", "event_end"], errors="ignore", inplace=True
    )

    # Fill numeric with median, object with "Unknown"
    df[df.select_dtypes("number").columns] = df.select_dtypes("number").fillna(
        df.median(numeric_only=True)
    )
    df[df.select_dtypes("object").columns] = df.select_dtypes("object").fillna(
        "Unknown"
    )

    # Remove IQR outliers
    for c in [
        "duration",
        "plan_duration",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[c] >= q1 - 1.5 * iqr) & (df[c] <= q3 + 1.5 * iqr)]

    # Feature engineering
    df["trip_hour"] = df["start_time"].dt.hour
    df["trip_dayofweek"] = df["start_time"].dt.dayofweek
    df["is_weekend"] = df["trip_dayofweek"].isin([5, 6])
    df["trip_date"] = df["start_time"].dt.date

    df["trip_distance_km"] = df.apply(
        lambda r: geodesic(
            (r["start_lat"], r["start_lon"]), (r["end_lat"], r["end_lon"])
        ).km
        if not pd.isnull(r["start_lat"]) and not pd.isnull(r["end_lat"])
        else 0,
        axis=1,
    )
    df["trip_speed_kmph"] = (df["trip_distance_km"] / df["duration"]) * 60
    df["trip_speed_kmph"].replace([np.inf, -np.inf], 0, inplace=True)

    df["duration_bin"] = pd.cut(
        df["duration"],
        bins=[0, 5, 15, 30, 60, 120, np.inf],
        labels=["<5m", "5-15m", "15-30m", "30-60m", "1-2h", ">2h"],
    )
    df["is_subscriber"] = df["passholder_type"].str.contains("Subscriber", na=False)

    return df


# ─── 3b. TRANSFORM – Stations ─────────────────────────────────────────────────
def transform_station_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Transforming station data…")

    drop_cols = [
        "station_address",
        "station_state",
        "trikes_available",
        "is_event_based",
        "is_virtual",
        "is_visible",
        "is_archived",
        "kiosk_public_status",
        "kiosk_status",
        "kiosk_connection_status",
        "kiosk_unresponsive_time",
        "kiosk_type",
        "open_time",
        "close_time",
        "public_text",
        "time_zone",
        "client_version",
        "data_snapshot_time",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    for c in ["station_name"]:
        df[c] = df[c].fillna(
            df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
        )

    for c in [
        "total_docks",
        "bikes_available",
        "docks_available",
        "classic_bikes_available",
        "smart_bikes_available",
        "electric_bikes_available",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median()).astype(int)

    return df


# ─── 4. LOAD into DuckDB ─────────────────────────────────────────────────────
def load_data_to_duckdb(trips_df: pd.DataFrame, stations_df: pd.DataFrame):
    con = duckdb.connect(str(DUCKDB_PATH), read_only=False)
    con.execute("DROP TABLE IF EXISTS trips")
    con.execute("DROP TABLE IF EXISTS stations")

    logging.info("Loading trips table")
    con.register("t", trips_df)
    con.execute("CREATE TABLE trips AS SELECT * FROM t")

    if stations_df is not None:
        logging.info("Loading stations table")
        con.register("s", stations_df)
        con.execute("CREATE TABLE stations AS SELECT * FROM s")

    logging.info("Creating view trips_with_stations")
    con.execute("""
        CREATE OR REPLACE VIEW trips_with_stations AS
        SELECT 
            t.*, 
            COALESCE(s.station_name, 'Unknown') AS start_station_name
        FROM trips t
        LEFT JOIN stations s
          ON t.start_station = s.station_id
    """)
    con.close()


# ─── 5. VALIDATION ────────────────────────────────────────────────────────────
def test_duckdb_query():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    trip_cnt = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    station_cnt = con.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    logging.info(f"Total trips in DuckDB: {trip_cnt:,}")
    logging.info(f"Total stations in DuckDB: {station_cnt:,}")
    con.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    logging.info("=== METRO BIKE SHARE ETL START ===")

    extract_all_zip_files()

    raw_trips = load_and_combine_csvs()
    if raw_trips is None or raw_trips.empty:
        logging.error("No trip data—exiting.")
        return

    trips_df = transform_bike_data(raw_trips)

    stations_df = load_station_data()
    if stations_df is not None:
        stations_df = transform_station_data(stations_df)

    load_data_to_duckdb(trips_df, stations_df)

    logging.info("ETL complete. Running validation…")
    test_duckdb_query()
    print(f"\n✔️  DuckDB database ready at: {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
