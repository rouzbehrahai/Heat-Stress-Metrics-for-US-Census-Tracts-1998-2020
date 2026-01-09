#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""technical_validation.py
Key features
------------
- Reuses existing logic for PRISM-day time alignment (12Z–12Z).
- Validates:
  * PRISM-aligned gridded meteorology vs ISD-Lite (NOAA)
  * ERA5-Land meteorology vs ISD-Lite (NOAA)
  * PRISM-aligned meteorology / heat-stress indices vs USCRN heat01 (NOAA)  [optional if files exist]
  * Radiation-driven UTCI (thermofeel) vs SURFRAD-derived station TMRT/UTCI  [optional if files exist]
- Writes per-window run artifacts under:
    <ROOT>/runs/<year>/<tag>/
  plus a combined table at:
    <ROOT>/tables/technical_validation_table.csv

----------------------------------
Many pipelines write heat-stress metrics (HI/WBGT/UTCI) in *long form* per PRISM-day:
    heatstress_long_prismday_YYYY-MM-DD.parquet

with schema:
    lat, lon, day, month, year, time, temp_C_used, rh_pct_used, HI_C, WBGT_C, UTCI_C

This script can read those files for USCRN HI/WBGT validation and SURFRAD UTCI validation
when you pass:
    --prefer-long-heatstress
(and optionally override the filename pattern with --heatstress-long-pattern).

IMPORTANT (duplicate station grid-cells)
---------------------------------------
When mapping stations to a grid, multiple stations can map to the same grid cell
(e.g., ERA grid is coarser; several stations can share one cell). Earlier versions
of this script built a {key -> station_index} dict directly from the station list.
If duplicate keys existed, the dict dropped duplicates, but kept "original" station
indices, which can exceed the dict length and cause IndexError during extraction.

This version fixes that by building a *unique-cell index* and then broadcasting
cell values back to station-level rows.

Dependencies
------------
Required: numpy, pandas, pyarrow, scipy, pyproj
Optional: requests (for USCRN listing), thermofeel (for UTCI-related tasks),
          tqdm (for nicer progress bars)
------------
Some of this script was written using an AI Jupiter Notebook plug-in 
Primary author: Rouzbeh Rahai
Development period: 2025-2026
"""  # noqa: E501

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError(
        "pyarrow is required. Install with: pip install pyarrow"
    ) from e


# ======================================================================================
# Logging / progress helpers
# ======================================================================================

def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _maybe_tqdm(it: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    """Use tqdm if available, else passthrough."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


# ======================================================================================
# Shared numeric helpers
# ======================================================================================

def _ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def infer_season(d: pd.Timestamp) -> str:
    m = int(pd.Timestamp(d).month)
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def format_window_label(start_day: pd.Timestamp, end_day: pd.Timestamp) -> str:
    """Human-ish label used in the output table."""
    s = pd.Timestamp(start_day)
    e = pd.Timestamp(end_day)
    season = infer_season(s)
    # If same month, compress day range
    if s.month == e.month:
        return f"{s.year} {s.strftime('%b')} {s.day:02d}–{e.day:02d} ({season})"
    return f"{s.year} {s.strftime('%b %d')}–{e.strftime('%b %d')} ({season})"


# -----------------------------
# PRISM-day time helpers (12Z–12Z windows)
# -----------------------------

def prismday_utc_times(D: pd.Timestamp) -> pd.DatetimeIndex:
    """24 hourly timestamps for PRISM-day D, defined as [D-12h, D+11h] in UTC."""
    D = _ts(D)
    start = D - pd.Timedelta(hours=12)
    return pd.date_range(start, periods=24, freq="h")


def prismday_window_utc(D: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open UTC time window for PRISM-day D: [D-12h, D+12h)."""
    D = _ts(D)
    return (D - pd.Timedelta(hours=12), D + pd.Timedelta(hours=12))


def make_day_index(start_day: str, end_day: str) -> pd.DatetimeIndex:
    s = _ts(start_day)
    e = _ts(end_day)
    if e < s:
        raise ValueError(f"end_day must be >= start_day. Got {s.date()} to {e.date()}")
    if s.year != e.year:
        raise ValueError(f"start_day and end_day must be in the same year. Got {s.date()} to {e.date()}")
    return pd.date_range(s, e, freq="D")


def run_tag(days: pd.DatetimeIndex) -> str:
    if len(days) == 1:
        return days[0].strftime("%Y-%m-%d")
    return f"{days[0].strftime('%Y-%m-%d')}_to_{days[-1].strftime('%Y-%m-%d')}"


# -----------------------------
# Vapor pressure helper (Buck equation) – consistent with your pipeline
# -----------------------------

def es_hPa(T_C: np.ndarray) -> np.ndarray:
    T = np.asarray(T_C, dtype=np.float64)
    out = np.empty_like(T, dtype=np.float64)
    w = T >= 0.0
    Tw = T[w]
    out[w] = 6.1121 * np.exp((18.678 - Tw / 234.5) * (Tw / (257.14 + Tw)))
    Ti = T[~w]
    out[~w] = 6.1115 * np.exp((23.036 - Ti / 333.7) * (Ti / (279.82 + Ti)))
    return out.astype(np.float32)


# ======================================================================================
# Download/cache helper
# ======================================================================================

def download_if_needed(url: str, dest: Path, retries: int = 1, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    last_err: Optional[Exception] = None
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(retries + 1):
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            log(f"Downloading: {url}")
            urllib.request.urlretrieve(url, tmp.as_posix())
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(dest)
                return dest
            raise RuntimeError(f"Downloaded file is empty: {url}")
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                continue
            raise last_err


# ======================================================================================
# STATIC GRID (build once) + KDTree mapping
# ======================================================================================

@dataclass
class SavedGridMeta:
    ll_decimals: int
    crs_src: str = "EPSG:4326"
    crs_dst: str = "EPSG:5070"
    n_points: int = 0
    source_parquet: str = ""


def build_keys(lat: np.ndarray, lon: np.ndarray, key_decimals: int) -> np.ndarray:
    key_scale = 10 ** int(key_decimals)
    key_mult = 1_000_000_000
    lat_i = np.round(lat.astype(np.float64) * key_scale).astype(np.int64)
    lon_i = np.round(lon.astype(np.float64) * key_scale).astype(np.int64)
    return lat_i * key_mult + lon_i


def read_latlon_unique(parquet_path: Path, ll_decimals: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["lat", "lon"])
    lat = np.round(df["lat"].to_numpy(np.float64), ll_decimals).astype(np.float32)
    lon = np.round(df["lon"].to_numpy(np.float64), ll_decimals).astype(np.float32)
    keys = build_keys(lat, lon, ll_decimals)
    _, first_idx = np.unique(keys, return_index=True)
    return lat[first_idx], lon[first_idx]


def project_to_meters(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer  # type: ignore
    except Exception as e:
        raise RuntimeError("pyproj is required. Install with: pip install pyproj") from e

    tfm = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tfm.transform(lon.astype(np.float64), lat.astype(np.float64))
    return x.astype(np.float32), y.astype(np.float32)


def save_grid(out_prefix: Path, lat: np.ndarray, lon: np.ndarray, x: np.ndarray, y: np.ndarray, meta: SavedGridMeta) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_prefix.with_suffix(".npz"), lat=lat, lon=lon, x=x, y=y)
    out_prefix.with_suffix(".json").write_text(json.dumps(meta.__dict__, indent=2))


def find_sample_model_parquet(work_dir: Path) -> Path:
    hits = sorted(work_dir.glob("t2m_prismday_*.parquet"))
    if not hits:
        raise FileNotFoundError(f"No model sample found in {work_dir} matching t2m_prismday_*.parquet")
    return hits[0]


def find_sample_era_parquet(era_base: Path) -> Path:
    hits = sorted((era_base / "t2m").glob("*/*.parquet"))
    if not hits:
        raise FileNotFoundError(f"No ERA sample found in {(era_base/'t2m')} matching */*.parquet")
    return hits[0]


@dataclass
class GridContext:
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ll_decimals: int
    tree: Any
    tfm: Any


def load_grid(prefix: Path) -> GridContext:
    try:
        from pyproj import Transformer  # type: ignore
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:
        raise RuntimeError("pyproj and scipy are required. Install with: pip install pyproj scipy") from e

    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing grid files: {npz_path} and/or {json_path}")

    npz = np.load(npz_path)
    meta = json.loads(json_path.read_text())

    lat = npz["lat"].astype(np.float32)
    lon = npz["lon"].astype(np.float32)
    x = npz["x"].astype(np.float32)
    y = npz["y"].astype(np.float32)

    tfm = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    tree = cKDTree(np.column_stack([x.astype(np.float64), y.astype(np.float64)]))

    return GridContext(lat=lat, lon=lon, x=x, y=y, ll_decimals=int(meta["ll_decimals"]), tree=tree, tfm=tfm)


def map_points_to_nearest(ctx: GridContext, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = ctx.tfm.transform(lon.astype(np.float64), lat.astype(np.float64))
    dist_m, idx = ctx.tree.query(np.column_stack([x, y]), k=1)
    return ctx.lat[idx].astype(np.float32), ctx.lon[idx].astype(np.float32), dist_m.astype(np.float32)


# ======================================================================================
# ISD-Lite: station metadata and observations
# ======================================================================================

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def load_isd_history(cache_dir: Path, download_retries: int = 1) -> pd.DataFrame:
    local = cache_dir / "isd-history.csv"
    download_if_needed(ISD_HISTORY_URL, local, retries=download_retries)
    return pd.read_csv(local, dtype=str)


def filter_isd_history(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    conus_only: bool = True,
) -> pd.DataFrame:
    """Filter ISD history for stations that cover [start_date, end_date] inclusive."""
    out = df.copy()
    out["LAT"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["LON"] = pd.to_numeric(out["LON"], errors="coerce")
    out["BEGIN"] = pd.to_datetime(out["BEGIN"], format="%Y%m%d", errors="coerce")
    out["END"] = pd.to_datetime(out["END"], format="%Y%m%d", errors="coerce")
    out["USAF"] = out["USAF"].astype(str).str.zfill(6)
    out["WBAN"] = out["WBAN"].astype(str).str.zfill(5)

    out = out[(out["CTRY"] == "US") & out["LAT"].notna() & out["LON"].notna()]
    out = out[(out["BEGIN"] <= start_date) & (out["END"] >= end_date)]

    if conus_only:
        out = out[
            (out["LAT"].between(24.0, 50.5)) &
            (out["LON"].between(-125.0, -66.0))
        ]
    out = out.reset_index(drop=True)
    out["station_id"] = out["USAF"].astype(str) + "-" + out["WBAN"].astype(str)
    return out


def pick_spatially_diverse_indices(lat: np.ndarray, lon: np.ndarray, n: int, seed: int = 7) -> List[int]:
    """Greedy farthest-point sampling in (lat,lon)."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([lat.astype(np.float64), lon.astype(np.float64)])
    if len(pts) == 0:
        return []
    first = int(rng.integers(0, len(pts)))
    chosen = [first]
    d2 = np.sum((pts - pts[first]) ** 2, axis=1)
    for _ in range(1, min(n, len(pts))):
        nxt = int(np.argmax(d2))
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
    return chosen


def isd_lite_url(usaf: str, wban: str, year: int) -> str:
    return f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"


def read_isd_lite_station_year(usaf: str, wban: str, year: int, cache_dir: Path, download_retries: int = 1) -> pd.DataFrame:
    url = isd_lite_url(usaf, wban, year)
    dest = cache_dir / "isd_lite" / f"{year}" / f"{usaf}-{wban}-{year}.gz"
    download_if_needed(url, dest, retries=download_retries)

    cols = [
        "year","month","day","hour",
        "temp","dew","slp","wdir","wspd","sky","precip1","precip6"
    ]
    with gzip.open(dest, "rt") as f:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=cols, engine="python")

    def mv(s: pd.Series) -> pd.Series:
        return s.replace(-9999, np.nan)

    for c in ["temp", "dew"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["temp_C_obs"] = mv(df["temp"]) / 10.0
    df["td_C_obs"] = mv(df["dew"]) / 10.0

    df["time_utc"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()

    ea = es_hPa(df["td_C_obs"].to_numpy(np.float32))
    es = es_hPa(df["temp_C_obs"].to_numpy(np.float32))
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = 100.0 * (ea / np.maximum(es, 1e-6))

    df["ea_hPa_obs"] = ea
    df["rh_pct_obs"] = np.clip(rh, 0.0, 100.0).astype(np.float32)

    out = df[["time_utc", "temp_C_obs", "ea_hPa_obs", "rh_pct_obs"]].copy()
    out = out.dropna(subset=["temp_C_obs"])
    return out


def load_isd_obs_for_stations(
    stations: pd.DataFrame,
    year: int,
    cache_dir: Path,
    max_workers: int,
    download_retries: int = 1,
) -> pd.DataFrame:
    """Download+parse ISD-Lite for a list of stations, in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = [stations.iloc[i] for i in range(len(stations))]
    out_list: List[pd.DataFrame] = []

    def _one(row: pd.Series) -> Optional[pd.DataFrame]:
        sid = row["station_id"]
        try:
            df_obs = read_isd_lite_station_year(row["USAF"], row["WBAN"], year, cache_dir, download_retries=download_retries)
            if df_obs is None or len(df_obs) == 0:
                return None
            df_obs["station_id"] = sid
            return df_obs
        except Exception as e:
            log(f"  [skip ISD obs] {sid} -> {type(e).__name__}: {e}")
            return None

    log(f"ISD-Lite: loading {len(rows):,} stations with max_workers={max_workers} ...")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, r): r["station_id"] for r in rows}
        for fut in as_completed(futs):
            df_obs = fut.result()
            done += 1
            if done % 25 == 0 or done == len(rows):
                log(f"  ISD-Lite progress: {done}/{len(rows)} stations")
            if df_obs is not None and len(df_obs) > 0:
                out_list.append(df_obs)

    log(f"ISD-Lite: finished in {time.time()-t0:.1f}s")
    return pd.concat(out_list, ignore_index=True) if out_list else pd.DataFrame()


# ======================================================================================
# USCRN heat01 (NOAA): loader (from cached yearly parquet, or build-on-demand)
# ======================================================================================

USCRN_HEAT01_BASE_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/heat01/"
USCRN_EXCLUDE_PREFIXES = ("AK_", "HI_", "PR_", "VI_", "GU_", "AS_", "MP_")

USCRN_ALL_COLS = [
    "WBANNO",
    "DATE_TIME",  # YYYYMMDDHH (UTC)
    "LONGITUDE",
    "LATITUDE",
    "RELATIVE_HUMIDITY",
    "SURFACE_PRESSURE",
    "SOLAR_RADIATION",
    "ESTIMATED_10_METER_WIND_SPEED",
    "DRY_BULB_TEMPERATURE_C",
    "HEAT_INDEX_C",
    "APPARENT_TEMPERATURE_C",
    "WET_BULB_GLOBE_TEMPERATURE_C",
    "DRY_BULB_TEMPERATURE_F",
    "HEAT_INDEX_F",
    "APPARENT_TEMPERATURE_F",
    "WET_BULB_GLOBE_TEMPERATURE_F",
]
USCRN_USECOLS = [
    "WBANNO", "DATE_TIME", "LATITUDE", "LONGITUDE",
    "DRY_BULB_TEMPERATURE_C", "RELATIVE_HUMIDITY",
    "HEAT_INDEX_C", "WET_BULB_GLOBE_TEMPERATURE_C",
    "SURFACE_PRESSURE", "SOLAR_RADIATION", "ESTIMATED_10_METER_WIND_SPEED",
]


def _http_get_text(url: str, timeout: int = 60) -> str:
    """requests if available, else urllib."""
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")


def list_uscrn_heat01_station_files_conus() -> List[str]:
    html = _http_get_text(USCRN_HEAT01_BASE_URL, timeout=60)
    files = sorted(set(re.findall(r'href="(CRNHE0101-[^"]+\.csv)"', html)))
    conus = []
    for f in files:
        station_name = f.replace("CRNHE0101-", "").replace(".csv", "")
        if not any(station_name.startswith(p) for p in USCRN_EXCLUDE_PREFIXES):
            conus.append(f)
    return conus


def read_uscrn_heat01_csv(url: str) -> pd.DataFrame:
    """Read one station CSV and return a tidy DataFrame."""
    df = pd.read_csv(
        url,
        header=None,
        names=USCRN_ALL_COLS,
        usecols=[USCRN_ALL_COLS.index(c) for c in USCRN_USECOLS],
        dtype="string",
        na_values=[-9999, -9999.0, "-9999", " -9999", "NA", ""],
        on_bad_lines="skip",
        low_memory=False,
    )

    # Drop embedded header rows if present
    df = df[df["WBANNO"] != "WBANNO"].copy()

    num_cols = [
        "LATITUDE", "LONGITUDE", "DRY_BULB_TEMPERATURE_C", "RELATIVE_HUMIDITY",
        "HEAT_INDEX_C", "WET_BULB_GLOBE_TEMPERATURE_C", "SURFACE_PRESSURE",
        "SOLAR_RADIATION", "ESTIMATED_10_METER_WIND_SPEED",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["WBANNO"] = pd.to_numeric(df["WBANNO"], errors="coerce").astype("Int32")

    # Parse UTC timestamp (tz-aware), then convert to naive UTC for merging consistency
    ts = pd.to_datetime(df["DATE_TIME"], format="%Y%m%d%H", errors="coerce", utc=True)
    df["time_utc"] = ts.dt.tz_convert(None)

    df = df.dropna(subset=[
        "WBANNO", "time_utc", "LATITUDE", "LONGITUDE",
        "DRY_BULB_TEMPERATURE_C", "RELATIVE_HUMIDITY",
        "HEAT_INDEX_C", "WET_BULB_GLOBE_TEMPERATURE_C",
    ])

    # CONUS bbox guard
    df = df[
        (df["LATITUDE"].between(24.0, 50.5)) &
        (df["LONGITUDE"].between(-125.0, -66.0))
    ]

    df = df.rename(columns={
        "LATITUDE": "LAT",
        "LONGITUDE": "LON",
        "DRY_BULB_TEMPERATURE_C": "temp_C_obs",
        "RELATIVE_HUMIDITY": "rh_pct_obs",
        "HEAT_INDEX_C": "HI_C_obs",
        "WET_BULB_GLOBE_TEMPERATURE_C": "WBGT_C_obs",
        "SURFACE_PRESSURE": "sp_hPa_obs",
        "SOLAR_RADIATION": "srad_Wm2_obs",
        "ESTIMATED_10_METER_WIND_SPEED": "wind10_mps_est_obs",
    })

    # Vapor pressure from T and RH
    ea = (df["rh_pct_obs"].to_numpy(np.float32) / 100.0) * es_hPa(df["temp_C_obs"].to_numpy(np.float32))
    df["ea_hPa_obs"] = ea.astype(np.float32)

    # station_id as string
    df["station_id"] = df["WBANNO"].astype(int).astype(str).str.zfill(5)

    # float32 for big data
    for c in ["LAT", "LON", "temp_C_obs", "rh_pct_obs", "ea_hPa_obs", "HI_C_obs", "WBGT_C_obs"]:
        df[c] = df[c].astype("float32", copy=False)

    return df[["station_id", "time_utc", "LAT", "LON", "temp_C_obs", "ea_hPa_obs", "rh_pct_obs", "HI_C_obs", "WBGT_C_obs"]]


def prepare_uscrn_year(
    year: int,
    out_dir: Path,
    max_workers: int,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return USCRN hourly obs for `year` (CONUS), caching to parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"USCRN_heat01_CONUS_hourly_{year:04d}.parquet"

    if out_path.exists() and not force_rebuild:
        log(f"USCRN: reading cached {out_path}")
        df = pd.read_parquet(out_path)
        # Ensure correct dtype for merging
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
        df["station_id"] = df["station_id"].astype(str)
        return df

    log(f"USCRN: building yearly parquet for {year} (this may take a while the first time) ...")
    files = list_uscrn_heat01_station_files_conus()
    log(f"USCRN: found {len(files)} CONUS station CSVs")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(fname: str) -> pd.DataFrame:
        url = USCRN_HEAT01_BASE_URL + fname
        df = read_uscrn_heat01_csv(url)
        df = df[df["time_utc"].dt.year == year]
        return df

    frames: List[pd.DataFrame] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, f): f for f in files}
        for fut in as_completed(futs):
            done += 1
            if done % 20 == 0 or done == len(files):
                log(f"  USCRN progress: {done}/{len(files)} station files")
            try:
                df = fut.result()
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                log(f"  [skip USCRN] {futs[fut]} -> {type(e).__name__}: {e}")

    if not frames:
        log(f"USCRN: no data found for year={year}")
        return pd.DataFrame(columns=["station_id","time_utc","LAT","LON","temp_C_obs","ea_hPa_obs","rh_pct_obs","HI_C_obs","WBGT_C_obs"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["station_id", "time_utc"]).reset_index(drop=True)

    out.to_parquet(out_path, index=False, engine="pyarrow")
    log(f"USCRN: wrote cache -> {out_path} | rows={len(out):,} | wall={time.time()-t0:.1f}s")
    return out


# ======================================================================================
# SURFRAD derived hourly TMRT/UTCI loader
# ======================================================================================

def load_surfrad_hourly_tmrt_utci(year: int, surfrad_dir: Path) -> pd.DataFrame:
    """Load precomputed SURFRAD hourly TMRT/UTCI parquet for year.

    Cleans:
      - standardizes timestamp column to `time_utc`
      - standardizes station id to `station_id`
      - drops rows with non-finite LAT/LON (prevents KDTree failures)
    """
    p = surfrad_dir / f"SURFRAD_hourly_TMRT_UTCI_{year:04d}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing SURFRAD parquet: {p}")

    df = pd.read_parquet(p)

    # Normalize timestamp
    if "timestamp_utc" in df.columns and "time_utc" not in df.columns:
        df = df.rename(columns={"timestamp_utc": "time_utc"})
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    # Normalize station id (your file has `station`)
    if "station_id" not in df.columns:
        if "station" in df.columns:
            df["station_id"] = df["station"].astype(str)
        else:
            raise KeyError(f"SURFRAD parquet missing station/station_id column: {p}")

    # Normalize obs columns
    if "TMRT_C_obs" not in df.columns and "TMRT_C" in df.columns:
        df = df.rename(columns={"TMRT_C": "TMRT_C_obs"})
    if "UTCI_C_obs" not in df.columns and "UTCI_C" in df.columns:
        df = df.rename(columns={"UTCI_C": "UTCI_C_obs"})

    # Normalize lat/lon to LAT/LON
    if "LAT" not in df.columns and "lat" in df.columns:
        df = df.rename(columns={"lat": "LAT"})
    if "LON" not in df.columns and "lon" in df.columns:
        df = df.rename(columns={"lon": "LON"})

    # Coerce and drop non-finite coords (e.g., station sxf has NaN for full year)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    m = np.isfinite(df["LAT"].to_numpy(np.float64)) & np.isfinite(df["LON"].to_numpy(np.float64))
    n_drop = int((~m).sum())
    if n_drop > 0:
        bad_stations = df.loc[~m, "station_id"].astype(str).value_counts().head(10).to_dict()
        log(f"[warn] SURFRAD: dropping {n_drop:,} rows with non-finite LAT/LON (top stations: {bad_stations})")
    df = df.loc[m].copy()

    # Keep only what we need downstream
    keep = ["station_id", "time_utc", "LAT", "LON", "TMRT_C_obs", "UTCI_C_obs"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"SURFRAD parquet missing required columns after normalization: {missing}")

    return df[keep].copy()

# ======================================================================================
# Parquet extractors (optimized vs pandas conversion)
# ======================================================================================

def _rowgroup_take(table: pa.Table, take_idx: np.ndarray) -> pa.Table:
    # pa.array is safest for pyarrow version differences
    return table.take(pa.array(take_idx.astype(np.int64)))


@dataclass
class StationKeyIndex:
    """Mapping from station rows to a compact unique-grid-cell index.

    - key_to_uix maps each (rounded) lat/lon key -> unique cell index [0..n_unique-1]
    - station_uix provides, for each station row, the unique cell index it maps to
      (so you can broadcast cell values back to station-level arrays).
    """
    key_to_uix: Dict[int, int]
    station_uix: np.ndarray  # shape (n_stations,)
    n_unique: int
    n_total: int


def build_station_key_index(
    stn_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    key_decimals: int,
    label: str = "",
) -> StationKeyIndex:
    """Build a compact key index and per-station broadcast index.

    Fixes the "duplicate grid-cells" problem:
      many stations -> fewer unique cells, but we still need station-level outputs.
    """
    lat = stn_df[lat_col].to_numpy(np.float32)
    lon = stn_df[lon_col].to_numpy(np.float32)

    keys = build_keys(lat, lon, key_decimals).astype(np.int64)

    if not np.isfinite(keys.astype(np.float64)).all():
        raise ValueError(
            f"Non-finite station grid keys encountered for {label or 'stations'} "
            f"(check {lat_col}/{lon_col} for NaNs)."
        )

    keys_unique, inv = np.unique(keys, return_inverse=True)
    key_to_uix = {int(k): int(i) for i, k in enumerate(keys_unique)}

    n_total = int(keys.size)
    n_unique = int(keys_unique.size)
    n_dupe = n_total - n_unique
    if n_dupe > 0:
        log(
            f"[note] {label or 'stations'}: {n_total:,} stations map to {n_unique:,} unique grid cells "
            f"({n_dupe:,} duplicate station->cell mappings). Broadcasting predictions to all stations in a cell."
        )

    return StationKeyIndex(
        key_to_uix=key_to_uix,
        station_uix=inv.astype(np.int64, copy=False),
        n_unique=n_unique,
        n_total=n_total,
    )


def extract_prismday_var_points(
    path: Path,
    prefix: str,
    key_to_ix: Dict[int, int],
    key_decimals: int,
) -> np.ndarray:
    """Extract PRISM-day hourly values for requested keys from a PRISM-day parquet.

    Returns array shaped (24, n_keys) where n_keys = len(key_to_ix).
    """
    n = len(key_to_ix)
    out = np.full((n, 24), np.nan, dtype=np.float32)

    pf = pq.ParquetFile(path.as_posix())
    hour_cols = [f"{prefix}_ph{h:02d}" for h in range(24)]
    names = pf.schema.names
    missing = [c for c in ["lat", "lon"] + hour_cols if c not in names]
    if missing:
        raise KeyError(f"{path.name}: missing columns: {missing[:12]}")  # pragma: no cover

    wanted = np.fromiter(key_to_ix.keys(), dtype=np.int64)

    for rg in range(pf.num_row_groups):
        tab_ll = pf.read_row_group(rg, columns=["lat", "lon"])
        lat = tab_ll["lat"].to_numpy(zero_copy_only=False).astype(np.float64)
        lon = tab_ll["lon"].to_numpy(zero_copy_only=False).astype(np.float64)
        keys = build_keys(lat, lon, key_decimals)

        mask = np.isin(keys, wanted, assume_unique=False)
        if not np.any(mask):
            continue

        take_idx = np.flatnonzero(mask).astype(np.int64)

        tab_vals = pf.read_row_group(rg, columns=hour_cols)
        tab_vals = _rowgroup_take(tab_vals, take_idx)

        # keys for selected rows (vectorized)
        keys_sel = keys[take_idx]
        ix = np.fromiter((key_to_ix.get(int(k), -1) for k in keys_sel), dtype=np.int64, count=len(keys_sel))
        good = ix >= 0
        if not np.any(good):
            continue
        ix = ix[good]

        # Build (n_selected, 24)
        cols_np = [tab_vals[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in hour_cols]
        vals = np.column_stack(cols_np)  # (n_selected, 24)
        vals = vals[good, :]

        out[ix, :] = vals

    return out.T  # (24,n)


def extract_era_day_points(
    path: Path,
    var: str,
    key_to_ix: Dict[int, int],
    key_decimals: int,
) -> np.ndarray:
    """Extract ERA hourly values for requested keys from a UTC-day parquet.

    Returns array shaped (24, n_keys) where n_keys = len(key_to_ix).
    """
    n = len(key_to_ix)
    out = np.full((n, 24), np.nan, dtype=np.float32)

    pf = pq.ParquetFile(path.as_posix())
    hour_cols = [f"{var}_h{h:02d}" for h in range(24)]
    names = pf.schema.names
    missing = [c for c in ["lat", "lon"] + hour_cols if c not in names]
    if missing:
        raise KeyError(f"{path.name}: missing columns: {missing[:12]}")  # pragma: no cover

    wanted = np.fromiter(key_to_ix.keys(), dtype=np.int64)

    for rg in range(pf.num_row_groups):
        tab_ll = pf.read_row_group(rg, columns=["lat", "lon"])
        lat = np.round(tab_ll["lat"].to_numpy(zero_copy_only=False).astype(np.float64), key_decimals)
        lon = np.round(tab_ll["lon"].to_numpy(zero_copy_only=False).astype(np.float64), key_decimals)
        keys = build_keys(lat, lon, key_decimals)

        mask = np.isin(keys, wanted, assume_unique=False)
        if not np.any(mask):
            continue

        take_idx = np.flatnonzero(mask).astype(np.int64)

        tab_vals = pf.read_row_group(rg, columns=hour_cols)
        tab_vals = _rowgroup_take(tab_vals, take_idx)

        keys_sel = keys[take_idx]
        ix = np.fromiter((key_to_ix.get(int(k), -1) for k in keys_sel), dtype=np.int64, count=len(keys_sel))
        good = ix >= 0
        if not np.any(good):
            continue
        ix = ix[good]

        cols_np = [tab_vals[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in hour_cols]
        vals = np.column_stack(cols_np)
        vals = vals[good, :]

        out[ix, :] = vals

    return out.T  # (24,n)


def combine_utc_days_to_prism_hours(arr_m1: np.ndarray, arr_d: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr_d)
    out[0:12, :] = arr_m1[12:24, :]
    out[12:24, :] = arr_d[0:12, :]
    return out


def era_day_path(era_base: Path, var: str, day_utc: pd.Timestamp) -> Path:
    day_utc = _ts(day_utc)
    year_dir = era_base / var / f"{day_utc.year:04d}"
    candidates = [
        year_dir / f"{day_utc.strftime('%m-%d-%Y')}.parquet",
        year_dir / f"{day_utc.strftime('%Y-%m-%d')}.parquet",
        year_dir / f"{day_utc.strftime('%Y-%m-%d')}_{var}.parquet",
        year_dir / f"{day_utc.strftime('%m-%d-%Y')}_{var}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing ERA file var={var} day={day_utc.date()} (tried {len(candidates)} patterns)")


# ======================================================================================
# Metrics + table row construction
# ======================================================================================

def stats(err: pd.Series) -> Dict[str, float]:
    e = err.dropna()
    if len(e) == 0:
        return {"N": 0.0, "bias": np.nan, "mae": np.nan, "rmse": np.nan}
    return {
        "N": float(len(e)),
        "bias": float(e.mean()),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
    }


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


VAR_LABELS = {
    "temp_C": "Air temperature (°C)",
    "ea_hPa": "Vapor pressure (hPa)",
    "rh_pct": "Relative humidity (%)",
    "HI_C": "Heat Index (°C)",
    "WBGT_C": "Wet-Bulb Globe Temperature (°C)",
    "TMRT_C": "Mean radiant temperature (°C)",
    "UTCI_C": "UTCI (°C)",
}


def write_metrics_suite(
    merged: pd.DataFrame,
    out_dir: Path,
    tag: str,
    prefix: str,
    variables: Sequence[str],
    obs_suffix: str = "_obs",
    pred_suffix: Optional[str] = None,
) -> None:
    """Write overall / by-station / by-day metric CSVs.

    merged is expected to have:
      - station_id
      - prism_day (string) (optional; only if you want by-day)
      - for each variable v in variables:
          f"{v}{obs_suffix}" and f"{v}{pred_suffix or '_' + prefix}"
    """
    pred_suffix = pred_suffix or f"_{prefix}"

    # overall
    overall_rows = []
    for v in variables:
        obs = f"{v}{obs_suffix}"
        pred = f"{v}{pred_suffix}"
        if obs not in merged.columns or pred not in merged.columns:
            continue
        s = stats(merged[pred] - merged[obs])
        overall_rows.append({
            "var": v,
            f"{prefix}_N": int(s["N"]),
            f"{prefix}_bias": s["bias"],
            f"{prefix}_mae": s["mae"],
            f"{prefix}_rmse": s["rmse"],
        })
    pd.DataFrame(overall_rows).to_csv(out_dir / f"metrics_overall_{prefix}_{tag}.csv", index=False)

    # by station
    rows = []
    if "station_id" in merged.columns:
        for sid, g in merged.groupby("station_id"):
            for v in variables:
                obs = f"{v}{obs_suffix}"
                pred = f"{v}{pred_suffix}"
                if obs not in g.columns or pred not in g.columns:
                    continue
                s = stats(g[pred] - g[obs])
                rows.append({
                    "station_id": sid,
                    "var": v,
                    f"{prefix}_N": int(s["N"]),
                    f"{prefix}_bias": s["bias"],
                    f"{prefix}_mae": s["mae"],
                    f"{prefix}_rmse": s["rmse"],
                })
    pd.DataFrame(rows).to_csv(out_dir / f"metrics_by_station_{prefix}_{tag}.csv", index=False)

    # by PRISM day (if present)
    rows = []
    if "prism_day" in merged.columns:
        for d, g in merged.groupby("prism_day"):
            for v in variables:
                obs = f"{v}{obs_suffix}"
                pred = f"{v}{pred_suffix}"
                if obs not in g.columns or pred not in g.columns:
                    continue
                s = stats(g[pred] - g[obs])
                rows.append({
                    "prism_day": str(d),
                    "var": v,
                    f"{prefix}_N": int(s["N"]),
                    f"{prefix}_bias": s["bias"],
                    f"{prefix}_mae": s["mae"],
                    f"{prefix}_rmse": s["rmse"],
                })
    pd.DataFrame(rows).to_csv(out_dir / f"metrics_by_day_{prefix}_{tag}.csv", index=False)


def build_table_rows_from_merged(
    merged: pd.DataFrame,
    stations_meta: pd.DataFrame,
    dist_col: str,
    window_label: str,
    window_start: str,
    window_end: str,
    product_name: str,
    reference_name: str,
    distance_filter_label: str,
    variables: Sequence[str],
    pred_prefix: str,
) -> List[Dict[str, Any]]:
    """Compute per-variable station counts, distance stats, bias & RMSE for the manuscript table."""
    rows: List[Dict[str, Any]] = []
    for v in variables:
        obs_col = f"{v}_obs"
        pred_col = f"{v}_{pred_prefix}"
        if obs_col not in merged.columns or pred_col not in merged.columns:
            continue

        m = merged[["station_id", obs_col, pred_col]].copy()
        m = m.dropna(subset=[obs_col, pred_col])
        n_hours = int(len(m))
        if n_hours == 0:
            continue

        station_ids = m["station_id"].astype(str).unique().tolist()
        stn = stations_meta[stations_meta["station_id"].astype(str).isin(station_ids)].copy()

        dist = pd.to_numeric(stn[dist_col], errors="coerce").to_numpy(np.float64)
        dist_med = float(np.nanmedian(dist)) if np.isfinite(dist).any() else float("nan")
        dist_p90 = safe_percentile(dist, 90.0)

        err = m[pred_col] - m[obs_col]
        s = stats(err)

        rows.append({
            "Temporal window": window_label,
            "Window start": window_start,
            "Window end": window_end,
            "Product": product_name,
            "Reference": reference_name,
            "Variable": VAR_LABELS.get(v, v),
            "Stations (N)": int(stn["station_id"].nunique()),
            "Station-hours matched (N)": int(s["N"]),
            "Distance filter": distance_filter_label,
            "Distance median (m)": dist_med,
            "Distance p90 (m)": dist_p90,
            "Bias": s["bias"],
            "RMSE": s["rmse"],
        })
    return rows


# ======================================================================================
# Prediction extractors for each product family
# ======================================================================================

def extract_model_prismday_predictions_from_long_heatstress(
    work_dir: Path,
    days: pd.DatetimeIndex,
    stations: pd.DataFrame,
    model_grid: GridContext,
    heatstress_long_pattern: str,
    day_workers: int = 1,
    require_cols: Sequence[str] = ("temp_C_used", "rh_pct_used", "HI_C", "WBGT_C", "UTCI_C"),
    skip_missing: bool = True,
) -> pd.DataFrame:
    """
    Extract model predictions from *long-form* PRISM-day heatstress parquet files:
      heatstress_long_prismday_{YYYY-MM-DD}.parquet

    File schema (confirmed):
      lat, lon, day, month, year, time, temp_C_used, rh_pct_used, HI_C, WBGT_C, UTCI_C

    Returns a long DataFrame with:
      station_id, time_utc, prism_day,
      temp_C_model, rh_pct_model, HI_C_model, WBGT_C_model, UTCI_C_model
    (Columns included depend on require_cols.)
    """
    if "lat_model" not in stations.columns or "lon_model" not in stations.columns:
        raise KeyError("stations must include lat_model and lon_model columns for long heatstress extraction.")

        # --- normalize heatstress_long_pattern ONCE (robust) ---
    pattern = heatstress_long_pattern
    if isinstance(pattern, Path):
        pattern = pattern.name
    pattern = str(pattern)

    # Guard against a common formatting mistake: "{date.parquet}" -> "{date}.parquet"
    if "{date.parquet}" in pattern:
        pattern = pattern.replace("{date.parquet}", "{date}.parquet")

    if "{date}" not in pattern:
        raise ValueError(
            f"heatstress_long_pattern must contain '{{date}}'. Got: {pattern!r}. "
            "Expected like: 'heatstress_long_prismday_{date}.parquet'"
        )

    log(f"[debug] long heatstress pattern={pattern!r} (type={type(pattern)})")
    log(f"[debug] sys.executable={sys.executable}")
    # -------------------------------------------------------

    # Helpful for nohup debugging / env mismatch
    log(f"[debug] long heatstress pattern={pattern!r} (type={type(pattern)})")
    log(f"[debug] sys.executable={sys.executable}")
    # -------------------------------------------------------

    # -------------------------------------------------------------

    # Build key -> list[station_id] mapping (handle multiple stations mapping to same grid cell)
    key_dec = int(model_grid.ll_decimals)
    stn_ids = stations["station_id"].astype(str).to_numpy()
    stn_keys = build_keys(
        stations["lat_model"].to_numpy(np.float32),
        stations["lon_model"].to_numpy(np.float32),
        key_decimals=key_dec,
    ).astype(np.int64)

    key_to_sids: Dict[int, List[str]] = defaultdict(list)
    for k, sid in zip(stn_keys, stn_ids):
        key_to_sids[int(k)].append(str(sid))

    wanted = np.fromiter(key_to_sids.keys(), dtype=np.int64)

    # Map file columns to output column names
    col_rename = {
        "temp_C_used": "temp_C_model",
        "rh_pct_used": "rh_pct_model",
        "HI_C": "HI_C_model",
        "WBGT_C": "WBGT_C_model",
        "UTCI_C": "UTCI_C_model",
    }

    def _one_day(D: pd.Timestamp) -> Optional[pd.DataFrame]:
        D = _ts(D)
        p = work_dir / pattern.format(date=D.strftime("%Y-%m-%d"))
        if not p.exists():
            if skip_missing:
                log(f"[skip day] missing long heatstress file: {p}")
                return None
            raise FileNotFoundError(str(p))

        pf = pq.ParquetFile(p.as_posix())
        need_ll = ["lat", "lon"]
        need_rest = ["year", "month", "day", "time"] + list(require_cols)

        names = pf.schema.names
        missing = [c for c in (need_ll + need_rest) if c not in names]
        if missing:
            raise KeyError(f"{p.name}: missing columns: {missing}")

        frames: List[pd.DataFrame] = []

        for rg in range(pf.num_row_groups):
            tab_ll = pf.read_row_group(rg, columns=need_ll)
            lat = tab_ll["lat"].to_numpy(zero_copy_only=False).astype(np.float64)
            lon = tab_ll["lon"].to_numpy(zero_copy_only=False).astype(np.float64)
            keys = build_keys(lat, lon, key_decimals=key_dec).astype(np.int64)

            mask = np.isin(keys, wanted, assume_unique=False)
            if not np.any(mask):
                continue

            take_idx = np.flatnonzero(mask).astype(np.int64)

            tab = pf.read_row_group(rg, columns=need_rest)
            tab = _rowgroup_take(tab, take_idx)

            keys_sel = keys[take_idx]

            # Arrays for time components
            y = tab["year"].to_numpy(zero_copy_only=False)
            m = tab["month"].to_numpy(zero_copy_only=False)
            d = tab["day"].to_numpy(zero_copy_only=False)
            hh = tab["time"].to_numpy(zero_copy_only=False)

            # Timestamp as naive UTC
            ts = pd.to_datetime(dict(year=y, month=m, day=d, hour=hh), errors="coerce")

            # Collect required value arrays
            vals: Dict[str, np.ndarray] = {}
            for c in require_cols:
                vals[c] = tab[c].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

            # Expand rows if multiple stations share a grid cell
            rep_idx: List[int] = []
            sid_out: List[str] = []
            for j, k in enumerate(keys_sel):
                sids = key_to_sids.get(int(k), [])
                if not sids:
                    continue
                for sid in sids:
                    sid_out.append(sid)
                    rep_idx.append(j)

            if not rep_idx:
                continue

            rep = np.asarray(rep_idx, dtype=np.int64)
            block: Dict[str, Any] = {
                "station_id": np.asarray(sid_out, dtype=object),
                "time_utc": ts.iloc[rep].to_numpy(),
                "prism_day": np.repeat(D.date().isoformat(), len(rep)),
            }
            for c in require_cols:
                out_c = col_rename.get(c, c)
                block[out_c] = vals[c][rep]

            df = pd.DataFrame(block)
            df = df.dropna(subset=["time_utc"]).copy()
            frames.append(df)

        if not frames:
            log(f"[warn] long heatstress: no matches for day={D.date()} file={p.name}")
            return None

        out = pd.concat(frames, ignore_index=True)
        return out

    # Run days sequentially or in parallel
    if day_workers <= 1 or len(days) <= 1:
        frames = []
        for D in _maybe_tqdm(days, total=len(days), desc="MODEL(long) days"):
            df = _one_day(D)
            if df is not None and len(df) > 0:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = {ex.submit(_one_day, D): str(D.date()) for D in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and len(df) > 0:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_model_prismday_predictions(
    work_dir: Path,
    days: pd.DatetimeIndex,
    stations: pd.DataFrame,
    model_grid: GridContext,
    model_file_pattern: str,
    need_vars: Sequence[str],
    var_prefix_map: Dict[str, str],
    day_workers: int = 1,
) -> pd.DataFrame:
    """Extract PRISM-day aligned model predictions for requested vars.

    Returns a long DataFrame with station_id, time_utc, prism_day, and columns:
      - temp_C_model, ea_hPa_model, rh_pct_model etc depending on need_vars.
    """
    station_ids = stations["station_id"].astype(str).to_numpy()

    # Build a compact unique-cell mapping; broadcast cell values back to station-level.
    idx = build_station_key_index(
        stations,
        lat_col="lat_model",
        lon_col="lon_model",
        key_decimals=model_grid.ll_decimals,
        label="MODEL",
    )
    key_to_ix = idx.key_to_uix
    station_uix = idx.station_uix

    def _one_day(D: pd.Timestamp) -> Optional[pd.DataFrame]:
        D = _ts(D)
        times = prismday_utc_times(D)
        n = len(station_ids)
        block_cols: Dict[str, Any] = {
            "station_id": np.repeat(station_ids, 24),
            "time_utc": np.tile(times.to_numpy(), n),
            "prism_day": np.repeat(D.date().isoformat(), 24 * n),
        }

        # Read each needed var parquet and extract (24,n_unique), then broadcast to (24,n)
        extracted: Dict[str, np.ndarray] = {}
        for var in need_vars:
            prefix = var_prefix_map.get(var, var)
            fname = model_file_pattern.format(var=var, date=D.strftime("%Y-%m-%d"))
            p = work_dir / fname
            if not p.exists():
                log(f"[skip day] missing MODEL file for var={var} day={D.date()} -> {p}")
                return None

            arr_u = extract_prismday_var_points(p, prefix=prefix, key_to_ix=key_to_ix, key_decimals=model_grid.ll_decimals)  # (24, n_unique)
            arr = arr_u[:, station_uix]  # (24, n_stations)
            extracted[var] = arr

        # Standard derived outputs for meteorology
        if "t2m" in extracted:
            Tm = extracted["t2m"].astype(np.float32)
            block_cols["temp_C_model"] = Tm.T.reshape(-1)

        if "ea" in extracted:
            EAm = extracted["ea"].astype(np.float32)
            block_cols["ea_hPa_model"] = EAm.T.reshape(-1)

        if ("t2m" in extracted) and ("ea" in extracted):
            Tm = extracted["t2m"].astype(np.float32)
            EAm = extracted["ea"].astype(np.float32)
            RHm = 100.0 * (EAm / np.maximum(es_hPa(Tm), 1e-6))
            RHm = np.clip(RHm.astype(np.float32), 0.0, 100.0)
            block_cols["rh_pct_model"] = RHm.T.reshape(-1)

        # Heat-stress indices if requested (wide files)
        if "hi" in extracted:
            block_cols["HI_C_model"] = extracted["hi"].astype(np.float32).T.reshape(-1)
        if "wbgt" in extracted:
            block_cols["WBGT_C_model"] = extracted["wbgt"].astype(np.float32).T.reshape(-1)

        # UTCI / TMRT if requested (wide files)
        if "tmrt" in extracted:
            block_cols["TMRT_C_model"] = extracted["tmrt"].astype(np.float32).T.reshape(-1)
        if "utci" in extracted:
            block_cols["UTCI_C_model"] = extracted["utci"].astype(np.float32).T.reshape(-1)

        return pd.DataFrame(block_cols)

    # Run days sequentially or in parallel
    if day_workers <= 1 or len(days) <= 1:
        frames = []
        for D in _maybe_tqdm(days, total=len(days), desc="MODEL days"):
            df = _one_day(D)
            if df is not None and len(df) > 0:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = {ex.submit(_one_day, D): str(D.date()) for D in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and len(df) > 0:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_era_prismday_predictions(
    era_base: Path,
    days: pd.DatetimeIndex,
    stations: pd.DataFrame,
    era_grid: GridContext,
    need_vars: Sequence[str] = ("t2m", "d2m"),
    day_workers: int = 1,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Extract ERA5-Land predictions aligned to PRISM-day hours for t2m/d2m."""
    station_ids = stations["station_id"].astype(str).to_numpy()

    # Compact unique-cell mapping + broadcast back to station-level.
    idx = build_station_key_index(
        stations,
        lat_col="lat_era",
        lon_col="lon_era",
        key_decimals=era_grid.ll_decimals,
        label="ERA",
    )
    key_to_ix = idx.key_to_uix
    station_uix = idx.station_uix

    def _one_day(D: pd.Timestamp) -> Optional[pd.DataFrame]:
        D = _ts(D)
        Dm1 = D - pd.Timedelta(days=1)
        times = prismday_utc_times(D)
        n = len(station_ids)

        block_cols: Dict[str, Any] = {
            "station_id": np.repeat(station_ids, 24),
            "time_utc": np.tile(times.to_numpy(), n),
            "prism_day": np.repeat(D.date().isoformat(), 24 * n),
        }

        try:
            p_t2m_m1 = era_day_path(era_base, "t2m", Dm1)
            p_t2m_d = era_day_path(era_base, "t2m", D)
            p_d2m_m1 = era_day_path(era_base, "d2m", Dm1)
            p_d2m_d = era_day_path(era_base, "d2m", D)
        except FileNotFoundError as e:
            if skip_missing:
                log(f"[skip day] ERA missing file for {D.date()}: {e}")
                return None
            raise

        # Extract to unique cells (24, n_unique)
        era_t2m_m1_u = extract_era_day_points(p_t2m_m1, "t2m", key_to_ix, era_grid.ll_decimals)
        era_t2m_d_u = extract_era_day_points(p_t2m_d, "t2m", key_to_ix, era_grid.ll_decimals)
        era_d2m_m1_u = extract_era_day_points(p_d2m_m1, "d2m", key_to_ix, era_grid.ll_decimals)
        era_d2m_d_u = extract_era_day_points(p_d2m_d, "d2m", key_to_ix, era_grid.ll_decimals)

        # Combine to PRISM-day hours (still unique cells)
        TeK_u = combine_utc_days_to_prism_hours(era_t2m_m1_u, era_t2m_d_u)
        TdK_u = combine_utc_days_to_prism_hours(era_d2m_m1_u, era_d2m_d_u)

        Te_u = (TeK_u - 273.15).astype(np.float32)
        Td_u = (TdK_u - 273.15).astype(np.float32)

        EAe_u = es_hPa(Td_u)
        RHe_u = 100.0 * (EAe_u / np.maximum(es_hPa(Te_u), 1e-6))
        RHe_u = np.clip(RHe_u.astype(np.float32), 0.0, 100.0)

        # Broadcast to station-level (24, n_stations)
        Te = Te_u[:, station_uix]
        Td = Td_u[:, station_uix]
        EAe = EAe_u[:, station_uix]
        RHe = RHe_u[:, station_uix]

        block_cols.update({
            "temp_C_era": Te.T.reshape(-1),
            "ea_hPa_era": EAe.T.reshape(-1),
            "rh_pct_era": RHe.T.reshape(-1),
        })

        return pd.DataFrame(block_cols)

    if day_workers <= 1 or len(days) <= 1:
        frames = []
        for D in _maybe_tqdm(days, total=len(days), desc="ERA days"):
            df = _one_day(D)
            if df is not None and len(df) > 0:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = {ex.submit(_one_day, D): str(D.date()) for D in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and len(df) > 0:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ======================================================================================
# Per-year context objects (so multiple windows reuse work)
# ======================================================================================

@dataclass
class ISDYearContext:
    year: int
    stations_all: pd.DataFrame  # includes mapping + distance flags
    obs_all: pd.DataFrame       # full-year obs for selected stations (or cached)
    model_grid: GridContext
    era_grid: GridContext


def prepare_isd_year(
    year: int,
    model_grid: GridContext,
    era_grid: GridContext,
    root: Path,
    cache_dir: Path,
    conus_only: bool,
    n_stations_target: Optional[int],
    station_seed: int,
    max_dist_model_m: float,
    max_dist_era_m: float,
    obs_workers: int,
    download_retries: int,
    coverage_mode: str,
    cache_obs_parquet: bool,
) -> ISDYearContext:
    """Prepare station metadata + load ISD-Lite obs once for the year."""
    # Determine station coverage window used to filter ISD-history.
    if coverage_mode == "full_year":
        start_cov = pd.Timestamp(year=year, month=1, day=1)
        end_cov = pd.Timestamp(year=year, month=12, day=31)
    else:
        raise ValueError("coverage_mode must be 'full_year' for now; window-coverage can be added if needed.")

    meta = load_isd_history(cache_dir, download_retries=download_retries)
    meta_y = filter_isd_history(meta, start_cov, end_cov, conus_only=conus_only)
    log(f"ISD-history candidates (coverage={coverage_mode}): {len(meta_y):,}")

    # Map to grids
    mlat, mlon, dist_m_model = map_points_to_nearest(model_grid, meta_y["LAT"].to_numpy(np.float32), meta_y["LON"].to_numpy(np.float32))
    elat, elon, dist_m_era = map_points_to_nearest(era_grid, meta_y["LAT"].to_numpy(np.float32), meta_y["LON"].to_numpy(np.float32))

    meta_y["lat_model"] = mlat
    meta_y["lon_model"] = mlon
    meta_y["dist_m_model"] = dist_m_model

    meta_y["lat_era"] = elat
    meta_y["lon_era"] = elon
    meta_y["dist_m_era"] = dist_m_era

    meta_y["keep_model_800m"] = meta_y["dist_m_model"] <= float(max_dist_model_m)
    meta_y["keep_era_9km"] = meta_y["dist_m_era"] <= float(max_dist_era_m)
    meta_y["keep_both"] = meta_y["keep_model_800m"] & meta_y["keep_era_9km"]

    log(f"Candidates within model<= {max_dist_model_m:.0f}m: {int(meta_y['keep_model_800m'].sum()):,}/{len(meta_y):,}")
    log(f"Candidates within ERA  <= {max_dist_era_m:.0f}m: {int(meta_y['keep_era_9km'].sum()):,}/{len(meta_y):,}")
    log(f"Candidates within BOTH: {int(meta_y['keep_both'].sum()):,}/{len(meta_y):,}")

    # Choose spatially diverse subset (optional)
    candidates = meta_y.copy()
    if n_stations_target is not None:
        order = pick_spatially_diverse_indices(
            candidates["LAT"].to_numpy(np.float64),
            candidates["LON"].to_numpy(np.float64),
            n=min(int(n_stations_target), len(candidates)),
            seed=int(station_seed),
        )
        candidates = candidates.iloc[order].reset_index(drop=True)

    log(f"ISD: selected {len(candidates):,} stations for download/obs parsing")

    # Cache obs parquet keyed by the station selection settings
    obs_cache_dir = root / "obs_cache"
    obs_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"isd_lite_obs_{year}_n{n_stations_target if n_stations_target is not None else 'all'}_seed{station_seed}_conus{int(conus_only)}"
    obs_cache_path = obs_cache_dir / f"{cache_key}.parquet"

    if cache_obs_parquet and obs_cache_path.exists():
        log(f"ISD-Lite: reading cached obs parquet -> {obs_cache_path}")
        obs = pd.read_parquet(obs_cache_path)
        obs["time_utc"] = pd.to_datetime(obs["time_utc"], errors="coerce")
        obs["station_id"] = obs["station_id"].astype(str)
    else:
        obs = load_isd_obs_for_stations(
            candidates,
            year=year,
            cache_dir=cache_dir,
            max_workers=obs_workers,
            download_retries=download_retries,
        )
        if len(obs) == 0:
            raise RuntimeError(f"No ISD-Lite observations loaded for year={year}.")

        if cache_obs_parquet:
            obs.to_parquet(obs_cache_path, index=False, engine="pyarrow")
            log(f"ISD-Lite: wrote obs cache -> {obs_cache_path}")

    log(f"ISD-Lite: obs rows={len(obs):,} | stations_with_obs={obs['station_id'].nunique():,}")
    return ISDYearContext(year=year, stations_all=candidates, obs_all=obs, model_grid=model_grid, era_grid=era_grid)


# ======================================================================================
# Validation runners per window
# ======================================================================================

def _filter_obs_to_window(obs: pd.DataFrame, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    t0, _ = prismday_window_utc(start_day)
    _, t1 = prismday_window_utc(end_day)
    out = obs[(obs["time_utc"] >= t0) & (obs["time_utc"] < t1)].copy()
    return out


def validate_isd_window(
    ctx: ISDYearContext,
    work_dir: Path,
    era_base: Path,
    out_dir: Path,
    days: pd.DatetimeIndex,
    window_label: str,
    model_file_pattern: str,
    model_var_prefix_map: Dict[str, str],
    day_workers: int,
    write_merged_parquet: bool,
    write_merged_csv: bool,
    include_paired: bool,
    max_dist_model_m: float,
    max_dist_era_m: float,
) -> List[Dict[str, Any]]:
    """Run ISD-Lite validations for one window; return table rows."""
    year = ctx.year
    tag = run_tag(days)

    # Filter obs to this window
    obs = _filter_obs_to_window(ctx.obs_all, days.min(), days.max())
    if len(obs) == 0:
        log(f"ISD-Lite: no obs in window {tag} (year={year}); skipping")
        return []

    sids_with_obs = set(obs["station_id"].astype(str).unique().tolist())
    stations = ctx.stations_all[ctx.stations_all["station_id"].astype(str).isin(sids_with_obs)].copy().reset_index(drop=True)

    # Save station list for this window
    stations_out = out_dir / f"stations_used_ISD_{tag}.csv"
    stations.to_csv(stations_out, index=False)
    log(f"Wrote {stations_out} | stations={len(stations):,}")

    # Partition by distance rules
    stations_model = stations[stations["keep_model_800m"]].reset_index(drop=True)
    stations_era = stations[stations["keep_era_9km"]].reset_index(drop=True)
    stations_both = stations[stations["keep_both"]].reset_index(drop=True)

    log(f"ISD window stations: model<= {max_dist_model_m:.0f}m: {len(stations_model):,} | ERA<= {max_dist_era_m:.0f}m: {len(stations_era):,} | both: {len(stations_both):,}")

    def obs_for(stn_df: pd.DataFrame) -> pd.DataFrame:
        sids = set(stn_df["station_id"].astype(str).tolist())
        return obs[obs["station_id"].astype(str).isin(sids)].copy()

    all_table_rows: List[Dict[str, Any]] = []

    # 1) PRISM-aligned gridded meteorology (model) vs ISD-Lite
    if len(stations_model) > 0:
        log("Extracting MODEL meteorology predictions for ISD-Lite validation...")
        df_pred = extract_model_prismday_predictions(
            work_dir=work_dir,
            days=days,
            stations=stations_model,
            model_grid=ctx.model_grid,
            model_file_pattern=model_file_pattern,
            need_vars=["t2m", "ea"],
            var_prefix_map=model_var_prefix_map,
            day_workers=day_workers,
        )
        merged = df_pred.merge(obs_for(stations_model), on=["station_id", "time_utc"], how="inner")
        if len(merged) == 0:
            log("MODEL vs ISD: merged is empty; skipping")
        else:
            if write_merged_parquet:
                merged.to_parquet(out_dir / f"merged_hourly_model_ISD_{tag}.parquet", index=False)
            if write_merged_csv:
                merged.to_csv(out_dir / f"merged_hourly_model_ISD_{tag}.csv", index=False)

            write_metrics_suite(merged, out_dir, tag, prefix="model_ISD", variables=["temp_C", "ea_hPa", "rh_pct"], pred_suffix="_model")
            log("Wrote MODEL vs ISD metrics")

            all_table_rows += build_table_rows_from_merged(
                merged=merged,
                stations_meta=stations_model,
                dist_col="dist_m_model",
                window_label=window_label,
                window_start=str(days.min().date()),
                window_end=str(days.max().date()),
                product_name="PRISM-aligned gridded meteorology",
                reference_name="ISD-Lite",
                distance_filter_label="≤800 m",
                variables=["temp_C", "ea_hPa", "rh_pct"],
                pred_prefix="model",
            )

    # 2) ERA5-Land meteorology vs ISD-Lite
    if len(stations_era) > 0:
        log("Extracting ERA meteorology predictions for ISD-Lite validation...")
        df_pred = extract_era_prismday_predictions(
            era_base=era_base,
            days=days,
            stations=stations_era,
            era_grid=ctx.era_grid,
            day_workers=day_workers,
            skip_missing=True,
        )
        merged = df_pred.merge(obs_for(stations_era), on=["station_id", "time_utc"], how="inner")
        if len(merged) == 0:
            log("ERA vs ISD: merged is empty; skipping")
        else:
            if write_merged_parquet:
                merged.to_parquet(out_dir / f"merged_hourly_era_ISD_{tag}.parquet", index=False)
            if write_merged_csv:
                merged.to_csv(out_dir / f"merged_hourly_era_ISD_{tag}.csv", index=False)

            write_metrics_suite(merged, out_dir, tag, prefix="era_ISD", variables=["temp_C", "ea_hPa", "rh_pct"], pred_suffix="_era")
            log("Wrote ERA vs ISD metrics")

            all_table_rows += build_table_rows_from_merged(
                merged=merged,
                stations_meta=stations_era,
                dist_col="dist_m_era",
                window_label=window_label,
                window_start=str(days.min().date()),
                window_end=str(days.max().date()),
                product_name="ERA5-Land meteorology",
                reference_name="ISD-Lite",
                distance_filter_label="≤9 km",
                variables=["temp_C", "ea_hPa", "rh_pct"],
                pred_prefix="era",
            )

    # 3) Paired comparison (optional; not part of the manuscript table by default)
    if include_paired and (len(stations_both) > 0):
        log("Computing paired comparison (MODEL vs ERA) on shared ISD-Lite stations...")
        df_model = extract_model_prismday_predictions(
            work_dir=work_dir,
            days=days,
            stations=stations_both,
            model_grid=ctx.model_grid,
            model_file_pattern=model_file_pattern,
            need_vars=["t2m", "ea"],
            var_prefix_map=model_var_prefix_map,
            day_workers=day_workers,
        )
        df_era = extract_era_prismday_predictions(
            era_base=era_base,
            days=days,
            stations=stations_both,
            era_grid=ctx.era_grid,
            day_workers=day_workers,
            skip_missing=True,
        )
        df_both = df_model.merge(df_era, on=["station_id", "time_utc", "prism_day"], how="inner")
        merged = df_both.merge(obs_for(stations_both), on=["station_id", "time_utc"], how="inner")

        need = [
            "temp_C_obs", "ea_hPa_obs", "rh_pct_obs",
            "temp_C_model", "ea_hPa_model", "rh_pct_model",
            "temp_C_era", "ea_hPa_era", "rh_pct_era",
        ]
        merged = merged.dropna(subset=[c for c in need if c in merged.columns]).copy()
        if len(merged) == 0:
            log("Paired: merged empty after dropna; skipping paired metrics")
        else:
            rows = []
            for v in ["temp_C", "ea_hPa", "rh_pct"]:
                err_m = merged[f"{v}_model"] - merged[f"{v}_obs"]
                err_e = merged[f"{v}_era"] - merged[f"{v}_obs"]
                sm = stats(err_m)
                se = stats(err_e)
                rows.append({
                    "var": v,
                    "paired_N": int(sm["N"]),
                    "model_bias": sm["bias"], "model_mae": sm["mae"], "model_rmse": sm["rmse"],
                    "era_bias": se["bias"], "era_mae": se["mae"], "era_rmse": se["rmse"],
                    "rmse_era_minus_model": se["rmse"] - sm["rmse"],
                })
            pd.DataFrame(rows).to_csv(out_dir / f"metrics_overall_paired_ISD_{tag}.csv", index=False)
            log("Wrote paired comparison metrics (MODEL vs ERA)")

    return all_table_rows


def validate_uscrn_window(
    year: int,
    uscrn_df_year: pd.DataFrame,
    model_grid: GridContext,
    work_dir: Path,
    out_dir: Path,
    days: pd.DatetimeIndex,
    window_label: str,
    model_file_pattern: str,
    model_var_prefix_map: Dict[str, str],
    day_workers: int,
    write_merged_parquet: bool,
    write_merged_csv: bool,
    max_dist_model_m: float,
    run_met: bool,
    run_hsi: bool,
    prefer_long_heatstress: bool,
    heatstress_long_pattern: str,
) -> List[Dict[str, Any]]:
    """Validate PRISM-aligned products vs USCRN heat01 for one window."""
    tag = run_tag(days)

    # Filter obs to window union (PRISM-day union)
    obs = _filter_obs_to_window(uscrn_df_year, days.min(), days.max())
    if len(obs) == 0:
        log(f"USCRN: no obs in window {tag} (year={year}); skipping")
        return []

    # Station meta
    stn = obs.groupby("station_id", as_index=False).agg({"LAT": "first", "LON": "first"})
    stn["station_id"] = stn["station_id"].astype(str)

    # Map to model grid
    mlat, mlon, dist_m = map_points_to_nearest(model_grid, stn["LAT"].to_numpy(np.float32), stn["LON"].to_numpy(np.float32))
    stn["lat_model"] = mlat
    stn["lon_model"] = mlon
    stn["dist_m_model"] = dist_m
    stn["keep_model_800m"] = stn["dist_m_model"] <= float(max_dist_model_m)

    stn_kept = stn[stn["keep_model_800m"]].copy().reset_index(drop=True)
    if len(stn_kept) == 0:
        log("USCRN: no stations within model distance threshold; skipping")
        return []

    # Save station list
    stn_path = out_dir / f"stations_used_USCRN_{tag}.csv"
    stn_kept.to_csv(stn_path, index=False)
    log(f"Wrote {stn_path} | stations={len(stn_kept):,}")

    # Restrict obs to kept stations
    obs = obs[obs["station_id"].astype(str).isin(set(stn_kept["station_id"].tolist()))].copy()

    table_rows: List[Dict[str, Any]] = []

    # 1) PRISM-aligned gridded meteorology vs USCRN (wide t2m/ea files)
    if run_met:
        log("USCRN: extracting model meteorology predictions (t2m, ea) ...")
        pred_met = extract_model_prismday_predictions(
            work_dir=work_dir,
            days=days,
            stations=stn_kept.assign(
                # fields expected by extractor
                lat_model=stn_kept["lat_model"].to_numpy(np.float32),
                lon_model=stn_kept["lon_model"].to_numpy(np.float32),
            ),
            model_grid=model_grid,
            model_file_pattern=model_file_pattern,
            need_vars=["t2m", "ea"],
            var_prefix_map=model_var_prefix_map,
            day_workers=day_workers,
        )

        if pred_met is not None and len(pred_met) > 0:
            merged = pred_met.merge(obs, on=["station_id", "time_utc"], how="inner")
            if len(merged) == 0:
                log("USCRN met: merged empty; skipping")
            else:
                if write_merged_parquet:
                    merged.to_parquet(out_dir / f"merged_hourly_model_USCRN_met_{tag}.parquet", index=False)
                if write_merged_csv:
                    merged.to_csv(out_dir / f"merged_hourly_model_USCRN_met_{tag}.csv", index=False)

                write_metrics_suite(merged, out_dir, tag, prefix="model_USCRN_met", variables=["temp_C", "ea_hPa", "rh_pct"], pred_suffix="_model")
                table_rows += build_table_rows_from_merged(
                    merged=merged,
                    stations_meta=stn_kept,
                    dist_col="dist_m_model",
                    window_label=window_label,
                    window_start=str(days.min().date()),
                    window_end=str(days.max().date()),
                    product_name="PRISM-aligned gridded meteorology",
                    reference_name="USCRN (heat01)",
                    distance_filter_label="≤800 m",
                    variables=["temp_C", "ea_hPa", "rh_pct"],
                    pred_prefix="model",
                )

    # 2) PRISM-aligned heat-stress indices vs USCRN
    if run_hsi:
        if prefer_long_heatstress:
            log("USCRN hsi: using long heatstress_long_prismday_* files for HI/WBGT ...")
            pred_hsi = extract_model_prismday_predictions_from_long_heatstress(
                work_dir=work_dir,
                days=days,
                stations=stn_kept.assign(
                    lat_model=stn_kept["lat_model"].to_numpy(np.float32),
                    lon_model=stn_kept["lon_model"].to_numpy(np.float32),
                ),
                model_grid=model_grid,
                heatstress_long_pattern=heatstress_long_pattern,
                day_workers=day_workers,
                require_cols=("temp_C_used", "rh_pct_used", "HI_C", "WBGT_C"),
                skip_missing=True,
            )
        else:
            # Fallback: wide hi/wbgt files if they exist
            need_vars = ["t2m", "ea"]  # for temp/rh derived
            need_vars_extra = []
            for v in ["hi", "wbgt"]:
                test_fname = model_file_pattern.format(var=v, date=days.min().strftime("%Y-%m-%d"))
                if (work_dir / test_fname).exists():
                    need_vars_extra.append(v)
                else:
                    log(f"USCRN hsi: missing {v} files (e.g., {test_fname}); will skip {v} metrics")
            need_vars = need_vars + need_vars_extra

            pred_hsi = extract_model_prismday_predictions(
                work_dir=work_dir,
                days=days,
                stations=stn_kept.assign(
                    lat_model=stn_kept["lat_model"].to_numpy(np.float32),
                    lon_model=stn_kept["lon_model"].to_numpy(np.float32),
                ),
                model_grid=model_grid,
                model_file_pattern=model_file_pattern,
                need_vars=need_vars,
                var_prefix_map=model_var_prefix_map,
                day_workers=day_workers,
            )

        if pred_hsi is None or len(pred_hsi) == 0:
            log("USCRN hsi: predictions empty; skipping")
        else:
            merged = pred_hsi.merge(obs, on=["station_id", "time_utc"], how="inner")
            if len(merged) == 0:
                log("USCRN hsi: merged empty; skipping")
            else:
                if write_merged_parquet:
                    merged.to_parquet(out_dir / f"merged_hourly_model_USCRN_hsi_{tag}.parquet", index=False)
                if write_merged_csv:
                    merged.to_csv(out_dir / f"merged_hourly_model_USCRN_hsi_{tag}.csv", index=False)

                vars_avail = ["temp_C", "rh_pct", "HI_C", "WBGT_C"]
                write_metrics_suite(merged, out_dir, tag, prefix="model_USCRN_hsi", variables=vars_avail, pred_suffix="_model")
                table_rows += build_table_rows_from_merged(
                    merged=merged,
                    stations_meta=stn_kept,
                    dist_col="dist_m_model",
                    window_label=window_label,
                    window_start=str(days.min().date()),
                    window_end=str(days.max().date()),
                    product_name="PRISM-aligned heat-stress indices",
                    reference_name="USCRN (heat01)",
                    distance_filter_label="≤800 m",
                    variables=vars_avail,
                    pred_prefix="model",
                )

    return table_rows


def validate_surfrad_window(
    year: int,
    surfrad_df_year: pd.DataFrame,
    model_grid: GridContext,
    work_dir: Path,
    out_dir: Path,
    days: pd.DatetimeIndex,
    window_label: str,
    model_file_pattern: str,
    model_var_prefix_map: Dict[str, str],
    day_workers: int,
    write_merged_parquet: bool,
    write_merged_csv: bool,
    max_dist_model_m: float,
    prefer_long_heatstress: bool,
    heatstress_long_pattern: str,
) -> List[Dict[str, Any]]:
    """Validate radiation-driven UTCI/TMRT vs SURFRAD station-derived TMRT/UTCI."""
    tag = run_tag(days)

    obs = _filter_obs_to_window(surfrad_df_year, days.min(), days.max())
    if len(obs) == 0:
        log(f"SURFRAD: no obs in window {tag} (year={year}); skipping")
        return []

    stn = obs.groupby("station_id", as_index=False).agg({"LAT": "first", "LON": "first"})
    stn["station_id"] = stn["station_id"].astype(str)

    mlat, mlon, dist_m = map_points_to_nearest(model_grid, stn["LAT"].to_numpy(np.float32), stn["LON"].to_numpy(np.float32))
    stn["lat_model"] = mlat
    stn["lon_model"] = mlon
    stn["dist_m_model"] = dist_m
    stn["keep_model_800m"] = stn["dist_m_model"] <= float(max_dist_model_m)

    stn_kept = stn[stn["keep_model_800m"]].copy().reset_index(drop=True)
    if len(stn_kept) == 0:
        log("SURFRAD: no stations within model distance threshold; skipping")
        return []

    stn_path = out_dir / f"stations_used_SURFRAD_{tag}.csv"
    stn_kept.to_csv(stn_path, index=False)
    log(f"Wrote {stn_path} | stations={len(stn_kept):,}")

    obs = obs[obs["station_id"].astype(str).isin(set(stn_kept["station_id"].tolist()))].copy()

    if prefer_long_heatstress:
        log("SURFRAD: using long heatstress_long_prismday_* files for UTCI ... (TMRT not in long schema)")
        pred = extract_model_prismday_predictions_from_long_heatstress(
            work_dir=work_dir,
            days=days,
            stations=stn_kept.assign(
                lat_model=stn_kept["lat_model"].to_numpy(np.float32),
                lon_model=stn_kept["lon_model"].to_numpy(np.float32),
            ),
            model_grid=model_grid,
            heatstress_long_pattern=heatstress_long_pattern,
            day_workers=day_workers,
            require_cols=("UTCI_C",),
            skip_missing=True,
        )
        if pred is None or len(pred) == 0:
            log("SURFRAD: model UTCI predictions empty; skipping")
            return []
        merged = pred.merge(obs, on=["station_id", "time_utc"], how="inner")
        if len(merged) == 0:
            log("SURFRAD: merged empty; skipping")
            return []

        if write_merged_parquet:
            merged.to_parquet(out_dir / f"merged_hourly_model_SURFRAD_utci_{tag}.parquet", index=False)
        if write_merged_csv:
            merged.to_csv(out_dir / f"merged_hourly_model_SURFRAD_utci_{tag}.csv", index=False)

        vars_avail = ["UTCI_C"]
        write_metrics_suite(merged, out_dir, tag, prefix="model_SURFRAD_utci", variables=vars_avail, pred_suffix="_model")

        table_rows = build_table_rows_from_merged(
            merged=merged,
            stations_meta=stn_kept,
            dist_col="dist_m_model",
            window_label=window_label,
            window_start=str(days.min().date()),
            window_end=str(days.max().date()),
            product_name="Radiation-driven UTCI (thermofeel)",
            reference_name="SURFRAD",
            distance_filter_label="≤800 m",
            variables=vars_avail,
            pred_prefix="model",
        )
        return table_rows

    # Fallback: wide tmrt/utci files
    for v in ["tmrt", "utci"]:
        test_fname = model_file_pattern.format(var=v, date=days.min().strftime("%Y-%m-%d"))
        if not (work_dir / test_fname).exists():
            log(f"SURFRAD: missing required model file var={v} (e.g., {test_fname}); skipping SURFRAD validation")
            return []

    pred = extract_model_prismday_predictions(
        work_dir=work_dir,
        days=days,
        stations=stn_kept.assign(
            lat_model=stn_kept["lat_model"].to_numpy(np.float32),
            lon_model=stn_kept["lon_model"].to_numpy(np.float32),
        ),
        model_grid=model_grid,
        model_file_pattern=model_file_pattern,
        need_vars=["tmrt", "utci"],
        var_prefix_map=model_var_prefix_map,
        day_workers=day_workers,
    )

    merged = pred.merge(obs, on=["station_id", "time_utc"], how="inner")
    if len(merged) == 0:
        log("SURFRAD: merged empty; skipping")
        return []

    if write_merged_parquet:
        merged.to_parquet(out_dir / f"merged_hourly_model_SURFRAD_utci_{tag}.parquet", index=False)
    if write_merged_csv:
        merged.to_csv(out_dir / f"merged_hourly_model_SURFRAD_utci_{tag}.csv", index=False)

    vars_avail = ["TMRT_C", "UTCI_C"]
    write_metrics_suite(merged, out_dir, tag, prefix="model_SURFRAD_utci", variables=vars_avail, pred_suffix="_model")

    table_rows = build_table_rows_from_merged(
        merged=merged,
        stations_meta=stn_kept,
        dist_col="dist_m_model",
        window_label=window_label,
        window_start=str(days.min().date()),
        window_end=str(days.max().date()),
        product_name="Radiation-driven UTCI (thermofeel)",
        reference_name="SURFRAD",
        distance_filter_label="≤800 m",
        variables=vars_avail,
        pred_prefix="model",
    )
    return table_rows


# ======================================================================================
# CLI entrypoints
# ======================================================================================

def cmd_build_grids(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    era_base = Path(args.era_base)
    root = Path(args.root)
    grid_dir = root / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    model_sample = Path(args.model_sample) if args.model_sample else find_sample_model_parquet(work_dir)
    era_sample = Path(args.era_sample) if args.era_sample else find_sample_era_parquet(era_base)

    log(f"MODEL sample: {model_sample}")
    lat_m, lon_m = read_latlon_unique(model_sample, args.model_ll_decimals)
    x_m, y_m = project_to_meters(lat_m, lon_m)
    save_grid(
        grid_dir / "model_grid",
        lat_m, lon_m, x_m, y_m,
        SavedGridMeta(ll_decimals=args.model_ll_decimals, n_points=int(len(lat_m)), source_parquet=str(model_sample)),
    )
    log(f"Saved model grid -> {(grid_dir/'model_grid.npz')}")  # noqa: E501

    log(f"ERA sample: {era_sample}")
    lat_e, lon_e = read_latlon_unique(era_sample, args.era_ll_decimals)
    x_e, y_e = project_to_meters(lat_e, lon_e)
    save_grid(
        grid_dir / "era5land_grid",
        lat_e, lon_e, x_e, y_e,
        SavedGridMeta(ll_decimals=args.era_ll_decimals, n_points=int(len(lat_e)), source_parquet=str(era_sample)),
    )
    log(f"Saved ERA grid -> {(grid_dir/'era5land_grid.npz')}")  # noqa: E501

    log(f"Done. Grid files live in: {grid_dir}")


def _parse_var_prefix_map(items: Sequence[str]) -> Dict[str, str]:
    """Parse --var-prefix entries of the form var=prefix."""
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_validate(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    era_base = Path(args.era_base)
    root = Path(args.root)

    grid_dir = root / "grids"
    model_grid_prefix = Path(args.model_grid_prefix) if args.model_grid_prefix else (grid_dir / "model_grid")
    era_grid_prefix = Path(args.era_grid_prefix) if args.era_grid_prefix else (grid_dir / "era5land_grid")

    cache_dir = root / "station_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_grid = load_grid(model_grid_prefix)
    era_grid = load_grid(era_grid_prefix)

    # windows from CLI
    if not args.window:
        raise SystemExit("No windows provided. Use --window YYYY-MM-DD YYYY-MM-DD (repeatable).")

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in args.window:
        windows.append((_ts(s), _ts(e)))

    # group by year for reuse
    windows_by_year: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for s, e in windows:
        if s.year != e.year:
            raise SystemExit(f"Window crosses years (not supported): {s.date()} to {e.date()}")
        windows_by_year.setdefault(s.year, []).append((s, e))

    # Parse variable prefix map (for internal column prefixes). Defaults to identity.
    var_prefix_map = _parse_var_prefix_map(args.var_prefix or [])
    if not var_prefix_map:
        var_prefix_map = {"t2m": "t2m", "ea": "ea", "hi": "hi", "wbgt": "wbgt", "tmrt": "tmrt", "utci": "utci"}

    all_table_rows: List[Dict[str, Any]] = []

    # Prepare optional per-year caches
    isd_ctx_by_year: Dict[int, ISDYearContext] = {}
    uscrn_by_year: Dict[int, pd.DataFrame] = {}
    surfrad_by_year: Dict[int, pd.DataFrame] = {}

    # Preload USCRN/SURFRAD if requested
    for year in sorted(windows_by_year.keys()):
        if not args.skip_isd:
            isd_ctx_by_year[year] = prepare_isd_year(
                year=year,
                model_grid=model_grid,
                era_grid=era_grid,
                root=root,
                cache_dir=cache_dir,
                conus_only=args.conus_only,
                n_stations_target=args.n_stations,
                station_seed=args.station_seed,
                max_dist_model_m=args.max_dist_model_m,
                max_dist_era_m=args.max_dist_era_m,
                obs_workers=args.obs_workers,
                download_retries=args.download_retries,
                coverage_mode=args.isd_coverage,
                cache_obs_parquet=args.cache_isd_obs_parquet,
            )

        if not args.skip_uscrn:
            try:
                uscrn_dir = Path(args.uscrn_dir) if args.uscrn_dir else (root / "uscrn_cache")
                uscrn_by_year[year] = prepare_uscrn_year(
                    year,
                    uscrn_dir,
                    max_workers=args.obs_workers,
                    force_rebuild=args.force_rebuild_uscrn,
                )
                log(f"USCRN: loaded year={year} rows={len(uscrn_by_year[year]):,}")
            except Exception as e:
                log(f"[warn] USCRN not available for year={year}: {type(e).__name__}: {e}")
                uscrn_by_year[year] = pd.DataFrame()

        if not args.skip_surfrad:
            try:
                surfrad_dir = Path(args.surfrad_dir) if args.surfrad_dir else (root / "surfrad_cache")
                surfrad_by_year[year] = load_surfrad_hourly_tmrt_utci(year, surfrad_dir)
                log(f"SURFRAD: loaded year={year} rows={len(surfrad_by_year[year]):,}")
            except Exception as e:
                log(f"[warn] SURFRAD not available for year={year}: {type(e).__name__}: {e}")
                surfrad_by_year[year] = pd.DataFrame()

    # Run each window
    for year, wins in windows_by_year.items():
        for (start_day, end_day) in wins:
            days = make_day_index(str(start_day.date()), str(end_day.date()))
            tag = run_tag(days)
            window_label = format_window_label(start_day, end_day)
            out_dir = runs_dir / f"{year:04d}" / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            log("=" * 90)
            log(f"RUN window: {window_label} | year={year} | days={len(days)} | tag={tag}")
            if len(days) != 7:
                log(f"[note] window length is {len(days)} days (table design often uses 7-day windows)")
            log(f"Outputs -> {out_dir}")

            # ISD-Lite validations
            if not args.skip_isd:
                ctx = isd_ctx_by_year.get(year)
                if ctx is None:
                    log(f"[warn] missing ISD context for year={year}; skipping ISD")
                else:
                    rows = validate_isd_window(
                        ctx=ctx,
                        work_dir=work_dir,
                        era_base=era_base,
                        out_dir=out_dir,
                        days=days,
                        window_label=window_label,
                        model_file_pattern=args.model_file_pattern,
                        model_var_prefix_map=var_prefix_map,
                        day_workers=args.day_workers,
                        write_merged_parquet=args.write_merged_parquet,
                        write_merged_csv=args.write_merged_csv,
                        include_paired=args.paired_comparison,
                        max_dist_model_m=args.max_dist_model_m,
                        max_dist_era_m=args.max_dist_era_m,
                    )
                    all_table_rows.extend(rows)

            # USCRN validations
            if not args.skip_uscrn:
                dfy = uscrn_by_year.get(year, pd.DataFrame())
                if dfy is None or dfy.empty:
                    log(f"USCRN: no data for year={year}; skipping")
                else:
                    rows = validate_uscrn_window(
                        year=year,
                        uscrn_df_year=dfy,
                        model_grid=model_grid,
                        work_dir=work_dir,
                        out_dir=out_dir,
                        days=days,
                        window_label=window_label,
                        model_file_pattern=args.model_file_pattern,
                        model_var_prefix_map=var_prefix_map,
                        day_workers=args.day_workers,
                        write_merged_parquet=args.write_merged_parquet,
                        write_merged_csv=args.write_merged_csv,
                        max_dist_model_m=args.max_dist_model_m,
                        run_met=not args.skip_uscrn_met,
                        run_hsi=not args.skip_uscrn_hsi,
                        prefer_long_heatstress=args.prefer_long_heatstress,
                        heatstress_long_pattern=args.heatstress_long_pattern,
                    )
                    all_table_rows.extend(rows)

            # SURFRAD validations
            if not args.skip_surfrad:
                dfy = surfrad_by_year.get(year, pd.DataFrame())
                if dfy is None or dfy.empty:
                    log(f"SURFRAD: no data for year={year}; skipping")
                else:
                    rows = validate_surfrad_window(
                        year=year,
                        surfrad_df_year=dfy,
                        model_grid=model_grid,
                        work_dir=work_dir,
                        out_dir=out_dir,
                        days=days,
                        window_label=window_label,
                        model_file_pattern=args.model_file_pattern,
                        model_var_prefix_map=var_prefix_map,
                        day_workers=args.day_workers,
                        write_merged_parquet=args.write_merged_parquet,
                        write_merged_csv=args.write_merged_csv,
                        max_dist_model_m=args.max_dist_model_m,
                        prefer_long_heatstress=args.prefer_long_heatstress,
                        heatstress_long_pattern=args.heatstress_long_pattern,
                    )
                    all_table_rows.extend(rows)

            # Write per-window table extract for convenience
            if all_table_rows:
                df_all = pd.DataFrame(all_table_rows)
                df_win = df_all[
                    (df_all["Window start"] == str(days.min().date())) &
                    (df_all["Window end"] == str(days.max().date()))
                ].copy()
                if len(df_win) > 0:
                    p = out_dir / f"technical_validation_table_rows_{tag}.csv"
                    df_win.to_csv(p, index=False)
                    log(f"Wrote per-window table rows -> {p}")

    # Write combined table
    if not all_table_rows:
        log("No table rows produced. Check inputs / windows / file availability.")
        return

    table = pd.DataFrame(all_table_rows)

    # Sort for readability
    sort_cols = ["Window start", "Product", "Reference", "Variable"]
    table = table.sort_values(sort_cols).reset_index(drop=True)

    out_path = tables_dir / "technical_validation_table.csv"
    table.to_csv(out_path, index=False)
    log(f"Wrote combined table -> {out_path} | rows={len(table):,}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="technical_validation.py",
        description="Generate technical validation metrics + a manuscript-ready summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # build-grids
    pg = sub.add_parser("build-grids", help="Build static grid KDTree files (one-time).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pg.add_argument("--work-dir", required=True, help="Directory containing model PRISM-day parquets (e.g., t2m_prismday_*.parquet)")
    pg.add_argument("--era-base", required=True, help="ERA5-Land base directory (contains t2m/<year>/*.parquet etc)")
    pg.add_argument("--root", required=True, help="Validation root output directory (will create <root>/grids)")
    pg.add_argument("--model-ll-decimals", type=int, default=6)
    pg.add_argument("--era-ll-decimals", type=int, default=4)
    pg.add_argument("--model-sample", default=None, help="Optional explicit sample model parquet path")
    pg.add_argument("--era-sample", default=None, help="Optional explicit sample ERA parquet path")
    pg.set_defaults(func=cmd_build_grids)

    # validate
    pv = sub.add_parser("validate", help="Run validations and build the summary table.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pv.add_argument("--work-dir", required=True, help="Directory with model PRISM-day parquets")
    pv.add_argument("--era-base", required=True, help="ERA5-Land base directory")
    pv.add_argument("--root", required=True, help="Validation root directory (contains grids/, runs/, station_cache/)")
    pv.add_argument("--model-grid-prefix", default=None, help="Override model grid prefix (default: <root>/grids/model_grid)")
    pv.add_argument("--era-grid-prefix", default=None, help="Override ERA grid prefix (default: <root>/grids/era5land_grid)")
    pv.add_argument("--window", nargs=2, action="append", metavar=("START_DAY", "END_DAY"), help="PRISM-day window inclusive, e.g., --window 2010-07-15 2010-07-21 (repeatable)")
    pv.add_argument("--model-file-pattern", default="{var}_prismday_{date}.parquet", help="Filename pattern under --work-dir (vars: t2m, ea, hi, wbgt, tmrt, utci)")
    pv.add_argument("--var-prefix", action="append", default=None, help="Map variable name to column prefix in parquet, e.g. --var-prefix hi=hi --var-prefix wbgt=wbgt")

    # Long heatstress (HI/WBGT/UTCI in one long parquet)
    pv.add_argument("--prefer-long-heatstress", action="store_true", default=False,
                    help="Prefer long-form heatstress_long_prismday_{date}.parquet for HI/WBGT/UTCI validations (USCRN/SURFRAD).")
    pv.add_argument("--heatstress-long-pattern", default="heatstress_long_prismday_{date}.parquet",
                    help="Filename pattern under --work-dir for long-form heatstress PRISM-day parquets.")

    # ISD options
    pv.add_argument("--skip-isd", action="store_true", help="Skip ISD-Lite validations")
    pv.add_argument("--isd-coverage", choices=["full_year"], default="full_year", help="Station coverage requirement for ISD-history filtering")
    pv.add_argument("--conus-only", action="store_true", default=True)
    pv.add_argument("--n-stations", type=int, default=500, help="Target number of spatially diverse ISD stations (set 0 for all)")
    pv.add_argument("--station-seed", type=int, default=11)
    pv.add_argument("--max-dist-model-m", type=float, default=800.0)
    pv.add_argument("--max-dist-era-m", type=float, default=9000.0)
    pv.add_argument("--download-retries", type=int, default=1)
    pv.add_argument("--cache-isd-obs-parquet", action="store_true", default=True, help="Cache parsed ISD obs to <root>/obs_cache for fast reruns")
    pv.add_argument("--no-cache-isd-obs-parquet", dest="cache_isd_obs_parquet", action="store_false")

    # USCRN options
    pv.add_argument("--skip-uscrn", action="store_true", help="Skip USCRN (heat01) validations")
    pv.add_argument("--skip-uscrn-met", action="store_true", help="Skip PRISM-aligned meteorology vs USCRN rows")
    pv.add_argument("--skip-uscrn-hsi", action="store_true", help="Skip PRISM-aligned heat-stress indices vs USCRN rows")
    pv.add_argument("--uscrn-dir", default=None, help="Directory for cached USCRN yearly parquet files (default: <root>/uscrn_cache)")
    pv.add_argument("--force-rebuild-uscrn", action="store_true", help="Force rebuild USCRN year parquets (otherwise reuse cached)")  # noqa: E501

    # SURFRAD options
    pv.add_argument("--skip-surfrad", action="store_true", help="Skip SURFRAD UTCI/TMRT validations")
    pv.add_argument("--surfrad-dir", default=None, help="Directory containing SURFRAD_hourly_TMRT_UTCI_<year>.parquet")

    # Parallelism / output
    pv.add_argument("--obs-workers", type=int, default=max(1, (os.cpu_count() or 8)), help="Parallel workers for station obs downloads/parsing")
    pv.add_argument("--day-workers", type=int, default=1, help="Parallel workers for per-day extraction (IO-heavy; use with care)")
    pv.add_argument("--write-merged-parquet", action="store_true", default=False, help="Write merged hourly parquet (large)")
    pv.add_argument("--write-merged-csv", action="store_true", default=False, help="Write merged hourly CSV (very large; not recommended)")
    pv.add_argument("--paired-comparison", action="store_true", default=False, help="Also write paired MODEL-vs-ERA comparison metrics (ISD-Lite only)")

    pv.set_defaults(func=cmd_validate)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize n_stations
    if getattr(args, "n_stations", None) == 0:
        args.n_stations = None

    args.func(args)


if __name__ == "__main__":
    main()