#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HeatStress end-to-end PRISM-day -> tract averages (ONE combined output with AREA + POP weightings).

Outputs per PRISM day D (UTC window: D-1 12Z .. D 11Z):
  heatstress_tract_area_and_popweighted_<D-1>_<D>_popy<POPYEAR>_v<VINTAGE>.parquet

Time and Geo Columns:
  GEOID, lat, lon, year, month, day, time, 
  <metric>_area, <metric>_pop

Notes:
- Population weights come from annual WorldPop-on-PRISM files:
    worldpop_usa_pop_on_prism800m_<YEAR>_key<KeyDecimals>.parquet
  columns: key(int64), pop(float32)
- For years < 2000, falls back to YEAR=2000 if the requested year is missing.
- Keeps UNION of tracts with coverage under either weighting.

------------
Some of this script was written using an AI Jupiter Notebook plug-in.
Primary author: Rouzbeh Rahai
Development period: 2025-2026
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Small date helpers
# =============================================================================

def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()

def date_range_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = _ts(start)
    end = _ts(end)
    if end < start:
        raise ValueError(f"end < start: {end} < {start}")
    return [d.normalize() for d in pd.date_range(start, end, freq="D")]

def doy_to_date(year: int, doy: int) -> pd.Timestamp:
    if doy < 1 or doy > 366:
        raise ValueError("DOY must be 1..366")
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)

def prism_window_dates(D: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_date_utc, end_date_utc) as dates (not hours): (D-1, D)."""
    D = _ts(D)
    return (D - pd.Timedelta(days=1), D)


# =============================================================================
# Common geo helpers
# =============================================================================

def nearest_index(sorted_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    idx = np.searchsorted(sorted_grid, values)
    idx = np.clip(idx, 1, len(sorted_grid) - 1)
    left = sorted_grid[idx - 1]
    right = sorted_grid[idx]
    choose_left = (values - left) < (right - values)
    return idx - choose_left.astype(np.int64)

def es_hPa(T_C: np.ndarray) -> np.ndarray:
    """Buck saturation vapor pressure (hPa) over water/ice."""
    T = np.asarray(T_C, dtype=np.float64)
    out = np.empty_like(T, dtype=np.float64)

    w = T >= 0.0
    Tw = T[w]
    out[w] = 6.1121 * np.exp((18.678 - Tw / 234.5) * (Tw / (257.14 + Tw)))

    Ti = T[~w]
    out[~w] = 6.1115 * np.exp((23.036 - Ti / 333.7) * (Ti / (279.82 + Ti)))

    return out.astype(np.float32)


# =============================================================================
# Paths: ERA + PRISM + NSRDB + WorldPop
# =============================================================================

def mmddyyyy(ts: pd.Timestamp) -> str:
    return ts.strftime("%m-%d-%Y")

def yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")

def era_day_path(era_base: Path, var: str, day_utc: pd.Timestamp) -> Path:
    """
    Tries multiple filename conventions:
      - MM-DD-YYYY.parquet
      - YYYY-MM-DD.parquet
      - YYYY-MM-DD_<var>.parquet
      - MM-DD-YYYY_<var>.parquet
    """
    day_utc = _ts(day_utc)
    year_dir = era_base / var / f"{day_utc.year:04d}"

    candidates = [
        year_dir / f"{mmddyyyy(day_utc)}.parquet",
        year_dir / f"{yyyymmdd(day_utc)}.parquet",
        year_dir / f"{yyyymmdd(day_utc)}_{var}.parquet",
        year_dir / f"{mmddyyyy(day_utc)}_{var}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Missing ERA file for var='{var}' day={day_utc.date()}.\n"
        + "\n".join([f"  tried: {c}" for c in candidates])
    )

def prism_day_tif(prism_base: Path, folder: str, prism_day: pd.Timestamp) -> Path:
    prism_day = _ts(prism_day)
    return prism_base / folder / f"{prism_day.year:04d}" / f"{mmddyyyy(prism_day)}.tif"

def nsrdb_day_var_path(nsrdb_base: Path, day_utc: pd.Timestamp, var: str) -> Path:
    """
    Layout:
      NSRDB_BASE/<year>/<YYYY-MM-DD>/<file>.parquet
    """
    day_utc = _ts(day_utc)
    folder = nsrdb_base / f"{day_utc.year:04d}" / yyyymmdd(day_utc)
    name_map = {
        "ghi": "ghi.parquet",
        "dhi": "dhi.parquet",
        "dni": "dni.parquet",
        "sza": "solar_zenith_angle.parquet",
        "alb": "surface_albedo.parquet",
    }
    if var not in name_map:
        raise ValueError(f"Unknown NSRDB var '{var}'. Expected one of {sorted(name_map)}")
    return folder / name_map[var]

def worldpop_year_path(worldpop_weights_dir: Path, year: int, key_decimals: int) -> Path:
    return worldpop_weights_dir / f"worldpop_usa_pop_on_prism800m_{year}_key{int(key_decimals)}.parquet"

def resolve_worldpop_year_path(
    worldpop_weights_dir: Path,
    year: int,
    key_decimals: int,
    pre2000_fallback_year: int = 2000,
    max_year: int = 2020,
) -> Tuple[int, Path]:
    """
    Returns (pop_year_used, path).
    - Try YEAR=year
    - If missing and year < 2000, try YEAR=pre2000_fallback_year
    - If year > max_year and file missing, fail.
    """
    p = worldpop_year_path(worldpop_weights_dir, year, key_decimals)
    if p.exists():
        return year, p

    if year < 2000:
        p2 = worldpop_year_path(worldpop_weights_dir, pre2000_fallback_year, key_decimals)
        if p2.exists():
            return pre2000_fallback_year, p2

    if year > max_year:
        raise FileNotFoundError(
            f"Requested pop year={year} but WorldPop Global_2000_2020 ends at {max_year}, "
            f"and no file found: {p}"
        )

    raise FileNotFoundError(f"Missing WorldPop weights for year={year}: {p}")


# =============================================================================
# STEP 1+2 fused: ERA t2m -> PRISM hourly T and PRISM Tdmean + ERA Td shape -> hourly ea
# =============================================================================

def lon_lat_from_window(transform, row_off: int, col_off: int, height: int, width: int):
    # rasterio Affine: x = c + (col + 0.5)*a ; y = f + (row + 0.5)*e
    a, e, c, f = transform.a, transform.e, transform.c, transform.f
    cols = np.arange(col_off, col_off + width)
    rows = np.arange(row_off, row_off + height)
    x = c + (cols + 0.5) * a
    y = f + (rows + 0.5) * e
    lon2d, lat2d = np.meshgrid(x, y)
    return lon2d.astype(np.float32), lat2d.astype(np.float32)

def load_era_day_C(era_base: Path, var: str, day_utc: pd.Timestamp) -> pd.DataFrame:
    """Load ERA UTC-day parquet (00..23Z) and convert K->C."""
    p = era_day_path(era_base, var, day_utc)
    cols = ["lat", "lon"] + [f"{var}_h{h:02d}" for h in range(24)]
    df = pd.read_parquet(p, columns=cols).copy()
    hcols = [f"{var}_h{h:02d}" for h in range(24)]
    df[hcols] = df[hcols].astype(np.float32) - 273.15
    return df

def build_era_prismday_anoms(era_base: Path, var: str, D: pd.Timestamp):
    """
    PRISM day D window = (D-1) 12..23 + (D) 00..11
    Returns ERA anomalies on ERA grid.
    """
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)

    dfD = load_era_day_C(era_base, var, D)
    dfM1 = load_era_day_C(era_base, var, Dm1)
    df = dfD.merge(dfM1, on=["lat", "lon"], suffixes=("_D", "_M1"))

    H = np.empty((len(df), 24), dtype=np.float32)
    for k, h in enumerate(range(12, 24)):
        H[:, k] = df[f"{var}_h{h:02d}_M1"].to_numpy(dtype=np.float32)
    for k, h in enumerate(range(0, 12), start=12):
        H[:, k] = df[f"{var}_h{h:02d}_D"].to_numpy(dtype=np.float32)

    era_lats = np.sort(df["lat"].unique().astype(np.float64))
    era_lons = np.sort(df["lon"].unique().astype(np.float64))
    lat_to_i = {v: i for i, v in enumerate(era_lats)}
    lon_to_j = {v: j for j, v in enumerate(era_lons)}

    T = np.full((len(era_lats), len(era_lons), 24), np.nan, dtype=np.float32)
    ii = df["lat"].map(lat_to_i).to_numpy()
    jj = df["lon"].map(lon_to_j).to_numpy()
    T[ii, jj, :] = H

    m = np.nanmean(T, axis=2)
    anom = T - m[..., None]
    Amin = np.nanmin(anom, axis=2)
    Amax = np.nanmax(anom, axis=2)
    valid = np.isfinite(Amin) & np.isfinite(Amax) & (Amax > Amin)
    return era_lats, era_lons, anom, Amin, Amax, valid

def load_era_day_Td_C(era_base: Path, dew_var: str, day_utc: pd.Timestamp) -> pd.DataFrame:
    p_td = era_day_path(era_base, dew_var, day_utc)
    cols = ["lat", "lon"] + [f"{dew_var}_h{h:02d}" for h in range(24)]
    df = pd.read_parquet(p_td, columns=cols).copy()
    if df.duplicated(subset=["lat", "lon"]).any():
        raise ValueError(f"ERA {dew_var} {day_utc.date()}: duplicate (lat,lon) rows")
    hcols = [f"{dew_var}_h{h:02d}" for h in range(24)]
    df[hcols] = df[hcols].astype(np.float32) - 273.15
    return df

def build_era_prismday_td_templates(era_base: Path, dew_var: str, D: pd.Timestamp):
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    dfD = load_era_day_Td_C(era_base, dew_var, D)
    dfM1 = load_era_day_Td_C(era_base, dew_var, Dm1)
    df = dfD.merge(dfM1, on=["lat", "lon"], suffixes=("_D", "_M1"))

    N = len(df)
    Td_ph = np.empty((N, 24), dtype=np.float32)
    for k, h in enumerate(range(12, 24)):
        Td_ph[:, k] = df[f"{dew_var}_h{h:02d}_M1"].to_numpy(dtype=np.float32)
    for k, h in enumerate(range(0, 12), start=12):
        Td_ph[:, k] = df[f"{dew_var}_h{h:02d}_D"].to_numpy(dtype=np.float32)

    good = np.isfinite(Td_ph).all(axis=1)
    Td_mean = np.full(N, np.nan, dtype=np.float32)
    if np.any(good):
        Td_mean[good] = Td_ph[good].mean(axis=1).astype(np.float32)

    era_lats = np.sort(df["lat"].unique().astype(np.float64))
    era_lons = np.sort(df["lon"].unique().astype(np.float64))
    lat_to_i = {v: i for i, v in enumerate(era_lats)}
    lon_to_j = {v: j for j, v in enumerate(era_lons)}

    TD = np.full((len(era_lats), len(era_lons), 24), np.nan, dtype=np.float32)
    TM = np.full((len(era_lats), len(era_lons)), np.nan, dtype=np.float32)
    valid = np.zeros((len(era_lats), len(era_lons)), dtype=bool)

    ii = df["lat"].map(lat_to_i).to_numpy()
    jj = df["lon"].map(lon_to_j).to_numpy()

    ii2 = ii[good]
    jj2 = jj[good]
    TD[ii2, jj2, :] = Td_ph[good]
    TM[ii2, jj2] = Td_mean[good]
    valid[ii2, jj2] = True
    return era_lats, era_lons, TD, TM, valid

def fuse_t2m_and_ea_prismday(
    D: pd.Timestamp,
    era_base: Path,
    prism_base: Path,
    prism_tdmean_folder: str,
    out_t2m: Path,
    out_ea: Path,
    window_size: int = 512,
    compression: str = "zstd",
    clip_tol: float = 1e-6,
    ea_floor_hpa: float = 0.05,
    test_max_windows: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[Path, Path]:
    """
    Outputs:
      - t2m parquet: lat, lon, prism_day, t2m_ph00..t2m_ph23 (°C)
      - ea  parquet: lat, lon, prism_day, ea_ph00..ea_ph23  (hPa)
    """
    D = _ts(D)
    prism_tmin = prism_day_tif(prism_base, "tmin", D)
    prism_tmax = prism_day_tif(prism_base, "tmax", D)
    prism_tdmean = prism_day_tif(prism_base, prism_tdmean_folder, D)
    for p in (prism_tmin, prism_tmax, prism_tdmean):
        if not p.exists():
            raise FileNotFoundError(f"Missing PRISM file: {p}")

    out_t2m.parent.mkdir(parents=True, exist_ok=True)
    out_ea.parent.mkdir(parents=True, exist_ok=True)
    for p in (out_t2m, out_ea):
        if p.exists():
            p.unlink()

    if not quiet:
        print("\n[Step1+2] Building ERA templates...")
    era_t_lats, era_t_lons, anom, Amin, Amax, valid_era_t = build_era_prismday_anoms(era_base, "t2m", D)
    era_e_lats, era_e_lons, td_cube, td_mean_cube, valid_era_e = build_era_prismday_td_templates(era_base, "d2m", D)

    schema_t = pa.schema(
        [("lat", pa.float32()), ("lon", pa.float32()), ("prism_day", pa.date32())]
        + [(f"t2m_ph{h:02d}", pa.float32()) for h in range(24)]
    )
    schema_e = pa.schema(
        [("lat", pa.float32()), ("lon", pa.float32()), ("prism_day", pa.date32())]
        + [(f"ea_ph{h:02d}", pa.float32()) for h in range(24)]
    )

    prism_day_np = np.datetime64(D.strftime("%Y-%m-%d"), "D")

    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise ImportError("rasterio is required for Step1+2.") from e

    windows_done = 0
    rows_t = 0
    rows_e = 0

    with rasterio.open(prism_tmin) as src_tmin, \
         rasterio.open(prism_tmax) as src_tmax, \
         rasterio.open(prism_tdmean) as src_tdmean, \
         pq.ParquetWriter(out_t2m.as_posix(), schema=schema_t, compression=compression) as w_t, \
         pq.ParquetWriter(out_ea.as_posix(), schema=schema_e, compression=compression) as w_e:

        if (src_tmin.width != src_tmax.width) or (src_tmin.height != src_tmax.height):
            raise ValueError("PRISM tmin/tmax rasters have different shapes.")
        if (src_tmin.width != src_tdmean.width) or (src_tmin.height != src_tdmean.height):
            raise ValueError("PRISM tdmean raster shape differs from tmin/tmax.")

        nd_tmin = src_tmin.nodata
        nd_tmax = src_tmax.nodata
        nd_td = src_tdmean.nodata

        transform = src_tmin.transform
        height, width = src_tmin.height, src_tmin.width

        for row_off in range(0, height, window_size):
            win_h = min(window_size, height - row_off)
            for col_off in range(0, width, window_size):
                win_w = min(window_size, width - col_off)
                window = Window(col_off, row_off, win_w, win_h)

                Tmin = src_tmin.read(1, window=window).astype(np.float32)
                Tmax = src_tmax.read(1, window=window).astype(np.float32)
                Tdmean = src_tdmean.read(1, window=window).astype(np.float32)

                if nd_tmin is not None:
                    Tmin[Tmin == nd_tmin] = np.nan
                if nd_tmax is not None:
                    Tmax[Tmax == nd_tmax] = np.nan
                if nd_td is not None:
                    Tdmean[Tdmean == nd_td] = np.nan

                valid_temp = np.isfinite(Tmin) & np.isfinite(Tmax) & (Tmax > Tmin)
                if not np.any(valid_temp):
                    continue

                lon2d, lat2d = lon_lat_from_window(transform, row_off, col_off, win_h, win_w)

                lat_v = lat2d[valid_temp].astype(np.float64)
                lon_v = lon2d[valid_temp].astype(np.float64)
                Tmin_v = Tmin[valid_temp].astype(np.float32)
                Tmax_v = Tmax[valid_temp].astype(np.float32)
                Tdmean_v_all = Tdmean[valid_temp].astype(np.float32)

                lat_idx_t = nearest_index(era_t_lats, lat_v)
                lon_idx_t = nearest_index(era_t_lons, lon_v)
                ok_t = valid_era_t[lat_idx_t, lon_idx_t]
                if not np.any(ok_t):
                    continue

                lat_v = lat_v[ok_t].astype(np.float32)
                lon_v = lon_v[ok_t].astype(np.float32)
                Tmin_v = Tmin_v[ok_t]
                Tmax_v = Tmax_v[ok_t]
                Tdmean_v_all = Tdmean_v_all[ok_t]
                lat_idx_t = lat_idx_t[ok_t]
                lon_idx_t = lon_idx_t[ok_t]

                a = anom[lat_idx_t, lon_idx_t, :].astype(np.float32)
                amin = Amin[lat_idx_t, lon_idx_t].astype(np.float32)
                amax = Amax[lat_idx_t, lon_idx_t].astype(np.float32)

                f = (a - amin[:, None]) / (amax - amin)[:, None]
                T_ph = Tmin_v[:, None] + f * (Tmax_v - Tmin_v)[:, None]
                T_ph = np.clip(T_ph, Tmin_v[:, None] - clip_tol, Tmax_v[:, None] + clip_tol)
                T_ph = np.clip(T_ph, Tmin_v[:, None], Tmax_v[:, None]).astype(np.float32, copy=False)

                out_t = {
                    "lat": lat_v.astype(np.float32, copy=False),
                    "lon": lon_v.astype(np.float32, copy=False),
                    "prism_day": np.full(len(lat_v), prism_day_np, dtype="datetime64[D]"),
                }
                for h in range(24):
                    out_t[f"t2m_ph{h:02d}"] = T_ph[:, h].astype(np.float32, copy=False)
                w_t.write_table(pa.Table.from_pydict(out_t, schema=schema_t))
                rows_t += len(lat_v)

                ok_td = np.isfinite(Tdmean_v_all)
                if np.any(ok_td):
                    lat_e = lat_v[ok_td].astype(np.float64)
                    lon_e = lon_v[ok_td].astype(np.float64)
                    Tdmean_e = Tdmean_v_all[ok_td].astype(np.float32)
                    T_ph_e = T_ph[ok_td, :].astype(np.float32, copy=False)

                    lat_idx_e = nearest_index(era_e_lats, lat_e)
                    lon_idx_e = nearest_index(era_e_lons, lon_e)
                    ok_e = valid_era_e[lat_idx_e, lon_idx_e]
                    if np.any(ok_e):
                        lat_e = lat_e[ok_e].astype(np.float32)
                        lon_e = lon_e[ok_e].astype(np.float32)
                        Tdmean_e = Tdmean_e[ok_e]
                        T_ph_e = T_ph_e[ok_e, :]
                        lat_idx_e = lat_idx_e[ok_e]
                        lon_idx_e = lon_idx_e[ok_e]

                        td_era_ph = td_cube[lat_idx_e, lon_idx_e, :]
                        td_era_mean = td_mean_cube[lat_idx_e, lon_idx_e]
                        td_shift = (Tdmean_e - td_era_mean).astype(np.float32)
                        td_ph_corr = (td_era_ph + td_shift[:, None]).astype(np.float32, copy=False)

                        ea_ph = es_hPa(td_ph_corr)
                        esTa_ph = es_hPa(T_ph_e)
                        ea_ph = np.clip(ea_ph, 0.0, esTa_ph)

                        floor = np.minimum(ea_floor_hpa, esTa_ph)
                        ea_ph = np.maximum(ea_ph, floor)
                        ea_ph = np.clip(ea_ph, floor - clip_tol, esTa_ph + clip_tol)
                        ea_ph = np.clip(ea_ph, floor, esTa_ph).astype(np.float32, copy=False)

                        out_e = {
                            "lat": lat_e.astype(np.float32, copy=False),
                            "lon": lon_e.astype(np.float32, copy=False),
                            "prism_day": np.full(len(lat_e), prism_day_np, dtype="datetime64[D]"),
                        }
                        for h in range(24):
                            out_e[f"ea_ph{h:02d}"] = ea_ph[:, h].astype(np.float32, copy=False)
                        w_e.write_table(pa.Table.from_pydict(out_e, schema=schema_e))
                        rows_e += len(lat_e)

                windows_done += 1
                if (not quiet) and ((windows_done % 25) == 0):
                    print(f"[Step1+2] windows={windows_done:,} rows_t2m={rows_t:,} rows_ea={rows_e:,}", flush=True)

                if test_max_windows is not None and windows_done >= test_max_windows:
                    break
            if test_max_windows is not None and windows_done >= test_max_windows:
                break

    if not quiet:
        print(f"[Step1] DONE: {out_t2m} rows={rows_t:,}")
        print(f"[Step2] DONE: {out_ea}  rows={rows_e:,}")
    return out_t2m, out_ea


# =============================================================================
# STEP 3: ERA -> PRISM-day hours -> IDW onto PRISM grid
# =============================================================================

@dataclass
class EraIdwContext:
    src_ll: pd.DataFrame
    tree: "cKDTree"
    transformer: "Transformer"

def normalize_ll(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    out = df.copy()
    out["lat"] = np.round(out["lat"].astype(np.float64), decimals).astype(np.float32)
    out["lon"] = np.round(out["lon"].astype(np.float64), decimals).astype(np.float32)
    return out

def read_era_day_df(era_base: Path, var: str, day_utc: pd.Timestamp, ll_decimals: int) -> pd.DataFrame:
    p = era_day_path(era_base, var, day_utc)
    cols = ["lat", "lon"] + [f"{var}_h{h:02d}" for h in range(24)]
    df = pd.read_parquet(p, columns=cols)
    return normalize_ll(df, decimals=ll_decimals)

def align_to_src_ll_and_extract_24h(src_ll: pd.DataFrame, df_var: pd.DataFrame, var: str) -> np.ndarray:
    if df_var.duplicated(subset=["lat", "lon"]).any():
        raise ValueError(f"{var}: duplicate (lat,lon) rows; cannot align safely.")
    hcols = [f"{var}_h{h:02d}" for h in range(24)]
    df_al = src_ll.merge(df_var[["lat", "lon"] + hcols], on=["lat", "lon"], how="left", sort=False)
    return df_al[hcols].to_numpy(dtype=np.float32, copy=False)

def combine_utc_days_to_prism_hours(arr_m1: np.ndarray, arr_d: np.ndarray) -> np.ndarray:
    out = np.empty((arr_d.shape[0], 24), dtype=np.float32)
    out[:, 0:12] = arr_m1[:, 12:24]
    out[:, 12:24] = arr_d[:, 0:12]
    return out

def build_prismday_source_values(era_base: Path, var: str, D: pd.Timestamp, src_ll: pd.DataFrame, ll_decimals: int) -> np.ndarray:
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    df_m1 = read_era_day_df(era_base, var, Dm1, ll_decimals)
    df_d = read_era_day_df(era_base, var, D, ll_decimals)
    arr_m1 = align_to_src_ll_and_extract_24h(src_ll, df_m1, var)
    arr_d = align_to_src_ll_and_extract_24h(src_ll, df_d, var)
    return combine_utc_days_to_prism_hours(arr_m1, arr_d)

def hourly_lw_flux_from_accum_arrays(
    str_day: np.ndarray,
    strd_day: np.ndarray,
    str_next: Optional[np.ndarray] = None,
    strd_next: Optional[np.ndarray] = None,
    clip_lwdown_neg_to_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, nhrs = str_day.shape
    net_j = np.full((n, 24), np.nan, dtype=np.float64)
    down_j = np.full((n, 24), np.nan, dtype=np.float64)

    net_j[:, 0] = str_day[:, 1]
    down_j[:, 0] = strd_day[:, 1]
    net_j[:, 1:23] = str_day[:, 2:24] - str_day[:, 1:23]
    down_j[:, 1:23] = strd_day[:, 2:24] - strd_day[:, 1:23]

    if (str_next is not None) and (strd_next is not None):
        net_j[:, 23] = str_next[:, 0] - str_day[:, 23]
        down_j[:, 23] = strd_next[:, 0] - strd_day[:, 23]

    net_w = (net_j / 3600.0).astype(np.float32)
    down_w = (down_j / 3600.0).astype(np.float32)
    up_w = (down_w - net_w).astype(np.float32)

    if clip_lwdown_neg_to_nan:
        down_w = np.where(down_w >= 0.0, down_w, np.nan).astype(np.float32)
    return down_w, up_w, net_w

def build_longwave_prismday_source_values(
    era_base: Path,
    D: pd.Timestamp,
    src_ll: pd.DataFrame,
    ll_decimals: int,
    clip_lwdown_neg_to_nan: bool = True,
    quiet: bool = False,
) -> Dict[str, np.ndarray]:
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    Dp1 = D + pd.Timedelta(days=1)

    df_str_m1 = read_era_day_df(era_base, "str", Dm1, ll_decimals)
    df_strd_m1 = read_era_day_df(era_base, "strd", Dm1, ll_decimals)
    df_str_d = read_era_day_df(era_base, "str", D, ll_decimals)
    df_strd_d = read_era_day_df(era_base, "strd", D, ll_decimals)

    str_m1 = align_to_src_ll_and_extract_24h(src_ll, df_str_m1, "str")
    strd_m1 = align_to_src_ll_and_extract_24h(src_ll, df_strd_m1, "strd")
    str_d = align_to_src_ll_and_extract_24h(src_ll, df_str_d, "str")
    strd_d = align_to_src_ll_and_extract_24h(src_ll, df_strd_d, "strd")

    lwdown_m1, lwup_m1, lwnet_m1 = hourly_lw_flux_from_accum_arrays(
        str_day=str_m1, strd_day=strd_m1, str_next=str_d, strd_next=strd_d,
        clip_lwdown_neg_to_nan=clip_lwdown_neg_to_nan,
    )

    try:
        df_str_p1 = read_era_day_df(era_base, "str", Dp1, ll_decimals)
        df_strd_p1 = read_era_day_df(era_base, "strd", Dp1, ll_decimals)
        str_p1 = align_to_src_ll_and_extract_24h(src_ll, df_str_p1, "str")
        strd_p1 = align_to_src_ll_and_extract_24h(src_ll, df_strd_p1, "strd")
        lwdown_d, lwup_d, lwnet_d = hourly_lw_flux_from_accum_arrays(
            str_day=str_d, strd_day=strd_d, str_next=str_p1, strd_next=strd_p1,
            clip_lwdown_neg_to_nan=clip_lwdown_neg_to_nan,
        )
    except FileNotFoundError:
        lwdown_d, lwup_d, lwnet_d = hourly_lw_flux_from_accum_arrays(
            str_day=str_d, strd_day=strd_d, str_next=None, strd_next=None,
            clip_lwdown_neg_to_nan=clip_lwdown_neg_to_nan,
        )
        if not quiet:
            print(f"[lw] D+1 missing; last-hour for day D not bridged (OK for PRISM-day).")

    return {
        "lwdown": combine_utc_days_to_prism_hours(lwdown_m1, lwdown_d),
        "lwnet":  combine_utc_days_to_prism_hours(lwnet_m1, lwnet_d),
    }

def idw_24h(values_24: np.ndarray, w: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    n_tar, k = idxs.shape
    num = np.zeros((n_tar, 24), dtype=np.float32)
    den = np.zeros((n_tar, 24), dtype=np.float32)
    for kk in range(k):
        idx_k = idxs[:, kk]
        wk = w[:, kk][:, None]
        vk = values_24[idx_k, :]
        good = np.isfinite(vk)
        num += wk * np.where(good, vk, 0.0).astype(np.float32, copy=False)
        den += wk * good.astype(np.float32, copy=False)
    out = np.full_like(num, np.nan, dtype=np.float32)
    m = den > 0.0
    out[m] = (num[m] / den[m]).astype(np.float32, copy=False)
    return out

def build_era_idw_context(era_base: Path, D: pd.Timestamp, ll_decimals: int = 6, quiet: bool = False) -> EraIdwContext:
    try:
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("pyproj and scipy required for ERA IDW.") from e

    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    if not quiet:
        print("[ERA-IDW] Building KDTree from sp grid...")

    try:
        df_sp_src = read_era_day_df(era_base, "sp", D, ll_decimals)
    except FileNotFoundError:
        df_sp_src = read_era_day_df(era_base, "sp", Dm1, ll_decimals)

    src_ll = df_sp_src[["lat", "lon"]].copy()
    src_lat = src_ll["lat"].to_numpy(np.float64, copy=False)
    src_lon = src_ll["lon"].to_numpy(np.float64, copy=False)

    tfm = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    src_x, src_y = tfm.transform(src_lon, src_lat)
    src_xy = np.column_stack([src_x, src_y]).astype(np.float64)
    tree = cKDTree(src_xy)
    return EraIdwContext(src_ll=src_ll, tree=tree, transformer=tfm)

def run_era_idw_prismday(
    D: pd.Timestamp,
    era_base: Path,
    prism_grid: Path,
    out_dir: Path,
    ctx: EraIdwContext,
    vars_instant: Sequence[str] = ("sp", "u10", "v10"),
    ll_decimals: int = 6,
    batch_rows: int = 750_000,
    compression: str = "zstd",
    k_neighbors: int = 4,
    idw_power: float = 1.0,
    eps: float = 1e-6,
    cutoff_m: float = 8000.0,
    clip_lwdown_neg_to_nan: bool = True,
    overwrite: bool = True,
    quiet: bool = False,
) -> Dict[str, Path]:
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as e:
        raise ImportError("scipy required for ERA IDW.") from e

    D = _ts(D)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf_grid = pq.ParquetFile(prism_grid.as_posix())
    if not quiet:
        print(f"\n[Step3] ERA IDW -> PRISM grid | day={D.date()} | rowgroups={pf_grid.num_row_groups}")

    values_map: Dict[str, np.ndarray] = {}
    for var in vars_instant:
        values_map[var] = build_prismday_source_values(era_base, var, D, ctx.src_ll, ll_decimals)

    lw = build_longwave_prismday_source_values(
        era_base, D, ctx.src_ll, ll_decimals,
        clip_lwdown_neg_to_nan=clip_lwdown_neg_to_nan, quiet=quiet,
    )
    values_map.update(lw)

    out_paths: Dict[str, Path] = {k: out_dir / f"{k}_prismday_{D.strftime('%Y-%m-%d')}.parquet" for k in values_map.keys()}

    schemas: Dict[str, pa.Schema] = {}
    writers: Dict[str, Optional[pq.ParquetWriter]] = {k: None for k in values_map.keys()}

    for k, p in out_paths.items():
        if p.exists() and overwrite:
            p.unlink()
        schemas[k] = pa.schema(
            [pa.field("lat", pa.float32()), pa.field("lon", pa.float32()), pa.field("prism_day", pa.date32())]
            + [pa.field(f"{k}_ph{h:02d}", pa.float32()) for h in range(24)]
        )

    prism_day_np = np.datetime64(D.strftime("%Y-%m-%d"), "D")
    tfm = ctx.transformer
    tree = ctx.tree

    for rg in range(pf_grid.num_row_groups):
        df_rg = pf_grid.read_row_group(rg, columns=["lat", "lon"]).to_pandas()
        n = len(df_rg)
        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            chunk = df_rg.iloc[start:end]
            lat_t = chunk["lat"].to_numpy(np.float64, copy=False)
            lon_t = chunk["lon"].to_numpy(np.float64, copy=False)
            xt, yt = tfm.transform(lon_t, lat_t)
            tar_xy = np.column_stack([xt, yt]).astype(np.float64)

            dists, idxs = tree.query(tar_xy, k=k_neighbors, workers=-1)
            if k_neighbors == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            ok = np.min(dists, axis=1) <= cutoff_m
            w = (1.0 / np.maximum(dists, eps) ** idw_power).astype(np.float32, copy=False)
            w[~np.isfinite(w)] = 0.0

            base_lat = chunk["lat"].to_numpy(np.float32, copy=False)
            base_lon = chunk["lon"].to_numpy(np.float32, copy=False)
            base_day = np.full(len(base_lat), prism_day_np, dtype="datetime64[D]")

            for var_key, src_vals_ph in values_map.items():
                out_24 = idw_24h(src_vals_ph, w, idxs)
                out_24[~ok, :] = np.nan

                arrays = [pa.array(base_lat, pa.float32()), pa.array(base_lon, pa.float32()), pa.array(base_day, pa.date32())]
                for h in range(24):
                    arrays.append(pa.array(out_24[:, h].astype(np.float32, copy=False), pa.float32()))
                names = ["lat", "lon", "prism_day"] + [f"{var_key}_ph{h:02d}" for h in range(24)]
                table = pa.Table.from_arrays(arrays, names=names).cast(schemas[var_key])

                if writers[var_key] is None:
                    writers[var_key] = pq.ParquetWriter(out_paths[var_key].as_posix(), schemas[var_key], compression=compression)
                writers[var_key].write_table(table)

    for wtr in writers.values():
        if wtr is not None:
            wtr.close()

    if not quiet:
        print("[Step3] DONE.")
        for k, p in out_paths.items():
            print("  -", k, "->", p)
    return out_paths


# =============================================================================
# STEP 4: NSRDB -> PRISM-day hours -> IDW onto PRISM grid
# =============================================================================

@dataclass
class NsrdbIdwContext:
    meta: pd.DataFrame
    tree: "cKDTree"
    transformer: "Transformer"

TS_COLS_48 = [f"ts_{i:03d}" for i in range(48)]

def nsrdb_ts48_to_hourly(arr48: np.ndarray, kind: str) -> np.ndarray:
    arr = np.asarray(arr48, dtype=np.float64)
    if kind == "flux":
        return (0.5 * (arr[:, 0::2] + arr[:, 1::2])).astype(np.float32)
    if kind == "albedo":
        alb = np.clip(arr / 1000.0, 0.0, 1.0)
        return (0.5 * (alb[:, 0::2] + alb[:, 1::2])).astype(np.float32)
    if kind == "sza":
        deg = arr / 100.0
        cosz = np.cos(np.deg2rad(deg))
        cosz_h = 0.5 * (cosz[:, 0::2] + cosz[:, 1::2])
        return np.rad2deg(np.arccos(np.clip(cosz_h, -1.0, 1.0))).astype(np.float32)
    raise ValueError(f"Unknown kind '{kind}'")

def build_nsrdb_prismday_source_values(
    nsrdb_base: Path,
    meta_points: pd.DataFrame,
    D: pd.Timestamp,
    var: str,
    kind: str,
) -> np.ndarray:
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    p_m1 = nsrdb_day_var_path(nsrdb_base, Dm1, var)
    p_d = nsrdb_day_var_path(nsrdb_base, D, var)
    if not p_m1.exists():
        raise FileNotFoundError(f"Missing NSRDB file: {p_m1}")
    if not p_d.exists():
        raise FileNotFoundError(f"Missing NSRDB file: {p_d}")

    df_m1 = pd.read_parquet(p_m1, columns=["point"] + TS_COLS_48)
    df_d = pd.read_parquet(p_d, columns=["point"] + TS_COLS_48)
    df_m1 = meta_points[["point"]].merge(df_m1, on="point", how="left", sort=False)
    df_d = meta_points[["point"]].merge(df_d, on="point", how="left", sort=False)

    arr48_m1 = df_m1[TS_COLS_48].to_numpy(np.float64, copy=False)
    arr48_d = df_d[TS_COLS_48].to_numpy(np.float64, copy=False)
    hourly_m1 = nsrdb_ts48_to_hourly(arr48_m1, kind=kind)
    hourly_d = nsrdb_ts48_to_hourly(arr48_d, kind=kind)
    return combine_utc_days_to_prism_hours(hourly_m1, hourly_d)

def build_nsrdb_idw_context(nsrdb_meta: Path, quiet: bool = False) -> NsrdbIdwContext:
    try:
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("pyproj and scipy required for NSRDB IDW.") from e

    meta = pd.read_parquet(nsrdb_meta, columns=["point", "latitude", "longitude"])
    meta = meta.dropna(subset=["latitude", "longitude"]).copy()
    meta = meta.sort_values(["point"]).reset_index(drop=True)

    tfm = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    src_lat = meta["latitude"].to_numpy(np.float64, copy=False)
    src_lon = meta["longitude"].to_numpy(np.float64, copy=False)
    sx, sy = tfm.transform(src_lon, src_lat)
    src_xy = np.column_stack([sx, sy]).astype(np.float64)

    tree = cKDTree(src_xy)
    if not quiet:
        print(f"[NSRDB-IDW] points: {len(src_xy):,}")
    return NsrdbIdwContext(meta=meta, tree=tree, transformer=tfm)

def run_nsrdb_idw_prismday(
    D: pd.Timestamp,
    nsrdb_base: Path,
    prism_grid: Path,
    out_dir: Path,
    ctx: NsrdbIdwContext,
    batch_rows: int = 750_000,
    compression: str = "zstd",
    k_neighbors: int = 4,
    idw_power: float = 1.0,
    eps: float = 1e-6,
    cutoff_m: float = 4000.0,
    overwrite: bool = True,
    quiet: bool = False,
) -> Dict[str, Path]:
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as e:
        raise ImportError("scipy required for NSRDB IDW.") from e

    D = _ts(D)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf_grid = pq.ParquetFile(prism_grid.as_posix())

    specs = {
        "ghi": ("flux", "ghi"),
        "dhi": ("flux", "dhi"),
        "dni": ("flux", "dni"),
        "sza": ("sza", "sza"),
        "alb": ("albedo", "alb"),
    }

    values_map: Dict[str, np.ndarray] = {}
    for var, (kind, prefix) in specs.items():
        values_map[prefix] = build_nsrdb_prismday_source_values(nsrdb_base, ctx.meta, D, var=var, kind=kind)

    out_paths: Dict[str, Path] = {k: out_dir / f"NSRDB_{k}_prismday_{D.strftime('%Y-%m-%d')}.parquet" for k in values_map.keys()}
    for p in out_paths.values():
        if p.exists() and overwrite:
            p.unlink()

    schemas = {
        k: pa.schema(
            [pa.field("lat", pa.float32()), pa.field("lon", pa.float32()), pa.field("prism_day", pa.date32())]
            + [pa.field(f"{k}_ph{h:02d}", pa.float32()) for h in range(24)]
        )
        for k in values_map.keys()
    }

    prism_day_np = np.datetime64(D.strftime("%Y-%m-%d"), "D")
    tfm = ctx.transformer
    tree = ctx.tree

    for var_key, vals_ph in values_map.items():
        out_path = out_paths[var_key]
        schema = schemas[var_key]
        with pq.ParquetWriter(out_path.as_posix(), schema=schema, compression=compression) as writer:
            for rg in range(pf_grid.num_row_groups):
                df_rg = pf_grid.read_row_group(rg, columns=["lat", "lon"]).to_pandas()
                n = len(df_rg)
                for start in range(0, n, batch_rows):
                    end = min(start + batch_rows, n)
                    chunk = df_rg.iloc[start:end]

                    lat_t = chunk["lat"].to_numpy(np.float64, copy=False)
                    lon_t = chunk["lon"].to_numpy(np.float64, copy=False)
                    xt, yt = tfm.transform(lon_t, lat_t)
                    tar_xy = np.column_stack([xt, yt]).astype(np.float64)

                    dists, idxs = tree.query(tar_xy, k=k_neighbors, workers=-1)
                    if k_neighbors == 1:
                        dists = dists[:, None]
                        idxs = idxs[:, None]

                    ok = np.min(dists, axis=1) <= cutoff_m
                    w = (1.0 / np.maximum(dists, eps) ** idw_power).astype(np.float32, copy=False)
                    w[~np.isfinite(w)] = 0.0

                    out_24 = idw_24h(vals_ph, w, idxs)
                    out_24[~ok, :] = np.nan

                    base_lat = chunk["lat"].to_numpy(np.float32, copy=False)
                    base_lon = chunk["lon"].to_numpy(np.float32, copy=False)
                    base_day = np.full(len(base_lat), prism_day_np, dtype="datetime64[D]")

                    arrays = [pa.array(base_lat, pa.float32()), pa.array(base_lon, pa.float32()), pa.array(base_day, pa.date32())]
                    for h in range(24):
                        arrays.append(pa.array(out_24[:, h].astype(np.float32, copy=False), pa.float32()))
                    names = ["lat", "lon", "prism_day"] + [f"{var_key}_ph{h:02d}" for h in range(24)]
                    writer.write_table(pa.Table.from_arrays(arrays, names=names).cast(schema))

    if not quiet:
        print("[Step4] DONE.")
        for k, p in out_paths.items():
            print("  -", k, "->", p)
    return out_paths


# =============================================================================
# STEP 6: Heat stress metrics on PRISM-day hours -> LONG parquet
# =============================================================================

def build_keys(lat: np.ndarray, lon: np.ndarray, key_decimals: int = 6) -> np.ndarray:
    key_scale = 10 ** int(key_decimals)
    key_mult = 1_000_000_000
    lat_i = np.round(lat.astype(np.float64) * key_scale).astype(np.int64)
    lon_i = np.round(lon.astype(np.float64) * key_scale).astype(np.int64)
    return lat_i * key_mult + lon_i

def prism_time_vectors(D: pd.Timestamp, time_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D = _ts(D)
    if time_mode.lower() == "prism":
        YEAR_BY_H = np.full(24, int(D.year), dtype=np.int16)
        MONTH_BY_H = np.full(24, int(D.month), dtype=np.int8)
        DAY_BY_H = np.full(24, int(D.day), dtype=np.int8)
        TIME_BY_H = np.arange(24, dtype=np.int8)
    elif time_mode.lower() == "utc":
        utc_base = pd.Timestamp(D) - pd.Timedelta(hours=12)
        utc_hours = pd.date_range(utc_base, periods=24, freq="H")
        YEAR_BY_H = utc_hours.year.astype(np.int16)
        MONTH_BY_H = utc_hours.month.astype(np.int8)
        DAY_BY_H = utc_hours.day.astype(np.int8)
        TIME_BY_H = utc_hours.hour.astype(np.int8)
    else:
        raise ValueError("time_mode must be 'prism' or 'utc'")
    return YEAR_BY_H, MONTH_BY_H, DAY_BY_H, TIME_BY_H

def to_3d(a_2d: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a_2d.astype(np.float32, copy=False)[:, :, None])

def apply_utci_validity_mask(
    t2m_K: np.ndarray,
    wind10: np.ndarray,
    rh_pct: np.ndarray,
    e_hPa: np.ndarray,
    mrt_K: np.ndarray,
    utci_C: np.ndarray,
) -> np.ndarray:
    TA_MIN_C = -50.0
    TA_MAX_C = 50.0
    DMRT_MIN_C = -30.0
    DMRT_MAX_C = 70.0
    RH_MIN_PCT_EXCLUSIVE = 5.0
    RH_MAX_PCT = 100.0
    E_MAX_HPA_EXCLUSIVE = 50.0
    U10_MIN_MS = 0.5
    U10_MAX_MS = 17.0

    t2m_C = t2m_K - 273.15
    mrt_C = mrt_K - 273.15
    dmrt = mrt_C - t2m_C

    bad = np.zeros_like(utci_C, dtype=bool)
    bad |= (t2m_C < TA_MIN_C) | (t2m_C > TA_MAX_C)
    bad |= (dmrt < DMRT_MIN_C) | (dmrt > DMRT_MAX_C)
    bad |= (wind10 < U10_MIN_MS) | (wind10 > U10_MAX_MS)
    bad |= (~np.isfinite(rh_pct)) | (rh_pct <= RH_MIN_PCT_EXCLUSIVE) | (rh_pct > RH_MAX_PCT)
    bad |= (~np.isfinite(e_hPa)) | (e_hPa >= E_MAX_HPA_EXCLUSIVE)
    bad |= (~np.isfinite(utci_C))

    out = utci_C.astype(np.float32, copy=True)
    out[bad] = np.nan
    return out

def find_hourly_cols_any_prefix(path: Path, prefixes: Sequence[str]) -> Tuple[str, List[str]]:
    import re
    hour_re = re.compile(r"_(ph|h)(\d{2})$")
    pf = pq.ParquetFile(path.as_posix())
    names = pf.schema.names
    for pref in prefixes:
        hits = []
        for name in names:
            if not name.startswith(f"{pref}_"):
                continue
            m = hour_re.search(name)
            if not m:
                continue
            hh = int(m.group(2))
            hits.append((name, hh))
        hits = [c for c, hh in sorted(hits, key=lambda x: x[1])]
        if len(hits) == 24:
            return pref, hits
    raise ValueError(f"{path.name}: could not find 24 hourly columns for prefixes={list(prefixes)}")

def read_hourly_arrays(path: Path, prefixes: Sequence[str], key_decimals: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, hour_cols = find_hourly_cols_any_prefix(path, prefixes)
    cols = ["lat", "lon"] + hour_cols
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")
    lat = df["lat"].to_numpy(np.float64, copy=False)
    lon = df["lon"].to_numpy(np.float64, copy=False)
    keys = build_keys(lat, lon, key_decimals=key_decimals)
    order = np.argsort(keys)
    keys = keys[order]
    lat = lat[order].astype(np.float32, copy=False)
    lon = lon[order].astype(np.float32, copy=False)
    data = df[hour_cols].to_numpy(np.float32, copy=False)[order, :].T
    return lat, lon, keys, np.ascontiguousarray(data)

def intersect_all_keys(keys_list: List[np.ndarray]) -> np.ndarray:
    common = keys_list[0]
    for k in keys_list[1:]:
        common = np.intersect1d(common, k, assume_unique=True)
        if common.size == 0:
            break
    return common

def indices_for_common(keys_sorted: np.ndarray, common_keys_sorted: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(keys_sorted, common_keys_sorted)
    if np.any(idx < 0) or np.any(idx >= keys_sorted.size) or not np.all(keys_sorted[idx] == common_keys_sorted):
        raise RuntimeError("Key alignment failed.")
    return idx

def compute_heatstress_long_prismday(
    D: pd.Timestamp,
    t2m_path: Path,
    ea_path: Path,
    sp_path: Path,
    u10_path: Path,
    v10_path: Path,
    lwdown_path: Path,
    lwnet_path: Path,
    ghi_path: Path,
    dhi_path: Path,
    dni_path: Path,
    sza_path: Path,
    alb_path: Path,
    out_path: Path,
    time_mode: str = "utc",
    key_decimals: int = 6,
    chunk_points: int = 50_000,
    compression: str = "zstd",
    test_max_points: Optional[int] = None,
    quiet: bool = False,
) -> Path:
    D = _ts(D)

    try:
        from metpy.calc import heat_index
        from metpy.units import units
    except ImportError as e:
        raise ImportError("metpy required for Heat Index.") from e
    try:
        from WBGT import WBGT_Liljegren
    except ImportError as e:
        raise ImportError("WBGT package required.") from e
    try:
        import thermofeel as tf
    except ImportError as e:
        raise ImportError("thermofeel required for UTCI.") from e

    if not quiet:
        print(f"\n[Step6] Aligning hourly inputs | day={D.date()}")

    lat_t2m, lon_t2m, key_t2m, t2m_2d = read_hourly_arrays(t2m_path, ["t2m"], key_decimals)
    _, _, key_ea, ea_2d = read_hourly_arrays(ea_path, ["ea"], key_decimals)

    _, _, key_sp, sp_2d = read_hourly_arrays(sp_path, ["sp"], key_decimals)
    _, _, key_u10, u10_2d = read_hourly_arrays(u10_path, ["u10"], key_decimals)
    _, _, key_v10, v10_2d = read_hourly_arrays(v10_path, ["v10"], key_decimals)

    _, _, key_ghi, ghi_2d = read_hourly_arrays(ghi_path, ["ghi"], key_decimals)
    _, _, key_dhi, dhi_2d = read_hourly_arrays(dhi_path, ["dhi"], key_decimals)
    _, _, key_dni, dni_2d = read_hourly_arrays(dni_path, ["dni"], key_decimals)
    _, _, key_sza, sza_2d = read_hourly_arrays(sza_path, ["sza"], key_decimals)
    _, _, key_alb, alb_2d = read_hourly_arrays(alb_path, ["alb"], key_decimals)

    _, _, key_lwd, lwd_2d = read_hourly_arrays(lwdown_path, ["lwdown"], key_decimals)
    _, _, key_lwn, lwn_2d = read_hourly_arrays(lwnet_path, ["lwnet"], key_decimals)

    common = intersect_all_keys([
        key_t2m, key_ea, key_sp, key_u10, key_v10,
        key_ghi, key_dhi, key_dni, key_sza, key_alb,
        key_lwd, key_lwn
    ])
    if common.size == 0:
        raise RuntimeError("No overlap across inputs.")

    idx_t2m = indices_for_common(key_t2m, common)
    idx_ea = indices_for_common(key_ea, common)
    idx_sp = indices_for_common(key_sp, common)
    idx_u10 = indices_for_common(key_u10, common)
    idx_v10 = indices_for_common(key_v10, common)
    idx_ghi = indices_for_common(key_ghi, common)
    idx_dhi = indices_for_common(key_dhi, common)
    idx_dni = indices_for_common(key_dni, common)
    idx_sza = indices_for_common(key_sza, common)
    idx_alb = indices_for_common(key_alb, common)
    idx_lwd = indices_for_common(key_lwd, common)
    idx_lwn = indices_for_common(key_lwn, common)

    LAT = lat_t2m[idx_t2m]
    LON = lon_t2m[idx_t2m]

    T2M_C = np.ascontiguousarray(t2m_2d[:, idx_t2m].astype(np.float32, copy=False))
    EA_HPA = np.ascontiguousarray(ea_2d[:, idx_ea].astype(np.float32, copy=False))

    SP_PA = np.ascontiguousarray(sp_2d[:, idx_sp].astype(np.float32, copy=False))
    U10 = np.ascontiguousarray(u10_2d[:, idx_u10].astype(np.float32, copy=False))
    V10 = np.ascontiguousarray(v10_2d[:, idx_v10].astype(np.float32, copy=False))
    WS10 = np.ascontiguousarray(np.hypot(U10, V10).astype(np.float32, copy=False))

    GHI = np.ascontiguousarray(ghi_2d[:, idx_ghi].astype(np.float32, copy=False))
    DHI = np.ascontiguousarray(dhi_2d[:, idx_dhi].astype(np.float32, copy=False))
    DNI = np.ascontiguousarray(dni_2d[:, idx_dni].astype(np.float32, copy=False))
    SZA = np.ascontiguousarray(sza_2d[:, idx_sza].astype(np.float32, copy=False))
    ALB = np.ascontiguousarray(alb_2d[:, idx_alb].astype(np.float32, copy=False))
    LWD = np.ascontiguousarray(lwd_2d[:, idx_lwd].astype(np.float32, copy=False))
    LWN = np.ascontiguousarray(lwn_2d[:, idx_lwn].astype(np.float32, copy=False))

    npoints = T2M_C.shape[1]
    if test_max_points is not None:
        npoints = min(npoints, int(test_max_points))

    es = es_hPa(T2M_C[:, :npoints])
    es = np.maximum(es, 1e-6)
    RH = (100.0 * (EA_HPA[:, :npoints] / es)).astype(np.float32)
    RH = np.clip(RH, 0.0, 100.0).astype(np.float32, copy=False)

    YEAR_BY_H, MONTH_BY_H, DAY_BY_H, TIME_BY_H = prism_time_vectors(D, time_mode=time_mode)

    schema = pa.schema([
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("day", pa.int8()),
        pa.field("month", pa.int8()),
        pa.field("year", pa.int16()),
        pa.field("time", pa.int8()),
        pa.field("temp_C_used", pa.float32()),
        pa.field("rh_pct_used", pa.float32()),
        pa.field("HI_C", pa.float32()),
        pa.field("WBGT_C", pa.float32()),
        pa.field("UTCI_C", pa.float32()),
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    tasks = [(s, min(s + chunk_points, npoints)) for s in range(0, npoints, chunk_points)]
    WIND_IS_2M = False
    WBGT_RETURNS_KELVIN = True

    with pq.ParquetWriter(out_path.as_posix(), schema=schema, compression=compression, use_dictionary=False) as writer:
        for (start, end) in tasks:
            sl = slice(start, end)
            n = end - start

            T_c = T2M_C[:, sl]
            RHc = RH[:, sl]
            TAS_K = (T_c + 273.15).astype(np.float32, copy=False)

            T_f = T_c * (9.0 / 5.0) + 32.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hi_q = heat_index(T_f * units.degF, RHc * units.percent, mask_undefined=False).to("degC")
            hi_mag = hi_q.m
            if hasattr(hi_mag, "filled"):
                hi_mag = hi_mag.filled(np.nan)
            HI_C = np.asarray(hi_mag, dtype=np.float32)

            ghi = GHI[:, sl]
            dhi = DHI[:, sl]
            sza = SZA[:, sl]

            # cosine zenith angle (from NSRDB SZA)
            cza = np.cos(np.deg2rad(sza)).astype(np.float32, copy=False)

            # Kong/Huber: set czda<=0 to an arbitrary negative value (e.g., -0.5)
            # to avoid division-by-zero / instability in direct-beam terms
            czda = np.where(cza <= 0.0, -0.5, cza).astype(np.float32, copy=False)

            # direct-beam fraction: avoid nonsense when ghi is tiny/zero
            GHI_MIN = 1.0  # W/m^2 (you can set 5.0 if you want stricter)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ghi > GHI_MIN, dhi / ghi, 0.0)

            fdir_frac = np.clip(1.0 - ratio, 0.0, 1.0).astype(np.float32, copy=False)
            fdir_frac = np.where(ghi > GHI_MIN, fdir_frac, 0.0).astype(np.float32, copy=False)


            wbgt_3d = WBGT_Liljegren(
                to_3d(TAS_K),
                to_3d(RHc),
                to_3d(SP_PA[:, sl]),
                to_3d(WS10[:, sl]),
                to_3d(ghi),
                to_3d(fdir_frac),
                to_3d(czda),
                WIND_IS_2M,
            )
            wbgt_2d = np.asarray(wbgt_3d, dtype=np.float32)[:, :, 0]
            WBGT_C = (wbgt_2d - 273.15).astype(np.float32, copy=False) if WBGT_RETURNS_KELVIN else wbgt_2d
            # mask rare solver failures / non-physical values
            WBGT_C[(WBGT_C < -60.0) | (WBGT_C > 80.0)] = np.nan

            w10 = WS10[:, sl]
            e = EA_HPA[:, sl]
            dn = DNI[:, sl]
            a = ALB[:, sl]
            lwd = LWD[:, sl]
            lwn = LWN[:, sl]

            invalid_inputs = (
                ~np.isfinite(TAS_K) | ~np.isfinite(w10) | ~np.isfinite(RHc) | ~np.isfinite(e) |
                ~np.isfinite(ghi) | ~np.isfinite(dn) | ~np.isfinite(sza) | ~np.isfinite(a) |
                ~np.isfinite(lwd) | ~np.isfinite(lwn)
            )

            t2mK_s = np.where(np.isfinite(TAS_K), TAS_K, 273.15).astype(np.float32, copy=False)
            w10_s = np.where(np.isfinite(w10), w10, 1.0).astype(np.float32, copy=False)
            e_s = np.where(np.isfinite(e), e, 10.0).astype(np.float32, copy=False)

            a_s = np.clip(np.where(np.isfinite(a), a, 0.2), 0.0, 1.0).astype(np.float32, copy=False)
            sza_s = np.where(np.isfinite(sza), sza, 180.0).astype(np.float32, copy=False)
            cossza = np.cos(np.deg2rad(sza_s)).astype(np.float32, copy=False)
            cossza = np.where(cossza > 0.0, cossza, 0.0).astype(np.float32, copy=False)

            ssrd = np.where(ghi > 0.0, ghi, 0.0).astype(np.float32, copy=False)
            ssr = ((1.0 - a_s) * ssrd).astype(np.float32, copy=False)
            dsrp = np.where(dn > 0.0, dn, 0.0).astype(np.float32, copy=False)
            fdir_wm2 = (dsrp * cossza).astype(np.float32, copy=False)

            mrt_K = tf.calculate_mean_radiant_temperature(ssrd, ssr, dsrp, lwd, fdir_wm2, lwn, cossza).astype(np.float32)
            utci_K = tf.calculate_utci(t2mK_s, w10_s, mrt_K, ehPa=e_s).astype(np.float32)
            utci_C = tf.kelvin_to_celsius(utci_K).astype(np.float32)
            UTCI_C = apply_utci_validity_mask(t2mK_s, w10_s, RHc.astype(np.float32, copy=False), e_s, mrt_K, utci_C)
            UTCI_C[invalid_inputs] = np.nan

            lat = LAT[sl].astype(np.float32, copy=False)
            lon = LON[sl].astype(np.float32, copy=False)

            lat_long = np.repeat(lat, 24)
            lon_long = np.repeat(lon, 24)
            day_long = np.tile(DAY_BY_H, n)
            month_long = np.tile(MONTH_BY_H, n)
            year_long = np.tile(YEAR_BY_H, n)
            time_long = np.tile(TIME_BY_H, n)

            temp_long = T_c.T.reshape(-1).astype(np.float32, copy=False)
            rh_long = RHc.T.reshape(-1).astype(np.float32, copy=False)
            hi_long = HI_C.T.reshape(-1).astype(np.float32, copy=False)
            wbgt_long = WBGT_C.T.reshape(-1).astype(np.float32, copy=False)
            utci_long = UTCI_C.T.reshape(-1).astype(np.float32, copy=False)

            table = pa.Table.from_arrays(
                [
                    pa.array(lat_long, pa.float32()),
                    pa.array(lon_long, pa.float32()),
                    pa.array(day_long, pa.int8()),
                    pa.array(month_long, pa.int8()),
                    pa.array(year_long, pa.int16()),
                    pa.array(time_long, pa.int8()),
                    pa.array(temp_long, pa.float32()),
                    pa.array(rh_long, pa.float32()),
                    pa.array(hi_long, pa.float32()),
                    pa.array(wbgt_long, pa.float32()),
                    pa.array(utci_long, pa.float32()),
                ],
                schema=schema,
            )
            writer.write_table(table)

    if not quiet:
        print(f"[Step6] DONE: {out_path}")
    return out_path


# =============================================================================
# STEP 7: Tract AREA-weighted AND POP-weighted averages -> ONE combined parquet
# =============================================================================

@dataclass
class TractMapping:
    keys_sorted: np.ndarray
    tract_code_sorted: np.ndarray
    geoid_by_code: np.ndarray
    rep_ll: Optional[Dict[str, Tuple[np.float32, np.float32]]]

def load_tract_mapping(map_dir: Path, vintage: int, key_decimals: int, quiet: bool = False) -> TractMapping:
    map_path = map_dir / f"gridkey_to_tractcode_v{vintage}_key{key_decimals}.parquet"
    lut_path = map_dir / f"tractcode_to_geoid_v{vintage}.parquet"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing tract map: {map_path}")
    if not lut_path.exists():
        raise FileNotFoundError(f"Missing tract LUT: {lut_path}")

    tab = pq.read_table(map_path.as_posix(), columns=["key", "tract_code"])
    keys = tab["key"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    codes = tab["tract_code"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    if keys.size and np.any(keys[1:] < keys[:-1]):
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        codes = codes[order]

    lut = pq.read_table(lut_path.as_posix(), columns=["tract_code", "GEOID"]).to_pandas().sort_values("tract_code")
    geoid_by_code = lut["GEOID"].astype(str).to_numpy()

    if not quiet:
        print(f"[tract-map] keys={keys.size:,} tracts={geoid_by_code.size:,}")
    return TractMapping(keys_sorted=keys, tract_code_sorted=codes, geoid_by_code=geoid_by_code, rep_ll=None)

def load_rep_points(tract_shp: Path, area_crs: str = "EPSG:5070", ll_crs: str = "EPSG:4326") -> Dict[str, Tuple[np.float32, np.float32]]:
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas required to compute representative points.") from e

    gdf = gpd.read_file(tract_shp)
    geoid_field = "GISJOIN2" if "GISJOIN2" in gdf.columns else ("GISJOIN" if "GISJOIN" in gdf.columns else None)
    if geoid_field is None:
        raise ValueError(f"{tract_shp}: expected GISJOIN2 or GISJOIN field.")
    gdf = gdf[[geoid_field, "geometry"]].copy().to_crs(area_crs)
    rep = gdf.geometry.representative_point()
    rep_ll = gpd.GeoSeries(rep, crs=area_crs).to_crs(ll_crs)

    geoid = gdf[geoid_field].astype(str).to_numpy()
    lat = rep_ll.y.astype(np.float32).to_numpy()
    lon = rep_ll.x.astype(np.float32).to_numpy()
    return {geoid[i]: (lat[i], lon[i]) for i in range(len(geoid))}

def load_worldpop_pop_aligned_to_mapping(
    worldpop_weights_dir: Path,
    prism_day_year: int,
    mapping_keys_sorted: np.ndarray,
    key_decimals: int,
    pre2000_fallback_year: int = 2000,
    quiet: bool = False,
) -> Tuple[int, np.ndarray]:
    pop_year, p = resolve_worldpop_year_path(
        worldpop_weights_dir=worldpop_weights_dir,
        year=int(prism_day_year),
        key_decimals=int(key_decimals),
        pre2000_fallback_year=int(pre2000_fallback_year),
    )
    if not quiet:
        print(f"[pop] loading: year={pop_year} file={p.name}")

    tab = pq.read_table(p.as_posix(), columns=["key", "pop"])
    keys = tab["key"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    pop = tab["pop"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    if keys.size and np.any(keys[1:] < keys[:-1]):
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        pop = pop[order]

    if keys.size == mapping_keys_sorted.size and np.array_equal(keys, mapping_keys_sorted):
        return pop_year, pop

    idx = np.searchsorted(keys, mapping_keys_sorted)
    ok = (idx >= 0) & (idx < keys.size) & (keys[idx] == mapping_keys_sorted)
    if not np.all(ok):
        raise RuntimeError("WorldPop keys do not cover mapping keys; check key_decimals/grid alignment.")
    return pop_year, pop[idx].astype(np.float32, copy=False)


def tract_area_and_pop_weighted_from_long(
    in_path: Path,
    out_path: Path,
    mapping: TractMapping,
    pop_by_key_sorted: Optional[np.ndarray],
    D: pd.Timestamp,
    time_mode: str = "utc",
    key_decimals: int = 6,
    batch_rows: int = 2_000_000,
    compression: str = "zstd",
    use_coslat_weights_if_degrees: bool = True,
    quiet: bool = False,
) -> Path:
    """
    Combined output:
      GEOID, lat, lon, day, month, year, time,
      <value>_area, <value>_pop
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Missing heatstress long parquet: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    pf = pq.ParquetFile(in_path.as_posix())
    names = pf.schema.names

    required = ["lat", "lon", "year", "month", "day", "time"]
    for c in required:
        if c not in names:
            raise KeyError(f"Missing required column {c}. Found: {names}")

    sch = pf.schema_arrow
    value_cols: List[str] = []
    for c in names:
        if c in required:
            continue
        t = sch.field(c).type
        if pa.types.is_dictionary(t):
            t = t.value_type
        if pa.types.is_floating(t) or pa.types.is_integer(t):
            value_cols.append(c)
    if not value_cols:
        raise RuntimeError("No numeric value columns found to aggregate.")

    YEAR_BY_H, MONTH_BY_H, DAY_BY_H, TIME_BY_H = prism_time_vectors(_ts(D), time_mode=time_mode)
    tkeys = (
        YEAR_BY_H.astype(np.int64) * 1_000_000
        + MONTH_BY_H.astype(np.int64) * 10_000
        + DAY_BY_H.astype(np.int64) * 100
        + TIME_BY_H.astype(np.int64)
    ).astype(np.int64)
    H = int(tkeys.size)

    keys_sorted = mapping.keys_sorted
    tract_code_sorted = mapping.tract_code_sorted
    geoid_by_code = mapping.geoid_by_code
    ntracts = int(geoid_by_code.size)

    do_pop = pop_by_key_sorted is not None
    if do_pop and (pop_by_key_sorted.shape[0] != keys_sorted.shape[0]):
        raise ValueError("pop_by_key_sorted must align to mapping.keys_sorted.")

    ngroups = ntracts * H
    sums_area = {c: np.zeros(ngroups, dtype=np.float64) for c in value_cols}
    wgts_area = {c: np.zeros(ngroups, dtype=np.float64) for c in value_cols}
    if do_pop:
        sums_pop = {c: np.zeros(ngroups, dtype=np.float64) for c in value_cols}
        wgts_pop = {c: np.zeros(ngroups, dtype=np.float64) for c in value_cols}
    else:
        sums_pop = {}
        wgts_pop = {}

    cols_to_read = ["lat", "lon", "year", "month", "day", "time"] + value_cols

    def make_tkey(y, m, d, t):
        return (y.astype(np.int64) * 1_000_000 + m.astype(np.int64) * 10_000 + d.astype(np.int64) * 100 + t.astype(np.int64))

    rows_seen = 0
    rows_matched = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols_to_read, use_threads=True):
        df = batch.to_pandas()
        lat = df["lat"].to_numpy(np.float64, copy=False)
        lon = df["lon"].to_numpy(np.float64, copy=False)
        key = build_keys(lat, lon, key_decimals=key_decimals)

        idx = np.searchsorted(keys_sorted, key)
        match = (idx >= 0) & (idx < keys_sorted.size) & (keys_sorted[idx] == key)

        tract_code = np.full(lat.size, -1, dtype=np.int32)
        tract_code[match] = tract_code_sorted[idx[match]]

        tk = make_tkey(df["year"].to_numpy(), df["month"].to_numpy(), df["day"].to_numpy(), df["time"].to_numpy())
        tix = np.searchsorted(tkeys, tk)
        valid_t = (tix >= 0) & (tix < H) & (tkeys[tix] == tk)

        valid = (tract_code >= 0) & valid_t
        rows_seen += lat.size
        rows_matched += int(valid.sum())
        if not np.any(valid):
            continue

        tract_v = tract_code[valid].astype(np.int64, copy=False)
        tix_v = tix[valid].astype(np.int64, copy=False)
        group = tract_v * H + tix_v

        if use_coslat_weights_if_degrees:
            w_area = np.cos(np.deg2rad(lat[valid])).astype(np.float64, copy=False)
            w_area = np.where(np.isfinite(w_area) & (w_area > 0.0), w_area, 0.0)
        else:
            w_area = np.ones(int(valid.sum()), dtype=np.float64)

        if do_pop:
            w_pop = pop_by_key_sorted[idx[valid]].astype(np.float64, copy=False)
            w_pop = np.where(np.isfinite(w_pop) & (w_pop > 0.0), w_pop, 0.0)

        for c in value_cols:
            v = df[c].to_numpy(np.float64, copy=False)[valid]

            good_a = np.isfinite(v) & (w_area > 0.0)
            if np.any(good_a):
                gid = group[good_a]
                vv = v[good_a]
                ww = w_area[good_a]
                sums_area[c] += np.bincount(gid, weights=vv * ww, minlength=ngroups)
                wgts_area[c] += np.bincount(gid, weights=ww, minlength=ngroups)

            if do_pop:
                good_p = np.isfinite(v) & (w_pop > 0.0)
                if np.any(good_p):
                    gid = group[good_p]
                    vv = v[good_p]
                    ww = w_pop[good_p]
                    sums_pop[c] += np.bincount(gid, weights=vv * ww, minlength=ngroups)
                    wgts_pop[c] += np.bincount(gid, weights=ww, minlength=ngroups)

    if not quiet:
        print(f"[Step7] streamed rows={rows_seen:,} matched={rows_matched:,} ({rows_matched/rows_seen:.3%})")

    means_area: Dict[str, np.ndarray] = {}
    for c in value_cols:
        out = np.full(ngroups, np.nan, dtype=np.float32)
        m = wgts_area[c] > 0
        out[m] = (sums_area[c][m] / wgts_area[c][m]).astype(np.float32)
        means_area[c] = out.reshape(ntracts, H)

    means_pop: Dict[str, np.ndarray] = {}
    if do_pop:
        for c in value_cols:
            out = np.full(ngroups, np.nan, dtype=np.float32)
            m = wgts_pop[c] > 0
            out[m] = (sums_pop[c][m] / wgts_pop[c][m]).astype(np.float32)
            means_pop[c] = out.reshape(ntracts, H)

    # UNION keep-set
    cov_area = np.zeros(ntracts, dtype=np.float64)
    for c in value_cols:
        cov_area = np.maximum(cov_area, wgts_area[c].reshape(ntracts, H).sum(axis=1))
    keep_area = cov_area > 0

    if do_pop:
        cov_pop = np.zeros(ntracts, dtype=np.float64)
        for c in value_cols:
            cov_pop = np.maximum(cov_pop, wgts_pop[c].reshape(ntracts, H).sum(axis=1))
        keep_pop = cov_pop > 0
    else:
        keep_pop = np.zeros(ntracts, dtype=bool)

    keep = keep_area | keep_pop
    keep_codes = np.nonzero(keep)[0].astype(np.int32)
    n_keep = int(keep_codes.size)

    y = (tkeys // 1_000_000).astype(np.int16)
    m = ((tkeys % 1_000_000) // 10_000).astype(np.int8)
    d = ((tkeys % 10_000) // 100).astype(np.int8)
    t = (tkeys % 100).astype(np.int8)

    out_year = np.tile(y, n_keep).astype(np.int16, copy=False)
    out_month = np.tile(m, n_keep).astype(np.int8, copy=False)
    out_day = np.tile(d, n_keep).astype(np.int8, copy=False)
    out_time = np.tile(t, n_keep).astype(np.int8, copy=False)

    geoid_keep = geoid_by_code[keep_codes]

    if mapping.rep_ll is None:
        lat_keep = np.full(n_keep, np.nan, dtype=np.float32)
        lon_keep = np.full(n_keep, np.nan, dtype=np.float32)
    else:
        ll = mapping.rep_ll
        lat_keep = np.array([ll.get(g, (np.nan, np.nan))[0] for g in geoid_keep], dtype=np.float32)
        lon_keep = np.array([ll.get(g, (np.nan, np.nan))[1] for g in geoid_keep], dtype=np.float32)

    out_lat = np.repeat(lat_keep, H).astype(np.float32, copy=False)
    out_lon = np.repeat(lon_keep, H).astype(np.float32, copy=False)

    dict_values = pa.array(geoid_keep.tolist(), type=pa.string())
    dict_index = pa.array(np.repeat(np.arange(n_keep, dtype=np.int32), H), type=pa.int32())
    geoid_arr = pa.DictionaryArray.from_arrays(dict_index, dict_values)

    out_fields = [
        pa.field("GEOID", pa.dictionary(pa.int32(), pa.string())),
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("day", pa.int8()),
        pa.field("month", pa.int8()),
        pa.field("year", pa.int16()),
        pa.field("time", pa.int8()),
    ]
    for c in value_cols:
        out_fields.append(pa.field(f"{c}_area", pa.float32()))
        out_fields.append(pa.field(f"{c}_pop", pa.float32()))
    out_schema = pa.schema(out_fields)

    out_dict: Dict[str, object] = {
        "GEOID": geoid_arr,
        "lat": out_lat,
        "lon": out_lon,
        "day": out_day,
        "month": out_month,
        "year": out_year,
        "time": out_time,
    }

    nrows = n_keep * H
    for c in value_cols:
        out_dict[f"{c}_area"] = means_area[c][keep_codes].reshape(-1).astype(np.float32, copy=False)
        if do_pop:
            out_dict[f"{c}_pop"] = means_pop[c][keep_codes].reshape(-1).astype(np.float32, copy=False)
        else:
            out_dict[f"{c}_pop"] = np.full(nrows, np.nan, dtype=np.float32)

    out_table = pa.Table.from_pydict(out_dict, schema=out_schema)
    if not quiet:
        print(f"[Step7] Writing COMBINED -> {out_path} rows={out_table.num_rows:,}")
    pq.write_table(out_table, out_path.as_posix(), compression=compression)
    return out_path


# =============================================================================
# Pipeline config + preflight + orchestration
# =============================================================================

@dataclass
class PipelineConfig:
    # Inputs
    era_base: Path
    prism_base: Path
    prism_tdmean_folder: str
    nsrdb_base: Path
    nsrdb_meta: Path
    prism_grid: Path

    # WorldPop
    worldpop_weights_dir: Path
    worldpop_pre2000_year: int
    do_pop_weighted: bool

    # Tracts
    tract_map_dir: Path
    tract_vintage: int
    tract_shp: Path
    load_rep_points: bool

    # Outputs
    work_dir: Path
    out_dir: Path

    # Behavior
    time_mode: str
    key_decimals: int
    ll_decimals_era: int
    keep_fused: bool
    keep_ancillary: bool
    keep_long: bool
    overwrite: bool
    quiet: bool

    # Tunables
    window_size: int
    compression: str
    idw_batch_rows: int
    idw_k: int
    idw_power: float
    era_cutoff_m: float
    nsrdb_cutoff_m: float
    heatstress_chunk_points: int
    heatstress_test_max_points: Optional[int]
    fuse_test_max_windows: Optional[int]

def preflight_day(cfg: PipelineConfig, D: pd.Timestamp) -> List[str]:
    D = _ts(D)
    Dm1 = D - pd.Timedelta(days=1)
    missing: List[str] = []

    # PRISM
    for folder in ("tmin", "tmax", cfg.prism_tdmean_folder):
        p = prism_day_tif(cfg.prism_base, folder, D)
        if not p.exists():
            missing.append(str(p))

    # ERA vars needed
    for var in ("t2m", "d2m", "sp", "u10", "v10", "str", "strd"):
        for day in (Dm1, D):
            try:
                era_day_path(cfg.era_base, var, day)
            except FileNotFoundError as e:
                missing.append(str(e).splitlines()[0])

    # NSRDB vars for D-1 and D
    for var in ("ghi", "dhi", "dni", "sza", "alb"):
        for day in (Dm1, D):
            p = nsrdb_day_var_path(cfg.nsrdb_base, day, var)
            if not p.exists():
                missing.append(str(p))

    for p in (cfg.nsrdb_meta, cfg.prism_grid):
        if not p.exists():
            missing.append(str(p))

    # WorldPop
    if cfg.do_pop_weighted:
        try:
            _, p = resolve_worldpop_year_path(cfg.worldpop_weights_dir, int(D.year), cfg.key_decimals, cfg.worldpop_pre2000_year)
            if not p.exists():
                missing.append(str(p))
        except FileNotFoundError as e:
            missing.append(str(e))

    # tract maps
    map_path = cfg.tract_map_dir / f"gridkey_to_tractcode_v{cfg.tract_vintage}_key{cfg.key_decimals}.parquet"
    lut_path = cfg.tract_map_dir / f"tractcode_to_geoid_v{cfg.tract_vintage}.parquet"
    for p in (map_path, lut_path):
        if not p.exists():
            missing.append(str(p))

    if cfg.load_rep_points and (not cfg.tract_shp.exists()):
        missing.append(str(cfg.tract_shp))

    return missing


def process_one_day(
    cfg: PipelineConfig,
    D: pd.Timestamp,
    era_ctx: EraIdwContext,
    nsrdb_ctx: NsrdbIdwContext,
    tract_map: TractMapping,
    pop_cache: "OrderedDict[int, np.ndarray]",
) -> Path:
    D = _ts(D)

    start_d, end_d = prism_window_dates(D)
    if not cfg.quiet:
        print("\n" + "=" * 100)
        print(f"PRISM_DAY={D.date()} (UTC window: {start_d.date()} 12Z .. {end_d.date()} 11Z)")
        print("=" * 100)

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    t2m_path = cfg.work_dir / f"t2m_prismday_{D.strftime('%Y-%m-%d')}.parquet"
    ea_path = cfg.work_dir / f"ea_prismday_{D.strftime('%Y-%m-%d')}.parquet"
    era_out_dir = cfg.work_dir / "era_idw_prismday"
    nsrdb_out_dir = cfg.work_dir / "nsrdb_idw_prismday"
    hs_long = cfg.work_dir / f"heatstress_long_prismday_{D.strftime('%Y-%m-%d')}.parquet"

    pop_year_used: Optional[int] = None
    if cfg.do_pop_weighted:
        pop_year_used, _ = resolve_worldpop_year_path(
            cfg.worldpop_weights_dir, int(D.year), int(cfg.key_decimals), pre2000_fallback_year=int(cfg.worldpop_pre2000_year)
        )

    if pop_year_used is not None:
        tract_out = cfg.out_dir / (
            f"heatstress_tract_area_and_popweighted_{start_d.strftime('%Y-%m-%d')}_{end_d.strftime('%Y-%m-%d')}"
            f"_popy{int(pop_year_used)}_v{cfg.tract_vintage}.parquet"
        )
    else:
        tract_out = cfg.out_dir / (
            f"heatstress_tract_area_and_popweighted_{start_d.strftime('%Y-%m-%d')}_{end_d.strftime('%Y-%m-%d')}"
            f"_v{cfg.tract_vintage}.parquet"
        )

    if tract_out.exists() and (not cfg.overwrite):
        if not cfg.quiet:
            print(f"[SKIP] exists: {tract_out}")
        return tract_out

    # Step1+2
    if (not t2m_path.exists()) or (not ea_path.exists()) or cfg.overwrite:
        fuse_t2m_and_ea_prismday(
            D=D,
            era_base=cfg.era_base,
            prism_base=cfg.prism_base,
            prism_tdmean_folder=cfg.prism_tdmean_folder,
            out_t2m=t2m_path,
            out_ea=ea_path,
            window_size=cfg.window_size,
            compression=cfg.compression,
            test_max_windows=cfg.fuse_test_max_windows,
            quiet=cfg.quiet,
        )

    # Step3
    era_outputs = run_era_idw_prismday(
        D=D,
        era_base=cfg.era_base,
        prism_grid=cfg.prism_grid,
        out_dir=era_out_dir,
        ctx=era_ctx,
        ll_decimals=cfg.ll_decimals_era,
        batch_rows=cfg.idw_batch_rows,
        compression=cfg.compression,
        k_neighbors=cfg.idw_k,
        idw_power=cfg.idw_power,
        cutoff_m=cfg.era_cutoff_m,
        overwrite=cfg.overwrite,
        quiet=cfg.quiet,
    )

    # Step4
    nsrdb_outputs = run_nsrdb_idw_prismday(
        D=D,
        nsrdb_base=cfg.nsrdb_base,
        prism_grid=cfg.prism_grid,
        out_dir=nsrdb_out_dir,
        ctx=nsrdb_ctx,
        batch_rows=cfg.idw_batch_rows,
        compression=cfg.compression,
        k_neighbors=cfg.idw_k,
        idw_power=cfg.idw_power,
        cutoff_m=cfg.nsrdb_cutoff_m,
        overwrite=cfg.overwrite,
        quiet=cfg.quiet,
    )

    sp_path = era_outputs["sp"]
    u10_path = era_outputs["u10"]
    v10_path = era_outputs["v10"]
    lwdown_path = era_outputs["lwdown"]
    lwnet_path = era_outputs["lwnet"]

    ghi_path = nsrdb_outputs["ghi"]
    dhi_path = nsrdb_outputs["dhi"]
    dni_path = nsrdb_outputs["dni"]
    sza_path = nsrdb_outputs["sza"]
    alb_path = nsrdb_outputs["alb"]

    # Step6
    if (not hs_long.exists()) or cfg.overwrite:
        compute_heatstress_long_prismday(
            D=D,
            t2m_path=t2m_path,
            ea_path=ea_path,
            sp_path=sp_path,
            u10_path=u10_path,
            v10_path=v10_path,
            lwdown_path=lwdown_path,
            lwnet_path=lwnet_path,
            ghi_path=ghi_path,
            dhi_path=dhi_path,
            dni_path=dni_path,
            sza_path=sza_path,
            alb_path=alb_path,
            out_path=hs_long,
            time_mode=cfg.time_mode,
            key_decimals=cfg.key_decimals,
            chunk_points=cfg.heatstress_chunk_points,
            compression=cfg.compression,
            test_max_points=cfg.heatstress_test_max_points,
            quiet=cfg.quiet,
        )

    # pop weights
    pop_aligned: Optional[np.ndarray] = None
    if cfg.do_pop_weighted:
        assert pop_year_used is not None
        if pop_year_used in pop_cache:
            pop_cache.move_to_end(pop_year_used)
            pop_aligned = pop_cache[pop_year_used]
        else:
            pop_year_used2, pop_aligned = load_worldpop_pop_aligned_to_mapping(
                cfg.worldpop_weights_dir, int(D.year), tract_map.keys_sorted, cfg.key_decimals, cfg.worldpop_pre2000_year, quiet=cfg.quiet
            )
            pop_cache[pop_year_used2] = pop_aligned
            pop_cache.move_to_end(pop_year_used2)
            while len(pop_cache) > 2:
                pop_cache.popitem(last=False)

    # Step7 combined
    tract_area_and_pop_weighted_from_long(
        in_path=hs_long,
        out_path=tract_out,
        mapping=tract_map,
        pop_by_key_sorted=pop_aligned,
        D=D,
        time_mode=cfg.time_mode,
        key_decimals=cfg.key_decimals,
        compression=cfg.compression,
        quiet=cfg.quiet,
    )

    # cleanup
    if not cfg.keep_long and hs_long.exists():
        hs_long.unlink(missing_ok=True)
    if not cfg.keep_ancillary:
        for p in list(era_outputs.values()) + list(nsrdb_outputs.values()):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
    if not cfg.keep_fused:
        for p in (t2m_path, ea_path):
            p.unlink(missing_ok=True)

    if not cfg.quiet:
        print(f"\n✅ Final combined tract output: {tract_out}")
    return tract_out


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="heatstress_prismday_to_tract.py",
        description="End-to-end PRISM-day HeatStress -> tract combined (area+pop) parquet.",
    )

    g = p.add_argument_group("Date selection")
    g.add_argument("--start", type=str, default=None, help="Start PRISM day (YYYY-MM-DD)")
    g.add_argument("--end", type=str, default=None, help="End PRISM day (YYYY-MM-DD), inclusive")
    g.add_argument("--year", type=int, default=None, help="Alternative: year for DOY mode")
    g.add_argument("--doy-start", type=int, default=None, help="Day-of-year start (1..366)")
    g.add_argument("--doy-end", type=int, default=None, help="Day-of-year end (1..366), inclusive")

    g = p.add_argument_group("Paths (defaults)")
    g.add_argument("--era-base", type=Path, default=Path("/home/rr692/WeatherData_ERALand"))
    g.add_argument("--prism-base", type=Path, default=Path("/home/rr692/PRISM Data"))
    g.add_argument("--prism-tdmean-folder", type=str, default="tdmean")
    g.add_argument("--nsrdb-base", type=Path, default=Path("/home/rr692/NSRDB_Data_Extraction/stage1_parquet_4km"))
    g.add_argument("--nsrdb-meta", type=Path, default=Path("/home/rr692/NSRDB_point_lat_lon_meta.parquet"))
    g.add_argument("--prism-grid", type=Path, default=Path("/home/rr692/DEM/prism800m_validcells_mean_elevation.parquet"))

    g.add_argument("--worldpop-weights-dir", type=Path, default=Path("/home/rr692/HeatStress_Output/worldpop_prism800m_weights_key6"))
    g.add_argument("--worldpop-pre2000-year", type=int, default=2000)
    g.add_argument("--no-pop-weighted", action="store_true", help="Disable pop-weighting (pop columns become NaN).")

    g.add_argument("--tract-map-dir", type=Path, default=Path("/home/rr692/HeatStress_Output/tract_maps_static"))
    g.add_argument("--tract-vintage", type=int, default=1990)
    g.add_argument("--tract-shp", type=Path, default=Path("/home/rr692/Census Data/nhgis0002_shape/nhgis0002_shape/nhgis0002_shapefile_tl2000_us_tract_1990/US_tract_1990.shp"))

    g.add_argument("--work-dir", type=Path, default=Path("/home/rr692/HeatStress_Output/work_test"))
    g.add_argument("--out-dir", type=Path, default=Path("/home/rr692/HeatStress_Output/tract_outputs"))

    g = p.add_argument_group("Behavior")
    g.add_argument("--time-mode", type=str, default="utc", choices=["utc", "prism"])
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--preflight-only", action="store_true")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--keep-fused", action="store_true")
    g.add_argument("--keep-ancillary", action="store_true")
    g.add_argument("--keep-long", action="store_true")
    g.add_argument("--no-rep-points", action="store_true")

    g = p.add_argument_group("Performance knobs")
    g.add_argument("--window-size", type=int, default=512)
    g.add_argument("--compression", type=str, default="zstd")
    g.add_argument("--key-decimals", type=int, default=6)
    g.add_argument("--ll-decimals-era", type=int, default=6)
    g.add_argument("--idw-batch-rows", type=int, default=750_000)
    g.add_argument("--idw-k", type=int, default=4)
    g.add_argument("--idw-power", type=float, default=1.0)
    g.add_argument("--era-cutoff-m", type=float, default=8000.0)
    g.add_argument("--nsrdb-cutoff-m", type=float, default=4000.0)
    g.add_argument("--heatstress-chunk-points", type=int, default=50_000)
    g.add_argument("--heatstress-test-max-points", type=int, default=None)
    g.add_argument("--fuse-test-max-windows", type=int, default=None)

    return p

def parse_dates_from_args(args: argparse.Namespace) -> List[pd.Timestamp]:
    if args.year is not None or args.doy_start is not None or args.doy_end is not None:
        if args.year is None or args.doy_start is None:
            raise ValueError("DOY mode requires --year and --doy-start")
        doy_end = args.doy_end if args.doy_end is not None else args.doy_start
        start = doy_to_date(args.year, args.doy_start)
        end = doy_to_date(args.year, doy_end)
        return date_range_inclusive(start, end)

    if args.start is None:
        raise ValueError("Provide --start/--end or --year/--doy-start.")
    start = _ts(args.start)
    end = _ts(args.end) if args.end is not None else start
    return date_range_inclusive(start, end)

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    dates = parse_dates_from_args(args)

    cfg = PipelineConfig(
        era_base=args.era_base,
        prism_base=args.prism_base,
        prism_tdmean_folder=args.prism_tdmean_folder,
        nsrdb_base=args.nsrdb_base,
        nsrdb_meta=args.nsrdb_meta,
        prism_grid=args.prism_grid,

        worldpop_weights_dir=args.worldpop_weights_dir,
        worldpop_pre2000_year=int(args.worldpop_pre2000_year),
        do_pop_weighted=(not args.no_pop_weighted),

        tract_map_dir=args.tract_map_dir,
        tract_vintage=int(args.tract_vintage),
        tract_shp=args.tract_shp,
        load_rep_points=(not args.no_rep_points),

        work_dir=args.work_dir,
        out_dir=args.out_dir,

        time_mode=args.time_mode,
        key_decimals=int(args.key_decimals),
        ll_decimals_era=int(args.ll_decimals_era),
        keep_fused=bool(args.keep_fused),
        keep_ancillary=bool(args.keep_ancillary),
        keep_long=bool(args.keep_long),
        overwrite=bool(args.overwrite),
        quiet=bool(args.quiet),

        window_size=int(args.window_size),
        compression=str(args.compression),
        idw_batch_rows=int(args.idw_batch_rows),
        idw_k=int(args.idw_k),
        idw_power=float(args.idw_power),
        era_cutoff_m=float(args.era_cutoff_m),
        nsrdb_cutoff_m=float(args.nsrdb_cutoff_m),
        heatstress_chunk_points=int(args.heatstress_chunk_points),
        heatstress_test_max_points=args.heatstress_test_max_points,
        fuse_test_max_windows=args.fuse_test_max_windows,
    )

    any_missing = False
    for D in dates:
        miss = preflight_day(cfg, D)
        if miss:
            any_missing = True
            print(f"\n[PRECHECK] Missing inputs for PRISM_DAY={_ts(D).date()}:")
            for m in miss[:200]:
                print("  -", m)
        else:
            if not cfg.quiet:
                print(f"[PRECHECK] OK inputs for PRISM_DAY={_ts(D).date()}")

    if args.preflight_only:
        return 1 if any_missing else 0
    if any_missing:
        print("\nERROR: Missing required inputs. Re-run with --preflight-only.")
        return 2

    era_ctx = build_era_idw_context(cfg.era_base, dates[0], ll_decimals=cfg.ll_decimals_era, quiet=cfg.quiet)
    nsrdb_ctx = build_nsrdb_idw_context(cfg.nsrdb_meta, quiet=cfg.quiet)

    tract_map = load_tract_mapping(cfg.tract_map_dir, cfg.tract_vintage, cfg.key_decimals, quiet=cfg.quiet)
    if cfg.load_rep_points:
        tract_map.rep_ll = load_rep_points(cfg.tract_shp)
    else:
        tract_map.rep_ll = None

    pop_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    outputs: List[Path] = []
    t0 = time.time()
    for D in dates:
        outputs.append(process_one_day(cfg, D, era_ctx, nsrdb_ctx, tract_map, pop_cache))

    if not cfg.quiet:
        print("\nALL DONE.")
        print(f"Days processed: {len(outputs)} | wall time: {time.time() - t0:.1f}s")
        for p in outputs:
            print(" -", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
