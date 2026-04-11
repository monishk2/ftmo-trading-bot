#!/usr/bin/env python3
"""
data/news_calendar.py
=====================
Generate a calendar of high-impact USD economic events (Jan 2022 – Apr 2025).

All dates are hardcoded from BLS / BEA / FOMC published release schedules.
Saves: data/news_calendar.parquet

Columns
-------
  datetime    – event timestamp in US/Eastern (tz-aware)
  event_type  – 'NFP' | 'FOMC' | 'CPI' | 'PPI' | 'RETAIL_SALES' | 'GDP_ADV' | 'PCE' | 'ISM_PMI'
  impact      – 'VERY_HIGH' | 'HIGH'
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytz

ET = pytz.timezone("US/Eastern")


# ── helpers ───────────────────────────────────────────────────────────────────

def _first_friday(year: int, month: int) -> date:
    """First Friday of the given year/month."""
    d = date(year, month, 1)
    delta = (4 - d.weekday()) % 7   # Monday=0, Friday=4
    return d + timedelta(days=delta)


def _first_biz_day(year: int, month: int) -> date:
    """First Monday–Friday of the given year/month."""
    d = date(year, month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _loc(d: date, time_str: str) -> pd.Timestamp:
    """Localize a date + 'HH:MM' string to US/Eastern, DST-safe."""
    h, m = int(time_str[:2]), int(time_str[3:5])
    naive = pd.Timestamp(d.year, d.month, d.day, h, m)
    try:
        return ET.localize(naive, is_dst=True)
    except Exception:
        return ET.localize(naive, is_dst=False)


# ── NFP – first Friday of each month, 08:30 ET ───────────────────────────────

def _nfp_events() -> list[tuple]:
    rows = []
    for year in range(2022, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 4:
                break
            rows.append((_loc(_first_friday(year, month), "08:30"), "NFP", "VERY_HIGH"))
    return rows


# ── FOMC – ~8/year, 14:00 ET (rate decision day) – hardcoded from FRB calendar ─

_FOMC_DATES = [
    # 2022
    date(2022,  1, 26), date(2022,  3, 16), date(2022,  5,  4),
    date(2022,  6, 15), date(2022,  7, 27), date(2022,  9, 21),
    date(2022, 11,  2), date(2022, 12, 14),
    # 2023
    date(2023,  2,  1), date(2023,  3, 22), date(2023,  5,  3),
    date(2023,  6, 14), date(2023,  7, 26), date(2023,  9, 20),
    date(2023, 11,  1), date(2023, 12, 13),
    # 2024
    date(2024,  1, 31), date(2024,  3, 20), date(2024,  5,  1),
    date(2024,  6, 12), date(2024,  7, 31), date(2024,  9, 18),
    date(2024, 11,  7), date(2024, 12, 18),
    # 2025
    date(2025,  1, 29), date(2025,  3, 19),
]


def _fomc_events() -> list[tuple]:
    return [(_loc(d, "14:00"), "FOMC", "VERY_HIGH") for d in _FOMC_DATES]


# ── CPI – ~12th–15th each month, 08:30 ET – hardcoded from BLS schedule ──────

_CPI_DATES = [
    # 2022
    date(2022,  1, 12), date(2022,  2, 10), date(2022,  3, 10),
    date(2022,  4, 12), date(2022,  5, 11), date(2022,  6, 10),
    date(2022,  7, 13), date(2022,  8, 10), date(2022,  9, 13),
    date(2022, 10, 13), date(2022, 11, 10), date(2022, 12, 13),
    # 2023
    date(2023,  1, 12), date(2023,  2, 14), date(2023,  3, 14),
    date(2023,  4, 12), date(2023,  5, 10), date(2023,  6, 13),
    date(2023,  7, 12), date(2023,  8, 10), date(2023,  9, 13),
    date(2023, 10, 12), date(2023, 11, 14), date(2023, 12, 12),
    # 2024
    date(2024,  1, 11), date(2024,  2, 13), date(2024,  3, 12),
    date(2024,  4, 10), date(2024,  5, 15), date(2024,  6, 12),
    date(2024,  7, 11), date(2024,  8, 14), date(2024,  9, 11),
    date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
    # 2025
    date(2025,  1, 15), date(2025,  2, 12), date(2025,  3, 12),
    date(2025,  4, 10),
]


def _cpi_events() -> list[tuple]:
    return [(_loc(d, "08:30"), "CPI", "VERY_HIGH") for d in _CPI_DATES]


# ── PPI – ~1 day after CPI, 08:30 ET ─────────────────────────────────────────

_PPI_DATES = [
    # 2022
    date(2022,  1, 13), date(2022,  2, 15), date(2022,  3, 15),
    date(2022,  4, 13), date(2022,  5, 12), date(2022,  6, 14),
    date(2022,  7, 14), date(2022,  8, 11), date(2022,  9, 14),
    date(2022, 10, 14), date(2022, 11, 15), date(2022, 12,  9),
    # 2023
    date(2023,  1, 18), date(2023,  2, 16), date(2023,  3, 15),
    date(2023,  4, 13), date(2023,  5, 11), date(2023,  6, 14),
    date(2023,  7, 13), date(2023,  8, 11), date(2023,  9, 14),
    date(2023, 10, 11), date(2023, 11, 15), date(2023, 12, 14),
    # 2024
    date(2024,  1, 12), date(2024,  2, 16), date(2024,  3, 14),
    date(2024,  4, 11), date(2024,  5, 14), date(2024,  6, 13),
    date(2024,  7, 12), date(2024,  8, 13), date(2024,  9, 12),
    date(2024, 10, 11), date(2024, 11, 14), date(2024, 12, 12),
    # 2025
    date(2025,  1, 16), date(2025,  2, 13), date(2025,  3, 13),
]


def _ppi_events() -> list[tuple]:
    return [(_loc(d, "08:30"), "PPI", "HIGH") for d in _PPI_DATES]


# ── Retail Sales – ~mid-month, 08:30 ET ──────────────────────────────────────

_RETAIL_DATES = [
    # 2022
    date(2022,  1, 14), date(2022,  2, 16), date(2022,  3, 16),
    date(2022,  4, 14), date(2022,  5, 17), date(2022,  6, 15),
    date(2022,  7, 15), date(2022,  8, 17), date(2022,  9, 15),
    date(2022, 10, 14), date(2022, 11, 16), date(2022, 12, 15),
    # 2023
    date(2023,  1, 18), date(2023,  2, 15), date(2023,  3, 15),
    date(2023,  4, 14), date(2023,  5, 16), date(2023,  6, 15),
    date(2023,  7, 18), date(2023,  8, 15), date(2023,  9, 14),
    date(2023, 10, 17), date(2023, 11, 15), date(2023, 12, 14),
    # 2024
    date(2024,  1, 17), date(2024,  2, 15), date(2024,  3, 14),
    date(2024,  4, 15), date(2024,  5, 15), date(2024,  6, 18),
    date(2024,  7, 16), date(2024,  8, 15), date(2024,  9, 17),
    date(2024, 10, 17), date(2024, 11, 15), date(2024, 12, 17),
    # 2025
    date(2025,  1, 16), date(2025,  2, 14), date(2025,  3, 17),
]


def _retail_events() -> list[tuple]:
    return [(_loc(d, "08:30"), "RETAIL_SALES", "HIGH") for d in _RETAIL_DATES]


# ── GDP Advance – quarterly ~4 weeks after quarter end, 08:30 ET ─────────────

_GDP_ADV_DATES = [
    date(2022,  1, 27),  # Q4 2021
    date(2022,  4, 28),  # Q1 2022
    date(2022,  7, 28),  # Q2 2022
    date(2022, 10, 27),  # Q3 2022
    date(2023,  1, 26),  # Q4 2022
    date(2023,  4, 27),  # Q1 2023
    date(2023,  7, 27),  # Q2 2023
    date(2023, 10, 26),  # Q3 2023
    date(2024,  1, 25),  # Q4 2023
    date(2024,  4, 25),  # Q1 2024
    date(2024,  7, 25),  # Q2 2024
    date(2024, 10, 30),  # Q3 2024
    date(2025,  1, 30),  # Q4 2024
]


def _gdp_events() -> list[tuple]:
    return [(_loc(d, "08:30"), "GDP_ADV", "VERY_HIGH") for d in _GDP_ADV_DATES]


# ── PCE – last Friday of following month, 08:30 ET ───────────────────────────

_PCE_DATES = [
    # 2022
    date(2022,  1, 28), date(2022,  2, 25), date(2022,  3, 25),
    date(2022,  4, 29), date(2022,  5, 27), date(2022,  6, 30),
    date(2022,  7, 29), date(2022,  8, 26), date(2022,  9, 30),
    date(2022, 10, 28), date(2022, 11, 23), date(2022, 12, 23),
    # 2023
    date(2023,  1, 27), date(2023,  2, 24), date(2023,  3, 31),
    date(2023,  4, 28), date(2023,  5, 26), date(2023,  6, 30),
    date(2023,  7, 28), date(2023,  8, 31), date(2023,  9, 29),
    date(2023, 10, 27), date(2023, 11, 22), date(2023, 12, 22),
    # 2024
    date(2024,  1, 26), date(2024,  2, 29), date(2024,  3, 29),
    date(2024,  4, 26), date(2024,  5, 31), date(2024,  6, 28),
    date(2024,  7, 26), date(2024,  8, 30), date(2024,  9, 27),
    date(2024, 10, 31), date(2024, 11, 27), date(2024, 12, 20),
    # 2025
    date(2025,  1, 31), date(2025,  2, 28), date(2025,  3, 28),
]


def _pce_events() -> list[tuple]:
    return [(_loc(d, "08:30"), "PCE", "HIGH") for d in _PCE_DATES]


# ── ISM Manufacturing PMI – first business day of month, 10:00 ET ─────────────

def _ism_events() -> list[tuple]:
    rows = []
    for year in range(2022, 2026):
        for month in range(1, 13):
            if year == 2025 and month > 4:
                break
            rows.append((_loc(_first_biz_day(year, month), "10:00"), "ISM_PMI", "HIGH"))
    return rows


# ── Builder ───────────────────────────────────────────────────────────────────

def build_calendar() -> pd.DataFrame:
    rows = (
        _nfp_events()
        + _fomc_events()
        + _cpi_events()
        + _ppi_events()
        + _retail_events()
        + _gdp_events()
        + _pce_events()
        + _ism_events()
    )
    df = pd.DataFrame(rows, columns=["datetime", "event_type", "impact"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_calendar()
    out = os.path.join(os.path.dirname(__file__), "news_calendar.parquet")
    df.to_parquet(out)
    print(f"Saved {len(df)} events to {out}")
    print("\nEvents per type:")
    print(df.groupby("event_type").size().sort_values(ascending=False).to_string())
    print(f"\nDate range: {df['datetime'].min()} → {df['datetime'].max()}")
    print("\nFirst 10:")
    print(df.head(10).to_string())
