"""
Dukascopy CSV → Parquet converter.

Dukascopy exports tick/OHLCV data with millisecond UTC timestamps in one of two formats:
  1. Gmt time;Open;High;Low;Close;Volume
     "01.01.2023 00:00:00.000";1.07012;1.07034;1.06998;1.07021;123.45
  2. Date & Time (UTC);Open;High;Low;Close;Volume   (no milliseconds)

Usage:
    python data/download_data.py --input ~/Downloads/EURUSD_15m.csv \
                                 --output data/historical/EURUSD_15m.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Dukascopy timestamp formats to try in order
# ---------------------------------------------------------------------------
_TS_FORMATS = [
    "%d.%m.%Y %H:%M:%S.%f",   # 01.01.2023 00:00:00.000  (milliseconds)
    "%d.%m.%Y %H:%M:%S",      # 01.01.2023 00:00:00
    "%Y.%m.%d %H:%M:%S.%f",   # 2023.01.01 00:00:00.000
    "%Y.%m.%d %H:%M:%S",      # 2023.01.01 00:00:00
    "%Y-%m-%d %H:%M:%S",      # 2023-01-01 00:00:00
]


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Try each known Dukascopy format and return a UTC-aware datetime series."""
    for fmt in _TS_FORMATS:
        try:
            parsed = pd.to_datetime(series.str.strip().str.strip('"'), format=fmt, utc=True)
            return parsed
        except (ValueError, TypeError):
            continue
    # Last resort: let pandas infer
    return pd.to_datetime(series.str.strip().str.strip('"'), infer_datetime_format=True, utc=True)


def load_dukascopy_csv(path: Path) -> pd.DataFrame:
    """
    Parse a Dukascopy OHLCV CSV into a clean DataFrame.

    Returned columns: datetime (UTC, tz-aware), open, high, low, close, volume
    """
    # Dukascopy uses semicolons as delimiters
    raw = pd.read_csv(
        path,
        sep=";",
        skipinitialspace=True,
        thousands=",",
        na_values=["", "NA", "N/A"],
    )

    # Normalise column names: strip whitespace, lowercase
    raw.columns = [c.strip().lower() for c in raw.columns]

    # Identify the timestamp column — Dukascopy labels it "gmt time" or similar
    ts_col = None
    for candidate in ["gmt time", "date & time (utc)", "datetime", "date", "time", "timestamp"]:
        if candidate in raw.columns:
            ts_col = candidate
            break
    if ts_col is None:
        # Fall back to first column
        ts_col = raw.columns[0]

    df = pd.DataFrame()
    df["datetime"] = _parse_timestamp(raw[ts_col])

    # Map OHLCV columns — Dukascopy uses "open", "high", "low", "close", "volume"
    col_map = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    for target, source in col_map.items():
        if source in raw.columns:
            df[target] = pd.to_numeric(raw[source], errors="coerce")
        else:
            raise ValueError(
                f"Expected column '{source}' not found. Available columns: {list(raw.columns)}"
            )

    # Drop rows with any NaN in OHLCV
    before = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing OHLCV values.")

    # Sort by time ascending
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Basic sanity checks — raises ValueError on failure."""
    assert (df["high"] >= df["low"]).all(), "Found rows where high < low"
    assert (df["high"] >= df["open"]).all(), "Found rows where high < open"
    assert (df["high"] >= df["close"]).all(), "Found rows where high < close"
    assert (df["low"] <= df["open"]).all(), "Found rows where low > open"
    assert (df["low"] <= df["close"]).all(), "Found rows where low > close"
    assert df["datetime"].is_monotonic_increasing, "Timestamps are not sorted ascending"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Dukascopy OHLCV CSV to a clean Parquet file."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to the Dukascopy CSV file (e.g. ~/Downloads/EURUSD_15m.csv)"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Destination parquet path (e.g. data/historical/EURUSD_15m.parquet)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip OHLCV sanity checks (not recommended)"
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    df = load_dukascopy_csv(input_path)
    print(f"  Parsed {len(df):,} rows  |  {df['datetime'].min()} → {df['datetime'].max()}")

    if not args.skip_validation:
        print("Validating OHLCV integrity...")
        validate_ohlcv(df)
        print("  All checks passed.")

    print(f"Saving: {output_path}")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  Done. File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
