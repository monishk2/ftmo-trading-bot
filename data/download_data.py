"""
Dukascopy CSV → Parquet converter (npx dukascopy-node format).

Expected input format (comma-delimited, Unix millisecond timestamps):
    timestamp,open,high,low,close,volume
    1577836800000,1.12076,1.12076,1.12076,1.12076,0

Stored output: UTC-aware datetimes in Parquet (snappy compressed).

Usage — single file:
    python data/download_data.py \\
        --input ~/Downloads/combined_eurusd.csv \\
        --output data/historical/EURUSD_15m.parquet

Usage — combine multiple files into one output:
    python data/download_data.py \\
        --combine ~/Downloads/eurusd_2023.csv ~/Downloads/eurusd_2024.csv \\
        --output data/historical/EURUSD_15m.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_dukascopy_csv(path: Path) -> pd.DataFrame:
    """Parse a dukascopy-node OHLCV CSV into a clean UTC-aware DataFrame.

    Handles:
    - Comma-delimited (from npx dukascopy-node)
    - timestamp column as Unix milliseconds
    - Semicolon-delimited legacy Dukascopy exports with string timestamps

    Returned columns: datetime (UTC tz-aware), open, high, low, close, volume
    """
    # Sniff delimiter: check first non-empty line
    with open(path) as fh:
        header = fh.readline().strip()
    sep = "," if "," in header else ";"

    raw = pd.read_csv(
        path,
        sep=sep,
        skipinitialspace=True,
        thousands=",",
        na_values=["", "NA", "N/A"],
    )
    raw.columns = [c.strip().lower() for c in raw.columns]

    # ------------------------------------------------------------------
    # Parse timestamp column
    # ------------------------------------------------------------------
    ts_col = None
    for candidate in ["timestamp", "gmt time", "date & time (utc)", "datetime", "date", "time"]:
        if candidate in raw.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = raw.columns[0]

    if ts_col == "timestamp" and pd.api.types.is_numeric_dtype(raw[ts_col]):
        # Unix milliseconds → UTC datetime
        datetimes = pd.to_datetime(raw[ts_col], unit="ms", utc=True)
    else:
        # String timestamp — try known formats then fall back to inference
        _TS_FORMATS = [
            "%d.%m.%Y %H:%M:%S.%f",
            "%d.%m.%Y %H:%M:%S",
            "%Y.%m.%d %H:%M:%S.%f",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        datetimes = None
        series = raw[ts_col].astype(str).str.strip().str.strip('"')
        for fmt in _TS_FORMATS:
            try:
                datetimes = pd.to_datetime(series, format=fmt, utc=True)
                break
            except (ValueError, TypeError):
                continue
        if datetimes is None:
            datetimes = pd.to_datetime(series, utc=True)

    df = pd.DataFrame({"datetime": datetimes})

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in raw.columns:
            raise ValueError(
                f"Expected column '{col}' not found. Available: {list(raw.columns)}"
            )
        df[col] = pd.to_numeric(raw[col], errors="coerce")

    before = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing OHLCV values.")

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Basic sanity checks — raises AssertionError on failure."""
    assert (df["high"] >= df["low"]).all(), "Found rows where high < low"
    assert (df["high"] >= df["open"]).all(), "Found rows where high < open"
    assert (df["high"] >= df["close"]).all(), "Found rows where high < close"
    assert (df["low"] <= df["open"]).all(), "Found rows where low > open"
    assert (df["low"] <= df["close"]).all(), "Found rows where low > close"
    assert df["datetime"].is_monotonic_increasing, "Timestamps are not sorted ascending"


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"  Saved {len(df):,} rows → {output_path}  ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dukascopy-node OHLCV CSV(s) to a clean Parquet file."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i", type=Path,
        help="Single CSV input file",
    )
    group.add_argument(
        "--combine", nargs="+", type=Path, metavar="CSV",
        help="Two or more CSV files to concatenate before saving",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Destination parquet path (e.g. data/historical/EURUSD_15m.parquet)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip OHLCV sanity checks (not recommended)",
    )
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()

    if args.input:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading: {input_path}")
        df = load_dukascopy_csv(input_path)
        print(f"  Parsed {len(df):,} rows  |  {df['datetime'].min()} → {df['datetime'].max()}")

    else:  # --combine
        frames = []
        for p in args.combine:
            p = p.expanduser().resolve()
            if not p.exists():
                print(f"ERROR: File not found: {p}", file=sys.stderr)
                sys.exit(1)
            print(f"Loading: {p}")
            chunk = load_dukascopy_csv(p)
            print(f"  Parsed {len(chunk):,} rows  |  {chunk['datetime'].min()} → {chunk['datetime'].max()}")
            frames.append(chunk)

        df = pd.concat(frames, ignore_index=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Combined: {len(df):,} rows  |  {df['datetime'].min()} → {df['datetime'].max()}")

    if not args.skip_validation:
        print("Validating OHLCV integrity...")
        validate_ohlcv(df)
        print("  All checks passed.")

    print(f"Saving: {output_path}")
    save_parquet(df, output_path)


if __name__ == "__main__":
    main()
