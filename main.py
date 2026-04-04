"""
FTMO Trading Bot — entry point.

Usage:
    python main.py --mode backtest --strategy all
    python main.py --mode backtest --strategy london_breakout
    python main.py --mode walkforward
    python main.py --mode live
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — file + console
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "main.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTMO Prop Firm Trading Bot")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["backtest", "walkforward", "live"],
        help="Execution mode",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "london_breakout", "fvg_retracement"],
        help="Strategy to run (backtest mode only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting FTMO Trading Bot | mode=%s strategy=%s", args.mode, args.strategy)

    if args.mode == "backtest":
        logger.info("Backtest mode — not yet implemented. Build backtesting/ modules first.")
        # TODO: from backtesting.backtester import Backtester
    elif args.mode == "walkforward":
        logger.info("Walk-forward mode — not yet implemented.")
        # TODO: from backtesting.walk_forward import WalkForwardOptimizer
    elif args.mode == "live":
        logger.info("Live mode — not yet implemented. Build execution/ modules first.")
        # TODO: from execution.ctrader_connector import CTraderConnector
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
