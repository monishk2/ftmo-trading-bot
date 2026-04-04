# FTMO Prop Firm Trading Bot

## Project Goal
Build a complete Python trading system that backtests and live-executes two structural forex strategies on FTMO prop firm accounts via cTrader. The system must enforce FTMO risk rules as hard constraints that cannot be overridden.

## Owner Context
- Monish, 22, CS/ML degree, strong Python skills
- Goal: Pass FTMO challenge and scale to $25K+/month across multiple prop firms
- Mac user — NO MetaTrader5 Python package (Windows only). Use cTrader Open API for live execution.
- Backtesting uses CSV data from Dukascopy — no broker connection needed for backtesting.

## Tech Stack
- Python 3.11+
- pandas, numpy for data processing
- plotly for visualization
- ta (technical analysis library) for indicators
- ctrader-open-api for live execution (Mac compatible)
- schedule for automation
- requests for API calls

## Architecture
ftmo-trading-bot/
├── CLAUDE.md
├── config/
│   ├── strategy_params.json
│   ├── ftmo_rules.json
│   └── instruments.json
├── data/
│   ├── historical/          # CSV files from Dukascopy (gitignored)
│   └── download_data.py
├── strategies/
│   ├── base_strategy.py
│   ├── london_open_breakout.py
│   ├── fvg_retracement.py
│   └── regime_filter.py
├── backtesting/
│   ├── backtester.py
│   ├── metrics.py
│   ├── report_generator.py
│   └── walk_forward.py
├── execution/
│   ├── ctrader_connector.py
│   ├── order_manager.py
│   ├── position_sizer.py
│   └── ftmo_guardian.py
├── monitoring/
│   ├── dashboard.py
│   ├── alerts.py
│   └── trade_journal.py
├── tests/
│   ├── test_strategies.py
│   ├── test_ftmo_guardian.py
│   ├── test_position_sizer.py
│   └── test_backtester.py
├── main.py
└── requirements.txt

## Critical Design Rules
1. FTMO Guardian has ABSOLUTE AUTHORITY over all trades. No module can bypass it.
2. Every parameter must be in JSON config files. No magic numbers in code.
3. Backtesting must simulate FTMO rules (daily loss limit, total drawdown).
4. Always add realistic spread + slippage in backtests (spread from instruments.json + random 0-0.5 pip slippage).
5. Log everything — every trade, every guardian check, every error.
6. Fail safe: if anything unexpected happens, close all positions and stop trading.
7. Use cTrader Open API for live execution, NOT MetaTrader5.

## Strategy Context (WHY these strategies have structural edge)

### Strategy 1: London Open Breakout
During Asian session (00:00-03:00 EST), EUR/USD trades in a tight range because institutional volume is low. At London open (03:00 EST), European banks start executing client orders, creating a predictable volatility spike. This is STRUCTURAL — institutional order flow at session opens is mechanical (banks MUST execute client orders at open), not a statistical pattern that gets arbitraged away.

### Strategy 2: Fair Value Gap (FVG) Retracement
When large institutional orders move price quickly, they create Fair Value Gaps — 3-candle patterns where candle 1's wick doesn't overlap candle 3's wick. Price revisits these gaps because other institutions want to fill at those levels. This is STRUCTURAL — large orders mechanically can't fill in one go, so gaps are a permanent market feature.

### Regime Filter
ATR percentile + ADX determine if market is trending (favor breakout), ranging (favor FVG retracement), or dead (sit out). This prevents the "strategy stopped working" problem by adapting to conditions.

## FTMO Rules (HARD LIMITS — Guardian must enforce these)
- Max daily loss: 5% of starting daily balance (Guardian triggers at 4% as safety buffer)
- Max total drawdown: 10% of initial balance (Guardian triggers at 9%)
- Min trading days: 4 (for challenge)
- Profit target: 10% (challenge), 5% (verification)
- No time limit on challenge
- EAs/bots are explicitly allowed by FTMO
- cTrader is supported by FTMO

## Commands
- `python main.py --mode backtest --strategy all` — Run full backtest
- `python main.py --mode backtest --strategy london_breakout` — Single strategy
- `python main.py --mode walkforward` — Walk-forward optimization
- `python main.py --mode live` — Start live trading via cTrader
- `python -m pytest tests/` — Run all tests
