# Deployment Guide — FTMO Trading Bot

## Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Platform Decision Matrix](#2-platform-decision-matrix)
3. [The5ers — Gold Multi-Strategy (cTrader)](#3-the5ers--gold-multi-strategy-ctrader)
4. [FTMO — Gold Multi-Strategy (cTrader)](#4-ftmo--gold-multi-strategy-ctrader)
5. [Apex Trader Funding — NAS100 IB Breakout (Tradovate)](#5-apex-trader-funding--nas100-ib-breakout-tradovate)
6. [Paper Trade Verification](#6-paper-trade-verification)
7. [Going Live Checklist](#7-going-live-checklist)
8. [Monitoring & Kill Procedures](#8-monitoring--kill-procedures)
9. [Funded Account Scaling Plan](#9-funded-account-scaling-plan)

---

## 1. Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         config/live_config.json         │
                    └──────────────┬──────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
  The5ers / FTMO            Apex / Topstep              Paper Trade
  (cTrader cBot)            (Tradovate REST)            Verification
         │                         │                         │
  ctrader_bots/             execution/                execution/
  gold_multi_strategy.py    tradovate_connector.py    paper_trade_verifier.py
  nas100_ib_breakout.py
```

**Two independent execution paths:**
- **cTrader Automate** — cloud-hosted Python cBots, always on, no VPS needed
- **Tradovate REST + WebSocket** — NQ futures for Apex/Topstep, requires always-on process

---

## 2. Platform Decision Matrix

### 2.1 Which The5ers Product?

> **Do NOT use The5ers Bootcamp (3-step) if your goal is ≤ 3 weeks.**
>
> The Bootcamp requires passing 3 *sequential* phases. If each phase has a 62% pass rate,
> you need 0.62³ = **24% cumulative** probability — worse than gold-only, and each phase
> takes ~18 days, so minimum total time is **54+ days**.
>
> Use **The5ers Hyper Growth** (1-phase evaluation). All numbers below are for Hyper Growth.

### 2.2 Platform Comparison — Getting Funded in ≤ 3 Weeks

| Rank | Platform | Program | Strategy | Account | Pass Rate | Eval Cost | Expected Cost | Avg Days |
|---|---|---|---|---|---|---|---|---|
| **1** | **The5ers** | **Hyper Growth $25K** | **Gold + NAS100 combined** | **$25K** | **61.6%** | **$95** | **$154** | **~18** |
| 2 | Apex | $50K batch (×5) | NAS100 IB only | $50K | 57.2% (batch) | $100 | $100 | ~13 |
| 3 | The5ers | Hyper Growth $25K | Gold only | $25K | 52% | $95 | $183 | ~18 |
| 4 | Apex | $100K batch (×3) | NAS100 IB only | $100K | 39.4% (batch) | $105 | $267 | ~13 |
| 5 | FTMO | Standard $10K | Gold only | $10K | 57% | $550 | $965 | ~15 |

**Recommendation for <3 weeks at lowest cost:**
- **First choice**: The5ers Hyper Growth $25K with **both cBots running simultaneously** (Gold + NAS100)
  → 61.6% pass for $95/eval ($154 expected cost). The combined strategy is the whole reason we built both cBots.
- **Second choice**: 5× Apex $50K at $100 total if you want futures (NQ) instead of CFDs.
  Slightly lower pass rate (57%) but faster (~13 days) and funded accounts scale well.
- **Both together**: Buy 1× The5ers Hyper Growth ($95) AND 5× Apex $50K ($100) simultaneously for $195 total.
  P(at least one funded account somewhere) ≈ 1 − (0.384 × 0.428) = **84%**.

---

## 3. The5ers Hyper Growth — Combined Gold + NAS100 (cTrader)

> **IMPORTANT: Run BOTH cBots on the same The5ers account simultaneously.**
> Gold trades XAUUSD H1; NAS100 trades US100 M5. They are uncorrelated and use
> separate position slots. The combined 61.6% pass rate (vs 52% gold-only) comes
> from running both concurrently at Gold 0.5% risk + NAS100 1.0% risk.

### 3.1 Account Setup
1. Sign up at the5ers.com → **Hyper Growth $25K** plan ($95)
   - Rules: 6% profit target ($1,500), 5% static DD ($1,250 max loss ever from start)
   - No time limit, no minimum trading days
2. Download **cTrader** from The5ers dashboard (their branded version)
3. Log in with The5ers credentials — balance will show $25,000

### 3.2 Upload Both cBots

**Bot 1 — Gold Multi-Strategy:**
1. Open cTrader → **Automate** → **My Robots** → **New** → **Python cBot**
2. Name: `GoldMultiStrategy`
3. Paste contents of `ctrader_bots/gold_multi_strategy.py`
4. Click **Build** → **Run in Cloud**
5. Attach to **XAUUSD H1** chart

**Bot 2 — NAS100 IB Breakout:**
1. **My Robots** → **New** → **Python cBot**
2. Name: `NAS100IbBreakout`
3. Paste contents of `ctrader_bots/nas100_ib_breakout.py`
4. Click **Build** → **Run in Cloud**
5. Attach to **US100 M5** chart (verify The5ers symbol name — may be "NAS100" or "US100")

Both bots run independently. They share the same account balance for DD tracking
but each manages its own position. cTrader cloud keeps both alive 24/7.

### 3.3 Configure Gold cBot Parameters

| Parameter | The5ers Value | Notes |
|---|---|---|
| SymbolName | XAUUSD | Confirm symbol with broker |
| RiskPerTradePct | **0.50** | $125/trade at $25K |
| MaxTradesPerDay | 3 | A+B+C combined cap |
| DailyKillPct | **3.5** | Stop at $875 intraday loss (leaves $375 buffer before 5% breach) |
| RrTp1 | 1.0 | TP1 at 1R |
| RrTp2 | 2.5 | Final TP at 2.5R |
| Tp1Pct | 0.50 | Close 50% at TP1 |
| TrailAtrMult | 2.0 | V4 trail |
| ABufPips | 50.0 | |
| AMinRangePips | 500.0 | |
| AMaxRangePips | 2000.0 | |
| ASlCapPips | 1000.0 | |
| ARegimeAtrPct | 40.0 | |
| BSlBufPips | 30.0 | |
| BSlCapPips | 500.0 | |
| BSlMinPips | 20.0 | |
| CSlPips | 200.0 | |

### 3.4 Configure NAS100 cBot Parameters

| Parameter | The5ers Value | Notes |
|---|---|---|
| SymbolName | US100 | Check broker — may be "NAS100" |
| RiskPerTradePct | **1.00** | $250/trade at $25K |
| RrRatio | 2.5 | |
| DailyKillPct | **3.5** | Combined kill: each bot monitors own DD |
| IbAdrRatio | 99.0 | No IB filter (raw NY open breakout) |
| BufferPoints | 5.0 | |
| MaxSlPoints | 80.0 | |
| MinSlPoints | 15.0 | |
| VolSmaPeriod | 20 | |

> **Combined daily kill logic note:** Each cBot independently checks its own account
> equity vs starting balance. If gold loses 2% AND NAS100 loses 1.5% on the same day,
> both bots will check against the same account equity — the combined 3.5% kill is
> redundant protection. Consider setting both `DailyKillPct=3.5` to ensure neither
> bot opens new trades once total daily DD approaches 4%.

### 3.5 The5ers Hyper Growth Rules (enforced by the bots)

| Rule | Limit | Bot Response |
|---|---|---|
| **Static DD floor** | Account equity can NEVER go below $23,750 ($25K × 95%) | Kill switch closes all positions at $875 daily loss (3.5%) |
| **Profit target** | +$1,500 (6% of $25K) | Manually stop both bots and request payout |
| **No time limit** | — | Bots run until target or DD breach |
| **No minimum days** | — | Can pass in 1 day if lucky |

> **When you hit $1,500 profit:** Stop both cBots immediately, do NOT keep trading.
> The5ers pays out from first profit — no waiting period on Hyper Growth.

### 3.6 cTrader Python Dependencies
No pip installs needed — `ctrader_automate` is injected by the cloud runtime.

### 3.7 Verification Before Going Live
Run paper trade verifier for at least 3 days:
```bash
venv/bin/python execution/paper_trade_verifier.py \
  --strategy gold --start $(date -v-7d +%Y-%m-%d) --end $(date +%Y-%m-%d)

venv/bin/python execution/paper_trade_verifier.py \
  --strategy nas100 --start $(date -v-7d +%Y-%m-%d) --end $(date +%Y-%m-%d)
```

---

## 4. FTMO — Gold Multi-Strategy (cTrader)

### 4.1 Account Setup
1. Sign up at ftmo.com → **Standard Challenge $10K** ($155)
2. Download **cTrader** (FTMO partner broker: Spotware / Purple Trading)
3. Credentials arrive by email within 1 business day

### 4.2 Upload the cBot
Same steps as §3.2 — the same `ctrader_bots/gold_multi_strategy.py` file works.

### 4.3 Parameter Differences vs The5ers

| Parameter | The5ers | FTMO |
|---|---|---|
| RiskPerTradePct | 0.50 | **1.50** |
| DailyKillPct | 4.0 | **4.0** |
| TotalDdKillPct | — | **9.0** (add a guardian check in code) |

### 4.4 FTMO Daily Loss Reference
FTMO uses **midnight CET balance** as the daily reference — NOT current balance.
The cBot approximates this with `_daily_start_balance = Account.Balance` at first bar
of the day. This is close but not exact. For production:

```
FTMO rule: Daily loss = Balance(00:00 CET) - min(Equity during day)
Guardian triggers at: 4% below 00:00 CET balance
```

Implement `utils/timezone.get_midnight_cet()` to record the exact reference daily.

### 4.5 FTMO Restrictions
- Min 4 trading days (the bot will trade Mon–Thu every week; this is satisfied)
- EAs explicitly allowed
- Scaling plan starts at $10K → $20K → $40K → $80K → $160K → $200K

---

## 5. Apex Trader Funding — NAS100 IB Breakout (Tradovate)

### 5.1 Account Setup
1. Sign up at apextraderfunding.com
2. Purchase **5× $50K Monthly Combine** simultaneously ($20 × 5 = $100)
   - Best value: 57.2% probability at least one funded within 13 trading days
   - Rules: $3K profit target | $2.5K max DD | $1K daily loss limit | 21 days
3. Link each account to Tradovate (Apex's partnered broker)

### 5.2 Tradovate App Registration
1. Go to trader.tradovate.com → Settings → API
2. Create a new application:
   - App Name: `NAS100IbBot`
   - Redirect URI: `http://localhost:8080`
3. Note the **App ID (CID)** and **Secret**

### 5.3 Environment Variables
```bash
# Add to ~/.zshrc or a .env file (NEVER commit to git)
export TRADOVATE_USER="your_username"
export TRADOVATE_PASS="your_password"
export TRADOVATE_APP_ID="your_app_name"
export TRADOVATE_APP_VERSION="1.0"
export TRADOVATE_CID="your_cid"
export TRADOVATE_SECRET="your_secret"
export TRADOVATE_ENV="demo"   # switch to "live" when ready
```

### 5.4 Install Dependencies
```bash
cd /Users/ashok/ftmo-trading-bot
venv/bin/pip install websocket-client requests
```

### 5.5 Test in Demo Mode
```bash
# First: set TRADOVATE_ENV=demo
source ~/.zshrc
venv/bin/python execution/tradovate_connector.py config/live_config.json apex
```

Expected output:
```
2026-01-15 09:15:00  INFO  TradovateConnector  Authenticated. account_id=12345
2026-01-15 09:15:01  INFO  NAS100IbLive  Running. Symbol=NQM5  Risk=2.0%  RR=2.5
2026-01-15 09:15:01  INFO  BarBuilder  WebSocket thread started for NQM5
```

### 5.6 Configure for Live
1. Set `TRADOVATE_ENV=live` in environment
2. Update `config/live_config.json` → `"apex"` → `"env": "live"`
3. For the first day: run with `"contracts": 1` (1 NQ contract = $20/point)
   - 1 NQ contract, 40-point SL = $800 risk per trade
   - At $50K, 1 contract ≈ 1.6% risk (slightly above 2% target — use MNQ for precise sizing)
4. For precise risk sizing: set `"use_mnq": true` (MNQ = $2/point, 10 contracts = 1 NQ)

### 5.7 Apex Rules Implementation
| Rule | Limit | Bot Behaviour |
|---|---|---|
| Daily Loss Limit | $1,000 | `daily_kill_pct=2.0%` of $50K stops at $1,000 |
| Max Drawdown | $2,500 (trailing EOD) | Monitor via heartbeat; close if equity near limit |
| Profit Target | $3,000 | Monitor; manually stop trading when reached |
| Time Limit | 21 trading days | Track day count; log warning after day 18 |
| No DLL failures | DLL pauses day | Bot already stops on DLL trigger |

### 5.8 Multiple Accounts
To run all 5 Apex accounts simultaneously, use `screen` or `tmux`:
```bash
# Terminal multiplexer (tmux)
tmux new-session -s apex1 -d "venv/bin/python execution/tradovate_connector.py config/apex_account1.json"
tmux new-session -s apex2 -d "venv/bin/python execution/tradovate_connector.py config/apex_account2.json"
# ... repeat for accounts 3-5
```

Create separate `config/apex_account{1..5}.json` files with different credentials.

### 5.9 NQ Contract Month Reference
```
H = March   (expires 3rd Friday of March)
M = June
U = September
Z = December
```
Current front-month (April 2026): `NQM6` (June 2026)
The connector auto-resolves this via `/contract/suggest`.

---

## 6. Paper Trade Verification

Before committing real capital, run 1–2 weeks in paper mode and verify signals match backtest.

### 6.1 Run the Verifier
```bash
venv/bin/python execution/paper_trade_verifier.py \
  --strategy gold \
  --start 2026-01-01 \
  --end 2026-01-31 \
  --config config/live_config.json
```

### 6.2 What It Checks
The verifier replays historical H1 bars through the same strategy logic and compares:
- Signal direction (long/short): must be 100% match
- Entry price: must be within 0.5% (spread/slippage tolerance)
- SL price: must be within 1%
- Number of signals per day: must match exactly

### 6.3 Pass Criteria
- Direction match: **100%**
- Price tolerance: **< 5% deviation on average**
- Signal count: **same number of signals ± 0 (exact)**

Any mismatch likely indicates:
- Timezone offset error (check cTrader's ET timezone vs your data timezone)
- VWAP anchor time mismatch (strategy uses 9:00 ET H1 bar; cBot should match)
- Asian range date boundary off-by-one (check hour 19/23 attribution)

---

## 7. Going Live Checklist

### Before First Live Trade
- [ ] Paper traded for minimum 5 trading days without discrepancies
- [ ] `paper_trade_verifier.py` passes all checks
- [ ] Account credentials tested in demo mode
- [ ] Kill switch confirmed working (tested with forced equity drawdown in demo)
- [ ] `config/live_config.json` credentials filled (or env vars set)
- [ ] Log directory exists: `mkdir -p logs/paper_trade logs/live`
- [ ] No open positions on account before starting bot
- [ ] Starting balance recorded (for DD monitoring reference)

### cTrader (Gold / The5ers / FTMO)
- [ ] cBot uploaded and compiled without errors
- [ ] Parameters verified against table in §3.3
- [ ] "Run in Cloud" enabled (keeps running even when cTrader desktop is closed)
- [ ] Test with single small trade (manually reduce RiskPerTradePct to 0.1% for first trade)

### Tradovate / Apex (NAS100)
- [ ] `TRADOVATE_ENV=live` set
- [ ] Authentication test passes: `python execution/tradovate_connector.py --auth-test`
- [ ] Front-month contract correctly resolved in logs
- [ ] Bracket orders tested in paper mode (verify SL + TP both attach)
- [ ] Account $50K balance confirmed in Tradovate
- [ ] 5 eval accounts started on same day (same Tuesday morning to sync timing)

---

## 8. Monitoring & Kill Procedures

### Daily Monitoring Routine (5 min/day)
```bash
# Check gold cBot status (cTrader Automate log)
# → Automate → My Robots → GoldMultiStrategy → Logs

# Check NAS100 connector
tail -f logs/live/tradovate_YYYY-MM-DD.log

# Check account equity
venv/bin/python -c "
import json
from execution.tradovate_connector import TradovateConnector
cfg = json.load(open('config/live_config.json'))['apex']
conn = TradovateConnector(cfg)
conn.authenticate()
print(conn.get_account_info())
"
```

### Emergency Kill Procedures

**Kill all NAS100 positions immediately:**
```bash
venv/bin/python -c "
import json
from execution.tradovate_connector import TradovateConnector
cfg = json.load(open('config/live_config.json'))['apex']
conn = TradovateConnector(cfg)
conn.authenticate()
conn.close_all_positions()
print('All positions closed.')
"
```

**Kill gold cBot (cTrader):**
1. Open cTrader Automate
2. Select GoldMultiStrategy → **Stop**
3. Verify no open XAUUSD positions in Positions tab
4. If positions remain: manually close them

### Automatic Kill Switch Triggers
| Event | Bot Response |
|---|---|
| Intraday equity DD ≥ 4% (gold) | Close all XAUUSD; stop new entries for the day |
| Intraday equity DD ≥ 2% (NAS100 Apex) | Close NQ position; stop new entries |
| Time stop (22:00 ET gold / 15:55 ET NAS100) | Close position; wait for next day |
| Bot crash / disconnect | cTrader cloud re-starts automatically; Tradovate: positions stay open with broker-side SL/TP |

---

## 9. Funded Account Scaling Plan

### Phase 1 — Apex Eval (Weeks 1–2)
- Run 5× $50K evals simultaneously at 2% risk
- Expected: 1 funded account within 13 trading days
- Cost: ~$100 per batch

### Phase 2 — Apex Funded (Months 1–3)
- Switch to 0.5% risk
- Wait 2+ months before first payout request (consistency rule)
- Expected income: $762/month per $100K funded account
- After scaling to 3× $100K: ~$2,286/month

### Phase 3 — The5ers Parallel (Month 2+)
- Once Apex funded and stable, start The5ers eval ($95)
- Gold Multi-Strategy at 0.5% risk
- Expected: +$581/month per $25K account
- Combined (NAS100 + Gold): **$1,343/month per $100K equivalent**

### Phase 4 — FTMO (Optional, Month 6+)
- Only if The5ers/Apex are consistently profitable
- FTMO is expensive ($550/eval) and has stricter intraday DD rules
- Use 1.5% risk; expected 57% pass rate; ~14.5 trading days to pass

### Income Projections (Funded)
| Configuration | Monthly Income |
|---|---|
| 1× Apex $100K (NAS100) | $762 |
| 2× The5ers $25K (Gold) | $1,162 |
| 1× Apex $100K + 2× The5ers | **$1,924** |
| 3× Apex $100K + 4× The5ers | **$4,610** |

---

## Appendix — Troubleshooting

### "ModuleNotFoundError: ctrader_automate"
The `ctrader_automate` module is only available inside cTrader's Python runtime.
Run the cBot by uploading it to cTrader Automate — NOT from the terminal.
The `if not _IN_CTRADER:` block at the bottom provides local stubs for linting.

### "Tradovate auth failed"
1. Verify credentials: `echo $TRADOVATE_USER`
2. Check if App ID requires approval (new apps need Tradovate review — takes 1-2 days)
3. Ensure you're using the correct environment: `demo` for paper, `live` for real

### "Symbol not found: NQM5"
The contract month letter changes quarterly. Update `config/live_config.json`:
```json
"symbol": "NQU6"   // September 2026 contract
```
Or set `"symbol": "NQ"` to auto-resolve via the connector.

### "SL too tight — no order placed"
NAS100 SL minimum is 15 points ($300/contract). If the IB is very wide or price
has moved far from IB boundary, sl_dist may be capped at min_sl_points. This is
expected behaviour — the bot skips marginal setups.

### "cBot shows 0 trades in live mode but backtest had trades"
Check the H1 bar timezone: cTrader may serve bars in UTC or local time depending
on broker. Verify that `bar_time.hour == 3` corresponds to 03:00 US/Eastern
(= 08:00 UTC). If not, adjust the entry window hour checks in the cBot.
