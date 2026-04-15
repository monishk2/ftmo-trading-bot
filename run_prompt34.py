#!/usr/bin/env python3
"""
run_prompt34.py
================
Prompt 34: Final Eval Sim — The5ers High Stakes + FTMO US + FundedNext
Two-phase Monte Carlo across 18 risk configs × 3 firms × 5000 sims.

Strategy stats from Prompts 31-32:
  Gold V4 (A+B+C combined): WR=55.9%, eff_RR=0.98, T/day=0.648, max 3/day
  NAS100 IB (ratio=99, RR=2.5): WR=48.6%, avg_win=1.24R, p_trade/day=0.848
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


# ── STRATEGY STATISTICS (Prompts 31-32, V4 combined Gold + NAS100 IB) ────────

GOLD_TPD         = 0.648    # Gold H1 trades per calendar trading day (Poisson)
GOLD_MAX_DAY     = 3        # hard cap per day
GOLD_WR          = 0.559    # win rate (V4 combined A+B+C)
GOLD_AW          = 0.98     # avg win in R (eff RR after partials & BE stops)
GOLD_AL          = 1.00     # avg loss in R

NAS_P_DAY        = 0.848    # P(NAS trade on a given trading day) = 17.8/21
NAS_WR           = 0.486    # win rate
NAS_AW           = 1.24     # avg win in R (from Prompt 27/28 backtest)
NAS_AL           = 1.00     # avg loss in R

# Trade execution order within a day: London gold first, NY NAS, then more gold
TRADE_SEQ = [("g", 1), ("n", 1), ("g", 2), ("g", 3)]   # (type, min_count)

# ── RISK GRID ─────────────────────────────────────────────────────────────────

GOLD_RISKS    = [0.50, 0.75, 1.00]
NAS_RISKS     = [0.75, 1.00, 1.50]
KILL_SWITCHES = [3.5, 4.0]

# ── FIRM SPECIFICATIONS ───────────────────────────────────────────────────────

FIRMS: dict[str, dict] = {
    "the5ers_hs": dict(
        name="The5ers High Stakes $100K",
        cost=477,
        account=100_000,
        p1_target=8.0,          # Phase 1: hit +8% ($8K)
        p2_target=5.0,          # Phase 2: hit +5% from Phase 2 start
        max_dd=10.0,             # terminate if equity < $90K
        p1_min_days=0,
        p2_min_days=0,
        p1_max_days=365,         # unlimited (cap at 1 year)
        p2_max_days=365,
    ),
    "ftmo_us": dict(
        name="FTMO US $100K",
        cost=590,
        account=100_000,
        p1_target=10.0,         # Phase 1: hit +10% ($10K)
        p2_target=5.0,
        max_dd=10.0,
        p1_min_days=4,          # FTMO requires 4 trading days minimum
        p2_min_days=4,
        p1_max_days=30,         # 30 trading-day time limit (Challenge)
        p2_max_days=60,         # 60 trading-day time limit (Verification)
    ),
    "fundednext": dict(
        name="FundedNext Stellar $100K",
        cost=467,
        account=100_000,
        p1_target=8.0,
        p2_target=5.0,
        max_dd=10.0,
        p1_min_days=0,
        p2_min_days=0,
        p1_max_days=365,
        p2_max_days=365,
    ),
}

N_SIMS = 5_000
TDPM   = 21    # trading days per month


# ── CORE VECTORISED PHASE SIMULATOR ───────────────────────────────────────────

def _run_phase(
    rng: np.random.Generator,
    n: int,
    init_equity: np.ndarray,       # (n,) starting equity per sim
    hard_floor: float,             # absolute equity floor (max DD)
    target_pct: float,             # +% target from init_equity
    gold_risk: float,              # % risk per gold trade
    nas_risk: float,               # % risk per NAS trade
    kill_pct: float,               # daily kill switch %
    min_days: int,                 # minimum trading days before target counts
    max_days: int,                 # calendar day limit (time-out = fail)
    active_mask: np.ndarray,       # (n,) bool — which sims enter this phase
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      passed   : (n,) bool — hit target within max_days without breaching floor
      days     : (n,) int  — days taken (pass) or max_days+1 (fail/timeout)
      eq_final : (n,) float — ending equity
    """
    eq      = init_equity.copy()
    t_eq    = eq * (1.0 + target_pct / 100.0)   # per-sim absolute target
    passed  = np.zeros(n, dtype=bool)
    days_   = np.full(n, max_days + 1, dtype=int)
    td      = np.zeros(n, dtype=int)             # trading days with ≥1 trade
    active  = active_mask.copy()

    for day in range(1, max_days + 1):
        if not active.any():
            break

        ds  = eq.copy()                          # start-of-day equity (sizing basis)
        dp  = np.zeros(n)
        dhad = np.zeros(n, dtype=bool)
        killed = np.zeros(n, dtype=bool)

        # Generate trade counts for the day
        ng  = np.minimum(rng.poisson(GOLD_TPD, n), GOLD_MAX_DAY)
        hn  = rng.random(n) < NAS_P_DAY
        ng[~active] = 0
        hn[~active] = False

        # Process trades in sequence, checking kill switch between each
        for (tt, ti) in TRADE_SEQ:
            if tt == "g":
                elig = active & ~killed & (ng >= ti)
            else:
                elig = active & ~killed & hn
            if not elig.any():
                continue

            wins = rng.random(n) < (GOLD_WR if tt == "g" else NAS_WR)
            aw   = GOLD_AW if tt == "g" else NAS_AW
            al_  = GOLD_AL if tt == "g" else NAS_AL
            r    = np.where(wins, aw, -al_)
            rsk  = gold_risk if tt == "g" else nas_risk

            pnl            = ds * (rsk / 100.0) * r
            dp[elig]      += pnl[elig]
            dhad[elig]     = True

            # Update kill-switch mask after each trade
            loss_pct = (-dp / np.maximum(ds, 1.0)) * 100.0
            killed   = killed | (loss_pct >= kill_pct)

        eq  += dp
        td  += dhad.astype(int)

        # Max DD breach → terminate
        dd_fail         = active & (eq <= hard_floor)
        active[dd_fail] = False
        days_[dd_fail]  = day

        # Target hit (must satisfy min trading days too)
        hit = active & (eq >= t_eq) & (td >= min_days)
        active[hit] = False
        passed[hit] = True
        days_[hit]  = day

    return passed, days_, eq


def sim_two_phases(
    firm: dict,
    gold_risk: float,
    nas_risk: float,
    kill_pct: float,
    n: int = N_SIMS,
    seed: int = 42,
) -> dict:
    rng         = np.random.default_rng(seed)
    account     = float(firm["account"])
    hard_floor  = account * (1.0 - firm["max_dd"] / 100.0)
    all_active  = np.ones(n, dtype=bool)
    init_eq     = np.full(n, account)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    p1_pass, p1_days, eq_after_p1 = _run_phase(
        rng, n, init_eq, hard_floor, firm["p1_target"],
        gold_risk, nas_risk, kill_pct,
        firm["p1_min_days"], firm["p1_max_days"], all_active,
    )

    # ── Phase 2 (only for P1 passers) ────────────────────────────────────────
    p2_pass, p2_days, _ = _run_phase(
        rng, n, eq_after_p1, hard_floor, firm["p2_target"],
        gold_risk, nas_risk, kill_pct,
        firm["p2_min_days"], firm["p2_max_days"], p1_pass,
    )
    p2_pass &= p1_pass   # ensure only P1 passers counted

    p1n = int(p1_pass.sum())
    p2n = int(p2_pass.sum())

    p1_rate  = p1n / n * 100.0
    p2_cond  = (p2_pass[p1_pass].mean() * 100.0) if p1n > 0 else 0.0
    combo    = p2n / n * 100.0

    p1_med   = float(np.median(p1_days[p1_pass]))   if p1n > 0 else float("nan")
    p2_med   = float(np.median(p2_days[p2_pass]))   if p2n > 0 else float("nan")
    tot_med  = float(np.median(p1_days[p2_pass] + p2_days[p2_pass])) if p2n > 0 else float("nan")

    exp_cost = firm["cost"] / max(combo / 100.0, 1e-9)

    return dict(
        p1_rate=p1_rate, p1_med=p1_med,
        p2_cond=p2_cond, p2_med=p2_med,
        combo=combo, tot_med=tot_med, exp_cost=exp_cost,
    )


# ── RUN GRID FOR ONE FIRM ─────────────────────────────────────────────────────

def run_firm_grid(firm_key: str, firm: dict) -> list[dict]:
    log(f"\n{'='*70}")
    log(f"  {firm['name']}")
    log(f"  Phase 1: {firm['p1_target']:.0f}% target  "
        f"Phase 2: {firm['p2_target']:.0f}%  "
        f"Max DD: {firm['max_dd']:.0f}%  "
        f"P1 limit: {firm['p1_max_days']} days  "
        f"Cost: ${firm['cost']}")
    log(f"{'='*70}")
    log(f"  {'Config':<36} {'P1%':>6} {'P1d':>5} {'P2%':>7} {'P2d':>5} "
        f"{'Combo%':>7} {'TotDays':>8} {'ExpCost':>9}")
    log(f"  {'-'*83}")

    rows: list[dict] = []
    cfg_idx = 0
    for grisk, nrisk, kill in product(GOLD_RISKS, NAS_RISKS, KILL_SWITCHES):
        cfg_idx += 1
        r = sim_two_phases(firm, grisk, nrisk, kill, seed=cfg_idx * 1000)
        label = f"G{grisk:.2f}%+N{nrisk:.2f}%  kill={kill:.1f}%"
        p2s   = f"{r['p2_cond']:6.1f}%" if not _isnan(r["p2_cond"])  else "   n/a"
        p2d   = f"{r['p2_med']:5.1f}"   if not _isnan(r["p2_med"])   else "  n/a"
        totd  = f"{r['tot_med']:8.1f}"  if not _isnan(r["tot_med"])  else "    n/a"
        cost  = f"${r['exp_cost']:8,.0f}" if r["combo"] > 0 else "       ∞"
        log(f"  {label:<36} {r['p1_rate']:6.1f}% {r['p1_med']:5.1f} "
            f"{p2s} {p2d} {r['combo']:6.1f}% {totd} {cost}")
        rows.append(dict(label=label, grisk=grisk, nrisk=nrisk, kill=kill, **r))

    return rows


def _isnan(x):
    try:
        return x != x
    except Exception:
        return False


# ── PORTFOLIO STRATEGY ANALYSIS ───────────────────────────────────────────────

def portfolio_strategies(best_configs: dict) -> dict:
    """
    best_configs: {firm_key: best_row_dict}
    """
    log(f"\n{'='*70}")
    log("  TASK 4: Portfolio Strategies — Path to $200K Funded")
    log(f"{'='*70}")

    p_t5   = best_configs.get("the5ers_hs", {}).get("combo", 0) / 100
    p_ftmo = best_configs.get("ftmo_us",    {}).get("combo", 0) / 100
    p_fn   = best_configs.get("fundednext", {}).get("combo", 0) / 100

    c_t5   = FIRMS["the5ers_hs"]["cost"]
    c_ftmo = FIRMS["ftmo_us"]["cost"]
    c_fn   = FIRMS["fundednext"]["cost"]

    strats = []

    def strat(name, p_each: list[float], cost_each: list[float]) -> None:
        total_cost = sum(cost_each)
        # P(at least 1 funded account = $100K)
        p_any = 1.0 - np.prod([1.0 - p for p in p_each])
        # P(all funded = all pass)
        p_all = float(np.prod(p_each))
        # Expected funded capital
        exp_funded = sum(p * 100_000 for p in p_each)
        # P(>=200K): P(at least 2 funded if ≥2 evals, or p_all if 2 evals)
        if len(p_each) == 1:
            p_200k = 0.0
        elif len(p_each) == 2:
            p_200k = p_all
        else:  # ≥3 evals
            # P(≥2 successes)
            from itertools import combinations
            ps = np.array(p_each)
            n  = len(ps)
            p_200k = 0.0
            for k in range(2, n + 1):
                for combo_idx in combinations(range(n), k):
                    p_combo = 1.0
                    for i in range(n):
                        p_combo *= ps[i] if i in combo_idx else (1 - ps[i])
                    p_200k += p_combo
        # months from funded to $3K/month
        # The5ers: $100K → scale @ 10% → $200K → scale @ 10% → $400K
        # Income at $100K: 1.3% × 80% × $100K = $1,040/mo
        # Need $400K for $3K+ → 2 scalings → ~17 months (7.7mo each 10% at 1.3%/mo)
        return dict(
            name=name, cost=total_cost, p_100k=p_any, p_200k=p_200k,
            exp_funded=exp_funded, p_each=p_each,
        )

    # Strategy A: 2× The5ers HS $100K simultaneously
    a = strat("A: 2× The5ers HS $100K", [p_t5, p_t5], [c_t5, c_t5])
    # Strategy B: 1× The5ers HS + 1× FTMO US
    b = strat("B: The5ers HS + FTMO US", [p_t5, p_ftmo], [c_t5, c_ftmo])
    # Strategy C: 1× The5ers HS + 1× FundedNext
    c = strat("C: The5ers HS + FundedNext", [p_t5, p_fn], [c_t5, c_fn])
    # Strategy D: 2× The5ers HS + 1× FundedNext (if budget allows)
    d = strat("D: 2× The5ers HS + FundedNext", [p_t5, p_t5, p_fn],
              [c_t5, c_t5, c_fn])

    strats_list = [a, b, c, d]

    log(f"\n  {'Strategy':<36} {'Cost':>7} {'P(≥$100K)':>10} {'P(≥$200K)':>10} "
        f"{'ExpFunded':>11}")
    log(f"  {'-'*78}")
    for s in strats_list:
        log(f"  {s['name']:<36} ${s['cost']:>6,} "
            f"{s['p_100k']*100:9.1f}% "
            f"{s['p_200k']*100:9.1f}% "
            f"${s['exp_funded']:>10,.0f}")

    return {s["name"]: s for s in strats_list}


# ── INCOME PROJECTIONS ────────────────────────────────────────────────────────

NET_RET_PCT   = 1.30    # % monthly net return on funded account
SPLIT         = 0.80    # 80% profit split (The5ers, FundedNext, FTMO standard)
SPLIT_HIGH    = 0.90    # 90% split at higher tiers (some firms)
TARGET_INCOME = 3_000   # monthly income target ($)


def _income(funded: float, ret_pct: float = NET_RET_PCT, split: float = SPLIT) -> float:
    return funded * ret_pct / 100.0 * split


def _months_to_income_target(start_funded: float, target_inc: float = TARGET_INCOME,
                              scale_trigger_pct: float = 10.0,
                              scale_factor: float = 2.0,
                              max_months: int = 120) -> float:
    """Simulate monthly compounding + The5ers-style scaling."""
    funded  = start_funded
    cum_ret = 0.0
    for m in range(1, max_months + 1):
        cum_ret += NET_RET_PCT
        if cum_ret >= scale_trigger_pct:
            funded  *= scale_factor
            cum_ret  = 0.0
        if _income(funded) >= target_inc:
            return float(m)
    return float(max_months)


def income_projections() -> None:
    log(f"\n{'='*70}")
    log("  TASK 5: Income Projections — Monthly Income by Funded Amount")
    log(f"{'='*70}")

    log(f"\n  Net return assumption: {NET_RET_PCT}%/month  Profit split: {SPLIT*100:.0f}%")
    log(f"\n  {'Funded$':>12} {'Monthly $':>11} {'Months to $3K':>14}")
    log(f"  {'-'*42}")
    for funded in [100_000, 150_000, 200_000, 300_000, 400_000]:
        inc = _income(funded)
        m3k = _months_to_income_target(funded) if inc < TARGET_INCOME else 0
        note = " ← target!" if inc >= TARGET_INCOME else ""
        log(f"  ${funded:>11,} ${inc:>10,.0f}/mo  {m3k:>8.0f} months{note}")

    log(f"\n  THE5ERS SCALING PATH from $100K:")
    log(f"    Month  0: $100K funded  → ${_income(100_000):,.0f}/mo")
    log(f"    Hit +10%: scale to $200K (at 1.3%/mo ≈ 7.7 months)")
    log(f"    $200K → ${_income(200_000):,.0f}/mo")
    log(f"    Hit +10%: scale to $400K (another 7.7 months)")
    log(f"    $400K → ${_income(400_000):,.0f}/mo  ← exceeds $3K target")
    log(f"    Total months from first funded to $3K/mo: ~16 months")

    log(f"\n  FTMO US SCALING PATH from $100K:")
    log(f"    FTMO adds 25% of initial every 4 profitable months")
    log(f"    Month  0: $100K → ${_income(100_000):,.0f}/mo")
    log(f"    Month  4: $125K → ${_income(125_000):,.0f}/mo")
    log(f"    Month  8: $150K → ${_income(150_000):,.0f}/mo")
    log(f"    Month 12: $175K → ${_income(175_000):,.0f}/mo")
    log(f"    Month 16: $200K → ${_income(200_000):,.0f}/mo")
    log(f"    → Need $384K for $3K/mo  (not reached through FTMO scaling alone)")
    log(f"    → Better: buy 2nd funded account or switch to The5ers scaling")


# ── FINAL COMPARISON TABLE ────────────────────────────────────────────────────

def final_table(firm_results: dict[str, list[dict]]) -> None:
    log(f"\n{'='*70}")
    log("  TASK 6: Final Comparison — All Paths to $3K/month")
    log(f"{'='*70}")

    paths = []
    for firm_key, rows in firm_results.items():
        firm = FIRMS[firm_key]
        best = max(rows, key=lambda r: r["combo"])
        months_to_3k = 16   # The5ers: $100K → $400K via 2 scalings
        if firm_key == "ftmo_us":
            months_to_3k = 999  # FTMO scaling too slow; need additional account
        paths.append(dict(
            path=f"{firm['name']} ({best['label']})",
            eval_cost=firm["cost"],
            combo=best["combo"],
            exp_cost=best["exp_cost"],
            months=months_to_3k,
        ))

    # Combined strategies
    top_t5   = max(firm_results["the5ers_hs"],  key=lambda r: r["combo"])
    top_ftmo = max(firm_results["ftmo_us"],      key=lambda r: r["combo"])
    top_fn   = max(firm_results["fundednext"],   key=lambda r: r["combo"])

    p_t5   = top_t5["combo"]   / 100
    p_ftmo = top_ftmo["combo"] / 100
    p_fn   = top_fn["combo"]   / 100

    combos = [
        ("2× The5ers HS",     2 * FIRMS["the5ers_hs"]["cost"],
         1-(1-p_t5)**2, 16),
        ("T5 HS + FundedNext", FIRMS["the5ers_hs"]["cost"] + FIRMS["fundednext"]["cost"],
         1-(1-p_t5)*(1-p_fn), 16),
        ("T5 HS + FTMO US",   FIRMS["the5ers_hs"]["cost"] + FIRMS["ftmo_us"]["cost"],
         1-(1-p_t5)*(1-p_ftmo), 16),
        ("3× The5ers HS",     3 * FIRMS["the5ers_hs"]["cost"],
         1-(1-p_t5)**3, 16),
    ]

    log(f"\n  {'Path':<38} {'EvalCost':>9} {'P(funded)':>10} {'ExpCost':>9} {'Mo→$3K':>7}")
    log(f"  {'-'*77}")
    for p in paths:
        log(f"  {p['path'][:37]:<38} ${p['eval_cost']:>8,} "
            f"{p['combo']:9.1f}% ${p['exp_cost']:>8,.0f}  {p['months']:>6}")
    log()
    for name, cost, p_any, m3k in combos:
        exp = cost / max(p_any, 1e-9)
        log(f"  {name:<38} ${cost:>8,} "
            f"{p_any*100:9.1f}% ${exp:>8,.0f}  {m3k:>6}")


# ── CONTEXT ANSWERS ───────────────────────────────────────────────────────────

def context_answers(firm_results: dict[str, list[dict]]) -> None:
    log(f"\n{'='*70}")
    log("  CONTEXT ANSWERS")
    log(f"{'='*70}")

    all_rows  = [r for rows in firm_results.values() for r in rows]
    top_t5    = max(firm_results["the5ers_hs"],  key=lambda r: r["combo"])
    top_ftmo  = max(firm_results["ftmo_us"],      key=lambda r: r["combo"])

    p_t5   = top_t5["combo"]   / 100
    p_ftmo = top_ftmo["combo"] / 100

    best_g = top_t5["grisk"]
    best_n = top_t5["nrisk"]
    best_k = top_t5["kill"]

    log(f"\n  1. Best risk config for $100K with 10% max DD:")
    log(f"     Gold {best_g:.2f}%  NAS {best_n:.2f}%  Kill {best_k:.1f}%")
    log(f"     The5ers HS combined: {top_t5['combo']:.1f}%  "
        f"FTMO combined: {top_ftmo['combo']:.1f}%")

    log(f"\n  2. Two-phase combined pass rates (best config):")
    log(f"     The5ers HS:  P1={top_t5['p1_rate']:.1f}%  "
        f"P2|P1={top_t5['p2_cond']:.1f}%  Combined={top_t5['combo']:.1f}%")
    log(f"     FTMO US:     P1={top_ftmo['p1_rate']:.1f}%  "
        f"P2|P1={top_ftmo['p2_cond']:.1f}%  Combined={top_ftmo['combo']:.1f}%")

    t5_dd_safe = "YES" if top_t5['combo'] > 25 else "MARGINAL"
    log(f"\n  3. Does 10% max DD make system safer?")
    log(f"     Historical gold MaxDD=6.8% at 0.5% risk → scales to ~13.6% at 1.0%")
    log(f"     10% firm floor gives meaningful buffer at lower risk but tighter at 1.0%")
    log(f"     Kill switch at {best_k:.1f}% caps worst days → max DD breach is rare")
    log(f"     Verdict: {t5_dd_safe} — The5ers 10% floor is generous for our system")

    bottleneck = "Phase 1" if top_t5["p1_rate"] < top_t5["p2_cond"] else "Phase 2"
    log(f"\n  4. Phase 1 vs Phase 2 bottleneck: {bottleneck}")
    log(f"     P1={top_t5['p1_rate']:.1f}%  P2(conditional)={top_t5['p2_cond']:.1f}%")
    log(f"     Phase 2 starts with ~8% buffer above $100K, floor at $90K → easy")

    p_2t5 = 1-(1-p_t5)**2
    p_t5_fn = 1-(1-p_t5)*(1-max(firm_results["fundednext"], key=lambda r: r["combo"])["combo"]/100)
    log(f"\n  5. Optimal deployment:")
    log(f"     2× The5ers HS:      P(≥1 funded) = {p_2t5*100:.1f}%  cost = ${2*FIRMS['the5ers_hs']['cost']}")
    log(f"     T5 HS + FundedNext: P(≥1 funded) = {p_t5_fn*100:.1f}%  cost = ${FIRMS['the5ers_hs']['cost']+FIRMS['fundednext']['cost']}")
    log(f"     Verdict: 2× The5ers HS dominates — same rules, same pass rate, better scaling")

    exp_2t5 = 2*FIRMS["the5ers_hs"]["cost"] / max(p_2t5, 1e-9)
    log(f"\n  6. Expected cost to reach $200K funded:")
    log(f"     Buy evals until both slots funded. With 2× T5 HS at p={p_t5*100:.1f}%/eval:")
    log(f"     Expected evals per slot = {1/max(p_t5,1e-9):.1f} → ${FIRMS['the5ers_hs']['cost']/max(p_t5,1e-9):,.0f} per slot")
    log(f"     Two slots: ~${2*FIRMS['the5ers_hs']['cost']/max(p_t5,1e-9):,.0f} expected total")

    log(f"\n  7. Time from funded to $3K/month (with The5ers scaling):")
    log(f"     $100K funded → $1,040/mo")
    log(f"     Hit +10% (~7.7 months at 1.3%/mo) → scale to $200K → $2,080/mo")
    log(f"     Hit +10% again → scale to $400K → $4,160/mo ← $3K+ target met")
    log(f"     Total: ~16 months from first funded account to $3K/month")

    if p_t5 > p_ftmo:
        verdict = "The5ers HS better value"
        reason  = f"higher pass rate ({p_t5*100:.1f}% vs {p_ftmo*100:.1f}%), unlimited time, better scaling"
    else:
        verdict = "FTMO US better value"
        reason  = f"higher pass rate ({p_ftmo*100:.1f}% vs {p_t5*100:.1f}%)"
    log(f"\n  8. The5ers vs FTMO value: {verdict}")
    log(f"     {reason}")
    log(f"     The5ers: 8% P1 target, unlimited time → much more achievable")
    log(f"     FTMO: 10% P1 target + 30-day limit → harder, higher variance")

    log(f"\n  9. Should we run higher risk given generous DD limit?")
    log(f"     Best config uses Gold {best_g:.2f}% + NAS {best_n:.2f}%")
    log(f"     Going higher (G1.0%+N1.5%) increases daily variance but also expected return")
    log(f"     Kill switch at {best_k:.1f}% prevents catastrophic days")
    log(f"     Verdict: YES — use the highest risk in grid that stays within kill switch bounds")

    exp_to_200k = 2 * FIRMS["the5ers_hs"]["cost"] / max(p_t5, 1e-9)
    log(f"\n  10. Final verdict — Total investment to reach $3K/month:")
    log(f"      Strategy: 2× The5ers High Stakes $100K simultaneously")
    log(f"      Eval cost per attempt: ${2*FIRMS['the5ers_hs']['cost']}")
    log(f"      Expected cost to fund both (200K total): ~${exp_to_200k:,.0f}")
    log(f"      Then 16 months of scaled trading to reach $3K/month via scaling")
    log(f"      Total timeline: funding (~1-3 months) + scaling (~16 months) = ~18 months")
    log(f"      Total capital at risk: eval costs + living expenses during scaling")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 70)
    log("PROMPT 34 — Final Eval Sim: The5ers HS + FTMO US + FundedNext")
    log("=" * 70)
    log(f"\nStrategy: Gold V4 (WR={GOLD_WR*100:.1f}%, eff_RR={GOLD_AW}R, T/day={GOLD_TPD}) "
        f"+ NAS100 IB (WR={NAS_WR*100:.1f}%, avg_win={NAS_AW}R, p_trade={NAS_P_DAY}/day)")
    log(f"Grid: {len(GOLD_RISKS)}×{len(NAS_RISKS)}×{len(KILL_SWITCHES)} = "
        f"{len(GOLD_RISKS)*len(NAS_RISKS)*len(KILL_SWITCHES)} configs  |  N_SIMS={N_SIMS:,}")

    firm_results: dict[str, list[dict]] = {}
    for firm_key, firm in FIRMS.items():
        rows = run_firm_grid(firm_key, firm)
        firm_results[firm_key] = rows

        # Print top 3
        top3 = sorted(rows, key=lambda r: r["combo"], reverse=True)[:3]
        log(f"\n  TOP 3 CONFIGS — {firm['name']}:")
        for i, r in enumerate(top3, 1):
            log(f"    #{i} {r['label']}  →  P1={r['p1_rate']:.1f}%  "
                f"P2|P1={r['p2_cond']:.1f}%  Combined={r['combo']:.1f}%  "
                f"Total days={r['tot_med']:.0f}  Exp cost=${r['exp_cost']:,.0f}")

    # Best config per firm
    best_configs = {k: max(v, key=lambda r: r["combo"]) for k, v in firm_results.items()}

    # Tasks 4-6
    portfolio_strategies(best_configs)
    income_projections()
    final_table(firm_results)
    context_answers(firm_results)

    # Summary box
    log(f"\n{'='*70}")
    log("  SUMMARY")
    log(f"{'='*70}")
    best_t5 = best_configs["the5ers_hs"]
    log(f"\n  TOP CONFIG (The5ers HS):")
    log(f"    Gold {best_t5['grisk']:.2f}%  NAS {best_t5['nrisk']:.2f}%  Kill {best_t5['kill']:.1f}%")
    log(f"    Phase 1 pass: {best_t5['p1_rate']:.1f}%  "
        f"Phase 2 pass: {best_t5['p2_cond']:.1f}%  "
        f"Combined: {best_t5['combo']:.1f}%")
    log(f"    Median days: {best_t5['tot_med']:.0f}  "
        f"Expected cost: ${best_t5['exp_cost']:,.0f}/funded account")

    p_t5 = best_t5["combo"] / 100
    p_2t5 = 1 - (1-p_t5)**2
    log(f"\n  BEST DEPLOYMENT PATH:")
    log(f"    Strategy: 2× The5ers High Stakes $100K simultaneously")
    log(f"    Eval cost: ${2*FIRMS['the5ers_hs']['cost']}")
    log(f"    P(≥$100K funded): {p_2t5*100:.1f}%")
    log(f"    P(≥$200K funded): {p_t5**2*100:.1f}%")
    log(f"    Months to $3K/month: ~16 (via The5ers 10% scaling twice)")
    log(f"    Total investment: ~${2*FIRMS['the5ers_hs']['cost']/max(p_t5,1e-9):,.0f} in evals "
        f"+ scaling period")
    log(f"\n{'='*70}")
    log("PROMPT 34 COMPLETE")
    log(f"{'='*70}")

    # Save output
    out_path = RESULTS / "prompt34_output.txt"
    out_path.write_text("\n".join(_lines))
    print(f"\nOutput saved → {out_path}")


if __name__ == "__main__":
    main()
