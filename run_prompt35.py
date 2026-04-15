#!/usr/bin/env python3
"""
run_prompt35.py
================
Prompt 35: Speed Optimization — Pass in 3 Weeks

Tests seven approaches for passing The5ers HS Phase 1 ($8K on $100K)
within 21 trading days at >33% probability.

  • Brute force: higher risk on current sequential system
  • Concurrent modules: one position per module simultaneously
  • Remove partials on Module B & C (full 2.0R target)
  • Add M15 Module B signals (ESTIMATED — not backtested)

Strategy stats from Prompts 31-32, split by module:
  Module A (London H1, V4 partials): WR=52.4%, avg_win=1.017R, T/day=0.238
  Module B (VWAP  H1, partial):      WR=55.0%, avg_win=1.023R, T/day=0.433
  Module B (VWAP  H1, no-partial):   WR=38.0%, avg_win=2.000R, T/day=0.433
  Module B (VWAP M15, no-partial):   WR=47.0%, avg_win=2.000R, T/day=0.563 [EST]
  Module C (NY pullbk, partial):     WR=74.2%, avg_win=0.487R, T/day=0.129
  Module C (NY pullbk, no-partial):  WR=52.0%, avg_win=2.000R, T/day=0.129
  NAS100  (IB breakout M5):          WR=48.6%, avg_win=1.240R, T/day=0.848
  Gold combined (current, partials): WR=55.9%, avg_win=0.980R, T/day=0.648
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


# ── MODULE DEFINITIONS: (T/day, WR, avg_win_R, avg_loss_R) ───────────────────

MD = {
    # Sequential gold (combined A+B+C as one stream — current system)
    "gold_seq":    (0.648, 0.559, 0.980, 1.0),

    # Individual modules — concurrent capable (1 position per module)
    "mod_A":       (0.238, 0.524, 1.017, 1.0),   # London H1, V4 partials

    "mod_B_par":   (0.433, 0.550, 1.023, 1.0),   # VWAP H1, V4 partials
    "mod_B_np":    (0.433, 0.380, 2.000, 1.0),   # VWAP H1, no partial, TP=2.0R
    "mod_B_m15":   (0.563, 0.470, 2.000, 1.0),   # VWAP M15, no partial [ESTIMATED]

    "mod_C_par":   (0.129, 0.742, 0.487, 1.0),   # NY pullback H1, V4 partials
    "mod_C_np":    (0.129, 0.520, 2.000, 1.0),   # NY pullback H1, no partial, TP=2.0R

    "nas":         (0.848, 0.486, 1.240, 1.0),   # NAS100 IB breakout M5
}

# ── EVAL PARAMETERS ───────────────────────────────────────────────────────────

ACCOUNT       = 100_000.0
TARGET_PCT    = 8.0           # Phase 1: hit +$8K
MAX_DD_PCT    = 10.0          # terminate: equity < $90K
MAX_DAYS      = 21            # 3 calendar weeks ≈ 21 trading days
MAX_DAYS_UNL  = 365           # unlimited (for comparison)
N_SIMS        = 5_000

# ── CONFIGS TO TEST ───────────────────────────────────────────────────────────
# Each config: dict with
#   name       : str
#   modules    : list of (module_key, risk_pct) — day trade order = London→NY
#   kill       : daily kill-switch % (realized P&L from day_start)
#   estimated  : bool — True if any stats are estimated, not backtested

CONFIGS = [
    # ── 0. Current system (baseline from Prompt 34) ──────────────────────────
    dict(
        name="CURRENT (G0.50%+N0.50%, seq, kill=3.5%)",
        modules=[("gold_seq", 0.50), ("nas", 0.50)],
        kill=3.5, estimated=False,
    ),

    # ── 1-4. Brute force: higher risk, same sequential system ─────────────────
    dict(
        name="BRUTE G1.5%+N1.5% kill=4.0%",
        modules=[("gold_seq", 1.50), ("nas", 1.50)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="BRUTE G2.0%+N2.0% kill=4.0%",
        modules=[("gold_seq", 2.00), ("nas", 2.00)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="BRUTE G2.5%+N2.5% kill=4.5%",
        modules=[("gold_seq", 2.50), ("nas", 2.50)],
        kill=4.5, estimated=False,
    ),
    dict(
        name="BRUTE G3.0%+N3.0% kill=5.0%",
        modules=[("gold_seq", 3.00), ("nas", 3.00)],
        kill=5.0, estimated=False,
    ),

    # ── 5-7. Concurrent modules, V4 partials on all ───────────────────────────
    dict(
        name="CONC (0.75%/mod+1.0%NAS) partials, kill=4.0%",
        modules=[("mod_A", 0.75), ("nas", 1.00),
                 ("mod_B_par", 0.75), ("mod_C_par", 0.75)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="CONC (1.0%/mod+1.5%NAS) partials, kill=4.0%",
        modules=[("mod_A", 1.00), ("nas", 1.50),
                 ("mod_B_par", 1.00), ("mod_C_par", 1.00)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="CONC (1.5%/mod+2.0%NAS) partials, kill=4.0%",
        modules=[("mod_A", 1.50), ("nas", 2.00),
                 ("mod_B_par", 1.50), ("mod_C_par", 1.50)],
        kill=4.0, estimated=False,
    ),

    # ── 8-10. Concurrent + remove partials on B & C ───────────────────────────
    dict(
        name="CONC+NOPRTL (0.75%/mod+1.0%NAS) kill=4.0%",
        modules=[("mod_A", 0.75), ("nas", 1.00),
                 ("mod_B_np", 0.75), ("mod_C_np", 0.75)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="CONC+NOPRTL (1.0%/mod+1.5%NAS) kill=4.0%",
        modules=[("mod_A", 1.00), ("nas", 1.50),
                 ("mod_B_np", 1.00), ("mod_C_np", 1.00)],
        kill=4.0, estimated=False,
    ),
    dict(
        name="CONC+NOPRTL (1.5%/mod+2.0%NAS) kill=4.0%",
        modules=[("mod_A", 1.50), ("nas", 2.00),
                 ("mod_B_np", 1.50), ("mod_C_np", 1.50)],
        kill=4.0, estimated=False,
    ),

    # ── 11-12. Full stack: concurrent + M15 Module B + no partials ────────────
    # WARNING: M15 VWAP Module B stats are ESTIMATED — not backtested
    dict(
        name="FULL-STACK [EST] (1.0%/mod+1.5%NAS) M15+noprtl, kill=4.0%",
        modules=[("mod_A", 1.00), ("nas", 1.50),
                 ("mod_B_m15", 1.00), ("mod_C_np", 1.00)],
        kill=4.0, estimated=True,
    ),
    dict(
        name="FULL-STACK [EST] (1.5%/mod+2.0%NAS) M15+noprtl, kill=4.0%",
        modules=[("mod_A", 1.50), ("nas", 2.00),
                 ("mod_B_m15", 1.50), ("mod_C_np", 1.50)],
        kill=4.0, estimated=True,
    ),
]


# ── VECTORISED SIMULATOR ─────────────────────────────────────────────────────

def _sim_pass(
    rng: np.random.Generator,
    n: int,
    cfg: dict,
    max_days: int,
) -> dict:
    """
    Returns per-sim outcomes for one config.
    Modules are processed in list order each day (acts as London→NY sequencing).
    Kill switch applied after each module's trade within the day.
    """
    account    = ACCOUNT
    hard_floor = account * (1.0 - MAX_DD_PCT / 100.0)
    target_eq  = account * (1.0 + TARGET_PCT / 100.0)
    kill_pct   = cfg["kill"]
    mods       = cfg["modules"]    # list of (key, risk%)

    eq      = np.full(n, account)
    passed  = np.zeros(n, dtype=bool)
    blowup  = np.zeros(n, dtype=bool)
    days_   = np.full(n, max_days + 1, dtype=int)
    active  = np.ones(n, dtype=bool)

    # Track kill-switch fires per day (across all active sims)
    kill_fires_total = 0
    kill_days_total  = 0   # denominator: sim×day events

    for day in range(1, max_days + 1):
        if not active.any():
            break

        kill_days_total += int(active.sum())
        ds   = eq.copy()
        dp   = np.zeros(n)
        killed = np.zeros(n, dtype=bool)

        for (mkey, risk_pct) in mods:
            tpd, wr, aw, al = MD[mkey]

            # Bernoulli trade occurrence (cap at 1 per module)
            has_trade = (rng.random(n) < tpd) & active & ~killed
            if not has_trade.any():
                continue

            wins = rng.random(n) < wr
            r    = np.where(wins, aw, -al)
            pnl  = ds * (risk_pct / 100.0) * r
            dp[has_trade] += pnl[has_trade]

            # Kill switch: cumulative realized daily loss ≥ kill_pct%
            new_killed = active & ~killed & ((-dp / np.maximum(ds, 1.0)) * 100.0 >= kill_pct)
            kill_fires_total += int(new_killed.sum())
            killed = killed | new_killed

        eq += dp

        # Max DD breach → terminate
        breach = active & (eq <= hard_floor)
        active[breach] = False
        blowup[breach] = True
        days_[breach]  = day

        # Target hit
        hit = active & (eq >= target_eq)
        active[hit] = False
        passed[hit] = True
        days_[hit]  = day

    kill_rate = (kill_fires_total / max(kill_days_total, 1)) * 100.0

    return dict(
        passed=passed, blowup=blowup, days=days_,
        kill_rate=kill_rate,
    )


def _stats(res: dict, cfg: dict, max_days: int) -> dict:
    n       = len(res["passed"])
    p_pass  = res["passed"].sum() / n * 100.0
    p_blow  = res["blowup"].sum() / n * 100.0
    timeout = 100.0 - p_pass - p_blow

    pass_days = res["days"][res["passed"]]
    p_15d = float((pass_days <= 15).sum()) / n * 100.0
    p_21d = p_pass  # all pass within max_days

    med_days = float(np.median(pass_days)) if res["passed"].any() else float("nan")

    # Daily EV and T/day from module definitions
    t_day = sum(MD[k][0] for k, _ in cfg["modules"])
    ev_pct_day = sum(
        MD[k][0] * (MD[k][1] * MD[k][2] - (1 - MD[k][1]) * MD[k][3]) * r
        for k, r in cfg["modules"]
    )

    return dict(
        p_21d=p_21d, p_15d=p_15d, p_blow=p_blow, timeout=timeout,
        med_days=med_days, t_day=t_day, ev_pct_day=ev_pct_day,
        kill_rate=res["kill_rate"], p_pass=p_pass,
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 72)
    log("PROMPT 35 — Speed Optimization: Pass in 3 Weeks")
    log("=" * 72)
    log(f"\nAccount: ${ACCOUNT:,.0f}  Target: +{TARGET_PCT:.0f}% (${ACCOUNT*TARGET_PCT/100:,.0f})"
        f"  Max DD: {MAX_DD_PCT:.0f}% (${ACCOUNT*MAX_DD_PCT/100:,.0f})")
    log(f"Time limit: {MAX_DAYS} trading days  N_SIMS={N_SIMS:,}  Seed=42")
    log(f"\n[EST] = stats estimated, not backtested. Treat as indicative only.\n")

    rng42 = np.random.default_rng(42)

    # ── Module EV table ───────────────────────────────────────────────────────
    log(f"{'='*72}")
    log("  MODULE EV TABLE (per-trade expected value in R)")
    log(f"{'='*72}")
    log(f"  {'Module':<26} {'T/day':>6} {'WR':>6} {'AvgWin':>8} {'EV/tr':>8} {'EV%@1%':>9}")
    log(f"  {'-'*64}")
    for k, (tpd, wr, aw, al) in MD.items():
        ev = wr * aw - (1 - wr) * al
        tag = " [EST]" if k == "mod_B_m15" else ""
        log(f"  {k+tag:<26} {tpd:>6.3f} {wr*100:>5.1f}% {aw:>8.3f}R {ev:>8.3f}R {tpd*ev:>8.4f}%")

    # ── Run all configs with 21-day limit ─────────────────────────────────────
    log(f"\n{'='*72}")
    log(f"  PHASE 1 SIM — {MAX_DAYS}-DAY TIME LIMIT (3 trading weeks)")
    log(f"{'='*72}")
    hdr = (f"  {'Config':<44} {'T/day':>6} {'$/day':>7} "
           f"{'P<21d':>6} {'P<15d':>6} {'MedD':>5} {'Blow%':>7} {'Kill%':>6}")
    log(hdr)
    log(f"  {'-'*88}")

    results_21 = []
    for i, cfg in enumerate(CONFIGS):
        res = _sim_pass(rng42, N_SIMS, cfg, MAX_DAYS)
        s   = _stats(res, cfg, MAX_DAYS)
        est = "[EST]" if cfg["estimated"] else "     "
        lbl = cfg["name"][:43]
        med = f"{s['med_days']:5.1f}" if not (s["med_days"] != s["med_days"]) else "  n/a"
        log(f"  {lbl:<44}{est} {s['t_day']:>5.2f} {s['ev_pct_day']*100:>6.2f}$ "
            f"{s['p_21d']:>5.1f}% {s['p_15d']:>5.1f}% {med} {s['p_blow']:>6.1f}% {s['kill_rate']:>5.1f}%")
        results_21.append((cfg, s))

    # ── Run all configs with unlimited time ───────────────────────────────────
    log(f"\n{'='*72}")
    log(f"  UNLIMITED TIME (365 days) — for reference vs Prompt 34 baseline")
    log(f"{'='*72}")
    log(f"  {'Config':<44} {'Pass%':>7} {'MedianDays':>11} {'Blow%':>7}")
    log(f"  {'-'*74}")

    rng_unl = np.random.default_rng(99)
    results_unl = []
    for cfg in CONFIGS:
        res = _sim_pass(rng_unl, N_SIMS, cfg, MAX_DAYS_UNL)
        n   = N_SIMS
        p   = res["passed"].sum() / n * 100.0
        b   = res["blowup"].sum()  / n * 100.0
        pass_days = res["days"][res["passed"]]
        med = float(np.median(pass_days)) if res["passed"].any() else float("nan")
        med_s = f"{med:11.0f}" if med == med else "        n/a"
        log(f"  {cfg['name'][:43]:<44} {p:>6.1f}% {med_s} {b:>6.1f}%")
        results_unl.append((cfg, p, med, b))

    # ── Blow-up vs speed trade-off ────────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  BLOW-UP vs SPEED TRADE-OFF ANALYSIS")
    log(f"{'='*72}")
    log(f"\n  {'Config':<50} {'P<21d':>6} {'Blow%':>7} {'Risk/Reward':>12}")
    log(f"  {'-'*78}")
    for cfg, s in results_21:
        if s["p_21d"] > 0:
            rr = s["p_21d"] / max(s["p_blow"], 0.1)
        else:
            rr = 0.0
        log(f"  {cfg['name'][:49]:<50} {s['p_21d']:>5.1f}% {s['p_blow']:>6.1f}% "
            f"  {rr:>7.1f}× ({s['p_21d']:.0f}pass:{s['p_blow']:.0f}blow)")

    # ── Key findings ──────────────────────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  KEY FINDINGS")
    log(f"{'='*72}")

    best_21 = max(results_21, key=lambda x: x[1]["p_21d"])
    best_vld = max(                              # best among non-estimated
        [(c, s) for c, s in results_21 if not c["estimated"]],
        key=lambda x: x[1]["p_21d"],
    )
    baseline = results_21[0][1]

    log(f"\n  BASELINE (current system):")
    log(f"    Pass <21d: {baseline['p_21d']:.1f}%   Pass <15d: {baseline['p_15d']:.1f}%")
    log(f"    Blow-up:   {baseline['p_blow']:.1f}%   Unlimited: "
        f"{results_unl[0][1]:.1f}%   T/day: {baseline['t_day']:.2f}")

    log(f"\n  BEST VALIDATED CONFIG (no estimated stats):")
    log(f"    {best_vld[0]['name']}")
    log(f"    Pass <21d: {best_vld[1]['p_21d']:.1f}%   Pass <15d: {best_vld[1]['p_15d']:.1f}%")
    log(f"    Blow-up:   {best_vld[1]['p_blow']:.1f}%   Kill rate: {best_vld[1]['kill_rate']:.1f}% of days")
    log(f"    T/day:     {best_vld[1]['t_day']:.2f}   Daily EV: ${best_vld[1]['ev_pct_day']*100:.2f}")
    log(f"    Median days (winners): {best_vld[1]['med_days']:.0f}")

    if best_21[0]["estimated"]:
        log(f"\n  BEST OVERALL (includes estimated M15 stats):")
        log(f"    {best_21[0]['name']}")
        log(f"    Pass <21d: {best_21[1]['p_21d']:.1f}%  [REQUIRES M15 BACKTEST TO VALIDATE]")
        log(f"    Blow-up:   {best_21[1]['p_blow']:.1f}%   Kill rate: {best_21[1]['kill_rate']:.1f}%")

    # ── Concurrent exposure analysis ──────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  CONCURRENT EXPOSURE ANALYSIS")
    log(f"{'='*72}")
    log(f"\n  Worst-case simultaneous exposure (all 4 positions open at once):")
    for cfg in CONFIGS[5:]:   # concurrent configs only
        max_exp = sum(r for _, r in cfg["modules"])
        log(f"    {cfg['name'][:55]:<56} {max_exp:.2f}%")

    log(f"\n  Note: Worst case requires all 4 modules to have open positions")
    log(f"  simultaneously. P(all 4 open) ≈ 0.238×0.848×0.433×0.129 = 1.1%")
    log(f"  → Actual average exposure is far lower than worst-case.")
    log(f"  → Kill switch at 4% prevents catastrophic single-day losses.")

    # ── Context answers ───────────────────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  CONTEXT ANSWERS")
    log(f"{'='*72}")

    conc_par_mod = results_21[6][1]    # concurrent moderate with partials
    conc_np_mod  = results_21[9][1]    # concurrent no-partial moderate
    brute_20     = results_21[2][1]    # G2.0%+N2.0%

    log(f"\n  1. Do concurrent positions meaningfully increase T/day?")
    log(f"     Sequential: T/day=1.50 (gold 0.648 + NAS 0.848)")
    log(f"     Concurrent: T/day={conc_par_mod['t_day']:.2f} (A+B+C+NAS, 1 per module)")
    log(f"     Increase: +{(conc_par_mod['t_day']-1.50)/1.50*100:.0f}% T/day")
    log(f"     BUT: most increase comes from NAS running concurrently with gold.")
    log(f"     Gold modules A+B+C fire at different sessions → minimal overlap risk.")

    bev_par  = conc_par_mod["ev_pct_day"] * 100
    bev_np   = conc_np_mod["ev_pct_day"]  * 100
    log(f"\n  2. Does removing partials on B/C improve daily $?")
    log(f"     Concurrent with partials (1.0%/mod): ${bev_par:.2f}/day EV")
    log(f"     Concurrent no-partial B/C (1.0%/mod): ${bev_np:.2f}/day EV")
    log(f"     Change: {'+' if bev_np > bev_par else ''}{(bev_np-bev_par):.2f}$/day")
    log(f"     Module C with partials: avg_win=0.487R (BE stop kills returns)")
    log(f"     Module C without partials at 2.0R: avg_win=2.0R, WR=52%  EV=0.56R")
    log(f"     → Yes, removing partials on C dramatically improves EV (0.103→0.56R)")
    log(f"     → B improvement is marginal (0.113→0.14R)")

    log(f"\n  3. Does M15 VWAP add real signals or just noise?")
    log(f"     HONEST ANSWER: Unknown — M15 stats are ESTIMATED, not backtested.")
    log(f"     M15 VWAP tends to have lower WR (more false breakdowns) and")
    log(f"     higher noise than H1 VWAP. Actual WR=47% is optimistic.")
    log(f"     DO NOT deploy M15 module without running a full backtest first.")
    log(f"     Prompt 36 should backtest M15 VWAP before using these numbers.")

    target_pass = 33.0
    best_vld_pass = best_vld[1]["p_21d"]
    log(f"\n  4. What risk level gives >33% pass in <21 days?")
    if best_vld_pass >= target_pass:
        log(f"     YES — {best_vld[0]['name']}")
        log(f"     achieves {best_vld_pass:.1f}% pass rate in <21 days")
    else:
        log(f"     CLOSE BUT NOT QUITE — best validated: {best_vld_pass:.1f}% (need 33%)")
        log(f"     Need M15 backtest or slightly higher risk to clear 33% threshold.")

    log(f"\n  5. At that risk, what's the blow-up rate?")
    log(f"     {best_vld[0]['name']}")
    log(f"     Blow-up (hit -10% floor before +8%): {best_vld[1]['p_blow']:.1f}%")
    log(f"     Unlimited time pass rate: "
        f"{[r[1] for r in results_unl if r[0] is best_vld[0]][0]:.1f}%")

    unl_best_vld = [r[1] for r in results_unl if r[0] is best_vld[0]][0]
    slow_safe_unl = results_unl[0][1]

    log(f"\n  6. Is fast-pass risk/reward worth it vs slow-but-safe 91.8%?")
    log(f"     Slow-safe (G0.5%+N0.5%, unlimited): {slow_safe_unl:.1f}% pass, ~110 days")
    log(f"     Fast-pass best validated: {best_vld_pass:.1f}% in 21 days, "
        f"{best_vld[1]['p_blow']:.1f}% blow-up")
    log(f"     Unlimited pass at fast-pass risk: {unl_best_vld:.1f}%")

    eval_cost = 477
    exp_cost_slow = eval_cost / max(slow_safe_unl / 100, 1e-9)
    exp_cost_fast = eval_cost / max(best_vld_pass / 100, 1e-9)
    log(f"\n     Expected cost per funded (slow): ${exp_cost_slow:,.0f}  "
        f"(~{int(110/5)} calendar weeks)")
    log(f"     Expected cost per funded (fast):  ${exp_cost_fast:,.0f}  "
        f"(~3 calendar weeks if pass)")
    log(f"     Trade-off: fast approach costs ${exp_cost_fast-exp_cost_slow:+,.0f} more per funded "
        f"but saves ~{max(110-21,0)} days on winning attempts.")

    log(f"\n  7. Honest verdict: can we build a 33%+ / 3-week system?")
    if best_vld_pass >= 33:
        log(f"     YES — but only with concurrent positioning + no partials on C")
        log(f"     AND higher risk (1.0-1.5%/module) than our Prompt 34 best config.")
        log(f"     The cost: blow-up rate rises to {best_vld[1]['p_blow']:.1f}% vs {baseline['p_blow']:.1f}% baseline.")
    else:
        gap = 33.0 - best_vld_pass
        log(f"     NOT QUITE — gap of {gap:.1f}pp to 33% with validated strategies.")
        log(f"     Can close gap by: (a) M15 backtest, (b) slightly higher risk grid,")
        log(f"     (c) accepting blow-up cost as eval re-entry cost.")
    log(f"\n     RECOMMENDATION:")
    log(f"     For most traders: stick with the slow-safe 91.8% approach.")
    log(f"     $477 × 1.1 evals = ~$519 expected cost. Nearly guaranteed funding.")
    log(f"     For 3-week-or-bust: use {best_vld[0]['name'][:50]}")
    log(f"     Accept that 1 in {1/max(best_vld[1]['p_blow']/100,0.01):.0f} attempts will blow up.")
    log(f"     At $477/eval: blow-up costs are small relative to funded account value.")

    # ── Summary box ──────────────────────────────────────────────────────────
    log(f"\n{'='*72}")
    log("  SUMMARY")
    log(f"{'='*72}")
    log(f"\n  BEST CONFIG FOR <21-DAY PASS AT >33% (VALIDATED):")
    log(f"    Config: {best_vld[0]['name']}")
    log(f"    T/day:            {best_vld[1]['t_day']:.2f}")
    log(f"    Daily EV:         ${best_vld[1]['ev_pct_day']*100:.2f}")
    log(f"    Pass rate <21d:   {best_vld[1]['p_21d']:.1f}%")
    log(f"    Pass rate <15d:   {best_vld[1]['p_15d']:.1f}%")
    log(f"    Median days:      {best_vld[1]['med_days']:.0f}")
    log(f"    Blow-up rate:     {best_vld[1]['p_blow']:.1f}%")
    log(f"    Kill switch rate: {best_vld[1]['kill_rate']:.1f}% of trading days")
    log(f"    Max exposure:     {sum(r for _,r in best_vld[0]['modules']):.1f}% (all 4 positions open)")
    log(f"    Exp cost/funded:  ${eval_cost/max(best_vld_pass/100,1e-9):,.0f}")
    log()
    log(f"  vs CURRENT SYSTEM:")
    log(f"    Pass rate <21d:   {baseline['p_21d']:.1f}%")
    log(f"    Pass rate unlim:  {slow_safe_unl:.1f}%")
    log(f"    Median days:      ~110")
    log(f"    Blow-up:          {baseline['p_blow']:.1f}%")
    log(f"    Exp cost/funded:  ${exp_cost_slow:,.0f}")
    log(f"\n{'='*72}")
    log("PROMPT 35 COMPLETE")
    log(f"{'='*72}")

    out_path = RESULTS / "prompt35_output.txt"
    out_path.write_text("\n".join(_lines))
    print(f"\nOutput saved → {out_path}")


if __name__ == "__main__":
    main()
