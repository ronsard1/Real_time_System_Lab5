"""
monte_carlo_nmr.py - Monte Carlo reliability of simplex, TMR, 5MR and 7MR across a failure-rate sweep
(Mini Project 5.2). Standard library only; matplotlib optional for the figures.

Fault model (Section 5.1). Each module fails at per-hour rate lambda. A fault is TRANSIENT with probability
P_TRANSIENT (down for 1 hour, self-clearing) or PERMANENT (down for REPAIR hours until repaired). The voter is
"at least K of N must agree": the system is down whenever more than N - K modules are down
(simplex 1-of-1, TMR 2-of-3, 5MR 3-of-5, 7MR 4-of-7).

Simulation. Module up-times are drawn as geometric numbers of hours (exactly the distribution produced by a
per-hour uniform draw below lambda), so only fault and repair instants are simulated (event-driven), which
makes long missions and many trials affordable. Each trial is one mission of HORIZON hours starting with all
modules up.

Statistics. System failures form a point process, so the pooled failure count is treated as Poisson and an exact
95 % interval is computed for the MTBF. Trials are added in batches until the MTBF interval half-width is at
most 10 % of the estimate (about 400 failures) or a budget of MAX_TRIALS is reached; when no failure is observed
a one-sided 95 % lower bound is reported. Mission reliability R(T) uses a Wilson 95 % interval.

    python monte_carlo_nmr.py     -> results_5_2.txt, sweep_results.csv, figures/
"""
import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
P_TRANSIENT = 0.7
REPAIR = 24
HORIZON = 20_000                       # hours per trial (one mission)
MISSION = 1_000                        # hours, for mission reliability R(T)
RATES = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
CONFIGS = [("simplex", 1, 1), ("TMR", 3, 2), ("5MR", 5, 3), ("7MR", 7, 4)]
TARGET_REL_HALF = 0.10
MIN_TRIALS, MAX_TRIALS, BATCH = 200, 60_000, 200
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- simulation
def geom(rng, lam):
    """Hours until the first per-hour draw below lam (support 1, 2, ...)."""
    return int(math.log(1.0 - rng.random()) / math.log(1.0 - lam)) + 1


def module_down_intervals(rng, lam, horizon):
    out, t = [], 0
    while True:
        t += geom(rng, lam)                       # up period ends with a fault
        if t >= horizon:
            return out
        d = 1 if rng.random() < P_TRANSIENT else REPAIR
        out.append((t, min(t + d, horizon), d == 1))
        t += d


def trial(rng, lam, n, k):
    """One mission. Returns (system failures, system downtime, time of first failure or None)."""
    ev = []
    for _ in range(n):
        for a, b, _ in module_down_intervals(rng, lam, HORIZON):
            ev.append((a, 1))
            ev.append((b, -1))
    if not ev:
        return 0, 0, None
    ev.sort(key=lambda e: (e[0], e[1]))           # at equal times process repairs (-1) before faults (+1)
    limit = n - k                                 # system down when more than n - k modules are down
    down = fails = 0
    downtime, since, first = 0, None, None
    for t, d in ev:
        before = down > limit
        down += d
        after = down > limit
        if after and not before:
            fails += 1
            since = t
            if first is None:
                first = t
        elif before and not after:
            downtime += t - since
    if down > limit:
        downtime += HORIZON - since
    return fails, downtime, first


# ---------------------------------------------------------------- analytical (Sections 5.1, 5.2)
def analytic(lam, n, k, discrete=False):
    """Steady-state formulas (independent modules). Availability = 1 - P(more than n-k down) is exact for any
    repair-time distribution. The failure rate counts arrivals of the fatal extra fault while n-k modules are down.
    discrete=True corrects that rate for the hour-step model: faults start only on whole hours, so a module that
    is down for d hours overlaps a later fault for d - 0.5 h on average (a fault in the same step as another is one
    event, not two). For 1-hour transients this halves the overlap rate."""
    m = P_TRANSIENT * 1 + (1 - P_TRANSIENT) * REPAIR            # mean module down time (MTTR of a module)
    q = lam * m / (1 + lam * m)                                 # steady-state module unavailability
    q_r = lam * (m - 0.5) / (1 + lam * m) if discrete else q    # probability "down" as seen by a new fault
    Q = sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(n - k + 1, n + 1))
    j = n - k                                                   # one more fault from here brings the system down
    rate = math.comb(n, j) * q_r ** j * (1 - q) ** (n - j) * (n - j) * lam
    mtbf = (1 - Q) / rate                                       # mean up time between system failures
    return dict(mtbf=mtbf, mttr=Q / rate, avail=1 - Q, rel=math.exp(-MISSION / mtbf), q=q)


def nonrepairable_rel(lam, n, k, T=MISSION):
    """Textbook mission reliability without repair: at least k of n modules survive T (R_m = e^-lambda T)."""
    r = math.exp(-lam * T)
    return sum(math.comb(n, j) * r ** j * (1 - r) ** (n - j) for j in range(k, n + 1))


# ---------------------------------------------------------------- statistics
def poisson_ci(k, conf=0.95):
    def cdf(n, mu):
        if mu <= 0:
            return 1.0
        return sum(math.exp(i * math.log(mu) - mu - math.lgamma(i + 1)) for i in range(n + 1))

    def solve(f, lo, hi):
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    a = (1 - conf) / 2
    lower = 0.0 if k == 0 else solve(lambda mu: 1 - cdf(k - 1, mu) < a, 0, 2 * k + 10)
    upper = solve(lambda mu: cdf(k, mu) > a, 0, 2 * k + 20)
    return lower, upper


def wilson(s, n, z=1.96):
    p = s / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def run_point(lam, n, k, rng):
    trials = fails = down = survived = 0
    while True:
        for _ in range(BATCH):
            f, d, first = trial(rng, lam, n, k)
            trials += 1
            fails += f
            down += d
            survived += first is None or first >= MISSION
        if trials >= MIN_TRIALS:
            if fails and 1.96 / math.sqrt(fails) <= TARGET_REL_HALF:
                break
            if trials >= MAX_TRIALS:
                break
    total = trials * HORIZON
    up = total - down
    lo, hi = poisson_ci(fails)
    if fails:
        mtbf, ci = up / fails, (up / hi, up / lo if lo > 0 else float("inf"))
    else:
        mtbf, ci = float("inf"), (up / 3.0, float("inf"))      # 3.0 = one-sided 95 % bound for zero events
    r_lo, r_hi = wilson(survived, trials)
    return dict(trials=trials, fails=fails, mtbf=mtbf, ci=ci, avail=up / total,
                mttr=down / fails if fails else float("nan"), rel=survived / trials, rel_ci=(r_lo, r_hi))


def fmt(x):
    return "inf" if x == float("inf") else f"{x:,.0f}"


def main():
    rng = random.Random(SEED)
    rows = []
    say(f"Faults: {P_TRANSIENT:.0%} transient (1 h), {1 - P_TRANSIENT:.0%} permanent ({REPAIR} h repair); mean module MTTR "
        f"{P_TRANSIENT + (1 - P_TRANSIENT) * REPAIR:.1f} h; trial = {HORIZON:,} h mission; R(T) for T = {MISSION} h")
    say(f"Trials per point: batches of {BATCH} until the 95 % MTBF interval half-width <= {TARGET_REL_HALF:.0%} "
        f"(about 400 failures), max {MAX_TRIALS:,}")
    say("=" * 170)
    say(f"{'lambda/h':>9s} {'config':8s}{'trials':>8s}{'fails':>7s}{'MTBF emp':>15s}{'(95 % CI)':>27s}{'textbook':>16s}{'dev':>8s}"
        f"{'discrete':>16s}{'dev':>8s}{'avail emp':>11s}{'avail anal':>11s}{'R(T) emp':>9s}{'R(T) an':>9s}{'R norep':>10s}")
    for lam in RATES:
        for name, n, k in CONFIGS:
            e, a, ad = run_point(lam, n, k, rng), analytic(lam, n, k), analytic(lam, n, k, discrete=True)
            nr = nonrepairable_rel(lam, n, k)
            dev = (e["mtbf"] - a["mtbf"]) / a["mtbf"] * 100 if e["fails"] else float("nan")
            devd = (e["mtbf"] - ad["mtbf"]) / ad["mtbf"] * 100 if e["fails"] else float("nan")
            ci = f"{fmt(e['ci'][0])} - {fmt(e['ci'][1])}"
            mt = fmt(e["mtbf"]) if e["fails"] else f"> {fmt(e['ci'][0])}"
            say(f"{lam:>9.0e} {name:8s}{e['trials']:>8,d}{e['fails']:>7,d}{mt:>15s} ({ci:>25s}){fmt(a['mtbf']):>16s}"
                f"{dev:>+7.1f}%{fmt(ad['mtbf']):>16s}{devd:>+7.1f}%{e['avail']:>11.7f}{a['avail']:>11.7f}"
                f"{e['rel']:>9.4f}{ad['rel']:>9.4f}{nr:>10.4f}")
            rows.append(dict(failure_rate=lam, config=name, n=n, k=k, trials=e["trials"], failures=e["fails"],
                             mtbf_emp=e["mtbf"], mtbf_ci_low=e["ci"][0], mtbf_ci_high=e["ci"][1], mtbf_analytic=a["mtbf"],
                             mtbf_dev_pct=dev, mtbf_analytic_discrete=ad["mtbf"], mtbf_dev_discrete_pct=devd, avail_emp=e["avail"], avail_analytic=a["avail"],
                             avail_dev_pct=(e["avail"] - a["avail"]) / a["avail"] * 100,
                             mttr_emp=e["mttr"], mttr_analytic=a["mttr"], rel_emp=e["rel"], rel_ci_low=e["rel_ci"][0],
                             rel_ci_high=e["rel_ci"][1], rel_analytic=ad["rel"], rel_nonrepairable=nr))
        say("")
    with open(os.path.join(HERE, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    say("Mission reliability without repair (textbook R_TMR = 3R^2 - 2R^3), T = 1000 h:")
    for lam in (1e-4, 5e-4, 6.93e-4, 1e-3, 5e-3):
        say(f"  lambda {lam:.2e} (lambda T = {lam * MISSION:5.2f}): simplex {nonrepairable_rel(lam, 1, 1):.4f}  TMR "
            f"{nonrepairable_rel(lam, 3, 2):.4f}  5MR {nonrepairable_rel(lam, 5, 3):.4f}  7MR {nonrepairable_rel(lam, 7, 4):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        cols = {"simplex": "#7f7f7f", "TMR": "#1f77b4", "5MR": "#ff7f0e", "7MR": "#2ca02c"}
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for name, _, _ in CONFIGS:
            r = [x for x in rows if x["config"] == name]
            lam = [x["failure_rate"] for x in r]
            ax[0].plot(lam, [x["mtbf_analytic_discrete"] for x in r], "-", color=cols[name], lw=1)
            ax[0].plot(lam, [x["mtbf_analytic"] for x in r], ":", color=cols[name], lw=1)
            em = [(x["failure_rate"], x["mtbf_emp"], x["mtbf_ci_low"], x["mtbf_ci_high"]) for x in r if x["failures"]]
            if em:
                ax[0].errorbar([e[0] for e in em], [e[1] for e in em],
                               yerr=[[e[1] - e[2] for e in em], [min(e[3], 1e15) - e[1] for e in em]],
                               fmt="o", ms=4, color=cols[name], capsize=2, label=name)
            lb = [(x["failure_rate"], x["mtbf_ci_low"]) for x in r if not x["failures"]]
            if lb:
                ax[0].plot([e[0] for e in lb], [e[1] for e in lb], "^", color=cols[name], mfc="none", ms=6)
            ax[1].plot(lam, [1 - x["avail_analytic"] for x in r], "-", color=cols[name], lw=1)
            ax[1].plot(lam, [max(1 - x["avail_emp"], 1e-12) for x in r], "o", ms=4, color=cols[name], label=name)
            ax[2].plot(lam, [x["rel_analytic"] for x in r], "-", color=cols[name], lw=1)
            ax[2].plot(lam, [x["rel_emp"] for x in r], "o", ms=4, color=cols[name], label=name)
        for a_, t in zip(ax, ("MTBF (h): points = simulation (95 % CI), line = discrete\nformula, dotted = textbook, triangle = lower bound",
                              "Unavailability 1 - A", "Mission reliability R(1000 h)")):
            a_.set_xscale("log")
            a_.set_xlabel("per-module failure rate (per hour)")
            a_.set_title(t, fontsize=8.5)
            a_.grid(alpha=0.3)
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[1].set_ylim(1e-12, 1)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", "fig_sweep.png"), dpi=140)
    except ImportError:
        pass
    open(os.path.join(HERE, "results_5_2.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
