"""
reliability_sim.py - TMR voting, reliability metrics and watchdog-based recovery (Mini Project 5.1).
Standard library only (random, statistics, math); matplotlib optional for one summary figure.

Model (hour steps, as in the procedure). Every hour an UP module draws a uniform number; below failure_rate
it fails. A transient fault (e.g. single-event upset) clears on the next hour; a permanent fault keeps the
module down for repair_delay hours. Simplex = one module. TMR = three independently seeded modules and a
majority voter: the system is down in any hour in which two or more modules are down.

    python reliability_sim.py        -> results_5_1.txt (+ figures/fig_5_1.png if matplotlib is installed)
"""
import math
import os
import random
import statistics

random.seed(42)                       # step 1: reproducible
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []
HOURS, TRIALS = 200_000, 30
LAMBDA = 0.001                        # per hour
REPAIR = 24                           # hours for a permanent fault


def say(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- step 2
def simulate_module(hours, failure_rate, fault_type, repair_delay=REPAIR, rng=random):
    """Return a list of booleans, True = module down in that hour. Transient and permanent faults are distinct."""
    down = [False] * hours
    remaining = 0
    for h in range(hours):
        if remaining > 0:                                # still down from an earlier fault
            down[h] = True
            remaining -= 1
            continue
        if rng.random() < failure_rate:
            down[h] = True
            if fault_type == "transient":
                remaining = 0                            # self-clearing: up again next hour
            elif fault_type == "permanent":
                remaining = repair_delay - 1             # stays down until repaired
            else:
                raise ValueError(fault_type)
    return down


def stats_of(down):
    """Failures = up->down transitions. Returns failures, downtime, MTBF (two definitions), availability."""
    fails = sum(1 for i, d in enumerate(down) if d and (i == 0 or not down[i - 1]))
    dt = sum(down)
    n = len(down)
    return dict(fails=fails, down=dt, mtbf_total=n / fails if fails else float("inf"),
                mtbf_up=(n - dt) / fails if fails else float("inf"), avail=(n - dt) / n)


# ---------------------------------------------------------------- steps 3-4
def simplex(hours, lam, ft, rng):
    return simulate_module(hours, lam, ft, rng=rng)


def tmr(hours, rates, ft, rng_seed, voter=None):
    mods = [simulate_module(hours, r, ft, rng=random.Random(rng_seed * 10 + i)) for i, r in enumerate(rates)]
    sysdown = [(a + b + c) >= 2 for a, b, c in zip(*mods)]
    if voter:                                            # lab task 2: voter timing overhead
        vr = random.Random(rng_seed * 7 + 99)
        sysdown = [d or voter(vr) for d in sysdown]
    return sysdown


# ---------------------------------------------------------------- step 6: analytical values
def q_down(lam, ft):
    m = 1 if ft == "transient" else REPAIR                # mean down time per fault
    return lam * m / (1 + lam * m)


def analytic_simplex(lam, ft):
    m = 1 if ft == "transient" else REPAIR
    return dict(mtbf=1 / lam, mttr=m, avail=(1 / lam) / (1 / lam + m))


def analytic_tmr(rates, ft):
    q = [q_down(l, ft) for l in rates]
    a, b, c = q
    p2 = a * b * (1 - c) + a * c * (1 - b) + b * c * (1 - a) + a * b * c     # P(two or more down)
    if ft == "transient":
        rate = p2                                         # each 2-of-3 hour is a new, 1-hour failure
    else:                                                 # one already down, one of the others fails
        rate = sum(q[i] * (1 - q[j]) * (1 - q[k]) * (rates[j] + rates[k])
                   for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)))
    return dict(p2=p2, mtbf=(1 - p2) / rate, avail=1 - p2)


def poisson_ci(k, conf=0.95):
    """Exact (Garwood) 95 % interval for a Poisson count k, by bisection on the Poisson CDF."""
    def cdf(n, mu):                                        # log-space, no underflow for large counts
        if mu <= 0:
            return 1.0
        return sum(math.exp(i * math.log(mu) - mu - math.lgamma(i + 1)) for i in range(n + 1))

    def solve(f, lo, hi):
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2
    a = (1 - conf) / 2
    lower = 0.0 if k == 0 else solve(lambda mu: (1 - cdf(k - 1, mu)) - a < 0 and 1 or -1, 0, 10 * k + 10)
    upper = solve(lambda mu: cdf(k, mu) - a > 0 and 1 or -1, 0, 10 * k + 20)
    return lower, upper


def transient_down_hours(hours, lam, rng):
    """Fast equivalent of simulate_module(..., 'transient'): every hour is independently down with probability
    lam, so the down hours can be generated by geometric gaps instead of 200 000 draws."""
    out, h = set(), -1
    while True:
        h += int(math.log(1 - rng.random()) / math.log(1 - lam)) + 1
        if h >= hours:
            return out
        out.add(h)


def run_trials(kind, ft, rates=None, voter=None):
    res = []
    for t in range(TRIALS):
        if kind == "simplex":
            d = simplex(HOURS, LAMBDA, ft, random.Random(1000 + t))
        else:
            d = tmr(HOURS, rates or [LAMBDA] * 3, ft, 5000 + t, voter)
        res.append(stats_of(d))
    return res


def summarise(res):
    fails = sum(r["fails"] for r in res)
    tot = HOURS * len(res)
    down = sum(r["down"] for r in res)
    per_trial = [r["mtbf_up"] for r in res if r["fails"]]
    return dict(fails=fails, mtbf_up=(tot - down) / fails if fails else float("inf"),
                mtbf_total=tot / fails if fails else float("inf"), avail=1 - down / tot,
                zero=sum(1 for r in res if r["fails"] == 0),
                spread=(min(per_trial), max(per_trial)) if per_trial else (float("inf"), float("inf")))


# ---------------------------------------------------------------- steps 7-9: watchdog
def simulate_watchdog(num_hangs, timeout, restart_time, period=1.0, rng=None):
    """Periodic task (heartbeat every `period` minutes). A hang at a random instant stops the heartbeat; the
    watchdog fires `timeout` minutes after the LAST heartbeat and forces a restart taking restart_time.
    Returns the MTTR samples (hang -> task running again)."""
    rng = rng or random.Random(7)
    samples = []
    for _ in range(num_hangs):
        since_last_beat = rng.uniform(0, period)          # hang happens part-way through a period
        detect = max(timeout - since_last_beat, 0)        # time from hang to watchdog expiry
        samples.append(detect + restart_time)
    return samples


def false_positive_rate(timeout, hours=2000, period=1.0, p_long=0.01, long_range=(2, 12), seed=3):
    """Lab task 3: a HEALTHY task occasionally runs long (legitimate heavy computation) and its next heartbeat
    is late. A restart triggered by such a gap is a false positive."""
    rng = random.Random(seed)
    beats = int(hours * 60 / period)
    fp = 0
    for _ in range(beats):
        gap = rng.uniform(*long_range) if rng.random() < p_long else period
        if gap > timeout:
            fp += 1
    return fp, fp / beats, fp / (hours / 24)


def main():
    say(f"Steps 5-6: {TRIALS} trials x {HOURS:,} h, failure rate {LAMBDA}/h per module, permanent-fault repair {REPAIR} h")
    say("=" * 112)
    say(f"{'config':24s}{'failures':>9s}{'MTBF up/fail':>14s}{'MTBF tot/fail':>15s}{'analytic MTBF':>15s}"
        f"{'avail emp':>11s}{'avail anal':>11s}{'trials 0 fail':>14s}")
    emp = {}
    for ft in ("transient", "permanent"):
        s = summarise(run_trials("simplex", ft))
        a = analytic_simplex(LAMBDA, ft)
        emp[("simplex", ft)] = s
        say(f"{'simplex, ' + ft:24s}{s['fails']:>9d}{s['mtbf_up']:>14,.0f}{s['mtbf_total']:>15,.0f}{a['mtbf']:>15,.0f}"
            f"{s['avail']:>11.6f}{a['avail']:>11.6f}{s['zero']:>14d}")
        s = summarise(run_trials("tmr", ft))
        a = analytic_tmr([LAMBDA] * 3, ft)
        emp[("tmr", ft)] = s
        say(f"{'TMR, ' + ft:24s}{s['fails']:>9d}{s['mtbf_up']:>14,.0f}{s['mtbf_total']:>15,.0f}{a['mtbf']:>15,.0f}"
            f"{s['avail']:>11.6f}{a['avail']:>11.6f}{s['zero']:>14d}")
        say(f"  {'':22s}TMR analytic P(>=2 of 3 down in an hour) = {a['p2']:.3e}; per-trial MTBF range "
            f"{s['spread'][0]:,.0f} - {s['spread'][1]:,.0f} h")
    say("  MTBF up/fail = uptime / failures (compare with 1/lambda); tot/fail = total hours / failures (includes repair time)")
    for key, label in ((("tmr", "transient"), "TMR transient"), (("tmr", "permanent"), "TMR permanent")):
        s = emp[key]
        lo, hi = poisson_ci(s["fails"])
        up = HOURS * TRIALS * s["avail"]
        say(f"  95 % CI on {label} MTBF from {s['fails']} failures: {up / hi:,.0f} - {up / lo:,.0f} h")
    say("  Convergence, TMR transient (fast equivalent generator, same model):")
    for n in (30, 100, 300, 1000):
        fails, up = 0, 0
        for t in range(n):
            r = random.Random(90000 + t)
            d = [transient_down_hours(HOURS, LAMBDA, r) for _ in range(3)]
            two = (d[0] & d[1]) | (d[0] & d[2]) | (d[1] & d[2])
            fails += len(two)                       # 1-hour episodes; adjacent hours are vanishingly rare
            up += HOURS - len(two)
        lo, hi = poisson_ci(fails)
        say(f"    {n:>5d} trials: {fails:>4d} failures, MTBF {up / fails:>9,.0f} h (95 % CI {up / hi:,.0f} - {up / lo:,.0f}); "
            f"analytic {analytic_tmr([LAMBDA] * 3, 'transient')['mtbf']:,.0f} h")
    fails, up = 0, 0
    for t in range(100):                               # hour-step model, fresh seeds
        s = stats_of(tmr(HOURS, [LAMBDA] * 3, "permanent", 700000 + t))
        fails += s["fails"]
        up += HOURS - s["down"]
    lo, hi = poisson_ci(fails)
    say(f"  TMR permanent, 100 further trials (hour steps, new seeds): {fails} failures, MTBF {up / fails:,.0f} h "
        f"(95 % CI {up / hi:,.0f} - {up / lo:,.0f}); analytic {analytic_tmr([LAMBDA] * 3, 'permanent')['mtbf']:,.0f} h")

    # ---- steps 7-9: watchdog, MTBF fixed at the empirical simplex permanent-fault value
    mtbf_h = emp[("simplex", "permanent")]["mtbf_up"]
    mtbf_min = mtbf_h * 60
    restart = 2.0                                         # minutes
    say("=" * 112)
    say(f"Steps 7-9: watchdog on a 1-min periodic task; MTBF fixed at the empirical simplex value {mtbf_h:,.0f} h; "
        f"restart {restart:.0f} min; 2000 hangs per setting")
    say(f"{'recovery':30s}{'mean MTTR (min)':>16s}{'availability':>14s}{'analytic A':>12s}{'downtime/year':>15s}")
    rng = random.Random(11)
    manual = [rng.expovariate(1 / 240) + restart for _ in range(2000)]     # operator notices after ~4 h on average
    rows = [("no watchdog (operator restart)", manual, 240 + restart)]
    for to in (1, 5, 10, 20):
        rows.append((f"watchdog timeout {to} min", simulate_watchdog(2000, to, restart, rng=random.Random(to)), to + restart))
    wd_rows = []
    for name, smp, target in rows:
        mttr = statistics.mean(smp)
        A = mtbf_min / (mtbf_min + mttr)
        A_an = (1 / LAMBDA * 60) / (1 / LAMBDA * 60 + target)
        wd_rows.append((name, mttr, A))
        say(f"{name:30s}{mttr:>16.2f}{A:>14.6f}{A_an:>12.6f}{(1 - A) * 525_960:>12.0f} min")
    say("  analytic A uses MTBF = 1/lambda and the target MTTR (timeout + restart). Expected detection delay is")
    say("  timeout - 0.5 min because the hang occurs on average half a period after the last heartbeat.")

    # ---- lab task 1: diverse modules
    say("=" * 112)
    say("Lab task 1: non-identical TMR modules (permanent faults); simplex reference uses the mean rate")
    say(f"{'module rates (per h)':30s}{'TMR fails':>10s}{'TMR MTBF emp':>14s}{'TMR MTBF anal':>15s}{'simplex MTBF':>14s}{'gain':>8s}")
    for rates in ([0.001] * 3, [0.0005, 0.001, 0.0015], [0.0002, 0.0008, 0.002], [0.001, 0.001, 0.01]):
        s = summarise(run_trials("tmr", "permanent", rates=rates))
        a = analytic_tmr(rates, "permanent")
        mean = sum(rates) / 3
        say(f"{str(rates):30s}{s['fails']:>10d}{s['mtbf_up']:>14,.0f}{a['mtbf']:>15,.0f}{1 / mean:>14,.0f}{s['mtbf_up'] * mean:>8.1f}x")

    # ---- lab task 2: voter overhead
    say("=" * 112)
    say("Lab task 2: voter overhead (permanent faults). Each hour the computation uses 90 % of its slot; the voter adds")
    say("an overhead. If computation + voter exceeds the slot the deadline is missed, which counts as a system failure.")
    voters = {"no voter cost": None,
              "bounded: fixed 5 % per hour": lambda r: 0.90 + 0.05 > 1.0,
              "unbounded: exponential, mean 2 %": lambda r: 0.90 + r.expovariate(1 / 0.02) > 1.0,
              "unbounded: exponential, mean 1 %": lambda r: 0.90 + r.expovariate(1 / 0.01) > 1.0}
    simplex_mtbf = emp[("simplex", "permanent")]["mtbf_up"]
    for name, v in voters.items():
        s = summarise(run_trials("tmr", "permanent", voter=v))
        say(f"  {name:34s} failures {s['fails']:>7d}  MTBF {s['mtbf_up']:>10,.0f} h  availability {s['avail']:.6f}"
            f"  ({'better' if s['mtbf_up'] > simplex_mtbf else 'WORSE'} than simplex {simplex_mtbf:,.0f} h)")

    # ---- lab task 3: false positives
    say("=" * 112)
    say("Lab task 3: watchdog false positives - healthy task, 1 % of periods run long (2-12 min) before the next heartbeat")
    for to in (5, 10, 20):
        fp, rate, per_day = false_positive_rate(to)
        say(f"  timeout {to:>2d} min: {fp:>5d} false restarts in 2000 h = {rate * 100:.3f} % of heartbeats, {per_day:.2f} per day;"
            f" true hangs detected: 100 %")

    # ---- lab task 4: availability versus MTTR
    say("=" * 112)
    say(f"Lab task 4: availability versus MTTR, MTBF fixed at {mtbf_h:,.0f} h")
    prev = None
    say(f"{'MTTR':>10s}{'availability':>14s}{'downtime/year':>16s}{'saved vs previous':>20s}")
    mttrs = [480, 240, 60, 20, 10, 5, 2, 1]
    for m in mttrs:
        A = mtbf_min / (mtbf_min + m)
        d = (1 - A) * 525_960
        say(f"{m:>7d} min{A:>14.6f}{d:>12.1f} min{'' if prev is None else f'{prev - d:>16.1f} min'}")
        prev = d

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        lab = ["simplex\ntransient", "TMR\ntransient", "simplex\npermanent", "TMR\npermanent"]
        vals = [emp[("simplex", "transient")]["mtbf_up"], emp[("tmr", "transient")]["mtbf_up"],
                emp[("simplex", "permanent")]["mtbf_up"], emp[("tmr", "permanent")]["mtbf_up"]]
        anv = [analytic_simplex(LAMBDA, "transient")["mtbf"], analytic_tmr([LAMBDA] * 3, "transient")["mtbf"],
               analytic_simplex(LAMBDA, "permanent")["mtbf"], analytic_tmr([LAMBDA] * 3, "permanent")["mtbf"]]
        x = range(4)
        ax[0].bar([i - 0.2 for i in x], vals, 0.4, label="empirical", color="#1f77b4")
        ax[0].bar([i + 0.2 for i in x], anv, 0.4, label="analytical", color="#ff7f0e")
        ax[0].set_yscale("log")
        ax[0].set_xticks(list(x))
        ax[0].set_xticklabels(lab, fontsize=8)
        ax[0].set_ylabel("MTBF (hours, log)")
        ax[0].set_title("Simplex vs TMR, 30 x 200 000 h", fontsize=9)
        ax[0].legend(fontsize=7)
        A = [mtbf_min / (mtbf_min + m) for m in mttrs]
        ax[1].plot(mttrs, [(1 - a) * 525_960 / 60 for a in A], "o-")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("MTTR (min, log)")
        ax[1].set_ylabel("downtime per year (h)")
        ax[1].set_title("MTTR lever, MTBF fixed", fontsize=9)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "fig_5_1.png"), dpi=140)
    except ImportError:
        pass
    open(os.path.join(HERE, "results_5_1.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
