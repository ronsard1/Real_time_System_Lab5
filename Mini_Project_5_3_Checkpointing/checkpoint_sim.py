"""
checkpoint_sim.py - checkpoint and rollback recovery for a long-running real-time task (Mini Project 5.3).
Standard library only; matplotlib optional for the figure.

Hypothetical mission (used for the recommendation memo): an on-board data-fusion job needs W = 28 800 s
(8 h) of computation and must finish within a deadline of 36 000 s (10 h). A checkpoint costs C = 30 s; after a
fault the task restores the last checkpoint in R = 60 s. Faults (radiation-induced upsets, software crashes)
arrive with a per-second probability of 1/MTBF.

run_with_checkpointing(I): every I seconds of progress the task pauses for C and records a checkpoint. A fault
during work or during a checkpoint rolls progress back to the last completed checkpoint and costs R; a fault
during recovery restarts the recovery. Time to the next fault is drawn as a geometric number of seconds, which
is exactly the per-second draw of the procedure (memoryless), so long intervals are simulated quickly.

    python checkpoint_sim.py     -> results_5_3.txt, completion_times.csv, figures/
"""
import csv
import math
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
W, DEADLINE, C, R = 28_800, 36_000, 30, 60
MTBFS = [1_800, 7_200, 28_800]
INTERVALS = sorted({int(round(30 * (28_800 / 30) ** (i / 24))) for i in range(25)})
TRIALS = 2_000
CAP = 20 * W                        # a run still unfinished after 20 x W is stopped and counted as a failure
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def time_to_fault(rng, p):
    return int(math.log(1.0 - rng.random()) / math.log(1.0 - p)) + 1


def run_with_checkpointing(interval, mtbf, rng, overhead=C, recovery=R):
    p = 1.0 / mtbf
    t, progress, faults = 0, 0, 0
    while progress < W:
        work = min(interval, W - progress)
        seg = work + (overhead if progress + work < W else 0)   # no checkpoint needed after the last segment
        f = time_to_fault(rng, p)
        if f > seg:                                             # segment and its checkpoint completed
            t += seg
            progress += work
        else:                                                   # fault: lose the segment, roll back, recover
            t += f
            faults += 1
            while True:
                g = time_to_fault(rng, p)
                if g > recovery:
                    t += recovery
                    break
                t += g                                          # fault during recovery: start it again
                faults += 1
        if t > CAP:
            return CAP, faults, False
    return t, faults, True


# ---------------------------------------------------------------- analytical approximations
def young(mtbf, c=C):
    """First order: T(I) ~ W + (W/I) C + (W/M)(I/2 + R); dT/dI = 0 gives I* = sqrt(2 C M)."""
    return math.sqrt(2 * c * mtbf)


def first_order_total(interval, mtbf):
    return W + (W / interval) * C + (W / mtbf) * (interval / 2 + R)


def exact_total(interval, mtbf):
    """Expected time for exponential faults that can also strike checkpoints and recoveries
    (renewal argument per segment): E[T] = (W/I) M e^(R/M) (e^((I+C)/M) - 1)."""
    return (W / interval) * mtbf * math.exp(R / mtbf) * (math.exp((interval + C) / mtbf) - 1)


def argmin(f, lo=1.0, hi=float(W)):
    a, b = math.log(lo), math.log(hi)
    for _ in range(200):                                        # golden-section search on log(I)
        m1, m2 = a + (b - a) * 0.382, a + (b - a) * 0.618
        if f(math.exp(m1)) < f(math.exp(m2)):
            b = m2
        else:
            a = m1
    return math.exp((a + b) / 2)


def main():
    rng = random.Random(55)
    say(f"Mission: W = {W:,} s of work, deadline {DEADLINE:,} s, checkpoint overhead C = {C} s, recovery R = {R} s; "
        f"{TRIALS} trials per point; {len(INTERVALS)} intervals from {INTERVALS[0]} s to {INTERVALS[-1]:,} s (= no checkpoint)")
    rows, summary = [], []
    for M in MTBFS:
        say("=" * 118)
        say(f"MTBF = {M:,} s ({M / 3600:g} h)")
        say(f"{'interval s':>11s}{'mean T s':>11s}{'95% CI':>18s}{'median':>9s}{'p95':>9s}{'P(T<=D)':>9s}{'faults':>8s}"
            f"{'unfinished':>11s}{'1st-order':>11s}{'exact model':>13s}")
        best = None
        for I in INTERVALS:
            res = [run_with_checkpointing(I, M, rng) for _ in range(TRIALS)]
            ts = [r[0] for r in res]
            mean = statistics.mean(ts)
            half = 1.96 * statistics.stdev(ts) / math.sqrt(TRIALS)
            q = statistics.quantiles(ts, n=20)
            unfinished = sum(1 for r in res if not r[2])
            ok = sum(1 for t in ts if t <= DEADLINE) / TRIALS
            fo, ex = first_order_total(I, M), exact_total(I, M)
            rows.append(dict(mtbf=M, interval=I, mean=mean, ci_low=mean - half, ci_high=mean + half, median=q[9], p95=q[18],
                             p_deadline=ok, mean_faults=statistics.mean(r[1] for r in res), unfinished=unfinished,
                             first_order=fo, exact=ex))
            if unfinished == 0 and (best is None or mean < best[1]):
                best = (I, mean, half)
            ex_s = f"{ex:>13,.0f}" if ex < 1e9 else f"{'> 1e9':>13s}"
            say(f"{I:>11,d}{mean:>11,.0f}{f'+/- {half:,.0f}':>18s}{q[9]:>9,.0f}{q[18]:>9,.0f}{ok:>9.3f}"
                f"{statistics.mean(r[1] for r in res):>8.1f}{unfinished:>11d}{fo:>11,.0f}{ex_s}")
        flat = [r["interval"] for r in rows if r["mtbf"] == M and r["unfinished"] == 0 and r["mean"] <= best[1] * 1.01]
        iy = young(M)
        ie = argmin(lambda i: exact_total(i, M))
        summary.append((M, best, (min(flat), max(flat)), iy, ie, exact_total(ie, M)))
        say(f"  empirical optimum {best[0]:,} s (mean {best[1]:,.0f} +/- {best[2]:,.0f} s); within 1 % of it: "
            f"{min(flat):,} - {max(flat):,} s")
        say(f"  Young sqrt(2CM) = {iy:,.0f} s;  exact-model optimum = {ie:,.0f} s (E[T] = {exact_total(ie, M):,.0f} s)")

    say("=" * 118)
    say("Shift of the optimum with MTBF:")
    say(f"{'MTBF s':>9s}{'empirical opt':>15s}{'1 % band':>18s}{'Young':>9s}{'exact model':>13s}{'E[T] at opt':>13s}"
        f"{'P(T<=D) at opt':>16s}")
    for M, best, band, iy, ie, et in summary:
        pd = next(r["p_deadline"] for r in rows if r["mtbf"] == M and r["interval"] == best[0])
        say(f"{M:>9,d}{best[0]:>15,d}{f'{band[0]:,} - {band[1]:,}':>18s}{iy:>9,.0f}{ie:>13,.0f}{et:>13,.0f}{pd:>16.3f}")

    with open(os.path.join(HERE, "completion_times.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        cols = {1_800: "#d62728", 7_200: "#1f77b4", 28_800: "#2ca02c"}
        for M, best, band, iy, ie, et in summary:
            r = [x for x in rows if x["mtbf"] == M and x["unfinished"] == 0]
            ax.plot([x["interval"] for x in r], [x["mean"] / 3600 for x in r], "o", ms=3.5, color=cols[M],
                    label=f"MTBF {M / 3600:g} h (simulated)")
            grid = [10 ** (math.log10(30) + i * (math.log10(28_800) - math.log10(30)) / 200) for i in range(201)]
            g = [(i, exact_total(i, M) / 3600) for i in grid if exact_total(i, M) < 1.6e5]
            ax.plot([a for a, _ in g], [b for _, b in g], "-", color=cols[M], lw=0.9)
            ax.axvline(iy, color=cols[M], ls=":", lw=0.9)
        ax.axhline(DEADLINE / 3600, color="black", ls="--", lw=0.8)
        ax.text(32, DEADLINE / 3600 + 0.15, "deadline 10 h", fontsize=7)
        ax.axhline(W / 3600, color="grey", ls="-", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylim(7.5, 16)
        ax.set_xlabel("checkpoint interval (s, log)")
        ax.set_ylabel("mean completion time (h)")
        ax.set_title("Completion time vs checkpoint interval: points = simulation, line = exact model, "
                     "dotted = Young", fontsize=8.5)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "fig_completion_vs_interval.png"), dpi=140)
    except ImportError:
        pass
    open(os.path.join(HERE, "results_5_3.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
