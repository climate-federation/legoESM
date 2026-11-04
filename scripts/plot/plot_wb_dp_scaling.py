#!/usr/bin/env python
"""WeatherBench data-parallel GPU ladder: s/step and efficiency per arm and lead.

Reads the trainer's timing logs (``logs/wbsurface/timing_<job>_<id>.out``):
arm and lead from the job name (``..._sfno..._72h...`` / ``..._ngcm..._6h...``),
GPU count from the launcher's ``[wb_dp] ... ranks=N`` line (1 when absent, the
single-GPU jobs).  s/step = median of the per-step times (slowest rank when
ranks each log the step) after the first three (compile + warm-up).  One
sample per GPU per step, so ideal scaling is a constant s/step and
efficiency = t(1 GPU) / t(N GPUs).

    python scripts/plot/plot_wb_dp_scaling.py --logs 'logs/wbsurface/timing_t9*.out' \
        --out logs/wbsurface/scaling/wb_dp_scaling.png
"""
from __future__ import annotations

import argparse
import glob
import re
import statistics
from pathlib import Path

# Okabe-Ito (no red-green pair)
LEAD_COLOR = {6: "#0072B2", 12: "#009E73", 24: "#E69F00", 72: "#CC79A7"}
ARMS = ("ngcm", "sfno")
_NAME = re.compile(r"timing_[^_]+_(ngcm|sfno)[a-z0-9]*_(\d+)h")
_STEP = re.compile(r"epoch (\d+) step (\d+)/\d+ loss=\S+ ([0-9.]+) s")
_RANKS = re.compile(r"\[wb_dp\].* ranks=(\d+)")
N_WARM = 3


def parse(path):
    """``(arm, lead_h, gpus, s_per_step, n_steps)`` or None (no steady step)."""
    m = _NAME.search(Path(path).name)
    if m is None:
        return None
    text = Path(path).read_text(errors="replace")
    # Ranks may each log a step: keep one time per (epoch, step), the slowest
    # (the step ends when the last rank does).
    per = {}
    for e, s, t in _STEP.findall(text):
        per[(int(e), int(s))] = max(per.get((int(e), int(s)), 0.0), float(t))
    steps = [per[k] for k in sorted(per)]
    if len(steps) <= N_WARM:
        return None
    r = _RANKS.search(text)
    gpus = int(r.group(1)) if r else 1
    return m.group(1), int(m.group(2)), gpus, statistics.median(steps[N_WARM:]), len(steps)


def collect(pattern, exclude=()):
    pts = {}
    for p in sorted(glob.glob(pattern)):
        if any(e in Path(p).name for e in exclude):
            continue
        r = parse(p)
        if r is not None:
            pts[r[:3]] = r[3]          # a later job of the same point wins
    return pts


def make_figure(pts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(ARMS), figsize=(5.5 * len(ARMS), 7.5),
                             squeeze=False)
    for j, arm in enumerate(ARMS):
        top, bot = axes[0, j], axes[1, j]
        for lead in sorted({k[1] for k in pts if k[0] == arm}):
            curve = sorted((g, t) for (a, h, g), t in pts.items()
                           if a == arm and h == lead)
            g, t = zip(*curve)
            c = LEAD_COLOR.get(lead, "#000000")
            top.plot(g, t, "o-", color=c, label=f"{lead} h")
            base = dict(curve).get(1)
            if base is not None:
                top.plot(g, [base] * len(g), "--", color=c, lw=0.8)
                bot.plot(g, [base / x for x in t], "o-", color=c, label=f"{lead} h")
        bot.axhline(1.0, ls="--", color="#000000", lw=0.8, label="ideal")
        for ax in (top, bot):
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
            ax.set_xlabel("GPUs (one sample each per step)")
        top.set_yscale("log")
        top.set_ylabel("s per training step")
        top.set_title(f"{arm}: step time (dashed = ideal, 1-GPU value)")
        bot.set_ylabel("parallel efficiency  t(1)/t(N)")
        bot.set_ylim(0, 1.1)
        top.legend(title="lead")
        bot.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", default="logs/wbsurface/timing_t9*.out")
    p.add_argument("--exclude", nargs="*", default=(),
                   help="substrings of log names to skip (superseded configs)")
    p.add_argument("--out", default="logs/wbsurface/scaling/wb_dp_scaling.png")
    a = p.parse_args(argv)
    pts = collect(a.logs, a.exclude)
    if not pts:
        raise SystemExit(f"no timing points in {a.logs}")
    for (arm, lead, g), t in sorted(pts.items()):
        print(f"POINT {arm} {lead}h {g} GPU {t:.1f} s/step")
    print(make_figure(pts, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
