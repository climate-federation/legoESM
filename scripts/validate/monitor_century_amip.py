"""Monitor a chained century AMIP run: health, forcing-era switches, CMOR flush.

The century chain (``scripts/cluster/levante/amip_mpas_gpu_chain.sbatch`` with
``CENTURY_DECK=1``) self-chains ~35 twelve-hour links.  Each link exit flushes
COMPLETED CMOR months (``_maybe_wallclock_exit`` ->
``flush_cmip_monthly(write=True)``), so the run is scoreable as it goes; the
terminal writer only fires at the final day.

This reports, from the run directory alone (no model import, no GPU):

* simulated day / year reached, wall-clock throughput per link,
* the day-line health series (mass-weighted mean T, CWV, |u|max, T range),
* drift check: mean T and CWV trend over the last N years,
* which forcing era each link selected (the ``CENTURY_DECK`` banner),
* CMOR flush inventory (variables x months written so far),
* volcano watch: whether the run has passed Agung/El Chichon/Pinatubo.

Usage
-----
    python scripts/validate/monitor_century_amip.py <run_dir> [--start-year 1923]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Major stratospheric eruptions in the CMIP6 volcanic deck, for the watch list.
VOLCANOES = {"Agung": 1963, "El Chichon": 1982, "Pinatubo": 1991}
_RESUME_RE = re.compile(r"Resuming from .*?checkpoint_day_0*(\d+)\.npz")
_DAY_RE = re.compile(
    r"^\s*Day\s+([\d.]+):\s*T=\[([\d.]+),([\d.]+)\]K\s+mean=([\d.]+)K"
    r".*?\|u\|_max=([\d.]+)m/s\s+CWV=([\d.]+)")


def parse_day_lines(run_dir: Path):
    """(day, T_min, T_max, T_mean, u_max, CWV) from every link log, ordered.

    Each link restarts its in-link day counter at 1, so the absolute day is
    ``resume_day + in_link_day``.  The resume day is read from the launcher's
    own ``Resuming from ... checkpoint_day_NNNN.npz`` line rather than
    accumulated across links: a link that RESTARTS FROM AN EARLIER CHECKPOINT
    (what happens after a blow-up — the century resumed at day 180 after
    dying at 182) would otherwise be added on top of the failed attempt and
    the run would appear further along than it is.  A cumulative stitch
    reported day 204 for a run genuinely at 182.
    """
    rows = []
    for log in sorted(run_dir.glob("slurm-*.out"),
                      key=lambda p: p.stat().st_mtime):
        text = log.read_text(errors="replace")
        m_resume = _RESUME_RE.search(text)
        base = float(m_resume.group(1)) if m_resume else 0.0
        for line in text.splitlines():
            m = _DAY_RE.match(line)
            if m:
                vals = [float(g) for g in m.groups()]
                vals[0] += base                # absolute day
                rows.append(tuple(vals))
    if not rows:
        return np.empty((0, 6))
    arr = np.array(rows)
    # Order by absolute day; a re-run of an already-simulated stretch
    # (post-blow-up restart) legitimately repeats days, so keep the LAST
    # value seen for each day rather than assuming monotonicity.
    order = np.argsort(arr[:, 0], kind="stable")
    return arr[order]


def era_switches(run_dir: Path):
    out = []
    for log in sorted(run_dir.glob("slurm-*.out")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("CENTURY_DECK:"):
                out.append(line.strip())
    return out


def cmor_inventory(run_dir: Path):
    cmor = run_dir / "cmor"
    if not cmor.is_dir():
        return {}
    inv = {}
    for table in sorted(p for p in cmor.iterdir() if p.is_dir()):
        files = sorted(table.glob("*.nc"))
        inv[table.name] = [f.name.split("_")[0] for f in files]
    return inv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--start-year", type=int, default=1923)
    ap.add_argument("--trend-years", type=float, default=5.0,
                    help="window for the drift check [simulated years]")
    args = ap.parse_args(argv)

    run = args.run_dir
    if not run.is_dir():
        raise SystemExit(f"no such run directory: {run}")

    arr = parse_day_lines(run)
    if arr.size == 0:
        print(f"{run.name}: no day lines yet")
        return 0
    day, t_min, t_max, t_mean, u_max, cwv = arr.T
    last = day[-1]
    year = args.start_year + int(last // 365)
    print(f"=== {run.name}: day {last:.0f}/36500  (simulated year {year}, "
          f"{last / 365.0:.2f} yr of 100)")
    print(f"    latest: mean T {t_mean[-1]:.2f} K | CWV {cwv[-1]:.1f} kg/m2 | "
          f"T [{t_min[-1]:.1f}, {t_max[-1]:.1f}] | |u|max {u_max[-1]:.1f} m/s")

    # Drift: compare the trailing window against the one before it.
    win = args.trend_years * 365.0
    if last > 2 * win:
        recent = day > last - win
        prior = (day > last - 2 * win) & (day <= last - win)
        print(f"    drift (last {args.trend_years:g} yr vs previous): "
              f"mean T {t_mean[recent].mean() - t_mean[prior].mean():+.3f} K, "
              f"CWV {cwv[recent].mean() - cwv[prior].mean():+.3f} kg/m2")
    else:
        print(f"    drift: needs > {2 * args.trend_years:g} yr, have "
              f"{last / 365.0:.2f}")

    # Stability guards — the historic failure signatures of this campaign.
    if not np.all(np.isfinite(arr)):
        print("    !! NON-FINITE day-line entry — blow-up")
    if t_min.min() < 150.0 or t_max.max() > 340.0:
        print(f"    !! T excursion: min {t_min.min():.1f} K, "
              f"max {t_max.max():.1f} K (sane band 150-340)")

    eras = era_switches(run)
    if eras:
        print(f"    forcing eras ({len(eras)} link starts):")
        for line in eras[-3:]:
            print(f"      {line}")
    passed = [f"{n} ({y})" for n, y in VOLCANOES.items() if year > y]
    upcoming = [(n, y) for n, y in VOLCANOES.items() if year <= y]
    if passed:
        print(f"    volcanoes simulated: {', '.join(passed)}")
    if upcoming:
        n, y = min(upcoming, key=lambda kv: kv[1])
        print(f"    next volcano: {n} {y} at ~day "
              f"{(y - args.start_year) * 365:.0f}")

    inv = cmor_inventory(run)
    if inv:
        for table, vars_ in inv.items():
            print(f"    CMOR {table}: {len(vars_)} files "
                  f"({', '.join(sorted(set(vars_))[:8])}...)")
    else:
        print("    CMOR: nothing flushed yet (first flush at the first link "
              "exit; the terminal write is day 36500)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
