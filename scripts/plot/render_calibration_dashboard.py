#!/usr/bin/env python
"""Render (or re-render) the calibration dashboard from its JSONL log.

The live campaign calls :meth:`CalibrationTracker.end_iteration` itself; this
script is for everything else:

* rebuilding the page after a job was killed (the JSONL survives, so the page
  can always be reconstructed);
* watching a campaign whose driver is not calling the tracker
  (``--watch 300`` re-renders every five minutes);
* ``--demo``, which writes a SYNTHETIC campaign exhibiting every pathology the
  page exists to catch — a parameter pinned at its bound, an iteration where
  half the ensemble blew up, a loss that turns back upward, and one statistic
  dominating the fit. Look at that page before trusting the real one: a
  dashboard that has only ever been shown healthy data is not a dashboard
  anyone should trust.

Examples::

    python scripts/plot/render_calibration_dashboard.py results/calib_stage3
    python scripts/plot/render_calibration_dashboard.py results/calib_stage3 \
        --from-registry --watch 300
    python scripts/plot/render_calibration_dashboard.py /tmp/demo --demo

No JAX unless ``--from-registry`` is passed (the registry import pulls it in),
so this runs on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

from legoesm.training.calibration_tracking import (
    DEFAULT_PIN_THRESHOLD,
    MEMBERS_FILENAME,
    CalibrationTracker,
    MemberRecord,
    StatisticRecord,
    render_dashboard,
    resolve_param_bounds,
)

# --- demo campaign ----------------------------------------------------------
# Bounds are the REAL registered ones from
# ``legoesm.atmosphere.physics.clouds.config.__param_spec__`` so the demo page
# has the same geometry as a live one; they are inlined rather than imported so
# --demo needs no JAX.
DEMO_BOUNDS: dict[str, tuple[float, float]] = {
    "atm.clouds.CloudConfig.rh_crit": (0.5, 0.99),
    "atm.clouds.CloudConfig.alpha_xr": (10.0, 1000.0),
    "atm.clouds.CloudConfig.p_xr": (0.05, 1.0),
    "atm.clouds.CloudConfig.gamma_xr": (0.1, 1.0),
    "atm.clouds.CloudConfig.q_c_diagnostic": (5.0e-5, 1.5e-3),
}

# (observed value, structural sigma, member scatter, fitted?) — sigmas as
# ``observation_error.structural_sigma`` would produce them (5% of the
# reference; tas absolute 0.7 K; net_toa excluded from the fit).
#
# Member scatter is in the statistic's OWN UNITS, never a fraction of its
# value: a member-to-member spread of "2% of 287 K" would be 5.7 K, which is
# the precise mistake ``observation_error`` exists to forbid (a fraction of a
# quantity with an arbitrary zero is not an error scale). These are plausible
# member-to-member spreads of a global annual mean.
DEMO_STATISTICS: dict[str, tuple[float, float, float, bool]] = {
    "rsut": (98.856, 4.943, 2.00, True),
    "rlut": (240.513, 12.026, 1.50, True),
    "pr": (3.086, 0.154, 0.06, True),
    "prw": (24.273, 1.214, 0.35, True),
    "tas": (287.0, 0.700, 0.25, True),  # const-ok: observed tas [K], not R_d
    "net_toa": (0.9, float("nan"), 1.20, False),
}


def build_demo_campaign(run_dir: str, *, n_iterations: int = 6,
                        n_members: int = 20, seed: int = 20260807,
                        force: bool = False) -> str:
    """Write a synthetic but pathological campaign; returns the run dir.

    REFUSES to write into a directory that already holds a campaign, unless
    ``force``. The log is append-only and the summary keeps the LAST record per
    ``(iteration, member)``, so synthetic members do not merely pad a real
    campaign — they REPLACE its members at the same indices, and every
    rendering of a multi-day run silently becomes fiction. One mistyped run
    directory is all it takes.

    The story it tells, deliberately: ``rsut`` carries a bias no cloud
    parameter can close, so the optimiser pushes ``rh_crit`` up against its
    0.99 ceiling and jams there; the ensemble strays into an unstable corner
    at iteration 3 and half the members blow up; and the loss stops improving
    and turns back upward at the end. Every one of those is a "stop spending"
    signal, and the page has to make each one obvious.
    """
    existing = os.path.join(run_dir, MEMBERS_FILENAME)
    if os.path.exists(existing) and not force:
        raise SystemExit(
            f"refusing --demo: {existing} already holds a campaign. Synthetic "
            "members would overwrite the real ones at the same (iteration, "
            "member) indices and falsify every rendering. Pick an empty "
            "directory, or pass --force-demo if you really mean to.")
    rng = random.Random(seed)
    tracker = CalibrationTracker(
        run_dir, param_bounds=DEMO_BOUNDS,
        title="Calibration campaign (SYNTHETIC DEMO)",
        run_label="Synthetic data — every pathology on purpose. Not a real run.")

    # Parameter ensemble means per iteration. rh_crit walks to its ceiling and
    # sticks; the others settle mid-range.
    rh_crit = [0.72, 0.80, 0.87, 0.93, 0.972, 0.983]
    alpha_xr = [400.0, 470.0, 520.0, 545.0, 552.0, 556.0]
    p_xr = [0.40, 0.36, 0.33, 0.32, 0.315, 0.313]
    gamma_xr = [0.55, 0.52, 0.50, 0.495, 0.492, 0.491]
    q_c = [6.0e-4, 5.4e-4, 5.0e-4, 4.85e-4, 4.80e-4, 4.78e-4]

    # Model statistics per iteration: everything converges onto the reference
    # except rsut, which stalls ~19 W/m2 high — the structurally unreachable
    # bias that drives the pinning.
    # The last entry of each path is WORSE than the second-to-last: the loss
    # turns back upward, which is the other "stop spending" signal.
    model_paths = {
        "rsut": [138.0, 128.0, 122.0, 119.5, 118.4, 119.9],
        "rlut": [232.0, 235.5, 238.0, 239.6, 240.2, 241.2],
        "pr": [3.55, 3.40, 3.25, 3.16, 3.12, 3.19],
        "prw": [27.9, 26.6, 25.4, 24.8, 24.5, 24.9],
        "tas": [289.6, 288.9, 288.1, 287.6, 287.3, 287.6],
        "net_toa": [-14.2, -9.8, -6.1, -4.0, -3.2, -4.1],
    }
    # Shrinking member scatter — convergence — that widens again at the end,
    # which is what "the optimiser is thrashing" looks like.
    scatter = [1.00, 0.72, 0.50, 0.36, 0.30, 0.44]

    for iteration in range(n_iterations):
        for member in range(n_members):
            jitter = scatter[iteration]

            def draw(mean: float, rel: float, jitter: float = jitter) -> float:
                return mean * (1.0 + rel * jitter * rng.gauss(0.0, 1.0))

            parameters = {
                "atm.clouds.CloudConfig.rh_crit":
                    min(draw(rh_crit[iteration], 0.035), 0.99),
                "atm.clouds.CloudConfig.alpha_xr":
                    draw(alpha_xr[iteration], 0.16),
                "atm.clouds.CloudConfig.p_xr": draw(p_xr[iteration], 0.14),
                "atm.clouds.CloudConfig.gamma_xr":
                    draw(gamma_xr[iteration], 0.12),
                "atm.clouds.CloudConfig.q_c_diagnostic":
                    draw(q_c[iteration], 0.18),
            }

            # Iteration 3 walks into an unstable corner: half the ensemble dies.
            blew_up = (iteration == 3 and member % 2 == 0)
            if blew_up:
                tracker.log_member(MemberRecord(
                    iteration=iteration, member=member, parameters=parameters,
                    loss=float("nan"), statistics=(), blew_up=True,
                    timestamp=1.0e9 + 86400.0 * iteration + 60.0 * member,
                    note="simulation produced non-finite state"))
                continue

            stats, loss = [], 0.0
            for name, (observed, sigma, spread,
                       fitted) in DEMO_STATISTICS.items():
                model = (model_paths[name][iteration]
                         + spread * jitter * rng.gauss(0.0, 1.0))
                if fitted:
                    contribution = ((model - observed) / sigma) ** 2
                    loss += contribution
                else:
                    contribution = float("nan")
                stats.append(StatisticRecord(
                    name=name, model=model, observed=observed,
                    contribution=contribution, fitted=fitted))
            tracker.log_member(MemberRecord(
                iteration=iteration, member=member, parameters=parameters,
                loss=loss, statistics=tuple(stats),
                timestamp=1.0e9 + 86400.0 * iteration + 60.0 * member))
    tracker.end_iteration()
    if tracker.errors:
        raise RuntimeError(
            f"{tracker.errors} tracking error(s) while building the demo; see "
            "tracking_errors.log — a demo that silently swallowed its own "
            "failures would prove nothing")
    return run_dir


def load_bounds(path: str) -> dict[str, tuple[float, float]]:
    """Read ``{"param": [lo, hi]}`` from a JSON file."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    bounds: dict[str, tuple[float, float]] = {}
    for name, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"bounds for {name!r} must be a [lo, hi] pair, got {pair!r}")
        low, high = float(pair[0]), float(pair[1])
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            raise ValueError(
                f"bounds for {name!r} must be finite with hi > lo, got "
                f"[{low}, {high}]")
        bounds[str(name)] = (low, high)
    return bounds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="campaign directory holding "
                                        f"{MEMBERS_FILENAME}")
    parser.add_argument("--title", default="Calibration campaign",
                        help="page title")
    parser.add_argument("--run-label", default="",
                        help="one-line description shown under the title")
    parser.add_argument("--bounds-json", default=None,
                        help='JSON file of {"param": [lo, hi]} bounds')
    parser.add_argument("--from-registry", action="store_true",
                        help="resolve bounds from the __param_spec__ registry "
                             "(imports JAX)")
    parser.add_argument("--pin-threshold", type=float,
                        default=DEFAULT_PIN_THRESHOLD,
                        help="fraction of the bound-to-bound range within "
                             "which a parameter counts as pinned "
                             f"(default {DEFAULT_PIN_THRESHOLD})")
    parser.add_argument("--no-events", action="store_true",
                        help="skip the TensorBoard event files")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                        help="re-render on this interval until interrupted")
    parser.add_argument("--demo", action="store_true",
                        help="write a synthetic pathological campaign into "
                             "run_dir first, then render it (refuses if the "
                             "directory already holds a campaign)")
    parser.add_argument("--force-demo", action="store_true",
                        help="allow --demo to write into a directory that "
                             "already holds a campaign (it will overwrite "
                             "members at the same indices)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.pin_threshold < 0.0 or args.pin_threshold >= 0.5:
        raise SystemExit(
            f"--pin-threshold must be in [0, 0.5), got {args.pin_threshold}")
    if args.watch is not None and args.watch <= 0.0:
        raise SystemExit(f"--watch must be positive, got {args.watch}")

    bounds: dict[str, tuple[float, float]] = {}
    if args.bounds_json:
        bounds.update(load_bounds(args.bounds_json))
    if args.from_registry:
        bounds.update(resolve_param_bounds())

    if args.demo:
        if args.bounds_json or args.from_registry:
            print("[render_calibration_dashboard] --demo supplies its own "
                  "bounds; ignoring the ones given", file=sys.stderr)
        bounds = dict(DEMO_BOUNDS)
        build_demo_campaign(args.run_dir, force=args.force_demo)

    jsonl = os.path.join(args.run_dir, MEMBERS_FILENAME)
    if not os.path.exists(jsonl):
        print(f"[render_calibration_dashboard] no {jsonl} yet — rendering an "
              "empty page", file=sys.stderr)

    try:
        while True:
            path = render_dashboard(
                args.run_dir, bounds, title=args.title,
                run_label=args.run_label, pin_threshold=args.pin_threshold,
                write_events=not args.no_events)
            print(path)
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        # Ctrl-C anywhere in the loop, not only inside the sleep.
        print("[render_calibration_dashboard] interrupted", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
