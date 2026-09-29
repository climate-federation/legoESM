#!/usr/bin/env python
"""GATE: DINO's certified from-rest MONTH, scored against NEMO, must not drift.

WHY THIS EXISTS.  The GYRE NEMO-fidelity lane carried DINO's certified
from-rest month unmeasured for weeks.  Nothing in the landing battery ran
DINO from rest, so a change that moved the day-30 wet 3-D temperature
distance from NEMO from ``2.040e-03 K`` to ``6.982e-03 K`` landed, survived
two GitHub-main merges, and was only found when someone re-ran the year by
hand.  The owner was one recipe line, and the cost was a factor 3.42.  The
receipt is ``docs/ocean/fidelity/testcases/
nemo_testcases_l2_gyre_dino_month_regression_receipt.md``.

WHAT IT RUNS.  The certified protocol, nothing else and nothing new:
``scripts/run/run_dino.py`` on the card
``scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml`` with recipe
``nemo_dino_kamm_mlf``, 30 days from rest at ``dt = 2700 s`` on NEMO's own
199x52 DINO frame, scored at day 30 against NEMO's ``RUN_TRAJ`` ``kt = 960``
restart by ``twin_nemo_ts_maps.py``.  Both of those are existing committed
programs; this gate only sequences them and reads the comparator's JSON.

THE BAR.  ``BAR_T3D_K`` is the value the lane is certified at, plus the
tolerance below.  A landing that pushes the day-30 wet 3-D temperature rms
above the bar fails the gate.  Never relax the bar to make a landing pass:
move it DOWN when a landing improves the number, and record both numbers in
that round's receipt.

COST AND OPT-IN.  One 960-step GPU run, about 5 minutes of model time plus
compile.  It is therefore opt-in: without ``LEGOESM_DINO_MONTH_GATE=1`` in
the environment the gate prints SKIPPED and exits 0, so the operator's
``land.sh`` can call it unconditionally and let the environment decide which
landings pay for it.  Physics landings (anything under ``packages/ocean`` or
``packages/core``) should set it; docs-only landings should not.

USAGE
-----
    LEGOESM_DINO_MONTH_GATE=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
        python scripts/validate/ocean_fidelity/dino_1226/\
dino_fromrest_month_gate.py --work-dir <scratch>

    python scripts/validate/ocean_fidelity/dino_1226/\
dino_fromrest_month_gate.py --self-test     # no GPU, no model run

``--self-test`` is the non-vacuity check: it drives :func:`verdict` with a
planted number above the bar and requires it to FAIL, and with the certified
number and requires it to PASS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# The certified day-30 wet 3-D temperature rms against NEMO's RUN_TRAJ kt=960
# record, in kelvin.  Provenance: the 2026-09-29 repair measurement in
# docs/ocean/fidelity/testcases/
# nemo_testcases_l2_gyre_dino_month_regression_receipt.md, taken with exactly
# the protocol below.  It also matches the pinned #1728 artifact (2.039e-03 K)
# to the printed digits.
CERTIFIED_T3D_K = 2.040288765e-03
# Run-to-run reproducibility on one pinned GPU is bit-exact for this card, so
# the tolerance only has to absorb a driver/XLA rebuild.  10 % of the
# certified value, which is still 3x smaller than the regression this gate
# was written to catch.
BAR_T3D_K = CERTIFIED_T3D_K * 1.10

CARD = "scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml"
RECIPE = "nemo_dino_kamm_mlf"
DAYS = 30
DT_SECONDS = 2700.0
NEMO_KT = 960
DEFAULT_NEMO_RUN = ("/data/abyssal/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
                    "cfgs/DINO/RUN_TRAJ")
ENV_FLAG = "LEGOESM_DINO_MONTH_GATE"


def verdict(measured_k: float, bar_k: float = BAR_T3D_K):
    """Return ``(passed, line)`` for one measured day-30 rms.

    Pure, so the non-vacuity self-test and the unit test can drive it without
    a GPU.  A measurement exactly at the bar passes; anything above fails.
    """
    if not (measured_k == measured_k) or measured_k < 0.0:      # NaN or absurd
        return False, (f"DINO from-rest month: measured {measured_k!r} is not "
                       "a usable rms -- the run or the comparator is broken")
    passed = measured_k <= bar_k
    return passed, (
        f"DINO from-rest month day-{DAYS} wet 3-D T rms vs NEMO kt={NEMO_KT}: "
        f"{measured_k:.9e} K against bar {bar_k:.9e} K "
        f"(certified {CERTIFIED_T3D_K:.9e} K) -- "
        f"{'PASS' if passed else 'FAIL'}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_month(work_dir: Path, nemo_run: str) -> float:
    """Run the certified protocol and return the comparator's day-30 T3D rms."""
    root = repo_root()
    run_dir = work_dir / "run"
    score_dir = work_dir / "score"
    subprocess.run(
        [sys.executable, "scripts/run/run_dino.py", "--config", CARD,
         "--grid", "latlon", "--recipe", RECIPE, "--n-lon", "50",
         "--nemo-faithful-grid", "--allow-multiyear", "--dt", str(DT_SECONDS),
         "--days", str(DAYS), "--snapshot-every-days", str(DAYS),
         "--output-dir", str(run_dir)],
        cwd=root, check=True)
    subprocess.run(
        [sys.executable,
         "scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py",
         "--run-dino-dir", str(run_dir), "--day", str(DAYS),
         "--nemo-kt", str(NEMO_KT), "--nemo-run", nemo_run,
         "--output-dir", str(score_dir)],
        cwd=root, check=True)
    sidecar = score_dir / f"twin_nemo_ts_maps_day{DAYS}_kt{NEMO_KT}.json"
    report = json.loads(sidecar.read_text())
    return float(report["statistics"]["T3D"]["rms_difference"])


def self_test() -> int:
    """Non-vacuity: the bar must reject the regression it was written for."""
    ok, line = verdict(CERTIFIED_T3D_K)
    assert ok, f"the certified number must pass its own bar: {line}"
    bad, line_bad = verdict(6.981690958e-03)
    assert not bad, f"the 2026-09 regression must fail the bar: {line_bad}"
    nan_ok, _ = verdict(float("nan"))
    assert not nan_ok, "a NaN measurement must fail, not pass"
    print("SELF-TEST OK  " + line)
    print("SELF-TEST OK  " + line_bad)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="scratch directory for the run and the score")
    parser.add_argument("--nemo-run", default=DEFAULT_NEMO_RUN)
    parser.add_argument("--self-test", action="store_true",
                        help="drive verdict() with planted numbers; no GPU")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if os.environ.get(ENV_FLAG) != "1":
        print(f"SKIPPED  DINO from-rest month gate is opt-in; set {ENV_FLAG}=1 "
              "on physics landings to run it (about 15 minutes on one GPU)")
        return 0
    work_dir = args.work_dir
    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="dino-month-gate-")
        work_dir = Path(tmp.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        measured = run_month(work_dir, args.nemo_run)
    finally:
        if tmp is not None:
            tmp.cleanup()
    passed, line = verdict(measured)
    print(line)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
