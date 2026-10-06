#!/usr/bin/env python3
"""Replay a run from its checkpoint with the per-process budget ledger on, for one
vertical band, over a whole window -- the model's own per-step ledger
(diagnostics.process_ledger), accumulated over EVERY step, so the physics
increment applied once per physics cadence and the dynamics of the steps in
between are both counted (a one-step snapshot cannot do that).

The band is the driver's own ``--budget-ledger-sigma-band``; on the hybrid
coordinate it selects interfaces by A_half + B_half, i.e. by pressure / p_ref at
p_s = p_ref.  The run's own launch arguments (mpas_onestep_param_grad.launch_argv)
are reused with --distributed removed: the ledger is single-rank only.

Run from the RUN'S checkout (cwd) with its library on PYTHONPATH.

Usage: ledger_band_replay.py <run> <start_day> <n_days> <sigma_lo> <sigma_hi> <out_dir>
Writes <out_dir>/budget_ledger_columns.npz (per-column mean rates over n_days).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpas_onestep_param_grad as H  # noqa: E402  (launch_argv)


def build_argv(run, day, n_days, lo, hi, out):
    argv = H.launch_argv(run, day, Path(out))        # ends with --days 1
    while "--distributed" in argv:
        argv.remove("--distributed")
    i = argv.index("--days")
    argv[i + 1] = str(n_days + 1)    # one extra day so the n_days diag write lands
    argv += ["--budget-ledger", "--budget-ledger-sigma-band", str(lo), str(hi),
             "--diag-days", str(n_days)]
    return argv


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 6:
        raise SystemExit(__doc__)
    run, day, n_days, lo, hi, out = a[0], int(a[1]), int(a[2]), float(a[3]), float(a[4]), a[5]
    Path(out).mkdir(parents=True, exist_ok=True)
    run_argv = build_argv(run, day, n_days, lo, hi, out)
    ra = Path.cwd() / "scripts" / "run"
    if not (ra / "run_amip.py").exists():
        raise SystemExit(f"run from the run's checkout: no {ra}/run_amip.py")
    sys.path.insert(0, str(ra))
    import run_amip
    print(f"run_amip from {run_amip.__file__}\nargv: {run_argv}", flush=True)
    return run_amip.main(run_argv)


if __name__ == "__main__":
    sys.exit(main())
