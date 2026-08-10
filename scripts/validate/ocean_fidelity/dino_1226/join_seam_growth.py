#!/usr/bin/env python
"""JOIN-level growth instrument: per-step divergence vs NEMO's own per-step
restarts, with a self-perturbation NOISE FLOOR (skill Rule 3).

WHY THIS EXISTS
---------------
The #1226/#1492 fidelity gate is 53 rows, and EVERY row compares ONE operator
at ONE step with NEMO's own inputs bracketing it.  That instrument is
structurally blind to JOIN defects -- composition, call ordering, time-level
handoff, commit points -- because a join error is invisible to a test that
hands the stage its inputs.  Both of this campaign's CONFIRMED defects lived
in joins (the surface-placement combine that discarded 56% of every surface
flux; step composition).

``multistep_replay.run_replay`` (committed) already integrates legoESM from
NEMO's bridged day-0 leap-frog state and compares the END-OF-STEP state to
NEMO's OWN restart at each step.  What it did NOT have is a DRIVER that
(a) runs the full 32-step RUN_TWIN_STEP1 window, (b) reports the GROWTH
(divergence(k)/divergence(1)) per field, and (c) establishes the NOISE FLOOR
the growth must be judged against.  Without (c) a growth curve is
uninterpretable: legoESM's DINO channel is a chaotic system, and ANY
perturbation grows.

THE DISCRIMINATOR (Rule 3/Rule 4)
---------------------------------
Arm ``TWIN`` (accumulating): bridge ONCE at kt=IC_STEP, integrate k=1..N,
compare each k against NEMO's restart at IC_STEP+k.  Divergence here is
``injected(k) + amplified(everything injected before k)`` -- the two are
confounded.

Arm ``RESET`` (per-step injection, no accumulation): RE-BRIDGE from NEMO's
OWN restart at IC_STEP+k-1, take exactly ONE step, compare against NEMO's
restart at IC_STEP+k.  Every entry starts from NEMO's exact state, so the
divergence is the PURE per-step source with zero inherited history.  This
is the arm that separates the two hypotheses:

  * ``RESET`` flat and ~= ``TWIN(k=1)``, ``TWIN`` growing  -> a CONSTANT
    per-step join source, plus amplification.  The join is re-injected
    every step; fixing it removes the whole curve.
  * ``RESET`` decaying toward 0 while ``TWIN`` grows       -> a START-UP
    (handshake / bridging) artifact only; the ongoing step is faithful.
  * ``RESET`` growing with k                               -> the source is
    state-dependent (regime, not composition).

Both arms change exactly ONE variable (whether the state is re-seeded);
protocol, window, masks, metric and model config are byte-identical
(Rule 7).

RETRACTED CONTROL (recorded per Rule 11): the first version of this script
used a ``1e-12 * |T|`` IC perturbation as a "noise floor".  It returned
numbers BIT-IDENTICAL to the twin arm in every field at every k (ratio
1.000e+00), i.e. it measured nothing: over a 32-step (1-day) window with a
measured amplification of only ~4x, a 1e-11 K seed stays ~1e-11 K and never
reaches the 5 printed digits.  The perturbation floor is not a usable
control on this window; the RESET arm replaces it.

BLIND SPOTS (Rule 2 -- stated BEFORE the numbers)
-------------------------------------------------
  * END-OF-STEP only.  NEMO's per-step restarts expose the state AFTER the
    time-level swap.  Every intra-step seam (ssh_nxt -> wzv -> dyn_spg ->
    dyn_zdf -> ssh_atf -> tra_atf) is a BLACK BOX here.  Intra-step
    localisation needs the ``stp_dump_*`` chain, which NEMO writes only at
    ``kt == nit000`` (stpmlf.F90:719/755/799) -- i.e. step 1 only.  The
    momentum half of that chain is measured by the committed
    ``acc_momentum_budget.py``; the tracer half and the ssh/e3t half are not.
  * Two joins whose per-step errors CANCEL at the swap are invisible.
  * A join defect that only activates in a regime the 32-step (1-day) window
    never enters (seasonal forcing, deep convection) is invisible.
  * ``max|.|`` is a single-cell reduction: it can peak at DIFFERENT physical
    locations at different k.  The argmax index is printed alongside so a
    "localised" claim can be checked rather than assumed.

PRECONDITIONS
-------------
fp64 (``PrecisionPolicy.fp64()`` -- ``JAX_ENABLE_X64=1`` alone is NOT enough,
Rule 1c) and ``LEGOESM_NEMO_E3T=both``; both are asserted by
``multistep_replay.build_replay_ic`` and re-printed here.

Usage
-----
    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \\
        scripts/validate/ocean_fidelity/dino_1226/join_seam_growth.py [n_steps]
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

FIELDS = ("max_dT_now", "max_dS_now", "max_du_now", "max_dv_now",
          "max_deta_now", "max_den", "max_de3t",
          "max_dT_before", "max_du_before")


RUN_GDB = os.environ.get(
    "DINO_NEMO_RUN_GDB",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB")


def seam_two_ww() -> None:
    """SEAM S10 vs S18: NEMO diagnoses ``ww`` TWICE per step and the two are
    consumed by DIFFERENT stages.

    ``wzv`` call 1 (stpmlf.F90:244) runs BEFORE ``dyn_spg``, so its
    ``r3t(Kaa)`` (wzv_MLF's ``r1_Dt*e3t_0*(r3t(Kaa)-r3t(Kbb))`` term,
    sshwzv.F90:216-217) comes from ``ssh_nxt``'s FIRST-GUESS ssh(Naa)
    (sshwzv.F90:125).  Call 2 (stpmlf.F90:315) runs AFTER ``dyn_spg_ts``
    has REPLACED ssh(Naa) with the barotropic time-average
    (dynspg_ts.F90:603 zeroes it, :991 accumulates it) and after the 2nd
    ``dom_qco_r3c`` (stpmlf.F90:303).  Call 1's ww feeds ``dyn_zad``
    (inside ``dyn_adv``); call 2's ww feeds ``tra_adv``.  legoESM computes
    ONE vertical velocity -- so IF the two NEMO ww's differ materially, the
    choice of which one we reproduce is a join defect.

    This function is pure arithmetic on NEMO's OWN two dumps: no legoESM
    quantity enters, so there is no staggering/units/time-level mapping to
    get wrong (the one instrument risk is the record shape, which is
    asserted against the file size).
    """
    jpi, jpj, jpk, hls = 56, 203, 36, 2
    a = np.fromfile(f"{RUN_GDB}/wzv_dump_ww_call1.bin", dtype="<f8")
    b = np.fromfile(f"{RUN_GDB}/wzv_dump_ww_call2.bin", dtype="<f8")
    assert a.size == b.size == jpi * jpj * jpk, (a.size, b.size)
    a = a.reshape(jpk, jpj, jpi)[:, hls:-hls, hls:-hls]
    b = b.reshape(jpk, jpj, jpi)[:, hls:-hls, hls:-hls]
    rms = lambda x: float(np.sqrt((x ** 2).mean()))
    print("\n--- SEAM S10 vs S18: NEMO's two per-step ww (RUN_GDB, kt=nit000) ---")
    print(f"  RMS(ww call1)={rms(a):.4e}  RMS(ww call2)={rms(b):.4e}  "
          f"RMS(call2-call1)={rms(b - a):.4e}  rel={rms(b - a) / rms(b):.6f}  "
          f"max|diff|={float(np.abs(b - a).max()):.3e}  "
          f"corr={float(np.corrcoef(a.ravel(), b.ravel())[0, 1]):.8f}")


def _growth_table(name: str, res: dict[str, np.ndarray]) -> None:
    kt = res["kt"]
    n = len(kt)
    print(f"\n--- {name}: divergence vs NEMO restart, per step ---")
    print(f"{'field':<16s} {'k=1':>11s} {'k=2':>11s} {'k=%d' % (n // 2):>11s} "
          f"{'k=%d' % n:>11s} {'k_N/k_1':>10s} {'per-step':>10s}")
    for f in FIELDS:
        a = res[f]
        g = a[-1] / a[0] if a[0] > 0 else float("nan")
        per = g ** (1.0 / max(n - 1, 1)) if np.isfinite(g) and g > 0 else float("nan")
        print(f"{f:<16s} {a[0]:11.4e} {a[1]:11.4e} {a[n // 2 - 1]:11.4e} "
              f"{a[-1]:11.4e} {g:10.3f} {per:10.4f}")


def main(argv: list[str]) -> int:
    n_steps = int(argv[1]) if len(argv) > 1 else 32

    from multistep_replay import (
        STEPS_PER_DAY,
        have_step1_artifacts,
        run_replay,
    )

    if not have_step1_artifacts():
        print("SKIP: RUN_TRAJ/RUN_TWIN_STEP1 oracle artifacts not present")
        return 0
    if n_steps > STEPS_PER_DAY:
        raise SystemExit(f"n_steps={n_steps} exceeds RUN_TWIN_STEP1's "
                         f"{STEPS_PER_DAY}-step window")

    from legoesm.core.precision import get_policy
    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"n_steps = {n_steps}")

    if os.path.isfile(f"{RUN_GDB}/wzv_dump_ww_call1.bin"):
        seam_two_ww()

    twin = run_replay(n_steps)
    _growth_table("TWIN (bridge once, accumulate)", twin)

    reset = _run_reset_arm(n_steps)
    _growth_table("RESET (re-bridge from NEMO every step, 1 step each)", reset)

    print("\n--- TWIN / RESET: inherited vs per-step-injected ---")
    print(f"{'field':<16s} {'reset k=1':>11s} {'reset k=%d' % n_steps:>11s} "
          f"{'reset trend':>12s} {'twin k=%d' % n_steps:>11s} "
          f"{'twin/reset':>11s}")
    for f in FIELDS:
        t, r = twin[f], reset[f]
        trend = r[-1] / r[0] if r[0] > 0 else float("nan")
        ratio = t[-1] / r[-1] if r[-1] > 0 else float("inf")
        print(f"{f:<16s} {r[0]:11.4e} {r[-1]:11.4e} {trend:12.3f} "
              f"{t[-1]:11.4e} {ratio:11.3f}")
    print("\nREAD: reset trend ~1 => a CONSTANT per-step join source "
          "(re-injected every step). reset trend << 1 => a start-up/handshake "
          "artifact only. twin/reset = how much of the day-32 divergence is "
          "inherited amplification rather than that step's own injection.")
    return 0


def _run_reset_arm(n_steps: int) -> dict[str, np.ndarray]:
    """PURE per-step injection: re-bridge from NEMO's restart every step.

    Implemented by advancing ``multistep_replay.IC_STEP`` and calling the
    COMMITTED ``run_replay(1)`` once per step -- the bridge, the forcing
    application, the model config and the comparison code are byte-identical
    to the TWIN arm (``IC_STEP`` is the only thing that moves).  No
    comparison logic is re-implemented here.
    """
    import multistep_replay as mr

    base = mr.IC_STEP
    out: dict[str, list] = {}
    try:
        for j in range(n_steps):
            mr.IC_STEP = base + j
            r = mr.run_replay(1)
            for k, v in r.items():
                out.setdefault(k, []).append(v[0])
    finally:
        mr.IC_STEP = base
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
