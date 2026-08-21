#!/usr/bin/env python
"""#1455 SD: the 90-day twin on NEMO's TRUE T-DEPTH ladder (the ownership A/B arm).

ONE VARIABLE vs the recorded analytic-ladder arm (``arm3_bn2.npz``): the
``LEGOESM_NEMO_E3T`` mode handed to
``nemo_state_bridge.effective_vertical_scale_factors``.  Everything else --
recipe, restart, --bridge-before, --save-3d, fp64 policy, forcing -- is the
recorded ``kamm_twin_90d.run_twin`` protocol, imported not copied.

WHICH KNOB (measured, not inferred; printed by this script every run):
  off        e3t_1d thickness + gdept_1d T-depth  (the recorded arms' ladder)
  e3t_only   e3t_0   thickness + gdept_1d T-depth
  gdept_only e3t_1d  thickness + gdept_0  T-depth  <- the TRUE T-DEPTH ladder
  both       e3t_0   thickness + gdept_0  T-depth
The T-depth ladders differ ONLY below k=25 (max rel 3.95%, max 104.97 m at
depth); the thickness ladders differ by up to 70.4 m (14.8% rel).

THE RECORDED INSTABILITY IS THE THICKNESS LADDER, NOT THIS ONE.  Addendum 33's
four-way isolation (day-10 max|u|): off 0.60 STABLE, gdept_only 0.61 STABLE,
e3t_only 1.74 GROWING, both 1.61 GROWING.

  RETRACTED 2026-08-21 (#1455), as a RECORD OF A RUN, not as a claim: those two
  GROWING rows DID NOT REPRODUCE.  Four 90-day arms on the same restart,
  differing only in this variable, all ran STABLE to day 90 at 0.6332 / 0.6341 /
  0.6344 / 0.6347 m/s peak speed, and the two end arms were re-confirmed under
  an fp64 precision policy at 0.6332 ("off") and 0.6350 ("both").  Nothing
  approached 1.6-1.7 m/s at any point.
  PLAUSIBLE discriminator, NOT measured: the seasonal-clock fix landed
  2026-08-20, one day AFTER the a9e289d8e measurements below, so the GROWING
  arms were forced exactly antiphase to NEMO's season and the later stable ones
  were not.  The run that would settle it is this arm at the LEGACY clock
  (DINO_TWIN_SEASONAL_KT0=0): if day-10 peak speed returns to ~1.6 m/s only
  there, the recorded instability was the clock and can be retired by name.
  Until that runs, the two GROWING rows are unexplained, not explained.  So ``gdept_only`` -- the arm the
#1455 four-arm PGF isolation showed collapses the southern pressure-gradient
gap 88x (-6.13 -> -0.07 m3/s2) -- is the one true-ladder arm that is NOT
blocked by the restart-start instability.

MEASURED AT HEAD a9e289d8e (fp64, nemo_dino_kamm_mlf, --bridge-before --save-3d,
production surface placement, wind on, day-0 gate exact on both arms):

  STEP 2 -- STABILITY.  The true T-DEPTH ladder runs the FULL 90 days from the
  NEMO restart.  max|u| 0.667 (d1) .. 0.665 (d10) .. 0.634 (d90); max|eta|
  0.830 -> 0.872; finite everywhere; STABLE=True.  The recorded thickness-
  ladder signature (0.69 -> 1.24 by d3, 2.23 by d20, saturating ~3.0) does NOT
  appear.  This CONFIRMS addendum 33's own four-way isolation rather than
  contradicting it: only ``e3t_only``/``both`` were ever unstable.  The
  campaign's "the true-ladder A/B is blocked on the restart instability" is
  therefore FALSE FOR THE T-DEPTH LADDER -- it was only ever true for the
  thickness ladder.

  STEP 3 -- OWNERSHIP A/B vs the analytic-ladder arm (arm3_bn2, same HEAD,
  same protocol, ONE variable = LEGOESM_NEMO_E3T off -> gdept_only).
  Continuity control (Rule 1e) passed first: southern_depth_mean_map.py
  reproduced arm3_bn2's committed day-90 group deficit -1.4835 Sv (committed
  -1.483) before either arm was read.

    acceptance_gate_90d.py, |candidate - NEMO d90|:
      metric                     analytic     true T-depth
      ACC [Sv]                    1.740        0.773
      upper contrast             1.828e-3     1.988e-3
      deep  contrast             2.288e-4     2.691e-4
      S-band surface sigma MAX   2.886e-3     3.063e-3
      S-band surface sigma MEAN  2.302e-2     2.302e-2
    (gate level 5x: 0/5 PASS on BOTH arms -- the ladder does not clear it.)

    ** RETRACTED BY MY OWN FOLLOW-UP, BEFORE IT WAS REPORTED: "the true ladder
    removes 56% of the ACC error" is FALSE AS A STATEMENT ABOUT THE PHYSICS. **
    The recorded ACC metric takes the MEDIAN over longitudes of a FULL-SECTION
    transport, which is not additive across latitude groups, so it cannot be
    split by band.  Re-reduced with the MEAN (additive; it reproduces the same
    total to 0.09 Sv) the day-90 transport error decomposes as [Sv]:

      row group              NEMO      analytic  true T-depth   err(an)  err(true)
      south      0..13      8.8398      7.3564       7.1875     -1.4835    -1.6523
      channel   14..48     29.8411     29.6948      31.8116     -0.1463    +1.9705
      north     49..end    26.5948     26.4848      25.5954     -0.1100    -0.9994
      TOTAL               65.2757     63.5360      64.5945     -1.7397    -0.6812

    EVERY band's error gets LARGER in magnitude on the true T-depth ladder --
    south 1.48 -> 1.65, channel 0.15 -> 1.97 (13x), north 0.11 -> 1.00 (9x).
    The total only shrinks because the channel's NEW +1.97 Sv OVERSHOOT
    cancels the south and north deficits.  This is the Rule-8 ladder paradox
    ("the true ladder tracks NEMO worse") reproduced and now LOCALISED: the
    faithful geometry is worse in every band and the aggregate hides it by
    cancellation.

    southern band (T-rows 0..13), day-90 group transport deficit vs NEMO:
      analytic ladder   -1.4835 Sv   (captures 52.1% of NEMO's +3.097 Sv)
      true T-depth      -1.6523 Sv   (captures 46.7%)                <- WORSE
    per-row zonal-mean dubar stays one-signed on BOTH arms (13/13 rows
    negative; |zonal mean|/std 1.11 -> 1.19), i.e. the near-uniform deficit
    does NOT vanish; it grows.

  VERDICT (CONFIRMED).  The T-depth ladder DOES NOT OWN the southern gyre
  deficit, in the sense this A/B was built to test: making the ladder faithful
  does not cure the deficit -- it makes it 11% WORSE and leaves it just as
  zonally uniform (13/13 rows one-signed on both arms).  "Exonerated" would be
  too strong, and is not claimed: the ladder is not causally inert here, it
  moves the deficit the WRONG way.  It remains the owner of the
  depth-integrated pressure-gradient BOOKKEEPING (the -6 -> -0.07 m3/s2
  four-arm result stands).  So the ~-0.61 m3/s2 near-uniform southern torque
  still has NO named owner: not the wind, not the bottom drag, not any
  momentum operator, and now not the depth ladder.

  ONE-VARIABLE CONTROL, verified not assumed: of the geometry the bridge
  builds, ONLY the T-depth ladder changes between the arms --
  ``z_coord.t_depth_ref`` max|d| = 104.969 m, ``z_coord.dz_ref`` max|d| = 0.0,
  ``H_bathy`` max|d| = 0.0.  A reviewer flagged a possible hidden second
  variable (``gm_redi_latlon_cgrid.py:486`` selects a DIFFERENT N2 formula on
  ``t_depth_ref is not None``); REFUTED by instantiating both bridges --
  ``t_depth_ref`` is non-None in BOTH arms, so no formula switch rides along.
  The ladder's many CONSUMERS (EOS pressure depth, PGF quadrature, GM/Redi
  slope depth, MLE depth) all move together, which is what changing the ladder
  MEANS: one variable, several consumers, not several variables.

FAIL-CLOSED CONTROLS (all fatal):
  C1 fp64 policy asserted via precision_gate.require_fp64.
  C2 LEGOESM_NEMO_E3T must be set EXPLICITLY (require_explicit_e3t_mode); the
     realized ladder is printed with a sha1 fingerprint of BOTH e3t and gdept
     and compared against the "off" ladder, and the A/B is asserted
     non-trivial (the selected mode must actually change a ladder).  NOTE the
     cross-contamination check is ONE-SIDED by design: it asserts that
     ``gdept_only`` leaves the THICKNESS ladder untouched, but there is NO
     mirror assertion that ``e3t_only`` leaves the T-depth ladder untouched --
     ``e3t_only``/``both`` are accepted values here and are not gated that way.
  C3 the day-0 gate + before-level bridge verify (kamm_twin_90d's own).
  C4 per-day stability watch: kamm_twin_90d prints max|u|/max|eta|/finite each
     day and aborts on non-finite / T>60.

Run (fp64, ~4 min on one GPU):
  CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=gdept_only \
    .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
    scripts/validate/ocean_fidelity/dino_1226/true_tdepth_ladder_twin.py \
    nemo_dino_kamm_mlf OUT.npz --days 90 --save-3d --bridge-before
"""
import hashlib
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_explicit_e3t_mode,
    require_fp64,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask  # noqa: E402
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    effective_vertical_scale_factors,
)
import kamm_twin_90d as K  # noqa: E402  (the recorded twin runner)


def _sha(a) -> str:
    return hashlib.sha1(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:12]


def print_realized_ladder(run_traj: str, mode: str) -> None:
    """Print + gate the ladder the bridge will actually build (skill Rule 10)."""
    g = read_nemo_mesh_mask(f"{run_traj}/mesh_mask.nc", nn_hls=0)
    tmask = np.asarray(g.tmask) > 0.5
    e3t_off, td_off, _ = effective_vertical_scale_factors(g, tmask, mode="off")
    e3t, td, src = effective_vertical_scale_factors(g, tmask, mode=mode)
    d_e3t = float(np.max(np.abs(e3t - e3t_off)))
    d_td = float(np.max(np.abs(td - td_off)))
    rel_td = float(np.max(np.abs((td - td_off) / np.maximum(td_off, 1e-30))))
    print(f"[LADDER] LEGOESM_NEMO_E3T={mode!r}  source={src}")
    print(f"[LADDER]   e3t   dtype={e3t.dtype} sha={_sha(e3t)}  "
          f"(off: sha={_sha(e3t_off)})  max|d| vs off = {d_e3t:.4f} m")
    print(f"[LADDER]   gdept dtype={td.dtype} sha={_sha(td)}  "
          f"(off: sha={_sha(td_off)})  max|d| vs off = {d_td:.4f} m "
          f"(max rel {rel_td:.5f})")
    if e3t.dtype != np.float64 or td.dtype != np.float64:
        raise SystemExit("FATAL C1: the ladder is not float64")
    if mode == "off":
        return
    if d_e3t == 0.0 and d_td == 0.0:
        raise SystemExit(
            f"FATAL C2: mode {mode!r} produced a ladder IDENTICAL to 'off' -- "
            "the A/B would be perturbing nothing (mesh_mask has no e3t_0/"
            "gdept_0?)")
    if mode == "gdept_only" and d_e3t != 0.0:
        raise SystemExit(
            f"FATAL C2: 'gdept_only' changed the THICKNESS ladder by {d_e3t:.4f} "
            "m -- it must change ONLY the T-depth ladder")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    require_fp64(context="true T-depth ladder twin")
    mode = require_explicit_e3t_mode("true T-depth ladder twin")
    args = K._parse_args(argv)
    print_realized_ladder(args.run_traj, mode)
    stable = K.run_twin(
        args.recipe, args.out, n_days=args.days, save_3d=args.save_3d,
        run_traj=args.run_traj, run_stepdump=args.run_stepdump,
        bridge_tke=args.bridge_tke, bridge_before=args.bridge_before,
        vmix_scheme=args.vmix_scheme, use_gm_redi=args.use_gm_redi,
        surface_tendency_placement=args.surface_tendency_placement,
        perturb_seed=args.perturb_seed,
    )
    print(f"[VERDICT] LEGOESM_NEMO_E3T={mode!r}  90-day twin STABLE={stable}")
    return 0 if stable else 1


def _self_check():
    """Runnable non-vacuity check for the ladder gate (no GPU, no restart)."""
    import types
    saved = globals()["effective_vertical_scale_factors"]
    saved_mesh = globals()["read_nemo_mesh_mask"]
    ok = np.arange(1.0, 37.0)
    try:
        globals()["effective_vertical_scale_factors"] = (
            lambda g, m, mode=None: (ok, ok, "e3t_0"))
        globals()["read_nemo_mesh_mask"] = lambda p, nn_hls=0: types.SimpleNamespace(
            tmask=np.ones((2, 2, 36)))
        try:
            print_realized_ladder("/nonexistent", "gdept_only")
        except SystemExit as e:
            print(f"OK: identical-ladder A/B is FATAL -- {e}")
        else:
            raise AssertionError("gate is VACUOUS: identical ladders passed")
        globals()["effective_vertical_scale_factors"] = (
            lambda g, m, mode=None: (ok, ok, "e3t_0") if mode == "off"
            else (ok + 1.0, ok + 1.0, "e3t_0"))
        try:
            print_realized_ladder("/nonexistent", "gdept_only")
        except SystemExit as e:
            print(f"OK: gdept_only touching the THICKNESS ladder is FATAL -- {e}")
        else:
            raise AssertionError("gate is VACUOUS: a thickness change passed")
    finally:
        globals()["effective_vertical_scale_factors"] = saved
        globals()["read_nemo_mesh_mask"] = saved_mesh


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "self-check":
        _self_check()
        sys.exit(0)
    sys.exit(main())
