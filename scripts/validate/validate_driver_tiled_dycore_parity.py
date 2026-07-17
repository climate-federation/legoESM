"""Full-segment parity lane for the tiled cube dycore (P4 increment 1c).

Codex round-14 HIGH on increment 1b: under sub-face tiling the compiled
segment runs with ``device_config=None`` (no pinned in/out shardings), so
the OUTER scan composition around the shard_map'd tiled core was never
device-validated — the driver seam is gated behind
``LEGOESM_TILED_DYCORE_EXPERIMENTAL=1`` until this lane passes.  This
script IS that validation: the SAME tiny cubed-sphere AMIP config is run
end-to-end through ``ModelDriver`` twice on one process with 24 virtual
CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=24``):

* ``--mode untiled`` — ``enable_tiled_dycore=False`` (today's >6-device
  GSPMD-auto baseline), final state saved to ``--out``;
* ``--mode tiled``   — ``enable_tiled_dycore=True`` (the adapter's tiled
  D-grid SSP-RK3 core inside the same compiled segment), compared to the
  ``--ref`` file.

Exit 0 on parity within per-field tolerances; non-zero with a per-field
max-abs-diff table otherwise.

PARITY CONTRACT: the adapter-level gate
(tests/parallel/test_tiled_cc_step_adapter.py) measured the tiled-vs-
serial step difference to be EXACT float32 storage ulps from
accumulation-order rounding (tendencies bit-identical; T 1 ulp @250 K,
p_s 2 ulps @1e5 Pa per step, winds ~1e-5 through the PGF), bounded at
u/v 2e-5 / T 1e-4 / p_s 0.06 abs over 3 steps.  This lane defaults to 5
steps with ~2.5x those bounds (chaos-free horizon); it validates the
SEGMENT composition (scan + physics + fixers + carry around the tiled
core), not the core itself — term regressions live at the adapter gate.

Config envelope: the driver's auto-computed diffusion is DISABLED
(``hyperdiff_scale=div_damp_scale=a_h_scale=0``) because the tiled base
cut omits those damping terms and ``make_tiled_cc_step`` refuses any
nonzero coefficient (loud, tested).  The default conservation fixer stays
ON: the driver helper hands the adapter a fixer-off inner copy (the
segment applies the target-anchored fixer outside dynamics), mirroring
the untiled inner-model copy exactly.

Run (compute node, both modes in one job):
    export XLA_FLAGS=--xla_force_host_platform_device_count=24
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python \
        scripts/validate/validate_driver_tiled_dycore_parity.py \
        --mode untiled --out /tmp/tiled_parity_ref.npz
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python \
        scripts/validate/validate_driver_tiled_dycore_parity.py \
        --mode tiled --ref /tmp/tiled_parity_ref.npz
"""

from __future__ import annotations

import argparse
import os
import sys

N_DEVICES = 24          # (6, 2, 2) mesh -> kt=2 sub-face tiling
_FIVE_STEPS_DAYS = 5 * 600.0 / 86400.0

# f32-honest per-field abs tolerances at the default 5-step horizon —
# ~2.5x the adapter gate's measured 3-step rounding envelope (see module
# docstring).  Overridable per-field via --tol u=... T=... .
_DEFAULT_TOL = {"u": 5e-5, "v": 5e-5, "T": 2.5e-4, "p_s": 0.15}


def _build_config(workdir: str, resolution: int = 8, nlev: int = 4):
    """Tiny C8/L4 dry-ish gray config INSIDE the tiled base-cut envelope —
    identical for both modes except ``enable_tiled_dycore``.  ``resolution``
    scales the bench sizes (C48 = production-ish per-tile work)."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere", resolution=resolution, nlev=nlev,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(
            discretization="centered", dt=600.0,
            # Tiled base-cut envelope: zero the factory-computed damping
            # (hyperdiff / div-damp / A_h) — make_tiled_cc_step refuses
            # nonzero coefficients (the tiled bodies omit those terms).
            hyperdiff_scale=0.0, div_damp_scale=0.0, a_h_scale=0.0,
        ),
        # diag/checkpoint OFF: keeps the reference byte-comparable (no
        # host cadence) and matches the cs_spmd parity lane.
        output=OutputConfig(output_dir=workdir, diag_days=0,
                            checkpoint_days=0),
        days=_FIVE_STEPS_DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", microphysics="none",
        cloud_scheme="none", gravity_wave_drag="none",
        n_devices=N_DEVICES,
    )


def _final_state_arrays(driver) -> dict:
    """Final prognostic state -> host numpy (single-process lane)."""
    import numpy as np
    import jax

    out = {}
    st = driver.state
    for name in st._fields:
        obj = getattr(st, name)
        data = getattr(obj, "data", obj)
        if data is None or not isinstance(data, jax.Array):
            continue
        out[name] = np.asarray(data)
    for name in ("q_v", "q_c", "q_r"):
        data = getattr(driver, name, None)
        if data is not None and isinstance(data, jax.Array):
            out[f"tracer_{name}"] = np.asarray(data)
    return out


def _parse_tols(pairs) -> dict:
    tol = dict(_DEFAULT_TOL)
    for p in pairs or ():
        k, _, v = p.partition("=")
        if not _ or k not in tol:
            raise SystemExit(
                f"--tol expects field=value with field in {sorted(tol)}; "
                f"got {p!r}")
        tol[k] = float(v)
    return tol


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("untiled", "tiled"), required=True)
    p.add_argument("--out", default=None,
                   help="npz path for the final state (untiled mode).")
    p.add_argument("--ref", default=None,
                   help="untiled-reference npz to compare against "
                        "(tiled mode).")
    p.add_argument("--workdir", default="/tmp/tiled_dycore_parity")
    p.add_argument("--days", type=float, default=_FIVE_STEPS_DAYS,
                   help="Run length [days]; default = five 600 s steps "
                        "(the calibrated rounding-envelope horizon).")
    p.add_argument("--tol", nargs="*", default=None, metavar="FIELD=ABS",
                   help="Per-field abs tolerance overrides, e.g. "
                        "u=1e-4 p_s=0.3 (defaults: "
                        + " ".join(f"{k}={v:g}"
                                   for k, v in _DEFAULT_TOL.items()) + ").")
    p.add_argument("--resolution", type=int, default=8,
                   help="Cube face resolution N (default 8; must divide "
                        "by kt=2). C48 approximates production per-tile "
                        "work for the np24 bench.")
    p.add_argument("--nlev", type=int, default=4,
                   help="Vertical levels (default 4).")
    p.add_argument("--bench", action="store_true",
                   help="Timing-only run: no reference write/compare "
                        "(parity gates are calibrated for the 5-step "
                        "horizon; bench horizons exceed them). Use the "
                        "printed wall receipt with two-duration "
                        "subtraction.")
    args = p.parse_args(argv)

    # Argument contract FIRST — fail before the (expensive) driver setup.
    if args.mode == "untiled" and not args.out and not args.bench:
        print("ERROR: --mode untiled requires --out (or --bench)")
        return 2
    if args.mode == "tiled" and not args.ref and not args.bench:
        print("ERROR: --mode tiled requires --ref (or --bench)")
        return 2
    tol = _parse_tols(args.tol)

    # INFO logging so the lane log carries the driver's routing receipts
    # (the 'Tiled dycore step ROUTED ...' / 'tiled dycore step ACTIVE'
    # lines) — the parity table alone proves numbers, not that the tiled
    # path actually built.
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.mode == "tiled":
        # Deliberate opt-in: this lane IS the device validation the
        # experimental gate (codex round-14 HIGH) demands.
        os.environ["LEGOESM_TILED_DYCORE_EXPERIMENTAL"] = "1"

    import numpy as np
    import jax

    if len(jax.devices()) < N_DEVICES:
        print(f"ERROR: lane needs {N_DEVICES} devices "
              f"(XLA_FLAGS=--xla_force_host_platform_device_count="
              f"{N_DEVICES}); got {len(jax.devices())}.")
        return 2

    if args.resolution % 2 != 0 or args.resolution < 4:
        print(f"ERROR: --resolution must be even and >= 4 (kt=2 tiling); "
              f"got {args.resolution}")
        return 2
    workdir = f"{args.workdir}/{args.mode}"
    cfg = _build_config(workdir, resolution=args.resolution,
                        nlev=args.nlev)
    cfg = cfg._replace(days=args.days,
                       enable_tiled_dycore=(args.mode == "tiled"))
    cfg.validate_strict()

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(cfg, output_dir=workdir)
    driver.setup()
    import time as _time
    _t0 = _time.perf_counter()
    driver.run()
    _wall = _time.perf_counter() - _t0
    # Post-setup dt: the driver CFL-clamps dt internally at higher
    # resolutions — step counts must use the dt that actually ran.
    _dt_eff = float(driver.config.dycore.dt)
    _steps = args.days * 86400.0 / _dt_eff
    # Two-duration subtraction receipt: run the SAME mode at two --days
    # values; (wall2-wall1)/(steps2-steps1) cancels compile + setup (the
    # persistent XLA cache makes both runs compile-warm anyway).  The
    # subtracted number is an END-TO-END driver.run() per-step — it still
    # includes per-SEGMENT host cadence (forcing repack, stability
    # checks; the forcing_update_days fallback cadence), identical in
    # both modes, so the tiled-vs-untiled CONTRAST is fair but the
    # absolute value is not a pure step-kernel time (codex round-17 Low).
    print(f"[{args.mode}] run wall {_wall:.2f} s over {_steps:.0f} steps "
          f"(naive {1e3 * _wall / max(_steps, 1):.0f} ms/step incl. "
          "compile — use two-duration subtraction; end-to-end, incl. "
          "per-segment host cadence)")

    if args.bench:
        return 0

    state = _final_state_arrays(driver)

    if args.mode == "untiled":
        np.savez(args.out, **state)
        print(f"[untiled] wrote {len(state)} fields -> {args.out}")
        return 0

    ref = np.load(args.ref)
    failed = []
    print(f"[tiled] parity vs {args.ref} "
          f"({args.days * 86400.0 / 600.0:.0f} steps):")
    for name in sorted(state):
        if name not in ref.files:
            print(f"  {name:14s} MISSING in reference — FAIL")
            failed.append(name)
            continue
        d = np.abs(state[name] - ref[name])
        loc = np.unravel_index(int(np.argmax(d)), d.shape)
        base = name.removeprefix("tracer_")
        gate = tol.get(base)
        verdict = ("gated" if gate is not None else "info ")
        ok = gate is None or float(d.max()) < gate
        print(f"  {name:14s} max|diff| {float(d.max()):.3e} at {loc} "
              f"[{verdict}{'' if ok else ' FAIL'}]")
        if not ok:
            failed.append(name)
    # Every gated field must exist in BOTH states — a missing prognostic
    # is a broken lane, never a pass.
    missing_gated = [k for k in tol
                     if k not in state or k not in ref.files]
    if missing_gated:
        print(f"  ERROR: gated fields absent from a state: "
              f"{missing_gated}")
        failed.extend(missing_gated)
    if failed:
        print(f"PARITY FAIL: {failed}")
        return 1
    print("PARITY OK: tiled segment matches the untiled baseline within "
          "the f32 rounding envelope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
