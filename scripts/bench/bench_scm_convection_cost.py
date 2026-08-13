"""Per-call cost of each convection scheme in the single-column SCM geometry.

WHY THIS EXISTS.  In the RCEMIP1 tuning campaign
(``scripts/cluster/scm_rce_paper/convtune_arms.sbatch``, job 9376354) nine of
ten schemes cost 220-312 s per 100-day evaluation and ``zhang_mcfarlane`` cost
1114 s -- 4.4x the pack, 18.5 h of wall clock for 60 evaluations.  Node
contention is already refuted by the job accounting (ZM's co-tenant on g235
exited after 6 minutes, so ZM had the node essentially to itself and was still
the slowest), which leaves a scheme-intrinsic cost.

THE HYPOTHESIS THIS PROBE TESTS.  ZM's dilute-parcel CAPE
(``convection/_zm_dilute.py``, ``use_dilute_cape=True`` by default) nests a
Newton solve inside an unrolled loop inside a level scan:

    lax.scan over nlev levels                     (sequential)
      for _ in range(_NIT_LHEAT = 2)              (unrolled into the body)
        _invert_entropy -> fori_loop(_NEWTON_ITERS = 20)   (sequential)
          3 x _moist_entropy                      (centered finite difference)

= nlev x 2 x 20 x 3 = 8,880 ``_moist_entropy`` evaluations per convection
call at nlev=74, in a dependency chain of depth 2,960.  No other scheme has
this shape.  The claim is that this is expensive *in a single column
specifically*, because with ncol=1 every one of those evaluations is a scalar
op with no vector width to amortize the loop and dispatch overhead, while in
the global model the identical op count is spread across 10^4-10^5 columns.

THREE CONTROLS, because a structural reading of the source is not a measured
cause (CLAUDE.md: a proposed mechanism must survive a perturbation test):

1.  ON/OFF -- ``use_dilute_cape=False`` selects the legacy undilute
    moist-adiabat CAPE and removes the nest entirely.  The hypothesis predicts
    ZM collapses into the pack.
2.  SCALING -- ``_NEWTON_ITERS`` is the trip count of the inner solve.  The
    hypothesis predicts cost roughly linear in it.  The probe asserts the
    patch actually changed the timing rather than silently failing to retrace,
    because an ineffective monkeypatch would produce a flat scaling that reads
    as a refutation.
3.  VECTOR WIDTH -- the same measurement at ncol = 1, 64, 1024.  The
    hypothesis predicts the ZM-to-pack cost RATIO collapses as ncol grows.  A
    ratio that stays flat refutes the "no vector width to amortize" half of
    the claim and would mean ZM is simply intrinsically expensive everywhere,
    which is a different finding with a different fix.

Nothing here interprets its own numbers: the probe prints the table and the
ratios, and the verdict is drawn afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm import constants                                     # noqa: E402
from legoesm.atmosphere.physics import (                          # noqa: E402
    ConvectionConfig, GravityWaveDragConfig, MicrophysicsConfig, PhysicsConfig,
    RadiationConfig, TurbulenceConfig, make_physics,
)
from legoesm.core.field import Field                              # noqa: E402
from legoesm.core.state import HydrostaticState                   # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate        # noqa: E402

# The campaign's own geometry: the CRM reference has 74 levels and the SCM is
# one column.  Both are read from the campaign rather than chosen here.
DEFAULT_NLEV = 74
DEFAULT_DT = 600.0
P_SFC = 101_480.0

SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)


def build_state(nlev: int, ncol: int):
    """``ncol`` identical tropical columns, RH 0.85, 6.5 K/km to a 200 K cap.

    Built through the SHARED sigma factory: ``SigmaCoordinate`` is a
    NamedTuple with seven derived fields and constructing it positionally is
    how the sibling reachability probe first crashed.
    """
    from legoesm.thermo import saturation_mixing_ratio

    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full, dtype=float)
    p_full = sigma_full * P_SFC
    z = -(constants.R_d * 290.0 / constants.g) * np.log(
        np.maximum(sigma_full, 1e-6))
    T = np.maximum(300.0 - 6.5e-3 * z, 200.0)
    q_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T),
                                               jnp.asarray(p_full)))
    q_v = np.maximum(0.85 * q_sat, 1e-11)

    dims4 = ("face", "x", "y", "level")
    dims3 = ("face", "x", "y")
    shape4 = (1, ncol, 1, nlev)
    shape3 = (1, ncol, 1)

    def b4(row):
        return jnp.broadcast_to(jnp.asarray(row).reshape(1, 1, 1, nlev),
                                shape4)

    state = HydrostaticState(
        u=Field(data=jnp.full(shape4, 5.0), name="u", dims=dims4),
        v=Field(data=jnp.zeros(shape4), name="v", dims=dims4),
        T=Field(data=b4(T), name="T", dims=dims4),
        p_s=Field(data=jnp.full(shape3, P_SFC), name="p_s", dims=dims3),
        phis=Field(data=jnp.zeros(shape3), name="phis", dims=dims3),
        tracers={
            "q_v": Field(data=b4(q_v), name="q_v", dims=dims4),
            "q_c": Field(data=jnp.zeros(shape4), name="q_c", dims=dims4),
            "q_r": Field(data=jnp.zeros(shape4), name="q_r", dims=dims4),
        },
    )
    return state, sigma


def _physics(scheme: str, dt: float, sub_overrides: dict | None = None):
    conv_kwargs: dict = {"scheme": scheme}
    if sub_overrides:
        base = getattr(ConvectionConfig(scheme=scheme), scheme)
        conv_kwargs[scheme] = base._replace(**sub_overrides)
    return make_physics(
        PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(**conv_kwargs),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        ),
        model_type="hydrostatic", dt=dt,
    )


def _activity(fn, state, sigma) -> dict:
    """Column-integrated |dT/dt| and |dq_v/dt| for the scheme's tendency.

    A scheme whose trigger never fires on this column returns an all-zero
    tendency, and timing it measures the cost of the INACTIVE branch.  Such a
    scheme must not enter the "pack" that ZM is compared against -- at ncol=1
    the pack median was originally ``kuo``, which cannot convect in a single
    column at all (codex review, finding 14).  ``finite`` alone does not catch
    this: an all-zero result is perfectly finite.
    """
    tend, _ = fn(state, None, sigma)
    dT = np.abs(np.asarray(tend.dT_dt.data, dtype=float)).sum()
    tt = tend.tracer_tendencies or {}
    dq = (np.abs(np.asarray(tt["q_v"].data, dtype=float)).sum()
          if "q_v" in tt else 0.0)
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(tend)]
    return {
        "sum_abs_dT_dt": float(dT),
        "sum_abs_dqv_dt": float(dq),
        "finite": bool(all(np.all(np.isfinite(x)) for x in leaves)),
    }


# Below these a tendency is indistinguishable from an untriggered scheme.
# Summed over the column, so both are very low bars -- an active scheme clears
# them by orders of magnitude (the weakest measured is dca at 3.6e-07 K/s).
# A scheme counts as active if EITHER channel fires: keying activity on
# temperature alone would misclassify a moisture-only closure as inactive and
# silently drop it from the pack (codex round 2, finding 10).
ACTIVITY_FLOOR_K_PER_S = 1.0e-12
ACTIVITY_FLOOR_KG_PER_KG_PER_S = 1.0e-15


def time_scheme(scheme: str, state, sigma, dt: float, *, repeats: int,
                warmups: int = 3, sub_overrides: dict | None = None) -> dict:
    """Compile once, warm up, then time ``repeats`` calls.

    The compile is timed separately and EXCLUDED from the per-call number: a
    tuning evaluation compiles once and then integrates 14,400 steps, so a
    per-call figure contaminated by compile would not be the quantity that
    explains the campaign's wall clock.

    The jitted function returns EVERY leaf of the tendency pytree, not just
    ``dT_dt``.  Returning one field lets XLA dead-code-eliminate whatever the
    scheme computes only for moisture, condensate or momentum, which would
    understate the schemes that produce the most outputs -- a per-scheme bias
    in exactly the comparison this bench exists to make (codex finding 8).
    """
    fn = _physics(scheme, dt, sub_overrides)

    @jax.jit
    def step(st):
        tend, _ = fn(st, None, sigma)
        return tuple(jax.tree_util.tree_leaves(tend))

    def _block(outs):
        for o in outs:
            o.block_until_ready()

    t0 = time.perf_counter()
    outs = step(state)
    _block(outs)
    compile_s = time.perf_counter() - t0

    for _ in range(warmups):
        _block(step(state))

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(step(state))
        samples.append(time.perf_counter() - t0)

    act = _activity(fn, state, sigma)
    ms = [1e3 * s for s in samples]
    return {
        "scheme": scheme,
        "compile_s": compile_s,
        # Median is the steady-state kernel latency; the MEAN is the estimator
        # that extrapolates to a campaign's total wall clock, because 14,400
        # sequential calls accumulate every slow one (codex finding 12).
        "call_ms": statistics.median(ms),
        "call_ms_mean": statistics.fmean(ms),
        "call_ms_min": min(ms),
        "call_ms_max": max(ms),
        "n_leaves": len(outs),
        "active": (act["sum_abs_dT_dt"] > ACTIVITY_FLOOR_K_PER_S
                   or act["sum_abs_dqv_dt"] > ACTIVITY_FLOOR_KG_PER_KG_PER_S),
        **act,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nlev", type=int, default=DEFAULT_NLEV)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--warmups", type=int, default=3,
                   help="untimed calls after compile, before the samples")
    p.add_argument("--ncols", type=int, nargs="*", default=[1, 64, 1024])
    p.add_argument("--newton-iters", type=int, nargs="*", default=[5, 10, 20],
                   help="_NEWTON_ITERS values for the scaling control")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    provenance = {
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "nlev": args.nlev, "dt": args.dt, "repeats": args.repeats,
    }
    print(json.dumps(provenance, indent=1))
    payload: dict = {"provenance": provenance}

    # ---------------------------------------------------------------- #
    # 1. All ten schemes, at every column count.
    # ---------------------------------------------------------------- #
    by_ncol: dict[int, list[dict]] = {}
    for ncol in args.ncols:
        state, sigma = build_state(args.nlev, ncol)
        nbytes = sum(int(np.asarray(x).nbytes)
                     for x in jax.tree_util.tree_leaves(state))
        rows = [time_scheme(s, state, sigma, args.dt, repeats=args.repeats,
                            warmups=args.warmups)
                for s in SCHEMES]
        by_ncol[ncol] = rows
        print(f"\n===== batch width ncol = {ncol} "
              f"(nlev {args.nlev}, dt {args.dt:.0f} s, "
              f"state {nbytes / 1024:.1f} KiB) =====")
        print(f"{'scheme':18s}{'compile [s]':>13s}{'median [ms]':>13s}"
              f"{'mean [ms]':>11s}{'sum|dT/dt|':>12s}{'sum|dqv/dt|':>13s}"
              f"  status")
        for r in sorted(rows, key=lambda r: r["call_ms"]):
            # An inactive scheme's timing is the cost of its untriggered
            # branch, NOT a scheme cost, so it is labelled in the table
            # itself rather than only in a footnote -- a bare number here can
            # be copied out and published as if it meant something (codex
            # round 2, finding 12).
            if r["active"]:
                med, mean = f"{r['call_ms']:13.3f}", f"{r['call_ms_mean']:11.3f}"
                status = "active"
            else:
                med, mean = f"{'N/A':>13s}", f"{'N/A':>11s}"
                status = f"INACTIVE (branch cost {r['call_ms']:.3f} ms)"
            print(f"{r['scheme']:18s}{r['compile_s']:13.2f}{med}{mean}"
                  f"{r['sum_abs_dT_dt']:12.3e}{r['sum_abs_dqv_dt']:13.3e}"
                  f"  {status}")
        inactive = [r["scheme"] for r in rows if not r["active"]]
        if inactive:
            print(f"  EXCLUDED from the pack (no tendency on this column, so "
                  f"their timing is the cost of the inactive branch): "
                  f"{', '.join(inactive)}")
        # ZM against the ACTIVE others.  Including an untriggered scheme makes
        # the baseline the cost of doing nothing; at ncol=1 the median of all
        # nine was `kuo`, which cannot convect in a single column at all
        # (codex round 1, finding 14).  Both reductions are printed: the
        # median is the steady-state kernel latency, the MEAN is what
        # extrapolates to a campaign's wall clock over 14,400 sequential
        # calls, and a ratio that disagrees between them is a long-tail
        # signal rather than a scheme cost (codex round 2, finding 9).
        zm_row = next(r for r in rows if r["scheme"] == "zhang_mcfarlane")
        pack_rows = [r for r in rows
                     if r["scheme"] != "zhang_mcfarlane" and r["active"]]
        if pack_rows and zm_row["active"]:
            p_med = statistics.median([r["call_ms"] for r in pack_rows])
            p_mean = statistics.median([r["call_ms_mean"] for r in pack_rows])
            print(f"  zhang_mcfarlane / median({len(pack_rows)} active others)"
                  f" = {zm_row['call_ms'] / p_med:.2f}x by median, "
                  f"{zm_row['call_ms_mean'] / p_mean:.2f}x by mean")
        else:
            print("  no active comparison scheme — ratio NOT computed")
    payload["by_ncol"] = {str(k): v for k, v in by_ncol.items()}

    # ---------------------------------------------------------------- #
    # 2. ON/OFF control: the dilute-parcel CAPE is what costs.
    # ---------------------------------------------------------------- #
    state1, sigma1 = build_state(args.nlev, 1)
    on = time_scheme("zhang_mcfarlane", state1, sigma1, args.dt,
                     repeats=args.repeats, warmups=args.warmups,
                     sub_overrides={"use_dilute_cape": True})
    off = time_scheme("zhang_mcfarlane", state1, sigma1, args.dt,
                      repeats=args.repeats, warmups=args.warmups,
                      sub_overrides={"use_dilute_cape": False})
    print("\n===== control 1: dilute-parcel CAPE ON vs OFF (ncol=1) =====")
    print(f"  use_dilute_cape=True   {on['call_ms']:9.3f} ms  "
          f"active={on['active']}")
    print(f"  use_dilute_cape=False  {off['call_ms']:9.3f} ms  "
          f"active={off['active']}")
    print(f"  ratio ON/OFF = {on['call_ms'] / off['call_ms']:.2f}x")
    if not (on["active"] and off["active"]):
        print("  WARNING: an arm produced no tendency — this compares an "
              "active scheme against a no-op, not two CAPE closures.")
    payload["dilute_control"] = {"on": on, "off": off}

    # ---------------------------------------------------------------- #
    # 3. SCALING control: cost vs the inner solve's trip count.
    # ---------------------------------------------------------------- #
    from legoesm.atmosphere.physics.convection import _zm_dilute

    original = _zm_dilute._NEWTON_ITERS
    scaling = []
    try:
        for n in args.newton_iters:
            _zm_dilute._NEWTON_ITERS = n
            # A fresh jit per setting: reusing one would return the cached
            # executable traced at the previous count and report a flat
            # scaling, which reads exactly like a refutation.
            jax.clear_caches()
            r = time_scheme("zhang_mcfarlane", state1, sigma1, args.dt,
                            repeats=args.repeats, warmups=args.warmups,
                            sub_overrides={"use_dilute_cape": True})
            r["newton_iters"] = n
            scaling.append(r)
    finally:
        _zm_dilute._NEWTON_ITERS = original
        jax.clear_caches()

    print("\n===== control 2: cost vs _NEWTON_ITERS (ncol=1, dilute ON) =====")
    for r in scaling:
        print(f"  _NEWTON_ITERS={r['newton_iters']:3d}  "
              f"{r['call_ms']:9.3f} ms")
    spread = max(r["call_ms"] for r in scaling) / \
        min(r["call_ms"] for r in scaling)
    print(f"  spread over {args.newton_iters} = {spread:.2f}x")
    if spread < 1.15:
        print("  WARNING: the trip count barely moved the cost. Either the "
              "monkeypatch did not retrace, or the inner solve is not where "
              "the time goes -- do NOT read this as a confirmation.")
    payload["newton_scaling"] = scaling

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
