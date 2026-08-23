#!/usr/bin/env python
"""#1455: the implicit vertical momentum solve's u/v-face control volume.

Two modes, one file, because the second is only interpretable next to the
first.

``--geometry`` (no model run, seconds)
    Census of the defect on the SHIPPED ``nemo_dino_kamm_mlf`` geometry: how
    many u/v faces carry a non-zero layer thickness at a level their own
    both-cells-wet mask CLOSES, what fraction of the reference ladder that
    thickness is, and how much it inflates the column divisor
    ``hu = sum_k e3u_0*umask`` (NEMO ``src/OCE/DOM/domain.F90:145``).

    It also answers the two questions a reviewer asks about the fix:
      * is masking the average identical to the ``min_cell_to_uface`` house
        rule on this coordinate?  (It is -- full-step z, no partial cells.)
      * can masking newly zero a thickness the drag-in-matrix bottom-level
        indicator divides by?  (``nemo_bottom_drag_rate_faces``,
        ocean_pe_latlon_cgrid.py:3318-3332.)

``--steps N --out ARM.npz`` then ``--diff A.npz B.npz`` (one model run each)
    One-variable A/B of the whole model on the shipped card, from NEMO's own
    day-180 restart through the recorded twin harness
    (``kamm_twin_90d._build_twin_state``, ``--bridge-before``, fp64).  Run one
    arm with the fix in the tree and one with it reverted, then diff the
    arrays -- NOT printed summaries.

    RECORDED (commit 65d6703de vs its parent 4e652b101).  NOTE: these two
    numbers were taken BEFORE the loop below threaded the card's surface
    tracer forcing, so they are a difference between two arms that both
    omitted it.  As a DIFFERENCE they stand -- the omission was common to
    both -- but they are not the shipped card's trajectory, and the 32-step
    amplification in particular belongs to a configuration the card does not
    run.  Re-run them if the number is to be quoted as the card's.
      1 step  : T, S, eta max|A-B| = 0.0 exactly; u 1.45e-15, v 8.33e-16 m/s
                -- about six ulp on a 0.75 m/s field, i.e. round-off.
      32 steps: u 3.78e-03, v 8.39e-03, T 3.27e-05 -- the model amplifying that
                one-ulp perturbation over a day, the same mechanism the 0.091 Sv
                ACC noise floor measures.  NOT a signal.

WHY THE ONE-STEP RESULT IS THE ANSWER AND NOT A DISAPPOINTMENT.  Inside
``_apply_implicit_vertical_mixing`` the column divisor feeds only the
``zdf_baroclinic_only`` split, which subtracts a column mean ``c`` and adds it
back, while ``zdf_drag_in_matrix`` subtracts ``E*c`` from the same solve input.
The tridiagonal satisfies ``A*1 = (I+E)*1`` (implicit_solver.py:329-336:
``b = 1 + alpha + beta + extra_diag``, ``a = -alpha``, ``c = -beta``), so the
solve returns ``A^-1 u`` for ANY ``c``.

SCOPE OF THAT ARGUMENT, because "no live consumer" would be too strong and an
earlier version of this note said it.  The cancellation is exact only while
three things hold, and a reader should check them before reusing the claim:
the column mean is subtracted and restored with the SAME thickness; the
operator is linear in u and acts as the identity on the depth-uniform mode --
which a thickness-dependent viscosity or a quadratic bottom drag breaks; and
the barotropic solver's own operators never see the inflated thickness.

And the divisor is inert only for the depth-integrated transport this A/B
scores.  A wrong column depth does NOT cancel in the vertical velocity from
continuity, the free-surface divergence, the bottom drag (which scales with
the bottom cell's own thickness), the overturning below the last full cell,
the discrete pressure gradient across a partial step, or any
thickness-weighted transport diagnostic on the affected columns.  This probe
measures one metric; it exonerates the operator for that metric and for
nothing else.

This probe does NOT print a verdict.  The interpretation belongs in the
analysis, after these controls pass.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _geometry(recipe: str) -> None:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.grids.operators_latlon_cgrid import (
        interp_cell_to_uface, interp_cell_to_vface,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, min_cell_to_uface, min_cell_to_vface,
    )
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_grid, dino_lat_lon_vertical,
    )

    cfg = dino_config_for_recipe(recipe)
    grid = dino_lat_lon_grid(cfg)
    zc = dino_lat_lon_vertical(grid, cfg)
    print(f"recipe={recipe} coord={type(zc).__name__} nlev={zc.n_levels} "
          f"vertical_coordinate={cfg.vertical_coordinate}")

    # eta = 0 => the z-star Jacobian is 1, so the staircase is isolated from
    # the free-surface stretch and every fraction below is exact.
    dz_cell = jnp.asarray(zc.h_partial)
    dz_ref = jnp.asarray(zc.dz_ref)
    act_u, act_v = compute_face_masks_3d(zc.is_active, grid)

    for name, now, fixed, minrule, act in (
        ("u", interp_cell_to_uface(dz_cell),
         None, min_cell_to_uface(dz_cell), act_u),
        ("v", interp_cell_to_vface(dz_cell, grid),
         None, min_cell_to_vface(dz_cell, grid), act_v),
    ):
        fixed = now * act.astype(now.dtype)
        closed = (now > 1e-9) & (act < 0.5)
        n = int(closed.sum())
        print(f"\n[{name}-faces] non-zero thickness at a CLOSED level: {n}")
        if n:
            frac = now[closed] / jnp.broadcast_to(dz_ref, now.shape)[closed]
            print(f"  fraction of the reference ladder: min "
                  f"{float(frac.min()):.4f} max {float(frac.max()):.4f} "
                  f"mean {float(frac.mean()):.4f}")
        print(f"  masked average == min_cell_to_{name}face: "
              f"{bool(jnp.allclose(fixed, minrule))}")

        H_now, H_fix = now.sum(-1), fixed.sum(-1)
        wet = H_fix > 1.0
        d = (H_now - H_fix)[wet]
        print(f"  wet {name}-columns {int(wet.sum())}, inflated "
              f"{int((d > 1e-6).sum())}")
        print(f"  column-depth inflation [m]: mean {float(d.mean()):.3f} "
              f"max {float(d.max()):.3f}; as a fraction of depth: mean "
              f"{float((d / H_fix[wet]).mean()):.5f} max "
              f"{float((d / H_fix[wet]).max()):.5f}")

        # Does masking newly zero a thickness the drag term divides by?
        bot = jnp.asarray(zc.bottom_level)
        if name == "u":
            inner = jnp.minimum(jnp.roll(bot, 1, axis=1), bot)
            bot_f = jnp.concatenate([inner, inner[:, 0:1]], axis=1)
        else:
            inner = jnp.minimum(bot[:-1], bot[1:])
            bot_f = jnp.pad(inner, ((1, 1), (0, 0)), constant_values=0)
        lev = jnp.arange(zc.n_levels)
        is_bot = lev[None, None, :] == bot_f[..., jnp.newaxis]
        clash = is_bot & (act < 0.5)
        print(f"  drag bottom-level indicator on a CLOSED face: "
              f"{int(clash.sum())}  (already zero pre-fix: "
              f"{int((clash & (now <= 1e-12)).sum())}, newly zeroed: "
              f"{int((clash & (now > 1e-12)).sum())})")


def _run_arm(recipe: str, steps: int, out: str) -> None:
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    from kamm_twin_90d import (
        DT, RESTART_FILE, RUN_STEPDUMP, RUN_TRAJ, _build_twin_state,
        seasonal_t0_seconds,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
    )

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, RUN_TRAJ, RUN_STEPDUMP, bridge_before=True)
    print(f"zdf_baroclinic_only={mc.zdf_baroclinic_only} "
          f"zdf_drag_in_matrix={mc.zdf_drag_in_matrix} "
          f"reconcile={mc.barotropic.barotropic_reconcile_target} "
          f"after={mc.barotropic.barotropic_after_reconcile}", flush=True)
    # STEP THE WAY THE HARNESS STEPS, not a simplification of it. This card
    # places its surface tendency on the leap-frog right-hand side, which
    # REQUIRES the forcing to be applied with ``return_rate=True`` and the
    # rate threaded into the step; calling ``model.step`` without it omits
    # every heat, salt and solar tendency for the whole run. The arms would
    # still differ only in the variable under test -- both omit the same
    # thing -- but the docstring's claim to be an A/B "of the whole model on
    # the shipped card" would be false, and the amplification quoted at 32
    # steps would be an amplification in a configuration the card never runs.
    # Found in review; the two branches below mirror kamm_twin_90d verbatim.
    _placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    t0 = seasonal_t0_seconds(f"{RUN_STEPDUMP}/{RESTART_FILE}")
    for n in range(steps):
        _t = t0 + (n + 1) * DT
        if _placement == "leapfrog_rhs":
            st, _ext = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=_t,
                return_rate=True)
            st = model.step(st, DT, surface_forcing=sf,
                            external_tracer_rate=_ext, t_seconds=_t)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=_t)
            st = model.step(st, DT, surface_forcing=sf, t_seconds=_t)
    np.savez(out, u=np.asarray(st.u.data), v=np.asarray(st.v.data),
             T=np.asarray(st.T.data), S=np.asarray(st.S.data),
             eta=np.asarray(st.eta.data))
    print(f"wrote {out} after {steps} steps of dt={DT}s")


def _diff(pa: str, pb: str) -> None:
    a, b = np.load(pa), np.load(pb)
    keys = sorted(set(a.files) & set(b.files))
    if not keys:
        raise SystemExit(f"no common arrays between {pa} and {pb}")
    worst = 0.0
    for k in keys:
        x = np.asarray(a[k], dtype=np.float64)
        y = np.asarray(b[k], dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise SystemExit(f"non-finite value in array {k!r} -- FATAL")
        ad = np.abs(x - y)
        d = float(ad.max())
        worst = max(worst, d)
        print(f"  {k:5s} shape={x.shape} max|A-B|={d:.6e} "
              f"at {np.unravel_index(int(ad.argmax()), ad.shape)} "
              f"nnz={int((ad > 0).sum())}")
    print(f"WORST max|A-B| over every array = {worst:.6e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("A.npz", "B.npz"))
    args = ap.parse_args(argv)

    if args.geometry:
        _geometry(args.recipe)
        return
    if args.diff:
        _diff(*args.diff)
        return
    if args.steps is not None:
        if not args.out:
            raise SystemExit("--steps needs --out")
        _run_arm(args.recipe, args.steps, args.out)
        return
    raise SystemExit("pick one of --geometry / --steps --out / --diff")


if __name__ == "__main__":
    main()
