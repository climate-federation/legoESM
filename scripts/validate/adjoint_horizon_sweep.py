#!/usr/bin/env python
"""Does the training adjoint blow up because of CHAOS or because of a KERNEL?

The lat-lon AIMIP trainer carries a documented receipt: "a 24 h rollout
explodes the adjoint to NaN through the 144-step differentiable dycore even
when the forward is finite (1-step grad is fine; 144-step is NaN — job
8533825)" (``scripts/run/run_aimip_latlon.py``), and the supervision horizon
was cut to 6 h because of it.  That receipt has two incompatible readings and
nobody has separated them:

* **growth** — the adjoint obeys a linear equation whose modes grow like
  ``exp(lambda_max * T)`` even while the nonlinear forward stays bounded, so it
  overflows.  If this is the cause, the horizon is a genuine physical limit and
  the cure is damping, checkpointing or a shorter window.
* **kernel** — some non-smooth kernel (a limiter, a positivity clamp, a
  ``sqrt``/``pow`` at zero, a division by a vanishing denominator, a
  ``where`` guarding an already-NaN branch) hands back a NaN cotangent.  Chaos
  cannot manufacture a NaN while the adjoint is still SMALL, so a NaN at a
  small norm refutes the growth reading outright, and the cure is a one-line
  kernel fix rather than a shorter horizon.

A global gradient norm cannot tell these apart: it collapses to ``inf``/``NaN``
the moment any single leaf does.  This sweep therefore records the per-leaf
report at every horizon and classifies the failure.

THE SECOND ARM: RAYLEIGH FRICTION
---------------------------------
The lat-lon training rollout runs with ``fric_decay == 1`` — no Rayleigh drag —
for a blunt reason: ``training_driver.build_training_segment`` DEFAULTS it to
``jnp.ones`` and no caller on this lane ever passes one.  Neither
``train_physics_params``' ``make_run_seg`` nor ``run_aimip_latlon.train_variant``
forwards ``driver.fric_decay``; only ``scale_build.py`` does, for its own
neural modes.

An earlier draft of this probe blamed the #931 surface-drag gate in
``ModelDriver._create_friction`` (which zeroes the drag whenever a BL scheme
owns surface momentum).  That is RETRACTED: the gate landed in commit
``7b61d8ac2`` on 2026-07-11, three weeks AFTER the 2026-06-20 receipt commit
``24abc4949`` that recorded job 8533825, so it cannot be the explanation for
that measurement.  It is not the mechanism; the missing argument is.

The ``column_nn`` variant makes the undamped case sharpest.  Its physics
pipeline is REPLACED by a network whose final layer is zero-initialised in both
weight and bias, so at initialisation it emits EXACTLY zero tendencies and
contributes no momentum drag at all.  That is close to the configuration
``build_training_segment`` documents as producing NaN adjoints (#797 bug 11) —
though not literally an undamped bare dycore, because the segment still runs a
saturation adjustment every step and the dycore still runs its polar Fourier
filter, both of which are dissipative and both of which are themselves
candidate non-smooth-kernel suspects.

So the sweep runs each configuration twice, changing ONE thing: the Rayleigh
profile handed to the segment builder, ``ones`` (as production does) versus the
Held-Suarez profile the driver would build if no BL scheme owned surface
momentum.  Everything else — physics lane, initial condition, target, timestep,
loss — is byte-identical between the two arms.

WHAT CONFIRMS AND WHAT REFUTES
------------------------------
* NaN/inf first appears at the same horizon in BOTH friction arms, classified
  ``kernel``, at an adjoint norm comparable to the short-horizon norm
  -> growth REFUTED; it is a kernel defect and no amount of damping fixes it.
* The norm grows exponentially and overflows, classified ``growth``, and the
  damped arm survives materially further -> growth CONFIRMED, damping is the
  lever, and the fitted rate gives the usable horizon.
* Both arms finite out to 144 steps -> the receipt does not reproduce at this
  resolution, and the original measurement's configuration must be recovered
  before anything is built on it.

Usage (GPU node; see scripts/cluster/scaling_ginsburg/adjoint_horizon_sweep.sbatch):

    python scripts/validate/adjoint_horizon_sweep.py \
        --variants classical dycore --n-lat 32 --n-lev 8 \
        --horizons 1 2 4 8 16 36 72 144
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("adjoint_sweep")


def _load_latlon_driver_module():
    """Import ``run_aimip_latlon.py`` as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_aimip_latlon_mod", REPO / "scripts" / "run" / "run_aimip_latlon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def held_suarez_fric_decay(config, sigma_full, dt):
    """The Rayleigh profile ``_create_friction`` WOULD build with no BL scheme.

    Deliberately NOT read off ``driver.fric_decay``: that accessor returns the
    #931-gated value, which is exactly ``ones`` for every lane this sweep
    targets, so both arms would be identical and the comparison vacuous.  The
    formula is the driver's, restated at the one place the gate is bypassed on
    purpose, and the gate's own condition is asserted so this cannot silently
    diverge from it.
    """
    k_f_max = config.k_BL_max_per_day / 86400.0
    k_free = config.k_free_per_day / 86400.0
    k_f = k_free + k_f_max * jnp.maximum(
        0.0, (sigma_full - config.sigma_b) / (1.0 - config.sigma_b))
    return jnp.exp(-k_f * dt)


def sweep_one(loss_for_horizon, horizons, control, label):
    """Run one arm: per-horizon loss, adjoint norm, leaf report, classification."""
    from legoesm.training.grad_horizon import (
        blowup_horizon, diagnose_blowup, estimate_growth_rate, grad_max_norm,
        leaf_grad_report, unfittable_horizons,
    )
    rows = []
    norms: dict[int, float] = {}
    # The SHORTEST horizon in the sweep is the amplitude reference every later
    # horizon is judged against: "the adjoint is still small" is meaningless
    # without it, and it is the one horizon the repo's own receipt reports as
    # healthy.  Captured on the first iteration below.
    reference_max: float | None = None
    for n in sorted(horizons):
        loss, grads = eqx.filter_value_and_grad(
            lambda c, _n=n: loss_for_horizon(_n, c))(control)
        report = leaf_grad_report(eqx.filter(grads, eqx.is_inexact_array))
        # Max-norm, not L2: squaring overflows fp32 above ~1.8e19 and would
        # report a finite-but-large adjoint as a blow-up.
        gnorm = grad_max_norm(report)
        if reference_max is None and jnp.isfinite(gnorm):
            reference_max = gnorm
        verdict = diagnose_blowup(report, reference_max)
        norms[n] = gnorm
        bad = sorted(k for k, (_, ok) in report.items() if not ok)
        rows.append({
            "horizon": int(n),
            "loss": float(loss),
            "loss_finite": bool(jnp.isfinite(loss)),
            "grad_norm": gnorm,
            "verdict": verdict,
            "n_leaves": len(report),
            "n_bad_leaves": len(bad),
            "first_bad_leaves": bad[:5],
            "max_finite_leaf": max(
                (v for v, ok in report.values() if ok), default=0.0),
        })
        log.info(
            "  %-18s n=%4d  loss=%12.4e [%s]  |grad|=%12.4e  %-7s  bad=%d/%d %s",
            label, n, float(loss),
            "fwd ok" if bool(jnp.isfinite(loss)) else "fwd NaN",
            gnorm, verdict, len(bad), len(report),
            (bad[0] if bad else ""))
    return {
        "arm": label,
        "rows": rows,
        "reference_max": reference_max,
        "blowup_horizon": blowup_horizon(norms),
        "unfittable_horizons": unfittable_horizons(norms),
        "growth_rate_per_step": estimate_growth_rate(norms),
    }


def build_arms(args, mod, variant):
    """Return ``(loss_for_horizon_factory, control)`` for one physics variant.

    The factory takes a ``fric_decay`` array and returns
    ``loss_for_horizon(n_steps, control) -> scalar``, so the two friction arms
    differ in that one array and nothing else.
    """
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.training.losses import combined_loss
    from legoesm.training.trainable_params import TrainablePhysicsParams

    # The lane's CLI is reproduced verbatim so the configuration under test is
    # the launcher's, not a hand-built lookalike.  ``--output-dir`` is forced to
    # a probe-private directory: the driver refuses to start when an existing
    # run manifest disagrees with the resolved config, and the shared
    # ``results/aimip_latlon`` manifest is from a June 2026 run whose schemes
    # differ from today's defaults.
    cli = mod.build_parser().parse_args(
        [*shlex.split(args.cli_args), "--variants", "classical",
         "--output-dir", str(REPO / args.out / "driver")])
    cli.n_lat, cli.n_lev = args.n_lat, args.n_lev
    config = mod.build_latlon_config(cli)
    # No assert on ``turbulence`` here.  An earlier draft asserted it on the
    # theory that the #931 gate is what zeroes the drag; it is not (see the
    # module docstring), so that assert guarded an invariant with no bearing on
    # whether the two arms differ.  The guard that actually binds is the
    # allclose check on the two fric_decay arrays in ``main``.
    driver = ModelDriver(config)
    driver.setup()
    grid, sigma, model = driver.grid, driver.sigma, driver.model
    # The launcher caps dt by the GRAVITY-WAVE CFL, not just the model's
    # advective clamp; skipping that cap makes the forward blow up at higher
    # --n-lat and a forward NaN would then be misread as an adjoint finding.
    from legoesm import constants as _const
    from legoesm.core.cfl import cfl_max_dt
    dy_min = float(_const.R_earth) * float(jnp.pi) / float(args.n_lat)
    dt = min(float(getattr(model, "effective_dt", cli.dt)),
             cfl_max_dt(dy_min, wave_speed=400.0, cfl_number=0.7))
    sigma_full = jnp.asarray(sigma.sigma_full)

    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)
    # The loader SIZES the carry's tracer and turbulence slots from these, so
    # omitting them hands the rollout a different state vector than the lane's.
    ics, targets, forcings = mod.load_window_pairs(
        grid, sigma, [(2015, 0, 1)], rollout_hours=6, forcing_ctx=ctx,
        grid_kind=cli.grid, era5_zarr=cli.era5_zarr,
        microphysics=cli.microphysics, turbulence=cli.turbulence)
    loss_config = mod.make_loss_config(cli)

    if variant == "classical":
        pipe = build_physics_pipeline(grid, sigma, config)
        step_unified = pipe.build_step_unified()
        control = TrainablePhysicsParams.from_defaults()
        seg_kwargs_for = lambda c: c.to_segment_kwargs()
        # run_aimip_latlon hands the CLASSICAL trainer its own microphysics and
        # radiation cadence (``--microphysics kessler``, ``--rad-update-steps
        # 6``).  The throwaway June probes hardcoded ``microphysics="none"`` and
        # a cadence of 1, which is a DIFFERENT MODEL: no Kessler means the
        # saturation adjustment carries condensation instead, and a cadence of 1
        # puts six times as many radiation calls into the backward graph.  A
        # horizon measured on that model cannot be quoted about this lane.
        seg_static = dict(microphysics=cli.microphysics,
                          rad_update_steps=cli.rad_update_steps)
    elif variant == "dycore":
        # Neural column physics at INITIALISATION emits ~zero tendencies, so the
        # rollout is the bare dycore — the #797 epoch-0 configuration.
        from legoesm.atmosphere.physics.neural_physics import (
            NeuralPhysics, make_neural_step_unified,
        )
        from legoesm.core.grid_adapters import make_adapter
        nn = NeuralPhysics(
            nlev=args.n_lev, hidden_dim=cli.nn_hidden,
            n_layers=cli.nn_layers, residual_scale=cli.nn_residual_scale,
            key=jax.random.PRNGKey(cli.nn_seed))
        step_unified = make_neural_step_unified(nn, make_adapter(grid))
        control = nn
        seg_kwargs_for = lambda c: {}
        # The column_nn lane calls build_training_segment with NO physics
        # kwargs, so ITS production settings are that helper's own defaults.
        # Different from the classical lane on purpose — matching each lane is
        # the point.
        seg_static = dict(microphysics="none", rad_update_steps=1)
    else:
        raise ValueError(
            f"unknown variant {variant!r}; supports 'classical', 'dycore'.")

    def _segment(fric_decay, c):
        from legoesm.driver.compiled_segments import build_segment_fn
        step = (make_neural_step_unified(c, make_adapter(grid))
                if variant == "dycore" else step_unified)
        return build_segment_fn(
            model=model, step_unified=step, grid=grid,
            sigma_full=sigma_full, dsigma=jnp.asarray(sigma.dsigma), dt=dt,
            fix_moisture=False, fix_mass=False,
            fric_decay=fric_decay, qv_smooth_coeff=0.0, **seg_static,
            lat=grid.grid_lat, lon=grid.grid_lon, start_day=0.0,
            # Production passes gradient_checkpoint=True (the sqrt-N remat
            # path).  Matching it keeps the backward graph the lane's own; the
            # throwaway June probes used False, which is a second changed
            # variable on top of the horizon.
            gradient_checkpoint=True, **seg_kwargs_for(c))

    def factory(fric_decay, wrt):
        """Return ``(loss_for_horizon, control)`` for one differentiation target.

        ``wrt="ic"`` differentiates with respect to the INITIAL STATE.  This is
        the adjoint the receipt is actually about, and — decisively — every leaf
        of it carries gradient.  ``wrt="params"`` differentiates the lane's own
        control, which is the quantity training moves but is a BAD instrument
        here: under today's defaults four of the six trainable scalars are
        structurally disconnected from the loss (the two SBM parameters are dead
        because convection is mass-flux, and the two bulk exchange coefficients
        are dead because the turbulence scheme owns the surface), so they return
        exact zeros.  A classifier asking "are the surviving leaves still small"
        would then be reading leaves that are zero by construction and would
        answer "small" no matter what the physics did.  Both are recorded; the
        initial-state sweep is the one to believe.
        """
        if wrt == "ic":
            seg = _segment(fric_decay, control)

            def loss_for_horizon(n_steps, ic):
                return combined_loss(seg.raw(ic, n_steps, forcings[0]),
                                     targets[0], sigma_full, grid=grid,
                                     config=loss_config)
            return loss_for_horizon, ics[0]

        def loss_for_horizon(n_steps, c):
            seg = _segment(fric_decay, c)
            return combined_loss(seg.raw(ics[0], n_steps, forcings[0]),
                                 targets[0], sigma_full, grid=grid,
                                 config=loss_config)
        return loss_for_horizon, control

    return factory, config, sigma_full, dt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=["classical", "dycore"],
                   choices=["classical", "dycore"])
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 36, 72, 144],
                   help="rollout lengths in dycore steps (144 = 24 h at dt=600)")
    p.add_argument("--n-lat", type=int, default=32)
    p.add_argument("--n-lev", type=int, default=8)
    p.add_argument(
        "--wrt", nargs="+", default=["ic", "params"], choices=["ic", "params"],
        help="what to differentiate.  'ic' (the initial state) is the adjoint "
             "the receipt is about and every leaf of it is live; 'params' is "
             "what training moves but several of its leaves are structurally "
             "dead under the default scheme set.")
    p.add_argument("--out", default="results/adjoint_horizon_sweep")
    p.add_argument(
        "--cli-args", default="",
        help="extra flags forwarded VERBATIM to run_aimip_latlon's parser, as "
             "ONE quoted string (split with shlex), e.g. "
             "--cli-args '--convection sbm --radiation gray'.  A nargs='*' "
             "list cannot carry these: argparse stops consuming at the first "
             "token beginning with '-', so the flags would be rejected as "
             "unknown arguments and the arm would never run.  The June 2026 "
             "receipt this probe re-examines ran gray radiation + SBM "
             "convection; today's defaults are rrtmgp + mass flux, so the two "
             "are different physics and are measured as separate, separately "
             "labelled arms.")
    args = p.parse_args()

    mod = _load_latlon_driver_module()
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
    except Exception:
        sha = "unknown"

    results = {"git_sha": sha, "argv": sys.argv[1:],
               "cli_args": args.cli_args,
               "x64": bool(jax.config.jax_enable_x64), "arms": []}
    for variant in args.variants:
        factory, config, sigma_full, dt = build_arms(args, mod, variant)
        arms = {
            "fric_off": jnp.ones_like(sigma_full),
            "fric_on": held_suarez_fric_decay(config, sigma_full, dt),
        }
        # A/B hygiene: the whole comparison is void if the two arrays coincide,
        # which is exactly what happens if the #931 gate is ever removed or the
        # Held-Suarez coefficients go to zero.  Checked, not assumed.
        if bool(jnp.allclose(arms["fric_on"], arms["fric_off"])):
            raise SystemExit(
                "the damped and undamped arms are numerically identical "
                f"(fric_on in [{float(jnp.min(arms['fric_on'])):.6g}, "
                f"{float(jnp.max(arms['fric_on'])):.6g}]); there is no A/B "
                "here and any result would be a comparison of a thing with "
                "itself")
        for arm_name, fric in arms.items():
          for wrt in args.wrt:
            log.info("=== variant=%s arm=%s wrt=%s  fric_decay[min,max]="
                     "[%.6f, %.6f]", variant, arm_name, wrt,
                     float(jnp.min(fric)), float(jnp.max(fric)))
            loss_fn, control = factory(fric, wrt)
            out = sweep_one(loss_fn, args.horizons, control,
                            f"{variant}/{arm_name}/{wrt}")
            out["wrt"] = wrt
            out["variant"] = variant
            out["fric_decay_min"] = float(jnp.min(fric))
            # Provenance next to the number: the schemes are what make one
            # arm's blow-up horizon comparable (or not) to another's.
            out["schemes"] = {
                k: str(getattr(config, k, None)) for k in
                ("convection", "turbulence", "microphysics", "radiation",
                 "cloud_scheme")}
            out["dt"] = dt
            results["arms"].append(out)

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "sweep.json"
    # allow_nan=False would raise; NaN is not valid JSON, so unfitted rates are
    # written as null and the reader cannot mistake a bare NaN token for data.
    def _json_safe(o):
        if isinstance(o, float):
            if o != o:
                return None          # NaN: "not fitted"
            if o == float("inf"):
                return "inf"         # a blown-up norm is DATA, not missing
            if o == float("-inf"):
                return "-inf"
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json_safe(v) for v in o]
        return o
    dest.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False))
    log.info("wrote %s", dest)

    log.info("=== SUMMARY ===")
    for a in results["arms"]:
        log.info("  %-34s blowup_at=%-5s unfittable=%-10s growth=%.4e /step",
                 a["arm"], a["blowup_horizon"], a["unfittable_horizons"],
                 a["growth_rate_per_step"])


if __name__ == "__main__":
    main()
