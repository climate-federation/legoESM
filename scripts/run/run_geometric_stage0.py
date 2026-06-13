"""GEOMETRIC calibration — Stage 0 truth generator (the twin's reference).

Spins up the 2° Veros-ACC channel with the GEOMETRIC mesoscale closure
(``EKEConfig.closure="geometric"``, Torres et al. 2025) at a KNOWN "true"
``GeometricConfig`` and writes the twin-experiment reference:

  * ``truth_snapshot.npz`` — the full ocean state at the end of the averaging
    window (every Field leaf), for the exact-IC / attractor-drop member protocol
    (``docs/planning/geometric_calibration_campaign.md`` §3.1).
  * ``observables.npz`` — the ETKI observation vector: time-mean AND temporal-std
    maps of T, S, η and the depth-integrated EKE field over the averaging window
    (§3.2, equal-weighted normalized maps; NO domain-integrated scalars), plus
    the HELD-OUT metrics (ACC transport, total KE) as time series for the
    early-stopping monitor.

This is the Stage-0 protocol shakedown: truth = the model itself at
``GeometricConfig`` defaults; a later step perturbs the parameters and recovers
them with ETKI (``legoesm.training.etki``) under the §3 protocol.

Reuses the SHARED ocean diagnostics (``barotropic_streamfunction``,
``compute_energy_budget``, ``compute_layer_thickness``) — the same reductions the
Phase-G free-run harness (``scripts/validate/ocean_fidelity/run_acc_freerun.py``)
applies — so truth observables and member observables are computed identically.
The AB2 + rigid-lid carry seeding mirrors that harness's ``_run_legoesm`` (the
recipe's faithful ``with_surface_forcing=True`` stepping is ab2 / rigid_lid /
dt_mom_ratio=9).

Run in float64 (the audit's non-negotiable for ocean gradient/calibration work).

Usage::

    # de-risk wiring + geometric stability (short, GPU 0):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0.py \
        --smoke --out results/ocean/geometric_stage0/smoke

    # truth generation (1.5-yr spin-up, 800-day averaging window, GPU 0):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run/run_geometric_stage0.py \
        --spin-years 1.5 --avg-days 800 --sample-every-days 5 \
        --out results/ocean/geometric_stage0/truth
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.0
# ACC channel re-entrant band for the Drake-passage transport (the default
# -65..-45 band is OUTSIDE this idealized channel; matches run_acc_freerun.py).
_ACC_DRAKE_LAT_S = -40.0
_ACC_DRAKE_LAT_N = -22.0


def _seed_freerun_carries(model, state, cfg, grid):
    """Seed the AB2 + rigid-lid carries so the jitted scan keeps a constant
    pytree (None -> Field/array mid-scan would crash ``jax.lax.scan``).

    Mirrors ``run_acc_freerun._run_legoesm`` (lines 151-168): the faithful ACC
    free-run composition is ``outer_integrator="ab2"`` +
    ``barotropic_solver="rigid_lid"``, both of which carry prior-step state that
    must exist as arrays from step 0.
    """
    import jax.numpy as jnp
    from legoesm.core.field import Field

    if cfg.outer_integrator == "ab2" and state.T_incr_prev is None:
        def _z(d):
            return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                         dims=d.dims, units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))

    if cfg.barotropic_solver == "rigid_lid" and state.psi is None:
        rl = model._ensure_rigid_lid_data(state)
        _zV = jnp.zeros((grid.n_lat + 1, grid.n_lon + 1), dtype=state.u.data.dtype)
        _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                               dpsin=_zI, dpsin_prev=_zI)
    return state


def _layer_thickness_np(state, z_coord):
    from legoesm.ocean.vertical import compute_layer_thickness
    return np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord))


def _held_out_metrics(state, z_coord, grid, *, rho_0, g_val):
    """ACC transport [Sv] + total KE [J] via the SHARED diagnostics (the
    held-out early-stopping monitor; NOT part of the ETKI observation vector).

    ``rho_0``/``g_val`` are the recipe's Veros constants so the KE monitor uses
    the SAME ρ₀/g the model integrates with (adversarial-review M2)."""
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
    from legoesm.ocean.budgets import compute_energy_budget

    u = np.asarray(state.u.data)
    mask = np.asarray(state.land_mask.data)
    h = _layer_thickness_np(state, z_coord)
    lat = np.degrees(np.asarray(grid.lat))

    psi = np.asarray(barotropic_streamfunction(u, h, mask, grid))   # (n_lat,n_lon) [Sv]
    band = (lat >= _ACC_DRAKE_LAT_S) & (lat <= _ACC_DRAKE_LAT_N)
    acc_T = (float(psi[band, :].max() - psi[band, :].min())
             if band.any() else float("nan"))
    eb = compute_energy_budget(state, z_coord, grid_type="latlon", grid=grid,
                               rho_0=rho_0, g_val=g_val)
    return acc_T, float(eb.KE)


class _Accumulator:
    """Streaming sum / sum-of-squares for time-mean + temporal-std maps (fp64).

    Two-moment streaming (mean = Σx/n, std = sqrt(max(Σx²/n − mean², 0))) is
    adequate in float64 for the O(100) samples of the averaging window; the
    ``maximum(·, 0)`` guards the catastrophic-cancellation floor.
    """

    def __init__(self, fields):
        self._fields = tuple(fields)
        self._sum: dict[str, np.ndarray] = {}
        self._sumsq: dict[str, np.ndarray] = {}
        self._n = 0

    def add(self, sample: dict[str, np.ndarray]):
        for k in self._fields:
            x = np.asarray(sample[k], dtype=np.float64)
            if k not in self._sum:
                self._sum[k] = np.zeros_like(x)
                self._sumsq[k] = np.zeros_like(x)
            self._sum[k] += x
            self._sumsq[k] += x * x
        self._n += 1

    @property
    def n(self) -> int:
        return self._n

    def finalize(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k in self._fields:
            mean = self._sum[k] / self._n
            var = self._sumsq[k] / self._n - mean * mean
            out[f"mean_{k}"] = mean
            out[f"std_{k}"] = np.sqrt(np.maximum(var, 0.0))
        return out


_OBSERVABLE_FIELDS = ("T", "S", "psi", "eke")


def _sample_maps(state) -> dict[str, np.ndarray]:
    """The observable fields pulled to host: T, S, ψ, depth-integrated EKE.

    The §3 observation vector calls for "T, S, η (or interface-equivalent)". The
    faithful ACC recipe runs the RIGID-LID barotropic solver, under which the
    free surface η ≡ 0 by construction (the barotropic mode is the streamfunction
    ψ, not η) — so η is a degenerate all-zero observable (its temporal std is 0,
    which would blow up the ETKI R^{-1/2} normalization). ψ is the correct
    interface-equivalent: the barotropic transport structure, and the field most
    directly sensitive to GEOMETRIC's ``kappa_u`` barotropic-production term.

    For the GEOMETRIC closure ``state.eke`` IS the depth-integrated 2-D budget
    variable (eke.py:730, ``eke_3d=False``), so it is the EKE map directly.
    """
    return {
        "T": np.asarray(state.T.data, dtype=np.float64),
        "S": np.asarray(state.S.data, dtype=np.float64),
        "psi": np.asarray(state.psi, dtype=np.float64),     # rigid-lid SSH-analog
        "eke": np.asarray(state.eke.data, dtype=np.float64),
    }


def run_stage0(*, spin_years: float, avg_days: float, sample_every_days: float,
               dt: float, geom_overrides: dict[str, float], out_dir: Path):
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    from legoesm.ocean.fidelity.veros_acc_recipe import (
        build_acc_recipe, geometric_eke_config,
    )
    from legoesm.ocean.physics.lateral_mixing.eke import GeometricConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        geom = GeometricConfig(**geom_overrides) if geom_overrides else GeometricConfig()
        recipe = build_acc_recipe(with_surface_forcing=True,
                                  eke_override=geometric_eke_config(geom))
        cfg = recipe.model_config
        model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
        if cfg.coriolis_scheme == "explicit_ab2":
            model.check_coriolis_stability(dt)
        state = _seed_freerun_carries(model, recipe.initial_state, cfg, recipe.grid)
        sf = recipe.wind_forcing

        # 2 tracer steps/day at dt=43200; jit a 1-day block for granular NaN checks.
        steps_per_block = max(1, int(round(_SECONDS_PER_DAY / dt)))

        def _body(st, _):
            return model.step(st, dt, surface_forcing=sf), None

        @partial(jax.jit, static_argnames=("n",))
        def _block(st, n):
            st, _ = jax.lax.scan(_body, st, None, length=n)
            return st

        def _advance_days(state, n_days, *, phase, sample_cb=None,
                          sample_every_days=None):
            """Integrate n_days in 1-day blocks; optional host-side sampling."""
            t0 = time.time()
            last_sample_day = -1e30
            for d in range(1, int(round(n_days)) + 1):
                state = _block(state, steps_per_block)
                jax.block_until_ready(state.u.data)
                if not bool(jnp.all(jnp.isfinite(state.u.data))):
                    raise RuntimeError(
                        f"GEOMETRIC ACC blew up during {phase} at day {d}")
                if (sample_cb is not None
                        and d - last_sample_day >= sample_every_days - 1e-9):
                    sample_cb(state, d)
                    last_sample_day = d
                if d % 30 == 0 or d == int(round(n_days)):
                    umax = float(jnp.max(jnp.abs(state.u.data)))
                    eke = np.asarray(state.eke.data)
                    emax = float(np.nanmax(eke))
                    print(f"  [{phase}] day {d:6d}/{int(round(n_days))} | "
                          f"max|u|={umax:.4f} m/s | max EKE={emax:.3e} | "
                          f"{time.time()-t0:6.0f}s", flush=True)
            return state

        # --- Spin-up to the attractor ---
        spin_days = spin_years * _DAYS_PER_YEAR
        print(f"== GEOMETRIC ACC truth: spin-up {spin_days:.0f} days "
              f"(α={geom.alpha}, c_eps={geom.c_eps_geometric}, "
              f"kappa_u={geom.kappa_u}, kappa_e={geom.kappa_e}, "
              f"rossby={geom.rossby_factor}), dt={dt:.0f}s fp64 ==", flush=True)
        state = _advance_days(state, spin_days, phase="spin")

        # --- Averaging window: accumulate observable maps + held-out series ---
        print(f"== averaging window {avg_days:.0f} days, "
              f"sample every {sample_every_days:.0f} days ==", flush=True)
        acc = _Accumulator(_OBSERVABLE_FIELDS)
        held = {"day": [], "acc_transport_Sv": [], "total_KE_J": []}

        def _sample(state, day):
            acc.add(_sample_maps(state))
            aT, KE = _held_out_metrics(state, recipe.z_coord, recipe.grid,
                                       rho_0=cfg.constants.rho_0,
                                       g_val=cfg.constants.g)
            held["day"].append(float(day + spin_days))
            held["acc_transport_Sv"].append(aT)
            held["total_KE_J"].append(KE)

        state = _advance_days(state, avg_days, phase="avg", sample_cb=_sample,
                              sample_every_days=sample_every_days)
        if acc.n < 2:
            raise RuntimeError(
                f"only {acc.n} observable sample(s) — increase --avg-days or "
                "decrease --sample-every-days")

        # --- Persist truth snapshot for exact-IC members ---
        # Save BOTH the Field leaves AND the plain-array carries (psi/dpsi/dpsin
        # rigid-lid streamfunction + AB2 increments): the exact-IC member protocol
        # restarts from this state, so the barotropic/AB2 carry IS part of the IC.
        # (Without these the member re-seeds them to zero and re-converges the
        # rigid-lid solve over a short transient — adversarial-review M1.)
        snap = {}
        for f in state._fields:
            obj = getattr(state, f)
            if obj is None:
                continue
            if hasattr(obj, "data"):              # Field leaf
                snap[f] = np.asarray(obj.data)
            elif isinstance(obj, jnp.ndarray):    # plain-array carry
                snap[f] = np.asarray(obj)
        np.savez_compressed(out_dir / "truth_snapshot.npz", **snap)

        # --- Persist observables (ETKI vector) + held-out series ---
        obs = acc.finalize()
        obs["held_day"] = np.asarray(held["day"])
        obs["held_acc_transport_Sv"] = np.asarray(held["acc_transport_Sv"])
        obs["held_total_KE_J"] = np.asarray(held["total_KE_J"])
        obs["n_samples"] = np.asarray(acc.n)
        np.savez_compressed(out_dir / "observables.npz", **obs)

        meta = {
            "geometric_true_params": {
                "alpha": geom.alpha, "c_eps_geometric": geom.c_eps_geometric,
                "kappa_u": geom.kappa_u, "kappa_e": geom.kappa_e,
                "rossby_factor": geom.rossby_factor,
            },
            "spin_days": spin_days, "avg_days": avg_days,
            "sample_every_days": sample_every_days, "dt_s": dt,
            "n_samples": acc.n,
            "acc_transport_mean_Sv": float(np.mean(held["acc_transport_Sv"])),
            "total_KE_mean_J": float(np.mean(held["total_KE_J"])),
            "grid": [int(recipe.grid.n_lat), int(recipe.grid.n_lon)],
        }
        (out_dir / "truth_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n== TRUTH WRITTEN to {out_dir} ==")
        print(f"   observables: {sorted(k for k in obs)}")
        print(f"   ACC transport (held-out mean): "
              f"{meta['acc_transport_mean_Sv']:.1f} Sv")
        print(f"   total KE (held-out mean): {meta['total_KE_mean_J']:.3e} J")
        return meta
    finally:
        set_policy(_prev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spin-years", type=float, default=1.5,
                    help="Spin-up duration [model years] before averaging (default 1.5).")
    ap.add_argument("--avg-days", type=float, default=800.0,
                    help="Averaging-window length [model days] (default 800).")
    ap.add_argument("--sample-every-days", type=float, default=5.0,
                    help="Observable sampling cadence within the window [days].")
    ap.add_argument("--dt", type=float, default=43200.0,
                    help="Tracer timestep [s] (default 43200 = Veros dt_tracer; "
                         "dt_mom = dt/9 via the recipe's dt_mom_ratio).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for truth_snapshot.npz / observables.npz.")
    ap.add_argument("--smoke", action="store_true",
                    help="Wiring + geometric-stability smoke: 5-day spin, 5-day "
                         "window, daily sampling (overrides the duration args).")
    # GeometricConfig 'true' parameter overrides (default = paper-calibrated).
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--c-eps-geometric", type=float, default=None)
    ap.add_argument("--kappa-u", type=float, default=None)
    ap.add_argument("--kappa-e", type=float, default=None)
    ap.add_argument("--rossby-factor", type=float, default=None)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    geom_overrides = {}
    for cli, field in (("alpha", "alpha"), ("c_eps_geometric", "c_eps_geometric"),
                       ("kappa_u", "kappa_u"), ("kappa_e", "kappa_e"),
                       ("rossby_factor", "rossby_factor")):
        v = getattr(args, cli)
        if v is not None:
            geom_overrides[field] = v

    if args.smoke:
        spin_years, avg_days, sample_every = 5.0 / _DAYS_PER_YEAR, 5.0, 1.0
    else:
        spin_years, avg_days, sample_every = (args.spin_years, args.avg_days,
                                              args.sample_every_days)

    run_stage0(spin_years=spin_years, avg_days=avg_days,
               sample_every_days=sample_every, dt=args.dt,
               geom_overrides=geom_overrides, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
