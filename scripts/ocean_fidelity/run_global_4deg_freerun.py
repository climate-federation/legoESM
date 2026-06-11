"""global_4deg transfer test: legoESM free run vs the banked Veros oracle.

Runs the legoESM Veros-global_4deg recipe
(``legoesm.ocean.fidelity.veros_global_4deg_recipe``) forward with the REAL
Veros forcing assets (``~/.veros/assets/global_4deg/``) and records yearly
metrics in the SAME schema as the banked Veros oracle runner
(``.physics-validator/transfer_global_4deg/run_10yr_4deg.py`` →
``stock_10yr_4deg.json``), so per-year ratios are apples-to-apples.

Data prep replicates Veros ``set_initial_conditions`` EXACTLY (SCOPING §C):
file ``.T`` transpose + z-flip; qnec sentinel (≤ −1e10 → 0); qnet sign flip +
sentinel + annual-mean imbalance removal over ALL interior cells then wet-mask
(must reproduce ``mean_flux = 7.860702e-02 W/m²`` — ASSERTED); T/S IC masked
by the active cells; Trenberth tau_x/tau_y from the forcing file (NOT the
ecmwf TX3/TY3).

Monthly forcing is interpolated PER STEP INSIDE the jitted scan
(SegmentForcing pattern): the 12-month stacks are closure-constant device
arrays; the month indices/weights are traced from the model time via the
Veros ``get_periodic_interval`` replica (360-day year, month-START anchoring)
— constant pytree, no per-month recompiles.  Veros calls ``set_forcing`` at
the START of each tracer step with ``vs.time`` = k·dt_tracer (time is
incremented after the step), so step k uses t = k·86400 s.  Wind stress is
passed as ``tau_x = −taux`` (the external seam flips the atmosphere-convention
stress back to Veros's on-ocean ``+taux``; ACC seam).  Heat/salt go through
the ``flux_feedback`` channels (q_prescribed / q_feedback / T_feedback_target
/ S_restore_target); ``q_net`` stays None (trace-time double-count guard).
The TKE surface forcing |τ/ρ₀|^1.5 flows from the tau channels automatically.

Mimicry residuals (documented, harness-level — SCOPING §A.1/§A.5): Veros
loads the T-point tau file into its u-face ``surface_taux`` WITHOUT
interpolation (a half-cell shift Veros itself makes) and averages u-face tau
back to T points for the TKE forcing; legoESM interpolates T-point tau to
u-faces and feeds the T-point tau to TKE directly.  Half-cell smoothing,
climate-negligible.

Metric conventions (mirroring the oracle runner): volume weights are the
Veros FULL-CELL ``dxt·dyt·cos(yt)·dzt·maskT`` (the snapped partial-cell
coordinate is full-cell by construction); u/v enter the KE sum at their
native faces indexed by the T cell exactly as Veros's ``u[i,j,k]`` (east/
north face of cell (i,j)); ψ is the rigid-lid prognostic streamfunction
[m³/s → Sv] with the SAME boundary gauge (ψ=0 mainland + per-island
constants) — any residual is the documented ψ-gauge note from the ACC
program.  mean_tke/mean_eke: legoESM carries both on the nlev−1 INTERIOR
interfaces while Veros's W-grid has nz levels (incl. the surface W point), so
the legoESM mean uses the cell-below weights (area·dzt[k+1]·maskT[k+1]) and
misses Veros's surface W contribution — a documented convention residual on
a diagnostic ratio (Veros's own banked mean_tke even goes NEGATIVE, so this
metric is shape-compared only).

Usage::

    # 10-day smoke (build + stability; CPU ok):
    JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/ocean_fidelity/run_global_4deg_freerun.py --smoke

    # 10-year run on GPU 0 (GPU 1 is occupied!):
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 nohup .venv/bin/python \
        scripts/ocean_fidelity/run_global_4deg_freerun.py --years 10 \
        --out .physics-validator/transfer_global_4deg/legoesm_10yr_4deg.json &
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import numpy as np

ASSETS = os.path.expanduser("~/.veros/assets/global_4deg")
FORCING_NC = os.path.join(ASSETS, "forcing_4deg_global_open_itf.nc")
ECMWF_NC = os.path.join(ASSETS, "ecmwf_4deg_monthly_nc4.nc")

# Veros's own logged annual-mean heat-flux imbalance (probe-verified to all
# printed digits) — the data-prep equality gate.
EXPECTED_MEAN_FLUX = 7.860702e-02   # [W/m²]
_MEAN_FLUX_ATOL = 1.0e-8

DAYS_PER_YEAR = 360.0               # Veros global_4deg forcing year
SECONDS_PER_DAY = 86400.0


def _read_forcing(var):
    import h5netcdf
    with h5netcdf.File(FORCING_NC, "r") as f:
        return np.array(f.variables[var]).T      # -> (x, y[, z/months])


def load_and_prepare():
    """netCDF reads + the Veros-verbatim data prep.  Returns (recipe,
    forcing_stacks) where each stack is (12, n_lat, n_lon) in legoESM layout
    (wall rows zero)."""
    import h5netcdf
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        build_global_4deg_recipe,
        mask_forcing_sentinel,
        remove_qnet_imbalance,
        replicate_veros_kbot,
        veros_area_t,
        veros_xy_to_legoesm,
        NY,
    )

    bath = _read_forcing("bathymetry")                      # (x, y)
    salt_xyz = _read_forcing("salinity")[:, :, ::-1]        # Veros z-order
    temp_xyz = _read_forcing("temperature")[:, :, ::-1]

    recipe = build_global_4deg_recipe(bath, salt_xyz, temp_xyz)
    n_lat = recipe.grid.n_lat

    # --- wind stress (Trenberth, from the forcing file; T-point data) ---
    taux = _read_forcing("tau_x")                           # (x, y, 12)
    tauy = _read_forcing("tau_y")

    # --- heat-flux pieces (Veros set_initial_conditions order) ---
    with h5netcdf.File(ECMWF_NC, "r") as f:
        qnec = np.array(f.variables["Q3"]).T                # (x, y, 12)
    qnec = mask_forcing_sentinel(qnec)

    qnet = mask_forcing_sentinel(-_read_forcing("q_net"))   # sign flip FIRST
    yt = -78.0 + 4.0 * np.arange(NY)                        # Veros yt[2:-2]
    area_col = veros_area_t(yt)                             # (y,)
    area_xy = np.broadcast_to(area_col[None, :], bath.shape)
    kbot = replicate_veros_kbot(bath, salt_xyz)
    wet_surf = (kbot > 0).astype(np.float64)                # maskT surface
    qnet, mean_flux = remove_qnet_imbalance(qnet, area_xy, wet_surf)
    if abs(mean_flux - EXPECTED_MEAN_FLUX) > _MEAN_FLUX_ATOL:
        raise AssertionError(
            f"qnet imbalance {mean_flux:.6e} W/m² != Veros's logged "
            f"{EXPECTED_MEAN_FLUX:.6e} (data-prep drift)")
    print(f"  qnet annual-mean imbalance removed: {mean_flux:.6e} W/m² "
          f"(== Veros log)")

    # --- SST / SSS monthly climatology ---
    sst = _read_forcing("sst")
    sss = _read_forcing("sss")

    def to_stack(arr_xy12):
        """(x, y, 12) -> (12, n_lat, n_lon) legoESM layout, wall rows 0."""
        months = [veros_xy_to_legoesm(arr_xy12[:, :, m], n_lat)
                  for m in range(arr_xy12.shape[2])]
        return jnp.asarray(np.stack(months, axis=0))

    stacks = {
        "taux": to_stack(taux), "tauy": to_stack(tauy),
        "qnet": to_stack(qnet), "qnec": to_stack(qnec),
        "sst": to_stack(sst), "sss": to_stack(sss),
    }
    return recipe, stacks


# ---------------------------------------------------------------------------
# Metrics — the oracle runner's schema (run_10yr_4deg.py:metrics)
# ---------------------------------------------------------------------------


def compute_metrics(state, recipe) -> dict:
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_DDZ, veros_area_t,
    )

    ia = np.asarray(recipe.z_coord.is_active)[1:-1, :, :]   # interior (40,90,15)
    lat = np.degrees(np.asarray(recipe.grid.lat))[1:-1]
    area = veros_area_t(lat)[:, None]                        # (40,1)
    dz = GLOBAL4_DDZ                                         # full-cell (snap)
    vol = area[:, :, None] * dz[None, None, :] * ia          # (40,90,15)

    u = np.asarray(state.u.data)        # (42, 91, 15)
    v = np.asarray(state.v.data)        # (43, 90, 15)
    T = np.asarray(state.T.data)[1:-1]
    S = np.asarray(state.S.data)[1:-1]

    # u/v at the T-cell's east/north face — Veros's u[i,j,k]/v[i,j,k].
    u_cell = u[1:-1, 1:, :]             # east faces of interior cells
    v_cell = v[2:-1, :, :]              # north faces of interior rows 1..40

    # Veros maskU/maskV (min rule) for max|u|.
    ia_e = np.minimum(ia, np.roll(ia, -1, axis=1))           # east-face wet
    rho0 = float(recipe.model_config.rho_0)

    def wmean(x, w):
        sw = w.sum()
        return float((x * w).sum() / max(sw, 1e-30))

    speed2 = u_cell ** 2 + v_cell ** 2

    # ψ: the rigid-lid prognostic streamfunction [m³/s] on vertices, same
    # boundary gauge as Veros (ψ=0 mainland + island constants).
    psi = np.asarray(state.psi) if state.psi is not None else np.zeros((1,))
    psi_min, psi_max = float(psi.min() / 1e6), float(psi.max() / 1e6)

    # tke/eke: interior interfaces (nlev-1) with cell-below Veros weights
    # (see module docstring — documented convention residual).
    out_we = {}
    for name in ("tke", "eke"):
        fld = getattr(state, name)
        if fld is None:
            out_we[f"mean_{name}"] = float("nan")
            continue
        e = np.asarray(fld.data)[1:-1, :, :]                 # (40,90,14)
        w_if = vol[:, :, 1:]                                  # cell below
        out_we[f"mean_{name}"] = wmean(e, w_if)

    return dict(
        psi_min_sv=psi_min,
        psi_max_sv=psi_max,
        psi_range_sv=psi_max - psi_min,
        total_ke_j=0.5 * rho0 * float((speed2 * vol).sum()),
        vol_mean_T=wmean(T, vol),
        vol_mean_S=wmean(S, vol),
        max_abs_u=float(np.max(np.abs(u_cell * ia_e))),
        sfc_T_mean=wmean(T[..., :1], vol[..., :1]),          # surface layer
        mean_tke=out_we["mean_tke"],
        mean_eke=out_we["mean_eke"],
        finite=bool(np.isfinite(u).all() and np.isfinite(T).all()
                    and np.isfinite(S).all()),
    )


# ---------------------------------------------------------------------------
# Forward integration
# ---------------------------------------------------------------------------


def run(years: float, out_path: str | None, compare_path: str | None,
        tracer_advection: str | None = None,
        tke_advection: str | None = None,
        coriolis: str | None = None,
        eke_mode: str | None = None) -> int:
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        DT_TRACER_S, get_periodic_interval_weights,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    t0 = time.perf_counter()
    recipe, stacks = load_and_prepare()
    cfg = recipe.model_config
    # ---- STABILITY FALLBACKS (deviations from the Veros-faithful recipe) ----
    # The fully faithful composition (centered tracers + TKE superbee
    # advection) is NOT yet stable in legoESM on this variable-bathymetry
    # global domain: the unlimited centered tracer advection seeds surface
    # grid noise that the GM/Redi+EKE chain amplifies into local cold spots
    # (NaN ~day 170), and the W-grid TKE superbee advection goes unstable on
    # the resulting transient w (wCFL > 1; NaN ~day 19).  Veros runs the same
    # schemes stably because its spin-up stays ~100x calmer (probe-measured
    # max|w| 4.6e-4 vs our 5e-3 m/s) — root cause under investigation.
    # The flags below are LOUD deviations the comparison must report.
    if tracer_advection is not None and tracer_advection != cfg.tracer_advection:
        print(f"!! STABILITY FALLBACK: tracer_advection="
              f"{tracer_advection!r} (Veros-faithful recipe: "
              f"{cfg.tracer_advection!r})")
        cfg = cfg._replace(tracer_advection=tracer_advection)
    vm = cfg.physics.vertical_mixing
    if (tke_advection is not None
            and tke_advection != vm.tke.advection_scheme):
        print(f"!! STABILITY FALLBACK: TKE advection_scheme="
              f"{tke_advection!r} (Veros-faithful recipe: "
              f"{vm.tke.advection_scheme!r})")
        cfg = cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=vm._replace(
                tke=vm.tke._replace(advection_scheme=tke_advection))))
    if coriolis is not None and coriolis != cfg.coriolis_scheme:
        print(f"!! ABLATION: coriolis_scheme={coriolis!r} "
              f"(Veros-faithful recipe: {cfg.coriolis_scheme!r})")
        cfg = cfg._replace(coriolis_scheme=coriolis)
    if eke_mode == "vanilla":
        print("!! ABLATION: EKE sources vanilla (parameterized GM source, "
              "no K_diss_h recycling)")
        gm = cfg.gm_redi
        cfg = cfg._replace(gm_redi=gm._replace(eke=gm.eke._replace(
            source_kdiss_h=False, kdiss_h_flux_form=False,
            gm_source_mode="parameterized")))
        _tke = cfg.physics.vertical_mixing.tke._replace(source_eke_diss=False)
        cfg = cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=cfg.physics.vertical_mixing._replace(tke=_tke)))
    elif eke_mode == "off":
        print("!! ABLATION: EKE OFF (constant kappa_GM = kappa_Redi = 1000)")
        cfg = cfg._replace(gm_redi=cfg.gm_redi._replace(eke=None))
        _tke = cfg.physics.vertical_mixing.tke._replace(source_eke_diss=False)
        cfg = cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=cfg.physics.vertical_mixing._replace(tke=_tke)))
    recipe = recipe._replace(model_config=cfg)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    # explicit_ab2 Coriolis margin: |f|·dt_mom ≈ 0.26 at 78° — report it.
    model.check_coriolis_stability(DT_TRACER_S)

    state = recipe.initial_state
    # AB2 increment carry seeded to zero (constant scan pytree; the first step
    # is the 1.6× forward-Euler seed) — run_acc_freerun pattern.
    if state.T_incr_prev is None:
        _z = lambda d: Field(data=jnp.zeros_like(d.data),
                             name=d.name + "_incr_prev", dims=d.dims,
                             units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    # Rigid lid: static island data (host flood-fill) + ψ carry.
    rl = model._ensure_rigid_lid_data(state)
    print(f"  rigid-lid islands found: {rl.nisle} (Veros oracle: 5)")
    if state.psi is None:
        _zV = jnp.zeros((recipe.grid.n_lat + 1, recipe.grid.n_lon + 1),
                        dtype=state.u.data.dtype)
        _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                               dpsin=_zI, dpsin_prev=_zI)

    taux, tauy = stacks["taux"], stacks["tauy"]
    qnet, qnec = stacks["qnet"], stacks["qnec"]
    sst, sss = stacks["sst"], stacks["sss"]

    def _forcing_at(t_s):
        """Compose the per-step OceanSurfaceForcing from the traced model
        time (Veros set_forcing_kernel: interpolate, then the scheme applies
        feedback + ice mask on the in-step T)."""
        n1, f1, n2, f2 = get_periodic_interval_weights(t_s)

        def interp(stack):
            return (f1 * jnp.take(stack, n1, axis=0)
                    + f2 * jnp.take(stack, n2, axis=0))

        return OceanSurfaceForcing(
            # external seam flips atmosphere-convention tau: pass −taux so the
            # ocean feels Veros's +taux (ACC seam; TKE uses |tau| only).
            tau_x=-interp(taux),
            tau_y=-interp(tauy),
            q_prescribed=interp(qnet),
            q_feedback=interp(qnec),
            T_feedback_target=interp(sst),
            S_restore_target=interp(sss),
            # q_net stays None — the flux_feedback double-count guard.
        )

    @partial(jax.jit, static_argnames=("n",))
    def _block(st, step0, n):
        def body(carry, k):
            t = (step0 + k).astype(jnp.float64) * DT_TRACER_S
            sf = _forcing_at(t)
            return model.step(carry, DT_TRACER_S, surface_forcing=sf), None
        st, _ = jax.lax.scan(body, st, jnp.arange(n))
        return st

    total_days = years * DAYS_PER_YEAR
    total_steps = int(round(total_days * SECONDS_PER_DAY / DT_TRACER_S))
    block_steps = 30                                  # 30 days per jit block
    year_steps = int(round(DAYS_PER_YEAR * SECONDS_PER_DAY / DT_TRACER_S))

    print(f"== legoESM global_4deg free run: {total_days:.0f} days "
          f"({total_steps} steps, dt_tracer={DT_TRACER_S:.0f} s, "
          f"dt_mom={DT_TRACER_S/cfg.dt_mom_ratio:.0f} s) ==")

    yearly = []
    done = 0
    t_run0 = time.perf_counter()
    while done < total_steps:
        n = min(block_steps, total_steps - done)
        state = _block(state, jnp.asarray(done, dtype=jnp.int64), n)
        jax.block_until_ready(state.u.data)
        done += n
        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            print(f"!! legoESM blew up after {done} steps "
                  f"({done * DT_TRACER_S / SECONDS_PER_DAY:.0f} days)")
            m = compute_metrics(state, recipe)
            m["year"] = done * DT_TRACER_S / SECONDS_PER_DAY / DAYS_PER_YEAR
            m["blowup"] = True
            yearly.append(m)
            break
        day = done * DT_TRACER_S / SECONDS_PER_DAY
        if done % year_steps == 0 or done == total_steps:
            m = compute_metrics(state, recipe)
            m["year"] = done / year_steps if done % year_steps == 0 else (
                done * DT_TRACER_S / SECONDS_PER_DAY / DAYS_PER_YEAR)
            m["wall_s"] = round(time.perf_counter() - t_run0, 1)
            yearly.append(m)
            print(f"[day {day:6.0f}] KE={m['total_ke_j']:.4e} "
                  f"psi=[{m['psi_min_sv']:.1f},{m['psi_max_sv']:.1f}] Sv "
                  f"T={m['vol_mean_T']:.4f} S={m['vol_mean_S']:.4f} "
                  f"max|u|={m['max_abs_u']:.3f} eke={m['mean_eke']:.3e} "
                  f"finite={m['finite']} wall={m['wall_s']:.0f}s", flush=True)
        elif done % (block_steps * 3) == 0:
            umax = float(jnp.max(jnp.abs(state.u.data)))
            print(f"  day {day:6.0f}: max|u|={umax:.4f} m/s "
                  f"({(time.perf_counter()-t_run0):.0f}s)", flush=True)

    out = dict(
        years=years,
        yearly=yearly,
        total_wall_h=round((time.perf_counter() - t0) / 3600.0, 3),
        config=dict(
            dt_tracer=DT_TRACER_S, dt_mom_ratio=cfg.dt_mom_ratio,
            outer_integrator=cfg.outer_integrator,
            barotropic_solver=cfg.barotropic_solver,
            coriolis_scheme=cfg.coriolis_scheme, ab2_scope=cfg.ab2_scope,
            eos=cfg.eos, nisle=int(rl.nisle),
        ),
    )
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {out_path}")

    if compare_path and yearly:
        _print_comparison(yearly, compare_path)
    return 0 if (yearly and yearly[-1]["finite"]) else 1


def _print_comparison(yearly, compare_path):
    with open(compare_path) as f:
        ref = json.load(f)["yearly"]
    keys = ("total_ke_j", "psi_min_sv", "psi_max_sv", "psi_range_sv",
            "vol_mean_T", "vol_mean_S", "max_abs_u", "mean_eke")
    print(f"\n== per-year ratios (legoESM / Veros oracle) ==")
    hdr = "year " + " ".join(f"{k:>13s}" for k in keys)
    print(hdr)
    for m in yearly:
        yr = m.get("year")
        if yr is None or float(yr) != int(float(yr)):
            continue
        rv = next((r for r in ref if r.get("year") == int(float(yr))), None)
        if rv is None:
            continue
        yr = int(float(yr))
        cells = []
        for k in keys:
            denom = rv[k]
            cells.append(f"{m[k] / denom:13.3f}" if denom else f"{'n/a':>13s}")
        print(f"{int(yr):4d} " + " ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true",
                    help="10-day stability smoke (overrides --years)")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    ap.add_argument("--compare", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        ".physics-validator/transfer_global_4deg/stock_10yr_4deg.json"),
        help="banked Veros oracle JSON for the per-year ratio table")
    ap.add_argument("--tracer-advection", default=None,
                    choices=("centered", "tvd"),
                    help="Override the recipe's tracer advection (stability "
                         "fallback: 'tvd'; Veros-faithful: 'centered').")
    ap.add_argument("--tke-advection", default=None, choices=("none", "superbee"),
                    help="Override the TKE advection (stability fallback: "
                         "'none'; Veros-faithful: 'superbee').")
    ap.add_argument("--coriolis", default=None,
                    choices=("explicit_ab2", "matsuno_split"),
                    help="Override the Coriolis scheme (ablation diagnostic; "
                         "Veros-faithful recipe: explicit_ab2).")
    ap.add_argument("--eke", default=None, choices=("vanilla", "off"),
                    help="EKE ablation diagnostics: 'vanilla' = parameterized "
                         "GM source, no K_diss_h recycling; 'off' = constant "
                         "kappa_GM=kappa_Redi=1000 (eke=None).")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

    years = (10.0 / DAYS_PER_YEAR) if args.smoke else args.years
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        return run(years, args.out,
                   None if args.smoke else args.compare,
                   tracer_advection=args.tracer_advection,
                   tke_advection=args.tke_advection,
                   coriolis=args.coriolis,
                   eke_mode=args.eke)
    finally:
        set_policy(prev)


if __name__ == "__main__":
    raise SystemExit(main())
