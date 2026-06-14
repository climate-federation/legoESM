"""global_flexible transfer test: legoESM free run vs the rescaled Veros oracle.

Runs the legoESM Veros-global_flexible recipe
(``legoesm.ocean.fidelity.veros_global_flexible_recipe``) forward with the
REAL Veros assets (``~/.veros/assets/global_flexible/``: ETOPO5 + the 1°
interpolated forcing file) at the documented comparison resolution
(90×40×20, dt 1800/14400 — see the recipe module docstring for the rescale
judgments), and records yearly metrics in the SAME schema as the banked
oracle runner (``.physics-validator/transfer_global_flexible/
run_oracle_flexible.py`` → ``stock_10yr_flexible.json``).

Data prep replicates the Veros setup EXACTLY at the rescaled resolution
(set_topography / set_initial_conditions / set_forcing_kernel — all via the
recipe's transcribed pure helpers): ETOPO5 clamp+gaussian+threshold+lon-shift
+nearest-interp, kbot = 1+argmin|z−zt| + the marginal-sea morphology,
trilinear T/S IC, bilinear monthly forcing, the load-time sign flips
(qnet/qsol NEGATED), surface-mask multiplication, and the MIT-grid one-cell
tau shift.  VERIFICATION GATE (--verify-against): the prepared kbot must be
BIT-IDENTICAL to the live oracle's ``vs.kbot`` (and zt/dyt/maskT/forcing
stacks to float tolerance) from the oracle's ``*_setup_arrays.npz`` — the
proven 4deg gate, extended to the stretched grid.

Heat-ownership (the new q_solar channel): Veros keeps the SOLAR-INCLUSIVE
total in qnet and redistributes with the pen(0)=0 zero-column-sum profile;
legoESM passes ``q_prescribed = qnet − qsol`` (non-solar remainder) +
``q_solar = qsol`` through ``flux_feedback(penetrative_shortwave=True)``
(I(0)=1 full-column deposit) — cell-by-cell algebraically identical (see
``OceanSurfaceForcing.q_solar``).  The ice mask (evaluated on the total flux)
gates surface AND solar column inside the scheme.

Monthly forcing is interpolated PER STEP INSIDE the jitted scan from the
traced model time (SegmentForcing pattern; Veros ``get_periodic_interval``
replica, 360-day year, month-START anchoring; step k uses t = k·dt_tracer).
Wind stress passes as ``tau_x = −taux`` (the external seam flips back to
Veros's on-ocean ``+taux``); the same half-cell u-face/T-point tau mimicry
note as the 4deg harness applies (Veros loads the shifted T-point data into
its u-face ``surface_taux`` without interpolation; legoESM interpolates
T-point→faces and feeds T-point tau to TKE).

Metric conventions mirror the oracle runner on the STRETCHED grid: volume
weights ``dxt·dyt(row)·cos(yt)·dzt·maskT`` (full-cell by the kbot snap);
u/v at their native faces indexed by the T cell; ψ in Sv with the same
boundary gauge; mean_tke/mean_eke with the documented nlev−1 interior-
interface convention residual (shape-compare only).

Usage::

    # 10-day smoke (CPU ok):
    JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/\
run_global_flexible_freerun.py --smoke \
        --verify-against .physics-validator/transfer_global_flexible/\
stock_10yr_flexible_setup_arrays.npz

    # 10-year run on GPU 1:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 nohup python \
        scripts/validate/ocean_fidelity/run_global_flexible_freerun.py \
        --years 10 --out .physics-validator/transfer_global_flexible/\
legoesm_10yr_flexible.json &
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import numpy as np

ASSETS = os.path.expanduser("~/.veros/assets/global_flexible")
TOPO_NC = os.path.join(ASSETS, "ETOPO5_Ice_g_gmt4.nc")
FORCING_NC = os.path.join(ASSETS, "forcing_1deg_global_interpolated.nc")

DAYS_PER_YEAR = 360.0               # Veros forcing year
SECONDS_PER_DAY = 86400.0


def _read_nc(path, var):
    import h5netcdf
    with h5netcdf.File(path, "r") as f:
        return np.array(f.variables[var], dtype="float").T   # Veros _get_data


def load_and_prepare(verify_against: str | None = None):
    """netCDF reads + the Veros-verbatim data prep at the comparison
    resolution.  Returns (recipe, stacks); each stack is (12, n_lat, n_lon)
    in legoESM layout (wall rows zero)."""
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        NZ,
        build_global_flexible_recipe,
        global_flexible_dzt_veros,
        prepare_global_flexible_topography,
        replicate_veros_kbot_flexible,
        veros_full_axes,
        veros_interpolate,
    )
    from legoesm.ocean.fidelity.veros_layout_common import (
        veros_mit_tau_shift,
        veros_xy_to_legoesm,
    )
    from legoesm.ocean.fidelity.veros_state_bridge import (
        veros_u_centered_z_centres,
    )

    xt_full, yt_full, _yu_full = veros_full_axes()
    xt_i, yt_i = xt_full[2:-2], yt_full[2:-2]
    dz_ref = global_flexible_dzt_veros()[::-1]
    zt = veros_u_centered_z_centres(dz_ref)[::-1]      # Veros k=0 deepest

    # ---- set_topography ----
    topo_x = _read_nc(TOPO_NC, "x")
    topo_y = _read_nc(TOPO_NC, "y")
    topo_z = _read_nc(TOPO_NC, "z")
    z_interp = prepare_global_flexible_topography(
        topo_x, topo_y, topo_z, xt_i, yt_i)
    kbot = replicate_veros_kbot_flexible(z_interp)
    wet_surf = (kbot > 0).astype(np.float64)           # maskT surface (x, y)

    # ---- set_initial_conditions: data-subset slicing (verbatim) ----
    xt_forc = _read_nc(FORCING_NC, "xt")
    yt_forc = _read_nc(FORCING_NC, "yt")
    zt_forc = _read_nc(FORCING_NC, "zt")[::-1]
    sl_x = slice(
        max(0, int(np.argmax(xt_forc >= xt_full.min())) - 1),
        len(xt_forc) - max(0, int(np.argmax(xt_forc[::-1] <= xt_full.max())) - 1))
    sl_y = slice(
        max(0, int(np.argmax(yt_forc >= yt_full.min())) - 1),
        len(yt_forc) - max(0, int(np.argmax(yt_forc[::-1] <= yt_full.max())) - 1))
    xt_forc = xt_forc[sl_x]
    yt_forc = yt_forc[sl_y]

    def get_sub(var):
        return _read_nc(FORCING_NC, var)[sl_x, sl_y, ...]

    # T/S IC: trilinear to (xt, yt, zt)
    t_grid = (xt_i, yt_i, zt)
    temp_data = veros_interpolate(
        (xt_forc, yt_forc, zt_forc), get_sub("temperature")[..., ::-1], t_grid)
    salt_data = veros_interpolate(
        (xt_forc, yt_forc, zt_forc), get_sub("salinity")[..., ::-1], t_grid)

    recipe = build_global_flexible_recipe(z_interp, temp_data, salt_data)

    # ---- monthly forcing fields (x, y, 12), Veros interp + signs/masks ----
    months = np.arange(12)
    time_grid = (xt_i, yt_i, months)

    def interp_t(var):
        return veros_interpolate(
            (xt_forc, yt_forc, months), get_sub(var), time_grid)

    taux = interp_t("tau_x")
    tauy = interp_t("tau_y")
    qnet = -interp_t("q_net") * wet_surf[:, :, None]     # sign flip + maskT
    qnec = interp_t("dqdt") * wet_surf[:, :, None]
    qsol = -interp_t("swf") * wet_surf[:, :, None]       # sign flip + maskT
    sst = interp_t("sst") * wet_surf[:, :, None]
    sss = interp_t("sss") * wet_surf[:, :, None]

    # MIT-grid one-cell tau shift (set_forcing_kernel), once at prep time.
    taux_s, tauy_s = veros_mit_tau_shift(taux, tauy, x_cyclic=True)

    # ---- verification gate vs the live oracle setup dump ----
    if verify_against:
        ref = np.load(verify_against)
        kb_ref = ref["kbot"][2:-2, 2:-2].astype(kbot.dtype)
        if not np.array_equal(kbot, kb_ref):
            n_bad = int((kbot != kb_ref).sum())
            raise AssertionError(
                f"kbot mismatch vs oracle: {n_bad} cells differ "
                f"(harness hist {np.bincount(kbot.ravel(), minlength=NZ + 1)}, "
                f"oracle hist {np.bincount(kb_ref.ravel(), minlength=NZ + 1)})")
        np.testing.assert_allclose(zt, ref["zt"], rtol=0, atol=1e-9,
                                   err_msg="zt mismatch vs oracle")
        np.testing.assert_allclose(yt_full, ref["yt"], rtol=0, atol=1e-9,
                                   err_msg="yt mismatch vs oracle")
        # dyt: oracle stores metres (post degtom); grid.dy is the 2-cell span.
        degtom = float(recipe.model_config.constants.R_earth) * np.pi / 180.0
        # Compare the PURE dyt construction (bit-exact, fp64) rather than a
        # float32 grid.dy roundtrip - the roundtrip only passes under the fp64
        # policy and would spuriously fail (~8e-8) for any caller that has not
        # set it (review hardening).
        from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
            global_flexible_dyt_deg as _dyt_deg,
        )
        np.testing.assert_allclose(
            _dyt_deg(ny=ref["dyt"][2:-2].shape[0]),
            ref["dyt"][2:-2] / degtom,
            rtol=1e-12, err_msg="dyt mismatch vs oracle")
        # maskT (lat, lon, k=0 surface) vs oracle maskT (x, y, k=0 deepest)
        ia = np.asarray(recipe.z_coord.is_active)[1:-1]      # (NY, NX, NZ)
        maskT_ref = np.transpose(
            ref["maskT"][2:-2, 2:-2, ::-1], (1, 0, 2))       # → (y, x, z surf-first)
        if not np.array_equal(ia.astype(bool), maskT_ref.astype(bool)):
            raise AssertionError("maskT mismatch vs oracle")
        # forcing stacks (float-tolerance: same scipy ops, expect ~exact)
        for name, mine in (("taux", taux), ("tauy", tauy), ("qnet", qnet),
                           ("qnec", qnec), ("qsol", qsol), ("t_star", sst),
                           ("s_star", sss)):
            theirs = ref[name][2:-2, 2:-2, :]
            d = float(np.max(np.abs(mine - theirs)))
            if d > 1e-10:
                raise AssertionError(f"forcing stack {name} differs from "
                                     f"oracle by max {d:.3e}")
        kb_hist = np.bincount(kbot.ravel(), minlength=NZ + 1)
        print(f"  VERIFIED vs oracle dump: kbot BIT-IDENTICAL "
              f"(hist {list(kb_hist)}), zt/yt/dyt/maskT/forcing stacks match")

    def to_stack(arr_xy12):
        """(x, y, 12) → (12, n_lat, n_lon) legoESM layout, wall rows 0."""
        months_ll = [veros_xy_to_legoesm(arr_xy12[:, :, m])
                     for m in range(arr_xy12.shape[2])]
        return jnp.asarray(np.stack(months_ll, axis=0))

    stacks = {
        "taux": to_stack(taux_s), "tauy": to_stack(tauy_s),
        # Heat-ownership: prescribed channel = NON-solar remainder.
        "q_prescribed": to_stack(qnet - qsol),
        "q_solar": to_stack(qsol),
        "qnec": to_stack(qnec),
        "sst": to_stack(sst), "sss": to_stack(sss),
    }
    return recipe, stacks


# ---------------------------------------------------------------------------
# Metrics — the oracle runner's schema (run_oracle_flexible.py:metrics)
# ---------------------------------------------------------------------------


def compute_metrics(state, recipe) -> dict:
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        NX, global_flexible_dyt_deg, global_flexible_dzt_veros,
    )
    from legoesm.ocean.fidelity.veros_layout_common import veros_area_t

    ia = np.asarray(recipe.z_coord.is_active)[1:-1, :, :]   # interior
    lat = np.degrees(np.asarray(recipe.grid.lat))[1:-1]
    dyt_deg = global_flexible_dyt_deg()
    area = veros_area_t(lat, dx_deg=360.0 / NX, dyt_deg=dyt_deg)[:, None]  # (NY,1)
    dz = global_flexible_dzt_veros()[::-1]                   # full-cell (snap)
    vol = area[:, :, None] * dz[None, None, :] * ia

    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    T = np.asarray(state.T.data)[1:-1]
    S = np.asarray(state.S.data)[1:-1]

    # u/v at the T-cell's east/north face — Veros's u[i,j,k]/v[i,j,k].
    u_cell = u[1:-1, 1:, :]
    v_cell = v[2:-1, :, :]

    ia_e = np.minimum(ia, np.roll(ia, -1, axis=1))           # maskU (min rule)
    rho0 = float(recipe.model_config.rho_0)

    def wmean(x, w):
        sw = w.sum()
        return float((x * w).sum() / max(sw, 1e-30))

    speed2 = u_cell ** 2 + v_cell ** 2

    psi = np.asarray(state.psi) if state.psi is not None else np.zeros((1,))
    psi_min, psi_max = float(psi.min() / 1e6), float(psi.max() / 1e6)

    out_we = {}
    for name in ("tke", "eke"):
        fld = getattr(state, name)
        if fld is None:
            out_we[f"mean_{name}"] = float("nan")
            continue
        e = np.asarray(fld.data)[1:-1, :, :]                 # (NY, NX, NZ-1)
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
        sfc_T_mean=wmean(T[..., :1], vol[..., :1]),
        mean_tke=out_we["mean_tke"],
        mean_eke=out_we["mean_eke"],
        finite=bool(np.isfinite(u).all() and np.isfinite(T).all()
                    and np.isfinite(S).all()),
    )


# ---------------------------------------------------------------------------
# Forward integration
# ---------------------------------------------------------------------------


def run(years: float, out_path: str | None, compare_path: str | None,
        verify_against: str | None,
        tracer_advection: str | None = None,
        tke_advection: str | None = None) -> int:
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        DT_TRACER_S, get_periodic_interval_weights,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    t0 = time.perf_counter()
    recipe, stacks = load_and_prepare(verify_against)
    cfg = recipe.model_config
    # Optional stability fallbacks (LOUD deviations, the 4deg pattern).
    if tracer_advection is not None and tracer_advection != cfg.tracer_advection:
        print(f"!! STABILITY FALLBACK: tracer_advection={tracer_advection!r} "
              f"(Veros-faithful recipe: {cfg.tracer_advection!r})")
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
    recipe = recipe._replace(model_config=cfg)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    model.check_coriolis_stability(DT_TRACER_S)

    state = recipe.initial_state
    if state.T_incr_prev is None:
        _z = lambda d: Field(data=jnp.zeros_like(d.data),
                             name=d.name + "_incr_prev", dims=d.dims,
                             units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    rl = model._ensure_rigid_lid_data(state)
    print(f"  rigid-lid islands found: {rl.nisle} (Veros oracle log: 4)")
    if state.psi is None:
        _zV = jnp.zeros((recipe.grid.n_lat + 1, recipe.grid.n_lon + 1),
                        dtype=state.u.data.dtype)
        _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                               dpsin=_zI, dpsin_prev=_zI)

    taux, tauy = stacks["taux"], stacks["tauy"]
    q_pres, q_sol = stacks["q_prescribed"], stacks["q_solar"]
    qnec = stacks["qnec"]
    sst, sss = stacks["sst"], stacks["sss"]

    def _forcing_at(t_s):
        n1, f1, n2, f2 = get_periodic_interval_weights(t_s)

        def interp(stack):
            return (f1 * jnp.take(stack, n1, axis=0)
                    + f2 * jnp.take(stack, n2, axis=0))

        return OceanSurfaceForcing(
            tau_x=-interp(taux),          # external seam flips back to +taux
            tau_y=-interp(tauy),
            q_prescribed=interp(q_pres),  # NON-solar remainder
            q_solar=interp(q_sol),        # penetrative Jerlov column
            q_feedback=interp(qnec),
            T_feedback_target=interp(sst),
            S_restore_target=interp(sss),
            # q_net / sw_down stay None — trace-time double-count guards.
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
    steps_per_day = SECONDS_PER_DAY / DT_TRACER_S
    block_steps = int(round(30 * steps_per_day))      # 30 days per jit block
    year_steps = int(round(DAYS_PER_YEAR * steps_per_day))

    print(f"== legoESM global_flexible free run: {total_days:.0f} days "
          f"({total_steps} steps, dt_tracer={DT_TRACER_S:.0f} s, "
          f"dt_mom={DT_TRACER_S / cfg.dt_mom_ratio:.0f} s) ==")

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
            m["year"] = (done // year_steps if done % year_steps == 0
                         else day / DAYS_PER_YEAR)
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
                  f"({(time.perf_counter() - t_run0):.0f}s)", flush=True)

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
            tracer_advection=cfg.tracer_advection,
            tke_advection=cfg.physics.vertical_mixing.tke.advection_scheme,
            penetrative_shortwave=True,
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
    print("\n== per-year ratios (legoESM / Veros oracle) ==")
    print("year " + " ".join(f"{k:>13s}" for k in keys))
    for m in yearly:
        yr = m.get("year")
        if yr is None or float(yr) != int(float(yr)):
            continue
        rv = next((r for r in ref if r.get("year") == int(float(yr))), None)
        if rv is None:
            continue
        cells = []
        for k in keys:
            denom = rv[k]
            cells.append(f"{m[k] / denom:13.3f}" if denom else f"{'n/a':>13s}")
        print(f"{int(float(yr)):4d} " + " ".join(cells))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    val_dir = os.path.join(repo, ".physics-validator/transfer_global_flexible")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true",
                    help="10-day stability smoke (overrides --years)")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    ap.add_argument("--compare",
                    default=os.path.join(val_dir, "stock_10yr_flexible.json"),
                    help="banked Veros oracle JSON for the ratio table")
    ap.add_argument("--verify-against",
                    default=os.path.join(
                        val_dir, "stock_10yr_flexible_setup_arrays.npz"),
                    help="oracle setup-arrays npz for the kbot/zt/forcing "
                         "bit-identity gate ('' to skip)")
    ap.add_argument("--tracer-advection", default=None,
                    choices=("centered", "tvd"))
    ap.add_argument("--tke-advection", default=None,
                    choices=("none", "superbee"))
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

    years = (10.0 / DAYS_PER_YEAR) if args.smoke else args.years
    verify = args.verify_against or None
    if verify and not os.path.exists(verify):
        print(f"!! verify-against npz not found ({verify}) — gate SKIPPED")
        verify = None
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        return run(years, args.out,
                   None if args.smoke else args.compare,
                   verify,
                   tracer_advection=args.tracer_advection,
                   tke_advection=args.tke_advection)
    finally:
        set_policy(prev)


if __name__ == "__main__":
    raise SystemExit(main())
