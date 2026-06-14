"""global_1deg transfer test: legoESM free run vs the stock Veros oracle.

Runs the legoESM Veros-global_1deg recipe
(``legoesm.ocean.fidelity.veros_global_1deg_recipe``) forward with the REAL
Veros asset (``~/.veros/assets/global_1deg/forcing_1deg_global.nc``) at the
STOCK resolution and dt (360x160x115, dt_mom = dt_tracer = 1800 s — this
row has no rescale path; the recipe module docstring documents the
explicit-AB2 Coriolis margin |f|*dt = 0.258 at 79.5 deg), and records
yearly metrics in the SAME schema as the banked oracle runner
(``.physics-validator/transfer_global_1deg/run_oracle_1deg.py`` ->
``stock_5yr_1deg.json``).

Data prep replicates the Veros setup EXACTLY (set_grid / set_topography /
set_initial_conditions / set_forcing_kernel via the recipe's transcribed
pure helpers): dz read from the file (surface-first -> dz_ref directly),
kbot = salt zero-count + bathymetry mask + kbot<nz fixup + the THREE
channel closures, file T/S straight onto the grid, the load-time sign
flips (qnet/qsol NEGATED), surface-mask multiplication, and the MIT-grid
one-cell tau shift.  NO regridding anywhere.

VERIFICATION GATES (--verify-against, the proven 4deg/flexible pattern):
  - kbot BIT-IDENTICAL to the live oracle ``vs.kbot``;
  - zt/yt/dyt/maskT to float tolerance / bit (masks);
  - all 7 monthly forcing stacks < 1e-10 vs the oracle ``vs.*`` arrays;
  - NEW (this row): the oracle's t=0 REALIZED forcing (VerosSetup.setup()
    calls set_forcing once): ``surface_taux/y`` (pins the MIT-shift ghost
    handling), ``forc_temp_surface``/``forc_salt_surface`` (pins the
    heat-ownership algebra qnet = q_prescribed + q_solar, the feedback
    term, the ice mask and cp0/rho0), and ``temp_source`` vs the
    transcribed divpen column (pins the zw pen-profile convention).

Heat-ownership (q_solar channel): Veros keeps the SOLAR-INCLUSIVE total in
qnet and redistributes with the pen(0)=0 zero-column-sum profile; legoESM
passes ``q_prescribed = qnet - qsol`` (non-solar remainder) + ``q_solar =
qsol`` through ``flux_feedback(penetrative_shortwave=True)`` (I(0)=1
full-column deposit) — cell-by-cell algebraically identical.  The ice mask
(evaluated on the total flux) gates surface AND solar column inside the
scheme.

Monthly forcing is interpolated PER STEP INSIDE the jitted scan from the
traced model time (SegmentForcing pattern; Veros ``get_periodic_interval``
replica, 360-day year, month-START anchoring; step k uses t = k*dt).
Wind stress passes as ``tau_x = -taux`` (the external seam flips back to
Veros's on-ocean ``+taux``); the same half-cell u-face/T-point tau mimicry
note as the 4deg/flexible harnesses applies.

Usage::

    # 10-day smoke (GPU 0; fits alongside the oracle with a small
    # XLA_PYTHON_CLIENT_MEM_FRACTION):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.18 \
        JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/\
run_global_1deg_freerun.py --smoke \
        --verify-against .physics-validator/transfer_global_1deg/\
stock_5yr_1deg_setup_arrays.npz

    # 5-year run on GPU 0 (after the oracle frees it):
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 nohup python \
        scripts/validate/ocean_fidelity/run_global_1deg_freerun.py \
        --years 5 --out .physics-validator/transfer_global_1deg/\
legoesm_5yr_1deg.json &
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import numpy as np

ASSETS = os.path.expanduser("~/.veros/assets/global_1deg")
FORCING_NC = os.path.join(ASSETS, "forcing_1deg_global.nc")

DAYS_PER_YEAR = 360.0               # Veros forcing year
SECONDS_PER_DAY = 86400.0

# Veros set_initial_conditions penetration-profile literals
# (global_1deg.py:182-184) — used ONLY by the divpen verification gate
# (the model column comes from the shared shortwave_penetration scheme,
# whose Jerlov "I" constants are asserted identical by the unit tests).
RPART_SHORTWAVE = 0.58
EFOLD1_SHORTWAVE = 0.35
EFOLD2_SHORTWAVE = 23.0


def _read_nc(path, var):
    import h5netcdf
    with h5netcdf.File(path, "r") as f:
        return np.array(f.variables[var], dtype="float").T   # Veros _read_forcing


def veros_divpen_shortwave(dz_ref: np.ndarray) -> np.ndarray:
    """Veros divpen_shortwave transcription (global_1deg.py:228-240), in
    VEROS z-order (k=0 deepest): pen = R*exp(zw/z1) + (1-R)*exp(zw/z2) with
    pen[surface] := 0, divpen[k] = (pen[k]-pen[k-1])/dzt[k], divpen[0] =
    pen[0]/dzt[0].  ``dz_ref`` is legoESM surface-first; zw is the Veros
    interface array (zw[k] = the TOP interface of Veros cell k... i.e. the
    cumulative depth, zw[-1] = 0 shifted: zw = cumsum(dzt) - H)."""
    dzt_veros = np.asarray(dz_ref, dtype=np.float64)[::-1]    # k=0 deepest
    zw = np.cumsum(dzt_veros) - dzt_veros.sum()               # top ifaces, zw[-1]=0
    pen = (RPART_SHORTWAVE * np.exp(zw / EFOLD1_SHORTWAVE)
           + (1.0 - RPART_SHORTWAVE) * np.exp(zw / EFOLD2_SHORTWAVE))
    pen[-1] = 0.0
    divpen = np.zeros_like(pen)
    divpen[1:] = (pen[1:] - pen[:-1]) / dzt_veros[1:]
    divpen[0] = pen[0] / dzt_veros[0]
    return divpen


def load_and_prepare(verify_against: str | None = None):
    """netCDF reads + the Veros-verbatim data prep (NO regridding).
    Returns (recipe, stacks); each stack is (12, n_lat, n_lon) in legoESM
    layout (wall rows zero)."""
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        NZ,
        VEROS_GLOBAL4_CP0,
        T_REST_S,
        build_global_1deg_recipe,
        veros_full_axes_1deg,
    )
    from legoesm.ocean.fidelity.veros_layout_common import (
        veros_mit_tau_shift,
        veros_xy_to_legoesm,
    )
    from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity.veros_state_bridge import (
        veros_u_centered_z_centres,
    )

    # ---- set_grid ----
    dz_file = _read_nc(FORCING_NC, "dz")                # (115,) surface-first
    zt = veros_u_centered_z_centres(dz_file)[::-1]      # Veros k=0 deepest

    # ---- set_topography ----
    bathy = _read_nc(FORCING_NC, "bathymetry")          # (x, y)
    salt_file = _read_nc(FORCING_NC, "salinity")        # (x, y, z) surf-first
    salt_veros = salt_file[:, :, ::-1]                  # k=0 deepest
    temp_veros = _read_nc(FORCING_NC, "temperature")[:, :, ::-1]

    recipe = build_global_1deg_recipe(dz_file, bathy, salt_veros, temp_veros)
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        replicate_veros_kbot_1deg,
    )
    kbot = replicate_veros_kbot_1deg(bathy, salt_veros)
    wet_surf = (kbot > 0).astype(np.float64)            # maskT surface (x, y)

    # ---- set_initial_conditions: monthly stacks, signs + surface mask ----
    taux = _read_nc(FORCING_NC, "tau_x")                # (x, y, 12), N/m^2
    tauy = _read_nc(FORCING_NC, "tau_y")
    qnet = -_read_nc(FORCING_NC, "q_net") * wet_surf[:, :, None]   # sign flip
    qnec = _read_nc(FORCING_NC, "dqdt") * wet_surf[:, :, None]
    qsol = -_read_nc(FORCING_NC, "swf") * wet_surf[:, :, None]     # sign flip
    sst = _read_nc(FORCING_NC, "sst") * wet_surf[:, :, None]
    sss = _read_nc(FORCING_NC, "sss") * wet_surf[:, :, None]

    # MIT-grid one-cell tau shift (set_forcing_kernel), once at prep time.
    taux_s, tauy_s = veros_mit_tau_shift(taux, tauy, x_cyclic=False)

    # ---- verification gates vs the live oracle setup dump ----
    if verify_against:
        ref = np.load(verify_against)
        kb_ref = ref["kbot"][2:-2, 2:-2].astype(kbot.dtype)
        if not np.array_equal(kbot, kb_ref):
            n_bad = int((kbot != kb_ref).sum())
            raise AssertionError(
                f"kbot mismatch vs oracle: {n_bad} cells differ")
        np.testing.assert_allclose(zt, ref["zt"], rtol=0, atol=1e-9,
                                   err_msg="zt mismatch vs oracle")
        xt_full, yt_full, _ = veros_full_axes_1deg()
        np.testing.assert_allclose(yt_full, ref["yt"], rtol=0, atol=1e-9,
                                   err_msg="yt mismatch vs oracle")
        np.testing.assert_allclose(xt_full, ref["xt"], rtol=0, atol=1e-9,
                                   err_msg="xt mismatch vs oracle")
        degtom = float(VEROS_CONSTANTS_CONFIG.R_earth) * np.pi / 180.0
        np.testing.assert_allclose(
            ref["dyt"][2:-2] / degtom, 1.0, rtol=1e-9,
            err_msg="oracle dyt is not the uniform 1 degree")
        # maskT (lat, lon, k=0 surface) vs oracle maskT (x, y, k=0 deepest)
        ia = np.asarray(recipe.z_coord.is_active)[1:-1]      # (NY, NX, NZ)
        maskT_ref = np.transpose(
            ref["maskT"][2:-2, 2:-2, ::-1], (1, 0, 2))       # (y, x, z surf-1st)
        if not np.array_equal(ia.astype(bool), maskT_ref.astype(bool)):
            raise AssertionError("maskT mismatch vs oracle")
        # forcing stacks (float-tolerance: same reads, expect ~exact)
        for name, mine in (("taux", taux), ("tauy", tauy), ("qnet", qnet),
                           ("qnec", qnec), ("qsol", qsol), ("t_star", sst),
                           ("s_star", sss)):
            theirs = ref[name][2:-2, 2:-2, :]
            d = float(np.max(np.abs(mine - theirs)))
            if d > 1e-10:
                raise AssertionError(f"forcing stack {name} differs from "
                                     f"oracle by max {d:.3e}")
        # divpen transcription (pins the zw pen-profile convention)
        np.testing.assert_allclose(
            veros_divpen_shortwave(dz_file), ref["divpen_shortwave"],
            rtol=0, atol=1e-12, err_msg="divpen_shortwave mismatch")
        # --- t=0 REALIZED forcing (set_forcing ran inside setup()) ---
        if "surface_taux0" in ref.files:
            stx = ref["surface_taux0"][2:-2, 2:-2]
            sty = ref["surface_tauy0"][2:-2, 2:-2]
            d = float(np.max(np.abs(taux_s[:, :, 0] - stx)))
            if d > 1e-10:
                raise AssertionError(
                    f"t=0 surface_taux mismatch (max {d:.3e}) — MIT-shift "
                    "ghost handling wrong (cyclic-roll vs zero-ghost)")
            d = float(np.max(np.abs(tauy_s[:, :, 0] - sty)))
            if d > 1e-10:
                raise AssertionError(f"t=0 surface_tauy mismatch ({d:.3e})")
            # heat/salt rates: replicate the kernel from MY stacks + T0/S0
            cp0, rho0 = VEROS_GLOBAL4_CP0, float(VEROS_CONSTANTS_CONFIG.rho_0)
            T0 = temp_veros[:, :, -1] * wet_surf       # surface, Veros k=-1
            S0 = salt_veros[:, :, -1] * wet_surf
            forc_T = (qnet[:, :, 0]
                      + qnec[:, :, 0] * (sst[:, :, 0] - T0)) * wet_surf \
                / cp0 / rho0
            forc_S = (1.0 / T_REST_S) * (sss[:, :, 0] - S0) * wet_surf \
                * float(dz_file[0])                     # dzt[-1] = surface
            ice = ((T0 * wet_surf) > -1.8) | (forc_T > 0)
            forc_T = forc_T * ice
            forc_S = forc_S * ice
            d = float(np.max(np.abs(
                forc_T - ref["forc_temp_surface0"][2:-2, 2:-2])))
            if d > 1e-12:
                raise AssertionError(
                    f"t=0 forc_temp_surface mismatch (max {d:.3e}) — "
                    "heat ownership / ice mask / cp0 algebra wrong")
            d = float(np.max(np.abs(
                forc_S - ref["forc_salt_surface0"][2:-2, 2:-2])))
            if d > 1e-12:
                raise AssertionError(
                    f"t=0 forc_salt_surface mismatch (max {d:.3e})")
            # 3-D penetrative source (Veros z-order): qsol*divpen*ice*maskT
            ia_veros = np.transpose(ia, (1, 0, 2))[:, :, ::-1]  # (x,y,k0-deep)
            src = (qsol[:, :, 0, None]
                   * ref["divpen_shortwave"][None, None, :]
                   * ice[:, :, None] * ia_veros) / cp0 / rho0
            d = float(np.max(np.abs(src - ref["temp_source0"][2:-2, 2:-2])))
            if d > 1e-12:
                raise AssertionError(
                    f"t=0 temp_source mismatch (max {d:.3e}) — solar column")
            extra = ", t=0 realized forcing (taux/tauy/heat/salt/solar)"
        else:
            extra = " (no t=0 realized-forcing arrays in dump)"
        kb_hist = np.bincount(kbot.ravel(), minlength=NZ + 1)
        print(f"  VERIFIED vs oracle dump: kbot BIT-IDENTICAL "
              f"(wet {int((kbot > 0).sum())} cols, hist head "
              f"{list(kb_hist[:8])}), zt/xt/yt/maskT/forcing stacks, "
              f"divpen{extra}")

    def to_stack(arr_xy12):
        """(x, y, 12) -> (12, n_lat, n_lon) legoESM layout, wall rows 0."""
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
# Metrics — the oracle runner's schema (run_oracle_1deg.py:metrics)
# ---------------------------------------------------------------------------


def compute_metrics(state, recipe) -> dict:
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import DXT_DEG, DYT_DEG
    from legoesm.ocean.fidelity.veros_layout_common import veros_area_t

    ia = np.asarray(recipe.z_coord.is_active)[1:-1, :, :]   # interior
    lat = np.degrees(np.asarray(recipe.grid.lat))[1:-1]
    area = veros_area_t(lat, dx_deg=DXT_DEG, dyt_deg=DYT_DEG)[:, None]  # (NY, 1)
    dz = np.asarray(recipe.z_coord.dz_ref)                   # full-cell (snap)
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
        tke_advection: str | None = None,
        tke_mxl_choice: int | None = None) -> int:
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        DT_S, get_periodic_interval_weights,
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
    vm = cfg.physics.vertical_mixing
    if (tke_mxl_choice is not None
            and tke_mxl_choice != vm.tke.tke_mxl_choice):
        # Generic experimentation override. The recipe ships the faithful
        # tke_mxl_choice=1 (Veros global_1deg.py:64) which now runs natively
        # — the distance-to-boundary cap (veros_mxl_choice1_boundary_cap)
        # makes it debt-safe, so NO fallback is needed. This knob only exists
        # to A/B the choice=2 recursion against choice=1; any use is a LOUD
        # deviation from the oracle, recorded in the output config.
        print(f"!! DEVIATION (experimentation): tke_mxl_choice={tke_mxl_choice!r} "
              f"(Veros-faithful recipe: {vm.tke.tke_mxl_choice!r})")
        cfg = cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=vm._replace(
                tke=vm.tke._replace(tke_mxl_choice=tke_mxl_choice))))
    recipe = recipe._replace(model_config=cfg)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    model.check_coriolis_stability(DT_S)

    state = recipe.initial_state
    if state.T_incr_prev is None:
        _z = lambda d: Field(data=jnp.zeros_like(d.data),
                             name=d.name + "_incr_prev", dims=d.dims,
                             units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    rl = model._ensure_rigid_lid_data(state)
    print(f"  rigid-lid islands found: {rl.nisle} "
          f"(Veros oracle log: 47 + main boundary)")
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
            t = (step0 + k).astype(jnp.float64) * DT_S
            sf = _forcing_at(t)
            return model.step(carry, DT_S, surface_forcing=sf), None
        st, _ = jax.lax.scan(body, st, jnp.arange(n))
        return st

    total_days = years * DAYS_PER_YEAR
    total_steps = int(round(total_days * SECONDS_PER_DAY / DT_S))
    steps_per_day = SECONDS_PER_DAY / DT_S                   # 48
    block_steps = int(round(5 * steps_per_day))              # 5 days per block
    year_steps = int(round(DAYS_PER_YEAR * steps_per_day))

    print(f"== legoESM global_1deg free run: {total_days:.0f} days "
          f"({total_steps} steps, dt={DT_S:.0f} s sync) ==")

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
                  f"({done * DT_S / SECONDS_PER_DAY:.0f} days)")
            m = compute_metrics(state, recipe)
            m["year"] = done * DT_S / SECONDS_PER_DAY / DAYS_PER_YEAR
            m["blowup"] = True
            yearly.append(m)
            break
        day = done * DT_S / SECONDS_PER_DAY
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
        elif done % (block_steps * 6) == 0:
            umax = float(jnp.max(jnp.abs(state.u.data)))
            print(f"  day {day:6.0f}: max|u|={umax:.4f} m/s "
                  f"({(time.perf_counter() - t_run0):.0f}s)", flush=True)

    out = dict(
        years=years,
        yearly=yearly,
        total_wall_h=round((time.perf_counter() - t0) / 3600.0, 3),
        config=dict(
            dt=DT_S, dt_mom_ratio=cfg.dt_mom_ratio,
            outer_integrator=cfg.outer_integrator,
            barotropic_solver=cfg.barotropic_solver,
            coriolis_scheme=cfg.coriolis_scheme, ab2_scope=cfg.ab2_scope,
            eos=cfg.eos, nisle=int(rl.nisle),
            tracer_advection=cfg.tracer_advection,
            tke_advection=cfg.physics.vertical_mixing.tke.advection_scheme,
            tke_mxl_choice=cfg.physics.vertical_mixing.tke.tke_mxl_choice,
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
    val_dir = os.path.join(repo, ".physics-validator/transfer_global_1deg")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true",
                    help="10-day stability smoke (overrides --years)")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    ap.add_argument("--compare",
                    default=os.path.join(val_dir, "stock_5yr_1deg.json"),
                    help="banked Veros oracle JSON for the ratio table")
    ap.add_argument("--verify-against",
                    default=os.path.join(
                        val_dir, "stock_5yr_1deg_setup_arrays.npz"),
                    help="oracle setup-arrays npz for the kbot/zt/forcing "
                         "bit-identity gate ('' to skip)")
    ap.add_argument("--tracer-advection", default=None,
                    choices=("centered", "tvd"))
    ap.add_argument("--tke-advection", default=None,
                    choices=("none", "superbee"))
    ap.add_argument("--tke-mxl-choice", type=int, default=None,
                    choices=(1, 2),
                    help="experimentation override for TKEConfig.tke_mxl_choice; "
                         "the recipe ships the faithful 1 (now debt-safe via "
                         "the distance-to-boundary cap), so this is NOT needed "
                         "for stability — only to A/B against choice=2")
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
                   tke_advection=args.tke_advection,
                   tke_mxl_choice=args.tke_mxl_choice)
    finally:
        set_policy(prev)


if __name__ == "__main__":
    raise SystemExit(main())
