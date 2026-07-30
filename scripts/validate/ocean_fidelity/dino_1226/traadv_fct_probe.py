"""#1226 traadv_fct component-row driver: fluxes, tendency (T), horizontal-only
tendency (new -- no probe for this existed before), vertical upstream flux,
and SALINITY -- all at the CORRECT k-alignment, all against NEMO's PURE
(Krhs-contamination-free) advective tendency reconstructed from its own dumped
fluxes (fct_dump_zw{x,y,z}_{up,anti}[_sal].bin, units 8826-8834/8990-8997,
cfgs/DINO/MY_SRC/traadv_fct.F90).

WHY A REWRITE (not a patch of the old scratchpad probes): the previously
*recorded* numbers in fidelity_bar_gate.py for these rows (fluxes 0.96-0.98,
tendency-T 0.994487, vertical-upstream-flux 0.997791, SALINITY 0.203) do not
reproduce when the existing scratchpad probes (probe_fct.py,
probe_nonosc_stages.py, check_wflux_offset0.py, probe_fct_sal.py) are re-run
against HEAD with the SAME dumps: every one of them lands at 4-5 nines
instead. Root cause isolated for the "fluxes" row specifically: the ORIGINAL
probe's alignment scan picked its k-offset by maximizing corr against NEMO's
RAW trd_final/trd_up dumps, which are Krhs-CONTAMINATED (pt(Krhs) already
carries tra_sbc/tra_qsr/etc added by earlier stpmlf.F90 calls before
tra_adv_fct runs) -- that contaminated signal peaks at the WRONG k-offset
(+1). The PURE reconstruction (this file), immune to that contamination,
sharply peaks at the correct offset (0) for every quantity checked. The
tendency-T/vertical-upstream-flux/SALINITY rows in the gate cite this same
PURE technique and the correct offset=0, yet recorded numbers between the
true 0.9999-class result and the wrong offset's ~0.92-0.96 -- unreproducible
transcription errors, not measurements of a different quantity (verified: no
env var / e3t mode / omega setting reproduces them). This file is the
single, permanent, re-runnable replacement.

Run with:
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
  python scripts/validate/ocean_fidelity/dino_1226/traadv_fct_probe.py
"""
import dataclasses

import netCDF4 as nc
import numpy as np
import jax.numpy as jnp

from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
)
from legoesm.ocean.vertical import compute_layer_thickness, diagnose_w_from_flux_div
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid, compute_face_masks_3d, min_cell_to_uface, min_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    add_bolus_to_advecting_flux,
)
from legoesm.ocean.advection import fct_tracer_advection
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    upwind_to_u_points, upwind_to_v_points,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
)

D = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
G = f"{D}/RUN_GDB"
JPK, JPJ, JPI = 35, 203, 56  # NEMO dump shape (halo-included, jpk x jpj x jpi)


def load_dump_full(name):
    raw = np.fromfile(f"{G}/{name}", dtype=np.float64).reshape(JPK, JPJ, JPI)
    return np.moveaxis(raw, 0, -1)  # (jpj, jpi, jpk), WITH halo


def load_dump(name):
    return load_dump_full(name)[2:-2, 2:-2, :]  # halo-stripped


def stats(L, N):
    finite = np.isfinite(L) & np.isfinite(N)
    L, N = L[finite], N[finite]
    n = L.size
    if n < 2:
        return dict(n=n, corr=np.nan)
    corr = np.corrcoef(L, N)[0, 1]
    absN = np.abs(N).sum()
    abs_ratio = np.abs(L).sum() / absN if absN != 0 else np.nan
    rms_ratio = (np.sqrt((L ** 2).mean()) / np.sqrt((N ** 2).mean())
                 if (N ** 2).mean() != 0 else np.nan)
    return dict(n=n, corr=corr, abs_ratio=abs_ratio, rms_ratio=rms_ratio)


def stats_at_offset(lego_field, nemo_field, offset, act):
    Ls, Ns = [], []
    for k in range(lego_field.shape[2]):
        kn = k + offset
        if 0 <= kn < nemo_field.shape[2]:
            wk = act[:, :, k]
            if not wk.any():
                continue
            Ls.append(lego_field[:, :, k][wk])
            Ns.append(nemo_field[:, :, kn][wk])
    if not Ls:
        return dict(n=0, corr=np.nan)
    return stats(np.concatenate(Ls), np.concatenate(Ns))


def nemo_div(Fx, Fy, Fz):
    """NEMO flux-divergence per traadv_fct.F90: ztra = -( dFx + dFy + dFz )."""
    dFx = np.empty_like(Fx)
    dFx[:, 1:, :] = Fx[:, 1:, :] - Fx[:, :-1, :]
    dFx[:, 0, :] = np.nan
    dFy = np.empty_like(Fy)
    dFy[1:, :, :] = Fy[1:, :, :] - Fy[:-1, :, :]
    dFy[0, :, :] = np.nan
    dFz = np.empty_like(Fz)
    dFz[:, :, :-1] = Fz[:, :, :-1] - Fz[:, :, 1:]
    dFz[:, :, -1] = Fz[:, :, -1]  # Fw(jpk) = 0
    return -(dFx + dFy + dFz)


def build_bridge_and_fluxes():
    grid = read_nemo_mesh_mask(f"{G}/mesh_mask.nc", nn_hls=0)
    R = f"{G}/DINO_00057600_restart.nc"
    now = read_nemo_restart(R, nn_hls=0)
    bef = read_nemo_restart_before(R, nn_hls=0)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    # #1226 (e72923faa retraction lesson): cfg.omega must reach the bridge's
    # grid.f construction, else grid.f (and the GM/Redi bolus folded into the
    # advecting transport below) silently reverts to the rounded default.
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    br_state = bridge_before_state_topo(br, grid, bef, periodic_i=True)
    br = br._replace(state=br_state)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    _grid = br.geometry
    z_coord = br.z_coord
    state = br.state
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    H_bathy = state.H_bathy.data

    T_now = jnp.asarray(state.T.data)
    S_now = jnp.asarray(state.S.data)
    T_bef = jnp.asarray(state.T_before.data)
    S_bef = jnp.asarray(state.S_before.data)
    eta_now = jnp.asarray(state.eta.data)

    h_k_old = compute_layer_thickness(
        eta_now, H_bathy, z_coord, min_water_column_m=mc.min_water_column_m,
    )
    u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, _grid)
    u_mask_3d = u_mask_3d.astype(T_now.dtype)
    v_mask_3d = v_mask_3d.astype(T_now.dtype)
    h_u_old = min_cell_to_uface(h_k_old)
    h_v_old = min_cell_to_vface(h_k_old, _grid)

    u_now = jnp.asarray(state.u.data)
    v_now = jnp.asarray(state.v.data)
    mass_flux_u = h_u_old * u_now * u_mask_3d
    mass_flux_v = h_v_old * v_now * v_mask_3d
    flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, _grid)
    w_baro = diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    model_dt = float(cfg.dt)
    p2dt = 2.0 * model_dt
    dT_gm, dS_gm, bolus = gm_redi_tracer_tendency_latlon(
        T_now, S_now, eta_now, H_bathy, _grid, z_coord, mc.gm_redi,
        eos=mc.eos, eos_linear=mc.eos_linear, mask=mask,
        u_mask=u_mask, v_mask=v_mask, rho_0=mc.rho_0, g=mc.g,
        omega=mc.omega,   # #1226: see the cfg/bridge omega note above.
        return_bolus_transport=True, dt=p2dt,
    )
    if bolus is None:
        raise RuntimeError(
            "bolus is None -- gm_bolus_advection != 'through_fct' or "
            "kappa_GM disabled; check the recipe config.")
    mass_flux_u_tr, mass_flux_v_tr, w_baro_tr = add_bolus_to_advecting_flux(
        bolus, mass_flux_u, mass_flux_v, u_mask_3d, v_mask_3d, _grid, z_coord,
    )
    act = np.asarray(z_coord.is_active)
    return dict(
        grid=_grid, z_coord=z_coord, state=state, act=act,
        T_now=T_now, S_now=S_now, T_bef=T_bef, S_bef=S_bef,
        h_k_old=h_k_old, mass_flux_u=mass_flux_u_tr, mass_flux_v=mass_flux_v_tr,
        w_baro=w_baro_tr, p2dt=p2dt,
    )


def nemo_pure_tendency_and_metrics(sal=False):
    """Load NEMO's dumped fluxes and reconstruct PURE (Krhs-uncontaminated)
    total/horizontal-only/vertical-only/upstream-only tendencies + metrics
    needed to non-dimensionalize lego's own divergence into a tendency."""
    suf = "_sal" if sal else ""
    Fu_up = load_dump_full(f"fct_dump_zwx_up{suf}.bin")
    Fv_up = load_dump_full(f"fct_dump_zwy_up{suf}.bin")
    Fw_up = load_dump_full(f"fct_dump_zwz_up{suf}.bin")
    Fu_anti = load_dump_full(f"fct_dump_zwx_anti{suf}.bin")
    Fv_anti = load_dump_full(f"fct_dump_zwy_anti{suf}.bin")
    Fw_anti = load_dump_full(f"fct_dump_zwz_anti{suf}.bin")
    Fu = Fu_up + Fu_anti
    Fv = Fv_up + Fv_anti
    Fw = Fw_up + Fw_anti

    div_final = nemo_div(Fu, Fv, Fw)[2:-2, 2:-2, :]
    div_up = nemo_div(Fu_up, Fv_up, Fw_up)[2:-2, 2:-2, :]
    # horizontal-only divergence (zero the vertical flux term): isolates the
    # xad+yad bucket the way trdtra.F90 would bucket ttrd_xad+ttrd_yad.
    div_h_only = nemo_div(Fu, Fv, np.zeros_like(Fw))[2:-2, 2:-2, :]
    div_v_only = nemo_div(np.zeros_like(Fu), np.zeros_like(Fv), Fw)[2:-2, 2:-2, :]

    mm = nc.Dataset(f"{G}/mesh_mask.nc")
    e1t = mm["e1t"][0].filled(np.nan)
    e2t = mm["e2t"][0].filled(np.nan)
    e3t_0 = np.moveaxis(mm["e3t_0"][0].filled(np.nan), 0, -1)
    tmask3 = np.moveaxis(mm["tmask"][0].filled(0), 0, -1).astype(float)
    mm.close()
    r = nc.Dataset(f"{G}/DINO_00057600_restart.nc")
    sshn = r.variables["sshn"][0].filled(np.nan)
    r.close()

    H_col = (e3t_0 * tmask3).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        e3t_now = e3t_0 * (1.0 + (sshn / np.where(H_col > 0, H_col, np.nan))[:, :, None])
    e1e2t = e1t * e2t
    jpkm1 = div_final.shape[2]

    def to_rate(div):
        return div / (e1e2t[:, :, None] * e3t_now[:, :, :jpkm1]) * tmask3[:, :, :jpkm1]

    return dict(
        trd_pure_final=to_rate(div_final),
        trd_pure_up=to_rate(div_up),
        trd_pure_h=to_rate(div_h_only),
        trd_pure_v=to_rate(div_v_only),
        Fu_up=Fu_up, Fv_up=Fv_up, Fw_up=Fw_up,
        Fu_anti=Fu_anti, Fv_anti=Fv_anti, Fw_anti=Fw_anti,
        e1e2t=e1e2t, e3t_now=e3t_now, jpkm1=jpkm1,
    )


def run(tracer_name, tracer_now, tracer_bef, ctx, sal):
    act = ctx["act"]
    _grid = ctx["grid"]
    h_k_old = ctx["h_k_old"]
    p2dt = ctx["p2dt"]
    eps = 1e-30

    nemo = nemo_pure_tendency_and_metrics(sal=sal)

    # --- lego full FCT tendency (production call, active_mask fix applied) ---
    div_h_fct, vert_div_fct = fct_tracer_advection(
        tracer_now, ctx["mass_flux_u"], ctx["mass_flux_v"], ctx["w_baro"],
        h_k_old, _grid, p2dt, high_order="centred2", tracer_before=tracer_bef,
        active_mask=jnp.asarray(act),
    )
    dT_total = np.asarray(-(div_h_fct + vert_div_fct) / jnp.maximum(h_k_old, eps))
    dT_h = np.asarray(-div_h_fct / jnp.maximum(h_k_old, eps))
    dT_v = np.asarray(-vert_div_fct / jnp.maximum(h_k_old, eps))

    # --- lego upstream-only tendency (pre-limiter, for the upstream/"fluxes" row) ---
    tr_u_low = upwind_to_u_points(tracer_bef, ctx["mass_flux_u"])
    tr_v_low = upwind_to_v_points(tracer_bef, ctx["mass_flux_v"])
    flux_u_low = ctx["mass_flux_u"] * tr_u_low
    flux_v_low = ctx["mass_flux_v"] * tr_v_low
    div_h_low = divergence_cgrid(flux_u_low, flux_v_low, _grid)
    nlev = tracer_now.shape[-1]
    w_int = ctx["w_baro"][..., 1:nlev]
    Tb_below, Tb_above = tracer_bef[..., 1:], tracer_bef[..., :-1]
    T_face_low = jnp.where(w_int > 0.0, Tb_below, Tb_above)
    F_vert_low_int = w_int * T_face_low
    pad_axes_v = ((0, 0),) * (F_vert_low_int.ndim - 1)
    F_vert_low = jnp.pad(F_vert_low_int, (*pad_axes_v, (1, 1)))
    vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]
    dT_up = np.asarray(-(div_h_low + vert_div_low) / jnp.maximum(h_k_old, eps))

    # --- per-face flux comparison, metric-scaled to NEMO's volume-flux units,
    # AT THE CORRECT offset=0 (see module docstring) ---
    dy_u = np.asarray(_grid.dy_u)
    dx_v = np.asarray(_grid.dx_v)
    area_t = getattr(_grid, "area_T", None)
    if area_t is None:
        area_t = np.asarray(_grid.dx_v[:-1, :]) * np.asarray(_grid.dy_u[:, :-1])
    else:
        area_t = np.asarray(area_t)
    lego_fu_east = np.asarray(flux_u_low)[:, 1:, :] * dy_u[:, 1:, None]
    lego_fv_north = np.asarray(flux_v_low)[1:, :, :] * dx_v[1:, :, None]
    lego_fw_full = np.asarray(F_vert_low) * area_t[:, :, None]

    r_fu = stats_at_offset(lego_fu_east, nemo["Fu_up"][2:-2, 2:-2, :], 0, act)
    r_fv = stats_at_offset(lego_fv_north, nemo["Fv_up"][2:-2, 2:-2, :], 0, act)
    r_fw = stats_at_offset(lego_fw_full, nemo["Fw_up"][2:-2, 2:-2, :], 0, act)

    r_final = stats_at_offset(dT_total, nemo["trd_pure_final"], 0, act)
    r_h = stats_at_offset(dT_h, nemo["trd_pure_h"], 0, act)
    r_v = stats_at_offset(dT_v, nemo["trd_pure_v"], 0, act)
    r_up = stats_at_offset(dT_up, nemo["trd_pure_up"], 0, act)

    print(f"\n{'=' * 78}\n{tracer_name}  (RUN_GDB kt=57601, offset=0, e3t={_e3t_mode()})\n{'=' * 78}")
    print(f"  upstream zonal flux (east face)      : {r_fu}")
    print(f"  upstream meridional flux (north face) : {r_fv}")
    print(f"  upstream vertical flux (area-scaled) : {r_fw}")
    print(f"  upstream-only tendency                : {r_up}")
    print(f"  FULL tendency (traadv_fct tendency)   : {r_final}")
    print(f"  HORIZONTAL-only tendency (NEW)         : {r_h}")
    print(f"  VERTICAL-only tendency                 : {r_v}")
    return dict(fu=r_fu, fv=r_fv, fw=r_fw, up=r_up, final=r_final, h=r_h, v=r_v)


def _e3t_mode():
    import os
    return os.environ.get("LEGOESM_NEMO_E3T", "off")


if __name__ == "__main__":
    ctx = build_bridge_and_fluxes()
    res_T = run("TEMPERATURE", ctx["T_now"], ctx["T_bef"], ctx, sal=False)
    res_S = run("SALINITY", ctx["S_now"], ctx["S_bef"], ctx, sal=True)

    print(f"\n{'=' * 78}\nSUMMARY (measuring commit: see MEASURED_AT in fidelity_bar_gate.py)\n{'=' * 78}")
    print(f"traadv_fct fluxes (T, per-face, offset=0): "
          f"u corr={res_T['fu']['corr']:.6f}/ratio={res_T['fu']['abs_ratio']:.6f}  "
          f"v corr={res_T['fv']['corr']:.6f}/ratio={res_T['fv']['abs_ratio']:.6f}  "
          f"w corr={res_T['fw']['corr']:.6f}/ratio={res_T['fw']['abs_ratio']:.6f}")
    print(f"traadv_fct tendency (T): corr={res_T['final']['corr']:.6f} "
          f"ratio={res_T['final']['abs_ratio']:.6f}")
    print(f"traadv_fct horizontal tend (T): corr={res_T['h']['corr']:.6f} "
          f"ratio={res_T['h']['abs_ratio']:.6f}")
    print(f"traadv_fct vertical upstream flux (T): corr={res_T['fw']['corr']:.6f} "
          f"ratio={res_T['fw']['abs_ratio']:.6f}")
    print(f"traadv_fct (SALINITY): corr={res_S['final']['corr']:.6f} "
          f"ratio={res_S['final']['abs_ratio']:.6f}")
