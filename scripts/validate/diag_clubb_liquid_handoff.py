"""Does CLUBB's PDF liquid reach the cloud optics, and what does the production
diagnostic condensate floor contribute?

    diag_clubb_liquid_handoff.py --config <deck> --restart <checkpoint.npz> [--floor-off]

Three measurements on ONE restored state, all area-weighted with the mesh cell
areas, NUMBERS ONLY (no verdict):

  1. L_rcm     column integral of the CLUBB PDF liquid ``rcm`` returned by
               ``clubb_step`` diagnostics over layers whose published liquid
               cloud fraction is positive                        [kg/m^2]
     L_qc      column integral of the host's prognostic ``q_c``   [kg/m^2]
     H         sum(max(rcm-q_c,0)*dp) / sum(rcm*dp)  over the same layers
  2. LWP/IWP that the optics actually receives, from ``compute_cloud_properties``
     with the deck's CloudConfig, and again with ``q_c_diagnostic=0`` so the
     diagnostic-condensate floor's own contribution is isolated, plus the
     implied in-cloud optical depth  tau = 3 LWP / (2 rho_w r_eff).
  3. The share of the cloud cover sitting at cloud fraction <= 1/n_subcolumns,
     which the deterministic sub-column sampler cannot represent.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
RHO_W = 1000.0


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_clubbprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_clubbprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    from legoesm import constants

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    _cfg_keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**_cfg_keys)
    parser.set_defaults(_config_keys=frozenset(_cfg_keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    print(f"[{a.label}] restored step={step0} day={day0}", flush=True)

    state = driver.state
    model = driver.model
    mesh = model.mesh
    sc = model.sigma_coord
    T = jnp.asarray(state.T.data)
    p_s = jnp.asarray(state.p_s.data)
    nCells, nlev = T.shape
    area = np.asarray(mesh.areaCell).reshape(nCells)
    A = area / area.sum()

    p_full = sc.pressure_at_full(p_s).reshape(nCells, nlev)
    p_half = sc.pressure_at_half(p_s).reshape(nCells, nlev + 1)
    dp = p_half[:, 1:] - p_half[:, :-1]

    def trac(name):
        if state.tracers is None or name not in state.tracers:
            return None
        r = state.tracers[name]
        d = r.data if hasattr(r, "data") else r
        return jnp.asarray(d).reshape(nCells, nlev)

    q_v = trac("q_v")
    q_c = trac("q_c")
    q_i = trac("q_i")
    n_c = trac("N_c")
    n_i = trac("N_i")
    if q_v is None:
        q_v = jnp.zeros_like(T)

    def colint(x):
        """Area-weighted mean column integral of a mixing ratio [kg/m^2]."""
        return float(np.sum(A * np.asarray(jnp.sum(x * dp, axis=1) / constants.g)))

    print(f"[{a.label}] ncol={nCells} nlev={nlev}")
    print(f"[{a.label}] prognostic q_c path   = {colint(q_c) if q_c is not None else float('nan'):.6e} kg/m2")
    print(f"[{a.label}] prognostic q_i path   = {colint(q_i) if q_i is not None else float('nan'):.6e} kg/m2")

    # ---- 1. the CLUBB PDF liquid -------------------------------------------
    from legoesm.driver.physics_pipeline import turbulence_config_for
    turb = turbulence_config_for(config)
    scheme_name = turb.scheme
    print(f"[{a.label}] turbulence scheme = {scheme_name}")
    rcm_done = False
    _aux = getattr(driver, "_carry_aux", {}) or {}
    def carry(name):
        v = _aux.get("physstate_" + name)
        if v is None:
            ps = getattr(driver, "_mpas_phys_state", None)
            v = getattr(ps, name, None) if ps is not None else None
        return v
    print(f"[{a.label}] restored carries: {sorted(k[len('physstate_'):] for k in _aux if k.startswith('physstate_'))}")
    moments_packed = carry("clubb_moments")
    if moments_packed is not None:
        from legoesm.atmosphere.physics.turbulence.clubb import (
            clubb_step, unpack_clubb_moments)
        from legoesm.atmosphere.physics._shared import (
            compute_rho as _compute_rho,
            compute_heights_from_sigma as _compute_heights_from_sigma)
        from legoesm.grids.voronoi import reconstruct_cell_velocity
        from legoesm.thermo import saturation_mixing_ratio
        u_cell, v_cell = reconstruct_cell_velocity(jnp.asarray(state.u.data), mesh)
        z_full, z_half = _compute_heights_from_sigma(T, p_half, q_v=q_v)
        rho = _compute_rho(T, p_full, q_v=q_v)
        _ov = carry("surface_T_sfc_override")
        T_sfc = jnp.asarray(_ov).reshape(nCells) if _ov is not None else T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        from legoesm.atmosphere.physics.turbulence.integration import materialize_sub_config
        clubb_cfg = getattr(materialize_sub_config(turb), "clubb", None)
        if clubb_cfg is None:
            raise SystemExit("no CLUBB sub-config on this deck")
        dt_phys = float(config.dycore.dt) * int(getattr(config, "physics_update_steps", 1) or 1)
        moments = unpack_clubb_moments(jnp.asarray(moments_packed))
        out = clubb_step(u_cell.reshape(nCells, nlev), v_cell.reshape(nCells, nlev),
                         T, q_v, moments, p_full, p_half, z_full, z_half,
                         T_sfc, q_sfc, rho, dt_phys, clubb_cfg)
        diags = out[5]
        rcm_a = np.asarray(diags["rcm"])          # ascending (bottom-up)
        cf_a = np.asarray(diags["cloud_frac"])
        rcm = rcm_a[:, ::-1][:, :nlev] if rcm_a.shape[1] >= nlev else rcm_a[:, ::-1]
        cfl = cf_a[:, ::-1][:, :nlev] if cf_a.shape[1] >= nlev else cf_a[:, ::-1]
        rcm = np.maximum(rcm, 0.0)
        mask = cfl > 0.0
        dpn = np.asarray(dp)
        qcn = np.asarray(q_c) if q_c is not None else np.zeros_like(rcm)
        L_rcm = float(np.sum(A * np.sum(np.where(mask, rcm, 0.0) * dpn, axis=1) / constants.g))
        L_qc_m = float(np.sum(A * np.sum(np.where(mask, qcn, 0.0) * dpn, axis=1) / constants.g))
        num = np.sum(A[:, None] * np.where(mask, np.maximum(rcm - qcn, 0.0), 0.0) * dpn)
        den = np.sum(A[:, None] * np.where(mask, rcm, 0.0) * dpn)
        H = float(num / den) if den > 0 else float("nan")
        _incl = np.where(mask, rcm, np.nan)
        print(f"[{a.label}] rcm raw shape={rcm_a.shape} nlev={nlev}; "
              f"grid-mean rcm where cf>0: mean={np.nanmean(_incl):.3e} max={np.nanmax(_incl):.3e} kg/kg; "
              f"cf where cf>0: mean={cfl[mask].mean():.3f}")
        print(f"[{a.label}] CLUBB PDF liquid  L_rcm = {L_rcm:.6e} kg/m2  "
              f"(cloudy layers only, cf>0 in {int(mask.sum())} cells)")
        print(f"[{a.label}] host q_c on those layers = {L_qc_m:.6e} kg/m2")
        print(f"[{a.label}] H = missing share of PDF liquid = {H:.4f}")
        rcm_done = True
    if not rcm_done:
        print(f"[{a.label}] no carried CLUBB moments on this state -> PDF liquid N/A")

    # ---- 2. what the optics receives, floor on and off ----------------------
    from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    _g = lambda n, d=None: getattr(config, n, d)
    ccfg = build_cloud_config(
        _g("cloud_scheme", "sundqvist"),
        convective_cloud=bool(_g("convective_cloud", False)),
        rh_crit=_g("cloud_rh_crit"), q_c_diagnostic=_g("cloud_q_c_diagnostic"),
        conv_cloud_max=_g("cloud_conv_cloud_max"),
        conv_cloud_condensate=_g("cloud_conv_cloud_condensate"),
        Nc_default=_g("cloud_Nc_default"),
        cloud_inhomogeneity_factor=_g("cloud_inhomogeneity_factor"),
        cloud_optics_inhomogeneity=_g("cloud_optics_inhomogeneity"),
        cloud_fsd=_g("cloud_fsd"),
        cloud_partial_coverage_optics=_g("cloud_partial_coverage_optics"),
        cloud_vertical_overlap_optics=_g("cloud_vertical_overlap_optics"),
        cloud_n_subcolumns=_g("cloud_n_subcolumns"),
        p_xr=_g("cloud_p_xr"), alpha_xr=_g("cloud_alpha_xr"),
        diagnostic_condensate_scheme=_g("cloud_diagnostic_condensate_scheme"),
        adiabatic_lwc_rate=_g("cloud_adiabatic_lwc_rate"),
        clubb_cf_override_strength=_g("cloud_clubb_cf_override_strength"),
        clubb_cf_override_floor=_g("cloud_clubb_cf_override_floor"),
        saturation_scheme=_g("cloud_saturation_scheme"),
    )
    cf_override = None
    _carried_cf = carry("cloud_fraction")
    if bool(getattr(config, "use_clubb_cloud_fraction", False)) and _carried_cf is not None:
        cf_override = jnp.asarray(_carried_cf).reshape(nCells, nlev)
        print(f"[{a.label}] using the CARRIED CLUBB cloud fraction as the optics override")

    def props(cfg):
        return compute_cloud_properties(
            T, p_full, q_v, dp, cfg, q_cloud=q_c, q_ice=q_i,
            n_ice=n_i, n_cloud=n_c, cloud_fraction_override=cf_override,
            p_half=p_half, lat=jnp.asarray(mesh.latCell).reshape(nCells))

    cp = props(ccfg)
    lwp = np.asarray(cp.lwp); iwp = np.asarray(cp.iwp)
    cf = np.asarray(cp.cloud_fraction)
    reff = np.asarray(cp.r_eff_liq)
    LWP = float(np.sum(A * lwp.sum(axis=1)))
    IWP = float(np.sum(A * iwp.sum(axis=1)))
    print(f"[{a.label}] optics scheme={ccfg.scheme} q_c_diagnostic={getattr(ccfg,'q_c_diagnostic',None)}")
    print(f"[{a.label}] optics LWP = {LWP:.6e} kg/m2   IWP = {IWP:.6e} kg/m2")
    tau = 1.5 * lwp / (RHO_W * np.maximum(reff, 1e-12))
    print(f"[{a.label}] implied liquid tau (column sum, grid-mean) = "
          f"{float(np.sum(A * tau.sum(axis=1))):.4f}   mean r_eff = "
          f"{float(np.sum(A[:, None] * reff * (lwp > 0)) / max(np.sum(A[:, None] * (lwp > 0)), 1e-30))*1e6:.2f} um")

    if getattr(ccfg, "q_c_diagnostic", 0.0) > 0.0:
        cfg0 = ccfg._replace(q_c_diagnostic=0.0)
        cp0 = props(cfg0)
        lwp0 = np.asarray(cp0.lwp); iwp0 = np.asarray(cp0.iwp)
        reff0 = np.asarray(cp0.r_eff_liq)
        LWP0 = float(np.sum(A * lwp0.sum(axis=1)))
        IWP0 = float(np.sum(A * iwp0.sum(axis=1)))
        tau0 = 1.5 * lwp0 / (RHO_W * np.maximum(reff0, 1e-12))
        print(f"[{a.label}] FLOOR OFF: LWP = {LWP0:.6e}  IWP = {IWP0:.6e} kg/m2")
        print(f"[{a.label}] FLOOR contributes {100.0*(LWP-LWP0)/max(LWP,1e-30):.2f}% of LWP, "
              f"{100.0*(IWP-IWP0)/max(IWP,1e-30):.2f}% of IWP")
        print(f"[{a.label}] tau floor-on = {float(np.sum(A*tau.sum(axis=1))):.4f}  "
              f"floor-off = {float(np.sum(A*tau0.sum(axis=1))):.4f}")

    # ---- 3. cover the sub-columns cannot see --------------------------------
    n_sub = int(getattr(config, "cloud_n_subcolumns", 0)
                or getattr(ccfg, "n_subcolumns", 0) or 8)
    thr = 1.0 / (2 * n_sub)
    small = (cf > 0.0) & (cf <= thr)
    tot = float(np.sum(A[:, None] * cf))
    sml = float(np.sum(A[:, None] * np.where(small, cf, 0.0)))
    print(f"[{a.label}] n_subcolumns={n_sub}  unrepresentable cf<= {thr:.4f}")
    print(f"[{a.label}] layer-summed cover: total={tot:.4f}  in unrepresentable layers={sml:.4f}"
          f"  share={100.0*sml/max(tot,1e-30):.2f}%")
    lwp_small = float(np.sum(A[:, None] * np.where(small, lwp, 0.0)))
    iwp_small = float(np.sum(A[:, None] * np.where(small, iwp, 0.0)))
    print(f"[{a.label}] water in those layers: LWP={lwp_small:.4e} ({100.0*lwp_small/max(LWP,1e-30):.2f}%)"
          f"  IWP={iwp_small:.4e} ({100.0*iwp_small/max(IWP,1e-30):.2f}%)")


if __name__ == "__main__":
    main()
