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
    ap.add_argument("--clubb-nsub", type=int, default=1,
                    help="macro/micro sub-steps for the closure call "
                         "(the run's cld_macmic_num_steps; 3 => 600 s on an 1800 s step)")
    ap.add_argument("--partition", action="store_true",
                    help="ALSO size TurbulenceConfig.liquid_partition: run the same "
                         "closure call with the host's q_c seeded into rt, take "
                         "the liquid it writes back, and push it through the "
                         "deck's own optics. This is the lever's effect, not an "
                         "upper bound built from an unseeded call.")
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
    q_c_partitioned_out = None
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
        _ns = max(1, int(a.clubb_nsub))
        _dts = dt_phys / _ns
        _u = u_cell.reshape(nCells, nlev); _v = v_cell.reshape(nCells, nlev)
        _T = T; _q = q_v; _m = moments; _rho = rho
        from legoesm.atmosphere.physics._shared import virtual_temperature
        print(f"[{a.label}] closure call: {_ns} sub-step(s) of {_dts:.1f} s "
              f"(the run's physics step is {dt_phys:.1f} s)")
        for _i in range(_ns):
            out = clubb_step(_u, _v, _T, _q, _m, p_full, p_half, z_full, z_half,
                             T_sfc, q_sfc, _rho, _dts, clubb_cfg)
            _du, _dv, _dT, _dq, _m, diags = out
            if _i < _ns - 1:
                _u = _u + _dts * _du; _v = _v + _dts * _dv
                _T = _T + _dts * _dT; _q = _q + _dts * _dq
                _tv = jnp.maximum(virtual_temperature(_T, _q), clubb_cfg.T0 * 0.5)
                _rho = p_full / (constants.R_d * _tv)
        # GRID MEAN, not the in-layer value: compute_cloud_cover divides rcm by
        # a vertical cloud fraction <= 1 at cloud edges, so diags["rcm"] is the
        # in-cloud water and integrating it as a grid mean inflates the
        # inventory and the missing-share (codex).
        rcm_a = np.asarray(diags["rcm_grid"])     # ascending (bottom-up)
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

        # ---- 1b. the LEVER itself ------------------------------------------
        # The band quoted before this came from an UNSEEDED call: rt held only
        # vapour, so the closure condensed from scratch and its rcm was an upper
        # bound. With the partition on, a layer that already holds liquid carries
        # it in rt, so the closure REPARTITIONS instead of condensing, which is
        # the quantity the lever actually delivers.
        if a.partition and q_c is not None:
            _u = u_cell.reshape(nCells, nlev); _v = v_cell.reshape(nCells, nlev)
            _T = T; _q = q_v; _m = moments; _rho = rho; _ql = q_c
            for _i in range(_ns):
                _du, _dv, _dT, _dq, _m, _dg = clubb_step(
                    _u, _v, _T, _q, _m, p_full, p_half, z_full, z_half,
                    T_sfc, q_sfc, _rho, _dts, clubb_cfg, q_c=_ql)
                _ql = _ql + _dts * _dg["dq_c_dt"]
                if _i < _ns - 1:
                    _u = _u + _dts * _du; _v = _v + _dts * _dv
                    _T = _T + _dts * _dT; _q = _q + _dts * _dq
                    _tv = jnp.maximum(virtual_temperature(_T, _q), clubb_cfg.T0 * 0.5)
                    _rho = p_full / (constants.R_d * _tv)
            q_c_partitioned = jnp.maximum(_ql, 0.0)
            _p0 = colint(q_c)
            _p1 = colint(q_c_partitioned)
            print(f"[{a.label}] PARTITION: q_c {_p0:.6e} -> {_p1:.6e} kg/m2  "
                  f"(x{_p1 / max(_p0, 1e-30):.3f})")
            _above900 = np.asarray(p_full) < 9.0e4
            _dpn = np.asarray(dp)
            _m900 = lambda x: float(np.sum(
                A * np.sum(np.where(_above900, np.asarray(x), 0.0) * _dpn, axis=1)
                / constants.g))
            print(f"[{a.label}] PARTITION above 900 hPa: q_c "
                  f"{_m900(q_c):.6e} -> {_m900(q_c_partitioned):.6e} kg/m2 "
                  f"(x{_m900(q_c_partitioned) / max(_m900(q_c), 1e-30):.3f})  "
                  f"[the sub-900-hPa band is contaminated by the probe's "
                  f"saturated surface humidity]")
            q_c_partitioned_out = q_c_partitioned
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

    if a.partition and q_c_partitioned_out is not None:
        _qcp_new = q_c_partitioned_out
        cp_p = compute_cloud_properties(
            T, p_full, q_v, dp, ccfg, q_cloud=_qcp_new, q_ice=q_i,
            n_ice=n_i, n_cloud=n_c, cloud_fraction_override=cf_override,
            p_half=p_half, lat=jnp.asarray(mesh.latCell).reshape(nCells))
        lwp_p = np.asarray(cp_p.lwp); reff_p = np.asarray(cp_p.r_eff_liq)
        LWP_p = float(np.sum(A * lwp_p.sum(axis=1)))
        tau_p = 1.5 * lwp_p / (RHO_W * np.maximum(reff_p, 1e-12))
        TAU = float(np.sum(A * tau.sum(axis=1)))
        TAU_p = float(np.sum(A * tau_p.sum(axis=1)))
        # LIQUID-ONLY sensitivity: the new liquid is pushed through the optics
        # against the ORIGINAL temperature, vapour and cloud fraction. The full
        # lever also moves T and q_v (and hence the fraction on the next step),
        # so this isolates the condensate's radiative weight rather than
        # predicting the run's response (codex).
        print(f"[{a.label}] PARTITION optics (LIQUID-ONLY sensitivity, "
              f"T/q_v/cloud-fraction held at their restart values):")
        print(f"[{a.label}] PARTITION optics: LWP {LWP:.6e} -> {LWP_p:.6e} kg/m2 "
              f"(x{LWP_p / max(LWP, 1e-30):.3f})")
        print(f"[{a.label}] PARTITION optics: liquid tau {TAU:.4f} -> {TAU_p:.4f} "
              f"(x{TAU_p / max(TAU, 1e-30):.3f})")

    # ---- 2b. WHERE the condensate is lost between the tracer and the optics --
    qcp = float(np.sum(A * np.asarray(jnp.sum(q_c * dp, axis=1) / constants.g))) if q_c is not None else float("nan")
    qip = float(np.sum(A * np.asarray(jnp.sum(q_i * dp, axis=1) / constants.g))) if q_i is not None else float("nan")
    cfg_raw = ccfg._replace(cloud_optics_inhomogeneity="constant",
                            cloud_inhomogeneity_factor=1.0)
    cpr = props(cfg_raw)
    LWPr = float(np.sum(A * np.asarray(cpr.lwp).sum(axis=1)))
    IWPr = float(np.sum(A * np.asarray(cpr.iwp).sum(axis=1)))
    print(f"[{a.label}] RETENTION tracer -> optics:")
    print(f"[{a.label}]   liquid: tracer {qcp:.6e} -> pre-thinning {LWPr:.6e} "
          f"({100.0*LWPr/max(qcp,1e-30):.1f}%) -> optics {LWP:.6e} "
          f"({100.0*LWP/max(qcp,1e-30):.1f}%); thinning keeps {100.0*LWP/max(LWPr,1e-30):.1f}%")
    print(f"[{a.label}]   ice:    tracer {qip:.6e} -> pre-thinning {IWPr:.6e} "
          f"({100.0*IWPr/max(qip,1e-30):.1f}%) -> optics {IWP:.6e} "
          f"({100.0*IWP/max(qip,1e-30):.1f}%); thinning keeps {100.0*IWP/max(IWPr,1e-30):.1f}%")
    _cfp = np.asarray(cp.cloud_fraction)
    _m = _cfp > 0.0
    print(f"[{a.label}]   cf used by optics: mean where>0 {_cfp[_m].mean():.4f}; "
          f"in-cloud liquid tau proxy "
          f"{float(np.mean(1.5*np.asarray(cpr.lwp)[_m]/np.maximum(_cfp[_m],1e-3)/(1000.0*np.maximum(np.asarray(cpr.r_eff_liq)[_m],1e-12)))):.3f}")

    # ---- 2c. convective plume condensate the arms carry --------------------
    for _n in ("conv_icwmr", "conv_mass_flux_up", "conv_precip"):
        _v = carry(_n)
        if _v is None:
            print(f"[{a.label}]   carry {_n}: absent")
            continue
        _v = np.asarray(_v)
        if _v.ndim == 2 and _v.shape[0] == nCells:
            print(f"[{a.label}]   carry {_n}: area-mean {float(np.sum(A[:, None]*_v)/_v.shape[1]):.6e} "
                  f"max {_v.max():.6e}  (column max area-mean {float(np.sum(A*_v.max(axis=1))):.6e})")
        else:
            print(f"[{a.label}]   carry {_n}: area-mean {float(np.sum(A*_v.reshape(nCells))):.6e} "
                  f"max {_v.max():.6e}")

    # ---- 2d. WHERE the water sits, on pressure so the two grids compare -----
    _edges = np.array([0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1100.]) * 100.0
    pm = np.asarray(p_full); dpn2 = np.asarray(dp)
    qcn2 = np.asarray(q_c) if q_c is not None else np.zeros_like(pm)
    qin2 = np.asarray(q_i) if q_i is not None else np.zeros_like(pm)
    cfn2 = np.asarray(cp.cloud_fraction)
    lwpn = np.asarray(cp.lwp)
    _rcm_b = rcm if rcm_done else np.zeros_like(pm)
    print(f"[{a.label}] pressure band | liquid path | ice path | mean cf where>0 | optics lwp | CLUBB rcm path")
    for lo, hi in zip(_edges[:-1], _edges[1:]):
        m = (pm >= lo) & (pm < hi)
        if not m.any():
            continue
        lp = float(np.sum(A[:, None] * np.where(m, qcn2 * dpn2, 0.0)) / constants.g)
        ip = float(np.sum(A[:, None] * np.where(m, qin2 * dpn2, 0.0)) / constants.g)
        mm = m & (cfn2 > 0.0)
        cfm = float(np.sum(A[:, None] * np.where(mm, cfn2, 0.0)) / max(np.sum(A[:, None] * mm), 1e-30))
        ow = float(np.sum(A[:, None] * np.where(m, lwpn, 0.0)))
        rp = float(np.sum(A[:, None] * np.where(m, _rcm_b * dpn2, 0.0)) / constants.g)
        print(f"[{a.label}]   {lo/100:6.0f}-{hi/100:6.0f} hPa | {lp:.4e} | {ip:.4e} | {cfm:.4f} | {ow:.4e} | {rp:.4e}")

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
