"""Water/energy closure of the coupled run in which the atmosphere owns land
and open water (``CoupledConfig.atm_owns_surface``, P5c).  Numbers only.

Run: the small MPAS coupled configuration of the owned-mode restart test
(level-3 mesh, 8 levels, dt 300 s, rrtmgp, louis, multilayer land on a
synthetic half-land mask, slab ocean, slab ice seeded 0.8 m x 30 %),
coupling window 1800 s, ``--days`` days (2), ``--micro`` microphysics
(kessler: the first scheme that rains here within the run).

Sampled every exchange (after the coupling step); every term is a global
area integral [kg water, J energy] over the whole run:

  P_atm, E_atm   the atmosphere's per-step surface precip / evaporation
  P_<tile>       the window precip the atmosphere split by tile
  E_ow, R_land   open-water evaporation and land runoff it exported
  H_recv         ice/lake evaporation the atmosphere received (H_{n-1})
  H_debit        ice/lake evaporation the coupler debited (H_n, incl. pending)
  fw_applied     ice/lake->ocean freshwater the ocean applied (lagged)
  fw_out         ice/lake->ocean freshwater the ice produced (incl. pending)
  q_ow           open-water heat the atmosphere exported (+ into the ocean)
  ocean_heat_extr  ice basal heat the dynamic path would take from the ocean

Rows (residuals; the probe prints no verdict):
  precip_split   sum_tiles P_<tile> - P_atm
  land_water     dW_land + pend_land - (P_land - E_land - R_land),
                 E_land = E_atm - E_ow - H_recv
  slab_heat      dH_slab - (q_ow + f_w Q_flux + f_w Q_freeze)
  atm_water      dW_atm - (E_atm - P_atm)                       (report)
  ice_water      dW_ice + fw_out + H_debit - P_lake              (report)
  system_water   d(W_atm + W_land + W_ice) + ocean_water_in + pending (report)

The ocean stores no water (slab), so the water it is handed counts as stored:
ocean_water_in = P_ow + P_ice - E_ow + R_land + fw_applied.  Pending at the end
= H_N.evap x window (owed to the atmosphere) + fw_N x window (owed to the
ocean), minus the same at the start (zero by the window-0 rule), + pend_land
(rain samples the land has not consumed + held land evaporation the land
debited but the atmosphere has not yet received; zero at the start).  Lake
fraction is zero in the coupled driver, so lake terms are 0.

``--energy-terms`` adds the land longwave-emission terms [J] (report only):
  E_land_atm     land emission the atmosphere's radiation assumed:
                 f_land eps_l sigma T_l^4 at the solve's boundary, held
                 between solves like the solve
  A_band_land    the land's share (by broadband emission) of the solve's
                 emission minus that broadband emission (band Planck vs
                 sigma T^4): eps_eff L_down - lw_net - eps_eff sigma T_rad^4
  E_land_charged emission the land charges: f_land eps_l sigma T_start^4 x
                 DT_LAND per land call (T_start = the land's own skin before
                 the call, eps_l = the emissivity the call exports, DT_LAND =
                 the rounded step count x DT)
  stale = E_land_charged - E_land_atm, band = A_band_land; the atmosphere
  and the land disagree on land emission by stale - band.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STORES = ("W_atm", "W_land", "W_ice", "H_slab")
# Water rows pass at |residual| <= this fraction of the run's precipitation.
# fp32 rounding alone leaves ~3e-4 (atm), so the bar is 1e-3 (vote
# 2026-11-23: author 1e-3, GLM 1e-3, codex 5e-4).
WATER_BAR_REL = 1.0e-3


def water_bar_failures(rel: dict) -> list:
    """Water rows of ``rel`` (residual / precipitation) outside the bar;
    a NaN row fails.  Pure."""
    return [r for r, v in rel.items()
            if r != "slab_heat" and not abs(v) <= WATER_BAR_REL]


def budget_rows(s0: dict, s1: dict, f: dict) -> dict:
    """Residual rows from start/end storages ``s0``/``s1`` (keys ``STORES``)
    and run-integrated fluxes ``f`` (keys as the module docstring).  Pure."""
    d = {k: s1[k] - s0[k] for k in STORES}
    e_land = f["E_atm"] - f["E_ow"] - f["H_recv"]
    p_split = f["P_ow"] + f["P_ice"] + f["P_lake"] + f["P_land"]
    ocean_in = (f["P_ow"] + f["P_ice"] - f["E_ow"] + f["R_land"]
                + f["fw_applied"])
    pending = f["pend_H"] + f["pend_fw"] + f["pend_land"]
    return {
        "precip_split": p_split - f["P_atm"],
        "land_water": (d["W_land"] + f["pend_land"]
                       - (f["P_land"] - e_land - f["R_land"])),
        "slab_heat": d["H_slab"] - (f["q_ow"] + f["q_flux"] + f["q_freeze"]),
        "atm_water": d["W_atm"] - (f["E_atm"] - f["P_atm"]),
        "ice_water": d["W_ice"] + f["fw_out"] + f["H_debit"] - f["P_lake"],
        "system_water": (d["W_atm"] + d["W_land"] + d["W_ice"] + ocean_in
                         + pending),
        "ocean_water_in": ocean_in,
        "pending": pending,
    }


def land_emission_terms(f_land, eps_land, T_land, eps_eff, T_rad, lw_down,
                        lw_net, sigma):
    """Per-cell [W/m2] land emission the radiation solve assumed with its
    boundary (broadband), and the land's share of the solve's emission minus
    that broadband emission.  Pure."""
    emit_bb = eps_eff * sigma * T_rad ** 4
    e_land = f_land * eps_land * sigma * T_land ** 4
    share = np.where(emit_bb > 0.0, e_land / np.where(emit_bb > 0.0, emit_bb,
                                                      1.0), 0.0)
    return e_land, share * (eps_eff * lw_down - lw_net - emit_bb)


def land_in_flight(f_land, area, win_sum_precip, held_evap, n, K, dt,
                   n_applied=None):
    """Water between land and atmosphere at a land-cadence phase [kg]: the
    ``n`` rain samples (``win_sum_precip``, sum of per-cell rates) the land
    has not consumed yet, plus the held land evaporation the land already
    debited for its ``K``-step interval but the atmosphere has applied for
    only ``n_applied`` steps (default ``n``: physics every step).  Both are
    owed to the land side.  Pure."""
    rain = (np.sum(f_land * win_sum_precip * area) * dt
            if n > 0 else 0.0)
    m = n if n_applied is None else n_applied
    return float(rain + np.sum(f_land * held_evap * area) * max(0, K - m)
                 * dt)


def steps_applied(call_step, n_steps, physics_every):
    """Steps the atmosphere has applied a land call's held evaporation by the
    end of step ``n_steps``: from the first physics step (``(s - 1) %
    physics_every == 0``) after the call's step on.  Pure."""
    s = call_step + 1
    s += (-(s - 1)) % physics_every
    return max(0, n_steps - s + 1)


def stores(a: dict, k: dict) -> dict:
    """Global storages from raw state arrays ``a`` and fixed weights ``k``.
    Pure numpy, so the injection control can perturb ``a`` directly."""
    rho_w, rho_i = k["rho_water"], k["rho_ice"]
    area, fl, fw = k["area"], k["f_land"], k["f_water"]
    w_atm = np.sum(a["q_tot"] * a["dp"], axis=-1) / k["g"]
    soil = np.sum(a["theta"] * k["dz"][None, :], axis=-1) * rho_w
    w_land = fl * (soil + a["snow"] + a["pond"] * rho_w + a["canopy"])
    w_ice = rho_i * a["h_ice"] * a["conc"] * fw
    h_slab = k["C_mix"] * a["T_slab"] * fw
    return {"W_atm": float(np.sum(w_atm * area)),
            "W_land": float(np.sum(w_land * area)),
            "W_ice": float(np.sum(w_ice * area)),
            "H_slab": float(np.sum(h_slab * area))}


def _np(x):
    return np.asarray(getattr(x, "data", x), dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Gate run (vote K5-Q2/V2): kessler 2 days; at 1-3 days morrison and at
    # 1 day kessler give exactly zero precipitation in this configuration.
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--micro", default="kessler")
    ap.add_argument("--warm-rain", default="kk2000",
                    help="morrison_warm_rain_scheme (morrison only)")
    ap.add_argument("--land-update-s", type=float, default=900.0)
    ap.add_argument("--convection", default="none",
                    help="convection scheme; 'none' is the gate run")
    # Real-world case (votes 2026-11-17, all 3-0 A): ERA5 IC through the
    # model's own ERA5->MPAS loader, a selectable vertical coordinate, and
    # the production land inputs.  Defaults keep the gate run unchanged;
    # every path is required when its option is selected (no hidden files).
    ap.add_argument("--ic", choices=("analytical", "era5"),
                    default="analytical")
    ap.add_argument("--ic-path", default="", help="ERA5 IC zarr (--ic era5)")
    ap.add_argument("--vertical-coord", default="hybrid")
    ap.add_argument("--nlev", type=int, default=None,
                    help="levels (default: the gate's 8)")
    ap.add_argument("--land", choices=("synthetic", "real"),
                    default="synthetic")
    ap.add_argument("--land-mask-file", default="")
    ap.add_argument("--clm-surfdata-path", default="")
    ap.add_argument("--surfdata", default="")
    ap.add_argument("--topography", default="")
    ap.add_argument("--trace", action="store_true",
                    help="print the land row per exchange")
    ap.add_argument("--rad-update-steps", type=int, default=1,
                    help="radiation solve cadence in steps (1 = the gate)")
    ap.add_argument("--energy-terms", action="store_true",
                    help="report the land LW emission terms")
    ap.add_argument("--grid", choices=("mpas", "fv3duo", "latlon"),
                    default="mpas",
                    help="mpas: the restart test's level-3 mesh; fv3duo: the "
                         "FV3 duo cube on the column lane (dycore knobs of "
                         "the production duo deck); latlon: the lat-lon "
                         "C-grid on the column lane (grid/dycore knobs of "
                         "the CMIP coupled decks, config/cmip)")
    ap.add_argument("--resolution", type=int, default=None,
                    help="mesh level (mpas, default 3), cube N (fv3duo) "
                         "or n_lat (latlon)")
    ap.add_argument("--dt", type=float, default=None,
                    help="dynamics step [s] (mpas default 300)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--deck", default="",
                    help="build the run from this coupled deck through "
                    "run_coupled (unknown flags go to it, e.g. data paths) "
                    "instead of the probe's own config; the slab heat row is "
                    "N/A unless the deck's ocean is the one-layer slab")
    ap.add_argument("--windows", type=int, default=0,
                    help="--deck: stop after this many coupling windows "
                    "(instead of --days)")
    ap.add_argument("--trace-atm-steps", action="store_true",
                    help="record the atmosphere's water change and E-P of "
                    "every step in the JSON (atm_steps)")
    ap.add_argument("--dump-step", type=int, nargs="+", default=[],
                    help="save step N's per-column ledger water rows, "
                    "evaporation, precipitation and water species (npz next "
                    "to --json)")
    ap.add_argument("--trace-positivity", action="store_true",
                    help="book the water the dycore's positivity stage (and "
                    "its vapour payment) changes, per call")
    args, deck_argv = ap.parse_known_args(argv)
    if args.windows and not args.deck:
        ap.error("--windows needs --deck")
    if deck_argv and not args.deck:
        ap.error(f"unrecognized arguments: {' '.join(deck_argv)}")
    if args.deck and args.energy_terms:
        ap.error("--energy-terms is not wired for --deck")
    if args.deck and args.days != int(args.days):
        ap.error("--deck needs whole --days (run_coupled takes an int)")
    own = ("micro", "warm_rain", "land_update_s", "convection", "ic",
           "ic_path", "vertical_coord", "nlev", "land", "rad_update_steps",
           "grid", "resolution", "dt")
    stray = [o for o in own if getattr(args, o) != ap.get_default(o)]
    if args.deck and stray:
        ap.error("the deck sets " + ", ".join(
            "--" + o.replace("_", "-") for o in stray))
    if args.deck:
        pass
    elif args.grid != "mpas" and (args.resolution is None or args.dt is None):
        ap.error(f"--grid {args.grid} needs --resolution and --dt")
    if args.ic == "era5" and not args.ic_path:
        ap.error("--ic era5 needs --ic-path")
    real_paths = ("land_mask_file", "clm_surfdata_path", "surfdata",
                  "topography")
    if args.deck:   # the same flags, the deck's own data paths
        for p in real_paths:
            if getattr(args, p):
                deck_argv += ["--" + p.replace("_", "-"), getattr(args, p)]
    elif args.land == "real":
        missing = [p for p in real_paths if not getattr(args, p)]
        if missing:
            ap.error("--land real needs " + ", ".join(
                "--" + p.replace("_", "-") for p in missing))
    elif any(getattr(args, p) for p in real_paths):
        ap.error("land paths are read only with --land real")

    import tempfile

    import jax
    import jax.numpy as jnp
    from _pytest.monkeypatch import MonkeyPatch

    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.unit import test_mpas_multilayer_land_port as lp
    from legoesm import constants
    from legoesm.coupler.coupler import initial_handback
    from legoesm.core.state import (
        MPAS_SFC_DIAG_BASE_KEYS, MPAS_SFC_DIAG_EXTRA_KEYS)
    from legoesm.coupler.lake.config import LakeConfig
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.ice.config import SeaIceConfig

    mp = MonkeyPatch()
    if args.deck:
        land_cfg = {}
    elif args.land == "synthetic":
        lp._patch_land_loaders(mp)
        land_cfg = dict(land_mask_path="synthetic.nc")
    else:
        land_cfg = dict(land_mask_path=args.land_mask_file,
                        clm_surfdata_path=args.clm_surfdata_path,
                        surfdata_path=args.surfdata,
                        topography=args.topography)
    if args.ic == "era5":
        land_cfg.update(ic="era5", ic_path=args.ic_path,
                        start_year=1979)  # vote 2026-11-17 Q7
    nlev = lp.MPAS_NLEV if args.nlev is None else args.nlev
    extra = ({"morrison_warm_rain_scheme": args.warm_rain}
             if args.micro == "morrison" else {})
    win = 1800.0
    DT = lp.DT if args.dt is None else args.dt
    if args.grid == "mpas":
        grid = GridConfig(grid_type="mpas", nlev=nlev,
                          resolution=args.resolution or lp.MPAS_RES,
                          vertical_coord=args.vertical_coord)
        dycore = DycoreConfig(discretization="mpas", dt=DT)
    elif args.grid == "latlon":
        # the CMIP coupled decks' lat-lon atmosphere (latlon_cgrid, no polar
        # filter, energy fixer on = the default); owned mode reads the
        # handback only on the column loop, hence column_lane
        grid = GridConfig(grid_type="latlon", nlev=nlev,
                          resolution=args.resolution,
                          vertical_coord=args.vertical_coord)
        dycore = DycoreConfig(discretization="latlon_cgrid", dt=DT,
                              column_lane=True)
    else:
        # the column lane of the duo production deck
        # (config/amip/amip_production_fv3duo_c24.yaml): every other duo
        # dycore field already sits at the deck's value by default
        grid = GridConfig(grid_type="cubed_sphere", nlev=nlev,
                          resolution=args.resolution,
                          vertical_coord=args.vertical_coord)
        dycore = DycoreConfig(discretization="fv3_duo", dt=DT,
                              column_lane=True, fv3_duo_fill=True)
    cfg = ExperimentConfig(
        grid=grid, dycore=dycore,
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=args.days, dataset="analytical", radiation="rrtmgp",
        convection=args.convection, turbulence="louis", microphysics=args.micro,
        precision="fp64", **land_cfg,
        use_multilayer_land=True, multilayer_n_layers=4,
        multilayer_soil_depth=2.0, land_beta_soil=True,
        land_surface_scheme="simple_seb", land_params_refresh=False,
        land_update_seconds=args.land_update_s,
        rad_update_steps=args.rad_update_steps, **extra)
    cc = PRESETS["aquaplanet"]()._replace(
        coupling_dt=win, atm_owns_surface=True, f_land_mode="analytical")
    if args.deck:
        from scripts.run.run_coupled import main as run_coupled_main
        d = run_coupled_main(
            ["--config", args.deck, "--days", str(int(args.days)),
             "--output", tempfile.mkdtemp(), *deck_argv], build_only=True)
        cfg, cc = d._atm.config, d.coupled_cfg
        if not cc.atm_owns_surface:
            raise SystemExit("--deck: the probe books the owned surface only")
        win = float(cc.coupling_dt)
        disc = cfg.dycore.discretization
        args.grid = ("fv3duo" if disc == "fv3_duo" else
                     "latlon" if cfg.grid.grid_type == "latlon" else "mpas")
        nlev = cfg.grid.nlev
        DT = float(cfg.dycore.dt)
        land_cfg, extra = {"deck": args.deck, "deck_argv": deck_argv}, {}
    else:
        d = CoupledESMDriver(cfg, cc, output_dir=tempfile.mkdtemp())
    d.setup()
    if args.windows:
        args.days = args.windows * win / 86400.0
        d._atm.config = d._atm.config._replace(days=args.days)
    if not args.deck:   # a deck keeps its own ice
        ice = d._sfc_state.ice
        d._sfc_state = d._sfc_state._replace(ice=ice._replace(
            h_ice=ice.h_ice.replace(data=jnp.full_like(ice.h_ice.data, 0.8)),
            concentration=ice.concentration.replace(
                data=jnp.full_like(ice.concentration.data, 0.3))))
        d._surface_handback = initial_handback(
            d._sfc_state, d._tile_config, d._ice_config or SeaIceConfig(),
            d._lake_config or LakeConfig())
    atm = d._atm
    # the driver's CFL clamps can shorten the step (the lat-lon pole cell):
    # every per-step sum below is over the step the model actually took
    DT_req, DT = DT, float(atm.config.dycore.dt)
    tc = d._tile_config
    ocfg = cc.ocean_config
    area = _np(atm.grid.grid_area).reshape(-1)
    n = area.size
    f_land = _np(tc.f_land).reshape(-1)
    f_lake = _np(tc.f_lake).reshape(-1)
    k = dict(area=area, f_land=f_land, f_water=1.0 - f_land - f_lake,
             dz=np.asarray(atm._land_soil_dz(), dtype=np.float64),
             # one-layer slab only: the two-layer slab's deep layer and its
             # restoring are not booked, so its heat row is N/A (NaN)
             C_mix=(ocfg.rho_ocean * ocfg.c_ocean * ocfg.h_mix
                    if ocfg.mode == "slab" else float("nan")),
             g=constants.g, rho_water=constants.rho_water,
             rho_ice=constants.rho_ice)
    print("CONFIG", json.dumps(dict(grid=args.grid,
                                    resolution=cfg.grid.resolution,
                                    days=args.days, micro=cfg.microphysics,
                                    convection=cfg.convection,
                                    ic=cfg.ic, ic_path=cfg.ic_path,
                                    vertical_coord=cfg.grid.vertical_coord,
                                    nlev=nlev, land=args.land,
                                    land_cfg=land_cfg,
                                    land_update_s=cfg.land_update_seconds,
                                    extra=extra, ncells=n, window=win,
                                    rad_update_steps=cfg.rad_update_steps,
                                    dt=DT, dt_requested=DT_req,
                                    f_lake_max=float(f_lake.max()),
                                    f_land_mean=float(f_land.mean()))))
    print(f"INSTR area_sum/4piR2-1 = "
          f"{area.sum() / (4 * np.pi * constants.R_earth ** 2) - 1:.3e}")
    WATER = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")
    carried = [s for s in WATER if atm.current_tracer(s) is not None]
    land0 = atm._land_ml_state
    print("STORES water species", carried, "land extras",
          {f: getattr(land0, f) is not None for f in (
              "surface_water", "W_canopy", "snow_bands", "ice_bands",
              "snow_ice_layers", "canopy_state")})
    if land0.snow_bands is not None or land0.ice_bands is not None \
            or land0.canopy_state is not None:
        raise SystemExit("land store not covered by this probe")

    def layer_dp():
        """The layer mass the lane carries [Pa]: the hybrid dp(p_s) of
        MPAS and the lat-lon C-grid (prognostic p_s); the duo's own
        prognostic delp (Lagrangian, moved by the water the physics
        applies), not the hybrid profile of its p_s."""
        if args.grid != "fv3duo":
            return _np(atm.sigma.layer_thickness_dp(atm.state.p_s.data)
                       ).reshape(n, -1)
        m = atm.model
        ci = slice(m.ng, m.ng + m.n)
        delp = np.asarray(m._faces6(atm.state.native)["state"]["delp"])
        return m._columns(delp[:, ci, ci]).reshape(n, -1)

    if args.grid == "fv3duo":
        hyb = _np(atm.sigma.layer_thickness_dp(atm.state.p_s.data)
                  ).reshape(n, -1)
        print(f"INSTR start max|delp - dp_hybrid(p_s)|/dp = "
              f"{np.max(np.abs(layer_dp() - hyb) / hyb):.3e}")

    def species_totals():
        dp = layer_dp()
        return {s: float(np.sum(np.sum(_np(atm.current_tracer(s)).reshape(n, -1)
                                       * dp, axis=-1) / constants.g * area))
                for s in carried}

    def arrays():
        dp = layer_dp()
        q = sum(_np(atm.current_tracer(s)).reshape(n, -1) for s in carried)
        L = atm._land_ml_state
        z = np.zeros(n)
        st = d._sfc_state.ice
        return dict(
            q_tot=q, dp=dp, theta=_np(L.theta_soil).reshape(n, -1),
            snow=_np(L.snow_depth).reshape(-1),
            pond=z if L.surface_water is None else _np(L.surface_water)
            .reshape(-1),
            canopy=z if L.W_canopy is None else _np(L.W_canopy).reshape(-1),
            h_ice=_np(st.h_ice).reshape(-1),
            conc=_np(st.concentration).reshape(-1),
            T_slab=_np(d._ocean_state.T_sfc).reshape(-1))

    n0 = len(MPAS_SFC_DIAG_BASE_KEYS)
    SL = {key: i for i, key in enumerate(MPAS_SFC_DIAG_BASE_KEYS)}
    SL.update({key: n0 + i for i, key in enumerate(MPAS_SFC_DIAG_EXTRA_KEYS)})
    F = {key: 0.0 for key in (
        "P_atm", "E_atm", "P_ow", "P_ice", "P_lake", "P_land", "E_ow",
        "R_land", "H_recv", "H_debit", "fw_applied", "fw_out", "q_ow",
        "q_flux", "q_freeze", "ocean_heat_extr", "pend_H", "pend_fw",
        "R_state", "pend_land", "boot_excess")}
    I = lambda x: float(np.sum(_np(x).reshape(-1) * area))  # noqa: E731
    F.update(E_land_atm=0.0, A_band_land=0.0, E_land_charged=0.0,
             E_land_debit=0.0)
    atm_steps, atm_prev = [], [0.0]
    hb_rec, solve, land_seen = {}, {}, {}
    if args.energy_terms:
        import legoesm.forcing.surface_utils as su
        _hsb = su.handback_surface_boundary

        def _rec_boundary(f_l, f_i, f_k, T_l, sst, eps, eps_l, *a, **kw):
            out = _hsb(f_l, f_i, f_k, T_l, sst, eps, eps_l, *a, **kw)
            hb_rec.update(f_land=_np(f_l).reshape(-1), T_land=_np(T_l)
                          .reshape(-1), eps_land=_np(eps_l).reshape(-1),
                          eps_eff=_np(out["sfc_emissivity"]).reshape(-1),
                          T_rad=_np(out["T_sfc_rad"]).reshape(-1))
            return out
        mp.setattr(su, "handback_surface_boundary", _rec_boundary)

    from legoesm.land.multilayer_land import land_skin_temperature
    dt_land = max(1, int(round(args.land_update_s / DT))) * DT

    def land_calls():
        """Book the emission of a land call made since the last look: from
        the land's own skin before it (sampled the step before) at the
        emissivity it exported (the boundary built after it)."""
        cur = getattr(atm, "_land_T_skin_last", None)
        if "T_pre" not in land_seen:     # the skin set at the run's start
            land_seen["obj"] = cur
            return
        if cur is not None and cur is not land_seen.get("obj"):
            c = (f_land * hb_rec["eps_land"] * constants.sigma_sb
                 * land_seen["T_pre"] ** 4)
            F["E_land_charged"] += I(c) * dt_land
            land_seen["c_pend"] = land_seen["c_last"] = c
            land_seen["n"] = land_seen.get("n", 0) + 1
            land_seen["obj"] = cur

    def eterm_step(sd):
        land_calls()
        # the skin the next land call (after this step) starts from
        land_seen["T_pre"] = _np(land_skin_temperature(
            atm._land_ml_state, atm.physics.land_ml_cfg)).reshape(-1)
        new_solve = n_steps[0] % args.rad_update_steps == 0
        if new_solve:
            solve.update(hb_rec, lw_down=_np(sd[SL["lw_down_sfc"]]).reshape(-1),
                         lw_net=_np(sd[SL["lw_net_sfc"]]).reshape(-1))
        e_land, band = land_emission_terms(
            solve["f_land"], solve["eps_land"], solve["T_land"],
            solve["eps_eff"], solve["T_rad"], solve["lw_down"],
            solve["lw_net"], constants.sigma_sb)
        if "e_first" not in land_seen:
            land_seen["e_first"] = I(e_land)
        if new_solve and land_seen.get("c_pend") is not None:
            # this solve's land emission vs the emission the land call
            # before it charged (the land exports its START-of-step skin
            # as the radiating temperature, so they are the same cells'
            # same T if the solve consumed that call's export)
            d = e_land - land_seen.pop("c_pend")
            land_seen["pair_sum"] = land_seen.get("pair_sum", 0.0) + I(d)
            land_seen["pair_max"] = max(land_seen.get("pair_max", 0.0),
                                        float(np.max(np.abs(d))))
            land_seen["n_pair"] = land_seen.get("n_pair", 0) + 1
        F["E_land_atm"] += I(e_land) * DT
        F["A_band_land"] += I(band) * DT

    pos_rec = {}
    if args.trace_positivity:
        # host-side totals of the water each stage changes, from inside the
        # jitted dycore step (area x dp x sum of water species)
        import legoesm.core.conservation as cons

        def _total(tr, dp, area):
            q = sum((f.data if hasattr(f, "data") else f)
                    for k_, f in tr.items() if k_ in WATER)
            return jnp.sum(jnp.asarray(area)[..., None] * dp * q) / constants.g

        def _book(name):
            def cb(b, a):
                pos_rec[name] = pos_rec.get(name, 0.0) + float(a) - float(b)
            return cb

        def _wrap(fn, name, area_of):
            def w(tracers, T, dp, *a, **kw):
                area = area_of(kw)          # before: the inner stage reads it
                out = fn(tracers, T, dp, *a, **kw)
                jax.debug.callback(_book(name), _total(tracers, dp, area),
                                   _total(out[0], dp, area))
                return out
            return w
        _awp, _pay = cons.apply_water_positivity, cons.pay_condensate_deficit_from_vapour
        _area2d = [None]

        def _area_kw(kw):
            _area2d[0] = kw["area"]
            return kw["area"]
        mp.setattr(cons, "apply_water_positivity",
                   _wrap(_awp, "positivity", _area_kw))
        mp.setattr(cons, "pay_condensate_deficit_from_vapour",
                   _wrap(_pay, "pay_from_vapour", lambda kw: _area2d[0]))
    if args.trace_positivity and args.grid == "latlon":
        # lat-lon dycore: the dp-weighted water each transport stage changes
        # (raw flux-form tendency, the polar filter on the tracer tendencies
        # and on dps), summed over every RK stage call [kg/s]
        import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as llm
        _tend0, _ff3, _ff2 = (llm.cgrid_latlon_pe_tendencies,
                              llm.fourier_filter_3d, llm.fourier_filter)
        ctx = {}

        def _cb(name):
            def cb(v):
                pos_rec[name] = pos_rec.get(name, 0.0) + float(v)
            return cb

        def tend_w(s_, grid_, sig, cfg_, **kw):
            out = _tend0(s_, grid_, sig, cfg_, **kw)
            dps, dq = out[3], out[4]
            area2 = jnp.asarray(grid_.area)
            dp = llm.dp_from_hybrid(sig, s_.p_s)
            dB = llm.dp_from_hybrid(sig, s_.p_s + 1.0) - dp
            names = [k_ for k_ in dq if k_ in WATER]
            qs = sum(s_.tracers[k_] for k_ in names)
            r = sum(jnp.sum(area2[..., None] * dq[k_] * dp) for k_ in names)
            r = (r + jnp.sum(area2 * jnp.sum(qs * dB, -1) * dps)) / constants.g
            jax.debug.callback(_cb("transport_raw_rate"), r)
            ctx.update(dp=dp, area=area2, qdB=jnp.sum(qs * dB, -1),
                       names=list(dq), n=0, water=set(names), in_tend=True)
            return out

        def ff3_w(field, grid_, mask):
            out = _ff3(field, grid_, mask)
            i = ctx.get("n", 99)
            ctx["n"] = i + 1
            j = i - 3                      # dT, du, dv come first
            if j >= len(ctx.get("names", ())) - 1:
                ctx["in_tend"] = False     # past this stage's last tracer
            if 0 <= j < len(ctx.get("names", ())) and \
                    ctx["names"][j] in ctx["water"]:
                jax.debug.callback(_cb("polar_filter_dq_rate"), jnp.sum(
                    ctx["area"][..., None] * (out - field) * ctx["dp"])
                    / constants.g)
            return out

        def ff2_w(field, grid_, mask):
            out = _ff2(field, grid_, mask)
            # dynamics stages only: the physics increment's dps filter is
            # an increment [Pa], not a stage rate
            if ctx.get("in_tend") and field.shape == ctx["qdB"].shape:
                jax.debug.callback(_cb("polar_filter_dps_rate"), jnp.sum(
                    ctx["area"] * ctx["qdB"] * (out - field)) / constants.g)
            return out
        mp.setattr(llm, "cgrid_latlon_pe_tendencies", tend_w)
        mp.setattr(llm, "fourier_filter_3d", ff3_w)
        mp.setattr(llm, "fourier_filter", ff2_w)
    model, orig = atm.model, atm.model.step
    n_steps = [0]
    dyn_dw = [np.zeros(2)]
    if args.trace and args.grid == "fv3duo":
        # attribution: the water the duo DYNAMICS alone changes per step
        # (bundle before vs after dyn.step, on its own delp)
        ci = slice(model.ng, model.ng + model.n)
        iw = [model.tracer_names.index(s) for s in carried]

        def w_bundle(b):
            f = model._faces6(b)
            dp = np.asarray(f["state"]["delp"])[:, ci, ci]
            q = sum(np.asarray(f["q"][i])[:, ci, ci] for i in iw)
            return np.array([float(np.sum(model._columns(x).sum(-1) * area)
                                   / constants.g) for x in (q * dp, dp)])
        dyn, dyn_step = model.dyn, model.dyn.step

        def traced_dyn_step(b, dt_):
            out_b = dyn_step(b, dt_)
            dyn_dw[0] += w_bundle(out_b) - w_bundle(b)
            return out_b
        dyn.step = traced_dyn_step

    pend_rec = [None]

    def land_check(call_step):
        """Book a new land call: the land's own evaporation debit (its held
        rate, debited ahead for the K steps the atmosphere then applies)."""
        he = (atm._land_window or {}).get("land_held_evap")
        if he is not None and he is not land_seen.get("held_obj"):
            land_seen["held_obj"] = he
            land_seen["call_step"] = call_step
            F["E_land_debit"] += I(f_land * _np(he).reshape(-1)) * max(
                1, int(round(cfg.land_update_seconds / DT))) * DT

    def finish_rec():
        """Complete the last step's record on the state it produced."""
        rec, pend_rec[0] = pend_rec[0], None
        if rec is None:
            return
        lw_, (ev_, pr_) = rec.pop("_led"), rec.pop("_sd")
        if args.trace_atm_steps:
            w_now = stores(arrays(), k)["W_atm"]
            rec["dW"] = w_now - atm_prev[0]
            atm_prev[0] = w_now
            rec["species"] = species_totals()
            if lw_ is not None:   # the physics' per-process water rows
                from legoesm.diagnostics.process_ledger import (
                    LEDGER_PROCESSES)
                rec["ledger"] = {pn: I(lw_[:, i]) * DT
                                 for i, pn in enumerate(LEDGER_PROCESSES)}
            atm_steps.append(rec)
        if rec["step"] in args.dump_step and args.json:
            np.savez(str(Path(args.json).with_suffix(
                f".step{rec['step']}.npz")),
                     ledger_water=(np.full((n, 1), np.nan) if lw_ is None
                                   else lw_),
                     evap=ev_, precip=pr_, area=area, f_land=f_land,
                     lat=_np(model.mesh.latCell).reshape(-1),
                     dp=layer_dp(),
                     **{sp: _np(atm.current_tracer(sp)).reshape(n, -1)
                        for sp in carried})

    def step(*a, **kw):
        finish_rec()
        out = orig(*a, **kw)
        sd = model._sfc_diag
        F["P_atm"] += I(sd[SL["precip"]]) * DT
        F["E_atm"] += I(sd[SL["evap_sfc"]]) * DT
        if n_steps[0] == 1:
            # the boot land call (end of step 1, published before step 2)
            # debits one DT of evaporation; the atmosphere applies that held
            # flux for K steps (attribution only)
            hb = (atm._land_window or {}).get("land_held_evap")
            if hb is not None:
                F["boot_excess"] = I(f_land * _np(hb).reshape(-1)) * (
                    max(1, int(round(cfg.land_update_seconds / DT))) - 1
                ) * DT
        # the land call runs after the dycore step, outside this hook: a
        # call first seen in step n+1 was made at the end of step n
        land_check(n_steps[0])
        # this step's fluxes; its state is read at the next call (the driver
        # installs the returned state only after this hook returns)
        rec = dict(step=n_steps[0] + 1,
                   EmP=(I(sd[SL["evap_sfc"]]) - I(sd[SL["precip"]])) * DT,
                   E=I(sd[SL["evap_sfc"]]) * DT, P=I(sd[SL["precip"]]) * DT)
        if "precip_conv_sfc" in SL and sd[SL["precip_conv_sfc"]] is not None:
            rec["P_conv"] = I(sd[SL["precip_conv_sfc"]]) * DT
        led = getattr(model, "_step_ledger", None)
        rec["_led"] = None if led is None else _np(led)[:, :, 0]
        rec["_sd"] = (_np(sd[SL["evap_sfc"]]).reshape(-1),
                      _np(sd[SL["precip"]]).reshape(-1))
        pend_rec[0] = rec
        L = atm._land_ml_state   # land's own runoff rate x f_land, per step
        F["R_state"] += float(np.sum(f_land * _np(
            L.runoff_surface + L.runoff_subsurface).reshape(-1) * area)) * DT
        if args.energy_terms:
            eterm_step(sd)
        n_steps[0] += 1
        return out

    model.step = step
    prev = {"H": d._surface_handback, "O": d._owned_to_ocean}
    n_ex = [0]

    def sample(driver, day, dt_seg):
        aux = atm._carry_aux
        for t in ("ow", "ice", "lake", "land"):
            F[f"P_{t}"] += I(aux[f"seg_precip_{t}"]) * dt_seg
        F["E_ow"] += I(aux["seg_evap_ow"]) * dt_seg
        F["R_land"] += I(aux["seg_runoff_land"]) * dt_seg
        F["q_ow"] += I(aux["seg_ow_sw_net"] + aux["seg_ow_lw_net"]
                       - aux["seg_shflx_ow"] - aux["seg_lhflx_ow"]) * dt_seg
        F["H_recv"] += I(prev["H"].evap) * dt_seg
        F["fw_applied"] += I(prev["O"].ice_lake_freshwater_flux) * dt_seg
        F["ocean_heat_extr"] += I(prev["O"].ocean_heat_extraction) * dt_seg
        H, O = d._surface_handback, d._owned_to_ocean
        F["H_debit"] += I(H.evap) * dt_seg
        F["fw_out"] += I(O.ice_lake_freshwater_flux) * dt_seg
        F["q_flux"] += float(np.sum(ocfg.Q_flux * k["f_water"] * area)) * dt_seg
        F["q_freeze"] += float(np.sum(
            _np(d._ocean_state.Q_freeze).reshape(-1) * k["f_water"] * area)
        ) * dt_seg
        prev["H"], prev["O"] = H, O
        n_ex[0] += 1
        if args.trace:   # per-window land row, to localise a land residual
            w = stores(arrays(), k)["W_land"]
            e_l = F["E_atm"] - F["E_ow"] - F["H_recv"]
            r = (w - trace["W"]) - ((F["P_land"] - trace["P"])
                                    - (e_l - trace["E"])
                                    - (F["R_land"] - trace["R"]))
            if args.grid == "fv3duo":
                wa = stores(arrays(), k)["W_atm"]
                print(f"TRACE_ATM ex {n_ex[0]:3d} dW_atm {wa - trace['WA']: .4e}"
                      f" E-P {(F['E_atm'] - F['P_atm']) - trace['EP']: .4e}"
                      f" dyn_dW {(dyn_dw[0] - trace['DY'])[0]: .4e}"
                      f" dyn_dM {(dyn_dw[0] - trace['DY'])[1]: .4e}")
                trace.update(WA=wa, EP=F["E_atm"] - F["P_atm"],
                             DY=dyn_dw[0].copy())
            print(f"TRACE ex {n_ex[0]:3d} day {day:.4f} dW_land {w - trace['W']: .4e}"
                  f" E_land {e_l - trace['E']: .4e} land_resid {r: .4e}")
            trace.update(W=w, E=e_l, P=F["P_land"], R=F["R_land"])

    a0 = arrays()
    s0 = stores(a0, k)
    atm_prev[0] = s0["W_atm"]
    sp0 = species_totals()
    trace = dict(W=s0["W_land"], E=0.0, P=0.0, R=0.0, WA=s0["W_atm"], EP=0.0,
                 DY=np.zeros(2))
    pend0 = (I(prev["H"].evap) * win
             + I(prev["O"].ice_lake_freshwater_flux) * win)
    status = d.run(segment_callback=sample)
    finish_rec()
    land_check(n_steps[0])   # a land call after the last step
    if abs(n_steps[0] * DT - args.days * 86400.0) > 1e-6 * DT:
        raise SystemExit(
            f"steps x dt = {n_steps[0]} x {DT:g} s != {args.days} d: the "
            "lane changed its step after setup; per-step sums are invalid")
    if args.energy_terms:
        land_calls()   # the call after the last step (its exported
        # emissivity: the last boundary seen; a run of whole windows ends on
        # a step whose boundary was built after the previous call)
        land_area_t = float(np.sum(f_land * area)) * n_steps[0] * DT
        stale = F["E_land_charged"] - F["E_land_atm"]
        print(f"ETERM land_calls {land_seen.get('n', 0)} steps {n_steps[0]} "
              f"K {args.land_update_s / DT:g} rad {args.rad_update_steps}")
        # lag bookkeeping, K=1 and rad=1 only (one charge per solve; at
        # other cadences a charge feeds several solves and it does not
        # telescope): stale = (last call's charge - first solve's emission)
        # x interval - sum over paired (solve - previous call)
        if dt_land == DT and args.rad_update_steps == 1:
            endpoint = (I(land_seen["c_last"]) * dt_land
                        - land_seen["e_first"] * DT)
            pairs = land_seen.get("pair_sum", 0.0) * DT
            print(f"ETERM paired {land_seen.get('n_pair', 0)} maxdiff "
                  f"{land_seen.get('pair_max', 0.0):.3e} W/m2 (solve vs the "
                  "call before it)")
            for kk, v in (("endpoint", endpoint), ("pairs", pairs),
                          ("stale_minus_lag", stale - (endpoint - pairs))):
                print(f"ETERM {kk:16s} {v: .6e} J  {v / land_area_t: .4e} "
                      "W/m2 (land mean)")
        else:
            print("ETERM lag identity not evaluated (needs K=1, rad=1)")
        for kk, v in (("E_land_atm", F["E_land_atm"]),
                      ("E_land_charged", F["E_land_charged"]),
                      ("stale", stale), ("band", F["A_band_land"]),
                      ("stale_minus_band", stale - F["A_band_land"])):
            print(f"ETERM {kk:16s} {v: .6e} J  {v / land_area_t: .4e} W/m2 "
                  "(land mean)")
    if args.grid == "fv3duo":
        hyb = _np(atm.sigma.layer_thickness_dp(atm.state.p_s.data)
                  ).reshape(n, -1)
        print(f"INSTR end max|delp - dp_hybrid(p_s)|/dp = "
              f"{np.max(np.abs(layer_dp() - hyb) / hyb):.3e}")
    a1 = arrays()
    s1 = stores(a1, k)
    sp1 = species_totals()
    # pending = final minus initial in-flight water (initial is 0 by the
    # window-0 rule; printed, and subtracted, so the row never assumes it)
    F["pend_H"] = I(prev["H"].evap) * win
    F["pend_fw"] = I(prev["O"].ice_lake_freshwater_flux) * win - pend0
    print(f"PENDING initial={pend0:.6e} kg")
    # Land <-> atmosphere in flight (the MPAS land cadence): rain samples the
    # land has not yet consumed, and the held land evaporation the land
    # already debited for the rest of its interval but the atmosphere has not
    # yet received.  None before the first land call (the run start).
    K = max(1, int(round(cfg.land_update_seconds / DT)))
    lw = atm._land_window or {}
    n_w = int(lw.get("land_win_n", 0))
    F["pend_land"] = land_in_flight(
        f_land, area,
        _np(lw["land_win_sum_precip_total"]).reshape(-1) if n_w > 0 else 0.0,
        _np(lw["land_held_evap"]).reshape(-1), n_w, K, DT,
        n_applied=steps_applied(
            land_seen["call_step"], n_steps[0],
            max(1, int(getattr(cfg, "physics_update_steps", 1) or 1))))
    print(f"PENDING land samples n={n_w} K={K} pend_land={F['pend_land']:.6e} kg")
    cw = getattr(atm, "_cpl_window", None) or {}
    if int(cw.get("cpl_win_n", 0)) != 0:
        raise SystemExit(f"run ends {int(cw['cpl_win_n'])} steps into a "
                         "coupling window; its samples are in no export. "
                         "Choose --days as a whole number of windows.")
    # H_debit / fw_out count H_0..H_N (every debit), the receivers
    # H_{-1}..H_{N-1}; the difference is the pending term.
    print("status", status, "steps", n_steps[0], "exchanges", n_ex[0])
    rows = budget_rows(s0, s1, F)
    # the land row on the land's own evaporation debits (not on the
    # atmosphere's receipt); its in-flight term is the unconsumed rain only
    rain_fl = land_in_flight(
        f_land, area,
        _np(lw["land_win_sum_precip_total"]).reshape(-1) if n_w > 0 else 0.0,
        np.zeros(n), n_w, K, DT)
    rows["land_water_debit"] = (s1["W_land"] - s0["W_land"] + rain_fl
                                - (F["P_land"] - F["E_land_debit"]
                                   - F["R_land"]))
    print(f"ROW land_water_debit abs={rows['land_water_debit']: .6e} "
          f"rel={rows['land_water_debit'] / (abs(F['P_atm']) or np.nan): .3e}"
          f" (rain in flight {rain_fl:.6e} kg)")

    # Controls: (1) null run: same state both ends -> every storage row 0.
    null = budget_rows(s0, s0, {key: 0.0 for key in F})
    # (2) injection: a known change of one state value per store.
    ctrl = {}
    i = int(np.argmax(f_land))
    j = int(np.argmax(k["f_water"]))
    for store, key, idx, delta, expect in (
            ("W_land", "theta", (i, 0), 1e-3,
             1e-3 * k["dz"][0] * constants.rho_water * f_land[i] * area[i]),
            ("W_ice", "h_ice", (j,), 1e-2,
             constants.rho_ice * 1e-2 * a1["conc"][j] * k["f_water"][j]
             * area[j]),
            ("W_atm", "q_tot", (j, 0), 1e-4,
             1e-4 * a1["dp"][j, 0] / constants.g * area[j]),
            ("H_slab", "T_slab", (j,), 0.1,
             0.1 * k["C_mix"] * k["f_water"][j] * area[j])):
        b = {kk: v.copy() for kk, v in a1.items()}
        b[key][idx] += delta
        got = stores(b, k)[store] - s1[store]
        ctrl[store] = (got - expect) / expect

    scale_w = abs(F["P_atm"]) or float("nan")
    scale_e = abs(F["q_ow"]) or float("nan")
    out = dict(status=status, steps=n_steps[0], exchanges=n_ex[0],
               stores0=s0, stores1=s1, fluxes=F, rows=rows,
               null=null, inject_rel_err=ctrl, atm_steps=atm_steps,
               species0=sp0, species1=sp1,
               rel={r: rows[r] / (scale_e if r == "slab_heat" else scale_w)
                    for r in ("precip_split", "land_water", "slab_heat",
                              "atm_water", "ice_water", "system_water")})
    for kk, v in F.items():
        print(f"FLUX {kk:16s} {v: .9e}")
    for kk in STORES:
        print(f"STORE {kk:7s} {s0[kk]: .12e} -> {s1[kk]: .12e}")
    for r, v in out["rel"].items():
        print(f"ROW {r:14s} abs={rows[r]: .6e} rel={v: .3e}")
    print(f"ATTRIB land_water - boot_excess = "
          f"{rows['land_water'] - F['boot_excess']: .6e} kg")
    for sp in carried:
        print(f"SPECIES {sp:4s} {sp0[sp]: .9e} -> {sp1[sp]: .9e}")
    if args.grid == "latlon":   # cell winds on the lat-lon column lane
        lat_c = _np(model.mesh.latCell).reshape(-1)
        cap = np.abs(lat_c) > np.deg2rad(60.0)
        uu = _np(atm.state.u.data).reshape(n, -1)
        vv = _np(atm.state.v.data).reshape(n, -1)
        print("STATE " + json.dumps(dict(
            max_abs_u=float(np.max(np.abs(uu))), max_abs_v=float(np.max(np.abs(vv))),
            polar_cap_qv_mean=float(np.sum(a1["q_tot"][cap] * a1["dp"][cap]
                                           * area[cap, None])
                                    / np.sum(a1["dp"][cap] * area[cap, None])),
            finite=bool(np.all(np.isfinite(uu)) and np.all(np.isfinite(vv))),
            min_species={sp: float(np.min(_np(atm.current_tracer(sp))))
                         for sp in carried})))
    print("NULL", {r: v for r, v in null.items()})
    if args.trace_positivity:
        print("POSITIVITY", {kk: f"{v:.6e}" for kk, v in pos_rec.items()})
    print("INJECT rel_err", ctrl)
    bad = water_bar_failures(out["rel"])
    print(f"BAR water |rel| <= {WATER_BAR_REL:g}:",
          "PASS" if not bad else f"FAIL {bad}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    mp.undo()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
