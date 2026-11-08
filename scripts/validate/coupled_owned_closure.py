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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STORES = ("W_atm", "W_land", "W_ice", "H_slab")


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


def land_in_flight(f_land, area, win_sum_precip, held_evap, n, K, dt):
    """Water between land and atmosphere at a land-cadence phase [kg]: the
    ``n`` rain samples (``win_sum_precip``, sum of per-cell rates) the land
    has not consumed yet, plus the held land evaporation the land already
    debited for its ``K``-step interval but the atmosphere still applies for
    ``K - n`` steps.  Both are owed to the land side.  Pure."""
    rain = (np.sum(f_land * win_sum_precip * area) * dt
            if n > 0 else 0.0)
    return float(rain + np.sum(f_land * held_evap * area) * (K - n) * dt)


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
    ap.add_argument("--trace", action="store_true",
                    help="print the land row per exchange")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

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
    lp._patch_land_loaders(mp)
    extra = ({"morrison_warm_rain_scheme": args.warm_rain}
             if args.micro == "morrison" else {})
    win = 1800.0
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=lp.MPAS_RES,
                        nlev=lp.MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=lp.DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=args.days, dataset="analytical", radiation="rrtmgp",
        convection="none", turbulence="louis", microphysics=args.micro,
        precision="fp64", land_mask_path="synthetic.nc",
        use_multilayer_land=True, multilayer_n_layers=4,
        multilayer_soil_depth=2.0, land_beta_soil=True,
        land_surface_scheme="simple_seb", land_params_refresh=False,
        land_update_seconds=args.land_update_s, **extra)
    cc = PRESETS["aquaplanet"]()._replace(
        coupling_dt=win, atm_owns_surface=True, f_land_mode="analytical")
    d = CoupledESMDriver(cfg, cc, output_dir=tempfile.mkdtemp())
    d.setup()
    ice = d._sfc_state.ice
    d._sfc_state = d._sfc_state._replace(ice=ice._replace(
        h_ice=ice.h_ice.replace(data=jnp.full_like(ice.h_ice.data, 0.8)),
        concentration=ice.concentration.replace(
            data=jnp.full_like(ice.concentration.data, 0.3))))
    d._surface_handback = initial_handback(
        d._sfc_state, d._tile_config, d._ice_config or SeaIceConfig(),
        d._lake_config or LakeConfig())
    atm = d._atm
    tc = d._tile_config
    ocfg = cc.ocean_config
    area = _np(atm.grid.grid_area).reshape(-1)
    n = area.size
    f_land = _np(tc.f_land).reshape(-1)
    f_lake = _np(tc.f_lake).reshape(-1)
    k = dict(area=area, f_land=f_land, f_water=1.0 - f_land - f_lake,
             dz=np.asarray(atm._land_soil_dz(), dtype=np.float64),
             C_mix=ocfg.rho_ocean * ocfg.c_ocean * ocfg.h_mix,
             g=constants.g, rho_water=constants.rho_water,
             rho_ice=constants.rho_ice)
    print("CONFIG", json.dumps(dict(days=args.days, micro=args.micro,
                                    land_update_s=args.land_update_s,
                                    extra=extra, ncells=n, window=win,
                                    dt=lp.DT, f_lake_max=float(f_lake.max()),
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

    def arrays():
        p_s = atm.state.p_s.data
        dp = _np(atm.sigma.layer_thickness_dp(p_s)).reshape(n, -1)
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
    model, orig = atm.model, atm.model.step
    n_steps = [0]

    def step(*a, **kw):
        out = orig(*a, **kw)
        sd = model._sfc_diag
        F["P_atm"] += I(sd[SL["precip"]]) * lp.DT
        F["E_atm"] += I(sd[SL["evap_sfc"]]) * lp.DT
        if n_steps[0] == 1:
            # the boot land call (end of step 1, published before step 2)
            # debits one DT of evaporation; the atmosphere applies that held
            # flux for K steps (attribution only)
            hb = (atm._land_window or {}).get("land_held_evap")
            if hb is not None:
                F["boot_excess"] = I(f_land * _np(hb).reshape(-1)) * (
                    max(1, int(round(cfg.land_update_seconds / lp.DT))) - 1
                ) * lp.DT
        L = atm._land_ml_state   # land's own runoff rate x f_land, per step
        F["R_state"] += float(np.sum(f_land * _np(
            L.runoff_surface + L.runoff_subsurface).reshape(-1) * area)) * lp.DT
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
            print(f"TRACE ex {n_ex[0]:3d} day {day:.4f} dW_land {w - trace['W']: .4e}"
                  f" E_land {e_l - trace['E']: .4e} land_resid {r: .4e}")
            trace.update(W=w, E=e_l, P=F["P_land"], R=F["R_land"])

    a0 = arrays()
    s0 = stores(a0, k)
    trace = dict(W=s0["W_land"], E=0.0, P=0.0, R=0.0)
    pend0 = (I(prev["H"].evap) * win
             + I(prev["O"].ice_lake_freshwater_flux) * win)
    status = d.run(segment_callback=sample)
    a1 = arrays()
    s1 = stores(a1, k)
    # pending = final minus initial in-flight water (initial is 0 by the
    # window-0 rule; printed, and subtracted, so the row never assumes it)
    F["pend_H"] = I(prev["H"].evap) * win
    F["pend_fw"] = I(prev["O"].ice_lake_freshwater_flux) * win - pend0
    print(f"PENDING initial={pend0:.6e} kg")
    # Land <-> atmosphere in flight (the MPAS land cadence): rain samples the
    # land has not yet consumed, and the held land evaporation the land
    # already debited for the rest of its interval but the atmosphere has not
    # yet received.  None before the first land call (the run start).
    K = max(1, int(round(cfg.land_update_seconds / lp.DT)))
    lw = atm._land_window or {}
    n_w = int(lw.get("land_win_n", 0))
    F["pend_land"] = land_in_flight(
        f_land, area,
        _np(lw["land_win_sum_precip_total"]).reshape(-1) if n_w > 0 else 0.0,
        _np(lw["land_held_evap"]).reshape(-1), n_w, K, lp.DT)
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
               null=null, inject_rel_err=ctrl,
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
    print("NULL", {r: v for r, v in null.items()})
    print("INJECT rel_err", ctrl)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    mp.undo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
