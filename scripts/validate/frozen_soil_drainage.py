"""Frozen-soil drainage with and without the CLM5 ice impedance (informational).

Builds the AMIP driver from run_amip arguments (everything after ``--``), takes
its set-up multilayer land state and configuration, and steps the land alone for
``--days`` under a fixed cold forcing (T_air 263 K, no precipitation, no
sunlight) at the production land step, once with the impedance exponent 0 and
once with ``--exponent``.  Columns: land (f_land > 0) whose start-of-run ice
fraction exceeds 0.3 in any of the top 5 layers, chosen once and shared by both
arms.  Reports area-weighted drainage and surface runoff [mm] and the water
budget residual of each arm; the budget check is first shown to catch a planted
missing drainage term.  No acceptance gate and no climate claim.

    python scripts/validate/frozen_soil_drainage.py --days 5 -- <run_amip args>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_BUDGET_TOL_KG_M2 = 1.0e-3        # per step, float64 (plan v3)
_ICE_FRAC_SELECT = 0.3
_TOP_LAYERS = 5
_DOY = 1.0                         # 1 January, the IC date
_COLD = dict(T_air=263.0, q_air=1.0e-3, lw_down=220.0, u=4.0)


def water_storage(theta, dz, pond, canopy, snow):
    """Column water [kg/m2]: soil + pond + canopy store + snow."""
    from legoesm import constants
    return (np.sum(theta * dz, axis=-1) + pond) * constants.rho_water + canopy + snow


def budget_residual(s0, s1, precip, evap, runoff, drainage, dt):
    """d(storage) - (P - E - R - D) dt [kg/m2]; fluxes in kg/m2/s."""
    return (s1 - s0) - (precip - evap - runoff - drainage) * dt


def step_runoff(drainage_state, freshwater, held):
    """(surface runoff, drainage) [kg/m2/s] of one land step.  A held column
    reverted its state (stale drainage field) and zeroed its response, so it
    moved no water: both are 0 there and the column is counted, not budgeted."""
    dr = np.where(held, 0.0, drainage_state)
    return np.where(held, 0.0, freshwater - dr), dr


def budget_verdict(resid_max, n_checked):
    """True/False against the tolerance; None when no column-step was budgeted."""
    if n_checked == 0:
        return None
    return resid_max < _BUDGET_TOL_KG_M2


def area_mean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _self_check():
    s0 = np.array([100.0]); d = np.array([2e-4]); dt = 1800.0
    s1 = s0 - d * dt
    assert abs(budget_residual(s0, s1, 0.0, 0.0, 0.0, d, dt)[0]) < 1e-12
    planted = budget_residual(s0, s1, 0.0, 0.0, 0.0, 0.0 * d, dt)
    assert abs(planted[0]) > _BUDGET_TOL_KG_M2, "budget check cannot see a missing drainage term"


def _driver(run_amip_argv):
    sys.path.insert(0, str(_ROOT / "scripts" / "run"))
    import run_amip as ra
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.run_config_yaml import load_yaml_config
    p = ra.build_arg_parser()
    pre, _ = p.parse_known_args(run_amip_argv)
    if pre.config:
        k = load_yaml_config(pre.config, p, example_keys="")
        p.set_defaults(**k)
        p.set_defaults(_config_keys=frozenset(k))
    a = p.parse_args(run_amip_argv)
    ra._resolve_land_model(a, p, run_amip_argv)
    a = ra._postprocess_args(a, p, run_amip_argv)
    d = ModelDriver(ra.build_config_from_args(a))
    d.setup()
    return d, a


def _run_arm(d, st0, cfg, idx, n_steps, dt):
    """Step the land alone on the packed land columns ``idx``, as the driver does."""
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.multilayer_land import (
        gather_land_columns, step_multilayer_land_with_diagnostics)
    ph = d.physics
    ncol = st0.theta_soil.shape[0]

    def pack(tree):
        return gather_land_columns(tree, idx, ncol)

    n = idx.shape[0]
    o = jnp.ones(n)
    p_s = 1.0e5 * o
    f = AtmToSurface(
        sw_down=0.0 * o, lw_down=_COLD["lw_down"] * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=_COLD["T_air"] * o, q_lowest=_COLD["q_air"] * o,
        u_lowest=_COLD["u"] * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * _COLD["T_air"]), cos_zenith=0.0 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)
    cfg_p, params_p, lat_p = pack(cfg), pack(ph.land_ml_params), pack(ph.land_ml_lat)
    carbon = getattr(ph, "land_ml_carbon", None)
    carbon_p = pack(carbon) if carbon is not None else None
    u_min = float(getattr(ph, "land_ml_u_min", 1.0))

    @jax.jit
    def step(st):
        new, resp, _, sfc = step_multilayer_land_with_diagnostics(
            st, f, cfg_p, u_min, dt, lat=lat_p, doy=_DOY,
            land_params=params_p, carbon_state=carbon_p)
        held = (jnp.zeros(n, bool) if sfc.held is None
                else jnp.asarray(sfc.held).reshape(-1))
        return new, resp.surface_mass_flux, resp.freshwater_flux, held

    st = pack(st0)
    for _ in range(n_steps):
        new, evap, fw, held = step(st)
        yield st, new, np.asarray(evap), np.asarray(fw), np.asarray(held)
        st = new


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: frozen_soil_drainage.py [--days N] [--exponent E] -- <run_amip args>")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--exponent", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv[:i])
    if not args.exponent > 0.0:
        raise SystemExit("--exponent must be > 0 (the comparison arm is e = 0)")
    _self_check()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.land.multilayer_land import soil_ice_log_impedance
    from legoesm.land.soil_grid import make_soil_grid

    d, ra_args = _driver(argv[i + 1:])
    cfg0 = d.physics.land_ml_cfg
    if not cfg0.thermal.enable_freeze_thaw:
        raise SystemExit("freeze/thaw is off in this configuration: the impedance is inert")
    dt = float(d.config.land_update_seconds) or float(d.config.dt)
    st0 = d._land_ml_state
    f_land = np.asarray(d._f_land).reshape(-1)
    idx_all = np.nonzero(f_land > 0.0)[0]
    from legoesm.grids.factory import create_grid
    area = np.asarray(create_grid("mpas", resolution=ra_args.resolution).areaCell)[idx_all]
    dz = np.asarray(make_soil_grid(cfg0.soil_grid).dz)
    li = np.asarray(soil_ice_log_impedance(
        st0.T_soil, st0.theta_soil,
        cfg0._replace(richards=cfg0.richards._replace(ice_impedance_exponent=1.0))))
    icefrac = (-li / np.log(10.0))[idx_all]
    mask = icefrac[:, :_TOP_LAYERS].max(axis=1) > _ICE_FRAC_SELECT
    if not mask.any():
        raise SystemExit("no frozen land columns selected")
    w = area * f_land[idx_all] * mask
    idx = jnp.asarray(idx_all)
    n_steps = int(round(args.days * 86400.0 / dt))

    def storage(s):
        can = np.asarray(s.W_canopy) if getattr(s, "W_canopy", None) is not None else 0.0
        return water_storage(np.asarray(s.theta_soil), dz, np.asarray(s.surface_water),
                             can, np.asarray(s.snow_depth))

    res = {}
    for e in (0.0, args.exponent):
        cfg = cfg0._replace(richards=cfg0.richards._replace(ice_impedance_exponent=e))
        drain = runoff = 0.0
        resid_max = 0.0
        n_held = n_checked = n_over = 0
        for s0, s1, evap, fw, held in _run_arm(d, st0, cfg, idx, n_steps, dt):
            # phase-aware vapour flux (sublimation at L_s) and the tile's total
            # runoff to the ocean (surface incl. snowmelt + drainage)
            ro, dr = step_runoff(np.asarray(s1.runoff_subsurface), fw, held)
            r = budget_residual(storage(s0), storage(s1), 0.0,
                                np.where(held, 0.0, evap), ro, dr, dt)
            ok = mask & ~held
            n_held += int(np.sum(mask & held))
            n_checked += int(np.sum(ok))
            n_over += int(np.sum(ok & (np.abs(r) >= _BUDGET_TOL_KG_M2)))
            if not np.all(np.isfinite(r[ok])):
                raise SystemExit(f"non-finite water budget in arm e={e:g}")
            if ok.any():
                resid_max = max(resid_max, float(np.max(np.abs(r[ok]))))
            drain += area_mean(dr * dt, w); runoff += area_mean(ro * dt, w)
        res[f"e={e:g}"] = dict(drainage_mm=drain, surface_runoff_mm=runoff,
                               held_column_steps=n_held,
                               max_budget_residual_kg_m2_per_step=resid_max,
                               budgeted_column_steps=n_checked,
                               over_tol_column_steps=n_over,
                               budget_within_tol=budget_verdict(resid_max, n_checked))

    ic = getattr(ra_args, "land_ic", None)
    out = dict(
        results=res, n_columns=int(mask.sum()), days=args.days, land_dt_s=dt,
        forcing=_COLD,
        provenance=dict(
            git_sha=subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
            land_ic=ic, land_ic_md5=(hashlib.md5(Path(ic).read_bytes()).hexdigest()
                                     if ic else None),
            run_amip_argv=argv[i + 1:]))
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
