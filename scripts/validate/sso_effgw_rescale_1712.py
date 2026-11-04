"""#1712: one-shot McFarlane ``effgw_oro`` rescale for a new subgrid-orography file.

    sso_effgw_rescale_1712.py --config <deck> --sso-old <nc> --sso-new <nc>
        --effgw-oro <run value> [--out result.json] [--extra <run_amip flags...>]

S = area-weighted GLOBAL mean of |tau| at the bottom interface, tau being the
McFarlane stress output (``mcfarlane_gwd(..., return_stress=True)``; it carries
``effgw = effgw_oro * landfrac``, so ocean columns contribute 0), evaluated on
the deck's initial state (no time step) through the production MPAS GWD factory
(``make_gwd_physics(gwd_config_for(cfg), "mpas", dt_phys)``); a shim only
switches ``return_stress`` on and records the call's inputs and ``tau``.

    effgw_oro_new = effgw_oro_old * S_old / S_new           (S_new == 0 raises)

KNOWN LIMIT (GLM, PR #2192): the ratio assumes S is linear in effgw_oro; GLM
cautioned it under-corrects where the tendency limiters (``umcfac``, ``tndmax``)
bind.  Measured
on the 14 CAM6 oracle columns: the limiters reshape the stress PROFILE (max rel
diff ~0.12) but leave the bottom-interface stress unchanged and exactly linear
in effgw_oro (CAM's redetermination carries a limited divergence down a layer),
so S departs from linearity only where a limiter binds in the BOTTOM layer.  The
probe reports the columns where limited and unlimited bottom stress differ,
S(2x)/S(1x) at production limiters, and S at effgw_oro_new against S_old.  A
global scalar target preserves the global integral, not the stress pattern.

Checks printed (numbers only, no verdict):
  * doubling: with the limiters disabled (umcfac, tndmax_per_day -> max float) a 2x
    effgw_oro doubles the bottom stress in every column; with production
    limiters, in every column whose bottom stress the limiters leave unchanged;
  * 3-column hand check: the bottom stress of three such columns
    recomputed in NumPy from the scheme's documented launch formula.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
# CAM gw_oro.F90 orohmin [m]: the launch gate the hand check reproduces.
_OROHMIN_M = 10.0  # coeff-ok: documented CAM constant (mcfarlane.py)


def area_mean_abs(tau_bottom, area) -> float:
    """Area-weighted global mean of |tau_bottom|."""
    a = np.asarray(area, dtype=np.float64)
    return float(np.sum(a * np.abs(np.asarray(tau_bottom, dtype=np.float64)))
                 / np.sum(a))


def rescale_effgw(effgw_old: float, s_old: float, s_new: float) -> float:
    """``effgw_old * s_old / s_new``; ``s_new == 0`` raises."""
    if s_new == 0.0:
        raise ValueError("S_new == 0: the new SSO file launches no orographic "
                         "stress anywhere; no rescale is defined")
    return float(effgw_old) * float(s_old) / float(s_new)


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_sso1712", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_sso1712"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_driver(deck, sso, extra):
    ra = _load_run_amip()
    import tempfile
    # Each build writes its own manifest: a fresh output dir per build, so
    # the second build (other SSO path) is not refused as a different run.
    argv = ["--config", deck, *extra, "--subgrid-orography-file", sso,
            "--output", tempfile.mkdtemp(prefix="sso1712_")]
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(deck, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    if getattr(args, "distributed", False) or getattr(
            args, "multicontroller", False):
        raise SystemExit("single-process probe: S is a sum over the local "
                         "mesh; drop --distributed")
    if getattr(args, "params", None):
        raise SystemExit("--params is not applied by this probe (run_amip "
                         "applies it around setup); refusing a stress that "
                         "would not be the run's")
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)
    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    return driver


def gwd_stress(driver, mcf_overrides=None):
    """Bottom-interface stress tau[:, -1] (and the captured mcfarlane inputs)
    from the production MPAS GWD factory on the driver's current state."""
    from legoesm.atmosphere.physics.gravity_wave_drag import integration as gi
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import (
        mcfarlane_gwd)
    from legoesm.driver.physics_pipeline import gwd_config_for

    cfg = driver.config
    gcfg = gwd_config_for(cfg)
    if gcfg.scheme != "mcfarlane":
        raise SystemExit(f"deck GWD scheme is {gcfg.scheme!r}, not mcfarlane")
    if mcf_overrides:
        gcfg = gcfg._replace(mcfarlane=gcfg.mcfarlane._replace(**mcf_overrides))
    dt_phys = float(cfg.dycore.dt) * int(cfg.physics_update_steps)
    rec = {}

    def shim(*a, **kw):
        out, tau = mcfarlane_gwd(*a, **kw, return_stress=True)
        rec["tau"] = np.asarray(tau)
        rec["args"], rec["kw"] = a, kw
        return out

    real = gi.get_gwd_fn
    with mock.patch.object(gi, "get_gwd_fn",
                           lambda c: (real(c)[0], shim, real(c)[2])):
        fn = gi.make_gwd_physics(gcfg, model_type="mpas", dt=dt_phys)
    model = driver.model
    fn(driver.state, model.mesh, model.sigma_coord, None)
    if rec["kw"].get("land_frac_col") is None:
        raise SystemExit("GWD call got no land fraction: S would include ocean")
    if rec["kw"].get("h_topo_col") is None:
        raise SystemExit("GWD call got no per-column sgh (no SSO file wired)")
    return rec, dt_phys


def hand_bottom_stress(rec, cols):
    """NumPy recomputation of effgw*tau0 (no limiter) for *cols*, from the
    captured inputs, using the scheme's own source/stability helpers."""
    import jax.numpy as jnp
    from legoesm import constants as c
    from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
        gw_prof, source_direction)
    from legoesm.atmosphere.physics.gravity_wave_drag.oro_source import (
        depth_averaged_oro_source)
    u, v, T, _pf, p_half, z_full = rec["args"][:6]
    cfg = rec["args"][10]
    kw = rec["kw"]
    u, v, T, p_half, z_full = (np.asarray(x)[cols] for x in
                               (u, v, T, p_half, z_full))
    lf = np.asarray(kw["land_frac_col"])[cols]
    sgh = np.clip(np.asarray(kw["h_topo_col"])[cols], 0.0, None) / np.sqrt(lf)
    hdsp = sgh * (2.0 if cfg.use_e3sm_hdsp in (True, "auto") else 1.0)
    pmid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dp = p_half[:, 1:] - p_half[:, :-1]
    _, _, nm, _ = gw_prof(jnp.asarray(T), jnp.asarray(pmid), jnp.asarray(p_half),
                          c.c_pd, c.R_d, c.g, 5.0e-5)  # coeff-ok: CAM n2min
    rho_mid = pmid / (c.R_d * T)
    rsrc, usrc, vsrc, nsrc, _ = depth_averaged_oro_source(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(rho_mid), jnp.asarray(hdsp),
        jnp.asarray(p_half), jnp.asarray(dp), jnp.asarray(z_full), nm,
        full_column=True)
    U, _, _ = source_direction(usrc, vsrc)
    U, rsrc, nsrc = (np.asarray(x) for x in (U, rsrc, nsrc))
    U_act = U / (1.0 + np.exp(-cfg.min_wind_sharpness * (U - cfg.min_wind)))
    h2 = np.minimum(hdsp ** 2, cfg.fcrit2 * U_act ** 2 / nsrc ** 2)
    tau0 = cfg.G_0 * rsrc * nsrc * cfg.k_wave * h2 * U_act * cfg.directional_spread
    tau0 = np.where(hdsp > _OROHMIN_M, tau0, 0.0)
    return cfg.effgw_oro * lf * tau0


def _emit(res, out):
    print(json.dumps(res, indent=1), flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sso-old", required=True)
    ap.add_argument("--sso-new", required=True)
    ap.add_argument("--effgw-oro", type=float, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    e0 = a.effgw_oro
    res = {"effgw_oro_old": e0, "sso_old": a.sso_old, "sso_new": a.sso_new}

    d_old = build_driver(a.config, a.sso_old, a.extra)
    from legoesm.driver.physics_pipeline import gwd_config_for
    e_run = gwd_config_for(d_old.config).mcfarlane.effgw_oro
    if e_run != e0:
        raise SystemExit(f"--effgw-oro {e0} is not the run's resolved "
                         f"effgw_oro {e_run}")
    area = np.asarray(d_old.model.mesh.areaCell).reshape(-1)
    r_old, dt_phys = gwd_stress(d_old, {"effgw_oro": e0})
    res["dt_phys_s"] = dt_phys
    res["S_old"] = area_mean_abs(r_old["tau"][:, -1], area)
    if res["S_old"] == 0.0:
        raise SystemExit("S_old == 0: the current SSO file launches no stress")
    del d_old

    d_new = build_driver(a.config, a.sso_new, a.extra)
    if not np.array_equal(np.asarray(d_new.model.mesh.areaCell).reshape(-1), area):
        raise SystemExit("the two builds have different meshes")
    run = lambda **o: gwd_stress(d_new, o)[0]["tau"][:, -1]  # noqa: E731
    r_new, _ = gwd_stress(d_new, {"effgw_oro": e0})
    t1 = r_new["tau"][:, -1]
    res["S_new"] = area_mean_abs(t1, area)
    e_new = rescale_effgw(e0, res["S_old"], res["S_new"])
    res["effgw_oro_new"] = e_new
    res["S_at_effgw_new"] = area_mean_abs(run(effgw_oro=e_new), area)
    res["S_at_effgw_new_over_S_old"] = res["S_at_effgw_new"] / res["S_old"]

    # Doubling.  Limiter-free (umcfac, tndmax -> max float): every column doubles.
    # Largest finite float, not inf: umcfac*|ubm| with ubm == 0 (calm source
    # column) would be inf*0 = NaN.
    big = float(np.finfo(np.float64).max)
    off = {"umcfac": big, "tndmax_per_day": big}
    f1 = run(effgw_oro=e0, **off)
    f2 = run(effgw_oro=2 * e0, **off)
    if not (np.all(np.isfinite(f1)) and np.all(np.isfinite(f2))):
        raise SystemExit("non-finite limiter-free bottom stress")
    nz = f1 != 0.0
    res["doubling_nolimit_max_rel_err"] = float(
        np.max(np.abs(f2[nz] / f1[nz] - 2.0))) if nz.any() else None
    res["n_cols_nonzero"] = int(nz.sum())
    # Production limiters: columns whose limited BOTTOM stress equals the
    # limiter-free one at both 1x and 2x (limiters may still bind aloft).
    t2 = run(effgw_oro=2 * e0)
    free = nz & np.isclose(t1, f1, rtol=1e-12, atol=0) & np.isclose(
        t2, f2, rtol=1e-12, atol=0)
    res["n_cols_bottom_unlimited"] = int(free.sum())
    res["doubling_prod_bottom_unlimited_max_rel_err"] = float(
        np.max(np.abs(t2[free] / t1[free] - 2.0))) if free.any() else None
    res["S_2x_over_S_1x"] = area_mean_abs(t2, area) / res["S_new"]
    res["S_frac_in_bottom_limited_cols"] = area_mean_abs(np.where(nz & ~free, t1, 0.0),
                                                  area) / res["S_new"]

    _emit(res, a.out)
    # 3-column hand check on bottom-unlimited columns: largest, median and
    # smallest nonzero stress.
    idx = np.flatnonzero(free)
    if idx.size == 0:
        raise SystemExit("no bottom-unlimited column: hand check impossible")
    order = idx[np.argsort(t1[idx])]
    cols = np.array(sorted({int(order[-1]), int(order[len(order) // 2]),
                            int(order[0])}))
    hand = hand_bottom_stress(r_new, cols)
    res["hand_cols"] = cols.tolist()
    res["hand_tau"] = hand.tolist()
    res["model_tau"] = t1[cols].tolist()
    res["hand_max_rel_err"] = float(np.max(np.abs(hand / t1[cols] - 1.0)))

    _emit(res, a.out)


if __name__ == "__main__":
    main()
