#!/usr/bin/env python
"""Per-term budget of the mass-weighted global-mean temperature tendency.

Attribution instrument for the measured zero-physics cooling of
-0.48 K/day (aimip_pressure_spinup --global-t): splits d<T>_m/dt into the
T equation's grid-space terms, closes the budget against the ACTUAL
one-step change of the reported metric, and prints each term in K/day and
W/m^2. The term(s) carrying the deficit are the rewrite target; the closure
row is the check that this instrument measures what the model does.

    d/dt <T>_m,  <T>_m = sum(w ps dsig T) / sum(w ps dsig)

has two parts: the mass-weighted mean of dT/dt, and the p_s-tendency
coupling  (<T dps/dt> - <T>_m <dps/dt>) / <ps>  (all sums area+dsigma
weighted). Every dT/dt term is evaluated EXACTLY as the RHS evaluates it
(same helpers, same sigma-dot input div + v.grad(lnps), same T floor),
so a term printed here is the term the model integrates.

Usage:
    JAX_ENABLE_X64=1 python scripts/validate/aimip_t_budget.py \
        --suite config/aimip/ace2/suite_curriculum_v2_anchored.yaml \
        --hours 6 --out results/aimip_t_budget.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "run"))


def _load_run_aimip():
    spec = importlib.util.spec_from_file_location(
        "run_aimip_tbudget", _REPO / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_aimip_tbudget"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", default="column_nn")
    p.add_argument("--eval-year", type=int, default=2017, dest="eval_year")
    p.add_argument("--hours", type=float, default=6.0,
                   help="Evaluate the budget at states sampled along a "
                        "zero-physics rollout of this length (t=0 and t=end),"
                        " so a spin-up-dependent term cannot hide.")
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpy as np

    ra = _load_run_aimip()
    from evaluations.wb_era5_cases import build_forecast_cases
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        _compute_omega_gaussian,
        _compute_sigma_dot_gaussian,
        _vertical_advection_sigma_gaussian,
        _vertical_advection_sigma_sb,
        spectral_pe_tendencies,
    )
    from legoesm.grids.gaussian import (
        create_gaussian_grid,
        sh_synthesis,
        sh_synthesis_3d,
        sh_synthesis_H,
        spectral_hyperdiffusion_3d,
        uv_from_vordiv_3d,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import TrainingERA5Config
    from legoesm.training.neural_gcm_spectral import spectral_rollout

    suite = ra._load_yaml(Path(args.suite))
    cfg = ra._merge(ra._load_yaml(Path(suite["base"])),
                    suite.get("cfg_overrides") or {})
    cfg = ra._merge(cfg, ra._load_yaml(
        Path(args.suite).parent / f"variant_{args.variant}.yaml"))
    cfg["aimip_variant"] = args.variant
    spec_cfg = ra._build_spectral_config(cfg)

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    pe_config = spec_cfg.pe_config
    dt = float(spec_cfg.dt)

    cadence = int(cfg.get("era5_cadence_hours", 6) or 6)
    cases = build_forecast_cases(
        TrainingERA5Config(dt_hours=cadence), grid, sigma,
        leads_hours=(cadence,), eval_year=args.eval_year,
        n_inits=1, init_stride_hours=24, resolution_deg=1.5)

    def zero_physics(state, grid_, sigma_, **_kw):
        import jax
        return jax.tree.map(jnp.zeros_like, state)

    kappa = constants.kappa
    cp = constants.c_pd
    g = constants.g
    a = grid.radius
    dsig = np.asarray(sigma.dsigma)
    w2 = np.asarray(grid.grid_area)
    w2 = w2 / w2.sum()

    def _mw(ps, X):
        """Mass-weighted global mean of a (nlat,nlon,nlev) field [units of X]."""
        num = (w2[:, :, None] * ps[:, :, None] * dsig[None, None, :] * X).sum()
        den = (w2[:, :, None] * ps[:, :, None] * dsig[None, None, :]).sum()
        return float(num / den)

    def _wm2(ps, X):
        """Column-integrated area mean of cp*X [W/m^2] (X in K/s)."""
        return float(cp / g * (w2[:, :, None] * ps[:, :, None]
                               * dsig[None, None, :] * X).sum())

    def budget(state):
        nlev = sigma.n_levels
        div = np.asarray(sh_synthesis_3d(grid, state.div_hat.data))
        T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data))
        # Same softplus floor the RHS applies (T_min default) — bias ~0.07 K
        # but pointwise identical to what the model integrates.
        _sp = 0.1
        T = np.asarray(jnp.asarray(T)
                       + _sp * jax.nn.softplus((pe_config.T_min - jnp.asarray(T)) / _sp))
        lnps = np.asarray(sh_synthesis(grid, state.lnps_hat.data))
        ps = np.exp(lnps)
        u_cos, v_cos = uv_from_vordiv_3d(grid, state.vor_hat.data,
                                         state.div_hat.data)
        cos3 = np.clip(np.asarray(grid.cos_lat)[:, None, None], 1e-8, None)
        u = np.asarray(u_cos) / cos3
        v = np.asarray(v_cos) / cos3

        # grad(lnps) EXACTLY as the RHS builds it (spectral lon derivative +
        # sh_synthesis_H theta derivative; a finite-difference dy fed a ~2%
        # mismatch into sigma_dot/omega/adiabatic rows — codex P2).
        lnps_hat = state.lnps_hat.data
        coslat = np.clip(np.asarray(grid.cos_lat)[:, None], 1e-8, None)
        dlnps_dx = np.asarray(sh_synthesis(
            grid, 1j * np.asarray(grid.ms) * lnps_hat)) / (np.asarray(a) * coslat)
        dlnps_dy = -np.asarray(sh_synthesis_H(grid, lnps_hat)) / (
            np.asarray(a) * coslat)
        v_grad_lnps = u * dlnps_dx[:, :, None] + v * dlnps_dy[:, :, None]

        sigma_dot, D_total = _compute_sigma_dot_gaussian(
            jnp.asarray(div + v_grad_lnps), sigma)
        sigma_range = 1.0 - float(sigma.sigma_half[0])
        dlnps_dt = -np.asarray(D_total)[..., 0] / sigma_range
        dps_dt = ps * dlnps_dt

        # Dispatch on the CONFIG's scheme — hard-importing the upwind helper
        # here mis-attributed the SB run's budget (the probed 'vertical
        # advection' row was a scheme the model never integrated, and the
        # difference leaked into 'horizontal_residual').
        _vadv = {"upwind": _vertical_advection_sigma_gaussian,
                 "sb_centered": _vertical_advection_sigma_sb}[
                     pe_config.vertical_advection_scheme]
        vert = np.asarray(_vadv(jnp.asarray(T), sigma_dot, sigma))
        omega = np.asarray(_compute_omega_gaussian(
            sigma_dot, jnp.asarray(ps), jnp.asarray(dps_dt), sigma))
        p_full = ps[:, :, None] * np.asarray(sigma.sigma_full)
        if pe_config.p_floor > 0:
            p_full = np.maximum(p_full, pe_config.p_floor)
        adiab_omega = kappa * T * omega / p_full
        adiab_vgrad = kappa * T * v_grad_lnps
        T_prime_div = (T - pe_config.si_T_ref) * div

        hd = np.zeros_like(T)
        if pe_config.hyperdiff_coeff > 0 and not pe_config.implicit_hyperdiff:
            hd_hat = spectral_hyperdiffusion_3d(
                grid, state.T_hat.data, pe_config.hyperdiff_coeff,
                pe_config.hyperdiff_order)
            hd = np.asarray(sh_synthesis_3d(grid, hd_hat))

        # TOTAL dT/dt from the real RHS (the closure reference for the
        # dT-part; horizontal flux term = total - everything else).
        tend = spectral_pe_tendencies(state, grid, sigma, pe_config, None)
        dT_total = np.asarray(sh_synthesis_3d(grid, tend.T_hat.data))
        dlnps_dt_rhs = np.asarray(sh_synthesis(grid, tend.lnps_hat.data))
        dps_dt_rhs = ps * dlnps_dt_rhs

        horiz = dT_total - vert - adiab_omega - adiab_vgrad - T_prime_div - hd

        Tm = _mw(ps, T)
        den = (w2[:, :, None] * ps[:, :, None] * dsig[None, None, :]).sum()
        ps_coupling = float(
            ((w2[:, :, None] * dps_dt_rhs[:, :, None] * dsig[None, None, :]
              * (T - Tm)).sum()) / den)

        day = 86400.0
        terms = {
            "vertical_advection": (_mw(ps, vert) * day, _wm2(ps, vert)),
            "adiabatic_omega": (_mw(ps, adiab_omega) * day, _wm2(ps, adiab_omega)),
            "adiabatic_vgrad_lnps": (_mw(ps, adiab_vgrad) * day, _wm2(ps, adiab_vgrad)),
            "T_prime_div": (_mw(ps, T_prime_div) * day, _wm2(ps, T_prime_div)),
            "hyperdiffusion": (_mw(ps, hd) * day, _wm2(ps, hd)),
            "horizontal_residual": (_mw(ps, horiz) * day, _wm2(ps, horiz)),
            "ps_tendency_coupling": (ps_coupling * day, cp / g * ps_coupling
                                     * float((w2 * ps).sum())),
            "TOTAL_dTdt_mass_weighted": (_mw(ps, dT_total) * day + ps_coupling * day,
                                         _wm2(ps, dT_total)),
        }
        return terms, T, ps

    results = {}
    case = cases[0]

    # t=0 budget
    terms0, _, _ = budget(case.init_state)
    results["t0"] = {k: {"K_per_day": v[0], "W_per_m2": v[1]}
                     for k, v in terms0.items()}

    # budget after a short zero-physics rollout (post-adjustment state)
    n_steps = int(round(args.hours * 3600.0 / dt))
    state_end = spectral_rollout(
        case.init_state, zero_physics, grid, sigma, pe_config, dt,
        n_steps, None, None)
    terms1, _, _ = budget(state_end)
    results[f"t{int(args.hours)}h"] = {
        k: {"K_per_day": v[0], "W_per_m2": v[1]} for k, v in terms1.items()}

    # CLOSURE: actual finite-difference change of the reported metric over
    # one step, vs the budget's TOTAL. The instrument is only trusted if
    # these agree (RK3 vs forward-Euler difference ~ O(dt * tendency
    # curvature) — expect % level, not factors).
    def _tm(state):
        T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data))
        ps = np.exp(np.asarray(sh_synthesis(grid, state.lnps_hat.data)))
        return _mw(ps, T)

    s1 = spectral_rollout(case.init_state, zero_physics, grid, sigma,
                          pe_config, dt, 1, None, None)
    fd = (_tm(s1) - _tm(case.init_state)) / dt * 86400.0
    results["closure"] = {
        "one_step_finite_difference_K_per_day": fd,
        "budget_total_t0_K_per_day": terms0["TOTAL_dTdt_mass_weighted"][0],
        "note": "FD includes RK3 + post-step chain (anchor); budget is the "
                "instantaneous RHS at t=0. Agreement to ~10% validates the "
                "instrument; the anchor and filter are OFF in this rollout "
                "call except the anchor inside spectral_rollout.",
    }
    results["meta"] = {
        "suite": args.suite, "eval_year": args.eval_year,
        "hours": args.hours, "n_levels": int(spec_cfg.n_levels),
        "sigma_top": float(spec_cfg.sigma_top), "dt": dt,
    }

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
