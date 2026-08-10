#!/usr/bin/env python
"""Per-step surface-pressure adjustment in the first hours of a forecast.

WHAT THIS DOES AND DOES NOT MEASURE — read before quoting a number.

It runs the dycore with ZERO physics and records the area-weighted RMS of
``p_s(t) - p_s(0)`` every step. That quantity is the model's EVOLUTION away from
its initial state. It is NOT an error: the real atmosphere also moves, and over
6 h the true change is ~258 Pa (persistence error at 6 h with the representation
floor removed). Measured here, the zero-physics dycore moves 313 Pa in 6 h —
the same order — so this probe CANNOT separate correct evolution from wrong
evolution. Only a comparison against ERA5 at the verification time can, and that
is what the WB2 scorecard already does.

It was written to test whether the arms' ~1000 Pa 6 h mslp error is a purely
dynamical initialisation shock. It does not answer that, and the first version
of this docstring claimed it would. What it DOES establish, and these stand:

  * **No land/ocean asymmetry.** 311.9 Pa over land vs 314.6 Pa over ocean at
    6 h, and equal at every step. The ``p_s * exp(delta_phis/(R_d T_sfc))``
    reconciliation to smoothed topography acts ONLY where topography was
    smoothed, so it cannot be the mechanism. REFUTED.
  * **Global mean p_s is conserved** to +0.15 Pa over 6 h, confirming the
    dry-mass anchor behaves in a free run.
  * **The evolution magnitude is not anomalous** — 313 Pa of dynamical motion
    against a true 6 h change of ~258 Pa. The dycore is not flinging the state
    around; whatever produces the ~1000 Pa error is not gross over-activity.

To actually attribute the 6 h error, the next probe must compare against the
ERA5 verification and decompose into amplitude vs phase/pattern — not measure
displacement from t=0.

Committed rather than run as a heredoc: this number decides where the next
GPU-hours go, so it has to be re-runnable against a changed dycore.

Usage:
    JAX_ENABLE_X64=1 python scripts/validate/aimip_pressure_spinup.py \
        --suite config/aimip/ace2/suite_curriculum_v2.yaml --variant column_nn \
        --hours 6 --n-cases 2 --out results/aimip_ps_spinup.json
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
        "run_aimip_spinup", _REPO / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_aimip_spinup"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", default="column_nn",
                   choices=("classical", "column_nn", "sfno_full"))
    p.add_argument("--eval-year", type=int, default=2017, dest="eval_year")
    p.add_argument("--n-cases", type=int, default=2, dest="n_cases")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument(
        "--vs-era5", action="store_true", dest="vs_era5",
        help="Instead of displacement from t=0, compare the 6 h forecast "
             "against the ERA5 verification and split the mslp error into a "
             "CASE-INVARIANT pattern and per-case residual. A large invariant "
             "fraction means a systematic base-state offset (correctable); a "
             "small one means genuine per-case forecast error.")
    p.add_argument(
        "--land-phis-threshold", type=float, default=1000.0,
        dest="land_phis_threshold",
        help="Surface geopotential [m2/s2] above which a WB2 cell counts as "
             "LAND. NOT 0: the spectral state's phis rings under truncation "
             "(min -1801 m2/s2 at T63) and bilinear regridding spreads it, so "
             "a >0 test calls 68%% of the AREA land against a true ~29%%. "
             "1000 m2/s2 (~100 m) reproduces 0.298 area-weighted.")
    p.add_argument(
        "--vs-reconciled", action="store_true", dest="vs_reconciled",
        help="Compare the forecast p_s DIRECTLY against the ERA5 state AFTER "
             "the same phis reconciliation the initial condition went through "
             "(case k+1's init_state, one cadence later). No sea-level "
             "reduction is involved, so this separates a real p_s base-state "
             "offset from amplification by the mslp diagnostic — and because "
             "the reconciliation is present on BOTH sides it cancels, so a "
             "fixed offset that VANISHES here is caused by the reconciliation "
             "while one that PERSISTS is the dycore's own.")
    p.add_argument("--smoothing-passes", type=int, default=4,
                   dest="smoothing_passes",
                   help="ERA5 orography smoothing passes (default 4). Sweeping "
                        "this is the perturbation test for the reconciliation "
                        "attribution: fewer passes = smaller delta_phis = "
                        "smaller p_s reconciliation, so the fixed offset "
                        "against raw ERA5 should shrink if that is the cause.")
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a trained params.eqx for --variant. When given, the "
             "rollout uses the TRAINED physics (built exactly as the WB2 eval "
             "builds it, incl. sponge + spectral filter + per-case forcing) "
             "instead of zero physics. Combined with --vs-reconciled this is "
             "the discriminator for 'physics doubles the 6 h mslp error': if "
             "the trained arm's DIRECT p_s error stays at the bare-dycore "
             "level (~200-460 Pa) the doubling is the sea-level-reduction "
             "diagnostic reading physics-modified T_lowest; if it reaches "
             "~1000 Pa the physics genuinely damages the p_s field.")
    p.add_argument(
        "--filter", action="store_true", dest="with_filter",
        help="Apply the suite's sponge + post-step spectral filter to the "
             "ZERO-physics rollout (they are always on when --checkpoint is "
             "given, matching the eval). NOTE: since the filter_lnps=False "
             "default landed, this filters vor/div/T only — it no longer "
             "reproduces the legacy lnps-filtered arm that measured the "
             "1108 Pa fixed offset (check out the pre-fix revision for "
             "that); it now serves as the post-fix control (measured "
             "228 Pa, vs 213 Pa unfiltered).")
    p.add_argument(
        "--global-t", action="store_true", dest="global_t",
        help="Mass-weighted GLOBAL-MEAN TEMPERATURE drift of the forecast "
             "against the ERA5 state at the SAME lead (both carried through "
             "the same ERA5->spectral path, so the transform cancels). This "
             "is the discriminator for a global-mean temperature drift: the "
             "real atmosphere's global mean barely moves over days, so a "
             "large model drift with ZERO physics implicates the dycore, "
             "while a drift that appears only with physics implicates the "
             "physics package. Reported per case and as a mean.")
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    import numpy as np
    import jax
    import jax.numpy as jnp

    ra = _load_run_aimip()
    from evaluations.wb_era5_cases import build_forecast_cases
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
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
    # sigma_top MUST come from the config: training/eval run 0.05 and the
    # factory default is 0.01 — a probe on the wrong vertical coordinate
    # measures a different model than the checkpoint (codex round-2 P0).
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    dt = float(spec_cfg.dt)
    n_steps = int(round(args.hours * 3600.0 / dt))
    # The integrated horizon must EQUAL the lead the verification uses, or the
    # two sides of the comparison span different windows and the difference is
    # a confound rather than a result.  dt=600 divides 6 h exactly; dt=700 does
    # not (31 steps = 21 700 s reported as "6 h").  Refuse rather than label.
    _achieved_s = n_steps * dt
    _want_s = args.hours * 3600.0
    if abs(_achieved_s - _want_s) > 1.0e-6 * max(_want_s, 1.0):
        raise SystemExit(
            f"--hours {args.hours} is not an integer number of dt={dt:g} s "
            f"steps: {n_steps} steps = {_achieved_s:g} s, but the ERA5 "
            f"verification is taken at {_want_s:g} s. Pick an --hours that "
            f"divides evenly (dt={dt:g} s -> {dt / 3600.0:g} h granularity), "
            "or the reported lead is not the lead that was integrated.")

    cadence = int(cfg.get("era5_cadence_hours", 6) or 6)
    # Verification must be fetched AT THE FORECAST LEAD: with a fixed
    # (cadence,) here, --hours 120 silently scored a 120 h forecast against
    # the 6 h truth (codex round-2 P1).
    verif_lead = int(round(args.hours))
    if abs(verif_lead - args.hours) > 1e-9 or verif_lead % cadence:
        raise SystemExit(
            f"--hours {args.hours} must be an integer multiple of the ERA5 "
            f"cadence ({cadence} h) so a verification snapshot exists.")
    cases = build_forecast_cases(
        TrainingERA5Config(dt_hours=cadence), grid, sigma,
        leads_hours=(verif_lead,), eval_year=args.eval_year,
        n_inits=args.n_cases, init_stride_hours=24, resolution_deg=1.5,
        smoothing_passes=args.smoothing_passes,
    )

    def zero_physics(state, grid_, sigma_, **_kw):
        """No physics at all — the whole point of the probe."""
        return jax.tree.map(jnp.zeros_like, state)

    # --- optional TRAINED physics (built exactly as the WB2 eval builds it) ---
    physics_fn = zero_physics
    rad_physics_fn = None
    rad_update_interval = 1
    sponge_factor = None
    spectral_filter = None
    use_case_forcing = False
    if args.with_filter and not args.checkpoint:
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            compute_spectral_filter,
            compute_sponge_factor,
        )
        pe_c = spec_cfg.pe_config
        if pe_c.sponge_tau > 0:
            sponge_factor = compute_sponge_factor(
                sigma.sigma_full, pe_c.sponge_sigma, pe_c.sponge_tau,
                spec_cfg.dt)
        if pe_c.spectral_filter_strength > 0:
            spectral_filter = compute_spectral_filter(
                grid.ls, grid.n_max, order=pe_c.spectral_filter_order,
                cutoff_fraction=pe_c.spectral_filter_strength)
    if args.checkpoint:
        import equinox as eqx
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            compute_spectral_filter,
            compute_sponge_factor,
        )

        wbe_spec = importlib.util.spec_from_file_location(
            "run_aimip_wb2_eval_spinup",
            _REPO / "scripts" / "validate" / "run_aimip_wb2_eval.py")
        wbe = importlib.util.module_from_spec(wbe_spec)
        sys.modules["run_aimip_wb2_eval_spinup"] = wbe
        wbe_spec.loader.exec_module(wbe)

        skeleton = wbe._build_skeleton(args.variant, cfg, spec_cfg, grid)
        trained = eqx.tree_deserialise_leaves(args.checkpoint, skeleton)
        if args.variant == "column_nn":
            from legoesm.training.neural_gcm_spectral import (
                make_column_mlp_spectral_physics,
            )
            physics_fn = make_column_mlp_spectral_physics(trained, grid)
            use_case_forcing = True
        elif args.variant == "classical":
            from legoesm.training.aimip_params import (
                make_aimip_classical_spectral_physics,
            )
            rad_update_interval = int(cfg.get("aimip_rad_update_interval", 1))
            built = make_aimip_classical_spectral_physics(
                trained, grid, spec_cfg.dt,
                radiation=str(cfg.get("aimip_radiation", "gray")),
                rad_update_interval_steps=rad_update_interval,
                convection_scheme=str(cfg.get("aimip_convection", "tiedtke")),
                turbulence_scheme=str(cfg.get("aimip_turbulence", "louis")),
                surface_bulk_scheme=str(
                    cfg.get("aimip_surface_bulk_scheme", "constant")),
                gwd_scheme=str(cfg.get("aimip_gwd", "mcfarlane")),
                microphysics_scheme=str(cfg.get("aimip_microphysics", "none")),
                cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
                land_mask=None,
                split_rad=rad_update_interval > 1,
            )
            if isinstance(built, tuple):
                physics_fn, rad_physics_fn = built
            else:
                physics_fn = built
                # mirror the eval: the non-split classical rollout receives
                # forcing_base; the split-rad branch does not.
                use_case_forcing = True
        else:
            raise SystemExit(
                f"--checkpoint supports classical/column_nn, not {args.variant}")
        # The trained arms ran (and were scored) WITH the sponge and spectral
        # filter; a checkpointed rollout without them would be a train/eval
        # mismatch, so mirror run_aimip_wb2_eval._build_rollout_fn here.
        pe_c = spec_cfg.pe_config
        if pe_c.sponge_tau > 0:
            sponge_factor = compute_sponge_factor(
                sigma.sigma_full, pe_c.sponge_sigma, pe_c.sponge_tau,
                spec_cfg.dt)
        if pe_c.spectral_filter_strength > 0:
            spectral_filter = compute_spectral_filter(
                grid.ls, grid.n_max, order=pe_c.spectral_filter_order,
                cutoff_fraction=pe_c.spectral_filter_strength)

    def _roll(init_state, n, forcing=None):
        kw = {}
        if rad_physics_fn is not None:
            kw.update(rad_physics_fn=rad_physics_fn,
                      rad_update_interval=rad_update_interval)
        if use_case_forcing and forcing is not None:
            kw.update(forcing_base=forcing)
        return spectral_rollout(
            init_state, physics_fn, grid, sigma, spec_cfg.pe_config,
            dt, n, sponge_factor, spectral_filter, **kw)

    area = jnp.asarray(grid.grid_area, dtype=jnp.float64)
    # Land/ocean split from the model's own surface geopotential: the p_s
    # reconciliation only acts where topography was smoothed, so an ocean signal
    # cannot come from it. phis is on the case's initial state.
    def _ps(state):
        return jnp.exp(sh_synthesis(grid, state.lnps_hat.data))

    def _rms(d, mask=None):
        w = area if mask is None else area * mask
        return float(jnp.sqrt(jnp.sum(w * d ** 2) / jnp.maximum(jnp.sum(w), 1e-30)))

    if args.global_t:
        from legoesm.grids.gaussian import sh_synthesis_3d

        lead_strides = int(round(args.hours / cadence))
        if abs(lead_strides * cadence - args.hours) > 1e-9 or lead_strides < 1:
            raise SystemExit(
                f"--hours {args.hours} must be a positive multiple of the "
                f"ERA5 cadence ({cadence} h): the truth state is another init "
                "state and only exists on the cadence grid.")
        cases_t = build_forecast_cases(
            TrainingERA5Config(dt_hours=cadence), grid, sigma,
            leads_hours=(cadence,), eval_year=args.eval_year,
            n_inits=args.n_cases + lead_strides, init_stride_hours=cadence,
            resolution_deg=1.5, smoothing_passes=args.smoothing_passes)

        dsig = np.asarray(sigma.dsigma if hasattr(sigma, "dsigma")
                          else np.diff(np.asarray(sigma.sigma_half)))
        w = np.asarray(area)
        w = w / w.sum()

        def _tbar(state):
            """Mass-weighted global-mean temperature [K].

            sigma-coordinate layer mass is p_s * dsigma / g, so the column
            weight is p_s * dsigma; the horizontal weight is the Gaussian
            cell area. Both are needed — an unweighted mean on this grid
            over-counts the poles and ignores that thick low layers hold
            most of the mass.
            """
            T = np.asarray(sh_synthesis_3d(grid, state.T_hat.data))
            ps = np.asarray(_ps(state))
            m = ps[:, :, None] * dsig[None, None, :]
            return float((w[:, :, None] * m * T).sum() / (w[:, :, None] * m).sum())

        rows = []
        for k in range(len(cases_t) - lead_strides):
            s0 = cases_t[k].init_state
            truth = cases_t[k + lead_strides].init_state
            fc = _roll(s0, n_steps, getattr(cases_t[k], "forcing", None))
            t0, tf, tt = _tbar(s0), _tbar(fc), _tbar(truth)
            rows.append({
                "case": k,
                "T_mean_init_K": t0,
                "T_mean_forecast_K": tf,
                "T_mean_era5_at_lead_K": tt,
                "model_drift_K": tf - t0,
                "era5_change_K": tt - t0,
                "error_vs_era5_K": tf - tt,
            })
        out = {
            "meta": {
                "what": "mass-weighted global-mean temperature: model drift "
                        "vs the true ERA5 change over the same window. "
                        "|model_drift| >> |era5_change| with zero physics "
                        "implicates the dycore; only-with-physics implicates "
                        "the physics package.",
                "suite": args.suite, "variant": args.variant,
                "eval_year": args.eval_year, "n_cases": len(rows),
                "hours": args.hours, "checkpoint": args.checkpoint,
                "sigma_top": float(spec_cfg.sigma_top),
                "n_levels": int(spec_cfg.n_levels),
            },
            "cases": rows,
            "mean_model_drift_K": float(np.mean([r["model_drift_K"] for r in rows])),
            "mean_era5_change_K": float(np.mean([r["era5_change_K"] for r in rows])),
            "mean_error_vs_era5_K": float(np.mean([r["error_vs_era5_K"] for r in rows])),
        }
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    if args.vs_reconciled:
        # Inits one cadence apart: case k+m's init_state IS the ERA5 truth at
        # case k's verification time for a forecast of m cadences, carried
        # through the SAME ERA5 -> spectral path (including the phis
        # reconciliation) as the IC.
        lead_strides = int(round(args.hours / cadence))
        if abs(lead_strides * cadence - args.hours) > 1e-9 or lead_strides < 1:
            raise SystemExit(
                f"--hours {args.hours} must be a positive multiple of the "
                f"ERA5 cadence ({cadence} h): the reconciled truth is another "
                "init state and only exists on the cadence grid.")
        cases_c = build_forecast_cases(
            TrainingERA5Config(dt_hours=cadence), grid, sigma,
            leads_hours=(cadence,), eval_year=args.eval_year,
            n_inits=args.n_cases + lead_strides, init_stride_hours=cadence,
            resolution_deg=1.5,
            # Same smoothing on the truth side, or the "reconciliation on
            # both sides cancels" claim breaks for --smoothing-passes != 4.
            smoothing_passes=args.smoothing_passes)
        errs = []
        for k in range(len(cases_c) - lead_strides):
            fc = _roll(cases_c[k].init_state, n_steps,
                       getattr(cases_c[k], "forcing", None))
            e = (np.asarray(_ps(fc))
                 - np.asarray(_ps(cases_c[k + lead_strides].init_state)))
            errs.append(e)
        E = np.stack(errs)
        mean_pat = E.mean(axis=0)
        # ALL statistics area-weighted: an unweighted mean on a Gaussian grid
        # over-counts the poles (CLAUDE.md area-weights gate; codex round-2).
        w = np.asarray(area)
        w = w / w.sum()

        def _wms(field2d, mask=None):
            ww = w if mask is None else w * mask
            s = ww.sum()
            if s <= 0.0:
                raise SystemExit(
                    "empty weight mask in p_s statistics (bad "
                    "--land-phis-threshold?) — a zero-weight mean would "
                    "print 0 Pa and read as a perfect forecast.")
            return float((ww * field2d ** 2).sum() / s)

        v_tot = float(np.mean([_wms(e) for e in E]))
        v_fix = _wms(mean_pat)
        v_res = float(np.mean([_wms(e - mean_pat) for e in E]))
        phis_g = np.asarray(sh_synthesis(grid, cases_c[0].init_state.phis_hat.data))
        thr = float(args.land_phis_threshold)
        land = (phis_g > thr).astype(np.float64)
        out = {
            "meta": {
                "what": "forecast p_s vs the RECONCILED ERA5 p_s at the "
                        "matching lead (case k+m's init state). No sea-level "
                        "reduction; the reconciliation is on both sides and "
                        "cancels. All statistics area-weighted.",
                "suite": args.suite, "variant": args.variant,
                "eval_year": args.eval_year, "n_cases": len(errs),
                "hours": args.hours, "land_phis_threshold": thr,
                "checkpoint": args.checkpoint,
                "sigma_top": float(spec_cfg.sigma_top),
            },
            "ps_rms_total_pa": float(np.sqrt(v_tot)),
            "ps_rms_fixed_pattern_pa": float(np.sqrt(v_fix)),
            "ps_rms_residual_pa": float(np.sqrt(v_res)),
            "fixed_fraction_of_variance": v_fix / max(v_tot, 1e-30),
            "ps_fixed_land_pa": float(np.sqrt(_wms(mean_pat, land))),
            "ps_fixed_ocean_pa": float(np.sqrt(_wms(mean_pat, 1.0 - land))),
            # Is the fixed pattern a CONSTANT offset or a structured field?
            # |mean| ~ RMS means a uniform shift; |mean| << RMS means structure.
            "ps_fixed_area_mean_pa": float((w * mean_pat).sum()),
            "ps_fixed_zonal_profile_pa": [
                float(v) for v in np.sqrt(np.mean(mean_pat ** 2, axis=1))],
            "lat_deg": [float(v) for v in np.rad2deg(np.asarray(grid.lat))],
        }
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    if args.vs_era5:
        from evaluations.wb_forecast import diagnose_and_regrid

        errs = []
        for case in cases:
            state = _roll(case.init_state, n_steps,
                          getattr(case, "forcing", None))
            fields, valid, _lat, _lon = diagnose_and_regrid(
                state, grid, sigma, resolution_deg=1.5)
            verif = case.verif_by_lead[verif_lead]
            e = np.asarray(fields["mslp"]) - np.asarray(verif["fields"]["mslp"])
            m = np.asarray(valid["mslp"]).astype(bool)
            errs.append(np.where(m, e, np.nan))
        E = np.stack(errs)                       # (n_cases, n_lat, n_lon)
        mean_pat = np.nanmean(E, axis=0)         # case-invariant component
        resid = E - mean_pat[None]
        # Variance split, over cells valid in every case.
        # LAND/OCEAN SPLIT OF THE FIXED PATTERN. This is the test that
        # implicates or clears the orography reconciliation
        # (``p_s * exp(delta_phis/(R_d T_sfc))``), which acts ONLY where
        # topography was smoothed. An earlier version of this analysis "refuted"
        # that mechanism using the land/ocean symmetry of p_s(t) - p_s(0) — the
        # WRONG QUANTITY, because the reconciliation happens AT t=0 and so
        # cancels out of a displacement from t=0 entirely. The ERA5-referenced
        # error is where its signature lives.
        ok = np.all(np.isfinite(E), axis=0)
        # Model orography regridded to the WB2 grid, so land can be masked on
        # the same grid the error lives on.
        phis_g = np.asarray(sh_synthesis(grid, cases[0].init_state.phis_hat.data))
        from evaluations.wb_regrid import regrid_to_wb2 as _rg
        # regrid_to_wb2 returns (field, tgt_lat, tgt_lon) — a 3-tuple, not an
        # array — and takes 1-D source coordinates in DEGREES while grid.lat /
        # grid.lon are in RADIANS. Both read from the source, after np.asarray
        # on the tuple raised "inhomogeneous shape (3,)".
        phis_wb2, _tlat, _tlon = _rg(
            jnp.asarray(phis_g),
            np.rad2deg(np.asarray(grid.lat)),
            np.rad2deg(np.asarray(grid.lon)),
            resolution_deg=1.5)
        phis_wb2 = np.asarray(phis_wb2)
        # cos(lat) weights on the WB2 lat-lon grid (CLAUDE.md area-weights
        # gate; matches the WB2 scorecard's own weighting). Weight zero where
        # any case is invalid so every statistic runs over the same cells.
        if not ok.any():
            raise SystemExit(
                "every mslp cell is invalid in at least one case — refusing "
                "to report statistics over an empty domain (a zero-weight "
                "mean would print 0 Pa and read as a perfect forecast).")
        w2 = np.cos(np.deg2rad(np.asarray(_tlat)))[:, None] * np.ones_like(phis_wb2)
        w2 = np.where(ok, w2, 0.0)
        w2 = w2 / w2.sum()

        def _wms2(field2d, mask=None):
            ww = w2 if mask is None else w2 * mask
            s = ww.sum()
            if s <= 0.0:
                raise SystemExit(
                    "empty weight mask in mslp statistics (bad "
                    "--land-phis-threshold?) — a zero-weight mean would "
                    "print 0 Pa and read as a perfect forecast.")
            return float(np.nansum(ww * field2d ** 2) / s)

        v_tot = float(np.mean([_wms2(np.where(ok, e, 0.0)) for e in E]))
        v_fix = _wms2(np.where(ok, mean_pat, 0.0))
        v_res = float(np.mean(
            [_wms2(np.where(ok, r, 0.0)) for r in resid]))
        _thr = float(args.land_phis_threshold)
        land_m = ((phis_wb2 > _thr) & ok).astype(np.float64)
        sea_m = ((phis_wb2 <= _thr) & ok).astype(np.float64)
        fix_land = (float(np.sqrt(_wms2(np.where(ok, mean_pat, 0.0), land_m)))
                    if land_m.any() else None)
        fix_sea = (float(np.sqrt(_wms2(np.where(ok, mean_pat, 0.0), sea_m)))
                   if sea_m.any() else None)

        out = {
            "mslp_fixed_pattern_land_pa": fix_land,
            "mslp_fixed_pattern_ocean_pa": fix_sea,
            "land_cells": int(land_m.sum()), "ocean_cells": int(sea_m.sum()),
            "land_phis_threshold": float(args.land_phis_threshold),
            "meta": {
                "what": "6 h ZERO-PHYSICS forecast vs ERA5: mslp error split "
                        "into a case-invariant pattern and a per-case "
                        "residual. High invariant fraction = systematic "
                        "base-state offset, not a skill failure.",
                "suite": args.suite, "variant": args.variant,
                "eval_year": args.eval_year, "n_cases": len(errs),
                "hours": args.hours, "n_max": int(spec_cfg.n_max),
                "n_levels": int(spec_cfg.n_levels),
                "checkpoint": args.checkpoint,
            },
            "mslp_rms_total_pa": float(np.sqrt(v_tot)),
            "mslp_rms_fixed_pattern_pa": float(np.sqrt(v_fix)),
            "mslp_rms_residual_pa": float(np.sqrt(v_res)),
            "fixed_fraction_of_variance": v_fix / max(v_tot, 1e-30),
        }
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    if args.checkpoint:
        raise SystemExit(
            "--checkpoint is not supported in the per-step displacement "
            "(series) mode: the series advances via independent 1-step "
            "rollouts, which resets the radiation cache/gating clock and the "
            "mass-anchor target every step — a trained-physics series would "
            "not be the model the checkpoint runs in eval. Use --vs-era5 or "
            "--vs-reconciled (single continuous rollout).")

    series = []
    for case in cases:
        s0 = case.init_state
        # The state carries ``phis_hat`` (SPECTRAL), not ``phis``. A
        # getattr(s0, "phis", None) returns None and silently disables the
        # land/ocean split — the exact silent-degradation pattern this campaign
        # keeps tripping over — so synthesise it and fail loudly if absent.
        if not hasattr(s0, "phis_hat"):
            raise AttributeError(
                "initial state has no phis_hat; the land/ocean split cannot be "
                "built and a whole-globe number alone would not discriminate "
                "the orography-reconciliation mechanism.")
        phis = sh_synthesis(grid, s0.phis_hat.data)
        # Use the FLAG, not a hardcoded 1.0: phis is synthesised from
        # phis_hat, so spectral ringing puts small positive values over ocean
        # and a fixed 1 m threshold misclassifies them as land — which is
        # exactly what --land-phis-threshold exists to raise.  The two other
        # call sites already honour it.
        land = (phis > float(args.land_phis_threshold)).astype(jnp.float64)
        ocean = 1.0 - land
        p0 = _ps(s0)
        rows = []
        state = s0
        for k in range(1, n_steps + 1):
            state = _roll(state, 1, getattr(case, "forcing", None))
            d = _ps(state) - p0
            rows.append({
                "step": k,
                "hours": k * dt / 3600.0,
                "rms_dps_pa": _rms(d),
                "rms_dps_land_pa": _rms(d, land),
                "rms_dps_ocean_pa": _rms(d, ocean),
                "mean_dps_pa": float(
                    jnp.sum(area * d) / jnp.sum(area)),
            })
        series.append(rows)

    # Average the per-step curves across cases.
    n = len(series[0])
    avg = []
    for i in range(n):
        row = {"step": series[0][i]["step"], "hours": series[0][i]["hours"]}
        for key in ("rms_dps_pa", "rms_dps_land_pa", "rms_dps_ocean_pa",
                    "mean_dps_pa"):
            vals = [s[i][key] for s in series if s[i][key] is not None]
            row[key] = float(np.mean(vals)) if vals else None
        avg.append(row)

    out = {
        "meta": {
            "what": "area-weighted RMS of p_s(t) - p_s(0) with ZERO physics; "
                    "isolates the purely dynamical adjustment of an ERA5 "
                    "analysis onto this dycore's balanced state.",
            "suite": args.suite, "variant": args.variant,
            "eval_year": args.eval_year, "n_cases": len(series),
            "dt_s": dt, "n_steps": n_steps, "hours": args.hours,
            "n_max": int(spec_cfg.n_max), "n_levels": int(spec_cfg.n_levels),
        },
        "series": avg,
    }
    text = json.dumps(out, indent=2)
    print(json.dumps(out["meta"], indent=2))
    for r in avg:
        if r["step"] <= 6 or r["step"] % 6 == 0:
            print(f"  t={r['hours']:5.2f} h  rms={r['rms_dps_pa']:9.1f} Pa"
                  f"  land={r['rms_dps_land_pa']:9.1f}"
                  f"  ocean={r['rms_dps_ocean_pa']:9.1f}"
                  f"  mean={r['mean_dps_pa']:+9.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
