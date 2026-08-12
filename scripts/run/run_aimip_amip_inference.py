#!/usr/bin/env python
"""Prescribed-SST AMIP inference for the trained AIMIP classical model, run to
the EXACT ai2cm/AIMIP Phase-1 protocol (https://github.com/ai2cm/AIMIP).

Protocol (not improvised):
* Forcing = the official ``ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc``
  (Zenodo 10.5281/zenodo.17065758): monthly ``sea_surface_temperature`` +
  ``sea_ice_cover`` centered at month start, LINEARLY interpolated to model time
  (``training.aimip_amip_forcing``). SST/sea-ice blended to a surface temperature
  and prescribed over OCEAN only (land keeps the model's own T).
* Period 00 UTC 1 Oct 1978 -> 1 Jan 2025, Gregorian; first 3 months (Oct-Dec
  1978) are spin-up and excluded from the recorded output.
* No time-varying CO2/GHG/solar (AIMIP-1 forbids it; radiation uses the model's
  fixed defaults).
* Ensemble members r1..r5 = five successive ERA5 initial-condition days.
* Output = monthly global-mean near-surface (lowest sigma level, 2 m proxy) air
  temperature -> annual means for the fleet-figure overlay.

All three AIMIP variants are AMIP-capable (``--variant``):
* ``classical`` — the surface-flux scheme anchors near-surface air T to the
  prescribed SST via ``surface_T_sfc_override``; transient GHG + ozone feed
  RRTMGP.
* ``column_nn`` / ``sfno_physics`` — the learned physics consumes the
  prescribed SST / sea-ice / insolation directly as input features
  (``physics_fn(..., forcing=...)``), the same forcing channel it was
  trained with (``aimip_surface_forcing``); no GHG/ozone (no physical
  radiation scheme to feed).

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_aimip_amip_inference.py \\
        --ckpt results/aimip_ace2loss_assembled/classical/params.eqx \\
        --start 1978-10-01 --end 1980-01-01 --record-from 1979-01-01 --member 0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("aimip-amip")
_SPD = 86400.0


def _load_yaml(path: Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


_VARIANTS = ("classical", "column_nn", "sfno_physics")


def _merged_cfg(suite_path: Path, variant: str = "classical") -> dict:
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown AMIP inference variant {variant!r}; expected one of "
            f"{_VARIANTS}."
        )
    suite = _load_yaml(suite_path)
    base = _load_yaml(Path(suite["base"]))
    base.update(suite.get("cfg_overrides", {}) or {})
    base.update(_load_yaml(suite_path.parent / f"variant_{variant}.yaml"))
    base.setdefault("nlev", base["n_levels"])  # naming debt: nlev vs n_levels
    return base


def _build_spec_cfg(cfg: dict):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig

    return NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]), n_levels=int(cfg["nlev"]), dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15, hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01, spectral_filter_order=8,
            # Forwarded so a suite's dry-mass anchor is not silently
            # ignored on the prescribed-SST lane; default False keeps
            # every existing AMIP run byte-identical.
            fix_mass=bool(cfg.get("fix_mass", False)),
            anchor_mass_to_initial=bool(
                cfg.get("anchor_mass_to_initial", False)),
            # Energy numerics, forwarded for the same reason: the code
            # defaults are already the conserving pair (sb_centered +
            # frictional heating), but a suite that pins the legacy
            # upwind form to reproduce a pre-2026-08-11 result was
            # SILENTLY IGNORED here while run_aimip honoured it.
            vertical_advection_scheme=str(
                cfg.get("vertical_advection_scheme", "sb_centered")),
            frictional_heating=bool(cfg.get("frictional_heating", True)),
        ),
        n_epochs=1, lr=float(cfg.get("aimip_lr", 3.0e-4)),
        weight_decay=float(cfg.get("aimip_weight_decay", 1.0e-5)),
        n_train_days=1, start_year=1979, loss_config=LossConfig(),
    )


def _date(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("results/aimip_ace2loss_assembled/classical/params.eqx"))
    ap.add_argument("--variant", type=str, default="classical",
                    choices=list(_VARIANTS),
                    help="AIMIP variant to run: classical (physics), "
                         "column_nn (per-column MLP physics), sfno_physics "
                         "(SFNO physics). NN variants need a checkpoint "
                         "trained WITH aimip_surface_forcing (T_sfc/sic/"
                         "insolation input features).")
    ap.add_argument("--suite", type=Path, default=Path("config/aimip/ace2/suite.yaml"))
    ap.add_argument("--forcing-path", type=str, default=None,
                    help="AIMIP forcing .nc (default: aimip_amip_forcing.DEFAULT_AIMIP_FORCING)")
    ap.add_argument("--cache-path", type=str,
                    default="results/aimip_forcing/gaussian_forcing.npz")
    ap.add_argument("--start", type=str, default="1978-10-01")
    ap.add_argument("--end", type=str, default="2025-01-01")
    ap.add_argument("--record-from", type=str, default="1979-01-01",
                    help="Exclude spin-up before this date from recorded output.")
    ap.add_argument("--member", type=int, default=0,
                    help="Ensemble member 0..4 = IC on the N-th successive ERA5 day.")
    ap.add_argument("--convection-scheme", type=str, default=None,
                    help="Override aimip_convection (attribution 2026-07-02: "
                         "untrained-edmf drives a +3.3 K/day column heating "
                         "runaway; trained tiedtke is near-balanced).")
    ap.add_argument("--rad-update-interval", type=int, default=36)
    ap.add_argument("--reinit-yearly", action="store_true",
                    help="Hindcast-IAV mode: reload the ERA5 initial condition "
                         "every Jan 1 (+member-offset days) instead of free-"
                         "running. Annual means then isolate the prescribed-"
                         "SST-driven interannual signal from multi-year model "
                         "drift — the drift-robust IAV metric for NN variants "
                         "whose free-running stability is still being "
                         "fine-tuned. Not the AIMIP Phase-1 free-run protocol; "
                         "outputs are written with a _reinit suffix.")
    ap.add_argument("--reinit-every-months", type=int, default=0,
                    help="Sub-annual hindcast reinit: reload the ERA5 IC on "
                         "day 1 of every N-th month (1 = monthly). For a "
                         "variant whose free-run runs away WITHIN a year "
                         "(SFNO: 328->385 K by April), yearly reinit is not "
                         "enough — monthly reinit measures the SST-forced "
                         "monthly anomaly before the runaway compounds, and "
                         "the 12-month mean is the IAV signal. Overrides "
                         "--reinit-yearly when >0.")
    ap.add_argument("--wall-limit-hours", type=float, default=0.0,
                    help="Stop cleanly after this many wall hours, write a "
                         "restart + .RESUME marker (0 = no limit). The 46-yr "
                         "run (~135 h) exceeds the 72 h SLURM ceiling, so the "
                         "sbatch self-chains on the marker.")
    ap.add_argument("--checkpoint-every-days", type=int, default=30,
                    help="Write the restart every N simulated days.")
    ap.add_argument("--restart-path", type=Path, default=None,
                    help="Base path for restart files (default: "
                         "<out-dir>/amip_restart_r<member>). If present at "
                         "startup, the run RESUMES from it.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Monthly CSV path (default: results/aimip_fleet_paper/"
                         "legoesm_<variant>_amip.csv).")
    args = ap.parse_args()
    _reinit_months = int(args.reinit_every_months)
    if _reinit_months <= 0 and args.reinit_yearly:
        _reinit_months = 12
    if args.out is None:
        _suffix = f"_reinit{_reinit_months}mo" if _reinit_months > 0 else ""
        args.out = Path(
            f"results/aimip_fleet_paper/legoesm_{args.variant}_amip{_suffix}.csv"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        compute_spectral_filter, compute_sponge_factor, spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.ml.training import load_checkpoint
    from legoesm.forcing.surface_utils import blend_surface_temperature
    from legoesm.training.aimip_amip_forcing import (
        DEFAULT_AIMIP_FORCING, ghg_vmr_at_year, interp_forcing_at,
        open_arco_era5, ozone_vmr_at_date, regrid_monthly_forcing_to_gaussian,
    )
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    from legoesm.training.campaign_driver import parse_bool_flag
    from legoesm.training.aimip_spatial import land_mask_from_phis
    from legoesm.training.era5_to_state import (
        TrainingERA5Config, era5_to_spectral_carry, load_era5_ic,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_amip_rollout,
    )

    cfg = _merged_cfg(args.suite, args.variant)
    if args.convection_scheme:
        if args.variant != "classical":
            raise SystemExit(
                "--convection-scheme only applies to --variant classical "
                "(NN variants replace the physics entirely)."
            )
        # Single override point: make_aimip_* AND the sizing PhysicsState both
        # read cfg["aimip_convection"].
        cfg["aimip_convection"] = args.convection_scheme
        logger.info(f"convection override: {cfg['aimip_convection']}")
    spec_cfg = _build_spec_cfg(cfg)
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    lat_w = np.asarray(grid.weights)
    ncol = len(grid.lat) * len(grid.lon)
    nlev = spec_cfg.n_levels
    dt = spec_cfg.dt
    n_steps_day = int(_SPD / dt)

    # --- exact AIMIP forcing on the Gaussian grid (monthly, cached) ---
    forcing_path = args.forcing_path or DEFAULT_AIMIP_FORCING
    times_ns, sst_m, sic_m, land = regrid_monthly_forcing_to_gaussian(
        forcing_path, grid, cache_path=args.cache_path,
    )
    ocean = land < 0.5
    logger.info(
        f"AIMIP forcing: {sst_m.shape[0]} months, {int(ocean.sum())}/{ncol} "
        f"ocean cells, grid T{spec_cfg.n_max} L{nlev}"
    )

    # --- initial condition: member = successive ERA5 day ---
    start = _date(args.start)
    ic_date = start + _dt.timedelta(days=int(args.member))
    wb2 = TrainingERA5Config().zarr_store
    ic_slice = load_era5_ic(wb2, ic_date.year, ic_date.month, ic_date.day)
    ic_carry = era5_to_spectral_carry(ic_slice, grid, sigma)
    state = carry_to_spectral_state(ic_carry, grid)
    land_mask = land_mask_from_phis(jnp.asarray(ic_carry.phis), smooth=True)
    sigma_full_np = np.asarray(sigma.sigma_full)

    pe = spec_cfg.pe_config
    sponge = (compute_sponge_factor(sigma.sigma_full, pe.sponge_sigma, pe.sponge_tau, dt)
              if pe.sponge_tau > 0 else None)
    sfilt = (compute_spectral_filter(grid.ls, grid.n_max, order=pe.spectral_filter_order,
                                     cutoff_fraction=pe.spectral_filter_strength)
             if pe.spectral_filter_strength > 0 else None)

    ds_o3 = None
    if args.variant == "classical":
        ds_o3 = open_arco_era5()       # historical ERA5 ozone (ARCO)

        # --- trained classical model (spatial_surface=True per ace2 config) ---
        template = AIMIPClassicalParams.from_defaults(spatial_surface=True)
        params = load_checkpoint(template, args.ckpt)
        non_rad_fn, rad_fn = make_aimip_classical_spectral_physics(
            params, grid, dt,
            radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
            rad_update_interval_steps=args.rad_update_interval,
            convection_scheme=str(cfg["aimip_convection"]),
            turbulence_scheme=str(cfg["aimip_turbulence"]),
            surface_bulk_scheme=str(
                cfg.get("aimip_surface_bulk_scheme", "constant")),
            gwd_scheme=str(cfg["aimip_gwd"]),
            microphysics_scheme=str(cfg["aimip_microphysics"]),
            cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
            # Honour the suite's ablation waiver: this driver only
            # REPLAYS a trained checkpoint, so a suite that trained
            # with a family off must be able to run here too.
            allow_unfilled_families=parse_bool_flag(
    cfg.get("aimip_allow_unfilled_families", False)),
            land_mask=land_mask, split_rad=True,
        )
        sizing_ps = init_physics_state(
            ncol, nlev,
            PhysicsConfig(
                radiation=RadiationConfig(scheme="none"),
                convection=ConvectionConfig(scheme=str(cfg["aimip_convection"])),
                turbulence=TurbulenceConfig(scheme=str(cfg["aimip_turbulence"])),
                microphysics=MicrophysicsConfig(scheme=str(cfg["aimip_microphysics"])),
                gravity_wave_drag=GravityWaveDragConfig(scheme=str(cfg["aimip_gwd"])),
            ),
        )

        seg = jax.jit(lambda st, sst_col, doy, ghg, o3: spectral_amip_rollout(
            st, non_rad_fn, rad_fn, grid, sigma, pe, dt, n_steps_day,
            sst_col=sst_col, sizing_phys_state=sizing_ps,
            day_of_year_base=doy, seconds_offset=0.0,
            rad_update_interval=args.rad_update_interval,
            sponge_factor=sponge, spectral_filter=sfilt, ghg_vmr=ghg, o3_vmr=o3,
        ))
    else:
        # --- trained learned-physics model (column_nn / sfno_physics) ---
        # The prescribed SST/sea-ice + orbital calendar reach the network
        # as INPUT FEATURES via spectral_rollout(forcing_base=...) — the
        # same channel it was trained with (aimip_surface_forcing).
        from legoesm.training.neural_gcm_spectral import spectral_rollout

        if args.variant == "column_nn":
            from legoesm.atmosphere.physics.neural_physics import (
                build_column_physics,
            )
            from legoesm.training.neural_gcm_spectral import (
                make_column_mlp_spectral_physics,
            )
            nn_template = build_column_physics(
                nlev=nlev,
                hidden_dim=int(cfg.get("nn_hidden_dim", 256)),
                n_layers=int(cfg.get("nn_n_layers", 4)),
                key=jax.random.PRNGKey(int(cfg.get("nn_seed", 0))),
            )
            nn_model = load_checkpoint(nn_template, args.ckpt)
            physics_fn = make_column_mlp_spectral_physics(nn_model, grid)
        else:  # sfno_physics
            from legoesm.ml.channel_packing import PE3DChannelSpec
            from legoesm.ml.sfno import SFNO, SFNOConfig
            from legoesm.training.neural_gcm_spectral import (
                N_SFNO_FORCING_CHANNELS, make_sfno_spectral_physics,
            )
            spec = PE3DChannelSpec(nlev=nlev)
            sfno_template = SFNO(
                SFNOConfig(
                    in_channels=spec.n_channels + N_SFNO_FORCING_CHANNELS,
                    out_channels=spec.n_channels,
                    # Same fallbacks as run_aimip._build_spectral_config so a
                    # suite that omits them still builds the architecture the
                    # checkpoint was trained with (codex).
                    embed_dim=int(cfg.get("sfno_embed_dim", 128)),
                    n_blocks=int(cfg.get("sfno_n_blocks", 4)),
                    mlp_expansion=int(cfg.get("sfno_mlp_expansion", 4)),
                    residual_prediction=False,
                ),
                grid, key=jax.random.PRNGKey(int(cfg.get("sfno_seed", 0))),
            )
            sfno_model = load_checkpoint(sfno_template, args.ckpt)
            physics_fn = make_sfno_spectral_physics(sfno_model, grid)

        seg = jax.jit(lambda st, t_sfc_col, sic_col, doy: spectral_rollout(
            st, physics_fn, grid, sigma, pe, dt, n_steps_day,
            sponge, sfilt,
            forcing_base={
                "T_sfc": t_sfc_col, "sic": sic_col,
                "day_of_year": doy,
                "seconds_of_day": jnp.asarray(0.0, jnp.float64),
            },
        ))
    T_ice = float(constants.T_freeze_ocean)

    def _surf_T(s):
        T = np.asarray(spectral_pe_to_grid(s, grid, sigma)["T"])[..., -1]
        return float(np.sum(np.mean(T, axis=-1) * lat_w) / np.sum(lat_w))

    # --- restart / chaining support (46-yr run ~135 h > 72 h SLURM ceiling) ---
    import equinox as eqx
    end = _date(args.end)
    record_from = _date(args.record_from)
    monthly: dict[tuple[int, int], list[float]] = {}
    _o3_cache: dict[tuple[int, int], jnp.ndarray] = {}  # current-month ozone
    day = start

    # Variant-qualified default: classical/column_nn/sfno_physics runs in the
    # same output dir must never deserialize each other's restart (codex).
    _reinit_tag = f"_reinit{_reinit_months}mo" if _reinit_months > 0 else ""
    restart_base = args.restart_path or args.out.with_name(
        f"amip_restart_{args.variant}{_reinit_tag}_r{args.member}"
    )
    restart_eqx = Path(f"{restart_base}.eqx")
    restart_json = Path(f"{restart_base}.json")
    resume_marker = Path(f"{restart_base}.RESUME")

    def _write_csvs():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            fh.write("year,month,global_mean_surfT_K\n")
            for (y, m) in sorted(monthly):
                fh.write(f"{y},{m},{float(np.mean(monthly[(y, m)])):.5f}\n")
        annual: dict[int, list[float]] = {}
        for (y, m), vals in monthly.items():
            annual.setdefault(y, []).append(float(np.mean(vals)))
        annual_out = args.out.with_name(args.out.stem + "_annual.csv")
        with annual_out.open("w") as fh:
            fh.write("year,annual_global_mean_surfT_K\n")
            for y in sorted(annual):
                if len(annual[y]) == 12:  # full years only
                    fh.write(f"{y},{float(np.mean(annual[y])):.5f}\n")
        return annual_out

    def _save_restart(next_day):
        # Atomic: write tmp then os.replace so a mid-write kill never leaves a
        # truncated restart. State leaves serialise against the IC template.
        restart_eqx.parent.mkdir(parents=True, exist_ok=True)
        tmp_e = restart_eqx.with_suffix(".eqx.tmp")
        eqx.tree_serialise_leaves(tmp_e, state)
        os.replace(tmp_e, restart_eqx)
        meta = {
            "next_day": next_day.isoformat(),
            "monthly": [[y, m, vals] for (y, m), vals in sorted(monthly.items())],
            # Provenance: which trained checkpoint produced this trajectory.
            # Resume refuses a mismatch (a NEW ckpt silently continuing an
            # OLD — possibly runaway — trajectory burned the first sfno
            # stable-sanity run).
            "ckpt": str(args.ckpt),
        }
        tmp_j = restart_json.with_suffix(".json.tmp")
        tmp_j.write_text(json.dumps(meta))
        os.replace(tmp_j, restart_json)
        _write_csvs()  # partial results visible mid-chain
        logger.info(f"Restart written at {next_day} -> {restart_eqx}")

    if restart_eqx.exists() and restart_json.exists():
        meta = json.loads(restart_json.read_text())
        _restart_ckpt = meta.get("ckpt")

        def _resolved(p):
            try:
                return str(Path(p).resolve())
            except Exception:
                return str(p)

        if (_restart_ckpt is not None
                and _resolved(_restart_ckpt) != _resolved(args.ckpt)):
            raise SystemExit(
                f"Restart {restart_eqx} was written by ckpt={_restart_ckpt} "
                f"but this run uses --ckpt {args.ckpt}. Refusing to resume a "
                f"different model's trajectory — delete the restart (and its "
                f"CSV) to start fresh, or pass --restart-path elsewhere."
            )
        # ``state`` currently holds the IC -> identical pytree structure, so it
        # is the deserialisation template.
        state = eqx.tree_deserialise_leaves(restart_eqx, state)
        day = _date(meta["next_day"])
        monthly = {(int(y), int(m)): list(vals) for y, m, vals in meta["monthly"]}
        logger.info(f"RESUMED from {restart_eqx}: continuing at {day} "
                    f"({len(monthly)} months recorded)")

    t0_wall = time.time()
    wall_stopped = False
    stop = False
    # --- Gregorian daily loop with linearly-interpolated prescribed SST ---
    while day < end and not stop:
        # Hindcast-IAV mode: fresh ERA5 IC on day 1 of every N-th month
        # (member = successive IC days, same convention as the initial
        # condition). Each segment runs prescribed-SST from a reanalysis
        # anchor, so the recorded means carry the SST-driven interannual
        # signal without multi-year (or, at N=1, within-year) model drift.
        if (_reinit_months > 0 and day.day == 1 and day != start
                and (day.month - 1) % _reinit_months == 0):
            _rd = day + _dt.timedelta(days=int(args.member))
            _slice = load_era5_ic(wb2, _rd.year, _rd.month, _rd.day)
            state = carry_to_spectral_state(
                era5_to_spectral_carry(_slice, grid, sigma), grid,
            )
            logger.info(f"{day}: REINIT from ERA5 {_rd} "
                        f"(hindcast-IAV mode, every {_reinit_months}mo)")
        # Prescribed SST/sea-ice = the monthly forcing LINEARLY interpolated to
        # this day, held across the day's 144 dycore steps. SST varies ~0.1 K/day,
        # so daily granularity is <0.01 K from per-step interpolation and matches
        # standard AMIP daily-SST practice (codex noted the per-step alternative;
        # not worth re-architecting the rollout for a sub-0.01 K effect on tas).
        day_ns = np.datetime64(day).astype("datetime64[ns]").astype(np.int64)
        sst_c = interp_forcing_at(times_ns, sst_m, day_ns)
        sic_c = np.clip(interp_forcing_at(times_ns, sic_m, day_ns), 0.0, 1.0)
        t_sfc = np.asarray(blend_surface_temperature(sst_c, sic_c, T_ice))
        override = np.where(ocean, t_sfc, np.nan).astype(np.float64)
        doy = jnp.asarray(float(day.timetuple().tm_yday))
        if args.variant == "classical":
            # Transient historical GHG for this year (physical RRTMGP needs it
            # to produce the warming trend; traced scalars -> no retrace).
            ghg = {k: jnp.asarray(float(v))
                   for k, v in ghg_vmr_at_year(day.year).items()}
            # Transient historical ozone, refreshed monthly (ARCO read + regrid
            # is the cost; ozone is a monthly field so once/month suffices).
            key = (day.year, day.month)
            if key not in _o3_cache:
                _o3_cache.clear()  # keep only the current month
                # plev->sigma interp uses the CURRENT surface pressure (p_s
                # evolves over the multi-decade run; codex) — not the fixed
                # IC p_s.
                p_s_now = np.asarray(
                    spectral_pe_to_grid(state, grid, sigma)["p_s"]
                ).reshape(-1)
                _o3_cache[key] = jnp.asarray(
                    ozone_vmr_at_date(ds_o3, day, grid, sigma_full_np, p_s_now)
                )
            state = seg(state, jnp.asarray(override), doy, ghg, _o3_cache[key])
        else:
            # NN variants: the same ocean-masked T_sfc (+ sea-ice) enters the
            # network as input features; land cells (NaN) fall back to the
            # lowest-level air-T proxy inside the physics fn.
            sic_col = np.where(ocean, sic_c, 0.0).astype(np.float64)
            state = seg(state, jnp.asarray(override), jnp.asarray(sic_col), doy)
        if day >= record_from:
            st = _surf_T(state)
            monthly.setdefault((day.year, day.month), []).append(st)
            if not np.isfinite(st):
                logger.error(f"{day}: NON-FINITE surfT; stopping")
                stop = True
        if day.day == 1:
            logger.info(f"{day}: running (member r{args.member + 1})")
        day += _dt.timedelta(days=1)

        # Periodic restart (every N simulated days) + wall-limit clean exit.
        if (day - start).days % max(args.checkpoint_every_days, 1) == 0:
            _save_restart(day)
        if (args.wall_limit_hours > 0
                and (time.time() - t0_wall) > args.wall_limit_hours * 3600.0):
            _save_restart(day)
            wall_stopped = True
            logger.info(
                f"Wall limit {args.wall_limit_hours}h reached at {day}; "
                f"restart saved — resubmit to continue."
            )
            break

    # --- write monthly + annual means + chaining marker ---
    annual_out = _write_csvs()
    logger.info(f"Wrote {args.out} ({len(monthly)} months) + {annual_out}")
    if wall_stopped and day < end:
        resume_marker.write_text(day.isoformat())  # sbatch self-chains on this
        logger.info(f"RESTART_NEEDED -> {resume_marker}")
    else:
        resume_marker.unlink(missing_ok=True)
        if day >= end:
            logger.info("Run reached --end; chain complete.")


if __name__ == "__main__":
    main()
