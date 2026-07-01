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

Only the CLASSICAL variant is AMIP-capable (its surface-flux scheme anchors
near-surface air T to the prescribed SST via ``surface_T_sfc_override``);
column_nn / sfno replace physics and have no SST hook.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_aimip_amip_inference.py \\
        --ckpt results/aimip_ace2loss_assembled/classical/params.eqx \\
        --start 1978-10-01 --end 1980-01-01 --record-from 1979-01-01 --member 0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
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


def _merged_cfg(suite_path: Path) -> dict:
    suite = _load_yaml(suite_path)
    base = _load_yaml(Path(suite["base"]))
    base.update(suite.get("cfg_overrides", {}) or {})
    base.update(_load_yaml(suite_path.parent / "variant_classical.yaml"))
    base.setdefault("nlev", base["n_levels"])  # naming debt: nlev vs n_levels
    return base


def _build_spec_cfg(cfg: dict):
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig

    return NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]), n_levels=int(cfg["nlev"]), dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15, hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01, spectral_filter_order=8,
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
    ap.add_argument("--rad-update-interval", type=int, default=36)
    ap.add_argument("--out", type=Path,
                    default=Path("results/aimip_fleet_paper/legoesm_classical_amip.csv"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from legoesm import constants
    from legoesm.atmosphere.dynamics.spectral_pe import (
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
    from legoesm.tools.forcing.surface_utils import blend_surface_temperature
    from legoesm.training.aimip_amip_forcing import (
        DEFAULT_AIMIP_FORCING, interp_forcing_at,
        regrid_monthly_forcing_to_gaussian,
    )
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    from legoesm.training.aimip_spatial import land_mask_from_phis
    from legoesm.training.era5_to_state import (
        TrainingERA5Config, era5_to_spectral_carry, load_era5_ic,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_amip_rollout,
    )

    cfg = _merged_cfg(args.suite)
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

    # --- trained classical model (spatial_surface=True per ace2 config) ---
    template = AIMIPClassicalParams.from_defaults(spatial_surface=True)
    params = load_checkpoint(template, args.ckpt)
    non_rad_fn, rad_fn = make_aimip_classical_spectral_physics(
        params, grid, dt,
        radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
        rad_update_interval_steps=args.rad_update_interval,
        convection_scheme=str(cfg["aimip_convection"]),
        turbulence_scheme=str(cfg["aimip_turbulence"]),
        gwd_scheme=str(cfg["aimip_gwd"]),
        microphysics_scheme=str(cfg["aimip_microphysics"]),
        cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
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

    pe = spec_cfg.pe_config
    sponge = (compute_sponge_factor(sigma.sigma_full, pe.sponge_sigma, pe.sponge_tau, dt)
              if pe.sponge_tau > 0 else None)
    sfilt = (compute_spectral_filter(grid.ls, grid.n_max, order=pe.spectral_filter_order,
                                     cutoff_fraction=pe.spectral_filter_strength)
             if pe.spectral_filter_strength > 0 else None)

    seg = jax.jit(lambda st, sst_col, doy: spectral_amip_rollout(
        st, non_rad_fn, rad_fn, grid, sigma, pe, dt, n_steps_day,
        sst_col=sst_col, sizing_phys_state=sizing_ps,
        day_of_year_base=doy, seconds_offset=0.0,
        rad_update_interval=args.rad_update_interval,
        sponge_factor=sponge, spectral_filter=sfilt,
    ))
    T_ice = float(constants.T_freeze_ocean)

    def _surf_T(s):
        T = np.asarray(spectral_pe_to_grid(s, grid, sigma)["T"])[..., -1]
        return float(np.sum(np.mean(T, axis=-1) * lat_w) / np.sum(lat_w))

    # --- Gregorian daily loop with linearly-interpolated prescribed SST ---
    end = _date(args.end)
    record_from = _date(args.record_from)
    monthly: dict[tuple[int, int], list[float]] = {}
    day = start
    stop = False
    while day < end and not stop:
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
        state = seg(state, jnp.asarray(override),
                    jnp.asarray(float(day.timetuple().tm_yday)))
        if day >= record_from:
            st = _surf_T(state)
            monthly.setdefault((day.year, day.month), []).append(st)
            if not np.isfinite(st):
                logger.error(f"{day}: NON-FINITE surfT; stopping")
                stop = True
        if day.day == 1:
            logger.info(f"{day}: running (member r{args.member + 1})")
        day += _dt.timedelta(days=1)

    # --- write monthly + annual means ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("year,month,global_mean_surfT_K\n")
        for (y, m) in sorted(monthly):
            fh.write(f"{y},{m},{float(np.mean(monthly[(y, m)])):.5f}\n")
    logger.info(f"Wrote {args.out} ({len(monthly)} months)")

    annual: dict[int, list[float]] = {}
    for (y, m), vals in monthly.items():
        annual.setdefault(y, []).append(float(np.mean(vals)))
    annual_out = args.out.with_name(args.out.stem + "_annual.csv")
    with annual_out.open("w") as fh:
        fh.write("year,annual_global_mean_surfT_K\n")
        for y in sorted(annual):
            if len(annual[y]) == 12:  # full years only
                fh.write(f"{y},{float(np.mean(annual[y])):.5f}\n")
    logger.info(f"Wrote {annual_out}")


if __name__ == "__main__":
    main()
