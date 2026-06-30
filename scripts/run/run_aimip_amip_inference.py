#!/usr/bin/env python
"""Prescribed-SST AMIP inference for the trained AIMIP classical model.

Runs the spectral-PE classical model (the AMIP-capable variant — its surface
flux scheme anchors near-surface air T to a prescribed SST; the column_nn / sfno
variants REPLACE physics and have no SST hook, so they are free-running and not
run here) over the AIMIP protocol period with monthly ERA5 SST prescribed, then
writes the annual global-mean near-surface (2 m proxy = lowest sigma level) air
temperature so it can be overlaid on the AIMIP Phase-1 fleet figure (paper
arXiv:2605.06944 Fig 3) as an anomaly from the model's OWN 1979-2014 mean.

Mechanism (see ``training.neural_gcm_spectral.spectral_amip_rollout``):
the prescribed monthly SST is injected as a TRACED arg into BOTH the
surface-flux/turbulence scheme (``phys_state.surface_T_sfc_override``) and
radiation (``forcing['T_sfc']`` + seasonal ``day_of_year``), so a single JIT'd
month-long segment is reused across the whole multi-decade run (SegmentForcing
doctrine — no retrace when the monthly SST/calendar changes). Over land the
override is NaN -> the model keeps its own surface temperature.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_aimip_amip_inference.py \\
        --ckpt results/aimip_ace2loss_assembled/classical/params.eqx \\
        --suite config/aimip/ace2/suite.yaml \\
        --start-year 1979 --end-year 2024 \\
        --out results/aimip_fleet_paper/legoesm_classical_amip.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("aimip-amip")

_SECONDS_PER_DAY = 86400.0


def _load_yaml(path: Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _merged_cfg(suite_path: Path) -> dict:
    suite = _load_yaml(suite_path)
    base = _load_yaml(Path(suite["base"]))
    base.update(suite.get("cfg_overrides", {}) or {})
    # variant_classical overlay (edmf/louis/mcfarlane/sundqvist/xu_randall).
    overlay = _load_yaml(suite_path.parent / "variant_classical.yaml")
    base.update(overlay)
    # CLAUDE.md naming debt: spectral config reads "nlev"; AIMIP yaml uses
    # "n_levels".
    base.setdefault("nlev", base["n_levels"])
    return base


def _build_spec_cfg(cfg: dict):
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig

    return NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]),
        n_levels=int(cfg["nlev"]),
        dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15,
            hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
        ),
        n_epochs=1,
        lr=float(cfg.get("aimip_lr", 3.0e-4)),
        weight_decay=float(cfg.get("aimip_weight_decay", 1.0e-5)),
        n_train_days=1,
        start_year=int(cfg.get("train_year", 1979)),
        loss_config=LossConfig(),
    )


def _time_index(ds, year: int, month: int, day: int = 15, max_days: int = 20):
    """Nearest ERA5 time index to (year, month, day), or raise if the store does
    not actually cover that month (else nearest-selection silently reuses the
    last available timestamp for out-of-range years — codex review #1)."""
    import pandas as pd
    times = pd.DatetimeIndex(ds.time.values)
    target = pd.Timestamp(year=year, month=month, day=day)
    if target < times[0] or target > times[-1]:
        raise ValueError(
            f"{year}-{month:02d} outside ERA5 store range "
            f"[{times[0].date()} .. {times[-1].date()}]"
        )
    idx = int(np.argmin(np.abs(times - target)))
    if abs((times[idx] - target).days) > max_days:
        raise ValueError(
            f"no ERA5 timestamp within {max_days}d of {year}-{month:02d}"
        )
    return idx


def _month_day_of_year(month: int) -> float:
    # Mid-month day-of-year for a 360-day (30-day-month) calendar — matches the
    # fixed 30-day segment length so seasonal insolation tracks the segment.
    return (month - 1) * 30.0 + 15.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("results/aimip_ace2loss_assembled/classical/params.eqx"))
    ap.add_argument("--suite", type=Path, default=Path("config/aimip/ace2/suite.yaml"))
    ap.add_argument("--start-year", type=int, default=1979)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--days-per-month", type=int, default=30)
    ap.add_argument("--rad-update-interval", type=int, default=36)
    ap.add_argument("--out", type=Path,
                    default=Path("results/aimip_fleet_paper/legoesm_classical_amip.csv"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.aimip_spatial import land_mask_from_phis
    from legoesm.training.era5_to_state import (
        TrainingERA5Config,
        era5_to_spectral_carry,
        load_era5_slice,
        open_era5_zarr,
        regrid_2d_to_gaussian,
    )
    from legoesm.atmosphere.dynamics.spectral_pe import (
        compute_spectral_filter,
        compute_sponge_factor,
        spectral_pe_to_grid,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state,
        spectral_amip_rollout,
    )

    cfg = _merged_cfg(args.suite)
    spec_cfg = _build_spec_cfg(cfg)
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    lat_w = np.asarray(grid.weights)
    n_lat, n_lon = len(grid.lat), len(grid.lon)
    ncol = n_lat * n_lon
    nlev = spec_cfg.n_levels
    dt = spec_cfg.dt
    n_steps_seg = args.days_per_month * int(_SECONDS_PER_DAY / dt)

    era5_cfg = TrainingERA5Config(zarr_store=cfg.get("zarr_store", TrainingERA5Config.zarr_store))
    ds = open_era5_zarr(era5_cfg.zarr_store)

    # --- IC: start-year January ---
    ic_idx = _time_index(ds, args.start_year, 1, 1)
    ic_slice = load_era5_slice(era5_cfg, ic_idx)
    ic_carry = era5_to_spectral_carry(ic_slice, grid, sigma)
    state = carry_to_spectral_state(ic_carry, grid)

    # Land mask (1 = land) from surface geopotential, for the spatial surface
    # params AND to keep prescribed SST over OCEAN only.
    land_mask = land_mask_from_phis(jnp.asarray(ic_carry.phis), smooth=True)
    ocean = np.asarray(land_mask).reshape(-1) < 0.5

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
        land_mask=land_mask,
        split_rad=True,
    )

    # Sizing PhysicsState (per-scheme carry shapes; values zero, SST injected
    # inside the rollout). Scheme strings only -> no trained params needed.
    sizing_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme=str(cfg["aimip_convection"])),
        turbulence=TurbulenceConfig(scheme=str(cfg["aimip_turbulence"])),
        microphysics=MicrophysicsConfig(scheme=str(cfg["aimip_microphysics"])),
        gravity_wave_drag=GravityWaveDragConfig(scheme=str(cfg["aimip_gwd"])),
    )
    sizing_ps = init_physics_state(ncol, nlev, sizing_cfg)

    pe = spec_cfg.pe_config
    sponge = (
        compute_sponge_factor(sigma.sigma_full, pe.sponge_sigma, pe.sponge_tau, dt)
        if pe.sponge_tau > 0 else None
    )
    sfilt = (
        compute_spectral_filter(
            grid.ls, grid.n_max, order=pe.spectral_filter_order,
            cutoff_fraction=pe.spectral_filter_strength,
        ) if pe.spectral_filter_strength > 0 else None
    )

    seg = jax.jit(
        lambda st, sst_col, doy, off: spectral_amip_rollout(
            st, non_rad_fn, rad_fn, grid, sigma, pe, dt, n_steps_seg,
            sst_col=sst_col, sizing_phys_state=sizing_ps,
            day_of_year_base=doy, seconds_offset=off,
            rad_update_interval=args.rad_update_interval,
            sponge_factor=sponge, spectral_filter=sfilt,
        )
    )

    def _surf_T(s):
        f = spectral_pe_to_grid(s, grid, sigma)
        T = np.asarray(f["T"])[..., -1]  # lowest sigma level (2 m proxy)
        zonal = np.mean(T, axis=-1)
        return float(np.sum(zonal * lat_w) / np.sum(lat_w))

    rows = []  # (year, month, global_mean_surfT)
    cum_seconds = 0.0
    stop = False
    for year in range(args.start_year, args.end_year + 1):
        if stop:
            break
        for month in range(1, 13):
            try:
                idx = _time_index(ds, year, month)
                sl = load_era5_slice(era5_cfg, idx)
                sst_g = np.asarray(
                    regrid_2d_to_gaussian(sl.sst, sl.lat, sl.lon, grid)
                ).reshape(-1)
                # Guard against a zero-filled SST (load_era5_slice returns zeros
                # when skin_temperature is absent — codex review #2): a 0 K
                # "ocean" would be silently prescribed. Require finite, physical
                # ocean SST.
                ocean_sst = sst_g[ocean]
                # Per-cell + mean validation (codex review): a few 0 K cells
                # from an absent skin_temperature would pass a mean-only check.
                # >150 K per cell catches 0 K-fill without rejecting the coldest
                # real ocean / sea-ice / coastline-mismatch cells (~180-271 K);
                # the >270 K mean keeps the global ocean physically plausible.
                if (
                    (not np.all(np.isfinite(ocean_sst)))
                    or (not np.all(ocean_sst > 150.0))
                    or float(ocean_sst.mean()) < 270.0
                ):
                    raise ValueError(
                        f"implausible ERA5 ocean SST (mean "
                        f"{float(np.nanmean(ocean_sst)):.1f} K, min "
                        f"{float(np.nanmin(ocean_sst)):.1f} K) — missing "
                        f"skin_temperature?"
                    )
            except Exception as exc:  # out-of-range / missing SST -> stop cleanly
                logger.warning(
                    f"{year}-{month:02d}: ERA5 SST unavailable ({exc!r}); stopping"
                )
                stop = True
                break
            # Prescribe over ocean only; NaN over land -> model keeps own T.
            override = np.where(ocean, sst_g, np.nan).astype(np.float64)
            state = seg(
                state, jnp.asarray(override),
                jnp.asarray(_month_day_of_year(month)),
                jnp.asarray(cum_seconds),
            )
            cum_seconds += args.days_per_month * _SECONDS_PER_DAY
            t_sfc = _surf_T(state)
            rows.append((year, month, t_sfc))
            if month == 12:
                logger.info(f"{year}: Dec global-mean surfT = {t_sfc:.3f} K")
            if not np.isfinite(t_sfc):
                logger.error(f"{year}-{month:02d}: NON-FINITE surfT; stopping")
                stop = True
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("year,month,global_mean_surfT_K\n")
        for y, m, t in rows:
            fh.write(f"{y},{m},{t:.5f}\n")
    logger.info(f"Wrote {args.out} ({len(rows)} months)")

    # Annual means -> companion CSV (what the fleet overlay consumes).
    annual = {}
    for y, m, t in rows:
        annual.setdefault(y, []).append(t)
    annual_out = args.out.with_name(args.out.stem + "_annual.csv")
    with annual_out.open("w") as fh:
        fh.write("year,annual_global_mean_surfT_K\n")
        for y in sorted(annual):
            if len(annual[y]) == 12:  # full years only
                fh.write(f"{y},{float(np.mean(annual[y])):.5f}\n")
    logger.info(f"Wrote {annual_out}")


if __name__ == "__main__":
    main()
