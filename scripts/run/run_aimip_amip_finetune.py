#!/usr/bin/env python
"""Stability fine-tune of the AIMIP classical model for free-running AMIP.

The ace2loss classical model was trained on 6-12 h forecasts with ``T_sfc=T_air``
(surface fluxes ~0), so run FREE with prescribed SST it drifts to a ~500 K
radiative-runaway equilibrium. This driver fine-tunes the ~58 scheme knobs + 65
spatial-surface coefficients so the model holds a balanced climate under the
EXACT AIMIP forcing:

* the rollout uses the protocol prescribed SST (``training.aimip_amip_forcing``,
  Zenodo 17065758) so surface fluxes are ACTIVE during training (unlike the
  original training);
* rollouts are multi-day, so the radiative drift manifests and is penalised by
  standard ERA5 state matching at the lead (``spectral_state_vs_carry_loss``);
* a modest global-mean near-surface-T drift term directly discourages the runaway
  (energy-balance proxy; ``--w-drift``).

Gradients flow to the physics params via ``make_aimip_classical_spectral_physics``
(built inside the loss so leaves are traced) and the checkpointed
``spectral_amip_rollout(use_checkpoint=True)``. Writes the fine-tuned params for
the AMIP inference driver to consume.

Usage (proof-of-concept)::

    JAX_ENABLE_X64=1 python scripts/run/run_aimip_amip_finetune.py \\
        --init-ckpt results/aimip_ace2loss_assembled/classical/params.eqx \\
        --out-ckpt results/aimip_amip_finetune/classical_stable.eqx \\
        --epochs 8 --rollout-days 3 --n-samples 6 --lr 1e-4 --w-drift 1.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("aimip-finetune")
_SPD = 86400.0


def _load_yaml(p):
    with Path(p).open() as fh:
        return yaml.safe_load(fh) or {}


def _merged_cfg(suite_path: Path) -> dict:
    suite = _load_yaml(suite_path)
    base = _load_yaml(Path(suite["base"]))
    base.update(suite.get("cfg_overrides", {}) or {})
    base.update(_load_yaml(suite_path.parent / "variant_classical.yaml"))
    base.setdefault("nlev", base["n_levels"])
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-ckpt", type=Path,
                    default=Path("results/aimip_ace2loss_assembled/classical/params.eqx"))
    ap.add_argument("--out-ckpt", type=Path,
                    default=Path("results/aimip_amip_finetune/classical_stable.eqx"))
    ap.add_argument("--suite", type=Path, default=Path("config/aimip/ace2/suite.yaml"))
    ap.add_argument("--forcing-path", type=str, default=None)
    ap.add_argument("--cache-path", type=str,
                    default="results/aimip_forcing/gaussian_forcing.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--rollout-days", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--first-year", type=int, default=1979)
    ap.add_argument("--last-year", type=int, default=2014)  # training period only
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--w-drift", type=float, default=1.0)
    ap.add_argument("--rad-update-interval", type=int, default=36)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    from legoesm.forcing.surface_utils import blend_surface_temperature
    from legoesm import constants
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
        spectral_state_vs_carry_loss,
    )

    cfg = _merged_cfg(args.suite)
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    spec_cfg = NeuralGCMSpectralConfig(
        n_max=int(cfg["n_max"]), n_levels=int(cfg["nlev"]), dt=float(cfg["dt"]),
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=2.5e15, hyperdiff_order=2, time_integrator="ssp_rk3",
            spectral_filter_strength=0.01, spectral_filter_order=8,
        ),
        n_epochs=1, n_train_days=1, start_year=args.first_year,
        loss_config=LossConfig(),
    )
    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)
    sigma_full = jnp.asarray(sigma.sigma_full)
    lat_w = jnp.asarray(grid.weights)
    ncol = len(grid.lat) * len(grid.lon)
    nlev = spec_cfg.n_levels
    dt = spec_cfg.dt
    n_steps = args.rollout_days * int(_SPD / dt)
    pe = spec_cfg.pe_config

    times_ns, sst_m, sic_m, land = regrid_monthly_forcing_to_gaussian(
        args.forcing_path or DEFAULT_AIMIP_FORCING, grid, cache_path=args.cache_path,
    )
    ocean = land < 0.5
    T_ice = float(constants.T_freeze_ocean)
    wb2 = TrainingERA5Config().zarr_store

    def _override_at(date: _dt.date):
        ns = np.datetime64(date).astype("datetime64[ns]").astype(np.int64)
        sst = interp_forcing_at(times_ns, sst_m, ns)
        sic = np.clip(interp_forcing_at(times_ns, sic_m, ns), 0.0, 1.0)
        tsfc = np.asarray(blend_surface_temperature(sst, sic, T_ice))
        return np.where(ocean, tsfc, np.nan).astype(np.float64)

    # --- build training samples: (IC state, target carry, prescribed SST, doy) ---
    years = np.linspace(args.first_year, args.last_year, args.n_samples).astype(int)
    samples = []
    phis0 = None
    for k, yr in enumerate(years):
        d0 = _dt.date(int(yr), 1 + (k % 12), 15)  # spread across seasons
        d1 = d0 + _dt.timedelta(days=args.rollout_days)
        ic_carry = era5_to_spectral_carry(load_era5_ic(wb2, d0.year, d0.month, d0.day), grid, sigma)
        tgt_carry = era5_to_spectral_carry(load_era5_ic(wb2, d1.year, d1.month, d1.day), grid, sigma)
        ic_state = carry_to_spectral_state(ic_carry, grid)
        mid = d0 + _dt.timedelta(days=args.rollout_days // 2)
        samples.append((
            ic_state, tgt_carry,
            jnp.asarray(_override_at(mid)),
            jnp.asarray(float(d0.timetuple().tm_yday)),
        ))
        if phis0 is None:
            phis0 = jnp.asarray(ic_carry.phis)
    logger.info(f"Built {len(samples)} fine-tune samples, rollout={args.rollout_days}d")

    land_mask = land_mask_from_phis(phis0, smooth=True)
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
    sponge = (compute_sponge_factor(sigma.sigma_full, pe.sponge_sigma, pe.sponge_tau, dt)
              if pe.sponge_tau > 0 else None)
    sfilt = (compute_spectral_filter(grid.ls, grid.n_max, order=pe.spectral_filter_order,
                                     cutoff_fraction=pe.spectral_filter_strength)
             if pe.spectral_filter_strength > 0 else None)

    params0 = load_checkpoint(
        AIMIPClassicalParams.from_defaults(spatial_surface=True), args.init_ckpt,
    )

    def _gm_surf_T_state(s):
        T = spectral_pe_to_grid(s, grid, sigma)["T"][..., -1]  # (n_lat, n_lon)
        return jnp.sum(jnp.mean(T, axis=-1) * lat_w) / jnp.sum(lat_w)

    def _gm_surf_T_carry(c):
        T = jnp.asarray(c.T)[..., -1]
        return jnp.sum(jnp.mean(T, axis=-1) * lat_w) / jnp.sum(lat_w)

    def loss_fn(params):
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
        total = 0.0
        for ic_state, tgt_carry, sst_col, doy in samples:
            pred = spectral_amip_rollout(
                ic_state, non_rad_fn, rad_fn, grid, sigma, pe, dt, n_steps,
                sst_col=sst_col, sizing_phys_state=sizing_ps, day_of_year_base=doy,
                rad_update_interval=args.rad_update_interval,
                sponge_factor=sponge, spectral_filter=sfilt, use_checkpoint=True,
            )
            state_loss = spectral_state_vs_carry_loss(
                pred, tgt_carry, grid, sigma, sigma_full, spec_cfg.loss_config,
            )
            drift = (_gm_surf_T_state(pred) - _gm_surf_T_carry(tgt_carry)) ** 2
            total = total + state_loss + args.w_drift * drift
        return total / len(samples)

    # Pre-warm the RRTMGP optics-table cache OUTSIDE filter_jit with CONCRETE
    # params: the NetCDF gas-optics load is module-cached by static file paths,
    # but the first (inside-trace) call would hit the tracer state and crash with
    # TracerArrayConversionError. Mirrors _train_spectral_loop's warm-up.
    try:
        _warm = make_aimip_classical_spectral_physics(
            params0, grid, dt,
            radiation=str(cfg.get("aimip_radiation", "rrtmgp")),
            rad_update_interval_steps=args.rad_update_interval,
            convection_scheme=str(cfg["aimip_convection"]),
            turbulence_scheme=str(cfg["aimip_turbulence"]),
            gwd_scheme=str(cfg["aimip_gwd"]),
            microphysics_scheme=str(cfg["aimip_microphysics"]),
            cloud_scheme=str(cfg.get("aimip_cloud", "xu_randall")),
            land_mask=land_mask, split_rad=True,
        )
        del _warm
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        logger.warning(f"RRTMGP optics warm-up raised {exc!r}; continuing")

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr))
    opt_state = opt.init(eqx.filter(params0, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(params, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(params, eqx.is_inexact_array),
        )
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    params = params0
    for epoch in range(args.epochs):
        params, opt_state, loss = step(params, opt_state)
        logger.info(f"epoch {epoch:3d}: loss={float(loss):.6f}")
        if not np.isfinite(float(loss)):
            logger.error("non-finite loss; stopping")
            break

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(params, args.out_ckpt)
    logger.info(f"Wrote fine-tuned params: {args.out_ckpt}")


if __name__ == "__main__":
    main()
