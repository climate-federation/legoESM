"""Differentiable COUPLED land calibration: train z0 / Vc_max25 / g1 / LCMA through a
real AMIP run WITH atmospheric feedback (``build_segment_fn(...).raw`` +
``eqx.filter_value_and_grad`` + MUON).

This is the coupled counterpart to the OFFLINE single-column calibrator
(``train_multilayer_land_era5.py``).  Offline forcing is ERA5-prescribed, so it cannot
constrain the photosynthesis / stomatal params (Vc_max25 / g1 / LCMA) — they need the
two-way atmosphere<->land feedback (a closed stomata warms+dries the surface, which the
prognostic boundary layer feels, which feeds back on the flux).  Here the down-welling
SW/LW, near-surface T/q, and wind reaching the land are MODEL-computed by the coupled
atmosphere, so all four surface-exchange parameters carry a real coupled gradient.

Mechanism (the gradient path the PR-#650 refactor unblocked):
  * ``ModelDriver.build_training_segment`` returns the NON-JIT ``.raw`` segment
    (AD-safe) + the seeded multilayer-land carry + the day's forcing.
  * The trainable per-PFT params -> a per-column ``LandSurfaceParams`` (only z0 /
    Vc_max25 / g1 / LCMA overridden; every other field stays at the baked CLM
    multilayer default) assigned to ``pipe.land_ml_params`` INSIDE the loss, so the
    segment reads a TRACED value at call time and gradients flow back.
  * The tile runs with ``bulk_scheme="most"`` (z0 active) + Farquhar stomata
    (``stomata.enabled`` + ``carbon="differland"`` + a PRESCRIBED carbon state ->
    fixed LAI; Vc_max25 / g1 / LCMA active) so all four params reach the surface flux.
  * Loss = land-area-weighted MSE of the model land-surface T
    (``carry.land_ml.T_soil[:, 0]``) vs the ERA5 skin-temperature climatology mapped
    to the model columns.

The default run is SHORT (a proof that the coupled gradient is correct + the loss
descends + the params move to physical values); ``--steps`` / ``--days`` scale it up
to a converged calibration on GPU.  Writes the recommended tuned per-PFT params to
``results/land_tuned_coupled.json`` — it does NOT mutate production defaults.

Run (CPU proof):
  PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python scripts/run/train_coupled_land_era5.py \
      --land-mask-file /tmp/era5_lsm.nc --resolution 12 --nlev 8 --steps 24 --iters 8
"""
from __future__ import annotations

import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.stomata import StomataConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.clm_surface_map import (
    load_clm_surface, download_clm_surfdata, _nearest_regrid,
)
from legoesm.ml.training import TrainingConfig, create_optimizer

# Reuse the offline calibrator's per-PFT bounds + sigmoid constrain + CLM-default raw
# inits (ONE shared bounded-parameter definition; no re-derivation).
import scripts.run.train_multilayer_land_era5 as ML

# The four coupled-trainable per-PFT (17) groups.  Everything else (albedo, emissivity,
# soil thermal, plant water-stress wp/fc, ...) stays at the baked CLM multilayer
# default — those are constrainable OFFLINE; only these four need the coupled feedback.
_TRAIN_KEYS = ("pft_z0", "pft_vcmax", "pft_g1", "pft_lcma")


def constrain4(p: dict) -> dict:
    """sigmoid-constrain the four trainable raw param groups into physical bounds."""
    return {k: ML.BOUNDS_EXT[k][0]
            + (ML.BOUNDS_EXT[k][1] - ML.BOUNDS_EXT[k][0]) * jax.nn.sigmoid(v)
            for k, v in p.items()}


def init_params4() -> dict:
    """Raw (unconstrained) inits at the CLM5 table defaults (the offline calibrator's
    init, restricted to the four coupled-trainable groups)."""
    full = ML.init_ext_params()
    return {k: full[k] for k in _TRAIN_KEYS}


def build_land_params(p: dict, base_lp, pft):
    """Per-column LandSurfaceParams: z0/Vc_max25/g1/LCMA from the trainable per-PFT
    params (PFT-weighted), every other field from the baked CLM multilayer default."""
    cp = constrain4(p)
    pw = lambda k: pft @ cp[k]                  # (ncol,17) @ (17,) -> (ncol,)
    return base_lp._replace(
        z0=pw("pft_z0"), Vc_max25=pw("pft_vcmax"),
        g1=pw("pft_g1"), LCMA=pw("pft_lcma"))


def load_era5_skt_columns(lat_deg, lon_deg, npz="/tmp/era5_diurnal.npz"):
    """ERA5 annual+diurnal-mean skin temperature [K] nearest-regridded to the model
    columns (reuses the CLM-map nearest-neighbour regrid)."""
    D = np.load(npz)
    skt = np.asarray(D["skin_temperature"]).mean((0, 1))   # (nlat, nlon)
    return np.asarray(_nearest_regrid(D["lat"], D["lon"], skt, lat_deg, lon_deg))


def setup_driver(args):
    """Build + setup the coupled AMIP model with the multilayer land tile switched to
    the TRAINABLE config (MOST z0 + Farquhar stomata + prescribed carbon)."""
    cfg = ExperimentConfig(
        grid=GridConfig(resolution=args.resolution, nlev=args.nlev),
        dycore=DycoreConfig(dt=args.dt),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        land_mask_path=args.land_mask_file,
        use_multilayer_land=True,
        multilayer_n_layers=args.n_layers, multilayer_soil_depth=args.soil_depth,
        # fp64 is REQUIRED for the backward: stiff clay van-Genuchten soils
        # (n_vg~1.09 -> m~0.083) make Se**(-1/m) and its reverse-mode derivative
        # overflow float32 (finite forward, NaN gradient) — exactly why the offline
        # calibrator runs JAX_ENABLE_X64=1.  The coupled gradient is finite in fp64.
        precision="fp64",
    )
    driver = ModelDriver(cfg, output_dir=args.output_dir)
    driver.setup()
    pipe = driver.physics
    if pipe.land_ml_cfg is None:
        raise SystemExit(
            "multilayer land tile is not active — pass --land-mask-file (the land "
            "mask is required to seed the soil columns).")
    # Switch the tile to the trainable surface-exchange config: MOST makes z0 active;
    # Farquhar stomata (enabled + differland carbon + a prescribed carbon state) make
    # Vc_max25/g1/LCMA active.  Thermal/hydraulics stay the per-cell CLM maps.
    pipe.land_ml_cfg = pipe.land_ml_cfg._replace(
        # PINNED to the scheme whose parameters this trainer fits.  The library
        # default is now the two-leaf canopy, which reads CANOPY properties
        # (rz0m x canopy height, band albedos) and NOT the per-PFT ``z0`` /
        # albedo / emissivity on LandSurfaceParams that this trainer optimises —
        # inheriting it would leave every trained leaf with zero gradient, which
        # the no-inert-parameters rule treats as a defect, not a nuisance.
        # Re-targeting this trainer at canopy parameters is a deliberate re-fit.
        surface_scheme=SimpleSEBConfig(),
        bulk_scheme="most",
        stomata=StomataConfig(enabled=True),
        carbon=CarbonConfig(scheme="differland"),
    )
    ncol = int(driver.grid.lat.size)
    pipe.land_ml_carbon = init_carbon_state((ncol,), pipe.land_ml_cfg.carbon)
    return driver, pipe


def main():
    # fp64 throughout (the stiff-clay VG backward overflows float32 — see setup_driver).
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--land-mask-file", default="/tmp/era5_lsm.nc")
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--nlev", type=int, default=8)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--soil-depth", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=24,
                    help="coupled segment length (radiation+land step every step). "
                         "Scale up (with --days on GPU) for a converged calibration.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--optimizer", default="muon",
                    choices=["muon", "muon_partitioned", "adamw", "adam"])
    ap.add_argument("--era5-npz", default="/tmp/era5_diurnal.npz")
    ap.add_argument("--output-dir", default="/tmp/coupled_land_train")
    ap.add_argument("--out", default="results/land_tuned_coupled.json")
    args = ap.parse_args()

    driver, pipe = setup_driver(args)
    ad = pipe.adapter

    # Per-column PFT fractions (map per-PFT trainable -> per-column) + ERA5 target.
    lat_rad = np.asarray(ad.flatten_2d(driver.grid.lat)).reshape(-1)
    lon_rad = np.asarray(ad.flatten_2d(driver.grid.lon)).reshape(-1)
    lat_deg, lon_deg = np.degrees(lat_rad), np.degrees(lon_rad)
    cmap = load_clm_surface(download_clm_surfdata(), lat_deg, lon_deg)
    pft = jnp.asarray(cmap["pft_fractions"])                     # (ncol, 17)
    skt_target = jnp.asarray(load_era5_skt_columns(lat_deg, lon_deg, args.era5_npz),
                             dtype=jnp.float32)
    # Weight: land fraction (only score land) x cos(lat) area weight.
    f_land = ad.flatten_2d(pipe.f_land).astype(jnp.float32)
    weight = f_land * jnp.cos(jnp.asarray(lat_rad, dtype=jnp.float32))
    base_lp = pipe.land_ml_params                               # baked CLM defaults

    run_seg, carry0, forcing = driver.build_training_segment(args.steps)

    def loss_fn(p):
        # Assign the TRACED params to the pipeline attribute the tile reads at call
        # time -> gradients flow back through the coupled segment to p.
        pipe.land_ml_params = build_land_params(p, base_lp, pft)
        final = run_seg(carry0, args.steps, forcing)
        pred = final.land_ml.T_soil[:, 0]                       # model land-surface T
        num = jnp.sum(weight * (pred - skt_target) ** 2)
        return num / jnp.sum(weight)

    vg = eqx.filter_value_and_grad(loss_fn)
    opt = create_optimizer(TrainingConfig(
        lr=args.lr, warmup_steps=max(1, args.iters // 10),
        total_steps=args.iters, optimizer=args.optimizer))
    p = init_params4()
    state = opt.init(p)

    print(f"# coupled land calibration: C{args.resolution}/L{args.nlev}, "
          f"{args.steps} steps, {pft.shape[0]} columns, optimizer={args.optimizer}",
          flush=True)
    for it in range(args.iters):
        loss, g = vg(p)
        # Drop the traced LandSurfaceParams the loss left on the pipeline attribute
        # (a deliberate side effect so the segment reads a traced value): restore a
        # CONCRETE value before the next trace, else the escaped tracer triggers
        # jax UnexpectedTracerError on the following iteration.
        pipe.land_ml_params = base_lp
        # First-iteration gradient sanity: every trainable group must carry a finite,
        # non-zero coupled gradient (else that param is not reaching the surface flux).
        if it == 0:
            print(f"#   loss0 {float(loss):.5f} finite={bool(np.isfinite(float(loss)))}",
                  flush=True)
            for k in _TRAIN_KEYS:
                gk = np.asarray(g[k])
                print(f"#   grad[{k}] max|.|={np.nanmax(np.abs(gk)):.3e} "
                      f"finite_frac={np.mean(np.isfinite(gk)):.2f} "
                      f"nonzero={bool(np.any(gk != 0))}", flush=True)
            for k in _TRAIN_KEYS:
                assert np.all(np.isfinite(np.asarray(g[k]))), f"{k}: non-finite gradient"
        updates, state = opt.update(g, state, p)   # MUON / weight-decay need params
        p = eqx.apply_updates(p, updates)
        print(f"# it {it:3d}  loss {float(loss):.4f}  "
              f"land-T-RMSE {float(jnp.sqrt(loss)):.3f} K", flush=True)

    tuned = {k: np.asarray(v).tolist() for k, v in constrain4(p).items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"# recommended coupled-tuned params -> {args.out} "
          f"(does NOT mutate production defaults)", flush=True)


if __name__ == "__main__":
    main()
