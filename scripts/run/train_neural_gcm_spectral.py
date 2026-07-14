#!/usr/bin/env python
"""Train a NeuralGCM: SFNO physics + spectral PE dynamical core.

Couples a learned SFNO physics parameterization to the differentiable
spectral primitive equation dynamical core and trains on ERA5 daily
snapshots via backpropagation through the full model.

Usage::

    JAX_ENABLE_X64=1 python scripts/train_neural_gcm_spectral.py

    # Custom config
    JAX_ENABLE_X64=1 python scripts/train_neural_gcm_spectral.py \\
        --n-max 42 --n-levels 10 --dt 1800 \\
        --sfno-embed 128 --sfno-blocks 4 \\
        --epochs 50 --lr 3e-4 \\
        --train-days 365 --year 2015 \\
        --cache-dir data/era5_cache

Requires JAX_ENABLE_X64=1 for spectral transforms (complex128).
"""

import argparse
import logging
import sys

import jax


def main():
    parser = argparse.ArgumentParser(
        description="Train NeuralGCM (SFNO + spectral PE dycore)",
    )

    # Grid
    parser.add_argument("--n-max", type=int, default=42,
                        help="Spectral truncation (T42~2.8deg, T48~2.5deg)")
    parser.add_argument("--n-levels", type=int, default=10,
                        help="Number of vertical levels")
    parser.add_argument("--sigma-top", type=float, default=0.05,
                        help="Model top sigma level")

    # Dynamics
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Dycore timestep [s] (600 for T42 explicit RK3)")
    parser.add_argument("--hyperdiff", type=float, default=2.5e15,
                        help="Hyperdiffusion coefficient")

    # SFNO
    parser.add_argument("--sfno-embed", type=int, default=128,
                        help="SFNO embedding dimension")
    parser.add_argument("--sfno-blocks", type=int, default=4,
                        help="Number of SFNO processor blocks")
    parser.add_argument("--sfno-mlp-expansion", type=int, default=4,
                        help="MLP expansion factor in SFNO blocks")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")

    # Data
    parser.add_argument("--train-days", type=int, default=365,
                        help="Number of training days")
    parser.add_argument("--year", type=int, default=2015,
                        help="ERA5 year to train on")
    parser.add_argument("--cache-dir", type=str, default="data/era5_cache",
                        help="ERA5 local cache directory")

    # Output
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/neural_gcm_spectral",
                        help="Checkpoint directory")
    parser.add_argument("--log-every", type=int, default=5,
                        help="Log every N epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    args = parser.parse_args()

    # Validate x64
    if not jax.config.x64_enabled:
        print(
            "ERROR: JAX x64 mode required for spectral transforms.\n"
            "Run with: JAX_ENABLE_X64=1 python scripts/train_neural_gcm_spectral.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.neural_gcm_spectral import (
        NeuralGCMSpectralConfig,
        train_neural_gcm_spectral,
    )
    from legoesm.training.losses import LossConfig

    config = NeuralGCMSpectralConfig(
        n_max=args.n_max,
        n_levels=args.n_levels,
        sigma_top=args.sigma_top,
        dt=args.dt,
        pe_config=SpectralPEConfig(
            hyperdiff_coeff=args.hyperdiff,
            hyperdiff_order=2,
            time_integrator="ssp_rk3",
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
        ),
        sfno_embed_dim=args.sfno_embed,
        sfno_n_blocks=args.sfno_blocks,
        sfno_mlp_expansion=args.sfno_mlp_expansion,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        n_train_days=args.train_days,
        start_year=args.year,
        loss_config=LossConfig(),
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
    )

    logging.getLogger(__name__).info(
        f"NeuralGCM training: T{config.n_max} L{config.n_levels}, "
        f"SFNO({config.sfno_embed_dim}d, {config.sfno_n_blocks} blocks), "
        f"dt={config.dt}s, {config.n_train_days} days, "
        f"{config.n_epochs} epochs"
    )

    sfno, loss_history = train_neural_gcm_spectral(
        config=config,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    logging.getLogger(__name__).info(
        f"Training complete. Final loss: {loss_history[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
