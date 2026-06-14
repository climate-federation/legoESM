#!/usr/bin/env python
"""Architecture-size sweep for column_nn and sfno_physics variants.

Reuses the production multi-step + CRPS training config
(``aimip_era5_multistep.yaml``) but parameterises the network sizes.
For each (variant, size) combination the sweep runs the full
``run_aimip._train_variant`` + ``_evaluate_variant`` flow and appends
the result to a unified sweep scorecard:

    results/aimip_arch_sweep_001/aimip_arch_sweep.json

Each entry's ``variant`` key is suffixed with the arch tag (e.g.
``column_nn_S``, ``sfno_physics_L``) so downstream plotting can
distinguish them from the production scorecard.

Usage::

    JAX_ENABLE_X64=1 python scripts/run_aimip_arch_sweep.py \\
        --base config/aimip/aimip_era5_multistep.yaml

The sweep grid is intentionally compact (4 sizes per variant -> 8
trainings) so it fits in roughly the same wall-clock as the
3-variant production scorecard at this resolution.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import jax
import yaml


logger = logging.getLogger("aimip-arch-sweep")


# Arch grid -- (size_tag, override_dict) per variant.  ``override_dict``
# values are merged on top of the base + variant overlay.
ARCH_GRID = {
    "column_nn": [
        ("S",  dict(nn_hidden_dim=128, nn_n_layers=2)),
        ("M",  dict(nn_hidden_dim=256, nn_n_layers=4)),
        ("L",  dict(nn_hidden_dim=512, nn_n_layers=4)),
        ("XL", dict(nn_hidden_dim=512, nn_n_layers=6)),
    ],
    "sfno_physics": [
        ("S",  dict(sfno_embed_dim=64,  sfno_n_blocks=2, sfno_mlp_expansion=4)),
        ("M",  dict(sfno_embed_dim=128, sfno_n_blocks=4, sfno_mlp_expansion=4)),
        ("L",  dict(sfno_embed_dim=256, sfno_n_blocks=4, sfno_mlp_expansion=4)),
        ("XL", dict(sfno_embed_dim=256, sfno_n_blocks=6, sfno_mlp_expansion=4)),
    ],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(overlay)
    return out


def _run_one(
    variant: str,
    size_tag: str,
    overrides: dict[str, Any],
    base_cfg: dict[str, Any],
    variant_overlay: dict[str, Any],
    output_dir: Path,
    cache_dir: str,
) -> dict[str, Any]:
    """Train + evaluate one (variant, arch_size) combination."""
    from scripts.run.run_aimip import _train_variant, _evaluate_variant

    cfg = _merge(base_cfg, variant_overlay)
    cfg = _merge(cfg, overrides)
    cfg["aimip_variant"] = variant
    arch_tag = f"{variant}_{size_tag}"
    # Per-arch checkpoint directory so models don't overwrite each other.
    cfg["output_dir"] = str(output_dir)
    cfg["log_every"] = 1

    logger.info("=" * 70)
    logger.info(f"Sweep entry: {arch_tag}  overrides={overrides}")
    logger.info("=" * 70)

    t0 = time.time()
    model, loss_history = _train_variant(variant, cfg, cache_dir)
    train_elapsed = time.time() - t0

    eval_test = _evaluate_variant(variant, model, cfg, cache_dir, period="test")
    eval_train = _evaluate_variant(variant, model, cfg, cache_dir, period="train")

    from legoesm.ml.training import save_checkpoint
    ckpt = output_dir / arch_tag / "params.eqx"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, ckpt)

    return {
        "variant_base": variant,
        "size": size_tag,
        "overrides": overrides,
        "train_loss_history": [float(x) for x in loss_history],
        "train_seconds": train_elapsed,
        "eval_metrics": eval_test,
        "eval_metrics_train_period": eval_train,
        "checkpoint": str(ckpt),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=Path("config/aimip/aimip_era5_multistep.yaml"),
        help="Base config (multi-step + CRPS).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/aimip_arch_sweep_001"),
    )
    parser.add_argument(
        "--variants", type=str, default="",
        help="Comma-separated subset of {column_nn,sfno_physics}.",
    )
    parser.add_argument(
        "--sizes", type=str, default="",
        help="Comma-separated subset of size tags (S,M,L,XL).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for spectral training.")

    base_cfg = _load_yaml(args.base)
    cache_dir = base_cfg.get("cache_dir", ".cache/aimip_era5")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chosen_variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants else list(ARCH_GRID)
    )
    chosen_sizes = (
        [s.strip() for s in args.sizes.split(",") if s.strip()]
        if args.sizes else None
    )

    results: dict[str, Any] = {
        "base_config": str(args.base),
        "sweep_grid": {
            k: [{"size": s, "overrides": o} for (s, o) in v]
            for k, v in ARCH_GRID.items()
        },
        "entries": {},
    }

    scorecard_path = args.output_dir / "aimip_arch_sweep.json"
    if scorecard_path.exists():
        try:
            with scorecard_path.open() as fh:
                existing = json.load(fh)
            results["entries"] = dict(existing.get("entries", {}))
            logger.info(f"Merging into existing scorecard ({len(results['entries'])} entries)")
        except Exception as exc:
            logger.warning(f"Existing scorecard unreadable, starting fresh ({exc!r})")

    for variant in chosen_variants:
        if variant not in ARCH_GRID:
            logger.warning(f"Variant {variant!r} not in sweep grid; skipping")
            continue
        variant_overlay = _load_yaml(
            args.base.parent / f"variant_{variant}.yaml"
        )
        for size_tag, overrides in ARCH_GRID[variant]:
            arch_tag = f"{variant}_{size_tag}"
            if chosen_sizes and size_tag not in chosen_sizes:
                continue
            if arch_tag in results["entries"]:
                logger.info(f"  {arch_tag}: skip (already present in scorecard)")
                continue
            try:
                entry = _run_one(
                    variant, size_tag, overrides,
                    base_cfg, variant_overlay,
                    args.output_dir, cache_dir,
                )
                results["entries"][arch_tag] = entry
                # Checkpoint the running scorecard after every entry so a
                # job-kill doesn't wipe completed sweep entries.
                with scorecard_path.open("w") as fh:
                    json.dump(results, fh, indent=2)
                logger.info(f"  {arch_tag} done -- scorecard now {len(results['entries'])} entries")
            except Exception as exc:
                logger.error(f"  {arch_tag} FAILED: {exc!r}")
                results["entries"][arch_tag] = {
                    "variant_base": variant, "size": size_tag,
                    "overrides": overrides,
                    "error": repr(exc),
                }
                with scorecard_path.open("w") as fh:
                    json.dump(results, fh, indent=2)

    logger.info(f"Wrote sweep scorecard: {scorecard_path}")
    print("\n" + "=" * 78)
    print("Arch sweep summary (sorted by held-out T_sfc RMSE)")
    print("=" * 78)
    rows = []
    for tag, entry in results["entries"].items():
        em = (entry.get("eval_metrics") or {})
        rmse = em.get("rmse", {}).get("T_sfc", {}).get("mean", float("nan"))
        bias = em.get("bias", {}).get("T_sfc", {}).get("mean", float("nan"))
        rows.append((tag, rmse, bias))
    rows.sort(key=lambda r: (float("inf") if r[1] != r[1] else r[1]))
    print(f"{'arch_tag':20s}  {'T_sfc RMSE [K]':>16s}  {'T_sfc bias [K]':>16s}")
    for tag, rmse, bias in rows:
        print(f"{tag:20s}  {rmse:>16.3f}  {bias:>16.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
