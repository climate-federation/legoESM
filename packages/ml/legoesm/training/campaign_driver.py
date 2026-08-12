"""Shared campaign machinery for the WB + AIMIP training drivers (D1).

One implementation behind both entry scripts
(``scripts/run/run_weatherbench_campaign.py`` and ``scripts/run/run_aimip.py``):
suite validation (dispatch-hardened), checkpoint resolution (EMA-aware),
artifact accounting, and the staged train/eval/plot executor. The entry
scripts keep only their arg-parse + campaign-specific argv builders.

Import-light and JAX-free at module import (login-node + CI testable);
anything heavy is imported inside functions.

Design: docs/superpowers/specs/2026-07-19-unified-wb-aimip-training-design.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# Training cores. spectral is the default differentiable core; latlon is the
# production C-grid PE core. cubed_sphere / mpas are RESERVED: the ERA5
# loader side exists (era5_to_cubedsphere_carry) but no segment/training
# path is wired — selecting one is an explicit NotImplementedError, never a
# silent fallback.
TRAINING_CORES = ("spectral", "latlon")
RESERVED_CORES = ("cubed_sphere", "mpas")

# Variant/mode vocabulary across both campaigns.
AIMIP_VARIANTS = ("classical", "column_nn", "sfno_physics", "sfno_full")
WB_MODES = ("physics", "neural_gcm", "sfno")

# Radiation schemes a CLASSICAL (parameterization-swap) training run may
# declare. The swap campaign holds radiation FIXED at rrtmgp (user
# requirement: scheme swaps always run under rrtmgp so convection/turbulence
# comparisons are not confounded by the radiation backend). Smoke/debug runs
# may escape via allow_non_rrtmgp (the T21 smoke contract uses gray for
# ~10x cheaper compile).
CLASSICAL_RADIATION = "rrtmgp"


def parse_bool_flag(value) -> bool:
    """A YAML/CLI flag -> bool, parsing the string spellings correctly.

    ``bool("false")`` is True, so a quoted flag would otherwise mean the
    opposite of what it says. Lives here (import-light, JAX-free) because four
    drivers need the SAME answer for the same key — hand-rolled copies had
    already diverged on a numeric ``2``.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def validate_training_core(core: str) -> str:
    """Dispatch-hardened training-core selection.

    Unknown -> ``ValueError``; reserved (loader exists, training path not
    wired) -> ``NotImplementedError`` stating exactly what is missing.
    """
    if core in TRAINING_CORES:
        return core
    if core in RESERVED_CORES:
        raise NotImplementedError(
            f"training_core={core!r} is reserved: the ERA5 loader side exists "
            "(era5_to_state.era5_to_cubedsphere_carry) but no training "
            "segment path is wired for this grid yet. Use one of "
            f"{list(TRAINING_CORES)}."
        )
    raise ValueError(
        f"Unknown training_core (dycore backend) {core!r}; valid = "
        f"{list(TRAINING_CORES)} (reserved: {list(RESERVED_CORES)})."
    )


def validate_classical_radiation(
    radiation: str,
    *,
    smoke: bool = False,
    allow_non_rrtmgp: bool = False,
) -> str:
    """Enforce the classical-mode radiation pin (design D1).

    A classical parameterization-swap run must use rrtmgp so scheme
    comparisons share the radiation backend. ``smoke`` runs and an explicit
    ``allow_non_rrtmgp`` (debug) escape are exempt — both are logged by the
    caller, never silent.
    """
    if radiation == CLASSICAL_RADIATION or smoke or allow_non_rrtmgp:
        return radiation
    raise ValueError(
        f"Invalid radiation {radiation!r} for classical training: must be "
        f"{CLASSICAL_RADIATION!r} — the parameterization-swap campaign holds "
        "the radiation backend fixed so scheme comparisons are unconfounded. "
        "Set aimip_radiation: rrtmgp, or pass smoke/allow_non_rrtmgp for a "
        "debug run."
    )


def latest_checkpoint(out_dir, *, prefer_ema: bool = False):
    """Newest ``epoch_NNNN.eqx`` in ``out_dir`` (highest epoch), or None.

    With ``prefer_ema=True``, return the ``epoch_NNNN_ema.eqx`` sibling of
    the newest raw checkpoint when it exists (evaluate-the-EMA-weights
    doctrine, D3); the raw file otherwise.
    """
    d = Path(out_dir)
    if not d.is_dir():
        return None
    ckpts = sorted(
        (p for p in d.glob("epoch_*.eqx") if not p.stem.endswith("_ema")),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        return None
    newest = ckpts[-1]
    if prefer_ema:
        ema = newest.with_name(f"{newest.stem}_ema.eqx")
        if ema.exists():
            return str(ema)
    return str(newest)


def nonempty(path) -> bool:
    """A delegate's output counts only if it exists AND is non-empty — a
    zero-byte file is a failed/interrupted write, not a real artifact."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def mpi_barrier(nproc: int) -> None:
    """Collective barrier when running under MPI; no-op single-process.

    Load-bearing between a multi-rank TRAIN stage and rank-0 EVAL: rank 0
    must not evaluate until every rank has flushed checkpoints. Failures
    RAISE (a swallowed barrier lets rank 0 score incomplete checkpoints).
    """
    if nproc <= 1:
        return
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()


def run_staged_campaign(
    *,
    modes,
    stages,
    out_root: str,
    train_fn,
    eval_fn,
    plot_fn,
    mode_out_dir_fn,
    scorecard_path_fn,
    allow_missing_artifacts: bool = False,
    prefer_ema: bool = False,
    log=None,
):
    """Generic staged executor shared by the WB and AIMIP campaigns.

    ``train_fn(mode)`` runs on every rank (collective data-parallel loops);
    ``eval_fn(mode, checkpoint)`` and ``plot_fn(family_scorecards)`` run on
    rank 0 only, after a barrier. A requested stage that produced nothing is
    a FAILURE (SystemExit) unless ``allow_missing_artifacts``.

    Returns the ``{mode: scorecard_path}`` map of families that produced a
    scorecard.
    """
    import logging

    from legoesm.training.data_parallel import mpi_rank_size

    rank, nproc = mpi_rank_size()
    if log is None:
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING
        )
        log = logging.getLogger("campaign")

    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
        log.info(
            "campaign: modes=%s stages=%s ranks=%d out=%s",
            list(modes), list(stages), nproc, out_root,
        )

    if "train" in stages:
        for mode in modes:
            if rank == 0:
                log.info("=== TRAIN %s ===", mode)
            train_fn(mode)

    mpi_barrier(nproc)

    family_scorecards: dict = {}
    missing: list = []
    if rank == 0:
        if "eval" in stages:
            for mode in modes:
                ckpt = latest_checkpoint(
                    mode_out_dir_fn(mode), prefer_ema=prefer_ema
                )
                if ckpt is None:
                    log.warning(
                        "=== EVAL %s SKIPPED: no checkpoint in %s "
                        "(train it first) ===",
                        mode, mode_out_dir_fn(mode),
                    )
                    missing.append(f"eval:{mode} (no checkpoint)")
                    continue
                log.info("=== EVAL %s (ckpt=%s) ===", mode, ckpt)
                eval_fn(mode, ckpt)
                sc = scorecard_path_fn(mode)
                if nonempty(sc):
                    family_scorecards[mode] = sc
                else:
                    log.warning(
                        "=== EVAL %s produced no scorecard at %s ===", mode, sc
                    )
                    missing.append(f"eval:{mode} (no scorecard written)")
        else:
            for mode in modes:
                sc = scorecard_path_fn(mode)
                if nonempty(sc):
                    family_scorecards[mode] = sc

        if "train" in stages and "eval" not in stages:
            for mode in modes:
                if latest_checkpoint(mode_out_dir_fn(mode)) is None:
                    missing.append(f"train:{mode} (no checkpoint written)")

        if "plot" in stages:
            if not family_scorecards:
                log.warning(
                    "=== PLOT SKIPPED: no family scorecards found under %s ===",
                    out_root,
                )
                missing.append("plot (no family scorecards)")
            else:
                out_png = plot_fn(family_scorecards)
                if out_png is not None and not nonempty(out_png):
                    log.warning("=== PLOT produced no figure at %s ===", out_png)
                    missing.append("plot (no figure written)")

        if missing and not allow_missing_artifacts:
            raise SystemExit(
                "campaign: requested artifacts missing: "
                + "; ".join(missing)
                + " (pass --allow-missing-artifacts to downgrade to a warning)."
            )

    return family_scorecards
