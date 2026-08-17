"""Learned physics + spectral PE dycore: SFNO or column MLP modes.

Couples a learned physics parameterization to the spectral primitive
equation dynamical core, with end-to-end gradient flow.  Two modes:

- **SFNO** (Kochkov et al. 2024): global SFNO predicts tendencies for
  all variables (u, v, T, q, lnps) on the Gaussian grid.
- **Column MLP** (Rasp et al. 2018): per-column MLP predicts T
  tendencies from local column state (T, u, v, q, p_s).  No spatial
  coupling — physically motivated for subgrid physics.

Both modes share the same pipeline::

    ERA5 (daily snapshots)
      -> carry_to_spectral_state   [SH analysis]
      -> spectral_rollout           [lax.scan: dycore + learned physics]
      -> spectral_state_vs_carry_loss  [SH synthesis + MSE vs ERA5 target]
      -> eqx.filter_value_and_grad -> optax update

Relationship to other modules
-----------------------------
- ``sfno_dycore_coupling.py`` / ``atmosphere.physics.neural_physics``: couple
  learned physics to *grid-space* dycores via ``build_segment_fn``.
- ``sfno_pe.py``: SFNO replaces the entire dycore.
- This module: learned physics *augments* the spectral PE dycore via
  ``spectral_pe_tendencies(..., physics_tendency=learned_output)``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    SpectralPEConfig,
    spectral_pe_tendencies,
    spectral_pe_to_grid,
    compute_spectral_filter,
    compute_sponge_factor,
    apply_sponge_filter,
    apply_spectral_filter_to_state,
    apply_filter_to_tracers,
    anchor_lnps_to_mass,
    global_dry_mass,
)
from legoesm import constants
from legoesm.core.conservation import conservative_positive_clip
from legoesm.core.field import Field
from legoesm.core.tracers import (
    make_full_moisture_registry,
    make_moisture_registry,
)
from legoesm.grids.gaussian import (
    GaussianGrid,
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    vordiv_from_uv_exact_3d,
)
from legoesm.grids.vertical import SigmaCoordinate, create_sigma_coordinate
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec, pack_pe_state, unpack_pe_output
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.training.era5_to_state import (
    TrainingERA5Config,
    era5_to_spectral_carry,
)
from legoesm.training.curriculum import build_curriculum_epoch_plan
from legoesm.training.ema import ema_update, init_ema
from legoesm.training.losses import LossConfig, level_weights
from legoesm.training.data_parallel import mpi_abort_on_uncaught

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class NeuralGCMSpectralConfig(NamedTuple):
    """Configuration for NeuralGCM spectral training."""
    # Grid
    n_max: int = 42              # Spectral truncation (T42 ~ 2.8 deg)
    n_levels: int = 10           # Vertical levels
    sigma_top: float = 0.05      # Model top sigma (~ 50 hPa)

    # Dynamics
    dt: float = 600.0            # Dycore timestep [s] (CFL-safe for T42 explicit RK3)
    pe_config: SpectralPEConfig = SpectralPEConfig(
        hyperdiff_coeff=2.5e15,
        hyperdiff_order=2,
        time_integrator="ssp_rk3",
        spectral_filter_strength=0.01,
        spectral_filter_order=8,
    )

    # SFNO architecture
    sfno_embed_dim: int = 128
    sfno_n_blocks: int = 4
    sfno_mlp_expansion: int = 4
    # MC-Dropout rate inside every block's MLP (U-Cast, arXiv:2604.09041:
    # 10 %, flat across 5-15 %).  0.0 keeps the network deterministic and the
    # forward pass bit-identical to every checkpoint trained before this knob.
    sfno_dropout: float = 0.0
    # U-Cast STAGE 2: the last N epochs train on the ensemble almost-fair CRPS
    # instead of the deterministic loss.  Their split is 8 epochs against 100
    # (a "short probabilistic fine-tuning"), and the curriculum is worth 3.4 %
    # CRPS + 3x faster convergence than CRPS from scratch.  0 = off (pure
    # stage 1, every existing arm).  Requires sfno_dropout > 0: MC-Dropout is
    # the ensemble source, and at p=0 every member is identical so the CRPS
    # degenerates to the MAE (refused at loop entry, not silently trained).
    # Per-branch LR / weight-decay multipliers for optimizer=muon_partitioned
    # (U-Cast: Muon 3e-3 / wd 0.1 vs AdamW 3e-4 / wd 0.03 on the 1-D group).
    # 1.0 = the single shared value, i.e. every existing suite unchanged.
    muon_lr_scale: float = 1.0
    adamw_lr_scale: float = 1.0
    # 0.0 preserves the pre-split behaviour (optax muon's own default is no
    # decay); see the TrainingConfig field comment.
    muon_weight_decay_scale: float = 0.0
    adamw_weight_decay_scale: float = 1.0
    # Number of PAST states fed alongside the current one (U-Cast / GraphCast
    # feed two inputs; we fed one). 0 = every existing arm, byte-identical.
    sfno_history_steps: int = 0
    crps_finetune_epochs: int = 0
    # Members drawn per optimizer step during stage 2.  U-Cast uses M=2 and
    # reports M=4 as "only marginal" — the cost is linear in M.
    crps_ensemble_size: int = 2

    # Training
    n_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    optimizer: str = "adamw"     # adamw | adam | muon — passed to ml.training.create_optimizer
    warmup_steps: int = 100      # used by the warmup-cosine schedule
    # Early stopping (AIMIP-style "stop if no improvement after months
    # of training" -- one epoch = full pass over the windowed train
    # set).  ``early_stop_patience <= 0`` disables early stopping.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1.0e-3
    # Optional list of ``(year, day_offset, n_days)`` windows for
    # contiguous-within-window multi-year sampling without temporal
    # leakage across windows.  ``None`` falls back to legacy
    # single-window behaviour (``start_year``+``n_train_days``).
    windows: tuple | None = None
    # NeuralGCM-style rollout curriculum INSIDE the main training loop:
    # ``((lead_hours, n_epochs), ...)`` phases, short leads first. Each
    # phase supervises a single autoregressive rollout to ``lead_hours``
    # against the ERA5 target at that lead (targets for every phase lead
    # are loaded up front). This is where fleet models get their
    # free-running stability — as part of TRAINING, not a bolt-on
    # fine-tune. ``None`` keeps the fixed ``loss_config.multi_step_hours``
    # behaviour. Overrides ``n_epochs`` (total = sum of phase epochs).
    rollout_curriculum: tuple | None = None
    # Streaming/chunked data: when >0, the windowed training set is loaded
    # and trained in chunks of this many WINDOWS per chunk (dense all-years
    # sampling would not fit in host RAM as one list). 0 = load everything
    # at once (legacy).
    chunk_windows: int = 0
    # Internal perf flag (not a science knob): when True and chunk_windows>0,
    # chunk k+1 is loaded (GCS read + regrid) on a background host thread while
    # chunk k trains, overlapping ~0.5 h/chunk of serial load time with GPU
    # compute (#985 item 2). Doubles the in-RAM chunk footprint (1-ahead buffer).
    # Off by default so a running chain's behaviour is unchanged until the owner
    # opts in at a link boundary. Ordering + start_chunk resume are identical to
    # the serial path (see _prefetch_iter).
    chunk_prefetch: bool = False
    # Data-parallel (#985 item 1): when True AND launched under a multi-rank MPI
    # job (nproc>1), each chunk's samples are sharded across ranks and gradients
    # are averaged every step -> ~N x throughput on the 72 h-lead phase's GPU
    # time.  Off / single-rank -> byte-identical serial path.  This shifts the
    # effective batch from 1 (serial SGD) to N (one synced update per N samples),
    # so it is a training-trajectory change, not just a speedup — opt in per run.
    data_parallel: bool = False
    # EMA of the trainable weights (U-Cast/GenCast convention, see
    # training/ema.py): when > 0, maintain ema <- d*ema + (1-d)*model after
    # every optimizer step and save ``epoch_NNNN_ema.eqx`` beside each
    # per-epoch checkpoint; on the chunked path the EMA is folded INTO the
    # atomic ``chunk_latest.eqx`` payload (no separate file). Evaluation
    # prefers the EMA weights when present. 0.0 = off (legacy behaviour,
    # byte-identical trajectory).
    ema_decay: float = 0.0

    # Data
    n_train_days: int = 365      # Number of daily IC/target pairs
    start_year: int = 2015       # ERA5 year to use

    # Rollout supervision horizon.  Legacy semantics: each IC/target
    # pair is integrated for ``rollout_days * 24`` simulated hours.
    # v12 adds ``rollout_hours`` for sub-daily horizons (AIMIP /
    # NeuralGCM convention is 6-hour pairs).  When ``rollout_hours``
    # is set explicitly it takes precedence; otherwise the loader
    # falls back to ``rollout_days * 24``.  The ERA5 snapshots are
    # loaded at the ``TrainingERA5Config.dt_hours`` cadence (6 h by
    # default), so ``rollout_hours`` must be a multiple of 6.
    rollout_days: int = 1
    rollout_hours: int = 0   # 0 -> use rollout_days * 24

    # Per-group learning-rate multiplier for AIMIP spatial-surface
    # coefficients.  When the trainable model carries a
    # ``spatial_surface`` field (AIMIPClassicalParams under
    # ``spatial_surface=True``), these coefficients live in a
    # low-rank Legendre x Fourier basis whose gradient magnitudes are
    # typically much smaller than the sigmoid-bounded scheme scalars,
    # so the base AIMIP LR (~3e-4) leaves them effectively untrained.
    # A multiplier of 5-10 brings spatial-coef updates to the same
    # rough scale as scheme-knob updates per step.  Default 1.0
    # disables the per-group split (single-LR behavior).
    spatial_lr_scale: float = 1.0

    # Step period between radiation re-evaluations during the rollout.
    # When > 1, ``make_physics_fn`` is expected to return a tuple
    # ``(non_rad_fn, rad_fn)`` and :func:`spectral_rollout` gates the
    # rad call via ``lax.cond`` on the scan step index, caching the
    # last heating tendency between recomputes.  This drops RRTMGP
    # backward-pass memory and forward compute by ``N`` x at the cost
    # of holding the radiation tendency constant for the gating window
    # (acceptable for AIMIP forecast losses where the rad time scale
    # is ~hours, not seconds).  Default 1 = compute every step
    # (legacy combined physics_fn).
    rad_update_interval: int = 1

    # Loss
    loss_config: LossConfig = LossConfig()

    # Logging
    log_every: int = 5
    checkpoint_dir: str = "checkpoints/neural_gcm_spectral"


# =============================================================================
# Resume helpers: scan an output dir for the highest-numbered
# epoch_NNNN.eqx written by ``_train_spectral_loop`` /
# ``_train_sfno_full_loop`` (per-epoch checkpoints).  ``maybe_resume_model``
# loads it into the supplied model template so callers can continue a
# previously-killed training run from epoch ``last_done+1``.
# =============================================================================

def find_latest_epoch_checkpoint(ckpt_dir):
    """Find the highest-numbered ``epoch_NNNN.eqx`` in ``ckpt_dir``.

    Returns ``(epoch:int, path:Path)`` for the latest checkpoint, or
    ``None`` if the directory is missing / empty / contains no files
    that match the ``epoch_<int>.eqx`` pattern.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    best = None
    for p in ckpt_dir.glob("epoch_*.eqx"):
        try:
            ep = int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if best is None or ep > best[0]:
            best = (ep, p)
    return best


def label_aimip_spatial_params(params):
    """Label every leaf ``"spatial"`` / ``"base"`` for the per-group LR split.

    Detection is purely STRUCTURAL — any leaf whose tree path contains
    ``spatial_surface`` is a spatial coefficient — so the partition works for
    any module that nests an :class:`AIMIPSpatialSurfaceParams` under that
    attribute, at any depth.
    """
    def _label(path, _leaf):
        path_str = "/".join(
            str(getattr(p, "name", p) if hasattr(p, "name") else p)
            for p in path
        )
        return "spatial" if "spatial_surface" in path_str else "base"
    return jax.tree_util.tree_map_with_path(_label, params)


def has_spatial_surface_leaves(model) -> bool:
    """True when ``model`` carries AIMIP spatial-surface coefficients.

    Uses the SAME path test :func:`label_aimip_spatial_params` partitions on,
    not a top-level ``getattr``: a wrapper model (``AIMIPTrainableBundle``)
    nests the coefficients one level down, so the attribute lookup returned
    None and the scaled LR was silently dropped while the leaves still existed.
    """
    return "spatial" in jax.tree_util.tree_leaves(
        label_aimip_spatial_params(eqx.filter(model, eqx.is_array))
    )


def maybe_resume_model(model_template, resume_from_dir):
    """Load latest epoch checkpoint from ``resume_from_dir`` into ``model_template``.

    Returns ``(model, start_epoch)``.  When no checkpoint exists or
    ``resume_from_dir`` is ``None``, returns ``(model_template, 0)``.

    The caller is responsible for constructing ``model_template`` with
    the same pytree structure as the saved model.
    """
    if resume_from_dir is None:
        return model_template, 0
    latest = find_latest_epoch_checkpoint(resume_from_dir)
    if latest is None:
        return model_template, 0
    epoch_done, ckpt_path = latest
    from legoesm.ml.training import load_checkpoint
    model = load_checkpoint(model_template, ckpt_path)
    start_epoch = epoch_done + 1
    logger.info(
        f"Resume: loaded {ckpt_path} (last completed epoch={epoch_done}); "
        f"continuing at epoch {start_epoch}"
    )
    return model, start_epoch


# =============================================================================
# Mid-epoch (per-CHUNK) checkpoint for the dense/chunked all-years trainer.
#
# The per-epoch ``epoch_NNNN.eqx`` above is fine when one epoch fits inside
# one walltime link.  For the DENSE all-years T106 config a single epoch is
# ~13 h of GCS-streamed chunks while the Derecho main queue caps walltime at
# 12 h (#942): a kill during chunk 8/8 loses the whole epoch and the
# self-chaining resubmit restarts epoch 0 forever -> zero progress.
#
# ``chunk_latest.eqx`` closes that gap.  After every CHUNK completes we save
# enough to resume EXACTLY where the kill happened -- and, unlike the
# per-epoch model-only checkpoint, we save the OPTIMIZER STATE too, so the
# Adam/MUON moments and the warmup+cosine step counter continue unbroken
# (an epoch-boundary resume off ``epoch_NNNN.eqx`` re-inits them; a chunk
# resume off ``chunk_latest.eqx`` does not).  The saved position ``(epoch,
# next_chunk)`` is the chunk to RESUME AT; the last chunk of epoch e is
# normalised to ``(e+1, 0)``.  The write is atomic (temp + os.replace via
# ``save_checkpoint``) so a walltime kill mid-write can't corrupt it.
#
# What a chunk resume reproduces exactly: model weights + optimizer state
# are restored bit-for-bit; the chunk/sample order is a deterministic,
# unshuffled partition of ``config.windows`` (``_make_chunk_loader``) so
# skipping the already-done chunks and replaying the rest yields the same
# training trajectory an uninterrupted run would have taken.  There is no
# per-step RNG (SFNO/rollout are deterministic, M=1 "CRPS" is MAE) and the
# curriculum position is a pure function of ``epoch`` (``epoch_plan``), so
# ``(epoch, next_chunk)`` is the complete resume state.
# =============================================================================

MIDEPOCH_CHECKPOINT_NAME = "chunk_latest.eqx"


def _save_midepoch_checkpoint(
    ckpt_dir, model, opt_state, epoch, next_chunk, ema_model=None
):
    """Atomically save the mid-epoch (per-chunk) resume state.

    Serialises ``(model, opt_state, [ema_model,] epoch, next_chunk)`` as ONE
    payload via :func:`legoesm.ml.training.save_checkpoint` (temp file +
    atomic ``os.replace``), so the model weights, the optimizer state, the
    EMA weights and the resume position land together or not at all — there
    is no torn raw/EMA pair (the EMA is inside the same atomic write).

    When ``ema_model`` is None the legacy 4-tuple layout is written (EMA
    disabled); when provided a 5-tuple with the EMA between opt_state and
    the position ints. ``epoch``/``next_chunk`` give the position to RESUME
    AT (the last chunk of epoch ``e`` is stored as ``(e+1, 0)``).
    """
    from legoesm.ml.training import save_checkpoint
    ckpt_dir = Path(ckpt_dir)
    _epoch_i = jnp.asarray(int(epoch), dtype=jnp.int32)
    _chunk_i = jnp.asarray(int(next_chunk), dtype=jnp.int32)
    if ema_model is None:
        payload = (model, opt_state, _epoch_i, _chunk_i)
    else:
        payload = (model, opt_state, ema_model, _epoch_i, _chunk_i)
    save_checkpoint(payload, ckpt_dir / MIDEPOCH_CHECKPOINT_NAME)


def _load_midepoch_checkpoint(
    ckpt_dir, model_template, opt_state_template, ema_template=None
):
    """Load the mid-epoch resume state, or ``None`` if absent.

    Returns ``(model, opt_state, epoch:int, next_chunk:int)`` when
    ``ema_template`` is None, else ``(model, opt_state, ema_model, epoch,
    next_chunk)`` (``ema_model`` is None if the on-disk payload predates
    EMA). ROBUST to a layout/arity mismatch in EITHER direction — an EMA
    5-tuple resumed with EMA disabled, or a legacy 4-tuple resumed with EMA
    enabled — by trying the requested layout first and the other on any
    deserialise failure (equinox raises assorted exception types on a
    template mismatch, so the fallback is deliberately broad).
    """
    if ckpt_dir is None:
        return None
    path = Path(ckpt_dir) / MIDEPOCH_CHECKPOINT_NAME
    if not path.exists():
        return None
    _zero = jnp.asarray(0, dtype=jnp.int32)
    _tmpl4 = (model_template, opt_state_template, _zero, _zero)
    # A usable EMA template for the 5-tuple attempt: the caller's, or a
    # copy of the model template (structure is all that matters for
    # deserialise; leaves are overwritten from disk).
    _ema_t = ema_template if ema_template is not None else model_template
    _tmpl5 = (model_template, opt_state_template, _ema_t, _zero, _zero)

    def _read4():
        m, o, e, c = eqx.tree_deserialise_leaves(str(path), _tmpl4)
        return m, o, None, int(e), int(c)

    def _read5():
        m, o, em, e, c = eqx.tree_deserialise_leaves(str(path), _tmpl5)
        return m, o, em, int(e), int(c)

    # ALWAYS attempt the 5-tuple layout FIRST, then fall back to 4-tuple.
    # Direction matters because equinox accepts a PREFIX template without
    # rejecting trailing file leaves (patrick-kidger/equinox#136): reading a
    # genuine 5-tuple with the 4-tuple template could silently consume the
    # EMA's leaves as the position ints. 5-first avoids that — a real
    # 5-tuple deserialises cleanly, and a genuine 4-tuple read with the
    # 5-tuple template RELIABLY raises (the file's int32 epoch scalar lands
    # where the 5-tuple template expects the EMA model's float array leaf,
    # a dtype/shape mismatch), so control reaches the 4-tuple fallback.
    _last_exc = None
    for _reader in (_read5, _read4):
        try:
            m, o, em, e, c = _reader()
        except Exception as exc:  # equinox mismatch: broad by design
            _last_exc = exc
            continue
        if ema_template is None:
            # Caller does not want the EMA (EMA disabled this run) — drop it.
            return m, o, int(e), int(c)
        return m, o, em, int(e), int(c)
    raise RuntimeError(
        f"Could not deserialise {path} as either the 5-tuple or 4-tuple "
        f"mid-epoch layout: {_last_exc!r}"
    )


# =============================================================================
# State conversion: SegmentCarry -> SpectralHydrostaticState
# =============================================================================

def carry_to_spectral_state(
    carry,
    grid: GaussianGrid,
    include_tracers: bool = True,
) -> SpectralHydrostaticState:
    """Convert a SegmentCarry (grid-space) to SpectralHydrostaticState.

    Performs SH analysis to transform grid-space fields (u, v, T, p_s,
    phis) into spectral coefficients (vor_hat, div_hat, T_hat, lnps_hat,
    phis_hat).

    When ``include_tracers=True`` (the default), the SegmentCarry's
    moisture fields (``q_v``, ``q_c``, ``q_r``) are packaged as
    grid-space ``Field`` entries on ``state.tracers``.  This matches
    the spectral PE dycore's tracer-pytree convention: tracers stay on
    the grid even though the prognostic dynamics fields live in
    spectral space.  Set ``include_tracers=False`` to fall back to the
    pre-tracer dry pipeline (``state.tracers=None``).

    Parameters
    ----------
    carry : SegmentCarry
        Grid-space state from ERA5 ingestion.
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    include_tracers : bool
        Whether to attach ``carry.q_v`` / ``q_c`` / ``q_r`` as grid-
        space ``Field`` entries on ``state.tracers``.

    Returns
    -------
    SpectralHydrostaticState
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "carry_to_spectral_state requires JAX_ENABLE_X64=True for "
            "float64 spectral transforms."
        )

    u = carry.u.astype(jnp.float64)
    v = carry.v.astype(jnp.float64)
    T = carry.T.astype(jnp.float64)
    p_s = carry.p_s.astype(jnp.float64)
    phis = carry.phis.astype(jnp.float64)

    # log(surface pressure) -> spectral
    lnps = jnp.log(jnp.maximum(p_s, 1.0))
    lnps_hat = sh_analysis(grid, lnps)

    # Temperature -> spectral
    T_hat = sh_analysis_3d(grid, T)

    # Surface geopotential -> spectral
    phis_hat = sh_analysis(grid, phis)

    # (u, v) -> (vor_hat, div_hat) via the EXACT left-inverse of
    # the spectral wind synthesis (spectral_pe_to_grid / uv_from_vordiv_3d).
    # The plain Bourke ``oc2``/``dmu`` analysis (vordiv_from_uv_3d) is NOT an
    # exact left-inverse at the truncation boundary and amplifies pole-row
    # wind error ~×21/pass at T85, which makes the WB2 eval round trip
    # (era5 -A-> state -S-> carry -A-> state) explode (#976).
    # ``vordiv_from_uv_exact_3d`` solves the per-m least-squares system so the
    # carry<->state round trip is idempotent, pole rows included.
    vor_hat, div_hat = vordiv_from_uv_exact_3d(grid, u, v)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    grid_dims_3d = ("lat", "lon", "level")

    tracers = None
    if include_tracers:
        # The species carried through the time stepping follow the MICROPHYSICS
        # SCHEME, as they already do in the production driver: the carry's
        # optional hydrometeor / number fields are seeded (non-None) by
        # ``prognostic_carry_seeds`` only when the selected scheme needs more
        # than the three warm-rain slots, so PRESENCE ON THE CARRY *is* the
        # scheme test.  Names and units come from the registry that defines
        # them (``make_full_moisture_registry``) rather than being retyped.
        #
        # This used to hardcode q_v/q_c/q_r.  A nine-species scheme (morrison,
        # thompson, p3, seifert_beheng, fast_sbm) then ran against a
        # three-species state: the microphysics bridge emits a tendency only
        # for a key already present on ``state.tracers``, so ice, snow, graupel
        # and the number concentrations restarted from zero at every
        # evaluation while the vapour and cloud sinks that produced them — and
        # the latent heat they released — were kept.  Neither water nor energy
        # closed, and nothing raised.
        _units = {t.name: t.units for t in make_full_moisture_registry().tracers}
        tracers = {
            name: Field(
                getattr(carry, name).astype(jnp.float64),
                name=name, dims=grid_dims_3d, units=units,
            )
            for name, units in _units.items()
            if getattr(carry, name, None) is not None
        }

    return SpectralHydrostaticState(
        vor_hat=Field(vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(lnps_hat, name="lnps_hat", dims=dims_2d, units="-"),
        phis_hat=Field(phis_hat, name="phis_hat", dims=dims_2d, units="m2/s2"),
        tracers=tracers,
    )


def spectral_state_to_carry(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
):
    """Convert a SpectralHydrostaticState to a grid-space SegmentCarry.

    Inverse of :func:`carry_to_spectral_state` (SH synthesis of the
    prognostic spectral fields back to the Gaussian grid; tracers are
    already grid-space).  Used by the WB scale trainer's spectral
    training core (#817) so a rolled-out spectral state can be scored
    by the carry-vs-carry losses (``combined_loss``) against an
    ``era5_to_spectral_carry`` target — the pred and target then share
    the exact packing (zero held/accum diagnostics, ``step_index=0``)
    that ``era5_to_spectral_carry`` uses, so the loss only ever sees
    the physical fields (u, v, T, q_v, p_s) differ.
    """
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    fields = spectral_pe_to_grid(state, grid, sigma_coord)
    u, v, T, p_s = fields["u"], fields["v"], fields["T"], fields["p_s"]
    phis = fields["phis"]

    grid_dims_3d = ("lat", "lon", "level")
    grid_dims_2d = ("lat", "lon")
    hstate = HydrostaticState(
        u=Field(u, name="u", dims=grid_dims_3d, units="m/s"),
        v=Field(v, name="v", dims=grid_dims_3d, units="m/s"),
        T=Field(T, name="T", dims=grid_dims_3d, units="K"),
        p_s=Field(p_s, name="p_s", dims=grid_dims_2d, units="Pa"),
        phis=Field(phis, name="phis", dims=grid_dims_2d, units="m2/s2"),
    )

    def _tracer(name):
        if state.tracers is not None and name in state.tracers:
            tr = state.tracers[name]
            return tr.data if hasattr(tr, "data") else tr
        return jnp.zeros_like(T)

    # Give back exactly the species the state carried, so a rolled-out carry
    # has the same structure as the ``era5_to_spectral_carry`` initial
    # condition it is scored against.  Absent species stay ``None`` rather than
    # becoming zeros: ``None`` is what marks a slot as unused for this scheme,
    # and zeros would make a three-species run's carry structurally different
    # from its own initial condition.
    # Set difference against the warm-rain registry, not a positional slice:
    # the three always-present species are defined by name, so a reordering of
    # the full registry cannot silently shift which ones this treats as
    # optional.
    _warm = set(make_moisture_registry().names)
    _extra = {
        name: _tracer(name)
        for name in make_full_moisture_registry().names
        if name not in _warm
        and state.tracers is not None and name in state.tracers
    }

    shape_3d = T.shape
    shape_2d = p_s.shape
    return pack_carry(
        hstate,
        q_v=_tracer("q_v"),
        q_c=_tracer("q_c"),
        q_r=_tracer("q_r"),
        **_extra,
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


# =============================================================================
# SFNO as spectral physics
# =============================================================================

# Surface-forcing planes appended to the SFNO input (prescribed-SST /
# AMIP pathway): T_sfc, sea-ice fraction, TOA insolation.
N_SFNO_FORCING_CHANNELS = 3
# Per-plane normalization: T [K] ~300, sic already in [0,1], insolation
# [W/m^2] ~1400 (matches the column-MLP feature scales in
# ``atmosphere.physics.neural_physics``).  Kept as a plain Python tuple, NOT a
# module-top ``jnp.array``: an import-time device allocation is forbidden by
# test_no_module_top_jax_alloc (crashes the import chain on the experimental
# Metal backend).  It enters the graph lazily via ``jnp.asarray`` at the use
# site below, cast to ``in_scale``'s dtype so the concatenate result dtype is
# identical to the former module-top ``jnp.array`` (in both x32 and x64).
_SFNO_FORCING_INPUT_SCALE = (300.0, 1.0, 1400.0)


def _channel_input_scale(nlev: int) -> jnp.ndarray:
    """Per-channel normalization factors for PE3DChannelSpec.

    Divides each channel by its typical magnitude so the SFNO sees O(1)
    inputs regardless of physical units.  Layout: [u, v, T, q, lnps, phis].
    """
    return jnp.concatenate([
        jnp.full(nlev, 30.0),       # u [m/s]
        jnp.full(nlev, 30.0),       # v [m/s]
        jnp.full(nlev, 300.0),      # T [K]
        jnp.full(nlev, 0.01),       # q [kg/kg]
        jnp.array([12.0]),          # lnps [-]
        jnp.array([50000.0]),       # phis [m2/s2]
    ])


def _channel_tendency_scale(nlev: int) -> jnp.ndarray:
    """Per-channel output scaling from O(1) SFNO output to physical tendencies.

    Typical atmospheric physics tendency magnitudes per second.
    """
    return jnp.concatenate([
        jnp.full(nlev, 1e-4),       # du/dt [m/s^2]
        jnp.full(nlev, 1e-4),       # dv/dt [m/s^2]
        jnp.full(nlev, 1e-4),       # dT/dt [K/s]
        jnp.full(nlev, 1e-7),       # dq/dt [kg/kg/s]
        jnp.array([1e-6]),          # dlnps/dt [1/s]
        jnp.array([0.0]),           # dphis/dt (static)
    ])


def make_sfno_spectral_physics(sfno: SFNO, grid: GaussianGrid):
    """Create a physics_fn compatible with SpectralPEModel.step().

    The returned function has signature::

        physics_fn(state, grid, sigma_coord, forcing=None)
            -> SpectralHydrostaticState  (tendencies)

    Includes input normalization and output scaling so that a
    randomly-initialized SFNO produces stable, small tendencies.

    Internally:
    1. ``pack_pe_state`` transforms spectral state to grid-space tensor
    2. Three surface-forcing planes are appended (T_sfc, sea-ice
       fraction, TOA insolation) — the prescribed-SST / AMIP pathway
       that gives the SFNO its interannual-variability response.  The
       state channels stay ``PE3DChannelSpec`` (4*nlev+2); the SFNO is
       constructed with ``in_channels = n_channels + 3`` and
       ``out_channels = n_channels``.
    3. Input is normalized to O(1) per channel
    4. SFNO forward pass (residual_prediction=False) produces O(1) output
    5. Output is scaled to physical tendency magnitudes
    6. ``unpack_pe_output`` converts to spectral tendencies

    ``forcing`` dict (traced; see ``spectral_amip_rollout``):
    ``"T_sfc"`` (ncol,) prescribed surface T over ocean / NaN over land
    (→ lowest-level air T proxy), ``"sic"`` (ncol,) sea-ice fraction,
    ``"day_of_year"`` + ``"seconds_of_day"`` scalars (seasonal + diurnal
    insolation via the shared orbital helper).  Without forcing
    (idealized / legacy tests): T_sfc proxy = lowest model level,
    sic = 0, insolation = S_0 (the historical constant input).

    Parameters
    ----------
    sfno : SFNO
        Spherical Fourier Neural Operator (eqx.Module).
    grid : GaussianGrid
        Grid for SH transforms inside pack/unpack.

    Returns
    -------
    callable
        Physics function for the spectral PE dycore.
    """
    from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle

    # nlev from the OUTPUT channels (pure state tendencies, 4*nlev+2);
    # in_channels additionally carries the N_SFNO_FORCING_CHANNELS
    # forcing planes.
    nlev = (sfno.config.out_channels - 2) // 4
    if (sfno.config.in_channels != sfno.config.out_channels + N_SFNO_FORCING_CHANNELS
            or (sfno.config.out_channels - 2) % 4 != 0
            or nlev < 1):
        raise ValueError(
            f"SFNO physics expects out_channels = 4*nlev + 2 (PE state "
            f"layout) and in_channels = out_channels + "
            f"{N_SFNO_FORCING_CHANNELS} (forcing planes); got "
            f"in={sfno.config.in_channels}, out={sfno.config.out_channels}."
        )
    in_scale = _channel_input_scale(nlev)
    out_scale = _channel_tendency_scale(nlev)
    spec = PE3DChannelSpec(nlev=nlev)
    lat2d = jnp.broadcast_to(grid.lat[:, None], (len(grid.lat), len(grid.lon)))
    lon2d = jnp.broadcast_to(grid.lon[None, :], (len(grid.lat), len(grid.lon)))

    def physics_fn(state, grid_, sigma_coord, forcing=None):
        packed = pack_pe_state(state, grid_)
        n_lat, n_lon = packed.shape[:2]
        # Lowest-level air T from the packed T channel (proxy for T_sfc
        # over land / when no forcing is prescribed).
        t_lowest = packed[..., spec.T_slice][..., -1]
        if forcing is not None:
            # Sanitize the inactive branch BEFORE the select: jnp.where
            # propagates NaN cotangents from the untaken branch in reverse
            # mode (codex HIGH) — land-NaN T_sfc would poison the training
            # gradient through the proxy path.
            t_raw = jnp.nan_to_num(
                forcing["T_sfc"].reshape(n_lat, n_lon), nan=0.0,
            )
            t_sfc = jnp.where(
                jnp.isfinite(forcing["T_sfc"]).reshape(n_lat, n_lon),
                t_raw, t_lowest,
            )
            sic = jnp.clip(
                jnp.nan_to_num(forcing["sic"].reshape(n_lat, n_lon), nan=0.0),
                0.0, 1.0,
            )
            mu0 = cos_zenith_angle(
                lat2d, lon2d,
                forcing["day_of_year"], forcing["seconds_of_day"] / 3600.0,
            )
            insol = constants.S_0 * jnp.maximum(mu0, 0.0)
        else:
            t_sfc = t_lowest
            sic = jnp.zeros((n_lat, n_lon), dtype=packed.dtype)
            insol = jnp.full((n_lat, n_lon), constants.S_0, dtype=packed.dtype)
        packed_in = jnp.concatenate(
            [packed,
             t_sfc[..., None], sic[..., None], insol[..., None]],
            axis=-1,
        )
        # Normalize inputs to O(1)
        packed_norm = packed_in / jnp.maximum(
            jnp.concatenate([
                in_scale,
                jnp.asarray(_SFNO_FORCING_INPUT_SCALE, dtype=in_scale.dtype),
            ]), 1e-10,
        )
        # SFNO forward: O(1) in, O(1) out
        output_norm = sfno(packed_norm.astype(jnp.float32), grid_)
        # Scale to physical tendency magnitudes
        output = output_norm * out_scale
        tend = unpack_pe_output(
            output.astype(jnp.float64), state, grid_, mode="tendencies",
        )
        # DRY-MASS constraint (ACE2-style budget fixer): physics must not
        # create or destroy air. The SFNO's raw dlnps/dt carries a nonzero
        # global mean -> a secular surface-pressure (mass) drift that
        # compounds over climate-length integrations and feeds the
        # radiative-runaway class. Project out the (l=0, m=0) spectral
        # coefficient = the global mean of the lnps tendency (the standard
        # lnps-mean proxy for dry-mass conservation on sigma coordinates).
        # NOTE: this is an APPROXIMATE fixer — dry mass ∝ ∫p_s dA, so the
        # exact constraint is area_mean(p_s·dlnps/dt)=0; zeroing the
        # area-mean dlnps/dt leaves a residual O(p_s') where p_s' is the
        # spatial p_s anomaly (~5%). It removes the secular global-mean
        # drift mode (the instability driver) and stays differentiable.
        _mean_mask = jnp.asarray(
            (grid.ls == 0) & (grid.ms == 0), dtype=tend.lnps_hat.data.dtype,
        )
        lnps_fixed = tend.lnps_hat.data * (1.0 - _mean_mask)
        return tend._replace(lnps_hat=tend.lnps_hat.replace(data=lnps_fixed))

    return physics_fn


# =============================================================================
# Column MLP as spectral physics (Rasp et al. 2018 style)
# =============================================================================

# The column MLP physics component lives in the physics directory:
#   legoesm.atmosphere.physics.neural_physics
# Re-export the coupling function for training convenience.
from legoesm.atmosphere.physics.neural_physics import (  # noqa: E402
    make_column_physics_fn as make_column_mlp_spectral_physics,
    build_column_physics,
)


def make_turbulence_only_spectral_physics(dt,
                                          turbulence_scheme="smagorinsky",
                                          turbulence_cfg=None):
    """A spectral ``physics_fn`` carrying ONLY the turbulence scheme (#1464).

    NAMED FOR WHAT IT IS, not for what it is used for. This was first written
    as ``make_momentum_only_*``, which was wrong: MEASURED on a sheared state,
    ``smagorinsky`` returns vor/div tendencies of 1.44e-9 / 1.30e-9 AND a
    temperature tendency of 1.64e-5 — the scheme diffuses heat as well as
    momentum. The T part is real physics and is not zeroed here; it is the
    CONSUMER that drops it, because ``make_column_physics_fn`` takes only
    vor/div/lnps from its momentum source and always keeps the network's
    thermodynamics.

    Built to pair with ``make_column_mlp_spectral_physics``, whose network has
    no momentum head and therefore returns zero vor/div — a model with no
    surface drag at all. Passing this in gives the learned arm the SAME
    momentum sink the classical arm gets by default
    (``TurbulenceConfig.scheme = "smagorinsky"``, inherited whenever a
    ``PhysicsConfig`` is built with only radiation and convection named), so
    the two arms differ in their THERMODYNAMICS and nothing else.

    Radiation, convection, microphysics and GWD are explicitly ``"none"`` here
    — the network owns those. Being explicit matters: the confound this fixes
    came from a sub-config that was never named and so kept its default.

    Parameters
    ----------
    dt : float
        Physics timestep [s], as for the other builders here.
    turbulence_scheme : str
        Any scheme ``TurbulenceConfig`` accepts. Validated by the factory's own
        dispatch, which raises on an unknown name.
    turbulence_cfg : TurbulenceConfig, optional
        A fully-built config; overrides ``turbulence_scheme`` when given.
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag import (
        GravityWaveDragConfig,
    )

    turb = (turbulence_cfg if turbulence_cfg is not None
            else TurbulenceConfig(scheme=turbulence_scheme))
    # STATELESS ONLY. This wrapper passes no PhysicsState and discards the
    # carry the scheme returns, so a TKE-family scheme would re-seed its
    # prognostic TKE from the floor on EVERY call — a silently different
    # closure, not the one named (codex). Refuse rather than run it wrong;
    # threading the carry is the fix if one of these is ever wanted here.
    _STATEFUL = ("tke", "mynn25", "clubb", "clubb_lite", "edmf")
    if turb.scheme in _STATEFUL:
        raise ValueError(
            f"make_turbulence_only_spectral_physics: {turb.scheme!r} is a "
            f"PROGNOSTIC scheme and this builder threads no PhysicsState, so "
            f"its carry would be re-seeded from the floor every step. Use a "
            f"stateless scheme (e.g. 'smagorinsky', 'louis', "
            f"'holtslag_boville', 'ysu') or thread the carry first.")
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=turb,
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    raw_fn = make_physics(cfg, model_type="spectral_pe", dt=dt)

    def physics_fn(state, grid_, sigma_coord):
        result = raw_fn(state, grid_, sigma_coord)
        return result[0] if isinstance(result, tuple) else result

    return physics_fn


# =============================================================================
# Physics-based parameterizations with trainable parameters
# =============================================================================

def make_physics_params_spectral_physics(params, grid, dt, *,
                                         radiation: str = "rrtmgp"):
    """Create a spectral PE physics_fn from trainable physics parameters.

    Rebuilds the combined physics (radiation + convection + turbulence + ...)
    with the current parameter values as JAX arrays so that gradients
    flow through the physics computations back to the parameters.

    The trainable parameters are injected into the scheme configs:
    - ``sbm_tau_c``, ``sbm_RH_ref`` → SBM convection timescale/humidity
    Gray radiation takes NO trained value since 2026-08-11 (its optical depths
    ``tau_equator``/``tau_pole`` left the trainable set with it).

    Parameters
    ----------
    params : TrainablePhysicsParams
        Current trainable parameter values (eqx.Module).
    grid : GaussianGrid
        Grid for spectral transforms.
    dt : float
        Dycore timestep [s].

    Returns
    -------
    callable
        ``physics_fn(state, grid, sigma_coord) -> SpectralHydrostaticState``
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, GrayRadiationConfig,
    )
    from legoesm.atmosphere.physics.convection.config import (
        ConvectionConfig, SBMConfig,
    )

    p = params.as_dict()

    # Radiation backend. This function HARD-WIRED gray, so the WeatherBench
    # `physics` arm ran gray no matter what its config said — the campaign pin
    # could not see it (codex, 2026-08-12). WB/AIMIP now get rrtmgp; gray stays
    # selectable for the smoke path and for non-campaign callers, and takes no
    # trained value either way (its optical depths left DEFAULT_TRAINABLE).
    if radiation == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        rad_cfg = RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig())
    elif radiation == "gray":
        rad_cfg = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())
    else:
        raise ValueError(
            f"unsupported radiation {radiation!r} for the trainable-physics "
            f"spectral path: must be one of ('rrtmgp', 'gray')")

    # Build SBM convection config with trainable timescale + RH
    sbm_cfg = SBMConfig(
        tau_c=p.get('sbm_tau_c', 7200.0),
        rh_ref=p.get('sbm_RH_ref', 0.7),
    )
    conv_cfg = ConvectionConfig(scheme="sbm", sbm=sbm_cfg)

    physics_config = PhysicsConfig(radiation=rad_cfg, convection=conv_cfg)
    # The trainable SURFACE leaves reach rrtmgp as per-call overrides — writing
    # them into RRTMGPConfig would key its solver cache on a tracer. Without
    # this they are inert on the rrtmgp path (codex): gray consumed the blended
    # albedo directly, so flipping the backend silently froze albedo_ocean /
    # albedo_ice. C_H / C_E ride the surface-flux config, not radiation.
    _albedo = p.get("albedo_ocean")
    _kwargs = {}
    if _albedo is not None and radiation == "rrtmgp":
        _kwargs["sfc_albedo_override"] = _albedo
    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt,
                          **_kwargs)

    def physics_fn(state, grid_, sigma_coord):
        result = raw_fn(state, grid_, sigma_coord)
        return result[0] if isinstance(result, tuple) else result

    return physics_fn


# =============================================================================
# Differentiable spectral rollout
# =============================================================================

def _compute_tracer_filter(
    grid: GaussianGrid,
    pe_config: SpectralPEConfig,
    spectral_filter: jnp.ndarray | None,
    dt: float,
) -> jnp.ndarray | None:
    """Build the per-SH-mode multiplicative filter applied to grid-space
    tracers in :func:`spectral_rollout`.

    Mirrors :meth:`SpectralPrimitiveEquationModel._ensure_tracer_filter`:
    combines the spectral exponential filter (de-aliasing) with the
    implicit hyperdiffusion factor ``exp(-nu · eig · dt_eff)``.  Returns
    ``None`` when neither knob is active (caller should then skip the
    SH round-trip on tracers entirely).
    """
    components = []
    if spectral_filter is not None:
        components.append(spectral_filter)
    if pe_config.hyperdiff_coeff > 0:
        nu = pe_config.hyperdiff_coeff
        order = pe_config.hyperdiff_order
        eig = (grid.ls * (grid.ls + 1) / grid.radius ** 2) ** order
        integrator = pe_config.time_integrator.lower()
        dt_eff = 2.0 * dt if 'leapfrog' in integrator else dt
        components.append(jnp.exp(-nu * eig * dt_eff))
    if not components:
        return None
    combined = components[0]
    for c in components[1:]:
        combined = combined * c
    return combined


def _add_phys_tendencies(a, b):
    """Sum two physics tendency :class:`SpectralHydrostaticState`s.

    Both inputs share the dycore-tendency layout used by
    ``make_physics`` (Equinox Fields wrapping per-mode/per-level
    arrays).  This helper walks the field tree and adds the
    underlying ``.data`` arrays so that "non-rad" and "rad-only"
    physics_fn outputs can be combined under :func:`spectral_rollout`
    when the rad branch is gated by ``rad_update_interval > 1``.
    """
    def _sum(x, y):
        if hasattr(x, "data") and hasattr(y, "data"):
            return x.replace(data=x.data + y.data)
        return x + y
    return jax.tree_util.tree_map(
        _sum, a, b, is_leaf=lambda obj: hasattr(obj, "data"),
    )


def _make_spectral_integrator(pe_config, grid, sigma_coord, dt, integrator_name):
    """Build the per-step ``(state, tendency_fn) -> new_state`` integrator for the
    differentiable spectral training rollouts.

    #817: when ``pe_config.semi_implicit`` the WB neural_gcm/sfno training lane
    uses the pure Hoskins-Simmons SSP-RK3 **semi-implicit** step
    (:func:`ssp_rk3_step_si`) instead of the explicit
    :func:`dispatch_integrator`.  Treating the fast gravity-wave terms implicitly
    damps the explicit core's tangent-linear (adjoint) growth — the ~x1.3/step
    e-folding that overflowed ``value_and_grad`` to NaN past ~6 h even while the
    forward stayed finite (the physics lane trained only because its
    parameterised turbulence damped the same tangent-linear system).

    The SI matrices depend ONLY on grid / sigma / T_ref / alpha / dt (NOT on the
    trainable params), so they are precomputed ONCE here — outside any
    ``lax.scan`` — and closed over; the in-scan step is then a constant-matrix,
    traced-RHS linear solve, cheap and clean under reverse-mode AD.  We use the
    PURE ``ssp_rk3_step_si`` (single-level carry), NOT the model's ``leapfrog_si``
    path, which mutates ``self._state_prev`` and is not ``lax.scan``/AD-safe.
    ``si_substeps`` internal sub-steps are honoured (matching the model's
    ``_do_step``), sub-stepping at ``dt / si_substeps`` with SI matrices built for
    that sub-step dt.
    """
    if not pe_config.semi_implicit:
        def _integrate_explicit(state, tendency_fn):
            return dispatch_integrator(state, tendency_fn, dt, integrator_name)
        return _integrate_explicit

    from legoesm.timestepping.semi_implicit import (
        precompute_si_matrices,
        ssp_rk3_step_si,
    )

    n_sub = int(pe_config.si_substeps)  # static Python int
    if n_sub < 1:
        raise ValueError(
            f"si_substeps must be >= 1, got {pe_config.si_substeps!r}")
    dt_si = dt / n_sub
    si_data = precompute_si_matrices(
        grid, sigma_coord, pe_config.si_T_ref, pe_config.si_alpha, dt_si,
    )

    def _integrate_si(state, tendency_fn):
        if n_sub == 1:
            return ssp_rk3_step_si(state, tendency_fn, dt_si, si_data, grid)

        def _one(s, _):
            return ssp_rk3_step_si(s, tendency_fn, dt_si, si_data, grid), None

        s, _ = jax.lax.scan(_one, state, None, length=n_sub)
        return s

    return _integrate_si


def positive_tracers(tracers, sigma_coord):
    """Every water species non-negative after the transport + filter step.

    The spectral core's tracer transport and the post-step SH-round-trip
    filter are both NON-MONOTONE: at sharp moisture gradients they overshoot
    (Gibbs), leaving small negative mixing ratios every step (measured
    q_v ~ -7e-4 kg/kg on a 2016-09-01 ERA5 start at T63). Downstream physics
    has no negative-water state, and a plain ``max(q, 0)`` would convert the
    overshoot into a compounding spurious source (the MPAS-century
    +30 kg/m2/yr water lesson on ``conservative_positive_clip``, and the
    N_i -> 1e193 -> NaN number lesson on ``TestAllTracersBorrowed``).

    EVERY per-mass tracer — the six water mixing ratios AND the number
    concentrations, which are stored per mass [#/kg] — gets the per-column
    conserving borrow: clip to zero, then rescale the column's positive cells
    so the dsigma-weighted column integral is unchanged (``p_s/g`` is
    constant per column and cancels in the ratio). Eligibility comes from
    the SHARED rule (``is_borrow_eligible_tracer``), not a local units
    split: a units-based "numbers clip freely" branch here would repeat the
    2026-07-26 exclusion that the century measurement reversed (codex,
    2026-08-16 round 1, P0).

    Pure-sigma coordinates only: with a hybrid coordinate the layer mass is
    ``dA·p_ref + dB·p_s`` (per-column), and a flat ``dsigma`` weight would
    conserve the wrong physical integral — refuse loudly rather than
    mis-conserve (codex round 1, P1).

    Container-type-preserving like ``apply_filter_to_tracers`` (Field stays
    Field, raw array stays raw; Field detection requires BOTH ``data`` and
    ``replace``, matching that helper). Unknown tracer names raise: a new
    species must state its positivity class rather than inherit one silently.
    """
    from legoesm.core.conservation import is_borrow_eligible_tracer

    if tracers is None:
        return None
    if not isinstance(sigma_coord, SigmaCoordinate):
        raise ValueError(
            "positive_tracers: only pure-sigma coordinates are supported — "
            f"got {type(sigma_coord).__name__}. A hybrid coordinate needs "
            "per-column layer-mass weights (dA·p_ref + dB·p_s), not dsigma.")
    dsigma = sigma_coord.dsigma
    out = {}
    for name, val in tracers.items():
        if not is_borrow_eligible_tracer(name):
            raise ValueError(
                f"positive_tracers: tracer {name!r} is not a known per-mass "
                "species (BORROW_ELIGIBLE_TRACERS); classify it before "
                "running it through the spectral training core.")
        is_field = hasattr(val, "data") and hasattr(val, "replace")
        arr = val.data if is_field else val
        arr, _created = conservative_positive_clip(arr, dsigma, axis=-1)
        out[name] = val.replace(data=arr) if is_field else arr
    return out


def spectral_rollout(
    initial_state: SpectralHydrostaticState,
    physics_fn,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    pe_config: SpectralPEConfig,
    dt: float,
    n_steps: int,
    sponge_factor: jnp.ndarray | None = None,
    spectral_filter: jnp.ndarray | None = None,
    rad_physics_fn=None,
    rad_update_interval: int = 1,
    *,
    sim_time_offset_seconds: float = 0.0,
    forcing_base: dict | None = None,
    phys_state_in=None,
    return_phys_state: bool = False,
) -> SpectralHydrostaticState:
    """Roll out spectral PE + SFNO physics for n_steps using lax.scan.

    Each step:
    1. Compute combined tendencies (dynamics + SFNO physics)
    2. Integrate with SSP-RK3 (or configured integrator)
    3. Apply sponge layer damping (if enabled)
    4. Apply spectral filter (if enabled)
    5. Apply tracer filter (combined spectral + hyperdiff via SH round-
       trip) when ``initial_state.tracers`` is non-empty AND either
       ``spectral_filter`` or ``pe_config.hyperdiff_coeff > 0`` is on.

    Gradient checkpointing is applied per step so memory scales as
    O(1) per step rather than O(n_steps).

    PROGNOSTIC PHYSICS STATE. When ``physics_fn`` carries the
    ``with_phys_state`` / ``init_phys_state`` markers (the classical
    split-rad factory attaches them), the scan carry additionally threads a
    :class:`PhysicsState` (CLUBB wp2/TKE, Bechtold's organization +
    stochastic state, the GWD spectrum, the PDF cloud fraction), so
    stateful schemes keep their memory across steps instead of running at
    their cold-start floors. Contract (deliberate first-order operator
    split, made explicit per codex review): every RK stage of a step reads
    the STEP-INITIAL physics memory; the updated memory is harvested once
    per step from the pre-step state, and only that harvest enters the
    next step (Bechtold's PRNG advances exactly once per step — stage
    evaluations reuse the frozen key and their state outputs are
    discarded). ``phys_state_in`` seeds the thread (None -> the scheme
    floors via ``init_phys_state``); ``return_phys_state=True`` returns
    ``(final_state, final_phys_state)`` so a CHAINED multi-segment loss
    can carry the memory across segments instead of resetting it every
    lead (codex P0).

    Parameters
    ----------
    initial_state : SpectralHydrostaticState
        Initial condition in spectral space.
    physics_fn : callable
        ``(state, grid, sigma_coord) -> SpectralHydrostaticState``
        Physics-tendency function for every-step schemes.  When
        ``rad_physics_fn`` is supplied this should be the *non-rad*
        contribution; otherwise it is the combined contribution.
    grid : GaussianGrid
    sigma_coord : SigmaCoordinate
    pe_config : SpectralPEConfig
    dt : float
        Time step [s].
    n_steps : int
        Number of steps to integrate.
    sponge_factor : array or None
        Precomputed sponge damping factors per level.
    spectral_filter : array or None
        Precomputed exponential spectral filter.
    rad_physics_fn : callable or None
        Optional separate radiation-tendency callable.  When provided,
        the radiation contribution is evaluated only at step indices
        divisible by ``rad_update_interval`` (and at step 0); on
        intervening steps the cached previous heating tendency is
        added to ``physics_fn``'s non-rad contribution.  This keeps
        the backward-pass memory footprint and forward compute of the
        expensive RRTMGP solve under control on a 48-step daily
        rollout (one rad call instead of 48 when
        ``rad_update_interval=48``).  None disables gating; the
        legacy single-physics_fn path runs.
    rad_update_interval : int
        Step period between radiation re-evaluations.  Only used when
        ``rad_physics_fn`` is non-None.  Default 1 = compute every
        step (same as a single combined physics_fn).
    forcing_base : dict or None
        Optional prescribed surface forcing for learned-physics
        variants (column MLP / SFNO): ``{"T_sfc": (ncol,), "sic":
        (ncol,), "day_of_year": scalar, "seconds_of_day": scalar}``,
        all TRACED (SegmentForcing doctrine — per-sample values change
        without retrace).  Per step the calendar entries are advanced
        by the elapsed simulated time (``step_idx * dt +
        sim_time_offset_seconds``) and the dict is passed to
        ``physics_fn(..., forcing=...)``.  T_sfc / sic are held fixed
        across the rollout (daily-AMIP granularity).  Mutually
        exclusive with the rad-gating path (the classical AMIP rollout
        ``spectral_amip_rollout`` owns that combination).

    Returns
    -------
    SpectralHydrostaticState
        State after n_steps * dt seconds.
    """
    integrator_name = pe_config.time_integrator
    # #817: explicit dispatch_integrator, or the semi-implicit SSP-RK3-SI step
    # (precomputed ONCE, outside the scan) when pe_config.semi_implicit.
    _integrate = _make_spectral_integrator(
        pe_config, grid, sigma_coord, dt, integrator_name,
    )
    ms = grid.ms  # for sponge filter

    # Precompute the tracer filter (mirrors the precomputation done by
    # the model class for ordinary stepping).  ``None`` when neither
    # the spectral filter nor hyperdiffusion is enabled.
    tracer_filter = _compute_tracer_filter(
        grid, pe_config, spectral_filter, dt,
    )

    # DRY-MASS ANCHOR target, captured from THIS rollout's own initial state.
    # ``None`` = fixer off, which keeps the traced structure identical to every
    # pre-existing arm. Traced, not stored on any object, so it is safe under
    # jit/grad and is re-derived per forecast case (no cross-case staleness).
    _target_mass = (
        global_dry_mass(grid, initial_state.lnps_hat.data)
        if (pe_config.fix_mass and pe_config.anchor_mass_to_initial)
        else None
    )

    use_rad_gating = rad_physics_fn is not None and rad_update_interval > 1
    if forcing_base is not None and use_rad_gating:
        raise ValueError(
            "spectral_rollout: forcing_base is the learned-physics forcing "
            "path and cannot be combined with rad-gating; classical "
            "prescribed-SST runs go through spectral_amip_rollout."
        )

    if not use_rad_gating:
        # Legacy single-physics path -- physics_fn computes the full
        # tendency (radiation included or absent) every dycore step.
        _t_off = jnp.asarray(sim_time_offset_seconds, dtype=jnp.float64)

        def _forcing_at(step_idx):
            # Elapsed simulated time -> advancing day-of-year (seasonal
            # insolation, FRACTIONAL like spectral_amip_rollout so the
            # declination is continuous within a day — codex) + wrapped
            # seconds-of-day (diurnal phase).  day_of_year additionally
            # wraps to [1, 366) so a year-end IC never exceeds
            # cos_zenith_angle's documented 1-365 domain (the declination
            # is 365-periodic, so the wrap is phase-preserving).
            t = (step_idx.astype(jnp.float64) * dt + _t_off
                 + jnp.asarray(forcing_base["seconds_of_day"], jnp.float64))
            doy = (
                jnp.asarray(forcing_base["day_of_year"], jnp.float64)
                + t / 86400.0
            )
            return {
                "T_sfc": forcing_base["T_sfc"],
                "sic": forcing_base["sic"],
                "day_of_year": jnp.mod(doy - 1.0, 365.0) + 1.0,
                "seconds_of_day": jnp.mod(t, 86400.0),
            }

        def step_post(new_state):
            """Post-integration chain, shared by the stateless and stateful
            ungated bodies (expressions unchanged — extracted verbatim)."""
            # Implicit sponge damping at model top
            if sponge_factor is not None:
                new_state = apply_sponge_filter(new_state, sponge_factor, ms)

            # Exponential spectral filter on highest wavenumbers
            if spectral_filter is not None:
                new_state = apply_spectral_filter_to_state(
                    new_state, spectral_filter,
                )

            # Combined spectral + implicit hyperdiff applied to grid-space
            # tracers via one SH round-trip per tracer per step (no-op when
            # tracer_filter is None or state.tracers is None).
            if tracer_filter is not None and new_state.tracers is not None:
                new_state = new_state._replace(
                    tracers=apply_filter_to_tracers(
                        new_state.tracers, tracer_filter, grid,
                    )
                )

            # SPECIES POSITIVITY, every path, every step: the transport and
            # the filter above are non-monotone, so every species can leave
            # the step slightly negative. Conserving borrow for mass, plain
            # clip for numbers — see ``positive_tracers``. This subsumes the
            # earlier learned-arm-only q_v clip (whose "the classical paths
            # conserve by construction" rationale predated the nine-species
            # carry and was wrong for it).
            if new_state.tracers is not None:
                new_state = new_state._replace(
                    tracers=positive_tracers(
                        new_state.tracers, sigma_coord,
                    )
                )

            # DRY-MASS ANCHOR, last in the chain so it also absorbs what the
            # sponge / spectral / tracer filters above took out. Off unless a
            # suite asks for it, so every existing arm is byte-identical.
            #
            # It has to be HERE and not only on SpectralPrimitiveEquationModel:
            # AIMIP's classical and column_nn arms integrate through THIS
            # function, never that class, so the class-side fixer was
            # unreachable and both arms lost ~16 hPa of area-weighted mslp over
            # a 10-day forecast (2017 scorecards, 8 inits) while the arm with no
            # dycore lost 0.8 hPa. The target is captured from THIS rollout's
            # own initial state below — per forecast case, so a second case can
            # never anchor to the first case's mass (the staleness the
            # class-side ``_target_mass`` has by construction).
            if _target_mass is not None:
                new_state = anchor_lnps_to_mass(grid, new_state, _target_mass)
            return new_state

        def step_fn(state, step_idx):
            if forcing_base is not None:
                fc = _forcing_at(step_idx)
                def tendency_fn(s):
                    phys = physics_fn(s, grid, sigma_coord, forcing=fc)
                    return spectral_pe_tendencies(
                        s, grid, sigma_coord, pe_config, phys,
                    )
            else:
                def tendency_fn(s):
                    phys = physics_fn(s, grid, sigma_coord)
                    return spectral_pe_tendencies(
                        s, grid, sigma_coord, pe_config, phys,
                    )

            new_state = step_post(_integrate(state, tendency_fn))
            return new_state, None

        # PROGNOSTIC PHYSICS STATE on the ungated path too (interval-1 /
        # no-rad classical callers), same markers and operator-split
        # contract as the rad-gated body below — without this, an
        # interval-1 classical run would silently stay memoryless while
        # the gated one threads state (codex P1). Marker + forcing_base
        # has no caller (learned arms are markerless) and is refused
        # loudly rather than half-supported.
        _ps_entry_u = getattr(physics_fn, "with_phys_state", None)
        _ps_init_u = getattr(physics_fn, "init_phys_state", None)
        _thread_phys_u = _ps_entry_u is not None and _ps_init_u is not None
        if _thread_phys_u and forcing_base is not None:
            raise ValueError(
                "spectral_rollout: a phys-state-marked physics_fn with "
                "forcing_base has no supported path; thread forcing through "
                "the stateful entry first.")

        if _thread_phys_u:
            if phys_state_in is not None:
                phys0 = phys_state_in
            else:
                # Prefer the state-aware seed (shear-equilibrium TKE) over
                # the scheme-floor initializer: a floor seed cannot spin up
                # within a short window (sqrt-production bottleneck).
                _seed_u = getattr(physics_fn, "seed_phys_state", None)
                if _seed_u is not None:
                    phys0 = _seed_u(initial_state, grid, sigma_coord)
                else:
                    _ncol = int(grid.n_lat) * int(grid.n_lon)
                    _nlev = int(jnp.shape(sigma_coord.sigma_full)[0])
                    phys0 = _ps_init_u(_ncol, _nlev)

            def step_fn_stateful(carry, _):
                state, phys_state = carry
                # Harvest once per step on the pre-step state; stages read
                # the step-initial memory (same contract as the gated body).
                _, phys_state_new = _ps_entry_u(
                    state, grid, sigma_coord, phys_state)

                def tendency_fn(s):
                    phys = _ps_entry_u(s, grid, sigma_coord, phys_state)[0]
                    return spectral_pe_tendencies(
                        s, grid, sigma_coord, pe_config, phys,
                    )

                new_state = step_post(_integrate(state, tendency_fn))
                return (new_state, phys_state_new), None

            step_fn_ckpt = jax.checkpoint(
                step_fn_stateful,
                prevent_cse=True,
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            (final_state, final_ps), _ = jax.lax.scan(
                step_fn_ckpt, (initial_state, phys0), None, length=n_steps,
            )
            if return_phys_state:
                return final_state, final_ps
            return final_state

        if phys_state_in is not None or return_phys_state:
            raise ValueError(
                "spectral_rollout: phys_state_in/return_phys_state need a "
                "physics_fn carrying the with_phys_state/init_phys_state "
                "markers; this physics_fn has none.")

        # ``prevent_cse=True`` plus ``policy=nothing_saveable`` is the
        # most aggressive memory-saving mode: every intermediate is
        # recomputed during backward.  Necessary on a 48-step daily
        # rollout with RRTMGP enabled (the gas-optics + two-stream
        # solver allocate hundreds of GiB of activations otherwise).
        # Gray-radiation runs are insensitive to this choice.
        step_fn_ckpt = jax.checkpoint(
            step_fn,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        final_state, _ = jax.lax.scan(
            step_fn_ckpt, initial_state,
            jnp.arange(n_steps) if forcing_base is not None else None,
            length=None if forcing_base is not None else n_steps,
        )
        return final_state

    # Rad-gating path: ``physics_fn`` is the *non-rad* contribution,
    # evaluated every dycore step; ``rad_physics_fn`` is evaluated
    # only when ``step_idx % rad_update_interval == 0``, and its
    # output is cached in the scan carry for the intervening
    # ``rad_update_interval - 1`` steps.  This drops RRTMGP backward-
    # pass memory and forward compute by ``rad_update_interval`` x at
    # the cost of holding the radiation tendency constant for the
    # gating window (acceptable for the AIMIP daily forecast loss
    # where the rad time scale is ~hours, not seconds).
    # ``rad_physics_fn`` accepts an optional ``sim_time_seconds``
    # kwarg used by the radiation diurnal cycle in
    # :func:`_make_spectral_pe_radiation` -- threading the current
    # scan step's elapsed time lets each rad evaluation see the
    # correct hour-of-day cos(SZA) instead of the static IC time
    # (the latter was the v10 behaviour and froze the diurnal
    # pattern; v11 fix).  Callers whose rad function predates the
    # kwarg still work because they ignore extra kwargs via the
    # ``physics_fn(... , grid_fields=None, sim_time_seconds=0.0)``
    # default in the integration wrapper.
    def _call_rad(s, t_seconds):
        try:
            return rad_physics_fn(
                s, grid, sigma_coord, sim_time_seconds=t_seconds,
            )
        except TypeError:
            # Legacy rad_physics_fn signature without sim_time_seconds.
            return rad_physics_fn(s, grid, sigma_coord)

    init_rad_tendency = _call_rad(initial_state, sim_time_offset_seconds)

    # Cast the (Python-float) offset into the same dtype the gated
    # branch uses, so the radiation diurnal cycle sees a single
    # consistent dtype regardless of how the offset is supplied.
    _offset = jnp.asarray(sim_time_offset_seconds, dtype=jnp.float64)

    # PROGNOSTIC PHYSICS STATE thread — opt-in via markers the classical
    # split-rad factory attaches to its non-rad callable. Without them
    # (learned arms, older callers) the legacy stateless body below runs
    # verbatim. With them, the scan carry gains a PhysicsState and every
    # step feeds the previous step's prognostic physics memory back in:
    # CLUBB's wp2/TKE, Bechtold's conv_prog_profile + stochastic state,
    # the GWD spectrum, the PDF cloud fraction. Before this, the combined
    # wrapper's updated state was DISCARDED every step, so every stateful
    # scheme ran memoryless — CLUBB's turbulence energy sat at its floor
    # forever, i.e. the arm effectively had no boundary-layer mixing
    # (2026-08-17 scene-17 dissection; the same absent-component class as
    # the nine-species fix, which said "carrying turbulence energy across
    # steps on this path is separate work" — this is that work).
    _ps_entry = getattr(physics_fn, "with_phys_state", None)
    _ps_init = getattr(physics_fn, "init_phys_state", None)
    _thread_phys = _ps_entry is not None and _ps_init is not None

    def _post_step(new_state):
        """Shared post-integration chain (filters / positivity / anchor)."""
        if sponge_factor is not None:
            new_state = apply_sponge_filter(new_state, sponge_factor, ms)
        if spectral_filter is not None:
            new_state = apply_spectral_filter_to_state(
                new_state, spectral_filter,
            )
        if tracer_filter is not None and new_state.tracers is not None:
            new_state = new_state._replace(
                tracers=apply_filter_to_tracers(
                    new_state.tracers, tracer_filter, grid,
                )
            )

        # SPECIES POSITIVITY — same fixer as the ungated body; this is the
        # branch the WB and AIMIP classical arms actually run (split rad).
        if new_state.tracers is not None:
            new_state = new_state._replace(
                tracers=positive_tracers(new_state.tracers, sigma_coord)
            )

        # Same anchor as the ungated body above, and this is the branch the
        # AIMIP arms actually take: aimip_era5.yaml sets
        # aimip_rad_update_interval 36, so ``use_rad_gating`` is True whenever
        # a radiation physics_fn is supplied.
        if _target_mass is not None:
            new_state = anchor_lnps_to_mass(grid, new_state, _target_mass)
        return new_state

    def _rad_refresh(state, cached_rad_tendency, step_idx):
        # Refresh rad tendency at the start of every gating window.
        # ``lax.cond`` retains backward-mode differentiability through
        # the rad branch; on skipped steps the cached tensor flows
        # through unchanged.
        should_refresh = (step_idx % rad_update_interval) == 0
        # Cumulative simulated time = optional caller-supplied offset
        # plus per-step contribution from THIS rollout's scan index.
        # Multi-step autoregressive supervision passes the wall time
        # elapsed since the IC so segment k's rad call sees the right
        # solar phase (otherwise every segment starts at 00 UTC and
        # the diurnal cycle is frozen at the IC's time-of-day).
        sim_time_seconds = step_idx.astype(jnp.float64) * dt + _offset
        return jax.lax.cond(
            should_refresh,
            lambda _: _call_rad(state, sim_time_seconds),
            lambda _: cached_rad_tendency,
            operand=None,
        )

    def step_fn_gated(carry, step_idx):
        state, cached_rad_tendency = carry
        new_rad_tendency = _rad_refresh(state, cached_rad_tendency, step_idx)

        def tendency_fn(s):
            non_rad_phys = physics_fn(s, grid, sigma_coord)
            combined_phys = _add_phys_tendencies(non_rad_phys, new_rad_tendency)
            return spectral_pe_tendencies(
                s, grid, sigma_coord, pe_config, combined_phys,
            )

        new_state = _post_step(_integrate(state, tendency_fn))
        return (new_state, new_rad_tendency), None

    def step_fn_gated_stateful(carry, step_idx):
        state, cached_rad_tendency, phys_state = carry
        new_rad_tendency = _rad_refresh(state, cached_rad_tendency, step_idx)

        # Harvest the updated prognostic physics state ONCE per step, on
        # the PRE-STEP state (operator-split convention: all RK stages of
        # this step read the same step-initial physics memory). Stage 1's
        # in-integrator physics evaluation has identical inputs, so XLA
        # may CSE the pair; if not, this costs one extra non-rad physics
        # evaluation per step — correctness over compute here.
        _, phys_state_new = _ps_entry(state, grid, sigma_coord, phys_state)

        def tendency_fn(s):
            non_rad_phys = _ps_entry(s, grid, sigma_coord, phys_state)[0]
            combined_phys = _add_phys_tendencies(non_rad_phys, new_rad_tendency)
            return spectral_pe_tendencies(
                s, grid, sigma_coord, pe_config, combined_phys,
            )

        new_state = _post_step(_integrate(state, tendency_fn))
        return (new_state, new_rad_tendency, phys_state_new), None

    if _thread_phys:
        if phys_state_in is not None:
            phys0 = phys_state_in
        else:
            # Prefer the state-aware seed (shear-equilibrium TKE) over the
            # scheme-floor initializer: a floor seed cannot spin up within
            # a short window (sqrt-production bottleneck; GLM option c).
            _seed = getattr(physics_fn, "seed_phys_state", None)
            if _seed is not None:
                phys0 = _seed(initial_state, grid, sigma_coord)
            else:
                _ncol = int(grid.n_lat) * int(grid.n_lon)
                _nlev = int(jnp.shape(sigma_coord.sigma_full)[0])
                phys0 = _ps_init(_ncol, _nlev)
        step_fn_ckpt = jax.checkpoint(
            step_fn_gated_stateful,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        (final_state, _, final_ps), _ = jax.lax.scan(
            step_fn_ckpt,
            (initial_state, init_rad_tendency, phys0),
            jnp.arange(n_steps),
        )
        if return_phys_state:
            return final_state, final_ps
        return final_state

    if phys_state_in is not None or return_phys_state:
        raise ValueError(
            "spectral_rollout: phys_state_in/return_phys_state need a "
            "physics_fn carrying the with_phys_state/init_phys_state "
            "markers (the classical split-rad factory attaches them); this "
            "physics_fn has none, so the state would be silently ignored.")

    step_fn_ckpt = jax.checkpoint(
        step_fn_gated,
        prevent_cse=True,
        policy=jax.checkpoint_policies.nothing_saveable,
    )

    (final_state, _), _ = jax.lax.scan(
        step_fn_ckpt,
        (initial_state, init_rad_tendency),
        jnp.arange(n_steps),
    )
    return final_state


def spectral_amip_rollout(
    initial_state: SpectralHydrostaticState,
    non_rad_fn,
    rad_fn,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    pe_config: SpectralPEConfig,
    dt: float,
    n_steps: int,
    *,
    sst_col: jnp.ndarray,
    sizing_phys_state,
    day_of_year_base: jnp.ndarray | float = 0.0,
    seconds_offset: jnp.ndarray | float = 0.0,
    rad_update_interval: int = 36,
    sponge_factor: jnp.ndarray | None = None,
    spectral_filter: jnp.ndarray | None = None,
    ghg_vmr: dict | None = None,
    o3_vmr: jnp.ndarray | None = None,
    use_checkpoint: bool = False,
) -> SpectralHydrostaticState:
    """Prescribed-SST AMIP rollout.

    ``use_checkpoint=False`` (default) is inference (no autodiff). Set
    ``use_checkpoint=True`` for the **stability fine-tune**: each step is
    ``jax.checkpoint``-wrapped (nothing-saveable) so reverse-mode AD through a
    multi-day prescribed-SST rollout fits in memory (the trained physics params
    flow via ``non_rad_fn`` / ``rad_fn``).

    Mirrors :func:`spectral_rollout`'s rad-gated path but injects a prescribed
    sea-surface temperature ``sst_col`` (shape ``(ncol,)``) into BOTH surface
    processes:

    * surface-flux / turbulence — via ``phys_state.surface_T_sfc_override``
      (``non_rad_fn`` reads it; this is what anchors near-surface air T to the
      prescribed SST, the essence of the AMIP protocol); and
    * radiation — via the per-step ``forcing['T_sfc']`` dict (``rad_fn``),
      together with the seasonal + diurnal calendar
      (``forcing['day_of_year'/'seconds_of_day']``) advanced from
      ``day_of_year_base`` + ``seconds_offset`` by the scan step index.

    ``sst_col``, ``day_of_year_base`` and ``seconds_offset`` are TRACED args, so
    a single JIT'd segment is reused across every month of a multi-decade run
    (SegmentForcing doctrine — no retrace when the monthly SST / calendar
    changes). ``sizing_phys_state`` is a template :class:`PhysicsState` (correct
    per-scheme carry shapes, zero-valued) built once by the caller with
    ``init_physics_state``; the traced ``sst_col`` is injected as its
    ``surface_T_sfc_override`` here. No gradient checkpointing (inference only),
    so this is markedly cheaper per step than the training rollout.

    ``non_rad_fn`` / ``rad_fn`` are the split-radiation pair returned by
    ``make_aimip_classical_spectral_physics(..., split_rad=True)``; both forward
    the optional ``phys_state`` / ``forcing`` kwargs used here.
    """
    integrator_name = pe_config.time_integrator
    # #817: explicit dispatch_integrator, or the semi-implicit SSP-RK3-SI step
    # (precomputed ONCE, outside the scan) when pe_config.semi_implicit.
    _integrate = _make_spectral_integrator(
        pe_config, grid, sigma_coord, dt, integrator_name,
    )
    ms = grid.ms  # for sponge filter
    tracer_filter = _compute_tracer_filter(grid, pe_config, spectral_filter, dt)

    # Dry-mass anchor target for THIS rollout, captured from its own initial
    # state (see spectral_rollout for the full rationale). ``None`` = off, which
    # keeps every existing AMIP run byte-identical.
    _amip_target_mass = (
        global_dry_mass(grid, initial_state.lnps_hat.data)
        if (pe_config.fix_mass and pe_config.anchor_mass_to_initial)
        else None
    )

    # Inject the (traced) prescribed SST as the surface-temperature anchor.
    # Prescribed SST is NaN over land; map those to the finite no-override
    # sentinel so the PERSISTED physics state stays finite (#911) — land
    # columns then fall back to the model surface T in both resolvers.
    from legoesm.atmosphere.physics.physics_state import NO_SFC_T_OVERRIDE
    # isfinite (not isnan): map NaN AND +/-Inf to the sentinel so the persisted
    # state is strictly finite (codex).
    _sst_override = jnp.where(jnp.isfinite(sst_col), sst_col, NO_SFC_T_OVERRIDE)
    phys_state = sizing_phys_state._replace(surface_T_sfc_override=_sst_override)
    _doy0 = jnp.asarray(day_of_year_base, dtype=jnp.float64)
    _off = jnp.asarray(seconds_offset, dtype=jnp.float64)

    def _forcing_at(step_idx):
        # Elapsed simulated time -> advancing day-of-year (seasonal insolation)
        # + wrapped seconds-of-day (diurnal cycle).
        t = step_idx.astype(jnp.float64) * dt + _off
        fc = {
            "T_sfc": sst_col,
            "day_of_year": _doy0 + t / 86400.0,
            "seconds_of_day": jnp.mod(t, 86400.0),
        }
        # Prescribed transient GHG + ozone (RRTMGP is a physical scheme: it needs
        # the actual historical concentrations to produce the radiative-forcing
        # trend; constant across the segment). Read by the radiation factory as
        # ghg_vmr_override / o3_vmr_override.
        if ghg_vmr is not None:
            fc["ghg_vmr"] = ghg_vmr
        if o3_vmr is not None:
            fc["o3_vmr"] = o3_vmr
        return fc

    def step_fn(carry, step_idx):
        state, cached_rad = carry
        fc = _forcing_at(step_idx)
        should_refresh = (step_idx % rad_update_interval) == 0
        new_rad = jax.lax.cond(
            should_refresh,
            lambda _: rad_fn(state, grid, sigma_coord, forcing=fc),
            lambda _: cached_rad,
            operand=None,
        )

        def tendency_fn(s):
            non_rad = non_rad_fn(
                s, grid, sigma_coord, phys_state=phys_state, forcing=fc,
            )
            combined = _add_phys_tendencies(non_rad, new_rad)
            return spectral_pe_tendencies(
                s, grid, sigma_coord, pe_config, combined,
            )

        new_state = _integrate(state, tendency_fn)
        if sponge_factor is not None:
            new_state = apply_sponge_filter(new_state, sponge_factor, ms)
        if spectral_filter is not None:
            new_state = apply_spectral_filter_to_state(new_state, spectral_filter)
        if tracer_filter is not None and new_state.tracers is not None:
            new_state = new_state._replace(
                tracers=apply_filter_to_tracers(
                    new_state.tracers, tracer_filter, grid,
                )
            )
        # SPECIES POSITIVITY — same fixer as spectral_rollout, same reason
        # (non-monotone transport + filter), on the prescribed-SST lane.
        if new_state.tracers is not None:
            new_state = new_state._replace(
                tracers=positive_tracers(new_state.tracers, sigma_coord)
            )
        # Same dry-mass anchor as spectral_rollout, for the same reason. This is
        # the PRESCRIBED-SST lane (classical AMIP inference and AMIP
        # fine-tuning); leaving it out would make ``fix_mass`` a knob that
        # silently does nothing on exactly the runs where a secular
        # surface-pressure drift compounds longest (codex round 2).
        if _amip_target_mass is not None:
            new_state = anchor_lnps_to_mass(grid, new_state, _amip_target_mass)
        return (new_state, new_rad), None

    init_rad = rad_fn(
        initial_state, grid, sigma_coord, forcing=_forcing_at(jnp.asarray(0)),
    )
    _step = (
        jax.checkpoint(
            step_fn, prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        if use_checkpoint else step_fn
    )
    (final_state, _), _ = jax.lax.scan(
        _step, (initial_state, init_rad), jnp.arange(n_steps),
    )
    return final_state


# =============================================================================
# Loss function
# =============================================================================

def perturb_spectral_ic(state, key, config, sigma_full=None):
    """IC-perturbed ensemble member for the CRPS path.

    Adds Gaussian noise to the prognostic SPECTRAL coefficients (plus the
    grid-space moisture tracer), scaled per variable by
    ``config.ensemble_ic_noise x <var>_resid_scale``.

    HONEST LIMITS of this construct (codex review 2026-07-31) — it is a
    SENSITIVITY probe, not a calibrated analysis-error ensemble:
      * the noise is WHITE in spectral space, so it respects neither
        geostrophic/hydrostatic balance, nor vertical correlation, nor the
        spatial covariance of real analysis error;
      * the vor/div amplitude is applied mode-independently, so ``wind /
        R_earth`` is dimensionally suggestive but NOT a calibrated wind
        perturbation — the vor/div-to-wind operator is degree dependent;
      * a 6 h tendency scale is not an ERA5 analysis-error covariance;
      * the network was trained only on unperturbed analyses, so its response
        to these draws is out of distribution by construction.
    Treat the spread it produces as "sensitivity to an assumed IC covariance",
    never as calibrated forecast uncertainty, and measure the response before
    trusting it (see the response-study protocol in the ensemble dev-note).

    ``vor``/``div`` share the wind scale; ``lnps`` is perturbed in LOG space by
    ``ps_resid_scale / p_ref`` so the multiplicative effect on p_s matches the
    intended Pa-scale spread.
    """
    amp = float(getattr(config, "ensemble_ic_noise", 0.0) or 0.0)
    if amp <= 0.0:
        return state
    # One key PER FIELD and per real/imaginary part: reusing a key across the
    # real and imaginary draws made the complex noise perfectly correlated
    # (arg fixed at 45 deg) instead of circular (codex review 2026-07-31).
    k_vor, k_div, k_T, k_ps, k_q = jax.random.split(key, 5)
    wind = amp * float(config.wind_resid_scale)
    temp = amp * float(config.T_resid_scale)
    # ln(p_s) perturbation: dp/p with p ~ p_ref.
    lnps = amp * float(config.ps_resid_scale) / float(constants.p_ref)

    def _kick(field, k, scale):
        d = field.data
        kr, ki = jax.random.split(k)
        noise = (jax.random.normal(kr, d.shape, dtype=jnp.float64)
                 + 1j * jax.random.normal(ki, d.shape, dtype=jnp.float64))
        return field.replace(data=d + scale * noise.astype(d.dtype))

    out = state._replace(
        vor_hat=_kick(state.vor_hat, k_vor, wind / constants.R_earth),
        div_hat=_kick(state.div_hat, k_div, wind / constants.R_earth),
        T_hat=_kick(state.T_hat, k_T, temp),
        lnps_hat=_kick(state.lnps_hat, k_ps, lnps),
    )
    # MOISTURE. q_v is a GRID-space tracer, not one of the spectral prognostics
    # above, so it was silently left unperturbed while both this docstring and
    # LossConfig.ensemble_ic_noise claimed q was scaled (codex review
    # 2026-07-31). Perturb it in grid space and clamp to >= 0, since a negative
    # q corrupts virtual temperature and hence the whole column.
    if out.tracers:
        qs = float(config.q_resid_scale) * amp
        new_tracers = dict(out.tracers)
        for name, tr in out.tracers.items():
            if not name.startswith("q"):
                continue
            data = tr.data if hasattr(tr, "data") else tr
            kicked = jnp.maximum(
                data + qs * jax.random.normal(k_q, data.shape, data.dtype), 0.0)
            new_tracers[name] = (
                tr.replace(data=kicked) if hasattr(tr, "replace") else kicked)
        out = out._replace(tracers=new_tracers)
    return out


def ensemble_afcrps_loss(
    member_fields, target_carry, grid, config,
):
    """Area-weighted almost-fair CRPS over an ensemble of GRID-space forecasts.

    ``member_fields`` is the stacked ``(M, n_lat, n_lon, nlev)`` prediction for
    ONE variable; ``target_carry`` the matching truth. Delegates the estimator
    to :func:`legoesm.ml.loss.area_weighted_afcrps` — the finite-ensemble
    (almost-fair) correction is defined once, there, not re-derived here.

    M == 1 degenerates to the MAE, which is the documented deterministic limit;
    that is why ``ensemble_ic_noise`` must be > 0 for the term to mean anything.
    """
    from legoesm.ml.loss import area_weighted_afcrps

    return area_weighted_afcrps(
        member_fields, target_carry, grid.weights.astype(member_fields.dtype),
        alpha=float(getattr(config, "afcrps_alpha", 0.95)),
    )


def _persistence_normalized_loss_components(
    pred_state: SpectralHydrostaticState,
    ic_state: SpectralHydrostaticState,
    target_carry,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    sigma_full: jnp.ndarray,
    config: LossConfig = LossConfig(),
):
    """``L(pred, target) / L(x0, target)`` — a skill-score loss.

    Reuses :func:`_spectral_state_loss_components` for BOTH numerator and
    denominator, so there is exactly one definition of the loss numerics; the
    denominator is that same function applied with the INITIAL state as the
    "forecast". See ``LossConfig.persistence_normalize`` for why dividing (not
    subtracting) is the operation that matters.

    The denominator is a function of DATA ONLY (``ic_state``, ``target_carry``),
    so it contributes no gradient; ``stop_gradient`` makes that explicit and
    keeps the term safe if a future caller passes a model-dependent reference.

    A value of 1.0 means "no better than persistence"; < 1 is skill.

    Cost: one extra loss evaluation per sample — a spectral->grid diagnosis plus
    reductions on the IC. No extra rollout, so it is cheap next to a multi-step
    trajectory.
    """
    num, comps = _spectral_state_loss_components(
        pred_state, target_carry, grid, sigma_coord, sigma_full, config,
    )
    den, _ = _spectral_state_loss_components(
        ic_state, target_carry, grid, sigma_coord, sigma_full, config,
    )
    # A zero or negative floor re-opens the divide-by-zero this guard exists to
    # close, and an unbounded 1/den lets a nearly-stationary sample dominate the
    # gradient. Validate on the STATIC config value (codex review 2026-07-30).
    _floor = float(config.persistence_floor)
    if not _floor > 0.0:
        raise ValueError(
            "LossConfig.persistence_floor must be > 0 (it bounds 1/denominator "
            f"for a sample whose persistence error is ~0); got {_floor!r}.")
    den = jax.lax.stop_gradient(den) + jnp.float32(_floor)
    inv = jnp.float32(1.0) / den
    return num * inv, {k: v * inv for k, v in comps.items()}


def _resolve_crps_finetune(config, n_epochs_total, loss_config):
    """Validate + resolve the U-Cast stage-2 knobs. Returns ``(epochs, M)``.

    Called at ``train_sfno_full_spectral`` ENTRY (before the multi-hour host
    data load) as well as inside the loop, so an invalid stage-2 config fails
    in seconds rather than after ~4 h of ERA5 loading (codex review
    2026-08-02).  Every refusal here guards a SILENT no-op, not a crash.
    """
    epochs = getattr(config, "crps_finetune_epochs", 0)
    epochs = 0 if epochs is None else int(epochs)
    # NOT ``or 2``: that turns an explicit 0 into 2 and bypasses the M >= 2
    # guard below, silently training a 2-member ensemble the user disabled.
    m = getattr(config, "crps_ensemble_size", None)
    m = 2 if m is None else int(m)
    if epochs <= 0:
        return 0, m
    if float(getattr(config, "sfno_dropout", 0.0)) <= 0.0:
        raise ValueError(
            "crps_finetune_epochs > 0 requires sfno_dropout > 0: MC-Dropout "
            "is the ensemble source, and at p=0 every member is the same "
            "forecast, so the almost-fair CRPS degenerates to the MAE and the "
            "fine-tune is a silent no-op."
        )
    if m < 2:
        raise ValueError(
            f"crps_ensemble_size must be >= 2, got {m} (a 1-member 'ensemble' "
            "makes the CRPS the MAE)."
        )
    if epochs >= int(n_epochs_total):
        raise ValueError(
            f"crps_finetune_epochs={epochs} >= n_epochs={n_epochs_total}: "
            "stage 2 is a SHORT fine-tune of a deterministically pre-trained "
            "model (U-Cast: 8 epochs against 100), not the whole run."
        )
    # The stage-2 objective is built ONLY from the w_crps_* family. With all
    # of them zero the loss is a constant 0.0, the gradient is exactly zero,
    # and the fine-tune runs to completion having changed nothing.
    if not any(float(getattr(loss_config, f"w_crps_{v}", 0.0)) > 0.0
               for v in ("T", "u", "v", "q", "ps")):
        raise ValueError(
            "crps_finetune_epochs > 0 but every w_crps_* weight is 0, so the "
            "stage-2 CRPS objective is identically zero and the fine-tune "
            "would train on a null gradient. Use a preset that sets them "
            "(config/wb/loss_presets/ucast.yaml) or set them explicitly."
        )
    return epochs, m


def ensemble_crps_loss_components(
    member_states,
    target_carry,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    sigma_full: jnp.ndarray,
    config: LossConfig = LossConfig(),
):
    """Almost-fair CRPS over an ENSEMBLE of spectral states — U-Cast stage 2.

    ``member_states`` is a spectral state whose leaves carry a leading
    ``(M,)`` member axis (what ``jax.vmap`` over per-member dropout keys
    produces).  Returns ``(total, components)`` with the SAME component keys
    as :func:`_spectral_state_loss_components` so the training loop's
    accumulators and logging are unchanged; the CRPS lands in ``crps`` and
    ``mse``/``bias``/``spec_crps`` stay zero.

    This is the objective the U-Cast recipe (arXiv:2604.09041) fine-tunes on
    after the deterministic stage: "short probabilistic fine-tuning on the
    Continuous Ranked Probability Score (CRPS) using Monte Carlo Dropout",
    8 epochs against 100, with a training ensemble of M=2 (they report M=4
    gives "only marginal CRPS gains").

    DELIBERATE REUSE, so train and eval cannot drift:
      * the pointwise estimator is ``legoesm.ml.loss.almost_fair_crps`` — the
        same function ``evaluations.wb_forecast.score_ensemble_forecast``
        scores with;
      * the area/level weighting and the per-variable scale selection are
        COPIED from ``_spectral_state_loss_components`` (Gaussian-latitude
        quadrature x level weights; residual > full-field > none), so a CRPS
        number here is commensurate with the MAE number the deterministic
        stage reports.

    Per-variable weights come from the ``w_crps_*`` family (the M=1 limit of
    this term IS that MAE), so the ``ucast`` preset's weights carry over to
    stage 2 unchanged.
    """
    from legoesm.ml.loss import almost_fair_crps

    alpha = float(getattr(config, "afcrps_alpha", 0.95))
    lev_w = level_weights(sigma_full, config=config)
    lat_w = grid.weights.astype(jnp.float32)
    lat_w_sum = jnp.sum(lat_w)

    def _area_weighted_mean_3d(field):
        zonal = jnp.mean(field, axis=1)
        weighted_lat = jnp.sum(zonal * lat_w[:, None], axis=0) / lat_w_sum
        return jnp.sum(weighted_lat * lev_w) / lev_w.shape[0]

    def _area_weighted_mean_2d(field):
        zonal = jnp.mean(field, axis=1)
        return jnp.sum(zonal * lat_w) / lat_w_sum

    if getattr(config, "residual_normalize", False):
        T_scale = config.T_resid_scale
        wind_scale = config.wind_resid_scale
        q_scale = config.q_resid_scale
        ps_scale = config.ps_resid_scale
    elif config.normalize_by_scale:
        T_scale = config.T_scale
        wind_scale = config.wind_scale
        q_scale = config.q_scale
        ps_scale = config.ps_scale
    else:
        T_scale = wind_scale = q_scale = ps_scale = 1.0

    # Member states -> member grid fields.  vmap keeps the (M, ...) axis
    # leading, which is exactly what almost_fair_crps expects.
    member_fields = jax.vmap(
        lambda s: spectral_pe_to_grid(s, grid, sigma_coord)
    )(member_states)

    crps_loss = jnp.float32(0.0)
    for name, weight_attr, scale, mean_fn in (
        ("T", "w_crps_T", T_scale, _area_weighted_mean_3d),
        ("u", "w_crps_u", wind_scale, _area_weighted_mean_3d),
        ("v", "w_crps_v", wind_scale, _area_weighted_mean_3d),
        ("p_s", "w_crps_ps", ps_scale, _area_weighted_mean_2d),
    ):
        w = float(getattr(config, weight_attr, 0.0))
        if w <= 0.0:
            continue
        target = getattr(target_carry, "p_s" if name == "p_s" else name)
        field = almost_fair_crps(
            member_fields[name].astype(jnp.float32),
            target.astype(jnp.float32), alpha=alpha)
        crps_loss = crps_loss + w * mean_fn(field) / scale

    _w_q = float(getattr(config, "w_crps_q", 0.0))
    if _w_q > 0.0 and member_states.tracers is not None \
            and "q_v" in member_states.tracers:
        _qv = member_states.tracers["q_v"]
        qv = (_qv.data if hasattr(_qv, "data") else _qv).astype(jnp.float32)
        field = almost_fair_crps(
            qv, target_carry.q_v.astype(jnp.float32), alpha=alpha)
        crps_loss = crps_loss + _w_q * _area_weighted_mean_3d(field) / q_scale

    zero = jnp.float32(0.0)
    return crps_loss, {
        "mse": zero, "bias": zero, "crps": crps_loss, "spec_crps": zero,
    }


def _spectral_state_loss_components(
    pred_state: SpectralHydrostaticState,
    target_carry,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    sigma_full: jnp.ndarray,
    config: LossConfig = LossConfig(),
):
    """Compute ``spectral_state_vs_carry_loss`` + per-family component breakdown.

    The four returned families are mutually exclusive and sum to the
    total loss, so the per-epoch diagnostic remains exact rather than
    a recomputation.  Each family already includes the user-supplied
    ``w_*`` weights and the per-variable scale normalisation:

    - ``mse``      : Σ_v w_v · <(pred - target)²>_area / scale²
    - ``bias``     : Σ_v w_bias_v · (area-weighted mean error)² / scale²
    - ``crps``     : Σ_v w_crps_v · <|pred - target|>_area / scale
                     (grid-space M=1 fair-CRPS = MAE)

    ``<·>_area`` is the Gaussian-latitude + level weighted mean (the same
    weighting as the bias term); a plain ``jnp.mean`` would over-weight the
    poles.
    - ``spec_crps``: Σ_v w_spec_crps_v · <|SH(pred - target)|> / scale
                     (spectral M=1 fair-CRPS, requires JAX_ENABLE_X64=True)

    Returns
    -------
    total_loss : scalar
        Sum of the four component scalars (= legacy
        ``spectral_state_vs_carry_loss`` output, ignoring spec_crps
        when its weights are zero).
    components : dict[str, scalar]
        Keys ``mse``, ``bias``, ``crps``, ``spec_crps``; each is a JAX
        scalar suitable for ``has_aux=True`` accumulation.
    """
    lev_w = level_weights(sigma_full, config=config)

    # Spectral -> grid-space
    fields = spectral_pe_to_grid(pred_state, grid, sigma_coord)

    # Component accumulators.  Use float32 to match the legacy ``loss``
    # dtype; the bias/CRPS terms inherit this through arithmetic.
    mse_loss = jnp.float32(0.0)
    bias_loss = jnp.float32(0.0)
    crps_loss = jnp.float32(0.0)
    spec_crps_loss = jnp.float32(0.0)

    # Per-variable scale denominators — same convention as carry_mse
    # (iter-69 fix carried over to the spectral path so identical
    # LossConfigs behave consistently across spectral and grid losses).
    if getattr(config, "residual_normalize", False):
        # ACE2-style residual normalization: scale by the std of the 6 h
        # field CHANGE (tendency), not the full-field std — weights the
        # predictable tendency. Used by the ACE2-loss preset.
        # floor guards a misconfigured 0 residual scale (codex: no zero-div).
        T_norm = max(config.T_resid_scale ** 2, 1e-30)
        wind_norm = max(config.wind_resid_scale ** 2, 1e-30)
        q_norm = max(config.q_resid_scale ** 2, 1e-30)
        ps_norm = max(config.ps_resid_scale ** 2, 1e-30)
    elif config.normalize_by_scale:
        T_norm = config.T_scale ** 2
        wind_norm = config.wind_scale ** 2
        q_norm = config.q_scale ** 2
        ps_norm = config.ps_scale ** 2
    else:
        T_norm = wind_norm = q_norm = ps_norm = 1.0

    # Latitude-weighted mean over the Gaussian grid, used by the bias
    # penalty AND the MSE / CRPS terms below.  ``grid.weights`` are
    # Gaussian-quadrature latitude weights; combined with uniform
    # longitude weighting this gives an area-weighted global mean. Using a
    # plain ``jnp.mean`` for the MSE/CRPS over-weights the poles (the
    # Gaussian rows shrink toward the pole) and contradicts the bias term.
    lat_w = grid.weights.astype(jnp.float32)
    lat_w_sum = jnp.sum(lat_w)

    def _area_weighted_mean_3d(field):
        # (n_lat, n_lon, nlev) -> level-weighted scalar bias.
        # Average over lon, weight by Gaussian quadrature in lat,
        # then sum over level (already level-weighted via lev_w).
        zonal = jnp.mean(field, axis=1)        # (n_lat, nlev)
        weighted_lat = jnp.sum(zonal * lat_w[:, None], axis=0) / lat_w_sum
        return jnp.sum(weighted_lat * lev_w) / lev_w.shape[0]

    def _area_weighted_mean_2d(field):
        zonal = jnp.mean(field, axis=1)        # (n_lat,)
        return jnp.sum(zonal * lat_w) / lat_w_sum

    def _spec_mae_3d(field_grid):
        # Deterministic (M=1) fair-CRPS in spectral space.  SH analysis
        # requires float64 to match the dycore's spectral convention;
        # the |.| is taken on the complex coefficient and averaged
        # across all (n_sh, nlev) entries.  Cast back to float32 so it
        # composes with the float32 ``loss`` accumulator.
        coeffs = sh_analysis_3d(grid, field_grid.astype(jnp.float64))
        return jnp.mean(jnp.abs(coeffs)).astype(jnp.float32)

    # Per-variable bias contributions, recorded so the logged aggregate is
    # attributable (the classical arm's stuck bias=5.6 could not be assigned
    # to a variable from the aggregate alone — codex audit 2026-08-10).
    bias_terms = {}

    # Temperature: (n_lat, n_lon, nlev)
    dT = fields['T'].astype(jnp.float32) - target_carry.T
    mse_loss = mse_loss + config.w_T * _area_weighted_mean_3d(dT ** 2) / T_norm
    if config.w_bias_T > 0.0:
        bias_T = _area_weighted_mean_3d(dT)
        bias_terms['bias_T'] = config.w_bias_T * bias_T ** 2 / T_norm
        bias_loss = bias_loss + bias_terms['bias_T']

    # Winds: (n_lat, n_lon, nlev)
    du = fields['u'].astype(jnp.float32) - target_carry.u
    dv = fields['v'].astype(jnp.float32) - target_carry.v
    mse_loss = mse_loss + config.w_u * _area_weighted_mean_3d(du ** 2) / wind_norm
    mse_loss = mse_loss + config.w_v * _area_weighted_mean_3d(dv ** 2) / wind_norm
    if config.w_bias_u > 0.0:
        bias_u = _area_weighted_mean_3d(du)
        bias_terms['bias_u'] = config.w_bias_u * bias_u ** 2 / wind_norm
        bias_loss = bias_loss + bias_terms['bias_u']
    if config.w_bias_v > 0.0:
        bias_v = _area_weighted_mean_3d(dv)
        bias_terms['bias_v'] = config.w_bias_v * bias_v ** 2 / wind_norm
        bias_loss = bias_loss + bias_terms['bias_v']

    # Grid-space CRPS contribution (deterministic = MAE).  The fair-CRPS
    # for a single-member forecast (M=1) reduces to area-weighted |pred -
    # target|; the M>=2 fair-CRPS would subtract the ensemble
    # self-spread term ``(1/(2M(M-1))) Σ |x_m - x_m'|``, but we don't
    # have an ensemble here.  We normalise by the variable scale
    # (NOT scale^2) because |error| has units of the variable, not
    # variance.  Weights ``w_crps_{T,u,v,ps}`` default to 0; setting
    # them non-zero combines an MAE + MSE objective in the NeuralGCM
    # style.
    # CRPS scales honor residual_normalize too (codex: else the MAE terms
    # would keep full-field scales while the MSE switched to residual —
    # internally inconsistent if both are active).
    if getattr(config, "residual_normalize", False):
        T_scale = config.T_resid_scale
        wind_scale = config.wind_resid_scale
        ps_scale = config.ps_resid_scale
    elif config.normalize_by_scale:
        T_scale = config.T_scale
        wind_scale = config.wind_scale
        ps_scale = config.ps_scale
    else:
        T_scale = wind_scale = ps_scale = 1.0
    if config.w_crps_T > 0.0:
        crps_loss = crps_loss + config.w_crps_T * _area_weighted_mean_3d(jnp.abs(dT)) / T_scale
    if config.w_crps_u > 0.0:
        crps_loss = crps_loss + config.w_crps_u * _area_weighted_mean_3d(jnp.abs(du)) / wind_scale
    if config.w_crps_v > 0.0:
        crps_loss = crps_loss + config.w_crps_v * _area_weighted_mean_3d(jnp.abs(dv)) / wind_scale

    # Spectral-space CRPS (M=1 = MAE of coefficient differences).  Same
    # normalisation convention as the grid CRPS: scale (not scale²).
    # Skipped silently when ``JAX_ENABLE_X64`` is off — the
    # ``sh_analysis_3d`` path requires float64.  Users opting into
    # spec-CRPS without x64 see a clear failure inside the SH kernel,
    # which is preferable to silently dropping the term.
    if config.w_spec_crps_T > 0.0:
        spec_crps_loss = spec_crps_loss + (
            config.w_spec_crps_T * _spec_mae_3d(dT) / T_scale
        )
    if config.w_spec_crps_u > 0.0:
        spec_crps_loss = spec_crps_loss + (
            config.w_spec_crps_u * _spec_mae_3d(du) / wind_scale
        )
    if config.w_spec_crps_v > 0.0:
        spec_crps_loss = spec_crps_loss + (
            config.w_spec_crps_v * _spec_mae_3d(dv) / wind_scale
        )

    # Surface pressure: (n_lat, n_lon)
    dp = fields['p_s'].astype(jnp.float32) - target_carry.p_s
    mse_loss = mse_loss + config.w_ps * _area_weighted_mean_2d(dp ** 2) / ps_norm
    if config.w_crps_ps > 0.0:
        crps_loss = crps_loss + config.w_crps_ps * _area_weighted_mean_2d(jnp.abs(dp)) / ps_scale
    if config.w_bias_ps > 0.0:
        bias_ps = _area_weighted_mean_2d(dp)
        bias_terms['bias_ps'] = config.w_bias_ps * bias_ps ** 2 / ps_norm
        bias_loss = bias_loss + bias_terms['bias_ps']

    # Specific humidity (q_v): contribute to the loss only when the
    # predicted state actually carries a ``q_v`` tracer (i.e., the
    # rollout was set up via ``carry_to_spectral_state(..., include_
    # tracers=True)``).  Skipping silently when missing keeps the
    # function backward-compatible with dry training pipelines.
    if (
        pred_state.tracers is not None
        and "q_v" in pred_state.tracers
    ):
        _qv_raw = pred_state.tracers["q_v"]
        qv_grid = (
            _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        ).astype(jnp.float32)
        dq = qv_grid - target_carry.q_v
        mse_loss = mse_loss + config.w_q * _area_weighted_mean_3d(dq ** 2) / q_norm
        # CRPS / MAE term for q (M=1 fair-CRPS limit).  The tails of
        # the q error distribution are wider than for T or wind --
        # convective + microphysical noise concentrates errors in
        # extreme columns -- so the linear (MAE) penalty is more
        # appropriate than squared error for this variable.  Same
        # area-weighting convention as the MSE term so the two
        # contributions live in commensurate units.
        # Same scale SELECTION as T/u/v/ps above (residual > full-field >
        # none). This previously ignored ``residual_normalize`` and fell back
        # to 1.0, which under a residual-normalized MAE preset suppressed the
        # q term by ~1/q_resid_scale = 2000x — i.e. humidity was effectively
        # unsupervised while the config advertised an equal weight.
        if getattr(config, "residual_normalize", False):
            q_scale = config.q_resid_scale
        elif config.normalize_by_scale:
            q_scale = config.q_scale
        else:
            q_scale = 1.0
        if getattr(config, "w_crps_q", 0.0) > 0.0:
            crps_loss = crps_loss + config.w_crps_q * _area_weighted_mean_3d(jnp.abs(dq)) / q_scale
        # Spectral CRPS for q.  Penalises errors in the spectral
        # pattern of humidity that grid-MAE underweights when the
        # error is concentrated at small scales (convective noise).
        if getattr(config, "w_spec_crps_q", 0.0) > 0.0:
            spec_crps_loss = spec_crps_loss + (
                config.w_spec_crps_q * _spec_mae_3d(dq) / q_scale
            )

    total = mse_loss + bias_loss + crps_loss + spec_crps_loss
    return total, {
        "mse": mse_loss,
        "bias": bias_loss,
        "crps": crps_loss,
        "spec_crps": spec_crps_loss,
        **bias_terms,
    }


def spectral_state_vs_carry_loss(
    pred_state: SpectralHydrostaticState,
    target_carry,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    sigma_full: jnp.ndarray,
    config: LossConfig = LossConfig(),
) -> jnp.ndarray:
    """Scalar wrapper around :func:`_spectral_state_loss_components`.

    Kept for backward compatibility with eval scripts and tests that
    only need the total loss.  Training paths that want per-family
    diagnostics should call ``_spectral_state_loss_components``
    directly and use ``has_aux=True`` in
    ``eqx.filter_value_and_grad``.

    Parameters
    ----------
    pred_state : SpectralHydrostaticState
        Predicted state after rollout.
    target_carry : SegmentCarry
        Target ERA5 state on the Gaussian grid.
    grid : GaussianGrid
    sigma_coord : SigmaCoordinate
    sigma_full : (nlev,)
        Sigma levels for pressure weighting.
    config : LossConfig

    Returns
    -------
    scalar loss
    """
    total, _ = _spectral_state_loss_components(
        pred_state, target_carry, grid, sigma_coord, sigma_full, config,
    )
    return total


# =============================================================================
# Data loading
# =============================================================================

def _training_year_range(windows, config) -> tuple[int, int]:
    """``(min_year, max_year + 1)`` spanned by the windows THIS call loads.

    Prefer the explicit ``windows`` argument (the data ``load_training_data``
    actually reads this call) over ``config.windows`` — a caller can pass a
    ``windows`` set that differs from ``config.windows``, and the cache must
    cover what is read, not what is configured.  Falls back to
    ``start_year`` + ``n_train_days`` when neither is given.

    The ``+1`` on the max year covers TARGET snapshots that spill into the
    following year (the rollout lead / ``n_days`` stride).  The cache store is
    scoped by this range, so each distinct span is its own store and
    ``ensure_local_cache``'s existence check is never stale across spans.
    """
    wins = windows or getattr(config, "windows", None) or ()
    if wins:
        years = [int(w[0]) for w in wins]
        return (min(years), max(years) + 1)
    start = int(getattr(config, "start_year", 2015) or 2015)
    n_days = int(getattr(config, "n_train_days", 0) or 0)
    return (start, start + max(0, (n_days - 1) // 365) + 1)


def host_build_device(warn_label: str = "host-resident load"):
    """The JAX CPU device for host-side array construction, or ``None``.

    Shared preflight for every host-build site (#985 prefetch, #1155
    non-chunked loads): returns ``jax.devices("cpu")[0]`` when the CPU
    backend exists, else warns and returns ``None`` so the caller falls back
    to building on the compute device (roomy configs are unchanged;
    memory-tight ones need ``JAX_PLATFORMS=cuda,cpu``).
    """
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        logger.warning(
            f"{warn_label}: the JAX CPU backend is unavailable "
            "(JAX_PLATFORMS?), so arrays will be built on the compute device "
            "(GPU-OOM risk for dataset-sized loads). Add 'cpu' to "
            "JAX_PLATFORMS to enable host residency."
        )
        return None


def stage_sample(tree, device=None):
    """Public per-sample staging: move a sample pytree's ARRAY leaves onto
    the compute device (default: this process's first local device).

    Consumer-side half of the host-resident-dataset contract (#1155).
    Placement semantics (measured, jax 0.10): arrays built under
    ``jax.default_device(cpu)`` are UNCOMMITTED — a GPU-jitted step would
    transfer them per call anyway, so staging is not a crash guard. It IS
    the explicit-placement hygiene: it makes each sample's single H2D copy
    visible at the call site, keeps behavior identical if the dataset ever
    arrives COMMITTED (e.g. an explicit ``device_put(cpu)`` producer), and
    bounds peak device footprint to one sample. Wraps the #985
    ``_stage_tree``. Uses ``jax.local_devices()`` (not ``jax.devices()``)
    so a multi-process runtime never targets a non-addressable device.
    """
    if device is None:
        device = jax.local_devices()[0]
    return _stage_tree(tree, device)


def load_training_data(
    config: NeuralGCMSpectralConfig,
    grid: GaussianGrid,
    sigma: SigmaCoordinate,
    cache_dir: str = "data/era5_cache",
    windows: list | None = None,
    host_resident: bool = False,
):
    """Load ERA5 daily IC/target pairs and convert to spectral states.

    Opens the ERA5 Zarr store once and batch-loads all needed time slices,
    avoiding per-slice GCS connection overhead.

    ``host_resident=True`` builds every carry under the JAX CPU backend
    (``jax.default_device``) so a dataset-sized load never materialises on
    the GPU — the #1155 fix for the non-chunked trainers (they load ALL
    pairs up front; at T106 all-years that is ~130 GB device-resident, an
    unconditional OOM). Consumers pair it with per-sample staging
    (``stage_sample`` / the training loops' ``host_staged=True``) — see
    ``stage_sample`` for the honest placement semantics (explicit-placement
    hygiene, not a crash guard: default_device-built arrays are
    UNCOMMITTED). Trade-offs, accepted and measured against the
    alternative: the per-snapshot spectral transforms run on CPU during the
    load (the load is GCS-dominated in practice — job 6758505 measured
    ~4.4 s/snapshot pure streaming), and staged consumers re-transfer each
    sample per epoch (~seconds/epoch of H2D vs. hour-scale epochs). A CPU
    backend is REQUIRED: with no viable success path for a dataset-sized
    device build, an absent backend raises immediately instead of warning
    and then OOMing hours into the load (fail-fast).

    Parameters
    ----------
    config : NeuralGCMSpectralConfig
        Used for ``n_train_days`` / ``start_year`` when ``windows`` is None.
    grid : GaussianGrid
    sigma : SigmaCoordinate
    cache_dir : str
        Local cache directory for ERA5 Zarr data.
    windows : list[tuple[int, int, int]] | None
        Optional list of ``(year, day_offset, n_days)`` tuples.  When
        provided, each window is sampled CONTIGUOUSLY (no cross-window
        IC/target pairs, so no temporal leakage between windows) and
        the results are concatenated.  This is the AIMIP-style
        multi-year / multi-season sampling protocol.  When ``None``
        the legacy single-window behaviour applies.
    host_resident : bool
        Build every array under the JAX CPU backend (see the prose above:
        the #1155 fix for non-chunked full-dataset loads). Requires the
        CPU backend (raises otherwise). Consumers pair it with per-sample
        staging (``stage_sample`` / ``host_staged=True``).

    Returns
    -------
    ic_states : list[SpectralHydrostaticState]
        Initial conditions in spectral space.
    target_carries : list[SegmentCarry]
        Targets in grid space (for loss computation).
    ic_times : list[np.datetime64 | None]
        Wall-clock IC time per sample (None if the zarr time
        coordinate was unreadable) — consumed by the prescribed
        surface-forcing path.
    """
    if host_resident:
        _host = host_build_device("load_training_data(host_resident=True)")
        if _host is None:
            # No CPU backend = no viable success path for a dataset-sized
            # load (the device build is the #1155 OOM by construction, hit
            # only AFTER hours of streaming). Fail fast with the remedy.
            raise RuntimeError(
                "load_training_data(host_resident=True) requires the JAX "
                "CPU backend (add 'cpu' to JAX_PLATFORMS, e.g. "
                "JAX_PLATFORMS=cuda,cpu): a full-dataset build on the "
                "compute device OOMs at scale (#1155)."
            )
        # NOTE for signature growth: the recursive call forwards EVERY
        # kwarg explicitly — a new load_training_data parameter must be
        # added here too, or the host-resident path silently uses its
        # default (tested: the recursion pins host_resident=False).
        with jax.default_device(_host):
            return load_training_data(
                config, grid, sigma, cache_dir=cache_dir,
                windows=windows, host_resident=False,
            )

    import numpy as np
    from legoesm.training.era5_to_state import (
        open_era5_zarr, resolve_var, ERA5Slice, ensure_local_cache,
    )
    era5_config = TrainingERA5Config(dt_hours=6)

    # Open store once.  Wire the (previously DEAD) ``cache_dir`` to the local
    # ERA5 zarr cache: without it every epoch re-fetched the SAME windows from
    # GCS (#895, ~50 h/run wasted).  Scope the one-time download to the FULL
    # training span (``config.windows``, NOT the per-chunk ``windows`` arg) so
    # the single shared cache store is not built for one chunk's years and then
    # read stale for the others.  ensure_local_cache is idempotent (skips if the
    # store already exists), so only the first call pays the download.
    store = era5_config.zarr_store
    import os
    # WINDOW-scoped cache (#985): supersedes the year-span cache for the AIMIP
    # T106 workload — materialise ONLY the ~2,880 snapshots the training
    # windows touch (~0.5 TB) instead of full 6-hourly year spans (~10 TB), and
    # read them back BY TIMESTAMP.  The scoped store is built AFTER the absolute
    # time indices are known (below), so the calendar is read from the REMOTE
    # store here; ``LEGOESM_ERA5_WINDOW_CACHE=1`` opts in.
    _window_cache = bool(
        cache_dir and os.environ.get("LEGOESM_ERA5_WINDOW_CACHE")
    )
    # OPT-IN year-span cache (default OFF -> byte-identical to the GCS-every-
    # epoch path): LEGOESM_ERA5_LOCAL_CACHE=1 materialises each YEAR span once
    # and reads it locally thereafter (#895).  Superseded by the window cache
    # above when both are set.
    if not _window_cache and cache_dir and os.environ.get(
        "LEGOESM_ERA5_LOCAL_CACHE"
    ):
        # Year-SCOPE the cache store.  ensure_local_cache now keys on a
        # completeness marker (not bare path existence), so a walltime-killed
        # build is rebuilt, not read as fill-value NaNs (#942/#985); the
        # per-span subdir still keeps distinct spans from colliding.
        _yrs = _training_year_range(windows, config)
        _scoped = os.path.join(os.fspath(cache_dir), f"y{_yrs[0]}_{_yrs[1]}")
        store = str(ensure_local_cache(era5_config, _scoped, years=_yrs))
    ds = open_era5_zarr(store)

    # Resolve start offset from config.start_year (defaults to 2015).
    # The WeatherBench2 ERA5 zarr starts at 1959-01-01 00:00 UTC with
    # 6-hour spacing, so day 0 of year Y maps to time-index
    # round((Y - 1959) * 365.25 * 4).  Snap to the nearest available
    # time after that target to tolerate leap-day drift.
    def _year_to_idx(year: int) -> int:
        try:
            year_times = np.array(ds.time.values, dtype="datetime64[ns]")
            target_t = np.datetime64(f"{int(year):04d}-01-01")
            return int(np.searchsorted(year_times, target_t))
        except Exception as exc:
            heuristic = max(0, int(round((int(year) - 1959) * 365.25 * 4)))
            logger.warning(
                f"load_training_data: ds.time access failed ({exc!r}); "
                f"falling back to heuristic start index {heuristic} for "
                f"year {year}.  Verify the ERA5 zarr time coordinate is "
                f"accessible if downstream metrics look off."
            )
            return heuristic

    rollout_days = int(getattr(config, "rollout_days", 1) or 1)
    if rollout_days < 1:
        raise ValueError(
            f"config.rollout_days must be >= 1, got {rollout_days!r}."
        )
    # v12: ``rollout_hours`` lets the supervision horizon drop below
    # one day (AIMIP / NeuralGCM convention is 6 h).  Fallback to the
    # legacy daily horizon when not set.  ERA5 cadence is fixed at
    # ``era5_config.dt_hours`` (6 h by default); ``rollout_hours``
    # must be a multiple of that, and so must the per-IC stride
    # ``ic_stride_units`` (which advances the IC by one ERA5 snapshot
    # = 6 h between consecutive pairs).
    rollout_hours_cfg = int(getattr(config, "rollout_hours", 0) or 0)
    if rollout_hours_cfg > 0:
        rollout_hours = rollout_hours_cfg
    else:
        rollout_hours = rollout_days * 24
    era5_dt_hours = int(era5_config.dt_hours)
    if rollout_hours % era5_dt_hours != 0:
        raise ValueError(
            f"rollout_hours={rollout_hours} must be a multiple of "
            f"era5_dt_hours={era5_dt_hours}."
        )
    rollout_stride = rollout_hours // era5_dt_hours          # snapshot units between IC and target
    snapshots_per_day = 24 // era5_dt_hours                  # 4 at 6h cadence

    # GenCast-style multi-step autoregressive supervision: when
    # ``loss_config.multi_step_hours`` is non-empty, load one target
    # per lead in addition to the legacy ``rollout_stride`` target.  The
    # loss is then summed (optionally weighted) over all leads after
    # chained 6-hour rollouts.
    loss_cfg = getattr(config, "loss_config", None)
    multi_step_hours = tuple(
        int(h) for h in (getattr(loss_cfg, "multi_step_hours", ()) or ())
    )
    if multi_step_hours:
        max_lead_hours = max(multi_step_hours)
        if any(h % era5_dt_hours != 0 for h in multi_step_hours):
            raise ValueError(
                f"All multi_step_hours={multi_step_hours} must be multiples "
                f"of era5_dt_hours={era5_dt_hours}."
            )
        target_strides = tuple(h // era5_dt_hours for h in multi_step_hours)
        max_target_stride = max_lead_hours // era5_dt_hours
    else:
        target_strides = (rollout_stride,)
        max_target_stride = rollout_stride

    if windows:
        # AIMIP-style multi-window contiguous sampling.  Each window
        # spec is ``(year, day_offset, n_days)``.  v12: with sub-daily
        # rollouts we still describe windows in DAYS at the YAML
        # level, but expand each window to one IC per ERA5 snapshot
        # (4 per day at 6 h cadence).  Pairs are formed inside the
        # window only -- no cross-window leakage.
        window_specs: list[tuple[int, int, int]] = [
            (int(y), int(off), int(n)) for (y, off, n) in windows
        ]
        time_indices: list[int] = []
        window_offsets: list[tuple[int, int]] = []  # (start_in_time_indices, n_ics)
        for (year, day_offset, n_days_w) in window_specs:
            if n_days_w < 1:
                raise ValueError(
                    f"Window {(year, day_offset, n_days_w)} has n_days < 1."
                )
            start_idx_w = _year_to_idx(year) + int(day_offset) * snapshots_per_day
            base = len(time_indices)
            n_ics_w = n_days_w * snapshots_per_day
            n_snapshots_w = n_ics_w + max_target_stride
            time_indices.extend(
                start_idx_w + s for s in range(n_snapshots_w)
            )
            window_offsets.append((base, n_ics_w))
        n_pairs = sum(w[1] for w in window_offsets)
        logger.info(
            f"Loading {n_pairs} ERA5 pairs "
            f"(rollout={rollout_hours}h, era5 cadence={era5_dt_hours}h) "
            f"across {len(window_specs)} windows: "
            + ", ".join(f"{y}@day{o}+{n}" for (y, o, n) in window_specs)
            + f" ({len(time_indices)} snapshots)..."
        )
    else:
        n_days = config.n_train_days
        n_ics = n_days * snapshots_per_day
        start_idx = _year_to_idx(config.start_year)
        time_indices = [start_idx + s for s in range(n_ics + max_target_stride)]
        window_offsets = [(0, n_ics)]
        logger.info(
            f"Loading {n_ics} ERA5 pairs "
            f"(rollout={rollout_hours}h, era5 cadence={era5_dt_hours}h) "
            f"from year {config.start_year} (start_idx={start_idx}; "
            f"opening Zarr store once, reading {len(time_indices)} snapshots)..."
        )

    # Real wall-clock timestamp per sample POSITION (index into time_indices).
    # Used for ic_times in EVERY mode, and — in window-cache mode — to select
    # snapshots from the scoped store by timestamp instead of absolute index.
    try:
        _full_times = np.array(ds.time.values, dtype="datetime64[ns]")
        sample_times = [_full_times[t] for t in time_indices]
    except Exception as exc:  # zarr store without a readable time coord
        sample_times = None
        logger.warning(
            f"load_training_data: ds.time unavailable ({exc!r}); ic_times will "
            f"be None (prescribed-forcing training needs it)."
        )

    # Window-scoped read store (#985): build the cache from the REMOTE store at
    # exactly the (unique) touched indices, then read snapshots back BY
    # TIMESTAMP.  Reading by timestamp — not by a remapped integer position — is
    # robust to overlapping windows / target spillover that make time_indices
    # non-unique: the scoped store holds each timestamp once and ``.sel`` finds
    # it regardless of how many sample positions reference it.
    read_by_time = False
    read_ds = ds
    if _window_cache:
        if sample_times is None:
            raise RuntimeError(
                "LEGOESM_ERA5_WINDOW_CACHE needs a readable ds.time to scope "
                "the cache by timestamp."
            )
        from legoesm.training.era5_to_state import (
            selection_fingerprint, wait_for_cache,
        )
        _uniq = sorted({int(t) for t in time_indices})
        _yrs = _training_year_range(windows, config)
        # Fingerprint the EXACT selection + source config into BOTH the cache
        # dir and its marker, so two different window sets that happen to share a
        # year span + snapshot count never reuse each other's store (codex #985).
        _fp = selection_fingerprint(_uniq, era5_config)
        _scoped = os.path.join(
            os.fspath(cache_dir), f"ywin_{_yrs[0]}_{_yrs[1]}_{_fp}",
        )

        # Multi-rank: serialize the WRITE so the ranks don't race on the shared
        # `.building` dir / os.replace (codex #985).  Coordinate via the
        # FILESYSTEM MARKER, NOT an MPI barrier: this loader can run on the
        # background prefetch thread while the main thread is mid gradient-
        # allreduce on COMM_WORLD, and a barrier there would interleave with
        # those collectives and deadlock.  Rank 0 (or a single rank) builds; the
        # others poll for the marker.  ensure_local_cache is idempotent, so an
        # already-complete store just returns.
        from legoesm.training.data_parallel import mpi_rank_size
        _rank, _nproc = mpi_rank_size()
        if _nproc > 1 and _rank != 0:
            _cache_path = wait_for_cache(_scoped, len(_uniq), _fp)
        else:
            _cache_path = ensure_local_cache(
                era5_config, _scoped, years=_yrs,
                time_selection=_uniq, fingerprint=_fp,
            )
        read_ds = open_era5_zarr(str(_cache_path))
        read_by_time = True
        logger.info(
            f"Window-scoped ERA5 cache: {len(_uniq)} unique snapshots at "
            f"{_cache_path}"
        )

    # Read lat/lon and pressure levels (from the store actually read).
    lat = np.deg2rad(read_ds.lat.values.astype(np.float64))
    lon = np.deg2rad(read_ds.lon.values.astype(np.float64))
    plev_hPa = np.array(era5_config.levels, dtype=np.float64)
    plev_Pa = np.sort(plev_hPa * 100.0)
    level_dim = "level" if "level" in read_ds.dims else "pressure_level"

    # Load surface geopotential (static, no time dim)
    phis_var = resolve_var(read_ds, "geopotential_at_surface")
    if phis_var:
        phis_era5 = read_ds[phis_var].values.astype(np.float32)
    else:
        phis_era5 = np.zeros((len(lat), len(lon)), dtype=np.float32)

    def _load_one_snapshot(ds_t):
        """Regrid one already-selected ERA5 time slice to the model grid."""

        def _get_3d(name):
            r = resolve_var(ds_t, name)
            if r is None:
                return np.zeros((len(lat), len(lon), len(plev_Pa)))
            data = ds_t[r].sel({level_dim: list(era5_config.levels)}).values
            if data.ndim == 3:
                dims = list(ds_t[r].dims)
                spatial = {"lat", "lon", "latitude", "longitude"}
                level_axis = next(
                    (i for i, d in enumerate(dims) if d not in spatial), 0,
                )
                if level_axis != 2:
                    data = np.moveaxis(data, level_axis, -1)
            if plev_hPa[0] > plev_hPa[-1]:
                data = data[..., ::-1]
            return data.astype(np.float32)

        def _get_2d(name):
            r = resolve_var(ds_t, name)
            if r is None:
                r = resolve_var(read_ds, name)
                if r is None:
                    return np.zeros((len(lat), len(lon)), dtype=np.float32)
                return read_ds[r].values.squeeze().astype(np.float32)
            data = ds_t[r].values
            data = data.squeeze()
            while data.ndim > 2:
                data = data[0]
            return data.astype(np.float32)

        era5 = ERA5Slice(
            T=_get_3d("temperature"),
            u=_get_3d("u_component_of_wind"),
            v=_get_3d("v_component_of_wind"),
            q=_get_3d("specific_humidity"),
            p_s=_get_2d("surface_pressure"),
            sst=_get_2d("skin_temperature"),
            phis=phis_era5,
            lat=lat, lon=lon, plev_Pa=plev_Pa,
        )
        return era5_to_spectral_carry(era5, grid, sigma)

    # Load all snapshots.  Position ``j`` maps to absolute index
    # ``time_indices[j]`` in the full store; window-cache mode selects the SAME
    # snapshot from the scoped store by its wall-clock timestamp.
    import time as _time
    t0 = _time.time()
    carries = []
    for j in range(len(time_indices)):
        if read_by_time:
            ds_t = read_ds.sel(time=sample_times[j])
        else:
            ds_t = read_ds.isel(time=time_indices[j])
        carries.append(_load_one_snapshot(ds_t))
        if (j + 1) % 50 == 0:
            elapsed = _time.time() - t0
            logger.info(f"  Loaded {j+1}/{len(time_indices)} snapshots ({elapsed:.0f}s)")

    logger.info(f"Loaded {len(time_indices)} snapshots ({_time.time()-t0:.0f}s)")

    # Build IC/target pairs (per-window so no cross-window leakage).
    # Legacy path (``multi_step_hours`` empty): one target at
    # ``rollout_stride`` snapshots ahead, returned as a flat list of
    # carries.  Multi-step path: a tuple of K carries per IC, one per
    # lead in ``multi_step_hours``.  Downstream callers detect the
    # tuple form and run K chained 6-hour rollouts.
    # Wall-clock IC times (np.datetime64) per sample — consumed by the
    # prescribed-surface-forcing path (``build_amip_sample_forcings``) to
    # evaluate SST/sea-ice + the orbital calendar at each sample.  ``sample_times``
    # (position -> real timestamp) was computed up front from the full-store
    # calendar, so it is correct in every cache mode.
    ic_states = []
    target_carries: list = []
    ic_times: list = []
    n_total_pairs = 0
    multi_step = len(target_strides) > 1 or bool(multi_step_hours)
    for (base, n_w) in window_offsets:
        for d in range(n_w):
            ic_states.append(carry_to_spectral_state(carries[base + d], grid))
            ic_times.append(
                sample_times[base + d] if sample_times is not None else None
            )
            if multi_step:
                targets_seq = tuple(
                    carries[base + d + s] for s in target_strides
                )
                target_carries.append(targets_seq)
            else:
                target_carries.append(carries[base + d + target_strides[0]])
            n_total_pairs += 1

    if multi_step:
        logger.info(
            f"Built {n_total_pairs} IC/(target_seq) tuples "
            f"(multi-step leads={multi_step_hours}h) across "
            f"{len(window_offsets)} window(s)"
        )
    else:
        logger.info(
            f"Built {n_total_pairs} IC/target pairs "
            f"(rollout={rollout_hours}h) across {len(window_offsets)} window(s)"
        )
    return ic_states, target_carries, ic_times


def build_history_pairs(ic_states, ic_times, history_steps,
                        era5_cadence_hours):
    """Attach ``history_steps`` PAST states to each sample, dropping the rest.

    Returns ``(keep_indices, prev_states)`` where ``prev_states[j]`` is a tuple
    (OLDEST FIRST) of the states preceding ``ic_states[keep_indices[j]]``.

    Deliberately derived from what ``load_training_data`` ALREADY returns rather
    than re-fetching: the ERA5 window cache is keyed on the window list, so
    extending each window backwards by one snapshot would invalidate every cache
    on disk and re-download hours of data to gain ~5 % more samples (a 5-day
    window is 20 snapshots at 6 h cadence).

    Contiguity is established from the TIMESTAMPS, not from list adjacency:
    consecutive entries in ``ic_states`` cross a window boundary wherever the
    window list jumps, and pairing across that boundary would feed the network a
    "past" state months or years away from its current one. A sample is kept
    only when all ``history_steps`` predecessors are exactly one ERA5 cadence
    apart, so the first sample of every window is dropped.

    ``ic_times`` is REQUIRED (raises when absent or holding ``None``): without
    timestamps there is no way to tell a genuine predecessor from a
    cross-window one, and silently assuming adjacency is exactly the kind of
    plausible-but-wrong pairing that trains fine and forecasts badly.
    """
    history_steps = int(history_steps)
    if history_steps <= 0:
        return list(range(len(ic_states))), [()] * len(ic_states)
    if ic_times is None or any(t is None for t in ic_times):
        raise ValueError(
            "history_steps > 0 requires per-sample ic_times to prove two "
            "samples are adjacent IN TIME; without them a 'previous' state "
            "can silently come from a different window (or year)."
        )
    import numpy as _np

    step = _np.timedelta64(int(round(float(era5_cadence_hours) * 60)), "m")
    times = [_np.datetime64(t) for t in ic_times]
    keep, prevs = [], []
    for i in range(len(ic_states)):
        if i - history_steps < 0:
            continue
        if all(times[i - k] - times[i - k - 1] == step
               for k in range(history_steps)):
            keep.append(i)
            prevs.append(tuple(
                ic_states[i - history_steps + k] for k in range(history_steps)
            ))
    logger.info(
        "history_steps=%d: kept %d/%d samples (dropped %d that had no "
        "in-window predecessor at %g h cadence)",
        history_steps, len(keep), len(ic_states),
        len(ic_states) - len(keep), float(era5_cadence_hours),
    )
    return keep, prevs


# =============================================================================
# Training entry point
# =============================================================================

def _resolve_dp_context(config):
    """``(dp_on, rank, nproc, comm)`` for the chunked spectral trainer (#985).

    Data-parallel training is active only when ``config.data_parallel`` is set
    AND a multi-rank MPI launcher yields ``nproc > 1``.  Off / single-rank ->
    ``(False, 0, 1, None)``: the caller takes the serial fused-step path, which
    is byte-identical to the pre-#985 loop (the default run is unchanged).
    ``comm=None`` lets the reductions default to ``MPI.COMM_WORLD``.
    """
    if not bool(getattr(config, "data_parallel", False)):
        return False, 0, 1, None
    from legoesm.training.data_parallel import mpi_rank_size

    rank, nproc = mpi_rank_size()
    if nproc <= 1:
        return False, rank, 1, None
    return True, rank, nproc, None


def _dp_updates_per_epoch(chunk_sizes, nproc: int) -> int:
    """Per-rank optimizer updates in ONE epoch under data-parallel sharding:
    ``sum_chunks floor(chunk_size / nproc)`` (each rank does one update per
    LOCAL sample; drop_remainder discards ``chunk_size % nproc``).

    Raises if any chunk is smaller than the world size — a rank would then get
    an empty shard and the epoch would do zero updates (silent no-op).  Used to
    size the LR schedule to the real update count AND as the fail-fast guard.
    """
    sizes = [int(s) for s in chunk_sizes]
    if not sizes:
        raise ValueError("data_parallel: no chunks to train on.")
    if min(sizes) < nproc:
        raise ValueError(
            f"data_parallel needs every chunk >= the world size ({nproc} "
            f"ranks), but the smallest chunk has {min(sizes)} sample(s): with "
            f"drop_remainder sharding a rank would get an empty shard and the "
            f"epoch would do zero updates. Reduce ranks or raise chunk_windows."
        )
    return sum(s // nproc for s in sizes)


def _dp_chunk_sizes(chunk_loader, n_samples_epoch: int) -> list[int]:
    """Per-chunk sample counts used to size + guard the DP schedule.

    Sharding happens PER CHUNK, so DP needs the real per-chunk sizes — NOT the
    epoch total treated as one chunk (that would size the schedule wrong AND
    hide an undersized chunk that silently trains zero samples).  A ``chunk_loader``
    MUST therefore expose ``chunk_sizes`` (the built-in ``_make_chunk_loader``
    does); a custom loader that omits it is rejected rather than mis-sized.  The
    single in-memory pass (``chunk_loader is None``) is exactly one chunk.
    """
    if chunk_loader is None:
        return [int(n_samples_epoch)]
    cs = getattr(chunk_loader, "chunk_sizes", None)
    if cs is None:
        raise ValueError(
            "data_parallel requires the chunk loader to expose `chunk_sizes` "
            "(per-chunk sample counts) so per-chunk sharding is sized and "
            "guarded correctly; _make_chunk_loader sets it — a custom loader "
            "must too."
        )
    return [int(s) for s in cs]


@mpi_abort_on_uncaught  # a rank dying (OOM/NaN) MPI_Aborts the job instead of
# leaving its peers hung in the next gradient allreduce (#985)
def _train_spectral_loop(
    model: eqx.Module,
    make_physics_fn,
    grid: GaussianGrid,
    sigma: SigmaCoordinate,
    ic_states,
    target_carries,
    config: NeuralGCMSpectralConfig,
    *,
    start_epoch: int = 0,
    sample_forcings: list | None = None,
    chunk_loader=None,
    n_samples_total: int | None = None,
    resume_from_dir=None,
    host_staged: bool = False,
):
    """Shared training loop for any learned-physics model coupled to the
    spectral PE dycore.

    ``host_staged=True`` declares that the PROVIDED ``ic_states`` /
    ``target_carries`` / ``sample_forcings`` were built host-resident
    (``load_training_data(host_resident=True)``, #1155) — each sample is then
    moved to the compute device just before its step, exactly like the #985
    prefetch path (which signals the same thing via the chunk loader's
    ``host_staged`` attribute).

    Parameters
    ----------
    model : eqx.Module
        Learnable model (SFNO or NeuralPhysics).
    make_physics_fn : callable
        ``(model, grid) -> physics_fn`` factory.
    grid, sigma : grid and vertical coordinate.
    ic_states : list of SpectralHydrostaticState.
    target_carries : list of SegmentCarry.
    config : NeuralGCMSpectralConfig.
    sample_forcings : list or None
        Optional per-sample prescribed surface forcing, aligned with
        ``ic_states``: each entry is the traced dict consumed by
        ``spectral_rollout(forcing_base=...)`` (``T_sfc``/``sic``
        columns + ``day_of_year``/``seconds_of_day`` scalars at the
        sample's IC time).  Passed to ``_train_step`` as a TRACED
        pytree argument (SegmentForcing doctrine: one JIT trace, the
        values change per sample).  ``None`` = legacy unforced
        training (idealized inputs; physics_fn falls back to its
        internal proxies).
    chunk_loader : callable or None
        Streaming/chunked data source: a zero-arg callable returning a
        fresh iterator of ``(ic_states, target_carries, sample_forcings)``
        chunks each time it is called (one full pass = one epoch). Used
        for DENSE all-years sampling that does not fit in host RAM as a
        single list. When set, ``ic_states``/``target_carries``/
        ``sample_forcings`` may be None; ``n_samples_total`` must give
        the per-epoch sample count (optimizer schedule). Every chunk
        must produce samples of identical shapes so the jitted train
        step is reused across chunks (no retrace).
    n_samples_total : int or None
        Per-epoch sample count when ``chunk_loader`` is used.
    resume_from_dir : str | Path | None
        Directory scanned for a mid-epoch ``chunk_latest.eqx`` checkpoint
        (chunked path only).  When found and at least as advanced as
        ``start_epoch``, the model + optimizer state are restored and the
        already-trained chunks of the resumed epoch are skipped (#942).
        The per-chunk checkpoint is also WRITTEN to
        ``config.checkpoint_dir`` after every chunk on the chunked path.

    Returns
    -------
    model : updated eqx.Module
    loss_history : list[float]
    """
    if chunk_loader is None and sample_forcings is not None \
            and len(sample_forcings) != len(ic_states):
        raise ValueError(
            f"sample_forcings length {len(sample_forcings)} != "
            f"n_samples {len(ic_states)}."
        )
    if chunk_loader is not None and not n_samples_total:
        raise ValueError("chunk_loader requires n_samples_total.")
    sigma_full = jnp.asarray(sigma.sigma_full)
    pe_config = config.pe_config

    sponge_factor = None
    if pe_config.sponge_tau > 0:
        sponge_factor = compute_sponge_factor(
            sigma.sigma_full, pe_config.sponge_sigma,
            pe_config.sponge_tau, config.dt,
        )
    spectral_filter = None
    if pe_config.spectral_filter_strength > 0:
        spectral_filter = compute_spectral_filter(
            grid.ls, grid.n_max,
            order=pe_config.spectral_filter_order,
            cutoff_fraction=pe_config.spectral_filter_strength,
        )

    from legoesm.ml.training import TrainingConfig, create_optimizer

    n_steps_per_day = int(86400 / config.dt)
    rollout_days = int(getattr(config, "rollout_days", 1) or 1)
    rollout_hours_cfg = int(getattr(config, "rollout_hours", 0) or 0)
    rollout_hours = rollout_hours_cfg if rollout_hours_cfg > 0 else rollout_days * 24
    n_steps_rollout = int(round(rollout_hours * 3600.0 / config.dt))
    n_samples_epoch = int(n_samples_total or len(ic_states))
    curriculum = tuple(
        (int(h), int(ep)) for h, ep in (config.rollout_curriculum or ())
    ) or None
    n_epochs_total = (
        sum(ep for _, ep in curriculum) if curriculum else config.n_epochs
    )
    # Fail-early curriculum validation (before the optimizer schedule is
    # built, so a bad curriculum surfaces as ITS error, not a schedule
    # side-effect like decay_steps=0). The shared builder raises the
    # canonical errors; the plan itself is rebuilt later next to its use.
    if curriculum:
        build_curriculum_epoch_plan(
            curriculum,
            tuple(int(h) for h in (config.loss_config.multi_step_hours or ())),
            config.dt,
            0,
        )
    # --- data-parallel context (#985), resolved BEFORE the optimizer so the
    # warmup+cosine schedule is sized by the ACTUAL number of optimizer updates.
    # Under DP each rank performs one update per LOCAL sample, i.e. only
    # sum_chunks floor(chunk_size / nproc) updates per epoch — sizing the
    # schedule by the unsharded sample count would leave the LR ~nproc x too
    # high at the end of training.  Off / single-rank -> serial, byte-identical.
    dp_on, dp_rank, dp_nproc, dp_comm = _resolve_dp_context(config)
    _all_reduce_grad_mean = _global_sum_mpi = _shard_samples = None
    if dp_on:
        from legoesm.parallel.reductions import global_sum_mpi as _global_sum_mpi
        from legoesm.training.data_parallel import (
            all_reduce_grad_mean as _all_reduce_grad_mean,
            shard_samples as _shard_samples,
        )
        _chunk_sizes = _dp_chunk_sizes(chunk_loader, n_samples_epoch)
        _updates_per_epoch = _dp_updates_per_epoch(_chunk_sizes, dp_nproc)
        _dropped = sum(s % dp_nproc for s in _chunk_sizes)
        if _dropped:
            logger.warning(
                f"data_parallel drops {_dropped} remainder sample(s) per epoch "
                f"(chunk sizes not divisible by {dp_nproc} ranks)."
            )
        logger.info(
            f"Data-parallel training: rank {dp_rank}/{dp_nproc}; each chunk's "
            f"samples sharded across ranks (drop_remainder), gradients averaged "
            f"per step, checkpoints written by rank 0 only. "
            f"{_updates_per_epoch} updates/epoch/rank."
        )
    else:
        _updates_per_epoch = n_samples_epoch

    total_steps = max(1, n_epochs_total * max(1, _updates_per_epoch))
    base_optimizer = create_optimizer(TrainingConfig(
        lr=config.lr,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        optimizer=config.optimizer,
    ))

    # Per-group LR partitioning for AIMIP spatial-surface coefficients.
    # When ``model.spatial_surface`` is present and the user has asked
    # for a non-default ``spatial_lr_scale``, build a second optimizer
    # at the scaled LR and route spatial-coef leaves to it via
    # ``optax.multi_transform``.  Detection is purely structural: any
    # leaf whose path contains the substring ``spatial_surface`` is
    # treated as a spatial coefficient, so the partition works for any
    # eqx.Module that nests a :class:`AIMIPSpatialSurfaceParams` under
    # that attribute name.
    spatial_lr_scale = float(getattr(config, "spatial_lr_scale", 1.0) or 1.0)
    has_spatial = has_spatial_surface_leaves(model)
    if has_spatial and spatial_lr_scale != 1.0:
        spatial_optimizer = create_optimizer(TrainingConfig(
            lr=config.lr * spatial_lr_scale,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
            optimizer=config.optimizer,
        ))

        optimizer = optax.multi_transform(
            {"spatial": spatial_optimizer, "base": base_optimizer},
            label_aimip_spatial_params,
        )
        logger.info(
            f"Per-group LR: spatial coefs at lr={config.lr * spatial_lr_scale:.2e} "
            f"(x{spatial_lr_scale}), scalar/MLP at lr={config.lr:.2e}"
        )
    else:
        optimizer = base_optimizer

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- mid-epoch (per-chunk) resume (#942) ---------------------------------
    # On the chunked all-years path, prefer the ``chunk_latest.eqx`` written
    # after each chunk over the per-epoch ``epoch_NNNN.eqx`` the caller
    # resumed from: it carries the OPTIMIZER STATE (moments + LR-schedule
    # step count) and the exact ``(epoch, next_chunk)`` position, so a job
    # killed mid-epoch continues from the next un-done chunk with the
    # optimizer trajectory unbroken.  Honoured only when it is at least as
    # advanced as the epoch-granular resume (``m_epoch >= start_epoch``);
    # a stale one (older epoch) is ignored.
    # EMA is folded into the mid-epoch payload (atomic — no torn raw/EMA
    # pair), so its decay must be known before the mid-epoch load.
    ema_decay = float(getattr(config, "ema_decay", 0.0) or 0.0)
    _restored_ema = None            # EMA recovered from chunk_latest.eqx
    _midepoch_ema_present = False    # payload carried an EMA (vs legacy)
    resume_chunk = 0
    midepoch_restored = False
    if chunk_loader is not None and resume_from_dir is not None:
        _ema_tmpl = init_ema(model) if ema_decay > 0.0 else None
        _mid = _load_midepoch_checkpoint(
            resume_from_dir, model, opt_state, ema_template=_ema_tmpl,
        )
        if _mid is not None:
            if _ema_tmpl is not None:
                m_model, m_opt_state, m_ema, m_epoch, m_next_chunk = _mid
            else:
                m_model, m_opt_state, m_epoch, m_next_chunk = _mid
                m_ema = None
            if m_epoch >= start_epoch:
                model, opt_state = m_model, m_opt_state
                _restored_ema = m_ema
                _midepoch_ema_present = m_ema is not None
                start_epoch = m_epoch
                resume_chunk = m_next_chunk
                midepoch_restored = True
                logger.info(
                    f"Mid-epoch resume: restored model + optimizer state at "
                    f"epoch {start_epoch}, chunk {resume_chunk} "
                    f"({MIDEPOCH_CHECKPOINT_NAME}); skipping the "
                    f"{resume_chunk} already-trained chunk(s) of this epoch."
                )
            else:
                logger.info(
                    f"Ignoring stale {MIDEPOCH_CHECKPOINT_NAME} "
                    f"(epoch {m_epoch} < resume epoch {start_epoch})."
                )
    if start_epoch > 0 and not midepoch_restored:
        # No mid-epoch checkpoint to restore from (epoch-boundary resume off
        # a model-only ``epoch_NNNN.eqx``, or the non-chunked path): the
        # optimizer is re-init'd, so the warmup+cosine schedule replays from
        # step 0 while the epoch counter skips ahead.  Known limitation of a
        # model-only resume — harmless for weights, but LR != the unbroken
        # run.  The chunked path avoids this via ``chunk_latest.eqx`` above.
        logger.warning(
            f"Resume at epoch {start_epoch}: optimizer state is fresh; "
            f"the LR schedule restarts from step 0 (weights unaffected)."
        )
    loss_history = []

    # --- EMA of the trainable weights (D3, training/ema.py) --------------
    # Updated on the host after every optimizer step (identical on every
    # DP rank: same averaged gradient -> same model -> same EMA), saved
    # beside each checkpoint. Resume prefers a matching *_ema.eqx; falls
    # back to re-seeding from the restored raw model (logged) so a legacy
    # run directory keeps working.
    # --- EMA of the trainable weights (D3, training/ema.py) --------------
    # Updated on the host after every optimizer step (identical on every DP
    # rank: same averaged gradient -> same model -> same EMA). For the
    # chunked path the EMA lives INSIDE the atomic chunk_latest.eqx payload
    # (no torn raw/EMA pair); the per-epoch path saves epoch_NNNN_ema.eqx
    # beside epoch_NNNN.eqx and an epoch-boundary resume restores that.
    ema_model = None
    _ema_update_fn = None
    if ema_decay > 0.0:
        ema_model = init_ema(model)
        # One jitted EMA step reused across the loop (a bare per-step
        # partition/tree-map/combine on a large SFNO is host-dispatch bound).
        # decay is closed over -> single trace.
        _ema_update_fn = eqx.filter_jit(
            lambda e, m: ema_update(e, m, ema_decay)
        )
        _resumed_weights = midepoch_restored or start_epoch > 0
        if _restored_ema is not None:
            # From the atomic mid-epoch payload: exactly paired with the
            # restored model + position.
            ema_model = _restored_ema
            logger.info(
                f"EMA resume: restored from {MIDEPOCH_CHECKPOINT_NAME} "
                "(atomic payload)"
            )
        elif (
            resume_from_dir is not None
            and not midepoch_restored
            and start_epoch > 0
        ):
            # Epoch-boundary resume: the EMA sibling of the last completed
            # epoch's checkpoint.
            _cand = (
                Path(resume_from_dir) / f"epoch_{start_epoch - 1:04d}_ema.eqx"
            )
            if _cand.exists():
                ema_model = eqx.tree_deserialise_leaves(_cand, ema_model)
                logger.info(f"EMA resume: restored {_cand}")
            else:
                logger.warning(
                    f"EMA resume: no {_cand.name} — re-seeding the EMA from "
                    "the restored raw weights."
                )
        elif _resumed_weights and midepoch_restored and not _midepoch_ema_present:
            # Enabled EMA on the resume of a run trained without it (legacy
            # 4-tuple payload): reseed from the restored raw weights.
            logger.warning(
                "EMA resume: the mid-epoch checkpoint predates EMA — "
                "re-seeding the EMA from the restored raw weights."
            )

    logger.info(
        f"Training: {config.n_epochs} epochs, "
        f"{n_samples_epoch} samples/epoch, "
        f"{n_steps_per_day} dycore steps/day (dt={config.dt}s), "
        f"rollout_hours={rollout_hours} -> {n_steps_rollout} steps/sample"
    )

    # Build the JIT-compiled train step ONCE outside the per-sample
    # loop.  The previous implementation built ``loss_fn`` -- and
    # therefore the ``physics_fn`` Python closure tree -- on every
    # gradient call, hitting the ``CLAUDE.md`` anti-pattern documented
    # under "Never build closures inside training loops" and leaking
    # ~1.5 GiB of Python/XLA metadata per sample at T11 L6 under
    # RRTMGP (verified 2026-05-17; OOM-killed mid-epoch-1).  Wrapping
    # the whole step in :func:`eqx.filter_jit` traces the build path
    # exactly once and caches the resulting XLA graph; subsequent
    # samples are pure GPU-bound forward+backward+update calls.
    #
    # ``make_physics_fn`` is a Python callable captured in the closure
    # (non-array, not a JIT input).  ``optimizer`` is likewise an
    # ``optax.GradientTransformation`` captured in the closure --
    # ``filter_jit`` traces it as a static argument.
    rad_update_interval = int(
        getattr(config, "rad_update_interval", 1) or 1
    )

    # Pre-warm the RRTMGP optics-table cache OUTSIDE the
    # ``filter_jit``-wrapped train step.  ``RRTMGP.preload`` reads
    # NetCDF gas-optics tables (``bnd_limits_gpt`` et al.) at first
    # call and stashes the resulting ``optics_lib`` in a module-level
    # cache keyed by the static file paths.  When the first
    # invocation happens inside ``eqx.filter_value_and_grad`` the
    # NetCDF-load path hits the Equinox-tracer state and crashes with
    # ``TracerArrayConversionError`` (verified 2026-05-18).  A
    # concrete-args call here populates the cache so the inside-trace
    # calls hit the cache and skip the load entirely.
    try:
        _warmup_physics = make_physics_fn(model, grid)
        del _warmup_physics
    except Exception as exc:
        logger.warning(
            f"RRTMGP cache warm-up call raised {exc!r}; continuing "
            "(the first jitted train step will retry under tracing -- "
            "if that also fails you likely hit a non-AIMIP physics path)."
        )

    # GenCast-style autoregressive supervision: chain K rollouts of
    # ``segment_steps[k]`` dycore steps each (one per lead in
    # ``loss_config.multi_step_hours``) and sum a CRPS+MSE loss after
    # each segment.  M=1 ensemble => CRPS reduces to MAE (the per-
    # segment loss already mixes MSE + MAE via ``w_*`` and ``w_crps_*``).
    # ``segment_steps`` is empty / single-element on the legacy
    # single-step path; downstream code is identical in that case.
    loss_cfg_train = config.loss_config
    multi_step_hours_train = tuple(
        int(h) for h in (loss_cfg_train.multi_step_hours or ())
    )
    if multi_step_hours_train:
        prev = 0
        segment_steps = []
        for h in multi_step_hours_train:
            segment_steps.append(
                int(round((h - prev) * 3600.0 / config.dt))
            )
            prev = h
        if any(s <= 0 for s in segment_steps):
            raise ValueError(
                f"Non-positive segment length derived from "
                f"multi_step_hours={multi_step_hours_train}, dt={config.dt}."
            )
        ms_weights_cfg = tuple(loss_cfg_train.multi_step_weights or ())
        if ms_weights_cfg and len(ms_weights_cfg) != len(segment_steps):
            raise ValueError(
                f"multi_step_weights length {len(ms_weights_cfg)} != "
                f"len(multi_step_hours)={len(segment_steps)}."
            )
        ms_weights = (
            tuple(float(w) for w in ms_weights_cfg)
            if ms_weights_cfg else (1.0,) * len(segment_steps)
        )
        ms_weight_sum = float(sum(ms_weights))
        logger.info(
            f"Multi-step autoregressive supervision active: "
            f"leads={multi_step_hours_train}h "
            f"(segments={segment_steps} dycore steps, weights={ms_weights})"
        )
    else:
        segment_steps = ()
        ms_weights = ()
        ms_weight_sum = 0.0

    def _rollout_one_segment(state, physics, n_seg_steps, time_offset_seconds,
                             forcing_base=None, phys_state=None):
        """Run one autoregressive segment (n_seg_steps dycore steps).

        ``time_offset_seconds`` is the cumulative simulated time elapsed
        since the IC.  The gated-radiation branch adds this to the
        per-segment scan index so the diurnal cycle stays phase-correct
        across multi-step autoregressive rollouts (segment k of K
        evaluates rad at IC + k * 6h, not 00 UTC each time).  The same
        offset advances ``forcing_base``'s calendar inside
        ``spectral_rollout`` for the learned-physics forcing path.
        """
        if isinstance(physics, tuple):
            non_rad_fn, rad_fn = physics
            # PROGNOSTIC PHYSICS MEMORY across chained segments (codex P0):
            # a marked stateful physics_fn returns (state, phys_state) so
            # segment k+1 continues from segment k's memory instead of
            # re-seeding every seam. The caller threads ``phys_state``.
            if hasattr(non_rad_fn, "with_phys_state"):
                return spectral_rollout(
                    state, non_rad_fn, grid, sigma, pe_config,
                    config.dt, n_seg_steps,
                    sponge_factor, spectral_filter,
                    rad_physics_fn=rad_fn,
                    rad_update_interval=rad_update_interval,
                    sim_time_offset_seconds=time_offset_seconds,
                    phys_state_in=phys_state,
                    return_phys_state=True,
                )
            return spectral_rollout(
                state, non_rad_fn, grid, sigma, pe_config,
                config.dt, n_seg_steps,
                sponge_factor, spectral_filter,
                rad_physics_fn=rad_fn,
                rad_update_interval=rad_update_interval,
                sim_time_offset_seconds=time_offset_seconds,
            ), None
        return spectral_rollout(
            state, physics, grid, sigma, pe_config,
            config.dt, n_seg_steps,
            sponge_factor, spectral_filter,
            sim_time_offset_seconds=time_offset_seconds,
            forcing_base=forcing_base,
        ), None

    # --- rollout curriculum (NeuralGCM-style stability training) ---
    # Each phase supervises ONE autoregressive rollout to phase_lead hours
    # against the ERA5 target at that lead (single-lead loss).  Targets for
    # every phase lead were loaded up front: the caller must set
    # ``loss_config.multi_step_hours`` to the sorted set of curriculum
    # leads so ``load_training_data`` builds a target tuple per sample.
    # Flat epoch plan: (phase_lead_hours, target_index, n_steps) per
    # global epoch — resume (start_epoch) indexes into this plan. Shared
    # implementation with the sfno_full macro-step plan (curriculum.py).
    epoch_plan = build_curriculum_epoch_plan(
        curriculum, multi_step_hours_train, config.dt, n_epochs_total
    )
    if curriculum:
        logger.info(
            f"Rollout curriculum active: "
            + ", ".join(f"{h}h x{ep}" for h, ep in curriculum)
            + f" ({len(epoch_plan)} epochs total)"
        )

    def _loss_components(m, ic_spectral, target_carry, forcing_base, phase_spec):
        """Per-sample loss + components for ``phase_spec``.

        ``(None, None, None)`` = the default path (chained multi-step or
        legacy single-rollout).  A curriculum spec = single rollout of
        ``n_steps_phase`` scored against target index ``k_target``.  SHARED by
        the serial fused step and the data-parallel grad-only step, so both run
        byte-identical physics (nproc==1 DP reproduces serial exactly).
        """
        _, k_target, n_steps_phase = phase_spec
        physics = make_physics_fn(m, grid)
        if k_target is not None:
            # Curriculum phase: one rollout to the phase lead.
            tgt = (target_carry[k_target]
                   if type(target_carry) is tuple else target_carry)
            pred, _ = _rollout_one_segment(
                ic_spectral, physics, n_steps_phase, 0.0,
                forcing_base=forcing_base,
            )
            return _spectral_state_loss_components(
                pred, tgt, grid, sigma, sigma_full, loss_cfg_train,
            )
        if segment_steps:
            state = ic_spectral
            total = jnp.float32(0.0)
            comp_total: dict = {}
            t_offset = 0.0
            seg_ps = None    # physics memory chained across segments
            for k, n_seg in enumerate(segment_steps):
                state, seg_ps = _rollout_one_segment(
                    state, physics, n_seg, t_offset,
                    forcing_base=forcing_base, phys_state=seg_ps,
                )
                seg_loss, seg_comp = _spectral_state_loss_components(
                    state, target_carry[k], grid, sigma,
                    sigma_full, loss_cfg_train,
                )
                total = total + ms_weights[k] * seg_loss
                # Accumulate over the LOSS's keys (not a fixed template) so
                # optional per-variable terms (bias_T/u/v/ps) survive; the key
                # set is config-static, so the pytree structure is stable
                # across segments and steps.
                for key, v in seg_comp.items():
                    comp_total[key] = (
                        comp_total.get(key, jnp.float32(0.0))
                        + ms_weights[k] * v)
                t_offset = t_offset + float(n_seg) * config.dt
            inv = 1.0 / ms_weight_sum
            return total * inv, {k: v * inv for k, v in comp_total.items()}
        # Legacy single-step path.
        pred, _ = _rollout_one_segment(
            ic_spectral, physics, n_steps_rollout, 0.0,
            forcing_base=forcing_base,
        )
        return _spectral_state_loss_components(
            pred, target_carry, grid, sigma,
            sigma_full, loss_cfg_train,
        )

    def _make_train_step(phase_spec):
        """Jitted fused (grad + optax update) train step — the serial path."""
        def _train_step(model, opt_state, ic_spectral, target_carry, forcing_base):
            def loss_fn(m):
                return _loss_components(
                    m, ic_spectral, target_carry, forcing_base, phase_spec,
                )
            (loss, components), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True,
            )(model)
            grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
            updates, new_opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(model, eqx.is_array),
            )
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss, grad_norm, components

        return eqx.filter_jit(_train_step)

    def _make_dp_grad_step(phase_spec):
        """Jitted GRAD-ONLY step for data-parallel training.

        Returns ``(loss, components, grads)`` for THIS rank's sample; the caller
        averages ``grads`` across ranks (``all_reduce_grad_mean``) BEFORE the
        optax update, so the cross-rank collective sits between grad and update
        and the cheap update stays outside JIT (matches the WB DP path).
        """
        def _grad_step(model, ic_spectral, target_carry, forcing_base):
            def loss_fn(m):
                return _loss_components(
                    m, ic_spectral, target_carry, forcing_base, phase_spec,
                )
            (loss, components), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True,
            )(model)
            return loss, components, eqx.filter(grads, eqx.is_array)

        return eqx.filter_jit(_grad_step)

    # One jitted step per distinct phase spec (compile once, reuse across
    # that phase's epochs AND across chunks — shapes are constant).
    _step_cache: dict = {}
    _dp_step_cache: dict = {}

    def _train_step_for(spec):
        if spec not in _step_cache:
            _step_cache[spec] = _make_train_step(spec)
        return _step_cache[spec]

    def _dp_grad_step_for(spec):
        if spec not in _dp_step_cache:
            _dp_step_cache[spec] = _make_dp_grad_step(spec)
        return _dp_step_cache[spec]


    def _iter_epoch_data(start_chunk=0):
        """Yield (ic_states, target_carries, sample_forcings) chunks.

        ``start_chunk`` skips the first N chunks WITHOUT loading them
        (mid-epoch resume): the already-trained chunks of a resumed
        epoch are never re-streamed from GCS.
        """
        if chunk_loader is None:
            # A single in-memory pass == one chunk; nothing to skip.
            yield ic_states, target_carries, sample_forcings
        else:
            yield from chunk_loader(start_chunk=start_chunk)

    # Chunks per epoch is constant across epochs (deterministic window
    # partition); used to normalise the last chunk of epoch e to the
    # resume position (e+1, 0) for the mid-epoch checkpoint.
    n_chunks_per_epoch = (
        int(getattr(chunk_loader, "n_chunks", 1))
        if chunk_loader is not None else 1
    )

    # When the chunk loader host-stages chunks (prefetch, #985), each chunk
    # arrives on the CPU and must be moved onto the compute device at the point
    # of use in the loop below -- see the loop body for why this is not done in
    # the prefetch consumer.
    _host_staged = bool(getattr(chunk_loader, "host_staged", False)) or bool(host_staged)
    _compute_dev = jax.devices()[0] if _host_staged else None

    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = int(getattr(config, "early_stop_patience", 0) or 0)
    early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 1.0e-3))

    # (data-parallel context was resolved above, before the optimizer schedule.)

    if start_epoch >= n_epochs_total:
        logger.info(
            f"Resume: start_epoch={start_epoch} >= n_epochs={n_epochs_total}; "
            f"skipping training loop (already complete)."
        )
        return model, loss_history

    for epoch in range(start_epoch, n_epochs_total):
        phase_spec = epoch_plan[epoch]
        train_step = None if dp_on else _train_step_for(phase_spec)
        dp_grad_step = _dp_grad_step_for(phase_spec) if dp_on else None
        epoch_loss = 0.0
        epoch_components = {"mse": 0.0, "bias": 0.0, "crps": 0.0, "spec_crps": 0.0}
        t0 = time.time()
        grad_norm_val = 0.0

        sample_idx = -1
        # Mid-epoch resume: skip the chunks already trained in this epoch
        # (only the first resumed epoch has chunk_skip > 0).
        chunk_skip = resume_chunk if epoch == start_epoch else 0
        chunk_pos = chunk_skip - 1
        for chunk_ics, chunk_targets, chunk_forcings in _iter_epoch_data(chunk_skip):
            chunk_pos += 1  # absolute chunk index within the epoch
            # Data-parallel: deterministic CONTIGUOUS shard of this chunk's
            # samples to this rank (drop_remainder keeps ranks balanced so the
            # per-step gradient allreduce never deadlocks).  The shard depends
            # only on (rank, nproc, chunk order), all deterministic -> a resume
            # re-shards identically, so chunk_latest.eqx stays reproducible.
            if dp_on:
                _sh = _shard_samples(
                    list(range(len(chunk_ics))), dp_rank, dp_nproc,
                )
                _ics = [chunk_ics[i] for i in _sh]
                _tgts = [chunk_targets[i] for i in _sh]
                _forc = ([chunk_forcings[i] for i in _sh]
                         if chunk_forcings is not None else None)
            else:
                _ics, _tgts, _forc = chunk_ics, chunk_targets, chunk_forcings

            for chunk_i, (ic, target) in enumerate(zip(_ics, _tgts)):
                sample_idx += 1
                _fb = _forc[chunk_i] if _forc is not None else None
                if _host_staged:
                    # Host-staged chunks are built on the CPU (see
                    # _make_chunk_loader); move only THIS sample onto the compute
                    # device, right before its step. Peak device footprint is one
                    # sample (the prior sample's device arrays are freed when
                    # ic/target/_fb rebind), so a prefetched chunk never rides GPU
                    # memory -- the #985 OOM. Rebinding ic/target keeps this local.
                    ic = _stage_tree(ic, _compute_dev)
                    target = _stage_tree(target, _compute_dev)
                    if _fb is not None:
                        _fb = _stage_tree(_fb, _compute_dev)
                if dp_on:
                    # Local grad on this rank's sample -> average across ranks
                    # BEFORE the update, so every replica applies the identical
                    # gradient and stays in sync (no weight broadcast).
                    loss, components, grads = dp_grad_step(
                        model, ic, target, _fb,
                    )
                    grads = _all_reduce_grad_mean(
                        grads, dp_nproc, comm=dp_comm,
                    )
                    grad_norm = optax.global_norm(grads)
                    updates, opt_state = optimizer.update(
                        grads, opt_state, eqx.filter(model, eqx.is_array),
                    )
                    model = eqx.apply_updates(model, updates)
                else:
                    model, opt_state, loss, grad_norm, components = train_step(
                        model, opt_state, ic, target, _fb,
                    )
                if ema_model is not None:
                    ema_model = _ema_update_fn(ema_model, model)

                # --- NaN / Inf detection (outside JIT, values materialized) ---
                # A rank hitting NaN raises; the @mpi_abort_on_uncaught decorator
                # on this loop then MPI_Aborts the whole job, so its peers can't
                # hang in the next gradient allreduce (#985). A NaN-consensus
                # allreduce would only make the message tidier — skipped.
                loss_val = float(loss)
                if jnp.isnan(loss) or jnp.isinf(loss):
                    raise RuntimeError(
                        f"NaN/Inf loss detected at epoch {epoch}, sample {sample_idx} "
                        f"(loss={loss_val}). "
                        "Check CFL conditions, parameter bounds, and input data."
                    )
                grad_norm_val = float(grad_norm)
                if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                    raise RuntimeError(
                        f"NaN/Inf gradient detected at epoch {epoch}, sample "
                        f"{sample_idx} (grad_norm={grad_norm_val}). "
                        "Consider reducing learning rate or adding gradient clipping."
                    )

                epoch_loss += loss_val
                # Iterate the COMPONENTS (not the init keys) so optional
                # per-variable terms (bias_T/u/v/ps) accumulate too.
                for key in components:
                    epoch_components[key] = (
                        epoch_components.get(key, 0.0) + float(components[key]))

            # --- mid-epoch (per-chunk) checkpoint (#942) ---------------------
            # After EACH chunk on the chunked all-years path, atomically save
            # model + optimizer state + the resume position so a walltime kill
            # loses at most one chunk instead of the whole epoch.  The last
            # chunk of epoch e normalises to (e+1, 0).  (No-op on the single
            # in-memory path, whose epoch == one chunk == the per-epoch save.)
            # Rank 0 only under DP: every rank holds the identical replicated
            # model + optimizer state, so one writer is correct and avoids a
            # shared-filesystem write race.  All ranks resume by reading it.
            if chunk_loader is not None and dp_rank == 0:
                _next_chunk = chunk_pos + 1
                if _next_chunk >= n_chunks_per_epoch:
                    _save_epoch, _save_chunk = epoch + 1, 0
                else:
                    _save_epoch, _save_chunk = epoch, _next_chunk
                # EMA folded into the single atomic mid-epoch payload: the
                # raw weights, optimizer state, EMA weights and resume
                # position land together or not at all (no torn raw/EMA
                # pair — the whole class of "EMA out of sync with position"
                # bugs is eliminated by the atomic write).
                _save_midepoch_checkpoint(
                    config.checkpoint_dir, model, opt_state,
                    _save_epoch, _save_chunk, ema_model=ema_model,
                )
                logger.info(
                    f"Saved mid-epoch checkpoint {MIDEPOCH_CHECKPOINT_NAME} "
                    f"(epoch {epoch}, chunk {chunk_pos} done -> resume at "
                    f"epoch {_save_epoch}, chunk {_save_chunk})"
                )

        n_samples = max(sample_idx + 1, 1)
        if dp_on:
            # Global epoch means across ranks (n_samples/epoch_loss are per-rank
            # local under sharding).  All ranks call this collective in lockstep
            # — balanced shards guarantee equal per-rank step counts — so it can
            # never deadlock; the shared avg_loss also keeps any early-stop
            # decision identical on every rank.
            # Deterministic order over the ACTUAL keys (per-variable bias terms
            # included); identical on every rank for a fixed config, so the
            # collective shape is consistent.
            _keys = tuple(sorted(epoch_components))
            _acc = _global_sum_mpi(
                jnp.asarray(
                    [epoch_loss] + [epoch_components[k] for k in _keys]
                    + [float(n_samples)]
                ),
                comm=dp_comm,
            )
            _accl = [float(x) for x in _acc]
            _gn = max(_accl[-1], 1.0)
            avg_loss = _accl[0] / _gn
            avg_components = {k: _accl[1 + i] / _gn for i, k in enumerate(_keys)}
        else:
            avg_loss = epoch_loss / n_samples
            avg_components = {k: v / n_samples for k, v in epoch_components.items()}
        loss_history.append(avg_loss)

        if (epoch % config.log_every == 0 or epoch == n_epochs_total - 1) and (
            dp_rank == 0
        ):
            elapsed = time.time() - t0
            _lead_tag = (f" [lead={phase_spec[0]}h]"
                         if phase_spec[0] is not None else "")
            _extra = " ".join(
                f"{k}={avg_components[k]:.4f}"
                for k in sorted(avg_components)
                if k not in ("mse", "bias", "crps", "spec_crps"))
            logger.info(
                f"Epoch {epoch:4d}{_lead_tag}: loss={avg_loss:.6f} "
                f"(mse={avg_components['mse']:.4f} "
                f"bias={avg_components['bias']:.4f} "
                f"crps={avg_components['crps']:.4f} "
                f"spec_crps={avg_components['spec_crps']:.4f}"
                + (f" {_extra}" if _extra else "") + "), "
                f"grad_norm={grad_norm_val:.6e}, time={elapsed:.1f}s"
            )

        # Save a per-epoch checkpoint so the chained-resubmit driver
        # (run_aimip.py --resume) can pick up from epoch+1 if SLURM
        # walltime kills the job mid-training.  Rank 0 only under DP (identical
        # replicated model on every rank).
        from legoesm.ml.training import save_checkpoint
        ckpt_dir = Path(config.checkpoint_dir)
        if dp_rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.eqx"
            save_checkpoint(model, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            if ema_model is not None:
                ema_path = ckpt_dir / f"epoch_{epoch:04d}_ema.eqx"
                save_checkpoint(ema_model, ema_path)
                logger.info(f"Saved EMA checkpoint: {ema_path}")

        # AIMIP-style early stopping.  Stop when the rolling loss has
        # not improved by more than ``early_stop_min_delta`` for
        # ``early_stop_patience`` consecutive epochs.  DISABLED under a
        # rollout curriculum: loss magnitudes are not comparable across
        # phases (longer lead = bigger loss) and a stop would silently
        # skip the remaining — most stability-critical — phases; phase
        # epoch counts are the explicit budget instead.
        if early_stop_patience > 0 and not curriculum:
            if best_loss - avg_loss > early_stop_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(
                        f"Early stop at epoch {epoch}: no improvement "
                        f"> {early_stop_min_delta} for "
                        f"{early_stop_patience} consecutive epochs "
                        f"(best={best_loss:.6f}, last={avg_loss:.6f})."
                    )
                    break

    return model, loss_history


def _prefetch_iter(gen, buffer=1):
    """Run ``gen`` on a background host thread, at most ``buffer`` items ahead.

    A drop-in wrapper that overlaps the producer (chunk load: GCS read + regrid)
    with the consumer (training). Yields items in the SAME order the underlying
    generator produces them — the resume/ordering contract of ``_chunks`` is
    preserved exactly. A producer exception is re-raised on the consumer side
    (after the items already buffered), so a failed chunk load is not swallowed.

    RAM bound: the producer must ACQUIRE a permit before it calls ``next(gen)``,
    so it never loads more than ``buffer`` chunks ahead of the one the consumer
    holds — peak footprint is ``buffer + 1`` chunks (2 for the default), NOT the
    3 a plain ``Queue(maxsize=buffer)`` would reach (it eagerly loads one more
    before blocking on the full queue). The consumer releases a permit each time
    it takes an item.

    Cleanup: if the consumer stops early (``break`` / an exception in the
    training loop), the ``finally`` sets a stop flag, releases a permit, and
    drains one slot so a producer parked in ``acquire``/``put`` wakes and exits
    instead of leaking a blocked thread holding chunk memory.

    ponytail: stdlib threading + a bounded Queue + a load-gating semaphore; no
    executor pool, no asyncio. Ceiling: the cleanup runs from the generator's
    ``finally``, which fires when the iterator is closed/GC'd. The training loop
    consumes to exhaustion or the process exits, so this always fires there; a
    caller that ``break``s and then indefinitely RETAINS the live iterator would
    leave the producer parked — bounded harmless because the thread is a daemon
    (it never blocks process exit). Upgrade to an explicit context manager if a
    caller ever needs deterministic mid-iteration teardown.
    """
    import queue
    import threading

    buffer = max(1, int(buffer))                # buffer=0 would deadlock acquire
    q: "queue.Queue" = queue.Queue(maxsize=buffer)
    load_permit = threading.Semaphore(buffer)   # permits to LOAD the next item
    stop = threading.Event()
    _DONE = object()
    it = iter(gen)

    def _safe_put(msg):
        # Block for backpressure while the consumer is live, but NEVER block
        # forever: once the consumer has left (stop set) a full queue means
        # nobody will drain it, so drop the message rather than hang the thread.
        while not stop.is_set():
            try:
                q.put(msg, timeout=0.2)
                return
            except queue.Full:
                continue
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass

    def _produce():
        try:
            while True:
                load_permit.acquire()           # wait for room BEFORE loading
                if stop.is_set():
                    return
                try:
                    item = next(it)             # the chunk load happens here
                except StopIteration:
                    break
                except BaseException as exc:     # propagate load failure
                    _safe_put((None, exc))
                    return
                _safe_put((item, None))
        finally:
            _safe_put((_DONE, None))

    t = threading.Thread(target=_produce, name="chunk-prefetch", daemon=True)
    t.start()
    try:
        while True:
            item, exc = q.get()
            if exc is not None:
                raise exc
            if item is _DONE:
                return
            # Release BEFORE yielding (not after): the consumer has taken this
            # item, so the producer may load the next one WHILE the consumer
            # trains on this one — that overlap is the whole point. Releasing
            # after the yield would keep the producer blocked during training.
            load_permit.release()
            yield item
    finally:
        # Unblock a producer parked in acquire (permit) or put (drain a slot) so
        # it can observe stop and exit rather than leak.
        stop.set()
        load_permit.release()
        try:
            q.get_nowait()
        except queue.Empty:
            pass


def _stage_tree(tree, device):
    """``device_put`` only the ARRAY leaves of ``tree`` onto ``device``, leaving
    None / metadata leaves untouched.

    A chunk is ``(ic_states, targets, forcings)`` of pytrees whose leaves are
    mostly arrays but not exclusively (``forcings`` may be ``None``); a bare
    ``jax.device_put(tree, device)`` chokes on a non-array leaf, so filter to
    arrays via ``eqx.is_array``. Used to host-stage prefetched chunks (#985).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, tree
    )


def _make_chunk_loader(config, grid, sigma, cache_dir,
                       surface_forcing_path, forcing_cache_path):
    """Streaming/chunked data source for DENSE all-years training.

    Partitions ``config.windows`` into groups of ``config.chunk_windows``
    windows; each chunk is loaded (ERA5 zarr read + regrid), trained on,
    and freed before the next — so a dense sample set that would not fit
    in host RAM as one list (ACE2 trains on every 6-hour snapshot of
    decades; one T63 snapshot-carry is ~MBs) streams through a bounded
    footprint. Chunks share sample shapes, so the jitted train step is
    reused across chunks (no retrace).

    Returns ``(chunk_loader, n_samples_total)`` for
    ``_train_spectral_loop(chunk_loader=..., n_samples_total=...)``.
    """
    windows = [tuple(int(x) for x in w[:3]) for w in (config.windows or ())]
    if not windows:
        raise ValueError("chunk_windows > 0 requires config.windows.")
    cw = int(config.chunk_windows)
    groups = [tuple(windows[i:i + cw]) for i in range(0, len(windows), cw)]
    # Pairs per window = n_days * snapshots/day; the loader's ERA5 cadence
    # is fixed at 6 h (TrainingERA5Config(dt_hours=6)) -> 4/day.
    _SNAPSHOTS_PER_DAY = 4
    n_total = sum(n_days * _SNAPSHOTS_PER_DAY for (_, _, n_days) in windows)
    logger.info(
        f"Chunked training data: {len(windows)} windows -> "
        f"{len(groups)} chunks of <= {cw} windows "
        f"({n_total} samples/epoch)"
    )

    prefetch = bool(getattr(config, "chunk_prefetch", False))
    # Prefetch host-staging (#985): the background producer BUILDS the buffered
    # (next) chunk directly on the CPU -- via ``jax.default_device`` in _load_group
    # below -- instead of building it on the GPU and copying it down. That
    # distinction is the fix: a T106 4xA100 run OOM'd because the prefetched chunk
    # rode GPU memory during the current chunk's training peak, and a post-hoc
    # device_put(cpu) does NOT help (the arrays are constructed on the GPU first).
    # ``jax.default_device`` is thread-local, so the producer thread builds on the
    # CPU while the main thread trains on the GPU. Each SAMPLE is then moved onto
    # the compute device just before its step (the training loop), so a prefetched
    # chunk never rides GPU memory. Needs the JAX CPU backend -- add 'cpu' to
    # JAX_PLATFORMS under a CUDA-only launch; if it is unavailable we warn and
    # fall back to building on the compute device (a roomy config is unchanged).
    _host_dev = None
    if prefetch:
        # Shared preflight (#1155 refactor): one copy of the cpu-backend
        # resolution + warning for every host-build site. Prefetch keeps the
        # warn-and-fallback behavior (a chunk is bounded, unlike the
        # non-chunked full-dataset loads, which fail fast instead).
        _host_dev = host_build_device("chunk_prefetch host staging")

    def _load_group(group):
        # Build on the CPU when host-staging (thread-local default device), so the
        # buffered chunk is never constructed on the GPU (#985).
        _ctx = (jax.default_device(_host_dev) if _host_dev is not None
                else contextlib.nullcontext())
        with _ctx:
            ics, tgts, times = load_training_data(
                config, grid, sigma, cache_dir, windows=group,
            )
            forcings = _maybe_build_sample_forcings(
                surface_forcing_path, times, grid, forcing_cache_path,
            )
        return ics, tgts, forcings

    def _chunks(start_chunk=0):
        # ``start_chunk`` skips (does NOT load) the first N groups so a
        # mid-epoch resume never re-streams the already-trained chunks.
        # The partition ``groups`` is a fixed, unshuffled slice of
        # ``config.windows`` and ``load_training_data`` reads snapshots in
        # deterministic time order, so chunk k is byte-identical across
        # runs -> a resume replays the exact same trajectory.
        def _serial():
            for gi, group in enumerate(groups):
                if gi < start_chunk:
                    continue
                yield _load_group(group)

        # Prefetch preserves order + the start_chunk skip exactly (it only wraps
        # the same _serial generator), so the resume contract is unchanged.
        yield from (_prefetch_iter(_serial()) if prefetch else _serial())

    # True when chunks are built host-staged (prefetch + CPU backend): the
    # training loop must then device_put each SAMPLE onto the compute device at
    # the point of use (#985).
    _chunks.host_staged = _host_dev is not None
    # Number of chunks per epoch (constant): the mid-epoch checkpoint reads
    # this to normalise the last chunk of an epoch to the next epoch's start.
    _chunks.n_chunks = len(groups)
    # Per-chunk sample counts (deterministic from the window partition): the DP
    # path sizes the LR schedule by the exact per-rank update count and guards
    # against a chunk smaller than the world size (#985).
    _chunks.chunk_sizes = [
        sum(nd * _SNAPSHOTS_PER_DAY for (_, _, nd) in g) for g in groups
    ]
    return _chunks, n_total


def _maybe_build_sample_forcings(surface_forcing_path, ic_times, grid,
                                 forcing_cache_path):
    """Build per-sample prescribed surface forcing, or None when off.

    Deferred import: ``aimip_amip_forcing`` pulls xarray/scipy at call
    time only, keeping the unforced training path import-light.
    """
    if surface_forcing_path is None:
        return None
    from legoesm.training.aimip_amip_forcing import build_amip_sample_forcings
    forcings = build_amip_sample_forcings(
        ic_times, grid, forcing_path=surface_forcing_path,
        cache_path=forcing_cache_path,
    )
    logger.info(
        f"Prescribed surface forcing ACTIVE for {len(forcings)} samples "
        f"({surface_forcing_path})"
    )
    return forcings


def train_neural_gcm_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
    *,
    resume_from_dir=None,
    surface_forcing_path: str | None = None,
    forcing_cache_path: str | None = None,
):
    """Train NeuralGCM: SFNO physics + spectral PE dycore.

    Parameters
    ----------
    resume_from_dir : str | Path | None
        If set, scan this directory for the highest-numbered
        ``epoch_NNNN.eqx`` checkpoint and continue training from the
        next epoch.  Used by ``scripts/run/run_aimip.py --resume`` for
        chained-resubmission SLURM jobs.
    surface_forcing_path : str | None
        Path to the AIMIP monthly SST/sea-ice forcing NetCDF.  When
        set, per-sample prescribed surface forcing (T_sfc over ocean,
        sea-ice fraction, orbital calendar) is fed to the SFNO's
        forcing input planes during training — REQUIRED for a network
        that must respond to prescribed SST at AMIP inference time.
        ``None`` trains unforced (physics_fn internal proxies).
    forcing_cache_path : str | None
        Optional .npz cache for the regridded monthly forcing.

    Returns (trained_sfno, loss_history).
    """
    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    spec = PE3DChannelSpec(nlev=config.n_levels)
    # Channel layout, forcing planes and residual_prediction=False come from
    # the shared registry, so a WB run and an AIMIP run of this variant build
    # the same network (they had drifted to 256/8 vs 128/4).
    from legoesm.training.model_registry import build_variant
    sfno = build_variant(
        "sfno_physics", nlev=config.n_levels, grid=grid, seed=seed,
        overrides={
            "sfno_embed_dim": config.sfno_embed_dim,
            "sfno_n_blocks": config.sfno_n_blocks,
            "sfno_mlp_expansion": config.sfno_mlp_expansion,
        },
    )
    n_p = sum(x.size for x in jax.tree.leaves(eqx.filter(sfno, eqx.is_array)))
    logger.info(f"SFNO: {spec.n_channels}ch, {config.sfno_embed_dim}d, "
                f"{config.sfno_n_blocks} blocks, {n_p:,} params")

    sfno, start_epoch = maybe_resume_model(sfno, resume_from_dir)

    if int(config.chunk_windows or 0) > 0:
        chunk_loader, n_total = _make_chunk_loader(
            config, grid, sigma, cache_dir,
            surface_forcing_path, forcing_cache_path,
        )
        return _train_spectral_loop(
            sfno, make_sfno_spectral_physics,
            grid, sigma, None, None, config,
            start_epoch=start_epoch,
            chunk_loader=chunk_loader, n_samples_total=n_total,
            resume_from_dir=resume_from_dir,
        )

    ic_states, target_carries, ic_times = load_training_data(
        config, grid, sigma, cache_dir, windows=config.windows,
        host_resident=True,   # non-chunked full-dataset load (#1155)
    )
    sample_forcings = _maybe_build_sample_forcings(
        surface_forcing_path, ic_times, grid, forcing_cache_path,
    )

    return _train_spectral_loop(
        sfno, make_sfno_spectral_physics,
        grid, sigma, ic_states, target_carries, config,
        start_epoch=start_epoch,
        sample_forcings=sample_forcings,
        host_staged=True,   # dataset loaded host-resident above (#1155)
        resume_from_dir=resume_from_dir,   # EMA resume needs the dir too
    )


# Minimum per-channel std for the sfno_full Z-score normalization.  A constant
# channel (e.g. a spatially/temporally invariant surface field) has std 0, so
# ``(x - mean)/std`` would divide by zero; the shared
# ``compute_normalization_stats`` floors the VARIANCE at ``eps**2`` (finite
# forward AND gradient — see its docstring), so this ``eps`` is the std floor.
# 1e-6 is a numerics guard, not a tunable physics coefficient.
_SFNO_FULL_STD_FLOOR = 1.0e-6  # coeff-ok: div-by-zero guard for constant channels


def compute_sfno_full_norm_stats(ic_states, grid, sigma, std_floor=_SFNO_FULL_STD_FLOOR):
    """Per-channel Z-score stats for the sfno_full emulator, in EMULATOR order.

    The emulator normalises the packed PE channel tensor produced by
    :func:`legoesm.ml.channel_packing.pack_pe_state` (layout
    ``[u(nlev), v(nlev), T(nlev), q(nlev), lnps, phis]``), whose channels span
    ~8 orders of magnitude (surface pressure ~1e5 Pa vs specific humidity
    ~1e-3).  Training on the RAW tensor makes the network output explode
    (epoch-0 NaN).  This computes per-channel mean/std over the training ICs in
    that SAME packed order, so training and eval apply an identical transform.

    Each spectral ``SpectralHydrostaticState`` IC is packed to grid space
    ``(n_lat, n_lon, n_channels)``; statistics are taken over all sample +
    spatial cells (unweighted — the SFNO operates on the Gaussian grid
    uniformly).  Samples are STREAMED (two passes, one packed sample resident
    at a time) rather than stacked: a single on-device stack of all ICs is
    ~16 GiB at T106 all-years and OOMs the load-time stats step (#1155-class
    invariant: never materialise the full dataset on-device).  The variance is
    floored exactly as :func:`legoesm.ml.normalization.compute_normalization_stats`
    does (``var >= std_floor**2``) so a constant channel yields
    ``std == std_floor`` with a finite gradient (no div-by-zero, no ``0 * inf``
    NaN in the backward pass).  Numerically equivalent to the stacked
    two-pass form up to float summation order.

    Parameters
    ----------
    ic_states : list[SpectralHydrostaticState]
        Training initial conditions (spectral space); packed here.
    grid : GaussianGrid
    sigma : SigmaCoordinate
        Passed to ``pack_pe_state`` positionally for call-site symmetry
        (packing is on the model sigma levels; the arg is otherwise unused).
    std_floor : float
        Minimum per-channel std (div-by-zero guard).

    Returns
    -------
    NormalizationStats
        ``mean``/``std`` of shape ``(n_channels,)`` in packed-channel order.
    """
    import numpy as np
    from legoesm.ml.normalization import NormalizationStats

    if not ic_states:
        raise ValueError(
            "compute_sfno_full_norm_stats: ic_states is empty; cannot compute "
            "normalization statistics with no training data."
        )

    def _packed_f64(s):
        # One packed grid-space sample (n_lat, n_lon, n_ch), f64 to match the
        # spectral-transform precision and keep the streamed reduction stable.
        return jnp.asarray(pack_pe_state(s, grid, sigma), dtype=jnp.float64)

    # Pass 1: per-channel mean over sample + spatial cells. Accumulate the
    # channel-axis-preserving sum one sample at a time (peak device footprint:
    # one packed sample, ~14 MiB at T106, vs ~16 GiB for the full stack).
    ch_sum = None
    cell_count = 0
    for s in ic_states:
        p = _packed_f64(s)
        contrib = jnp.sum(p, axis=tuple(range(p.ndim - 1)))   # (n_ch,)
        ch_sum = contrib if ch_sum is None else ch_sum + contrib
        cell_count += int(np.prod(p.shape[:-1]))
    mean = ch_sum / cell_count

    # Pass 2: variance ABOUT THE MEAN (matches compute_normalization_stats'
    # mean((data-mean)**2), not the cancellation-prone E[x^2]-E[x]^2 form).
    ch_sqsum = None
    for s in ic_states:
        p = _packed_f64(s)
        contrib = jnp.sum((p - mean) ** 2, axis=tuple(range(p.ndim - 1)))
        ch_sqsum = contrib if ch_sqsum is None else ch_sqsum + contrib
    var = ch_sqsum / cell_count

    # Floor the VARIANCE before sqrt (identical convention to
    # compute_normalization_stats): var==0 -> d(sqrt)/dvar = inf -> 0*inf NaN
    # in the backward pass; clamping to std_floor**2 gives std>=std_floor with
    # a finite gradient.
    var_floor = float(std_floor) ** 2
    std = jnp.sqrt(jnp.maximum(var, var_floor))
    return NormalizationStats(mean=mean, std=std)


def train_sfno_full_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
    dt_sfno: float = 21600.0,
    *,
    resume_from_dir=None,
):
    """Train SFNO as a full atmospheric emulator (no dycore).

    Unlike :func:`train_neural_gcm_spectral` (where SFNO produces
    physics-tendency increments that the spectral PE dycore advances),
    here SFNO directly maps ``state_t -> state_{t+1}`` at the macro
    step ``dt_sfno`` (default 6 h, matching the standard SFNO/FourCastNet
    autoregressive cadence).  The wrapper
    :class:`SFNOPrimitiveEquationModel` applies post-hoc dry-air-mass
    and moisture-budget corrections after each step.

    The training loop mirrors :func:`_train_spectral_loop` (multi-step
    autoregressive supervision, bias / CRPS / spectral-CRPS loss,
    per-epoch checkpointing and early stopping) but uses
    ``model.step()`` for rollouts instead of ``spectral_rollout``.
    Segment lengths in the multi-step schedule are computed in SFNO
    macro-step units (``dt_sfno``), not the dycore micro-step
    (``config.dt``), so a 24 h lead at ``dt_sfno=6 h`` is 4 SFNO steps.

    Returns (trained_sfno, loss_history).
    """
    from legoesm.atmosphere.dynamics.neural.sfno_pe import (
        SFNOPrimitiveEquationConfig,
    )

    # FAIL EARLY on an invalid U-Cast stage-2 config: everything below this
    # line (host-resident ERA5 load, norm stats, sidecars) costs hours, and
    # the loop re-validates anyway.
    #
    # The epoch budget MUST be the loop's, i.e. len(epoch_plan) from the SAME
    # shared planner — not a hand-rolled sum. ``sum(ep for _, ep in rc)`` is
    # wrong for a curriculum containing a zero-epoch stage (the planner skips
    # it and falls back to n_epochs), which would reject a valid stage 2 here
    # and accept an invalid one there (codex review 2026-08-02).
    _resolve_crps_finetune(
        config,
        len(build_sfno_curriculum_epoch_plan(
            getattr(config, "rollout_curriculum", None),
            tuple(int(h) for h in (config.loss_config.multi_step_hours or ())),
            dt_sfno,
            config.n_epochs,
        )),
        config.loss_config,
    )

    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    spec = PE3DChannelSpec(nlev=config.n_levels)
    # in_channels carries (1 + history) copies of the packed state; the emulator
    # wrapper validates the two agree rather than mis-slicing.
    _history_steps = int(getattr(config, "sfno_history_steps", 0) or 0)
    # Shared registry (same source as the WB lane and as sfno_physics); the
    # arch config is read back so the emulator wrapper below cannot disagree
    # with the weights it wraps about channel counts.
    from legoesm.training.model_registry import (
        build_variant as _build_variant,
        sfno_arch_config as _sfno_arch_config,
    )
    _sfno_over = {
        "sfno_embed_dim": config.sfno_embed_dim,
        "sfno_n_blocks": config.sfno_n_blocks,
        "sfno_mlp_expansion": config.sfno_mlp_expansion,
        "sfno_dropout": config.sfno_dropout,
        "sfno_history_steps": _history_steps,
    }
    sfno_arch_cfg = _sfno_arch_config(
        "sfno_full", nlev=config.n_levels, overrides=_sfno_over)
    sfno = _build_variant("sfno_full", nlev=config.n_levels, grid=grid,
                          seed=seed, overrides=_sfno_over)
    n_p = sum(x.size for x in jax.tree.leaves(eqx.filter(sfno, eqx.is_array)))
    logger.info(
        f"SFNO full-emulator: {spec.n_channels}ch, {config.sfno_embed_dim}d, "
        f"{config.sfno_n_blocks} blocks, {n_p:,} params, "
        f"dt_sfno={dt_sfno:.0f}s ({dt_sfno/3600:.1f}h macro step)"
    )

    pe_emulator_cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_arch_cfg,
        mode="state_update",
        dt_sfno=dt_sfno,
        history_steps=_history_steps,
        correct_mass=True,
        # Not yet wired in the SFNO PE bridge (spectral moisture tracer needs
        # synthesis/clip/re-analysis); previously silently ignored.
        correct_moisture_budget=False,
        # ACE2 corrector step 1 (arXiv:2411.11268 §4.3): negative moisture
        # clamped to zero INSIDE the model, before the loss.
        clip_q=True,
        # Post-step damping of the vor/div cascade that spectral
        # differentiation of the network's u,v feeds each macro step. Read
        # from the SAME pe_config the dycore rollout uses (never a private
        # literal), which run_aimip._build_spectral_config derives from the
        # merged suite via the spectral_filter_strength / spectral_filter_order
        # keys (defaults 0.01 / 8). The WB2 eval bridge rebuilds that same
        # pe_config from the same suite, so the lanes cannot drift AND a suite
        # can actually override the damping. Without it a trained T63
        # checkpoint reaches 969 m/s winds by macro step 4 and NaNs at step
        # 12; see the SFNOPrimitiveEquationConfig docstring for the measured
        # per-step trace.
        spectral_filter_strength=config.pe_config.spectral_filter_strength,
        spectral_filter_order=config.pe_config.spectral_filter_order,
        # Per-channel Z-score normalization ON: the raw PE channel tensor spans
        # ~8 orders of magnitude (p_s ~1e5 Pa vs q ~1e-3), which drove the
        # epoch-0 NaN.  norm_stats are computed from the training ICs below and
        # persisted to a sidecar so the WB2 eval bridge applies the SAME
        # transform (state_update denormalises the network output as a full
        # state — correct, see sfno_pe.py:129-140,395-396).
        use_normalization=True,
    )

    sfno, start_epoch = maybe_resume_model(sfno, resume_from_dir)

    # Host-resident dataset (#1155): sfno_full loads ALL pairs up front (no
    # chunking) — at T106 all-years that is ~130 GB, an unconditional GPU OOM
    # if built on the compute device. Build on host; the loop stages each
    # sample to the GPU just before its step.
    ic_states, target_carries, _ic_times = load_training_data(
        config, grid, sigma, cache_dir, windows=config.windows,
        host_resident=True,
    )

    # Data-driven per-channel stats in packed-emulator order (same channel
    # layout the wrapper sees via pack_pe_state).  A resumed run recomputes
    # identical stats from the same deterministic windows and overwrites the
    # sidecar idempotently.
    # NORMALIZATION IS PART OF THE MODEL'S INTERFACE, not of the current
    # dataset. Recomputing it on resume silently moves the input coordinate
    # system out from under already-trained weights — which is exactly what a
    # FINE-TUNE on different years does: the OOD run (2015-2016, resumed from a
    # 1979-2014 base) recomputed stats from the new years, so the frozen weights
    # saw a different transform than they were trained in, and the loss opened
    # at 7.84 against the base's 6.30 (codex review 2026-07-31).
    #
    # So: if a sidecar already exists in the checkpoint dir AND we are resuming,
    # REUSE it. For an ordinary chained link this is a no-op (the same windows
    # give the same stats); for a fine-tune it is what keeps the model's input
    # mapping fixed.
    _stats_existing = Path(config.checkpoint_dir) / "norm_stats.npz"
    if start_epoch > 0 and _stats_existing.exists():
        from legoesm.ml.normalization import load_normalization_stats

        norm_stats = load_normalization_stats(_stats_existing)
        logger.info(
            "SFNO full-emulator: REUSING existing norm stats %s (resume/"
            "fine-tune) — recomputing them would change the model's input "
            "coordinate system under trained weights.", _stats_existing)
    else:
        norm_stats = compute_sfno_full_norm_stats(ic_states, grid, sigma)

    # Persist the stats next to the checkpoints (config.checkpoint_dir ==
    # {output_dir}/{aimip_variant}).  The eval bridge reloads norm_stats.npz
    # from this same directory so train + eval share ONE source of truth (the
    # eqx checkpoint serialises only the SFNO leaves, NOT the wrapper's
    # norm_stats — so a sidecar, not the checkpoint, carries the values).
    # Computed from the FULL (unsharded) window set on every rank, so the values
    # are identical everywhere — but N concurrent writers to one path can still
    # tear the file, so only rank 0 writes it.
    from legoesm.ml.normalization import save_normalization_stats
    _dp_on_stats, _rank_stats, _, _ = _resolve_dp_context(config)
    ckpt_dir = Path(config.checkpoint_dir)
    stats_path = ckpt_dir / "norm_stats.npz"
    if ((not _dp_on_stats) or _rank_stats == 0) and not (
            start_epoch > 0 and stats_path.exists()):
        # Never overwrite a sidecar that trained weights already depend on.
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_normalization_stats(norm_stats, stats_path)
    logger.info(
        f"SFNO full-emulator: saved per-channel norm stats "
        f"({norm_stats.mean.shape[-1]} channels) to {stats_path}"
    )

    # Architecture sidecar. The eqx checkpoint carries only the SFNO LEAVES,
    # and ``dropout`` is a STATIC field, so a checkpoint trained at p=0.1 loads
    # without error into a p=0.0 skeleton and vice versa — silently, because
    # the leaf layout is identical by construction. Everything else about the
    # architecture (embed_dim, n_blocks) would fail loudly on a shape mismatch;
    # the dropout rate would not. Record it so eval can refuse a mismatch
    # instead of reporting a miscalibrated ensemble (codex review 2026-08-01).
    #
    # NEVER overwritten on resume: a chained link that silently adopted a
    # changed ``sfno_dropout`` would leave every earlier epoch checkpoint in
    # this directory described by a sidecar that is false for them. A genuine
    # architecture change mid-run is a mistake, so it is refused, not recorded.
    arch_path = ckpt_dir / "sfno_arch.json"
    import json as _json

    _arch_record = {
        "in_channels": int(sfno_arch_cfg.in_channels),
        "out_channels": int(sfno_arch_cfg.out_channels),
        "embed_dim": int(sfno_arch_cfg.embed_dim),
        "n_blocks": int(sfno_arch_cfg.n_blocks),
        "mlp_expansion": int(sfno_arch_cfg.mlp_expansion),
        "residual_prediction": bool(sfno_arch_cfg.residual_prediction),
        "dropout": float(sfno_arch_cfg.dropout),
    }
    if arch_path.exists():
        _prev = _json.loads(arch_path.read_text())
        _changed = {
            k: (_prev[k], v) for k, v in _arch_record.items()
            if k in _prev and _prev[k] != v
        }
        _missing = sorted(set(_arch_record) - set(_prev))
        if _missing:
            logger.warning(
                "sfno_arch.json at %s predates fields %s; they cannot be "
                "verified against these checkpoints.", arch_path, _missing)
        if _changed:
            raise ValueError(
                f"architecture changed mid-run: {arch_path} records "
                f"{ {k: a for k, (a, _) in _changed.items()} } but this run "
                f"was configured with { {k: b for k, (_, b) in _changed.items()} }. "
                f"The checkpoints in this directory were trained with the "
                f"recorded architecture. Start a NEW output_dir for the new "
                f"configuration."
            )
    elif start_epoch > 0:
        # RESUMING a directory that predates the sidecar. Writing one now would
        # assert THIS run's architecture over checkpoints trained by an unknown
        # one — permanently mislabelling a pre-dropout checkpoint as
        # dropout-trained, which is precisely the guard being bypassed (codex
        # review 2026-08-01). Stay silent on disk, loud in the log.
        logger.warning(
            "resuming %s with no sfno_arch.json: the architecture of the "
            "existing checkpoints CANNOT be verified, and none will be "
            "recorded (this run configures dropout=%.3g). Evaluations of "
            "this directory are UNVERIFIED against their training "
            "architecture.", ckpt_dir, float(sfno_arch_cfg.dropout))
    elif (not _dp_on_stats) or _rank_stats == 0:
        # Atomic publish: other ranks test exists() then read immediately, so a
        # plain write_text can hand them a truncated file. os.replace is atomic
        # within a filesystem, so a reader sees either no file or a whole one.
        import os as _os

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _tmp = arch_path.with_suffix(f".json.tmp{_rank_stats}")
        _tmp.write_text(_json.dumps(_arch_record, indent=1))
        _os.replace(_tmp, arch_path)

    return _train_sfno_full_loop(
        sfno, pe_emulator_cfg, grid, sigma,
        ic_states, target_carries, config, dt_sfno,
        start_epoch=start_epoch,
        norm_stats=norm_stats,
        seed=seed,
    )


def build_sfno_curriculum_epoch_plan(
    rollout_curriculum,
    multi_step_hours,
    dt_sfno: float,
    n_epochs_fallback: int,
):
    """Flat per-epoch plan for the sfno_full rollout curriculum.

    Pure (jax-free) mirror of the epoch-plan build in
    :func:`_train_spectral_loop` (the dycore-mode curriculum, see
    neural_gcm_spectral.py `epoch_plan` at the ``if curriculum:`` block), but
    expressed in SFNO **macro** steps (``dt_sfno``) instead of dycore
    micro-steps (``config.dt``).

    Each NeuralGCM-style phase ``(lead_hours, n_epochs)`` supervises a single
    autoregressive rollout to ``lead_hours`` scored against the ERA5 target at
    that lead.  Targets for every phase lead are loaded up front, one per entry
    of ``multi_step_hours`` (``load_training_data`` returns a tuple of K carries
    per sample when ``multi_step_hours`` is set), so a phase's target is indexed
    by ``k_target = multi_step_hours.index(lead_hours)``.

    Parameters
    ----------
    rollout_curriculum : sequence of ``(lead_hours, n_epochs)`` or None
        Short-lead-first phases.  ``None`` (or empty) -> the caller keeps the
        fixed chained-multi-step behaviour; this returns a fallback plan of
        ``n_epochs_fallback`` no-op ``(None, None, None)`` entries so the epoch
        loop can index it uniformly.
    multi_step_hours : sequence of int
        The sorted set of loaded target leads (``loss_config.multi_step_hours``).
        EVERY curriculum lead must be a member, else its target was never loaded.
    dt_sfno : float
        SFNO macro (autoregressive) time step [s].  ``n_sfno_steps`` for a lead
        is ``round(lead_hours * 3600 / dt_sfno)`` (the same hours->steps
        conversion the dycore-mode plan uses with ``config.dt``).
    n_epochs_fallback : int
        Length of the no-op plan returned when ``rollout_curriculum`` is falsy.

    Returns
    -------
    list of ``(lead_hours, k_target, n_sfno_steps)``
        One entry per GLOBAL epoch (phase ``(h, ep)`` contributes ``ep`` copies
        of its spec).  Resume (``start_epoch``) indexes into this list.  When
        ``rollout_curriculum`` is falsy: ``[(None, None, None)] * n_epochs_fallback``.

    Raises
    ------
    ValueError
        If ``multi_step_hours`` is empty while a curriculum is given (no targets
        loaded), or if any curriculum lead is not in ``multi_step_hours``
        (its target was never loaded).  Same message style as the dycore-mode
        curriculum validation in :func:`_train_spectral_loop`.
    """
    # Delegates to the shared curriculum module; require_exact enforces the
    # whole-macro-step rule (a misaligned lead would otherwise silently
    # supervise the wrong horizon — see curriculum.build_curriculum_epoch_plan).
    return build_curriculum_epoch_plan(
        rollout_curriculum,
        multi_step_hours,
        dt_sfno,
        int(n_epochs_fallback),
        require_exact=True,
        dt_name="dt_sfno",
    )


@mpi_abort_on_uncaught  # a rank dying (OOM/NaN) MPI_Aborts the job
# instead of leaving its peers blocked forever in the next allreduce.
def _train_sfno_full_loop(
    sfno: SFNO,
    pe_emulator_cfg,
    grid: GaussianGrid,
    sigma: SigmaCoordinate,
    ic_states,
    target_carries,
    config: NeuralGCMSpectralConfig,
    dt_sfno: float,
    *,
    start_epoch: int = 0,
    host_staged: bool = True,
    norm_stats=None,
    seed: int = 0,
):
    """Training loop for SFNO full-atmosphere emulator (no dycore).

    ``host_staged`` (default True — the caller loads host-resident, #1155):
    each sample is moved to the compute device just before its step, so the
    full dataset never rides GPU memory.

    Parallel to :func:`_train_spectral_loop` but uses
    ``SFNOPrimitiveEquationModel.step(state, dt_sfno)`` for rollout
    instead of the dycore + physics-tendency assembly.  The multi-step
    segment schedule is reused verbatim from ``config.loss_config``,
    but segment lengths are converted into SFNO macro steps.
    """
    from legoesm.atmosphere.dynamics.neural.sfno_pe import (
        SFNOPrimitiveEquationModel,
    )
    from legoesm.ml.training import TrainingConfig, create_optimizer

    # Fail early (before the JIT trace) if normalization is requested without
    # stats — the wrapper raises the same contract deep in loss_fn otherwise.
    if pe_emulator_cfg.use_normalization and norm_stats is None:
        raise ValueError(
            "_train_sfno_full_loop: pe_emulator_cfg.use_normalization=True but "
            "norm_stats is None. Pass norm_stats (see "
            "compute_sfno_full_norm_stats) or set use_normalization=False."
        )

    sigma_full = jnp.asarray(sigma.sigma_full)
    loss_history = []

    # Multi-step segment schedule (same lead set as the dycore-mode
    # training, but expressed in SFNO macro steps).
    loss_cfg_train = config.loss_config
    multi_step_hours_train = tuple(
        int(h) for h in (loss_cfg_train.multi_step_hours or ())
    )
    if multi_step_hours_train:
        prev = 0
        segment_steps = []
        for h in multi_step_hours_train:
            n_sfno = int(round((h - prev) * 3600.0 / dt_sfno))
            if n_sfno <= 0:
                raise ValueError(
                    f"Non-positive SFNO segment derived from "
                    f"multi_step_hours={multi_step_hours_train}, "
                    f"dt_sfno={dt_sfno}s.  Use a smaller dt_sfno or "
                    f"larger leads."
                )
            segment_steps.append(n_sfno)
            prev = h
        ms_weights_cfg = tuple(loss_cfg_train.multi_step_weights or ())
        if ms_weights_cfg and len(ms_weights_cfg) != len(segment_steps):
            raise ValueError(
                f"multi_step_weights length {len(ms_weights_cfg)} != "
                f"len(multi_step_hours)={len(segment_steps)}."
            )
        ms_weights = (
            tuple(float(w) for w in ms_weights_cfg)
            if ms_weights_cfg else (1.0,) * len(segment_steps)
        )
        ms_weight_sum = float(sum(ms_weights))
        logger.info(
            f"SFNO full-emulator multi-step supervision: "
            f"leads={multi_step_hours_train}h "
            f"(segments={segment_steps} SFNO steps, weights={ms_weights})"
        )
    else:
        # Single-step fallback: roll out config.rollout_hours at dt_sfno.
        n_sfno_single = max(
            1,
            int(round(
                float(getattr(config, "rollout_hours", 0) or 24)
                * 3600.0 / dt_sfno
            )),
        )
        segment_steps = (n_sfno_single,)
        ms_weights = (1.0,)
        ms_weight_sum = 1.0
        logger.info(
            f"SFNO full-emulator single-segment supervision: "
            f"{n_sfno_single} SFNO steps "
            f"(~{n_sfno_single * dt_sfno / 3600:.1f} h)"
        )

    # --- rollout curriculum (NeuralGCM-style stability training) ---
    # When ``config.rollout_curriculum`` is set, each phase supervises ONE
    # autoregressive rollout to ``lead_hours`` scored against the ERA5 target at
    # that lead (single-lead loss), progressively longer.  Mirrors the dycore-
    # mode curriculum in ``_train_spectral_loop`` (see
    # ``build_sfno_curriculum_epoch_plan`` for the pure epoch-plan build shared
    # with the unit test), but in SFNO macro steps.  Targets for every phase
    # lead were loaded up front: the caller sets ``loss_config.multi_step_hours``
    # to the sorted set of curriculum leads so ``load_training_data`` builds a
    # target tuple per sample (``target_carry[k_target]`` picks the phase lead).
    # ``None`` keeps the fixed chained-multi-step behaviour above byte-identical.
    epoch_plan = build_sfno_curriculum_epoch_plan(
        config.rollout_curriculum,
        multi_step_hours_train,
        dt_sfno,
        config.n_epochs,
    )
    curriculum_on = bool(config.rollout_curriculum)
    n_epochs_total = len(epoch_plan)
    if curriculum_on:
        logger.info(
            "SFNO full-emulator rollout curriculum active: "
            + ", ".join(
                f"{h}h x{ep}"
                for h, ep in (
                    (int(h), int(ep)) for h, ep in config.rollout_curriculum
                )
            )
            + f" ({n_epochs_total} epochs total)"
        )

    # Optimizer schedule sized by the ACTUAL epoch count (curriculum overrides
    # config.n_epochs with sum-of-phase-epochs), so the warmup+cosine decay is
    # correct for a curriculum run (mirrors _train_spectral_loop, which sizes
    # total_steps by n_epochs_total).
    #
    # total_steps must count OPTIMIZER UPDATES, not samples. Serially those are
    # the same (one update per sample), but under data parallelism each update
    # consumes ``ranks x local_devices`` samples, so there are that many times
    # FEWER updates. Sizing the schedule by the sample count then ends the
    # cosine decay far above zero and lets ``warmup_steps`` swallow most of the
    # run — silently changing every DP trajectory (codex review 2026-07-30).
    # Resolved here, BEFORE the optimizer is built, because the sharding below
    # happens later in the function.
    _dp_probe_on, _, _dp_probe_nproc, _ = _resolve_dp_context(config)
    _dev_probe = jax.local_device_count() if bool(
        getattr(config, "data_parallel", False)) else 1
    _batch_probe = max(1, _dp_probe_nproc) * max(1, _dev_probe)
    _updates_per_epoch = max(1, (len(ic_states) // _batch_probe))
    total_steps = max(1, n_epochs_total * _updates_per_epoch)
    if _batch_probe > 1:
        logger.info(
            "SFNO full-emulator optimizer schedule: %d update(s)/epoch x %d "
            "epochs = %d total steps (global batch = %d rank(s) x %d device(s))",
            _updates_per_epoch, n_epochs_total, total_steps,
            _dp_probe_nproc, _dev_probe)
    optimizer = create_optimizer(TrainingConfig(
        lr=config.lr,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        optimizer=config.optimizer,
        # Per-branch LR / wd split for muon_partitioned (U-Cast runs Muon at
        # 10x the AdamW group's LR). All 1.0 by default -> unchanged.
        muon_lr_scale=config.muon_lr_scale,
        adamw_lr_scale=config.adamw_lr_scale,
        muon_weight_decay_scale=config.muon_weight_decay_scale,
        adamw_weight_decay_scale=config.adamw_weight_decay_scale,
    ))
    opt_state = optimizer.init(eqx.filter(sfno, eqx.is_array))

    logger.info(
        f"SFNO full-emulator training: {n_epochs_total} epochs, "
        f"{len(ic_states)} samples/epoch, dt_sfno={dt_sfno:.0f}s"
    )

    # Static Python bool (feature gating doctrine: a Python `if` on a static
    # flag, NOT jnp.where — the latter would trace BOTH loss branches).
    _persist_norm = bool(getattr(loss_cfg_train, "persistence_normalize", False))
    if _persist_norm:
        logger.info(
            "SFNO full-emulator: persistence-normalized (skill-score) loss "
            "active — L = L(pred,truth)/L(x0,truth); 1.0 == no better than "
            "persistence.")

    # Resolved HERE, not inherited: the architecture sizing computes its own
    # ``_history_steps`` inside ``train_sfno_full_spectral``, which is a
    # different function scope — reading it here was a NameError that only fired
    # once a test actually entered this loop.
    _history_steps = int(getattr(config, "sfno_history_steps", 0) or 0)

    _train_dropout = float(config.sfno_dropout) > 0.0
    # Distinct stream from the weight-init key (``PRNGKey(seed)``) so two arms
    # sharing a seed do not correlate their masks with their initialisation.
    _dropout_base_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 1)
    if _train_dropout:
        logger.info(
            "SFNO full-emulator: MC-Dropout ACTIVE during training "
            "(p=%.3f, fresh mask every macro step). Evaluation without a "
            "dropout key remains the deterministic mean forecast.",
            float(config.sfno_dropout))

    def _rollout_segment(state, model_wrapper, n_steps, key=None,
                         prev_state=None):
        """Iterate ``model_wrapper.step`` ``n_steps`` times via lax.scan.

        ``key`` carries the MC-Dropout randomness (U-Cast, arXiv:2604.09041,
        where dropout is active during training AND inference).  It is folded
        with the scan index so every macro step draws a FRESH mask — a single
        mask held across the trajectory would train the network against one
        fixed sub-network per sample instead of the dropout ensemble.  ``None``
        (the default, and the only value when ``sfno_dropout == 0``) is the
        deterministic path, bit-identical to before this argument existed.

        The scan body is ``jax.checkpoint``-wrapped (``prevent_cse=True``,
        ``policy=nothing_saveable``) — the SAME gradient-checkpoint pattern the
        dycore-mode training rollout uses (see ``spectral_rollout``'s
        ``step_fn_ckpt`` and ``run_amip_rollout``'s ``_step``). Without it a long
        curriculum rollout (e.g. 120 h = 20 SFNO macro steps at dt_sfno=6 h)
        stores every step's activations for reverse-mode AD and OOMs on a large
        SFNO. ``nothing_saveable`` recomputes every intermediate on the backward
        pass, trading compute for O(1)-per-step activation memory; the transform
        is a pure function of the differentiable SFNO weights, so gradients are
        unchanged (AD-exact). Insensitive on short (1-2 step) segments.
        """
        if _history_steps and prev_state is None:
            raise ValueError(
                f"sfno_history_steps={_history_steps} but no prev_state was "
                "supplied to _rollout_segment. The first macro step needs a "
                "REAL observed predecessor; defaulting it to the current state "
                "would feed the network a zero tendency and quietly train a "
                "different model.")
        if _history_steps == 0:
            def _body(s, i):
                step_key = None if key is None else jax.random.fold_in(key, i)
                return model_wrapper.step(s, dt_sfno, key=step_key), None

            step_ckpt = jax.checkpoint(
                _body,
                prevent_cse=True,
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            final, _ = jax.lax.scan(step_ckpt, state, jnp.arange(n_steps))
            return final

        # HISTORY MODE. The carry is (previous, current). After the first macro
        # step the "previous" state is the model's OWN prior prediction, not
        # ERA5 — which is the honest autoregressive setup: at inference nothing
        # else is available. Only step 0 sees a real observed predecessor, and
        # the caller supplies it as ``prev_state``.
        def _body_hist(carry, i):
            prev, cur = carry
            step_key = None if key is None else jax.random.fold_in(key, i)
            nxt = model_wrapper.step(
                cur, dt_sfno, key=step_key, prev_states=(prev,))
            return (cur, nxt), None

        step_ckpt = jax.checkpoint(
            _body_hist,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        (_, final), _ = jax.lax.scan(
            step_ckpt, (prev_state, state), jnp.arange(n_steps))
        return final

    def _build_wrapper(m):
        # Rebuild the wrapper inside the trace; ``grid``, ``sigma`` and
        # ``pe_emulator_cfg`` are static so the constructor introduces no new
        # array work, and the inner SFNO ``m`` is the differentiable target.
        # ``norm_stats`` is a NamedTuple of jnp arrays; it enters the trace as a
        # closed-over constant (identical stats every step). The (de)normalize
        # transform is differentiable, so gradients flow to the SFNO weights
        # unchanged (SegmentForcing doctrine: the stats are static per training,
        # not a per-iter changing arg).
        return SFNOPrimitiveEquationModel(
            grid=grid,
            sigma_coord=sigma,
            config=pe_emulator_cfg,
            sfno_model=m,
            norm_stats=norm_stats,
        )

    def _chained_loss(m, ic_spectral, target_carry, dropout_key=None):
        """Default (non-curriculum) chained multi-step loss — UNCHANGED path.

        ``dropout_key`` is folded per SEGMENT so the chained rollout does not
        reuse one mask across segments. (Named ``dropout_key``, not ``key``:
        the component loop below binds ``key``.)"""
        wrapper = _build_wrapper(m)
        state = ic_spectral
        total = jnp.float32(0.0)
        comp_total: dict = {}
        for k, n_seg in enumerate(segment_steps):
            state = _rollout_segment(
                state, wrapper, n_seg,
                key=(None if dropout_key is None
                     else jax.random.fold_in(dropout_key, k)),
            )
            if _persist_norm:
                # Skill-score loss: divide by the persistence error AT THIS
                # LEAD, so a long segment cannot dominate purely by magnitude.
                seg_loss, seg_comp = _persistence_normalized_loss_components(
                    state, ic_spectral, target_carry[k], grid, sigma,
                    sigma_full, loss_cfg_train,
                )
            else:
                seg_loss, seg_comp = _spectral_state_loss_components(
                    state, target_carry[k], grid, sigma,
                    sigma_full, loss_cfg_train,
                )
            total = total + ms_weights[k] * seg_loss
            # Key-general (mirrors the classical loop): optional per-variable
            # bias terms survive; the key set is config-static.
            for key, v in seg_comp.items():
                comp_total[key] = (
                    comp_total.get(key, jnp.float32(0.0)) + ms_weights[k] * v)
        inv = 1.0 / ms_weight_sum
        return total * inv, {k: v * inv for k, v in comp_total.items()}

    _crps_ft_epochs, _crps_M = _resolve_crps_finetune(
        config, n_epochs_total, loss_cfg_train)
    # Final-lead step count for the NON-curriculum stage-2 path, derived from
    # the lead in HOURS — never from sum(segment_steps). The segments are
    # per-increment ROUNDED step counts, so their sum only equals the final
    # lead when every increment lands exactly on the dt_sfno grid: at
    # dt_sfno=4 h, multi_step_hours=[6,12] rounds to [2,2] = 16 h against a
    # 12 h target (codex review 2026-08-02). Refuse the off-grid case with
    # the same exactness the curriculum planner enforces.
    #
    # Scoped to the NON-curriculum path: a curriculum always supplies its own
    # exact ``n_sfno_steps`` per phase, so this value is never read there and
    # rejecting an off-grid trailing target would be a spurious failure
    # (codex review 2026-08-02).
    _crps_final_lead_steps = None
    if _crps_ft_epochs > 0 and not config.rollout_curriculum:
        _final_lead_h = (multi_step_hours_train[-1] if multi_step_hours_train
                         else float(getattr(config, "rollout_hours", 0) or 24))
        _steps_f = float(_final_lead_h) * 3600.0 / dt_sfno
        _crps_final_lead_steps = int(round(_steps_f))
        if abs(_steps_f - _crps_final_lead_steps) > 1e-6 \
                or _crps_final_lead_steps < 1:
            raise ValueError(
                f"crps_finetune_epochs > 0 but the final supervised lead "
                f"({_final_lead_h} h) is not an exact multiple of "
                f"dt_sfno={dt_sfno:g}s (ratio {_steps_f}). Stage 2 would roll "
                f"to a different horizon than the target it scores. Pick "
                f"leads on the dt_sfno grid."
            )
    if _crps_ft_epochs > 0:
        logger.info(
            "SFNO full-emulator: U-Cast stage 2 — the last %d of %d epochs "
            "train on the %d-member MC-Dropout almost-fair CRPS "
            "(alpha=%.3g).", _crps_ft_epochs, n_epochs_total, _crps_M,
            float(getattr(loss_cfg_train, "afcrps_alpha", 0.95)))

    def _ensemble_crps_loss(m, ic_spectral, target_carry, k_target,
                            n_sfno_steps, dropout_key):
        """Stage-2 loss: M MC-Dropout rollouts scored by almost-fair CRPS.

        Each member gets its OWN dropout stream (``fold_in`` on the member
        index), so the members differ; the rollout itself is the same
        ``_rollout_segment`` the deterministic stage uses, which keeps the
        post-step filter/mass/clip chain identical between stages.
        """
        wrapper = _build_wrapper(m)
        tgt = (target_carry[k_target]
               if (k_target is not None and type(target_carry) is tuple)
               else (target_carry[-1] if type(target_carry) is tuple
                     else target_carry))
        n_steps = (n_sfno_steps if n_sfno_steps is not None
                   else _crps_final_lead_steps)

        def _one_member(member_key):
            return _rollout_segment(
                ic_spectral, wrapper, n_steps, key=member_key)

        member_keys = jax.random.split(dropout_key, _crps_M)
        member_states = jax.vmap(_one_member)(member_keys)
        return ensemble_crps_loss_components(
            member_states, tgt, grid, sigma, sigma_full, loss_cfg_train)

    def _curriculum_loss(m, ic_spectral, target_carry, k_target, n_sfno_steps,
                         dropout_key=None):
        """Single-lead curriculum loss: ONE rollout to ``n_sfno_steps`` scored
        against the ERA5 target at that lead.  ``target_carry`` is the K-tuple of
        carries built by ``load_training_data`` (one per multi_step_hours lead);
        ``k_target`` selects the phase lead's target.  A defensive fallback
        indexes tuple targets only (a non-tuple target with a curriculum would
        mean the loader disagreed with the plan — but the loader always returns
        a tuple when multi_step_hours is set, which the curriculum requires)."""
        wrapper = _build_wrapper(m)
        tgt = (target_carry[k_target]
               if type(target_carry) is tuple else target_carry)
        state = _rollout_segment(
            ic_spectral, wrapper, n_sfno_steps, key=dropout_key)
        if _persist_norm:
            return _persistence_normalized_loss_components(
                state, ic_spectral, tgt, grid, sigma, sigma_full,
                loss_cfg_train,
            )
        return _spectral_state_loss_components(
            state, tgt, grid, sigma, sigma_full, loss_cfg_train,
        )

    def _make_train_step(phase_spec):
        """Jitted fused (grad + optax update) train step for ``phase_spec``.

        ``(None, None, None)`` -> the default chained multi-step loss.  A
        curriculum spec ``(lead, k_target, n_sfno_steps)`` -> a single rollout
        to that lead.  One jitted step per DISTINCT ``n_sfno_steps`` (phase
        compiles once, reused across that phase's epochs) — mirrors
        ``_train_spectral_loop._train_step_for``.
        """
        # 4th element: this epoch trains on the stage-2 ensemble CRPS.
        _, k_target, n_sfno_steps, use_crps = phase_spec

        def _train_step(sfno_m, opt_state_in, ic_spectral, target_carry,
                        dropout_key=None):
            def loss_fn(m):
                if use_crps:
                    return _ensemble_crps_loss(
                        m, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dropout_key,
                    )
                if k_target is not None:
                    return _curriculum_loss(
                        m, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dropout_key=dropout_key,
                    )
                return _chained_loss(
                    m, ic_spectral, target_carry, dropout_key=dropout_key)

            (loss, components), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True,
            )(sfno_m)
            grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
            updates, new_opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state_in,
                eqx.filter(sfno_m, eqx.is_array),
            )
            new_model = eqx.apply_updates(sfno_m, updates)
            return new_model, new_opt_state, loss, grad_norm, components

        return eqx.filter_jit(_train_step)

    def _make_pmap_grad_step(phase_spec):
        """Local-device (pmap) GRAD step: one sample per GPU, mean-reduced.

        Levante GPU nodes carry 4x A100, and this needs NO MPI — it is the
        "multi-GPU-PER-PROCESS sub-case" that ``data_parallel.py`` documents.
        It reuses that module's reduction SEMANTICS (per-device
        ``value_and_grad`` then ``lax.pmean`` over an axis named "data"), but
        cannot call ``data_parallel_value_and_grad`` directly: that helper takes
        a loss WITHOUT aux and assumes an all-array parameter pytree, while our
        loss returns ``(loss, components)`` and the SFNO is an Equinox module
        carrying static (non-array) leaves that ``jnp.broadcast_to`` would
        choke on. So the model is split with ``eqx.partition`` and only the
        ARRAY half is replicated.

        Composes with the MPI path: pmean reduces across this process's local
        devices, ``all_reduce_grad_mean`` then reduces across ranks, giving the
        correct hierarchical mean (devices x ranks).
        """
        # 4th element: this epoch trains on the stage-2 ensemble CRPS.
        _, k_target, n_sfno_steps, use_crps = phase_spec
        _, static = eqx.partition(sfno, eqx.is_array)

        def _step(arrays, ic_spectral, target_carry, dropout_key):
            m = eqx.combine(arrays, static)
            # Per-DEVICE dropout key (in_axes=0 below): each device holds a
            # different sample and must draw a different mask, or the "batch"
            # is one mask applied n_dev times. Gated on the static Python flag
            # so a dropout-free run keeps the deterministic path exactly.
            dk = dropout_key if _train_dropout else None

            def loss_fn(mm):
                if use_crps:
                    return _ensemble_crps_loss(
                        mm, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dk,
                    )
                if k_target is not None:
                    return _curriculum_loss(
                        mm, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dropout_key=dk,
                    )
                return _chained_loss(
                    mm, ic_spectral, target_carry, dropout_key=dk)

            (loss, components), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True,
            )(m)
            grads = eqx.filter(grads, eqx.is_array)
            return (jax.lax.pmean(loss, "data"),
                    jax.lax.pmean(components, "data"),
                    jax.lax.pmean(grads, "data"))

        # in_axes=(None, 0, 0) is LOAD-BEARING: jax.pmap maps EVERY argument
        # over axis 0 by default, so passing the parameter arrays positionally
        # would SHARD the weights across devices instead of replicating them
        # (IndivisibleError, or — worse — a silent wrong split for any leaf whose
        # leading dim happens to equal n_dev). ``None`` broadcasts the params;
        # the sample batch keeps its leading device axis. Caught by codex review
        # 2026-07-30; the first version of the device test hid it by manually
        # broadcasting a toy model.
        return jax.pmap(_step, axis_name="data", in_axes=(None, 0, 0, 0))

    def _make_dp_grad_step(phase_spec):
        """Jitted GRAD-ONLY step for data-parallel training.

        Returns ``(loss, components, grads)`` for THIS rank's sample; the caller
        averages ``grads`` across ranks BEFORE the optax update, so the
        cross-rank collective sits between grad and update and the cheap update
        stays outside JIT. Mirrors ``_train_spectral_loop._make_dp_grad_step``.
        """
        # 4th element: this epoch trains on the stage-2 ensemble CRPS.
        _, k_target, n_sfno_steps, use_crps = phase_spec

        def _grad_step(sfno_m, ic_spectral, target_carry, dropout_key=None):
            def loss_fn(m):
                if use_crps:
                    return _ensemble_crps_loss(
                        m, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dropout_key,
                    )
                if k_target is not None:
                    return _curriculum_loss(
                        m, ic_spectral, target_carry, k_target, n_sfno_steps,
                        dropout_key=dropout_key,
                    )
                return _chained_loss(
                    m, ic_spectral, target_carry, dropout_key=dropout_key)

            (loss, components), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True,
            )(sfno_m)
            return loss, components, eqx.filter(grads, eqx.is_array)

        return eqx.filter_jit(_grad_step)

    # One jitted step per distinct phase spec (compile once, reuse across that
    # phase's epochs — shapes are constant within a phase).
    _step_cache: dict = {}
    _dp_step_cache: dict = {}

    def _train_step_for(spec):
        if spec not in _step_cache:
            _step_cache[spec] = _make_train_step(spec)
        return _step_cache[spec]

    def _dp_grad_step_for(spec):
        if spec not in _dp_step_cache:
            _dp_step_cache[spec] = _make_dp_grad_step(spec)
        return _dp_step_cache[spec]
    _pmap_step_cache: dict = {}

    def _pmap_grad_step_for(spec):
        if spec not in _pmap_step_cache:
            _pmap_step_cache[spec] = _make_pmap_grad_step(spec)
        return _pmap_step_cache[spec]

    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = int(getattr(config, "early_stop_patience", 0) or 0)
    early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 1.0e-3))

    # EMA of the SFNO weights (D3, training/ema.py) — same doctrine as
    # _train_spectral_loop: host-side update after every optimizer step,
    # epoch_NNNN_ema.eqx beside each checkpoint, resume prefers the EMA
    # file of the last completed epoch, else re-seeds from raw weights.
    ema_decay = float(getattr(config, "ema_decay", 0.0) or 0.0)
    ema_model = None
    _ema_update_fn = None
    if ema_decay > 0.0:
        ema_model = init_ema(sfno)
        _ema_update_fn = eqx.filter_jit(
            lambda e, m: ema_update(e, m, ema_decay)
        )
        if start_epoch > 0:
            _cand = (
                Path(config.checkpoint_dir)
                / f"epoch_{start_epoch - 1:04d}_ema.eqx"
            )
            if _cand.exists():
                ema_model = eqx.tree_deserialise_leaves(_cand, ema_model)
                logger.info(f"EMA resume: restored {_cand}")
            else:
                logger.warning(
                    "EMA resume: no *_ema.eqx found; re-seeding the EMA "
                    "from the restored raw weights."
                )

    if start_epoch >= n_epochs_total:
        logger.info(
            f"Resume: start_epoch={start_epoch} >= n_epochs={n_epochs_total}; "
            f"skipping training loop (already complete)."
        )
        return sfno, loss_history

    # --- data parallelism (1 GPU per MPI rank) -----------------------------
    # Mirrors _train_spectral_loop's DP path: shard the sample list, take a
    # GRAD-ONLY jitted step, average grads across ranks, then apply the cheap
    # optax update outside JIT. Every replica therefore applies the IDENTICAL
    # averaged gradient and stays bit-synchronised without a weight broadcast.
    # Off / single-rank -> (False, 0, 1, None) and the serial fused-step path
    # below is byte-identical to the pre-DP loop.
    dp_on, dp_rank, dp_nproc, dp_comm = _resolve_dp_context(config)
    _all_reduce_grad_mean = _global_sum_mpi = None
    if dp_on:
        from legoesm.parallel.reductions import global_sum_mpi as _global_sum_mpi
        from legoesm.training.data_parallel import (
            all_reduce_grad_mean as _all_reduce_grad_mean,
            shard_samples,
        )
        _n_global = len(ic_states)
        # drop_remainder keeps every rank's step count EQUAL. Unequal shards
        # would desynchronise the per-sample allreduce and deadlock.
        ic_states = shard_samples(ic_states, dp_rank, dp_nproc)
        target_carries = shard_samples(target_carries, dp_rank, dp_nproc)
        _dropped = _n_global - dp_nproc * len(ic_states)
        if not ic_states:
            raise ValueError(
                f"data_parallel: {_n_global} samples across {dp_nproc} ranks "
                "leaves this rank with none. Use fewer ranks or more windows."
            )
        logger.info(
            "SFNO full-emulator data-parallel: %d rank(s), %d/%d samples on "
            "this rank%s", dp_nproc, len(ic_states), _n_global,
            f", {_dropped} remainder sample(s) dropped per epoch"
            if _dropped else "")

    # LOCAL-device data parallelism (no MPI): one sample per GPU on this
    # process, gradients pmean'd across devices. Levante GPU nodes are 4x A100.
    # n_dev == 1 -> disabled, and the serial fused-step path below is unchanged.
    n_dev = jax.local_device_count() if bool(
        getattr(config, "data_parallel", False)) else 1
    pmap_on = n_dev > 1
    if pmap_on:
        logger.info(
            "SFNO full-emulator local-device parallel: %d device(s) x "
            "%d rank(s) = %d samples per optimizer update",
            n_dev, dp_nproc, n_dev * dp_nproc)
        _pmap_dropped = len(ic_states) % n_dev
        if _pmap_dropped:
            # NO SILENT CAPS: this is a second, independent drop on top of the
            # MPI shard remainder, and epoch metrics exclude both. Report the
            # combined per-epoch coverage loss rather than let the logs read as
            # if every sample were used.
            logger.warning(
                "SFNO full-emulator: %d sample(s)/epoch dropped on this rank to "
                "keep the pmap device axis constant (%d local sample(s) %% %d "
                "device(s)); %d dropped across %d rank(s) per epoch.",
                _pmap_dropped, len(ic_states), n_dev,
                _pmap_dropped * dp_nproc, dp_nproc)
        if len(ic_states) < n_dev:
            raise ValueError(
                f"data_parallel: {len(ic_states)} sample(s) on this rank but "
                f"{n_dev} local devices; each device needs one. Use fewer "
                "devices or more training windows.")

    def _stack_batch(samples):
        """(n_dev, ...) leading device axis from n_dev sample pytrees."""
        return jax.tree.map(lambda *xs: jnp.stack(xs), *samples)

    def _iter_updates():
        """Yield one (ic, target) unit per optimizer update.

        pmap: groups of ``n_dev`` stacked along a leading device axis. The
        remainder is DROPPED so the device axis is constant and the step
        compiles once (a ragged final group would retrace every epoch).
        Serial: single samples, exactly as before.
        """
        if not pmap_on:
            for pair in zip(ic_states, target_carries):
                yield pair
            return
        n_full = (len(ic_states) // n_dev) * n_dev
        for j in range(0, n_full, n_dev):
            yield (_stack_batch(ic_states[j:j + n_dev]),
                   _stack_batch(target_carries[j:j + n_dev]))

    def _is_rank0():
        return (not dp_on) or dp_rank == 0

    for epoch in range(start_epoch, n_epochs_total):
        # Curriculum: this epoch's phase spec (resume indexes into epoch_plan);
        # the per-phase jitted step compiles once and is reused within a phase.
        # Non-curriculum: epoch_plan[epoch] == (None, None, None) -> the default
        # chained-multi-step train_step for every epoch (single compile).
        # U-Cast stage 2 occupies the LAST crps_finetune_epochs epochs. The
        # flag rides in the phase spec so the per-phase jit cache keys on it:
        # the CRPS step has a different graph (M vmapped rollouts) and must
        # not reuse the deterministic step's compilation.
        _use_crps = epoch >= (n_epochs_total - _crps_ft_epochs) \
            if _crps_ft_epochs > 0 else False
        _spec = (*epoch_plan[epoch], _use_crps)
        train_step = None if (dp_on or pmap_on) else _train_step_for(_spec)
        dp_grad_step = (
            _dp_grad_step_for(_spec) if (dp_on and not pmap_on) else None)
        pmap_grad_step = _pmap_grad_step_for(_spec) if pmap_on else None
        epoch_loss = 0.0
        epoch_components = {"mse": 0.0, "bias": 0.0, "crps": 0.0, "spec_crps": 0.0}
        t0 = time.time()
        grad_norm_val = 0.0

        for sample_idx, (ic, target) in enumerate(_iter_updates()):
            # MC-Dropout key for THIS update. Folded on (epoch, sample) so no
            # two optimizer steps reuse a mask — and so a RESUMED run does not
            # replay the masks of the epochs it already trained. ``None`` when
            # the arm is dropout-free (deterministic path unchanged).
            # ``dp_rank`` is folded in FIRST: every rank walks the same
            # (epoch, sample_idx) counters over its OWN shard, so without it
            # all ranks would apply the identical mask to different samples —
            # correlated stochastic gradients across the very ranks the
            # data-parallel mean exists to decorrelate.
            dropout_key = (
                jax.random.fold_in(
                    jax.random.fold_in(
                        jax.random.fold_in(_dropout_base_key, dp_rank),
                        epoch),
                    sample_idx)
                if _train_dropout else None)
            if host_staged and not pmap_on:
                # Host-resident dataset (#1155): stage only THIS sample onto
                # the compute device; the prior sample's device copies free
                # when ic/target rebind. Peak device footprint: one sample.
                # Under pmap the batch already carries a leading device axis
                # and jax.pmap places each slice on its own device, so an
                # explicit single-device stage would fight that placement.
                ic = stage_sample(ic)
                target = stage_sample(target)
            if pmap_on:
                # One sample per LOCAL device; pmean inside the step. Then, if
                # an MPI world is also present, average across ranks too — the
                # two reductions compose into a (devices x ranks) mean.
                # One key per LOCAL device (in_axes=0). Always supplied so the
                # pmap signature is stable; with sfno_dropout=0 the step
                # ignores it and the forward pass is unchanged.
                _pmap_keys = jax.random.split(
                    (dropout_key if dropout_key is not None
                     else jax.random.PRNGKey(0)),
                    n_dev)
                loss, components, grads = pmap_grad_step(
                    eqx.filter(sfno, eqx.is_array), ic, target, _pmap_keys,
                )
                # pmean made every device identical -> take device 0.
                loss = loss[0]
                components = jax.tree.map(lambda a: a[0], components)
                grads = jax.tree.map(lambda a: a[0], grads)
                if dp_on:
                    grads = _all_reduce_grad_mean(
                        grads, dp_nproc, comm=dp_comm)
                grad_norm = optax.global_norm(grads)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(sfno, eqx.is_array),
                )
                sfno = eqx.apply_updates(sfno, updates)
            elif dp_on:
                # Local grad on this rank's sample -> cross-rank MEAN -> update.
                loss, components, grads = dp_grad_step(
                    sfno, ic, target, dropout_key)
                grads = _all_reduce_grad_mean(grads, dp_nproc, comm=dp_comm)
                grad_norm = optax.global_norm(grads)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(sfno, eqx.is_array),
                )
                sfno = eqx.apply_updates(sfno, updates)
            else:
                sfno, opt_state, loss, grad_norm, components = train_step(
                    sfno, opt_state, ic, target, dropout_key,
                )
            if ema_model is not None:
                ema_model = _ema_update_fn(ema_model, sfno)

            loss_val = float(loss)
            if jnp.isnan(loss) or jnp.isinf(loss):
                raise RuntimeError(
                    f"SFNO full-emulator: NaN/Inf loss at epoch {epoch}, "
                    f"sample {sample_idx} (loss={loss_val})."
                )
            grad_norm_val = float(grad_norm)
            if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                raise RuntimeError(
                    f"SFNO full-emulator: NaN/Inf gradient at epoch {epoch}, "
                    f"sample {sample_idx} (grad_norm={grad_norm_val})."
                )

            # ``loss`` under pmap is ALREADY a mean over the n_dev samples in
            # this update, so weight it by n_dev to keep epoch_loss a true SUM
            # over samples. Without this the epoch mean (and the cross-rank
            # reduction, which is weighted by n_samples) is off by n_dev.
            _w = n_dev if pmap_on else 1
            epoch_loss += loss_val * _w
            # Iterate the COMPONENTS (not the init keys) so optional
            # per-variable terms (bias_T/u/v/ps) accumulate too.
            for key in components:
                epoch_components[key] = (
                    epoch_components.get(key, 0.0) + float(components[key]) * _w)

        # Under pmap one iteration consumed n_dev samples; count SAMPLES (not
        # updates) so avg_loss stays a per-sample mean comparable across
        # device counts. The remainder dropped by _iter_updates is excluded.
        n_samples = max(
            (len(ic_states) // n_dev) * n_dev if pmap_on else len(ic_states), 1)
        if dp_on:
            # epoch_loss / n_samples are per-rank LOCAL under sharding. Reduce
            # to a global mean so the logged number is the real epoch loss AND
            # every rank takes the same early-stop branch — a rank-local
            # decision would desynchronise control flow and hang the next
            # allreduce. Balanced shards guarantee lockstep entry here.
            # Deterministic order over the ACTUAL keys (per-variable bias terms
            # included); identical on every rank for a fixed config, so the
            # collective shape is consistent.
            _keys = tuple(sorted(epoch_components))
            _acc = _global_sum_mpi(
                jnp.asarray(
                    [epoch_loss] + [epoch_components[k] for k in _keys]
                    + [float(n_samples)]
                ),
                comm=dp_comm,
            )
            _accl = [float(x) for x in _acc]
            _gn = max(_accl[-1], 1.0)
            avg_loss = _accl[0] / _gn
            avg_components = {k: _accl[1 + i] / _gn for i, k in enumerate(_keys)}
        else:
            avg_loss = epoch_loss / n_samples
            avg_components = {
                k: v / n_samples for k, v in epoch_components.items()}
        loss_history.append(avg_loss)

        if epoch % config.log_every == 0 or epoch == n_epochs_total - 1:
            elapsed = time.time() - t0
            _extra = " ".join(
                f"{k}={avg_components[k]:.4f}"
                for k in sorted(avg_components)
                if k not in ("mse", "bias", "crps", "spec_crps"))
            logger.info(
                f"Epoch {epoch:4d}: loss={avg_loss:.6f} "
                f"(mse={avg_components['mse']:.4f} "
                f"bias={avg_components['bias']:.4f} "
                f"crps={avg_components['crps']:.4f} "
                f"spec_crps={avg_components['spec_crps']:.4f}"
                + (f" {_extra}" if _extra else "") + "), "
                f"grad_norm={grad_norm_val:.6e}, time={elapsed:.1f}s"
            )

        # Per-epoch checkpoint: same cadence as ``_train_spectral_loop``
        # so the run_aimip.py --resume driver can pick up after a
        # walltime kill.
        # RANK 0 ONLY under DP. Every replica holds identical weights (the
        # gradient is averaged before the update), so N writers would race on
        # the same paths for no benefit — and a torn epoch_NNNN.eqx breaks the
        # --resume driver.
        if _is_rank0():
            from legoesm.ml.training import save_checkpoint
            ckpt_dir = Path(config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.eqx"
            save_checkpoint(sfno, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            if ema_model is not None:
                ema_path = ckpt_dir / f"epoch_{epoch:04d}_ema.eqx"
                save_checkpoint(ema_model, ema_path)
                logger.info(f"Saved EMA checkpoint: {ema_path}")

        # Early stopping is DISABLED under a rollout curriculum (mirrors
        # _train_spectral_loop): loss magnitudes are NOT comparable across phases
        # (a longer lead has a naturally larger loss), so a stop triggered right
        # after an early short-lead phase would silently skip the remaining — most
        # stability-critical — long phases (24/48/120 h). The per-phase epoch
        # counts are the explicit training budget instead.
        # Stage 2 lives in the LAST epochs, so an early stop during stage-1
        # pre-training silently skips the probabilistic fine-tune entirely and
        # yields a deterministic checkpoint labelled as a CRPS run (codex
        # review 2026-08-02). Suppress early stopping until stage 2 has begun;
        # inside stage 2 the loss is a different objective whose plateau
        # against stage-1's best is meaningless, so it stays suppressed.
        if early_stop_patience > 0 and not config.rollout_curriculum \
                and _crps_ft_epochs == 0:
            if best_loss - avg_loss > early_stop_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info(
                        f"Early stop at epoch {epoch}: no improvement "
                        f"> {early_stop_min_delta} for "
                        f"{early_stop_patience} consecutive epochs "
                        f"(best={best_loss:.6f}, last={avg_loss:.6f})."
                    )
                    break

    return sfno, loss_history


def train_column_mlp_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
    seed: int = 0,
    hidden_dim: int = 256,
    n_layers: int = 4,
    # None = let build_column_physics apply its own default. Restating
    # 0.01 here made a second source of truth that could drift from the
    # factory's (and from the WB lane, which never passed it) unnoticed.
    residual_scale: float | None = None,
    *,
    resume_from_dir=None,
    surface_forcing_path: str | None = None,
    forcing_cache_path: str | None = None,
):
    """Train column MLP physics (Rasp 2018) + spectral PE dycore.

    ``surface_forcing_path`` / ``forcing_cache_path``: as in
    :func:`train_neural_gcm_spectral` — prescribed SST/sea-ice +
    orbital-calendar features per training sample (the AMIP /
    interannual-variability pathway).

    Returns (trained_neural_physics, loss_history).
    """
    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    from legoesm.training.model_registry import build_variant
    nn_phys = build_variant(
        "column_nn", nlev=config.n_levels, seed=seed,
        overrides={"nn_hidden_dim": hidden_dim, "n_layers": n_layers,
                   "residual_scale": residual_scale},
    )
    n_p = sum(x.size for x in jax.tree.leaves(eqx.filter(nn_phys, eqx.is_array)))
    logger.info(f"Column MLP: {config.n_levels} levels, {hidden_dim}d, "
                f"{n_layers} layers, {n_p:,} params")

    nn_phys, start_epoch = maybe_resume_model(nn_phys, resume_from_dir)

    if int(config.chunk_windows or 0) > 0:
        chunk_loader, n_total = _make_chunk_loader(
            config, grid, sigma, cache_dir,
            surface_forcing_path, forcing_cache_path,
        )
        return _train_spectral_loop(
            nn_phys, make_column_mlp_spectral_physics,
            grid, sigma, None, None, config,
            start_epoch=start_epoch,
            chunk_loader=chunk_loader, n_samples_total=n_total,
            resume_from_dir=resume_from_dir,
        )

    ic_states, target_carries, ic_times = load_training_data(
        config, grid, sigma, cache_dir, windows=config.windows,
        host_resident=True,   # non-chunked full-dataset load (#1155)
    )
    sample_forcings = _maybe_build_sample_forcings(
        surface_forcing_path, ic_times, grid, forcing_cache_path,
    )

    return _train_spectral_loop(
        nn_phys, make_column_mlp_spectral_physics,
        grid, sigma, ic_states, target_carries, config,
        start_epoch=start_epoch,
        sample_forcings=sample_forcings,
        host_staged=True,   # dataset loaded host-resident above (#1155)
        resume_from_dir=resume_from_dir,   # EMA resume needs the dir too
    )


def train_physics_params_spectral(
    config: NeuralGCMSpectralConfig = NeuralGCMSpectralConfig(),
    cache_dir: str = "data/era5_cache",
):
    """Train physics-based parameterization parameters + spectral PE dycore.

    Tunes the parameters of the combined physics schemes (SBM convection and
    the surface exchange/albedo knobs) by backpropagating through both the
    physics computations and the spectral dynamical core. Gray radiation runs
    at its documented defaults — it is not trained.

    Trainable parameters (via ``TrainablePhysicsParams``):
    - ``sbm_tau_c``: SBM convection relaxation timescale
    - ``sbm_RH_ref``: SBM convection reference relative humidity

    Returns (trained_params, loss_history).
    """
    # The IDEALIZED 2-family parameter set (convection + radiation), not the
    # campaign "classical" model — the registry's classical is the six-family
    # AIMIPClassicalParams since 2026-08-12.
    from legoesm.training.trainable_params import TrainablePhysicsParams

    grid = create_gaussian_grid(config.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(config.n_levels, sigma_top=config.sigma_top)

    params = TrainablePhysicsParams.from_defaults()
    n_p = len(params.raw_values)
    logger.info(f"Physics params: {n_p} trainable ({', '.join(params.raw_values)})")
    for k, v in params.as_dict().items():
        logger.info(f"  {k} = {float(v):.4f}")

    ic_states, target_carries, _ic_times = load_training_data(
        config, grid, sigma, cache_dir, windows=config.windows,
        host_resident=True,   # non-chunked full-dataset load (#1155)
    )

    dt = config.dt

    def _make_physics_fn(p, grid_):
        return make_physics_params_spectral_physics(p, grid_, dt)

    return _train_spectral_loop(
        params, _make_physics_fn,
        grid, sigma, ic_states, target_carries, config,
        host_staged=True,   # dataset loaded host-resident above (#1155)
    )
