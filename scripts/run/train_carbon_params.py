#!/usr/bin/env python
"""Stage-B (B2): gradient-calibrate the DifferLand SOM carbon parameters.

Warm-start from the Stage-A IC-map production ``CarbonConfig`` defaults and
gradient-tune the soil-organic-matter (SOM) parameters against OBSERVED
per-archetype soil organic carbon (SOC), so the freeze-floor / turnover /
transfer values are calibrated instead of hand-set.  The differentiable forward
map is Phase B1's
:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced` (per-archetype
equilibrium SOC as a function of TRACED ``CarbonConfig`` SOM leaves); the observed
target is the surfdata organic-carbon column, cover-weighted per archetype
(:mod:`legoesm.land.carbon.soc_observations`).

Doctrine (mirrors ``scripts/run/train_scm_rce_params.py`` / the SCM-RCE pattern):
the trained leaves are spliced into each group's ``CarbonConfig`` via
``param_collector.apply_param_overrides`` **inside** the loss (so the leaves are
traced -- the SegmentForcing override doctrine), the optimizer is MUON
(``legoesm.ml.training.create_optimizer``, warmup+cosine+clip), and the run writes
a RECOMMENDED ``tuned_carbon_parameters.json`` under ``results/`` -- it NEVER
mutates the production ``CarbonConfig`` defaults (production keeps static
Python-float leaves).

Modular loss registry: ``LOSSES = {obs_name: LossTerm(target, extractor, weight)}``
-- ``soc`` (``som_total`` vs observed SOC, cover-weighted MSE via
``legoesm.ml.loss.area_weighted_mse``) plus, under ``--with-sif``, ``sif`` (a
single-step per-archetype simulated-SIF forward vs observed SIF -- constrains
photosynthesis/GPP, the boreal-productivity lever).  Both are summed by ``_total_loss``
WITHOUT touching the loop; ``biomass`` / ``lai`` / ``d13c`` plug in the same way (add a
registry entry).

Login-node policy: the archetype spin-up JIT-compiles the coupled land+carbon
model and its reverse-mode; run this via ``sbatch`` / ``srun`` on a compute node,
never on the login node.  ``JAX_ENABLE_X64=1``.

Feasibility: gradient through ~100-200 archetypes x a multi-year spin-up x N
optimizer steps is heavy.  Keep it tractable with a SHORT training spin-up
(``--n-spinup 25``, ``jax.checkpoint`` on via ``remat=True``) and, if needed, a
REPRESENTATIVE subset of archetypes (``--max-archetypes``, stratified across the
PFT / climate range).  The tuned parameters apply globally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from legoesm import constants
from legoesm.ml.loss import area_weighted_mse
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.training.param_collector import build_trainable_params
from legoesm.training.trainable_params import TrainablePhysicsParams


# --- units / identifiers -------------------------------------------------------
_G_PER_KG = 1000.0                 # gC/m2 -> kgC/m2 (exact conversion)
CARBON_SCHEME_KEY = "land.carbon"  # param_collector scheme_key for CarbonConfig
SIF_SCHEME_KEY = "land.canopy.sif"  # param_collector scheme_key for SIFConfig (--with-sif)
# Organic-CARBON per organic-MATTER fraction (van Bemmelen 1/1.724 = 0.58): the
# raw CLM5 surfdata ORGANIC field is organic MATTER density (its units attribute
# declares 0.58 gC/gOM). The harmonized legoesm_surfdata "organic" derives from
# HWSD ORG_CARBON (already carbon) -> factor 1.0. --om-to-oc overrides either.
_VAN_BEMMELEN_OC_PER_OM = 0.58

# The SOM fields Stage-B v1 calibrates (traced through the spin-up); the same set
# B1's gradient smoke proved a finite, non-zero jax.grad for.
#
# ``Q10_het_exp`` is EXCLUDED here (unlike the B1 slow/--slow-spinup-grad set,
# which keeps it -- see tests/land/validation/test_carbon_calibration_gradient.py).
# Reason: Q10_het_exp enters carbon_cycle._temperate_modifier, which sets BOTH
# (a) the SOM cascade decomposition rate (captured live -- fast_analytic.
# analytic_som_soc calls som_decomposition_rate on the traced config every
# step) and (b) the litter -> SOM decomposition flux lit_to_som (carbon_cycle.py
# ~L565-595) -- part of the SOM active-pool INPUT.  Fast mode PRECOMPUTES
# lit_to_som_annual ONCE at default parameters
# (global_init.precompute_fast_analytic_inputs) and freezes it, so path (b) is
# invisible to jax.grad here: training Q10_het_exp in fast mode would see only a
# partial/biased gradient (path (a) only).  The remaining seven fields never
# appear in the litter equations, so their fast-mode gradient is the full
# gradient.  (cwd_humification_eff is unaffected: its wood input is stored
# SEPARATELY as a_wood_annual -- the NPP-to-wood allocation, independent of
# every SOM param, no approximation at all -- with the cwd_humification_eff
# multiply applied LIVE in analytic_som_soc, so its gradient is exact too.)
SOM_FIELDS: tuple[str, ...] = (
    "tor_som_active", "tor_som_slow", "tor_som_passive",
    "f_active_to_slow", "f_slow_to_passive", "som_freeze_floor",
    "cwd_humification_eff",
)

# --- training defaults (modest; a compute-node short calibration) --------------
DEFAULT_STEPS = 60
DEFAULT_LR = 3.0e-2
DEFAULT_WARMUP_STEPS = 5
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_N_SPINUP = 25              # short TRAINING spin-up (remat on)
DEFAULT_N_VERIFY = 3
DEFAULT_DT = 3600.0               # sub-daily spin-up timestep [s]
DEFAULT_N_LAYERS = 10
DEFAULT_SOIL_DEPTH = 3.0          # soil column depth [m]
DEFAULT_MAX_ARCHETYPES = 40       # representative-subset cap (0 = keep all)
# Default chunk size for the --slow-spinup-grad EXACT chunked gradient
# accumulation: the batched coupled-model reverse-mode VJP SEGFAULTS on the CPU
# XLA backend at >=8 archetypes (a crash, not OOM), so each per-chunk VJP is
# bounded to this many archetypes.  Value = the LARGEST crash-free CPU chunk size
# from the Step-0 bisect (scripts/tmp/_bisect_carbon_chunk.py); fewer chunks =
# fewer sequential VJPs = faster.  0 (or --grad-chunk 0) keeps the single-batch
# path for fast mode / tiny runs.
DEFAULT_GRAD_CHUNK = 4
DEFAULT_OUTDIR = Path("results/carbon_calibration")
# Loss-term weights. SOC (kgC/m2, O(10)) and SIF (umol/m2/s, O(10)) live on different
# scales, so the summed cover-weighted MSE weights each term; SOC is the reference (1).
W_SOC = 1.0
DEFAULT_SIF_WEIGHT = 1.0           # --sif-weight; balances the sif MSE against soc
GRAD_NONZERO_TOL = 1.0e-14
# Mean analytic-vs-spin-up SOC relative error above which the fast forward is
# flagged as an unreliable proxy for the model (a loud run-log warning, not a
# hard fail: the unit test gates the tiny-world match; the run surfaces the
# production-scale fidelity so a large mismatch is never silently trusted).
_MATCH_WARN_REL = 0.15
# Line-search scales for a monotone-decreasing MUON step (mirrors the SCM-RCE
# trainer): try the full step first, then progressively shorter ones.
_LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)

# --- quick (unit-test) overrides ----------------------------------------------
QUICK_STEPS = 1
QUICK_N_SPINUP = 6
QUICK_N_VERIFY = 2
QUICK_N_LAYERS = 6
QUICK_SOIL_DEPTH = 2.0
QUICK_MAX_ARCHETYPES = 3


# ===========================================================================
# Modular loss registry
# ===========================================================================
class LossTerm(NamedTuple):
    """One observation term of the calibration loss.

    ``target`` is the per-archetype observed value (``(n_arch,)`` in the same
    units as ``extractor``'s output).  ``extractor(preds) -> (n_arch,)`` pulls the
    modelled counterpart from the predictions bundle.  ``weight`` is the scalar term
    weight in the summed loss.  The registry is a dict of these so
    ``biomass`` / ``lai`` / ``sif`` / ``d13c`` add WITHOUT touching the loop.

    ``mask`` (optional ``(n_arch,)`` of 1.0/0.0) marks archetypes where THIS term's
    observation is present.  It multiplies the term's per-archetype squared error, so an
    archetype missing THIS observable (but present in another) contributes 0 to this term
    yet still counts in the SINGLE shared cover-weight denominator -- letting one stream's
    gaps not discard another stream's signal (the observations need not overlap).  ``None``
    = every archetype observed (the default / SOC-only path)."""
    target: jax.Array
    extractor: Callable[[Any], jax.Array]
    weight: float
    mask: Any = None


def _soc_extractor(preds) -> jax.Array:
    """Model per-archetype SOC [kgC/m2] = total SOM / 1000 (gC/m2 -> kgC/m2).

    Reads the SLOW forward's equilibrium ``CarbonState`` from the predictions bundle
    (``preds["eq"]``); the FAST closed-form path supplies SOC directly via
    :func:`_fast_soc_extractor`.
    """
    from legoesm.land.carbon.config import som_total
    return som_total(preds["eq"]) / _G_PER_KG


def _fast_soc_extractor(preds) -> jax.Array:
    """Model per-archetype SOC [kgC/m2] from the FAST closed-form forward.

    The fast-analytic path computes SOC (``analytic_som_soc / 1000``) directly, so the
    predictions bundle carries it under ``preds["soc"]`` (already kgC/m2)."""
    return preds["soc"]


def _sif_extractor(preds) -> jax.Array:
    """Model per-archetype simulated SIF [umol/m2/s] from the predictions bundle.

    The (single-step, spin-up-free) SIF forward is evaluated in the loss and stored under
    ``preds["sif"]`` (see :func:`legoesm.land.carbon.sif_forward.simulate_archetype_sif`)."""
    return preds["sif"]


def cover_weighted_mse(
    pred: jax.Array, target: jax.Array, cover_weight: jax.Array,
) -> jax.Array:
    """Cover-weighted MSE over archetypes: ``sum_a w_a (pred_a-obs_a)^2 / sum_a w_a``.

    Reuses the shared spherical helper ``ml.loss.area_weighted_mse`` by mapping the
    ARCHETYPE axis onto its latitude axis (``n_lon = n_channels = 1``): with those
    trailing singleton axes the no-mask branch returns exactly
    ``sum(w*sq)/sum(w)`` (see ``area_weighted_mse``), i.e. the cover-weighted mean
    squared SOC residual.  ``cover_weight[a]`` is archetype ``a``'s total land
    cover (:func:`legoesm.land.carbon.soc_observations.per_archetype_cover_weight`).
    """
    pred3 = jnp.reshape(pred, (-1, 1, 1))
    target3 = jnp.reshape(target, (-1, 1, 1))
    return area_weighted_mse(pred3, target3, jnp.asarray(cover_weight))


def _total_loss(preds, losses: dict[str, LossTerm], cover_weight: jax.Array) -> jax.Array:
    """Sum the registry: ``sum_k w_k * cover_weighted_mse(extractor_k(preds), target_k)``.

    ``preds`` is the loss's predictions bundle (``{"eq": CarbonState}`` /
    ``{"soc": ...}`` for the SOC forward, plus ``{"sif": ...}`` when ``--with-sif``); each
    term's ``extractor`` pulls the modelled counterpart it needs.  Adding a term is a
    registry entry -- this summation never changes."""
    total = jnp.asarray(0.0, dtype=cover_weight.dtype)
    csum = jnp.sum(cover_weight)
    for term in losses.values():
        pred = term.extractor(preds)
        if term.mask is None:
            total = total + term.weight * cover_weighted_mse(
                pred, term.target, cover_weight)
        else:
            # Masked term: single GLOBAL cover-weight denominator (sum over ALL active
            # archetypes), numerator masked to this term's observed archetypes.
            total = total + term.weight * jnp.sum(
                cover_weight * term.mask * (pred - term.target) ** 2) / csum
    return total


def _weighted_sse(preds, losses: dict[str, LossTerm], cover_weight: jax.Array) -> jax.Array:
    """UN-normalized cover-weighted SSE: ``sum_k w_k sum_a cover_a (pred_{k,a}-obs_{k,a})^2``.

    The NUMERATOR of :func:`_total_loss` (which divides this by ``sum_a cover_a``):
    ``_total_loss(eq, losses, cover) == _weighted_sse(eq, losses, cover) /
    sum(cover)`` because every term shares the SAME cover-weight denominator.  The
    chunked EXACT-gradient accumulation (:func:`_make_chunked_slow`) sums this
    per-chunk UN-normalized SSE (+ its grad) across chunks and divides by the
    GLOBAL cover-weight sum ONCE at the end, so the chunked gradient equals the
    unchunked cover-weighted-MSE gradient exactly.  Kept un-normalized (a direct
    ``sum(cover*(pred-obs)^2)`` rather than ``cover_weighted_mse * sum_chunk_cover``)
    so a chunk whose archetypes ALL have zero cover contributes a clean 0, not a
    0/0 NaN from a per-chunk normalisation.
    """
    cover = jnp.asarray(cover_weight)
    total = jnp.asarray(0.0, dtype=cover.dtype)
    for term in losses.values():
        pred = term.extractor(preds)
        sq = cover * (pred - term.target) ** 2
        if term.mask is not None:
            sq = sq * term.mask     # zero this term's missing-observation archetypes
        total = total + term.weight * jnp.sum(sq)
    return total


# ===========================================================================
# Trainable parameters (warm-started from the CarbonConfig production defaults)
# ===========================================================================
def _filter_params(
    params: TrainablePhysicsParams, keep_names: set[str],
) -> TrainablePhysicsParams:
    constraints = [c for c in params.constraints if c.name in keep_names]
    raw_values = {c.name: params.raw_values[c.name] for c in constraints}
    return TrainablePhysicsParams(raw_values=raw_values, constraints=constraints)


def build_carbon_trainables(
    *, som_only: bool = True, with_sif: bool = False,
) -> TrainablePhysicsParams:
    """Tier-``extended`` ``land.carbon`` (+ optional ``land.canopy.sif``) trainables,
    warm-started from the production ``CarbonConfig`` / ``SIFConfig`` defaults
    (``build_trainable_params`` seeds each raw leaf from the live NamedTuple default via
    the inverse sigmoid).

    ``som_only`` (default) restricts the CARBON set to the seven fast-analytic-valid SOM
    fields (:data:`SOM_FIELDS`) -- the Stage-B v1 calibration target -- so the optimizer
    is not handed the ~30 DALEC phenology/allocation leaves that barely move equilibrium
    SOC (nor ``Q10_het_exp``, whose fast-mode gradient is partial -- see the comment on
    :data:`SOM_FIELDS`).  Pass ``som_only=False`` to expose the full tier-2 carbon set
    (only valid with ``--slow-spinup-grad``; see :func:`_finalize_args`).

    ``with_sif`` additionally includes the tier-1/2 ``SIFConfig`` fluorescence params
    (``kn0``/``kn_beta``/``kn_gamma``/``max_electron_yield``/``escape_probability``/
    ``kf``/``kd``/``kp``) -- ALWAYS kept in full (the SIF forward is a separate single-step
    diagnostic, not a SOM-pool field, so ``som_only`` never filters them).  Overrides are
    applied traced INSIDE the loss; production defaults are untouched.
    """
    scheme_keys = {CARBON_SCHEME_KEY}
    if with_sif:
        scheme_keys.add(SIF_SCHEME_KEY)
    params = build_trainable_params(
        active_scheme_keys=scheme_keys, tier="extended", dtype=jnp.float64)
    sif_names = {c.name for c in params.constraints
                 if c.name.startswith(SIF_SCHEME_KEY + ".")}
    if with_sif and not sif_names:
        raise ValueError(
            f"--with-sif requested but no {SIF_SCHEME_KEY} trainables at tier 'extended'; "
            f"is the spec module registered in param_collector.SPEC_MODULES?")
    if not som_only:
        # Full tier-2 carbon set (+ all SIF fluorescence params when --with-sif).
        return params
    # SOM-only carbon subset, PLUS all SIF fluorescence params.
    keep = {f"{CARBON_SCHEME_KEY}.{f}" for f in SOM_FIELDS} | sif_names
    have = {c.name for c in params.constraints}
    missing = {f"{CARBON_SCHEME_KEY}.{f}" for f in SOM_FIELDS} - have
    if missing:
        raise ValueError(
            f"SOM fields absent from the land.carbon tier-extended trainables: "
            f"{sorted(missing)}; known={sorted(have)}")
    return _filter_params(params, keep)


def _carbon_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{CarbonConfig field -> traced constrained scalar}`` for the loss."""
    return params.to_overrides().get(CARBON_SCHEME_KEY, {})


def _sif_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{SIFConfig field -> traced constrained scalar}`` for the sif loss term (empty
    when ``--with-sif`` is off / no SIF params are trained)."""
    return params.to_overrides().get(SIF_SCHEME_KEY, {})


def _make_sif_forward(table):
    """Build ``sif_forward(sif_overrides) -> (n_arch,)`` simulated SIF [umol/m2/s] for the
    given (possibly chunk-sliced) archetype table.

    Precomputes the static per-archetype coupled-Farquhar leaf state ONCE
    (:func:`legoesm.land.carbon.sif_forward.build_sif_forward`), then each call splices the
    TRACED SIF overrides into a fresh ``SIFConfig`` (production defaults untouched) and
    applies the differentiable ``leaf_sif`` kernel -- a single-step graph, no spin-up."""
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import build_sif_forward
    from legoesm.training.param_collector import apply_param_overrides

    sif_fn = build_sif_forward(table)

    def sif_forward(sif_overrides):
        cfg = apply_param_overrides(SIFConfig(), sif_overrides)
        return sif_fn(cfg)

    return sif_forward


def _param_values(params: TrainablePhysicsParams) -> dict[str, float]:
    """Constrained physical values keyed by BARE field name (drops the scheme key)."""
    physical = params.as_dict()
    return {c.field: float(physical[c.name]) for c in params.constraints}


# ===========================================================================
# Loss + gradient diagnostics
# ===========================================================================
def make_loss_fn(
    *,
    table,
    losses: dict[str, LossTerm],
    cover_weight: jax.Array,
    sif_forward=None,
    n_spinup: int,
    n_verify: int,
    dt: float,
    n_layers: int,
    soil_depth: float,
):
    """Build ``loss_fn(params) -> scalar`` with the overrides applied INSIDE
    (traced) via ``equilibrate_archetypes_traced`` -- ``jax.grad`` flows to the SOM
    leaves through the coupled ``lax.scan`` spin-up + the analytic slow-pool reset.

    ``sif_forward`` (when ``--with-sif``) is the single-step SIF forward evaluated on the
    TRACED SIF overrides; its prediction is added to the bundle under ``preds["sif"]`` and
    the ``sif`` registry term sums into the loss.  The SIF forward has NO spin-up scan (it
    does not touch ``equilibrate_archetypes_traced``), and the SIF params do not enter the
    SOC forward, so the two gradients are block-independent.
    """
    from legoesm.land.carbon.global_init import equilibrate_archetypes_traced

    def loss_fn(params: TrainablePhysicsParams) -> jax.Array:
        overrides = _carbon_overrides(params)
        eq = equilibrate_archetypes_traced(
            table, overrides, n_spinup=n_spinup, n_verify=n_verify,
            dt=dt, n_layers=n_layers, soil_depth=soil_depth)
        preds = {"eq": eq}
        if sif_forward is not None:
            preds["sif"] = sif_forward(_sif_overrides(params))
        return _total_loss(preds, losses, cover_weight)

    return loss_fn


def _chunk_bounds(n: int, chunk: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` archetype-index chunks of at most ``chunk``.

    ``chunk <= 0`` or ``chunk >= n`` -> ONE chunk ``[(0, n)]`` (the single-batch
    path).  A chunk bounds the number of archetypes in one
    :func:`~legoesm.land.carbon.global_init.equilibrate_archetypes_traced` call
    (hence one reverse-mode VJP): each archetype's coupled spin-up is INDEPENDENT
    of the others (its own PFT physiology + climate forcing column; the batched
    ``vmap`` is per-column elementwise), so any contiguous partition yields the
    identical per-archetype equilibrium SOC.
    """
    if chunk <= 0 or chunk >= n:
        return [(0, n)]
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


def _make_chunked_slow(target_bundle, spin, *, chunk: int):
    """EXACT chunked value-and-grad + loss for the slow grad-through-spin-up forward.

    The cover-weighted-MSE loss is LINEAR over archetypes with a SINGLE global
    denominator ``sum_a cover_a`` (shared by every loss term)::

        grad = (1/sum_w) * sum_chunk grad[ sum_{a in chunk} cover_a (pred_a-obs_a)^2 ]

    so each per-chunk term is a coupled-spin-up VJP over ONLY that chunk's
    archetypes.  This pre-splits the archetype table (+ its cover weights + each
    loss term's target) into contiguous chunks of ``chunk`` archetypes; no
    reverse-mode VJP ever spans more than ``chunk`` archetypes (the CPU batched VJP
    segfaults at >=8).  Each chunk's UN-normalized weighted SSE (:func:`_weighted_sse`)
    + its gradient are accumulated (scalar add; pytree add), and the total is
    divided by the GLOBAL cover-weight sum ONCE at the end.  The result EQUALS the
    unchunked ``eqx.filter_value_and_grad`` of the cover-weighted MSE exactly (to
    floating point) -- the ``sum_w`` normalisation is applied once, never per chunk.

    Returns ``(value_and_grad_fn, loss_eval_fn)``:
      * ``value_and_grad_fn(params) -> (loss, grads)`` -- ``grads`` is a
        ``TrainablePhysicsParams`` pytree with the SAME structure
        ``eqx.filter_value_and_grad`` returns (static ``constraints`` intact,
        ``raw_values`` = summed/normalised grad arrays), so the training loop's
        ``_grad_stats`` / ``optax.global_norm`` / ``optimizer.update`` consume it
        unchanged.
      * ``loss_eval_fn(params) -> loss`` -- the chunked FORWARD-only loss for the
        backtracking line search (so it never batches more than ``chunk``
        archetypes either); numerically identical to the ``loss`` above.
    """
    from legoesm.land.carbon.global_init import equilibrate_archetypes_traced

    table = target_bundle["table"]
    losses = target_bundle["losses"]
    cover_weight = target_bundle["cover_weight"]
    n_arch = int(np.asarray(table.pft_id).shape[0])
    # GLOBAL cover-weight sum -- the single normaliser, applied ONCE (never per
    # chunk).  Concrete (built from the fixed cover weights, not traced).
    total_w = jnp.sum(cover_weight)
    bounds = _chunk_bounds(n_arch, chunk)
    # Optional single-step SIF forward per chunk (--with-sif); the SIF forward is
    # per-archetype + spin-up-free, so slicing it to the chunk's archetypes is exact and
    # adds no coupled-model VJP (the crash was the spin-up batch, not this leaf-sif graph).
    make_sif = target_bundle.get("make_sif_forward")

    # Pre-slice each chunk's STATIC table + its cover-weight / per-term target
    # slices ONCE (the archetype table + membership are non-differentiated).
    chunk_specs = []
    for (s, e) in bounds:
        ctable = _slice_table(table, np.arange(s, e))
        cw = cover_weight[s:e]
        cterms = {k: LossTerm(target=t.target[s:e], extractor=t.extractor,
                              weight=t.weight,
                              mask=(None if t.mask is None else t.mask[s:e]))
                  for k, t in losses.items()}
        csif = make_sif(ctable) if make_sif is not None else None
        chunk_specs.append((ctable, cw, cterms, csif))

    def _chunk_sse(ctable, cw, cterms, csif):
        """UN-normalized cover-weighted SSE over ONLY this chunk's archetypes."""
        def sse(params: TrainablePhysicsParams) -> jax.Array:
            overrides = _carbon_overrides(params)
            eq = equilibrate_archetypes_traced(ctable, overrides, **spin)
            preds = {"eq": eq}
            if csif is not None:
                preds["sif"] = csif(_sif_overrides(params))
            return _weighted_sse(preds, cterms, cw)
        return sse

    def value_and_grad_fn(params: TrainablePhysicsParams):
        total_sse = jnp.asarray(0.0, dtype=cover_weight.dtype)
        total_grad = None
        for (ctable, cw, cterms, csif) in chunk_specs:
            sse, grad = eqx.filter_value_and_grad(
                _chunk_sse(ctable, cw, cterms, csif))(params)
            total_sse = total_sse + sse
            total_grad = grad if total_grad is None else jax.tree_util.tree_map(
                lambda a, b: a + b, total_grad, grad)
        # Normalise ONCE by the global cover-weight sum: loss = sum_SSE / sum_w,
        # grad = sum_grad / sum_w -> exactly the unchunked cover-weighted-MSE grad.
        return total_sse / total_w, _scale_updates(total_grad, 1.0 / total_w)

    def loss_eval_fn(params: TrainablePhysicsParams) -> jax.Array:
        total_sse = jnp.asarray(0.0, dtype=cover_weight.dtype)
        for (ctable, cw, cterms, csif) in chunk_specs:
            total_sse = total_sse + _chunk_sse(ctable, cw, cterms, csif)(params)
        return total_sse / total_w

    return value_and_grad_fn, loss_eval_fn


def _build_grad_and_loss_fns(args, target_bundle, spin, precomputed):
    """Return ``(value_and_grad_fn, loss_eval_fn)`` for the active forward.

    Slow forward with ``--grad-chunk C > 0`` AND more than ``C`` archetypes ->
    the EXACT chunked accumulation (:func:`_make_chunked_slow`; bounds every
    coupled-spin-up VJP to <=C archetypes, past the CPU batched-VJP crash).  Fast
    forward, ``--grad-chunk 0``, or a table already <= C archetypes -> the
    single-batch ``eqx.filter_value_and_grad`` of the full loss (the unchanged
    production path -- fast mode is never chunked).
    """
    n_arch = int(np.asarray(target_bundle["table"].pft_id).shape[0])
    if (not args.fast_analytic) and args.grad_chunk > 0 and n_arch > args.grad_chunk:
        return _make_chunked_slow(target_bundle, spin, chunk=args.grad_chunk)
    loss_fn = _build_loss_fn(args, target_bundle, spin, precomputed)
    return eqx.filter_value_and_grad(loss_fn), loss_fn


# ===========================================================================
# Fast closed-form forward (B2-fast): differentiate the analytic SOM
# equilibrium, NOT the coupled spin-up.  The spin-up is run ONCE (no grad) to
# record the SOM-parameter-INDEPENDENT inputs; each optimizer step then costs a
# handful of vector ops instead of ~tens of minutes of grad-through-spin-up.
# ===========================================================================
def make_fast_analytic_forward(precomputed) -> Callable[[TrainablePhysicsParams], jax.Array]:
    """Build ``model_soc(params) -> (n_arch,)`` SOC [kgC/m2] from the closed-form
    SOM equilibrium.

    The trained SOM overrides are spliced into ``CarbonConfig`` via
    ``apply_param_overrides`` INSIDE the returned closure (traced --
    SegmentForcing/SCM-RCE override doctrine), then
    :func:`legoesm.land.carbon.fast_analytic.analytic_som_soc` forward-substitutes
    the cascade equilibrium from the precomputed inputs.  Production defaults are
    untouched (a fresh ``CarbonConfig`` per call; the overrides are the only
    substituted leaves).
    """
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.fast_analytic import analytic_som_soc
    from legoesm.training.param_collector import apply_param_overrides

    def model_soc(params: TrainablePhysicsParams) -> jax.Array:
        # Fresh CarbonConfig per call, overrides spliced in via _replace (traced,
        # immutable NamedTuple): production defaults are never mutated.
        cfg = apply_param_overrides(
            CarbonConfig(scheme="differland"), _carbon_overrides(params))
        return analytic_som_soc(precomputed, cfg) / _G_PER_KG   # gC/m2 -> kgC/m2

    return model_soc


def make_fast_loss_fn(
    precomputed, *, losses: dict[str, LossTerm], cover_weight: jax.Array,
    sif_forward=None,
) -> Callable[[TrainablePhysicsParams], jax.Array]:
    """Fast-analytic calibration loss, routed through the SAME registry as the slow path.

    The SOC term is the closed-form per-archetype SOM SOC (``make_fast_analytic_forward``,
    stored under ``preds["soc"]`` for :func:`_fast_soc_extractor`); ``jax.grad`` flows to
    the seven fast-valid SOM leaves (:data:`SOM_FIELDS`) through the analytic cascade at
    ~zero cost -- no spin-up in the gradient loop.  The SOM ``biomass`` / ``lai`` terms
    remain a v2 concern (the closed form is SOM-specific), but the single-step ``sif`` term
    (``--with-sif``) composes here too: it is spin-up-free, so it adds ``preds["sif"]`` and
    sums via ``_total_loss`` exactly as in the slow path.  With ``sif_forward=None`` and
    only ``soc`` in the registry this equals the prior SOC-only fast loss.
    """
    model_soc = make_fast_analytic_forward(precomputed)

    def loss_fn(params: TrainablePhysicsParams) -> jax.Array:
        preds = {"soc": model_soc(params)}
        if sif_forward is not None:
            preds["sif"] = sif_forward(_sif_overrides(params))
        return _total_loss(preds, losses, cover_weight)

    return loss_fn


def _build_loss_fn(args, target_bundle, spin, precomputed):
    """Return ``loss_fn(params) -> scalar`` for the active forward -- the fast
    closed-form SOC loss when ``--fast-analytic``, else the slow spin-up loss.  Both
    route through the registry ``_total_loss``; ``--with-sif`` adds the single-step SIF
    forward (built on the full table) to the predictions bundle in either mode."""
    make_sif = target_bundle.get("make_sif_forward")
    sif_forward = make_sif(target_bundle["table"]) if make_sif is not None else None
    if args.fast_analytic:
        return make_fast_loss_fn(
            precomputed, losses=target_bundle["losses"],
            cover_weight=target_bundle["cover_weight"], sif_forward=sif_forward)
    return make_loss_fn(
        table=target_bundle["table"], losses=target_bundle["losses"],
        cover_weight=target_bundle["cover_weight"], sif_forward=sif_forward, **spin)


def _grad_stats(grads: TrainablePhysicsParams, *, tol: float) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for c in grads.constraints:
        g = grads.raw_values[c.name]
        abs_max = float(jnp.max(jnp.abs(g)))
        finite = bool(jnp.all(jnp.isfinite(g)))
        stats[c.name] = {
            "abs_max": abs_max, "finite": finite,
            "nonzero": bool(finite and abs_max > tol),
        }
    return stats


def _assert_bounds(params: TrainablePhysicsParams) -> None:
    physical = params.as_dict()
    for c in params.constraints:
        v = physical[c.name]
        if not bool(jnp.all(jnp.isfinite(v))):
            raise RuntimeError(f"{c.name} non-finite after transform")
        lo = jnp.asarray(c.min_val, dtype=v.dtype)
        hi = jnp.asarray(c.max_val, dtype=v.dtype)
        if not bool(jnp.all((v >= lo) & (v <= hi))):
            raise RuntimeError(
                f"{c.name}={np.asarray(v)} escaped bounds "
                f"({c.min_val}, {c.max_val})")


def _scale_updates(updates, scale: float):
    return jax.tree_util.tree_map(lambda x: x * scale, updates)


# ===========================================================================
# Observed-SOC target assembly (fixed, built once, non-diff)
# ===========================================================================
def _resolve_om_to_oc(args) -> float:
    """Preset-aware organic-matter->carbon factor (``--om-to-oc`` overrides).

    Default depends on the surfdata source: raw CLM5 ``ORGANIC`` is organic MATTER
    -> ``_VAN_BEMMELEN_OC_PER_OM`` (0.58); harmonized ``legoesm_surfdata``
    ``organic`` derives from HWSD ORG_CARBON (already carbon) -> 1.0.  An explicit
    ``--om-to-oc`` wins for a source whose convention differs.
    """
    if args.om_to_oc is not None:
        return float(args.om_to_oc)
    return _VAN_BEMMELEN_OC_PER_OM if args.surfdata_preset == "clm5_surfdata" else 1.0


def _load_surfdata_organic(surf_path: str, *, om_to_oc: float,
                           soil_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell organic-carbon column ``(ncell, n_layer)`` [kgC/m3] + layer
    thicknesses ``dz`` ``(n_layer,)`` [m] from a raw CLM5 surfdata NetCDF.

    ``ORGANIC(nlevsoi, lsmlat, lsmlon)`` is reshaped to the SAME row-major
    ``(ncell = nlat*nlon, n_layer)`` order the cover loader
    (:func:`build_global_carbon_ic._load_clm5_cover_soil`) uses, so cell indices
    align with the archetype membership.  ``om_to_oc`` scales organic MATTER ->
    organic CARBON (van Bemmelen ~ 0.58 = 1/1.724); default ``1.0`` treats
    ``ORGANIC`` as already carbon (the ``soc_observations`` convention).  Layer
    thicknesses reuse the MODEL soil grid (``make_soil_grid``) at ``n_layer``
    levels over ``soil_depth`` -- so the observed column integrates over the SAME
    depth the modelled ``som_total`` represents (no re-derived geometry); the raw
    CLM5 surfdata carries no ``DZSOI``.
    """
    import xarray as xr

    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid

    ds = xr.open_dataset(surf_path, decode_times=False)
    try:
        # Raw CLM5 surfdata names the field "ORGANIC"; the harmonized
        # legoesm_surfdata schema uses lowercase "organic" -- accept either.
        var = next((v for v in ("ORGANIC", "organic") if v in ds), None)
        if var is None:
            raise SystemExit(
                f"surfdata {surf_path} has no ORGANIC/organic variable; available "
                f"soil vars: {sorted(v for v in ds.data_vars if 'ORG' in v.upper() or 'SOIL' in v.upper())}"
                f" (full: {sorted(ds.data_vars)[:40]}...)")
        org = np.asarray(ds[var].values, dtype=float)   # (nlev, nlat, nlon)
    finally:
        ds.close()
    if org.ndim != 3:
        raise SystemExit(
            f"ORGANIC expected (nlev, nlat, nlon); got shape {org.shape}.")
    n_layer, n_lat, n_lon = org.shape
    # (ncell, n_layer) row-major (i_lat, i_lon) -- matches the cover reshape.
    organic = np.moveaxis(org, 0, -1).reshape(n_lat * n_lon, n_layer)
    organic = np.nan_to_num(organic, nan=0.0) * float(om_to_oc)
    grid = make_soil_grid(SoilGridConfig(
        n_layers=n_layer, total_depth=float(soil_depth), growth_factor=1.5))
    dz = np.asarray(grid.dz, dtype=float)
    return organic, dz


def _synthetic_observed_soc(table) -> np.ndarray:
    """A deterministic, physically-oriented per-archetype observed SOC [kgC/m2]
    for ``--dry-run-synthetic`` (no data files): colder archetypes hold more SOC,
    so the freeze-floor / turnover parameters have signal to fit.
    """
    mat_c = np.asarray(table.mat_k, float) - constants.T_freeze
    # 15 kgC/m2 baseline (deliberately offset from the model's default warm-soil
    # equilibrium so the synthetic dry-run has a clear residual + gradient signal)
    # + 0.5 per degC below freezing (cold soils retain more carbon).
    return 15.0 + 0.5 * np.maximum(0.0, -mat_c)


def _stratified_subsample(table, max_archetypes: int, *, seed: int) -> np.ndarray:
    """Indices of a representative subset spanning the PFT / climate range.

    Keeps up to ``ceil(max/n_pft)`` archetypes per PFT, evenly spaced along that
    PFT's mean-annual-temperature order (so warm..cold extremes survive), capped
    to ``max_archetypes`` total.  The tuned parameters apply globally, so a
    representative subset suffices for the gradient signal.
    """
    pft = np.asarray(table.pft_id, int)
    mat = np.asarray(table.mat_k, float)
    n_arch = pft.shape[0]
    if max_archetypes <= 0 or n_arch <= max_archetypes:
        return np.arange(n_arch)
    uniq = np.unique(pft)
    per = max(1, int(np.ceil(max_archetypes / uniq.size)))
    keep: list[int] = []
    for p in uniq:
        idx = np.where(pft == p)[0]
        idx = idx[np.argsort(mat[idx])]                 # cold..warm within PFT
        take = min(per, idx.size)
        pick = np.linspace(0, idx.size - 1, take).round().astype(int)
        keep.extend(int(idx[j]) for j in np.unique(pick))
    keep = sorted(set(keep))
    if len(keep) > max_archetypes:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(keep, size=max_archetypes, replace=False).tolist())
    return np.asarray(keep, int)


def _slice_table(table, idx: np.ndarray):
    from legoesm.land.carbon.global_init import ArchetypeTable
    idx = np.asarray(idx, int)
    return ArchetypeTable(**{f: np.asarray(getattr(table, f))[idx]
                             for f in table._fields})


def build_target(args) -> dict[str, Any]:
    """Assemble the calibration target: the (possibly subsampled) ArchetypeTable,
    the observed per-archetype SOC [kgC/m2], and the per-archetype cover weight.

    Reuses the driver's real-data path (``build_global_carbon_ic.load_real_inputs``
    / ``_load_clm5_cover_soil``) + ``build_archetypes`` for the table + membership,
    then ``soc_observations`` for the cover-weighted observed SOC.  Returns a dict
    with ``table``, ``losses`` (the registry), ``cover_weight``, ``pft_id``, and
    provenance.
    """
    from legoesm.land.carbon.climate_features import reduce_climatology_to_features
    from legoesm.land.carbon.global_init import build_archetypes
    from legoesm.land.carbon.soc_observations import (
        per_archetype_cover_weight, per_archetype_observed_soc,
    )

    import scripts.data.build_global_carbon_ic as ic

    if args.dry_run_synthetic:
        inputs = ic.build_synthetic_inputs()
        source = "synthetic"
    else:
        # --archetypes only supplies geometry/provenance; the table + membership
        # are (re)built from the real cover so they are guaranteed consistent with
        # the ORGANIC grid (a saved archetypes.npz carries no cell membership).
        inputs = ic.load_real_inputs(args)
        source = f"rebuild:{args.surfdata_preset}"

    features = reduce_climatology_to_features(
        inputs.monthly_t_k, inputs.monthly_precip, inputs.monthly_sw,
        inputs.monthly_netrad)
    table, cell_id, cell_w = build_archetypes(
        inputs.pft_weights, features, inputs.soil_class, inputs.land_mask,
        k_per_pft=args.k_per_pft, w_min=args.w_min, seed=args.seed)
    n_arch_full = int(np.asarray(table.pft_id).shape[0])

    cover_full = per_archetype_cover_weight(cell_id, cell_w, n_arch=n_arch_full)
    if args.dry_run_synthetic:
        observed_full = _synthetic_observed_soc(table)
    else:
        # The observed-SOC organic column is read on the surfdata's NATIVE grid,
        # which matches the clm5_surfdata cover path (also native, no regrid).  The
        # legoesm_surfdata cover path REGRIDS to --resolution-deg, so a native-grid
        # organic read would mismatch the cover ncell unless the file is already on
        # the target grid -- gate it explicitly (a documented follow-up: regrid the
        # organic column through the same GlobalSurfaceData.organic path) rather
        # than emit a confusing downstream "grid mismatch".
        if args.surfdata_preset != "clm5_surfdata":
            raise SystemExit(
                f"observed-SOC organic column is only wired for the native-grid "
                f"'clm5_surfdata' preset; '{args.surfdata_preset}' regrids the "
                f"cover to --resolution-deg and the organic column is not yet "
                f"regridded to match (follow-up). Use --surfdata-preset "
                f"clm5_surfdata for Stage-B calibration.")
        om_to_oc = _resolve_om_to_oc(args)
        print(f"[target] organic-matter->carbon factor om_to_oc={om_to_oc:.4g} "
              f"(preset={args.surfdata_preset})")
        organic, dz = _load_surfdata_organic(
            args.surf_path, om_to_oc=om_to_oc, soil_depth=args.soil_depth)
        if organic.shape[0] != inputs.pft_weights.shape[0]:
            raise SystemExit(
                f"ORGANIC ncell {organic.shape[0]} != cover ncell "
                f"{inputs.pft_weights.shape[0]}; grid mismatch.")
        observed_full = per_archetype_observed_soc(
            organic, dz, cell_id, cell_w, n_arch=n_arch_full)

    # Observed SIF target (--with-sif): a per-archetype cover-weighted mean, mirroring
    # the SOC target.  Synthetic in the dry-run; else a gridded satellite SIF product,
    # which (like the SOC organic column) is only wired for the native-grid clm5_surfdata
    # cover.  Built on the FULL table (subsample below).
    observed_sif_full = None
    if args.with_sif:
        from legoesm.land.carbon.sif_observations import (
            load_gridded_sif, per_archetype_observed_sif, synthetic_observed_sif,
        )
        if args.dry_run_synthetic:
            observed_sif_full = synthetic_observed_sif(table)
        else:
            if args.surfdata_preset != "clm5_surfdata":
                raise SystemExit(
                    "--with-sif observed SIF is only wired for the native-grid "
                    "'clm5_surfdata' preset (the gridded SIF must match the cover grid); "
                    f"'{args.surfdata_preset}' regrids the cover. Use clm5_surfdata.")
            ncell = int(inputs.pft_weights.shape[0])
            sif_cell = load_gridded_sif(
                args.sif_obs, ncell=ncell, sif_var=(args.sif_var or None))
            observed_sif_full = per_archetype_observed_sif(
                sif_cell, cell_id, cell_w, n_arch=n_arch_full)

    keep = _stratified_subsample(table, args.max_archetypes, seed=args.seed)
    table = _slice_table(table, keep)
    observed = np.asarray(observed_full)[keep]
    cover = np.asarray(cover_full)[keep]
    observed_sif = (np.asarray(observed_sif_full)[keep]
                    if observed_sif_full is not None else None)

    # Sanitise with PER-STREAM finite masks.  The cover weight is SHARED across registry
    # terms, so an archetype is ACTIVE if it has ANY finite observation (finite_any) with
    # positive cover; each term then MASKS the archetypes where ITS OWN observation is
    # missing.  So a SOC-observed-but-SIF-missing archetype keeps its SOC signal (and vice
    # versa) -- the streams need not overlap -- while the single global active-cover sum
    # stays the denominator (preserving the chunked-gradient exactness).  Targets are
    # sanitised NaN->0 where their own mask is 0, so 0*mask never poisons the sum.
    finite_soc = np.isfinite(observed)
    finite_any = finite_soc
    finite_sif = None
    if observed_sif is not None:
        finite_sif = np.isfinite(observed_sif)
        finite_any = finite_soc | finite_sif
    cover = np.where(finite_any & (cover > 0.0), cover, 0.0)
    observed = np.where(finite_soc, observed, 0.0)
    if observed_sif is not None:
        observed_sif = np.where(finite_sif, observed_sif, 0.0)
    if not np.any(cover > 0.0):
        raise SystemExit(
            "no archetype has a finite observed "
            + ("SOC or SIF" if observed_sif is not None else "SOC")
            + " with positive cover; cannot fit.")
    if observed_sif is not None and not np.any(finite_sif & (cover > 0.0)):
        # --with-sif was requested but NO archetype has a usable SIF observation (an
        # all-gap product, or a grid that does not overlap the land cover). Fail loud
        # rather than silently train a dead SIF term that reports a fake 0 RMSE.
        raise SystemExit(
            "--with-sif but no archetype has a finite observed SIF with positive cover "
            "(the gridded SIF product may be all-gap over the land cover or on a "
            "mismatched grid); provide a SIF product overlapping the surfdata cover, or "
            "drop --with-sif.")

    # Mode-aware SOC extractor: the FAST closed form supplies SOC directly (preds["soc"]);
    # the SLOW forward supplies the equilibrium CarbonState (preds["eq"]).
    soc_extractor = _fast_soc_extractor if args.fast_analytic else _soc_extractor
    # SOC mask only matters when another stream keeps an archetype active where SOC is
    # missing; SOC-only, the cover-zeroing already drops NaN-SOC archetypes (mask=None,
    # the unchanged single-stream path).
    soc_mask = (jnp.asarray(finite_soc, dtype=jnp.float64)
                if observed_sif is not None else None)
    losses = {
        "soc": LossTerm(target=jnp.asarray(observed, dtype=jnp.float64),
                        extractor=soc_extractor, weight=W_SOC, mask=soc_mask),
        # TODO (Stage-B v2): add `biomass` (CLM5 monthly biomass), `lai`, `d13c` entries
        # here -- each a LossTerm(target, extractor, weight[, mask]) -- and the loop picks
        # them up with no further change.
    }
    make_sif_forward = None
    if observed_sif is not None:
        # SIF is added to the SAME registry (summed via _total_loss, no loop change); its
        # simulated forward is the single-step, spin-up-free _make_sif_forward.
        losses["sif"] = LossTerm(
            target=jnp.asarray(observed_sif, dtype=jnp.float64),
            extractor=_sif_extractor, weight=args.sif_weight,
            mask=jnp.asarray(finite_sif, dtype=jnp.float64))
        make_sif_forward = _make_sif_forward

    n_kept = int(table.pft_id.shape[0])
    n_active = int(np.sum(cover > 0.0))
    soc_active = finite_soc & (cover > 0.0)
    sif_note = ""
    if observed_sif is not None:
        sif_active = finite_sif & (cover > 0.0)
        n_soc = int(np.sum(soc_active))
        n_sif = int(np.sum(sif_active))
        sif_lo, sif_hi = ((np.min(observed_sif[sif_active]),
                           np.max(observed_sif[sif_active])) if n_sif else (float("nan"),) * 2)
        sif_note = (f"; SOC obs on {n_soc}, SIF obs on {n_sif} of {n_active}; "
                    f"observed SIF [{sif_lo:.2f}, {sif_hi:.2f}] umol/m2/s (w={args.sif_weight:g})")
    soc_lo, soc_hi = ((np.min(observed[soc_active]), np.max(observed[soc_active]))
                      if np.any(soc_active) else (float("nan"),) * 2)
    print(f"[target] source={source} archetypes: {n_arch_full} built -> "
          f"{n_kept} kept ({n_active} active); "
          f"observed SOC [{soc_lo:.2f}, {soc_hi:.2f}] kgC/m2{sif_note}")
    return {
        "table": table,
        "losses": losses,
        "cover_weight": jnp.asarray(cover, dtype=jnp.float64),
        "observed_soc": observed,
        "observed_sif": observed_sif,
        "make_sif_forward": make_sif_forward,
        "pft_id": np.asarray(table.pft_id, int),
        "source": source,
        "n_arch_full": n_arch_full,
        "n_arch_kept": n_kept,
    }


# ===========================================================================
# Scorecard + output
# ===========================================================================
def _per_pft_soc(pft_id, values, cover_weight) -> dict[str, float]:
    """Cover-weighted mean SOC [kgC/m2] per PFT name over the kept archetypes."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    out: dict[str, float] = {}
    vals = np.asarray(values, float)
    cov = np.asarray(cover_weight, float)
    for p in np.unique(pft_id):
        sel = (pft_id == p) & (cov > 0.0)
        name = CLM5_PFT_NAMES[int(p)] if int(p) < len(CLM5_PFT_NAMES) else f"pft{p}"
        if not np.any(sel):
            continue
        w = cov[sel]
        out[name] = float(np.sum(w * vals[sel]) / np.sum(w))
    return out


def _slow_model_soc(params, target_bundle, spin, *, grad_chunk: int = 0) -> np.ndarray:
    """Per-archetype modelled SOC [kgC/m2] via the SLOW grad-through-spin-up
    forward (no grad here -- scorecard evaluation only).

    ``grad_chunk > 0`` chunks the forward the SAME way the chunked trainer chunks
    the VJP, so the scorecard never batches more than ``grad_chunk`` archetypes
    either (the batched coupled model is the CPU crash risk).  The per-archetype
    SOC is concatenated in table order and is identical to a single-batch forward
    (independent columns)."""
    from legoesm.land.carbon.global_init import equilibrate_archetypes_traced
    table = target_bundle["table"]
    n_arch = int(np.asarray(table.pft_id).shape[0])
    overrides = _carbon_overrides(params)
    socs = []
    for (s, e) in _chunk_bounds(n_arch, grad_chunk):
        ctable = _slice_table(table, np.arange(s, e))
        eq = equilibrate_archetypes_traced(ctable, overrides, **spin)
        socs.append(np.asarray(_soc_extractor({"eq": eq})))
    return np.concatenate(socs) if len(socs) > 1 else socs[0]


def _build_model_soc_fn(args, target_bundle, spin, precomputed):
    """Return ``model_soc_fn(params) -> (n_arch,) kgC/m2`` for the active forward
    (fast closed-form when ``--fast-analytic``, else the slow spin-up -- chunked to
    ``--grad-chunk`` archetypes so the scorecard forward matches the trainer)."""
    if args.fast_analytic:
        fwd = make_fast_analytic_forward(precomputed)
        return lambda params: np.asarray(fwd(params))
    return lambda params: _slow_model_soc(
        params, target_bundle, spin, grad_chunk=args.grad_chunk)


def _precompute_and_check(table, initial_params, spin, cover_weight=None):
    """Run the ONE-TIME default-parameter precompute spin-up and VERIFY the
    closed-form SOM SOC reproduces the spin-up equilibrium at the defaults.

    Returns ``(FastAnalyticInputs, match_dict)``.  The match is reported but never
    auto-fails here -- the calibration proceeds and the scorecard/report surface
    the fidelity so a large mismatch is visible rather than silently shipped.

    Reports THREE fidelity metrics because the raw per-archetype mean relative
    error over-states the surrogate's error for the calibration's purpose:
    * ``mean_abs_rel_err`` -- raw mean |analytic-real|/|real|; INFLATED by
      near-zero-SOM (cold/dead) archetypes (a tiny absolute miss on a ~0 pool is a
      huge %), which contribute ~0 to the cover-weighted loss.
    * ``cover_weighted_rel_err`` -- the LOSS-RELEVANT metric:
      ``sqrt(Σ w·(analytic-real)²/Σ w) / (Σ w·real/Σ w)`` -- the cover-weighted
      RMS SOC error relative to the cover-weighted mean SOC, i.e. how well the
      surrogate tracks the actual cover-weighted-MSE objective.
    * ``high_som_mean_rel_err`` -- mean |rel| over archetypes with real SOC > 5
      kgC/m2 (the meaningful-stock cells).
    """
    from legoesm.land.carbon.global_init import precompute_fast_analytic_inputs

    t0 = time.time()
    precomputed, real_som = precompute_fast_analytic_inputs(
        table, n_spinup=spin["n_spinup"], n_verify=spin["n_verify"],
        dt=spin["dt"], n_layers=spin["n_layers"], soil_depth=spin["soil_depth"])
    # Closed-form SOC at the warm-started production defaults vs the spin-up
    # equilibrium SOM, both in kgC/m2 (the warm start reproduces the defaults to
    # sigmoid-clamp precision, so this is the analytic-vs-spin-up fidelity).
    model_soc = make_fast_analytic_forward(precomputed)
    analytic_kg = np.asarray(model_soc(initial_params))
    real_kg = np.asarray(real_som) / _G_PER_KG
    abs_err = np.abs(analytic_kg - real_kg)
    denom = np.maximum(np.abs(real_kg), 1e-6)
    rel = abs_err / denom
    # Loss-relevant: cover-weighted RMS SOC error / cover-weighted mean SOC.
    if cover_weight is not None:
        w = np.asarray(cover_weight, float)
        wsum = float(np.sum(w)) or 1.0
        cw_rmse = float(np.sqrt(np.sum(w * (analytic_kg - real_kg) ** 2) / wsum))
        cw_mean_real = float(np.sum(w * real_kg) / wsum)
        cover_weighted_rel = cw_rmse / max(abs(cw_mean_real), 1e-6)
    else:
        cover_weighted_rel = float("nan")
    hi = real_kg > 5.0  # meaningful-stock archetypes
    high_som_mean_rel = float(np.mean(rel[hi])) if bool(np.any(hi)) else float("nan")
    match = {
        "n_archetypes": int(real_kg.shape[0]),
        "mean_abs_rel_err": float(np.mean(rel)),
        "max_abs_rel_err": float(np.max(rel)),
        "cover_weighted_rel_err": cover_weighted_rel,
        "high_som_mean_rel_err": high_som_mean_rel,
        "n_high_som": int(np.sum(hi)),
        "max_abs_err_kgC_m2": float(np.max(abs_err)),
        "spinup_som_kgC_m2_range": [float(np.min(real_kg)), float(np.max(real_kg))],
        "precompute_seconds": float(time.time() - t0),
    }
    print(f"[fast-analytic] precompute {match['precompute_seconds']:.1f}s; "
          f"analytic-vs-spin-up SOC match over {match['n_archetypes']} archetypes: "
          f"raw-mean|rel|={match['mean_abs_rel_err']:.2%} (inflated by near-zero cells)  "
          f"COVER-WEIGHTED|rel|={match['cover_weighted_rel_err']:.2%}  "
          f"high-SOM(>5)|rel|={match['high_som_mean_rel_err']:.2%} (n={match['n_high_som']})  "
          f"(spin-up SOM {match['spinup_som_kgC_m2_range'][0]:.1f}..."
          f"{match['spinup_som_kgC_m2_range'][1]:.1f} kgC/m2)")
    if match["mean_abs_rel_err"] > _MATCH_WARN_REL:
        print(f"[WARN] fast-analytic forward diverges from the spin-up "
              f"(mean|rel|={match['mean_abs_rel_err']:.1%} > {_MATCH_WARN_REL:.0%}); "
              f"the closed form may not faithfully track the model -- inspect "
              f"before trusting the tuned parameters (raise --n-spinup / --dt).")
    return precomputed, match


def _write_scorecard(path: Path, *, target_bundle, initial_params, tuned_params,
                     model_soc_fn) -> dict[str, Any]:
    pft_id = target_bundle["pft_id"]
    cover = np.asarray(target_bundle["cover_weight"])
    observed = target_bundle["observed_soc"]
    soc_default = model_soc_fn(initial_params)
    soc_tuned = model_soc_fn(tuned_params)

    # Weight SOC diagnostics by cover MASKED to the SOC-observed archetypes (matches the
    # loss's soc mask); mask=None (SOC-only path) -> cover unchanged.
    soc_mask = target_bundle["losses"]["soc"].mask
    w_soc = cover if soc_mask is None else cover * np.asarray(soc_mask)
    obs_pft = _per_pft_soc(pft_id, observed, w_soc)
    def_pft = _per_pft_soc(pft_id, soc_default, w_soc)
    tuned_pft = _per_pft_soc(pft_id, soc_tuned, w_soc)
    per_pft = {
        name: {
            "observed_kgC_m2": obs_pft.get(name),
            "modeled_default_kgC_m2": def_pft.get(name),
            "modeled_tuned_kgC_m2": tuned_pft.get(name),
        }
        for name in sorted(set(obs_pft) | set(def_pft) | set(tuned_pft))
    }

    def _wrmse(a, b, weight):
        wsum = float(np.sum(weight))
        if wsum <= 0.0:
            return float("nan")     # no scored archetypes -> "no score", never a fake 0
        return float(np.sqrt(np.sum(
            weight * (np.asarray(a) - np.asarray(b)) ** 2) / wsum))
    rmse_default = _wrmse(soc_default, observed, w_soc)
    rmse_tuned = _wrmse(soc_tuned, observed, w_soc)

    init_vals = _param_values(initial_params)
    tuned_vals = _param_values(tuned_params)
    params_delta = {
        f: {"default": init_vals[f], "tuned": tuned_vals[f]}
        for f in sorted(tuned_vals)
    }
    scorecard = {
        "source": target_bundle["source"],
        "n_archetypes_full": target_bundle["n_arch_full"],
        "n_archetypes_kept": target_bundle["n_arch_kept"],
        "cover_weighted_soc_rmse_kgC_m2": {
            "default": rmse_default, "tuned": rmse_tuned,
            "improvement": rmse_default - rmse_tuned,
        },
        "per_pft_soc": per_pft,
        "parameters_default_vs_tuned": params_delta,
    }
    # Optional SIF scorecard (--with-sif): cover-weighted simulated-SIF RMSE vs observed,
    # default vs tuned (single-step forward; surfaces whether the sif term improved).
    make_sif = target_bundle.get("make_sif_forward")
    observed_sif = target_bundle.get("observed_sif")
    if make_sif is not None and observed_sif is not None:
        sif_fwd = make_sif(target_bundle["table"])
        sif_default = np.asarray(sif_fwd(_sif_overrides(initial_params)))
        sif_tuned = np.asarray(sif_fwd(_sif_overrides(tuned_params)))
        # Weight by cover MASKED to the SIF-observed archetypes (matches the loss's sif
        # mask) so missing-SIF cells never enter the RMSE.
        sif_mask = np.asarray(target_bundle["losses"]["sif"].mask)
        w_sif = cover * sif_mask
        scorecard["cover_weighted_sif_rmse_umol_m2_s"] = {
            "default": _wrmse(sif_default, observed_sif, w_sif),
            "tuned": _wrmse(sif_tuned, observed_sif, w_sif),
            "improvement": (_wrmse(sif_default, observed_sif, w_sif)
                            - _wrmse(sif_tuned, observed_sif, w_sif)),
        }
    path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    return scorecard


def _write_tuned_json(path: Path, *, tuned_params, initial_params, loss_history,
                      meta) -> dict[str, Any]:
    """Human-readable RECOMMENDED tuned params (never mutates CarbonConfig/SIFConfig)."""
    from legoesm.training.param_collector import build_registry
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.canopy.sif import SIFConfig

    registry = {m.qualified_name: m for m in build_registry()}
    # Per-scheme production defaults (carbon + SIF); a field is looked up in the config
    # its scheme_key names, so SIF fields (kn0, ...) resolve against SIFConfig, not
    # CarbonConfig (which would KeyError).
    defaults_by_scheme = {
        CARBON_SCHEME_KEY: CarbonConfig()._asdict(),
        SIF_SCHEME_KEY: SIFConfig()._asdict(),
    }
    tuned_vals = _param_values(tuned_params)
    init_vals = _param_values(initial_params)
    rows = []
    tuned_fields: dict[str, float] = {}
    for c in tuned_params.constraints:
        meta_c = registry[c.name]
        tuned = tuned_vals[c.field]
        if not np.isfinite(tuned):
            raise RuntimeError(f"tuned {c.field} is non-finite: {tuned}")
        tuned_fields[c.field] = tuned
        rows.append({
            "field": c.field,
            "scheme_key": c.scheme_key,
            "units": meta_c.units,
            "production_default": float(defaults_by_scheme[c.scheme_key][c.field]),
            "warm_start": init_vals[c.field],
            "tuned": tuned,
            "lower": float(c.min_val),
            "upper": float(c.max_val),
        })
    scheme_keys = sorted({c.scheme_key for c in tuned_params.constraints})
    payload = {
        "note": ("RECOMMENDED DifferLand carbon"
                 + ("+SIF" if SIF_SCHEME_KEY in scheme_keys else "")
                 + " calibration -- a recommendation, NOT a mutation of the production "
                 "CarbonConfig/SIFConfig defaults."),
        "scheme_key": CARBON_SCHEME_KEY,
        "scheme_keys": scheme_keys,
        "optimizer": meta["optimizer"],
        "training": meta["training"],
        "tuned_carbon_parameters": tuned_fields,
        "parameters": rows,
        "loss_history": [float(x) for x in loss_history],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _plot_loss(path: Path, losses: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.arange(len(losses)), losses, marker="o", color="#1f4e79")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cover-weighted calibration MSE  (soc[+sif], summed)")
    ax.set_title("Carbon SOM calibration loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ===========================================================================
# Training loop
# ===========================================================================
def train(args: argparse.Namespace) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    spin = {
        "n_spinup": args.n_spinup, "n_verify": args.n_verify, "dt": args.dt,
        "n_layers": args.n_layers, "soil_depth": args.soil_depth,
    }

    t_assemble = time.time()
    target_bundle = build_target(args)
    print(f"[target] assembled in {time.time() - t_assemble:.1f}s")

    initial_params = build_carbon_trainables(
        som_only=not args.all_carbon_params, with_sif=args.with_sif)
    params = initial_params
    n_sif = sum(c.name.startswith(SIF_SCHEME_KEY + ".") for c in params.constraints)
    print(f"[preflight] {len(params.constraints)} trainable params "
          f"(SOM-only={not args.all_carbon_params}, with_sif={args.with_sif}"
          f"{f', {n_sif} SIF' if args.with_sif else ''}); mode="
          f"{'fast-analytic' if args.fast_analytic else 'slow-spinup-grad'}; "
          f"n_spinup={args.n_spinup}, max_archetypes={args.max_archetypes}")

    # Fast-analytic: run the spin-up ONCE (no grad) to record the SOM-parameter-
    # INDEPENDENT inputs, then differentiate the closed-form cascade equilibrium.
    precomputed = None
    match_info: dict[str, Any] = {}
    if args.fast_analytic:
        precomputed, match_info = _precompute_and_check(
            target_bundle["table"], initial_params, spin,
            cover_weight=target_bundle["cover_weight"])

    # (value_and_grad_fn, loss_eval_fn) for the active forward. In --slow-spinup-grad
    # with --grad-chunk C>0 and >C archetypes these are the EXACT chunked
    # accumulation (each per-chunk VJP <=C archetypes, past the CPU batched-VJP
    # crash); otherwise the single-batch eqx.filter_value_and_grad of the full loss.
    # Neither closes over `params`, so both survive the preflight param filter below.
    value_and_grad_fn, loss_eval_fn = _build_grad_and_loss_fns(
        args, target_bundle, spin, precomputed)
    if not args.fast_analytic and args.grad_chunk > 0:
        n_arch_kept = target_bundle["n_arch_kept"]
        n_chunks = len(_chunk_bounds(n_arch_kept, args.grad_chunk))
        print(f"[preflight] slow-spinup-grad chunked: {n_chunks} chunk(s) of "
              f"<= {args.grad_chunk} archetypes over {n_arch_kept} kept "
              f"(EXACT cover-weighted-MSE gradient, normalised once at the end)")

    # Preflight: finite + non-zero gradient gate (drop dead DOFs before MUON).
    t0 = time.time()
    pre_loss, pre_grads = value_and_grad_fn(params)
    pre_loss_val = float(pre_loss)
    if not np.isfinite(pre_loss_val):
        raise RuntimeError(f"preflight loss non-finite: {pre_loss_val}")
    pre_stats = _grad_stats(pre_grads, tol=args.grad_nonzero_tol)
    keep_names = {n for n, s in pre_stats.items() if s["nonzero"]}
    frozen = {n: ("non-finite grad" if not s["finite"] else "zero preflight grad")
              for n, s in pre_stats.items() if not s["nonzero"]}
    if not keep_names:
        raise RuntimeError(
            f"no carbon param has a finite non-zero gradient: {pre_stats}")
    if frozen:
        print(f"[preflight] freezing {sorted(frozen)} ({sorted(set(frozen.values()))})")
    params = _filter_params(params, keep_names)
    _assert_bounds(params)
    print(f"[preflight] loss={pre_loss_val:.8g}  trainable={len(params.constraints)}  "
          f"({time.time() - t0:.1f}s)")

    optimizer = create_optimizer(TrainingConfig(
        lr=args.lr, warmup_steps=args.warmup_steps,
        total_steps=max(args.steps, 1), grad_clip_norm=args.grad_clip_norm,
        optimizer=args.optimizer))
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    loss_history = [pre_loss_val]
    no_improve = 0
    print(f"[train] step=0 loss={pre_loss_val:.8g}")
    for step in range(1, args.steps + 1):
        ts = time.time()
        loss, grads = value_and_grad_fn(params)
        loss_val = float(loss)
        gstats = _grad_stats(grads, tol=args.grad_nonzero_tol)
        bad = {n: s for n, s in gstats.items() if not s["finite"]}
        if bad:
            raise RuntimeError(f"non-finite gradient at step {step}: {bad}")
        grad_norm = float(optax.global_norm(eqx.filter(grads, eqx.is_array)))
        updates, opt_state_next = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(params, eqx.is_array))
        # Backtracking line search: accept the first scale that reduces the loss
        # and keeps every param in-bounds -> monotone-decreasing, never divergent.
        best_params, best_loss, accepted = params, loss_val, False
        for scale in _LINE_SEARCH_SCALES:
            cand = eqx.apply_updates(params, _scale_updates(updates, scale))
            _assert_bounds(cand)
            cand_loss = float(loss_eval_fn(cand))
            if np.isfinite(cand_loss) and cand_loss < best_loss:
                best_params, best_loss, accepted = cand, cand_loss, True
                break
        # ALWAYS advance the optimizer schedule + momentum so the LR warmup can
        # ramp: a non-reducing EARLY step (tiny warmup LR, or a first step before
        # MUON momentum builds) must NOT be mistaken for convergence. Only stop
        # after ``--patience`` CONSECUTIVE non-reducing steps.
        opt_state = opt_state_next
        if accepted:
            params = best_params
            loss_history.append(best_loss)
            no_improve = 0
            print(f"[train] step={step} loss={best_loss:.8g} |g|={grad_norm:.3e} "
                  f"time={time.time() - ts:.1f}s")
        else:
            no_improve += 1
            loss_history.append(loss_val)
            print(f"[train] step={step} no-reduce {no_improve}/{args.patience} "
                  f"loss={loss_val:.8g} |g|={grad_norm:.3e} time={time.time() - ts:.1f}s")
            if no_improve >= args.patience:
                print(f"[train] stop step={step}: no reduction for "
                      f"{args.patience} consecutive steps (loss={loss_val:.8g})")
                break

    _assert_bounds(params)
    improved = loss_history[-1] < loss_history[0]

    meta = {
        "optimizer": {
            "name": args.optimizer, "lr": args.lr,
            "warmup_steps": args.warmup_steps, "total_steps": args.steps,
            "grad_clip_norm": args.grad_clip_norm,
            "schedule": "linear warmup then cosine decay to zero",
        },
        "training": {
            "forward": "fast_analytic" if args.fast_analytic else "slow_spinup_grad",
            "n_spinup": args.n_spinup, "n_verify": args.n_verify, "dt": args.dt,
            "n_layers": args.n_layers, "soil_depth": args.soil_depth,
            "max_archetypes": args.max_archetypes,
            "grad_chunk": (args.grad_chunk if not args.fast_analytic else 0),
            "som_only": not args.all_carbon_params,
            "with_sif": bool(args.with_sif),
            "sif_weight": (args.sif_weight if args.with_sif else None),
            "loss_terms": sorted(target_bundle["losses"]),
            "source": target_bundle["source"],
            "n_archetypes_full": target_bundle["n_arch_full"],
            "n_archetypes_kept": target_bundle["n_arch_kept"],
            "frozen_params": sorted(frozen),
            "initial_loss": loss_history[0], "final_loss": loss_history[-1],
            "loss_improved": bool(improved),
            "analytic_vs_spinup_match": match_info,
        },
    }
    model_soc_fn = _build_model_soc_fn(args, target_bundle, spin, precomputed)
    tuned_json = args.outdir / "tuned_carbon_parameters.json"
    payload = _write_tuned_json(
        tuned_json, tuned_params=params, initial_params=initial_params,
        loss_history=loss_history, meta=meta)
    scorecard = _write_scorecard(
        args.outdir / "scorecard.json", target_bundle=target_bundle,
        initial_params=initial_params, tuned_params=params,
        model_soc_fn=model_soc_fn)
    _plot_loss(args.outdir / "loss_curve.png", loss_history)

    print(f"[done] wrote {tuned_json}")
    sfl = payload["tuned_carbon_parameters"].get("som_freeze_floor")
    sfl_default = _default_carbon_value("som_freeze_floor")
    sfl_str = f"{sfl:.4g}" if sfl is not None else "frozen@default"
    print(f"[result] loss {loss_history[0]:.6g} -> {loss_history[-1]:.6g} "
          f"(improved={improved}); som_freeze_floor "
          f"default={sfl_default:.4g} -> tuned={sfl_str}; "
          f"SOC RMSE {scorecard['cover_weighted_soc_rmse_kgC_m2']['default']:.4g} -> "
          f"{scorecard['cover_weighted_soc_rmse_kgC_m2']['tuned']:.4g} kgC/m2")
    if not improved:
        print("[WARN] training loss did NOT decrease -- inspect the gradient / lr; "
              "the tuned JSON reflects the warm start only.")
    return {"payload": payload, "scorecard": scorecard,
            "loss_history": loss_history, "improved": improved}


def _default_carbon_value(field: str) -> float:
    from legoesm.land.carbon.config import CarbonConfig
    return float(CarbonConfig()._asdict()[field])


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- input mode ---
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild the archetype table + membership from the real "
                        "cover/climate (default real path)")
    p.add_argument("--archetypes", type=str, default="",
                   help="prior archetypes.npz (geometry/provenance only; table + "
                        "membership are rebuilt from --surf-path for ORGANIC "
                        "consistency)")
    p.add_argument("--dry-run-synthetic", action="store_true",
                   help="tiny fabricated world + synthetic observed SOC (no files)")
    # --- real cover/soil inputs (reused via build_global_carbon_ic) ---
    p.add_argument("--surfdata-preset", type=str, default="clm5_surfdata",
                   choices=("clm5_surfdata", "legoesm_surfdata"))
    p.add_argument("--surf-path", "--surfdata", dest="surf_path", type=str,
                   default="", help="surfdata NetCDF (raw CLM5 for clm5_surfdata)")
    p.add_argument("--veg-path", type=str, default="")
    p.add_argument("--resolution-deg", type=float, default=1.0)
    # --- climate inputs ---
    p.add_argument("--climate-from-latitude", action="store_true",
                   help="zonal DEMONSTRATION climate (no climatology NetCDF)")
    p.add_argument("--climatology", type=str, default="",
                   help="monthly-mean T/precip/SW/netrad NetCDF (science-grade)")
    p.add_argument("--clim-t-var", type=str, default="tas")
    p.add_argument("--clim-precip-var", type=str, default="pr")
    p.add_argument("--clim-sw-var", type=str, default="rsds")
    p.add_argument("--clim-netrad-var", type=str, default="netrad")
    p.add_argument("--clim-lat", type=str, default="lat")
    p.add_argument("--clim-lon", type=str, default="lon")
    p.add_argument("--clim-t-in-celsius", action="store_true")
    # --- archetype build controls ---
    p.add_argument("--k-per-pft", type=int, default=12)
    p.add_argument("--w-min", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-archetypes", type=int, default=DEFAULT_MAX_ARCHETYPES,
                   help="cap on trained archetypes (representative subset; 0=all)")
    # --- observed-SOC target ---
    p.add_argument("--om-to-oc", type=float, default=None,
                   help="organic-matter -> organic-carbon factor for ORGANIC "
                        "(default: preset-aware -- 0.58 van Bemmelen for "
                        "clm5_surfdata organic MATTER, 1.0 for legoesm_surfdata "
                        "HWSD carbon)")
    # --- observed-SIF target (--with-sif) ---
    p.add_argument("--with-sif", dest="with_sif", action="store_true",
                   help="add the SIF (solar-induced fluorescence) observation stream: a "
                        "per-archetype simulated-SIF forward vs observed SIF, summed with "
                        "the SOC term (cover-weighted MSE), and train the SIFConfig "
                        "fluorescence params. Constrains photosynthesis/GPP (the "
                        "boreal-productivity lever). Works in fast OR slow mode (the SIF "
                        "forward is single-step, spin-up-free).")
    p.add_argument("--sif-obs", type=str, default="",
                   help="gridded SIF NetCDF on the surfdata grid [model photon-flux units] "
                        "(required for the REAL --with-sif path; --dry-run-synthetic uses "
                        "a fabricated target). Real source = TROPOMI/OCO-2/GOME-2 gridded "
                        "SIF (radiance->photon-flux + regrid is a data-prep follow-up).")
    p.add_argument("--sif-var", type=str, default="",
                   help="SIF variable name in --sif-obs (auto-detected if omitted)")
    p.add_argument("--sif-weight", type=float, default=DEFAULT_SIF_WEIGHT,
                   help="weight of the sif MSE term relative to soc (soc weight = 1; "
                        f"default {DEFAULT_SIF_WEIGHT}). SOC/SIF live on different scales.")
    # --- spin-up geometry (training forward pass) ---
    p.add_argument("--n-spinup", type=int, default=DEFAULT_N_SPINUP)
    p.add_argument("--n-verify", type=int, default=DEFAULT_N_VERIFY)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--soil-depth", type=float, default=DEFAULT_SOIL_DEPTH)
    p.add_argument("--grad-chunk", type=int, default=DEFAULT_GRAD_CHUNK,
                   help="--slow-spinup-grad ONLY: accumulate the EXACT "
                        "cover-weighted-MSE gradient over contiguous archetype "
                        "chunks of this many archetypes, bounding each coupled-"
                        "spin-up reverse-mode VJP to <=grad_chunk archetypes (the "
                        "CPU batched VJP segfaults at >=8). Each chunk's "
                        "UN-normalized weighted SSE grad sums, divided ONCE by the "
                        "global cover-weight sum -> EXACTLY the unchunked gradient. "
                        f"Default {DEFAULT_GRAD_CHUNK}; 0 = single batch (fast mode "
                        "/ tiny runs); ignored under --fast-analytic.")
    # --- optimizer ---
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--patience", type=int, default=25,
                   help="stop after this many CONSECUTIVE non-reducing steps "
                        "(guards the warmup ramp from a premature stop)")
    p.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--optimizer", choices=("muon", "muon_partitioned", "adam", "adamw"),
                   default="muon")
    p.add_argument("--grad-nonzero-tol", type=float, default=GRAD_NONZERO_TOL)
    p.add_argument("--all-carbon-params", action="store_true",
                   help="train the full tier-2 carbon set, not just the 7 "
                        "fast-analytic-valid SOM fields (requires "
                        "--slow-spinup-grad; see SOM_FIELDS)")
    # --- forward map: fast closed-form (default) vs slow grad-through-spin-up ---
    p.add_argument("--fast-analytic", dest="fast_analytic", action="store_true",
                   default=True,
                   help="DEFAULT: differentiate the closed-form SOM-cascade "
                        "equilibrium (one no-grad precompute spin-up, then many "
                        "cheap optimizer steps). The spin-up (n-spinup/n-verify/dt/"
                        "n-layers/soil-depth) is run ONCE to record the "
                        "SOM-parameter-independent inputs.")
    p.add_argument("--slow-spinup-grad", dest="fast_analytic", action="store_false",
                   help="use the B1 grad-through-coupled-spin-up forward "
                        "(equilibrate_archetypes_traced); correct but ~tens of "
                        "minutes per optimizer step.")
    # --- output ---
    p.add_argument("--output", "--outdir", dest="outdir", type=Path,
                   default=DEFAULT_OUTDIR)
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke config (unit test): 1 step, small spin-up")
    return p


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    # --fast-analytic (default True) is only gradient-valid for the seven
    # SOM_FIELDS: precompute_fast_analytic_inputs freezes every non-SOM (GPP /
    # phenology / allocation) input at the DEFAULT tier-2 params, so a tier-2
    # field --all-carbon-params would additionally expose has its effect baked
    # into that frozen precompute -- its fast-mode gradient would be stale or
    # exactly zero (dead DOF), not a real calibration signal.  Fail loudly
    # instead of silently training on a wrong/zero gradient; the full tier-2 set
    # remains available via the correct (but slow) grad-through-spin-up forward.
    if args.fast_analytic and args.all_carbon_params:
        raise SystemExit(
            "--fast-analytic --all-carbon-params is not supported: the fast "
            "closed-form forward has a valid gradient only for the SOM-pool-only "
            "fields in SOM_FIELDS. The non-SOM tier-2 params --all-carbon-params "
            "adds (GPP/phenology/allocation) have their effects baked into the "
            "frozen precompute, so their fast-mode gradient would be stale/zero. "
            "Use --slow-spinup-grad --all-carbon-params to train the full "
            "tier-2 carbon set with the correct (grad-through-spin-up) forward.")
    if args.quick:
        args.dry_run_synthetic = True
        args.steps = min(args.steps, QUICK_STEPS)
        args.n_spinup = min(args.n_spinup, QUICK_N_SPINUP)
        args.n_verify = min(args.n_verify, QUICK_N_VERIFY)
        args.n_layers = min(args.n_layers, QUICK_N_LAYERS)
        args.soil_depth = min(args.soil_depth, QUICK_SOIL_DEPTH)
        args.max_archetypes = min(args.max_archetypes, QUICK_MAX_ARCHETYPES) \
            if args.max_archetypes > 0 else QUICK_MAX_ARCHETYPES
        args.warmup_steps = min(args.warmup_steps, 0)
        # A --quick world is a handful of archetypes -- one batch fits; keep the
        # single-batch path (the chunked accumulation is exercised by its own
        # direct unit test, not the quick smoke).
        args.grad_chunk = 0
    if not (args.dry_run_synthetic or args.rebuild or args.archetypes):
        raise SystemExit(
            "choose an input mode: --dry-run-synthetic, --rebuild "
            "(+--surf-path +--climatology/--climate-from-latitude), or "
            "--archetypes <npz> +--surf-path.")
    if not args.dry_run_synthetic and not args.surf_path:
        raise SystemExit("--surf-path is required for the real (--rebuild/"
                         "--archetypes) path.")
    # --with-sif REAL path needs a gridded SIF product; the dry-run fabricates one.
    if args.with_sif and not args.dry_run_synthetic and not args.sif_obs:
        raise SystemExit(
            "--with-sif on the real (--rebuild/--archetypes) path requires --sif-obs "
            "<gridded SIF NetCDF on the surfdata grid>; --dry-run-synthetic uses a "
            "fabricated SIF target instead.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _finalize_args(build_arg_parser().parse_args(argv))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
