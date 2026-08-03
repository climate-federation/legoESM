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
``legoesm.core.param_overrides.apply_param_overrides`` **inside** the loss (so the leaves are
traced -- the SegmentForcing override doctrine), the optimizer is MUON
(``legoesm.ml.training.create_optimizer``, warmup+cosine+clip), and the run writes
a RECOMMENDED ``tuned_carbon_parameters.json`` under ``results/`` -- it NEVER
mutates the production ``CarbonConfig`` defaults (production keeps static
Python-float leaves).

Modular loss registry: ``LOSSES = {obs_name: LossTerm(target, extractor, weight)}``
-- ``soc`` (``som_total`` vs observed SOC, cover-weighted MSE via
``legoesm.ml.loss.area_weighted_mse``) plus optional single-step streams: ``sif``
(``--with-sif``; simulated-SIF vs observed SIF -- photosynthesis/GPP, the boreal-
productivity lever), ``biomass`` / ``lai`` (``--with-biomass`` / ``--with-lai``;
closed-form live-pool vs observed biomass/LAI -- DALEC allocation/residence), and ``d13c``
(``--with-d13c``; simulated leaf carbon-isotope discrimination vs observed leaf delta13C --
constrains Ci/Ca, training the C3 stomatal water-use-efficiency params AND the C4 bundle-sheath
leakiness phi; C4 archetypes use a FAITHFUL C4 Farquhar-Cerling discrimination and CONTRIBUTE
to the loss -- no longer masked).  All are summed by ``_total_loss`` WITHOUT touching
the loop; a new stream plugs in the same way (add a registry entry).

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
import hashlib
import json
import os
import sys
import time
import zipfile
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
from legoesm.ml.training import (
    TrainingConfig,
    configure_jax_compilation_cache,
    create_optimizer,
)
from legoesm.training.param_collector import build_trainable_params
from legoesm.training.trainable_params import TrainablePhysicsParams


# --- units / identifiers -------------------------------------------------------
_G_PER_KG = 1000.0                 # gC/m2 -> kgC/m2 (exact conversion)
CARBON_SCHEME_KEY = "land.carbon"  # param_collector scheme_key for CarbonConfig
SIF_SCHEME_KEY = "land.canopy.sif"  # param_collector scheme_key for SIFConfig (--with-sif)
STOMATA_SCHEME_KEY = "land.stomata"  # param_collector scheme_key for StomataConfig (--with-d13c)
D13C_SCHEME_KEY = "land.d13c"        # param_collector scheme_key for D13CConfig (--with-d13c; C4 phi)
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
# ``permafrost_protection_min`` / ``permafrost_frozen_fraction_threshold`` (the
# perennial-frost / anaerobic SOM protection knobs) are FULLY fast-valid: the
# closed form scales each pool's turnover by f_perma(phi, config) with phi (the
# annual frozen fraction) FROZEN/param-independent, so f_perma is EXACT in these
# two leaves (no litter-input approximation like Q10_het_exp's) -- jax.grad flows
# the full gradient (verified: test_carbon_cycle.TestPerennialFrostProtection.
# test_f_perma_differentiable_in_perma_params).  They are THE direct high-latitude
# SOC levers, so they belong in the default fast SOC-calibration set.
SOM_FIELDS: tuple[str, ...] = (
    "tor_som_active", "tor_som_slow", "tor_som_passive",
    "f_active_to_slow", "f_slow_to_passive", "som_freeze_floor",
    "cwd_humification_eff",
    "permafrost_protection_min", "permafrost_frozen_fraction_threshold",
)

# The LIVE-pool (foliage/root/wood) fields the biomass/LAI streams (--with-biomass/
# --with-lai) calibrate through the closed-form live-pool forward
# (legoesm.land.carbon.live_pool_forward): the DALEC allocation partition (f_fol/f_root),
# the live-pool residence rates (leaf_lifespan / tor_root / tor_wood) and the leaf-mass-per-
# area LCMA.  Their gradient flows analytically through the single-step live-pool
# equilibrium C_pool = a_pool*NPP/k_pool, so they are FAST-valid (like SOM_FIELDS through
# analytic_som_soc) and added to the fast trainable set only when a live-pool stream is on.
# The preflight freezes any that the requested stream(s) leave with a zero gradient (e.g.
# tor_wood under --with-lai alone: LAI depends only on C_fol).
#
# EXCLUDED (deliberately, like Q10_het_exp is from SOM_FIELDS):
#   * ``f_lab`` -- the labile fraction.  The closed-form foliage uses the DIRECT allocation
#     a_fol only (the labile buffer's NET annual contribution to the standing foliage is
#     second-order and largely drained to nighttime respiration -- verified by the fidelity
#     gate), so LAI is INSENSITIVE to f_lab, and f_lab's only biomass effect (diluting the
#     structural root/wood allocation) goes through a DIFFERENT pathway than the model's
#     labile-buffer respiration.  Its fast-mode gradient is therefore unreliable; train it
#     via the slow grad-through-spin-up path (--slow-spinup-grad --all-carbon-params) instead.
#
# APPROXIMATION (frozen-NPP partial gradient, documented): npp_pos_annual is recorded ONCE
# at the DEFAULT physiology and held fixed while these leaves vary.  NPP is UPSTREAM of
# allocation, so freezing it is EXACT for the direct allocation/residence/LCMA -> pool
# effect the gradient follows.  It omits a SECOND-ORDER feedback (the pools set LAI = C_fol/
# LCMA -> GPP and the maintenance respiration -> NPP), exactly the partial-gradient class
# that excludes Q10_het_exp from the fast SOM set and freezes the SIF forward's leaf state;
# the fidelity gate (_live_pool_match) confirms the closed form tracks the spin-up
# equilibrium at the defaults, so the calibration follows the correct leading-order
# gradient direction.  A fully-coupled NPP feedback needs the slow forward (out of scope --
# the fast live-pool forward is single-step by design).
LIVE_POOL_FIELDS: tuple[str, ...] = (
    "f_fol", "f_root", "leaf_lifespan", "tor_root", "tor_wood", "LCMA",
)

# The STOMATAL water-use-efficiency (WUE) fields the leaf-delta13C stream (--with-d13c)
# calibrates through the single-step leaf-isotope-discrimination forward
# (legoesm.land.carbon.d13c_forward): the Ball-Berry residual conductance g0 and slope g1_bb.
# Leaf delta13C constrains Ci/Ca (a NEW lever vs SIF/live-pool), and Ci depends on these
# conductance parameters, so the gradient reaches them through the coupled Farquhar solve
# (re-solved per step -- the forward CANNOT freeze Ci, unlike SIF).  Restricted to the WUE
# knobs on purpose: the rest of the tier-2 StomataConfig set is photosynthetic capacity /
# kinetics (Vc_max25 is applied PER-ARCHETYPE in the d13c forward, so a scalar override would
# be wrong; J_max25 / Arrhenius / Jarvis fields are not WUE).  A Medlyn forward would train
# g1_med instead -- a documented follow-up (the forward uses the Ball-Berry path here).
STOMATA_WUE_FIELDS: tuple[str, ...] = ("g0", "g1_bb")

# The C4 leaf-delta13C lever the --with-d13c stream calibrates through the FAITHFUL C4
# discrimination branch (legoesm.land.carbon.d13c_forward.leaf_d13c_c4_from_ci_ca): the
# bundle-sheath leakiness phi.  A NEW, INDEPENDENT lever vs the C3 stomatal WUE fields above
# (g0/g1_bb move ONLY the C3 archetypes' Ci/Ca; phi moves ONLY the C4 archetypes' fixed-Ci/Ca
# discrimination), so the two are trained together but decoupled by the is_c4 selector.  Only
# phi is tunable; the C4 Ci/Ca setpoint (D13CConfig.ci_ca_c4) is FIXED (spec-excluded) to
# avoid a non-identifiable (phi, Ci/Ca) degeneracy -- see legoesm.land.carbon.config.D13CConfig.
D13C_LEAKINESS_FIELDS: tuple[str, ...] = ("phi_c4_leakiness",)

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
# Persistent-compilation-cache floor [s]: only XLA compiles SLOWER than this are
# written to disk.  The per-archetype-group coupled-land-model compiles are the
# expensive graphs here but each is only ~O(10 s) -- well BELOW the correction
# campaign's 30 s rrtmgp-tuned floor, which would cache NOTHING for the carbon
# trainer.  1 s (JAX's own default) caches every non-trivial carbon graph while
# still skipping sub-second kernels that are not worth the cache I/O.
DEFAULT_CACHE_MIN_COMPILE_SECS = 1.0
# --- fast-analytic precompute RESULT cache ------------------------------------
# The one-time default-parameter precompute (precompute_fast_analytic_inputs)
# runs ONE coupled-land spin-up per (is_woody, is_evergreen, soil_class) group
# -- ~150 s at 12 archetypes, ~22 min at 40 -- and its output (the FastAnalyticInputs
# AND the reference `real_som` equilibrium) is DETERMINISTIC and independent of the
# TUNABLE/trained SOM parameters: the precompute always runs at the PRODUCTION
# DEFAULTS and the trained params are varied only later in the closed form, never
# re-spun.  We therefore cache the RESULT arrays to disk keyed on the archetype
# table + spin config and reload them on a re-run (seconds) instead of re-spinning.
#
# BUMP _PRECOMPUTE_CACHE_VERSION whenever the spin-up / coupled-land carbon step
# physics OR its default parameters change -- specifically
# packages/land/legoesm/land/carbon/spinup.py (run_semi_analytic_spinup /
# analytic_slow_pool_equilibrium), the coupled carbon step (make_archetype_step_fn
# / carbon_cycle.step_carbon), the stationary-year re-integration in
# precompute_fast_analytic_inputs itself, the archetype-batch construction
# (iter_archetype_batches), OR the DEFAULT CarbonConfig / multilayer-land parameter
# VALUES those consume (the recorded result is the DEFAULT-parameter equilibrium, so
# a changed default silently changes it).  The key is over INPUTS not code, so such
# a change with UNCHANGED table+spin inputs would otherwise serve a STALE precompute;
# the version bump forces a MISS (recompute).  Mirrors the XLA compilation cache's
# HLO-in-key safety (see configure_jax_compilation_cache).
_PRECOMPUTE_CACHE_VERSION = "v7"  # v7: + perennial-frost/anaerobic SOM protection (f_perma on the SOM modifier, keyed on the annual frozen fraction) -- coupled spin-up equilibrium changes for cold archetypes + new soil_frozen_fraction field in FastAnalyticInputs (v6 was the r_maint_* CUE recalibration)
DEFAULT_PRECOMPUTE_CACHE_DIR = (
    os.environ.get("CARBON_PRECOMPUTE_CACHE_DIR", "")
    or str(REPO_ROOT / ".cache" / "carbon_precompute"))
# Loss-term weights. SOC (kgC/m2, O(10)) and SIF (umol/m2/s, O(10)) live on different
# scales, so the summed cover-weighted MSE weights each term; SOC is the reference (1).
W_SOC = 1.0
DEFAULT_SIF_WEIGHT = 1.0           # --sif-weight; balances the sif MSE against soc
# biomass (kgC/m2, O(1-10)) and LAI (m2/m2, O(1)) live on their own scales, so each term
# is weighted against the reference SOC term (weight 1); tune per run if a term dominates.
DEFAULT_BIOMASS_WEIGHT = 1.0       # --biomass-weight; balances the biomass MSE against soc
DEFAULT_LAI_WEIGHT = 1.0           # --lai-weight; balances the lai MSE against soc
# leaf delta13C (permil, O(-25 to -30)) lives on its own (negative) scale; weight it against
# the reference SOC term (weight 1). A residual of a few permil vs a SOC residual of O(10)
# kgC/m2 -> the default weight keeps the delta13C term from being swamped; tune per run.
DEFAULT_D13C_WEIGHT = 1.0          # --d13c-weight; balances the d13c MSE against soc
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


def _biomass_extractor(preds) -> jax.Array:
    """Model per-archetype live biomass [kgC/m2] from the predictions bundle.

    The (single-step, spin-up-free) closed-form live-pool forward is evaluated in the loss
    and stores ``(biomass, lai)`` under ``preds["biomass"]`` / ``preds["lai"]`` (see
    :func:`legoesm.land.carbon.live_pool_forward.compute_biomass_lai`)."""
    return preds["biomass"]


def _lai_extractor(preds) -> jax.Array:
    """Model per-archetype simulated LAI [m2/m2] from the predictions bundle
    (``preds["lai"]``; the closed-form ``C_fol / LCMA``)."""
    return preds["lai"]


def _d13c_extractor(preds) -> jax.Array:
    """Model per-archetype simulated leaf delta13C [permil] from the predictions bundle.

    The (single-step, spin-up-free) leaf carbon-isotope-discrimination forward is evaluated
    in the loss and stored under ``preds["d13c"]`` (see
    :func:`legoesm.land.carbon.d13c_forward.simulate_archetype_d13c`)."""
    return preds["d13c"]


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


def _cover_weighted_rel_err(pred, ref, cover_weight=None) -> float:
    """Cover-weighted RMS error relative to the cover-weighted mean reference (numpy).

    ``sqrt(sum w (pred-ref)^2 / sum w) / |sum w ref / sum w|`` -- the loss-relevant fidelity
    metric shared by the analytic-vs-spin-up SOC and live-pool (biomass/LAI) match checks
    (:func:`_precompute_and_check`): how well a closed-form surrogate tracks the actual
    cover-weighted-MSE objective, normalised so near-zero cells do not inflate it.  A
    uniform weight (``cover_weight=None``) reduces to the plain RMS / mean."""
    p = np.asarray(pred, float)
    r = np.asarray(ref, float)
    w = np.ones_like(r) if cover_weight is None else np.asarray(cover_weight, float)
    wsum = float(np.sum(w)) or 1.0
    rmse = float(np.sqrt(np.sum(w * (p - r) ** 2) / wsum))
    mean_ref = float(np.sum(w * r) / wsum)
    return rmse / max(abs(mean_ref), 1e-6)


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
    with_biomass: bool = False, with_lai: bool = False, with_d13c: bool = False,
) -> TrainablePhysicsParams:
    """Tier-``extended`` ``land.carbon`` (+ optional ``land.canopy.sif`` / ``land.stomata``)
    trainables, warm-started from the production ``CarbonConfig`` / ``SIFConfig`` /
    ``StomataConfig`` defaults (``build_trainable_params`` seeds each raw leaf from the live
    NamedTuple default via the inverse sigmoid).

    ``som_only`` (default) restricts the CARBON set to the seven fast-analytic-valid SOM
    fields (:data:`SOM_FIELDS`) -- the Stage-B v1 calibration target -- so the optimizer
    is not handed the ~30 DALEC phenology/allocation leaves that barely move equilibrium
    SOC (nor ``Q10_het_exp``, whose fast-mode gradient is partial -- see the comment on
    :data:`SOM_FIELDS`).  Pass ``som_only=False`` to expose the full tier-2 carbon set
    (only valid with ``--slow-spinup-grad``; see :func:`_finalize_args`).

    ``with_biomass`` / ``with_lai`` additionally include the seven live-pool
    allocation/residence/LCMA fields (:data:`LIVE_POOL_FIELDS`) -- the biomass/LAI streams'
    calibration target -- which are FAST-valid through the closed-form live-pool forward
    (their gradient is analytic; NPP is frozen upstream of allocation).  Kept in the
    ``som_only`` set so the fast path can train them; the preflight freezes any left with a
    zero gradient by the requested stream(s).

    ``with_sif`` additionally includes the tier-1/2 ``SIFConfig`` fluorescence params
    (``kn0``/``kn_beta``/``kn_gamma``/``max_electron_yield``/``escape_probability``/
    ``kf``/``kd``/``kp``) -- ALWAYS kept in full (the SIF forward is a separate single-step
    diagnostic, not a SOM-pool field, so ``som_only`` never filters them).

    ``with_d13c`` additionally includes the ``StomataConfig`` water-use-efficiency subset
    (:data:`STOMATA_WUE_FIELDS`: ``g0`` / ``g1_bb``) -- and ONLY that subset (the rest of the
    tier-2 stomata set is photosynthetic capacity / kinetics, and ``Vc_max25`` is applied
    per-archetype in the delta13C forward, so a scalar override would be wrong).  All
    overrides are applied traced INSIDE the loss; production defaults are untouched.
    """
    scheme_keys = {CARBON_SCHEME_KEY}
    if with_sif:
        scheme_keys.add(SIF_SCHEME_KEY)
    if with_d13c:
        scheme_keys.add(STOMATA_SCHEME_KEY)   # C3 WUE (g0/g1_bb)
        scheme_keys.add(D13C_SCHEME_KEY)      # C4 leakiness (phi)
    params = build_trainable_params(
        active_scheme_keys=scheme_keys, tier="extended", dtype=jnp.float64)
    have = {c.name for c in params.constraints}
    sif_names = {n for n in have if n.startswith(SIF_SCHEME_KEY + ".")}
    if with_sif and not sif_names:
        raise ValueError(
            f"--with-sif requested but no {SIF_SCHEME_KEY} trainables at tier 'extended'; "
            f"is the spec module registered in param_collector.SPEC_MODULES?")
    # Stomata WATER-USE-EFFICIENCY subset only (the C3 d13c lever); never the full stomata set.
    stomata_wue = {f"{STOMATA_SCHEME_KEY}.{f}" for f in STOMATA_WUE_FIELDS}
    # C4 bundle-sheath leakiness phi (the C4 d13c lever); ci_ca_c4 is spec-excluded, so it is
    # never a trainable here.
    d13c_leak = {f"{D13C_SCHEME_KEY}.{f}" for f in D13C_LEAKINESS_FIELDS}
    if with_d13c:
        missing_st = stomata_wue - have
        if missing_st:
            raise ValueError(
                f"--with-d13c requested but stomata WUE fields {sorted(missing_st)} absent "
                f"from the {STOMATA_SCHEME_KEY} tier-extended trainables; known="
                f"{sorted(n for n in have if n.startswith(STOMATA_SCHEME_KEY + '.'))}. "
                f"Is the spec module registered in param_collector.SPEC_MODULES?")
        missing_d = d13c_leak - have
        if missing_d:
            raise ValueError(
                f"--with-d13c requested but C4 leakiness field(s) {sorted(missing_d)} absent "
                f"from the {D13C_SCHEME_KEY} tier-extended trainables; known="
                f"{sorted(n for n in have if n.startswith(D13C_SCHEME_KEY + '.'))}. "
                f"Is the spec module (legoesm.land.carbon.config:D13CConfig) registered in "
                f"param_collector.SPEC_MODULES?")
    keep_stomata = stomata_wue if with_d13c else set()
    keep_d13c = d13c_leak if with_d13c else set()
    if not som_only:
        # Full tier-2 carbon set (+ all SIF fluorescence params when --with-sif), but the
        # stomata part restricted to the WUE subset (never the per-archetype-incompatible
        # Vc_max25 / kinetics).  keep_non_stomata already carries the land.d13c phi.  With
        # --with-d13c off this returns the prior full set.
        keep_non_stomata = {n for n in have if not n.startswith(STOMATA_SCHEME_KEY + ".")}
        return _filter_params(params, keep_non_stomata | keep_stomata | keep_d13c)
    # SOM-only carbon subset, PLUS all SIF fluorescence params, PLUS the live-pool fields
    # when a biomass/LAI stream is on, PLUS the stomata WUE + C4 phi subsets when --with-d13c.
    carbon_fields = set(SOM_FIELDS)
    if with_biomass or with_lai:
        carbon_fields |= set(LIVE_POOL_FIELDS)
    keep = ({f"{CARBON_SCHEME_KEY}.{f}" for f in carbon_fields}
            | sif_names | keep_stomata | keep_d13c)
    missing = {f"{CARBON_SCHEME_KEY}.{f}" for f in carbon_fields} - have
    if missing:
        raise ValueError(
            f"carbon fields absent from the land.carbon tier-extended trainables: "
            f"{sorted(missing)}; known={sorted(have)}")
    return _filter_params(params, keep)


def _carbon_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{CarbonConfig field -> traced constrained scalar}`` for the loss."""
    return params.to_overrides().get(CARBON_SCHEME_KEY, {})


def _sif_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{SIFConfig field -> traced constrained scalar}`` for the sif loss term (empty
    when ``--with-sif`` is off / no SIF params are trained)."""
    return params.to_overrides().get(SIF_SCHEME_KEY, {})


def _stomata_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{StomataConfig field -> traced constrained scalar}`` for the d13c loss term's C3
    branch (empty when ``--with-d13c`` is off / no stomata WUE params are trained)."""
    return params.to_overrides().get(STOMATA_SCHEME_KEY, {})


def _d13c_config_overrides(params: TrainablePhysicsParams) -> dict[str, jax.Array]:
    """``{D13CConfig field -> traced constrained scalar}`` for the d13c loss term's C4 branch
    (the leakiness ``phi_c4_leakiness``; empty when ``--with-d13c`` is off)."""
    return params.to_overrides().get(D13C_SCHEME_KEY, {})


def _make_sif_forward(table):
    """Build ``sif_forward(sif_overrides) -> (n_arch,)`` simulated SIF [umol/m2/s] for the
    given (possibly chunk-sliced) archetype table.

    Precomputes the static per-archetype coupled-Farquhar leaf state ONCE
    (:func:`legoesm.land.carbon.sif_forward.build_sif_forward`), then each call splices the
    TRACED SIF overrides into a fresh ``SIFConfig`` (production defaults untouched) and
    applies the differentiable ``leaf_sif`` kernel -- a single-step graph, no spin-up."""
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import build_sif_forward
    from legoesm.core.param_overrides import apply_param_overrides

    sif_fn = build_sif_forward(table)

    def sif_forward(sif_overrides):
        cfg = apply_param_overrides(SIFConfig(), sif_overrides)
        return sif_fn(cfg)

    return sif_forward


def _make_d13c_forward(table):
    """Build ``d13c_forward(params) -> (n_arch,)`` simulated leaf delta13C [permil] for the
    given (possibly chunk-sliced) archetype table (BOTH C3 and C4 pathways).

    Precomputes the static per-archetype CLIMATE forcing + the static ``is_c4`` selector ONCE
    (:func:`legoesm.land.carbon.d13c_forward.build_d13c_forward`), then each call splices the
    TRACED overrides into fresh configs (production defaults untouched) and evaluates the
    per-archetype C3/C4 discrimination:

    * C3 archetypes -- the stomata WUE overrides (``g0`` / ``g1_bb``) go into a ``StomataConfig``
      and the forward RE-SOLVES the coupled Farquhar system (``Ci`` depends on the trained
      conductance, so it cannot be frozen like the SIF leaf state); ``Vc_max25`` is applied
      per-archetype inside the forward (NOT a trained scalar).
    * C4 archetypes -- the leakiness override (``phi_c4_leakiness``) goes into a ``D13CConfig``
      and drives the fixed-Ci/Ca Farquhar-Cerling C4 discrimination.

    Takes the full ``params`` (not a single scheme slice like the SIF forward) because the two
    pathways draw from two schemes (``land.stomata`` + ``land.d13c``).  Still a single-step
    graph, no spin-up; the g1_bb gradient flows through the C3 archetypes and the phi gradient
    through the C4 archetypes."""
    from legoesm.land.carbon.config import D13CConfig
    from legoesm.land.carbon.d13c_forward import build_d13c_forward
    from legoesm.land.stomata import StomataConfig
    from legoesm.core.param_overrides import apply_param_overrides

    d13c_fn = build_d13c_forward(table)

    def d13c_forward(params):
        st_cfg = apply_param_overrides(
            StomataConfig(enabled=True, stomata_model="ball_berry"),
            _stomata_overrides(params))
        d13c_cfg = apply_param_overrides(D13CConfig(), _d13c_config_overrides(params))
        return d13c_fn(st_cfg, d13c_cfg)

    return d13c_forward


def _make_live_pool_forward(precomputed):
    """Build ``live_pool_forward(carbon_overrides) -> (biomass, lai)`` [kgC/m2, m2/m2] from
    the precomputed live-pool inputs (``npp_pos_annual`` + ``is_woody``).

    Precomputes nothing new (the NPP + woody flag are recorded ONCE by the fast-analytic
    precompute, param-independent), then each call splices the TRACED CARBON overrides into
    a fresh ``CarbonConfig`` (production defaults untouched) and applies the differentiable
    closed-form live-pool equilibrium -- a single-step graph, no spin-up.  Uses the SAME
    ``land.carbon`` overrides as the SOC forward (biomass/LAI train the allocation/residence/
    LCMA leaves), mirroring :func:`_make_sif_forward` (which uses the disjoint SIF slice)."""
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import build_live_pool_forward
    from legoesm.core.param_overrides import apply_param_overrides

    live_fn = build_live_pool_forward(precomputed)

    def live_pool_forward(carbon_overrides):
        cfg = apply_param_overrides(CarbonConfig(scheme="differland"), carbon_overrides)
        return live_fn(cfg)   # (biomass, lai)

    return live_pool_forward


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
    d13c_forward=None,
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
    the ``sif`` registry term sums into the loss.  ``d13c_forward`` (when ``--with-d13c``) is
    the single-step leaf-delta13C forward on the TRACED stomata WUE overrides, added under
    ``preds["d13c"]``.  Both single-step forwards have NO spin-up scan (they do not touch
    ``equilibrate_archetypes_traced``) and their params do not enter the SOC forward, so the
    gradients are block-independent.
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
        if d13c_forward is not None:
            preds["d13c"] = d13c_forward(params)
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
    # Optional single-step SIF / leaf-delta13C forwards per chunk (--with-sif / --with-d13c);
    # both are per-archetype + spin-up-free, so slicing them to the chunk's archetypes is exact
    # and adds no coupled-model VJP (the crash was the spin-up batch, not the leaf-sif / small
    # Farquhar-resolve graphs).
    make_sif = target_bundle.get("make_sif_forward")
    make_d13c = target_bundle.get("make_d13c_forward")

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
        cd13c = make_d13c(ctable) if make_d13c is not None else None
        chunk_specs.append((ctable, cw, cterms, csif, cd13c))

    def _chunk_sse(ctable, cw, cterms, csif, cd13c):
        """UN-normalized cover-weighted SSE over ONLY this chunk's archetypes."""
        def sse(params: TrainablePhysicsParams) -> jax.Array:
            overrides = _carbon_overrides(params)
            eq = equilibrate_archetypes_traced(ctable, overrides, **spin)
            preds = {"eq": eq}
            if csif is not None:
                preds["sif"] = csif(_sif_overrides(params))
            if cd13c is not None:
                preds["d13c"] = cd13c(params)
            return _weighted_sse(preds, cterms, cw)
        return sse

    def value_and_grad_fn(params: TrainablePhysicsParams):
        total_sse = jnp.asarray(0.0, dtype=cover_weight.dtype)
        total_grad = None
        for (ctable, cw, cterms, csif, cd13c) in chunk_specs:
            sse, grad = eqx.filter_value_and_grad(
                _chunk_sse(ctable, cw, cterms, csif, cd13c))(params)
            total_sse = total_sse + sse
            total_grad = grad if total_grad is None else jax.tree_util.tree_map(
                lambda a, b: a + b, total_grad, grad)
        # Normalise ONCE by the global cover-weight sum: loss = sum_SSE / sum_w,
        # grad = sum_grad / sum_w -> exactly the unchunked cover-weighted-MSE grad.
        return total_sse / total_w, _scale_updates(total_grad, 1.0 / total_w)

    def loss_eval_fn(params: TrainablePhysicsParams) -> jax.Array:
        total_sse = jnp.asarray(0.0, dtype=cover_weight.dtype)
        for (ctable, cw, cterms, csif, cd13c) in chunk_specs:
            total_sse = total_sse + _chunk_sse(ctable, cw, cterms, csif, cd13c)(params)
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
    from legoesm.core.param_overrides import apply_param_overrides

    def model_soc(params: TrainablePhysicsParams) -> jax.Array:
        # Fresh CarbonConfig per call, overrides spliced in via _replace (traced,
        # immutable NamedTuple): production defaults are never mutated.
        cfg = apply_param_overrides(
            CarbonConfig(scheme="differland"), _carbon_overrides(params))
        return analytic_som_soc(precomputed, cfg) / _G_PER_KG   # gC/m2 -> kgC/m2

    return model_soc


def make_fast_loss_fn(
    precomputed, *, losses: dict[str, LossTerm], cover_weight: jax.Array,
    sif_forward=None, d13c_forward=None, live_pool_forward=None,
) -> Callable[[TrainablePhysicsParams], jax.Array]:
    """Fast-analytic calibration loss, routed through the SAME registry as the slow path.

    The SOC term is the closed-form per-archetype SOM SOC (``make_fast_analytic_forward``,
    stored under ``preds["soc"]`` for :func:`_fast_soc_extractor`); ``jax.grad`` flows to
    the seven fast-valid SOM leaves (:data:`SOM_FIELDS`) through the analytic cascade at
    ~zero cost -- no spin-up in the gradient loop.  The single-step ``sif`` term
    (``--with-sif``), ``d13c`` term (``--with-d13c``) and ``biomass`` / ``lai`` terms
    (``--with-biomass`` / ``--with-lai``) compose here too: each is spin-up-free, so
    ``sif_forward`` adds ``preds["sif"]``, ``d13c_forward`` adds ``preds["d13c"]`` (from the
    traced STOMATA overrides), and ``live_pool_forward`` adds ``preds["biomass"]`` +
    ``preds["lai"]`` (from the traced CARBON overrides), summed via ``_total_loss`` exactly as
    in the slow path.  With every optional forward ``None`` and only ``soc`` in the registry
    this equals the prior SOC-only fast loss.
    """
    model_soc = make_fast_analytic_forward(precomputed)

    def loss_fn(params: TrainablePhysicsParams) -> jax.Array:
        preds = {"soc": model_soc(params)}
        if sif_forward is not None:
            preds["sif"] = sif_forward(_sif_overrides(params))
        if d13c_forward is not None:
            preds["d13c"] = d13c_forward(params)
        if live_pool_forward is not None:
            biomass, lai = live_pool_forward(_carbon_overrides(params))
            preds["biomass"] = biomass
            preds["lai"] = lai
        return _total_loss(preds, losses, cover_weight)

    return loss_fn


def _build_loss_fn(args, target_bundle, spin, precomputed):
    """Return ``loss_fn(params) -> scalar`` for the active forward -- the fast
    closed-form SOC loss when ``--fast-analytic``, else the slow spin-up loss.  Both
    route through the registry ``_total_loss``; ``--with-sif`` adds the single-step SIF
    forward (built on the full table) in either mode, and ``--with-biomass`` / ``--with-lai``
    add the single-step live-pool forward (fast mode only -- it needs the NPP precompute)."""
    make_sif = target_bundle.get("make_sif_forward")
    sif_forward = make_sif(target_bundle["table"]) if make_sif is not None else None
    make_d13c = target_bundle.get("make_d13c_forward")
    d13c_forward = make_d13c(target_bundle["table"]) if make_d13c is not None else None
    live_pool_forward = target_bundle.get("live_pool_forward")
    if args.fast_analytic:
        return make_fast_loss_fn(
            precomputed, losses=target_bundle["losses"],
            cover_weight=target_bundle["cover_weight"], sif_forward=sif_forward,
            d13c_forward=d13c_forward, live_pool_forward=live_pool_forward)
    return make_loss_fn(
        table=target_bundle["table"], losses=target_bundle["losses"],
        cover_weight=target_bundle["cover_weight"], sif_forward=sif_forward,
        d13c_forward=d13c_forward, **spin)


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


def _build_observed_sif(args, table, inputs, cell_id, cell_w, n_arch_full):
    """Per-archetype observed SIF [umol/m2/s] (``--with-sif``), or ``None``.

    Synthetic in the dry-run; else a gridded satellite SIF product cover-weighted per
    archetype (native-grid ``clm5_surfdata`` only, like the SOC organic column)."""
    if not args.with_sif:
        return None
    from legoesm.land.carbon.sif_observations import (
        load_gridded_sif, per_archetype_observed_sif, synthetic_observed_sif,
    )
    if args.dry_run_synthetic:
        return synthetic_observed_sif(table)
    if args.surfdata_preset != "clm5_surfdata":
        raise SystemExit(
            "--with-sif observed SIF is only wired for the native-grid 'clm5_surfdata' "
            f"preset (the gridded SIF must match the cover grid); '{args.surfdata_preset}' "
            "regrids the cover. Use clm5_surfdata.")
    ncell = int(inputs.pft_weights.shape[0])
    sif_cell = load_gridded_sif(args.sif_obs, ncell=ncell, sif_var=(args.sif_var or None))
    return per_archetype_observed_sif(sif_cell, cell_id, cell_w, n_arch=n_arch_full)


def _build_observed_biomass(args, table, inputs, cell_id, cell_w, n_arch_full):
    """Per-archetype observed live biomass [kgC/m2] (``--with-biomass``), or ``None``.

    Synthetic in the dry-run; else a gridded biomass product cover-weighted per archetype
    (native-grid ``clm5_surfdata`` only).  The real ESA-CCI Biomass / GEDI AGB fetcher +
    AGB->carbon / above-ground->total conversion is a documented FOLLOW-UP (see
    :mod:`legoesm.land.carbon.biomass_observations`); ``--biomass-obs`` supplies a
    pre-regridded product in carbon units."""
    if not args.with_biomass:
        return None
    from legoesm.land.carbon.biomass_observations import (
        load_gridded_biomass, per_archetype_observed_biomass, synthetic_observed_biomass,
    )
    if args.dry_run_synthetic:
        return synthetic_observed_biomass(table)
    if args.surfdata_preset != "clm5_surfdata":
        raise SystemExit(
            "--with-biomass observed biomass is only wired for the native-grid "
            f"'clm5_surfdata' preset; '{args.surfdata_preset}' regrids the cover. "
            "Use clm5_surfdata.")
    ncell = int(inputs.pft_weights.shape[0])
    bio_cell = load_gridded_biomass(
        args.biomass_obs, ncell=ncell, biomass_var=(args.biomass_var or None))
    return per_archetype_observed_biomass(bio_cell, cell_id, cell_w, n_arch=n_arch_full)


def _build_observed_lai(args, table, inputs, cell_id, cell_w, n_arch_full):
    """Per-archetype observed LAI [m2/m2] (``--with-lai``), or ``None``.

    Synthetic in the dry-run; else the CLM5 surfdata ``MONTHLY_LAI`` (annual-mean, per-PFT)
    cover-weighted per archetype (native-grid ``clm5_surfdata`` only).  ``MONTHLY_LAI``
    lives in the surfdata itself, so ``--lai-obs`` defaults to ``--surf-path`` (no new
    download)."""
    if not args.with_lai:
        return None
    from legoesm.land.carbon.lai_observations import (
        load_surfdata_lai, per_archetype_observed_lai, synthetic_observed_lai,
    )
    if args.dry_run_synthetic:
        return synthetic_observed_lai(table)
    if args.surfdata_preset != "clm5_surfdata":
        raise SystemExit(
            "--with-lai observed LAI is only wired for the native-grid 'clm5_surfdata' "
            f"preset (MONTHLY_LAI on the cover grid); '{args.surfdata_preset}' regrids the "
            "cover. Use clm5_surfdata.")
    ncell = int(inputs.pft_weights.shape[0])
    n_pft = int(inputs.pft_weights.shape[1])
    lai_path = args.lai_obs or args.surf_path   # MONTHLY_LAI lives in the surfdata itself
    lai_cell_pft = load_surfdata_lai(
        lai_path, ncell=ncell, n_pft=n_pft, lai_var=(args.lai_var or None))
    return per_archetype_observed_lai(lai_cell_pft, cell_id, cell_w, n_arch=n_arch_full)


def _build_observed_d13c(args, table, inputs, cell_id, cell_w, n_arch_full):
    """Per-archetype observed leaf delta13C [permil] (``--with-d13c``), or ``None``.

    Synthetic in the dry-run; else a gridded leaf-delta13C product cover-weighted per archetype
    (native-grid ``clm5_surfdata`` only).  The real sparse leaf-delta13C fetcher (a Cornwell-
    style leaf-delta13C compilation / a global ecosystem-delta13C map, + the ecosystem->leaf
    reconciliation) is a documented FOLLOW-UP (see
    :mod:`legoesm.land.carbon.d13c_observations`); ``--d13c-obs`` supplies a pre-regridded
    LEAF-delta13C product.

    C4 archetypes are NO LONGER masked: the simulated forward now applies a FAITHFUL C4
    Farquhar-Cerling discrimination (:mod:`legoesm.land.carbon.d13c_forward`), so a C4
    archetype WITH an observation CONTRIBUTES to the d13c term (calibrating the C4 leakiness
    ``phi``).  The per-term finite-mask still drops genuinely-missing (NaN) observations only.
    """
    if not args.with_d13c:
        return None
    from legoesm.land.carbon.d13c_observations import (
        c4_archetype_mask,
        load_gridded_d13c,
        per_archetype_observed_d13c,
        synthetic_observed_d13c,
    )
    if args.dry_run_synthetic:
        observed = np.asarray(synthetic_observed_d13c(table), dtype=float)
    else:
        if args.surfdata_preset != "clm5_surfdata":
            raise SystemExit(
                "--with-d13c observed leaf delta13C is only wired for the native-grid "
                f"'clm5_surfdata' preset (the gridded delta13C must match the cover grid); "
                f"'{args.surfdata_preset}' regrids the cover. Use clm5_surfdata.")
        ncell = int(inputs.pft_weights.shape[0])
        d13c_cell = load_gridded_d13c(
            args.d13c_obs, ncell=ncell, d13c_var=(args.d13c_var or None))
        observed = np.asarray(
            per_archetype_observed_d13c(d13c_cell, cell_id, cell_w, n_arch=n_arch_full),
            dtype=float)
    # C4 archetypes are INCLUDED (faithful C4 forward); report how many now contribute (a
    # finite observed target with the C4 branch active) instead of being masked out.
    c4 = c4_archetype_mask(table.pft_id)
    n_c4_obs = int(np.sum(c4 & np.isfinite(observed)))
    if n_c4_obs:
        print(f"[target] {n_c4_obs} C4 archetype(s) CONTRIBUTE to the delta13C term via the "
              f"faithful C4 Farquhar-Cerling discrimination (calibrates C4 leakiness phi)")
    return observed


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

    # Observed SIF / biomass / LAI targets (--with-sif / --with-biomass / --with-lai): each
    # a per-archetype cover-weighted mean mirroring the SOC target.  Synthetic in the
    # dry-run; else a gridded product / surfdata field, which (like the SOC organic column)
    # is only wired for the native-grid clm5_surfdata cover.  Built on the FULL table
    # (subsample below).  ``units`` is only for the run-log note.
    observed_sif_full = _build_observed_sif(args, table, inputs, cell_id, cell_w, n_arch_full)
    observed_biomass_full = _build_observed_biomass(
        args, table, inputs, cell_id, cell_w, n_arch_full)
    observed_lai_full = _build_observed_lai(
        args, table, inputs, cell_id, cell_w, n_arch_full)
    observed_d13c_full = _build_observed_d13c(
        args, table, inputs, cell_id, cell_w, n_arch_full)

    keep = _stratified_subsample(table, args.max_archetypes, seed=args.seed)
    table = _slice_table(table, keep)
    observed = np.asarray(observed_full)[keep]
    cover = np.asarray(cover_full)[keep]

    def _sub(arr):
        return np.asarray(arr)[keep] if arr is not None else None

    # Optional single-step streams: (name, observed, extractor, weight, units, forward-tag).
    # ``forward`` = "sif" builds the SIF forward; "live_pool" builds the live-pool forward
    # (shared by biomass + lai); "d13c" builds the leaf-isotope-discrimination forward.
    extra_specs = [
        ("sif", _sub(observed_sif_full), _sif_extractor, args.sif_weight,
         "umol/m2/s", "sif"),
        ("biomass", _sub(observed_biomass_full), _biomass_extractor,
         args.biomass_weight, "kgC/m2", "live_pool"),
        ("lai", _sub(observed_lai_full), _lai_extractor, args.lai_weight,
         "m2/m2", "live_pool"),
        ("d13c", _sub(observed_d13c_full), _d13c_extractor, args.d13c_weight,
         "permil", "d13c"),
    ]
    present = [s for s in extra_specs if s[1] is not None]

    # Sanitise with PER-STREAM finite masks.  The cover weight is SHARED across registry
    # terms, so an archetype is ACTIVE if it has ANY finite observation (finite_any) with
    # positive cover; each term then MASKS the archetypes where ITS OWN observation is
    # missing.  So a SOC-observed-but-SIF-missing archetype keeps its SOC signal (and vice
    # versa) -- the streams need not overlap -- while the single global active-cover sum
    # stays the denominator (preserving the chunked-gradient exactness).  Targets are
    # sanitised NaN->0 where their own mask is 0, so 0*mask never poisons the sum.
    finite_soc = np.isfinite(observed)
    finite = {name: np.isfinite(obs) for (name, obs, *_rest) in present}
    finite_any = finite_soc
    for name in finite:
        finite_any = finite_any | finite[name]
    cover = np.where(finite_any & (cover > 0.0), cover, 0.0)
    observed = np.where(finite_soc, observed, 0.0)
    # Re-materialise each present stream's sanitised target (NaN->0 where masked out).
    present = [
        (name, np.where(finite[name], np.asarray(obs), 0.0), ext, w, units, fwd)
        for (name, obs, ext, w, units, fwd) in present
    ]
    if not np.any(cover > 0.0):
        streams = "/".join(["SOC"] + [n.upper() for (n, *_r) in present])
        raise SystemExit(
            f"no archetype has a finite observed {streams} with positive cover; cannot fit.")
    for (name, obs, *_rest) in present:
        if not np.any(finite[name] & (cover > 0.0)):
            # The stream was requested but NO archetype has a usable observation (an all-gap
            # product, or a grid that does not overlap the land cover). Fail loud rather than
            # silently train a dead term that reports a fake 0 RMSE.
            raise SystemExit(
                f"--with-{name} but no archetype has a finite observed {name.upper()} with "
                f"positive cover (the product may be all-gap over the land cover or on a "
                f"mismatched grid); provide a {name.upper()} product overlapping the "
                f"surfdata cover, or drop --with-{name}.")

    # Mode-aware SOC extractor: the FAST closed form supplies SOC directly (preds["soc"]);
    # the SLOW forward supplies the equilibrium CarbonState (preds["eq"]).
    soc_extractor = _fast_soc_extractor if args.fast_analytic else _soc_extractor
    # SOC mask only matters when another stream keeps an archetype active where SOC is
    # missing; SOC-only, the cover-zeroing already drops NaN-SOC archetypes (mask=None,
    # the unchanged single-stream path).
    soc_mask = (jnp.asarray(finite_soc, dtype=jnp.float64) if present else None)
    losses = {
        "soc": LossTerm(target=jnp.asarray(observed, dtype=jnp.float64),
                        extractor=soc_extractor, weight=W_SOC, mask=soc_mask),
    }
    # Each present single-step stream is added to the SAME registry (summed via _total_loss,
    # no loop change); its simulated forward is the spin-up-free _make_sif_forward /
    # _make_live_pool_forward.  ``d13c`` etc. plug in the same way.
    make_sif_forward = None
    make_d13c_forward = None
    needs_live_pool = False
    for (name, obs, ext, w, _units, fwd) in present:
        losses[name] = LossTerm(
            target=jnp.asarray(obs, dtype=jnp.float64), extractor=ext, weight=w,
            mask=jnp.asarray(finite[name], dtype=jnp.float64))
        if fwd == "sif":
            make_sif_forward = _make_sif_forward
        elif fwd == "d13c":
            make_d13c_forward = _make_d13c_forward
        elif fwd == "live_pool":
            needs_live_pool = True

    n_kept = int(table.pft_id.shape[0])
    n_active = int(np.sum(cover > 0.0))
    soc_active = finite_soc & (cover > 0.0)
    notes = []
    for (name, obs, _ext, w, units, _fwd) in present:
        act = finite[name] & (cover > 0.0)
        n_obs = int(np.sum(act))
        lo, hi = ((np.min(obs[act]), np.max(obs[act])) if n_obs else (float("nan"),) * 2)
        notes.append(f"{name.upper()} obs on {n_obs} [{lo:.2f}, {hi:.2f}] {units} "
                     f"(w={w:g})")
    extra_note = ("; " + "; ".join(notes)) if notes else ""
    soc_lo, soc_hi = ((np.min(observed[soc_active]), np.max(observed[soc_active]))
                      if np.any(soc_active) else (float("nan"),) * 2)
    print(f"[target] source={source} archetypes: {n_arch_full} built -> "
          f"{n_kept} kept ({n_active} active); "
          f"observed SOC [{soc_lo:.2f}, {soc_hi:.2f}] kgC/m2{extra_note}")
    by_name = {name: obs for (name, obs, *_r) in present}
    return {
        "table": table,
        "losses": losses,
        "cover_weight": jnp.asarray(cover, dtype=jnp.float64),
        "observed_soc": observed,
        "observed_sif": by_name.get("sif"),
        "observed_biomass": by_name.get("biomass"),
        "observed_lai": by_name.get("lai"),
        "observed_d13c": by_name.get("d13c"),
        "make_sif_forward": make_sif_forward,
        "make_d13c_forward": make_d13c_forward,
        "needs_live_pool": needs_live_pool,
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


def _precompute_cache_key(table, spin: dict) -> str:
    """Stable SHA-256 digest over the precompute's INPUTS (never its tunable SOM
    parameters -- the cached result is SOM-parameter-INDEPENDENT by construction).

    Key components, in a FIXED and documented order (any change of order changes
    the key, so it must never be reordered):

      1. ``_PRECOMPUTE_CACHE_VERSION`` -- the stale-physics guard (bump on any
         spin-up / coupled-carbon-step change; see its definition).
      2. the six NUMERIC ``ArchetypeTable`` fields in declaration order
         (``pft_id, mat_k, map_yr, t_seasonal_amp_k, aridity, sw_mean_w``), each
         as ``np.ascontiguousarray(...).tobytes()`` -- tagged with the field name,
         dtype, and shape so two distinct tables can never alias to one digest
         (a raw byte concatenation without tags could).
      3. the string ``soil_class`` field as ``"|".join(soil_class)`` (per-archetype
         texture keys; ``|`` is not a soil-class token).
      4. the spin config ``(n_spinup, n_verify, dt, n_layers, soil_depth)`` formatted
         deterministically (``repr`` on the floats keeps full precision stable).

    The digest is a pure function of these inputs, so it is identical across
    processes/hosts for the same table+spin (deterministic hashing).
    """
    h = hashlib.sha256()
    h.update(b"CARBON_PRECOMPUTE")
    h.update(_PRECOMPUTE_CACHE_VERSION.encode())
    for name in ("pft_id", "mat_k", "map_yr", "t_seasonal_amp_k", "aridity",
                 "sw_mean_w"):
        arr = np.ascontiguousarray(getattr(table, name))
        h.update(f"|{name}:{arr.dtype}:{arr.shape}|".encode())
        h.update(arr.tobytes())
    soil_class = "|".join(str(s) for s in np.asarray(table.soil_class).ravel())
    h.update(b"|soil_class|")
    h.update(soil_class.encode("utf-8"))
    spin_key = (f"|spin|{spin['n_spinup']}|{spin['n_verify']}|{spin['dt']!r}|"
                f"{spin['n_layers']}|{spin['soil_depth']!r}|")
    h.update(spin_key.encode("utf-8"))
    return h.hexdigest()


def _load_or_precompute(table, spin: dict, *, cache_dir: str, rebuild: bool):
    """Deterministic RESULT cache around ``precompute_fast_analytic_inputs``.

    The precompute is a PURE, SOM-parameter-INDEPENDENT function of the archetype
    table + spin config that runs one default-parameter coupled-land spin-up per
    ``(is_woody, is_evergreen, soil_class)`` group -- ~150 s at 12 archetypes,
    ~22 min at 40.  Its ``(FastAnalyticInputs, real_som)`` result is cached to
    ``<cache_dir>/<key>.npz`` (key from :func:`_precompute_cache_key`) so a re-run
    with the same inputs reloads in ~1 s instead of re-spinning.

    ``precompute_fast_analytic_inputs`` stays PURE (no disk I/O): the cache is a
    driver concern and lives here.  ``rebuild=True`` (``--rebuild-precompute``)
    forces a recompute + overwrite; an empty ``cache_dir`` disables the cache.

    Returns ``(FastAnalyticInputs, real_som)`` -- the SAME contract as the wrapped
    precompute, reconstructed EXACTLY (dtypes, ``dt_days`` as a Python float,
    ``real_som`` shape) on a cache hit.
    """
    # Deferred imports keep module import cheap and match the precompute's own
    # (function-scope) import discipline.
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs
    from legoesm.land.carbon.global_init import precompute_fast_analytic_inputs

    key = _precompute_cache_key(table, spin)
    key8 = key[:8]
    path = Path(cache_dir) / f"{key}.npz" if cache_dir else None

    if path is not None and path.exists() and not rebuild:
        t0 = time.time()
        try:
            with np.load(path) as z:
                precomputed = FastAnalyticInputs(
                    lit_to_som_annual=jnp.asarray(z["lit_to_som_annual"]),
                    a_wood_annual=jnp.asarray(z["a_wood_annual"]),
                    soil_T_traj=jnp.asarray(z["soil_T_traj"]),
                    soil_frozen_fraction=jnp.asarray(z["soil_frozen_fraction"]),
                    precip=jnp.asarray(z["precip"]),
                    dt_days=float(z["dt_days"]),
                    npp_pos_annual=jnp.asarray(z["npp_pos_annual"]),
                    is_woody=jnp.asarray(z["is_woody"]),
                    is_evergreen=jnp.asarray(z["is_evergreen"]),
                    live_ref_C_fol=jnp.asarray(z["live_ref_C_fol"]),
                    live_ref_C_root=jnp.asarray(z["live_ref_C_root"]),
                    live_ref_C_wood=jnp.asarray(z["live_ref_C_wood"]),
                )
                real_som = jnp.asarray(z["real_som"])
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            # A corrupt / truncated / partially-written npz (a writer killed under an
            # OLD non-atomic version, a disk fault, or a missing key from a schema
            # change) must NOT crash every future run: drop the bad file and fall
            # through to a clean recompute (which re-saves a good one atomically).
            print(f"[fast-analytic] precompute cache {key8} unreadable "
                  f"({type(exc).__name__}); recomputing", flush=True)
            try:
                path.unlink()
            except OSError:
                pass
        else:
            print(f"[fast-analytic] precompute CACHE HIT ({key8}) loaded in "
                  f"{time.time() - t0:.1f}s", flush=True)
            return precomputed, real_som

    # MISS (absent / unreadable / --rebuild-precompute) or cache disabled: recompute.
    t0 = time.time()
    precomputed, real_som = precompute_fast_analytic_inputs(
        table, n_spinup=spin["n_spinup"], n_verify=spin["n_verify"],
        dt=spin["dt"], n_layers=spin["n_layers"], soil_depth=spin["soil_depth"])
    elapsed = time.time() - t0
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save the FastAnalyticInputs arrays + the dt_days scalar + real_som + the
        # live-pool inputs/reference (npp_pos_annual / is_woody / live_ref_*).
        # dt_days is a Python float -> a 0-d float64 array round-trips back to
        # float() exactly.  Write to a unique temp file then os.replace onto the
        # final path: os.replace is ATOMIC within a directory, so a crashed or
        # concurrent writer can never leave a TRUNCATED npz that a later run would
        # np.load into a corrupt-cache crash (the reader only ever sees a complete
        # file or none).  The '.npz' suffix on the temp keeps np.savez from
        # appending a second one.
        tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
        try:
            np.savez(
                tmp,
                lit_to_som_annual=np.asarray(precomputed.lit_to_som_annual),
                a_wood_annual=np.asarray(precomputed.a_wood_annual),
                soil_T_traj=np.asarray(precomputed.soil_T_traj),
                soil_frozen_fraction=np.asarray(precomputed.soil_frozen_fraction),
                precip=np.asarray(precomputed.precip),
                dt_days=np.asarray(precomputed.dt_days, dtype=np.float64),
                real_som=np.asarray(real_som),
                npp_pos_annual=np.asarray(precomputed.npp_pos_annual),
                is_woody=np.asarray(precomputed.is_woody),
                is_evergreen=np.asarray(precomputed.is_evergreen),
                live_ref_C_fol=np.asarray(precomputed.live_ref_C_fol),
                live_ref_C_root=np.asarray(precomputed.live_ref_C_root),
                live_ref_C_wood=np.asarray(precomputed.live_ref_C_wood),
            )
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"[fast-analytic] precompute {elapsed:.1f}s (saved cache {key8})",
              flush=True)
    else:
        print(f"[fast-analytic] precompute {elapsed:.1f}s (cache disabled)",
              flush=True)
    return precomputed, real_som


def _live_pool_match(precomputed, initial_params, cover_weight) -> dict[str, Any]:
    """Closed-form live-pool (biomass/LAI) vs the spin-up's annual-mean equilibrium.

    Builds the closed-form live-pool forward at the warm-started production defaults and
    compares its biomass [kgC/m2] / LAI [m2/m2] against the precompute's recorded annual-mean
    reference pools (``live_ref_*``), cover-weighted (the SAME metric the SOC gate uses).
    ``ref_biomass`` uses the SAME pool set as the forward (``C_fol + C_root + C_wood``) and
    ``ref_lai = C_fol / LCMA`` at the default LCMA, so this is an apples-to-apples
    closed-form-vs-spin-up fidelity.  Returns the fidelity-gate metrics for match_info / the
    run log."""
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.live_pool_forward import build_live_pool_forward
    from legoesm.core.param_overrides import apply_param_overrides

    live_fwd = build_live_pool_forward(precomputed)
    cfg_def = apply_param_overrides(
        CarbonConfig(scheme="differland"), _carbon_overrides(initial_params))
    biomass_cf, lai_cf = (np.asarray(x) for x in live_fwd(cfg_def))
    ref_C_fol = np.asarray(precomputed.live_ref_C_fol)
    ref_C_root = np.asarray(precomputed.live_ref_C_root)
    ref_C_wood = np.asarray(precomputed.live_ref_C_wood)
    ref_biomass = (ref_C_fol + ref_C_root + ref_C_wood) / _G_PER_KG      # kgC/m2
    ref_lai = ref_C_fol / float(cfg_def.LCMA)                            # m2/m2
    w = np.asarray(cover_weight, float) if cover_weight is not None else None
    return {
        "biomass_cover_weighted_rel_err": _cover_weighted_rel_err(biomass_cf, ref_biomass, w),
        "lai_cover_weighted_rel_err": _cover_weighted_rel_err(lai_cf, ref_lai, w),
        "spinup_biomass_kgC_m2_range": [float(np.min(ref_biomass)),
                                        float(np.max(ref_biomass))],
        "spinup_lai_range": [float(np.min(ref_lai)), float(np.max(ref_lai))],
    }


def _precompute_and_check(table, initial_params, spin, cover_weight=None, *,
                          cache_dir: str = "", rebuild: bool = False):
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
    t0 = time.time()
    # Load the SOM-parameter-INDEPENDENT precompute from the deterministic result
    # cache (or run + save it on a miss / --rebuild-precompute); the cache prints
    # its own CACHE HIT / saved line.  `precompute_seconds` below therefore times
    # the load-or-compute (seconds on a hit, ~150 s on a cold miss).
    precomputed, real_som = _load_or_precompute(
        table, spin, cache_dir=cache_dir, rebuild=rebuild)
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
    # --- Live-pool (biomass/LAI) closed-form-vs-spin-up fidelity gate ---------------
    # The closed-form live pools at the warm-started defaults vs the spin-up's ANNUAL-MEAN
    # equilibrium live pools (precomputed.live_ref_*), cover-weighted (same metric as SOC).
    # ALWAYS reported (cheap, single-step) so the live-pool forward's fidelity is visible
    # even without --with-biomass/--with-lai; a large mismatch means the closed form or the
    # NPP/allocation wiring is wrong -- surfaced (loud WARN), never silently shipped.
    lp_match = _live_pool_match(precomputed, initial_params, cover_weight)
    match.update(lp_match)
    print(f"[fast-analytic] precompute {match['precompute_seconds']:.1f}s; "
          f"analytic-vs-spin-up SOC match over {match['n_archetypes']} archetypes: "
          f"raw-mean|rel|={match['mean_abs_rel_err']:.2%} (inflated by near-zero cells)  "
          f"COVER-WEIGHTED|rel|={match['cover_weighted_rel_err']:.2%}  "
          f"high-SOM(>5)|rel|={match['high_som_mean_rel_err']:.2%} (n={match['n_high_som']})  "
          f"(spin-up SOM {match['spinup_som_kgC_m2_range'][0]:.1f}..."
          f"{match['spinup_som_kgC_m2_range'][1]:.1f} kgC/m2)")
    print(f"[fast-analytic] live-pool closed-form vs spin-up (annual-mean): biomass "
          f"COVER-WEIGHTED|rel|={lp_match['biomass_cover_weighted_rel_err']:.2%} (spin-up "
          f"{lp_match['spinup_biomass_kgC_m2_range'][0]:.1f}.."
          f"{lp_match['spinup_biomass_kgC_m2_range'][1]:.1f} kgC/m2)  LAI "
          f"COVER-WEIGHTED|rel|={lp_match['lai_cover_weighted_rel_err']:.2%} (spin-up "
          f"{lp_match['spinup_lai_range'][0]:.2f}..{lp_match['spinup_lai_range'][1]:.2f} "
          f"m2/m2)")
    if max(lp_match["biomass_cover_weighted_rel_err"],
           lp_match["lai_cover_weighted_rel_err"]) > _MATCH_WARN_REL:
        print(f"[WARN] live-pool closed form diverges from the spin-up "
              f"(biomass|rel|={lp_match['biomass_cover_weighted_rel_err']:.1%}, "
              f"lai|rel|={lp_match['lai_cover_weighted_rel_err']:.1%} > {_MATCH_WARN_REL:.0%}); "
              f"the closed-form live-pool equilibrium may not faithfully track the model -- "
              f"inspect before trusting --with-biomass/--with-lai tuned parameters.")
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
    # Optional leaf-delta13C scorecard (--with-d13c): cover-weighted simulated-delta13C RMSE
    # vs observed, default vs tuned (single-step forward; surfaces whether the d13c term
    # improved).  Weighted by cover MASKED to the delta13C-observed archetypes (matches the
    # loss's d13c finite-mask, which now drops only genuinely-missing observations -- C4
    # archetypes with an observation ARE included via the faithful C4 forward), so only
    # unobserved cells are excluded from the RMSE.
    make_d13c = target_bundle.get("make_d13c_forward")
    observed_d13c = target_bundle.get("observed_d13c")
    if make_d13c is not None and observed_d13c is not None:
        d13c_fwd = make_d13c(target_bundle["table"])
        d13c_default = np.asarray(d13c_fwd(initial_params))
        d13c_tuned = np.asarray(d13c_fwd(tuned_params))
        d13c_mask = np.asarray(target_bundle["losses"]["d13c"].mask)
        w_d13c = cover * d13c_mask
        scorecard["cover_weighted_d13c_rmse_permil"] = {
            "default": _wrmse(d13c_default, observed_d13c, w_d13c),
            "tuned": _wrmse(d13c_tuned, observed_d13c, w_d13c),
            "improvement": (_wrmse(d13c_default, observed_d13c, w_d13c)
                            - _wrmse(d13c_tuned, observed_d13c, w_d13c)),
        }
    # Optional biomass/LAI scorecard (--with-biomass/--with-lai): cover-weighted RMSE of the
    # single-step live-pool forward vs observed, default vs tuned (surfaces whether the term
    # improved).  Both share ONE live-pool forward (biomass, lai) from the bundle.
    live_pool_forward = target_bundle.get("live_pool_forward")
    if live_pool_forward is not None:
        biomass_def, lai_def = (np.asarray(x) for x in
                                live_pool_forward(_carbon_overrides(initial_params)))
        biomass_tun, lai_tun = (np.asarray(x) for x in
                                live_pool_forward(_carbon_overrides(tuned_params)))
        for name, key, obs, pred_def, pred_tun in (
            ("biomass", "cover_weighted_biomass_rmse_kgC_m2",
             target_bundle.get("observed_biomass"), biomass_def, biomass_tun),
            ("lai", "cover_weighted_lai_rmse_m2_m2",
             target_bundle.get("observed_lai"), lai_def, lai_tun),
        ):
            if obs is None:
                continue
            obs = np.asarray(obs)
            # Weight by cover MASKED to this stream's observed archetypes (matches the loss
            # mask) so missing cells never enter the RMSE.
            w_term = cover * np.asarray(target_bundle["losses"][name].mask)
            scorecard[key] = {
                "default": _wrmse(pred_def, obs, w_term),
                "tuned": _wrmse(pred_tun, obs, w_term),
                "improvement": (_wrmse(pred_def, obs, w_term)
                                - _wrmse(pred_tun, obs, w_term)),
            }
    path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    return scorecard


def _write_tuned_json(path: Path, *, tuned_params, initial_params, loss_history,
                      meta) -> dict[str, Any]:
    """Human-readable RECOMMENDED tuned params (never mutates CarbonConfig/SIFConfig)."""
    from legoesm.training.param_collector import build_registry
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.stomata import StomataConfig

    registry = {m.qualified_name: m for m in build_registry()}
    # Per-scheme production defaults (carbon + SIF + stomata); a field is looked up in the
    # config its scheme_key names, so SIF fields (kn0, ...) resolve against SIFConfig and
    # stomata WUE fields (g0, g1_bb) against StomataConfig, not CarbonConfig (which would
    # KeyError).
    defaults_by_scheme = {
        CARBON_SCHEME_KEY: CarbonConfig()._asdict(),
        SIF_SCHEME_KEY: SIFConfig()._asdict(),
        STOMATA_SCHEME_KEY: StomataConfig()._asdict(),
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
                 + ("+d13c-stomata" if STOMATA_SCHEME_KEY in scheme_keys else "")
                 + " calibration -- a recommendation, NOT a mutation of the production "
                 "CarbonConfig/SIFConfig/StomataConfig defaults."),
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
    # Enable JAX's PERSISTENT compilation cache BEFORE the first compile (build_target
    # / the fast-analytic precompute below): the archetype spin-up compiles a separate
    # coupled-land-model XLA graph per (is_woody, is_evergreen, soil_class) group, so a
    # diverse real archetype set costs ~20 min of COLD compiles every launch. With a
    # shared on-disk cache those compiles are written once and reused -> seconds on a
    # re-run. Empty dir => no-op (default behavior). Must precede any JAX compilation.
    cache_dir = configure_jax_compilation_cache(
        args.compilation_cache_dir, args.cache_min_compile_secs)
    if cache_dir is not None:
        print(f"[carbon] JAX persistent compilation cache: {cache_dir} (caching compiles > "
              f"{args.cache_min_compile_secs:g}s) — per-archetype coupled-land-model compiles "
              "are written once and reused across launches.", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    spin = {
        "n_spinup": args.n_spinup, "n_verify": args.n_verify, "dt": args.dt,
        "n_layers": args.n_layers, "soil_depth": args.soil_depth,
    }

    t_assemble = time.time()
    target_bundle = build_target(args)
    print(f"[target] assembled in {time.time() - t_assemble:.1f}s")

    initial_params = build_carbon_trainables(
        som_only=not args.all_carbon_params, with_sif=args.with_sif,
        with_biomass=args.with_biomass, with_lai=args.with_lai,
        with_d13c=args.with_d13c)
    params = initial_params
    n_sif = sum(c.name.startswith(SIF_SCHEME_KEY + ".") for c in params.constraints)
    # d13c stream trains TWO schemes: C3 stomata WUE (land.stomata) + C4 leakiness (land.d13c).
    n_d13c_stomata = sum(c.name.startswith(STOMATA_SCHEME_KEY + ".") for c in params.constraints)
    n_d13c_phi = sum(c.name.startswith(D13C_SCHEME_KEY + ".") for c in params.constraints)
    stream_note = "".join(
        f", {s}" for s, on in (("sif", args.with_sif), ("biomass", args.with_biomass),
                               ("lai", args.with_lai), ("d13c", args.with_d13c)) if on)
    print(f"[preflight] {len(params.constraints)} trainable params "
          f"(SOM-only={not args.all_carbon_params}, streams=soc{stream_note}"
          f"{f', {n_sif} SIF' if args.with_sif else ''}"
          f"{f', {n_d13c_stomata} stomata + {n_d13c_phi} C4-phi' if args.with_d13c else ''}); mode="
          f"{'fast-analytic' if args.fast_analytic else 'slow-spinup-grad'}; "
          f"n_spinup={args.n_spinup}, max_archetypes={args.max_archetypes}")

    # Fast-analytic: run the spin-up ONCE (no grad) to record the SOM-parameter-
    # INDEPENDENT inputs, then differentiate the closed-form cascade equilibrium.
    precomputed = None
    match_info: dict[str, Any] = {}
    if args.fast_analytic:
        precomputed, match_info = _precompute_and_check(
            target_bundle["table"], initial_params, spin,
            cover_weight=target_bundle["cover_weight"],
            cache_dir=args.precompute_cache_dir,
            rebuild=args.rebuild_precompute)
    # Build the single-step live-pool forward ONCE (from the precomputed NPP + woody flag)
    # and attach it to the bundle -- shared by the fast loss (biomass/lai preds) and the
    # scorecard.  Requires the precompute (fast mode); --finalize gates biomass/lai to fast.
    if target_bundle.get("needs_live_pool") and precomputed is not None:
        target_bundle["live_pool_forward"] = _make_live_pool_forward(precomputed)

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
            "with_biomass": bool(args.with_biomass),
            "biomass_weight": (args.biomass_weight if args.with_biomass else None),
            "with_lai": bool(args.with_lai),
            "lai_weight": (args.lai_weight if args.with_lai else None),
            "with_d13c": bool(args.with_d13c),
            "d13c_weight": (args.d13c_weight if args.with_d13c else None),
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
    # --- observed-biomass target (--with-biomass); FAST mode only (needs the NPP precompute) ---
    p.add_argument("--with-biomass", dest="with_biomass", action="store_true",
                   help="add the live-biomass observation stream: a per-archetype "
                        "closed-form live-pool biomass forward [kgC/m2] vs observed biomass, "
                        "summed with the SOC term (cover-weighted MSE), training the DALEC "
                        "allocation/residence/LCMA params. Constrains the standing-biomass "
                        "carbon. FAST-analytic mode only (the live-pool forward needs the "
                        "one-time NPP precompute). Real ESA-CCI/GEDI AGB is a fetcher "
                        "follow-up; --dry-run-synthetic fabricates a target.")
    p.add_argument("--biomass-obs", type=str, default="",
                   help="gridded biomass NetCDF on the surfdata grid [kgC/m2] (required for "
                        "the REAL --with-biomass path; --dry-run-synthetic fabricates one). "
                        "Real source = ESA-CCI Biomass / GEDI L4B AGB (AGB->carbon + "
                        "above-ground->total + regrid is a data-prep follow-up).")
    p.add_argument("--biomass-var", type=str, default="",
                   help="biomass variable name in --biomass-obs (auto-detected if omitted)")
    p.add_argument("--biomass-weight", type=float, default=DEFAULT_BIOMASS_WEIGHT,
                   help="weight of the biomass MSE term relative to soc (soc weight = 1; "
                        f"default {DEFAULT_BIOMASS_WEIGHT}).")
    # --- observed-LAI target (--with-lai); FAST mode only; MONTHLY_LAI is in the surfdata ---
    p.add_argument("--with-lai", dest="with_lai", action="store_true",
                   help="add the LAI observation stream: a per-archetype closed-form LAI "
                        "forward (C_fol/LCMA) vs observed LAI, summed with the SOC term "
                        "(cover-weighted MSE), training the DALEC allocation/leaf-residence/"
                        "LCMA params. Observed = the CLM5 surfdata MONTHLY_LAI (annual-mean, "
                        "per-PFT) already in --surf-path (NO new download). FAST-analytic "
                        "mode only (the live-pool forward needs the NPP precompute).")
    p.add_argument("--lai-obs", type=str, default="",
                   help="surfdata/NetCDF carrying MONTHLY_LAI on the cover grid (defaults to "
                        "--surf-path, where MONTHLY_LAI lives); --dry-run-synthetic "
                        "fabricates a target.")
    p.add_argument("--lai-var", type=str, default="",
                   help="LAI variable name in --lai-obs (default MONTHLY_LAI)")
    p.add_argument("--lai-weight", type=float, default=DEFAULT_LAI_WEIGHT,
                   help="weight of the lai MSE term relative to soc (soc weight = 1; "
                        f"default {DEFAULT_LAI_WEIGHT}).")
    # --- observed leaf-delta13C target (--with-d13c) ---
    p.add_argument("--with-d13c", dest="with_d13c", action="store_true",
                   help="add the leaf carbon-isotope discrimination (delta13C) observation "
                        "stream: a per-archetype simulated leaf-delta13C forward [permil] vs "
                        "observed leaf delta13C, summed with the SOC term (cover-weighted MSE), "
                        "and train the StomataConfig water-use-efficiency params (Ball-Berry "
                        "g0/g1_bb). Constrains Ci/Ca (a NEW lever vs SIF/biomass). Works in "
                        "fast OR slow mode (the forward is single-step, spin-up-free). C4 "
                        "archetypes are MASKED (model Farquhar is C3-only; C4 is a follow-up).")
    p.add_argument("--d13c-obs", type=str, default="",
                   help="gridded leaf-delta13C NetCDF on the surfdata grid [permil] (required "
                        "for the REAL --with-d13c path; --dry-run-synthetic uses a fabricated "
                        "target). Real source = a sparse leaf-delta13C compilation (e.g. "
                        "Cornwell 2018) / ecosystem-delta13C map (regrid + ecosystem->leaf "
                        "reconciliation is a data-prep follow-up).")
    p.add_argument("--d13c-var", type=str, default="",
                   help="delta13C variable name in --d13c-obs (auto-detected if omitted)")
    p.add_argument("--d13c-weight", type=float, default=DEFAULT_D13C_WEIGHT,
                   help="weight of the d13c MSE term relative to soc (soc weight = 1; "
                        f"default {DEFAULT_D13C_WEIGHT}). SOC/delta13C live on different scales.")
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
    # --- compute / JAX ---
    p.add_argument("--compilation-cache-dir",
                   default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
                   help="dir for JAX's PERSISTENT on-disk compilation cache so the "
                        "~20-min cold compile of the per-archetype coupled-land-model "
                        "graphs is written ONCE and reused across launches (a re-run "
                        "with the same code/backend drops to seconds). MUST be a SHARED-"
                        "filesystem path when a compute node writes it. Defaults to "
                        "$JAX_COMPILATION_CACHE_DIR; empty = disabled (unchanged behavior).")
    p.add_argument("--cache-min-compile-secs", type=float,
                   default=DEFAULT_CACHE_MIN_COMPILE_SECS,
                   help="(--compilation-cache-dir) cache only XLA compiles slower than "
                        f"this [s] (default {DEFAULT_CACHE_MIN_COMPILE_SECS:g}). The carbon "
                        "coupled-land-model per-group graphs are ~O(10 s) each, well below "
                        "the correction campaign's 30 s rrtmgp floor, so keep this small or "
                        "the cache stays empty.")
    p.add_argument("--precompute-cache-dir",
                   default=DEFAULT_PRECOMPUTE_CACHE_DIR,
                   help="dir for the deterministic RESULT cache of the fast-analytic "
                        "precompute (the SOM-parameter-INDEPENDENT default-parameter "
                        "spin-up, ~150 s at 12 archetypes / ~22 min at 40). Its result "
                        "arrays are cached per (archetype-table + spin-config) key so a "
                        "re-run reloads in ~1 s instead of re-spinning. Defaults to "
                        "$CARBON_PRECOMPUTE_CACHE_DIR or <repo>/.cache/carbon_precompute "
                        "(gitignored); empty = disabled (always recompute).")
    p.add_argument("--rebuild-precompute", action="store_true",
                   help="force a recompute + overwrite of the precompute RESULT cache "
                        "(--precompute-cache-dir). SEPARATE from --rebuild: --rebuild "
                        "selects the archetype-table INPUT mode and is REQUIRED on the "
                        "real path (so overloading it would disable the cache on every "
                        "real run); --rebuild-precompute bypasses ONLY the result cache. "
                        "Also bump _PRECOMPUTE_CACHE_VERSION when the spin-up physics "
                        "changes (the key is over inputs, not code).")
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
        # Keep the smoke HERMETIC: exercise the REAL (fast) precompute and never
        # read/write the shared repo result cache (its hit/miss logic has its own
        # direct unit test).  A fabricated world's precompute is sub-second anyway.
        args.precompute_cache_dir = ""
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
    # --with-d13c REAL path needs a gridded leaf-delta13C product; the dry-run fabricates one.
    if args.with_d13c and not args.dry_run_synthetic and not args.d13c_obs:
        raise SystemExit(
            "--with-d13c on the real (--rebuild/--archetypes) path requires --d13c-obs "
            "<gridded leaf-delta13C NetCDF [permil] on the surfdata grid>; "
            "--dry-run-synthetic uses a fabricated delta13C target instead.")
    # --with-biomass/--with-lai need the FAST closed-form live-pool forward, which is built
    # from the one-time NPP precompute recorded ONLY in --fast-analytic mode.  Fail loudly
    # rather than register a biomass/lai loss term whose prediction the slow path never
    # populates (a KeyError in the loss).  The allocation/residence/LCMA leaves are
    # fast-valid through the closed-form live-pool equilibrium (NPP frozen upstream), so
    # fast mode is the correct home; the slow grad-through-spin-up path trains them via
    # --slow-spinup-grad --all-carbon-params (SOC only, no biomass/lai term).
    if (args.with_biomass or args.with_lai) and not args.fast_analytic:
        raise SystemExit(
            "--with-biomass / --with-lai require --fast-analytic (default): the closed-form "
            "live-pool biomass/LAI forward is built from the one-time NPP precompute that "
            "only the fast path records. Drop --slow-spinup-grad, or drop "
            "--with-biomass/--with-lai.")
    # --with-biomass REAL path needs a gridded biomass product; the dry-run fabricates one.
    # (--with-lai reads MONTHLY_LAI from --surf-path, already required on the real path.)
    if args.with_biomass and not args.dry_run_synthetic and not args.biomass_obs:
        raise SystemExit(
            "--with-biomass on the real (--rebuild/--archetypes) path requires --biomass-obs "
            "<gridded biomass NetCDF [kgC/m2] on the surfdata grid>; --dry-run-synthetic "
            "uses a fabricated biomass target instead.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _finalize_args(build_arg_parser().parse_args(argv))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
