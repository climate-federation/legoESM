"""Interface between legoESM and the CLM-ML-JAX multilayer canopy model.

This module provides :func:`compute_clm_ml_canopy_fluxes`, which translates
legoESM's ``AtmToSurface`` forcing into the ``mlcanopy_type`` input
container, calls ``MLCanopyFluxes``, and maps the output back to a
``SurfaceFluxOutput``.

All imports of ``clm_ml_jax`` / ``multilayer_canopy`` are **lazy** (inside
function bodies) so that the rest of legoESM continues to import cleanly
without the optional ``canopy`` extra installed.

Variable-unit conventions
-------------------------
- legoESM ``psi_soil``: matric potential [m], negative for unsaturated
- CLM ``smp_l``:         matric potential [mm], negative for unsaturated
- legoESM ``K_unsat``:  hydraulic conductivity [m/s]
- CLM ``hk_l``:          hydraulic conductivity [mm/s]
- CLM ``swskyb``, ``swskyd``: direct/diffuse SW per waveband [W/m²]
- CLM patch/column/gridcell indices: 1-based (index 0 unused)
"""

from __future__ import annotations

import math
import numbers
import sys
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.canopy.config import (
    VALID_CLM_ML_TURBULENCE_SCHEMES,
    CLMMLCanopyConfig,
)
from legoesm.land.canopy.sif import SIFConfig, multilayer_canopy_sif
from legoesm.land.canopy.state import CanopyState
from legoesm.land.surface_scheme import SurfaceFluxOutput
from legoesm.thermo import saturation_specific_humidity

# CLM-ML unfilled array elements carry spval = 1e36; treat anything above this
# guard (or non-finite) as invalid padding and drop it from the SIF sum.
_SPVAL_GUARD = 1.0e30

if TYPE_CHECKING:
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig

# ---------------------------------------------------------------------------
# Physics contract
# ---------------------------------------------------------------------------

__physics_contract__ = {
    "units": {
        "shflx": "W/m² (positive upward, sensible heat to atmosphere)",
        "lhflx": "W/m² (positive upward, latent heat to atmosphere)",
        "G_soil": "W/m² (positive downward into soil)",
        "lw_up": "W/m² (positive upward, outgoing longwave)",
        "T_canopy_air": "K (PAI-weighted mean within-canopy air temperature)",
        "gpp": "gC/m²/s (gross primary production, zero in darkness)",
    },
    "signs": {
        "shflx": "positive = atmosphere gains heat",
        "lhflx": "positive = atmosphere gains moisture-equivalent energy",
        "G_soil": "positive = soil gains heat",
        "lw_up": "positive = upward emission from surface",
    },
    "conserves": "energy (Rnet = shflx + lhflx + G_soil + stflx_air + stflx_veg per timestep)",
    # Differentiable via the JAX-native diff path, GATED on
    # ``CLMMLCanopyConfig.differentiable=True`` (single column): the interface
    # passes a GridInfo (``grid=``) so ``MLCanopyFluxes`` runs ``lax.scan`` +
    # ``jax.checkpoint`` and the forcing→flux map (incl. trainable Vcmax25/g1) is
    # on the ``jax.grad`` tape.  Production default (``differentiable=False``)
    # stays forward-only (host-syncing checks, no tape).  Verified: forward/diff
    # flux parity ~1e-14 and FD grad rel_err <0.01% (TestCLMMLDifferentiability).
    "differentiable": True,
    "reference": "Bonan et al. (2021), GMD, CLM-ML v2",
    "idealized_test": "tests/land/unit/test_canopy.py::TestCLMMLInterface",
}

# ---------------------------------------------------------------------------
# Module-level CLM initialization guard and topology cache
# ---------------------------------------------------------------------------

# Minimum canopy-top geometry height [m].  A prescribed/climatology ``htop`` can
# be 0 on a bare or surfdata-uncovered column; CLM-ML then derives
# ``hbot = hbot_frac * htop = 0`` and the ``hbot < htop`` layering invariant
# collapses to a zero-thickness canopy.  Floor ``htop`` to this small positive
# value so the geometry stays valid — LAI is 0 on those columns, so the canopy
# contributes no fluxes regardless of the nominal height.  Matches the two-leaf
# path's ``HC_MIN_M`` (0.1 m) in ``boundary_data/_internals``.
_HTOP_GEOM_MIN_M: float = 0.1

_CLM_INITIALIZED: bool = False

# Topology cache: avoid re-running _setup_clm_topology when the grid has not
# changed between timesteps.  Key = (ncol, lat[0], lon[0]) — sufficient to
# detect a new grid allocation; full lat/lon arrays are not hashed here for
# performance.  Reset when any element changes.
_last_topology_key: tuple | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_clm_initialized() -> None:
    """Call CLM phase-1 initialization exactly once."""
    global _CLM_INITIALIZED
    if _CLM_INITIALIZED:
        return
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import clm_varpar_init
    from legoesm.land.canopy.clm_ml_backend.offline_driver import clmSoilOptionMod

    # Use CLM4.5 physics so nlevsoi=10 matches legoESM's default 10-layer soil.
    clmSoilOptionMod.clm_phys = "CLM4_5"
    clm_varpar_init()

    # Initialize MLpftcon and psihat look-up tables.
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI
    from legoesm.land.canopy.clm_ml_backend.clm_src_main import pftconMod
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLpftconMod

    pftconMod.pftcon = pftconMod.Init()
    MLpftconMod.MLpftcon = MLpftconMod.Init()
    LookupPsihatINI()

    # Initialize orbital parameters for year 2001 (non-leap, 365 days).
    # Must match the start_date_ymd=20010101 epoch used in _setup_clm_time so
    # that CLM's shr_orb_cosz gets consistent orbital geometry (year 2000 is
    # a leap year and shifts caldays after Feb 28 by one day).
    from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import shr_orb_params
    import legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb as _varorb

    # shr_orb_params returns (eccen, obliq_deg, mvelp_deg, obliqr, lambm0, mvelpp)
    eccen, _obliq, _mvelp, obliqr, lambm0, mvelpp = shr_orb_params(2001)  # coeff-ok: non-leap reference year 2001 for CLM orbital parameters epoch
    _varorb.eccen = float(eccen)
    _varorb.obliqr = float(obliqr)
    _varorb.mvelpp = float(mvelpp)
    _varorb.lambm0 = float(lambm0)

    _CLM_INITIALIZED = True


# ---------------------------------------------------------------------------
# Canopy-airspace turbulence scheme
# ---------------------------------------------------------------------------
# CLM-ML's Harman & Finnigan roughness-sublayer (RSL) formulation is, term by
# term, Monin-Obukhov similarity PLUS a roughness-sublayer correction ψ̂
# (Bonan et al. 2018 appendix A2, eqs. A16/A19):
#
#     psim = -psim1 + psim2 + c1*psihat_m(za) - c1*psihat_m(hc) + vkc/beta
#     psic = -psic1 + psic2 + c1*psihat_h(za) - c1*psihat_h(hc)
#
# so ψ̂ ≡ 0 removes the RSL term exactly, leaving MOST ψ.  ψ̂ is evaluated by
# bilinear interpolation of a lookup table, and every consumer in
# ``MLCanopyTurbulenceMod`` reads one of the four module globals below: the JAX
# ``_LookupPsihat{M,H}`` called by ``_GetPsiRSL``, and the pure-Python
# ``_LookupPsihat{M,H}_scalar`` called by the Obukhov root solvers.  Swapping
# all four switches the momentum AND scalar corrections across the eager,
# scalar and differentiable paths at once — including the ``psim_hat2`` return
# that normalises the WITHIN-canopy wind profile, which a patch of
# ``_GetPsiRSL`` alone would leave inconsistent.
#
# SCOPE — what "most" does NOT do (do not overclaim in comparisons): it removes
# ψ̂ only.  β = u*/u(h), the displacement height, and the u(hc) = u*/β canopy-top
# anchor still come from Harman & Finnigan canopy-drag theory, and the
# within-canopy mixing-length closure is untouched (CLM-ML has no alternative).
# So "most" is "CLM-ML with the roughness-sublayer correction disabled", NOT a
# reproduction of the two-leaf/big-leaf roughness-length MOST surface layer;
# residual differences against the big-leaf scheme are NOT attributable to
# canopy physiology alone.
#
# We deliberately do NOT touch ``MLclm_varctl.turb_type``: upstream implements
# only ``turb_type == 1`` and every other value calls ``endrun``.
#
# clm-ml-jax exposes no public accessor for these tables, so they are addressed
# by name, the pristine snapshot is checked for non-vacuity, and the applied
# scheme is verified through the REAL lookup functions — a silent no-op here
# would mean ``turbulence_scheme="most"`` quietly ran RSL physics.
_PSIHAT_TABLE_ATTRS: tuple[str, ...] = (
    "psigridM", "psigridH", "_psigridM_jax", "_psigridH_jax",
)

# Probe coordinates used to verify a scheme actually took effect, in the
# lookup table's own coordinates: normalised height (z-hc)/(hc-d) and stability
# (hc-d)/L.  Any interior point works; this one is well inside the tabulated
# unstable range, where ψ̂ is comfortably non-zero.
_PSIHAT_PROBE_ZDT: float = 0.5   # coeff-ok: table-interior probe coordinate
_PSIHAT_PROBE_DTL: float = -0.2  # coeff-ok: table-interior probe coordinate

# Pristine RSL tables, snapshotted once after LookupPsihatINI.  PRIVATE: CLM is
# always handed a fresh copy, never this object, because upstream
# ``LookupPsihatINI`` assigns into ``psigrid*`` IN PLACE and would otherwise
# overwrite the snapshot (or a shared zero table) behind our back.
_PSIHAT_RSL: dict[str, Any] | None = None
# Scheme in force the first time a DIFFERENTIABLE step was built.  Diff mode
# bakes the psihat table into the jaxpr as a trace-time constant, so a compiled
# or differentiated function cannot follow a later switch.
_DIFF_TURBULENCE_SCHEME: str | None = None
# Backend gs_type in force from the last ``_apply_stomatal_model``; used to clear
# the leaf-kernel lru_caches only on an actual stomatal-model switch.
_APPLIED_GS_TYPE: int | None = None

# Last turbulence scheme concretely APPLIED + VERIFIED to the process-global ψ̂
# tables by an eager (non-traced) _apply_turbulence_scheme call.  A traced step
# (jax.jit / jax.grad) must NOT re-run the host-side float() probe on the now-traced
# tables; instead it asserts against this record that the eager cold-start step
# already installed the requested scheme.  Distinct from _DIFF_TURBULENCE_SCHEME,
# which is the (stricter) trace-time LOCK committed only for a traced step.
_APPLIED_TURBULENCE_SCHEME: str | None = None


def _psihat_probe() -> dict[str, float]:
    """Evaluate ψ̂ through every real lookup entry point (JAX and scalar).

    Keyed by entry point so a PARTIAL failure can be named: if the root-solver
    lookups still return RSL values while ``_GetPsiRSL`` returns MOST (or vice
    versa), the run is an inconsistent hybrid — worse than either scheme — and
    the caller must be told exactly which path disagreed.
    """
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyTurbulenceMod as _turb
    return {
        name: float(getattr(_turb, name)(_PSIHAT_PROBE_ZDT, _PSIHAT_PROBE_DTL))
        for name in ("_LookupPsihatM", "_LookupPsihatH",
                     "_LookupPsihatM_scalar", "_LookupPsihatH_scalar")
    }


def _apply_canopy_layering(canopy_config: CLMMLCanopyConfig) -> None:
    """Install the canopy layer counts CLM-ML builds its vertical structure from.

    ``CLMMLCanopyConfig.nlevmlcan`` / ``nlayer_above`` select the backend's
    EXPLICIT-COUNT mode: ``nlayer_within = nlevmlcan - nlayer_above`` layers
    spanning ``0..htop`` plus ``nlayer_above`` layers from ``htop`` to the
    reference height.  Without this the backend falls back to its
    height-increment mode (``dz_tall = 0.5 m``), which for a real forest canopy
    (htop ~ 27 m) makes ~54 layers whose beta-distribution LAI tails fall below
    ``dpai_min`` — those layers are zeroed and the run dies with
    ``initVerticalStructure: canopy layer has zero plant area index``.

    Both module namespaces are written because ``MLinitVerticalMod`` does
    ``from MLclm_varctl import nlayer_within, nlayer_above`` — a BY-VALUE import,
    so setting only the ``MLclm_varctl`` attribute is a silent no-op and the
    layering would stay at the backend default.

    Re-applied on every eager interface call (like the turbulence tables):
    these are process-global CLM module state shared by every column and
    config, so setting them once would leak the first caller's layering into a
    later differently-configured run in the same process.  The mutation is a
    host-side Python write — under ``jax.jit`` it happens at trace time, not per
    compiled execution, so a jitted rollout that never re-traces will not
    re-apply it (fine: the vertical structure is built once, at the cold-start
    step, and the counts do not change within a run).  Concurrent runs with
    different layerings in one process are unsafe unless externally serialised.
    """
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl as _ml_ctl
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLinitVerticalMod as _init_vert

    n_above = int(canopy_config.nlayer_above)
    n_within = int(canopy_config.nlevmlcan) - n_above
    for _mod in (_ml_ctl, _init_vert):
        _mod.nlayer_within = n_within
        _mod.nlayer_above = n_above


def _apply_stomatal_model(canopy_config: CLMMLCanopyConfig) -> None:
    """Install the leaf stomatal-conductance model (``gs_type``) into the backend.

    Maps ``CLMMLCanopyConfig.stomatal_model`` -> backend ``gs_type``
    {medlyn:0, ball_berry:1, wue:2}.  Patches BOTH ``MLclm_varctl.gs_type`` AND
    ``MLLeafPhotosynthesisMod.gs_type`` because the photosynthesis module does
    ``from MLclm_varctl import gs_type`` BY VALUE — a module-level copy the leaf
    kernel branches on — so setting only ``MLclm_varctl.gs_type`` is a silent
    no-op (the same by-value trap as the layering counts).

    Re-applied every step (process-global CLM state shared by every column and
    config).  Under ``"medlyn"`` the traced per-site ``vcmaxpft_jax`` injection
    is live (the backend threads it only through the Medlyn path); the default
    ``"wue"`` keeps the water-use-efficiency optimisation.
    """
    from legoesm.land.canopy.config import CLM_ML_STOMATAL_GS_TYPE
    scheme = canopy_config.stomatal_model
    if scheme not in CLM_ML_STOMATAL_GS_TYPE:
        raise ValueError(
            f"unknown CLM-ML stomatal_model {scheme!r}; must be one of "
            f"{tuple(CLM_ML_STOMATAL_GS_TYPE)}")
    gs = CLM_ML_STOMATAL_GS_TYPE[scheme]
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl as _ml_ctl
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafPhotosynthesisMod as _photo
    _ml_ctl.gs_type = gs
    _photo.gs_type = gs
    # The backend's leaf-kernel factories are ``functools.lru_cache``d on their
    # parameter tuple but NOT on ``gs_type`` (the returned kernel closes over it
    # as a static Python branch).  So SWITCHING stomatal model in one process
    # could otherwise reuse a kernel compiled for the previous model with the new
    # model's parameters.  Clear those caches whenever the applied gs_type
    # actually changes (tracked in ``_APPLIED_GS_TYPE`` so the common no-switch
    # case pays nothing).  This is a serial-process guard; concurrent mixed-model
    # calls remain unsupported (the state is process-global) — one stomatal model
    # per process is the contract.  Upstream fix: key the caches on gs_type.
    global _APPLIED_GS_TYPE
    if _APPLIED_GS_TYPE is not None and _APPLIED_GS_TYPE != gs:
        for _name in dir(_photo):
            _fn = getattr(_photo, _name, None)
            if hasattr(_fn, "cache_clear"):
                _fn.cache_clear()
    _APPLIED_GS_TYPE = gs


def _apply_turbulence_scheme(scheme: str, *, differentiable: bool = False) -> None:
    """Point CLM-ML's ψ̂ lookup tables at the selected turbulence scheme.

    Called on EVERY canopy step, not once at init: the tables are process-global
    CLM module state, so a one-shot mutation would leak the first caller's
    scheme into every later column, config and run in the same process.

    ``differentiable`` marks a step that will be traced.  Under ``jax.jit`` /
    ``jax.grad`` the table is captured as a trace-time constant, so switching
    schemes afterwards cannot reach an already-compiled function; that is
    refused loudly instead of silently returning the other scheme's physics.
    """
    if scheme not in VALID_CLM_ML_TURBULENCE_SCHEMES:
        raise ValueError(
            f"unknown CLM-ML turbulence_scheme {scheme!r}; the canopy-airspace "
            f"turbulence scheme must be one of {VALID_CLM_ML_TURBULENCE_SCHEMES}")

    global _PSIHAT_RSL

    if differentiable:
        if (_DIFF_TURBULENCE_SCHEME is not None
                and _DIFF_TURBULENCE_SCHEME != scheme):
            raise RuntimeError(
                "CLM-ML differentiable mode cannot switch turbulence_scheme "
                f"within a process (was {_DIFF_TURBULENCE_SCHEME!r}, now "
                f"{scheme!r}). The psihat lookup table is captured as a "
                "trace-time constant, so an already-traced/compiled step would "
                "keep running the OLD scheme while reporting the new one. Run "
                "one scheme per process, or rebuild the traced function in a "
                "fresh interpreter.")
        # NOTE: only CHECKED here, never committed.  The lock is committed by
        # _commit_diff_turbulence_scheme immediately before the step is traced,
        # so no path that raises first can lock the process to a scheme it never
        # actually compiled.

    # The tables must be populated before they can be snapshotted: at import
    # MLCanopyTurbulenceMod allocates psigrid* as ZEROS, so snapshotting a
    # pre-init module would pin "rsl_bonan" to a MOST table forever.
    _ensure_clm_initialized()
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyTurbulenceMod as _turb

    if _PSIHAT_RSL is None:
        snapshot: dict[str, Any] = {}
        for attr in _PSIHAT_TABLE_ATTRS:
            table = getattr(_turb, attr, None)
            if table is None:
                raise RuntimeError(
                    "CLM-ML turbulence-scheme selection requires "
                    f"MLCanopyTurbulenceMod.{attr} (an RSL psihat lookup table) "
                    "to be populated, but it is absent or None. Either "
                    "LookupPsihatINI has not run, or the installed clm-ml-jax "
                    "stores psihat differently — in which case "
                    "turbulence_scheme='most' would NOT remove the "
                    "roughness-sublayer term and must not be trusted.")
            if not bool(np.any(np.asarray(table))):
                raise RuntimeError(
                    f"CLM-ML psihat table MLCanopyTurbulenceMod.{attr} is "
                    "all zeros, so the roughness-sublayer correction is already "
                    "absent. Snapshotting it as the 'rsl_bonan' reference would "
                    "silently pin BOTH schemes to Monin-Obukhov. Expected "
                    "LookupPsihatINI to have loaded the RSL lookup tables.")
            snapshot[attr] = table.copy()
        _PSIHAT_RSL = snapshot

    for attr in _PSIHAT_TABLE_ATTRS:
        pristine = _PSIHAT_RSL[attr]
        if scheme == "rsl_bonan":
            # Fresh copy: never hand CLM the snapshot itself.
            table = pristine.copy()
        elif isinstance(pristine, jax.Array):
            table = jnp.zeros_like(pristine)
        else:
            # Fresh zeros per apply, so an in-place upstream write can never
            # turn a cached "most" table back into RSL values.
            table = np.zeros_like(pristine)
        setattr(_turb, attr, table)

    # Verify through the REAL lookup functions rather than trusting that setting
    # those four names still controls ψ̂.  Catches an upstream rename or analytic
    # reimplementation, which would otherwise degrade to silently running the
    # wrong scheme.  Run on EVERY apply, not only when the scheme changes: the
    # lookup functions are module attributes that anything (a later import, a
    # test, a plugin) can rebind at any time, and a cached "already verified"
    # verdict would not notice.  Four scalar lookups per canopy step is noise
    # next to CLM-ML's eager host-synchronising solver.
    probe = _psihat_probe()
    if scheme == "most":
        bad = {k: v for k, v in probe.items() if v != 0.0}
        if bad:
            raise RuntimeError(
                "turbulence_scheme='most' did not remove the roughness-sublayer "
                f"term; these psihat lookups are still non-zero: {bad}. The "
                f"installed clm-ml-jax no longer sources ψ̂ solely from "
                f"{_PSIHAT_TABLE_ATTRS}; the MOST option must not be trusted.")
    else:
        # ANY zero path is a defect, not just all of them: a mix means the
        # Obukhov root solver and _GetPsiRSL would run different theories.
        bad = {k: v for k, v in probe.items() if v == 0.0}
        if bad:
            raise RuntimeError(
                "turbulence_scheme='rsl_bonan' left psihat at zero for "
                f"{sorted(bad)}, so that path silently runs Monin-Obukhov while "
                "the rest runs the roughness-sublayer correction.")

    # Record the concretely-applied + verified scheme so a later TRACED step can
    # assert the tables match without re-running the host float() probe (which
    # cannot run inside a jax.jit trace — the lookups become tracers).
    global _APPLIED_TURBULENCE_SCHEME
    _APPLIED_TURBULENCE_SCHEME = scheme


def _assert_turbulence_scheme_for_trace(scheme: str) -> None:
    """Trace-safe turbulence-scheme check for the warm traceable path.

    The ψ̂ tables are process-global and were applied + VERIFIED concretely by the
    eager cold-start step that built the warm ``canopy_state`` (see
    :func:`_apply_turbulence_scheme`).  A traced (jax.jit / jax.grad) step must NOT
    re-run the host-side ``float()`` probe on the now-traced tables, but it MUST
    still refuse a scheme it cannot honour:

    * ``_APPLIED_TURBULENCE_SCHEME != scheme`` — the tables currently installed do
      NOT match the requested scheme (no eager step applied it, or a different
      scheme was applied since), so a trace would silently run the wrong physics.
    * ``_DIFF_TURBULENCE_SCHEME`` set to another scheme — the process is already
      locked to a different scheme that an already-compiled step keeps running.

    Both are hard errors (no silent degrade), mirroring the eager probe's contract.

    OPERATIONAL CONSTRAINT (codex round-2): the ψ̂ tables are process-global and this
    check is a string label, not a lock over the assert→trace→table-capture region.
    It is sound for the intended deployment — one serialized process running one
    turbulence scheme — but is NOT thread-safe: a concurrent thread mutating the
    tables between this assert and the backend trace could capture the wrong scheme.
    Run CLM-ML canopy rollouts single-threaded / one-scheme-per-process.
    """
    if scheme not in VALID_CLM_ML_TURBULENCE_SCHEMES:
        raise ValueError(
            f"unknown CLM-ML turbulence_scheme {scheme!r}; the canopy-airspace "
            f"turbulence scheme must be one of {VALID_CLM_ML_TURBULENCE_SCHEMES}")
    if _APPLIED_TURBULENCE_SCHEME != scheme:
        raise RuntimeError(
            "CLM-ML traceable step requested turbulence_scheme="
            f"{scheme!r}, but the process-global psihat tables currently hold "
            f"{_APPLIED_TURBULENCE_SCHEME!r} (set by the last eager apply). The "
            "host-side ψ̂ verification probe cannot run inside a jax.jit trace, so "
            "the scheme must be applied + verified by an eager (differentiable=False, "
            "cold-start) step in THIS process before a traced/jitted rollout. Run "
            "one forward cold step first, then reuse its canopy_state.")
    if (_DIFF_TURBULENCE_SCHEME is not None
            and _DIFF_TURBULENCE_SCHEME != scheme):
        raise RuntimeError(
            "CLM-ML traceable mode cannot switch turbulence_scheme within a "
            f"process (locked to {_DIFF_TURBULENCE_SCHEME!r}, now {scheme!r}). The "
            "psihat lookup table is captured as a trace-time constant, so an "
            "already-traced/compiled step keeps running the OLD scheme.")


def _commit_diff_turbulence_scheme(scheme: str) -> None:
    """Lock the process to ``scheme`` for differentiable mode.

    Call this ONLY at the point where the traced step is about to be built —
    after every diff-mode preflight check has passed.  Committing earlier (e.g.
    inside :func:`_apply_turbulence_scheme`) would let a call that raises during
    preflight lock the process to a scheme it never compiled, and then wrongly
    reject a later, valid run under the other scheme.
    """
    global _DIFF_TURBULENCE_SCHEME
    _DIFF_TURBULENCE_SCHEME = scheme


def resolve_num_ml_steps(canopy_config: CLMMLCanopyConfig, dt: float) -> int:
    """CLM-ML sub-steps to take within one legoESM step of length ``dt``.

    ``num_ml_steps=None`` (the default) derives the count so the canopy runs at
    its design sub-step ``dtime_ml_target_s`` regardless of the host timestep.
    That matters because the canopy air-space storage term is stiff on a
    timescale of minutes: driving it at the host ``dt`` (1800 s is typical)
    makes the term a numerical artefact that buffers energy through the day and
    releases it at night.  See ``CLMMLCanopyConfig.num_ml_steps``.

    An explicit count that implies a sub-step COARSER than the design value is
    REFUSED, unless ``allow_coarse_ml_substep`` opts in (for reproducing a
    published or legacy configuration verbatim), in which case it warns.
    """
    # Shared validator: rejects bools, non-numbers, nan and inf as well as <= 0.
    # Guarded explicitly rather than trusted because an inf target would make
    # ceil() return 1 and silently defeat the very check this function exists
    # to enforce, and a bool dt would be read as a 1-second step.
    from legoesm.core.setup_selector import require_positive_finite
    # NB the shared validator accepts None as "not set / use the default"; both
    # of these are REQUIRED here, so reject None first — otherwise it would fall
    # through to float() and raise TypeError instead of the documented error.
    for _name, _value in (("CLM-ML canopy step dt", dt),
                          ("CLMMLCanopyConfig.dtime_ml_target_s",
                           canopy_config.dtime_ml_target_s)):
        if _value is None:
            raise ValueError(f"{_name} must be a finite number > 0, got None")
        require_positive_finite(_name, _value)
    dt = float(dt)
    target = float(canopy_config.dtime_ml_target_s)

    ratio = dt / target
    if not math.isfinite(ratio):
        # e.g. a denormal target (5e-324) overflows the division; ceil() would
        # raise OverflowError instead of the documented validation error.
        raise ValueError(
            f"CLM-ML sub-step count dt/dtime_ml_target_s is not finite "
            f"(dt={dt!r}, target={target!r})")

    if canopy_config.num_ml_steps is None:
        # CEILING, not round: rounding to nearest can land on a count whose
        # sub-step is COARSER than the target (dt=750 -> round(2.5)=2 -> 375 s),
        # which is exactly the regime this field exists to avoid.
        return max(1, math.ceil(ratio))

    n_sub = canopy_config.num_ml_steps
    # Explicit means explicit: silently int()-coercing would accept a 1.9 or a
    # YAML "6" and run a different sub-step than the config asked for.  Any
    # integral type is fine (a count assembled from numpy is still a count);
    # bool is not, since ``num_ml_steps=True`` is a mistake, not a request for
    # one sub-step.
    if isinstance(n_sub, bool) or not isinstance(n_sub, numbers.Integral):
        raise TypeError(
            "CLMMLCanopyConfig.num_ml_steps must be an integer (or None to "
            f"derive it from dtime_ml_target_s), got {n_sub!r} of type "
            f"{type(n_sub).__name__}")
    n_sub = int(n_sub)
    if n_sub < 1:
        raise ValueError(
            "CLMMLCanopyConfig.num_ml_steps must be >= 1 (or None to derive it "
            f"from dtime_ml_target_s), got {n_sub!r}")

    effective = dt / n_sub
    if effective > target:
        msg = (
            f"CLM-ML canopy sub-step would be {effective:.12g} s "
            f"(dt={dt:.12g} s / num_ml_steps={n_sub}), coarser than the "
            f"{target:.12g} s the scheme is designed for. The canopy air-space "
            "storage term is stiff on a timescale of minutes, so it buffers "
            "energy through the day and releases it at night — inflating "
            "nighttime latent heat and delaying the sensible-heat peak (at "
            "US-MMS this cost 58% of the latent-heat diurnal skill). Pass "
            "num_ml_steps=None to derive the sub-step automatically.")
        if canopy_config.allow_coarse_ml_substep is not True:
            # `is not True`, not falsiness: opting into a known-degraded
            # configuration must be an explicit bool, so a stray "false"
            # string cannot enable it by truthiness.
            # Refuse rather than warn: a warning is routinely suppressed or
            # buried in batch output, and this silently degrades production
            # fluxes.  allow_coarse_ml_substep=True is the deliberate opt-in.
            raise ValueError(
                msg + " Set allow_coarse_ml_substep=True to run anyway (e.g. "
                "to reproduce a legacy configuration verbatim).")
        warnings.warn(msg + " Running anyway: allow_coarse_ml_substep=True.",
                      RuntimeWarning, stacklevel=2)
    return n_sub


def _setup_clm_topology(
    ncol: int,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    dz_soil: np.ndarray,
    z_soil: np.ndarray,
    z_ref: float,
    pft_clm: int = 7,
) -> None:
    """Set up CLM module-level topology singletons for ``ncol`` columns.

    Uses a 1:1 mapping: patch index ``p`` = column index ``c`` = gridcell
    index ``g`` = legoESM column ``i + 1`` (1-based).

    Parameters
    ----------
    ncol : int
        Number of legoESM columns.
    lat_deg : np.ndarray
        Latitude of each column [degrees], shape ``(ncol,)``.
    lon_deg : np.ndarray
        Longitude of each column [degrees, -180..180 or 0..360],
        shape ``(ncol,)``.  Used by the internal solar zenith calculation.
    dz_soil : np.ndarray
        Soil layer thicknesses [m], shape ``(n_layers,)``.
    z_soil : np.ndarray
        Soil layer mid-point depths from surface [m], shape ``(n_layers,)``.
    z_ref : float
        Atmospheric reference height [m].
    pft_clm : int
        CLM PFT index (1-based) applied to all columns.  Controls Vcmax25
        and plant hydraulic parameters via the MLpftcon lookup table.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main import ColumnType as _col_mod
    from legoesm.land.canopy.clm_ml_backend.clm_src_main import GridcellType as _grc_mod
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import column_type
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevsno, nlevgrnd, nlevsoi
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval

    n_layers = len(dz_soil)

    # ---- patch ----
    # patch arrays: 1-based, index 0 unused
    col_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    gc_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    itype_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    for i in range(ncol):
        p = i + 1  # 1-based patch index
        col_arr[p] = p       # column = patch (1:1)
        gc_arr[p] = p        # gridcell = patch (1:1)
        itype_arr[p] = pft_clm  # CLM PFT from config (was hardcoded to 13)

    patch.column = jnp.array(col_arr, dtype=jnp.int32)
    patch.gridcell = jnp.array(gc_arr, dtype=jnp.int32)
    patch.itype = jnp.array(itype_arr, dtype=jnp.int32)

    # ---- col ----
    # Shape: (ncol+1, nlevsno+nlevgrnd+1)
    nth = nlevsno + nlevgrnd + 1
    snl_np = np.zeros(ncol + 1, dtype=np.int32)  # no snow
    snl_np[0] = ispval
    dz_np = np.full((ncol + 1, nth), np.nan)
    z_np = np.full((ncol + 1, nth), np.nan)
    zi_np = np.full((ncol + 1, nth), np.nan)
    nbedrock_np = np.full(ncol + 1, ispval, dtype=np.int32)

    # Soil layers in CLM convention:
    # Python index j (1-based) → Fortran layer j → depth from surface
    # nlevsno=5 in CLM4.5, so Python j=6..15 are soil layers 1..10 for CLM4.5
    # BUT in standalone mode snl=0, so Fortran layer 1 = Python index nlevsno+1 = 6
    # _GetCLMVar: j = int(snl[c]) + 1 = 1 → Python index 1 (no snow offset applied)
    # This seems inconsistent... let me use the simpler j=1..n_layers directly.
    # _GetCLMVar line: j = int(_snl_np[c]) + 1 = 1, then t_soisno_col[c, 1]
    # So the convention in standalone CLM-ML-JAX is: j=1 is the top soil layer,
    # regardless of the snow layer allocation. The snl offset is not applied here.

    for i in range(ncol):
        c = i + 1  # 1-based column index
        for j in range(1, n_layers + 1):
            dz_np[c, j] = float(dz_soil[j - 1])
            z_np[c, j] = float(z_soil[j - 1])
        # Fill remaining CLM soil layers (n_layers+1..nlevsoi) with the deepest
        # legoESM layer rather than leaving them as NaN.  SoilResistance reads
        # dz[c, j] for all j=1..nlevsoi; NaN propagates to gradients via jnp.where.
        for j in range(n_layers + 1, nlevsoi + 1):
            dz_np[c, j] = float(dz_soil[-1])
            z_np[c, j] = float(z_soil[-1]) + float(dz_soil[-1]) * (j - n_layers)
        # Interface depths: zi[c, 0] = 0 (surface), zi[c, j] = cumulative depth
        zi_np[c, 0] = 0.0
        cum = 0.0
        for j in range(1, nlevsoi + 1):
            cum += dz_np[c, j]
            zi_np[c, j] = cum
        nbedrock_np[c] = n_layers

    _col_mod.col = column_type(
        snl=jnp.array(snl_np, dtype=jnp.int32),
        dz=jnp.array(dz_np, dtype=jnp.float64),
        z=jnp.array(z_np, dtype=jnp.float64),
        zi=jnp.array(zi_np, dtype=jnp.float64),
        nbedrock=jnp.array(nbedrock_np, dtype=jnp.int32),
    )

    # ---- grc ----
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.GridcellType import GridcellType as gridcell_type
    latdeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    londeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    for i in range(ncol):
        g = i + 1
        latdeg_np[g] = float(lat_deg[i]) if i < len(lat_deg) else 0.0
        londeg_np[g] = float(lon_deg[i]) if i < len(lon_deg) else 0.0
    _grc_mod.grc = gridcell_type(
        latdeg=jnp.array(latdeg_np, dtype=jnp.float64),
        londeg=jnp.array(londeg_np, dtype=jnp.float64),
    )


def _setup_clm_time(dt: float, doy: float, step_count: int) -> None:
    """Configure the CLM time manager so that ``get_curr_calday(0) ≈ doy + 1``.

    CLM calday convention: calday 1.000 = 0Z on Jan 1.  So if legoESM
    provides ``doy`` as a 0-based float (0.0 = Jan 1, 120.0 = May 1 in a
    non-leap year), the correct CLM calday is ``doy + 1``.

    We encode this by pinning ``start_date_ymd = 20000101`` (calday=1) and
    choosing ``itim`` such that::

        get_curr_calday(0) = 1 + itim * dt / 86400 ≈ doy + 1
        → itim = round(doy * 86400 / dt)

    This fixes the previous bug where ``itim = step_count`` (1, 2, 3 …)
    caused the internal calday to always be near January 1 regardless of the
    actual simulation date, producing wrong solar zenith angles and hence
    wrong sun/shade fractions and per-layer radiation profiles.
    """
    import legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager as _tm

    _tm.dtstep = int(dt)
    # Encode actual day-of-year into itim so that get_curr_calday returns
    # the correct calendar day for solar zenith computation.
    _tm.itim = max(1, round(doy * 86400.0 / max(dt, 1.0)))
    # Use year 2001 (non-leap) as the reference epoch for both start_date and
    # curr_date.  Year 2000 is a leap year (366 days); the CLM time manager's
    # get_curr_date() recomputes curr_date_ymd from itim*dtstep using isleap(),
    # so keeping start_date_ymd=20000101 caused all caldays after Feb 28 to be
    # 1 day behind (Apr 30 instead of May 1 for doy=120).
    _tm.start_date_ymd = 20010101  # coeff-ok: CLM time epoch: Jan 1 2001 (non-leap year, 365 days)
    _tm.start_date_tod = 0
    # Derive curr_date_ymd from doy (non-leap 2001 epoch).
    import datetime as _dt
    _jan1 = _dt.date(2001, 1, 1)  # coeff-ok: CLM time epoch: Jan 1 2001 (non-leap year, 365 days)
    _curr = _jan1 + _dt.timedelta(days=int(doy))
    _tm.curr_date_ymd = int(_curr.strftime("%Y%m%d"))
    _tm.curr_date_tod = int((doy * 86400.0) % 86400)


def _compute_cos_zenith(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    doy: float,
) -> np.ndarray:
    """Compute cosine of solar zenith angle per column.

    .. note::
        **Legacy function** — used only in unit tests
        (``tests/land/unit/test_canopy.py::TestSolarGeometry``).
        The production path uses :func:`_compute_virtual_lon_deg` with
        CLM's Kepler ``shr_orb_cosz`` for round-trip consistency.
        Do NOT delete this function without updating the test class.

    Uses the Spencer (1971) declination formula.  Accuracy is ±0.01 in
    cos_zen, sufficient for the Weiss–Norman clearness-index partition.

    Parameters
    ----------
    lat_deg : np.ndarray  shape (ncol,), latitude in degrees
    lon_deg : np.ndarray  shape (ncol,), longitude in degrees (-180..180 or 0..360)
    doy     : float, 0-based day of year (0.0 = Jan 1 00:00 UTC)

    Returns
    -------
    cos_zen : np.ndarray  shape (ncol,), clamped to [0, 1]
    """
    lat_r = np.deg2rad(lat_deg)
    # Solar declination — Spencer (1971).
    # doy is 0-based (0.0 = Jan 1 00:00), so Spencer's d_n = doy + 1 and
    # B = 2π*(d_n-1)/365 = 2π*doy/365.  The previous (doy-1) was wrong by 1 day.
    B = 2.0 * np.pi * doy / 365.0  # coeff-ok: Julian year length (365 days, non-leap 2001 epoch)
    decl = (0.006918                # coeff-ok: Spencer (1971, J. Appl. Meteorol.) declination polynomial
            - 0.399912 * np.cos(B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.070257 * np.sin(B)  # coeff-ok: Spencer (1971) declination polynomial
            - 0.006758 * np.cos(2 * B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.000907 * np.sin(2 * B)  # coeff-ok: Spencer (1971) declination polynomial
            - 0.002697 * np.cos(3 * B)  # coeff-ok: Spencer (1971) declination polynomial
            + 0.00148  * np.sin(3 * B))  # coeff-ok: Spencer (1971) declination polynomial
    # Fractional time of day (UTC hours from doy fractional part)
    frac = doy % 1.0           # 0.0 = midnight, 0.5 = noon UTC
    utc_hour = frac * 24.0     # coeff-ok: exact hours-per-day conversion (24 h/day)
    # Local solar time hour angle (degrees, 0=noon)
    lon_norm = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
    ha_deg = (utc_hour - 12.0) * 15.0 + lon_norm  # coeff-ok: exact degrees-per-hour (360°/24h=15°/h)
    ha_r = np.deg2rad(ha_deg)
    cos_zen = (np.sin(lat_r) * np.sin(decl)
               + np.cos(lat_r) * np.cos(decl) * np.cos(ha_r))
    return np.maximum(cos_zen, 0.0)


def _compute_virtual_lon_deg(
    cos_zen_forcing: np.ndarray,
    lat_deg: np.ndarray,
    caldaym1: float,
) -> np.ndarray:
    """Invert shr_orb_cosz to find a longitude that reproduces cos_zen_forcing.

    Uses CLM's own Kepler orbital mechanics (``shr_orb_decl``) and the same
    ``caldaym1`` that ``MLCanopyFluxesMod._MLCanopyForcing`` will use, so the
    inversion is exact rather than approximated by Spencer (1971).

    Must be called after ``_ensure_clm_initialized()`` (sets ``clm_varorb``
    orbital parameters used by ``shr_orb_decl``).

    The CLM formula (``shr_orb_cosz``):
        cosz = sin(lat)*sin(decl) - cos(lat)*cos(decl)*cos(jday_frac*2π + lon)
    Inversion:
        lon  = arccos((sin(lat)*sin(decl) − cosz) / (cos(lat)*cos(decl)))
               − jday_frac * 2π

    Both arccos branches produce the same cosz when substituted back, so the
    principal branch (arccos → [0, π]) is unambiguous.  The arccos result is
    well-defined even at night (``cos_arg`` is clipped to [−1, 1]) so no
    nighttime guard is needed.

    Parameters
    ----------
    cos_zen_forcing : np.ndarray  shape (ncol,), ≥ 0 (clamp before calling)
    lat_deg : np.ndarray  shape (ncol,), latitude [°]
    caldaym1 : float
        CLM calendar day at the *beginning* of the current timestep.
        Equals ``get_curr_calday(offset=-dt)`` after ``_setup_clm_time``.
        Computed as ``1 + (itim - 1) * dt / 86400`` where
        ``itim = max(1, round(doy * 86400 / dt))``.

    Returns
    -------
    lon_deg : np.ndarray  shape (ncol,), virtual longitude [°, −180..180]
    """
    from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import shr_orb_decl
    import legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb as _varorb

    lat_r = np.deg2rad(lat_deg)

    # CLM Kepler orbital mechanics — exact match with _MLCanopyForcing
    declinm1, _ = shr_orb_decl(caldaym1, _varorb.eccen, _varorb.mvelpp,
                                _varorb.lambm0, _varorb.obliqr)
    decl = float(declinm1)

    jday_frac = caldaym1 % 1.0  # fractional part of caldaym1 → [0, 1)

    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_decl = np.full_like(lat_r, np.sin(decl))
    cos_decl = np.full_like(lat_r, np.cos(decl))

    denom = cos_lat * cos_decl
    denom_safe = np.where(np.abs(denom) > 1e-6, denom,
                          np.where(denom >= 0, 1e-6, -1e-6))
    cos_arg = np.clip((sin_lat * sin_decl - cos_zen_forcing) / denom_safe, -1.0, 1.0)

    lon_rad = np.arccos(cos_arg) - jday_frac * 2.0 * np.pi
    lon_rad = ((lon_rad + np.pi) % (2.0 * np.pi)) - np.pi  # → [−π, π]
    return np.degrees(lon_rad)


def _estimate_beam_fraction(
    sw_down: jnp.ndarray,
    cos_zen: np.ndarray,
) -> jnp.ndarray:
    """Estimate direct-beam fraction from clearness index (Erbs et al. 1982).

    Clearness index  kt = SW_down / (S0 * cos_zen)  where S0 = 1361 W/m².
    Erbs et al. (1982, Solar Energy 28:293-302) give the *diffuse* fraction Id/I:

        kt ≤ 0.22:  Id/I = 1 - 0.09*kt
        0.22 < kt ≤ 0.80:
            Id/I = 0.9511 - 0.1604*kt + 4.388*kt² - 16.638*kt³ + 12.336*kt⁴
        kt > 0.80:  Id/I = 0.165

    Direct fraction: f_dir = 1 - Id/I, clamped to [0, 1].
    Returns f_dir per column.

    JAX-native: ``sw_down`` is kept as a traced ``jnp`` array so that
    ``d(f_dir)/d(sw_down)`` (via the clearness index ``kt``) stays on the
    ``jax.grad`` tape.  ``cos_zen`` is solar geometry (a non-differentiated
    constant); passing a NumPy array is fine — ``jnp`` ops upcast it.
    """
    S0 = constants.S_0  # solar constant [W/m²]
    sw = jnp.asarray(sw_down, dtype=jnp.float64)
    cos_zen_clamped = jnp.maximum(jnp.asarray(cos_zen, dtype=jnp.float64), 0.01)  # coeff-ok: minimum cos_zen floor to avoid division by zero in clearness index
    sw_toa = S0 * cos_zen_clamped
    kt = jnp.where(sw > 1.0, jnp.minimum(sw / sw_toa, 1.0), 0.0)

    # Erbs et al. (1982, Solar Energy 28:293-302) diffuse-fraction polynomial
    id_over_i_low = 1.0 - 0.09 * kt  # coeff-ok: Erbs et al. (1982, Solar Energy 28:293) low-kt regime
    id_over_i_mid = (0.9511 - 0.1604 * kt + 4.388 * kt**2  # coeff-ok: Erbs et al. (1982) mid-kt polynomial
                     - 16.638 * kt**3 + 12.336 * kt**4)     # coeff-ok: Erbs et al. (1982) mid-kt polynomial
    id_over_i_high = jnp.full_like(kt, 0.165)  # coeff-ok: Erbs et al. (1982) high-kt (clear-sky) limit
    id_over_i = jnp.where(kt <= 0.22, id_over_i_low,  # coeff-ok: Erbs et al. (1982) kt regime threshold
                jnp.where(kt <= 0.80, id_over_i_mid, id_over_i_high))  # coeff-ok: Erbs et al. (1982) kt regime threshold
    f_dir = jnp.clip(1.0 - id_over_i, 0.0, 1.0)

    # At night (sw_down < 1 W/m²) force beam fraction to zero
    return jnp.where(sw < 1.0, 0.0, f_dir)


def _sw_partition(
    sw_down: jnp.ndarray,
    f_vis: float = 0.46,  # coeff-ok: observation-based VIS fraction (Weiss & Norman 1985; ~0.46 climatological mean)
    f_dir: float = -1.0,
    cos_zen: np.ndarray | None = None,
    f_dir_fallback: float = 0.30,  # coeff-ok: Erbs et al. (1982) overcast-sky beam fraction fallback; prefer CLMMLCanopyConfig.f_dir_noclearness_fallback
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Partition total downwelling SW into direct/diffuse × VIS/NIR bands.

    Parameters
    ----------
    sw_down : jnp.ndarray
        Total downwelling shortwave [W/m²], shape ``(ncol,)``.
    f_vis : float
        Fraction of total SW in the visible (PAR) band [0.4–0.7 µm].
        Observation-based climatological value is 0.46 (not 0.50).
    f_dir : float
        Direct-beam fraction of total SW.
        -1.0  → auto-estimated from ``cos_zen`` via clearness index
                 (Weiss & Norman 1985; Erbs et al. 1982).  This is the
                 physically correct default for ESM applications.
        0–1   → fixed override (use only for idealised runs).
    cos_zen : np.ndarray | None
        Cosine of solar zenith angle per column, shape ``(ncol,)``.
        Required when ``f_dir < 0``.  Ignored otherwise.
    f_dir_fallback : float
        Beam fraction used when ``f_dir < 0`` but ``cos_zen`` is unavailable.
        Provided from ``CLMMLCanopyConfig.f_dir_noclearness_fallback``.

    Returns
    -------
    swskyb_vis, swskyb_nir : jnp.ndarray
        Direct beam SW in VIS and NIR bands [W/m²].
    swskyd_vis, swskyd_nir : jnp.ndarray
        Diffuse SW in VIS and NIR bands [W/m²].

    JAX-native: ``sw_down`` is kept traced end-to-end so ``d(swsky*)/d(sw_down)``
    flows on the ``jax.grad`` tape (needed for differentiability w.r.t. the SW
    forcing).  ``f_vis``/``f_dir``/``f_dir_fallback`` are static Python floats, so
    the ``if f_dir < 0.0`` branch is resolved at trace time (not a traced select).
    """
    sw = jnp.asarray(sw_down, dtype=jnp.float64)

    if f_dir < 0.0:
        # Physics-based estimate using clearness index
        if cos_zen is None:
            # Fallback: assume overcast (conservative, no zenith info)
            f_dir_arr = jnp.full_like(sw, f_dir_fallback)
        else:
            f_dir_arr = _estimate_beam_fraction(sw, cos_zen)
    else:
        f_dir_arr = jnp.full_like(sw, float(f_dir))

    vis = f_vis * sw
    nir = (1.0 - f_vis) * sw
    swskyb_vis = f_dir_arr * vis
    swskyb_nir = f_dir_arr * nir
    swskyd_vis = (1.0 - f_dir_arr) * vis
    swskyd_nir = (1.0 - f_dir_arr) * nir
    return swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir


def _build_stubs(
    ncol: int,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    T_soil_top: jnp.ndarray,
    psi_soil: jnp.ndarray | None,
    theta_soil: jnp.ndarray | None,
    dz_soil: np.ndarray,
    soil_hydraulics: Any | None,
    cos_zen: np.ndarray | None = None,
    T_soil_all: jnp.ndarray | None = None,
    t_a10_prior: jnp.ndarray | None = None,
    lai_override: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Build minimal CLM input stub objects from legoESM state.

    All stubs are simple Python namespaces; only the fields accessed by
    ``_GetCLMVar``, ``initVerticalStructure``, ``SoilResistance``, and the
    soil relative humidity block are populated.

    Returns
    -------
    dict with keys: atm2lnd, wateratm2lndbulk, surfalb, soilstate,
                    waterstatebulk, temperature, frictionvel, canopystate,
                    energyflux, waterfluxbulk, solarabs, waterdiagnosticbulk
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevsoi, ivis, inir, nlevgrnd, nlevsno
    # NOTE: ivis=1, inir=2 in CLM

    np_ = ncol + 1  # 1-based patch dimension

    # ---- SW partitioning ----
    swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir = _sw_partition(
        forcing.sw_down,
        f_vis=float(canopy_config.f_vis),
        f_dir=float(canopy_config.f_dir),
        cos_zen=cos_zen,
        f_dir_fallback=float(canopy_config.f_dir_noclearness_fallback),
    )

    # shape (np_, numrad+1) = (np_, 3); CLM uses ivis=1, inir=2
    numrad = 2
    swskyb_col = np.zeros((np_, numrad + 1))
    swskyd_grc = np.zeros((np_, numrad + 1))
    swskyb_col_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    swskyd_grc_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)

    for i in range(ncol):
        p = i + 1  # 1-based
        swskyb_col_jax = swskyb_col_jax.at[p, ivis].set(swskyb_vis[i])
        swskyb_col_jax = swskyb_col_jax.at[p, inir].set(swskyb_nir[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, ivis].set(swskyd_vis[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, inir].set(swskyd_nir[i])

    # ---- CO2 and O2 partial pressures ----
    # forc_pco2[g] = CO2 partial pressure [Pa]
    # co2_ppmv × p_surface × 1e-6 → Pa
    # In _GetCLMVar: co2ref_cur = forc_pco2[g] / pbot * 1e6 → umol/mol (round-trips)
    co2_pa = forcing.co2_ppmv * forcing.p_surface * 1.0e-6  # Pa
    o2_pa = canopy_config.o2ref * 1.0e-3 * forcing.p_surface  # coeff-ok: exact unit conversion mmol/mol → mol/mol (1e-3); Pa = mol/mol × p_surface

    # ---- 1-D forcing arrays (1-based) ----
    def _pad1(arr):
        """Pad array to shape (ncol+1,) with 0 at index 0."""
        out = jnp.zeros(np_, dtype=jnp.float64)
        return out.at[1:].set(arr.astype(jnp.float64))

    forc_u = _pad1(forcing.u_lowest)
    forc_v = _pad1(forcing.v_lowest)
    forc_pco2 = _pad1(co2_pa)
    forc_po2 = _pad1(o2_pa)
    forc_solad_col = swskyb_col_jax   # (np_, 3)  direct SW
    forc_solai_grc = swskyd_grc_jax   # (np_, 3)  diffuse SW
    forc_t = _pad1(forcing.T_lowest)
    forc_pbot = _pad1(forcing.p_surface)
    forc_lwrad = _pad1(forcing.lw_down)
    forc_q = _pad1(forcing.q_lowest)
    forc_rain = _pad1(forcing.precip_total - forcing.precip_snow)
    forc_snow = _pad1(forcing.precip_snow)

    # ---- Ground albedo ----
    # Use land_params or config fallback
    # albgrd_col / albgri_col are the SUB-CANOPY SOIL (ground) spectral albedos,
    # used as the bottom boundary in the CLM-ML two-stream RT solver.
    # These must NOT be set from land_params.albedo_veg (vegetation broadband
    # albedo) — doing so was a prior bug.  Use configurable soil albedo defaults:
    # CLM4.5 lookup for loam soil: VIS ≈ 0.10, NIR ≈ 0.20.
    albgrd_vis = float(canopy_config.albgrd_vis_default)
    albgrd_nir = float(canopy_config.albgrd_nir_default)
    albgrd_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    albgri_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        albgrd_col = albgrd_col.at[c, ivis].set(albgrd_vis)
        albgrd_col = albgrd_col.at[c, inir].set(albgrd_nir)
        albgri_col = albgri_col.at[c, ivis].set(albgrd_vis)
        albgri_col = albgri_col.at[c, inir].set(albgrd_nir)

    # ---- Soil state ----
    # Use nlevgrnd (total ground layers = 15 in CLM4.5) as the allocation size,
    # not nlevsoi (10 soil layers only).  MLSoilTemperatureMod writes to
    # thk[c, j] for j=1..nlevgrnd and would OOB on a nlevsoi+1-sized array.
    # Similarly t_soisno_col is declared with shape nlevsno+nlevgrnd in the type.
    # Both arrays are filled only for j=1..nlevsoi; deeper layers keep default.
    n_layers = len(dz_soil)
    smp_l_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)   # [mm]
    hk_l_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)    # [mm/s]
    rootfr_patch = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)
    h2osoi_ice = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)   # no ice
    # Soil thermal conductivity [W/m/K] — CLM4.5 Table 3.3 moist loam default.
    # Physical range 0.2–2.0 W/m/K; 0.9–1.5 for wetter/sandier soils.
    thk_col = jnp.full((np_, nlevgrnd + 1),
                       float(canopy_config.thk_soil_default_W_m_K),
                       dtype=jnp.float64)

    root_depth = float(land_config.root_depth)
    z_centers = np.array([float(z) for z in dz_soil], dtype=np.float64).cumsum() - dz_soil / 2

    # Root fraction exponential profile (same as legoESM multilayer_land.py)
    root_frac_np = np.exp(-z_centers / root_depth)
    root_frac_np = root_frac_np / root_frac_np.sum()

    # Pad root fraction to nlevsoi layers and renormalize so sum == 1.0.
    # When n_layers < nlevsoi, the deepest legoESM fraction is replicated into
    # extra CLM layers; without renormalization sum(rootfr) > 1 and btran is
    # overweighted.
    n_clm_soil = nlevsoi
    rootfr_padded = np.zeros(n_clm_soil, dtype=np.float64)
    for j in range(n_clm_soil):
        jl = min(j, n_layers - 1)
        rootfr_padded[j] = float(root_frac_np[jl])
    rootfr_total = rootfr_padded.sum()
    if rootfr_total > 0:
        rootfr_padded /= rootfr_total  # guarantee sum == 1.0

    # The COUPLED path supplies PER-COLUMN, PER-LAYER hydraulics
    # (build_soil_hydraulics -> (ncol, n_layer) fields: theta_sat/psi_sat/b_ch/
    # K_sat), while the single-site path (run_lmip) supplies SCALAR fields.  Slice
    # the config down to the current (column, layer) before hydraulic_conductivity
    # so it returns a SCALAR K — otherwise the full (ncol, n_layer) params broadcast
    # K to (ncol, ...) and ``hk_l_col.at[p, j].set(K)`` (a scalar slot) fails with
    # "Cannot broadcast to shape with fewer dimensions".  Scalars / the
    # retention_curve string / per-layer-only fields pass through unchanged.
    def _hydraulics_at(cfg_hyd, ci, li):
        def _sel(x):
            # The shipped producers emit ONLY scalars (default SoilHydraulicsConfig)
            # or 2-D (ncol, L) fields (build_soil_hydraulics -> (ncol, n_layer);
            # clm_hydraulics_config -> (ncol, 1) layer-broadcast).  Clamp BOTH dims
            # so a size-1 column- or layer-broadcast dim never goes out of bounds
            # (min(ci, ncol-1) is a no-op for a real ncol-sized dim).
            nd = getattr(x, "ndim", 0)
            if nd == 0:
                return x                       # scalar / retention_curve string
            if nd == 1:
                # 1-D hydraulics fields are not emitted by the shipped producers;
                # treat a length-ncol vector as per-column, else per-layer (clamped).
                return x[ci] if x.shape[0] == ncol else x[min(li, x.shape[0] - 1)]
            return x[min(ci, x.shape[0] - 1), min(li, x.shape[1] - 1)]
        return jax.tree_util.tree_map(_sel, cfg_hyd)

    for i in range(ncol):
        p = i + 1  # 1-based patch = column
        # Fill all nlevsoi layers so that layers beyond n_layers don't stay at
        # smp=0 mm (saturated) which biases btran via the weighted-average.
        for j in range(1, nlevsoi + 1):
            if psi_soil is not None and j - 1 < psi_soil.shape[1]:
                # psi [m] → smp_l [mm]; preserve sign (negative for unsaturated)
                smp_l_col = smp_l_col.at[p, j].set(psi_soil[i, j - 1] * 1000.0)
            else:
                smp_l_col = smp_l_col.at[p, j].set(float(canopy_config.smp_default_mm))

            if (soil_hydraulics is not None and psi_soil is not None
                    and theta_soil is not None and j - 1 < theta_soil.shape[1]):
                from legoesm.land.soil_hydraulics import hydraulic_conductivity
                K = hydraulic_conductivity(psi_soil[i, j - 1], theta_soil[i, j - 1],
                                           _hydraulics_at(soil_hydraulics, i, j - 1))
                # Keep K traced (no float()) so d(hk)/d(psi,theta) stays on the
                # jax.grad tape; numerically identical to the prior float() cast.
                hk_l_col = hk_l_col.at[p, j].set(K * 1000.0)  # m/s → mm/s
            else:
                hk_l_col = hk_l_col.at[p, j].set(float(canopy_config.hk_default_mm_s))

            rootfr_patch = rootfr_patch.at[p, j].set(rootfr_padded[j - 1])

    # ---- Soil evaporative resistance (Sellers-Lockwood formula) ----
    # The CLM-ML SurfaceResistanceMod uses rs = exp(8.206 - 4.255*Se) [s/m]
    # where Se = effective saturation.  For a silty clay loam with porosity 0.464
    # and the default smp = -50000 mm = -0.49 MPa, Se ≈ 0.15, so
    # rs ≈ exp(8.206 - 4.255*0.15) ≈ exp(7.568) ≈ 1927 s/m.
    # For saturated soil (smp ≈ 0): rs ≈ exp(8.206 - 4.255) ≈ 52 s/m.
    # We use 2000 s/m as the default to match the moderate-stress config default,
    # replacing the previous 100 s/m (matched to saturated soil only).
    _rs_base = float(canopy_config.soilresis_default_s_m)
    soilresis_col = jnp.full(np_, _rs_base, dtype=jnp.float64)

    # ---- Temperature ----
    # t_a10_patch is the 10-day running mean canopy air temperature [K].
    # On first call (canopy_state=None) we have no prior state, so we fall back
    # to instantaneous T_lowest.  The caller must extract t_a10 from canopy_state
    # and pass it here on subsequent steps via the t_a10_prior argument.
    t_a10_patch = _pad1(t_a10_prior if t_a10_prior is not None else forcing.T_lowest)
    # Fill all nlevsoi soil temperature layers.  Previously only layer 1 was set;
    # layers 2–10 = 0 K corrupted thermal gradient and soil heat flux.
    # t_soisno_col shape must be (np_, nlevgrnd+1) to match CLM-ML type declaration.
    t_soisno_col = jnp.zeros((np_, nlevgrnd + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        # No float() casts below: soil temperature is kept traced so
        # d(flux)/d(T_soil) / d(T_soil_top) flows on the jax.grad tape.  Values
        # are identical to the prior float() path in the eager (production) mode.
        if T_soil_all is not None and T_soil_all.shape[1] >= 1:
            n_fill = min(T_soil_all.shape[1], nlevsoi)
            for j in range(1, n_fill + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(T_soil_all[i, j - 1])
            # Layers n_fill+1..nlevsoi: repeat deepest legoESM layer
            deepest_T = T_soil_all[i, n_fill - 1]
            for j in range(n_fill + 1, nlevgrnd + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(deepest_T)
        else:
            # Fallback: fill all layers with surface soil temperature
            for j in range(1, nlevgrnd + 1):
                t_soisno_col = t_soisno_col.at[c, j].set(T_soil_top[i])

    # ---- canopystate ----
    htop_patch = jnp.zeros(np_, dtype=jnp.float64)
    elai_patch = jnp.zeros(np_, dtype=jnp.float64)
    esai_patch = jnp.zeros(np_, dtype=jnp.float64)
    for i in range(ncol):
        p = i + 1
        # NO float() on the per-column land_params / lai_override values: on the
        # COUPLED jitted path these are TRACED (LAI/htop/SAI flow through the land
        # state + surface blend in step_unified), so float() raises
        # ConcretizationTypeError.  Keep them as jnp/array scalars — .at[p].set
        # accepts a tracer, a numpy scalar (eager) or a Python-float fallback alike,
        # and the value is identical to the old float() path in eager mode.  Canopy
        # STRUCTURE (ncan/ntop/nbot) is NOT built from these here on the traceable
        # path — it comes from the concrete grid_info — so a traced htop/LAI only
        # feeds per-column DATA (RSL reference height, dpai), never a slice bound.
        htop_v = (land_params.htop[i] if land_params is not None and land_params.htop is not None
                  else 5.0)  # coeff-ok: 5 m fallback canopy height (CLM4.5 DBF-temperate default)
        # A prescribed/climatology htop can be 0 on a bare or uncovered column;
        # floor it so hbot = hbot_frac*htop stays < htop (valid CLM-ML layering).
        # LAI is 0 there, so the nominal height changes no canopy flux.  jnp.maximum
        # (not max()) so a traced htop_v does not break the Python comparison; pin
        # the floor to the (float64) patch dtype so a float32 htop_v does not store a
        # float32-rounded 0.1 (~1.5e-9 m drift vs the old float64 max()) (codex).
        htop_v = jnp.maximum(htop_v, jnp.asarray(_HTOP_GEOM_MIN_M, dtype=htop_patch.dtype))
        # LAI precedence: prognostic ``lai_override`` (C_fol / LCMA from the
        # DifferLand carbon pool) > prescribed ``LandSurfaceParams.LAI``
        # climatology > scalar fallback.  Canopy STRUCTURE (htop/SAI) stays
        # prescribed either way (the carbon cycle produces no allometric map).
        if lai_override is not None:
            lai_v = lai_override[i]
        elif land_params is not None and land_params.LAI is not None:
            lai_v = land_params.LAI[i]
        else:
            lai_v = 2.0  # coeff-ok: LAI=2 fallback (no prescribed/prognostic LAI)
        sai_v  = (land_params.SAI[i] if land_params is not None and land_params.SAI is not None
                  else 0.5)
        htop_patch = htop_patch.at[p].set(htop_v)
        elai_patch = elai_patch.at[p].set(lai_v)
        esai_patch = esai_patch.at[p].set(sai_v)

    # ---- frictionvel ----
    forc_hgt_u_patch = jnp.full(np_, float(land_config.z_ref), dtype=jnp.float64)

    # ---- Build stub namespaces ----
    atm2lnd = SimpleNamespace(
        forc_u_grc=forc_u,
        forc_v_grc=forc_v,
        forc_pco2_grc=forc_pco2,
        forc_po2_grc=forc_po2,
        forc_solad_downscaled_col=forc_solad_col,
        forc_solai_grc=forc_solai_grc,
        forc_t_downscaled_col=forc_t,
        forc_pbot_downscaled_col=forc_pbot,
        forc_lwrad_downscaled_col=forc_lwrad,
    )
    wateratm2lndbulk = SimpleNamespace(
        forc_q_downscaled_col=forc_q,
        forc_rain_downscaled_col=forc_rain,
        forc_snow_downscaled_col=forc_snow,
    )
    surfalb = SimpleNamespace(
        albgrd_col=albgrd_col,
        albgri_col=albgri_col,
    )
    soilstate = SimpleNamespace(
        smp_l_col=smp_l_col,
        hk_l_col=hk_l_col,
        rootfr_patch=rootfr_patch,
        soilresis_col=soilresis_col,
        thk_col=thk_col,
    )
    waterstatebulk = SimpleNamespace(
        h2osoi_ice_col=h2osoi_ice,
    )
    temperature = SimpleNamespace(
        t_a10_patch=t_a10_patch,
        t_soisno_col=t_soisno_col,
    )
    frictionvel = SimpleNamespace(
        forc_hgt_u_patch=forc_hgt_u_patch,
    )
    canopystate = SimpleNamespace(
        htop_patch=htop_patch,
        elai_patch=elai_patch,
        esai_patch=esai_patch,
    )
    # Output receiver stubs (MLCanopyFluxes writes to these only when mlcan_to_clm=1)
    energyflux = SimpleNamespace(
        taux_patch=jnp.zeros(np_),
        tauy_patch=jnp.zeros(np_),
        eflx_lh_tot_patch=jnp.zeros(np_),
        eflx_sh_tot_patch=jnp.zeros(np_),
        eflx_lwrad_out_patch=jnp.zeros(np_),
    )
    waterfluxbulk = SimpleNamespace(
        qflx_evap_tot_patch=jnp.zeros(np_),
    )
    solarabs = SimpleNamespace(
        fsa_patch=jnp.zeros(np_),
    )
    waterdiagnosticbulk = SimpleNamespace(
        q_ref2m_patch=jnp.zeros(np_),
    )

    return dict(
        atm2lnd=atm2lnd,
        wateratm2lndbulk=wateratm2lndbulk,
        surfalb=surfalb,
        soilstate=soilstate,
        waterstatebulk=waterstatebulk,
        temperature=temperature,
        frictionvel=frictionvel,
        canopystate=canopystate,
        energyflux=energyflux,
        waterfluxbulk=waterfluxbulk,
        solarabs=solarabs,
        waterdiagnosticbulk=waterdiagnosticbulk,
    )


def _init_mlcanopy(ncol: int, stubs: dict, canopy_config: CLMMLCanopyConfig) -> Any:
    """Allocate and cold-start a fresh ``mlcanopy_type`` instance.

    Called only on the first legoESM step (``canopy_state.mlcanopy is None``).
    Sets ``htop``, ``hbot``, and beta-distribution shape parameters for the
    vertical PAD profile, then calls ``init_cold`` for leaf water potential
    and intercepted water initialisation.
    """
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import create_mlcanopy, init_cold
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import nlevmlcan
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval

    mlcanopy = create_mlcanopy(1, ncol)

    # Set canopy geometry for each patch.
    ztop = mlcanopy.ztop_canopy
    zbot = mlcanopy.zbot_canopy
    # getPADparameters assigns PFT-default beta params only when pbeta < 0.
    # create_mlcanopy initializes them to spval (1e36), so force them negative
    # to trigger the PFT lookup in getPADparameters.
    pbeta_lai = jnp.full_like(mlcanopy.pbeta_lai_canopy, -1.0)
    pbeta_sai = jnp.full_like(mlcanopy.pbeta_sai_canopy, -1.0)

    canopy = stubs["canopystate"]
    for i in range(ncol):
        p = i + 1
        htop_v = float(canopy.htop_patch[p])
        hbot_v = canopy_config.hbot_frac * htop_v  # hbot_frac from CLMMLCanopyConfig (default 0.1)

        ztop = ztop.at[p].set(htop_v)
        zbot = zbot.at[p].set(hbot_v)

    # Initialize root_biomass_canopy to a physical value.
    # create_mlcanopy initializes it to spval=1e36, which causes SoilResistance
    # to compute rld ≈ 1e37 m/m³ and a soil-root conductance of ~1e33 — the
    # plant resistance then dominates correctly but layer-wise soil ET is wrong.
    # Default: 300 g/m² (temperate deciduous tree, Jackson et al. 1997).
    root_biomass = jnp.full_like(mlcanopy.root_biomass_canopy,
                                  float(canopy_config.root_biomass_default_g_m2))
    # root_biomass_canopy is initialized to spval for p=0; keep that convention
    # and only fill 1-based patch indices.
    for i in range(ncol):
        root_biomass = root_biomass.at[i + 1].set(
            float(canopy_config.root_biomass_default_g_m2)
        )

    mlcanopy = mlcanopy._replace(
        ztop_canopy=ztop,
        zbot_canopy=zbot,
        pbeta_lai_canopy=pbeta_lai,
        pbeta_sai_canopy=pbeta_sai,
        root_biomass_canopy=root_biomass,
    )

    # Call init_cold to set initial leaf water potential and intercepted water.
    # Signature: init_cold(mlcanopy_inst, begp, endp)
    mlcanopy = init_cold(mlcanopy, 1, ncol)

    return mlcanopy


def _extract_clm_ml_sif(
    mlcanopy: Any, ncol: int, sif_cfg: SIFConfig,
) -> jnp.ndarray | None:
    """Top-of-canopy SIF from the CLM-ML per-(layer, leaf) photosynthesis.

    Reads the ``mlcanopy_inst`` per-(patch, layer, leaf) arrays ``je_leaf``
    (the model's native electron-transport rate) and ``apar_leaf`` (absorbed
    PAR per unit leaf area), plus the per-(patch, layer) ``dpai_profile`` /
    ``fracsun_profile`` — the CLM-ml ``MLCanopyFluxesType`` names the
    ``multilayer_canopy`` JAX port mirrors faithfully.  SIF is a **pure
    consumer** of the Bonan model's Farquhar solution: it uses ``je_leaf``
    directly and never re-inverts ``je`` from ``An``/``Ci`` (nor re-derives
    ``Gamma*``).  Sums the per-leaf SIF weighted by leaf area (``dpai *
    fracsun`` sunlit, ``dpai * (1 - fracsun)`` shaded) the SAME way CLM-ML sums
    ``anet_leaf * dpai`` into canopy GPP, via the shared
    :func:`~legoesm.land.canopy.sif.multilayer_canopy_sif` core.

    hasattr-guarded: any missing field returns ``None`` (SIF is opt-in and must
    never crash the flux path if a port version renames a field).  Unfilled
    layers/leaves carry ``spval = 1e36`` and are masked to zero leaf area with a
    sanitised ``je``/``apar`` so they drop out of the sum cleanly.
    """
    required = ("je_leaf", "apar_leaf", "dpai_profile", "fracsun_profile")
    if not all(hasattr(mlcanopy, f) for f in required):
        return None

    # Patch axis is 1-based here: the legoESM interface builds mlcanopy via
    # create_mlcanopy(1, ncol) (begp=1), so column i (0..ncol-1) lives at array
    # index i+1 and index 0 is the unused pad — matching every sibling read in
    # _extract_surface_fluxes.  The layer (1:nlevmlcan) and leaf (1:nleaf) blocks
    # likewise skip index 0.  Leaf order: il=1 sunlit, il=2 shaded.
    def _lv(name):  # (ncol, nlev, nleaf)
        return jnp.stack([getattr(mlcanopy, name)[i + 1, 1:, 1:] for i in range(ncol)])

    def _pr(name):  # (ncol, nlev)
        return jnp.stack([getattr(mlcanopy, name)[i + 1, 1:] for i in range(ncol)])

    je, apar = _lv("je_leaf"), _lv("apar_leaf")
    dpai, fracsun = _pr("dpai_profile"), _pr("fracsun_profile")

    # Per-(layer, leaf) leaf-area weight from the layer PAI and sunlit fraction.
    leaf_area = jnp.stack([dpai * fracsun, dpai * (1.0 - fracsun)], axis=-1)  # (ncol,nlev,nleaf)

    # Mask unfilled / spval elements: zero their leaf area and sanitise je/apar
    # so nothing non-finite reaches the SIF core.
    valid = (
        jnp.isfinite(je) & jnp.isfinite(apar) & jnp.isfinite(leaf_area)
        & (jnp.abs(je) < _SPVAL_GUARD) & (apar < _SPVAL_GUARD) & (jnp.abs(leaf_area) < _SPVAL_GUARD)
    )
    je = jnp.where(valid, je, 0.0)
    apar = jnp.where(valid, jnp.maximum(apar, 0.0), 0.0)
    leaf_area = jnp.where(valid, jnp.maximum(leaf_area, 0.0), 0.0)

    def _flat(a):  # (ncol, nlev*nleaf)
        return a.reshape(a.shape[0], -1)

    return multilayer_canopy_sif(_flat(je), _flat(apar), _flat(leaf_area), sif_cfg)


def _extract_surface_fluxes(
    mlcanopy: Any,
    ncol: int,
    forcing: AtmToSurface,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    canopy_config: CLMMLCanopyConfig | None = None,
) -> SurfaceFluxOutput:
    """Map ``mlcanopy_type`` output fields to a ``SurfaceFluxOutput``.

    Sign conventions
    ----------------
    - ``shflx_canopy``: sensible heat flux [W/m²], positive into atmosphere
    - ``lhflx_canopy``: latent heat flux [W/m²], positive into atmosphere
    - ``gsoi_soil``:    ground heat flux [W/m²], positive into soil (CLM)
    - ``lwup_canopy``:  upwelling LW [W/m²]
    - ``gppveg_canopy``: GPP [µmol CO₂/m²/s]
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import ivis, inir

    # Per-column arrays → (ncol,)
    shflx = jnp.stack([mlcanopy.shflx_canopy[i + 1] for i in range(ncol)])
    lhflx = jnp.stack([mlcanopy.lhflx_canopy[i + 1] for i in range(ncol)])
    G_soil = jnp.stack([mlcanopy.gsoi_soil[i + 1] for i in range(ncol)])
    lw_up = jnp.stack([mlcanopy.lwup_canopy[i + 1] for i in range(ncol)])
    # GPP: µmol CO2/m²/s → gC/m²/s  (standard atomic weight of C = constants.M_C g/mol)
    _M_C_g_per_mol = constants.M_C
    gpp_umol = jnp.stack([mlcanopy.gppveg_canopy[i + 1] for i in range(ncol)])
    gpp = gpp_umol * (_M_C_g_per_mol * 1.0e-6)  # µmol/m²/s → gC/m²/s

    # Friction velocity → momentum flux components
    ustar = jnp.stack([mlcanopy.ustar_canopy[i + 1] for i in range(ncol)])
    rho_a = forcing.rho_lowest  # moist air density, includes virtual-T correction
    tau_total = rho_a * ustar ** 2  # [N/m²]
    # Split tau between x and y proportional to wind components
    u = forcing.u_lowest
    v = forcing.v_lowest
    ws = jnp.sqrt(u ** 2 + v ** 2 + 1.0e-6)
    tau_x = tau_total * u / ws
    tau_y = tau_total * v / ws

    # Net radiation: absorbed SW + net LW
    # rnet_canopy is the full-column net radiation (canopy + soil).
    # swveg_canopy and swsoi_soil are indexed (patch, band) where band
    # indices ivis=1, inir=2; index 0 holds spval and must be skipped.
    rnet = jnp.stack([mlcanopy.rnet_canopy[i + 1] for i in range(ncol)])
    sw_net = jnp.stack(
        [
            mlcanopy.swveg_canopy[i + 1, ivis]
            + mlcanopy.swveg_canopy[i + 1, inir]
            + mlcanopy.swsoi_soil[i + 1, ivis]
            + mlcanopy.swsoi_soil[i + 1, inir]
            for i in range(ncol)
        ]
    )
    lw_net = rnet - sw_net

    # Canopy-mean albedo: 1 - SW_absorbed / SW_down
    sw_down = forcing.sw_down
    albedo = jnp.where(sw_down > 1.0, 1.0 - sw_net / (sw_down + 1.0e-6), 0.15)  # coeff-ok: nighttime/low-light albedo fallback (~0.15 broadband for vegetated surface)

    # Surface temperature: use soil surface T (updated by CLM-ML)
    T_surface = jnp.stack([mlcanopy.tg_soil[i + 1] for i in range(ncol)])

    # q_surface: surface specific humidity [kg/kg].
    # CLM-ML computes rhg_soil = exp(psis*Mw/(R*Tg)) (Philip formula) for the
    # soil surface relative humidity.  Using qsat(Tg) ignores this and
    # overestimates soil evaporation by ~30% at smp=-50000 mm (rhg≈0.70).
    # Use rhg_soil from mlcanopy if available; fall back to qsat only if absent.
    q_sat_surface = saturation_specific_humidity(T_surface, forcing.p_surface)
    if hasattr(mlcanopy, "rhg_soil"):
        rhg = jnp.stack([mlcanopy.rhg_soil[i + 1] for i in range(ncol)])
        # Clamp rhg to [0, 1] — spval=1e36 indicates uninitialised
        rhg = jnp.clip(rhg, 0.0, 1.0)
        q_surface = rhg * q_sat_surface
    else:
        q_surface = q_sat_surface

    # Emissivity and roughness from canopy
    emissivity = jnp.full(ncol, float(land_config.emissivity_land))
    z0 = jnp.stack([mlcanopy.z0m_canopy[i + 1] for i in range(ncol)])

    # Optional solar-induced fluorescence (passive TOC diagnostic).  Static gate
    # on the config leaf; per-(layer,leaf) sum shares the two-leaf/big-leaf core.
    sif = None
    sif_cfg = getattr(canopy_config, "sif", None) if canopy_config is not None else None
    if sif_cfg is not None:
        sif = _extract_clm_ml_sif(mlcanopy, ncol, sif_cfg)

    # Canopy heat storage: needed for energy balance closure in the coupler.
    # Rnet = SH + LH + G_soil + stflx_air + stflx_veg
    # Extract from mlcanopy if the fields exist (they are always computed by
    # MLCanopyFluxes but named differently in old CLM-ML-JAX versions).
    if hasattr(mlcanopy, "stflx_air_canopy"):
        stflx_air = jnp.stack([mlcanopy.stflx_air_canopy[i + 1] for i in range(ncol)])
    else:
        stflx_air = None
    if hasattr(mlcanopy, "stflx_veg_canopy"):
        stflx_veg = jnp.stack([mlcanopy.stflx_veg_canopy[i + 1] for i in range(ncol)])
    else:
        stflx_veg = None

    # T_canopy_air: aerodynamic exchange temperature for SH coupling to atmosphere.
    # taveg_canopy is the PAI-weighted mean within-canopy air temperature — the
    # best available proxy for the canopy exchange node Tc in H = rho*cp*(Tc-Ta)/Ra.
    # (tair_profile is not written back to the output NamedTuple by MLCanopyFluxes.)
    if hasattr(mlcanopy, "taveg_canopy"):
        T_canopy_air = jnp.stack([mlcanopy.taveg_canopy[i + 1] for i in range(ncol)])
        # Guard against cold-start spval=1e36 (no PAI → taveg=0, or uninitialized)
        T_canopy_air = jnp.where(
            (T_canopy_air > 1e30) | (T_canopy_air < 100.0),
            T_surface, T_canopy_air,
        )
    else:
        T_canopy_air = T_surface  # fallback: tg_soil

    # Below-canopy GROUND latent heat [W/m²]: CLM-ML's soil-surface evaporation
    # (``lhsoi_soil``, throttled by the Philip rhg_soil humidity), separate from
    # the leaf transpiration + canopy-water evaporation folded into ``lhflx``.
    # Exposing it lets the multilayer driver route the ground component to
    # snowpack sublimation (L_s) over snow while leaf transpiration (LE_canopy =
    # lhflx − LE_soil) draws soil water at L_v — the same phase-split the two-leaf
    # scheme gets.  ``lhsoi_soil`` is a CORE MLSoilFluxes output (always present
    # in a compatible clm-ml-jax), so read it directly: a missing field raises
    # loudly here rather than silently reverting the driver to the pre-F13 all-
    # L_v-soil routing (which would drain soil water for ground evaporation that
    # should sublimate from the snowpack).
    LE_soil = jnp.stack([mlcanopy.lhsoi_soil[i + 1] for i in range(ncol)])
    LE_canopy = lhflx - LE_soil

    return SurfaceFluxOutput(
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        sw_net=sw_net,
        lw_net=lw_net,
        lw_up=lw_up,
        G_soil=G_soil,
        T_surface=T_surface,
        q_surface=q_surface,
        albedo=albedo,
        emissivity=emissivity,
        z0=z0,
        gpp=gpp,
        sif=sif,
        T_canopy_air=T_canopy_air,
        stomatal_ratio=jnp.ones(ncol),
        stflx_air=stflx_air,
        stflx_veg=stflx_veg,
        LE_canopy=LE_canopy,
        LE_soil=LE_soil,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def extract_clm_ml_grid_info(canopy_state: CanopyState, patch: int | None = None) -> Any:
    """Extract the concrete canopy structure (``ncan``/``ntop``/``nbot``) from a warm state.

    Call this ONCE on a concrete (non-traced) warm-start ``CanopyState`` — e.g.
    the state returned by the first forward step — and thread the result through
    every subsequent traceable step via ``compute_clm_ml_canopy_fluxes(...,
    grid_info=...)``.  This keeps the structural integers CONCRETE even when the
    carried ``canopy_state.mlcanopy`` becomes a tracer in a jitted/differentiated
    rollout (otherwise ``int(tracer)`` raises ``ConcretizationTypeError``).

    ``ncan``/``ntop``/``nbot`` VARY per column (they are functions of canopy
    height / PFT beta-distribution, built per patch in ``initVerticalStructure``),
    so a multi-column run needs one ``GridInfo`` per column.

    Parameters
    ----------
    canopy_state:
        A concrete warm-started state (``canopy_state.mlcanopy`` populated by a
        prior forward step).  Must NOT be a tracer.
    patch:
        Optional 1-based patch index.  When given, extract ONLY that patch and
        return a single ``GridInfo`` (single-site diff / back-compat).  When
        ``None`` (default), auto-detect: a single-column state returns one
        ``GridInfo``; a multi-column state (ncol>1) returns a TUPLE of per-column
        ``GridInfo`` aligned with the 1-based patch order (entry ``c`` -> patch
        ``c+1``), which the traceable path threads into its per-column loop.

    Returns
    -------
    GridInfo | tuple[GridInfo, ...]
        One ``GridInfo`` (single column / explicit ``patch``) or a per-column
        tuple (ncol>1).  ``multilayer_canopy.MLclm_varctl.GridInfo`` fields are
        concrete Python ints.
    """
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import GridInfo
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch as _patch
    if canopy_state is None or canopy_state.mlcanopy is None:
        raise ValueError(
            "extract_clm_ml_grid_info needs a warm-started canopy_state whose "
            "mlcanopy is populated; got None. Run one forward step first."
        )
    m = canopy_state.mlcanopy
    # 1-based patch dim: arrays are (ncol+1,) with index 0 unused (begp=1).
    ncol = int(m.ncan_canopy.shape[0]) - 1
    # PFT per patch (patch.itype installed by _setup_clm_topology at the cold step)
    # — threaded so the physics need no int(patch.itype[p]) under a traced scan p.
    _itype = np.asarray(_patch.itype)

    def _gi(p: int):
        return GridInfo(
            p=int(p),
            ncan=int(m.ncan_canopy[p]),
            ntop=int(m.ntop_canopy[p]),
            nbot=int(m.nbot_canopy[p]),
            pft=int(_itype[p]),
        )

    try:
        if patch is not None:
            return _gi(patch)
        if ncol == 1:
            return _gi(1)
        return tuple(_gi(c + 1) for c in range(ncol))
    except jax.errors.ConcretizationTypeError as exc:  # pragma: no cover - guard
        raise RuntimeError(
            "extract_clm_ml_grid_info must be called on a CONCRETE canopy_state "
            "(outside jax.grad tracing), not on a traced carry."
        ) from exc


def compute_clm_ml_canopy_fluxes(
    T_soil_top: jnp.ndarray,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    canopy_state: CanopyState | None,
    dt: float,
    T_soil: jnp.ndarray | None = None,
    psi_soil: jnp.ndarray | None = None,
    theta_soil: jnp.ndarray | None = None,
    lat: jnp.ndarray | None = None,
    lon: jnp.ndarray | None = None,
    doy: float = 0.0,
    lai_override: jnp.ndarray | None = None,
    vcmaxpft_jax: jnp.ndarray | None = None,
    g1_medlyn_jax: jnp.ndarray | None = None,
    grid_info: Any | None = None,
) -> tuple[SurfaceFluxOutput, CanopyState]:
    """Compute canopy fluxes via the CLM-ML-JAX multilayer canopy model.

    Parameters
    ----------
    T_soil_top:
        Soil surface temperature [K], shape ``(ncol,)``.
    forcing:
        Atmospheric forcing from the coupler.
    canopy_config:
        Static CLM-ML-JAX configuration (``CLMMLCanopyConfig``).
    land_config:
        Parent ``MultiLayerLandConfig`` (for fallback soil parameters).
    land_params:
        Per-column ``LandSurfaceParams`` (or ``None`` for config defaults).
        Must supply ``LAI``, ``SAI``, ``htop`` for the canopy scheme.

        Note: CLM-ML derives its own within-canopy wind (from ``forcing.u_lowest``/
        ``v_lowest``) and hydraulic water-stress (btran, from ``psi_soil`` via
        ``smp_l_col``), so the driver's ``w_frac_rz`` / ``wind_speed`` diagnostics
        are NOT inputs here and are intentionally not accepted.
    canopy_state:
        Previous-step canopy state.  ``None`` or ``mlcanopy is None``
        triggers cold-start allocation.
    dt:
        Timestep [s].
    T_soil:
        Full soil temperature profile [K], shape ``(ncol, n_layers)``.
    psi_soil:
        Soil matric potential [m], shape ``(ncol, n_layers)``.
    theta_soil:
        Volumetric water content [m³/m³], shape ``(ncol, n_layers)``.
    lat:
        Latitude [degrees], shape ``(ncol,)`` or ``None``.
    lon:
        Longitude [degrees], shape ``(ncol,)`` or ``None``.
        Used for solar zenith angle and SW partitioning.  Pass ``None``
        to default to 0° (Greenwich); this is incorrect for most sites.
    doy:
        Day of year (0-based float, e.g. 120.0 = May 1 in a non-leap year).
    vcmaxpft_jax:
        Optional trainable per-PFT Vcmax25 override [µmol/m²/s], shape
        ``(mxpft+1,)`` — replaces the module-global ``MLpftcon.vcmaxpft`` lookup
        so ``jax.grad`` can flow into Vcmax25.  Injected as a TRACED leaf from
        the loss (SegmentForcing doctrine); ``None`` keeps the PFT default.
    g1_medlyn_jax:
        Optional trainable per-PFT Medlyn ``g1`` override [kPa^0.5], same shape
        contract as ``vcmaxpft_jax``.  Only active when the canopy stomatal model
        is Medlyn (``MLclm_varctl.gs_type == 0``); inert under the default WUE
        conductance (``gs_type == 2``).  ``None`` keeps the PFT default.
    grid_info:
        Optional concrete ``GridInfo(p, ncan, ntop, nbot)`` structural constants
        for the differentiable path.  REQUIRED for a MULTI-STEP differentiated
        rollout: when a ``jax.grad`` loss unrolls ≥2 canopy steps and carries the
        returned ``CanopyState`` forward, ``canopy_state.mlcanopy`` is a tracer, so
        the structural ints cannot be read from it — extract them once from the
        (concrete) warm-start state via :func:`extract_clm_ml_grid_info` and pass
        the same object each step.  ``None`` (single warm step whose state is a
        captured constant) reads the ints off the concrete template.

    Returns
    -------
    surface_out : SurfaceFluxOutput
        Canopy surface energy balance fluxes for the legoESM coupler.
    new_canopy_state : CanopyState
        Updated prognostic state to carry forward to the next step.
    """
    # ---- Fail-early on the static config, BEFORE any CLM state is touched ----
    # Both of these raise on a bad static value.  They run at function entry
    # rather than next to their point of use so an invalid config aborts before
    # _ensure_clm_initialized / _setup_clm_time have mutated process-global CLM
    # module state, leaving it consistent for the caller's next attempt.
    canopy_config.validate()
    n_ml_steps = resolve_num_ml_steps(canopy_config, dt)

    # ---- Phase-1 CLM initialization (once) ----
    _ensure_clm_initialized()

    # Lazy imports of CLM-ML-JAX entry point
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type
    from legoesm.land.soil_grid import make_soil_grid

    ncol = T_soil_top.shape[0]

    # ---- Differentiable-mode gate (static, resolved here — never traced) ----
    # ``differentiable=True`` opts a training run into the JAX-native diff path
    # (``grid=`` + ``lax.scan``).  It requires (a) a WARM canopy_state whose
    # ``mlcanopy`` already carries the vertical structure (ncan/ntop/nbot) — the
    # first cold-start step always runs forward-only to build it — and (b) a
    # single column: the diff path reads one concrete ``(ncan, ntop, nbot)``.
    # A multi-column diff request is a hard error (rather than silently degrading
    # to the forward path), matching the dispatch-hardening rule.  NOTE: this is
    # NOT vmap-able — the interface runs host-side setup that mutates CLM module
    # globals (topology singletons ``patch``/``col``/``grc``, orbital params,
    # ``_last_topology_key``), which ``jax.vmap`` cannot batch.  Multi-column
    # training must loop columns OUTSIDE ``jax.grad`` and accumulate per-column
    # gradients.
    _want_diff = bool(getattr(canopy_config, "differentiable", False))
    if _want_diff and ncol != 1:
        raise ValueError(
            "CLMMLCanopyConfig.differentiable=True is single-column only "
            f"(ncol == 1); got ncol={ncol}. The CLM-ML interface runs host-side "
            "setup that mutates CLM module globals (topology, orbital params), so "
            "it cannot be jax.vmap-ed over columns. For multi-column training, "
            "differentiate one column at a time (loop columns outside jax.grad "
            "and accumulate per-column gradients)."
        )
    _warm_started = canopy_state is not None and canopy_state.mlcanopy is not None
    _diff_mode = _want_diff and _warm_started  # ncol == 1 guaranteed above

    # ---- Traceable (jax.jit) forward path -------------------------------------
    # A warm-started step whose caller threads a concrete ``grid_info`` opts into
    # the fully device-native path (jnp forcing, ``grid=``, ``cos_zenith_device=``)
    # even WITHOUT jax.grad — this is what a jax.jit scan-over-time uses to keep the
    # whole segment on device (SegmentForcing / warm-start doctrine).  ``_diff_mode``
    # stays the stricter grad-only subset (it additionally runs the AD-capability
    # and geometry-tracer guards) and is SINGLE-COLUMN only (multi-column training
    # would loop columns outside jax.grad).  Both share the de-hosted per-step
    # machinery; the cold-start (first) step still runs the eager host build below.
    #
    # S2 (multi-column traceable jit-forward): canopy columns are PHYSICALLY
    # INDEPENDENT (no horizontal coupling — each patch is a standalone 1-D canopy),
    # so a warm ncol>1 step threads ONE GridInfo per column (a length-ncol tuple
    # from ``extract_clm_ml_grid_info``) and the call site below loops the columns,
    # invoking the proven single-patch kernel once per column with ``filter=[c+1]``,
    # ``grid=gi[c]`` and a per-column ``cos_zenith_device``.  ncan/ntop/nbot VARY
    # per column, which is exactly why the per-column GridInfo tuple is required.
    # NOTE: GridInfo is a NamedTuple (hence a tuple), so distinguish a SINGLE
    # GridInfo from a TUPLE-of-GridInfos by the ``ncan`` attribute (present on a
    # GridInfo, absent on the outer tuple/list) — an isinstance(tuple) check would
    # wrongly unpack a lone GridInfo into its four int fields.
    _gi_list = None
    if grid_info is not None:
        _gi_list = [grid_info] if hasattr(grid_info, "ncan") else list(grid_info)
    _percolumn_ok = _gi_list is not None and len(_gi_list) == ncol
    # jax.grad training stays single-column; jit-forward supports ncol>=1 given a
    # per-column GridInfo.
    _traceable = (
        (_diff_mode and ncol == 1)
        or (_warm_started and grid_info is not None and (ncol == 1 or _percolumn_ok))
    )
    _traceable_multi = _traceable and ncol > 1  # the per-column loop path

    # ---- Multi-column-under-trace backstop -----------------------------------
    # ncol>1 traceable requires a per-column GridInfo tuple (the structural ints
    # vary per column).  Without it, ncol>1 falls to the EAGER host path below,
    # whose ``np.array(forcing.*)`` marshalling raises a cryptic
    # TracerArrayConversionError on traced forcing deep in the backend.  Detect
    # that case — ncol>1, NOT traceable, with ANY traced forcing/soil leaf — and
    # fail with actionable guidance.  This is the single chokepoint every driver
    # path funnels through, covering the object-config drivers (coupled_esm_driver
    # / earth_system_driver) that pass a MultiLayerLandConfig(surface_scheme=
    # CLMMLCanopyConfig()) directly.  EAGER multi-column forward (concrete arrays,
    # offline global) stays valid — only a TRACED ncol>1 WITHOUT a per-column
    # GridInfo is rejected.
    if ncol != 1 and not _traceable:
        _traced_ncol_leaf = next(
            (
                _n
                for _n, _v in (
                    [(f"forcing.{_f}", getattr(forcing, _f)) for _f in forcing._fields]
                    + [("T_soil_top", T_soil_top), ("T_soil", T_soil),
                       ("psi_soil", psi_soil), ("theta_soil", theta_soil)]
                )
                if _v is not None and isinstance(_v, jax.core.Tracer)
            ),
            None,
        )
        if _traced_ncol_leaf is not None:
            raise NotImplementedError(
                "CLM-ML multilayer canopy under jax.jit needs a per-column GridInfo "
                f"for ncol>1; got ncol={ncol} with a traced leaf "
                f"({_traced_ncol_leaf}) but grid_info="
                f"{'None' if grid_info is None else f'len={len(_gi_list)}'} "
                "(expected a length-ncol tuple from extract_clm_ml_grid_info on the "
                "warm-start state).  Warm-start one forward step, extract the "
                "per-column structure, and thread it via grid_info=.  (jax.grad "
                "training through the canopy remains single-column.)"
            )

    # Solar geometry (lat / lon / doy / cos_zenith) is a NON-differentiated
    # static input BY DESIGN: it is consumed by host-side CLM orbital setup
    # (_setup_clm_time / _setup_clm_topology / shr_orb_cosz), not by the traced
    # canopy physics.  You do not train through the Sun's position — it is fixed
    # by lat/lon/time.  If a caller differentiates the WHOLE AtmToSurface pytree
    # (so cos_zenith becomes a tracer), the host ``np.array(forcing.cos_zenith)``
    # below would raise a cryptic TracerArrayConversionError.  Detect it here and
    # fail with an actionable message instead.  (Differentiating the physical
    # forcing leaves — T_lowest, sw_down, q, u, v, lw_down, p, co2 — is fully
    # supported; only geometry must stay concrete.)
    # lat/lon feed HOST-side CLM topology/orbital setup (np.array(lat),
    # _setup_clm_topology) that runs at trace time on ANY traceable step (diff OR
    # jit-forward), so a TRACED lat/lon (e.g. jax.jit over lat) would raise a cryptic
    # TracerArrayConversionError.  Guard them on every traceable step.  (dt is
    # already forced concrete upstream by resolve_num_ml_steps' math.ceil.)
    # cos_zenith is NOT checked here: on the traceable path it is a DEVICE input
    # (cos_zenith_device, jnp.asarray + stop_gradient), so it MAY be a tracer — the
    # normal case when forcing is built inside the jitted step.  Codex round-2 HIGH.
    if _want_diff or _traceable:
        for _geo_name, _geo_val in (("lat", lat), ("lon", lon)):
            if _geo_val is not None and isinstance(_geo_val, jax.core.Tracer):
                raise ValueError(
                    f"CLM-ML traceable/diff path: {_geo_name} is a jax tracer, but "
                    "it feeds host-side CLM topology/orbital setup and must be a "
                    "NON-differentiated STATIC input. Keep geometry concrete (close "
                    "over it, or jax.lax.stop_gradient it before the jit/grad boundary); "
                    "vary only the physical forcing leaves and doy per step."
                )
    # Diff-TRAINING contract: additionally keep cos_zenith concrete so a grad over the
    # whole AtmToSurface pytree does not try to train through the Sun's position.
    # (The jit-forward traceable path handles a traced cos_zenith via stop_gradient.)
    if _want_diff and forcing.cos_zenith is not None and isinstance(
            forcing.cos_zenith, jax.core.Tracer):
        raise ValueError(
            "CLM-ML diff mode: forcing.cos_zenith is a jax tracer, but solar geometry "
            "is a NON-differentiated static input. Differentiate only the physical "
            "forcing leaves (T_lowest, sw_down, q, u, v, lw_down, p, co2); close over "
            "or stop_gradient cos_zenith before the grad boundary."
        )

    # Cold-start-under-grad guard.  ``differentiable=True`` needs a WARM state:
    # the first (cold) step builds the canopy vertical structure via host-side
    # Python/NumPy (``_init_mlcanopy`` + ``np.array(forcing.*)`` marshalling) and
    # cannot run on the jax.grad tape.  When ``_want_diff`` but the state is cold,
    # ``_diff_mode`` falls back to the forward path, whose ``np.array(forcing.
    # T_lowest)`` (t_a10 running mean) would raise a cryptic
    # TracerArrayConversionError under grad.  Detect a traced PHYSICAL forcing
    # leaf here and give the actionable warm-start-first guidance instead
    # (mirrors the multi-column / geometry guards; the "no cryptic degrade" bar).
    if _want_diff and not _warm_started:
        # Scan EVERY differentiable input so a cold-start grad w.r.t. ANY of them
        # fails loudly here rather than slipping into the forward init path and
        # raising a cryptic tracer/concretization error downstream.  Cover ALL
        # forcing leaves GENERICALLY (via ``_fields`` — a hand-maintained subset
        # kept missing leaves that ``_build_stubs`` consumes, e.g. precip_total/
        # precip_snow → forc_rain/forc_snow), plus the soil state, prognostic LAI
        # and the trainable Vcmax25/g1 arrays.  Geometry (cos_zenith/lat/lon) is
        # already handled by the guard above (it fires first if traced).
        _named_inputs = [
            (f"forcing.{_field}", getattr(forcing, _field))
            for _field in forcing._fields
        ] + [
            ("T_soil_top", T_soil_top),
            ("T_soil", T_soil),
            ("psi_soil", psi_soil),
            ("theta_soil", theta_soil),
            ("lai_override", lai_override),
            ("vcmaxpft_jax", vcmaxpft_jax),
            ("g1_medlyn_jax", g1_medlyn_jax),
        ]
        _traced_leaf = next(
            (
                _name
                for _name, _val in _named_inputs
                if _val is not None and isinstance(_val, jax.core.Tracer)
            ),
            None,
        )
        if _traced_leaf is not None:
            raise ValueError(
                "CLMMLCanopyConfig.differentiable=True needs a WARM-started "
                f"canopy_state, but {_traced_leaf} is a jax tracer on a COLD "
                "start (canopy_state is None / mlcanopy not built). The first "
                "cold step builds the canopy vertical structure via host-side "
                "Python/NumPy and cannot run on the jax.grad tape. Run ONE forward "
                "(cold) step OUTSIDE the grad region to warm-start the state, then "
                "differentiate subsequent steps — thread the concrete structure "
                "via grid_info=extract_clm_ml_grid_info(state0)."
            )

    # ---- Build soil grid data ----
    # The soil grid is STATIC structure (a pure function of the static soil_grid
    # config).  Under jax.jit, make_soil_grid's jnp ops would otherwise yield
    # tracers that the host-side structural arithmetic downstream (z_centers +
    # root-fraction profile in _build_stubs, the topology cache key) cannot
    # float()/np.array()/tobytes().  ensure_compile_time_eval evaluates this
    # constant subgraph at trace time, so dz/z stay concrete numpy in EVERY mode
    # (eager, jax.grad and jax.jit) — no per-mode branch needed.
    with jax.ensure_compile_time_eval():
        grid = make_soil_grid(land_config.soil_grid)
        dz_soil = np.array(grid.dz, dtype=np.float64)     # (n_layers,)
        z_soil = np.array(grid.z_node, dtype=np.float64)   # (n_layers,)
    n_layers = int(dz_soil.shape[0])

    global _last_topology_key
    if _traceable:
        # ---- Warm traceable step (jax.jit / global path) ----------------------
        # Removes the two per-step HOST dependencies that break a jax.jit trace:
        #  (1) solar zenith — threaded to the kernel as a DEVICE array
        #      (``cos_zenith_device``) instead of the host shr_orb_cosz recompute;
        #  (2) per-step CLM time globals — at met_type==0 the backend's calendar
        #      interpolation days are all 0 (curr_calday unused; verified in
        #      MLCanopyFluxesMod's time block), so nothing per-step reads them.
        # It STILL re-installs THIS (config, grid)'s CLM topology + step size every
        # trace (concrete host writes, executed once at trace time — cheap): the
        # kernel reads process-global ``col.z/zi``, ``patch.itype`` (pft) and
        # ``get_step_size()``, but ``grid_info`` carries only (p,ncan,ntop,nbot).
        # A DIFFERENT config's cold-start left in those globals would otherwise be
        # traced against this warm state (silent wrong structure/params — codex
        # CRITICAL), and a different caller's ``dt`` would corrupt ``num_ml_steps``.
        if int(canopy_config.met_type) != 0:
            raise ValueError(
                "CLM-ML traceable/jit path supports met_type==0 (external coupler "
                f"forcing) only; got met_type={canopy_config.met_type}. Other "
                "met_types interpolate CLM forcing on curr_calday, a per-step CLM "
                "time global that cannot be written from a traced doy.")
        if lon is not None:
            raise ValueError(
                "CLM-ML traceable/jit path requires lon=None: it takes the solar "
                "zenith straight from forcing.cos_zenith (device).  The lon-supplied "
                "path instead computes zenith from lat/lon/doy and IGNORES "
                "forcing.cos_zenith, so allowing lon here would silently diverge "
                "from the eager result. Pass cos_zenith via forcing, leave lon=None.")
        # Solar zenith is GEOMETRY (non-differentiated by contract): stop_gradient
        # so a jax.grad over forcing.cos_zenith cannot leak a gradient through
        # arccos into the radiation.  (The differentiable=True path already rejects
        # a traced geometry leaf; this also covers the differentiable=False
        # traceable path.)  (ncol,), aligned with the 1-based filter built below.
        cos_zen = jnp.maximum(
            jax.lax.stop_gradient(jnp.asarray(forcing.cos_zenith, dtype=jnp.float64)),
            0.0)
        # Re-install topology + step size for THIS (config, grid).  lat is geometry
        # (concrete); lon is None here, so grc.londeg is irrelevant to the
        # device-zenith path (structure only).  Soil grid is concrete (built under
        # ensure_compile_time_eval above).  The _setup_* helpers write CLM module
        # globals (col.snl, col.z/zi, patch.itype, grc.*, the step size) via jnp, so
        # they must run under ensure_compile_time_eval too: otherwise those writes
        # would be TRACERS under jax.jit and the backend's np.asarray(col.snl) etc.
        # would raise.  Every input is concrete, so this is a pure trace-time setup.
        _lat_deg = (np.array(lat, dtype=np.float64) if lat is not None
                    else np.zeros(ncol, dtype=np.float64))
        _lon_deg = np.zeros(ncol, dtype=np.float64)
        with jax.ensure_compile_time_eval():
            _setup_clm_time(dt, 0.0, 0)  # concrete: get_step_size()==dt; calday unused
            _setup_clm_topology(ncol, _lat_deg, _lon_deg, dz_soil, z_soil,
                                float(land_config.z_ref),
                                pft_clm=int(canopy_config.pft_clm))
        # Invalidate the eager topology cache: this traceable step overwrote the
        # process-global topology (with lon_deg=0), so a LATER eager call whose
        # cached _last_topology_key still matches would skip _setup_clm_topology and
        # run against these (device-path) globals.  Forcing None makes the next
        # eager call re-install its own topology (codex round-2 HIGH).
        _last_topology_key = None
    else:
        # ---- Latitude / longitude arrays ----
        if lat is not None:
            lat_deg = np.array(lat, dtype=np.float64)
        else:
            lat_deg = np.zeros(ncol, dtype=np.float64)

        # ---- CLM global state setup ----
        # _setup_clm_time must precede _compute_virtual_lon_deg because the latter
        # calls shr_orb_decl with orbital params set by _ensure_clm_initialized and
        # needs caldaym1 derived from the itim we are about to write.
        z_ref = float(land_config.z_ref)
        _setup_clm_time(dt, doy, 0)
        # caldaym1 matches what _MLCanopyForcing computes internally:
        #   get_curr_calday(offset=-int(dtime_clm)) = 1 + (itim-1) * dt / 86400
        # This is distinct from doy%1 by exactly one CLM step (dt/86400 days).
        _itim = max(1, round(doy * 86400.0 / max(dt, 1.0)))
        _caldaym1 = 1.0 + (_itim - 1) * dt / 86400.0

        if lon is not None:
            lon_deg = np.array(lon, dtype=np.float64)
            # Use CLM's own Kepler shr_orb_cosz at caldaym1 — same declination and
            # phase as CLM's internal solar_zen_forcing, so SW partitioning is
            # consistent with beam extinction kb=0.5/coszen in MLCanopyFluxes.
            from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import shr_orb_cosz as _clm_cosz, shr_orb_decl as _clm_decl
            import legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb as _varorb_loc
            _declinm1, _ = _clm_decl(_caldaym1, _varorb_loc.eccen, _varorb_loc.mvelpp,
                                      _varorb_loc.lambm0, _varorb_loc.obliqr)
            _pi = np.pi
            cos_zen = np.maximum(np.array([
                float(_clm_cosz(_caldaym1,
                                float(lat_deg[i]) * _pi / 180.0,
                                float(lon_deg[i]) * _pi / 180.0,
                                float(_declinm1)))
                for i in range(ncol)
            ], dtype=np.float64), 0.0)
        else:
            # Use coupler-provided cos_zenith directly — avoids 100× error in beam
            # extinction (kb=0.5/coszen) that occurs when lon defaults to 0° (Greenwich).
            # Then invert the CLM shr_orb_cosz formula (using CLM's own Kepler declination
            # and caldaym1) so that CLM's internal solar_zen_forcing[p] reproduces the
            # forcing.  Both arccos branches give the same cosz, so the virtual longitude
            # is numerically correct even if not geographically meaningful.
            cos_zen = np.maximum(np.array(forcing.cos_zenith, dtype=np.float64), 0.0)
            lon_deg = _compute_virtual_lon_deg(cos_zen, lat_deg, _caldaym1)

        # Topology cache key: capture EVERY input _setup_clm_topology consumes, over
        # ALL columns — not just column 0.  Sampling only lat_deg[0]/lon_deg[0] (the
        # prior key) reused stale per-column topology when two grids shared ncol and
        # a column-0 coordinate but differed in interior columns (silent wrong lat/
        # lon → wrong solar zenith for those columns).  ``tobytes()`` gives an exact,
        # cheap, hashable signature of the full arrays; include the soil grid, z_ref
        # and pft so a config change also forces a rebuild.  (In the virtual-longitude
        # path lon_deg is re-derived each step and changes, so this still rebuilds
        # per step there — same as before; no new cost.)
        _topo_key = (
            ncol,
            lat_deg.tobytes(), lon_deg.tobytes(),
            dz_soil.tobytes(), z_soil.tobytes(),
            float(z_ref), int(canopy_config.pft_clm),
        )
        if _topo_key != _last_topology_key:
            _setup_clm_topology(ncol, lat_deg, lon_deg, dz_soil, z_soil, z_ref,
                                pft_clm=int(canopy_config.pft_clm))
            _last_topology_key = _topo_key

    # ---- Propagate 10-day running mean temperature for Vcmax acclimation ----
    # MLCanopyFluxes copies t_a10_patch into tacclim_forcing on output — reading
    # it back would apply the filter twice per step (effective ~20d, not 10d).
    # Instead we carry the running mean explicitly in CanopyState.t_a10_arr and
    # update it here:
    #   T_a10_new = (1 - alpha) * T_a10_old + alpha * T_lowest
    #   alpha = dt / (10 * 86400)  (10-day e-folding, CLM default)
    alpha = min(dt / (10.0 * 86400.0), 1.0)
    if _traceable:
        # Traced running mean: t_a10 is the Vcmax temperature-acclimation state,
        # a real function of T_lowest, so keep it on device (a host
        # np.array(forcing.T_lowest) would raise on a tracer — scope item A).
        T_low = jnp.asarray(forcing.T_lowest, dtype=jnp.float64)
        if canopy_state is not None and canopy_state.t_a10_arr is not None:
            t_a10_prev = jnp.asarray(canopy_state.t_a10_arr, dtype=jnp.float64)
            t_a10_now = (1.0 - alpha) * t_a10_prev + alpha * T_low
        else:
            t_a10_now = T_low
        t_a10_prior = t_a10_now
    else:
        T_lowest_np = np.array(forcing.T_lowest, dtype=np.float64)
        if canopy_state is not None and canopy_state.t_a10_arr is not None:
            t_a10_prev = np.array(canopy_state.t_a10_arr, dtype=np.float64)
            t_a10_now = ((1.0 - alpha) * t_a10_prev + alpha * T_lowest_np).astype(np.float64)
        else:
            # Cold start: initialize to instantaneous T (will converge in ~10 days)
            t_a10_now = T_lowest_np.copy()
        t_a10_prior = jnp.array(t_a10_now)

    # ---- Build stub CLM instances ----
    soil_hyd = land_config.hydraulics
    stubs = _build_stubs(
        ncol, forcing, canopy_config, land_config, land_params,
        T_soil_top, psi_soil, theta_soil, dz_soil, soil_hyd,
        cos_zen=cos_zen,
        T_soil_all=T_soil,
        t_a10_prior=t_a10_prior,
        lai_override=lai_override,
    )

    # ---- Install the canopy layer counts ----
    # Must precede _init_mlcanopy: the cold-start call builds the vertical
    # structure and reads these process-global counts.
    _apply_canopy_layering(canopy_config)

    # ---- Allocate / retrieve mlcanopy_type ----
    if canopy_state is None or canopy_state.mlcanopy is None:
        mlcanopy = _init_mlcanopy(ncol, stubs, canopy_config)
    else:
        mlcanopy = canopy_state.mlcanopy

    # ---- Decomposition bounds ----
    bounds = bounds_type(begg=1, endg=ncol, begl=1, endl=ncol,
                         begc=1, endc=ncol, begp=1, endp=ncol)

    # ---- filter_exposedvegp: all columns (1-based) ----
    filter_exposedvegp = list(range(1, ncol + 1))
    num_exposedvegp = ncol

    # ---- Set MLclm_varctl global settings ----
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl as _ml_ctl
    _ml_ctl.runge_kutta_type = canopy_config.runge_kutta_type
    _ml_ctl.met_type = canopy_config.met_type
    # dtime_ml: sub-step length. Must divide dt evenly.  Count resolved (and
    # validated) at function entry above.
    _ml_ctl.dtime_ml = dt / n_ml_steps
    _ml_ctl.mlcan_to_clm = 0  # we read output directly from mlcanopy_type
    # DIFFERENTIABLE_MODE is a static intent flag (currently vestigial upstream —
    # every physics module switches on the ``grid`` argument, not this global —
    # but set it to match the mode so any future read is consistent).
    _ml_ctl.DIFFERENTIABLE_MODE = bool(_diff_mode)

    # ---- Select the canopy-airspace turbulence scheme ----
    # Eager path: (re-)apply + VERIFY every step because the psihat tables are
    # process-global CLM state shared by every column and config, so a one-shot
    # mutation would leak the first caller's scheme.  Traceable path: the tables
    # were already applied + verified concretely by the eager cold-start step that
    # built this warm state; the host-side float() probe cannot run inside a jax.jit
    # trace, so only ASSERT the scheme matches (and is not locked to another) — no
    # re-apply / re-probe.  The trace-time lock is committed after the step below.
    if _traceable:
        _assert_turbulence_scheme_for_trace(canopy_config.turbulence_scheme)
    else:
        _apply_turbulence_scheme(
            canopy_config.turbulence_scheme, differentiable=False)

    # ---- Select the leaf stomatal-conductance model (gs_type) ----
    # Process-global like the turbulence tables; "medlyn" activates the traced
    # vcmaxpft_jax injection path below.
    _apply_stomatal_model(canopy_config)

    # ---- Build GridInfo for the traceable (jax.jit / jax.grad) path ----
    # Structural ints must be concrete Python ints extracted BEFORE tracing (int()
    # on a tracer raises ConcretizationTypeError).  The warm-start template
    # ``mlcanopy`` is a captured constant under jax.grad, so these reads are
    # concrete; under a jax.jit scan the carried ``mlcanopy`` is a tracer, so the
    # caller MUST thread ``grid_info`` (extract once from the warm state).  Mirrors
    # make_clm_ml_forward (MLCanopyFluxesMod.py:2098).
    if _traceable:
        from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import GridInfo
        # Capability guard: the differentiable path is only CORRECT with a
        # clm-ml-jax build whose ``_CanopyFluxesDiagnostics`` runs in diff mode
        # (``grid=`` parameter).  Older builds return from ``MLCanopyFluxes``
        # BEFORE diagnostics in diff mode, leaving the canopy-integrated outputs
        # (shflx/lhflx/gpp/rnet/…) stale — a silent wrong-answer.  clm-ml-jax is
        # not on PyPI (local install), so we cannot pin a version; probe the
        # capability directly and fail LOUDLY instead.
        import inspect as _inspect
        from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyFluxesMod as _mlmod
        if "grid" not in _inspect.signature(_mlmod._CanopyFluxesDiagnostics).parameters:
            raise RuntimeError(
                "CLM-ML traceable mode requires a clm-ml-jax build "
                "whose _CanopyFluxesDiagnostics accepts grid= (runs canopy-flux "
                "diagnostics on the jax.grad/jax.jit tape). The installed clm-ml-jax "
                "returns before diagnostics in differentiable mode, which would leave "
                "shflx/lhflx/gpp/rnet stale. Update clm-ml-jax to a revision including "
                "the differentiable-diagnostics fix (adds grid= to _CanopyFluxesDiagnostics)."
            )
        # The traceable path also threads a DEVICE solar zenith (cos_zenith_device)
        # into MLCanopyFluxes so the forcing step needs no per-step host orbital
        # recompute (the last per-step host op).  A build with grid= but without
        # cos_zenith_device would TypeError cryptically on the call below; fail with
        # actionable guidance instead (matches the "no cryptic degrade" bar).
        if "cos_zenith_device" not in _inspect.signature(_mlmod.MLCanopyFluxes).parameters:
            raise RuntimeError(
                "CLM-ML traceable mode (jax.jit / global path) requires a clm-ml-jax "
                "build whose MLCanopyFluxes accepts cos_zenith_device= (device solar "
                "zenith, so solar_zen_forcing is set WITHOUT the host shr_orb_cosz "
                "recompute that breaks a jax.jit trace). Update clm-ml-jax to a "
                "revision that adds cos_zenith_device= to MLCanopyFluxes / _GetCLMVar."
            )
        # ``dpai_profile.shape`` is static, so the range check works either way.
        _ncan_max = int(mlcanopy.dpai_profile.shape[1])
        if _traceable_multi:
            # Multi-column (S2): one validated GridInfo per column, mapped to the
            # 1-based patch position (entry c -> patch c+1) so a tuple built in any
            # order is realigned here.  ncan/ntop/nbot VARY per column.
            if vcmaxpft_jax is not None or g1_medlyn_jax is not None:
                raise NotImplementedError(
                    "CLM-ML multi-column (ncol>1) traceable forward does not support "
                    "the trainable-param overrides (vcmaxpft_jax / g1_medlyn_jax): "
                    "those belong to the single-column jax.grad training path. "
                    "Differentiate one column at a time."
                )
            # Realign by each GridInfo's OWN patch index (``.p``), NOT tuple
            # position: the loop below drives column c -> patch c+1, so a tuple
            # built in any order must still supply patch (c+1)'s structure.  Keying
            # off position and overwriting p would silently apply another column's
            # ncan/ntop/nbot to the wrong patch (codex).  Require exactly patches
            # 1..ncol, one per column.
            _by_p = {int(_g.p): _g for _g in _gi_list}
            if set(_by_p) != set(range(1, ncol + 1)):
                raise ValueError(
                    f"CLM-ML multi-column grid_info covers patches {sorted(_by_p)}, "
                    f"but exactly 1..{ncol} are required (one GridInfo per column, "
                    "patch index c+1). Build it with extract_clm_ml_grid_info(warm_state)."
                )
            _grids = []
            for _c in range(ncol):
                _g = _by_p[_c + 1]
                _ncan_c = int(_g.ncan)
                if not (1 <= _ncan_c <= _ncan_max):
                    raise ValueError(
                        f"CLM-ML traceable multi-column: column {_c} (patch {_c + 1}) "
                        f"has ncan={_ncan_c} (valid 1..{_ncan_max}) — the per-column "
                        "grid_info is not a warm-started structure. Extract it from a "
                        "warm forward step via extract_clm_ml_grid_info(state0)."
                    )
                _grids.append(GridInfo(p=_c + 1, ncan=_ncan_c,
                                       ntop=int(_g.ntop), nbot=int(_g.nbot),
                                       pft=int(_g.pft)))
        else:
            # Single column.  Structural ints (ncan/ntop/nbot) must be CONCRETE
            # Python ints — the diff path reads them at trace time.  Two sources:
            #  (1) caller-supplied ``grid_info`` (REQUIRED for a multi-step
            #      differentiated rollout: the carried ``mlcanopy`` is a tracer,
            #      so reading ints off it would raise); or
            #  (2) the warm template ``mlcanopy`` when it is concrete.
            _p = int(filter_exposedvegp[0])
            if grid_info is not None:
                _g0 = _gi_list[0]
                _ncan_p = int(_g0.ncan)
                _ntop_p = int(_g0.ntop)
                _nbot_p = int(_g0.nbot)
                _pft_p = int(_g0.pft)
            else:
                from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import (
                    patch as _patch_s)
                _pft_p = int(np.asarray(_patch_s.itype)[_p])
                try:
                    _ncan_p = int(mlcanopy.ncan_canopy[_p])
                    _ntop_p = int(mlcanopy.ntop_canopy[_p])
                    _nbot_p = int(mlcanopy.nbot_canopy[_p])
                except jax.errors.ConcretizationTypeError as exc:
                    raise RuntimeError(
                        "CLM-ML diff mode: the canopy_state.mlcanopy structural ints "
                        "(ncan/ntop/nbot) are TRACED — this happens when a differentiated "
                        "loss unrolls MULTIPLE canopy steps and carries the returned state "
                        "as the jax.grad tape's carry. Extract the concrete structural "
                        "ints once from the warm-start state and thread them through every "
                        "step via grid_info=extract_clm_ml_grid_info(state0)."
                    ) from exc
            if not (1 <= _ncan_p <= _ncan_max):
                raise ValueError(
                    "CLM-ML diff mode needs a warm-started canopy_state whose vertical "
                    f"structure is initialised; got ncan={_ncan_p} (valid 1..{_ncan_max}). "
                    "Run one forward (differentiable=False or cold-start) step first."
                )
            _grids = [GridInfo(p=_p, ncan=_ncan_p, ntop=_ntop_p, nbot=_nbot_p,
                               pft=_pft_p)]
    else:
        _grids = None

    # ---- Call MLCanopyFluxes ----
    # Only forward the diff-mode / trainable-param kwargs when they are actually
    # used, so the DEFAULT forward-only path keeps the exact call signature the
    # interface has always used (``_o2ref_py`` only) and does NOT depend on the
    # newer ``grid=``/``vcmaxpft_jax=``/``g1_MED_jax=`` upstream API surface
    # (Codex P1: an older clm-ml-jax would otherwise TypeError on every call,
    # including production forward-only runs).  Diff mode is already gated by the
    # capability guard above.
    # Per-site Vcmax25 override (config) — the "beyond the global PFT table"
    # value.  Build ``vcmaxpft_jax`` from the module lookup with this PFT's entry
    # replaced by the site value; the backend's nitrogen-profile routine
    # (``CanopyNitrogenProfile``) selects the supplied array over the global
    # table BEFORE leaf photosynthesis, regardless of stomatal model, so this
    # feeds photosynthesis under WUE, Medlyn and Ball-Berry alike.  A per-site
    # FIXED value goes here; a TRAINABLE Vcmax25 is supplied as the traced
    # ``vcmaxpft_jax`` argument (from a training loop), which takes precedence.
    if canopy_config.vcmax25_override is not None and vcmaxpft_jax is None:
        from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLpftconMod as _pftmod
        _pft = int(canopy_config.pft_clm)
        _vlen = int(_pftmod.MLpftcon.vcmaxpft.shape[0])
        if not (0 <= _pft < _vlen):
            raise ValueError(
                f"CLMMLCanopyConfig.pft_clm={_pft} out of range for the MLpftcon "
                f"vcmaxpft table (0..{_vlen - 1}); cannot apply vcmax25_override")
        vcmaxpft_jax = _pftmod.MLpftcon.vcmaxpft.at[_pft].set(
            float(canopy_config.vcmax25_override))

    def _call_mlcanopy(mlc, flt, num, gridobj, cosz):
        _opt_kwargs: dict[str, Any] = {}
        if gridobj is not None:
            _opt_kwargs["grid"] = gridobj
        if _traceable:
            # Device solar zenith (cos), aligned with ``flt`` (entry k -> patch
            # flt[k]).  Lets _GetCLMVar set solar_zen_forcing WITHOUT the host
            # shr_orb_cosz recompute that reads per-step orbital globals.
            _opt_kwargs["cos_zenith_device"] = cosz
        if vcmaxpft_jax is not None:
            _opt_kwargs["vcmaxpft_jax"] = vcmaxpft_jax
        if g1_medlyn_jax is not None:
            _opt_kwargs["g1_MED_jax"] = g1_medlyn_jax
        return MLCanopyFluxes(
            bounds=bounds,
            num_exposedvegp=num,
            filter_exposedvegp=flt,
            atm2lnd_inst=stubs["atm2lnd"],
            canopystate_inst=stubs["canopystate"],
            soilstate_inst=stubs["soilstate"],
            temperature_inst=stubs["temperature"],
            waterstatebulk_inst=stubs["waterstatebulk"],
            waterfluxbulk_inst=stubs["waterfluxbulk"],
            energyflux_inst=stubs["energyflux"],
            frictionvel_inst=stubs["frictionvel"],
            surfalb_inst=stubs["surfalb"],
            solarabs_inst=stubs["solarabs"],
            mlcanopy_inst=mlc,
            wateratm2lndbulk_inst=stubs["wateratm2lndbulk"],
            waterdiagnosticbulk_inst=stubs["waterdiagnosticbulk"],
            _o2ref_py=float(canopy_config.o2ref),
            **_opt_kwargs,
        )

    if _traceable_multi:
        # Canopy columns are physically INDEPENDENT (no horizontal coupling), so the
        # proven single-patch kernel runs once per column with ``grid.p`` = that
        # column's patch index, the ``grid=`` kernel writing ONLY that patch.
        #
        # S3 fast path — jax.lax.scan over columns (O(1) compile in ncol): when the
        # columns share vertical structure (ncan/ntop/nbot/pft uniform — the common
        # explicit-count-layering case, MLinitVerticalMod's _ntop=nlayer_within /
        # _ncan=_ntop+nlayer_above being htop-INDEPENDENT), the scan carry is the
        # shared mlcanopy and the ONLY traced input is the column index p.  The body
        # traces ONCE, so the HLO is O(1) in ncol (vs the S2 Python loop's O(ncol)
        # unroll — the AMIP-scale compile wall).  ncan/ntop/nbot/pft stay CONCRETE
        # (closed-over ``_g0``), so NO per-layer masking or dynamic pft-gather is
        # needed here.  Numerically a NO-OP vs the loop (validated column-for-column
        # against independent single-column runs).
        #
        # S2 fallback — when structure VARIES across columns (heterogeneous
        # PFT/nbot), the concrete-structure scan would be wrong, so fall back to the
        # per-column Python loop: correct, only O(ncol) compile there.  Letting the
        # scan handle that case needs traced-nbot radiation masking + dynamic pft
        # gathers (deferred — see docs/land/clm_ml_s3_masked_vmap_plan.md).
        _g0 = _grids[0]
        _uniform_structure = all(
            (_g.ncan == _g0.ncan and _g.ntop == _g0.ntop
             and _g.nbot == _g0.nbot and _g.pft == _g0.pft)
            for _g in _grids
        )
        # The scan needs a CONCRETE per-column PFT (grid.pft, closed into the scan
        # body): under a traced grid.p the Solar/Longwave host fallback
        # int(patch.itype[grid.p]) would fail.  extract_clm_ml_grid_info always
        # supplies pft>=0; a hand-built grid_info with the pft=-1 sentinel falls back
        # to the S2 loop (concrete p, so its host itype read is valid) — never wrong,
        # only slower.
        _pft_ok = _g0.pft >= 0
        if (_uniform_structure and _pft_ok
                and bool(getattr(canopy_config, "scan_columns", True))):
            # xs: per-column patch index (1..ncol — TRACED under the scan) paired
            # with that column's cos(zenith).  filter=[1] is a STATIC dummy — under
            # grid= the kernel indexes the column by grid.p (the traced xs), NOT the
            # filter; cos is reshaped to the length-1 slice the kernel's
            # cos_zenith_device contract expects.
            _p_xs = jnp.arange(1, ncol + 1)

            def _col_scan_body(_mlc, _xs):
                _p_c, _cosz_c = _xs
                _g_c = GridInfo(p=_p_c, ncan=_g0.ncan, ntop=_g0.ntop,
                                nbot=_g0.nbot, pft=_g0.pft)
                _mlc = _call_mlcanopy(_mlc, [1], 1, _g_c,
                                      jnp.reshape(_cosz_c, (1,)))
                return _mlc, None

            mlcanopy_new, _ = jax.lax.scan(
                _col_scan_body, mlcanopy, (_p_xs, cos_zen))
        else:
            mlcanopy_new = mlcanopy
            for _c in range(ncol):
                mlcanopy_new = _call_mlcanopy(
                    mlcanopy_new, [_c + 1], 1, _grids[_c],
                    cos_zen[_c:_c + 1] if _traceable else None)
    else:
        _grid0 = _grids[0] if _grids is not None else None
        mlcanopy_new = _call_mlcanopy(
            mlcanopy, filter_exposedvegp, num_exposedvegp, _grid0, cos_zen)

    if _traceable:
        # MLCanopyFluxes RETURNED, so a traced artefact really was built with the
        # psihat table this scheme installed. Only now may the process be locked:
        # committing before the call would lock a scheme that a failed trace
        # never compiled, and then wrongly reject a later valid run.
        _commit_diff_turbulence_scheme(canopy_config.turbulence_scheme)

    # ---- Extract SurfaceFluxOutput ----
    surface_out = _extract_surface_fluxes(
        mlcanopy_new, ncol, forcing, land_config, land_params,
        canopy_config=canopy_config)

    return surface_out, CanopyState(mlcanopy=mlcanopy_new, t_a10_arr=t_a10_now)
