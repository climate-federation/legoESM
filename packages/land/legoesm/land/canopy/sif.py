"""Solar-Induced chlorophyll Fluorescence (SIF) — optional canopy diagnostic.

Ported from the **BEPS-SIF** model (Qiu & Zhang, `yongguangzhang/BEPS-SIF-model`,
``photosyn_gs.c::SIF_y``), which itself uses the van der Tol et al. (2014)
empirical light-saturation fluorescence-yield parameterisation.

This module computes leaf- and canopy-level SIF **from an existing Farquhar
solution** — net assimilation ``An``, intercellular CO2 ``Ci``, the CO2
compensation point ``Gamma*`` and absorbed PAR.  It deliberately does **not**
re-implement photosynthesis, electron transport or stomatal conductance; those
come from the caller (``canopy/photosynthesis.py`` for the two-leaf canopy,
``land/stomata.py`` for the SimpleSEB big-leaf).  SIF is a passive diagnostic:
it never feeds back into the prognostic land state.

Model (all per leaf-class, following BEPS-SIF ``SIF_y``)::

    je   = An * (Ci + 2*Gamma*) / (Ci - Gamma*)        # actual electron transport, >= 0
    x    = 1 - je / (max_electron_yield * APAR)         # degree of light saturation, [0,1]
    Kn   = kn0 * (1+kn_beta) * x^kn_gamma / (x^kn_gamma + kn_beta)   # NPQ (van der Tol 2014)
    fm   = kf / (kf + kd + Kn)                          # steady-state fluorescence yield
    ps   = kp / (kf + kp + kd) * (1 - x)                # photochemical drain
    fs   = fm * (1 - ps)                                # fluorescence yield
    SIF  = fs * APAR                                    # emitted leaf SIF photon flux

Canopy total is the sunlit + shaded sum (two-leaf), the single big-leaf value
(SimpleSEB), or the leaf-area-weighted sum over canopy layers × sunlit/shaded
(CLM-ML multilayer), scaled by an optional escape probability ``fesc``
(emitted -> observed top-of-canopy).  ``fesc = 1`` (default) returns the
*emitted* SIF; it is clamped to ``[0, 1]`` at the aggregation sites (a
probability).  All paths share the same per-leaf fluorescence core
(:func:`leaf_sif_from_je` = fluorescence yield × APAR), so a single-element
canopy reduces exactly to the big-leaf.

Electron transport ``je`` reaches the core two ways:

- **Big-leaf / two-leaf** (SimpleSEB, two-leaf canopy): no native ``je`` is
  exposed, so :func:`leaf_sif` *inverts* it from the Farquhar solution,
  ``je = An*(Ci+2*Gamma*)/(Ci-Gamma*)`` (BEPS-SIF ``photosyn_gs``).
- **CLM-ML multilayer**: the Bonan model already solved electron transport, so
  :func:`multilayer_canopy_sif` consumes its native ``je_leaf`` + ``apar_leaf``
  **directly** — SIF stays a pure consumer of that Farquhar solution and never
  re-implements it.

The inversion is **C3-style** (uses the C3 CO2 compensation point ``Gamma*``),
matching BEPS-SIF ``photosyn_gs`` (single ``gammac``).  The SimpleSEB big-leaf
path is C3-only, so this is exact there.  For a mixed two-leaf canopy
(``fC4 > 0``) it is applied to the blended C3/C4 ``An`` as a documented
BEPS-parity approximation — there is no separate C4 fluorescence path (out of
scope; ``An`` for C4 dominated cells still yields a plausible SIF, but the
``je`` proxy is not physically rigorous for the C4 fraction).  The multilayer
path sidesteps this entirely by using the model's own ``je``.

Units: ``An``/``je``/``APAR``/``SIF`` in ``umol m-2 s-1`` (photon flux for
APAR/SIF); ``Ci``/``Gamma*`` in ``umol mol-1`` (mole fraction — they must share
units, which the two-leaf and SimpleSEB Farquhar paths both satisfy).  The
BEPS-SIF ``0.05`` denominator coefficient is exposed as ``max_electron_yield``:
BEPS applied it to *incident* PPFD, whereas legoESM feeds *absorbed* PAR, so
retune it (and ``fesc``) against satellite SIF for absolute matching.

All functions are pure JAX, JIT-compatible and differentiable (safe-guarded
divisions use the double-``where`` trick so reverse-mode gradients stay finite
at ``APAR -> 0`` and ``Ci -> Gamma*``).

Faithfulness (oracle: BEPS-SIF ``photosyn_gs.c::SIF_y`` + je/xxn block, Qiu 2015)
--------------------------------------------------------------------------------
The steady-state fluorescence-yield core :func:`fluorescence_yield` is a
**byte-faithful** port of the BEPS-SIF ``SIF_y`` C function — every coefficient
equals the source verbatim::

    kf = 0.05;  kd = 0.95;  kp = 4.0;
    ps_sif = kp/(kf+kp+kd)*(1.0-xxn);
    kk = exp(log(xxn)*2.83);                    // = xxn**2.83
    kn = 2.48*(1+0.114)*kk/(kk+0.114);          // van der Tol (2014) NPQ
    fm = kf/(kf+kd+kn);
    fs = fm*(1.0-ps_sif);

and :func:`actual_electron_transport` reproduces the BEPS proxy
``je = aphoto*(ci+2*gammac)/(ci-gammac)`` (with a floor at 0) in the NORMAL
regime.  Note ``kn_beta = 0.114`` is the value hardcoded in ``SIF_y`` — NOT the
``0.0114`` of van der Tol's drought-stressed parameter set (a factor-of-ten
trap).  ``tests/land/unit/test_canopy_sif_faithful.py`` pins the full ``fs(x)``
curve, ``je``, ``x`` and the composed ``leaf_sif = fs(x)*APAR`` against an
INDEPENDENT scalar reimplementation to round-off (rel ``1e-12``), with every
coefficient (``kn0``/``kn_beta``/``kn_gamma``/``kf``/``kd``/``kp``/the ``2*Γ*``
factor/the ``0.05`` scale) canaried so the pin is provably non-vacuous.

FAITHFUL where it counts — the byte-identical ``fs(x)``, ``je`` and ``x`` hold to
round-off across the strictly-nonsingular NORMAL regime ``Ci - Gamma* > eps`` AND
``APAR > eps`` (``eps = 1e-9``).  The departures below fall into (a) absolute
calibration (#1, #5); (b) the singular EDGES ``Ci -> Gamma*`` / ``APAR -> 0``
(the #2 knife-edge, #3), where BEPS's raw C divisions hit ``0/0`` (NaN) or
``je/0`` (``±inf``) at EXACT equality and a differentiable model must guard the
``0/0`` to keep the reverse-mode gradient (and forward value) finite; and (c) the
physically-degenerate at/below-compensation region ``Ci <= Gamma*`` (#2) —
NONSINGULAR (e.g. ``Ci=40, Gamma*=43`` has denom ``-3``) but where BEPS's
``if(je<0)`` floor leaks a sign-flipped ``je`` artifact that the module instead
sends to ``je=0``.  #4 (the symmetric ``[0,1]`` x-clip) is a robustness guard
that spans (b) and the ordinary regime: its LOWER clip fires at the
light-saturation edge (``je > mey*APAR``, matching BEPS's ``xxn<0 -> 0``), while
its UPPER clip fires only on the native-``je`` API
(:func:`leaf_sif_from_je`) for a caller-supplied ``je<0`` at ORDINARY ``APAR``
(not an edge).  Each departure is canaried in that test.
(``APAR`` is absorbed PAR, ``>= 0`` by definition — the ``SIF >= 0`` guarantee and
this characterisation both assume ``APAR >= 0``; a negative ``APAR`` is out of
contract.  When the ``Ci -> Gamma*`` and ``APAR -> 0`` guards overlap, the dark
guard #3 is applied LAST, so ``APAR <= eps`` forces ``x = 0`` even where #2 would
otherwise give ``x = 1``.)

DEPARTURES from BEPS-SIF ``SIF_y``:

1. **Absorbed PAR, not incident PPFD** (the only normal-regime departure).
   BEPS forms BOTH the light-saturation denominator and the emitted flux from
   INCIDENT PPFD — ``xxn = 1 - je/(iphoton*0.05)`` and
   ``*xSIF = SIF_y(xxn)*iphoton`` with ``iphoton = 4.55*0.5*rad_leaf``.  legoESM
   feeds ABSORBED PAR to both (``x = 1 - je/(max_electron_yield*APAR)``,
   ``SIF = fs*APAR``), keeping the shared ``0.05`` as the tunable
   ``max_electron_yield``.  The yield *shape* ``fs(x)`` is unchanged, so re-tune
   ``max_electron_yield`` (and ``fesc``) against satellite SIF for absolute
   matching.  A reformulation, not a bug.
2. **``je`` floored to 0 across the whole at/below-compensation region
   ``Ci <= Gamma*`` (and the ``(Gamma*, Gamma*+eps]`` knife-edge)**, via a
   ``denom = Ci - Gamma* > eps`` guard.  This SUPERSEDES BEPS's cruder
   ``if(je<0) je=0`` floor, which only catches ``An<0`` when ``Ci > Gamma*``
   (positive denominator): for ``Ci < Gamma*`` the sign of ``je`` FLIPS, so
   BEPS's floor leaks two artifacts that legoESM instead sends to ``je=0``
   (``x=1`` for ``APAR > eps``, no photosynthetic electron transport at/below
   compensation — the physical answer):
     - ``An < 0`` (respiring leaf below compensation): BEPS ``je = neg/neg`` is
       spuriously POSITIVE (e.g. ``An=-1, Ci=40, Gamma*=43 -> je=+42``, so BEPS
       ``x=0``) — a whole half-plane, not an ``eps`` set.
     - ``An > 0`` on the knife-edge ``Ci in (Gamma*, Gamma*+eps]``: BEPS ``je``
       is huge-positive, then masked to ``x=0`` by its own ``xxn<0 -> 0`` clip.
   The guard is a double-``where`` (not a one-sided ``max(denom,eps)``, which
   would flip the sign into a huge ``je``) so the reverse-mode gradient at
   ``Ci=Gamma*`` stays finite.  At EXACT ``Ci = Gamma*`` BEPS computes ``0/0``
   (NaN for ``An=0``) or ``±inf`` (IEEE, ``An!=0``); the guard returns the finite
   ``je=0`` regardless of ``An`` (``x=1`` when ``APAR > eps``; ``x=0`` if also
   ``APAR <= eps``, per #3).  Both regions are physically degenerate.
3. **Dark guard ``APAR <= eps -> x = 0``** (mirror of #2 at ``APAR -> 0``).  For
   any ``APAR <= eps`` the module returns ``x=0`` rather than the C formula
   ``1 - je/(APAR*0.05)``, via a double-``where`` for a finite gradient at
   ``APAR=0``.  At EXACT ``APAR = 0`` BEPS computes ``je/0`` — ``0/0`` (NaN) once
   ``je`` is floored to 0, or ``+inf`` for a residual ``je>0``; the guard returns
   the finite ``x=0`` (SIF ``0``).  For ``0 < APAR <= eps`` BEPS's ``x -> 1``
   limit (at ``je=0``) differs from the guard's ``x=0``, but the emitted
   ``fs(0)*APAR = O(eps)`` makes the SIF flux error negligible.
4. **Symmetric x-clip.**  BEPS clips only ``xxn<0 -> 0`` (relying on
   ``je>=0 => xxn<=1``); legoESM clips ``[0,1]`` — redundantly, in BOTH
   :func:`degree_of_light_saturation` and (at its input) :func:`fluorescence_yield`
   — so ``x`` can never leave ``[0,1]``.  On the ``An``/``Ci`` inversion path
   (:func:`leaf_sif`) ``je>=0`` so the upper clip is a no-op; on the native-``je``
   API (:func:`leaf_sif_from_je`, the CLM-ML path) a caller-supplied ``je<0`` with
   ``APAR > eps`` gives ``x>1`` and the upper clip keeps it in ``SIF_y``'s
   ``[0,1]`` domain (for ``APAR <= eps`` the dark guard #3 sets ``x=0`` first, so
   the upper clip is not involved).
5. **Escape probability ``fesc``.**  An ADDITIVE emitted->observed
   top-of-canopy factor (Yang & van der Tol 2016), clamped ``[0,1]``, applied at
   the canopy-aggregation sites; BEPS ``SIF_y`` has no such factor.  ``fesc = 1``
   (the default) gives the raw, un-attenuated legoESM emitted SIF — which equals
   BEPS's ``fs*iphoton`` form ONLY under the absorbed=incident identification of
   departure #1 (i.e. it does not by itself undo that reformulation).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Safe-division / floor epsilon.  A regulariser (never a tunable), below the
# param-spec float-eligibility so it can never reach the trainable collector.
_SIF_EPS = 1e-9


__param_spec__ = {
    "SIFConfig": {
        "scheme_key": "land.canopy.sif",
        "params": {
            # van der Tol (2014) empirical NPQ curve — the primary satellite-SIF
            # calibration knobs (tier 1).
            "kn0": {"units": "1", "bounds": (0.5, 6.0), "tunable_tier": 1, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
            "kn_beta": {"units": "1", "bounds": (0.01, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
            "kn_gamma": {"units": "1", "bounds": (0.5, 10.0), "tunable_tier": 1, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
            # Maximum electron yield per unit absorbed PAR (BEPS-SIF `0.05`
            # denominator) — sets the light-saturation scale; tier 1.
            "max_electron_yield": {"units": "1", "bounds": (0.01, 0.2), "tunable_tier": 1, "transform": "sigmoid", "category": "fluorescence", "reference": "BEPS-SIF photosyn_gs.c (iphoton*0.05)", "shape": None},
            # Canopy SIF escape probability fesc (emitted -> observed TOC); tier 1.
            # Sigmoid with the default (1.0 = no escape loss) STRICTLY INTERIOR to
            # the bounds so it round-trips through the trainable collector's
            # transform inverse; the upper bound 1.05 is only that interior margin
            # — the physical fesc is clamped to [0, 1] at every use site.
            "escape_probability": {"units": "1", "bounds": (0.1, 1.05), "tunable_tier": 1, "transform": "sigmoid", "category": "canopy_structure", "reference": "van der Tol 2014 / Yang & van der Tol 2016 fesc", "shape": None},
            # Photophysical de-excitation rate constants (fluorescence / thermal /
            # photochemistry) — only their ratios matter, so held in the extended
            # tier (tier 2) to avoid degeneracy with kn0 in core training.
            "kf": {"units": "1", "bounds": (0.01, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
            "kd": {"units": "1", "bounds": (0.5, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
            "kp": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fluorescence", "reference": "van der Tol 2014 GB; BEPS-SIF SIF_y", "shape": None},
        },
    },
}


class SIFConfig(NamedTuple):
    """Solar-induced fluorescence parameters (BEPS-SIF / van der Tol 2014).

    Presence of a ``SIFConfig`` on the canopy config enables the SIF
    diagnostic; ``None`` disables it (default).  Defaults reproduce the
    BEPS-SIF ``SIF_y`` coefficients.
    """

    # --- van der Tol (2014) NPQ curve: Kn = kn0*(1+beta)*x^gamma/(x^gamma+beta) ---
    kn0: float = 2.48
    kn_beta: float = 0.114
    kn_gamma: float = 2.83

    # --- light-saturation scale: x = 1 - je/(max_electron_yield*APAR) ---
    max_electron_yield: float = 0.05

    # --- emitted -> observed top-of-canopy escape probability (fesc) ---
    escape_probability: float = 1.0

    # --- photophysical rate constants (fluorescence / thermal / photochemical) ---
    kf: float = 0.05
    kd: float = 0.95
    kp: float = 4.0


def fluorescence_yield(x: jax.Array, cfg: SIFConfig) -> jax.Array:
    """Steady-state fluorescence yield ``fs`` (van der Tol 2014 / BEPS-SIF).

    Parameters
    ----------
    x   : degree of light saturation in ``[0, 1]`` (0 = light-limited, all
          absorbed light drives photochemistry; 1 = light-saturated).
    cfg : :class:`SIFConfig`.

    Returns
    -------
    fs : dimensionless fluorescence yield in ``(0, 1)``.
    """
    x = jnp.clip(x, 0.0, 1.0)
    # Guard power(0, kn_gamma): d/d(kn_gamma) x^gamma = x^gamma*log(x) is
    # 0*(-inf) = NaN at x=0, which x commonly hits at light saturation
    # (je >= max_electron_yield*apar).  kn_gamma is tunable, so that NaN would
    # poison training gradients.  x=0 -> x^gamma=0 with zero gamma-gradient is
    # the analytic limit (gamma>0), so mask x=0 off the differentiable path.
    x_pos = x > 0.0
    xg = jnp.where(x_pos, jnp.power(jnp.where(x_pos, x, 1.0), cfg.kn_gamma), 0.0)
    kn = cfg.kn0 * (1.0 + cfg.kn_beta) * xg / (xg + cfg.kn_beta)
    fm = cfg.kf / (cfg.kf + cfg.kd + kn)
    ps = cfg.kp / (cfg.kf + cfg.kp + cfg.kd) * (1.0 - x)
    return fm * (1.0 - ps)


def actual_electron_transport(
    An: jax.Array, Ci: jax.Array, gamma_star: jax.Array,
) -> jax.Array:
    """Actual electron-transport proxy ``je`` from net assimilation (BEPS-SIF).

    ``je = An * (Ci + 2*Gamma*) / (Ci - Gamma*)``, floored at 0 (BEPS-SIF
    ``if je<0: je=0``).  ``Ci`` and ``Gamma*`` must share units (mole fraction).

    When ``Ci <= Gamma*`` (dawn/dusk, stress) the true denominator is <= 0, so
    the signed ``je`` is negative for ``An > 0`` and must floor to 0 — the leaf
    is at/below its compensation point (no net carboxylation).  We mask ``je``
    to 0 there with a double-``where``: this both reproduces the BEPS floor AND
    keeps the reverse-mode gradient finite (never divides by the ``~0``
    denominator).  A one-sided ``max(denom, eps)`` would be WRONG here — it flips
    the sign, turning the ``je<0`` case into a huge positive ``je``.
    """
    denom = Ci - gamma_star
    valid = denom > _SIF_EPS
    safe_denom = jnp.where(valid, denom, 1.0)
    je_raw = An * (Ci + 2.0 * gamma_star) / safe_denom
    return jnp.where(valid, jnp.maximum(je_raw, 0.0), 0.0)


def degree_of_light_saturation(
    je: jax.Array, apar: jax.Array, cfg: SIFConfig,
) -> jax.Array:
    """Degree of light saturation ``x = 1 - je/(max_electron_yield*APAR)``.

    Clipped to ``[0, 1]``.  ``APAR -> 0`` (dark) returns 0 via a double-``where``
    so the reverse-mode gradient stays finite (no ``0/0``).
    """
    denom = cfg.max_electron_yield * apar
    lit = apar > _SIF_EPS
    safe_denom = jnp.where(lit, denom, 1.0)
    x = jnp.where(lit, 1.0 - je / safe_denom, 0.0)
    return jnp.clip(x, 0.0, 1.0)


def leaf_sif_from_je(
    je: jax.Array, apar: jax.Array, cfg: SIFConfig,
) -> jax.Array:
    """Emitted leaf SIF photon flux [umol m-2 s-1] from a KNOWN ``je``.

    ``SIF = fluorescence_yield(x) * APAR`` with ``x =
    degree_of_light_saturation(je, APAR)``.  Use this when the caller already
    has the electron-transport rate — e.g. the CLM-ML multilayer canopy's
    native ``je_leaf``; :func:`leaf_sif` is the An/Ci/Gamma* wrapper that
    *inverts* je first (BEPS-SIF style, for callers without a native je such as
    the SimpleSEB big-leaf).  Does NOT apply the escape probability.
    ``APAR = 0`` gives ``SIF = 0`` exactly.
    """
    x = degree_of_light_saturation(je, apar, cfg)
    return fluorescence_yield(x, cfg) * apar


def leaf_sif(
    An: jax.Array, Ci: jax.Array, gamma_star: jax.Array, apar: jax.Array,
    cfg: SIFConfig,
) -> jax.Array:
    """Emitted leaf (or single leaf-class) SIF photon flux [umol m-2 s-1].

    ``SIF = fluorescence_yield(x) * APAR`` with ``je`` inverted from the Farquhar
    solution (:func:`actual_electron_transport`).  Does NOT apply the escape
    probability (aggregate with :func:`two_leaf_canopy_sif` or multiply by
    ``cfg.escape_probability`` at the call site for the observed TOC value).
    ``APAR = 0`` gives ``SIF = 0`` exactly.
    """
    return leaf_sif_from_je(actual_electron_transport(An, Ci, gamma_star), apar, cfg)


def two_leaf_canopy_sif(
    An_sun: jax.Array, Ci_sun: jax.Array, gstar_sun: jax.Array, apar_sun: jax.Array,
    An_sh: jax.Array, Ci_sh: jax.Array, gstar_sh: jax.Array, apar_sh: jax.Array,
    cfg: SIFConfig,
) -> jax.Array:
    """Observed top-of-canopy SIF for a two-leaf (sunlit+shaded) canopy.

    ``(SIF_sun + SIF_sh) * fesc``.  The sunlit/shaded ``An`` and ``APAR`` are
    the canopy-integrated per-class quantities (per unit ground area), so the
    sum is the canopy total with no extra LAI weighting.  ``fesc`` is clamped to
    ``[0, 1]`` (a probability) so a hand-set or training-perturbed config can
    never produce negative or amplified SIF.
    """
    sif_sun = leaf_sif(An_sun, Ci_sun, gstar_sun, apar_sun, cfg)
    sif_sh = leaf_sif(An_sh, Ci_sh, gstar_sh, apar_sh, cfg)
    fesc = jnp.clip(cfg.escape_probability, 0.0, 1.0)
    return (sif_sun + sif_sh) * fesc


def multilayer_canopy_sif(
    je: jax.Array, apar: jax.Array, leaf_area: jax.Array, cfg: SIFConfig,
) -> jax.Array:
    """Observed top-of-canopy SIF for a multi-layer canopy (CLM-ML).

    Consumes the multilayer model's OWN electron-transport rate ``je`` (CLM-ML
    ``je_leaf``) and absorbed PAR ``apar`` (``apar_leaf``) directly — it does
    NOT re-invert ``je`` from ``An``/``Ci`` (that is the big-leaf
    :func:`leaf_sif` path, for callers without a native ``je``).  This keeps SIF
    a *pure consumer* of the Bonan multilayer canopy's Farquhar solution.

    Leaf-area-weighted sum of per-(layer, leaf-class) emitted SIF, × escape
    probability ``fesc``.  All arrays broadcast over the canopy elements
    (layers × sunlit/shaded) and the SIF is summed over the LAST axis, so pass
    ``(ncol, n_elem)`` to get per-column ``(ncol,)`` SIF.

    ``apar`` is per unit **leaf** area (``apar_leaf``) and ``leaf_area`` is that
    element's leaf-area index (``dpai_profile * fracsun`` for sunlit,
    ``dpai_profile * (1 − fracsun)`` for shaded), so ``leaf_sif * leaf_area`` is
    the per-**ground** contribution — matching how CLM-ML sums ``anet_leaf *
    dpai`` into canopy GPP.  Invalid / unfilled elements must be passed with
    ``leaf_area = 0`` (their sanitized ``je``/``apar`` then drop out).

    Reduces EXACTLY to :func:`leaf_sif_from_je` × ``fesc`` for a single element
    with ``leaf_area = 1``; feeding ``je = actual_electron_transport(An, Ci,
    Gamma*)`` then ties it to the big-leaf :func:`leaf_sif` — the cross-check
    that validates the multilayer path against the big-leaf / two-leaf paths
    (identical fluorescence core).

    JE CONVENTION (validated at CHATS7 — do NOT rescale ``je_leaf``):
    ``je_leaf`` is the model's FULL Farquhar electron-transport rate ``J``,
    whereas the big-leaf inversion is the BEPS-SIF *proxy*
    ``An*(Ci+2*Gamma*)/(Ci-Gamma*)`` ≈ ``J/4`` — so the two je definitions differ
    ~4-5x in absolute scale.  Despite that, feeding ``je_leaf`` directly is
    CORRECT.  With ``max_electron_yield`` (BEPS-calibrated ~0.05) applied to
    ABSORBED PAR, the full-J ``je_leaf`` drives ``x`` onto its 0-clamp for the
    high-light daytime elements, where ``fluorescence_yield(0)`` is je-INDEPENDENT
    (``SIF ~ fs(0)*APAR*fesc``) so the je scale drops out there.  Rescaling
    ``je_leaf`` to the proxy convention (÷4) instead lifts ``x`` off the clamp at
    the sub-saturated sunrise/sunset steps, raising the yield and OVERSHOOTING the
    multilayer SIF -- it BREAKS the agreement rather than improving it.  An EC-site
    diurnal cross-check (``scripts/validate/compare_ml_bigleaf_ec.py``, CHATS7)
    confirms the direction INTERNALLY to the multilayer path (same APAR, same leaf
    areas, so it isolates the je convention): the ÷4-rescaled canopy SIF OVERSHOOTS
    the native-``je_leaf`` SIF by ~15 % on the diurnal mean -- more near the
    sub-saturated sunrise/sunset steps where ``x`` lifts off the clamp -- so the
    native ``je_leaf`` is the correct feed and the ÷4 rescale breaks it (the
    ~15 %/~5x magnitudes are EMPIRICAL, not derivable from the code).  This
    native-vs-÷4 test is SEPARATE from how the multilayer SIF MAGNITUDE compares to
    the two-leaf big-leaf: that cross-scheme match is set by canopy STRUCTURE (the
    two-leaf's single green LAI absorbs less than CLM-ML's plant-area profile) and
    runs ~10 % below, tracking the latent-heat/radiation bias -- NOT a je-convention
    effect.  (An earlier ~1 % cross-scheme match was an artifact of driving the
    two-leaf with plant-area index; green LAI is physiology-correct and exposes the
    structural ~10 %.)
    ``max_electron_yield`` and ``fesc`` remain tier-1 trainables for ABSOLUTE
    calibration against satellite SIF (a separate concern from the je convention).
    """
    per_leaf = leaf_sif_from_je(je, apar, cfg)
    fesc = jnp.clip(cfg.escape_probability, 0.0, 1.0)
    return jnp.sum(per_leaf * leaf_area, axis=-1) * fesc


__physics_contract__ = {
    "summary": (
        "Solar-induced chlorophyll fluorescence (SIF) as an optional passive "
        "canopy diagnostic, ported from BEPS-SIF (Qiu & Zhang) using the "
        "van der Tol (2014) empirical light-saturation fluorescence yield. "
        "Computes leaf/canopy SIF from the existing Farquhar solution (An, Ci, "
        "Gamma*, absorbed PAR); does not re-implement photosynthesis or stomatal "
        "conductance and never feeds back into the prognostic state. Sunlit+shaded "
        "sum for the two-leaf canopy; single leaf for SimpleSEB; leaf-area-weighted "
        "layer x sun/shade sum for the CLM-ML multilayer canopy; optional escape "
        "probability fesc (clamped to [0,1]) converts emitted to observed "
        "top-of-canopy SIF. LIMITATION: the je inversion is C3-style (uses "
        "Gamma*) as in BEPS-SIF; exact for C3 (default fC4=0 and the C3-only "
        "SimpleSEB path), a documented BEPS-parity approximation for fC4>0 "
        "(no separate C4 fluorescence path)."
    ),
    "inputs": {
        "An": "umol/m^2/s", "Ci": "umol/mol", "gamma_star": "umol/mol",
        "apar": "umol/m^2/s (>= 0; absorbed PAR)",
        "leaf_area": "m^2/m^2 (>= 0; leaf-area index, multilayer path only)",
    },
    "outputs": {"sif": "umol/m^2/s"},
    "sign_convention": (
        "SIF >= 0 (emission) FOR APAR >= 0 (absorbed PAR; the input domain). je "
        "floored at 0; degree of light saturation x in [0,1]; fluorescence yield "
        "fs in (0,1); SIF = fs*APAR so APAR=0 -> SIF=0. A negative APAR is out of "
        "contract (SIF = fs(0)*APAR < 0) — absorbed PAR is non-negative by "
        "definition. The multilayer path (multilayer_canopy_sif) additionally "
        "weights by leaf_area (LAI); leaf_area >= 0 is likewise required for "
        "SIF >= 0 (both APAR and LAI are non-negative by definition)."
    ),
    "conserves": [],  # passive diagnostic, not a conservation law
    "differentiable": True,
    "reference": (
        "van der Tol, Berry, Campbell & Rascher (2014) J. Geophys. Res. "
        "Biogeosci. 119, 2312-2327; Qiu & Zhang BEPS-SIF model "
        "(github.com/yongguangzhang/BEPS-SIF-model, photosyn_gs.c SIF_y); "
        "Yang & van der Tol (2016) RSE 209 for the escape probability fesc."
    ),
    "idealized_test": (
        "tests/land/unit/test_canopy_sif.py: SIF > 0 in light and == 0 in the "
        "dark (APAR=0); SIF scales ~linearly with APAR at fixed saturation; "
        "fluorescence yield in [0.005, 0.06]; x monotonic in je; fesc scales the "
        "output; finite grad wrt An and APAR."
    ),
}
