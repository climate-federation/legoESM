"""Internal-tide mixing parameterization (Jayne & St-Laurent 2001).

Adds an abyssal diapycnal-diffusivity contribution

    K_tidal(x, y, z) = Γ · q · E_BT(x, y) · F(x, y, z) / (ρ_0 · N²(x, y, z))

where:

* ``E_BT(x, y)`` [W/m²] is the prescribed barotropic-to-baroclinic
  tidal-energy conversion rate.  Provided externally (e.g. from
  Egbert-Erofeeva TPXO + a topographic-roughness model) or via
  :func:`synthetic_baroclinic_tide_energy_from_bathy` for tests.
* ``q`` is the local-dissipation fraction (≈ 1/3; remainder
  radiates as low-mode internal-tide energy).
* ``Γ`` is the Osborn mixing efficiency (≈ 0.2).
* ``F(x, y, z)`` is a normalised vertical structure function that
  concentrates the dissipation near the bottom.  J-S-L 2001 use an
  exponential decay scale ``h_decay ≈ 500 m``:

      F(z) = exp(−(H − z) / h_decay) / N_norm

  with depth ``z`` positive-downward on the wet column ``z ∈ [0, H]``
  (``z = H`` at the seafloor, ``z = 0`` at the surface) and the
  column-integral normaliser ``N_norm = ∫_0^H exp(−(H − z')/h_decay) dz'``.
* ``N²`` is the local Brunt-Väisälä frequency.  Strong stratification
  → small ``K``; weak stratification → larger ``K`` capped by
  ``K_max``.

References
----------
Jayne, S. R., & St-Laurent, L. C. (2001). Parameterizing tidal
dissipation over rough topography. *Geophys. Res. Lett.*, 28(5),
811–814.

Simmons, H. L., Jayne, S. R., St-Laurent, L. C., & Weaver, A. J.
(2004). Tidally driven mixing in a numerical model of the ocean
general circulation. *Ocean Modelling*, 6(3–4), 245–263.

Faithfulness
------------
Oracle: the CVMix Aug-2012 documentation (Griffies et al.) Ch. 5, which writes
the Simmons (2004) scheme (== Jayne & St-Laurent 2001), in this module's
positive-downward depth convention (``z ∈ [0, H]``; ``z = H`` at the seafloor),
as::

    kappa = q * Gamma * E * F(z) / (rho * N^2)                (CVMix Eq. 5.22)
    F(z)  = e^(-(H-z)/zeta) / [zeta * (1 - e^(-H/zeta))]      (Eq. 5.27)

where H is the column/bottom depth and the denominator ``zeta*(1-e^(-H/zeta))``
is the ANALYTIC column integral of the numerator ``e^(-(H-z)/zeta)`` over the
wet column (so ``∫_0^H F dz = 1``).  q = 1/3 (Eq. 5.19, St-Laurent 2002),
Gamma = 0.2 (Eq. 5.14, Osborn 1980), zeta = 500 m (Eq. 5.30).
``tests/ocean/unit/test_tidal_simmons_faithful.py`` pins the closed form to
round-off against an independent reimplementation.

FAITHFUL:
- The bottom-intensified exponential structure — the NORMALISED F matches
  Eq. 5.27 exactly; ``_exp_decay_structure`` returns the numerator
  exp(-(H-z)/h_decay) rescaled by a per-column constant (a log-sum-exp shift
  for underflow safety) that cancels in the normalisation — and the full
  K = Gamma*q*E_BT*F/(rho_0*N^2) (Eq. 5.22).
- Config defaults Gamma = 0.2, q_local = 1/3, h_decay_m = 500, K_max = 5e-3
  m^2/s all match CVMix (Eqs. 5.14 / 5.19 / 5.30 / Sec. 5.3.3).

DEPARTURES (documented, canaried in the test):
- Normalisation is DISCRETE (``F / sum_k F_raw*h_k``, ``normalize_structure``),
  not CVMix's ANALYTIC denominator ``zeta*(1-e^(-H/zeta))`` (Eq. 5.27); the two
  agree in the fine-grid limit and differ by the mid-point-quadrature error on a
  coarse column.  Both integrate F to unity by construction.
- Boussinesq density: Eq. 5.22 divides by the LOCAL in-situ ``rho``; this module
  divides by the configurable REFERENCE density ``rho_0`` (``config.rho_0``,
  default ``constants.rho_ocean``) — the standard Boussinesq substitution.
- ``N_squared_min = 1e-7`` s^-2 is HIGHER than Simmons' suggested floor 1e-8
  (Sec. 5.3.5) — a stronger regularisation of the near-bottom singularity.
- A ``K_max`` cap with NO background floor (caller adds background kappa), the
  ``max(H-z, 0)`` clamp (dry levels below the seafloor return F_raw=1 and are
  then zeroed by ``normalize_structure``'s wet mask), and an ``h_decay >= 1e-6``
  floor are AD/robustness guards not in the analytic oracle.

Differentiability: ``differentiable`` is True in the AD sense — JAX returns
finite selected-branch subgradients everywhere.  The form is NOT everywhere
smooth: ``jnp.maximum`` (the H-z and N^2 floors), ``jnp.clip`` (the K_max cap),
and the wet/dry ``jnp.where`` thresholds are kinked; gradients at those
boundaries are one-sided but finite (checked in the faithfulness test).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Jayne & St-Laurent (2001) internal-tide diapycnal mixing: a "
        "bottom-intensified diffusivity K = Gamma*q*E_BT*F(z)/(rho_0*N^2) from a "
        "prescribed barotropic-to-baroclinic tidal-energy conversion and an "
        "exponential near-bottom vertical structure."
    ),
    "inputs": {
        "E_BT_W_per_m2": "W/m^2", "layer_depths_m": "m", "h_partial_m": "m",
        "H_bathy_m": "m", "N_squared": "1/s^2",
    },
    "outputs": {"K_tidal": "m^2/s"},
    "sign_convention": (
        "K_tidal >= 0; strong stratification (large N^2) reduces K; the vertical "
        "structure F(z) is bottom-intensified and normalised so its column "
        "integral partitions E_BT exactly; capped at K_max with NO background "
        "floor (the caller adds background kappa on top); zero where E_BT=0 or "
        "the column is dry; depths positive downward."
    ),
    # Pure diffusivity producer: nothing conserved; the budget closes in the
    # diffusion solver.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Jayne, S. R. & St-Laurent, L. C. (2001), GRL 28(5), 811-814; "
        "Simmons et al. (2004), Ocean Modelling 6, 245-263"
    ),
    "idealized_test": (
        "tests/unit/test_tidal_mixing.py — K peaks near the seafloor and decays "
        "upward with scale h_decay; E_BT=0 or a dry column gives zero K; larger "
        "N^2 lowers K."
    ),
}


# St Laurent (2002) tidal-mixing scheme reference defaults.
_STLAURENT_RMS_ROUGHNESS_M = 250.0
_STLAURENT_U_TIDE_M_S = 0.02
_STLAURENT_N_BOTTOM_PER_S = 1.0e-3
_STLAURENT_ROUGHNESS_SCALE_M = 3000.0

__param_spec__ = {
    "TidalMixingConfig": {
        "scheme_key": "ocean.vm.tidal",
        "excluded": {
            "N_squared_min": "numerics: floor/cap",
        },
        "params": {
            "Gamma": {"units": "1", "bounds": (0.066, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "St Laurent (2002) tidal mixing", "shape": None},
            "K_max": {"units": "m^2/s", "bounds": (0.00165, 0.015), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "St Laurent (2002) tidal mixing", "shape": None},
            "h_decay_m": {"units": "m", "bounds": (165.0, 1500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "St Laurent (2002) tidal mixing", "shape": None},
            "q_local": {"units": "1", "bounds": (0.11, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "vertical_mixing", "reference": "St Laurent (2002) tidal mixing", "shape": None},
        },
    },
}


class TidalMixingConfig(NamedTuple):
    """Configuration for the Jayne & St-Laurent abyssal tidal mixing scheme.

    Disabled by default — when integrated into the production
    vertical-mixing dispatcher the caller must set ``enabled=True``
    and supply a per-cell ``E_BT`` field (or pass the synthetic
    helper output).

    Gamma, q_local, h_decay_m and K_max defaults match Simmons et al.
    2004 OGCM production values (CVMix Eqs. 5.14 / 5.19 / 5.30 / Sec.
    5.3.3).  ``N_squared_min`` (1e-7) is a STRONGER floor than Simmons'
    suggested 1e-8 (Sec. 5.3.5); ``rho_0`` is a Boussinesq reference
    density, not the Eq. 5.22 in-situ ``rho``.
    """
    enabled: bool = False
    Gamma: float = 0.2           # Osborn mixing efficiency [-]
    q_local: float = 1.0 / 3.0   # Local dissipation fraction [-]
    h_decay_m: float = 500.0     # Exponential vertical decay scale [m]
    K_max: float = 5.0e-3        # Cap on tidal κ [m²/s]
    N_squared_min: float = 1.0e-7  # Stratification floor [1/s²] to
                                    # prevent K → ∞ in convective cells.
    rho_0: float = constants.rho_ocean
    # NOTE: this scheme intentionally has NO background-κ floor.  The
    # caller adds the chosen background κ (constant / KPP / Richardson)
    # on top of the tidal contribution; floors live there so that dry
    # cells and columns where ``E_BT = 0`` produce zero ``K_tidal``.


# ==============================================================================
# Vertical structure function
# ==============================================================================

def _exp_decay_structure(
    layer_depths_m: jnp.ndarray,
    H_bathy_m: jnp.ndarray,
    h_decay_m: float,
    h_partial_m: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Bottom-intensified exponential decay function ``F(z)``.

    Builds a per-column normalised profile that decays upward away
    from the seafloor with e-folding scale ``h_decay_m``.  The
    column integral ``∫ F dz`` is normalised to one so the total
    energy ``E_BT`` is fully partitioned across the column.

    Parameters
    ----------
    layer_depths_m : array ``(..., nlev)``
        Depth of each layer centre (positive downward) [m].
    H_bathy_m : array ``(...)``
        Local sea-floor depth (positive downward) [m].  Broadcasts
        against the layer axis.
    h_decay_m : float
        e-folding scale [m].
    h_partial_m : array ``(..., nlev)`` or None
        Layer thickness [m], used ONLY to identify wet cells
        (``h_partial > 0``) for the log-sum-exp rebase baseline (see
        Notes).  When None the baseline is taken over ALL levels — safe
        for a full wet column; the physical caller
        (:func:`compute_tidal_diffusivity`) always passes it so that a
        column with dry below-bottom levels rebases over the WET part.

    Returns
    -------
    F : array ``(..., nlev)``
        Unnormalised bottom-intensified structure, PROPORTIONAL to the
        Eq. 5.27 numerator ``exp(-(H-z)/h_decay)`` on wet cells (zero on
        dry cells when ``h_partial_m`` is supplied).  It is rescaled by a
        per-column constant (see below) for numerical stability; the
        constant cancels exactly in :func:`normalize_structure`, so the
        NORMALISED ``F`` (and hence ``K``) is unchanged.  ``Σ_k F·dz``
        is set to 1 by that later step.

    Notes
    -----
    Numerical stability: the raw exponent is rebased by the per-column
    MINIMUM distance-above-bottom over WET cells (the deepest wet cell) —
    a log-sum-exp shift — so the largest exponential is ``exp(0)=1`` and
    cannot underflow.  Without it, a decay scale ``h`` far below the grid
    spacing (e.g. the ``1e-6`` floor reached by a pathological
    ``h_decay_m`` outside the supported ``[165, 1500]`` m ``__param_spec__``
    bounds) would underflow ``exp(-(H-z)/h)`` to zero at EVERY wet level,
    collapsing the column to zero mixing instead of the correct
    ``h → 0`` limit (all dissipation concentrated in the deepest wet
    cell).  The baseline is taken over WET cells only: dry below-bottom
    levels clamp to distance 0 and would otherwise pin the baseline to 0,
    re-admitting the underflow on cell-centred columns that sit above the
    seafloor.  Because the shift is a per-column constant it divides out
    of the normalisation exactly, leaving the physical profile untouched
    for supported ``h``.  Dry cells are set to 0 here (rather than the
    overflowing ``exp(+dist_ref/h)``) so the downstream ``F_raw·0`` in
    :func:`normalize_structure` cannot become ``inf·0 = NaN``.
    """
    h = jnp.maximum(h_decay_m, 1.0e-6)
    z = layer_depths_m
    H = H_bathy_m[..., None]                      # (..., 1)
    # Distance above the sea-floor (positive going up from bottom).
    dist_from_bottom = jnp.maximum(H - z, 0.0)
    # Log-sum-exp rebase over WET cells: subtract the smallest wet distance
    # (the deepest wet cell) so max(raw)=1 and tiny h cannot underflow the
    # column to all-zero.  The constant exp(dist_ref/h) cancels in
    # normalize_structure.  ``h_partial_m is None`` is a static (argument-level)
    # check — the None branch is for standalone use on a full wet column.
    if h_partial_m is not None:
        wet = h_partial_m > 0.0
        far = jnp.max(dist_from_bottom, axis=-1, keepdims=True)
        dist_ref = jnp.min(
            jnp.where(wet, dist_from_bottom, far), axis=-1, keepdims=True,
        )
        arg = -(dist_from_bottom - dist_ref) / h
        # DOUBLE-WHERE (grad-safe): dry cells have dist < dist_ref, so their
        # exponent +|Δ|/h would overflow to inf for tiny h; masking the OUTPUT
        # alone leaves a ``0·inf = NaN`` cotangent through the discarded branch
        # under reverse-mode AD.  Mask the EXPONENT to 0 (exp -> 1, finite) AND
        # the output to 0 so both the value and the gradient stay finite.
        arg = jnp.where(wet, arg, 0.0)
        raw = jnp.where(wet, jnp.exp(arg), 0.0)
    else:
        dist_ref = jnp.min(dist_from_bottom, axis=-1, keepdims=True)
        raw = jnp.exp(-(dist_from_bottom - dist_ref) / h)
    return raw


def normalize_structure(
    F_raw: jnp.ndarray,
    h_partial: jnp.ndarray,
) -> jnp.ndarray:
    """Normalise the structure function over the water column.

    ``F_normalised[..., k] = F_raw[..., k] / Σ_k (F_raw[..., k] · h_k)``

    Ensures ``Σ_k F·h = 1`` per column so the column integral of
    ``E_BT · F`` recovers ``E_BT`` exactly.  Columns with zero total
    thickness (dry / pre-grid columns) return zero everywhere.

    Per-level DRY cells (``h_partial <= 0``, e.g. z-levels below the
    partial bottom cell) are forced to ``F = 0``.  They contribute
    nothing to the column integral (``F_raw·0``) but ``_exp_decay_structure``
    still returns a nonzero ``F_raw`` there (the ``max(H-z,0)`` clamp
    caps it at 1), so without this mask a dry level would carry
    ``F = F_raw/integral ≠ 0`` and leak a spurious ``K`` into the
    diffusion solver.  A dry cell is defined by ``h_partial <= 0``
    consistently: NEGATIVE sentinel thicknesses are clamped OUT of the
    integral denominator (``h_wet``) as well as zeroed on output, so a
    dry level can never alter the wet-column partition.  Wet-level
    values are unchanged, so the partition is identical.  The
    ``integral > 1e-12`` guard makes a degenerate near-zero column
    (essentially no water) return zero everywhere rather than divide by
    a vanishing denominator.
    """
    h_wet = jnp.where(h_partial > 0.0, h_partial, 0.0)
    integral = jnp.sum(F_raw * h_wet, axis=-1, keepdims=True)
    safe = jnp.where(integral > 1.0e-12, integral, 1.0)
    F = jnp.where(integral > 1.0e-12, F_raw / safe, 0.0)
    # Zero per-level dry cells so they cannot leak K downstream.
    F = jnp.where(h_partial > 0.0, F, 0.0)
    return F


# ==============================================================================
# Main entry point
# ==============================================================================

def compute_tidal_diffusivity(
    E_BT_W_per_m2: jnp.ndarray,
    layer_depths_m: jnp.ndarray,
    h_partial_m: jnp.ndarray,
    H_bathy_m: jnp.ndarray,
    N_squared: jnp.ndarray,
    *,
    config: TidalMixingConfig,
    land_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Diapycnal diffusivity from the Jayne & St-Laurent (2001) scheme.

    K_tidal = Γ · q · E_BT · F(z) / (ρ_0 · N²)

    Parameters
    ----------
    E_BT_W_per_m2 : array ``(...)``
        Barotropic-to-baroclinic tide-energy conversion rate
        [W/m²].  Spatial map; usually from a coarse-grained
        observational product or :func:`synthetic_baroclinic_tide_energy_from_bathy`.
    layer_depths_m : array ``(..., nlev)``
        Layer-centre depth (positive downward) [m].
    h_partial_m : array ``(..., nlev)``
        Layer thickness [m].  Sum equals ``H_bathy`` per column.
    H_bathy_m : array ``(...)``
        Sea-floor depth [m].
    N_squared : array ``(..., nlev)``
        Brunt-Väisälä frequency squared [1/s²].  Floored at
        ``config.N_squared_min`` to prevent ``K → ∞`` in
        unstratified cells.
    config : :class:`TidalMixingConfig`
    land_mask : array ``(...)`` or None
        Optional 1 = ocean / 0 = land mask applied multiplicatively.

    Returns
    -------
    K_tidal : array ``(..., nlev)``
        Diapycnal diffusivity contribution [m²/s], capped at
        ``config.K_max`` with no background floor.  Zero where
        ``E_BT`` is zero, where the column is dry, or where
        ``land_mask=0``.  Callers add background κ (e.g. from
        ``KPPConfig.K_bg``) separately.
    """
    F_raw = _exp_decay_structure(
        layer_depths_m, H_bathy_m, config.h_decay_m, h_partial_m,
    )
    F = normalize_structure(F_raw, h_partial_m)

    N2_safe = jnp.maximum(N_squared, config.N_squared_min)

    numerator = (
        config.Gamma * config.q_local * E_BT_W_per_m2[..., None] * F
    )
    K = numerator / (config.rho_0 * N2_safe)
    # Cap only (no background floor — the caller adds background κ
    # on top of this contribution).  Zero where E_BT is zero or
    # where ``normalize_structure`` returned zero (dry columns).
    K = jnp.clip(K, 0.0, config.K_max)

    # Force STRUCTURAL zeros to EXACTLY zero via ``where`` (not a multiply):
    # any cell where ``F == 0`` — a dry level (h_partial<=0) OR a degenerate
    # near-zero column (integral<=1e-12) — must produce K=0.  Masked /
    # below-bottom cells commonly carry N²=NaN, where ``0 / NaN = NaN`` would
    # otherwise leak NaN into the diffusion solver despite F=0; ``where``
    # selects the literal 0.0 regardless of NaN.
    K = jnp.where(F > 0.0, K, 0.0)
    # Land mask: amplitude-preserving (a fractional mask scales K) AND
    # NaN-robust where fully land (mask<=0 forces literal 0.0).
    if land_mask is not None:
        m = land_mask[..., None]
        K = jnp.where(m > 0.0, K * m, 0.0)
    return K


def brunt_vaisala_cell_from_eos(
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    layer_thickness_m: jnp.ndarray,
    eos_fn,
    *,
    rho_0: float,
    n_squared_min: float,
    g: float = constants.g,
) -> jnp.ndarray:
    """Cell-centred buoyancy frequency N² [1/s²] from the model EOS.

    Feeds :func:`compute_tidal_diffusivity` (K ∝ 1/N²). Uses the
    model-selected EOS (Wright, linear, …) so the tidal diffusivity's
    stratification is CONSISTENT with the density the dynamical core
    integrates — replacing a hardcoded linear ρ-anomaly (issue #1111).

    Locally-referenced (MOM/NEMO convention): the two levels bounding an
    interface are evaluated at the SHARED interface pressure, so the vertical
    density difference reflects T,S stratification ALONE and does not
    double-count adiabatic compressibility. All geometry is derived from the
    SAME (live) thickness field so the interface is self-consistent: the depth
    of interface ``k+½`` is the running sum of thicknesses down to the bottom
    of cell ``k`` (``cumsum(thickness)[..., k]``), the exact ``z_{k+½}`` on a
    stretched grid; interface pressure is the hydrostatic reference
    ``rho_0 g z_iface`` (positive-down).

    Parameters
    ----------
    T_degC, S_psu : array ``(..., nlev)``
        Potential temperature [degC] and salinity [PSU] at cell centres.
    layer_thickness_m : array ``(..., nlev)``
        Cell thickness [m]. Interface depths and the centre-to-centre spacing
        ``dz`` (half-sum of adjacent thicknesses) are both derived from it, so
        the pressure reference and ``dz`` use one consistent (live) grid.
    eos_fn : callable
        Model EOS ``fn(T, S, p_Pa) -> in-situ density [kg/m³]`` (e.g. from
        :func:`legoesm.ocean.eos.make_eos_fn`).
    rho_0 : float
        Boussinesq reference density [kg/m³].
    n_squared_min : float
        Stratification floor [1/s²] (``config.N_squared_min``) — guards
        ``K → ∞`` in weakly / unstably stratified columns.
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    array ``(..., nlev)``
        N² at cell centres, floored at ``n_squared_min``.
    """
    nlev = T_degC.shape[-1]
    if nlev < 2:
        # Degenerate single layer — N² undefined; return the floor so the
        # downstream K is finite (the diffusion is a no-op anyway).
        return jnp.full_like(T_degC, n_squared_min)

    # Interface z_{k+½} = running sum of thicknesses to the bottom of cell k,
    # from the SAME thickness field as dz (no static/live grid mixing).
    z_iface = jnp.cumsum(layer_thickness_m, axis=-1)[..., :-1]
    p_iface = rho_0 * g * z_iface                       # hydrostatic ref [Pa]
    dz = 0.5 * (layer_thickness_m[..., :-1] + layer_thickness_m[..., 1:])
    dz = jnp.where(dz > 1.0e-12, dz, 1.0)               # centre spacing floor
    rho_upper = eos_fn(T_degC[..., :-1], S_psu[..., :-1], p_iface)
    rho_lower = eos_fn(T_degC[..., 1:], S_psu[..., 1:], p_iface)
    n2_iface = -(g / rho_0) * (rho_upper - rho_lower) / dz   # (..., nlev-1)

    # Interfaces → cell centres: replicate top/bottom, average the interior.
    n2_top = n2_iface[..., :1]
    n2_bot = n2_iface[..., -1:]
    if n2_iface.shape[-1] >= 2:
        n2_avg = 0.5 * (n2_iface[..., :-1] + n2_iface[..., 1:])
        n2_cell = jnp.concatenate([n2_top, n2_avg, n2_bot], axis=-1)
    else:
        # nlev == 2 → one interface → broadcast to both cells.
        n2_cell = jnp.broadcast_to(n2_iface, T_degC.shape)
    return jnp.maximum(n2_cell, n_squared_min)


# ==============================================================================
# Synthetic E_BT for tests / dev runs
# ==============================================================================

def synthetic_baroclinic_tide_energy_from_bathy(
    H_bathy_m: jnp.ndarray,
    *,
    rms_roughness_m: float = _STLAURENT_RMS_ROUGHNESS_M,
    u_tide_m_s: float = _STLAURENT_U_TIDE_M_S,
    N_bottom_per_s: float = _STLAURENT_N_BOTTOM_PER_S,
    rho_0: float = constants.rho_ocean,
    deep_threshold_m: float = 1000.0,
    roughness_scale_m: float = _STLAURENT_ROUGHNESS_SCALE_M,
) -> jnp.ndarray:
    """Synthetic ``E_BT`` field for spin-up smoke runs.

    Approximates the Simmons et al. 2004 conversion-rate formula:

        E_BT ≈ ρ_0 · κ_h · ⟨h²⟩ · N_b · u_tide² / 2

    where ``κ_h = 2π/L_topo`` is the topographic wavenumber and
    ``⟨h²⟩`` is the bathymetry roughness variance.  Without a
    topographic spectrum this helper uses a single
    ``rms_roughness_m`` value and modulates by depth so deep abyssal
    regions get the standard ~1 mW/m² and shallow shelves get zero.

    Parameters
    ----------
    H_bathy_m : array ``(...)``
        Local bathymetry [m].  Shallow regions (``H < deep_threshold_m``)
        produce zero ``E_BT``.
    rms_roughness_m : float
        Sub-grid topographic roughness ``√⟨h²⟩`` [m].  Default
        250 m matches Simmons 2004's global mean.
    u_tide_m_s : float
        Barotropic tidal current amplitude [m/s].  Default 0.02 m/s
        (~2 cm/s) is the global-mean M2.
    N_bottom_per_s : float
        Bottom Brunt-Väisälä frequency [1/s].
    rho_0 : float
        Reference seawater density [kg/m³].
    deep_threshold_m : float
        Bathymetry below this depth gets the full ``E_BT``; shallower
        cells ramp linearly down to zero at the coast.
    roughness_scale_m : float
        Topographic wavelength controlling ``κ_h = 2π / L_topo`` [m].
        Default 3 km matches typical mid-ocean-ridge spectra.

    Returns
    -------
    E_BT : array (same shape as ``H_bathy_m``)
        Synthetic baroclinic conversion rate [W/m², ≥ 0].
    """
    # Jayne & St. Laurent (2001) / Simmons et al. (2004) conversion rate
    #   E = ½·ρ₀·κ_h·⟨h²⟩·N_b·⟨u²⟩   [W/m²]
    # κ_h enters LINEARLY (not squared): kg/m³·(1/m)·m²·(1/s)·m²/s² = kg/s³ =
    # W/m². Squaring κ_h yields W/m³ (~477× too weak at these defaults).
    kappa_h = 2.0 * np.pi / max(roughness_scale_m, 1.0)
    E_uniform = (
        0.5
        * rho_0
        * kappa_h
        * (rms_roughness_m ** 2)
        * N_bottom_per_s
        * (u_tide_m_s ** 2)
    )
    depth_ramp = jnp.clip(
        H_bathy_m / max(deep_threshold_m, 1.0), 0.0, 1.0,
    )
    return E_uniform * depth_ramp
