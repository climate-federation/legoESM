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

  with ``N_norm = ∫_{-H}^{0} exp(−(H − z')/h_decay) dz'``.
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

    Defaults match Simmons et al. 2004 OGCM production values.
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

    Returns
    -------
    F : array ``(..., nlev)``
        Vertical structure function such that ``Σ_k F[..., k]·dz[k]
        ≈ 1`` for each wet column (caller-supplied ``dz`` are folded
        in by the column-integral normalisation step).
    """
    h = jnp.maximum(h_decay_m, 1.0e-6)
    z = layer_depths_m
    H = H_bathy_m[..., None]                      # (..., 1)
    # Distance above the sea-floor (positive going up from bottom).
    dist_from_bottom = jnp.maximum(H - z, 0.0)
    raw = jnp.exp(-dist_from_bottom / h)
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
    """
    integral = jnp.sum(F_raw * h_partial, axis=-1, keepdims=True)
    safe = jnp.where(integral > 1.0e-12, integral, 1.0)
    F = jnp.where(integral > 1.0e-12, F_raw / safe, 0.0)
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
        layer_depths_m, H_bathy_m, config.h_decay_m,
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

    if land_mask is not None:
        K = K * land_mask[..., None]
    return K


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
