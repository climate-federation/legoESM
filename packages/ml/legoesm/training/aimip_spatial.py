"""Spatially-varying AIMIP surface and radiation parameters.

This module exposes a low-rank lat-lon basis for the AIMIP "classical"
surface and radiation knobs that were previously global scalars
(``Cd_neutral``, ``Ch_neutral``, ``z0``, ``sfc_emissivity``,
``sfc_albedo``).

The motivation is the cold T-bias and synoptic-pattern residual that
the global-scalar classical variant cannot close against SFNO (see
``results/aimip_001/aimip_scorecard.json``).  SFNO learns lat-lon
kernels; classical knobs are scalars.  Adding a low-rank spatial
parameterization to the surface-energy-balance fields lets the
classical variant capture the dominant spatial structure (land vs
ocean, equator-pole gradients, zonal asymmetries from topography) at
a small parameter cost (~13 coefficients per field).

Basis: real-valued products of Legendre polynomials in
``sin(lat)`` and Fourier modes in ``lon`` up to ``l_max=4`` /
``m_max=2``.  Total 13 modes (1 constant + 4 zonal Legendre + 4
zonal-1 longitude modes + 4 zonal-2 longitude modes).  Each basis
function is normalized to unit area-weighted RMS so the raw
coefficients have a uniform scale across modes.

Land-mask gating: each :class:`SpatialField` accepts an optional land
mask in ``evaluate``.  When provided, the field is the learned
spatial value on land columns and falls back to the global scalar
``f_0`` over the ocean.  This matches the user's request that
surface parameter flexibility be "over land" — ocean surface fluxes
are dominated by prescribed SST + a stability-dependent bulk
formula and benefit less from extra learnable land-surface roughness.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ----------------------------------------------------------------------
# Low-rank Legendre × Fourier basis on a Gaussian grid
# ----------------------------------------------------------------------

# Default basis specification.  13 modes balances flexibility against
# overfitting risk on a single-GPU AIMIP training budget.  Extend by
# raising ``l_max`` / ``m_max`` in :func:`spatial_basis`.
_DEFAULT_L_MAX = 4
_DEFAULT_M_MAX = 2


def _legendre_norm(l: int, sin_lat: jax.Array) -> jax.Array:
    """Normalized real Legendre polynomial ``P_l(sin(lat))``.

    Uses Bonnet's recurrence to evaluate ``P_l`` then rescales so the
    resulting basis function has unit area-weighted L^2 norm on the
    sphere::

        (1/2) integral_{-1}^{1} P_l(x)^2 dx = 1/(2l+1)

    so dividing by ``1/sqrt(2l+1)`` gives unit-norm.  We use this
    natural rescaling here so the raw coefficients have a roughly
    isotropic gradient scale.
    """
    P = jnp.ones_like(sin_lat)
    if l == 0:
        return P
    P_prev = P
    P = sin_lat
    for n in range(1, l):
        P_next = ((2 * n + 1) * sin_lat * P - n * P_prev) / (n + 1)
        P_prev = P
        P = P_next
    return jnp.sqrt(2 * l + 1) * P


def spatial_basis(
    grid,
    l_max: int = _DEFAULT_L_MAX,
    m_max: int = _DEFAULT_M_MAX,
) -> jax.Array:
    """Build the real Legendre x Fourier lat-lon basis.

    Returns
    -------
    array, shape ``(n_basis, n_lat, n_lon)``
        Stacked basis functions.  Layout:
            [ P_0, P_1, ..., P_{l_max},
              cos(lon), sin(lon), P_1*cos(lon), P_1*sin(lon),
              cos(2*lon), sin(2*lon), P_1*cos(2*lon), P_1*sin(2*lon),
              ...
              cos(m_max*lon), sin(m_max*lon),
              P_1*cos(m_max*lon), P_1*sin(m_max*lon) ]
        Each function is normalized by ``sqrt(2l+1)`` (Legendre part)
        so coefficients have a roughly uniform magnitude scale.
    """
    sin_lat = jnp.sin(grid.lat2d).astype(jnp.float64)  # (n_lat, n_lon)
    lon = grid.lon2d.astype(jnp.float64)

    fns: list[jax.Array] = []
    # Zonal (m=0) Legendre modes 0..l_max
    for l in range(l_max + 1):
        fns.append(_legendre_norm(l, sin_lat))
    # Longitude modes m=1..m_max paired with l=0 (constant in lat) and
    # l=1 (P_1(sin_lat) = sqrt(3)*sin_lat).
    for m in range(1, m_max + 1):
        cos_m = jnp.cos(m * lon)
        sin_m = jnp.sin(m * lon)
        fns.append(cos_m)
        fns.append(sin_m)
        P1 = _legendre_norm(1, sin_lat)
        fns.append(P1 * cos_m)
        fns.append(P1 * sin_m)
    return jnp.stack(fns, axis=0)


def n_basis(l_max: int = _DEFAULT_L_MAX, m_max: int = _DEFAULT_M_MAX) -> int:
    """Number of basis functions for the given truncation."""
    return (l_max + 1) + 4 * m_max


# ----------------------------------------------------------------------
# Learnable spatial field
# ----------------------------------------------------------------------


class SpatialField(eqx.Module):
    """Learnable lat-lon field via a low-rank Legendre x Fourier basis.

    ``evaluate(grid, land_mask=None)`` returns ``(n_lat, n_lon)``::

        z(lat, lon) = sum_k coeffs[k] * basis_k(lat, lon)
        z_bounded   = tanh(z)                             # in (-1, 1)
        if transform == "log_perturb":
            value = f_0 * exp(scale * z_bounded)          # multiplicative
        else:  # "shift"
            value = f_0 + scale * z_bounded               # additive

    The ``tanh`` keeps coefficients well-behaved under gradient
    descent and bounds the field range to ``[f_0 * exp(-scale),
    f_0 * exp(+scale)]`` (multiplicative) or ``[f_0 - scale, f_0 +
    scale]`` (additive).

    Parameters
    ----------
    coeffs : jax.Array
        Raw learnable coefficients, shape ``(n_basis,)``.
    f_0 : float
        Baseline (global) value.
    scale : float
        Maximum log-perturbation magnitude (for ``log_perturb``) or
        maximum additive perturbation (for ``shift``).
    transform : {"log_perturb", "shift"}
        Range mapping.  Use ``"log_perturb"`` for strictly positive
        quantities (Cd, z0, etc.); ``"shift"`` for bounded quantities
        already centered (albedo, emissivity).
    l_max, m_max : int
        Basis truncation (default 4 / 2 -> 13 coeffs).
    """

    coeffs: jax.Array
    f_0: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    transform: Literal["log_perturb", "shift"] = eqx.field(static=True)
    l_max: int = eqx.field(static=True)
    m_max: int = eqx.field(static=True)

    @staticmethod
    def from_defaults(
        f_0: float,
        scale: float,
        transform: Literal["log_perturb", "shift"] = "log_perturb",
        l_max: int = _DEFAULT_L_MAX,
        m_max: int = _DEFAULT_M_MAX,
        dtype: jnp.dtype = jnp.float32,
        init_std: float = 0.0,
        key: jax.Array | None = None,
    ) -> "SpatialField":
        """Build a spatial field with zero or small-random initial coefs.

        ``init_std`` controls the standard deviation of an isotropic
        Gaussian initialization in coefficient space.  ``init_std=0``
        (default) keeps the spatial field equal to the baseline
        ``f_0`` everywhere -- useful as a sanity-check fallback and
        for the AIMIP regression that the spatial mode reduces to the
        scalar mode at zero coefficients.  Non-zero ``init_std``
        breaks the symmetry of zero gradient at zero coefficients
        and lets the optimizer explore the spatial degrees of
        freedom from step one.
        """
        nb = n_basis(l_max, m_max)
        if init_std > 0.0:
            if key is None:
                key = jax.random.PRNGKey(0)
            coeffs = init_std * jax.random.normal(key, (nb,), dtype=dtype)
        else:
            coeffs = jnp.zeros(nb, dtype=dtype)
        return SpatialField(
            coeffs=coeffs,
            f_0=float(f_0),
            scale=float(scale),
            transform=transform,
            l_max=l_max,
            m_max=m_max,
        )

    def evaluate(
        self,
        grid,
        land_mask: jax.Array | None = None,
        f_0_override: "jax.Array | float | None" = None,
    ) -> jax.Array:
        """Synthesize the lat-lon field from current coefficients.

        Parameters
        ----------
        grid : GaussianGrid
        land_mask : array or None
            Optional soft (0..1) land mask.  When provided, the
            spatial value applies only on land columns; ocean
            columns fall back to the baseline ``f_0``.
        f_0_override : scalar JAX array or float or None
            When non-None, replaces the static ``self.f_0`` for this
            evaluation.  Lets AIMIP plumb the *trained* scalar
            value (sigmoid-bounded leaf in ``AIMIPClassicalParams``)
            as the baseline so the scalar gradient flows through
            the spatial field's gradient and the bias-penalty loss
            can move the scalar knob.  Without this, the static
            ``self.f_0`` is the only baseline and the scalar trained
            leaf is effectively bypassed under
            ``aimip_spatial_surface: true``.
        """
        basis = spatial_basis(grid, self.l_max, self.m_max)
        z = jnp.einsum("k,kij->ij", self.coeffs.astype(basis.dtype), basis)
        z_bounded = jnp.tanh(z)
        f_0_eff = self.f_0 if f_0_override is None else f_0_override
        if self.transform == "log_perturb":
            value = f_0_eff * jnp.exp(self.scale * z_bounded)
        elif self.transform == "shift":
            value = f_0_eff + self.scale * z_bounded
        else:
            raise ValueError(
                f"Unknown transform {self.transform!r} "
                "(expected 'log_perturb' or 'shift')."
            )
        if land_mask is not None:
            land = land_mask.astype(value.dtype)
            value = value * land + f_0_eff * (1.0 - land)
        return value


# ----------------------------------------------------------------------
# Bundle of AIMIP-spatial surface fields
# ----------------------------------------------------------------------


class SpatialFieldSpec(NamedTuple):
    """Static spec for one spatial surface knob."""
    f_0: float
    scale: float
    transform: Literal["log_perturb", "shift"]


# Per-field defaults.  The ``scale`` parameter bounds the
# perturbation magnitude:
#   - log_perturb fields: value in ``[f_0 * exp(-scale), f_0 * exp(+scale)]``.
#   - shift fields:       value in ``[f_0 - scale, f_0 + scale]``.
#
# v6 (2026-05-18): shift-transform scales tightened after the v5 run
# at T11 L8 produced a +1.07 K warm T bias driven by the surface
# energy-balance fields over-correcting (was -2.74 K cold at the T21
# baseline; the per-group LR x5 plus 15-epoch budget pushed them
# past the bias-zero crossing).  Narrower bounds on
# ``sfc_emissivity`` / ``sfc_albedo`` constrain the optimizer's
# spatial perturbation to the physically plausible cold-side
# correction range.
# Log-perturb fields (Cd, Ch, z0) drive momentum and BL physics, not
# the radiation budget, so their scales are left alone.
# ``albedo_ocean`` / ``albedo_ice`` were removed in the 2026-06
# dead-code audit: their evaluated fields were never consumed by
# ``make_aimip_classical_spectral_physics``.
_FIELD_SPECS: dict[str, SpatialFieldSpec] = {
    "Cd_neutral":     SpatialFieldSpec(f_0=1.5e-3, scale=0.7,  transform="log_perturb"),
    "Ch_neutral":     SpatialFieldSpec(f_0=1.5e-3, scale=0.7,  transform="log_perturb"),
    "z0":             SpatialFieldSpec(f_0=1.0e-4, scale=2.3,  transform="log_perturb"),
    "sfc_emissivity": SpatialFieldSpec(f_0=0.95,   scale=0.02, transform="shift"),
    "sfc_albedo":     SpatialFieldSpec(f_0=0.20,   scale=0.05, transform="shift"),
}


SPATIAL_FIELD_NAMES = tuple(_FIELD_SPECS.keys())


class AIMIPSpatialSurfaceParams(eqx.Module):
    """Bundle of learnable spatial surface and surface-radiation fields.

    Holds a :class:`SpatialField` for each name in
    :data:`SPATIAL_FIELD_NAMES`.  ``evaluate`` returns a dict mapping
    name -> ``(n_lat, n_lon)`` array suitable for substitution into
    :class:`SurfaceLayerConfig` and :class:`GrayRadiationConfig`.
    """

    fields: dict[str, SpatialField]

    @staticmethod
    def from_defaults(
        l_max: int = _DEFAULT_L_MAX,
        m_max: int = _DEFAULT_M_MAX,
        dtype: jnp.dtype = jnp.float32,
        init_std: float = 0.0,
        key: jax.Array | None = None,
    ) -> "AIMIPSpatialSurfaceParams":
        """Build the AIMIP spatial-surface bundle.

        Each :class:`SpatialField` receives an independently-keyed
        random init when ``init_std > 0``; the key is split per field
        so swapping the field set leaves earlier inits stable across
        runs that touch different field names.
        """
        if init_std > 0.0 and key is None:
            key = jax.random.PRNGKey(0)
        keys = (
            jax.random.split(key, len(_FIELD_SPECS))
            if (init_std > 0.0 and key is not None) else (None,) * len(_FIELD_SPECS)
        )
        return AIMIPSpatialSurfaceParams(
            fields={
                name: SpatialField.from_defaults(
                    f_0=spec.f_0,
                    scale=spec.scale,
                    transform=spec.transform,
                    l_max=l_max,
                    m_max=m_max,
                    dtype=dtype,
                    init_std=init_std,
                    key=keys[i],
                )
                for i, (name, spec) in enumerate(_FIELD_SPECS.items())
            }
        )

    def evaluate(
        self,
        grid,
        land_mask: jax.Array | None = None,
        baselines: dict[str, "jax.Array | float"] | None = None,
    ) -> dict[str, jax.Array]:
        """Evaluate all spatial fields on the grid.

        ``baselines`` is an optional dict mapping field-name to a
        scalar (or array) that overrides the static ``f_0`` for that
        field.  AIMIP passes the trained sigmoid-bounded scalar
        leaves here so the spatial fields perturb around the
        trainable scalar baseline; missing names fall back to the
        per-field static ``f_0``.
        """
        baselines = baselines or {}
        return {
            name: field.evaluate(
                grid, land_mask=land_mask,
                f_0_override=baselines.get(name),
            )
            for name, field in self.fields.items()
        }

    def n_trainable(self) -> int:
        """Total trainable coefficients across all spatial fields."""
        return sum(f.coeffs.size for f in self.fields.values())


def land_mask_from_phis(
    phis: jax.Array,
    *,
    smooth: bool = True,
    sharpness: float = 1.0e-2,
) -> jax.Array:
    """Derive a (soft) land mask from surface geopotential.

    ``phis = g * z_s`` so positive ``phis`` indicates surface above sea
    level (land).  For differentiability we default to a smooth
    sigmoid in ``phis`` with a sharpness chosen so the transition zone
    is roughly one model layer wide (~10 m elevation).

    Parameters
    ----------
    phis : jax.Array
        Surface geopotential, shape ``(n_lat, n_lon)`` [m^2/s^2].
    smooth : bool
        If True, use a sigmoid (differentiable).  If False, use a
        hard step (non-differentiable).
    sharpness : float
        Sigmoid sharpness in 1/(m^2/s^2).  At ``sharpness=1e-2`` the
        transition width in elevation is ~10 m (since
        d(sigmoid)/d(phis)|_0 = sharpness/4 and phis = g*z_s).

    Returns
    -------
    jax.Array
        Land fraction in [0, 1], shape ``(n_lat, n_lon)``.
    """
    if smooth:
        return jax.nn.sigmoid(sharpness * phis)
    return jnp.where(phis > 0.0, 1.0, 0.0)


def grid_with_zm_land_fraction(grid, convection_scheme: str):
    """``grid`` carrying the column land fraction Zhang-McFarlane needs.

    ZM picks its autoconversion coefficient per column from the land fraction
    and refuses to run without one; the AIMIP Gaussian grid carries none.  The
    fraction is the ERA5 land-sea mask of the AIMIP store, regridded exactly as
    the WB classical arm feeds it to ZM.  Not ``phis > 0``: on the smoothed
    orography that marked 72 % of T21 columns as land.  Any other convection
    scheme gets ``grid`` back unchanged, with no store access.
    """
    if convection_scheme != "zhang_mcfarlane":
        return grid
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice
    from legoesm.training.scale_build import prescribed_surface_planes

    sl = load_era5_slice(TrainingERA5Config(load_land_frac=True), 0)
    land = prescribed_surface_planes(sl, grid)["land_frac"]
    return grid._replace(land_frac=land.reshape(-1))
