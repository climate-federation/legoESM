"""Small-planet (reduced-radius) grid helpers for Wedi-Smolarkiewicz tests.

Wedi & Smolarkiewicz (2009) introduced a "small-planet" framework in
which the Earth radius and rotation rate are simultaneously rescaled,

    R_eff = R / X,    Ω_eff = Ω · X,

with ``X = 125`` typical, so that an integration time of one Earth-day
on the small planet corresponds to ~12 days on the full Earth.  This
allows expensive idealised tests (Held-Suarez, Klemp 2015 supercell)
to be run at a fraction of the wall-clock cost while preserving the
non-dimensional Rossby and inertial timescales.

This module provides thin convenience wrappers around each grid
factory that apply the rescaling consistently.  Higher-level test
drivers (e.g. matrix-script ``run_climate_idealized``) compose these
helpers with the existing Held-Suarez forcing to deliver the
**WS09-SP** small-planet Held-Suarez entry of the Hughes (2026)
catalog.

Reference
---------
Wedi, N. P., & Smolarkiewicz, P. K. (2009). A framework for testing
global non-hydrostatic models. *QJRMS*, 135(639), 469-484.
"""

from __future__ import annotations

from legoesm import constants


# ---------------------------------------------------------------------------
# Default scaling factor (DCMIP / Wedi-Smolarkiewicz convention)
# ---------------------------------------------------------------------------

DEFAULT_SCALE_FACTOR = 125.0


def scaled_radius(factor: float = DEFAULT_SCALE_FACTOR) -> float:
    """Return the small-planet radius ``R_earth / factor`` [m]."""
    return float(constants.R_earth) / float(factor)


def scaled_omega(factor: float = DEFAULT_SCALE_FACTOR) -> float:
    """Return the small-planet rotation rate ``Ω · factor`` [rad/s]."""
    return float(constants.Omega) * float(factor)


# ---------------------------------------------------------------------------
# Grid factories with simultaneous (radius, Ω) rescaling
# ---------------------------------------------------------------------------


def make_small_planet_cubed_sphere(
    n: int,
    factor: float = DEFAULT_SCALE_FACTOR,
    **kwargs,
):
    """Cubed-sphere grid with R/Ω scaled by ``factor`` and ``factor·Ω``."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(
        n,
        radius=scaled_radius(factor),
        omega=scaled_omega(factor),
        **kwargs,
    )


def make_small_planet_gaussian(
    n_max: int,
    factor: float = DEFAULT_SCALE_FACTOR,
    **kwargs,
):
    """Gaussian (spectral) grid with R/Ω scaled by ``factor``.

    Both the radius and the Coriolis-baking rotation rate are
    rescaled per Wedi & Smolarkiewicz (2009).  The factory takes an
    ``omega`` keyword; we pass ``Ω · factor`` so spectral PE runs see
    the small-planet ``f = 2·(Ω·X)·sin(lat)``.
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    return create_gaussian_grid(
        n_max,
        radius=scaled_radius(factor),
        omega=scaled_omega(factor),
        **kwargs,
    )
