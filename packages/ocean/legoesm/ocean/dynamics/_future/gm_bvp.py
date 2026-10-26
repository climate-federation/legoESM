"""Ferrari et al. (2010) boundary-value problem for the GM eddy streamfunction.

PARKED in ``_future/`` (ponytail #11, user-approved 2026-10-02): not wired —
no production driver, factory or registry imports this module, and its tests
are skipped.  Wire it into production (moving it back) or delete it.

What this replaces
------------------
The DIAGNOSTIC GM closure sets the eddy streamfunction pointwise from the
local neutral slope, ``Gamma = kappa * S``, and then relies on slope tapering
to keep it finite where the stratification vanishes -- in the mixed layer and
against the bottom, exactly where the slope is largest and least meaningful.

Ferrari, Griffies, Nurser & Vallis (2010) instead make ``Gamma`` the SOLUTION
of a vertical boundary-value problem,

    c^2 d2(Gamma)/dz2 - N^2 Gamma = (g / rho_0) * grad_h(rho) * kappa,
    Gamma = 0 at the surface and at the bottom,

so the streamfunction satisfies its boundary conditions by construction
rather than by tapering, and the elliptic operator spreads the eddy transport
over a vertical scale ``c / N`` instead of leaving it pointwise.  Where the
stratification is strong the second-derivative term is negligible and the
solution collapses to the diagnostic limit: with ``N^2 Gamma = -(g/rho_0)
grad_h(rho) kappa`` and ``S = -grad_h(rho) / d(rho)/dz``, ``N^2 = -(g/rho_0)
d(rho)/dz``, the right-hand side is ``-N^2 kappa S`` and ``Gamma -> kappa S``.
That limit is pinned by a test.

Provenance
----------
Transcribed from FESOM2's ``oce_fer_gm.F90`` (``fer_solve_Gamma``,
``fer_gamma2vel``, and the ``cm`` block of ``init_Redi_GM``), which is the
implementation legoESM is being compared against; the discretisation,
boundary rows, right-hand side averaging and wave-speed estimate follow it
term for term.  See ``docs/ocean/fidelity/fesom2_gap_analysis.md``.

Reference
---------
Ferrari, R., S. M. Griffies, A. J. G. Nurser and G. K. Vallis (2010): A
boundary-value problem for the parameterized mesoscale eddy transport.
Ocean Modelling, 32, 143-156.

Layout
------
Layer quantities carry ``nlev`` in the trailing axis; interface quantities
carry ``nlev + 1``.  ``Gamma`` and ``N^2`` live on INTERFACES, the horizontal
density gradient and the layer thicknesses on LAYERS -- the same staggering
FESOM uses, and the reason the right-hand side averages the two layers that
straddle an interface.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve_batched

__physics_contract__ = {
    "summary": (
        "Ferrari et al. (2010) boundary-value problem for the Gent-McWilliams "
        "eddy streamfunction: c^2 Gamma_zz - N^2 Gamma = (g/rho_0) grad_h(rho) "
        "kappa with Gamma = 0 at both vertical boundaries, replacing the "
        "diagnostic Gamma = kappa*S."
    ),
    "inputs": {
        "sigma_h": "kg/m^4 (horizontal gradient of potential density)",
        "N2": "1/s^2",
        "cfg.kappa_gm": "m^2/s",
        "c_wave": "m/s",
    },
    "outputs": {
        "Gamma": "m^2/s (eddy streamfunction, per horizontal direction)",
        "u_bolus": "m/s",
    },
    "sign_convention": (
        "z is POSITIVE UP; layer thicknesses are positive. Gamma is the "
        "streamfunction whose vertical derivative is the bolus velocity, "
        "u* = d(Gamma_x)/dz, so a positive d(Gamma)/dz is an eastward bolus "
        "flow. The right-hand side carries the sign of grad_h(rho), which for "
        "a density surface sloping up to the north gives a northward eddy "
        "transport that flattens it -- the GM sign convention."
    ),
    # Tracer variance is not created: the transport is the curl of a
    # streamfunction that VANISHES at both boundaries, so the bolus velocity is
    # non-divergent in the vertical integral and moves no net volume through the
    # column. The zero boundary condition is what makes that exact here, where
    # the diagnostic form relies on tapering.
    "conserves": ["tracer", "volume"],
    # The tridiagonal solve is thomas_solve_batched (custom VJP, cuSPARSE on
    # GPU); dry columns get identity rows rather than a data-dependent branch,
    # so shapes are static and gradients finite.
    "differentiable": True,
    "reference": "Ferrari, Griffies, Nurser & Vallis (2010), Ocean Modelling 32, 143-156",
    "idealized_test": (
        "tests/ocean/unit/test_gm_bvp.py: the strong-stratification limit "
        "recovers Gamma = kappa*S; the boundary rows hold Gamma = 0; the "
        "assembled system is checked against a dense matrix solve."
    ),
}

__param_spec__ = {
    "GMBVPConfig": {
        "scheme_key": "ocean.lat.gm_bvp",
        "excluded": {
            "n2_floor": "numerics: regulariser on the BVP diagonal [1/s^2] "
                        "(a solver floor keeping an unstratified column non-singular, "
                        "not a closure coefficient)",
        },
        "params": {
            "c_min": {
                "units": "m/s", "bounds": (0.01, 2.0), "tunable_tier": 2,
                "transform": "softplus", "category": "lateral_mixing",
                "reference": "FESOM2 namelist.oce K_GM_cmin", "shape": None,
            },
            "mode_number": {
                "units": "1", "bounds": (1.0, 6.0), "tunable_tier": 2,
                "transform": "softplus", "category": "lateral_mixing",
                "reference": "FESOM2 namelist.oce K_GM_cm", "shape": None,
            },
        },
    }
}

# --- Ferrari et al. (2010) BVP ------------------------------------------------
#: Diagonal regulariser.  FESOM uses max(N^2, 1e-8) so an unstratified column
#: cannot produce a singular row; without it b -> -(a + c) and the operator
#: loses its zeroth-order term.
_N2_FLOOR_DEFAULT: float = 1.0e-8


class GMBVPConfig(NamedTuple):
    """Ferrari (2010) streamfunction BVP settings.

    Defaults follow FESOM2's CORE2 ``namelist.oce``: ``K_GM_cmin = 0.1``,
    ``K_GM_cm = 3.0``.
    """
    #: Floor on the baroclinic wave speed [m/s] (FESOM ``K_GM_cmin``).
    c_min: float = 0.1
    #: Which baroclinic mode sets the vertical scale (FESOM ``K_GM_cm``).
    #: The WKB first-mode speed is divided by this.
    mode_number: float = 3.0
    #: Floor on N^2 in the BVP diagonal [1/s^2].
    n2_floor: float = _N2_FLOOR_DEFAULT


def baroclinic_wave_speed(N2_interface, dz_layer, cfg: GMBVPConfig):
    """WKB baroclinic gravity-wave speed [m/s].

    ``c = (1/pi) * integral(N dz) / mode_number``, floored at ``c_min`` --
    FESOM's ``cm`` block.  The integral is trapezoidal over layers using the
    two interface values that bound each layer, and ``N^2`` is clipped at zero
    before the square root so a statically unstable column contributes nothing
    rather than a NaN.

    Parameters
    ----------
    N2_interface : (..., nlev + 1) buoyancy frequency squared [1/s^2].
    dz_layer : (..., nlev) layer thicknesses [m], positive.

    Returns
    -------
    (...,) wave speed [m/s].
    """
    n = jnp.sqrt(jnp.maximum(N2_interface, 0.0))
    integrand = 0.5 * (n[..., :-1] + n[..., 1:])          # (..., nlev)
    c1 = jnp.sum(integrand * dz_layer, axis=-1) / jnp.pi
    return jnp.maximum(c1 / cfg.mode_number, cfg.c_min)


def _assemble(sigma_h_layer, N2_interface, kappa_interface, dz_layer,
              c_wave, wet_interface, cfg: GMBVPConfig):
    """Tridiagonal coefficients and RHS for ONE horizontal direction.

    Returns ``(a, b, c, d)``, each ``(..., nlev + 1)``, with the system on the
    trailing axis as :func:`thomas_solve_batched` requires.
    """
    c2 = jnp.asarray(c_wave)[..., None] ** 2               # (..., 1)

    # Interface spacing quantities.  h_above[k] / h_below[k] are the layer
    # thicknesses straddling interface k; dz_centre[k] is the distance between
    # the two layer CENTRES, which is what the second derivative divides by.
    h_above = dz_layer[..., :-1]                           # for interfaces 1..nlev-1
    h_below = dz_layer[..., 1:]
    dz_centre = 0.5 * (h_above + h_below)

    # Dry layers carry dz = 0, so the divisions below would produce inf (and,
    # once multiplied by a masked-out zero, NaN gradients).  Floor the
    # DENOMINATORS only: every row that uses a floored value is forced to an
    # identity row a few lines down, so the substituted 1.0 never reaches the
    # solution -- it only keeps inf/NaN out of the array and out of the VJP.
    h_above_s = jnp.where(h_above > 0.0, h_above, 1.0)
    h_below_s = jnp.where(h_below > 0.0, h_below, 1.0)
    dz_centre_s = jnp.where(dz_centre > 0.0, dz_centre, 1.0)

    a_in = c2 / (h_above_s * dz_centre_s)
    c_in = c2 / (h_below_s * dz_centre_s)
    n2_in = jnp.maximum(N2_interface[..., 1:-1], cfg.n2_floor)
    b_in = -a_in - c_in - n2_in

    # RHS on interior interfaces: (g/rho_0) * mean of the two straddling
    # layers' horizontal density gradient * kappa at the interface.
    r = constants.g / constants.rho_ocean
    sig_in = 0.5 * (sigma_h_layer[..., :-1] + sigma_h_layer[..., 1:])
    d_in = r * sig_in * kappa_interface[..., 1:-1]

    # Pad the boundary rows back on: Gamma = 0 at the top and bottom
    # interfaces (identity row, zero RHS).
    zero = jnp.zeros_like(a_in[..., :1])
    one = jnp.ones_like(zero)
    a = jnp.concatenate([zero, a_in, zero], axis=-1)
    c = jnp.concatenate([zero, c_in, zero], axis=-1)
    b = jnp.concatenate([one, b_in, one], axis=-1)
    d = jnp.concatenate([zero, d_in, zero], axis=-1)

    # Dry interfaces become identity rows too, AND so does the deepest WET
    # interface -- that interface IS the seafloor, i.e. the bottom boundary
    # where Gamma = 0 is imposed.  Padding identity onto index ``nlev`` alone
    # only gets the bottom BC right for a full-depth column; on a partial
    # column the seafloor row would otherwise be assembled as an INTERIOR row
    # whose sub-diagonal reaches into the dry region, which both violates
    # Gamma(bottom) = 0 and leaves a spurious bolus velocity in the first dry
    # layer (codex counterexample: wet = [T,T,T,F,F] gave Gamma[2] = 3.83e-2
    # instead of 0, and a wet-column integral of -3.83e-2 instead of 0).
    #
    # A branch here would be data-dependent; masking keeps the shape static and
    # the gradient finite, and an identity row returns Gamma = 0 which is what
    # the caller wants anyway.
    wet = wet_interface
    # ``below[k]`` = "interface k+1 is also wet"; False past the array end, so
    # the deepest wet interface is exactly ``wet & ~below``.
    below = jnp.concatenate(
        [wet[..., 1:], jnp.zeros_like(wet[..., :1])], axis=-1)
    interior = wet & below
    a = jnp.where(interior, a, 0.0)
    c = jnp.where(interior, c, 0.0)
    b = jnp.where(interior, b, 1.0)
    d = jnp.where(interior, d, 0.0)
    return a, b, c, d


def solve_gm_streamfunction(
    sigma_x_layer,
    sigma_y_layer,
    N2_interface,
    kappa_interface,
    dz_layer,
    *,
    wet_interface=None,
    cfg: GMBVPConfig = GMBVPConfig(),
    c_wave=None,
):
    """Solve the Ferrari (2010) BVP for both streamfunction components.

    Parameters
    ----------
    sigma_x_layer, sigma_y_layer : (..., nlev)
        Horizontal gradient of potential density [kg/m^4], on LAYERS.
    N2_interface : (..., nlev + 1)
        Buoyancy frequency squared [1/s^2], on INTERFACES.
    kappa_interface : (..., nlev + 1)
        GM thickness diffusivity [m^2/s], on INTERFACES.
    dz_layer : (..., nlev)
        Layer thicknesses [m], positive.
    wet_interface : (..., nlev + 1) bool or None
        True where the interface is in the water column.  ``None`` treats the
        whole column as wet.  Dry interfaces return ``Gamma = 0``.
    cfg : GMBVPConfig
    c_wave : (...,) or None
        Baroclinic wave speed [m/s].  Computed from ``N2_interface`` via
        :func:`baroclinic_wave_speed` when omitted.

    Returns
    -------
    (Gamma_x, Gamma_y), each ``(..., nlev + 1)`` [m^2/s], zero at the top and
    bottom boundaries by construction.
    """
    if wet_interface is None:
        wet_interface = jnp.ones_like(N2_interface, dtype=bool)
    if c_wave is None:
        c_wave = baroclinic_wave_speed(N2_interface, dz_layer, cfg)

    out = []
    for sigma in (sigma_x_layer, sigma_y_layer):
        a, b, c, d = _assemble(sigma, N2_interface, kappa_interface, dz_layer,
                               c_wave, wet_interface, cfg)
        out.append(thomas_solve_batched(a, b, c, d))
    return out[0], out[1]


def bolus_velocity(Gamma_x, Gamma_y, dz_layer):
    """Layer bolus velocity from the streamfunction, ``u* = d(Gamma)/dz``.

    ``z`` is positive up and ``Gamma`` is indexed from the surface down, so
    the difference is ``(Gamma[k] - Gamma[k+1]) / dz[k]`` -- FESOM's
    ``fer_gamma2vel`` (its node average over a triangle is grid-specific and
    is not part of this).

    Because ``Gamma`` vanishes at both boundaries, the depth integral of the
    bolus velocity over a column telescopes to zero: the parameterisation
    moves no net volume, which is the property the diagnostic form only gets
    approximately.

    Returns
    -------
    (u_bolus, v_bolus), each ``(..., nlev)`` [m/s].
    """
    # A dry layer has dz = 0 and Gamma = 0 at both of its interfaces, so the
    # quotient is 0/0 -> NaN.  Floor the denominator: the numerator is already
    # exactly zero there, so the velocity is zero for the right reason and the
    # NaN never enters the column (or its gradient).
    dz_s = jnp.where(dz_layer > 0.0, dz_layer, 1.0)
    u = (Gamma_x[..., :-1] - Gamma_x[..., 1:]) / dz_s
    v = (Gamma_y[..., :-1] - Gamma_y[..., 1:]) / dz_s
    return u, v
