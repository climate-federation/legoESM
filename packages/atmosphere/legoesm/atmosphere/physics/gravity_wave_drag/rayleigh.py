"""Rayleigh friction gravity wave drag.

Simplest GWD parameterization: applies linear drag proportional to wind
speed in the boundary layer and an upper-atmosphere sponge layer.

Follows the same pattern as held_suarez.py for sigma-based drag profiles.

Faithfulness to Held & Suarez (1994)
------------------------------------
FAITHFUL to HS94 — the boundary-layer MOMENTUM tendency:
  * ``k_bl = k_max · max(0, (σ − σ_b)/(1 − σ_b))`` applied to BOTH wind
    components (``du_dt = −k_bl·u``, ``dv_dt = −k_bl·v``) is EXACTLY HS94's
    low-level Rayleigh friction ``k_v = k_f · max(0, (σ − σ_b)/(1 − σ_b))``
    (their momentum-drag eq.); it matches the model's own ``held_suarez.py``
    forcing term. The DEFAULT constants are the HS94 values: ``k_max = 1/day =
    k_f`` and ``σ_b = 0.7`` (== ``held_suarez.K_F`` / ``held_suarez.SIGMA_B``).
ADDITIONS / DEPARTURES from HS94:
  * **the BL frictional HEATING is NOT HS94**: the KE removed by the BL drag
    returns as heat ``dT_dt = -(u du_bl + v dv_bl)/c_pd``. HS94 has no
    KE-compensating heating (its temperature forcing is a separate Newtonian
    relaxation), so the FAITHFUL claim for the BL part is the MOMENTUM tendency.
  * the upper **SPONGE is NOT HS94**; it is a port of Isca v1.0
    ``damping_driver.f90`` subroutine ``rayleigh`` (Polvani & Kushner 2002
    sponge): ``k_sp = sponge_k ((p_sb - p)/p_sb)^2`` for ``p < p_sb =
    sponge_pbottom_Pa``, applied only on levels ``1..nlev_rayfric`` where
    ``nlev_rayfric`` is the level whose reference pressure (sigma *
    ``constants.p_atm_std``, Isca's ``PSTD_MKS`` profile) is nearest ``2 p_sb``.
    Heating = Isca's ``do_conserve_energy`` time-centred form ``-((u + dt/2
    du_sp) du_sp + (v + dt/2 dv_sp) dv_sp)/c_pd``, always on here (energy
    conservation). Defaults ``p_sb = 50 Pa`` (Isca / PK02) and ``sponge_k =
    1/(0.5 day)``. With 50 Pa the sponge is INERT on any grid whose top level
    sits below 50 Pa (e.g. CAM L32, top full level ~364 Pa).
  * DEPARTURE (smoothing, RULE 5): ``max(0, (p_sb - p)/p_sb)`` uses the C1
    ``positive_part_c1`` at ``sponge_sharpness`` (width 1/sharpness = 0.01);
    ``sponge_sharpness = inf`` is Isca's hard form (oracle test).
  * ``conserves = ["energy"]``: Rayleigh friction is a direct drag with no wave;
    the drag and sponge convert resolved KE to heat, both tracked. The heating
    returns the KE the step removes (time-centred in the sponge), so
    ``c_pd sum(rho dT_dt dz) = eps_gwd - dt/2 sum(rho (du_sp^2 + dv_sp^2) dz)``
    with ``eps_gwd`` the instantaneous KE loss of the GWDOutput contract.
    Momentum is NOT conserved (a sink).
  * the level cap is a hard integer switch (Isca's), piecewise constant in
    ``p_sb`` and ``p_s``: its gradient is zero, only the cutoff is smoothed.
NUMERICS: the ``k_bl`` UPPER clip at 1 is a redundant guard (``(σ−σ_b)/(1−σ_b)
≤ 1`` for any physical ``σ ≤ 1``); the ``p_sfc`` floor at 1 Pa.
Oracle: ``tests/atmosphere/test_rayleigh_sponge_isca_oracle.py``.

References
----------
- Held, I. M., & Suarez, M. J. (1994). A proposal for the intercomparison of
  the dynamical cores of atmospheric general circulation models. Bull. Amer.
  Meteor. Soc., 75, 1825-1830 (the σ_b/k_f low-level Rayleigh friction).
- Polvani, L. M., & Kushner, P. J. (2002). Tropospheric response to
  stratospheric perturbations in a relatively simple general circulation
  model. Geophys. Res. Lett., 29, 1114 (the sponge); code: Isca v1.0
  (5868cfd) src/atmos_param/damping_driver/damping_driver.f90 ``rayleigh``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
from legoesm.core.smooth import positive_part_c1

# Machine-checked scheme contract (see tests/test_physics_contracts.py and
# docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md). The architect pins
# units/signs/conservation/reference; the body must honour it.
__physics_contract__ = {
    "summary": (
        "Rayleigh-friction gravity-wave drag: a linear momentum sink in the "
        "boundary layer plus the Polvani-Kushner (Isca damping_driver) sponge "
        "near the model top."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "Drag opposes the wind: du_dt = -k(sigma)*u (and v), so du_dt has the "
        "opposite sign to u; eps_gwd (frictional dissipation -> heating) is "
        "positive-definite."
    ),
    # Momentum is NOT conserved (a drag/sponge removes momentum to the surface /
    # absorbs it at the top), but total ENERGY is: the kinetic energy lost by the
    # mean flow is returned as frictional heating, dT_dt = -(u*du_dt + v*dv_dt)/c_pd
    # (eps_gwd = column-integrated KE loss). Unlike the c!=0 launched-SPECTRUM GWD
    # schemes (hines.py/prognostic_spectral.py = ["none"], whose propagating waves
    # carry off vertical energy flux the limiter discards) Rayleigh is a DIRECT
    # drag with NO wave — the drag + sponge convert RESOLVED mean-flow KE directly
    # to heat (both tracked), so the resolved-KE->heat closure IS the full energy
    # budget. (The c=0 orographic schemes mcfarlane.py/lindzen.py ALSO conserve
    # energy — a stationary wave carries zero energy flux.) (The sponge heating
    # the removed KE is a design choice; a physical radiation sponge would
    # remove the GW energy from the domain instead.) So conserved quantity = energy.
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Held & Suarez (1994) low-level sigma-drag Rayleigh friction "
        "(k_f, sigma_b); upper sponge = Isca v1.0 damping_driver rayleigh "
        "(Polvani & Kushner 2002)."
    ),
    "idealized_test": (
        "rest state (u=v=0) -> zero tendency; for u>0 the boundary-layer drag is "
        "non-positive and its magnitude ramps from 0 at sigma=sigma_b to k_max at "
        "the surface."
    ),
}


def rayleigh_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: RayleighConfig,
) -> GWDOutput:
    """Compute Rayleigh friction GWD tendencies.

    Parameters
    ----------
    u, v : jax.Array
        Wind components [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full, z_half : jax.Array
        Heights [m], shapes (ncol, nlev) and (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    lat : jax.Array
        Latitude [rad], shape (ncol,).
    dt : float
        Time step [s].
    config : RayleighConfig

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Sigma coordinate
    p_sfc = p_half[:, -1:]  # (ncol, 1)
    sigma = p_full / jnp.clip(p_sfc, 1.0, None)

    # Boundary layer drag: ramps from 0 at sigma_b to k_max at surface
    k_bl = config.k_max * jnp.clip(
        (sigma - config.sigma_b) / (1.0 - config.sigma_b), 0.0, 1.0
    )

    # Upper sponge: Isca v1.0 damping_driver ``rayleigh`` (Polvani & Kushner 2002).
    # Levels 1..nlev_rayfric only, nlev_rayfric = level whose reference pressure
    # (full levels at ps = PSTD_MKS) is nearest 2*sponge_pbottom (Fortran minloc,
    # first index == jnp.argmin); there k = rfactr*((p_sb - p)/p_sb)**2 for p < p_sb.
    p_sb = config.sponge_pbottom_Pa
    pref = sigma * constants.p_atm_std
    k_cap = jnp.argmin(jnp.abs(pref - 2.0 * p_sb), axis=1)[:, None]
    in_cap = jnp.arange(nlev)[None, :] <= k_cap
    frac = positive_part_c1((p_sb - p_full) / p_sb, config.sponge_sharpness)
    k_sponge = jnp.where(in_cap, config.sponge_k * frac**2, 0.0)

    du_bl, dv_bl = -k_bl * u, -k_bl * v
    du_sp, dv_sp = -k_sponge * u, -k_sponge * v
    du_dt = du_bl + du_sp
    dv_dt = dv_bl + dv_sp

    # Frictional heating. BL: -(u du + v dv)/c_pd. Sponge: Isca's time-centred
    # do_conserve_energy form -((u + dt/2 du) du + (v + dt/2 dv) dv)/c_pd, always on.
    dT_dt = (
        -(u * du_bl + v * dv_bl)
        - ((u + 0.5 * dt * du_sp) * du_sp + (v + 0.5 * dt * dv_sp) * dv_sp)
    ) / constants.c_pd

    # Column dissipation, GWDOutput contract (instantaneous KE loss). The column heating
    # c_pd*sum(rho dT dz) is smaller by dt/2*sum(rho (du_sp^2 + dv_sp^2) dz): the
    # time-centred sponge heating returns the KE the explicit step removes.
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
