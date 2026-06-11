"""Super-droplet initialization spectra (ERF ``SDInitialization`` port).

Constant-multiplicity samplers: every super-droplet gets the same multiplicity
``ξ = N_total/n_sd`` and its attribute (mass or dry radius) is drawn from the
*true* distribution by inverse-transform sampling — exactly ERF's
``SDMultiplicityType::constant`` path (and PySDM's ``ConstantMultiplicity``):

* ``sample_exponential_mass`` — ``m = m_min − Δ·ln(1−U)``, ``Δ = m_mean−m_min``
  (the Shima-2009 box initial spectrum);
* ``sample_lognormal_radius`` — truncated log-normal dry radius via the exact
  inverse CDF ``r = μ·exp(σ·√2·erfinv(2u−1))`` with ``u`` remapped to
  ``[CDF(r_min), CDF(r_max)]`` (ERF uses an erfinv *approximation* as a GPU
  workaround; we use the exact ``jax.scipy.special.erfinv`` — the intended
  distribution, not the workaround's bias).

State builders assemble ready :class:`SuperDropletState` ensembles:
``exponential_water_droplets`` (cloud water spectrum, e.g. the Golovin box)
and ``lognormal_aerosol_droplets`` (dry aerosol mode carried as
``solute_mass``, wet radius set near the dry size for the condensation solver
to equilibrate/activate).

ERF's ``SDMultiplicityType::sampled`` (uniform-in-log importance sampling with
PDF-weighted multiplicities, better tail coverage) is NOT ported yet — its
multiplicity renormalization lives in the particle-injection path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import erfinv

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState

__physics_contract__ = {
    "summary": (
        "Constant-multiplicity super-droplet initialization spectra: "
        "exponential-in-mass and truncated log-normal-in-radius inverse-CDF "
        "samplers and the corresponding SuperDropletState builders."
    ),
    "inputs": {
        "n_sd": "1",
        "n_total": "1 (represented droplets in the sampled volume/mass)",
        "mass_mean": "kg",
        "r_mean": "m",
        "geom_std": "1",
        "key": "1 (jax.random PRNG key)",
    },
    "outputs": {"multiplicity": "1", "radius": "m", "solute_mass": "kg"},
    "sign_convention": (
        "All sampled masses/radii are positive; multiplicities are the equal "
        "share n_total/n_sd; expected represented totals match the requested "
        "distribution moments."
    ),
    "conserves": ["none"],
    "differentiable": False,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF ERF_SDInitialization.H (constant-multiplicity path)",
    "idealized_test": (
        "Exponential sample mean -> mass_mean (MC tolerance); lognormal sample "
        "median -> r_mean and log-std -> ln(geom_std); truncated samples lie "
        "strictly within [r_min, r_max]; deterministic for a fixed key."
    ),
}

# (4/3)π — sphere volume prefactor (pure geometry).
_FOUR_THIRDS_PI = 4.0 / 3.0 * jnp.pi
_SQRT2 = jnp.sqrt(2.0)


def sample_exponential_mass(
    key: jax.Array,
    n_sd: int,
    mass_mean: float | jax.Array,
    mass_min: float | jax.Array = 0.0,
    dtype=jnp.float64,
) -> jax.Array:
    """Draw ``n_sd`` masses from ``m_min + Exp(mean = m_mean − m_min)`` [kg].

    Inverse transform ``m = m_min − Δ·ln(1−U)`` (ERF uses ``ln(u)`` with
    ``u ∈ (0,1]``; jax uniform is ``[0,1)`` so ``1−U`` gives the identical
    distribution without ``log(0)``).
    """
    if n_sd < 1:
        raise ValueError(f"n_sd must be >= 1, got {n_sd}")
    if not (float(mass_mean) > float(mass_min) >= 0.0):
        raise ValueError(
            f"need mass_mean > mass_min >= 0, got mean={mass_mean}, min={mass_min}")
    delta = jnp.asarray(mass_mean, dtype) - jnp.asarray(mass_min, dtype)
    U = random.uniform(key, (n_sd,), dtype=dtype)
    # log1p(-U) == log(1-U) to roundoff (~1e-15 relative — NOT bit-identical
    # to the naive form, but more accurate near U=0).
    return jnp.asarray(mass_min, dtype) - delta * jnp.log1p(-U)


def _lognormal_cdf(r, r_mean, sigma):
    from jax.scipy.special import erf
    return 0.5 * (1.0 + erf(jnp.log(r / r_mean) / (sigma * _SQRT2)))


def sample_lognormal_radius(
    key: jax.Array,
    n_sd: int,
    r_mean: float | jax.Array,
    geom_std: float | jax.Array,
    r_min: float | jax.Array | None = None,
    r_max: float | jax.Array | None = None,
    dtype=jnp.float64,
) -> jax.Array:
    """Draw ``n_sd`` radii from a (truncated) log-normal [m].

    ``r = r_mean·exp(σ·√2·erfinv(2u−1))`` with ``σ = ln(geom_std)`` and ``u``
    remapped to ``[CDF(r_min), CDF(r_max)]`` when truncation bounds are given
    (the ERF truncated-inverse-CDF construction, with the exact erfinv).
    ``r_mean`` is the *median* (geometric mean) of the distribution.
    ``geom_std`` must be > 1 (a degenerate σ=0 'log-normal' has no consistent
    truncated-CDF inverse).
    """
    if n_sd < 1:
        raise ValueError(f"n_sd must be >= 1, got {n_sd}")
    if not float(r_mean) > 0.0:
        raise ValueError(f"r_mean must be > 0, got {r_mean}")
    if not float(geom_std) > 1.0:
        raise ValueError(f"geom_std must be > 1, got {geom_std}")
    if r_min is not None and not float(r_min) > 0.0:
        raise ValueError(f"r_min must be > 0, got {r_min}")
    if r_min is not None and r_max is not None and not float(r_max) > float(r_min):
        raise ValueError(f"need r_max > r_min, got [{r_min}, {r_max}]")
    r_mean = jnp.asarray(r_mean, dtype)
    sigma = jnp.log(jnp.asarray(geom_std, dtype))
    u = random.uniform(key, (n_sd,), dtype=dtype)
    cdf_lo = _lognormal_cdf(jnp.asarray(r_min, dtype), r_mean, sigma) \
        if r_min is not None else jnp.asarray(0.0, dtype)
    cdf_hi = _lognormal_cdf(jnp.asarray(r_max, dtype), r_mean, sigma) \
        if r_max is not None else jnp.asarray(1.0, dtype)
    u_trunc = cdf_lo + u * (cdf_hi - cdf_lo)
    # Keep erfinv off the exact ±1 poles (open-interval guard).
    u_trunc = jnp.clip(u_trunc, 1.0e-15, 1.0 - 1.0e-15)
    z = erfinv(2.0 * u_trunc - 1.0) * _SQRT2
    return r_mean * jnp.exp(sigma * z)


def exponential_water_droplets(
    key: jax.Array,
    n_sd: int,
    n_total: float,
    mass_mean: float,
    dtype=jnp.float64,
) -> SuperDropletState:
    """Equal-multiplicity cloud-water ensemble with an exponential mass
    spectrum (the canonical Shima-2009 box initial condition).

    ``n_total`` is the number of real droplets represented (count in the box,
    or per kg for a unit parcel); each super-droplet gets ``ξ = n_total/n_sd``.
    """
    if not float(n_total) >= 0.0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    mass = sample_exponential_mass(key, n_sd, mass_mean, dtype=dtype)
    radius = jnp.cbrt(mass / (_FOUR_THIRDS_PI * constants.rho_water))
    o = jnp.ones((n_sd,), dtype=dtype)
    return SuperDropletState(
        multiplicity=o * (n_total / n_sd),
        radius=radius,
        solute_mass=jnp.zeros((n_sd,), dtype=dtype),
        active=o,
    )


def lognormal_aerosol_droplets(
    key: jax.Array,
    n_sd: int,
    n_total: float,
    r_dry_median: float,
    geom_std: float,
    solute_density: float,
    r_min: float | None = None,
    r_max: float | None = None,
    wet_radius_factor: float = 1.0,
    dtype=jnp.float64,
) -> SuperDropletState:
    """Equal-multiplicity dry-aerosol ensemble with a (truncated) log-normal
    dry-radius mode carried as ``solute_mass``.

    ``solute_mass = (4/3)π ρ_s r_dry³``; the initial wet radius is
    ``wet_radius_factor · r_dry`` (callers equilibrate to the haze radius via
    the condensation solver — see the activation-parcel validator — or start
    near-dry and let the first steps relax it).
    """
    if not float(n_total) >= 0.0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    if not float(solute_density) > 0.0:
        raise ValueError(f"solute_density must be > 0, got {solute_density}")
    if not float(wet_radius_factor) >= 1.0:
        raise ValueError(
            f"wet_radius_factor must be >= 1 (wet >= dry), got {wet_radius_factor}")
    r_dry = sample_lognormal_radius(key, n_sd, r_dry_median, geom_std,
                                    r_min=r_min, r_max=r_max, dtype=dtype)
    m_s = _FOUR_THIRDS_PI * solute_density * r_dry**3
    o = jnp.ones((n_sd,), dtype=dtype)
    return SuperDropletState(
        multiplicity=o * (n_total / n_sd),
        radius=wet_radius_factor * r_dry,
        solute_mass=m_s,
        active=o,
    )
