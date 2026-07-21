"""Monte-Carlo sampling kernels for the 3D ray tracer: phase functions,
scattering-frame rotation, and Lambertian surface reflection.

Pure JAX, differentiable-shaped (no Python control flow on traced values; uses
``jnp.where``/``lax`` selection). All directions are unit 3-vectors ``(ux,uy,uz)``
with ``z`` increasing upward.

Phase-1 fidelity choice: a single Henyey-Greenstein phase function with the
RRTMGP total asymmetry ``g`` is used for *all* scattering (gas+cloud+aerosol).
HG with ``g=0`` is exactly isotropic, so this is correct for the slab,
conservation, and two-stream-limit tests. Explicit Rayleigh (1+cos^2) gas
scattering is a documented Phase-1.1 refinement, not a skipped edge case.
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

# --- Henyey-Greenstein (numerics) ---
# Below this |g| the HG inversion is replaced by the isotropic limit to avoid the
# 1/(2g) singularity; HG -> isotropic as g -> 0 so the switch is continuous.
_G_ISOTROPIC_EPS = 1.0e-3
# Strict-interior cap on the asymmetry fed to the HG inverse CDF.  The mixture
# un-mixing ``g_cloud = g / (1 - rayleigh_frac)`` can reach exactly +1 under an
# inclusive clip, but ``henyey_greenstein_mu`` has ``s = (1-g^2)/(1-g+2gu) =
# 0/0`` at (g=+1, u=0) — and u=0 IS attainable from ``jax.random.uniform``'s
# [0, 1) range — which NaN-poisons the photon direction and (via jnp.where's
# both-branch evaluation) reverse-mode AD.  Keep g strictly inside (-1, 1);
# 1e-6 leaves ``1 - g`` well resolved in both float32 and float64.
_HG_G_MAX = 1.0 - 1.0e-6


def henyey_greenstein_mu(key: Array, g: Array) -> Array:
  """Sample the scattering-angle cosine ``mu = cos(theta)`` from the HG phase
  function with asymmetry ``g`` (per-photon scalar).

  ``mu = (1 + g^2 - s^2) / (2g)`` with ``s = (1-g^2)/(1-g+2g u)``; isotropic
  ``mu = 1 - 2u`` in the ``|g| < eps`` limit.
  """
  u = jax.random.uniform(key, dtype=g.dtype)
  # Clip g off the forward/backward singularity g=+-1 at the SOURCE so EVERY
  # caller is protected (the pure-HG ``scatter_direction`` path as well as the
  # mixed path): at exactly g=+-1 with u=0, s=(1-g^2)/(1-g+2gu)=0/0 -> NaN. At
  # g=_HG_G_MAX, u=0 the analytic limit is mu=-1 (finite).
  g = jnp.clip(g, -_HG_G_MAX, _HG_G_MAX)
  # safe_g keeps the |g|<eps branch finite: jnp.where evaluates BOTH branches,
  # so a raw 1/(2g) at g=0 would yield NaN and poison reverse-mode AD even though
  # the isotropic branch is selected. Use 0.5 (NOT +-1: g=+-1 gives s=0/0 at
  # u=0); any g with 0<|g|<1 keeps s and 1/(2g) finite for all u in [0,1).
  safe_g = jnp.where(jnp.abs(g) < _G_ISOTROPIC_EPS, 0.5, g)
  s = (1.0 - safe_g * safe_g) / (1.0 - safe_g + 2.0 * safe_g * u)
  mu_hg = (1.0 + safe_g * safe_g - s * s) / (2.0 * safe_g)
  mu_iso = 1.0 - 2.0 * u
  return jnp.where(jnp.abs(g) < _G_ISOTROPIC_EPS, mu_iso, mu_hg)


def rayleigh_mu(key: Array, dtype) -> Array:
  """Sample the scattering-angle cosine from the Rayleigh phase function
  ``p(mu) = (3/8)(1 + mu^2)`` (molecular/gas scattering).

  Closed-form inverse CDF: the depressed cubic ``mu^3 + 3 mu + (4 - 8u) = 0``
  has the single real root ``t - 1/t`` with ``t = cbrt(term + sqrt(term^2+1))``,
  ``term = 4u - 2``.
  """
  u = jax.random.uniform(key, dtype=dtype)
  term = 4.0 * u - 2.0
  t = jnp.cbrt(term + jnp.sqrt(term * term + 1.0))
  return t - 1.0 / t


def _rotate_scatter(direction: Array, mu: Array, psi: Array) -> Array:
  """Rotate ``direction`` by polar cosine ``mu`` and azimuth ``psi`` in the
  scattering frame (Wang et al. 1995). The near-vertical-axis branch
  (``|uz| ~ 1``) uses the degenerate frame to avoid 1/sqrt(1-uz^2) -> inf."""
  sin_theta = jnp.sqrt(jnp.maximum(1.0 - mu * mu, 0.0))
  cps, sps = jnp.cos(psi), jnp.sin(psi)
  ux, uy, uz = direction[0], direction[1], direction[2]
  denom = jnp.sqrt(jnp.maximum(1.0 - uz * uz, 0.0))
  safe_denom = jnp.where(denom > 0.0, denom, 1.0)
  vx = ux * mu + sin_theta * (ux * uz * cps - uy * sps) / safe_denom
  vy = uy * mu + sin_theta * (uy * uz * cps + ux * sps) / safe_denom
  vz = uz * mu - sin_theta * cps * denom
  wx = sin_theta * cps
  wy = sin_theta * sps
  wz = jnp.sign(uz) * mu
  near_axis = denom < 1.0e-6
  new = jnp.where(near_axis, jnp.stack([wx, wy, wz]),
                  jnp.stack([vx, vy, vz]))
  return new / jnp.linalg.norm(new)


def scatter_direction(key: Array, direction: Array, g: Array) -> Array:
  """Rotate ``direction`` by a sampled Henyey-Greenstein angle (asymmetry
  ``g``) about a uniform azimuth, returning the new unit direction."""
  k_mu, k_psi = jax.random.split(key)
  mu = henyey_greenstein_mu(k_mu, g)
  psi = 2.0 * jnp.pi * jax.random.uniform(k_psi, dtype=direction.dtype)
  return _rotate_scatter(direction, mu, psi)


def scatter_direction_mixed(
    key: Array, direction: Array, g: Array, rayleigh_frac: Array,
    mie_cdf: Array | None = None, mie_ang: Array | None = None,
    r_eff: Array | None = None,
) -> Array:
  """Scatter by an explicit mixture: with probability ``rayleigh_frac`` use the
  Rayleigh (1+mu^2) phase (gas), else the CLOUD phase. The cloud phase is the
  microhh Mie LUT (``mie_cdf``/``mie_ang`` band slices + cell ``r_eff`` [um])
  when supplied, else Henyey-Greenstein with the un-mixed cloud asymmetry
  ``g_cloud = g / (1 - rayleigh_frac)`` (Rayleigh contributes g=0). Mirrors the
  oracle's gas-Rayleigh / cloud-Mie split (aerosol-as-HG 3-way is future; for
  cloud-dominated LES scattering the non-Rayleigh part is cloud Mie).
  """
  k_pick, k_ray, k_cloud, k_psi = jax.random.split(key, 4)
  dtype = direction.dtype
  is_rayleigh = jax.random.uniform(k_pick, dtype=dtype) < rayleigh_frac
  mu_rayleigh = rayleigh_mu(k_ray, dtype)
  if mie_cdf is not None and mie_ang is not None and r_eff is not None:
    from legoesm.atmosphere.physics.radiation.mc3d import mie
    mu_cloud = mie.mie_sample_cos(
        jax.random.uniform(k_cloud, dtype=dtype), r_eff, mie_cdf, mie_ang)
  else:
    one_m = jnp.clip(1.0 - rayleigh_frac, _G_ISOTROPIC_EPS, 1.0)
    # Clip STRICTLY inside (-1, 1): g/one_m >= 1 (cloud-dominated cell) fed
    # to the HG sampler at exactly +-1 hits the 0/0 singularity documented
    # at _HG_G_MAX above.
    g_cloud = jnp.clip(g / one_m, -_HG_G_MAX, _HG_G_MAX)
    mu_cloud = henyey_greenstein_mu(k_cloud, g_cloud)
  # Distinct sub-keys: jnp.where evaluates BOTH branches, so sharing one key
  # would correlate the two draws.
  mu = jnp.where(is_rayleigh, mu_rayleigh, mu_cloud)
  psi = 2.0 * jnp.pi * jax.random.uniform(k_psi, dtype=dtype)
  return _rotate_scatter(direction, mu, psi)


def lambertian_reflect(key: Array, dtype) -> Array:
  """Sample an upward (+z) cosine-weighted direction for a Lambertian surface
  reflection. ``cos(theta) = sqrt(u1)`` gives the cosine weighting."""
  k1, k2 = jax.random.split(key)
  u1 = jax.random.uniform(k1, dtype=dtype)
  u2 = jax.random.uniform(k2, dtype=dtype)
  cos_t = jnp.sqrt(u1)
  sin_t = jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))
  phi = 2.0 * jnp.pi * u2
  return jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t])


def isotropic_direction(key: Array, dtype) -> Array:
  """Sample a direction uniformly on the unit sphere (LW thermal emission)."""
  k1, k2 = jax.random.split(key)
  mu = 1.0 - 2.0 * jax.random.uniform(k1, dtype=dtype)
  sin_t = jnp.sqrt(jnp.maximum(1.0 - mu * mu, 0.0))
  phi = 2.0 * jnp.pi * jax.random.uniform(k2, dtype=dtype)
  return jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), mu])


def free_path(key: Array, majorant: Array) -> Array:
  """Sample a free-flight distance ``-ln(u)/majorant`` for Woodcock tracking."""
  u = jax.random.uniform(key, dtype=majorant.dtype)
  # ``1-u`` keeps the argument in (0,1] so the log is finite.
  return -jnp.log(1.0 - u) / majorant
