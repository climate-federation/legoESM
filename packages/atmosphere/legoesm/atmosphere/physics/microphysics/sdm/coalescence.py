"""Collision-coalescence: Shima (2009) Monte-Carlo super-droplet algorithm.

Faithful JAX port of the ERF ``SuperDropletPC::Coalescence`` Monte-Carlo scheme
(``Source/Particles/ERF_SuperDropletPCCoalescence.cpp``) for a single
well-mixed cell. One step:

1. draw a random permutation of the ``n`` super-droplets and form
   ``L = ⌊n/2⌋`` non-overlapping candidate pairs;
2. order each pair so ``i`` has the larger multiplicity (``ξ_i ≥ ξ_j``);
3. evaluate the collision kernel ``K(R_i, R_j[, Δv])``;
4. form the Shima scaled coalescence probability over the step
   ``P = ξ_i · (K/V_cell) · [½ n(n-1)/⌊n/2⌋] · dt`` — the bracket scales the
   ``L`` sampled candidate pairs up to all ``C(n,2)`` possible pairs;
5. draw an integer collision count ``γ = ⌊P⌋ + 𝟙[U(0,1) < P-⌊P⌋]`` (stochastic
   rounding), capped at ``⌊ξ_i/ξ_j⌋``;
6. apply the multiplicity-conserving coalescence update (Shima eq. for ξ_i≥ξ_j):
   the larger-ξ droplet loses multiplicity (``ξ_i -= γ ξ_j``) and the smaller-ξ
   droplet grows by absorbing ``γ`` of the larger droplet's volume
   (``R_j ← (γ R_i³ + R_j³)^{1/3}``, ``m_j += γ m_i``); the degenerate
   ``ξ_i = γ ξ_j`` case splits the pair into two equal-size droplets.

Because the candidate pairs are non-overlapping, the per-pair updates are
independent and applied with conflict-free scatters.

**Exact invariant:** the represented water mass ``Σ_i ξ_i m_i`` is conserved to
machine precision; the represented number ``Σ_i ξ_i`` decreases. This is *not*
differentiable (random permutation + stochastic integer γ); it is a pure
function of an explicit ``jax.random`` key (user said differentiability is not
required for coalescence).

References
----------
* Shima et al. (2009) QJRMS 135:1307-1320 (esp. §4, the SDM algorithm).
* ERF ``Source/Particles/ERF_SuperDropletPCCoalescence.cpp`` (oracle).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState
from legoesm.atmosphere.physics.microphysics.sdm.kernels import (
    collision_kernel,
    terminal_velocity,
)

__physics_contract__ = {
    "summary": (
        "Shima (2009) Monte-Carlo collision-coalescence step for a well-mixed "
        "cell: random pairing, kernel-weighted scaled probability, stochastic "
        "integer collision count, multiplicity-conserving coalescence."
    ),
    "inputs": {
        "multiplicity": "1",
        "radius": "m",
        "solute_mass": "kg",
        "V_cell": "m^3",
        "rho": "kg/m^3",
        "p": "Pa",
        "T": "K",
        "dt": "s",
        "key": "1 (jax.random PRNG key)",
    },
    "outputs": {"multiplicity": "1", "radius": "m", "solute_mass": "kg"},
    "sign_convention": (
        "Coalescence only merges droplets: the represented number Σξ "
        "decreases (or is unchanged), radii grow, multiplicities stay >= 0."
    ),
    "conserves": ["mass"],
    "differentiable": False,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletPCCoalescence",
    "idealized_test": (
        "Golovin additive kernel from any IC: ensemble-mean Σξ(t)/Σξ(0) = "
        "exp(-(b/ρ_w)·L·t) (L = Σξm/V_cell conserved); represented water mass "
        "Σξm conserved to machine precision; deterministic for a fixed key."
    ),
}


def coalescence_step(
    state: SuperDropletState,
    V_cell: float | jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    key: jax.Array,
    cfg: SDMConfig,
) -> SuperDropletState:
    """One Shima Monte-Carlo collision-coalescence step for a single cell.

    Parameters
    ----------
    state : SuperDropletState
        Droplet ensemble for one well-mixed cell (all attributes ``(n_sd,)``).
    V_cell : float
        Cell volume [m³] over which the droplets are mixed.
    rho, p, T : float
        Ambient air density [kg/m³], pressure [Pa], temperature [K] — used only
        by the ``cloud_rain_shima`` terminal velocity for the hydrodynamic
        kernels' relative speed; ignored by the Golovin kernel.
    dt : float
        Time step [s].
    key : jax.Array
        ``jax.random`` PRNG key (consumed; pass a fresh split each step).
    cfg : SDMConfig
        Scheme configuration (selects the collision kernel; static argument).
    """
    xi = state.multiplicity
    R = state.radius
    s = state.solute_mass
    active = state.active
    n = xi.shape[0]

    if n < 2:
        return state  # nothing to pair

    dtype = R.dtype
    L = n // 2  # number of candidate pairs

    k_perm, k_gamma = random.split(key)
    perm = random.permutation(k_perm, n)
    ia = perm[0:2 * L:2]   # first member of each pair
    ib = perm[1:2 * L:2]   # second member of each pair

    xi_a, xi_b = xi[ia], xi[ib]
    a_is_big = xi_a >= xi_b
    big = jnp.where(a_is_big, ia, ib)    # larger-multiplicity index
    small = jnp.where(a_is_big, ib, ia)  # smaller-multiplicity index

    xi_big = xi[big]
    xi_small = xi[small]
    R_big = R[big]
    R_small = R[small]
    s_big = s[big]
    s_small = s[small]
    act_pair = active[big] * active[small]

    # Relative speed for hydrodynamic kernels (Golovin ignores it).
    if cfg.collision_kernel == "golovin":
        dv = jnp.zeros((L,), dtype=dtype)
    else:
        rho_a = jnp.asarray(rho, dtype=dtype)
        p_a = jnp.asarray(p, dtype=dtype)
        T_a = jnp.asarray(T, dtype=dtype)
        v_big = terminal_velocity(R_big, rho_a, p_a, T_a, cfg)
        v_small = terminal_velocity(R_small, rho_a, p_a, T_a, cfg)
        dv = jnp.abs(v_big - v_small)

    K = collision_kernel(R_big, R_small, dv, cfg)

    # Shima scaled probability: ⌊n/2⌋ candidate pairs represent all C(n,2) pairs.
    scaling = 0.5 * n * (n - 1) / L
    P = xi_big * (K / V_cell) * scaling * dt
    P = P * act_pair  # inactive pairs cannot coalesce

    # Stochastic integer collision count γ, capped so the prey is not over-consumed.
    floor_P = jnp.floor(P)
    frac = P - floor_P
    u = random.uniform(k_gamma, (L,), dtype=dtype)
    gamma = floor_P + (u < frac).astype(dtype)
    xi_small_safe = jnp.maximum(xi_small, 1.0)  # avoid 0-division for inactive slots
    gamma = jnp.minimum(gamma, jnp.floor(xi_big / xi_small_safe))
    gamma = jnp.where((xi_small > 0.0) & (act_pair > 0.0), gamma, 0.0)
    gamma = jnp.maximum(gamma, 0.0)

    has_coal = gamma > 0.0
    # Degenerate equal-multiplicity split when ξ_big == γ ξ_small.
    excess = xi_big - gamma * xi_small
    case_split = has_coal & (excess <= 0.0)

    R_merged = jnp.cbrt(gamma * R_big**3 + R_small**3)
    s_merged = s_small + gamma * s_big

    # New multiplicities.
    xi_big_new = jnp.where(
        has_coal,
        jnp.where(case_split, jnp.floor(xi_small / 2.0), xi_big - gamma * xi_small),
        xi_big,
    )
    xi_small_new = jnp.where(
        has_coal & case_split,
        xi_small - jnp.floor(xi_small / 2.0),
        xi_small,
    )
    # Smaller-ξ droplet always grows to the merged size; in the split case the
    # larger-ξ droplet becomes the merged size too.
    R_small_new = jnp.where(has_coal, R_merged, R_small)
    R_big_new = jnp.where(case_split, R_merged, R_big)
    s_small_new = jnp.where(has_coal, s_merged, s_small)
    s_big_new = jnp.where(case_split, s_merged, s_big)

    # Conflict-free scatter back (candidate pairs are non-overlapping).
    xi = xi.at[big].set(xi_big_new).at[small].set(xi_small_new)
    R = R.at[big].set(R_big_new).at[small].set(R_small_new)
    s = s.at[big].set(s_big_new).at[small].set(s_small_new)
    # A droplet whose multiplicity has reached zero represents nothing -> inactive.
    active = jnp.where(xi > 0.0, active, 0.0)

    return state._replace(multiplicity=xi, radius=R, solute_mass=s, active=active)


def represented_number(state: SuperDropletState) -> jax.Array:
    """Total number of real droplets represented, ``Σ_i active_i ξ_i`` [-]."""
    return jnp.sum(state.active * state.multiplicity)
