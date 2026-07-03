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

**Invariant:** the represented water mass ``Σ_i ξ_i m_i`` is conserved to
floating-point round-off — the merge is algebraically exact in both branches,
but mass recomputed from ``cbrt(γR_i³+R_j³)³`` carries ~1e-12 relative
round-off — and the represented number ``Σ_i ξ_i`` is non-increasing. The
default mode is *not* differentiable (random permutation + stochastic integer
γ). The opt-in ``SDMConfig(collision_mode="deterministic")`` replaces integer
γ with a smooth expected coalescence increment for fixed candidate pairs; it is
reverse-mode differentiable with respect to particle radii/multiplicities and
thermodynamic inputs, away from normal piecewise kernel/cap boundaries.

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
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    water_mass_per_droplet,
)
from legoesm.atmosphere.physics.microphysics.sdm.kernels import (
    brownian_kernel,
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
    # Contract bool = the DEFAULT (stochastic) path = forward-only. Opt-in
    # deterministic mode uses expected pair increments + has JAX VJPs for radii/
    # multiplicity/thermodynamics; pair selection stays discrete wrt order.
    "differentiable": False,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletPCCoalescence",
    "idealized_test": (
        "Golovin additive kernel from any IC: ensemble-mean Σξ(t)/Σξ(0) = "
        "exp(-(b/ρ_w)·L·t) and 2nd mass moment Σξm²(t) = Σξm²(0)·exp(2(b/ρ_w)Lt) "
        "(L = Σξm/V_cell conserved); represented water mass Σξm conserved to "
        "floating-point round-off; deterministic for a fixed key."
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
    n = xi.shape[0]

    if n < 2:
        return state  # nothing to pair

    dtype = state.radius.dtype
    L = n // 2  # number of candidate pairs

    if cfg.collision_mode == "stochastic":
        k_perm, k_gamma = random.split(key)
        perm = random.permutation(k_perm, n)
    elif cfg.collision_mode == "deterministic":
        k_gamma = key
        perm = jnp.arange(n, dtype=jnp.int32)
    else:
        raise ValueError(
            f"Unknown SDM collision_mode: {cfg.collision_mode!r} "
            "(expected 'stochastic' or 'deterministic')"
        )
    ia = perm[0:2 * L:2]   # first member of each pair
    ib = perm[1:2 * L:2]   # second member of each pair

    # Shima scaled probability: ⌊n/2⌋ candidate pairs represent all C(n,2) pairs.
    scaling = jnp.full((L,), 0.5 * n * (n - 1) / L, dtype=dtype)
    valid_pair = jnp.ones((L,), dtype=dtype)
    if cfg.collision_mode == "deterministic":
        return coalescence_step_pairs_deterministic(
            state, ia, ib, valid_pair, scaling, V_cell, rho, p, T, dt, cfg)
    return coalescence_step_pairs(
        state, ia, ib, valid_pair, scaling, V_cell, rho, p, T, dt, k_gamma, cfg)


def _pair_kernel_probability_inputs(
    state: SuperDropletState,
    ia: jax.Array,
    ib: jax.Array,
    valid_pair: jax.Array,
    pair_scaling: float | jax.Array,
    V_cell: float | jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
):
    """Gather pair state and compute the Shima scaled coalescence probability."""
    xi = state.multiplicity
    R = state.radius
    s = state.solute_mass
    active = state.active
    dtype = R.dtype

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
    act_pair = active[big] * active[small] * jnp.asarray(valid_pair, dtype=dtype)

    # Relative speed for hydrodynamic kernels (Golovin ignores it).
    if cfg.collision_kernel == "golovin":
        dv = jnp.zeros_like(R_big)
    else:
        rho_a = jnp.asarray(rho, dtype=dtype)
        p_a = jnp.asarray(p, dtype=dtype)
        T_a = jnp.asarray(T, dtype=dtype)
        v_big = terminal_velocity(R_big, rho_a, p_a, T_a, cfg)
        v_small = terminal_velocity(R_small, rho_a, p_a, T_a, cfg)
        dv = jnp.abs(v_big - v_small)

    K = collision_kernel(R_big, R_small, dv, cfg)
    if cfg.include_brownian:
        # ERF adds the Brownian coagulation coefficient on top of the selected
        # kernel (k_val += k_brown), using the TOTAL droplet mass (water+solute).
        T_a = jnp.asarray(T, dtype=dtype)
        p_a = jnp.asarray(p, dtype=dtype)
        m_big = water_mass_per_droplet(state)[big] + s_big
        m_small = water_mass_per_droplet(state)[small] + s_small
        K = K + brownian_kernel(R_big, R_small, m_big, m_small, p_a, T_a)

    scaling = jnp.asarray(pair_scaling, dtype=dtype)
    V_cell = jnp.asarray(V_cell, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    P = xi_big * (K / V_cell) * scaling * dt
    P = P * act_pair
    return (big, small, xi_big, xi_small, R_big, R_small, s_big, s_small,
            act_pair, P)


def coalescence_step_pairs(
    state: SuperDropletState,
    ia: jax.Array,
    ib: jax.Array,
    valid_pair: jax.Array,
    pair_scaling: float | jax.Array,
    V_cell: float | jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    key: jax.Array,
    cfg: SDMConfig,
) -> SuperDropletState:
    """Apply the Shima update to a fixed set of non-overlapping candidate pairs.

    ``valid_pair`` masks candidate pairs that should be skipped, while
    ``pair_scaling`` supplies the local ``C(n,2)/floor(n/2)`` scale factor for
    each candidate. Invalid pairs may overlap valid pairs; they contribute zero
    additive updates and therefore cannot overwrite a real coalescence result.
    """
    xi = state.multiplicity
    R = state.radius
    s = state.solute_mass
    active = state.active
    dtype = R.dtype
    (big, small, xi_big, xi_small, R_big, R_small, s_big, s_small,
     act_pair, P) = _pair_kernel_probability_inputs(
        state, ia, ib, valid_pair, pair_scaling, V_cell, rho, p, T, dt, cfg)

    # Stochastic integer collision count γ, capped so the prey is not over-consumed.
    floor_P = jnp.floor(P)
    frac = P - floor_P
    u = random.uniform(key, ia.shape, dtype=dtype)
    gamma = floor_P + (u < frac).astype(dtype)
    # Cap at ⌊ξ_big/ξ_small⌋ so the larger-multiplicity droplet is not
    # over-consumed. Guard the floating-point quotient against rounding up to a
    # value whose product exceeds ξ_big (which would let the split branch CREATE
    # mass): drop one collision if γ_cap·ξ_small overshoots ξ_big. This makes
    # ``excess = ξ_big − γ ξ_small >= 0`` hold for every pair.
    xi_small_safe = jnp.maximum(xi_small, 1.0)  # avoid 0-division for inactive slots
    gamma_cap = jnp.floor(xi_big / xi_small_safe)
    gamma_cap = jnp.where(gamma_cap * xi_small > xi_big, gamma_cap - 1.0, gamma_cap)
    gamma_cap = jnp.maximum(gamma_cap, 0.0)
    gamma = jnp.minimum(gamma, gamma_cap)
    gamma = jnp.where((xi_small > 0.0) & (act_pair > 0.0), gamma, 0.0)

    has_coal = gamma > 0.0
    # Degenerate full-consumption split when ξ_big == γ ξ_small. Because the cap
    # above guarantees excess >= 0, ``excess <= 0`` selects exactly that case,
    # so the split branch conserves mass (no negative-excess mass creation).
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

    # Additive scatter is robust to invalid overlapping pairs: their deltas are
    # exactly zero, while valid Shima candidate pairs are non-overlapping.
    xi = xi.at[big].add(jnp.where(has_coal, xi_big_new - xi_big, 0.0))
    xi = xi.at[small].add(jnp.where(has_coal & case_split,
                                    xi_small_new - xi_small, 0.0))
    R = R.at[big].add(jnp.where(case_split, R_big_new - R_big, 0.0))
    R = R.at[small].add(jnp.where(has_coal, R_small_new - R_small, 0.0))
    s = s.at[big].add(jnp.where(case_split, s_big_new - s_big, 0.0))
    s = s.at[small].add(jnp.where(has_coal, s_small_new - s_small, 0.0))
    # A droplet whose multiplicity has reached zero represents nothing -> inactive.
    active = jnp.where(xi > 0.0, active, 0.0)

    return state._replace(multiplicity=xi, radius=R, solute_mass=s, active=active)


def coalescence_step_pairs_deterministic(
    state: SuperDropletState,
    ia: jax.Array,
    ib: jax.Array,
    valid_pair: jax.Array,
    pair_scaling: float | jax.Array,
    V_cell: float | jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> SuperDropletState:
    """Apply a deterministic mean-field coalescence update to candidate pairs.

    For each non-overlapping pair, the Shima scaled probability ``P`` is used
    as the expected collision count. To keep multiplicities non-negative while
    preserving differentiability, the expected count is smoothly saturated at
    the available ratio ``xi_big / xi_small``:

    ``gamma = cap * (1 - exp(-P / cap))``.

    In the normal rare-collision regime this is ``gamma ~= P``; in certain
    collision probes it approaches the full available coalescence. The update is
    algebraically mass-conserving for fractional ``gamma``.
    """
    xi = state.multiplicity
    R = state.radius
    s = state.solute_mass
    active = state.active
    dtype = R.dtype
    (big, small, xi_big, xi_small, R_big, R_small, s_big, s_small,
     act_pair, P) = _pair_kernel_probability_inputs(
        state, ia, ib, valid_pair, pair_scaling, V_cell, rho, p, T, dt, cfg)

    cap = jnp.where(xi_small > 0.0, xi_big / xi_small, 0.0)
    cap = jnp.maximum(cap, 0.0)
    # Deterministic γ = the TRUE capped Shima expectation E[γ] = min(P, cap)
    # (codex 2026-06-13: the earlier smooth surrogate cap·(1−e^(−P/cap)) was
    # BIASED low — P=0.7 gave 0.50 vs the stochastic mean 0.71). min() is a kink
    # at P=cap but differentiable a.e. (sub-gradient), keeping the deterministic
    # path jax.grad-able while now matching the stochastic ensemble mean.
    gamma = jnp.where(
        (cap > 0.0) & (act_pair > 0.0),
        jnp.minimum(P, cap),
        0.0,
    )
    gamma = jnp.minimum(gamma, cap)
    has_coal = gamma > 0.0

    xi_big_new = xi_big - gamma * xi_small
    R_small_new = jnp.cbrt(gamma * R_big**3 + R_small**3)
    s_small_new = s_small + gamma * s_big

    xi = xi.at[big].add(jnp.where(has_coal, xi_big_new - xi_big, 0.0))
    R = R.at[small].add(jnp.where(has_coal, R_small_new - R_small, 0.0))
    s = s.at[small].add(jnp.where(has_coal, s_small_new - s_small, 0.0))
    active = jnp.where(xi > jnp.asarray(0.0, dtype), active, 0.0)
    xi = jnp.maximum(xi, jnp.asarray(0.0, dtype))

    return state._replace(multiplicity=xi, radius=R, solute_mass=s, active=active)


def represented_number(state: SuperDropletState) -> jax.Array:
    """Total number of real droplets represented, ``Σ_i active_i ξ_i`` [-]."""
    return jnp.sum(state.active * state.multiplicity)
