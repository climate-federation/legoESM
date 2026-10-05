"""1.5-order TKE sub-grid-scale closure for the plane LES (gap #4).

PARKED in ``_future/`` (ponytail #16, user-approved 2026-10-02): not wired —
no production driver, factory or registry imports this module, and its tests
are skipped.  Wire it into production (moving it back) or delete it.

Deardorff (1980) / Lilly (1967) prognostic-TKE sub-grid closure, the alternative
to the Smagorinsky-Lilly SGS in
:func:`legoesm.atmosphere.dynamics.les.compressible_euler_plane._compute_smagorinsky_K_m_plane`
called for in ``docs/COMPARE_REANALYSIS.md`` §3/gap #4 (the one genuinely new
dynamics-side piece needed for boundary-layer LES, where a prognostic-TKE
closure is generally preferred over the diagnostic Smagorinsky).

The closure carries a prognostic sub-grid TKE ``e`` and sets the eddy viscosity
``ν_t = C_k ℓ √e`` with a stratification-limited mixing length ``ℓ``; ``e``
evolves under shear production, buoyancy production, and dissipation
``ε = C_ε e^{3/2}/ℓ``.  Conventions MATCH the Smagorinsky closure so the two are
interchangeable: strain magnitude ``|S|² = 2 S_ij S_ij`` and the sub-grid ``N²``
from :func:`legoesm.atmosphere.dynamics.les.compressible_euler_plane.sgs_brunt_vaisala_sq`.

**Consistency check (in the tests):** the *local-equilibrium* limit (production =
dissipation, neutral ``ℓ = Δ``) reduces exactly to Smagorinsky-Lilly with
``C_s = (C_k³/C_ε)^{1/4}`` ≈ 0.19 for the default constants — i.e. this closure is
a strict generalization that adds non-equilibrium TKE memory.

All functions are pure-JAX, AD-safe (gradient-safe ``sqrt`` at ``e = 0``), and
jit/vmap-friendly.  Closure constants live in :class:`TKESGSConfig`.  Wiring the
prognostic ``e`` carry + the ``turbulence_closure="tke_1.5"`` dispatch into the
plane dycore time loop is the integration step (the closure math lives here).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Safety floor keeping the dissipation length and ε denominators finite.
_LENGTH_FLOOR = 1.0e-30


class TKESGSConfig(NamedTuple):
    """1.5-order TKE closure constants (Deardorff 1980 / Lilly 1967).

    ``c_k`` sets ``ν_t = c_k ℓ √e``; the dissipation coefficient is the
    Deardorff length-dependent ``C_ε = c_eps_0 + c_eps_1 · ℓ/Δ``; the turbulent
    Prandtl number is ``Pr_t = 1/(1 + 2ℓ/Δ)`` (so ``K_h = (1 + 2ℓ/Δ) ν_t``);
    the stable mixing length is ``ℓ = min(Δ, stable_length_coeff·√e/N)``.
    """

    c_k: float = 0.1
    c_eps_0: float = 0.19
    c_eps_1: float = 0.51
    stable_length_coeff: float = 0.76


def validate_tke_config(config: TKESGSConfig) -> None:
    """Reject non-physical closure constants (negative/zero viscosity, ε, ℓ).

    The closure functions here are per-cell/per-step hot kernels and therefore
    assume a valid config; this is the explicit precondition gate the dycore
    integration MUST call **once** when it builds the SGS scheme (the
    ``turbulence_closure="tke_1.5"`` setup path), not inside the time loop.

    ``c_k`` and ``stable_length_coeff`` must be positive.  The dissipation
    ``C_ε = c_eps_0 + c_eps_1·ℓ/Δ`` is linear in ``ℓ/Δ ∈ [0, 1]``, so it stays
    positive on the whole interval iff both endpoints are positive: ``c_eps_0 > 0``
    (at ``ℓ=0``) and ``c_eps_0 + c_eps_1 > 0`` (at ``ℓ=Δ``).
    """
    if config.c_k <= 0.0:
        raise ValueError(f"TKESGSConfig.c_k must be > 0, got {config.c_k}.")
    if config.stable_length_coeff <= 0.0:
        raise ValueError(
            f"TKESGSConfig.stable_length_coeff must be > 0, got "
            f"{config.stable_length_coeff}."
        )
    if config.c_eps_0 <= 0.0 or config.c_eps_0 + config.c_eps_1 <= 0.0:
        raise ValueError(
            f"TKESGSConfig dissipation C_ε must stay positive: need c_eps_0>0 "
            f"and c_eps_0+c_eps_1>0, got {config.c_eps_0}, {config.c_eps_1}."
        )


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """Gradient-safe ``sqrt``: exact for ``x>0``, zero (with zero grad) at ``x<=0``."""
    pos = x > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, jnp.ones_like(x))), jnp.zeros_like(x))


def tke_mixing_length(
    delta: jax.Array,
    tke: jax.Array,
    n2: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """SGS mixing length ``ℓ`` [m]: grid scale ``Δ``, limited under stable ``N²``.

    Neutral / unstable (``N² ≤ 0``): ``ℓ = Δ``.  Stable (``N² > 0``):
    ``ℓ = min(Δ, c_s·√e/N)`` (Deardorff stability limit), so mixing is
    suppressed in strongly stratified layers.  AD-safe.
    """
    delta = jnp.asarray(delta)
    tke = jnp.asarray(tke, dtype=delta.dtype)
    n2 = jnp.asarray(n2, dtype=delta.dtype)
    coeff = jnp.asarray(config.stable_length_coeff, dtype=delta.dtype)
    stable = n2 > 0
    n = _safe_sqrt(n2)
    n_safe = jnp.where(stable, jnp.maximum(n, jnp.asarray(_LENGTH_FLOOR, delta.dtype)), jnp.ones_like(n))
    ell_stable = jnp.minimum(delta, coeff * _safe_sqrt(tke) / n_safe)
    return jnp.where(stable, ell_stable, delta)


def tke_eddy_viscosity(
    tke: jax.Array,
    length: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """SGS momentum eddy viscosity ``ν_t = C_k ℓ √e`` [m²/s] (AD-safe)."""
    tke = jnp.asarray(tke)
    length = jnp.asarray(length, dtype=tke.dtype)
    c_k = jnp.asarray(config.c_k, dtype=tke.dtype)
    return c_k * length * _safe_sqrt(tke)


def tke_turbulent_prandtl(
    length: jax.Array,
    delta: jax.Array,
) -> jax.Array:
    """Turbulent Prandtl number ``Pr_t = 1/(1 + 2ℓ/Δ)`` (Deardorff).

    ``K_h = ν_t/Pr_t = (1 + 2ℓ/Δ) ν_t`` — heat is mixed MORE than momentum as
    the mixing length shrinks relative to the grid scale.
    """
    length = jnp.asarray(length)
    delta = jnp.asarray(delta, dtype=length.dtype)
    delta_safe = jnp.maximum(delta, jnp.asarray(_LENGTH_FLOOR, dtype=length.dtype))
    return 1.0 / (1.0 + 2.0 * length / delta_safe)


def tke_dissipation_rate(
    tke: jax.Array,
    length: jax.Array,
    delta: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """SGS dissipation ``ε = (c_eps_0 + c_eps_1 ℓ/Δ) e^{3/2}/ℓ`` [m²/s³] (AD-safe)."""
    tke = jnp.asarray(tke)
    length = jnp.asarray(length, dtype=tke.dtype)
    delta = jnp.asarray(delta, dtype=tke.dtype)
    c_eps = config.c_eps_0 + config.c_eps_1 * length / jnp.maximum(
        delta, jnp.asarray(_LENGTH_FLOOR, dtype=tke.dtype)
    )
    length_safe = jnp.maximum(length, jnp.asarray(_LENGTH_FLOOR, dtype=tke.dtype))
    e32 = tke * _safe_sqrt(tke)  # e^{3/2}, AD-safe at e=0
    return c_eps * e32 / length_safe


def tke_tendency(
    tke: jax.Array,
    strain_mag_sq: jax.Array,
    n2: jax.Array,
    length: jax.Array,
    delta: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """Local SGS-TKE tendency ``∂e/∂t = P_shear + P_buoy − ε`` [m²/s³].

    ``P_shear = ν_t·|S|²`` (production by resolved shear), ``P_buoy =
    −(ν_t/Pr_t)·N²`` (buoyancy: a sink under stable ``N²>0``, a source under
    unstable ``N²<0``), ``ε`` the dissipation.  Excludes SGS-TKE *transport*
    (handled by the dycore's diffusion of ``e``) — this is the local source
    term.  ``|S|² = 2 S_ij S_ij`` and ``N²`` follow the Smagorinsky conventions.
    """
    nu_t = tke_eddy_viscosity(tke, length, config)
    pr_t = tke_turbulent_prandtl(length, delta)
    p_shear = nu_t * jnp.asarray(strain_mag_sq, dtype=nu_t.dtype)
    p_buoy = -(nu_t / pr_t) * jnp.asarray(n2, dtype=nu_t.dtype)
    eps = tke_dissipation_rate(tke, length, delta, config)
    return p_shear + p_buoy - eps


def tke_equilibrium(
    strain_mag_sq: jax.Array,
    n2: jax.Array,
    length: jax.Array,
    delta: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """Local-equilibrium SGS TKE (production = dissipation) at fixed ``ℓ``.

    Solving ``ν_t(|S|² − N²/Pr_t) = ε`` for ``e``:
    ``e_eq = (C_k/C_ε) ℓ² (|S|² − N²/Pr_t)``, floored at zero (no turbulence
    when the forcing is sub-critical).  ``tke_tendency(e_eq, …) = 0`` by
    construction **for the GIVEN ``ℓ``**.

    Caveat (stable branch): :func:`tke_mixing_length` makes ``ℓ`` itself a
    function of ``e`` when ``N² > 0``, so a self-consistent stable steady state
    ``(ℓ, e)`` is a fixed point requiring iteration — ``e_eq`` at a frozen ``ℓ``
    is the equilibrium *for that ℓ*, not the coupled fixed point.  This is the
    intended building block: the prognostic scheme reaches the coupled state
    dynamically (it recomputes ``ℓ(e)`` each step), and the neutral/unstable
    drop-in :func:`tke_equilibrium_eddy_viscosity` (``ℓ = Δ``, no ``e``-coupling)
    is exact.
    """
    length = jnp.asarray(length)
    delta = jnp.asarray(delta, dtype=length.dtype)
    s2 = jnp.asarray(strain_mag_sq, dtype=length.dtype)
    n2 = jnp.asarray(n2, dtype=length.dtype)
    pr_t = tke_turbulent_prandtl(length, delta)
    c_eps = config.c_eps_0 + config.c_eps_1 * length / jnp.maximum(
        delta, jnp.asarray(_LENGTH_FLOOR, dtype=length.dtype)
    )
    forcing = s2 - n2 / pr_t
    e_eq = (config.c_k / c_eps) * length**2 * forcing
    return jnp.maximum(e_eq, jnp.zeros_like(e_eq))


def tke_equilibrium_eddy_viscosity(
    strain_mag_sq: jax.Array,
    n2: jax.Array,
    delta: jax.Array,
    config: TKESGSConfig = TKESGSConfig(),
) -> jax.Array:
    """Diagnostic (neutral ``ℓ = Δ``) local-equilibrium eddy viscosity ``ν_t``.

    A drop-in alternative to the Smagorinsky viscosity that needs NO prognostic
    ``e`` carry: ``ν_t = √(C_k³/C_ε) · Δ² · √(|S|² − N²/Pr_t)``, which equals the
    Smagorinsky-Lilly form ``(C_s Δ)²√(|S|² − Pr·N²)`` with ``C_s² = √(C_k³/C_ε)``
    and ``Pr = 1/Pr_t``.  The full prognostic scheme (``tke_tendency`` +
    ``tke_eddy_viscosity``) adds non-equilibrium memory on top of this limit.
    """
    delta = jnp.asarray(delta)
    e_eq = tke_equilibrium(strain_mag_sq, n2, delta, delta, config)
    return tke_eddy_viscosity(e_eq, delta, config)


def smagorinsky_equivalent_cs(config: TKESGSConfig = TKESGSConfig()) -> float:
    """The Smagorinsky ``C_s`` the neutral-equilibrium TKE closure reproduces.

    ``C_s = (C_k³/C_ε)^{1/4}`` with ``C_ε = c_eps_0 + c_eps_1`` at ``ℓ = Δ``
    (≈ 0.19 for the defaults — the canonical LES Smagorinsky constant).
    """
    c_eps = config.c_eps_0 + config.c_eps_1
    return float((config.c_k**3 / c_eps) ** 0.25)
