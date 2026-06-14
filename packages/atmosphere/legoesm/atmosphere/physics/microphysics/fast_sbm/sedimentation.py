"""Per-bin sedimentation (oracle ``FALFLUXHUCM_Z``).

The oracle advects every bin down the column with an upstream
flux-divergence scheme, substepped to the bin's fall CFL
(``NSUB = int(2·dt/min_k(dz_k/V_k)) + 1``); the flux through the bottom
interface is the surface precipitation.

The port REUSES the shared positivity-limited upstream helper
``microphysics.output.sedimentation_tendency`` (same upstream
flux-divergence; its FCT-style cap guarantees positivity at any Courant
number) applied per bin over a STATIC number of substeps
(``n_substeps``): adaptive trip counts are not reverse-mode
differentiable in JAX, so the oracle's runtime ``NSUB`` becomes a config
constant. At per-substep CFL < 1 (the resolved regime) the two are the
same scheme; beyond that the limiter caps what the oracle would resolve
with more substeps — documented deviation, bounded by the positivity
guarantee.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.output import (
    sedimentation_tendency,
)

__physics_contract__ = {
    "summary": (
        "Per-bin gravitational settling of the liquid spectrum down the "
        "column with surface precipitation (oracle FALFLUXHUCM_Z; "
        "upstream flux divergence, positivity-limited, substepped)."
    ),
    "inputs": {
        "q_bins": "kg/kg per bin, (ncol, nlev, n_bins), level 0 = top",
        "rho": "kg/m^3 (ncol, nlev)",
        "v_term": "m/s per bin (n_bins,), positive downward",
        "dz": "m (ncol, nlev)",
        "dt": "s",
    },
    "outputs": {
        "dq_bins_dt": "kg/kg/s per bin",
        "surface_precip": "kg/m^2/s (ncol,)",
    },
    "sign_convention": (
        "v_term > 0 moves mass toward the surface (last level index); "
        "column water column-integral loss exactly equals the surface "
        "precipitation flux; q stays nonnegative for nonnegative input."
    ),
    "conserves": ["mass"],
    "differentiable": True,
    "reference": (
        "WRF module_mp_fast_sbm.F FALFLUXHUCM_Z (GSFC upstream method); "
        "shared helper microphysics/output.py sedimentation_tendency"
    ),
    "idealized_test": (
        "Zero velocity is a fixed point; uniform fall in a uniform column "
        "translates mass downward and precipitates the bottom layer's "
        "content; column water + accumulated precip is invariant to "
        "roundoff; no bin goes negative at CFL > 1."
    ),
}


def sediment_bins(
    q_bins: jax.Array,
    rho: jax.Array,
    v_term: jax.Array,
    dz: jax.Array,
    dt: float | jax.Array,
    n_substeps: int = 4,
) -> tuple[jax.Array, jax.Array]:
    """Settle all bins for one step; returns ``(dq_bins_dt, precip)``.

    ``n_substeps`` is static (oracle's adaptive NSUB is not reverse-mode
    differentiable); per-substep CFL = ``v dt / (dz n_substeps)``.

    ``v_term`` may be per-bin ``(n_bins,)`` (same fall speed at every level)
    or per-cell-per-bin ``(ncol, nlev, n_bins)`` — the oracle's level-
    dependent ``VR1(K,KR)``, recommended; the column adapter passes the
    latter so density/pressure vary the fall speed with height.
    """
    dt_sub = dt / n_substeps
    ncol, nlev, n_bins = q_bins.shape
    if v_term.ndim == 1:
        v_field = jnp.broadcast_to(v_term, (ncol, nlev, n_bins))
    else:
        v_field = v_term

    def one_bin(q_k, v_k):
        # q_k, v_k: (ncol, nlev) — this bin's field and per-level fall speed.
        def body(carry, _):
            q, acc = carry
            tend, sflux = sedimentation_tendency(
                q, rho, v_k, dz, dt_sub, return_surface_flux=True)
            return (q + dt_sub * tend, acc + sflux * dt_sub), None

        (q_end, precip_mass), _ = jax.lax.scan(
            body, (q_k, jnp.zeros(q_k.shape[0], dtype=q_k.dtype)),
            None, length=n_substeps)
        return q_end, precip_mass

    # Move bins to the front, scan the static substeps per bin.
    q_t = jnp.moveaxis(q_bins, -1, 0)            # (n_bins, ncol, nlev)
    v_t = jnp.moveaxis(v_field, -1, 0)           # (n_bins, ncol, nlev)
    q_end_t, precip_t = jax.vmap(one_bin)(q_t, v_t)
    q_end = jnp.moveaxis(q_end_t, 0, -1)
    dq_dt = (q_end - q_bins) / dt
    surface_precip = jnp.sum(precip_t, axis=0) / dt     # (ncol,)
    return dq_dt, surface_precip
