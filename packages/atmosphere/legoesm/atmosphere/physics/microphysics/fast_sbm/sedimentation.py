"""Per-bin sedimentation (oracle ``FALFLUXHUCM_Z``).

Two entry points:

* :func:`falfluxhucm_z` — the faithful port of the oracle routine
  (``module_mp_fast_sbm.F`` l. 5324-5382): unlimited upstream flux
  divergence on ``chem = f/ρ_air``, per-bin adaptive ``NSUB`` (a bin is
  skipped when no level holds ``chem >= 1e-20`` cm⁻³ g⁻¹ per g cm⁻³). The
  data-dependent ``NSUB`` uses the Morrison sedimentation sub-step machinery
  unchanged (user decision 2026-10-20): a static masked ``fori_loop`` of
  length ``MorrisonConfig.sed_cfl_substeps_max`` (default) and Morrison's
  strict error path (``equinox.error_if`` + NaN poison) when a bin needs
  more substeps, or when the oracle itself STOPs (``TFALL >= 1e10``).
  Oracle-tested; NOT yet called by the column adapter (see column status).
* :func:`sediment_bins` — the column adapter's NOT YET FAITHFUL surrogate
  described below (positivity-limited, static substeps).

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

import equinox as eqx
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


# Oracle FALFLUXHUCM_Z literals (CGS): TFALL=1.E10 start value and STOP
# threshold [s]; VFALL+1.E-20 [cm/s]; IFALL test chem >= 1.E-20 on
# ffx = FF/rhocgs [cm^-3 g^-1 / (g cm^-3)].
_TFALL_START_S = 1.0e10
_VFALL_EPS_CM_S = 1.0e-20
_IFALL_MIN_CGS = 1.0e-20
_M_TO_CM = 100.0
# chem SI [m^-3 kg^-1 / (kg m^-3)] -> CGS [cm^-3 g^-1 / (g cm^-3)]: 1e-6.
_CHEM_SI_TO_CGS = 1.0e-6


def falfluxhucm_z(
    chem: jax.Array,
    v_term: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Oracle ``FALFLUXHUCM_Z`` for one column; returns ``(chem_new, nsub)``.

    Args:
        chem: ``(nlev, n_bins)`` ``f/ρ_air`` [m⁻³ kg⁻¹ / (kg m⁻³)], level 0 =
            TOP (the oracle's ``ffx_z = chem_new/rhocgs``, in SI).
        v_term: ``(nlev, n_bins)`` fall speed [m/s], positive downward.
        rho: ``(nlev,)`` air density [kg/m³].
        dz: ``(nlev,)`` layer thickness [m] (WRF ``dz8w``); level heights
            are rebuilt as the oracle does, ``z_k = Σ_{j<k} dz_j + dz_k/2``
            from the surface.
        dt: step [s].

    Upstream flux divergence, bottom-up index ``k`` (oracle):
    ``DWFLUX(k) = −(ρ_k V_k c_k − ρ_{k+1} V_{k+1} c_{k+1})/(ρ_k (z_{k+1}−z_k))``,
    top ``−ρ V c/(ρ (z_top − z_top−1))``, ``NSUB = int(2 dt/TFALL) + 1`` with
    ``TFALL = min_k z_k/(V_k + 1e-20)`` (absolute height, CGS). No positivity
    limiter (as the oracle). ``nsub`` is 0 for skipped bins.
    """
    # Static substep cap of the masked loop (reverse-mode AD needs a static
    # length): the Morrison sedimentation sub-step cap, reused unchanged.
    # Function-scope import: microphysics.config imports this package.
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    max_substeps = MorrisonConfig._field_defaults["sed_cfl_substeps_max"]
    dtype = jnp.result_type(chem)
    dt = jnp.asarray(dt, dtype)
    c0 = chem[::-1]                      # bottom-up, as the oracle
    v = v_term[::-1]
    r = rho[::-1][:, None]
    h_cm = dz[::-1] * _M_TO_CM
    z_full = jnp.concatenate([jnp.zeros((1,), h_cm.dtype), jnp.cumsum(h_cm)[:-1]])
    z_cm = (z_full + 0.5 * h_cm)[:, None]
    tfall = jnp.minimum(
        _TFALL_START_S,
        jnp.min(z_cm / (v * _M_TO_CM + _VFALL_EPS_CM_S), axis=0))     # (n_bins,)
    ifall = jnp.any(c0 * _CHEM_SI_TO_CGS >= _IFALL_MIN_CGS, axis=0)
    nsub_f = jnp.floor(2.0 * dt / tfall) + 1.0
    overflow = jnp.any(ifall & ((nsub_f > max_substeps)
                                | (tfall >= _TFALL_START_S)))
    nsub = jnp.where(ifall, jnp.minimum(nsub_f, max_substeps), 0.0)
    dtfall = dt / jnp.maximum(nsub, 1.0)
    z = z_cm / _M_TO_CM
    dz_up = jnp.concatenate([z[1:] - z[:-1], z[-1:] - z[-2:-1]], axis=0)

    def substep(n, c):
        flux = r * v * c
        flux_above = jnp.concatenate([flux[1:], jnp.zeros_like(flux[:1])], axis=0)
        dwflux = -(flux - flux_above) / (r * dz_up)
        return jnp.where(n < nsub, c + dwflux * dtfall, c)

    c_new = jax.lax.fori_loop(0, max_substeps, substep, c0)[::-1]
    c_new = eqx.error_if(
        c_new, overflow,
        "fast_sbm FALFLUXHUCM_Z: a bin needs more fall substeps than the "
        f"static cap {max_substeps} (MorrisonConfig.sed_cfl_substeps_max "
        "default), or the oracle STOPs (TFALL >= 1e10: a populated bin with "
        "zero fall speed); shorten the physics step")
    poison = jnp.where(overflow, jnp.nan, 1.0).astype(dtype)
    return c_new * poison, nsub.astype(jnp.int32)


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
