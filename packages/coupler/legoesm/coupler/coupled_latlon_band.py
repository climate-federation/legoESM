"""Minimal coupled atmosphere + slab-ocean step on a SHARED lat-lon grid,
latitude-band MPI — the first coupled MPI step (C10 scaling-lane groundwork).

The audit's "coupled atm+ocean scaling bench lane" was build-first because no
coupled MPI step existed: the atm band step (``make_latlon_mpi_step``) exposes
only ``step_fn(state, dt)`` and threads no ocean coupling.  This module builds
the coupled step the SIMPLEST correct way — EXPLICIT coupling at the interval
boundary, exactly the pattern the production ``CoupledESMDriver._segment_hook``
uses (couple between segments, not inside the jitted dynamics):

1. the atm C-grid dycore runs ``n_atm_substeps`` band-MPI steps (halo-exchanged
   dynamics; the step is compiled ONCE — no per-interval recompile);
2. an EXPLICIT band-local sensible-heat exchange couples the atm surface layer
   to a co-located slab-ocean SST.

Because the atm and slab ocean share the SAME lat-lon grid and the SAME
latitude-band decomposition, the coupling is POINTWISE per cell and fully
BAND-LOCAL: each rank couples its own band with NO cross-rank communication and
NO regrid.  The slab SST is a per-cell mixed-layer temperature (the
column-local ``simple_ocean`` energy balance in its simplest sensible-only
form).

Energy is conserved EXACTLY per cell by construction: the surface layer gains
``F*dt`` and the ocean loses ``F*dt`` (same flux ``F``, opposite signs), so
``C_atm·ΔT_sfc_air + C_ocean·ΔSST = 0`` at every column — hence globally
(area-weighted) too.  This is the invariant the conservation gate checks, and
the property a real coupled model must preserve.

This is CORRECTNESS + capability infrastructure (serial==band-MPI parity +
conservation gate), not a production physics envelope; the SST-in / flux-out
threading needed to fold real radiation + bulk fluxes into the SAME jitted step
(no interval recompile, for a production timing bench) is the documented
follow-up in ``docs/performance/scaling/SCALING_STATUS_AUDIT.md``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


class CoupledSlabConfig(NamedTuple):
    """Sensible-heat coupling + slab mixed-layer parameters.

    The exchange is a bulk sensible-heat flux ``F = k_exchange·(SST − T_sfc_air)``
    [W/m²] into the atmospheric surface layer; the ocean loses the same ``F``.
    All fields carry explicit physical units; defaults reference
    :mod:`legoesm.constants` where a physical constant exists.
    """

    # Bulk sensible-heat transfer coefficient [W/m²/K].
    k_exchange: float = 20.0
    # Atmospheric surface-layer heat capacity per area [J/m²/K]
    # (order ``constants.c_pd · Δp/g`` for a ~100 hPa surface layer ≈ 1.0e6).
    c_atm_area: float = 1.02e6
    # Slab-ocean mixed-layer heat capacity per area [J/m²/K]
    # (order rho_ocean·c_ocean·h_mix for a ~10 m mixed layer ≈ 4.1e7).
    c_ocean_area: float = 4.09e7
    # Ocean freezing floor [K] (SST clamped at/above this).
    t_freeze_ocean_K: float = constants.T_freeze_ocean


def apply_surface_coupling(atm_state, sst, cfg: CoupledSlabConfig,
                           dt_couple: float):
    """One EXPLICIT sensible-heat coupling update (band-local, per cell).

    ``atm_state`` is a C-grid ``CGridLatLonHydrostaticState`` (rank-local band);
    ``sst`` is the co-located slab SST ``(n_lat_local, n_lon)`` [K].  The atm
    surface-layer temperature is the lowest model level ``T[..., -1]``.  Returns
    ``(atm_state_new, sst_new, clamp_energy)``: the surface-air T and SST updated
    by the equal-and-opposite flux ``F = k_exchange·(SST − T_sfc_air)``, plus the
    per-cell energy the freezing clamp injected [J/m²].

    The SENSIBLE EXCHANGE alone conserves the coupled surface energy exactly
    (per cell, to float round-off):
    ``c_atm_area·ΔT_sfc_air + c_ocean_area·ΔSST_free = dt·F − dt·F = 0``.

    The freezing clamp is a NON-conservative numerical floor on the diagnosed
    ``atm + water`` budget: when the free SST would fall below ``t_freeze_ocean``
    the water freezes and the released latent heat keeps SST at the floor — so
    the clamp ADDS ``clamp_energy = c_ocean_area·(SST_clamped − SST_free) ≥ 0``
    to the surface (water) budget, with the equal-and-opposite debit carried by
    a sea-ice reservoir this minimal slab does NOT model.  Returning
    ``clamp_energy`` makes the budget auditable: ``Δ(atm + water) = Σ
    clamp_energy`` exactly (the conservation gate runs warm SST so the clamp
    never fires, and a dedicated test checks the clamp bookkeeping directly).
    """
    T = atm_state.T
    t_sfc_air = T[..., -1]                       # (n_lat_local, n_lon)
    flux = cfg.k_exchange * (sst - t_sfc_air)    # W/m² into the atmosphere
    t_sfc_air_new = t_sfc_air + dt_couple * flux / cfg.c_atm_area
    sst_free = sst - dt_couple * flux / cfg.c_ocean_area
    sst_new = jnp.maximum(sst_free, cfg.t_freeze_ocean_K)
    # Energy the clamp injected into the water budget (== the latent-fusion
    # debit a modelled sea-ice reservoir would carry); zero where no clamp.
    clamp_energy = cfg.c_ocean_area * (sst_new - sst_free)
    atm_state_new = atm_state._replace(
        T=T.at[..., -1].set(t_sfc_air_new))
    return atm_state_new, sst_new, clamp_energy


def coupling_energy(atm_state, sst, cfg: CoupledSlabConfig, area):
    """Area-weighted coupled surface energy
    ``Σ area·(c_atm·T_sfc_air + c_ocean·SST)`` [J] — the quantity the sensible
    EXCHANGE conserves EXACTLY.  The freezing clamp is the one non-conservative
    term: it is a SOURCE to this diagnosed budget (see
    :func:`apply_surface_coupling`), so ``Δ(this) = Σ area·clamp_energy``.
    ``area`` is the per-cell area ``(n_lat_local, n_lon)`` [m²] (rank-local
    band); the caller allreduces across ranks for the global total."""
    t_sfc_air = atm_state.T[..., -1]
    e_col = cfg.c_atm_area * t_sfc_air + cfg.c_ocean_area * sst
    return jnp.sum(area * e_col)


def make_coupled_latlon_band_step(
    atm_model, layout, cfg: CoupledSlabConfig, *,
    physics_fn: Callable | None = None,
    n_atm_substeps: int = 1,
):
    """Build a coupled atm + slab-ocean band-MPI step.

    ``atm_model`` is the rank-local band C-grid model; ``layout`` the
    :class:`LatLonBandLayout`.  Returns
    ``coupled_step(atm_state, sst, dt) -> (atm_state, sst)`` that advances the
    atm ``n_atm_substeps`` band-MPI steps then applies ONE explicit surface
    coupling over ``dt_couple = n_atm_substeps·dt`` (segment-boundary coupling).
    The atm step compiles once; the coupling is band-local host-side array math.

    ``n_atm_substeps`` must be a positive integer: a value ``< 1`` would take no
    atm step yet still report a "coupled" interval, so it is rejected here
    rather than silently producing a no-op loop (dispatch-hardening doctrine).
    """
    from legoesm.parallel.latlon_mpi import make_latlon_mpi_step

    if n_atm_substeps != int(n_atm_substeps) or int(n_atm_substeps) < 1:
        raise ValueError(
            f"n_atm_substeps must be a positive integer, got {n_atm_substeps!r}")
    n_sub = int(n_atm_substeps)

    atm_step = make_latlon_mpi_step(atm_model, layout, physics_fn=physics_fn)

    def coupled_step(atm_state, sst, dt):
        for _ in range(n_sub):
            atm_state = atm_step(atm_state, dt)
        atm_state, sst, _clamp_energy = apply_surface_coupling(
            atm_state, sst, cfg, dt * n_sub)
        return atm_state, sst

    return coupled_step
