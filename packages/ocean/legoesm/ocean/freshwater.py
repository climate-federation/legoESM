"""Freshwater forcing for the MPAS ocean model.

Handles precipitation, evaporation, land runoff, and ice melt/freeze
freshwater fluxes. Applies virtual salt flux to salinity and real
freshwater mass flux to the free surface.

Conventions
-----------
- All fluxes in kg/m2/s (mass flux per unit area).
- Positive = freshwater entering ocean (precip, runoff, ice melt).
- Evaporation is positive upward in the coupler, so E enters here
  as a positive value that *removes* freshwater from the ocean.

References
----------
- Griffies, S. M. (2004). Fundamentals of Ocean Climate Models, Ch. 12.
- Large, W. G. et al. (1997). J. Phys. Oceanogr., 27(11), 2418-2447.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FreshwaterForcing(NamedTuple):
    """Freshwater fluxes applied to the ocean surface.

    All fields have shape (nCells,) and units kg/m²/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m²/s].
    evap : jax.Array
        Evaporation rate [kg/m²/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m²/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m²/s], positive = melt.
    restoring : jax.Array
        OMIP-2 SSS-restoring virtual freshwater flux [kg/m²/s,
        positive INTO ocean].  Computed from
        :func:`legoesm.ocean.forcing.sss_restoring.compute_sss_restoring_flux`.
        Zero by default for backward compatibility with the
        legacy 4-component constructor.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray
    restoring: jnp.ndarray = None  # type: ignore[assignment]


def zero_freshwater(nCells: int) -> FreshwaterForcing:
    """Create zero freshwater forcing.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.

    Returns
    -------
    FreshwaterForcing
    """
    # Init helper: keep at the JAX default float dtype.  Callers running
    # under a non-default precision policy can ``cast_pytree`` the
    # result to match their state.
    z = jnp.zeros(nCells)
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z, restoring=z)


def net_freshwater_flux(
    fw: FreshwaterForcing, *, include_runoff: bool = True,
) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m²/s].

    F_fw = P - E + R + M + R_restore

    where P=precip, E=evaporation (positive up), R=runoff,
    M=ice melt, R_restore=SSS-restoring virtual FW flux (zero
    when ``restoring`` field is None or absent — legacy
    callers built without the SSS-restoring extension).

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m²/s], positive into ocean.
    """
    # ``include_runoff=False`` is NEMO's ``emp``: the evaporation-minus-
    # precipitation channel WITHOUT the river runoff, which NEMO carries
    # separately (``stp2d.F90:278-279`` forms ``emp - rnf`` for the sea
    # surface, ``sbcrnf.F90`` puts the same runoff in the horizontal
    # divergence, and ``trasbc.F90``'s dilution term reads ``emp`` alone).
    # The runoff is EXCLUDED FROM THE SUM rather than subtracted from it:
    # ``(x + r) - r`` is not bitwise ``x``, and on the ORCA2 record the
    # two spellings differ on 328 cells.
    if not include_runoff:
        base = fw.precip - fw.evap + fw.ice_fw
    else:
        base = fw.precip - fw.evap + fw.runoff + fw.ice_fw
    # ``restoring is None`` is a Python (trace-time) check — safe
    # under JIT because the field is structural metadata.
    if fw.restoring is None:
        return base
    return base + fw.restoring


def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    """Compute free-surface tendency from freshwater flux.

    deta/dt = F_fw / rho_0

    Parameters
    ----------
    fw : FreshwaterForcing
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Free-surface tendency [m/s].
    """
    return net_freshwater_flux(fw) / rho_0


def virtual_salt_flux(
    fw: FreshwaterForcing,
    S_ref: float | jnp.ndarray,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Compute virtual salt flux for the top ocean layer.

    dS/dt = -S_ref * F_fw / (rho_0 * dz_0)

    This approximation maintains volume while adjusting salinity
    to account for freshwater dilution/concentration.

    Parameters
    ----------
    fw : FreshwaterForcing
    S_ref : float
        Reference salinity [PSU].
    dz_0 : jax.Array, shape (nCells,)
        Top layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Salinity tendency [PSU/s] for top layer.
    """
    return virtual_salt_flux_from_net(
        net_freshwater_flux(fw), S_ref, dz_0, rho_0)


def virtual_closure_temperature_twin(fw, T_top, dz_0, rho_0: float, mask=None):
    """Top-cell temperature tendency [K/s] that the virtual-salt closure owes
    for the water whose HEAT CONTENT the surface heat flux already carries.

    Our OMIP heat flux is NEMO ``blk_oce_2``: ``-evap*cp*SST + rain*cp*theta_air``
    (+ snow terms), and the SSS-restoring channel adds ``F*cp*SST``
    (``sbcssr``).  Under a non-linear free surface that heat pairs with the
    dilution of temperature by the entering/leaving water; the virtual
    closure never dilutes temperature (the volume goes to eta only), so NEMO's
    linear-free-surface branch adds the counterpart back in ``trasbc``
    (``sbc_tsc(jp_tem) += emp*sst/rho0``).  Without it the virtual closure
    carries a spurious surface heat flux ``-(E-P) cp SST`` (~2-6 W/m2,
    cooling where evaporation exceeds rain; codex + GLM 2026-09-05).

    Only the channels whose heat is in the heat flux enter: precip, evap,
    restoring.  Runoff and ice melt water carry no heat-flux term in this
    coupling (they arrive at the local temperature) and are excluded.

        dT/dt|top = -(P - E + R_restore)/rho_0 * T_top / dz_0
    """
    F = jnp.asarray(fw.precip) - jnp.asarray(fw.evap)
    rest = getattr(fw, "restoring", None)
    if rest is not None:
        F = F + jnp.asarray(rest)
    dT = -(F / rho_0) * T_top / jnp.maximum(dz_0, 1.0e-3)
    return dT if mask is None else dT * mask


def real_freshwater_dilution_tendencies(
    F_rate: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    h_k: jnp.ndarray,
    mask: jnp.ndarray,
    h_floor: float = 1e-10,
    F_entry: jnp.ndarray | None = None,
    entry_heat: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Surface-dilution tracer tendencies of the real (volume) freshwater
    closure: the water enters the TOP cell, so the column stretch that the
    z-star step spreads uniformly must be paired with a downward transport
    of the resident water through every interface.

    NEMO's non-linear free surface (``sshwzv``/``traadv`` with vvl, ``trasbc``
    carrying no ``emp*sss`` term) dilutes the top cell by ``-S_1 F/(rho h_1)``
    and leaves the layers below at constant salinity and temperature.  In a
    z-star core the eta channel already thickens EVERY layer by
    ``(h_k/H) F/rho`` and the tracer step keeps ``h S`` per layer, i.e. a
    uniform dilution ``-S F/(rho H)`` that NEMO does not have.  The
    difference is a vertical transport

        W(k+1/2) = (F/rho) * H_below(k+1/2) / H        [m/s, downward]

    (``H_below`` = thickness below the interface, so ``W`` = ``F/rho`` at the
    surface and 0 at the bottom) carrying the UPWIND cell's tracer through
    each interface (the cell above when water enters, the cell below when it
    leaves); the surface interface carries S = 0 (pure water) and T = T_1
    (NEMO's ``emp`` heat convention: water added or removed at the surface
    temperature) for both signs.  Flux form per layer::

        dC_k/dt = (W(k-1/2) C_up(k-1/2) - W(k+1/2) C_up(k+1/2)) / h_k

    Column-integrated ``h S`` is unchanged (the fluxes telescope to the
    zero-salt surface flux and the zero bottom flux); for a column of uniform
    S the layers below the top keep S exactly and the top cell obtains
    ``-S_1 F/(rho h_1)`` once the z-star stretch is added.  First-order
    upwind in the vertical; the transport is ~1e-7 m/s so its CFL is ~1e-5.
    ``F_rate`` is the SAME rate the eta channel receives (normalised,
    including ice melt water and the water-flux SSS restoring); the ice SALT
    flux stays on its own surface channel.

    Parameters
    ----------
    F_rate : (...,) water flux into the ocean [m/s] (= ``freshwater_eta_tendency``)
    S, T : (..., nlev) tracers AFTER the advective/z-star step
    h_k : (..., nlev) live layer thickness (0 below the seafloor)
    mask : (...,) wet mask (1 = ocean)
    F_entry : (..., nlev) or None
        Per-level water ENTRY rate [m/s] summing to ``F_rate`` per column
        (NEMO ``sbc_rnf_div``: river water enters every level down to
        ``h_rnf``, see :func:`runoff_entry_profile`).  None = everything
        enters at the surface.  Each entering parcel dilutes its own layer
        (S = 0); the transport
        ``W(k+1/2) = sum_{j<=k} entry_j - F H_above(k+1/2)/H`` moves the
        resident water so the uniform z-star stretch is undone.

    entry_heat : (..., nlev) or None
        Temperature content carried by the entering water, ``entry * T_in``
        [K m/s].  None = ``entry * T`` of the receiving layer (the water
        arrives at the local temperature: no heat change -- runoff at SST,
        NEMO ``rnf_tsc``).  Channels whose heat content is ALREADY in the
        surface heat flux (rain at air temperature, evaporation at SST, the
        SSS-restoring water: NEMO ``blk_oce_2`` / ``sbcssr`` and our
        ``compute_omip2_surface_forcing``) must pass 0 for their share, or
        that heat is counted twice (codex review 2026-09-05).

    Returns
    -------
    (dS_dt, dT_dt) : (..., nlev) tendencies [PSU/s, K/s], zero on land
    """
    F = jnp.asarray(F_rate) * mask
    h = jnp.where(h_k > h_floor, h_k, 0.0)
    H = jnp.sum(h, axis=-1, keepdims=True)
    if F_entry is None:
        # everything enters at the surface
        entry = jnp.concatenate(
            [F[..., None], jnp.zeros_like(h[..., 1:])], axis=-1)
    else:
        entry = jnp.asarray(F_entry) * mask[..., None]
    # W(k+1/2) = water entered at or above cell k minus the uniform stretch
    # of the column above the interface; = F H_below/H when all enters at
    # the top; 0 at the bottom in every case (sum entry == F).
    H_above_if = jnp.cumsum(h, axis=-1)                  # interface under cell k
    W_below = (jnp.cumsum(entry, axis=-1)
               - F[..., None] * H_above_if / jnp.maximum(H, h_floor))
    # Interface tracer = UPWIND cell: the cell above for water entering
    # (W > 0, downward), the cell below for water leaving (evaporation, ice
    # growth: W < 0, upward).  Taking the cell above for both signs is
    # anti-diffusive under evaporation and sharpens the halocline (GLM
    # review 2026-09-05).  No transport crosses the SURFACE interface: the
    # entering water is a per-layer SOURCE carrying S = 0 and the heat
    # ``entry_heat`` (see the parameter docs).
    S_below_cell = jnp.concatenate([S[..., 1:], S[..., -1:]], axis=-1)
    T_below_cell = jnp.concatenate([T[..., 1:], T[..., -1:]], axis=-1)
    S_at_below = jnp.where(W_below >= 0.0, S, S_below_cell)
    T_at_below = jnp.where(W_below >= 0.0, T, T_below_cell)
    flux_S_below = W_below * S_at_below
    flux_T_below = W_below * T_at_below
    # no transport crosses the surface: the entering water is a SOURCE in
    # its layer (S = 0, T = that layer's T)
    flux_S_above = jnp.concatenate(
        [jnp.zeros_like(S[..., :1]), flux_S_below[..., :-1]], axis=-1)
    flux_T_above = jnp.concatenate(
        [jnp.zeros_like(T[..., :1]), flux_T_below[..., :-1]], axis=-1)
    inv_h = jnp.where(h > h_floor, 1.0 / jnp.maximum(h, h_floor), 0.0)
    heat_in = entry * T if entry_heat is None else jnp.asarray(entry_heat) * mask[..., None]
    dS = (flux_S_above - flux_S_below) * inv_h
    dT = (flux_T_above - flux_T_below + heat_in) * inv_h
    return dS * mask[..., None], dT * mask[..., None]


def virtual_salt_flux_from_net(
    F_fw: jnp.ndarray,
    S_ref: float | jnp.ndarray,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Top-layer virtual-salt tendency [PSU/s] from a PRECOMPUTED net freshwater
    flux ``F_fw`` [kg/m²/s, +INTO ocean] (vs :func:`virtual_salt_flux`, which
    builds the net from a ``FreshwaterForcing``).  Lets a caller normalize the
    net (e.g. :func:`normalize_freshwater_net`) before the closure.

        dS/dt = -S_ref * F_fw / (rho_0 * dz_0)
    """
    # Guard thin cells: on partial-cell grids, dz_0 can be O(cm) at
    # shallow coastal cells.  Dividing by tiny dz produces huge dS/dt.
    # Zero the tendency where dz_0 < 1mm (same guard as prescribed
    # surface forcing's is_ocean threshold).
    is_wet = dz_0 > 1.0e-3
    dz_safe = jnp.maximum(dz_0, 1.0e-3)
    return jnp.where(is_wet, -S_ref * F_fw / (rho_0 * dz_safe), 0.0)


def normalize_freshwater_net(
    F_fw: jnp.ndarray,
    area: jnp.ndarray,
    mask: jnp.ndarray,
    owned_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Remove the ocean-area-weighted global mean of a freshwater flux.

    Returns ``F_fw - mean(F_fw)`` so the area integral over the ``mask`` cells is
    exactly zero.  Applied to the virtual-salt closure this CONSERVES GLOBAL SALT:
    the salt-mass tendency per area is ``-S_ref * F_fw * 1e-3`` (the top-layer
    thickness cancels), so ``∮(-S_ref*(F_fw - F_mean)*area) = 0`` (Griffies: a
    redistributive surface freshwater flux changes no salt mass).  Using the SAME
    area-mean removal that the free-surface (eta) path applies keeps volume and
    salt normalization consistent.

    The caller must pass the EFFECTIVE WET mask (ocean cells the salt flux
    actually touches), so the mean removal matches the applied flux exactly --
    ``apply_freshwater_virtual_salt_top`` uses ``mask * (h_top > 1e-3)``.

    MPI correctness (codex finding #7 + round-2 #1)
    -----------------------------------------------
    The global mean MUST be taken over OWNED cells only on a sharded run --
    a plain local ``jnp.sum`` over each rank's (owned + halo) cells, then
    nothing, is single-rank-correct but UNDER/OVER-counts halo cells on MPI;
    a naive global allreduce of the unmasked local sum would DOUBLE-COUNT MPAS
    Voronoi halo cells.  Pass ``owned_mask`` (1.0 owned, 0.0 halo;
    cf. ``conservation_mpas``) to make the reduction MPI-correct: the local
    accumulators are restricted to owned cells and reduced with the MPAS-aware
    :func:`legoesm.parallel.reductions.global_sum_if_distributed`.  This is the
    canonical owned-cell reduction (used by ``conservation_mpas``/``eta_floor``)
    keyed on ``is_multi_process()`` -- NOT ``ocean_global_sum``, which keys on
    ``is_distributed()`` and would MISS the Voronoi/MPAS MPI layout (that path
    does not arm the global halo backend, so ``is_distributed()`` is False) and
    silently return a rank-local mean.  It is identity on one rank.

    ``owned_mask=None`` (the default) keeps the BIT-IDENTICAL single-rank local
    sum — correct for the single-GPU OMIP runs and every current caller.

    SCOPE: this helper provides the correct owned-cell reduction, but the
    production MPAS callers (``ocean_pe_mpas`` virtual-salt / runoff-spread,
    ``ocean_tendency_common`` top-layer wrapper) and the SEPARATE eta freshwater
    normalization (``ocean_model_mpas.step``) do NOT yet thread ``owned_mask``;
    until they do AND the eta path uses the same owned-cell reduction, an
    MPI-sharded MPAS ``normalize_freshwater`` run still normalizes volume (eta)
    and salt with different (rank-local) corrections.  Tracked as MPAS-MPI
    freshwater work, not part of the single-rank-correct default.
    """
    w = area * mask
    # One reduction shared with the salt correction (``_global_weighted_sums``):
    # owned-masked when the caller threads ``owned_mask`` (MPI layout / Voronoi
    # SPMD), psum over the armed ocean SPMD axis, allreduce on MPI, identity
    # serially.
    num, den = _global_weighted_sums(F_fw, jnp.ones_like(w), w, owned_mask=owned_mask)
    F_mean = num / jnp.maximum(den, 1.0e-10)
    return F_fw - F_mean * mask


_DEN_FLOOR = 1.0e-10   # empty-domain denominator floor [shared by both means]
_THIN_CELL_M = 1.0e-3  # top-cell thickness below which the closure is inert [m]


def _global_weighted_sums(num_field, den_field, w, owned_mask=None):
    """``(Σ num_field*w, Σ den_field*w)`` reduced correctly on EVERY backend.

    Extracted from :func:`normalize_freshwater_net` so the freshwater mean and
    the salinity-weighted salt correction share ONE reduction.  Re-deriving it
    is exactly how the salt correction silently lost the lat-SPMD ``psum`` and
    would have applied a different lambda per latitude band.

    ``owned_mask=None`` keeps the legacy serial/lat-SPMD arm (bit-identical);
    passing it takes the MPI owned-cell arm via ``global_sum_if_distributed``
    (allreduce SUM, full VJP -> AD-safe).
    """
    from legoesm.parallel.reductions import (
        global_sum_if_distributed, spmd_reduce_axis,
    )
    w_eff = w if owned_mask is None else w * owned_mask.astype(w.dtype)
    num_local = jnp.sum(num_field * w_eff)
    den_local = jnp.sum(den_field * w_eff)
    # Ocean SPMD lanes (lat-lon "lat" bands: exact shards, no halo rows;
    # Voronoi "device" blocks: owned-masked above): ONE packed psum over the
    # armed axis, checked FIRST because ``is_multi_process()`` is False under
    # one process.  Inert on serial / MPI / cube-spmd ("face" mesh).
    _ax = spmd_reduce_axis()
    if _ax is not None:
        import jax
        num_g, den_g = jax.lax.psum(jnp.stack([num_local, den_local]), _ax)
        return num_g, den_g
    if owned_mask is None:
        return num_local, den_local
    # MPI owned-cell arm: the MPAS-aware reduction (keyed on the armed
    # partition layout via ``is_multi_process``), identity on one rank.
    return global_sum_if_distributed(jnp.stack([num_local, den_local]))


def normalized_virtual_salt_flux(
    freshwater,
    S_ref: float | jnp.ndarray,
    h_top: jnp.ndarray,
    rho_0: float,
    area: jnp.ndarray,
    mask: jnp.ndarray,
    owned_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Top-layer virtual-salt tendency [PSU/s] with GLOBAL-SALT conservation.

    NOTE (local-S interaction): the zero-global-salt property holds EXACTLY
    only for a SCALAR ``S_ref`` (``S_ref * \u222bF' dA = 0`` for the zero-mean
    ``F'``).  With the ``freshwater_salinity="local"`` array the covariance
    ``\u222bS_local F' dA`` is generally nonzero — the same residual NEMO's own
    ``sfx = emp*sss`` convention carries; the normalization then removes the
    global VOLUME imbalance while the salt closure is NEMO-faithful rather
    than exactly conservative.  The model configs REJECT the
    ``local`` + ``normalize_freshwater=True`` combination outright
    (lat-lon ``_validate_config`` / MPAS tendency gate) until a joint
    volume+salt correction exists.

    Shared by the MPAS (``apply_freshwater_virtual_salt_top``) and lat-lon cores
    so the OMIP global-freshwater correction is implemented ONCE.  Removes the
    area-mean of the PHYSICAL freshwater (P-E+R+ice -- NOT the ``restoring``
    channel, a local relaxation that must not be globally redistributed) over the
    EFFECTIVE WET mask (cells the salt flux actually touches; the closure zeroes
    ``h_top<=1mm``) so the mean removal exactly matches the applied flux.  The
    (un-normalized) restoring channel is re-added.

    ``owned_mask`` (optional, codex finding #7) makes the global-mean reduction
    MPI/SPMD-correct over OWNED cells; ``None`` keeps the bit-identical
    single-rank local sum (see :func:`normalize_freshwater_net`).
    """
    F_phys = (freshwater.precip - freshwater.evap
              + freshwater.runoff + freshwater.ice_fw)
    wet = mask * (h_top > 1.0e-3).astype(mask.dtype)
    F_phys = normalize_freshwater_net(F_phys, area, wet, owned_mask=owned_mask)
    restoring = getattr(freshwater, "restoring", None)
    F_fw = F_phys if restoring is None else (F_phys + restoring)
    return virtual_salt_flux_from_net(F_fw, S_ref, h_top, rho_0)


def joint_volume_salt_virtual_salt_flux(
    freshwater,
    S_local: jnp.ndarray,
    h_top: jnp.ndarray,
    rho_0: float,
    area: jnp.ndarray,
    mask: jnp.ndarray,
    owned_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Top-layer virtual-salt tendency [PSU/s] with the LOCAL salinity closure
    AND both global budgets closed FOR THE PHYSICAL CHANNEL (restoring is
    deliberately excepted; see SCOPE below) -- the "joint volume+salt correction" that
    :func:`normalized_virtual_salt_flux` names as the missing piece.

    WHY THIS EXISTS.  The fixed-``S_ref`` closure removes salt at the S_ref rate
    no matter how little is present, so at a river mouth where ``S_local -> 0``
    nothing bounds it: the eORCA1 d90 state carries 111 cells with S < 0 (min
    -22.31 PSU), all in the top 6.6 m, all within 3 cells of land, the worst at
    the Amazon mouth.  NEMO's own convention (``tra_sbc``: ``sfx = emp*sss``)
    uses the LOCAL surface salinity, which decays toward zero and cannot cross
    it.  That convention was previously unusable here because ``local`` +
    ``normalize_freshwater=True`` is rejected: a zero-mean FRESHWATER flux does
    not give a zero-mean SALT flux once the multiplier varies in space.

    TWO CONSTRAINTS NEED TWO CORRECTIONS:

    * VOLUME -- ``F' = F - <F>_A`` (:func:`normalize_freshwater_net`) so
      ``∮F' dA = 0`` and the free-surface/volume budget is untouched.  This
      is exactly what the S_ref path already does, and is kept bit-identical.
    * SALT -- the salt-mass tendency per unit area is ``-S_local * F'`` (the
      top-layer thickness cancels), whose integral ``∮S_local F' dA`` is the
      nonzero covariance.  Remove ITS area mean as well, so
      ``∮(S_local F')' dA = 0`` exactly.

    Both means are removed with the SAME area-weighted, owned-cell reduction
    helper, so the MPI/SPMD correctness and AD-safety of the existing path carry
    over unchanged (``global_sum_mpi``, full VJP).

    THE CORRECTION IS SALINITY-WEIGHTED, NOT UNIFORM.  Subtracting a uniform
    offset ``<S F'>`` would close the budget but REINTRODUCE the very defect
    this fixes: at a cell with ``S_local = 0`` the salt flux would be
    ``-<S F'>``, which for a negative global mean drives that cell below zero.
    Instead the correction is distributed in proportion to the local salinity,

        b = max(S_local, 0)
        G = b*(F' - lambda),   lambda = ∮b F' dA / ∮b dA

    which gives ``∮G dA = 0`` EXACTLY, and ``G = 0`` exactly wherever ``S <= 0``.

    The basis ``b`` is NONNEGATIVE and is used in BOTH the numerator and the
    correction.  Two reasons, both load-bearing:

    * Using signed ``S`` in the denominator is unsafe precisely in the state
      this function repairs: with negative cells present ``∮S dA`` can pass
      through zero while ``∮S F' dA`` does not, which would silently drop the
      correction.
    * Using signed ``S`` in the numerator against a ``max(S,0)`` correction is a
      SUPPORT MISMATCH -- a cell with ``S < 0`` would get the flux ``S*F'`` and
      no correction, so it would be neither inert nor conservative.  Matching
      the support makes nonpositive cells fully inert, which is also the right
      physics: there is no salt there to remove.

    SCOPE OF THE TWO CLAIMS (narrowed after adversarial review):

    * The zero-global-salt property covers the PHYSICAL freshwater channel
      only.  ``restoring`` is a local relaxation that is deliberately NOT
      redistributed (same choice as the S_ref path), so a nonzero restoring
      channel does change total salt -- by design, not by accident.
    * ``G = 0`` at ``S = 0`` is a CONTINUOUS-TIME statement about the tendency.
      It removes the unbounded S_ref-rate extraction that drove cells to
      -22 PSU, but it is not by itself a positivity-preserving time
      integrator: an explicit step with a large enough ``dt`` can still
      undershoot from a small positive ``S``.  A positivity-preserving update
      or a dt bound remains the caller's responsibility.
    It is also the more physical choice: the redistribution acts where there is
    salt to move.  This is a REDISTRIBUTIVE correction in the Griffies sense --
    it changes no global salt, only where the salt sits -- and it is the same
    class of approximation the freshwater normalization already makes for
    volume.  It is opt-in, never silent.

    ``restoring`` is a LOCAL relaxation and must not be globally redistributed,
    so (as in the S_ref path) it is excluded from BOTH normalizations and added
    back afterwards, multiplied by the same local salinity.
    """
    F_phys = (freshwater.precip - freshwater.evap
              + freshwater.runoff + freshwater.ice_fw)
    wet = mask * (h_top > _THIN_CELL_M).astype(mask.dtype)
    # 1. volume: zero-area-mean freshwater (identical to the S_ref path)
    F_phys = normalize_freshwater_net(F_phys, area, wet, owned_mask=owned_mask)
    # 2. salt: remove ∮S F' dA, distributed in proportion to a NONNEGATIVE
    #    basis b = max(S_local, 0) so the correction vanishes exactly where the
    #    salinity does (a uniform offset would push zero-salinity cells
    #    negative) AND the denominator cannot cancel.  Using S itself as the
    #    basis is unsafe precisely in the situation this function exists to
    #    fix: with negative cells present, ∮S dA can pass through zero while
    #    ∮S F' dA does not, silently dropping the correction.
    #        lambda = ∮S F' dA / ∮b dA,   G = S*F' - lambda*b
    #    ∮G dA = ∮S F' dA - lambda*∮b dA = 0 exactly, and G = 0 wherever S <= 0.
    #    The reduction is the SHARED one (`_global_weighted_sums`) so lambda
    #    cannot drift from the freshwater mean's owned-cell / lat-SPMD handling.
    # The basis appears in BOTH the numerator and the correction.  Using signed
    # S in the numerator with a max(S,0) correction is a SUPPORT MISMATCH: a
    # cell with S < 0 would then receive the flux S*F' but no correction, so it
    # is neither inert nor conservative, and the "G = 0 where S <= 0" claim
    # would be false there (codex round-2 #2).
    basis = jnp.maximum(S_local, 0.0)
    num, den = _global_weighted_sums(basis * F_phys, basis, area * wet,
                                     owned_mask=owned_mask)
    # Empty/all-fresh domain: nothing to redistribute.  The nested `where`
    # keeps the reverse-mode VJP finite (the false branch divides by 1.0, not
    # by the vanishing denominator).
    _ok = den > _DEN_FLOOR
    lam = jnp.where(_ok, num / jnp.where(_ok, den, 1.0), 0.0)
    G_phys = (basis * (F_phys - lam)) * wet
    restoring = getattr(freshwater, "restoring", None)
    # `restoring` is masked by `wet` too: without it a dry cell that happens to
    # carry h_top > 1 mm would receive a salinity tendency (the `is_wet` gate
    # below tests thickness, NOT the land mask).
    G = G_phys if restoring is None else (G_phys + basis * restoring * wet)
    # dS/dt = -G / (rho_0 * dz_0), with the same thin-cell guard as
    # virtual_salt_flux_from_net (dz_0 can be O(cm) on partial cells).
    is_wet = h_top > _THIN_CELL_M
    dz_safe = jnp.maximum(h_top, _THIN_CELL_M)
    return jnp.where(is_wet, -G / (rho_0 * dz_safe), 0.0)


def resolve_runoff_spread_arg(config):
    """Resolve the NEMO runoff-depth spread argument from a model config.

    Shared by the lat-lon C-grid (``LatLonCGridOceanModel``) and the MPAS
    Voronoi core (``mpas_ocean_baroclinic_tendencies``) so the map/scalar
    SELECTION + MUTUAL-EXCLUSION guard live in ONE place — no per-grid
    copy-paste of the same control flow (repo rule: no duplicate numerics
    across grids).

    Reads two OPTIONAL config fields:

    * ``runoff_depth_spread_m`` — a FLAT scalar spread depth [m] (NEMO
      ``rn_dep_max`` flat mode; every river spreads over the same depth).
    * ``runoff_depth_spread_map`` — a PER-CELL NEMO ``ln_rnf_depth_ini`` map
      [m] (array or ``None``): depth proportional to the local climatological
      runoff maximum, so small Arctic/Siberian rivers stay near-surface while
      the Amazon spreads to ~150 m.  Build with
      :func:`legoesm.ocean.forcing.runoff_depth.nemo_runoff_depth_map`.

    The two are MUTUALLY EXCLUSIVE (a flat depth OR the per-cell map, not
    both).  Returns the value to pass as ``runoff_spread_m`` to
    :func:`runoff_spread_virtual_salt_tendency_3d`:

    * the map as a JAX array when ``runoff_depth_spread_map`` is set;
    * the scalar float when only ``runoff_depth_spread_m > 0``;
    * ``None`` when neither is active (caller uses the legacy top-cell
      closure — bit-identical to the pre-spread path).
    """
    _m = getattr(config, "runoff_depth_spread_m", 0.0)
    spread_m = float(_m) if _m is not None else 0.0
    spread_map = getattr(config, "runoff_depth_spread_map", None)
    if spread_map is not None and spread_m > 0.0:
        raise ValueError(
            "runoff_depth_spread_map and runoff_depth_spread_m are "
            "mutually exclusive — pick the NEMO ln_rnf_depth_ini "
            "per-cell map OR the flat spread depth.")
    if spread_map is not None:
        return jnp.asarray(spread_map)
    if spread_m > 0.0:
        return spread_m
    return None


def runoff_spread_layer_fractions(h_k, mask, spread):
    """NEMO ``sbcrnf`` runoff depth weights: per-level FRACTION of each level
    lying above the spread depth (a level straddling it counts its above-
    depth part), zero on dry levels and land, and the resulting
    ``h_rnf = min(spread, wet depth)``.  ``spread`` is a scalar or an
    already-broadcast ``(..., 1)`` per-cell depth.  Shared by the virtual
    (salt) and real (volume) closures so both spread a river over the same
    levels."""
    cum_above = jnp.cumsum(h_k, axis=-1) - h_k          # depth of level top
    h_safe = jnp.maximum(h_k, 1.0e-3)
    w_frac = jnp.clip((spread - cum_above) / h_safe, 0.0, 1.0)
    wet_lvl = (h_k > 1.0e-3) & (mask[..., None] > 0.5)
    w_frac = jnp.where(wet_lvl, w_frac, 0.0)
    h_rnf = jnp.sum(w_frac * h_k, axis=-1)
    return w_frac, h_rnf


def runoff_entry_profile(fw, h_k, mask, rho_0: float, runoff_spread_m):
    """Per-level water ENTRY rate [m/s] of the runoff channel under the real
    closure: NEMO ``sbc_rnf_div`` injects ``rnf/(rho0 h_rnf)`` into the
    divergence of every level down to ``h_rnf``, i.e. the river water enters
    each level in proportion to its thickness inside ``h_rnf``.  Returns
    ``(..., nlev)`` summing to ``runoff/rho_0`` per column (0 on land)."""
    _spread = jnp.asarray(runoff_spread_m)
    if _spread.ndim > 0:
        _spread = _spread[..., None]
    w_frac, h_rnf = runoff_spread_layer_fractions(h_k, mask, _spread)
    R = jnp.asarray(fw.runoff) / rho_0 * mask
    frac = w_frac * h_k / jnp.maximum(h_rnf, 1.0e-3)[..., None]
    return R[..., None] * frac


def real_freshwater_entry(fw, F_rate, h_k, mask, rho_0: float, T,
                          runoff_spread_m=None):
    """Per-level water entry ``(F_entry, entry_heat)`` for
    :func:`real_freshwater_dilution_tendencies` on either core.

    ``F_rate`` is the (normalised) total the eta channel received.  Runoff
    enters over NEMO's ``h_rnf`` when a spread depth is configured, all else
    at the surface.  Heat: rain/evaporation/restoring water carries ZERO
    tracer temperature (their heat content is in the surface heat flux,
    NEMO ``blk_oce_2`` / ``sbcssr``); runoff, ice melt water and the
    normalisation residual enter at the local temperature (NEMO ``rnf_tsc``
    at SST; no ice-melt heat-content term exists in our ice coupling either
    -- PLAUSIBLE, review item).
    """
    F = jnp.asarray(F_rate) * mask
    R = jnp.asarray(fw.runoff) / rho_0 * mask
    if runoff_spread_m is not None:
        R_entry = runoff_entry_profile(fw, h_k, mask, rho_0, runoff_spread_m)
    else:
        R_entry = jnp.concatenate(
            [R[..., None], jnp.zeros_like(h_k[..., 1:])], axis=-1)
    top_rest = F - jnp.sum(R_entry, axis=-1)
    F_entry = R_entry.at[..., 0].add(top_rest)
    # heat-in-flux channels (zero tracer temperature)
    pe = (jnp.asarray(fw.precip) - jnp.asarray(fw.evap)) / rho_0 * mask
    rest = getattr(fw, "restoring", None)
    pe = pe if rest is None else pe + jnp.asarray(rest) / rho_0 * mask
    # runoff enters at SST over the whole h_rnf (NEMO rnf_tsc = rnf*sst)
    entry_heat = (R_entry * T[..., :1]).at[..., 0].add((top_rest - pe) * T[..., 0])
    return F_entry, entry_heat


def runoff_spread_virtual_salt_tendency_3d(
    fw,
    S_ref: float | jnp.ndarray,
    h_k: jnp.ndarray,
    rho_0: float,
    mask: jnp.ndarray,
    *,
    runoff_spread_m: float,
    area: jnp.ndarray | None = None,
    normalize: bool = False,
    owned_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """FULL-COLUMN virtual-salt tendency [PSU/s] with the RUNOFF component
    spread over the top ``runoff_spread_m`` metres (NEMO ``rn_dep_max=150``).

    NEMO injects river runoff uniformly over the levels down to
    ``h_rnf = min(rn_dep_max, local depth)`` (sbcrnf.F90: ``phdivn[k] -=
    rnf/(rho0*h_rnf)`` for ``k <= nk_rnf``), instead of diluting a single
    surface cell — the Amazon/large-river plume otherwise sits too fresh,
    too shallow and too local.  Virtual-salt equivalent::

        dS/dt[k] = -S_ref * R / (rho_0 * h_rnf)   for spread levels k
        dS/dt[0] += -S_ref * F_top / (rho_0 * h_0) (all NON-runoff channels)

    Spread weights are FRACTIONAL per level (a level straddling the spread
    depth contributes only its above-depth fraction), so
    ``h_rnf = min(runoff_spread_m, wet column depth)`` exactly on any vertical
    grid (NEMO ``h_rnf = min(rn_dep_max, depth)``).  The COLUMN-INTEGRAL salt
    tendency is exactly ``-S_ref*(F_top + R)/rho_0`` — bit-identical
    conservation to the legacy top-cell closure, only the vertical
    distribution changes.

    NOTE (NEMO equivalence scope): this spreads the SALINITY dilution only;
    the freshwater VOLUME still enters the free surface at the top (the
    eta/barotropic path is unchanged), whereas NEMO injects the volume
    divergence at depth too.  Salt and volume conservation are identical;
    the small dynamic difference (where the volume convergence sits in the
    column) is a documented refinement, not a budget error.

    ``normalize=True`` reproduces :func:`normalized_virtual_salt_flux`'s
    GLOBAL-SALT closure exactly: the area-mean of the PHYSICAL flux
    (P-E+R+ice, runoff INCLUDED in the mean, restoring excluded) is removed
    from the TOP-cell channel, the raw runoff is spread — the total applied
    flux equals the legacy normalized total.

    Parameters
    ----------
    fw : FreshwaterForcing
    h_k : jax.Array, shape (..., nlev)
        ACTUAL layer thicknesses (partial-cell aware; 0 on dry levels).
    mask : jax.Array, shape (...,)
        Ocean mask (1 = ocean).
    runoff_spread_m : float or jax.Array, shape (...)
        Spread depth [m]. A SCALAR spreads every river over the same
        depth (NEMO rn_dep_max flat mode; must be > 0 — the caller gates
        the legacy top-cell path on a static config bool). A PER-CELL
        array is the NEMO ``ln_rnf_depth_ini`` mode — depth proportional
        to the local climatological runoff maximum (``h_rnf = rn_dep_max
        · rnf_max/rn_rnf_max``, floored at 1 m, capped at the local
        depth by the fractional-weight construction below), so small
        Arctic rivers stay near-surface while the Amazon spreads to
        150 m. Build the map with
        :func:`legoesm.ocean.forcing.runoff_depth.nemo_runoff_depth_map`.

    Returns
    -------
    jax.Array, shape (..., nlev)
        Salinity tendency; the caller multiplies by its land mask and
        integrates (``S += dt*dS`` or ``dS_dt += dS``).
    """
    _spread = jnp.asarray(runoff_spread_m)
    if _spread.ndim == 0:
        if float(runoff_spread_m) <= 0.0:
            raise ValueError(
                "runoff_spread_virtual_salt_tendency_3d requires "
                f"runoff_spread_m > 0 (got {runoff_spread_m}); the legacy "
                "top-cell closure handles the un-spread case.")
    else:
        # per-cell NEMO ln_rnf_depth_ini map: broadcast over levels; the
        # builder guarantees >= 1 m on wet cells (values <= top-cell
        # thickness degrade gracefully to the single-cell form).
        _spread = _spread[..., None]
    R = fw.runoff
    h_top = h_k[..., 0]
    # --- top-cell channels (everything but runoff) -------------------------
    F_top = fw.precip - fw.evap + fw.ice_fw
    restoring = getattr(fw, "restoring", None)
    if normalize:
        if area is None:
            raise ValueError("normalize=True requires `area`")
        # Mean over the SAME physical net + effective-wet mask as
        # normalized_virtual_salt_flux, so the global closure is unchanged.
        # Reuse normalize_freshwater_net (single reduction owner; MPI/SPMD-
        # correct when owned_mask is threaded — codex finding #7) instead of an
        # inline local sum.  ``F_top - F_mean*wet`` == normalize(F_top+R) - R
        # restricted to the wet mask: normalize subtracts the mean of (F_top+R)
        # from (F_top+R), and we keep R un-normalized (it is spread below), so
        # add R back after normalisation.
        wet = mask * (h_top > 1.0e-3).astype(mask.dtype)
        F_top = normalize_freshwater_net(
            F_top + R, area, wet, owned_mask=owned_mask) - R
    if restoring is not None:
        F_top = F_top + restoring
    dS_top = virtual_salt_flux_from_net(F_top, S_ref, h_top, rho_0)

    # --- runoff spread over the top runoff_spread_m ------------------------
    # Per-level FRACTIONAL weight (codex MED): a level straddling the spread
    # depth contributes only the fraction of its thickness above it, so
    # ``h_rnf = sum(w*h_k) = min(runoff_spread_m, wet column depth)`` exactly
    # (NEMO h_rnf = min(rn_dep_max, depth)) on any vertical grid — including
    # coarse/partial grids where whole-level inclusion would overshoot
    # (h=[100,100,...], spread=150 must give h_rnf=150, not 200).  The top
    # level always carries weight where wet, so shallow columns degrade
    # gracefully to the legacy single-cell form.  The conservation identity
    # is exact for any weights: sum_k (dS_col*w_k)*h_k = dS_col*h_rnf.
    w_frac, h_rnf = runoff_spread_layer_fractions(h_k, mask, _spread)
    wet_col = (h_top > 1.0e-3) & (mask > 0.5)
    h_rnf_safe = jnp.maximum(h_rnf, 1.0e-3)
    dS_rnf_col = jnp.where(wet_col, -S_ref * R / (rho_0 * h_rnf_safe), 0.0)
    dS_3d = dS_rnf_col[..., None] * w_frac
    # Land cells return exactly zero from the helper itself (codex LOW) —
    # callers' mask multiply is then a harmless no-op.
    return dS_3d.at[..., 0].add(dS_top * mask)


def salt_flux_salinity_tendency(salt_flux, dz_0, rho_0: float):
    """Top-layer salinity tendency from a REAL salt-mass flux [PSU/s].

    A salt mass flux ``F_salt`` [kg(salt)/m²/s, positive INTO the ocean] adds
    salt to the top layer of thickness ``dz_0``:

        d(S * 1e-3 * rho_0 * dz_0)/dt = F_salt   (PSU = g/kg -> kg/kg via 1e-3)
        => dS/dt = F_salt * 1e3 / (rho_0 * dz_0)

    Inverse of the ``sss_restoring`` convention
    (``salt_flux = rho_0 * dz * dS/dt * 1e-3``).  This is the REAL-salt channel
    (e.g. sea-ice brine rejection); distinct from ``virtual_salt_flux`` (the
    freshwater dilution proxy).  Same thin-cell guard (dz_0 < 1 mm -> 0).

    Parameters
    ----------
    salt_flux : array
        Salt-mass flux into the ocean [kg(salt)/m²/s].
    dz_0 : array
        Top-layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m³].

    Returns
    -------
    array
        Top-layer salinity tendency [PSU/s].
    """
    is_wet = dz_0 > 1.0e-3
    dz_safe = jnp.maximum(dz_0, 1.0e-3)
    return jnp.where(is_wet, salt_flux * 1.0e3 / (rho_0 * dz_safe), 0.0)


def freshwater_from_coupler(
    precip_total: jnp.ndarray,
    lhflx: jnp.ndarray,
    L_v: float,
    runoff_surface: jnp.ndarray | None = None,
    runoff_subsurface: jnp.ndarray | None = None,
    ice_state_old=None,
    ice_state_new=None,
    ice_config=None,
    ocean_mask: jnp.ndarray | None = None,
    dt: float = 1.0,
    surface_mass_flux: jnp.ndarray | None = None,
) -> FreshwaterForcing:
    """Compute freshwater forcing from coupler fields.

    Parameters
    ----------
    precip_total : jax.Array, shape (nCells,)
        Total precipitation [kg/m2/s].
    lhflx : jax.Array, shape (nCells,)
        Latent heat flux [W/m2], positive upward.  Used only as the
        fallback when ``surface_mass_flux`` is None — see audit F22.
    L_v : float
        Latent heat of vaporization [J/kg].  Fallback factor for
        ``lhflx``-based evap; ignored when ``surface_mass_flux`` is
        provided.
    runoff_surface : jax.Array or None, shape (nCells,)
        Surface runoff from land [kg/m2/s].
    runoff_subsurface : jax.Array or None, shape (nCells,)
        Subsurface runoff from land [kg/m2/s].
    ice_state_old, ice_state_new : SeaIceState or None
        Ice state before/after ice step. Used to compute ice freshwater.
    ice_config : SeaIceConfig or None
        Ice config with rho_ice.
    ocean_mask : jax.Array or None, shape (nCells,)
        Ocean mask (1=ocean). Used to restrict fluxes to ocean cells.
    dt : float
        Timestep [s]. Used for ice thickness change rate.
    surface_mass_flux : jax.Array or None, shape (nCells,)
        Phase-aware moisture mass flux from SurfaceToAtm
        [kg/m²/s, positive up].  Added in iter-16 of the
        Physical_Consistency cycle (audit F3).  Preferred over the
        ``lhflx / L_v`` back-derivation because the latter
        under-counts mass by ~13 % on sublimating tiles.  When
        provided, this overrides the lhflx-based evap calculation.

    Returns
    -------
    FreshwaterForcing
    """
    nCells = precip_total.shape[0]

    # Precipitation over ocean
    precip = precip_total

    # Evaporation: prefer the phase-aware surface_mass_flux when
    # available, fall back to lhflx / L_v for legacy callers.
    if surface_mass_flux is not None:
        evap = surface_mass_flux
    else:
        evap = lhflx / L_v

    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
    # dtype to the precip path so a missing runoff input does not
    # silently widen the freshwater forcing struct to f64 under x64.
    # Both fields are summed independently (codex adversarial review,
    # iter-1, bug #3) — previously, ``runoff_subsurface`` was silently
    # dropped whenever ``runoff_surface`` was ``None``.
    runoff = jnp.zeros(nCells, dtype=precip.dtype)
    if runoff_surface is not None:
        runoff = runoff + runoff_surface
    if runoff_subsurface is not None:
        runoff = runoff + runoff_subsurface

    # Ice freshwater: based on areal ice mass change.
    # ice_mass = rho_ice * h * A  (per unit area of grid cell)
    # ice_fw = -(ice_mass_new - ice_mass_old) / dt
    # Melting (mass decrease) puts freshwater into ocean (positive fw).
    # Supports both single-category and multi-category ice.
    if ice_state_old is not None and ice_state_new is not None and ice_config is not None:
        h_old = ice_state_old.h_ice.data
        h_new = ice_state_new.h_ice.data
        A_old = ice_state_old.concentration.data
        A_new = ice_state_new.concentration.data
        rho_ice = ice_config.rho_ice
        ice_mass_old = rho_ice * h_old * A_old
        ice_mass_new = rho_ice * h_new * A_new
        # Multi-category: h has more dims than precip_total; sum categories.
        # Stack the two ice-mass arrays once and reduce the trailing axes
        # together — each loop pass becomes one ``jnp.sum`` instead of two.
        n_extra = ice_mass_old.ndim - precip_total.ndim
        if n_extra > 0:
            _ice_pair = jnp.stack([ice_mass_old, ice_mass_new], axis=-1)
            for _ in range(n_extra):
                _ice_pair = jnp.sum(_ice_pair, axis=-2)
            ice_mass_old, ice_mass_new = _ice_pair[..., 0], _ice_pair[..., 1]
        ice_fw = -(ice_mass_new - ice_mass_old) / jnp.maximum(dt, 1e-10)
    else:
        ice_fw = jnp.zeros(nCells, dtype=precip.dtype)

    # Mask to ocean cells
    if ocean_mask is not None:
        precip = precip * ocean_mask
        evap = evap * ocean_mask
        runoff = runoff * ocean_mask
        ice_fw = ice_fw * ocean_mask

    return FreshwaterForcing(
        precip=precip,
        evap=evap,
        runoff=runoff,
        ice_fw=ice_fw,
        restoring=jnp.zeros_like(precip),
    )


def with_sss_restoring(
    fw: FreshwaterForcing,
    *,
    S_model_top: jnp.ndarray,
    S_target: jnp.ndarray,
    lat_deg: jnp.ndarray,
    lon_deg: jnp.ndarray,
    ice_concentration: jnp.ndarray,
    restoring_config,
) -> FreshwaterForcing:
    """Augment a ``FreshwaterForcing`` with the OMIP-2 SSS restoring FW flux.

    The restoring is added to the dedicated ``restoring`` field so
    ``net_freshwater_flux`` automatically picks it up.  Per-cell
    salt-mass exchange is delegated to the standard virtual-salt
    convention via ``virtual_salt_flux``: the restoring contribution
    is salt-conserving by construction because it acts on the FW
    side only.

    When ``restoring_config.enabled`` is False this is a no-op and
    the original ``fw`` is returned unchanged.

    Parameters
    ----------
    fw : FreshwaterForcing
        Existing freshwater forcing (precip - evap + runoff + ice_fw).
    S_model_top : array
        Top-layer model salinity [PSU] on the ocean's per-cell grid.
    S_target : array
        Target SSS climatology [PSU] interpolated onto the same grid.
    lat_deg, lon_deg : array
        Cell-centred latitude and longitude [°].
    ice_concentration : array
        Cell ice fraction in [0, 1] used to gate restoring under ice.
    restoring_config : SSSRestoringConfig
        Configuration for region masks, piston velocity, ice gating.

    Returns
    -------
    FreshwaterForcing
        Same forcing with ``restoring`` populated.
    """
    # Local import to avoid a cycle with ocean.forcing → ocean.freshwater.
    from legoesm.ocean.forcing.sss_restoring import compute_sss_restoring_flux

    if not restoring_config.enabled:
        return fw

    out = compute_sss_restoring_flux(
        S_model_top=S_model_top,
        S_target=S_target,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        ice_concentration=ice_concentration,
        config=restoring_config,
    )
    # Combine with any pre-existing ``restoring`` component (e.g.
    # multiple restoring channels stacked).
    prior = fw.restoring if fw.restoring is not None else jnp.zeros_like(out["freshwater_flux"])
    return fw._replace(restoring=prior + out["freshwater_flux"])


def refuse_multiprocess_eta_normalization(where: str) -> None:
    """Refuse rank-local eta normalization under multi-process execution.

    codex RED: the existing multi-rank fail-fast for freshwater normalization
    lives INSIDE the virtual-salt block, which the ``real_freshwater`` closure
    skips.  Both cores still normalize the eta/volume forcing with a RANK-LOCAL
    area mean, so a multi-rank run would silently apply a DIFFERENT correction
    on each rank -- a wrong number, not an error.  One helper so the two cores
    cannot drift apart (repo rule: factor shared logic, never copy-paste it).

    Remove this only when the eta normalization uses an owned-cell mask and a
    global reduction, exactly as the virtual-salt path will.
    """
    import jax as _jax

    from legoesm.parallel.reductions import is_multi_process, mpi_world_size

    if (is_multi_process() or mpi_world_size() > 1
            or _jax.process_count() > 1):
        raise NotImplementedError(
            f"{where}: normalize_freshwater with freshwater_closure="
            "'real_freshwater' uses a RANK-LOCAL area mean for the eta/volume "
            "forcing, which is incorrect across processes (each rank would "
            "subtract its own mean). Run single-process, or disable "
            "normalize_freshwater, until an owned-mask global reduction lands."
        )
