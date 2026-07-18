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


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
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
    if owned_mask is None:
        num_local = jnp.sum(F_fw * w)
        den_local = jnp.sum(w)
        # Lat-band shard_map body (ARMED lat SPMD halo backend): the sums
        # above are per-band PARTIALS over exact shards (band arrays carry no
        # halo rows — the halo is exchanged transiently inside the pad ops),
        # so psum them to the true global mean; a band-local mean would give
        # every band a different correction, breaking global salt
        # conservation and cross-band consistency.  Deliberately NOT
        # ``ocean_global_sum``: its ``is_distributed()`` arm would ALSO
        # allreduce route-A MPI rank-local sums, double-counting halo rows
        # (route-A threads ``owned_mask`` instead — the branch below).
        # Serial / MPI / cube-spmd ("face" mesh): inert -> the legacy
        # bit-identical local sum.
        from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
        if get_halo_backend() == "spmd":
            _mesh = get_spmd_mesh()
            if _mesh is not None and "lat" in tuple(_mesh.axis_names):
                import jax
                num_local, den_local = jax.lax.psum(
                    jnp.stack([num_local, den_local]), "lat")
        F_mean = num_local / jnp.maximum(den_local, 1.0e-10)
    else:
        # MPI/SPMD: restrict local accumulators to OWNED cells (no halo
        # double-count) then reduce globally.  Use the MPAS-AWARE reduction
        # ``global_sum_if_distributed`` (keyed on ``is_multi_process()``), NOT
        # ``ocean_global_sum``: the Voronoi/MPAS MPI path builds a partition
        # layout WITHOUT arming the global halo backend, so ``is_distributed()``
        # stays False there and ``ocean_global_sum`` would silently return the
        # RANK-LOCAL owned-cell mean (codex review #1).  ``global_sum_if_
        # distributed`` is the canonical owned-cell reduction used by
        # ``conservation_mpas`` / ``eta_floor``; it is identity on one rank
        # (so serial stays bit-identical) and is built on ``global_sum_mpi``
        # (allreduce SUM, full VJP -> AD-safe).
        from legoesm.parallel.reductions import global_sum_if_distributed
        w_owned = w * owned_mask.astype(w.dtype)
        num_local = jnp.sum(F_fw * w_owned)
        den_local = jnp.sum(w_owned)
        num, den = global_sum_if_distributed(jnp.stack([num_local, den_local]))
        F_mean = num / jnp.maximum(den, 1.0e-10)
    return F_fw - F_mean * mask


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
    cum_above = jnp.cumsum(h_k, axis=-1) - h_k          # depth of level top
    h_safe = jnp.maximum(h_k, 1.0e-3)
    w_frac = jnp.clip((_spread - cum_above) / h_safe, 0.0, 1.0)
    wet_lvl = (h_k > 1.0e-3) & (mask[..., None] > 0.5)
    w_frac = jnp.where(wet_lvl, w_frac, 0.0)
    h_rnf = jnp.sum(w_frac * h_k, axis=-1)
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
