"""Tendency probe harness for Phase G tier-2 fidelity comparison.

Given a frozen ocean state on the lat-lon C-grid, this module produces a
per-process breakdown of the legoESM tendencies — the legoESM-side input
to a tier-2 comparison against Veros (or any other reference model that
emits per-term diagnostics). The probe leverages the existing
``MomentumTendencyDiagnostics`` infrastructure (closure-tested in
``test_momentum_diagnostics_closure.py``) so by construction
``Σ momentum components == du/dt`` to machine precision.

The probe is timestepping-free: it calls
``latlon_cgrid_ocean_baroclinic_tendencies(..., diagnose_momentum=True)``
once on the input state, plus direct calls to ``coriolis_cgrid`` and
``compute_ocean_rho`` for components not in the baroclinic-tendency
breakdown.

For tracer per-process comparison, the caller constructs the recipe
config with the desired single process active (e.g.
``lateral_mixing scheme="none"``, ``vertical_mixing scheme="none"``,
``physics=None`` for tracer-advection-only). This is the same isolation
strategy Veros's per-term diagnostic dumps use.

Per-region metrics (``per_region_metrics``) supply the spatially-aware
breakdown — interior vs boundary vs equator vs mixed-layer vs abyssal —
needed because a single global L2 hides bugs that concentrate spatially.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import coriolis_cgrid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _static_kappa_redi_override,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
)
from legoesm.ocean.eos import compute_ocean_rho, make_eos_fn
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
    compute_isoneutral_K33_latlon,
)
from legoesm.ocean.state import LatLonCGridOceanConfig, LatLonCGridOceanState
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_ocean_jacobian,
)


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------


class LatLonProbeResult(NamedTuple):
    """Per-process tendencies on a frozen lat-lon C-grid ocean state.

    Momentum components come from ``MomentumTendencyDiagnostics``.
    Coriolis is computed separately since it is applied in the
    forward-backward step, not inside
    ``latlon_cgrid_ocean_baroclinic_tendencies``. Tracer tendencies
    are returned as totals; the caller isolates a single process by
    constructing a config in which only that process is active.

    Field naming uses Veros conventions where possible
    (``pgf_ke_u`` = KE-gradient + pressure gradient, etc.) so the
    side-by-side comparison report reads identically regardless of
    which side is being introspected.
    """
    # Momentum tendencies (du/dt, dv/dt) per process — m/s^2
    pgf_ke_u: jnp.ndarray            # KE gradient + pressure gradient
    pgf_ke_v: jnp.ndarray
    coriolis_u: jnp.ndarray          # f * v_at_u
    coriolis_v: jnp.ndarray          # -f * u_at_v
    vortcor_u: jnp.ndarray           # relative vorticity advection
    vortcor_v: jnp.ndarray
    vertadv_u: jnp.ndarray           # flux-form vertical momentum advection
    vertadv_v: jnp.ndarray
    ah_lap_u: jnp.ndarray            # harmonic lateral viscosity
    ah_lap_v: jnp.ndarray
    bh_bilap_u: jnp.ndarray          # biharmonic lateral viscosity
    bh_bilap_v: jnp.ndarray
    botdrag_u: jnp.ndarray           # bottom drag
    botdrag_v: jnp.ndarray
    av_vert_u: jnp.ndarray           # vertical viscosity (explicit path)
    av_vert_v: jnp.ndarray
    phys_u: jnp.ndarray              # surface-forcing physics
    phys_v: jnp.ndarray
    total_u: jnp.ndarray             # Σ PE components (excl. Coriolis)
    total_v: jnp.ndarray
    # Tracer tendencies — degC/s, PSU/s
    dT_dt_total: jnp.ndarray         # total dT/dt incl. GM/Redi (see below)
    dS_dt_total: jnp.ndarray
    # GM/Redi isopycnal-mixing tracer tendency. The lat-lon model applies this
    # in its tracer step (from the top-level config.gm_redi), NOT inside
    # baroclinic_tendencies — so the probe computes it explicitly to include it
    # in the tier-2 comparison, and folds it into dT_dt_total/dS_dt_total.
    dT_gm_redi: jnp.ndarray
    dS_gm_redi: jnp.ndarray
    # EOS-derived density at cell centres — kg/m^3
    rho: jnp.ndarray
    # Vertical-mixing (NEMO zdf) tracer tendency [degC/s, PSU/s] — the
    # backward-Euler vertical-diffusion increment (T_new - T_old)/dt_tracer
    # computed with the ACTIVE vertical-mixing K (TKE/KPP/constant closure),
    # i.e. the legoESM analogue of NEMO's ``ttrd_zdf`` tracer trend.  This is
    # the per-process field the NEMO tendency-match compares (tier-3 oracle);
    # zero when ``vertical_mixing scheme="none"`` (the Veros-ACC probe default).
    # Computed with the SAME shared helper the production implicit solve uses
    # (``implicit_vertical_diffusion_ocean``) — no re-derived numerics.
    # APPENDED LAST (never inserted mid-tuple) so existing positional/tuple
    # consumers and serialized snapshots keep their field order (codex MED).
    dT_zdf: jnp.ndarray = None
    dS_zdf: jnp.ndarray = None


def probe_latlon_cgrid(
    state: LatLonCGridOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    *,
    physics_fn=None,
    surface_forcing=None,
    sponge=None,
    dt: float = 300.0,
    dt_tracer: float | None = None,
    gm_redi_tracer_state: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> LatLonProbeResult:
    """Compute per-process tendencies on a frozen ocean state.

    Parameters
    ----------
    state : LatLonCGridOceanState
        Frozen ocean state (T, S, u, v, eta) to probe.
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
        Determines which scheme variants apply. For per-process
        tracer isolation, construct ``config`` with all but one
        process disabled.
    physics_fn : callable, optional
    surface_forcing : optional
    sponge : SpongeForcing, optional
    dt : float
        Timestep [s] — needed for the CFL-aware advection-flux build
        even though no time integration is performed.
    gm_redi_tracer_state : (T, S) or None
        Tracer fields fed to the GM/Redi (iso) tendency + implicit-K33
        solve ONLY — an oracle-fidelity override for models whose iso
        operator is evaluated at a different time level than the
        state's primary ``T``/``S`` (e.g. NEMO's leap-frog: the
        iso-neutral operator — density, N², slopes, gradients — is
        computed entirely on the *before* level ``Nbb``
        (``stpmlf.F90:199`` ``CALL ldf_slp(kstp, rhd, rn2b, Nbb, Nnn)``
        "before slope for standard operator"; ``traldf_iso_scheme.h90``
        gradients read ``pt_in(...,Kbb)``), while advection is
        evaluated on the *now* level ``Nnn``/``Kmm``
        (``traadv_fct.F90:354`` ``trd_tra(..., pt(:,:,:,jn,Kmm))``).
        ``None`` (default) uses ``state.T.data``/``state.S.data`` —
        byte-identical to the pre-existing behaviour for every other
        caller.

    Returns
    -------
    LatLonProbeResult
    """
    tendencies, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        state, grid, z_coord, config,
        physics_fn=physics_fn,
        surface_forcing=surface_forcing,
        sponge=sponge,
        dt=dt,
        diagnose_momentum=True,
    )

    cor_u, cor_v = coriolis_cgrid(
        state.u.data, state.v.data, grid,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
    )

    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    # Honor the recipe's EOS choice — ``compute_ocean_rho`` defaults to
    # Wright 1997 if no ``eos_fn`` is supplied; that's the wrong answer
    # when the recipe pins ``eos="veros_nonlin2"`` for ACC etc. Build
    # the EOS function from the config and pass it explicitly.
    _eos_depth = getattr(config, "eos_depth", "insitu")
    _eos_rho0 = getattr(config, "rho_0", None)
    # Geometric-depth NEMO path: EOS rho0 must match the p=rho0·g·gdept fed to
    # it (value cancels -> exact gdept).  "insitu" default -> no rho0 kwarg ->
    # byte-identical.
    _eos_mk_kw = {"rho0": _eos_rho0} if _eos_depth == "geometric" else {}
    eos_fn = make_eos_fn(
        eos=getattr(config, "eos", "wright"),
        eos_linear=getattr(config, "eos_linear", None),
        eos_veros_nonlin2=getattr(config, "eos_veros_nonlin2", None),
        eos_veros_nonlin3=getattr(config, "eos_veros_nonlin3", None),
        **_eos_mk_kw,
    )
    rho = compute_ocean_rho(state, z_coord, J, eos_fn=eos_fn,
                            eos_depth=_eos_depth, rho0=_eos_rho0)

    # GM/Redi isopycnal mixing — the lat-lon model applies this in its tracer
    # step (from the TOP-LEVEL config.gm_redi), not inside baroclinic_tendencies.
    # Compute it here so it (a) is exposed for the per-process iso comparison and
    # (b) is folded into the tracer totals to match what the model integrates.
    if getattr(config, "gm_redi", None) is not None:
        # T_iso/S_iso: the tracer fed to the iso operator.  Defaults to the
        # state's primary T/S (byte-identical to the pre-existing behaviour);
        # ``gm_redi_tracer_state`` lets an oracle-fidelity caller supply a
        # different time level (see the docstring / NEMO Kbb citation above).
        if gm_redi_tracer_state is not None:
            T_iso, S_iso = gm_redi_tracer_state
        else:
            T_iso, S_iso = state.T.data, state.S.data
        # Production (ocean_model_latlon_cgrid.py) ALWAYS threads the static
        # cos(lat) Redi override through every gm_redi_tracer_tendency_latlon /
        # compute_isoneutral_K33_latlon call site (e.g. lines 3551/3718/3756,
        # 6986/6994) — reuse the exact same exported helper so the probe never
        # silently runs with kappa_Redi held at its (wrong, non-cos-scaled)
        # equator value on any recipe with kappa_redi_lat_scaling=True.
        kappa_redi_override, kappa_redi_v_override = _static_kappa_redi_override(
            config.gm_redi, grid)
        dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(
            T_iso, S_iso, state.eta.data, state.H_bathy.data,
            grid, z_coord, config.gm_redi,
            eos=getattr(config, "eos", "wright"),
            eos_linear=getattr(config, "eos_linear", None),
            mask=state.land_mask.data,
            u_mask=state.u_mask.data, v_mask=state.v_mask.data,
            rho_0=config.constants.rho_0, g=config.constants.g,
            # #1226: config.omega (the field dino_lat_lon_model_config
            # actually threads a NEMO-recipe's pinned Omega into, mirroring
            # config.g/config.rho_0's own split from config.constants.*) --
            # NOT config.constants.Omega, which stays the unwired NamedTuple
            # default. Without this the probe silently reintroduces the
            # ldf_eiv kappa (aeiu) amplitude bias the fix removes in
            # production. getattr guards a bare LatLonCGridOceanConfig built
            # before the omega field existed (defaults to the same value
            # the field itself defaults to).
            omega=getattr(config, "omega", constants.Omega),
            kappa_redi_override=kappa_redi_override,
            kappa_redi_v_override=kappa_redi_v_override,
            dt=(dt_tracer if dt_tracer is not None else dt),
            eos_depth=_eos_depth,
        )
        if getattr(config.gm_redi, "implicit_K33", False):
            # When the vertical isoneutral diagonal is applied IMPLICITLY (Veros-
            # faithful), the explicit dT_gm above is skew-only.  Fold in the
            # implicit K_33 increment exactly as Veros folds it into dtemp_iso
            # ((new - old)/dt_tracer), so the per-process iso comparison stays
            # apples-to-apples.  Backward-Euler damping is dt-dependent ⇒ use the
            # tracer dt (Veros's dt_tracer), not the momentum dt.
            from legoesm.ocean.physics.vertical_mixing import (
                implicit_vertical_diffusion_ocean, build_dz_half,
            )
            dt_tr = dt_tracer if dt_tracer is not None else dt
            K33 = compute_isoneutral_K33_latlon(
                T_iso, S_iso, state.eta.data, state.H_bathy.data,
                grid, z_coord, config.gm_redi,
                eos=getattr(config, "eos", "wright"),
                eos_linear=getattr(config, "eos_linear", None),
                mask=state.land_mask.data,
                rho_0=config.constants.rho_0, g=config.constants.g,
                kappa_redi_override=kappa_redi_override,
                kappa_redi_v_override=kappa_redi_v_override,
                # #1226: same wall masks as the tendency call above.
                u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                dt=dt_tr,
                eos_depth=_eos_depth,
            )
            # Match the production model: zero K33 at non-wet interfaces
            # so partial-cell bottom cells never mix against below-bottom
            # values (ocean_model_latlon_cgrid masks the combined tracer
            # K with is_active[..., 1:]; without this, nemo_cap — which
            # no longer taper-kills bottom-adjacent slopes — leaks
            # through the probe; codex r7 P2).
            from legoesm.ocean.vertical import OceanPartialCellCoordinate
            if isinstance(z_coord, OceanPartialCellCoordinate):
                K33 = K33 * z_coord.is_active[..., 1:].astype(K33.dtype)
            dz_cell = z_coord.dz_ref * J[:, :, jnp.newaxis]
            dz_half = build_dz_half(dz_cell)
            mask3 = state.land_mask.data[:, :, jnp.newaxis]
            T_imp = implicit_vertical_diffusion_ocean(T_iso, K33, dz_cell, dz_half, dt_tr)
            S_imp = implicit_vertical_diffusion_ocean(S_iso, K33, dz_cell, dz_half, dt_tr)
            dT_gm = dT_gm + (T_imp - T_iso) / dt_tr * mask3
            dS_gm = dS_gm + (S_imp - S_iso) / dt_tr * mask3
    else:
        dT_gm = jnp.zeros_like(tendencies.dT_dt.data)
        dS_gm = jnp.zeros_like(tendencies.dS_dt.data)

    # --- Vertical-mixing (NEMO zdf) tracer tendency -------------------------
    # The legoESM analogue of NEMO's ``ttrd_zdf``: the backward-Euler vertical-
    # diffusion increment produced by the ACTIVE vertical-mixing closure at the
    # frozen state, (T_new - T_old)/dt_tracer.  Same construction as the
    # implicit-K33 block above (shared ``implicit_vertical_diffusion_ocean``),
    # but with the closure's tracer diffusivity K_v from the shared
    # ``compute_vertical_K_profiles`` dispatch — no re-derived numerics and no
    # duplicated closure.  scheme="none" (the Veros-ACC probe default) leaves
    # this exactly zero, so existing tier-2 comparisons are unchanged.
    _vm_cfg = getattr(getattr(config, "physics", None), "vertical_mixing", None)
    if _vm_cfg is not None and getattr(_vm_cfg, "scheme", "none") != "none":
        from legoesm.ocean.physics.vertical_mixing import (
            build_dz_half, compute_vertical_K_profiles,
            implicit_vertical_diffusion_ocean,
        )
        _dt_tr = dt_tracer if dt_tracer is not None else dt
        # K_v = the closure's TRACER diffusivity at this state (the same call
        # the production implicit solve makes).
        _K_v, _A_v = compute_vertical_K_profiles(
            state, z_coord, surface_forcing, config.physics,
        )
        _dz_cell = z_coord.dz_ref * J[:, :, jnp.newaxis]
        _dz_half = build_dz_half(_dz_cell)
        _mask3 = state.land_mask.data[:, :, jnp.newaxis]
        _T_zdf = implicit_vertical_diffusion_ocean(
            state.T.data, _K_v, _dz_cell, _dz_half, _dt_tr)
        _S_zdf = implicit_vertical_diffusion_ocean(
            state.S.data, _K_v, _dz_cell, _dz_half, _dt_tr)
        dT_zdf = (_T_zdf - state.T.data) / _dt_tr * _mask3
        dS_zdf = (_S_zdf - state.S.data) / _dt_tr * _mask3
    else:
        dT_zdf = jnp.zeros_like(tendencies.dT_dt.data)
        dS_zdf = jnp.zeros_like(tendencies.dS_dt.data)

    return LatLonProbeResult(
        pgf_ke_u=diag.KE_PGF_u.data,
        pgf_ke_v=diag.KE_PGF_v.data,
        coriolis_u=cor_u,
        coriolis_v=cor_v,
        vortcor_u=diag.vortcor_u.data,
        vortcor_v=diag.vortcor_v.data,
        vertadv_u=diag.vertadv_u.data,
        vertadv_v=diag.vertadv_v.data,
        ah_lap_u=diag.Ah_lap_u.data,
        ah_lap_v=diag.Ah_lap_v.data,
        bh_bilap_u=diag.Bh_bilap_u.data,
        bh_bilap_v=diag.Bh_bilap_v.data,
        botdrag_u=diag.botdrag_u.data,
        botdrag_v=diag.botdrag_v.data,
        av_vert_u=diag.Av_vert_u.data,
        av_vert_v=diag.Av_vert_v.data,
        phys_u=diag.phys_u.data,
        phys_v=diag.phys_v.data,
        total_u=diag.total_u.data,
        total_v=diag.total_v.data,
        dT_dt_total=tendencies.dT_dt.data + dT_gm,
        dS_dt_total=tendencies.dS_dt.data + dS_gm,
        dT_zdf=dT_zdf,
        dS_zdf=dS_zdf,
        dT_gm_redi=dT_gm,
        dS_gm_redi=dS_gm,
        rho=rho,
    )


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


class RegionMasks(NamedTuple):
    """Cell-centre region masks (shape ``(n_lat, n_lon, nlev)``).

    Masks are not necessarily disjoint — a cell can be both
    ``equator`` and ``mixed_layer``. ``wet`` is the union of all
    ocean cells (the natural denominator for a region's "share of
    the ocean" check).

    Use ``per_region_metrics`` to apply these to a tendency array; the
    array must share the cell-centre shape.
    """
    interior: jnp.ndarray
    boundary: jnp.ndarray
    equator: jnp.ndarray
    mixed_layer: jnp.ndarray
    abyssal: jnp.ndarray
    wet: jnp.ndarray


def build_region_masks(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    state: LatLonCGridOceanState,
    *,
    equator_lat_deg: float = 5.0,
    mixed_layer_depth_m: float = 100.0,
    abyssal_depth_m: float = 2000.0,
    boundary_dist_cells: int = 1,
) -> RegionMasks:
    """Build per-region masks for tier-2 comparison.

    Parameters
    ----------
    equator_lat_deg : float
        Cells with ``|lat| <= equator_lat_deg`` are in ``equator``.
    mixed_layer_depth_m : float
        Cells whose centre-depth is above ``mixed_layer_depth_m`` are
        in ``mixed_layer``.
    abyssal_depth_m : float
        Cells whose centre-depth is below ``abyssal_depth_m`` are in
        ``abyssal``.
    boundary_dist_cells : int
        Wet cells within this many cells of a dry cell are
        ``boundary``; remaining wet cells are ``interior``.

    Returns
    -------
    RegionMasks
        Boolean 3D arrays.
    """
    n_lat, n_lon, nlev = state.T.data.shape

    mask_2d = np.asarray(state.land_mask.data) > 0.5  # (n_lat, n_lon)
    lat_deg = np.degrees(np.asarray(grid.lat))         # (n_lat,)

    wet_3d = np.broadcast_to(mask_2d[:, :, None], (n_lat, n_lon, nlev))

    # Boundary: wet cells with at least one dry neighbour in (n_lat, n_lon).
    # Use a manual neighbour count instead of scipy.ndimage to keep the
    # dependency surface small.
    interior_2d = mask_2d.copy()
    for _ in range(boundary_dist_cells):
        interior_2d = (
            interior_2d
            & np.roll(mask_2d, 1, axis=0)
            & np.roll(mask_2d, -1, axis=0)
            & np.roll(mask_2d, 1, axis=1)
            & np.roll(mask_2d, -1, axis=1)
        )
    boundary_2d = mask_2d & ~interior_2d
    boundary_3d = np.broadcast_to(boundary_2d[:, :, None], (n_lat, n_lon, nlev))
    interior_3d = wet_3d & ~boundary_3d

    equator_lat_mask = np.abs(lat_deg) <= equator_lat_deg  # (n_lat,)
    equator_2d = equator_lat_mask[:, None] & mask_2d
    equator_3d = np.broadcast_to(equator_2d[:, :, None], (n_lat, n_lon, nlev))

    dz_ref = np.asarray(z_coord.dz_ref)
    z_centre = np.cumsum(dz_ref) - 0.5 * dz_ref  # (nlev,) positive downward
    ml_per_lev = z_centre <= mixed_layer_depth_m
    ab_per_lev = z_centre >= abyssal_depth_m
    ml_3d = wet_3d & ml_per_lev[None, None, :]
    ab_3d = wet_3d & ab_per_lev[None, None, :]

    return RegionMasks(
        interior=jnp.asarray(interior_3d),
        boundary=jnp.asarray(boundary_3d),
        equator=jnp.asarray(equator_3d),
        mixed_layer=jnp.asarray(ml_3d),
        abyssal=jnp.asarray(ab_3d),
        wet=jnp.asarray(wet_3d),
    )


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def per_region_metrics(
    legoesm: jnp.ndarray,
    ref: jnp.ndarray,
    masks: RegionMasks,
) -> dict[str, dict[str, float]]:
    """Compute L1/L2/L∞ + sign-match + pattern correlation per region.

    Both ``legoesm`` and ``ref`` must share the same shape, which must
    match the masks (cell-centre shape ``(n_lat, n_lon, nlev)``).

    Returns
    -------
    dict
        ``{region_name: {"L1": ..., "L2": ..., "Linf": ...,
        "sign_match": ..., "pattern_corr": ..., "n_cells": ...}}``.

        Regions with zero wet cells return ``nan`` for every metric and
        ``n_cells=0``.
    """
    if legoesm.shape != ref.shape:
        raise ValueError(
            f"legoesm shape {legoesm.shape} != ref shape {ref.shape}. "
            "Both arrays must be at the same staggered position; "
            "interpolate to a common grid before calling per_region_metrics."
        )

    out: dict[str, dict[str, float]] = {}
    diff = np.asarray(legoesm - ref)
    lego_np = np.asarray(legoesm)
    ref_np = np.asarray(ref)

    for region_name in masks._fields:
        region_mask = np.asarray(getattr(masks, region_name))
        if region_mask.shape != diff.shape:
            raise ValueError(
                f"Mask {region_name!r} shape {region_mask.shape} != array "
                f"shape {diff.shape}."
            )
        n_cells = int(region_mask.sum())
        if n_cells == 0:
            out[region_name] = {
                "L1": float("nan"), "L2": float("nan"), "Linf": float("nan"),
                "sign_match": float("nan"),
                "pattern_corr": float("nan"), "n_cells": 0,
            }
            continue

        d_region = diff[region_mask]
        l_region = lego_np[region_mask]
        r_region = ref_np[region_mask]

        L1 = float(np.mean(np.abs(d_region)))
        L2 = float(np.sqrt(np.mean(d_region ** 2)))
        Linf = float(np.max(np.abs(d_region)))

        # Sign match: fraction of cells where the two arrays have the
        # same sign (treating ``|x| < 1e-30`` as 'zero, agrees').
        eps = 1e-30
        l_sign = np.sign(np.where(np.abs(l_region) > eps, l_region, 0.0))
        r_sign = np.sign(np.where(np.abs(r_region) > eps, r_region, 0.0))
        sign_match = float(np.mean(l_sign == r_sign))

        # Pattern correlation (Pearson). Guard against constant fields.
        l_mean = float(np.mean(l_region))
        r_mean = float(np.mean(r_region))
        l_dev = l_region - l_mean
        r_dev = r_region - r_mean
        denom = float(np.sqrt(np.sum(l_dev ** 2) * np.sum(r_dev ** 2)))
        if denom > 0.0:
            pattern_corr = float(np.sum(l_dev * r_dev) / denom)
        else:
            pattern_corr = float("nan")

        out[region_name] = {
            "L1": L1, "L2": L2, "Linf": Linf,
            "sign_match": sign_match,
            "pattern_corr": pattern_corr,
            "n_cells": n_cells,
        }
    return out


def compare_probe_results(
    legoesm: LatLonProbeResult,
    ref: LatLonProbeResult,
    masks: RegionMasks,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare every field of two ``LatLonProbeResult``s region-by-region.

    Skips fields whose shape does not match the region masks (e.g.,
    u-face quantities on a cell-centre mask) and emits a single entry
    ``{"_skipped": True, "reason": "shape mismatch ..."}``. The caller
    is responsible for building face-aware masks if face-staggered
    comparison is required.

    Returns
    -------
    dict
        ``{field_name: {region_name: metrics_dict}}``.
    """
    mask_shape = masks.interior.shape  # all masks share this shape
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in legoesm._fields:
        lego_arr = getattr(legoesm, name)
        ref_arr = getattr(ref, name)
        if lego_arr.shape != mask_shape:
            out[name] = {
                "_skipped": {
                    "reason": (
                        f"array shape {lego_arr.shape} != mask shape "
                        f"{mask_shape} — supply face-aware masks to compare "
                        f"this field."
                    ),
                }
            }  # type: ignore[assignment]
            continue
        out[name] = per_region_metrics(lego_arr, ref_arr, masks)
    return out


# ---------------------------------------------------------------------------
# Face -> cell-centre interpolation for per-process MOMENTUM comparison (Q2)
# ---------------------------------------------------------------------------
#
# legoESM momentum tendencies live at velocity faces (u at n_lon+1 lon-faces,
# v at n_lat+1 lat-faces); bridged Veros tendencies live at Veros's u/v faces
# in (lat, lon, lev) layout. To use the cell-centre region masks, interpolate
# BOTH sides to cell centres.
#
# Documented delta (strategy doc §8 ledger): momentum tendencies do NOT match
# as cleanly as density. Density is point-wise in (T,S) -> matched to ~0.04
# kg/m^3. Momentum tendencies depend on each model's averaging stencils and
# formulation (legoESM vector-invariant vs Veros flux-form Coriolis/advection),
# so even a correct interpolation floors around interior corr ~0.9 (Coriolis
# anchor: corr 0.96). That floor is a genuine model difference, not a bug.


def u_face_to_centre(arr: jnp.ndarray) -> jnp.ndarray:
    """legoESM u-face (n_lat, n_lon+1, nlev) -> cell centre (n_lat, n_lon, nlev)."""
    a = jnp.asarray(arr)
    return 0.5 * (a[:, :-1, :] + a[:, 1:, :])


def v_face_to_centre(arr: jnp.ndarray) -> jnp.ndarray:
    """legoESM v-face (n_lat+1, n_lon, nlev) -> cell centre (n_lat, n_lon, nlev)."""
    a = jnp.asarray(arr)
    return 0.5 * (a[:-1, :, :] + a[1:, :, :])


def veros_u_face_to_centre(arr: jnp.ndarray) -> jnp.ndarray:
    """Bridged Veros u-tendency at u-faces (n_lat, n_lon, nlev) -> cell centre
    via periodic zonal averaging (Veros u[j] = east face of T-cell j)."""
    a = jnp.asarray(arr)
    return 0.5 * (a + jnp.roll(a, 1, axis=1))


def veros_v_face_to_centre(arr: jnp.ndarray) -> jnp.ndarray:
    """Bridged Veros v-tendency at v-faces (n_lat, n_lon, nlev) -> cell centre
    via meridional averaging (Veros v[i] = north face of T-cell i)."""
    a = jnp.asarray(arr)
    return 0.5 * (a + jnp.roll(a, 1, axis=0))


def weighted_sign_match(
    lego: jnp.ndarray, ref: jnp.ndarray, mask: jnp.ndarray, rel_floor: float = 0.1,
) -> float:
    """Sign-match restricted to cells where ``|ref| > rel_floor * max|ref|`` in
    the masked region. For stencil-floor momentum processes the plain
    sign-match is dominated by near-zero cells (sign is noise there); this
    counts only dynamically-significant cells. NOT a replacement for the plain
    metric — reported alongside it."""
    m = np.asarray(mask)
    r = np.asarray(ref)[m]
    l = np.asarray(lego)[m]
    if r.size == 0:
        return float("nan")
    thr = rel_floor * float(np.max(np.abs(r)))
    sig = np.abs(r) > thr
    if int(sig.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sign(l[sig]) == np.sign(r[sig])))


# Aggregation map: Veros process <-> the legoESM probe components summing to it.
#   du_cor <-> coriolis            (1:1)
#   du_adv <-> vortcor + vertadv   (advection: rel-vort flux + vertical adv)
#   du_mix <-> av_vert + botdrag   (vertical viscosity + bottom drag)
_MOMENTUM_AGG = {
    "coriolis_u": (("coriolis_u",), "u", "coriolis_u"),
    "coriolis_v": (("coriolis_v",), "v", "coriolis_v"),
    "du_adv": (("vortcor_u", "vertadv_u"), "u", "veros_du_adv"),
    "dv_adv": (("vortcor_v", "vertadv_v"), "v", "veros_dv_adv"),
    "du_mix": (("av_vert_u", "botdrag_u"), "u", "veros_du_mix"),
    "dv_mix": (("av_vert_v", "botdrag_v"), "v", "veros_dv_mix"),
}


def compare_momentum_at_centres(
    legoesm: LatLonProbeResult, veros_tend: dict, masks: RegionMasks,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-process momentum comparison at cell centres (Q2).

    Aggregates legoESM probe components to Veros's process groupings,
    interpolates both sides to cell centres, and returns per-region metrics
    (with an extra ``weighted_sign_match`` on the interior). Processes whose
    Veros counterpart is absent from ``veros_tend`` are skipped.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for proc, (lego_fields, kind, veros_key) in _MOMENTUM_AGG.items():
        if veros_key not in veros_tend:
            continue
        lego_sum = sum(getattr(legoesm, f) for f in lego_fields)
        lego_c = u_face_to_centre(lego_sum) if kind == "u" else v_face_to_centre(lego_sum)
        ref = veros_tend[veros_key]
        ref_c = (
            veros_u_face_to_centre(ref) if kind == "u" else veros_v_face_to_centre(ref)
        )
        metrics = per_region_metrics(lego_c, ref_c, masks)
        metrics["interior"]["weighted_sign_match"] = weighted_sign_match(
            lego_c, ref_c, masks.interior,
        )
        out[proc] = metrics
    return out


__all__ = (
    "LatLonProbeResult",
    "RegionMasks",
    "build_region_masks",
    "compare_momentum_at_centres",
    "compare_probe_results",
    "per_region_metrics",
    "probe_latlon_cgrid",
    "u_face_to_centre",
    "v_face_to_centre",
    "veros_u_face_to_centre",
    "veros_v_face_to_centre",
    "weighted_sign_match",
)
