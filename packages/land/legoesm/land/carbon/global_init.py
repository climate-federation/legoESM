"""Global carbon initial-condition map: per-PFT climate archetype builder.

Two offline stages, both run ONCE (never inside a JAX-traced model step):

* Stage A -- :func:`build_archetypes` (see
  ``docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md``): reduce
  the global ``(cell, pft)`` cover-weight x climate-feature space to a small set
  of per-PFT climate archetypes via k-means.  Pure numpy, no JAX.
* Stage B -- :func:`equilibrate_archetypes`: spin EACH archetype to a verified
  soil-carbon equilibrium with the shared semi-analytic driver
  (:func:`legoesm.land.carbon.spinup.run_semi_analytic_spinup`), batching
  archetypes that share ``(is_woody, is_evergreen, soil_class)`` into ONE vectorised
  ``step_multilayer_land`` column-block (few JAX compiles, not one per
  archetype).  The per-group construction (:func:`iter_archetype_batches`) and
  the coupled step (:func:`make_archetype_step_fn`) are factored into shared
  helpers so the drift/realism validator (``scripts/validate/
  global_carbon_ic_map.py``) re-integrates the IDENTICAL step without copying
  the config / physiology / forcing build.  The heavy JAX / land-model imports
  are deferred into those functions so Stage-A callers stay numpy-only.

The Stage-B equilibration has a DIFFERENTIABLE twin,
:func:`equilibrate_archetypes_traced`, that shares the identical construction /
step / spin-up but splices TRACED ``CarbonConfig`` SOM leaves in via
``apply_param_overrides`` so ``jax.grad`` of the per-archetype equilibrium SOC
w.r.t. those parameters flows end-to-end (the Stage-B v1 carbon-calibration
forward map; ``docs/land/stageB_carbon_calibration_plan.md``).
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from legoesm.land.carbon.climate_features import ClimateFeatures

_FEATURE_FIELDS = ("mat_k", "map_yr", "t_seasonal_amp_k", "aridity", "sw_mean_w")

# --- clustering defaults ---
# Lloyd-iteration cap: a loop-iteration count, never a config field or a bare
# kwarg-default literal (CLAUDE.md: "loop-iteration COUNTS are never
# config/trainable").
_KMEANS_MAX_ITER = 50
# Default minimum PFT cover weight for a (cell, PFT) pair to be "occupied"
# and included in clustering; caller-overridable via `build_archetypes(w_min=...)`.
_W_MIN_DEFAULT = 0.05
# Bare-ground PFT index in the CLM5_PFT_NAMES layout (column 0).  Bare is inert
# (Vc_max25 == 0, no live or soil carbon), so it is EXCLUDED from clustering and
# from equilibration; a bare cell/fraction keeps cell_archetype_id == -1 and
# contributes ZERO carbon in map_to_grid (F2).  This module assumes the CLM5 PFT
# axis with bare at index 0.
_BARE_PFT_ID = 0

# --- archetype equilibration (Stage B) defaults ---
# Time / unit conversions (exact).
_SECS_PER_DAY = 86400.0
_YEAR_DAYS = 365.0
_SECONDS_PER_YEAR = _YEAR_DAYS * _SECS_PER_DAY
_G_PER_KG = 1000.0                 # gC -> kgC
# Spin-up run controls (year counts / grid geometry are run parameters, not
# empirical coefficients -> module constants, never config/kwarg literals).
_N_SPINUP_DEFAULT = 200            # transient years before the analytic reset
_N_VERIFY_DEFAULT = 40             # verification years after the reset
_DT_DEFAULT = 3600.0               # sub-daily spin-up timestep [s]
_N_LAYERS_DEFAULT = 10             # soil layers
_SOIL_DEPTH_DEFAULT = 3.0          # soil column depth [m]
_SOIL_GROWTH_FACTOR = 1.5          # geometric soil-layer thickness growth
_U_MIN = 1.0                       # wind-speed floor [m/s] (matches run_lmip)
_DRIFT_WINDOW_MAX_YRS = 20         # cap on the drift-diagnostic averaging window
_DRIFT_FLOOR = 1e-9                # divide-safety floor for the drift fraction

# --- finidat->run carbon-IC loader: grid-match geometry (degrees, not physics) --
_DEG_PER_CIRCLE = 360.0            # full longitude circle [deg]
_DEG_HALF_CIRCLE = 180.0           # half circle [deg] (longitude wrap pivot)
_COORD_MATCH_ATOL_DEG = 1e-3       # grid-match lat/lon tolerance [deg]
# Single-point seeding: how far the nearest finidat land cell may be from the
# requested site before the match is refused.  3 deg is just over one cell of the
# 1.9x2.5 build grid's diagonal, so a legitimate coastal/island request still
# matches while an ocean-only or regional finidat cannot silently supply a cell
# from a different climate.
_POINT_MATCH_MAX_DEG = 3.0

# --- archetype spin-up carbon seeds [gC/m2] ---
# Conservative, BELOW-equilibrium initial pools that GROW IN.  The semi-analytic
# driver resets the slow wood/SOM pools analytically, so these seeds mostly set
# the transient; a modest woody seed keeps ``wood_litter > eps`` so the analytic
# wood solve engages, while herbaceous groups get NO wood (``init_carbon_state``
# zeroes it when ``woody=False``).  Over-seeding the fast pools is dangerous (it
# drives the biomass-maintenance death spiral -- see ``_biome_carbon_init`` in
# the ``scripts/validate/land_carbon_equilibrium.py`` harness).
_C_LAB_SEED = 100.0
_C_FOL_SEED = 150.0
_C_ROOT_SEED = 300.0
_C_WOOD_SEED = 3000.0              # woody groups only
_C_LIT_SEED = 500.0
_C_SOM_SEED = 5000.0

# PFT growth-form / leaf-habit classifiers (``is_woody`` / ``is_evergreen``) live
# in ``legoesm.land.surface_params`` next to ``CLM5_PFT_NAMES`` as the SINGLE
# source of truth shared with the per-pixel validator
# (``scripts/validate/land_carbon_equilibrium.py``); they are imported at
# function scope in ``iter_archetype_batches`` to keep this module's import
# numpy-only (surface_params pulls in JAX).


class ArchetypeTable(NamedTuple):
    pft_id: np.ndarray          # (n_arch,) int
    mat_k: np.ndarray
    map_yr: np.ndarray
    t_seasonal_amp_k: np.ndarray
    aridity: np.ndarray
    sw_mean_w: np.ndarray
    soil_class: np.ndarray      # (n_arch,) str


def _kmeans(x, k, seed, n_iter=_KMEANS_MAX_ITER):
    """Lloyd k-means on standardised (m, d) features; deterministic. Returns
    (labels (m,), centroids (k, d)) in standardised space."""
    rng = np.random.default_rng(seed)
    m = x.shape[0]
    k = min(k, m)
    idx = rng.permutation(m)[:k]
    cent = x[idx].copy()
    labels = np.zeros(m, int)
    for _ in range(n_iter):
        d = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new
            break
        labels = new
        for j in range(k):
            sel = labels == j
            if sel.any():
                cent[j] = x[sel].mean(0)
    return labels, cent


def build_archetypes(pft_weights, features, soil_class, land_mask, *,
                      k_per_pft=12, w_min=_W_MIN_DEFAULT, seed=0):
    """Cluster each PFT's occupied cells into ``k_per_pft`` climate archetypes.

    Bare ground (PFT index 0, ``_BARE_PFT_ID``) is excluded outright -- it is
    inert (``Vc_max25 == 0``), so it neither clusters nor equilibrates and a
    bare cell/fraction maps to zero carbon (F2).  Assumes the CLM5 PFT axis
    (bare at column 0).

    Parameters
    ----------
    pft_weights : array (ncell, npft)
        Fractional cover of each PFT in each cell.
    features : ClimateFeatures
        Per-cell climate features (see ``climate_features.py``), each
        ``(ncell,)``.
    soil_class : array (ncell,) str
        Per-cell soil texture key.
    land_mask : array (ncell,) bool
        True where the cell is land.
    k_per_pft : int
        Number of climate archetypes per PFT (upper bound; a PFT occupying
        fewer than ``k_per_pft`` cells gets fewer archetypes).
    w_min : float
        Minimum cover weight for a (cell, PFT) pair to be considered
        occupied and included in clustering.
    seed : int
        Base RNG seed; PFT ``p`` clusters with seed ``seed + p`` so results
        are reproducible and independent of loop order.

    Returns
    -------
    (ArchetypeTable, cell_archetype_id, cell_archetype_weight)
        ``cell_archetype_id`` is ``(ncell, npft)`` int, -1 where PFT ``p``
        has weight below ``w_min`` in cell ``c`` OR ``p`` is bare (always
        excluded), else the archetype index into the returned table.
        ``cell_archetype_weight`` is ``(ncell, npft)`` float, the cover weight
        (0 where id is -1, including the whole bare column).
    """
    pft_weights = np.asarray(pft_weights)
    ncell, npft = pft_weights.shape
    land = np.asarray(land_mask)
    feat = np.stack(
        [np.asarray(getattr(features, f)) for f in _FEATURE_FIELDS], axis=1
    )  # (ncell, 5)
    # PFTs that get a carbon archetype: everything EXCEPT bare ground (F2).
    cluster_pfts = [p for p in range(npft) if p != _BARE_PFT_ID]
    # Cells that actually enter clustering: land cells with >= w_min cover in at
    # least one clusterable (non-bare) PFT.  Standardise the features over ONLY
    # these cells -- ocean/masked and bare-only cells must NOT perturb the
    # mean/std metric the per-PFT k-means shares (F5).
    occupied = np.zeros(ncell, bool)
    for p in cluster_pfts:
        occupied |= pft_weights[:, p] >= w_min
    cluster_cells = occupied & land
    ref = feat[cluster_cells] if cluster_cells.any() else feat
    mu = ref.mean(0)
    sd = ref.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)  # coeff-ok: std floor
    feat_std = (feat - mu) / sd
    soil_class = np.asarray(soil_class, dtype=object)
    cell_id = np.full((ncell, npft), -1, int)
    cell_w = np.where((pft_weights >= w_min) & land[:, None], pft_weights, 0.0)
    cell_w[:, _BARE_PFT_ID] = 0.0        # bare contributes no carbon (F2)
    at = {f: [] for f in ("pft_id", *_FEATURE_FIELDS, "soil_class")}
    next_arch = 0
    for p in cluster_pfts:
        occ = np.where((pft_weights[:, p] >= w_min) & land)[0]
        if occ.size == 0:
            continue
        # NOTE (F4 follow-up): the k-means ASSIGNMENT is still unweighted; fully
        # cover-weighted k-means assignment is a documented follow-up.  The
        # per-archetype climate means + soil-class mode below ARE cover-weighted.
        labels, _ = _kmeans(feat_std[occ], k_per_pft, seed + p)
        for j in np.unique(labels):
            members = occ[labels == j]
            cell_id[members, p] = next_arch
            at["pft_id"].append(p)
            # Cover-weighted climate means: weight each member cell by PFT p's
            # cover there, so the archetype's climate reflects where the PFT
            # actually dominates, not every occupied cell equally (F4).  Members
            # have cover >= w_min > 0, so the weight sum is strictly positive.
            w_mem = pft_weights[members, p]
            for fi, fname in enumerate(_FEATURE_FIELDS):
                at[fname].append(
                    float(np.average(feat[members, fi], weights=w_mem)))
            # Cover-weighted modal soil texture: the class holding the MOST PFT-p
            # cover in the cluster (not the most member cells).
            member_soils = soil_class[members]
            cover_by_class = {
                s: float(w_mem[member_soils == s].sum())
                for s in np.unique(member_soils)
            }
            at["soil_class"].append(
                str(max(cover_by_class, key=cover_by_class.get)))
            next_arch += 1
    table = ArchetypeTable(
        pft_id=np.asarray(at["pft_id"], int),
        mat_k=np.asarray(at["mat_k"]), map_yr=np.asarray(at["map_yr"]),
        t_seasonal_amp_k=np.asarray(at["t_seasonal_amp_k"]),
        aridity=np.asarray(at["aridity"]), sw_mean_w=np.asarray(at["sw_mean_w"]),
        soil_class=np.asarray(at["soil_class"], dtype=object))
    return table, cell_id, cell_w


def dropped_cover_fraction(pft_weights, land_mask, *, w_min=_W_MIN_DEFAULT):
    """Per-cell vegetated land cover silently dropped from the carbon map (F3).

    :func:`build_archetypes` / :func:`map_to_grid` omit every (cell, PFT) pair
    whose cover is below ``w_min`` (no archetype -> zero carbon), and bare ground
    is excluded outright.  Many small sub-``w_min`` fractions in one cell can sum
    to material area, so this returns the ``(mean, max)`` over LAND cells of the
    total NON-BARE cover falling in ``0 < cover < w_min`` -- the vegetated land
    the archetype map does not represent.  A known Stage-A approximation
    (documented in ``docs/land/carbon_equilibrium_audit.md``); the follow-up
    maps each trace PFT to its nearest same-PFT archetype instead of dropping it.

    Parameters
    ----------
    pft_weights : array (ncell, npft)
        Per-gridcell fractional cover of each PFT.
    land_mask : array (ncell,) bool
        True where the cell is land.
    w_min : float
        Occupancy threshold (matches ``build_archetypes(w_min=...)``).

    Returns
    -------
    (mean_dropped, max_dropped) : (float, float)
        Mean and max over land cells of the dropped sub-``w_min`` non-bare cover
        fraction; ``(0.0, 0.0)`` when there are no land cells.
    """
    w = np.asarray(pft_weights, float)
    land = np.asarray(land_mask, bool)
    _ncell, npft = w.shape
    non_bare = np.ones(npft, bool)
    if npft > _BARE_PFT_ID:
        non_bare[_BARE_PFT_ID] = False
    sub = w[:, non_bare]
    dropped = np.where((sub > 0.0) & (sub < w_min), sub, 0.0).sum(axis=1)
    dropped = dropped[land]
    if dropped.size == 0:
        return 0.0, 0.0
    return float(dropped.mean()), float(dropped.max())


class ArchetypeBatch(NamedTuple):
    """One ``(is_woody, is_evergreen, soil_class)`` GROUP's shared coupled-step construction.

    Produced by :func:`iter_archetype_batches` and consumed BOTH by
    :func:`equilibrate_archetypes` and the drift validator
    (``scripts/validate/global_carbon_ic_map.py``), so the config / per-column
    physiology / batched forcing are built ONCE, never copied.
    """
    config: object          # MultiLayerLandConfig (scalar per group)
    land_params: object     # LandSurfaceParams, per-column (ncol_g,) arrays
    forcing_fn: object       # callable(doy, hour) -> AtmToSurface (batched)
    g_idx: np.ndarray       # (ncol_g,) archetype indices in table order
    steps_per_year: int     # round(seconds_per_year / dt)
    t_init: object          # (ncol_g,) land-state initial temperature [K]
    soil_frozen_fraction: object  # (ncol_g,) annual frozen fraction [-] (perennial-frost index)


_CANOPY_STOMATAL_MODELS = ("ball_berry", "medlyn")


def iter_archetype_batches(table: ArchetypeTable, *, n_layers, soil_depth, dt,
                           carbon_overrides=None,
                           nsc_gated_respiration=False,
                           cold_deciduous_dormancy=False,
                           leaf_c_resorption_frac=0.0,
                           stomatal_model="ball_berry"):
    """Build the per-``(is_woody, is_evergreen, soil_class)`` GROUP construction
    shared by the archetype equilibration (:func:`equilibrate_archetypes`) and
    the drift validator (``scripts/validate/global_carbon_ic_map.py``).

    Both must run the IDENTICAL coupled land+carbon step from a given IC, so
    the ``MultiLayerLandConfig``, the per-column ``LandSurfaceParams`` (CLM5 PFT
    physiology), and the ``vmap``-batched climatological ``forcing_fn`` are
    built ONCE here rather than copied into each caller.  Archetypes are grouped
    by ``(is_woody, is_evergreen, soil_class)`` -- the config knobs that must be
    scalar per group (the ``carbon.woody`` + ``carbon.evergreen`` flags and the
    soil hydraulics) -- so each group is one vectorised ``step_multilayer_land``
    column-block; the per-archetype PFT physiology and climate ride along as
    per-column arrays.

    Parameters
    ----------
    table : ArchetypeTable
        Stage-A archetypes (``pft_id`` indexes ``CLM5_PFT_NAMES``;
        ``soil_class`` keys ``SOIL_TEXTURE_VG``).
    n_layers : int
        Soil layers.
    soil_depth : float
        Soil column depth [m].
    dt : float
        Sub-daily timestep [s] (sets ``steps_per_year``).
    carbon_overrides : dict[str, jax.Array] | None
        Optional ``{CarbonConfig field name -> traced scalar}`` map applied to
        EACH group's ``CarbonConfig`` via
        :func:`legoesm.core.param_overrides.apply_param_overrides` BEFORE the
        step / spin-up is built.  ``None`` (default) is the static Stage-A/-B
        path -- the config keeps Python-float leaves and no ``legoesm.training``
        import happens.  A dict makes the named SOM fields (``tor_som_active`` /
        ``tor_som_slow`` / ``tor_som_passive`` / ``f_active_to_slow`` /
        ``f_slow_to_passive`` / ``som_freeze_floor`` / ``Q10_het_exp`` /
        ``cwd_humification_eff`` ...) TRACED arrays that flow through the coupled
        ``lax.scan`` spin-up, so ``jax.grad`` of the equilibrium SOC w.r.t. those
        parameters is computable (:func:`equilibrate_archetypes_traced`).
        ``apply_param_overrides`` raises on any field not on ``CarbonConfig``.

    Returns
    -------
    list[ArchetypeBatch]
        One batch per group; the union of the ``g_idx`` arrays covers every
        archetype exactly once (table order preserved within a group).
    """
    # Deferred (function-scope) imports: keep Stage-A / module import numpy-only.
    import jax
    import jax.numpy as jnp

    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    from legoesm.land.soil_thermal import SoilThermalConfig
    from legoesm.land.richards import RichardsConfig
    from legoesm.land.soil_texture import SOIL_TEXTURE_VG
    from legoesm.land.surface_params import (
        CLM5_PFT_NAMES, PARAM_NAMES, array_to_params, clm5_pft_table,
        is_cold_deciduous, is_evergreen, is_woody,
    )
    from legoesm.land.carbon.carbon_cycle import annual_frozen_fraction
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.stomata import StomataConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.climate_forcing import make_climatological_forcing

    # Canopy switch: the surface scheme is the two-leaf canopy (the land
    # default), and the big-leaf StomataConfig MIRRORS the same stomatal model
    # so any big-leaf re-derivation agrees with the canopy -- no hidden choice.
    if stomatal_model not in _CANOPY_STOMATAL_MODELS:
        raise ValueError(f"unknown stomatal_model {stomatal_model!r}; "
                         f"expected one of {_CANOPY_STOMATAL_MODELS}")
    surface_scheme = TwoLeafCanopyConfig(stomatal_model=stomatal_model).validate()
    stomata_cfg = StomataConfig(enabled=True, stomata_model=stomatal_model)

    pft_id = np.asarray(table.pft_id, int)
    # Bare ground is inert (no carbon) and must never be equilibrated; a bare
    # archetype indicates a mis-built table (build_archetypes drops bare) (F2).
    if np.any(pft_id == _BARE_PFT_ID):
        raise ValueError(
            f"iter_archetype_batches: archetype table contains bare ground "
            f"(pft_id == {_BARE_PFT_ID}); bare is inert and is excluded from "
            f"clustering + equilibration by build_archetypes. A bare archetype "
            f"indicates a mis-built table.")
    soil_class = np.asarray(table.soil_class, dtype=object)
    mat_k = np.asarray(table.mat_k, float)
    map_yr = np.asarray(table.map_yr, float)
    t_seas = np.asarray(table.t_seasonal_amp_k, float)
    sw_mean = np.asarray(table.sw_mean_w, float)
    n_arch = pft_id.shape[0]
    steps_per_year = int(round(_SECONDS_PER_YEAR / dt))
    pft_table = clm5_pft_table()  # (n_pft, 12) in PARAM_NAMES column order

    def _build_forcing_fn(mat_g, tamp_g, sw_g, precip_g):
        """Per-group batched climatological forcing.  The group's climate is
        bound as fn ARGS (not captured from the loop) so late binding cannot
        collapse every group's forcing onto the last group."""
        def forcing_fn(doy, hour):
            # vmap the single-column climate forcing over the group's
            # archetypes, then drop the trailing length-1 column axis.
            batched = jax.vmap(
                make_climatological_forcing, in_axes=(0, 0, 0, 0, None, None),
            )(mat_g, tamp_g, sw_g, precip_g, doy, hour)
            return jax.tree.map(
                lambda x: jnp.squeeze(x, axis=1) if x.ndim >= 2 else x, batched)
        return forcing_fn

    # Group archetypes by (woody, evergreen, soil_class) -> a shared scalar
    # config, preserving table order within each group.  The leaf-habit split
    # (evergreen vs deciduous) is a third key so tropical/needleleaf-EVERGREEN
    # archetypes equilibrate with continuous phenology while deciduous ones keep
    # the DALEC Gaussian pulse -- carbon.evergreen, like carbon.woody, is a
    # scalar per group.  (Herbaceous PFTs are never evergreen, so this adds at
    # most one extra woody group per soil texture, not a full doubling.)
    groups: dict = {}
    for a in range(n_arch):
        pft_name = CLM5_PFT_NAMES[pft_id[a]]
        key = (is_woody(pft_name), is_evergreen(pft_name),
               is_cold_deciduous(pft_name), str(soil_class[a]))
        groups.setdefault(key, []).append(a)

    batches: list[ArchetypeBatch] = []
    for (group_woody, group_evergreen, group_cold_deciduous, soil), members \
            in groups.items():
        g_idx = np.asarray(members, int)
        carbon_cfg = CarbonConfig(
            scheme="differland", woody=group_woody,
            evergreen=group_evergreen,
            cold_deciduous=group_cold_deciduous,
            nsc_gated_respiration=nsc_gated_respiration,
            cold_deciduous_dormancy=cold_deciduous_dormancy,
            leaf_c_resorption_frac=leaf_c_resorption_frac,
            C_lab_init=_C_LAB_SEED, C_fol_init=_C_FOL_SEED,
            C_root_init=_C_ROOT_SEED, C_wood_init=_C_WOOD_SEED,
            C_lit_init=_C_LIT_SEED, C_som_init=_C_SOM_SEED)
        if carbon_overrides:
            # Splice the TRACED SOM leaves into this group's config so they flow
            # through the coupled spin-up (SegmentForcing/SCM-RCE override
            # doctrine); apply_param_overrides raises on any unknown
            # CarbonConfig field.  The helper lives in legoesm.core, NOT
            # legoesm.training: importing it from the training layer dragged
            # land -> training -> tuning -> driver.config -> atmosphere and
            # broke the component-independence contract.
            from legoesm.core.param_overrides import apply_param_overrides
            carbon_cfg = apply_param_overrides(carbon_cfg, carbon_overrides)
        config = MultiLayerLandConfig(
            bulk_scheme="most",
            snow_albedo_feedback=True,
            soil_grid=SoilGridConfig(
                n_layers=n_layers, total_depth=soil_depth,
                growth_factor=_SOIL_GROWTH_FACTOR),
            hydraulics=SoilHydraulicsConfig(**SOIL_TEXTURE_VG[soil]),
            thermal=SoilThermalConfig(),
            richards=RichardsConfig(),
            carbon=carbon_cfg,
            stomata=stomata_cfg,
            surface_scheme=surface_scheme,
        )
        # Per-archetype PFT physiology -> per-column LandSurfaceParams.  The
        # CLM5 table rows are column-for-column PARAM_NAMES, so array_to_params
        # populates exactly the fields step_multilayer_land / compute_effective_
        # beta read (Vc_max25, LCMA, g1, root_depth, theta_wp, theta_fc,
        # albedo_veg, emissivity, z0).
        rows = pft_table[jnp.asarray(pft_id[g_idx])]  # (ncol_g, 12)
        land_params = array_to_params(rows, PARAM_NAMES)
        # Per-archetype climate (static per column).
        mat_g = jnp.asarray(mat_k[g_idx])
        tamp_g = jnp.asarray(t_seas[g_idx])
        sw_g = jnp.asarray(sw_mean[g_idx])
        precip_g = jnp.asarray(map_yr[g_idx] / _SECONDS_PER_YEAR)
        forcing_fn = _build_forcing_fn(mat_g, tamp_g, sw_g, precip_g)
        # Per-column perennial-frost index: the annual frozen fraction of THIS
        # group's climatological temperature cycle (mat + seasonal amplitude),
        # the same climate the forcing drives the spin-up with.  Computed here
        # (a priori, param-independent) and threaded into the coupled carbon
        # step so perennially-frozen high-latitude columns accumulate permafrost
        # SOC.  ``carbon_cfg`` supplies the (default, never-trained)
        # ``som_freeze_width_K`` -- IDENTICAL to the width
        # ``precompute_fast_analytic_inputs`` uses, so the closed-form surrogate
        # and this spin-up apply a byte-identical ``phi``.
        phi_g = annual_frozen_fraction(mat_g, tamp_g, carbon_cfg)
        # Archetypes carry no hemisphere; the land-state initial temperature is
        # the group mean-annual temperature (matches the NH-phased forcing).
        batches.append(ArchetypeBatch(
            config=config, land_params=land_params, forcing_fn=forcing_fn,
            g_idx=g_idx, steps_per_year=steps_per_year, t_init=mat_g,
            soil_frozen_fraction=phi_g))
    return batches


def make_archetype_step_fn(config, land_params, *, dt, soil_frozen_fraction=None):
    """Build the coupled land+carbon step for one archetype batch.

    Returns ``step_fn(state, carbon, forcing, doy) -> (new_state, new_carbon,
    diag)``: it advances ``step_multilayer_land`` (state + carbon) and
    reconstructs the carbon-flux breakdown consistently (the SAME per-column
    ``land_params`` GPP the coupled step used).  Shared by
    :func:`equilibrate_archetypes` (whose semi-analytic solve reads ``diag``)
    and the drift validator (which discards ``diag`` and keeps only the
    state/carbon trajectory), so BOTH run the byte-identical coupled step.

    Note: the state/carbon evolution is driven PURELY by
    ``step_multilayer_land``; ``diag`` is diagnostic only (it never feeds the
    update), so a caller that needs only the pool trajectory still integrates
    the exact same dynamics.
    """
    import jax.numpy as jnp

    from legoesm.land.soil_grid import make_soil_grid
    from legoesm.land.carbon_diagnostics import reconstruct_carbon_diagnostics
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics

    ncol_g = land_params.root_depth.shape[0]
    # Archetypes carry no hemisphere; run NH-phased (lat=0) consistently with
    # the NH-phased climatological forcing.  The equilibrium depends on the
    # seasonal amplitude + mean, not the phase.
    lat_g = jnp.zeros(ncol_g)
    grid = make_soil_grid(config.soil_grid)
    # Root-density weights matching step_multilayer_land's per-column form
    # (exp(-z/root_depth), normalised), so the reconstructed beta_soil sees
    # exactly the model's root profile.
    root_frac_g = jnp.exp(-grid.z_node[None, :] / land_params.root_depth[:, None])
    root_frac_g = root_frac_g / jnp.sum(root_frac_g, axis=-1, keepdims=True)
    theta_wp_g = land_params.theta_wp
    theta_fc_g = land_params.theta_fc
    beta_min = config.beta_min

    def step_fn(state, carbon, forcing, doy):
        new_state, _response, carbon_new, surface_out = (
            step_multilayer_land_with_diagnostics(
                state, forcing, config, _U_MIN, dt,
                lat=lat_g, carbon_state=carbon, doy=doy,
                land_params=land_params,
                soil_frozen_fraction=soil_frozen_fraction))
        # Reconstruct the flux breakdown consistently with the model: the GPP
        # is the surface scheme's own output -- the SAME array the coupled
        # step handed to step_carbon (two-leaf canopy, incl. any P-model
        # capacity; a big-leaf re-derivation here would silently disagree
        # with the step under any canopy scheme) -- AND the SAME per-column
        # soil_frozen_fraction so the reconstructed SOM decomposition losses
        # carry the SAME perennial-frost protection -- the analytic slow-pool
        # reset reads these losses to infer k_X, so they MUST be protected too
        # (else the reset would undo the protected spun-up SOM).
        diag = reconstruct_carbon_diagnostics(
            new_state, forcing, carbon, config, root_frac_g,
            theta_wp_g, theta_fc_g, beta_min, lat_g, doy, dt,
            spatial=True, land_params=land_params,
            soil_frozen_fraction=soil_frozen_fraction,
            gpp_override=surface_out.gpp)
        return new_state, carbon_new, diag

    return step_fn


def _spinup_batch(batch: ArchetypeBatch, *, n_spinup, n_verify, dt,
                  remat=False):
    """Run ONE archetype group's verified semi-analytic spin-up.

    The per-group construction (coupled step, IC, batched forcing) already lives
    in the shared :func:`iter_archetype_batches` / :func:`make_archetype_step_fn`
    helpers; this wraps the single :func:`~legoesm.land.carbon.spinup.
    run_semi_analytic_spinup` call so BOTH the static
    :func:`equilibrate_archetypes` (numpy scatter + QC) and the differentiable
    :func:`equilibrate_archetypes_traced` (jnp scatter, ``jax.grad``) drive the
    byte-identical spin-up -- the transfer-fraction kwargs are read from the
    (possibly overridden) group config in ONE place, never copy-pasted.

    Parameters
    ----------
    batch : ArchetypeBatch
        One group from :func:`iter_archetype_batches`.
    n_spinup, n_verify : int
        Transient / verification year counts.
    dt : float
        Sub-daily timestep [s].
    remat : bool
        Forwarded to :func:`~legoesm.land.carbon.spinup.run_semi_analytic_spinup`
        -- ``True`` checkpoints the per-year body so reverse-mode AD stays within
        a single-year memory budget (used by the traced path).

    Returns
    -------
    (final_carbon, annual)
        The verified-equilibrium :class:`CarbonState` ``(ncol_g,)`` and the
        per-verify-year ``annual`` diagnostics dict.
    """
    # Deferred (function-scope) imports: keep Stage-A / module import numpy-only.
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.carbon.spinup import run_semi_analytic_spinup
    from legoesm.land.multilayer_land import init_multilayer_land_state

    ncol_g = int(np.asarray(batch.g_idx).shape[0])
    step_fn = make_archetype_step_fn(
        batch.config, batch.land_params, dt=dt,
        soil_frozen_fraction=batch.soil_frozen_fraction)
    state0 = init_multilayer_land_state(ncol_g, batch.config, T_init=batch.t_init)
    carbon0 = init_carbon_state((ncol_g,), batch.config.carbon)
    _final_state, final_carbon, annual, _reset_fluxes = run_semi_analytic_spinup(
        step_fn, state0, carbon0, batch.forcing_fn,
        n_spinup=n_spinup, n_verify=n_verify,
        steps_per_year=batch.steps_per_year, dt=dt,
        cwd_humification_eff=batch.config.carbon.cwd_humification_eff,
        f_active_to_slow=batch.config.carbon.f_active_to_slow,
        f_slow_to_passive=batch.config.carbon.f_slow_to_passive,
        remat=remat)
    return final_carbon, annual


def equilibrate_archetypes(
    table: ArchetypeTable,
    *,
    n_spinup: int = _N_SPINUP_DEFAULT,
    n_verify: int = _N_VERIFY_DEFAULT,
    dt: float = _DT_DEFAULT,
    n_layers: int = _N_LAYERS_DEFAULT,
    soil_depth: float = _SOIL_DEPTH_DEFAULT,
    nsc_gated_respiration: bool = False,
    cold_deciduous_dormancy: bool = False,
    leaf_c_resorption_frac: float = 0.0,
    stomatal_model: str = "ball_berry",
    spinup_batch_fn=None,
    only_groups=None,
):
    """Spin every climate archetype to a verified soil-carbon equilibrium.

    ``spinup_batch_fn`` (same signature as :func:`_spinup_batch`) lets the
    driver wrap the per-group spin-up in a disk memo; this function itself
    stays pure.  ``only_groups`` (indices into the batch order) spins that
    subset and leaves the other archetypes at zero -- for running the groups
    as parallel jobs that each fill the driver's per-group cache.

    Runs the shared semi-analytic spin-up
    (:func:`legoesm.land.carbon.spinup.run_semi_analytic_spinup`) once per
    ``(is_woody, is_evergreen, soil_class)`` GROUP, batching that group's
    archetypes into a single vectorised ``step_multilayer_land`` column-block.
    There are only a few such groups (<= ~3 x n_soil_textures -- woody-evergreen,
    woody-deciduous, and herbaceous, which is never evergreen), so the whole
    global archetype table costs a handful of JAX compiles rather than one per
    archetype.

    Grouping rationale: ``MultiLayerLandConfig`` sub-configs (soil hydraulics,
    the ``carbon.woody`` + ``carbon.evergreen`` flags) are SCALARS, so
    archetypes that share the soil texture, the woody/herbaceous split, AND the
    evergreen/deciduous leaf habit can share one config.  Their
    PER-ARCHETYPE PFT physiology (``Vc_max25``, ``LCMA``, ``g1``,
    ``root_depth``, ``theta_wp``, ``theta_fc``, ``albedo_veg``, ``emissivity``,
    ``z0`` -- exactly the CLM5 PFT table columns) is supplied as per-column
    ``LandSurfaceParams`` arrays, which ``step_multilayer_land`` (and the
    reconstructed GPP via ``land_params``) consume.  Each archetype's climate
    (``mat_k``/``t_seasonal_amp_k``/``sw_mean_w``/``map_yr``) drives a
    ``vmap``-batched :func:`make_climatological_forcing`.

    Parameters
    ----------
    table : ArchetypeTable
        The Stage-A archetypes (``pft_id`` indexes ``CLM5_PFT_NAMES``;
        ``soil_class`` keys ``SOIL_TEXTURE_VG``).
    n_spinup, n_verify : int
        Transient and verification years for the semi-analytic spin-up.
    dt : float
        Sub-daily spin-up timestep [s].
    n_layers : int
        Soil layers.
    soil_depth : float
        Soil column depth [m].

    Returns
    -------
    (equilibrium, qc)
        ``equilibrium`` is a :class:`~legoesm.land.carbon.config.CarbonState`
        with per-archetype ``(n_arch,)`` pools.  ``qc`` is a dict of
        ``(n_arch,)`` numpy arrays -- ``gpp``/``npp`` [gC/m2/yr], ``som_kgC``,
        ``biomass_kgC`` [kgC/m2], and the total-carbon ``drift_frac_per_yr``
        over the verify segment (a drift near 0 confirms the equilibrium).
    """
    # Deferred (function-scope) imports: keep Stage-A / module import numpy-only.
    # The per-group construction (config / land_params / forcing), the coupled
    # step, and the per-group spin-up now live in the SHARED helpers above (also
    # used by the drift validator + the differentiable traced variant), so this
    # function only orchestrates + scatters.
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState, som_total

    n_arch = np.asarray(table.pft_id, int).shape[0]
    pool_fields = CarbonState._fields  # ("C_lab", ..., "C_som_passive")
    batches = iter_archetype_batches(
        table, n_layers=n_layers, soil_depth=soil_depth, dt=dt,
        nsc_gated_respiration=nsc_gated_respiration,
        cold_deciduous_dormancy=cold_deciduous_dormancy,
        leaf_c_resorption_frac=leaf_c_resorption_frac,
        stomatal_model=stomatal_model)

    # Archetype-ordered output accumulators (scattered per group via g_idx).
    pools_out = {p: np.zeros(n_arch) for p in pool_fields}
    qc = {k: np.zeros(n_arch)
          for k in ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr")}

    spinup = _spinup_batch if spinup_batch_fn is None else spinup_batch_fn
    for i_batch, batch in enumerate(batches):
        if only_groups is not None and i_batch not in only_groups:
            continue
        g_idx = batch.g_idx
        t0 = time.time()
        final_carbon, annual = spinup(
            batch, n_spinup=n_spinup, n_verify=n_verify, dt=dt)
        print(f"[global_carbon_ic] group {i_batch} ({g_idx.shape[0]} archetypes) "
              f"equilibrated in {time.time() - t0:.0f} s", flush=True)

        # Scatter the verified equilibrium pools back to archetype order.
        for p in pool_fields:
            pools_out[p][g_idx] = np.asarray(getattr(final_carbon, p))

        # Per-archetype QC from the verify segment (last-year fluxes + total-C
        # drift); pools taken from the returned equilibrium CarbonState.
        tc = sum(np.asarray(annual[p]) for p in pool_fields)  # (n_verify, ncol_g)
        win = max(2, min(_DRIFT_WINDOW_MAX_YRS, n_verify // 2))
        drift = (tc[-1] - tc[-win]) / (win * np.maximum(tc[-1], _DRIFT_FLOOR))
        biomass = sum(np.asarray(getattr(final_carbon, p))
                      for p in ("C_lab", "C_fol", "C_root", "C_wood"))

        qc["gpp"][g_idx] = np.asarray(annual["gpp"])[-1]
        qc["npp"][g_idx] = np.asarray(annual["npp"])[-1]
        qc["som_kgC"][g_idx] = np.asarray(som_total(final_carbon)) / _G_PER_KG
        qc["biomass_kgC"][g_idx] = biomass / _G_PER_KG
        qc["drift_frac_per_yr"][g_idx] = drift

    equilibrium = CarbonState(
        **{p: jnp.asarray(pools_out[p]) for p in pool_fields})
    return equilibrium, qc


def equilibrate_archetypes_traced(
    table: ArchetypeTable,
    carbon_overrides,
    *,
    n_spinup: int = _N_SPINUP_DEFAULT,
    n_verify: int = _N_VERIFY_DEFAULT,
    dt: float = _DT_DEFAULT,
    n_layers: int = _N_LAYERS_DEFAULT,
    soil_depth: float = _SOIL_DEPTH_DEFAULT,
):
    """DIFFERENTIABLE archetype equilibration: per-archetype equilibrium
    :class:`CarbonState` as a function of TRACED ``CarbonConfig`` SOM leaves.

    Identical physics to :func:`equilibrate_archetypes` (it shares the SAME
    :func:`iter_archetype_batches` construction, :func:`make_archetype_step_fn`
    coupled step, and :func:`_spinup_batch` semi-analytic driver), but the
    ``carbon_overrides`` are spliced into every group's ``CarbonConfig`` as TRACED
    arrays (via ``apply_param_overrides`` inside :func:`iter_archetype_batches`),
    so the returned pools -- in particular ``som_total(state)`` -- are
    differentiable w.r.t. the overridden SOM parameters and ``jax.grad`` /
    ``jax.jacobian`` flows end-to-end through the coupled ``lax.scan`` spin-up +
    the analytic slow-pool reset.  This is the Stage-B (v1) forward map: SOM
    parameters -> per-archetype equilibrium SOC.

    Differentiability contract
    --------------------------
    * The archetype table + cell membership are built ONCE by
      :func:`build_archetypes` (static numpy k-means) and are NOT differentiated
      -- only the equilibration is.
    * ``carbon_overrides`` (``{field name -> traced scalar}``, e.g. one scheme
      slice of ``TrainablePhysicsParams.to_overrides()['land.carbon']``) enters as
      TRACED leaves; production keeps static Python-float leaves (pass
      ``equilibrate_archetypes`` / ``carbon_overrides=None``).  The tunable SOM
      fields (``tor_som_active`` / ``tor_som_slow`` / ``tor_som_passive`` /
      ``f_active_to_slow`` / ``f_slow_to_passive`` / ``som_freeze_floor`` /
      ``Q10_het_exp`` / ``cwd_humification_eff``) are read by
      ``carbon_cycle.step_carbon_differland`` and the analytic reset, so the
      gradient flows through both the transient dynamics and the slow-pool solve.
    * Reverse-mode memory through the multi-year nested scan is bounded by
      ``remat=True`` (``jax.checkpoint`` on the per-year spin-up body) forwarded
      through :func:`_spinup_batch`; a short training spin-up is acceptable (the
      fast pools + the analytic slow reset do the heavy lifting).
    * The per-group pools are scattered into the archetype-ordered output with a
      differentiable ``jnp`` ``.at[g_idx].set`` (static indices, traced values)
      -- NO numpy on the traced path -- so AD is not severed.  No per-archetype
      QC (that reduction is numpy-only and diagnostic); use
      :func:`equilibrate_archetypes` for the QC bundle.

    Override-bound contract
    -----------------------
    ``carbon_overrides`` MUST be within each field's physical / ``__param_spec__``
    range -- in particular the transfer fractions ``f_active_to_slow`` /
    ``f_slow_to_passive`` / ``cwd_humification_eff`` and the floor
    ``som_freeze_floor`` in ``[0, 1]`` (the SOM-cascade sign convention).  The
    per-step Python fail-early guards
    (``carbon_cycle._freeze_modifier`` / :func:`~legoesm.land.carbon.config.
    validate_som_transfer_fractions`) canNOT re-check a JAX leaf (a Python bool on
    a tracer is impossible), so the bound is enforced STRUCTURALLY UPSTREAM by the
    constraint transform: the sanctioned producer
    ``training.param_collector.build_trainable_params(...).to_overrides()['land.carbon']``
    maps every trainable SOM leaf through a sigmoid onto its ``(lo, hi)`` bound for
    ALL raw inputs, so a value out of range cannot be produced.  A caller that
    hand-crafts an UNCONSTRAINED traced override bypasses that transform and is
    responsible for the bound itself (exactly as a direct
    ``CarbonConfig(f_active_to_slow=<raw>)`` construction would be).

    Parameters
    ----------
    table : ArchetypeTable
        The Stage-A archetypes (built once; static).
    carbon_overrides : dict[str, jax.Array]
        ``{CarbonConfig field name -> traced scalar}`` applied to every group's
        carbon config.  ``apply_param_overrides`` raises on an unknown field.
    n_spinup, n_verify : int
        Transient / verification year counts (a short ``n_spinup`` is acceptable
        for a training forward pass).
    dt : float
        Sub-daily spin-up timestep [s].
    n_layers : int
        Soil layers.
    soil_depth : float
        Soil column depth [m].

    Returns
    -------
    CarbonState
        Per-archetype ``(n_arch,)`` equilibrium pools; ``som_total(state)`` is the
        differentiable per-archetype SOC [gC/m2].
    """
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState

    n_arch = int(np.asarray(table.pft_id, int).shape[0])
    pool_fields = CarbonState._fields
    batches = iter_archetype_batches(
        table, n_layers=n_layers, soil_depth=soil_depth, dt=dt,
        carbon_overrides=carbon_overrides)

    # Differentiable archetype-ordered scatter: jnp zeros + .at[g_idx].set with
    # STATIC indices (g_idx from the numpy-built table) and TRACED pool values.
    pools_out = {p: jnp.zeros(n_arch) for p in pool_fields}
    for batch in batches:
        g_idx = jnp.asarray(np.asarray(batch.g_idx, int))
        # remat=True: bound reverse-mode memory to a single spin-up year.
        final_carbon, _annual = _spinup_batch(
            batch, n_spinup=n_spinup, n_verify=n_verify, dt=dt, remat=True)
        for p in pool_fields:
            pools_out[p] = pools_out[p].at[g_idx].set(getattr(final_carbon, p))

    return CarbonState(**pools_out)


def precompute_fast_analytic_inputs(
    table: ArchetypeTable,
    *,
    n_spinup: int = _N_SPINUP_DEFAULT,
    n_verify: int = _N_VERIFY_DEFAULT,
    dt: float = _DT_DEFAULT,
    n_layers: int = _N_LAYERS_DEFAULT,
    soil_depth: float = _SOIL_DEPTH_DEFAULT,
):
    """Record the SOM-parameter-INDEPENDENT inputs for the fast closed-form SOC
    forward (:func:`legoesm.land.carbon.fast_analytic.analytic_som_soc`).

    Runs ONE DEFAULT-parameter verified semi-analytic spin-up per
    ``(is_woody, is_evergreen, soil_class)`` group (NO grad -- the SAME
    :func:`iter_archetype_batches` construction, :func:`make_archetype_step_fn`
    coupled step, and :func:`~legoesm.land.carbon.spinup.run_semi_analytic_spinup`
    driver as :func:`equilibrate_archetypes`), then re-integrates ONE stationary
    year from the verified equilibrium recording, per sub-daily step:

    * the top-soil-layer temperature the SOM decomposition modifier sees
      (``new_state.T_soil[:, 0]`` -- the ``T`` passed to the coupled carbon step),

    and takes the litter -> SOM decomposition flux ``lit_to_som`` and the NPP-to-wood
    allocation ``a_wood`` (the CWD->SOM input driver, annual [gC/m2/yr]) from the
    LAST-TRANSIENT-year fluxes the analytic reset
    :func:`~legoesm.land.carbon.spinup.analytic_slow_pool_equilibrium` consumed
    (``run_semi_analytic_spinup``'s exposed ``reset_fluxes``), NOT the post-verify
    stationary year -- so the closed form reproduces that reset's active-pool INPUT
    (which sets the ``real_som`` target) exactly, rather than sampling a shifted
    phase of the still-equilibrating ~27-yr wood pool (leaving only the small ``k_X``
    soil-T residual; the passive-bias fix, see
    ``FastAnalyticInputs.a_wood_annual``).  It also records, per sub-daily step:
    * the model's own annual ALLOCATABLE NPP ``npp_pos_annual = sum_t max(NPP_day,
      0) dt`` [gC/m2/yr], the per-group ``is_woody`` / ``is_evergreen`` flags, and the
      ANNUAL-MEAN equilibrium LIVE pools ``live_ref_C_fol/C_root/C_wood`` [gC/m2] -- the
      inputs + reference for the closed-form live-pool (biomass/LAI) forward
      (:func:`legoesm.land.carbon.live_pool_forward.build_live_pool_forward`).

    Because the tunable SOM AND live-pool (allocation/residence/LCMA) parameters have
    NO feedback onto GPP / litterfall / soil energy / NPP, this trajectory + these
    inputs are independent of them, so
    :func:`~legoesm.land.carbon.fast_analytic.analytic_som_soc` and the live-pool
    forward can vary those parameters at ~zero cost while reproducing the model's
    equilibrium at the defaults.  The stationary year is re-integrated from the
    RETURNED equilibrium so the recorded turnover / annual-mean pools are evaluated at
    exactly the equilibrium (a tight match to the returned ``som_total`` and the
    live-pool reference).

    Parameters mirror :func:`equilibrate_archetypes` (the precompute should use a
    healthy ``n_spinup`` for a well-settled equilibrium; it is a ONE-TIME forward
    pass, not in the gradient loop).

    Returns
    -------
    (FastAnalyticInputs, som_total_equilibrium)
        The per-archetype fast-analytic inputs and the per-archetype REAL
        verified-equilibrium total SOM ``(n_arch,)`` [gC/m2]
        (``som_total(final_carbon)``) for the analytic-vs-spin-up match check.
    """
    # Deferred (function-scope) imports: keep Stage-A / module import numpy-only.
    import jax
    import jax.numpy as jnp

    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.carbon.config import som_total
    from legoesm.land.carbon.fast_analytic import FastAnalyticInputs
    from legoesm.land.carbon.spinup import run_semi_analytic_spinup, step_doy_hour
    from legoesm.land.multilayer_land import init_multilayer_land_state

    pft_id = np.asarray(table.pft_id, int)
    n_arch = pft_id.shape[0]
    steps_per_year = int(round(_SECONDS_PER_YEAR / dt))
    dt_days = dt / _SECS_PER_DAY

    # DEFAULT-parameter batches (carbon_overrides=None): the recorded inputs are
    # deliberately at the production defaults -- the SOM parameters are varied
    # later ONLY inside the closed form, never re-run.
    batches = iter_archetype_batches(
        table, n_layers=n_layers, soil_depth=soil_depth, dt=dt)

    lit_to_som_annual = np.zeros(n_arch)
    a_wood_annual = np.zeros(n_arch)
    soil_T_traj = np.zeros((n_arch, steps_per_year))
    # Perennial-frost index (annual frozen fraction) -- the SAME per-column
    # value iter_archetype_batches threads into the coupled step (both derived
    # from the archetype climate via annual_frozen_fraction), so the closed form
    # and the spin-up apply a byte-identical f_perma.
    soil_frozen_fraction = np.zeros(n_arch)
    real_som = np.zeros(n_arch)
    # Live-pool (biomass/LAI) forward inputs + reference (all param-independent):
    #   npp_pos_annual -- the model's own annual ALLOCATABLE NPP driving the pools;
    #   is_woody       -- the static per-group woody flag routing the structural
    #                     allocation remainder;
    #   live_ref_*     -- the spin-up's ANNUAL-MEAN equilibrium live pools, the
    #                     reference the closed-form live-pool forward is checked
    #                     against (the biomass/LAI fidelity gate).
    npp_pos_annual = np.zeros(n_arch)
    is_woody = np.zeros(n_arch)
    is_evergreen = np.zeros(n_arch)
    live_ref_C_fol = np.zeros(n_arch)
    live_ref_C_root = np.zeros(n_arch)
    live_ref_C_wood = np.zeros(n_arch)

    for batch in batches:
        g_idx = np.asarray(batch.g_idx, int)
        ncol_g = g_idx.shape[0]
        step_fn = make_archetype_step_fn(
            batch.config, batch.land_params, dt=dt,
            soil_frozen_fraction=batch.soil_frozen_fraction)
        state0 = init_multilayer_land_state(
            ncol_g, batch.config, T_init=batch.t_init)
        carbon0 = init_carbon_state((ncol_g,), batch.config.carbon)
        final_state, final_carbon, _spin_annual, reset_fluxes = run_semi_analytic_spinup(
            step_fn, state0, carbon0, batch.forcing_fn,
            n_spinup=n_spinup, n_verify=n_verify,
            steps_per_year=batch.steps_per_year, dt=dt,
            cwd_humification_eff=batch.config.carbon.cwd_humification_eff,
            f_active_to_slow=batch.config.carbon.f_active_to_slow,
            f_slow_to_passive=batch.config.carbon.f_slow_to_passive,
            remat=False)
        # The SOM active-pool inputs ``lit_to_som``/``a_wood`` are taken from the
        # LAST-TRANSIENT-year fluxes the analytic reset consumed (``reset_fluxes``,
        # the 4th return of run_semi_analytic_spinup), NOT from the post-verify
        # stationary year below.  ``analytic_slow_pool_equilibrium`` (which sets
        # ``real_som``, the target the closed form is checked against) builds
        # ``i_active = lit_to_som + cwd_humification_eff * a_wood`` from THESE fluxes,
        # so recording them makes ``analytic_som_soc`` reproduce that reference
        # reset's active-pool INPUT chain exactly (the residual is only the small
        # ``k_X`` soil-T difference).  Recording the post-verify ``a_wood`` instead
        # sampled a DIFFERENT
        # phase of the ~27-yr wood pool (the reset boosts ``C_wood`` -> higher
        # maintenance respiration -> lower NPP/allocation), so the surrogate's
        # ``a_wood`` ran ~8-11% below the value that built ``real_som`` for woody
        # columns and systematically UNDER-predicted the CWD-fed
        # active->slow->passive cascade -- the passive-pool bias this fixes.

        # Record ONE stationary year from the verified equilibrium: the top-soil
        # temperature the modifier sees + the param-independent live-pool inputs.
        # The state/carbon evolve through the byte-identical coupled step_fn; only the
        # recorded diagnostics differ from the spin-up scans.  ``max(diag.npp, 0)`` is
        # the model's OWN allocatable NPP (NPP_pos in step_carbon_differland), and the
        # per-step live pools ``cb_new.C_fol/C_root/C_wood`` are averaged over the year.
        # (``lit_to_som``/``a_wood`` come from ``reset_fluxes`` above, NOT from here.)
        def _record_step(carry, step_idx):
            st, cb = carry
            doy, hour = step_doy_hour(step_idx, dt)
            forcing = batch.forcing_fn(doy, hour)
            st_new, cb_new, diag = step_fn(st, cb, forcing, doy)
            npp_pos = jnp.maximum(diag.npp, 0.0)             # gC/m2/day allocatable NPP
            return (st_new, cb_new), (
                st_new.T_soil[:, 0],
                npp_pos, cb_new.C_fol, cb_new.C_root, cb_new.C_wood)

        _carry, (t_seq, npp_seq, cfol_seq, croot_seq, cwood_seq) = jax.lax.scan(
            _record_step, (final_state, final_carbon),
            jnp.arange(batch.steps_per_year))
        # Annual NPP input [gC/m2/yr] = sum over the year of the per-day rate;
        # live pools = ANNUAL MEAN over the stationary year [gC/m2].
        npp_ann = jnp.sum(npp_seq * dt_days, axis=0)         # (ncol_g,)
        cfol_mean = jnp.mean(cfol_seq, axis=0)               # (ncol_g,)
        croot_mean = jnp.mean(croot_seq, axis=0)             # (ncol_g,)
        cwood_mean = jnp.mean(cwood_seq, axis=0)             # (ncol_g,)

        # Reset-consistent SOM active-pool inputs (last-transient-year fluxes the
        # analytic reset consumed) -- see ``reset_fluxes`` note above.
        lit_to_som_annual[g_idx] = np.asarray(reset_fluxes.lit_to_som)
        a_wood_annual[g_idx] = np.asarray(reset_fluxes.a_wood)
        soil_T_traj[g_idx, :] = np.asarray(t_seq).T          # (ncol_g, steps_per_year)
        soil_frozen_fraction[g_idx] = np.asarray(batch.soil_frozen_fraction)
        real_som[g_idx] = np.asarray(som_total(final_carbon))
        npp_pos_annual[g_idx] = np.asarray(npp_ann)
        is_woody[g_idx] = float(bool(batch.config.carbon.woody))
        is_evergreen[g_idx] = float(bool(batch.config.carbon.evergreen))
        live_ref_C_fol[g_idx] = np.asarray(cfol_mean)
        live_ref_C_root[g_idx] = np.asarray(croot_mean)
        live_ref_C_wood[g_idx] = np.asarray(cwood_mean)

    # Per-archetype precip [kg/m2/s] matches the archetype forcing
    # (map_yr / seconds_per_year -- constant over the year, drives f_moist).
    precip = np.asarray(table.map_yr, float) / _SECONDS_PER_YEAR
    inputs = FastAnalyticInputs(
        lit_to_som_annual=jnp.asarray(lit_to_som_annual),
        a_wood_annual=jnp.asarray(a_wood_annual),
        soil_T_traj=jnp.asarray(soil_T_traj),
        soil_frozen_fraction=jnp.asarray(soil_frozen_fraction),
        precip=jnp.asarray(precip),
        dt_days=float(dt_days),
        npp_pos_annual=jnp.asarray(npp_pos_annual),
        is_woody=jnp.asarray(is_woody),
        is_evergreen=jnp.asarray(is_evergreen),
        live_ref_C_fol=jnp.asarray(live_ref_C_fol),
        live_ref_C_root=jnp.asarray(live_ref_C_root),
        live_ref_C_wood=jnp.asarray(live_ref_C_wood))
    return inputs, jnp.asarray(real_som)


def map_to_grid(cell_archetype_id, cell_archetype_weight, archetype_equilibria):
    """Cover-weighted map of per-archetype equilibrium pools onto the grid.

    ``pools[c] = sum_p w[c,p] * eq[id[c,p]]``, with ``id[c,p] == -1``
    (PFT ``p`` absent/below ``w_min`` in cell ``c``, see
    :func:`build_archetypes`) contributing 0.  NOT renormalised by
    vegetated fraction -- a cell's bare-ground remainder legitimately holds
    less carbon than a fully vegetated one.

    Parameters
    ----------
    cell_archetype_id : array (ncell, n_pft) int
        Archetype index per (cell, PFT); -1 where absent.
    cell_archetype_weight : array (ncell, n_pft) float
        Cover weight per (cell, PFT); 0 where ``cell_archetype_id`` is -1.
    archetype_equilibria : CarbonState (n_arch,)
        Per-archetype equilibrium pools from :func:`equilibrate_archetypes`.

    Returns
    -------
    CarbonState (ncell,)
        Cover-weighted pool mix per grid cell.
    """
    # Deferred (function-scope) import: only this Stage-C helper needs JAX,
    # keeping Stage-A (`build_archetypes`) callers numpy-only (module
    # docstring above).
    import jax.numpy as jnp

    cid = np.asarray(cell_archetype_id); cw = np.asarray(cell_archetype_weight, float)
    safe = np.where(cid >= 0, cid, 0)                    # gather index (masked below)
    mask = (cid >= 0).astype(float) * cw                 # (ncell, n_pft)
    fields = {}
    for f in archetype_equilibria._fields:
        vals = np.asarray(getattr(archetype_equilibria, f))    # (n_arch,)
        gathered = vals[safe]                                   # (ncell, n_pft)
        fields[f] = jnp.asarray((gathered * mask).sum(axis=1))  # (ncell,)
    return archetype_equilibria.__class__(**fields)


def map_to_grid_frozen_fraction(table, cell_archetype_id, cell_archetype_weight,
                                *, config=None):
    """Cover-weighted per-cell annual frozen fraction ``phi`` in ``[0, 1]``.

    Companion to :func:`map_to_grid`, but for the perennial-frost index ``phi``
    that seeds the permafrost/anaerobic SOM protection (``f_perma``, see
    :func:`legoesm.land.carbon.carbon_cycle.perennial_frost_protection`).  A
    coupled RUN reads this per-cell field from the finidat and threads it into
    the coupler's ``step_multilayer_land`` call so the running model applies the
    SAME protection that scaled the seeded equilibrium, preserving the deep
    permafrost SOC the IC seeds (instead of decomposing it back toward the
    unprotected equilibrium).

    Per-archetype ``phi`` is recomputed here from the archetype table climate
    (``mat_k`` / ``t_seasonal_amp_k``) via the SAME climate-only
    :func:`~legoesm.land.carbon.carbon_cycle.annual_frozen_fraction`.  ``phi``
    reads only ``config.som_freeze_width_K`` (SOM-parameter- and PFT-independent),
    so passing the SAME ``CarbonConfig`` the archetypes equilibrated at makes the
    recomputed ``phi`` BYTE-IDENTICAL to the value :func:`iter_archetype_batches`
    threaded into each archetype's coupled spin-up.  ``config=None`` (default)
    uses the production-default ``CarbonConfig``, byte-identical for a default
    build; a ``--tuned-params`` build that overrides ``som_freeze_width_K`` MUST
    pass the SAME overridden config here (the caller's equilibration config),
    else the mapped ``phi`` would drift from the seeded one.

    Reduction (INTENSIVE vs :func:`map_to_grid`'s extensive pools).  Carbon
    pools are stocks: a half-bare cell holds half the carbon, so ``map_to_grid``
    sums ``w * eq`` UNnormalised.  ``phi`` is an intensive climate FRACTION -- a
    half-bare permafrost cell has the SAME frozen climate, not half of it -- so
    it is a cover-weighted MEAN over the present PFTs (normalised by the
    vegetated weight), NOT a sum, which would spuriously dilute ``phi`` toward 0
    in partially-vegetated cells.  Cells with no vegetated cover (all PFTs below
    ``w_min``) get ``phi = 0`` (temperate default; they carry no seeded SOC to
    protect, and ``f_perma(0) ~ 1`` leaves any carbon there unchanged).

    Heterogeneous-cell fidelity (a KNOWN Stage-A approximation, NOT exact).  A
    single per-cell ``phi`` fed to the nonlinear ``f_perma`` cannot in general
    reproduce a mix of archetypes seeded at DIFFERENT ``phi`` (``f_perma`` of a
    mean != cover-weighted mean of ``f_perma``), and this is NOT rigorously
    bounded: :func:`build_archetypes` clusters PER PFT and stores cluster-wide
    cover-weighted-mean climates, so two PFTs sharing a cell can map to
    archetypes at materially different ``mat_k``/``amp`` (their cover-weighted
    mean climate is pulled toward each cluster's dominant-cover cells, see
    ``test_archetype_climate_mean_is_cover_weighted``).  The cover-weighted
    ``phi`` mean is a pragmatic single-value choice; it is not claimed to
    preserve a strongly heterogeneous cell's mixed-pool protection exactly.  It
    still fixes the FIRST-ORDER drift this wiring targets (``phi`` absent ->
    f_perma == 1 -> the WHOLE seeded high-latitude SOC decomposes, everywhere).
    Exact preservation of a heterogeneous mix would need per-PFT/subcolumn
    ``phi`` + carbon (not the single mixed pool the coupled run carries) or an
    ``f_perma``-inverse of the SOM-weighted mean protection; computing ``phi``
    from each cell's own climatology (from ``monthly_t_k``) is the clean
    single-value alternative that avoids the mean-of-centroids skew.

    Parameters
    ----------
    table : ArchetypeTable
        The archetype table (``mat_k`` / ``t_seasonal_amp_k`` climate) from
        :func:`build_archetypes`; its rows align with the archetype index space
        of ``cell_archetype_id``.
    cell_archetype_id : array (ncell, n_pft) int
        Archetype index per (cell, PFT); ``-1`` where absent (see
        :func:`build_archetypes`).
    cell_archetype_weight : array (ncell, n_pft) float
        Cover weight per (cell, PFT); 0 where ``cell_archetype_id`` is ``-1``.
    config : CarbonConfig, optional
        The ``CarbonConfig`` the archetypes equilibrated at (only
        ``som_freeze_width_K`` is read).  ``None`` -> the production default.

    Returns
    -------
    jnp.ndarray (ncell,)
        Cover-weighted per-cell ``phi`` in ``[0, 1]``.
    """
    # Deferred (function-scope) import: keep Stage-A (`build_archetypes`)
    # callers numpy-only, mirroring `map_to_grid`.
    import jax.numpy as jnp

    from legoesm.land.carbon.carbon_cycle import annual_frozen_fraction
    from legoesm.land.carbon.config import CarbonConfig

    cid = np.asarray(cell_archetype_id)
    ncell = cid.shape[0]
    mat = np.asarray(table.mat_k, float)
    # Degenerate all-bare / empty table: no archetype to gather -> phi = 0
    # everywhere (guard before the gather, which would index an empty array).
    if mat.size == 0:
        return jnp.zeros((ncell,))
    amp = np.asarray(table.t_seasonal_amp_k, float)
    cfg = config if config is not None else CarbonConfig(scheme="differland")
    # Climate-only phi at the equilibration width -- byte-identical to the
    # per-archetype phi the spin-up applied (annual_frozen_fraction reads only
    # som_freeze_width_K).
    phi_arch = np.asarray(
        annual_frozen_fraction(jnp.asarray(mat), jnp.asarray(amp), cfg),
        float)                                            # (n_arch,)

    cw = np.asarray(cell_archetype_weight, float)
    safe = np.where(cid >= 0, cid, 0)                     # gather index (masked below)
    w = (cid >= 0).astype(float) * cw                     # (ncell, n_pft)
    gathered = phi_arch[safe]                             # (ncell, n_pft)
    wsum = w.sum(axis=1)                                  # (ncell,)
    # Intensive: normalised cover-weighted MEAN (phi is a fraction, not a stock).
    # 1e-30 is a divide-by-zero safety floor only (the where already zeros the
    # unvegetated cells); no vegetated cover -> phi = 0.
    phi_cell = np.where(
        wsum > 0.0, (gathered * w).sum(axis=1) / np.maximum(wsum, 1e-30), 0.0)
    return jnp.asarray(phi_cell)


class PointCarbonMatch(NamedTuple):
    """Which finidat cell a single-column run was seeded from."""

    index: int                 # flat cell index into the finidat
    lat_deg: float             # the matched cell's latitude
    lon_deg: float             # the matched cell's longitude (as stored)
    distance_deg: float        # great-circle separation from the request [deg]
    is_land: bool              # whether the matched cell is flagged land
    # The soil column the pools were spun up on, so a caller can check its own
    # grid against it (None on a finidat that does not record them).  The pools
    # carry no vertical dimension, so a difference does not break the load -- it
    # means the equilibrium was reached under a different soil column.
    n_layers: int | None
    soil_depth_m: float | None
    # The cell's dominant cover and its cover fraction.  A finidat cell is a PFT
    # MIXTURE; a single-point run picks one veg_type.  Surfaced so that mismatch
    # is visible instead of hidden behind the word "equilibrium".
    dominant_pft: str | None
    dominant_pft_weight: float | None


def load_finidat_carbon_ic_at_point(path, lat_deg, lon_deg, *,
                                    max_distance_deg=_POINT_MATCH_MAX_DEG,
                                    require_land=True):
    """Initialise a SINGLE-COLUMN run from the nearest cell of a global finidat.

    ``load_finidat_carbon_ic`` is deliberately strict: a run's column count and
    per-column coordinates must match the finidat exactly, because the pools are
    per-area stocks pinned to the finidat's own cells.  A single-point LMIP run
    (``scripts/run/run_lmip.py``) has ONE column at an arbitrary ``(lat, lon)``,
    so it can never satisfy that -- but it can still be INITIALISED from the
    nearest cell, because the pools are per-area densities [gC/m2]: no area
    weighting or regridding is involved, we simply read the column that the
    spin-up equilibrated at the nearest place on Earth.

    This mirrors :func:`legoesm.land.boundary_data.point.surface_params_at_point`,
    which already picks the nearest surfdata gridcell for the same driver -- so a
    single-point run takes its cover, soil and carbon from the same neighbourhood.

    **This is a nearest-cell initialisation, NOT an equilibrium guarantee for the
    site.**  The pools equilibrated under the finidat CELL's grid-mean climate,
    its cover-weighted PFT MIXTURE and the build's soil class; a point run
    chooses a single ``veg_type`` and ``soil_texture`` at coordinates that need
    not resemble any of those.  The returned :class:`PointCarbonMatch` therefore
    reports the cell's soil column and dominant cover so the caller can see the
    mismatch, and the pools should be expected to drift toward the point
    configuration's own equilibrium -- much less far than from a cold start, but
    not to zero.

    Fail-loud, never a silent seed:

    * a file that is not a carbon finidat (any of the eight pools missing) raises
      -- unlike the auto-detecting strict loader, the caller asked for this file
      by name;
    * the nearest candidate must lie within ``max_distance_deg`` of the request,
      else raises (an ocean-only or regional finidat must not silently supply a
      cell from the far side of the planet);
    * with ``require_land`` (the default) only cells flagged land are candidates,
      so a coastal request cannot be seeded from an all-zero ocean column.

    Distance is the great-circle angle (haversine on the unit sphere), NOT a
    lat/lon Euclidean distance -- the latter mis-ranks candidates at high
    latitude, which is exactly where the carbon gradients are sharpest.

    Parameters
    ----------
    path : str | pathlib.Path
        The finidat ``global_carbon_ic.npz``.
    lat_deg, lon_deg : float
        The single column's coordinates.  Longitude is matched modulo 360, so a
        0..360 finidat and a -180..180 request describe the same place.
    max_distance_deg : float
        Reject a match farther than this great-circle angle [deg].
    require_land : bool
        Restrict candidates to cells whose ``land_mask`` is true (when the
        finidat carries one).

    Returns
    -------
    (CarbonState (1,), jnp.ndarray (1,) | None, PointCarbonMatch)
        The seeded pools [gC/m2] shaped for a one-column run, the matching
        permafrost ``phi`` (``None`` for a legacy finidat without it), and the
        provenance of the match.
    """
    import jax.numpy as jnp

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "lat" not in keys or "lon" not in keys:
            raise ValueError(
                f"{path!r} has no 'lat'/'lon' fields; a carbon finidat must carry "
                "per-cell coordinates to be matched to a point.")
        lat_fin = np.asarray(z["lat"], float).reshape(-1)
        lon_fin = np.asarray(z["lon"], float).reshape(-1)
        land = (np.asarray(z["land_mask"]).reshape(-1).astype(bool)
                if "land_mask" in keys else None)
        n_lay = int(z["n_layers"]) if "n_layers" in keys else None
        depth = float(z["soil_depth"]) if "soil_depth" in keys else None
        dom_pft = (np.asarray(z["dominant_pft"]).reshape(-1)
                   if "dominant_pft" in keys else None)
        pft_names = ([str(s) for s in np.asarray(z["pft_names"]).reshape(-1)]
                     if "pft_names" in keys else None)
        pft_w = (np.asarray(z["pft_weights"], float)
                 if "pft_weights" in keys else None)

    # Load the pools FIRST (the strict loader owns pool presence / per-cell shape
    # / phi validation), so the selection arrays below can be checked against the
    # authoritative cell count before anything is indexed with them.  Selecting
    # from a misaligned 'lat' would otherwise index a different pool column.
    carbon, phi = load_finidat_carbon_ic(path)
    if carbon is None:
        from legoesm.land.carbon.config import CarbonState
        raise ValueError(
            f"{path!r} is not a carbon finidat: it is missing one or more of the "
            f"eight pools {CarbonState._fields}.")
    ncell = int(np.asarray(getattr(carbon, carbon._fields[0])).shape[0])
    for name, arr in (("lat", lat_fin), ("lon", lon_fin), ("land_mask", land),
                      ("dominant_pft", dom_pft)):
        if arr is not None and arr.shape != (ncell,):
            raise ValueError(
                f"finidat {name!r} has shape {arr.shape} but the pools have "
                f"({ncell},); the selection arrays and the pools describe "
                "different cells, so a nearest-cell pick would be meaningless.")
    if pft_w is not None and pft_w.shape[0] != ncell:
        raise ValueError(
            f"finidat 'pft_weights' has shape {pft_w.shape} but the pools have "
            f"({ncell},) cells.")

    # Great-circle angle on the unit sphere (haversine): correct at the poles and
    # across the 0/360 seam, where a lat/lon Euclidean metric is not.
    phi1 = np.deg2rad(lat_fin)
    phi2 = np.deg2rad(float(lat_deg))
    dphi = phi1 - phi2
    dlam = np.deg2rad(((lon_fin - float(lon_deg) + 180.0) % 360.0) - 180.0)
    hav = (np.sin(dphi / 2.0) ** 2
           + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2)
    dist_deg = np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0))))

    candidates = dist_deg
    if require_land:
        if land is None:
            raise ValueError(
                f"{path!r} has no 'land_mask'; pass require_land=False to seed "
                "from the nearest cell regardless of surface type.")
        if not bool(land.any()):
            raise ValueError(f"{path!r} has no land cells to seed from.")
        candidates = np.where(land, dist_deg, np.inf)

    # Ties (an exact midpoint, or duplicated pole rows) go to the LOWEST flat
    # index, i.e. the finidat's own storage order.  Deterministic for a given
    # file, but it is file order and not a physical criterion -- documented
    # rather than silently relied upon.
    idx = int(np.argmin(candidates))
    d = float(candidates[idx])
    if not np.isfinite(d) or d > float(max_distance_deg):
        raise ValueError(
            f"nearest {'land ' if require_land else ''}cell of {path!r} is "
            f"{d:.3g} deg from ({lat_deg}, {lon_deg}), beyond "
            f"max_distance_deg={max_distance_deg}; refusing to seed a column from "
            "a distant climate.")

    carbon_pt = type(carbon)(
        **{f: jnp.asarray(getattr(carbon, f)[idx]).reshape(1)
           for f in carbon._fields})
    phi_pt = None if phi is None else jnp.asarray(phi[idx]).reshape(1)
    # The cell's dominant cover, so a caller can see WHAT vegetation the seeded
    # pools actually equilibrated under (a grid cell is a PFT mixture; a
    # single-point run picks one veg_type, and the two need not agree).
    dom_name = None
    dom_weight = None
    if dom_pft is not None:
        d_id = int(dom_pft[idx])
        if pft_names is not None and 0 <= d_id < len(pft_names):
            dom_name = pft_names[d_id]
        if pft_w is not None and 0 <= d_id < pft_w.shape[1]:
            dom_weight = float(pft_w[idx, d_id])
    match = PointCarbonMatch(
        index=idx, lat_deg=float(lat_fin[idx]), lon_deg=float(lon_fin[idx]),
        distance_deg=d, is_land=True if land is None else bool(land[idx]),
        n_layers=n_lay, soil_depth_m=depth,
        dominant_pft=dom_name, dominant_pft_weight=dom_weight)
    return carbon_pt, phi_pt, match


def load_finidat_carbon_ic(path, *, expect_ncol=None, target_lat_deg=None,
                           target_lon_deg=None, coord_atol_deg=_COORD_MATCH_ATOL_DEG):
    """Load a per-cell 8-pool :class:`CarbonState` (+ permafrost ``phi``) from a
    global-carbon finidat and STRICTLY grid-match it to a run's land columns.

    The inverse of :func:`map_to_grid` (the eight pools) +
    :func:`map_to_grid_frozen_fraction` (the ``soil_frozen_fraction`` ``phi``): it
    reads the per-cell finidat ``global_carbon_ic.npz`` (written by
    ``scripts/data/build_global_carbon_ic.py``) so a coupled run can INGEST the
    seeded equilibrium instead of cold-starting carbon, AND thread the SAME
    per-cell ``phi`` that scaled the seed into the coupler's permafrost/anaerobic
    SOM protection (:func:`make_coupler(land_soil_frozen_fraction=...)
    <legoesm.coupler.coupler.make_coupler>`) so the deep permafrost SOC PERSISTS.

    Detection (static feature gate).  A ``.npz`` is a "carbon finidat" iff it
    carries ALL eight :class:`~legoesm.land.carbon.config.CarbonState` pool fields.
    A file missing any pool (a legacy land restart, an unrelated ``.npz``) is NOT
    a carbon IC -> this returns ``(None, None)`` so the caller keeps its cold-start
    EXACTLY (a static Python branch, never a partial / silent seed).

    Grid match (the SIMPLER correct option: STRICT, fail-loud).  The finidat's
    per-cell count must equal ``expect_ncol`` (the run's land columns) and -- when
    ``target_lat_deg`` / ``target_lon_deg`` are supplied -- the finidat's per-cell
    ``lat`` / ``lon`` must match the run grid's columns element-for-element
    (latitude to ``coord_atol_deg``; longitude modulo 360, so a 0..360 vs
    -180..180 store is not a false mismatch).  ANY mismatch RAISES with the exact
    shapes / coords: the seeded pools are per-AREA stocks pinned to the finidat's
    own cells, so silently reshaping or scatter-broadcasting them onto a different
    (or differently-ordered) grid would corrupt the IC.  Cross-grid conservative
    regridding (lat-lon -> a different lat-lon / Gaussian) is a documented
    follow-up; a cubed-sphere / Voronoi run cannot match a lat-lon finidat and must
    build the carbon IC on its own grid.

    Parameters
    ----------
    path : str | pathlib.Path
        The finidat ``global_carbon_ic.npz``.
    expect_ncol : int, optional
        The run's land-column count (``math.prod(grid_shape)``); enforced exactly.
    target_lat_deg, target_lon_deg : array (ncol,), optional
        The run grid's per-column latitude / longitude [degrees] (row-major -- the
        SAME flatten the finidat used).  When given, the finidat coordinates must
        match column-for-column (else :class:`ValueError`).
    coord_atol_deg : float
        Absolute tolerance [deg] for the latitude / longitude match.

    Returns
    -------
    (CarbonState (ncol,) | None, jnp.ndarray (ncol,) | None)
        The seeded per-cell carbon pools [gC/m2] and the per-cell permafrost
        ``phi`` (``soil_frozen_fraction`` in ``[0, 1]``) if the finidat carries it,
        else ``phi`` is ``None`` (a legacy finidat with no ``phi`` -> the coupler
        runs unprotected, byte-identical).  ``(None, None)`` when ``path`` is not a
        carbon finidat.
    """
    # Deferred (function-scope) import: keep Stage-A callers numpy-only, mirroring
    # map_to_grid / map_to_grid_frozen_fraction.
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState

    pool_fields = CarbonState._fields
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        # Feature gate: not a carbon finidat unless EVERY pool is present.
        if not all(f in keys for f in pool_fields):
            return None, None
        pools = {f: np.asarray(z[f], dtype=np.float64).reshape(-1)
                 for f in pool_fields}
        ncell = pools[pool_fields[0]].shape[0]
        # Every pool must share the per-cell length (a corrupt / mismatched file).
        for f in pool_fields:
            if pools[f].shape != (ncell,):
                raise ValueError(
                    f"finidat carbon pool {f!r} has shape {pools[f].shape}; "
                    f"expected ({ncell},) to match {pool_fields[0]!r}.")
        lat_fin = (np.asarray(z["lat"], float).reshape(-1)
                   if "lat" in keys else None)
        lon_fin = (np.asarray(z["lon"], float).reshape(-1)
                   if "lon" in keys else None)
        phi = (np.asarray(z["soil_frozen_fraction"], float).reshape(-1)
               if "soil_frozen_fraction" in keys else None)

    # --- STRICT grid match (fail-loud; never a silent reshape / regrid) --------
    if expect_ncol is not None and ncell != int(expect_ncol):
        raise ValueError(
            f"finidat carbon IC has {ncell} cells but the run's land grid has "
            f"{int(expect_ncol)} columns; the seeded per-area pools are grid-bound "
            "-- build the IC on the run grid (match --resolution).  Cross-grid "
            "regridding is not yet supported for the carbon IC.")
    if target_lat_deg is not None:
        tlat = np.asarray(target_lat_deg, float).reshape(-1)
        if lat_fin is None:
            raise ValueError(
                "finidat carbon IC has no 'lat' field to verify the grid match "
                "against target_lat_deg; refusing to seed onto an unverified grid.")
        if tlat.shape != lat_fin.shape:
            raise ValueError(
                f"finidat 'lat' has shape {lat_fin.shape} but target_lat_deg has "
                f"{tlat.shape}; grid mismatch.")
        dlat = float(np.max(np.abs(lat_fin - tlat))) if ncell else 0.0
        if dlat > coord_atol_deg:
            raise ValueError(
                f"finidat latitude does not match the run grid (max |dlat|="
                f"{dlat:.4g} deg > {coord_atol_deg} deg); the carbon IC is on a "
                "different (or differently-ordered) grid.")
    if target_lon_deg is not None:
        tlon = np.asarray(target_lon_deg, float).reshape(-1)
        if lon_fin is None:
            raise ValueError(
                "finidat carbon IC has no 'lon' field to verify the grid match "
                "against target_lon_deg; refusing to seed onto an unverified grid.")
        if tlon.shape != lon_fin.shape:
            raise ValueError(
                f"finidat 'lon' has shape {lon_fin.shape} but target_lon_deg has "
                f"{tlon.shape}; grid mismatch.")
        # Longitude modulo 360 (a 0..360 vs -180..180 store is the SAME grid); the
        # signed angular difference wraps at the 0/360 seam.
        dlon = (lon_fin - tlon + 180.0) % 360.0 - 180.0
        dlon_max = float(np.max(np.abs(dlon))) if ncell else 0.0
        if dlon_max > coord_atol_deg:
            raise ValueError(
                f"finidat longitude does not match the run grid (max |dlon|="
                f"{dlon_max:.4g} deg > {coord_atol_deg} deg); the carbon IC is on a "
                "different (or differently-ordered) grid.")

    # --- phi validation (a malformed field fails loud, never silently seeds) ---
    if phi is not None:
        if phi.shape != (ncell,):
            raise ValueError(
                f"finidat 'soil_frozen_fraction' (phi) has shape {phi.shape}; "
                f"expected ({ncell},).")
        if not bool(np.all(np.isfinite(phi))):
            raise ValueError("finidat 'soil_frozen_fraction' (phi) must be finite.")
        lo = float(np.min(phi)); hi = float(np.max(phi))
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                "finidat 'soil_frozen_fraction' (phi) must be in [0, 1] (an annual "
                f"frozen fraction); got [{lo}, {hi}].")

    carbon = CarbonState(**{f: jnp.asarray(pools[f]) for f in pool_fields})
    phi_out = None if phi is None else jnp.asarray(phi)
    return carbon, phi_out
