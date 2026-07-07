"""Global carbon initial-condition map: per-PFT climate archetype builder.

Two offline stages, both run ONCE (never inside a JAX-traced model step):

* Stage A -- :func:`build_archetypes` (see
  ``docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md``): reduce
  the global ``(cell, pft)`` cover-weight x climate-feature space to a small set
  of per-PFT climate archetypes via k-means.  Pure numpy, no JAX.
* Stage B -- :func:`equilibrate_archetypes`: spin EACH archetype to a verified
  soil-carbon equilibrium with the shared semi-analytic driver
  (:func:`legoesm.land.carbon.spinup.run_semi_analytic_spinup`), batching
  archetypes that share ``(is_woody, soil_class)`` into ONE vectorised
  ``step_multilayer_land`` column-block (few JAX compiles, not one per
  archetype).  The per-group construction (:func:`iter_archetype_batches`) and
  the coupled step (:func:`make_archetype_step_fn`) are factored into shared
  helpers so the drift/realism validator (``scripts/validate/
  global_carbon_ic_map.py``) re-integrates the IDENTICAL step without copying
  the config / physiology / forcing build.  The heavy JAX / land-model imports
  are deferred into those functions so Stage-A callers stay numpy-only.
"""

from __future__ import annotations

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

# Woody-PFT classifier: trees and shrubs are woody; grasses and crops are
# herbaceous (mirrors ``_is_woody`` in the land_carbon_equilibrium harness).
_HERBACEOUS_TAGS = ("grass", "crop")


def _is_woody(pft_name: str) -> bool:
    """True for woody PFTs (trees/shrubs); False for grasses/crops."""
    return not any(tag in pft_name for tag in _HERBACEOUS_TAGS)


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
        has weight below ``w_min`` in cell ``c``, else the archetype index
        into the returned table. ``cell_archetype_weight`` is
        ``(ncell, npft)`` float, the cover weight (0 where id is -1).
    """
    pft_weights = np.asarray(pft_weights)
    ncell, npft = pft_weights.shape
    feat = np.stack(
        [np.asarray(getattr(features, f)) for f in _FEATURE_FIELDS], axis=1
    )  # (ncell, 5)
    # Standardise globally so per-PFT clusters share a metric.
    mu = feat.mean(0)
    sd = feat.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)  # coeff-ok: std floor
    feat_std = (feat - mu) / sd
    soil_class = np.asarray(soil_class, dtype=object)
    cell_id = np.full((ncell, npft), -1, int)
    cell_w = np.where((pft_weights >= w_min) & np.asarray(land_mask)[:, None], pft_weights, 0.0)
    at = {f: [] for f in ("pft_id", *_FEATURE_FIELDS, "soil_class")}
    next_arch = 0
    for p in range(npft):
        occ = np.where((pft_weights[:, p] >= w_min) & np.asarray(land_mask))[0]
        if occ.size == 0:
            continue
        labels, _ = _kmeans(feat_std[occ], k_per_pft, seed + p)
        for j in np.unique(labels):
            members = occ[labels == j]
            cell_id[members, p] = next_arch
            at["pft_id"].append(p)
            for fi, fname in enumerate(_FEATURE_FIELDS):
                at[fname].append(float(feat[members, fi].mean()))
            # modal soil texture in the cluster
            vals, cnts = np.unique(soil_class[members], return_counts=True)
            at["soil_class"].append(str(vals[cnts.argmax()]))
            next_arch += 1
    table = ArchetypeTable(
        pft_id=np.asarray(at["pft_id"], int),
        mat_k=np.asarray(at["mat_k"]), map_yr=np.asarray(at["map_yr"]),
        t_seasonal_amp_k=np.asarray(at["t_seasonal_amp_k"]),
        aridity=np.asarray(at["aridity"]), sw_mean_w=np.asarray(at["sw_mean_w"]),
        soil_class=np.asarray(at["soil_class"], dtype=object))
    return table, cell_id, cell_w


class ArchetypeBatch(NamedTuple):
    """One ``(is_woody, soil_class)`` GROUP's shared coupled-step construction.

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


def iter_archetype_batches(table: ArchetypeTable, *, n_layers, soil_depth, dt):
    """Build the per-``(is_woody, soil_class)`` GROUP construction shared by the
    archetype equilibration (:func:`equilibrate_archetypes`) and the drift
    validator (``scripts/validate/global_carbon_ic_map.py``).

    Both must run the IDENTICAL coupled land+carbon step from a given IC, so
    the ``MultiLayerLandConfig``, the per-column ``LandSurfaceParams`` (CLM5 PFT
    physiology), and the ``vmap``-batched climatological ``forcing_fn`` are
    built ONCE here rather than copied into each caller.  Archetypes are grouped
    by ``(is_woody, soil_class)`` -- the only two config knobs that must be
    scalar per group (the ``carbon.woody`` flag and the soil hydraulics) -- so
    each group is one vectorised ``step_multilayer_land`` column-block; the
    per-archetype PFT physiology and climate ride along as per-column arrays.

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
    )
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.stomata import StomataConfig
    from legoesm.land.climate_forcing import make_climatological_forcing

    pft_id = np.asarray(table.pft_id, int)
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

    # Group archetypes by (woody, soil_class) -> a shared scalar config,
    # preserving table order within each group.
    groups: dict = {}
    for a in range(n_arch):
        key = (_is_woody(CLM5_PFT_NAMES[pft_id[a]]), str(soil_class[a]))
        groups.setdefault(key, []).append(a)

    batches: list[ArchetypeBatch] = []
    for (group_woody, soil), members in groups.items():
        g_idx = np.asarray(members, int)
        config = MultiLayerLandConfig(
            bulk_scheme="most",
            snow_albedo_feedback=True,
            soil_grid=SoilGridConfig(
                n_layers=n_layers, total_depth=soil_depth,
                growth_factor=_SOIL_GROWTH_FACTOR),
            hydraulics=SoilHydraulicsConfig(**SOIL_TEXTURE_VG[soil]),
            thermal=SoilThermalConfig(),
            richards=RichardsConfig(),
            carbon=CarbonConfig(
                scheme="differland", woody=group_woody,
                C_lab_init=_C_LAB_SEED, C_fol_init=_C_FOL_SEED,
                C_root_init=_C_ROOT_SEED, C_wood_init=_C_WOOD_SEED,
                C_lit_init=_C_LIT_SEED, C_som_init=_C_SOM_SEED),
            stomata=StomataConfig(enabled=True, stomata_model="ball_berry"),
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
        # Archetypes carry no hemisphere; the land-state initial temperature is
        # the group mean-annual temperature (matches the NH-phased forcing).
        batches.append(ArchetypeBatch(
            config=config, land_params=land_params, forcing_fn=forcing_fn,
            g_idx=g_idx, steps_per_year=steps_per_year, t_init=mat_g))
    return batches


def make_archetype_step_fn(config, land_params, *, dt):
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
    from legoesm.land.multilayer_land import step_multilayer_land

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
        new_state, _response, carbon_new = step_multilayer_land(
            state, forcing, config, _U_MIN, dt,
            lat=lat_g, carbon_state=carbon, doy=doy, land_params=land_params)
        # Reconstruct the flux breakdown consistently with the model: pass
        # land_params so the GPP override uses the SAME per-archetype
        # Vc_max25/g1/LCMA the coupled step used.
        diag = reconstruct_carbon_diagnostics(
            new_state, forcing, carbon, config, root_frac_g,
            theta_wp_g, theta_fc_g, beta_min, lat_g, doy, dt,
            spatial=True, land_params=land_params)
        return new_state, carbon_new, diag

    return step_fn


def equilibrate_archetypes(
    table: ArchetypeTable,
    *,
    n_spinup: int = _N_SPINUP_DEFAULT,
    n_verify: int = _N_VERIFY_DEFAULT,
    dt: float = _DT_DEFAULT,
    n_layers: int = _N_LAYERS_DEFAULT,
    soil_depth: float = _SOIL_DEPTH_DEFAULT,
):
    """Spin every climate archetype to a verified soil-carbon equilibrium.

    Runs the shared semi-analytic spin-up
    (:func:`legoesm.land.carbon.spinup.run_semi_analytic_spinup`) once per
    ``(is_woody, soil_class)`` GROUP, batching that group's archetypes into a
    single vectorised ``step_multilayer_land`` column-block.  There are only a
    few such groups (<= ~2 x n_soil_textures), so the whole global archetype
    table costs a handful of JAX compiles rather than one per archetype.

    Grouping rationale: ``MultiLayerLandConfig`` sub-configs (soil hydraulics,
    the ``carbon.woody`` flag) are SCALARS, so archetypes that share both the
    soil texture and the woody/herbaceous split can share one config.  Their
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
    # The per-group construction (config / land_params / forcing) and the
    # coupled step now live in the SHARED helpers below (also used by the drift
    # validator), so this function only orchestrates + scatters.
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState
    from legoesm.land.carbon.carbon_cycle import init_carbon_state
    from legoesm.land.carbon.spinup import run_semi_analytic_spinup
    from legoesm.land.multilayer_land import init_multilayer_land_state

    n_arch = np.asarray(table.pft_id, int).shape[0]
    pool_fields = CarbonState._fields  # ("C_lab", ..., "C_som")
    batches = iter_archetype_batches(
        table, n_layers=n_layers, soil_depth=soil_depth, dt=dt)

    # Archetype-ordered output accumulators (scattered per group via g_idx).
    pools_out = {p: np.zeros(n_arch) for p in pool_fields}
    qc = {k: np.zeros(n_arch)
          for k in ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr")}

    for batch in batches:
        g_idx = batch.g_idx
        ncol_g = g_idx.shape[0]
        step_fn = make_archetype_step_fn(batch.config, batch.land_params, dt=dt)
        state0 = init_multilayer_land_state(
            ncol_g, batch.config, T_init=batch.t_init)
        carbon0 = init_carbon_state((ncol_g,), batch.config.carbon)
        _final_state, final_carbon, annual = run_semi_analytic_spinup(
            step_fn, state0, carbon0, batch.forcing_fn,
            n_spinup=n_spinup, n_verify=n_verify,
            steps_per_year=batch.steps_per_year, dt=dt,
            cwd_humification_eff=batch.config.carbon.cwd_humification_eff)

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
        qc["som_kgC"][g_idx] = np.asarray(final_carbon.C_som) / _G_PER_KG
        qc["biomass_kgC"][g_idx] = biomass / _G_PER_KG
        qc["drift_frac_per_yr"][g_idx] = drift

    equilibrium = CarbonState(
        **{p: jnp.asarray(pools_out[p]) for p in pool_fields})
    return equilibrium, qc


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
