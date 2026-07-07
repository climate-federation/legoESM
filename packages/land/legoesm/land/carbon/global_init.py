"""Global carbon initial-condition map: per-PFT climate archetype builder.

Stage A of the archetype semi-analytic spin-up (see
``docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md``): reduce the global
``(cell, pft)`` cover-weight x climate-feature space to a small set of
per-PFT climate archetypes via k-means, so that a cheap per-archetype
spin-up (a later task) can stand in for a per-cell spin-up. Pure numpy:
this runs once, offline, over a full global grid, not inside any
JAX-traced model step, so there is no autodiff/JIT requirement here.
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
