"""Cluster worst-performing columns by environment → one LES per representative.

LES cost dominates the loop (``docs/COMPARE_REANALYSIS.md`` §7): a single deep
LES is more expensive than the whole AMIP run, so running one per worst column is
wasteful when many flagged columns share an environment.  This groups the
worst-column manifest by its environment tags (SST, CAPE, bulk shear — the §6
"natural bridge") and picks **K representative columns** to spin off LES for; the
coefficient diagnosed for a representative is then applied to every column in its
cluster (compose with the feedback assembly / parameter field).

Selection is deterministic **farthest-first traversal** (Gonzalez's k-center
2-approximation), NOT k-means, for two reasons: (1) representatives are *real
columns* (we must force an LES with an actual column's environment, not a
synthetic centroid), and (2) it is deterministic + reproducible (no random
init).  The traversal is anchored at the **worst-scoring** column (so the most
biased environment is always sampled) and then repeatedly adds the column whose
environment is farthest from all chosen representatives, maximising coverage of
the environment diversity.

Environment features have very different magnitudes (SST ~300 K, CAPE ~10³ J/kg,
shear ~10 m/s), so they are z-score normalised (per-feature mean/std) before
distances are taken — otherwise CAPE alone would dominate.  A near-constant
feature (std below a tiny threshold) is *suppressed* (divided by 1.0, not its ~0
std), so FP-noise-level variation cannot dominate.  Pass explicit ``env_scales``
to override (same role as ``environment_kernel_field``'s ``length_scales``).

This is an offline, host-side *selection* step (like the manifest ranking) — not
in the autodiff path — so it uses NumPy, consistent with
:mod:`legoesm.training.column_manifest`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

# A feature whose spread is below this (numerically negligible / FP noise) is
# treated as NON-discriminating: it is divided by 1.0 (NOT by its ~0 std), so its
# centred ~0 values contribute ~0 distance — rather than amplifying FP-noise-level
# variation to dominate the clustering.  (This is deliberate suppression, not a
# floor-to-eps, which would do the opposite.)
_MIN_FEATURE_STD = 1.0e-12
# Two environments closer than this in NORMALISED space do not warrant separate
# LES; farthest-first stops once the farthest remaining column is this close to a
# chosen representative (avoids a duplicate / near-duplicate representative).
_DISTINCT_ENV_TOL = 1.0e-9


def _validate_env_scales(env_scales: np.ndarray, n_features: int) -> None:
    """Reject non-positive / non-finite / wrong-length explicit ``env_scales``."""
    if env_scales.shape != (n_features,):
        raise ValueError(
            f"env_scales must have shape ({n_features},), got {env_scales.shape}."
        )
    if not np.all(np.isfinite(env_scales)):
        raise ValueError(f"env_scales must be finite, got {env_scales}.")
    if np.any(env_scales <= 0.0):
        raise ValueError(f"env_scales must be strictly positive, got {env_scales}.")


class ColumnClusters(NamedTuple):
    """Environment clustering of a worst-column manifest.

    ``representative_indices`` index into the input ``records`` — run one LES per
    representative.  ``labels[i]`` is the position in ``representative_indices`` of
    the representative that record ``i`` is assigned to (its nearest in normalised
    environment space), so a diagnosed coefficient maps representative → cluster.
    """

    representative_indices: tuple[int, ...]   # which records get an LES
    labels: tuple[int, ...]                   # per record → index into the reps tuple
    n_clusters: int


def _environment_matrix(records: Sequence[Any]) -> np.ndarray:
    """``(n, 3)`` [SST, CAPE, bulk-shear] feature matrix from the manifest records."""
    return np.array(
        [
            [
                float(r.environment.sst_K),
                float(r.environment.cape_J_kg),
                float(r.environment.bulk_shear_m_s),
            ]
            for r in records
        ],
        dtype=np.float64,
    )


def _normalize(features: np.ndarray, env_scales: np.ndarray | None) -> np.ndarray:
    """Per-feature normalisation so the three environment axes are comparable.

    Default: z-score ``(x − mean) / std``; a feature whose std is below
    :data:`_MIN_FEATURE_STD` is SUPPRESSED (denominator 1.0 → its centred ~0
    values contribute ~0 distance), not amplified.  ``env_scales`` (length-3,
    validated positive+finite by the caller) overrides the denominator (the mean
    is still removed), letting a caller impose the same per-feature scales as
    ``environment_kernel_field``.
    """
    centered = features - features.mean(axis=0, keepdims=True)
    if env_scales is None:
        std = features.std(axis=0, keepdims=True)
        # Suppress numerically-non-discriminating features (see _MIN_FEATURE_STD).
        denom = np.where(std < _MIN_FEATURE_STD, 1.0, std)
    else:
        denom = env_scales.reshape(1, -1)
    return centered / denom


def cluster_columns_by_environment(
    records: Sequence[Any],
    n_clusters: int,
    *,
    env_scales: Sequence[float] | None = None,
) -> ColumnClusters:
    """Group worst columns by environment; return ``K`` representative columns + labels.

    ``records`` is the worst-column manifest (each exposing ``combined_score`` and
    ``environment.{sst_K, cape_J_kg, bulk_shear_m_s}``).  ``n_clusters`` is the LES
    budget (number of representatives); it is capped at ``len(records)`` and, if
    the environment has fewer distinct points than requested, fewer clusters are
    returned (``n_clusters`` in the result reflects the actual count) rather than
    emitting duplicate representatives.

    Deterministic farthest-first anchored at the worst-scoring column.  Raises on
    an empty manifest or a non-positive budget.
    """
    n = len(records)
    if n == 0:
        raise ValueError("cluster_columns_by_environment: empty manifest.")
    if int(n_clusters) <= 0:
        raise ValueError(
            f"cluster_columns_by_environment: n_clusters must be > 0, got {n_clusters}."
        )
    k = min(int(n_clusters), n)

    feats = _environment_matrix(records)
    if not np.all(np.isfinite(feats)):
        raise ValueError(
            "cluster_columns_by_environment: non-finite environment tag "
            "(SST/CAPE/shear) in the manifest — a NaN/inf would poison the "
            "normalisation + representative selection."
        )
    scales = None
    if env_scales is not None:
        scales = np.asarray(env_scales, dtype=np.float64)
        _validate_env_scales(scales, feats.shape[1])
    norm = _normalize(feats, scales)               # (n, 3)

    scores = np.array([float(r.combined_score) for r in records])
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "cluster_columns_by_environment: non-finite combined_score — a NaN/inf "
            "would corrupt the worst-scoring anchor."
        )
    first = int(np.argmax(scores))                 # anchor: the worst-scoring column

    reps = [first]
    min_dist = np.linalg.norm(norm - norm[first], axis=1)   # (n,)
    while len(reps) < k:
        cand = int(np.argmax(min_dist))            # farthest from all chosen reps
        if min_dist[cand] <= _DISTINCT_ENV_TOL:
            # Every remaining column is (near-)identical to a representative in
            # normalised env space → stop early rather than emit a duplicate.
            break
        reps.append(cand)
        min_dist = np.minimum(min_dist, np.linalg.norm(norm - norm[cand], axis=1))

    rep_pts = norm[reps]                            # (k', 3)
    dist_to_reps = np.linalg.norm(
        norm[:, None, :] - rep_pts[None, :, :], axis=-1
    )                                              # (n, k')
    labels = np.argmin(dist_to_reps, axis=1)       # nearest representative

    return ColumnClusters(
        representative_indices=tuple(int(i) for i in reps),
        labels=tuple(int(label) for label in labels),
        n_clusters=len(reps),
    )
