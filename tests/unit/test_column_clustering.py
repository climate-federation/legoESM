"""Unit tests for :mod:`legoesm.training.column_clustering`.

Environment clustering of the worst-column manifest into K representative
columns (one LES per representative): well-separated clusters, the worst-scoring
anchor, identity when K≥N, determinism, the duplicate-environment early stop, and
the guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.training.column_clustering import cluster_columns_by_environment
from legoesm.training.column_manifest import ColumnEnvironment, ColumnRecord


def _record(flat, score, sst, cape, shear):
    return ColumnRecord(
        flat_index=flat,
        grid_index=(flat,),
        lat_deg=0.0,
        lon_deg=0.0,
        time_index=0,
        combined_score=score,
        T_rmse_K=0.0,
        qv_rmse_kg_kg=0.0,
        wind_rmse_m_s=0.0,
        precip_err_mm_day=0.0,
        environment=ColumnEnvironment(sst_K=sst, cape_J_kg=cape, bulk_shear_m_s=shear),
    )


def _two_separated_groups():
    # Group A: cold/dry/low-shear; Group B: warm/moist/high-shear — far apart.
    a = [
        _record(0, 1.0, 280.0, 100.0, 2.0),
        _record(1, 0.9, 281.0, 120.0, 2.5),
        _record(2, 0.8, 279.0, 90.0, 1.8),
    ]
    b = [
        _record(3, 2.0, 302.0, 3000.0, 25.0),
        _record(4, 1.5, 301.0, 3200.0, 24.0),
        _record(5, 1.4, 303.0, 2900.0, 26.0),
    ]
    return a + b


def test_two_well_separated_clusters():
    records = _two_separated_groups()
    out = cluster_columns_by_environment(records, n_clusters=2)
    assert out.n_clusters == 2
    # Anchor (rep 0) is the worst-scoring column: flat_index 3, score 2.0.
    assert records[out.representative_indices[0]].flat_index == 3
    # The two reps come from DIFFERENT groups (one index <3, one >=3).
    rep_idx = sorted(out.representative_indices)
    assert (rep_idx[0] < 3) and (rep_idx[1] >= 3)
    # All group-A records share one label; all group-B records share the other.
    labels = np.asarray(out.labels)
    assert len(set(labels[:3].tolist())) == 1
    assert len(set(labels[3:].tolist())) == 1
    assert labels[0] != labels[3]


def test_anchor_is_worst_scoring_column():
    records = [
        _record(0, 0.5, 290.0, 500.0, 5.0),
        _record(1, 9.0, 295.0, 800.0, 8.0),   # worst score → anchor
        _record(2, 0.7, 285.0, 300.0, 3.0),
    ]
    out = cluster_columns_by_environment(records, n_clusters=1)
    assert out.n_clusters == 1
    assert out.representative_indices == (1,)
    # K=1 → every column assigned to the single representative.
    assert out.labels == (0, 0, 0)


def test_k_geq_n_is_identity():
    records = _two_separated_groups()
    out = cluster_columns_by_environment(records, n_clusters=10)  # capped at n=6
    assert out.n_clusters == 6
    assert sorted(out.representative_indices) == [0, 1, 2, 3, 4, 5]
    # Each column is its own representative.
    for i, label in enumerate(out.labels):
        assert out.representative_indices[label] == i


def test_duplicate_environments_stop_early():
    """K requested > distinct environments ⇒ fewer reps, no duplicate rep."""
    records = [
        _record(0, 3.0, 300.0, 1000.0, 10.0),
        _record(1, 2.0, 300.0, 1000.0, 10.0),  # identical env to 0
        _record(2, 1.0, 300.0, 1000.0, 10.0),  # identical env to 0
    ]
    out = cluster_columns_by_environment(records, n_clusters=3)
    assert out.n_clusters == 1  # all envs coincide → a single representative
    assert out.representative_indices == (0,)  # the worst-scoring anchor
    assert len(set(out.representative_indices)) == len(out.representative_indices)
    assert out.labels == (0, 0, 0)  # every column → the single representative


def test_single_record():
    out = cluster_columns_by_environment(
        [_record(0, 1.0, 300.0, 1000.0, 10.0)], n_clusters=3)
    assert out.representative_indices == (0,)
    assert out.labels == (0,)
    assert out.n_clusters == 1


def test_deterministic():
    records = _two_separated_groups()
    a = cluster_columns_by_environment(records, n_clusters=3)
    b = cluster_columns_by_environment(records, n_clusters=3)
    assert a == b


def test_farthest_first_tie_break_is_lowest_index_deterministic():
    """When two candidates are EQUIDISTANT from the chosen representatives, the
    farthest-first step must pick the LOWEST-index one, reproducibly.

    ``test_deterministic`` uses WELL-SEPARATED groups (no ties), so it cannot catch
    a non-deterministic tie-break.  But the clustering re-derives each round from the
    checkpointed manifest, so a tie-break that varied (random, or argmax-from-the-end)
    would make the SAME manifest spin off a DIFFERENT LES representative across re-runs
    — diagnosing a different coefficient and breaking campaign reproducibility (the
    clustering analog of the iter-212 ranking tie-break lock).  The k-center uses
    ``np.argmax(min_dist)``, which resolves ties to the lowest index.  Here two
    candidates sit at IDENTICAL distance (±10 K SST) on either side of the anchor, so
    the second representative must be the lower-index one.
    """
    # idx0: SST 310 (10 K from anchor); idx1: anchor (worst score); idx2: SST 290
    # (also 10 K) — idx0 and idx2 are exactly equidistant from the anchor.
    records = [
        _record(10, 0.5, 310.0, 1000.0, 10.0),
        _record(11, 9.0, 300.0, 1000.0, 10.0),   # anchor: highest combined_score
        _record(12, 0.5, 290.0, 1000.0, 10.0),
    ]
    out = cluster_columns_by_environment(records, n_clusters=2, env_scales=(1.0, 1.0, 1.0))
    # Anchor (idx 1) then the LOWER-index tied candidate (idx 0) — NOT idx 2.
    assert out.representative_indices == (1, 0)
    # Reproducible across calls (a fresh process re-deriving from the same manifest).
    again = cluster_columns_by_environment(records, n_clusters=2, env_scales=(1.0, 1.0, 1.0))
    assert again.representative_indices == out.representative_indices


def test_labels_in_range_and_reps_distinct():
    records = _two_separated_groups()
    out = cluster_columns_by_environment(records, n_clusters=3)
    assert len(set(out.representative_indices)) == len(out.representative_indices)
    assert all(0 <= label < len(out.representative_indices) for label in out.labels)
    assert len(out.labels) == len(records)


def test_each_representative_labels_to_own_cluster():
    """Each representative belongs to ITS OWN cluster: labels[reps[j]] == j (a rep is
    at distance 0 from itself in normalised env space, so it is its own nearest rep).
    A bug in the rep-ordering ↔ label-index mapping (or the dist_to_reps matrix) would
    put a representative in another cluster — invisible to the in-range / distinctness
    checks, caught here. Also confirms every cluster is non-empty (each j is used)."""
    records = _two_separated_groups()
    out = cluster_columns_by_environment(records, n_clusters=3)
    for j, rep in enumerate(out.representative_indices):
        assert out.labels[rep] == j, (j, rep, out.labels[rep])
    assert set(out.labels) == set(range(len(out.representative_indices)))


def test_env_scales_changes_representative():
    """env_scales re-weights the axes ⇒ a DIFFERENT 2nd representative — proving
    the override is load-bearing, not ignored.  Anchor (worst score) = rec 0."""
    records = [
        _record(0, 9.0, 300.0, 1000.0, 10.0),   # anchor
        _record(1, 1.0, 300.0, 5000.0, 10.5),   # far in CAPE, close in shear
        _record(2, 0.5, 300.0, 1050.0, 30.0),   # close in CAPE, far in shear
    ]
    # CAPE-dominated (scales=1) → rec 1 is farthest from the anchor.
    cape_dom = cluster_columns_by_environment(records, n_clusters=2, env_scales=(1.0, 1.0, 1.0))
    assert set(cape_dom.representative_indices) == {0, 1}
    # Down-weight CAPE 1000× → shear dominates → rec 2 is farthest instead.
    shear_dom = cluster_columns_by_environment(records, n_clusters=2, env_scales=(1.0, 1000.0, 1.0))
    assert set(shear_dom.representative_indices) == {0, 2}


def test_empty_manifest_raises():
    with pytest.raises(ValueError, match="empty manifest"):
        cluster_columns_by_environment([], n_clusters=2)


def test_nonpositive_n_clusters_raises():
    records = _two_separated_groups()
    with pytest.raises(ValueError, match="n_clusters must be > 0"):
        cluster_columns_by_environment(records, n_clusters=0)


def test_nonfinite_score_raises():
    records = _two_separated_groups()
    bad = records[:2] + [_record(99, float("nan"), 300.0, 1000.0, 10.0)]
    with pytest.raises(ValueError, match="non-finite combined_score"):
        cluster_columns_by_environment(bad, n_clusters=2)


def test_nonfinite_environment_raises():
    records = _two_separated_groups()
    bad = records[:2] + [_record(99, 1.0, float("inf"), 1000.0, 10.0)]
    with pytest.raises(ValueError, match="non-finite environment tag"):
        cluster_columns_by_environment(bad, n_clusters=2)


@pytest.mark.parametrize("scales", [(1.0, -1.0, 1.0), (1.0, 0.0, 1.0),
                                    (1.0, float("nan"), 1.0), (1.0, 1.0)])
def test_bad_env_scales_raise(scales):
    records = _two_separated_groups()
    with pytest.raises(ValueError, match="env_scales must"):
        cluster_columns_by_environment(records, n_clusters=2, env_scales=scales)
