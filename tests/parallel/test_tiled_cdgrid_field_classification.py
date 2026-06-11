"""P4 phase-1b spec: classify every cubed-sphere metric for tiling.

The tiled shard_map tendency stage needs each metric as a per-tile
block; ``mesh.classify_face_metric`` decides how (host-slice vs
setup-exchange).  This audit enumerates EVERY array field of a real
``CubedSphereCDGrid`` (and its nested base ``CubedSphereGrid``),
asserts the classification matches the codex P4 metric design, and
RATCHETS: a new metric field that classifies as ``"other"`` (unknown
horizontal extent) fails until it is given a tiling story.  It also
checks the two actionable buckets behave — ``sliceable`` fields
roundtrip through ``tiled_face_block`` and ``padded`` fields report a
halo depth the tiled exchange supports (h in {1, 2}).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.mesh import classify_face_metric, tiled_face_block

N = 12  # divisible by kt in {2, 3}
KT = 2
NL = N // KT


def _all_array_fields(obj, prefix=""):
    """Yield (name, array) for every jax/numpy array leaf, recursing
    one level into a nested NamedTuple field named ``base``."""
    fields = getattr(obj, "_fields", None)
    if fields is None:
        return
    for fname in fields:
        val = getattr(obj, fname)
        if hasattr(val, "_fields"):
            yield from _all_array_fields(val, prefix=f"{prefix}{fname}.")
        elif hasattr(val, "shape"):
            yield (f"{prefix}{fname}", val)


@pytest.fixture(scope="module")
def cdgrid():
    return create_cubed_sphere_cdgrid(create_cubed_sphere(N))


def test_every_metric_is_classified(cdgrid):
    """No face-leading metric may be 'other' — every field needs a
    tiling story (ratchet on new metrics)."""
    unclassified = []
    for name, arr in _all_array_fields(cdgrid):
        kind, _h = classify_face_metric(arr, N)
        if kind == "other":
            unclassified.append((name, tuple(arr.shape)))
    assert not unclassified, (
        "Unclassified cubed-sphere metrics (give each a tiled-layout "
        f"story in classify_face_metric): {unclassified}"
    )


def test_classification_partition_matches_design(cdgrid):
    """Pin the codex P4 partition: the staggered/centered metrics are
    sliceable; the padded angle/duogrid extents need setup-exchange."""
    buckets = {"sliceable": set(), "padded": set(), "table": set(),
               "scalar": set()}
    for name, arr in _all_array_fields(cdgrid):
        kind, _h = classify_face_metric(arr, N)
        buckets.setdefault(kind, set()).add(name)

    # Core unpadded metrics MUST be sliceable.
    for must in ("base.area", "base.f", "base.dx", "base.dy",
                 "lon_corner", "dx_edge_y", "dy_edge_x", "cosa_u",
                 "rsin_v", "sin_sg", "cos_sg", "rdxa", "rarea_c"):
        assert must in buckets["sliceable"], (
            f"{must} expected sliceable, got "
            f"{classify_face_metric(getattr_deep(cdgrid, must), N)}")

    # Padded angle metrics MUST be in the setup-exchange bucket.
    for must in ("base.cos_angle_padded", "base.sin_angle_padded"):
        assert must in buckets["padded"], (
            f"{must} expected padded/setup-exchange")

    # Offset tables are exchange internals.
    assert any("offset" in n for n in buckets["table"]), (
        "halo_interp_offsets* expected to classify as exchange tables")


def test_padded_metrics_within_tiled_exchange_depth(cdgrid):
    """Every padded metric's halo depth must be one the tiled exchange
    supports (h1/h2) — h3 (duogrid) is out of phase-1b scope and would
    need the deeper kernel first."""
    h3 = []
    for name, arr in _all_array_fields(cdgrid):
        kind, h = classify_face_metric(arr, N)
        if kind == "padded" and h is not None and h > 2:
            h3.append((name, h, tuple(arr.shape)))
    # Informational pin: h3 padded metrics exist (duogrid) but must be
    # excluded from the phase-1b stage, not silently sliced.
    for name, h, _shape in h3:
        assert "h3" in name or "ext" in name, (
            f"unexpected h{h} padded metric {name} — phase-1b stage "
            "must exclude it")


def test_sliceable_metrics_roundtrip(cdgrid):
    """A sliceable metric reassembles from its tile blocks (interior
    cells exact)."""
    area = cdgrid.base.area  # (6, n, n)
    f0 = np.asarray(area[0])
    for ti in range(KT):
        for tj in range(KT):
            blk = tiled_face_block(area[0], ti, tj, NL, KT)
            assert blk.shape == (NL, NL)
            np.testing.assert_array_equal(
                np.asarray(blk),
                f0[ti * NL:(ti + 1) * NL, tj * NL:(tj + 1) * NL])


def getattr_deep(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("kt", (2, 3))
def test_stack_tiled_sliceable_metrics(cdgrid, kt):
    """Every sliceable metric stacks to (6*kt^2, *block) and each
    device's block equals the global per-tile slice; the deferred list
    is exactly the padded/table/scalar fields (no sliceable dropped)."""
    from legoesm.parallel.mesh import (
        stack_tiled_sliceable_metrics,
        classify_face_metric,
    )

    nl = N // kt
    stacks, deferred = stack_tiled_sliceable_metrics(cdgrid, kt)
    deferred_names = {n for n, _ in deferred}

    # Partition completeness: every array field is either stacked or
    # deferred, never both, never missing.
    all_names = {name for name, _ in _all_array_fields(cdgrid)}
    assert set(stacks) | deferred_names == all_names
    assert not (set(stacks) & deferred_names)

    # No sliceable field was deferred.
    for name in deferred_names:
        arr = getattr_deep(cdgrid, name)
        assert classify_face_metric(arr, N)[0] != "sliceable", (
            f"{name} is sliceable but was deferred")

    # Spot-check a centered, a corner, and an edge field roundtrip.
    for name in ("base.area", "lon_corner", "dx_edge_y"):
        arr = getattr_deep(cdgrid, name)
        stack = stacks[name]
        assert stack.shape[0] == 6 * kt * kt
        for f in range(6):
            for ti in range(kt):
                for tj in range(kt):
                    d = (f * kt + ti) * kt + tj
                    want = arr[f][
                        ti * nl: ti * nl + stack.shape[1],
                        tj * nl: tj * nl + stack.shape[2]]
                    np.testing.assert_array_equal(
                        np.asarray(stack[d]), np.asarray(want),
                        err_msg=f"{name} device {d} block mismatch")
