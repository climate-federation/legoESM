"""Guards for round 161's face-ratio substitution and its producer walk.

Round 160 measured that the free-surface ratio at the u and v points, which
the velocity-indicator continuity solve rebuilds each face transport from,
differs from the oracle's own recorded one on every active face.  Round 161
substitutes the oracle's own ratio for exactly one continuity call and shows it
does NOT own the vertical-velocity residual, then decomposes the ratio
difference itself into the statement legoESM writes and the sea surface height
it feeds that statement.

These guards are structural and run without a GYRE step: that the walk's
producer substitution really is confined to the velocity indicator, that the
native/redundant face map it inverts is the one the shared helper writes, that
the ratio statement is linear in the sea surface height (which is why the
oracle's stage composition and legoESM's cannot differ by more than rounding),
and that both of the walk's plants exist and are refused unless they are known
names.  The measured numbers are in the round-161 receipt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.vertical import (
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_face_geometry_from_operands,
)

REPO = Path(__file__).resolve().parents[3]
WALK_PATH = (
    REPO / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_year_owners.py")


@pytest.fixture(scope="module")
def walk():
    spec = importlib.util.spec_from_file_location(
        "_gyre_year_owners_r161", WALK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operands(ny=5, nx=6, nlev=3):
    rng = np.random.default_rng(1161)
    e3u_0 = jnp.asarray(rng.uniform(40.0, 120.0, (ny, nx, nlev)))
    e3v_0 = jnp.asarray(rng.uniform(40.0, 120.0, (ny, nx, nlev)))
    umask3 = jnp.asarray(np.ones((ny, nx, nlev)))
    vmask3 = jnp.asarray(np.ones((ny, nx, nlev)))
    hu_0 = jnp.asarray(np.sum(np.asarray(e3u_0), axis=-1))
    hv_0 = jnp.asarray(np.sum(np.asarray(e3v_0), axis=-1))
    area_t = jnp.asarray(rng.uniform(1.0e9, 2.0e9, (ny, nx)))
    area_u = jnp.asarray(rng.uniform(1.0e9, 2.0e9, (ny, nx)))
    area_v = jnp.asarray(rng.uniform(1.0e9, 2.0e9, (ny, nx)))
    return e3u_0, e3v_0, umask3, vmask3, hu_0, hv_0, area_t, area_u, area_v


def test_the_walk_and_its_plants_exist(walk):
    """The round's two plants are named on the walk, and an unknown plant is
    refused rather than silently ignored."""
    assert walk.DEVELOPED_FACE_R3_PLANTS == ("face-r3-inert", "face-r3-tracer")
    assert callable(walk.developed_stage2_face_r3_walk)
    assert callable(walk.developed_process_rank_split)
    with pytest.raises(walk.GateError, match="unknown developed face-r3"):
        walk.developed_stage2_face_r3_walk(
            Path("/nonexistent"), Path("/nonexistent"), "0" * 40,
            Path("/nonexistent"), Path("/nonexistent"), plant="not-a-plant")


def test_the_native_face_map_the_walk_inverts_is_the_shared_one():
    """The walk installs the oracle's ratio on NEMO's NATIVE extent, which it
    reaches by dropping legoESM's redundant U wrap column and V south row.
    That inverse has to be the map the shared helper writes, or the
    substitution would be shifted by one face and would still 'work'."""
    ops = _operands()
    rng = np.random.default_rng(7)
    eta = jnp.asarray(rng.uniform(-0.4, 0.4, ops[6].shape))
    native = nemo_qco_live_face_geometry_from_operands(eta, *ops)
    _, _, one_plus_r3u, one_plus_r3v = nemo_qco_live_face_geometry_cgrid(
        eta, *ops)
    # The helper returns ``1 + ratio``, so the map is asserted on THAT
    # quantity, where it is exact.  Asserting it on the recovered ratio would
    # fail for a reason that has nothing to do with the map, which is how the
    # readout floor below was found.
    assert np.array_equal(
        np.asarray(one_plus_r3u)[:, 1:], 1.0 + np.asarray(native.r3u))
    assert np.array_equal(
        np.asarray(one_plus_r3v)[1:, :], 1.0 + np.asarray(native.r3v))
    # And the wrap/wall the map inserts is exactly what it claims to be.
    assert np.array_equal(
        np.asarray(one_plus_r3u)[:, 0], 1.0 + np.asarray(native.r3u)[:, -1])
    assert np.array_equal(
        np.asarray(one_plus_r3v)[0], np.ones_like(np.asarray(one_plus_r3v)[0]))


def test_recovering_the_ratio_from_the_stretching_factor_costs_a_last_place():
    """Why the walk reads the OFFLINE ratio off the native helper instead of
    subtracting one from the redundant one.  The ratio is of order 1e-05 and
    the factor is of order 1, so the round trip quantises the ratio at half a
    unit in the last place of 1.0 -- about 1e-16, three orders ABOVE the
    ratio's own last place.  A decomposition that ignored this would read its
    own arithmetic as a transcription difference."""
    ops = _operands()
    rng = np.random.default_rng(23)
    eta = jnp.asarray(rng.uniform(-0.4, 0.4, ops[6].shape))
    native = np.asarray(
        nemo_qco_live_face_geometry_from_operands(eta, *ops).r3u)
    _, _, one_plus_r3u, _ = nemo_qco_live_face_geometry_cgrid(eta, *ops)
    recovered = np.asarray(one_plus_r3u)[:, 1:] - 1.0
    error = np.max(np.abs(recovered - native))
    assert error > 0.0, "the round trip is lossless here; pick a harder case"
    assert error <= np.spacing(1.0)
    # And it is ORDERS above the quantity's own resolution, which is the
    # whole point.
    assert error > 10.0 * float(np.max(np.spacing(np.abs(native))))


def test_the_ratio_statement_is_linear_in_the_sea_surface_height():
    """NEMO builds ONE ratio per step and COMBINES ratios per stage
    (``stprk3_stg.f90:211``), while legoESM builds one ratio from the already
    combined height (``vertical.py:247``).  The walk's whole decomposition
    rests on those being the same number up to rounding, which is true only if
    the statement is linear in the height -- so that is asserted, not assumed."""
    ops = _operands()
    rng = np.random.default_rng(11)
    before = jnp.asarray(rng.uniform(-0.4, 0.4, ops[6].shape))
    after = jnp.asarray(rng.uniform(-0.4, 0.4, ops[6].shape))
    two_thirds, one_third = 2.0 / 3.0, 1.0 / 3.0
    combined_height = nemo_qco_live_face_geometry_from_operands(
        two_thirds * before + one_third * after, *ops)
    combined_ratio = (
        two_thirds * np.asarray(
            nemo_qco_live_face_geometry_from_operands(before, *ops).r3u)
        + one_third * np.asarray(
            nemo_qco_live_face_geometry_from_operands(after, *ops).r3u))
    left = np.asarray(combined_height.r3u)
    scale = float(np.max(np.abs(left)))
    assert scale > 0.0
    # Same number to within a few last places of the ratio's own size; NOT
    # bit-identical, which is the whole reason the walk measures the row.
    assert np.max(np.abs(left - combined_ratio)) <= 8.0 * np.spacing(scale)
    assert not np.array_equal(left, combined_ratio)


def test_the_ratio_statement_actually_reads_the_height(monkeypatch):
    """Non-vacuity for the linearity guard above: perturbing the height must
    move the ratio, or the guard would pass on a constant."""
    ops = _operands()
    base = jnp.asarray(np.full(ops[6].shape, 0.1))
    moved = jnp.asarray(np.full(ops[6].shape, 0.2))
    left = np.asarray(nemo_qco_live_face_geometry_from_operands(base, *ops).r3u)
    right = np.asarray(
        nemo_qco_live_face_geometry_from_operands(moved, *ops).r3u)
    assert not np.array_equal(left, right)
    assert np.max(np.abs(right - left)) > 0.0
