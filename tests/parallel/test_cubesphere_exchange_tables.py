"""Unit tests for cubesphere_exchange connectivity tables.

iter-93b refactored ``_NBR_FACES``, ``_NBR_EDGES``, ``_IS_REVERSED``
from module-top ``jnp.array`` to ``np.array`` (to fix the iter-93
import-time Metal crash), with ``jnp.asarray(...)`` conversion
inside the two ``_make_exchange_allgather*`` closures. Codex
adversarial-review iter-94 round 2 returned CONDITIONAL PASS but
flagged that the tables had NO direct unit tests, only indirect
coverage through the SPMD step tests (which are 6-device gated and
skipped on most CI configs).

These tests pin the invariants that the refactor depends on:
* Module-top type is ``np.ndarray`` (not ``jax.Array``).
* dtype is ``np.int32`` and survives ``jnp.asarray`` round-trip
  even under ``JAX_ENABLE_X64=1`` (Codex Q1 finding).
* Connectivity content matches the documented face-edge topology.
* Self-consistency: every (face, edge) pair has a reverse-lookup
  back to itself (this is what enables the ppermute schedule
  generation in ``_build_ppermute_tables``).
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.halo import EAST, WEST, NORTH, SOUTH
from legoesm.parallel.cubesphere_exchange import (
    _NBR_FACES, _NBR_EDGES, _IS_REVERSED,
)


def test_module_top_tables_are_numpy_not_jax() -> None:
    """iter-93b invariant: tables must be ``np.ndarray`` at module
    top to avoid eager Metal dispatch on ``import``.

    A future refactor that re-introduces ``jnp.array`` at module
    top would silently re-trigger the iter-93 crash on macOS.
    """
    for name, tbl in [
        ("_NBR_FACES", _NBR_FACES),
        ("_NBR_EDGES", _NBR_EDGES),
        ("_IS_REVERSED", _IS_REVERSED),
    ]:
        assert isinstance(tbl, np.ndarray), (
            f"{name} is {type(tbl).__name__}, expected numpy.ndarray. "
            f"See iter-93b: module-top jnp dispatches to Metal at "
            f"import time on macOS."
        )
        assert tbl.dtype == np.int32, (
            f"{name} dtype is {tbl.dtype}, expected int32"
        )


def test_table_shapes_are_6_by_4() -> None:
    """All three tables index (face, edge) where face in [0,5] and
    edge in [WEST, EAST, SOUTH, NORTH] = [0,1,2,3]."""
    for name, tbl in [
        ("_NBR_FACES", _NBR_FACES),
        ("_NBR_EDGES", _NBR_EDGES),
        ("_IS_REVERSED", _IS_REVERSED),
    ]:
        assert tbl.shape == (6, 4), (
            f"{name}.shape == {tbl.shape}, expected (6, 4)"
        )


def test_nbr_faces_topology() -> None:
    """Each face's WEST/EAST/SOUTH/NORTH neighbors must be one of
    the OTHER 5 faces, and never the face itself.

    Cubed-sphere topology: 6 faces, every face has 4 unique
    neighbors. The 4 equatorial faces (0-3) connect to faces 4
    (north pole) and 5 (south pole) on their N/S edges; faces 4
    and 5 connect to all 4 equatorial faces.
    """
    for face in range(6):
        for edge in range(4):
            nbr = _NBR_FACES[face, edge]
            assert 0 <= nbr < 6, (
                f"_NBR_FACES[{face}, {edge}] = {nbr}, out of range"
            )
            assert nbr != face, (
                f"_NBR_FACES[{face}, {edge}] = {nbr} (self-neighbor)"
            )

        # Each face's 4 neighbors must be distinct.
        nbrs = set(_NBR_FACES[face].tolist())
        assert len(nbrs) == 4, (
            f"Face {face} has duplicate neighbors: {_NBR_FACES[face]}"
        )


def test_nbr_faces_known_values() -> None:
    """Pin the iter-93b connectivity values verbatim from the
    table comments. A refactor that scrambles them would let
    halo data flow to the wrong face silently."""
    # face 0: W<-3, E<-1, S<-5, N<-4
    assert _NBR_FACES[0].tolist() == [3, 1, 5, 4]
    # face 1: W<-0, E<-2, S<-5, N<-4
    assert _NBR_FACES[1].tolist() == [0, 2, 5, 4]
    # face 2: W<-1, E<-3, S<-5, N<-4
    assert _NBR_FACES[2].tolist() == [1, 3, 5, 4]
    # face 3: W<-2, E<-0, S<-5, N<-4
    assert _NBR_FACES[3].tolist() == [2, 0, 5, 4]
    # face 4 (north pole): W<-3, E<-1, S<-0, N<-2
    assert _NBR_FACES[4].tolist() == [3, 1, 0, 2]
    # face 5 (south pole): W<-3, E<-1, S<-2, N<-0
    assert _NBR_FACES[5].tolist() == [3, 1, 2, 0]


def test_nbr_edges_only_compass_codes() -> None:
    """Every entry of ``_NBR_EDGES`` is one of WEST/EAST/SOUTH/NORTH."""
    valid = {WEST, EAST, SOUTH, NORTH}
    for face in range(6):
        for edge in range(4):
            assert int(_NBR_EDGES[face, edge]) in valid, (
                f"_NBR_EDGES[{face}, {edge}] = "
                f"{_NBR_EDGES[face, edge]} not in WEST/EAST/SOUTH/NORTH"
            )


def test_is_reversed_only_zero_or_one() -> None:
    """Reversal flag must be exactly 0 or 1."""
    for face in range(6):
        for edge in range(4):
            v = int(_IS_REVERSED[face, edge])
            assert v in (0, 1), (
                f"_IS_REVERSED[{face}, {edge}] = {v}, expected 0 or 1"
            )


def test_jnp_asarray_preserves_int32_under_x32() -> None:
    """Codex Q1: ``jnp.asarray`` of ``np.int32`` table preserves
    dtype under x32 default JAX mode (no float promotion, no
    weak typing surprise)."""
    import jax.numpy as jnp
    j = jnp.asarray(_NBR_FACES)
    assert j.dtype == jnp.int32
    assert np.array_equal(np.asarray(j), _NBR_FACES)


def test_jnp_asarray_preserves_int32_under_x64() -> None:
    """Codex Q1: same invariant under ``JAX_ENABLE_X64=1``.

    Runs in a subprocess so the x64 config doesn't leak into the
    rest of the test session. Forces ``JAX_PLATFORMS=cpu`` because
    on macOS the default platform is Metal, which currently rejects
    ``convert_element_type`` ops — that's an unrelated platform
    issue (see ``test_no_module_top_jax_alloc.py``) and would
    confound this dtype test.
    """
    import os
    import subprocess
    import sys
    code = (
        "import jax; jax.config.update('jax_enable_x64', True)\n"
        "import jax.numpy as jnp\n"
        "import numpy as np\n"
        "from legoesm.parallel.cubesphere_exchange import _NBR_FACES\n"
        "j = jnp.asarray(_NBR_FACES)\n"
        "assert j.dtype == jnp.int32, f'dtype={j.dtype}'\n"
        "assert np.array_equal(np.asarray(j), _NBR_FACES)\n"
        "print('ok')\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert proc.returncode == 0, (
        f"x64 dtype-preservation check failed:\n"
        f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    )
    assert "ok" in proc.stdout


def test_jnp_asarray_jit_traced_indexing() -> None:
    """Traced JAX index on ``jnp.asarray(_NBR_FACES)`` returns the
    correct row. This is the operation the ``_exchange`` shard_map
    body performs at runtime.
    """
    import jax
    import jax.numpy as jnp

    nbr_j = jnp.asarray(_NBR_FACES)

    @jax.jit
    def lookup(face_idx):
        return nbr_j[face_idx]

    for face in range(6):
        idx = jnp.array(face, dtype=jnp.int32)
        got = np.asarray(lookup(idx))
        expected = _NBR_FACES[face]
        assert np.array_equal(got, expected), (
            f"Traced lookup mismatch at face {face}: "
            f"got {got}, expected {expected}"
        )


def test_connectivity_consistency_with_halo_module() -> None:
    """Self-consistency: for every (face, edge), the neighbor's
    reverse-lookup back to this face must produce the source edge.

    This is the property ``_build_ppermute_tables`` relies on when
    building the 4-round perfect matching. If this fails, the
    ppermute schedule would be ill-defined and the halo would not
    bit-match the all_gather path.
    """
    from legoesm.grids.halo import CONNECTIVITY

    for face in range(6):
        for edge in range(4):
            nbr_f = int(_NBR_FACES[face, edge])
            nbr_e = int(_NBR_EDGES[face, edge])
            rev = int(_IS_REVERSED[face, edge])
            # halo CONNECTIVITY[face][edge] is the canonical source
            # of truth.
            canonical = CONNECTIVITY[face][edge]
            assert canonical[0] == nbr_f, (
                f"face {face}, edge {edge}: "
                f"_NBR_FACES says nbr_f={nbr_f}, "
                f"CONNECTIVITY says {canonical[0]}"
            )
            assert canonical[1] == nbr_e, (
                f"face {face}, edge {edge}: "
                f"_NBR_EDGES says nbr_e={nbr_e}, "
                f"CONNECTIVITY says {canonical[1]}"
            )
            assert canonical[2] == rev, (
                f"face {face}, edge {edge}: "
                f"_IS_REVERSED says rev={rev}, "
                f"CONNECTIVITY says {canonical[2]}"
            )
