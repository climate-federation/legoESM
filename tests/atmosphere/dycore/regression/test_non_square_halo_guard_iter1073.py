"""FV3_3D iter-1073: non-square halo guard coverage.

iter-1072 added a ``ValueError`` guard at ``pad_halo_4d`` entry to
turn the silent-corruption of non-square cubed-sphere data into a
loud failure.  Codex review of iter-1072 surfaced a BLOCKER:
several sibling 4D halo entry points (``pad_halo_mpi_4d``,
``pad_halo_vector_4d``, ``explicit_pad_halo_4d``,
``packed_pad_halo_4d``) bypass the guard and could still hit the
buggy helpers via direct calls.

iter-1073 added the same guard to each sibling entry.  This file
pins the contract by directly exercising every guarded entry point
with non-square input and asserting it raises ``ValueError`` with
the expected iter-1072 pointer in the error message.

Run with::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \\
        tests/test_non_square_halo_guard_iter1073.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


jax.config.update("jax_enable_x64", True)


# Non-square shapes representative of the iter-370 / cross_face_du_proj
# call sites: ``du_normal`` is ``(6, n+1, n, nlev)``;
# ``dv_normal`` is ``(6, n, n+1, nlev)``.
@pytest.fixture(params=[(5, 6), (6, 5)], ids=["n_x<n_y", "n_x>n_y"])
def non_square_4d(request):
    """A (6, n_x, n_y, nlev) array with deterministic per-face values."""
    n_x, n_y = request.param
    nlev = 3
    data = jnp.zeros((6, n_x, n_y, nlev), dtype=jnp.float64)
    for f in range(6):
        data = data.at[f].set(float(f + 1))
    return data, n_x, n_y


def _assert_iter1072_msg(exc: ValueError) -> None:
    """Every iter-1073 guard cites iter-1072 in its error message."""
    msg = str(exc)
    assert "iter-1072" in msg, (
        f"ValueError missing iter-1072 pointer for archaeology.  "
        f"Got: {msg!r}"
    )
    assert "square" in msg.lower(), (
        f"ValueError should mention the square requirement.  "
        f"Got: {msg!r}"
    )


class TestNonSquareHaloGuards:
    """Exercise each iter-1073 guard entry point."""

    def test_pad_halo_4d_rejects_non_square(self, non_square_4d):
        """iter-1072 entry-point guard."""
        from legoesm.grids.halo import pad_halo_4d

        data, _, _ = non_square_4d
        with pytest.raises(ValueError) as exc_info:
            pad_halo_4d(data)
        _assert_iter1072_msg(exc_info.value)

    def test_pad_halo_vector_4d_rejects_non_square_u(self, non_square_4d):
        """iter-1073 vector u_data guard."""
        from legoesm.grids.halo import pad_halo_vector_4d

        u_data, n_x, n_y = non_square_4d
        # v_data is square so failure must come from u_data check.
        v_data = jnp.zeros((6, n_x, n_x, 3), dtype=jnp.float64)
        # Pass dummy metric arrays; the guard fires before they're used.
        ang = jnp.zeros((6, n_x, n_y), dtype=jnp.float64)
        ang_p = jnp.zeros((6, n_x + 2, n_y + 2), dtype=jnp.float64)
        with pytest.raises(ValueError) as exc_info:
            pad_halo_vector_4d(u_data, v_data, ang, ang, ang_p, ang_p)
        _assert_iter1072_msg(exc_info.value)
        assert "u_data" in str(exc_info.value)

    def test_pad_halo_vector_4d_rejects_non_square_v(self, non_square_4d):
        """iter-1073 vector v_data guard."""
        from legoesm.grids.halo import pad_halo_vector_4d

        v_data, n_x, n_y = non_square_4d
        u_data = jnp.zeros((6, n_x, n_x, 3), dtype=jnp.float64)
        ang = jnp.zeros((6, n_x, n_y), dtype=jnp.float64)
        ang_p = jnp.zeros((6, n_x + 2, n_y + 2), dtype=jnp.float64)
        with pytest.raises(ValueError) as exc_info:
            pad_halo_vector_4d(u_data, v_data, ang, ang, ang_p, ang_p)
        _assert_iter1072_msg(exc_info.value)
        assert "v_data" in str(exc_info.value)

    def test_explicit_pad_halo_4d_rejects_non_square(self, non_square_4d):
        """iter-1073 SPMD entry-point guard."""
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d

        data, _, _ = non_square_4d
        # mesh=None is fine — the guard fires before mesh is dispatched.
        with pytest.raises(ValueError) as exc_info:
            explicit_pad_halo_4d(data, mesh=None)
        _assert_iter1072_msg(exc_info.value)

    def test_packed_pad_halo_4d_rejects_non_square(self, non_square_4d):
        """iter-1073 SPMD packed entry-point guard."""
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

        data, _, _ = non_square_4d
        with pytest.raises(ValueError) as exc_info:
            packed_pad_halo_4d(data, mesh=None)
        _assert_iter1072_msg(exc_info.value)

    def test_packed_pad_halo_4d_rejects_mixed_fields(self, non_square_4d):
        """iter-1073 multi-field packed guard: even one bad field
        triggers the ValueError with the offending field index in the
        message."""
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

        bad_data, n_x, n_y = non_square_4d
        good_data = jnp.zeros((6, n_x, n_x, 3), dtype=jnp.float64)
        with pytest.raises(ValueError) as exc_info:
            # Bad field at index 1.
            packed_pad_halo_4d(good_data, bad_data, mesh=None)
        _assert_iter1072_msg(exc_info.value)
        assert "field 1" in str(exc_info.value), (
            f"ValueError should identify the offending field index.  "
            f"Got: {exc_info.value}"
        )


class TestSquareDataStillWorks:
    """Sanity: the iter-1073 guards don't break square-data callers."""

    def test_pad_halo_4d_square_ok(self):
        from legoesm.grids.halo import pad_halo_4d

        n = 6
        data = jnp.zeros((6, n, n, 2), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f + 1))
        out = pad_halo_4d(data)
        assert out.shape == (6, n + 2, n + 2, 2)

    def test_pad_halo_vector_4d_square_ok(self):
        from legoesm.grids.halo import pad_halo_vector_4d

        n = 6
        u = jnp.ones((6, n, n, 2), dtype=jnp.float64)
        v = jnp.ones((6, n, n, 2), dtype=jnp.float64)
        cos_a = jnp.ones((6, n, n), dtype=jnp.float64)
        sin_a = jnp.zeros((6, n, n), dtype=jnp.float64)
        cos_a_p = jnp.ones((6, n + 2, n + 2), dtype=jnp.float64)
        sin_a_p = jnp.zeros((6, n + 2, n + 2), dtype=jnp.float64)
        u_pad, v_pad = pad_halo_vector_4d(u, v, cos_a, sin_a, cos_a_p, sin_a_p)
        assert u_pad.shape == (6, n + 2, n + 2, 2)
        assert v_pad.shape == (6, n + 2, n + 2, 2)
