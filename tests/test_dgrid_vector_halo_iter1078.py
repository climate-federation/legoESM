"""FV3_3D iter-1078: DGRID_NE vector halo with axis-swap component swap."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _build_distinct_per_face(n, nlev=1):
    u_d = jnp.ones((6, n, n + 1, nlev), dtype=jnp.float64)
    v_d = jnp.ones((6, n + 1, n, nlev), dtype=jnp.float64)
    for f in range(6):
        u_d = u_d.at[f].set(float(f + 1))
        v_d = v_d.at[f].set(float(f + 1) * 10.0)
    return u_d, v_d


class TestVectorHaloShapes:
    def test_output_shapes(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n, nlev = 4, 3
        u_d, v_d = _build_distinct_per_face(n, nlev)
        u_pad, v_pad = pad_halo_dgrid_vector_4d(u_d, v_d)
        assert u_pad.shape == (6, n + 2, n + 3, nlev)
        assert v_pad.shape == (6, n + 3, n + 2, nlev)

    def test_rejects_3d(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        with pytest.raises(ValueError, match="4D inputs"):
            pad_halo_dgrid_vector_4d(jnp.zeros((6, 4, 5)), jnp.zeros((6, 5, 4)))

    def test_rejects_wrong_face_count(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        with pytest.raises(ValueError, match="6 faces"):
            pad_halo_dgrid_vector_4d(jnp.zeros((3, 4, 5, 1)), jnp.zeros((3, 5, 4, 1)))

    def test_rejects_mismatched_shapes(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        with pytest.raises(ValueError, match="u_d expects"):
            pad_halo_dgrid_vector_4d(
                jnp.zeros((6, 4, 5, 1)), jnp.zeros((6, 4, 5, 1))
            )


class TestAxisSwapSigns:
    @pytest.fixture
    def padded(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n = 4
        u_d, v_d = _build_distinct_per_face(n)
        return n, pad_halo_dgrid_vector_4d(u_d, v_d)

    def test_face_1_north_u_halo_equals_plus_v4(self, padded):
        """Face 1 NORTH ↔ Face 4 EAST, sign_uv=+1.  u_1_halo ← +v_4."""
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[1, 1:n + 1, n + 2, 0]),
            np.full(n, 50.0),  # v_4 = 5*10
        )

    def test_face_1_north_v_halo_equals_minus_u4(self, padded):
        """sign_vu=-1.  v_1_halo ← -u_4."""
        n, (_, v_pad) = padded
        np.testing.assert_array_equal(
            np.asarray(v_pad[1, 1:n + 2, n + 1, 0]),
            np.full(n + 1, -5.0),  # -u_4 = -5
        )

    def test_face_4_east_u_halo_equals_minus_v1(self, padded):
        """Face 4 EAST ↔ Face 1 NORTH, sign_uv=-1.  u_4_halo ← -v_1."""
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[4, n + 1, 1:n + 2, 0]),
            np.full(n + 1, -20.0),  # -v_1 = -20
        )

    def test_face_4_east_v_halo_equals_plus_u1(self, padded):
        """sign_vu=+1.  v_4_halo ← +u_1."""
        n, (_, v_pad) = padded
        np.testing.assert_array_equal(
            np.asarray(v_pad[4, n + 2, 1:n + 1, 0]),
            np.full(n, 2.0),  # +u_1 = +2
        )

    def test_face_3_south_u_halo_equals_plus_v5(self, padded):
        """Face 3 SOUTH ↔ Face 5 WEST, sign_uv=+1.  u_3_halo ← +v_5."""
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[3, 1:n + 1, 0, 0]),
            np.full(n, 60.0),  # v_5 = 6*10
        )

    def test_face_3_north_u_halo_equals_minus_v4(self, padded):
        """Face 3 NORTH ↔ Face 4 WEST, sign_uv=-1, rev=True."""
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[3, 1:n + 1, n + 2, 0]),
            np.full(n, -50.0),  # -v_4 = -50
        )


class TestSameAxisStillFaithful:
    def test_face_0_west_u_halo_is_face_3_east(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n = 4
        u_d, v_d = _build_distinct_per_face(n)
        u_pad, _ = pad_halo_dgrid_vector_4d(u_d, v_d)
        np.testing.assert_array_equal(
            np.asarray(u_pad[0, 0, 1:n + 2, 0]),
            np.full(n + 1, 4.0),
        )

    def test_face_0_east_u_halo_is_face_1_west(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n = 4
        u_d, v_d = _build_distinct_per_face(n)
        u_pad, _ = pad_halo_dgrid_vector_4d(u_d, v_d)
        np.testing.assert_array_equal(
            np.asarray(u_pad[0, n + 1, 1:n + 2, 0]),
            np.full(n + 1, 2.0),
        )
