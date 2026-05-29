"""Tests for the explicit cubed-sphere SPMD halo exchange.

Validates that both the all_gather and ppermute backends produce
identical results to the local pad_halo, and that vector and packed
exchanges are correct.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _make_6way_mesh():
    """Create a 6-device face-sharded mesh on CPU."""
    devices = jax.devices("cpu")
    if len(devices) < 6:
        pytest.skip("Need at least 6 CPU devices "
                     "(set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
    mesh = jax.sharding.Mesh(
        np.array(devices[:6]).reshape(6),
        axis_names=("face",),
    )
    return mesh


def _shard_on_face(data, mesh):
    """Shard a (6, ...) array on the face axis."""
    P = jax.sharding.PartitionSpec
    spec = P("face", *((None,) * (data.ndim - 1)))
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.device_put(data, sharding)


@pytest.fixture
def mesh_6():
    return _make_6way_mesh()


# =======================================================================
# all_gather backend (default)
# =======================================================================

class TestAllGather3D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(42), (6, n, n))
        ref = np.array(_pad_halo_local(data))
        result = np.array(explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)

    def test_uniform(self, mesh_6):
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        data = jnp.ones((6, 8, 8)) * 7.0
        result = np.array(explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], 7.0)
        np.testing.assert_allclose(result[:, 0, 1:-1], 7.0, rtol=1e-6)
        np.testing.assert_allclose(result[:, -1, 1:-1], 7.0, rtol=1e-6)


class TestAllGather4D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        data = jax.random.normal(jax.random.PRNGKey(123), (6, 8, 8, 5))
        ref = np.array(_pad_halo_local_4d(data))
        result = np.array(explicit_pad_halo_4d(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)


# =======================================================================
# ppermute backend
# =======================================================================

class TestPpermute3D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            set_ppermute_default, explicit_pad_halo, _cache,
        )
        _cache.clear()
        set_ppermute_default(True)
        try:
            n = 8
            data = jax.random.normal(jax.random.PRNGKey(42), (6, n, n))
            ref = np.array(_pad_halo_local(data))
            result = np.array(
                explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10,
                                       err_msg="ppermute 3D differs from local")
        finally:
            set_ppermute_default(False)
            _cache.clear()


class TestPpermute4D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import (
            set_ppermute_default, explicit_pad_halo_4d, _cache,
        )
        _cache.clear()
        set_ppermute_default(True)
        try:
            data = jax.random.normal(jax.random.PRNGKey(7), (6, 8, 8, 5))
            ref = np.array(_pad_halo_local_4d(data))
            result = np.array(
                explicit_pad_halo_4d(_shard_on_face(data, mesh_6), mesh_6))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10,
                                       err_msg="ppermute 4D differs from local")
        finally:
            set_ppermute_default(False)
            _cache.clear()


# =======================================================================
# Packed multi-field exchange
# =======================================================================

class TestPackedExchange:
    def test_packed_matches_individual(self, mesh_6):
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_4d, packed_pad_halo_4d,
        )
        f1 = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8, 5))
        f2 = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8, 5))
        f3 = jax.random.normal(jax.random.PRNGKey(2), (6, 8, 8, 3))
        f1s, f2s, f3s = [_shard_on_face(f, mesh_6) for f in (f1, f2, f3)]
        ref1 = np.array(explicit_pad_halo_4d(f1s, mesh_6))
        ref2 = np.array(explicit_pad_halo_4d(f2s, mesh_6))
        ref3 = np.array(explicit_pad_halo_4d(f3s, mesh_6))
        r1, r2, r3 = [np.array(r) for r in packed_pad_halo_4d(f1s, f2s, f3s, mesh=mesh_6)]
        np.testing.assert_allclose(r1, ref1, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r2, ref2, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r3, ref3, rtol=1e-6, atol=1e-10)

    def test_packed_with_duogrid_matches_unpacked_with_duogrid(self, mesh_6):
        """Iter-84: packed SPMD halo with `duogrid=dg` must produce the
        same output as unpacked `pad_halo_4d(duogrid=dg)`. Before iter-84
        the packed path silently skipped the kinked-to-extended remap,
        so this test would have failed (first output would miss the
        duogrid correction while the reference applied it).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_4d
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d

        n, nlev = 8, 5
        grid = create_cubed_sphere(n, use_duogrid=True)
        dg = grid.duogrid
        assert dg is not None, "duogrid not active on grid"

        f1 = jax.random.normal(jax.random.PRNGKey(0), (6, n, n, nlev))
        f2 = jax.random.normal(jax.random.PRNGKey(1), (6, n, n, nlev))
        f1s, f2s = [_shard_on_face(f, mesh_6) for f in (f1, f2)]

        # Reference: unpacked pad_halo_4d with duogrid
        ref1 = np.array(pad_halo_4d(f1, duogrid=dg))
        ref2 = np.array(pad_halo_4d(f2, duogrid=dg))

        # Under test: packed SPMD halo with duogrid kwarg
        r1_s, r2_s = packed_pad_halo_4d(f1s, f2s, mesh=mesh_6, duogrid=dg)
        r1 = np.array(r1_s)
        r2 = np.array(r2_s)

        np.testing.assert_allclose(r1, ref1, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r2, ref2, rtol=1e-6, atol=1e-10)

    def test_packed_without_duogrid_unchanged(self, mesh_6):
        """Iter-84 safety: packed SPMD without duogrid kwarg must be
        bit-identical to the pre-iter-84 behaviour.
        """
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_4d, packed_pad_halo_4d,
        )
        n, nlev = 8, 5
        f = jax.random.normal(jax.random.PRNGKey(42), (6, n, n, nlev))
        fs = _shard_on_face(f, mesh_6)
        ref = np.array(explicit_pad_halo_4d(fs, mesh_6))
        (r,) = packed_pad_halo_4d(fs, mesh=mesh_6)
        np.testing.assert_array_equal(r, ref)


# =======================================================================
# Vector (u, v) exchange
# =======================================================================

class TestVectorExchange:
    def test_matches_scalar_reference(self, mesh_6):
        """Vector SPMD exchange must match the local pad_halo_vector_4d."""
        from legoesm.grids.halo import pad_halo_vector_4d, pad_halo_4d
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_vector_4d,
        )
        n, nlev = 8, 5
        u = jax.random.normal(jax.random.PRNGKey(10), (6, n, n, nlev))
        v = jax.random.normal(jax.random.PRNGKey(11), (6, n, n, nlev))
        # Synthetic angles (identity rotation for simplicity)
        cos_a = jnp.ones((6, n, n))
        sin_a = jnp.zeros((6, n, n))
        cos_a_p = jnp.ones((6, n + 2, n + 2))
        sin_a_p = jnp.zeros((6, n + 2, n + 2))

        # Reference: local path (rotates → pads each component → rotates back)
        u_ref, v_ref = pad_halo_vector_4d(
            u, v, cos_a, sin_a, cos_a_p, sin_a_p,
        )

        # Explicit SPMD path (packs both components → one exchange)
        us, vs = _shard_on_face(u, mesh_6), _shard_on_face(v, mesh_6)
        cos_as = _shard_on_face(cos_a, mesh_6)
        sin_as = _shard_on_face(sin_a, mesh_6)
        cos_aps = _shard_on_face(cos_a_p, mesh_6)
        sin_aps = _shard_on_face(sin_a_p, mesh_6)

        u_spmd, v_spmd = explicit_pad_halo_vector_4d(
            us, vs, cos_as, sin_as, cos_aps, sin_aps, mesh_6,
        )

        np.testing.assert_allclose(
            np.array(u_spmd), np.array(u_ref), rtol=1e-6, atol=1e-10,
            err_msg="vector u mismatch")
        np.testing.assert_allclose(
            np.array(v_spmd), np.array(v_ref), rtol=1e-6, atol=1e-10,
            err_msg="vector v mismatch")


# =======================================================================
# Halo=2 SPMD exchange
# =======================================================================

class TestHalo2SPMD:
    """SPMD halo=2 must match the local halo=2 reference.

    Regression: previously :func:`explicit_pad_halo` (halo=2) created an
    exchange object and silently discarded it, falling through to the
    local h2 helper.  On a sharded mesh the local helper has no
    neighbour-face data, so the halo cells contained junk.  The fix
    introduces a dedicated 2-deep-edge all_gather exchange.
    """

    def test_3d_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_h2
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(2024), (6, n, n))
        ref = np.array(_pad_halo_local_h2(data))
        result = np.array(
            explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6, halo=2),
        )
        np.testing.assert_allclose(
            result, ref, rtol=1e-6, atol=1e-10,
            err_msg="SPMD halo=2 3D differs from local h2 reference",
        )

    def test_4d_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_h2_4d
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        n, nlev = 8, 5
        data = jax.random.normal(jax.random.PRNGKey(2025), (6, n, n, nlev))
        ref = np.array(_pad_halo_local_h2_4d(data))
        result = np.array(
            explicit_pad_halo_4d(_shard_on_face(data, mesh_6), mesh_6, halo=2),
        )
        np.testing.assert_allclose(
            result, ref, rtol=1e-6, atol=1e-10,
            err_msg="SPMD halo=2 4D differs from local h2 reference",
        )


# =======================================================================
# Backend activation / dispatch integration
# =======================================================================

class TestBackendActivation:
    def test_activate_deactivate(self, mesh_6):
        from legoesm.grids.halo import get_halo_backend
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        assert get_halo_backend() in ("local", "spmd")
        activate_spmd_halo_backend(mesh_6)
        assert get_halo_backend() == "spmd"
        deactivate_spmd_halo_backend()
        assert get_halo_backend() == "local"

    def test_pad_halo_dispatches_to_spmd(self, mesh_6):
        from legoesm.grids.halo import pad_halo, _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(99), (6, n, n))
        ref = np.array(_pad_halo_local(data))
        activate_spmd_halo_backend(mesh_6)
        try:
            result = np.array(pad_halo(_shard_on_face(data, mesh_6)))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)
        finally:
            deactivate_spmd_halo_backend()


# =======================================================================
# SPMD with interp_offsets (iter-7 plumbing)
# =======================================================================

class TestSPMDWithOffsets:
    """Bit-exact equivalence with local pad when interp_offsets are
    forwarded.  Guards the iter-7 plumbing: ``pad_halo`` /
    ``pad_halo_4d`` SPMD branch passes ``offsets`` to
    ``explicit_pad_halo[_4d]``, which in turn dispatches to the
    ``with_offsets`` variant of the allgather kernel.
    """

    def test_halo1_3d_with_offsets(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(0), (6, n, n))
        offsets = jax.random.normal(
            jax.random.PRNGKey(1), (6, 4, n),
        ) * 0.3
        ref = np.array(_pad_halo_local(data, offsets))
        result = np.array(explicit_pad_halo(
            _shard_on_face(data, mesh_6), mesh_6,
            halo=1, interp_offsets=offsets,
        ))
        # Bit-exact: same arithmetic on each device's strips.
        np.testing.assert_array_equal(result, ref)

    def test_halo2_3d_with_offsets(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_h2
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(2), (6, n, n))
        offsets = jax.random.normal(
            jax.random.PRNGKey(3), (6, 4, 2, n),
        ) * 0.3
        ref = np.array(_pad_halo_local_h2(data, offsets))
        result = np.array(explicit_pad_halo(
            _shard_on_face(data, mesh_6), mesh_6,
            halo=2, interp_offsets=offsets,
        ))
        np.testing.assert_array_equal(result, ref)

    def test_halo2_4d_with_offsets(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_h2_4d
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        n, nlev = 8, 5
        data = jax.random.normal(jax.random.PRNGKey(4), (6, n, n, nlev))
        offsets = jax.random.normal(
            jax.random.PRNGKey(5), (6, 4, 2, n),
        ) * 0.3
        ref = np.array(_pad_halo_local_h2_4d(data, offsets))
        result = np.array(explicit_pad_halo_4d(
            _shard_on_face(data, mesh_6), mesh_6,
            halo=2, interp_offsets=offsets,
        ))
        np.testing.assert_array_equal(result, ref)

    def test_packed_halo1_4d_with_offsets(self, mesh_6):
        """``packed_pad_halo_4d`` with offsets matches per-field
        ``_pad_halo_local_4d`` calls.  Iter-8 plumbing.
        """
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
        n, c1, c2 = 8, 5, 3
        f1 = jax.random.normal(jax.random.PRNGKey(6), (6, n, n, c1))
        f2 = jax.random.normal(jax.random.PRNGKey(7), (6, n, n, c2))
        offsets = jax.random.normal(
            jax.random.PRNGKey(8), (6, 4, n),
        ) * 0.3
        ref1 = np.array(_pad_halo_local_4d(f1, offsets))
        ref2 = np.array(_pad_halo_local_4d(f2, offsets))
        f1_s = _shard_on_face(f1, mesh_6)
        f2_s = _shard_on_face(f2, mesh_6)
        out1, out2 = packed_pad_halo_4d(
            f1_s, f2_s, mesh=mesh_6, interp_offsets=offsets,
        )
        np.testing.assert_array_equal(np.array(out1), ref1)
        np.testing.assert_array_equal(np.array(out2), ref2)

    def test_packed_halo2_4d_with_offsets(self, mesh_6):
        """``packed_pad_halo_4d(halo=2, ...)`` matches per-field
        ``_pad_halo_local_h2_4d`` calls.  Iter-10 generalization to
        halo=2 lets PPM-transport call sites pack their q_i / q_j
        halo exchanges into a single SPMD collective.
        """
        from legoesm.grids.halo import _pad_halo_local_h2_4d
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
        n, c1, c2 = 8, 5, 3
        f1 = jax.random.normal(jax.random.PRNGKey(11), (6, n, n, c1))
        f2 = jax.random.normal(jax.random.PRNGKey(12), (6, n, n, c2))
        offsets = jax.random.normal(
            jax.random.PRNGKey(13), (6, 4, 2, n),
        ) * 0.3
        ref1 = np.array(_pad_halo_local_h2_4d(f1, offsets))
        ref2 = np.array(_pad_halo_local_h2_4d(f2, offsets))
        f1_s = _shard_on_face(f1, mesh_6)
        f2_s = _shard_on_face(f2, mesh_6)
        out1, out2 = packed_pad_halo_4d(
            f1_s, f2_s, mesh=mesh_6, interp_offsets=offsets, halo=2,
        )
        np.testing.assert_array_equal(np.array(out1), ref1)
        np.testing.assert_array_equal(np.array(out2), ref2)

    def test_pad_halo_pair_h2_local_fallback(self):
        """``pad_halo_pair_h2`` under the local backend falls back to
        two sequential ``pad_halo(halo=2)`` calls — bit-exact match.
        This guards the iter-11 PPM rewiring against accidental
        regressions on single-device runs.
        """
        from legoesm.grids.halo import (
            pad_halo, pad_halo_pair_h2, get_halo_backend,
        )
        # Verify we're in the local backend (or fall through)
        assert get_halo_backend() in ("local", "spmd")
        n = 8
        q1 = jax.random.normal(jax.random.PRNGKey(20), (6, n, n))
        q2 = jax.random.normal(jax.random.PRNGKey(21), (6, n, n))
        offsets = jax.random.normal(
            jax.random.PRNGKey(22), (6, 4, 2, n),
        ) * 0.3
        # Sequential references
        ref1 = np.array(pad_halo(q1, halo=2, interp_offsets=offsets))
        ref2 = np.array(pad_halo(q2, halo=2, interp_offsets=offsets))
        # Pair-pack (under local backend, falls back to sequential)
        out1, out2 = pad_halo_pair_h2(q1, q2, interp_offsets=offsets)
        np.testing.assert_array_equal(np.array(out1), ref1)
        np.testing.assert_array_equal(np.array(out2), ref2)

    def test_halo1_3d_with_offsets_ppermute(self, mesh_6):
        """ppermute backend with offsets matches local pad_halo
        bit-for-bit.  Iter-17 plumbed ``interp_offsets`` through the
        ppermute kernel so the bandwidth-optimal high-resolution path
        is also numerically equivalent to single-device.
        """
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo, set_ppermute_default,
        )
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(40), (6, n, n))
        offsets = jax.random.normal(
            jax.random.PRNGKey(41), (6, 4, n),
        ) * 0.3
        ref = np.array(_pad_halo_local(data, offsets))
        set_ppermute_default(True)
        try:
            result = np.array(explicit_pad_halo(
                _shard_on_face(data, mesh_6), mesh_6,
                halo=1, interp_offsets=offsets,
            ))
        finally:
            set_ppermute_default(False)
        np.testing.assert_array_equal(result, ref)

    def test_pad_halo_pair_h2_spmd(self, mesh_6):
        """``pad_halo_pair_h2`` under the SPMD backend produces a
        bit-exact match against the local pad_halo reference.  Guards
        the iter-11 packing path through PPM transport.
        """
        from legoesm.grids.halo import (
            pad_halo, pad_halo_pair_h2,
        )
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        n = 8
        q1 = jax.random.normal(jax.random.PRNGKey(30), (6, n, n))
        q2 = jax.random.normal(jax.random.PRNGKey(31), (6, n, n))
        offsets = jax.random.normal(
            jax.random.PRNGKey(32), (6, 4, 2, n),
        ) * 0.3
        # Sequential local references (no SPMD)
        ref1 = np.array(pad_halo(q1, halo=2, interp_offsets=offsets))
        ref2 = np.array(pad_halo(q2, halo=2, interp_offsets=offsets))
        activate_spmd_halo_backend(mesh_6)
        try:
            q1_s = _shard_on_face(q1, mesh_6)
            q2_s = _shard_on_face(q2, mesh_6)
            out1, out2 = pad_halo_pair_h2(q1_s, q2_s, interp_offsets=offsets)
            np.testing.assert_array_equal(np.array(out1), ref1)
            np.testing.assert_array_equal(np.array(out2), ref2)
        finally:
            deactivate_spmd_halo_backend()


# =======================================================================
# Tracer-leak-free SPMD halo under JIT (#327)
# =======================================================================

class TestTracerLeakFree:
    """Regression test for #327.

    The exchange-kernel factories (``_make_exchange_allgather``,
    ``_make_exchange_allgather_h2``, ``_make_exchange_ppermute``) call
    ``jnp.asarray`` in their outer bodies.  If ``_get_exchange`` is
    first called for a key from *inside* an active JIT trace (e.g. the
    first ``_step_cell_centre`` compilation), the resulting traced
    values get captured in the ``_exchange`` closure and stored in the
    module-level ``_cache`` — an ``UnexpectedTracerError`` (production
    job 25211021 crashed after 5 min).

    The fix pre-warms ``_cache`` inside ``activate_spmd_halo_backend``
    (outside any trace) for all relevant keys, so a later call from
    inside JIT is a cache hit and never invokes the factory.

    These tests JIT-compile a function that calls the explicit pad-halo
    exchange and assert it runs without raising.  Run under
    ``JAX_CHECK_TRACER_LEAKS=1`` (and
    ``XLA_FLAGS=--xla_force_host_platform_device_count=6``) for the
    sharpest regression signal — without the pre-warm fix the jitted
    call raises ``UnexpectedTracerError`` during tracing.
    """

    def test_no_tracer_leak_jitted_pad_halo_4d(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
            explicit_pad_halo_4d, _cache,
        )
        _cache.clear()
        # activate_spmd_halo_backend pre-warms _cache OUTSIDE any trace.
        activate_spmd_halo_backend(mesh_6, n=8)
        try:
            data = jax.random.normal(jax.random.PRNGKey(327), (6, 8, 8, 5))
            ref = np.array(_pad_halo_local_4d(data))
            data_s = _shard_on_face(data, mesh_6)

            @jax.jit
            def jitted(d):
                return explicit_pad_halo_4d(d, mesh_6)

            # Without the pre-warm fix this raises UnexpectedTracerError
            # while tracing (the factory's jnp.asarray runs in-trace and
            # the result is cached); with the fix the cache hit avoids it.
            result = np.array(jitted(data_s))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)
        finally:
            deactivate_spmd_halo_backend()
            _cache.clear()

    def test_no_tracer_leak_jitted_pad_halo_3d(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
            explicit_pad_halo, _cache,
        )
        _cache.clear()
        activate_spmd_halo_backend(mesh_6, n=8)
        try:
            data = jax.random.normal(jax.random.PRNGKey(328), (6, 8, 8))
            ref = np.array(_pad_halo_local(data))
            data_s = _shard_on_face(data, mesh_6)

            @jax.jit
            def jitted(d):
                return explicit_pad_halo(d, mesh_6)

            result = np.array(jitted(data_s))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)
        finally:
            deactivate_spmd_halo_backend()
            _cache.clear()
