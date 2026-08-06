"""FV3_3D iter-1078: DGRID_NE vector halo with axis-swap component swap.

2026-08-04 EXTENSION: the original constant-per-face tests were VALUE-BLIND
to two defect classes the transplant probe caught against certified FV3
truth (fv3_recon/transplant2_9311777.log):

1. node-axis ghost strips sourced the neighbour's SEAM line (which this
   face already stores) instead of the line one INWARD — invisible to a
   constant field, where both lines hold the same value;
2. the four half-turn seams (2,S),(2,N),(4,N),(5,S) copied covariant
   components UNSIGNED — no old test probed those strips.

``TestHalfTurnSeamSigns`` and ``TestNodeAxisSourceRowOneInward`` pin each
class with exact integers; ``TestValueLevelVsAnalyticSwcoreHalos`` is the
value-level gate against ``analytic_swcore_state``'s certified halos that
would have caught BOTH.
"""
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


def _build_linear_fields(n, nlev=1):
    """u varies along its NODE axis j; v along its node axis i.

    Exposes the normal-axis SOURCE-ROW choice at every seam: the shared
    seam line and the one-inward line differ by exactly 1.0, so a
    seam-line copy (the pre-2026-08-04 defect) is an exact integer error.
    """
    u = np.zeros((6, n, n + 1, nlev))
    v = np.zeros((6, n + 1, n, nlev))
    for f in range(6):
        u[f] = 100.0 * (f + 1) + np.arange(n + 1)[None, :, None]
        v[f] = 100.0 * (f + 1) + np.arange(n + 1)[:, None, None]
    return jnp.asarray(u), jnp.asarray(v)


# Covariant D-grid vector transform under create = rot90(native, k) — the
# INDEPENDENT adapter spec (deliberately NOT imported from dgrid_halo, so
# the comparison stays non-circular).  Frame tangents relabel k=1:
# (e1,e2)->(-e2,+e1) => u_c=-v_n, v_c=+u_n; composing gives the tables
# below; odd k also swaps the staggered arrays.  Full derivation:
# /burg-archive/glab/users/pg2328/fv3_recon/probe_transplant_NOTES.md §8.
# The same signs apply to the CONTRAVARIANT ua/va (the quarter-turn frame
# relabel is orthogonal, so co- and contravariant components transform
# identically).
_TRANSPLANT_SU = (1.0, -1.0, -1.0, 1.0)
_TRANSPLANT_SV = (1.0, 1.0, -1.0, -1.0)


def _transplant_analytic_winds(n, ng, u0, alpha):
    """Build the 6 native gridstructs + analytic states and transplant the
    PHYSICAL covariant D winds into the create layout.

    Returns ``(gs, st, u_c, v_c, PERM, ROT)`` with ``u_c (6, n, n+1)``,
    ``v_c (6, n+1, n)``.  Adapter: create face F = np.rot90(native tile
    PERM[F]+1, ROT[F]) (cubed_sphere.py GNOMONIC_ED_FACE_PERM/_ROT, pinned
    to the phase-2 literals) + the _TRANSPLANT_SU/_SV covariant law.
    """
    from legoesm.grids.cubed_sphere import (
        GNOMONIC_ED_FACE_PERM as PERM,
        GNOMONIC_ED_FACE_ROT as ROT,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )
    # pinned literals (test_fv3_native_metrics_phase2.py:297-298)
    assert tuple(PERM) == (0, 1, 3, 4, 2, 5)
    assert tuple(ROT) == (0, 0, 3, 3, 1, 0)
    gs = [build_fv3_native_gridstruct(n, ng, tile=t) for t in range(1, 7)]
    st = [analytic_swcore_state(g, u0=u0, alpha=alpha) for g in gs]
    u_c = np.empty((6, n, n + 1))
    v_c = np.empty((6, n + 1, n))
    for F in range(6):
        g, k = PERM[F], ROT[F]
        u_nat = st[g]["u"][ng:ng + n, ng:ng + n + 1]
        v_nat = st[g]["v"][ng:ng + n + 1, ng:ng + n]
        assert np.abs(u_nat).max() <= u0 * 1.001
        assert np.abs(v_nat).max() <= u0 * 1.001
        if k % 2 == 0:
            u_c[F] = _TRANSPLANT_SU[k] * np.rot90(u_nat, k)
            v_c[F] = _TRANSPLANT_SV[k] * np.rot90(v_nat, k)
        else:
            u_c[F] = _TRANSPLANT_SU[k] * np.rot90(v_nat, k)
            v_c[F] = _TRANSPLANT_SV[k] * np.rot90(u_nat, k)
    return gs, st, u_c, v_c, tuple(PERM), tuple(ROT)


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


class TestHalfTurnSeamSigns:
    """Half-turn (same-axis REVERSED) seams negate BOTH covariant components.

    CONNECTIVITY pairs: (2,S)<->(5,S) and (2,N)<->(4,N) — the neighbour's
    frame is rotated 180 degrees, so its +i/+j basis is the negation of
    this face's continued basis and covariant components cross with sign
    -1.  Certified vs analytic_swcore_state halos: transplant log line 260
    ``VEC v F=2 edge=S got=-2.5 want=+2.5`` (exact negation).  The pre-fix
    helper copied these strips UNSIGNED, so every assertion here was RED
    on it (ghosts read +value).  Constant per-face fields (u_f=f+1,
    v_f=10(f+1)) make the expectations exact.
    """

    @pytest.fixture(scope="class")
    def padded(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n = 4
        u_d, v_d = _build_distinct_per_face(n)
        return n, pad_halo_dgrid_vector_4d(u_d, v_d)

    @pytest.mark.parametrize("face,edge,nbr", [
        (2, "S", 5), (2, "N", 4), (4, "N", 2), (5, "S", 2),
    ])
    def test_half_turn_ghosts_negate_both_components(self, padded, face, edge, nbr):
        n, (u_pad, v_pad) = padded
        u_want = -float(nbr + 1)          # -u_nbr
        v_want = -10.0 * float(nbr + 1)   # -v_nbr
        if edge == "S":
            u_got = u_pad[face, 1:n + 1, 0, 0]
            v_got = v_pad[face, 1:n + 2, 0, 0]
        else:
            u_got = u_pad[face, 1:n + 1, n + 2, 0]
            v_got = v_pad[face, 1:n + 2, n + 1, 0]
        np.testing.assert_array_equal(np.asarray(u_got), np.full(n, u_want))
        np.testing.assert_array_equal(np.asarray(v_got), np.full(n + 1, v_want))


class TestNodeAxisSourceRowOneInward:
    """Node-axis ghosts source the neighbour row ONE INWARD of the seam.

    On a component's node-staggered axis (u: j, v: i) the outermost line
    IS the shared seam — both faces store that physical line — so the
    depth-1 ghost must carry the line one inward.  The pre-fix helper
    copied the seam line on EVERY seam including identity ones (transplant
    log line 176: create face 0 u S got=+3.5355 (shared edge)
    want=+3.2270 (one beyond); line 250 for v W; line 232 for the
    axis-swap class).  ``_build_linear_fields`` makes the two candidate
    rows differ by exactly 1.0, so each assertion is RED on a seam-line
    copy.
    """

    @pytest.fixture(scope="class")
    def padded(self):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        n = 4
        u_d, v_d = _build_linear_fields(n)
        return n, pad_halo_dgrid_vector_4d(u_d, v_d)

    def test_same_axis_identity_u_south_row(self, padded):
        """(0,S) <- (5,N), unreversed: ghost = u_5 at j=n-1, NOT j=n."""
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[0, 1:n + 1, 0, 0]),
            np.full(n, 600.0 + (n - 1)),
        )

    def test_same_axis_identity_v_west_row(self, padded):
        """(0,W) <- (3,E): ghost = v_3 at i=n-1, NOT i=n."""
        n, (_, v_pad) = padded
        np.testing.assert_array_equal(
            np.asarray(v_pad[0, 0, 1:n + 1, 0]),
            np.full(n, 400.0 + (n - 1)),
        )

    def test_axis_swap_u_south_sources_inward_v_row(self, padded):
        """(1,S) <- (5,E), sign_uv=-1: ghost = -v_5 at i=n-1, NOT i=n.

        v_5 depends only on i, so the reversed transverse order is
        value-invisible and the row choice is isolated.
        """
        n, (u_pad, _) = padded
        np.testing.assert_array_equal(
            np.asarray(u_pad[1, 1:n + 1, 0, 0]),
            np.full(n, -(600.0 + (n - 1))),
        )


class TestValueLevelVsAnalyticSwcoreHalos:
    """VALUE-LEVEL certification vs ``analytic_swcore_state`` halos.

    Certifies value-level semantics (not just stencil wiring): transplants
    each native tile's PHYSICAL covariant D-grid winds into the create
    layout, runs ``pad_halo_dgrid_vector_4d``, and compares every depth-1
    side ghost strip against the analytic state's OWN halos — which are
    evaluated directly at the kinked-node geometry, i.e. exactly what the
    FV3 mpp DGRID_NE exchange delivers (fv3_native_gridstruct.py
    docstring).  This is the unit-level port of
    fv3_recon/probe_transplant.py's TRANSPLANT_VEC section, which caught
    both 2026-08-04 defect classes (node-axis seam-row copies + missing
    half-turn signs, log fv3_recon/transplant2_9311777.log).

    Adapter (INDEPENDENT of dgrid_halo's tables — non-circular):
    - create face F = np.rot90(native tile PERM[F]+1, ROT[F])
      (cubed_sphere.py GNOMONIC_ED_FACE_PERM/_ROT, pinned below);
    - covariant components relabel with the frame: k=1 maps
      (e1,e2)->(-e2,+e1) so u_c=-v_n, v_c=+u_n; composing gives
      SU=(+,-,-,+), SV=(+,+,-,-) indexed by k, with the staggered
      arrays swapping for odd k (probe_transplant_NOTES.md §8);
    - alpha=pi/4 breaks the polar 4-fold symmetry (alpha=0 masks
      wrong-neighbour defects on the polar tiles by symmetry).
    """

    N = 12
    NG = 3
    U0 = 5.0
    ALPHA = np.pi / 4.0
    SU = _TRANSPLANT_SU
    SV = _TRANSPLANT_SV

    @staticmethod
    def _to_native_frame(uc, vc, k, n):
        """Continuous create coords -> native coords, create=rot90(native,k)."""
        if k == 0:
            return uc, vc
        if k == 1:
            return vc, n - uc
        if k == 2:
            return n - uc, n - vc
        return n - vc, uc

    @staticmethod
    def _classify(x):
        """Integer -> ('node', idx); half-integer -> ('cell', idx).  Exact."""
        x2 = 2.0 * x
        r = round(x2)
        assert x2 == float(r), x
        return ("node", r // 2) if r % 2 == 0 else ("cell", (r - 1) // 2)

    @pytest.fixture(scope="class")
    def transplant(self):
        _gs, st, u_c, v_c, PERM, ROT = _transplant_analytic_winds(
            self.N, self.NG, self.U0, self.ALPHA)
        return st, u_c, v_c, PERM, ROT

    def test_all_side_ghost_strips_match_certified_halos(self, transplant):
        from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
        st, u_c, v_c, PERM, ROT = transplant
        n, ng, u0 = self.N, self.NG, self.U0
        u_p, v_p = pad_halo_dgrid_vector_4d(
            jnp.asarray(u_c[..., None]), jnp.asarray(v_c[..., None]))
        u_p = np.asarray(u_p)[..., 0]
        v_p = np.asarray(v_p)[..., 0]

        # interiors bit-untouched
        np.testing.assert_array_equal(u_p[:, 1:-1, 1:-1], u_c)
        np.testing.assert_array_equal(v_p[:, 1:-1, 1:-1], v_c)

        def strips(field):
            # depth-1 side ghost (ic, jc) lists in field-local staggered
            # indices; diagonal corner cells excluded.
            if field == "u":   # (n, n+1): ic cell, jc node
                return {"W": [(-1, j) for j in range(n + 1)],
                        "E": [(n, j) for j in range(n + 1)],
                        "S": [(i, -1) for i in range(n)],
                        "N": [(i, n + 1) for i in range(n)]}
            return {"W": [(-1, j) for j in range(n)],
                    "E": [(n + 1, j) for j in range(n)],
                    "S": [(i, -1) for i in range(n + 1)],
                    "N": [(i, n) for i in range(n + 1)]}

        atol = 1e-9 * u0   # expected floor ~1e-13 (same numbers both sides)
        n_cmp = 0
        for field, padded in (("u", u_p), ("v", v_p)):
            for F in range(6):
                g, k = PERM[F], ROT[F]
                sgn = self.SU[k] if field == "u" else self.SV[k]
                for edge, cells in strips(field).items():
                    got = []
                    want = []
                    for ic, jc in cells:
                        uc_, vc_ = ((ic + 0.5, float(jc)) if field == "u"
                                    else (float(ic), jc + 0.5))
                        un, vn = self._to_native_frame(uc_, vc_, k, n)
                        ti, ii = self._classify(un)
                        tj, jj = self._classify(vn)
                        assert ti != tj, (field, F, edge, ic, jc)
                        arr = st[g]["u"] if ti == "cell" else st[g]["v"]
                        w = arr[ng + ii, ng + jj]
                        # BIG_NUMBER sentinel / physicality guard
                        assert abs(w) <= u0 * 1.001, (field, F, edge, ic, jc, w)
                        want.append(sgn * w)
                        got.append(padded[F, ic + 1, jc + 1])
                        n_cmp += 1
                    np.testing.assert_allclose(
                        np.asarray(got), np.asarray(want), rtol=0.0,
                        atol=atol,
                        err_msg=f"{field} F={F} T={g + 1} k={k} edge={edge}")
        # 2*(n+1) + 2*n strips per field per face
        assert n_cmp == 2 * 6 * (2 * (n + 1) + 2 * n)


class TestLocalGhostUaVaRingVsNumpyOracle:
    """s3 gate: the local_ghost_d2a ua/va ghost ring vs the certified
    NumPy ``d2a2c_vect`` (fv3_native_sw_core) on transplanted analytic
    winds — certifies value-level semantics of the Fortran-faithful LOCAL
    D→A route (``d2a2c_ua_va_halo_4d(..., local_ghost_d2a=True)``).

    CONSUMED-SET derivation (from ``_fv3_divergence_corner.py``, the
    ``uf``/``vf`` construction): ``uf_interior`` reads ``va_at_jface`` =
    0.5-pairs of ``va_h1`` — at INTERIOR j only (the ``is_uf_boundary``
    ``jnp.where`` discards the interior branch at j==0/n) and at ghost
    i_pad rows 0/n+1 → **va ring W/E COLUMNS at physical j**;
    ``vf_interior`` symmetrically reads ``ua_at_iface`` at interior i and
    ghost j_pad cols 0/n+1 → **ua ring S/N ROWS at physical i**.  Ring
    diagonals and the other two side strips of each field are never
    consumed.  ONLY those slots are compared here (the diagonals are
    additionally asserted poisoned).

    Truth: per native tile, ``d2a2c_vect(st.u, st.v, gs, bd)`` — the
    verbatim NumPy port whose UA/VA span two ghost rings with real
    metrics (fv3_native_sw_core.py:246-249); its corner OVERRIDES
    (:277-284/:333-340) touch only diagonal/ring-2 slots outside the
    compared set.  Mapping: create face F = rot90(tile PERM[F]+1, ROT[F])
    with the _TRANSPLANT_SU/_SV component law (contravariant == covariant
    under the orthogonal quarter-turn relabel).

    Tolerance: the JAX side runs on ``create_fv3_native_cubed_sphere`` +
    ``create_cubed_sphere_cdgrid(fv3_native_angles=True)`` whose angle
    metrics differ from the certified longdouble gridstruct build by
    ~3e-8 (reconcile_production_csw.py stage-1 note), giving a ~1e-6
    noise floor at u0=5 — atol=1e-5 sits ~10x above that and ~4 orders
    below a structural (row/sign/frame) error, which is O(0.1..1).
    """

    N = 12
    NG = 3
    U0 = 5.0
    ALPHA = np.pi / 4.0

    def test_consumed_ring_slots_match_numpy_d2a2c(self):
        from legoesm import constants
        from legoesm.core.fv3_native_sw_core import Bounds, d2a2c_vect
        from legoesm.core.fv3_sw_core import d2a2c_ua_va_halo_4d
        from legoesm.grids.cubed_sphere import create_fv3_native_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
        )

        n, ng, u0 = self.N, self.NG, self.U0
        gs, st, u_c, v_c, PERM, ROT = _transplant_analytic_winds(
            n, ng, u0, self.ALPHA)

        # NumPy oracle UA/VA per tile (full data domain, real metrics).
        bd = Bounds.single_tile(n, ng)
        oracle = []
        for t in range(6):
            ua_n, va_n, *_ = d2a2c_vect(
                st[t]["u"], st[t]["v"], gs[t], bd, n + 1, n + 1,
                dord4=True, grid_type=0)
            oracle.append((ua_n, va_n))

        # JAX side: the reconcile-proven recipe (create face 0 == tile 1).
        base = create_fv3_native_cubed_sphere(
            n, radius=float(constants.R_earth), use_duogrid=True,
            k2e_nord=4, dtype=jnp.float64)
        cd = create_cubed_sphere_cdgrid(base, fv3_native_angles=True)
        ua_h1, va_h1 = d2a2c_ua_va_halo_4d(
            jnp.asarray(u_c[..., None]), jnp.asarray(v_c[..., None]), cd,
            real_metric_ghosts=True, covariant_halo=True,
            local_ghost_d2a=True)
        ua_h1 = np.asarray(ua_h1)[..., 0]
        va_h1 = np.asarray(va_h1)[..., 0]
        assert ua_h1.shape == (6, n + 2, n + 2)

        # ring diagonals are poisoned by contract
        for ii in (0, -1):
            for jj in (0, -1):
                assert np.abs(ua_h1[:, ii, jj]).min() > 1e6
                assert np.abs(va_h1[:, ii, jj]).min() > 1e6

        def map_cell(ic, jc, k):
            if k == 0:
                return ic, jc
            if k == 1:
                return jc, n - 1 - ic
            if k == 2:
                return n - 1 - ic, n - 1 - jc
            return n - 1 - jc, ic

        atol = 1e-5
        n_cmp = 0
        for field, ring in (("ua", ua_h1), ("va", va_h1)):
            for F in range(6):
                g, k = PERM[F], ROT[F]
                sgn = (_TRANSPLANT_SU[k] if field == "ua"
                       else _TRANSPLANT_SV[k])
                # consumed strips: ua S/N ghost rows at physical i;
                # va W/E ghost columns at physical j.
                if field == "ua":
                    cells = ([(i, -1) for i in range(n)]
                             + [(i, n) for i in range(n)])
                else:
                    cells = ([(-1, j) for j in range(n)]
                             + [(n, j) for j in range(n)])
                got = []
                want = []
                for ic, jc in cells:
                    i_n, j_n = map_cell(ic, jc, k)
                    # component swap for odd k (contravariant pair)
                    swap = (k % 2 == 1)
                    src = (1 if (field == "va") != swap else 0)
                    nat = oracle[g][src][ng + i_n, ng + j_n]
                    assert abs(nat) < 1e3, (field, F, ic, jc, nat)
                    want.append(sgn * nat)
                    got.append(ring[F, ic + 1, jc + 1])
                    n_cmp += 1
                np.testing.assert_allclose(
                    np.asarray(got), np.asarray(want), rtol=0.0, atol=atol,
                    err_msg=f"{field} F={F} T={g + 1} k={k}")
        assert n_cmp == 2 * 6 * 2 * n
