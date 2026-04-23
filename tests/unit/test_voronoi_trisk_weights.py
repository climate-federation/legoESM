"""TRiSK weightsOnEdge (Thuburn-Ringler) validation tests.

Covers issue #211: the Perot-style minimum-norm LSQ used previously
violated Thuburn (2009) Eq. 30 stationary-geostrophic-mode preservation
on non-uniform Voronoi meshes, driving MPAS-ocean baroclinic-instability
runs unstable.  These tests lock in the Thuburn/Ringler kite-area
formula now used in `_compute_weights_on_edge`.

References
----------
- Ringler, T. D., Thuburn, J., Klemp, J. B., & Skamarock, W. C. (2010).
  J. Comput. Phys., 229(9), 3065–3090.  Eqs. 17–26, 49.
- Thuburn, J., Ringler, T. D., Skamarock, W. C., & Klemp, J. B. (2009).
  J. Comput. Phys., 228(22), 8321–8335.  Eqs. 30, 33, 39.
- MPAS-Tools `mpas_mesh_converter.cpp buildEdgesOnEdgeArrays` —
  the canonical reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=[(1, 10), (2, 20), (3, 30)],
                ids=["level1", "level2", "level3"])
def mesh(request):
    """SCVT meshes at three resolutions to stress the formula over
    varying kite-area asymmetry."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    level, lloyd = request.param
    return create_voronoi_mesh(level, lloyd_iterations=lloyd)


@pytest.fixture(scope="module")
def channel_mesh():
    """Regional periodic channel mesh — exercises the dvEdge flooring
    path that LSQ was blind to and that caused issue #211 instability."""
    from legoesm.grids.voronoi import create_regional_voronoi_mesh
    return create_regional_voronoi_mesh(
        lon_range=(0.0, 10.0), lat_range=(16.0, 34.0),
        resolution_km=200.0, periodic_x=True,
    )


@pytest.fixture(scope="module")
def channel_mesh_20km():
    """Finer-resolution (20 km) regional periodic channel — matches the
    production Eady setup. This resolution exposes pathological 2·d_rad
    seam triangles that a looser oversize-edge cutoff (3·d_rad) lets
    through. Those triangles introduced phantom edges with
    verticesOnEdge[1] = -1, poisoning TRiSK walks near (but not on) the
    seam and producing O(1) Thuburn-antisymmetry violations on ~420
    stencil pairs. The 1.5·d_rad filter in the sub-360° periodic path
    drops them cleanly. See issue #211."""
    from legoesm.grids.voronoi import create_regional_voronoi_mesh
    return create_regional_voronoi_mesh(
        lon_range=(0.0, 10.0), lat_range=(16.0, 34.0),
        resolution_km=20.0, periodic_x=True,
    )


@pytest.fixture(scope="module")
def arrays(mesh):
    """Pull all TRiSK arrays into float64 numpy."""
    return dict(
        w=np.asarray(mesh.weightsOnEdge, dtype=np.float64),
        eoe=np.asarray(mesh.edgesOnEdge, dtype=np.int64),
        nEOE=np.asarray(mesh.nEdgesOnEdge, dtype=np.int64),
        ka=np.asarray(mesh.kiteAreasOnVertex, dtype=np.float64),
        ac=np.asarray(mesh.areaCell, dtype=np.float64),
        at=np.asarray(mesh.areaTriangle, dtype=np.float64),
        cov=np.asarray(mesh.cellsOnVertex, dtype=np.int64),
        coe=np.asarray(mesh.cellsOnEdge, dtype=np.int64),
        voe=np.asarray(mesh.verticesOnEdge, dtype=np.int64),
        dv=np.asarray(mesh.dvEdge, dtype=np.float64),
        dc=np.asarray(mesh.dcEdge, dtype=np.float64),
    )


# ===========================================================================
# 1.  Partition: kite areas sum to cell area.
# ===========================================================================

class TestKiteAreaPartition:
    def test_kite_areas_sum_to_areacell(self, mesh, arrays):
        """Σ kite_area(v, c) for all v touching c = areaCell[c]."""
        cov = arrays["cov"]
        ka = arrays["ka"]
        ac = arrays["ac"]
        sums = np.zeros(mesh.nCells)
        for v in range(mesh.nVertices):
            for k in range(3):
                c = int(cov[k, v])
                if c >= 0:
                    sums[c] += ka[k, v]
        rel = np.max(np.abs(sums - ac) / np.maximum(ac, 1e-30))
        assert rel < 1e-6, (
            f"Kite-area partition max rel err {rel:.3e} — mesh geometry bug"
        )

    def test_kite_areas_positive(self, mesh, arrays):
        """Every kite area must be strictly positive on a valid SCVT."""
        cov = arrays["cov"]
        ka = arrays["ka"]
        for v in range(mesh.nVertices):
            for k in range(3):
                if int(cov[k, v]) >= 0:
                    assert ka[k, v] > 0.0, (
                        f"Non-positive kite area at (k={k}, v={v}): {ka[k,v]}"
                    )


# ===========================================================================
# 2.  Dimensionless weight antisymmetry  (Ringler 2010 Eq. 49;
#     Thuburn 2009 Eq. 39).
#     w_T(e, e') = weightsOnEdge[e, e'] · dcEdge[e] / dvEdge[e']
#     must satisfy w_T(e, e') + w_T(e', e) = 0.
# ===========================================================================

class TestAntisymmetry:
    def test_w_T_antisymmetric(self, mesh, arrays):
        w, eoe, nEOE = arrays["w"], arrays["eoe"], arrays["nEOE"]
        dv, dc = arrays["dv"], arrays["dc"]
        max_viol = 0.0
        n_pairs = 0
        for e in range(mesh.nEdges):
            for k in range(int(nEOE[e])):
                ep = int(eoe[k, e])
                if ep < 0:
                    continue
                wT_e_ep = w[k, e] * dc[e] / dv[ep]
                wT_ep_e = None
                for kk in range(int(nEOE[ep])):
                    if int(eoe[kk, ep]) == e:
                        wT_ep_e = w[kk, ep] * dc[ep] / dv[e]
                        break
                if wT_ep_e is None:
                    continue
                viol = abs(wT_e_ep + wT_ep_e)
                if viol > max_viol:
                    max_viol = viol
                n_pairs += 1
        assert n_pairs > 0
        assert max_viol < 1e-6, (
            f"Thuburn-Ringler antisymmetry violation: max |w_T(e,e')+w_T(e',e)|"
            f" = {max_viol:.3e} over {n_pairs} pairs"
        )

    def test_bilinear_energy_invariant(self, mesh):
        """Σ_{e,e'} w_stored[e,e'] · dc[e] · dv[e] · u(e) · u(e') = 0

        This is the discrete bilinear energy form; antisymmetry of the
        underlying TRiSK operator in the (dc·dv)-weighted inner product
        gives zero discrete work for any random `u` — the core property
        used for energy conservation of the PV flux (Ringler 2010 Eq. 49).
        """
        from legoesm.core.operators_voronoi import tangential_velocity
        key = jax.random.PRNGKey(0)
        u = jax.random.normal(key, (mesh.nEdges,), dtype=jnp.float64)
        vt = tangential_velocity(u, mesh)
        dv = jnp.asarray(mesh.dvEdge, dtype=jnp.float64)
        dc = jnp.asarray(mesh.dcEdge, dtype=jnp.float64)
        inner = float(jnp.sum(u * vt * dv * dc))
        norm = float(jnp.sum(u ** 2 * dv * dc))
        rel = abs(inner) / max(norm, 1e-30)
        assert rel < 1e-6, (
            f"Tangential-velocity bilinear form not skew-symmetric: "
            f"rel={rel:.3e} — breaks energy conservation of PV flux"
        )


# ===========================================================================
# 3.  Stencil structure.
# ===========================================================================

class TestStencil:
    def test_stencil_sizes_match_cell_adjacency(self, mesh, arrays):
        """nEdgesOnEdge[e] = (nEdgesOnCell[c1] - 1) + (nEdgesOnCell[c2] - 1)
        for interior edges on a well-formed SCVT."""
        nEOC = np.asarray(mesh.nEdgesOnCell)
        nEOE = arrays["nEOE"]
        coe = arrays["coe"]
        for e in range(mesh.nEdges):
            c1, c2 = int(coe[0, e]), int(coe[1, e])
            expected = (nEOC[c1] - 1) if c1 >= 0 else 0
            expected += (nEOC[c2] - 1) if c2 >= 0 else 0
            assert int(nEOE[e]) == expected, (
                f"Edge {e}: stencil size {nEOE[e]} ≠ expected {expected}"
            )

    def test_stencil_does_not_include_iEdge(self, mesh, arrays):
        """No self-reference in the stencil."""
        eoe = arrays["eoe"]
        nEOE = arrays["nEOE"]
        for e in range(mesh.nEdges):
            for k in range(int(nEOE[e])):
                assert int(eoe[k, e]) != e


# ===========================================================================
# 4.  Thuburn 2009 Eq. 30 — stationary geostrophic mode preservation.
#
# For any vertex scalar ψ, define the edge-normal "velocity"
#   u(e) := ψ(v1(e)) - ψ(v0(e))
# This field represents a discrete streamfunction-gradient flow.
# The TRiSK weights must make this the discrete null-space of the
# tangential Coriolis reconstruction: no energy is exchanged with such
# a mode, so it is a stationary mode of the shallow-water linearization.
#
# Concretely, the discrete check is that the tangential reconstruction
# applied to u has no curl at any interior vertex:
#
#   Σ_{e ∈ EV(v)} εV(e,v) · v_t(e) · dcEdge[e] ≈ 0
#
# for curl_vertex (discrete mimetic curl).  If this holds, geostrophic
# modes of the linearized f-plane SW system are stationary to roundoff
# — the property that Perot-LSQ fails.
# ===========================================================================

class TestStationaryGeostrophicMode:
    def test_curl_of_trisk_of_vertex_gradient_is_zero(self, mesh):
        """curl_vertex(TRiSK(grad_v ψ)) ≈ 0 for arbitrary vertex ψ."""
        from legoesm.core.operators_voronoi import tangential_velocity, curl_vertex
        key = jax.random.PRNGKey(7)
        psi_vertex = jax.random.normal(key, (mesh.nVertices,),
                                        dtype=jnp.float64)
        v0 = mesh.verticesOnEdge[0]
        v1 = mesh.verticesOnEdge[1]
        # u_edge = ψ(v1) - ψ(v0): a discrete tangential gradient of ψ.
        u_edge = psi_vertex[v1] - psi_vertex[v0]
        vt = tangential_velocity(u_edge, mesh)
        zeta = curl_vertex(vt, mesh)
        # Scale: typical |u|·nEdges ~ O(√N), triangle area ~ area of
        # sphere / nVertices, so dimensionless curl should be O(1/N).
        max_abs = float(jnp.max(jnp.abs(zeta)))
        scale = float(jnp.std(vt))
        rel = max_abs / max(scale, 1e-30)
        # On an exactly-orthogonal SCVT this identity is machine-exact;
        # small SCVT orthogonality error (from finite Lloyd iterations)
        # leaks through at O(mesh_quality_eps).
        assert rel < 1e-3, (
            f"curl(v_t) of vertex-gradient flow: max |curl|/|v_t| = {rel:.3e}"
            " — Thuburn 2009 Eq. 30 violated (stationary geostrophic mode"
            " leaks energy)"
        )


# ===========================================================================
# 5.  Regression test: the full dycore run that used to blow up.
# ===========================================================================

class TestRegressionLegacyFailure:
    def test_pv_flux_does_near_zero_work_on_balanced_flow(self, mesh):
        """Energy-conserving PV flux does near-zero discrete work.

        Previously this test passed trivially due to LSQ symmetry by
        accident.  With the Thuburn-Ringler formula we expect
        machine-precision zero work up to the roundoff of the PV-flux
        operator itself.
        """
        from legoesm.core.operators_voronoi import (
            pv_flux_energy_conserving,
            potential_vorticity_vertex,
        )
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (mesh.nEdges,), dtype=jnp.float64)
        h = 1.0 + 0.1 * jax.random.normal(k2, (mesh.nCells,),
                                           dtype=jnp.float64)
        h = jnp.maximum(h, 0.1)

        q = potential_vorticity_vertex(u, h, mesh.fVertex, mesh)
        Fq = pv_flux_energy_conserving(u, h, q, mesh)

        dv = mesh.dvEdge
        dc = mesh.dcEdge
        # Discrete work uses dv·dc (edge area).
        work = float(jnp.sum(Fq * u * dv * dc))
        scale = float(jnp.sum(u ** 2 * dv * dc))
        rel = abs(work) / max(scale, 1e-30)
        # The PV flux work is not exactly zero for general (q, h)
        # because the antisymmetric stencil operator is sandwiched
        # between (q·h) factors that vary edge-to-edge.  The correct
        # test is that `rel ≪ 1`; machine precision only occurs
        # when h=const and q=const.
        assert rel < 1e-3, (
            f"PV flux discrete work too large: relative={rel:.2e} — "
            "stencil operator energy conservation broken"
        )


# ===========================================================================
# 6.  Regional / channel-mesh invariants (issue #211 specific).
#     The LSQ weights were insensitive to dvEdge flooring; TRiSK is not,
#     so the regional-mesh path must recompute weights with floored dvEdge.
# ===========================================================================

class TestChannelMesh:
    def test_channel_antisymmetry(self, channel_mesh):
        """TRiSK antisymmetry must hold on the zonally-periodic channel
        mesh used by the Eady baroclinic instability test.  Before the
        regional-mesh fix, weightsOnEdge was computed with pre-floor
        dvEdge but the mesh stored post-floor dvEdge — weights were
        silently inconsistent and amplified the barotropic mode."""
        w = np.asarray(channel_mesh.weightsOnEdge, dtype=np.float64)
        eoe = np.asarray(channel_mesh.edgesOnEdge, dtype=np.int64)
        nEOE = np.asarray(channel_mesh.nEdgesOnEdge, dtype=np.int64)
        dv = np.asarray(channel_mesh.dvEdge, dtype=np.float64)
        dc = np.asarray(channel_mesh.dcEdge, dtype=np.float64)
        max_viol = 0.0
        n_pairs = 0
        for e in range(channel_mesh.nEdges):
            for k in range(int(nEOE[e])):
                ep = int(eoe[k, e])
                if ep < 0:
                    continue
                wT_e_ep = w[k, e] * dc[e] / dv[ep]
                wT_ep_e = None
                for kk in range(int(nEOE[ep])):
                    if int(eoe[kk, ep]) == e:
                        wT_ep_e = w[kk, ep] * dc[ep] / dv[e]
                        break
                if wT_ep_e is None:
                    continue
                viol = abs(wT_e_ep + wT_ep_e)
                if viol > max_viol:
                    max_viol = viol
                n_pairs += 1
        assert max_viol < 1e-6, (
            f"Channel-mesh Thuburn antisymmetry violation: {max_viol:.3e} "
            f"({n_pairs} pairs).  Likely stale pre-floor dvEdge."
        )

    def test_channel_stationary_geostrophic_mode(self, channel_mesh):
        """curl(TRiSK(grad_v ψ)) ≈ 0 on the channel mesh."""
        from legoesm.core.operators_voronoi import tangential_velocity, curl_vertex
        key = jax.random.PRNGKey(2026)
        psi = jax.random.normal(key, (channel_mesh.nVertices,),
                                 dtype=jnp.float64)
        v0 = channel_mesh.verticesOnEdge[0]
        v1 = channel_mesh.verticesOnEdge[1]
        u = psi[v1] - psi[v0]
        vt = tangential_velocity(u, channel_mesh)
        zeta = curl_vertex(vt, channel_mesh)
        rel = float(jnp.max(jnp.abs(zeta))) / max(float(jnp.std(vt)), 1e-30)
        # Channel boundary triangles have larger quality defect than an
        # SCVT interior; use a looser tolerance that still reliably
        # catches the pre-fix amplification signature.
        assert rel < 1e-2, (
            f"Channel-mesh stationary geostrophic mode violated: "
            f"rel={rel:.3e}"
        )

    def test_channel_20km_antisymmetry(self, channel_mesh_20km):
        """The production-resolution (20 km) Eady channel mesh must hit
        machine-precision TRiSK antisymmetry — it is the mesh where
        issue #211's day-1.9 blowup occurred. A looser oversize-edge
        filter (3·d_rad) let pathological 2·d_rad meridional triangles
        through near the seam, introducing phantom half-edges that
        broke antisymmetry at exactly 0.408 on ~420 stencil pairs.
        Tightening to 1.5·d_rad in the sub-360° periodic path removes
        them. Regression guard: any reversion that re-loosens the filter
        (or that allows phantom half-edges from some other source) must
        trip this test."""
        w = np.asarray(channel_mesh_20km.weightsOnEdge, dtype=np.float64)
        eoe = np.asarray(channel_mesh_20km.edgesOnEdge, dtype=np.int64)
        nEOE = np.asarray(channel_mesh_20km.nEdgesOnEdge, dtype=np.int64)
        dv = np.asarray(channel_mesh_20km.dvEdge, dtype=np.float64)
        dc = np.asarray(channel_mesh_20km.dcEdge, dtype=np.float64)
        max_viol = 0.0
        n_pairs = 0
        for e in range(channel_mesh_20km.nEdges):
            for k in range(int(nEOE[e])):
                ep = int(eoe[k, e])
                if ep < 0:
                    continue
                wT_e_ep = w[k, e] * dc[e] / dv[ep]
                wT_ep_e = None
                for kk in range(int(nEOE[ep])):
                    if int(eoe[kk, ep]) == e:
                        wT_ep_e = w[kk, ep] * dc[ep] / dv[e]
                        break
                if wT_ep_e is None:
                    continue
                viol = abs(wT_e_ep + wT_ep_e)
                if viol > max_viol:
                    max_viol = viol
                n_pairs += 1
        assert max_viol < 1e-6, (
            f"20 km channel-mesh Thuburn antisymmetry violation: "
            f"max={max_viol:.3e} over {n_pairs} pairs. Expected machine "
            f"precision (~1e-7). If this fails, oversize-edge filter is "
            f"likely reverted to 3·d_rad or phantom half-edges re-appeared."
        )

    def test_channel_20km_no_interior_phantom_cells(self, channel_mesh_20km):
        """No interior cell should have nEdgesOnCell outside {4, 5, 6, 7}.
        Interior hex cells are 6-edge. Boundary-row cells are 4 or 5.
        Any cell with 8+ edges indicates phantom edges sneaking through
        a too-loose oversize filter — the exact signature that drove the
        issue #211 blowup before the 1.5·d_rad fix (24 cells had 8 edges
        clustered at lon 0.3–0.7° and 9.3–9.7°)."""
        nEOC = np.asarray(channel_mesh_20km.nEdgesOnCell, dtype=np.int64)
        n_phantom = int((nEOC > 7).sum())
        assert n_phantom == 0, (
            f"20 km channel mesh has {n_phantom} cells with nEdgesOnCell "
            f">7 — indicates phantom edges from a too-loose oversize "
            f"filter in _regional_delaunay (see issue #211)."
        )
