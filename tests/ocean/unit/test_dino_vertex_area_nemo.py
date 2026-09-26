"""The vertex (F-cell) AREA under ``metric_convention="nemo_isotropic"`` (#1455).

WHAT THIS PINS.  NEMO builds DINO's F-point metrics in
``cfgs/DINO/MY_SRC/usrdef_hgr.F90``::

    zfj          = REAL( mjg(jj,0) - nn_jeq_s, wp ) + 0.5            ! :97
    pphif(ji,jj) = 1./rad * ASIN( TANH( rn_e1_deg *rad* zfj ) )      ! :109
    pe1f (ji,jj) = ra * rad * COS( rad * pphif(ji,jj) ) * rn_e1_deg  ! :114
    pe2f (ji,jj) = ra * rad * COS( rad * pphif(ji,jj) ) * rn_e1_deg  ! :118

and every operator that needs an F-cell area divides by the PRODUCT
``e1f*e2f``.  ``zfj`` and ``zvj`` (:96) carry the same ``+0.5`` row offset, so
``pphif == pphiv`` and ``e1f*e2f == e1v*e2v`` -- verified exactly on NEMO's own
dumped mesh, and the reason this fix reuses the corrected v-face width instead
of re-deriving a latitude.

THE DEFECT THIS TEST WOULD HAVE CAUGHT.  legoESM's ``"exact"`` convention
builds the vertex area as the true spherical cap between the two adjacent
TRACER latitudes, ``R^2*dlon*|sin(phi[j]) - sin(phi[j-1])|``.  On the Mercator
coordinate ``sin(phi) = tanh(dlon*j)`` gives ``d(sin phi)/dj = dlon*cos^2(phi)``,
so that cap is ``R^2*dlon^2`` times the EXACT INTERVAL INTEGRAL of ``cos^2(phi)``
while NEMO's product is ``R^2*dlon^2`` times its MIDPOINT value.  Exact
quadrature versus the midpoint rule on ``sech^2``, giving a relative gap

    (dlon^2 / 12) * (3 sin^2(phi_f) - 1)

which is +1.17e-05 median and 4.16e-05 max on this mesh and changes SIGN at
|phi| = 35.26 deg.  It is the same class as the v-face width defect -- a
midpoint taken in the wrong space -- with the twist that here legoESM's value
is the MORE accurate one and the oracle contract is to adopt NEMO's quadrature,
exactly as this convention already adopts NEMO's ``pe1t = pe2t``.

NON-VACUITY.  Every fidelity assertion below is paired with the SAME assertion
under ``metric_convention="exact"``, which must FAIL -- so the test provably
fires when the fix is removed rather than passing for an unrelated reason.

The hand-quoted NEMO values are read off NEMO's own mesh for the DINO R1 mesh
(195 x 48, equator on a T-point) at column i=26 and carried here as literals,
so this test needs no NEMO installation.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids import create_latlon_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    _vertex_dual_area_interior,
    curl_vertex_cgrid,
    pv_flux_al81_partial_cell,
    vertex_area_cgrid,
)
from legoesm.ocean.experiments.dino import (
    dino_lat_lon_grid,
    nemo_faithful_dino_config,
)

# NEMO's own ``e1f*e2f`` [m^2] on the DINO R1 mesh, column i=26, keyed by the
# legoESM vertex-row index it corresponds to.  NOTE these are all INTERIOR
# rows and all take NEMO's product; the wall rows the fix deliberately leaves
# on the spherical cap are 0 and n_lat, which NEMO's mesh does not carry.
# The correspondence is
# established by LATITUDE (asserted below), not by an assumed halo offset:
# NEMO's array carries a 2-row halo, so legoESM vertex row j is NEMO's row
# ``j+1``.  ``e1f`` at these rows equals the ``e1v`` this module's sibling
# (``test_dino_vface_zonal_width_nemo.py``) quotes, to the last digit -- that
# is the ``pphif == pphiv`` identity above, visible in the literals.
NEMO_E1E2F_AT_VERTEX = {
    1: (-68.9727620197, 1591968317.8563280106),    # southernmost INTERIOR row
    2: (-68.6110140362, 1644627135.1704397202),
    97: (-0.4999936539, 12364258960.0476951599),   # the near-equatorial row
    193: (68.6110140362, 1644627135.1704397202),
    194: (68.9727620197, 1591968317.8563280106),   # northernmost INTERIOR row
}

# The gap the fix removes, measured over the interior vertex rows of this mesh.
EXACT_CONVENTION_MAX_REL_GAP = 4.0965e-05

# NEMO's own earth radius and degree->radian factor (phycst.F90 :26, :37) and
# DINO's rn_e1_deg (usrdef_nam.F90:30 / namelist_cfg).  Quoted as NEMO's
# values, not legoESM's, because this module reproduces NEMO's mesh.
_RA, _RAD, _RN_E1_DEG = constants.R_earth, np.pi / 180.0, 1.0


@pytest.fixture(scope="module", autouse=True)
def _fp64_storage():
    """The oracle lane runs fp64 STORAGE and this comparison needs it: the
    default fp32 policy stores the metric to ~6e-8 relative, which bounds any
    agreement with NEMO far above the bars asserted here.  Pinned rather than
    inherited, exactly as the sibling width module does.
    """
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _dino_geometry(convention: str):
    """The NEMO-faithful DINO R1 grid + its C-grid geometry."""
    cfg = dataclasses.replace(
        nemo_faithful_dino_config(), metric_convention=convention)
    g = dino_lat_lon_grid(cfg=cfg)
    geom = create_latlon_geometry(
        n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon,
        lat_face_1d=g.lat_v, radius=g.radius,
        metric_convention=convention,
    )
    return g, geom


def _nemo_e1e2f(n_lat: int) -> np.ndarray:
    """NEMO's ``e1f*e2f`` rebuilt END TO END from ``usrdef_hgr.F90``, and
    INDEPENDENTLY of legoESM's own latitudes -- which is the point.  DINO's
    equator sits on a T-point, so for a grid of ``n_lat = 2K+1`` rows legoESM's
    vertex row ``j`` carries NEMO's half-integer index ``j - K - 0.5``.
    """
    K = (n_lat - 1) // 2
    zfj = np.arange(n_lat + 1, dtype=np.float64) - K - 0.5          # :97
    gphif = np.arcsin(np.tanh(_RN_E1_DEG * _RAD * zfj))             # :109
    e1f = _RA * _RAD * np.cos(gphif) * _RN_E1_DEG                   # :114
    return e1f * e1f                                                # :114 * :118


@pytest.fixture(scope="module")
def dino_iso(_fp64_storage):
    return _dino_geometry("nemo_isotropic")


@pytest.fixture(scope="module")
def dino_exact(_fp64_storage):
    return _dino_geometry("exact")


class TestConstructionMatchesNemo:
    def test_area_at_hand_quoted_nemo_points(self, dino_iso):
        """Pinned against NEMO's own array at five rows, matched by LATITUDE
        so the comparison is not resting on an assumed halo offset."""
        g, geom = dino_iso
        lat_v_deg = np.degrees(np.asarray(g.lat_v, dtype=np.float64))
        area_q = np.asarray(geom.area_q, dtype=np.float64)
        for j, (nemo_lat_deg, nemo_area) in NEMO_E1E2F_AT_VERTEX.items():
            assert abs(lat_v_deg[j] - nemo_lat_deg) < 1e-9, (
                f"row {j}: legoESM vertex latitude {lat_v_deg[j]:.10f} is not "
                f"NEMO's gphif {nemo_lat_deg:.10f} -- the rows being compared "
                f"are not the same rows")
            rel = abs(area_q[j, 26] - nemo_area) / nemo_area
            assert rel < 1e-12, (
                f"row {j} (lat {nemo_lat_deg:.4f}): vertex area "
                f"{area_q[j, 26]:.6f} vs NEMO's e1f*e2f {nemo_area:.6f} "
                f"-- relative {rel:.4e}")

    def test_whole_interior_not_only_the_quoted_rows(self, dino_iso,
                                                    dino_exact):
        """Five rows agreeing is the reading this campaign has had to retract
        before, so the comparison is made over every interior vertex row --
        against NEMO's transform rebuilt independently of legoESM's grid."""
        g, geom = dino_iso
        _, geom_x = dino_exact
        nemo = _nemo_e1e2f(g.n_lat)
        sl = slice(1, g.n_lat)          # interior rows; ends are wall rows
        rel_iso = np.abs(
            np.asarray(geom.area_q, np.float64)[sl, 26] - nemo[sl]) / nemo[sl]
        rel_exact = np.abs(
            np.asarray(geom_x.area_q, np.float64)[sl, 26] - nemo[sl]) / nemo[sl]
        assert rel_iso.max() < 1e-12, (
            f"interior max relative gap vs NEMO's closed form "
            f"{rel_iso.max():.4e}")
        assert rel_exact.max() > 1e-6, (
            "the 'exact' convention matches NEMO's closed form too -- the "
            "non-vacuity control is dead")
        # Pinned RELATIVELY (1%): the absolute size tracks the grid's radius
        # and row count, and an absolute bar here would go red on a change
        # that has nothing to do with this fix.
        assert abs(rel_exact.max() / EXACT_CONVENTION_MAX_REL_GAP - 1.0) < 1e-2, (
            f"the gap the fix removes moved: {rel_exact.max():.6e} vs the "
            f"pinned {EXACT_CONVENTION_MAX_REL_GAP:.6e}")

    def test_gap_is_the_midpoint_rule_not_something_else(self, dino_exact):
        """THE NAMED DIFF, asserted rather than asserted-about.  If the gap is
        the exact-quadrature-vs-midpoint-rule error claimed in this module's
        docstring, it must equal ``(dlon^2/12)(3 sin^2(phi_f) - 1)`` -- SIGNED,
        including the sign change at |phi| = 35.26 deg, which no scale-factor
        or radius error could reproduce.
        """
        g, geom_x = dino_exact
        nemo = _nemo_e1e2f(g.n_lat)
        sl = slice(1, g.n_lat)
        signed = (np.asarray(geom_x.area_q, np.float64)[sl, 26]
                  - nemo[sl]) / nemo[sl]
        lat_v = np.asarray(g.lat_v, dtype=np.float64)[sl]
        a = _RN_E1_DEG * _RAD
        predicted = (a ** 2 / 12.0) * (3.0 * np.sin(lat_v) ** 2 - 1.0)
        ratio = signed / predicted
        assert abs(np.median(ratio) - 1.0) < 1e-3, (
            f"the measured gap does not decompose into the midpoint-rule "
            f"prediction: median ratio {np.median(ratio):.6f}")
        # The sign change is the discriminating feature; a uniform relative
        # error (wrong radius, wrong dlon) cannot produce one.
        assert signed.min() < 0.0 < signed.max(), (
            "the gap does not change sign across the domain -- it is not the "
            "midpoint-rule error this fix is named for")
        crossing = np.degrees(np.arcsin(np.sqrt(1.0 / 3.0)))
        zero_lat = lat_v[np.argmin(np.abs(signed))]
        assert abs(abs(np.degrees(zero_lat)) - crossing) < 1.0, (
            f"the sign change sits at {np.degrees(zero_lat):.2f} deg, not at "
            f"the predicted +-{crossing:.2f} deg")

    def test_area_is_the_square_of_the_corrected_vface_width(self, dino_iso):
        """``pphif == pphiv`` (usrdef_hgr.F90 :97 vs :96), so NEMO's F-cell
        area IS the square of its v-face width.  Reproduce that exactly -- it
        is what licenses building this from ``cos_lat_v`` rather than from a
        separately derived F-point latitude.
        """
        g, geom = dino_iso
        sl = slice(1, g.n_lat)
        area = np.asarray(geom.area_q, np.float64)[sl, :g.n_lon]
        dx_v = np.asarray(geom.dx_v, np.float64)[sl]
        dy_v = np.asarray(geom.dy_v, np.float64)[sl]
        np.testing.assert_allclose(area, dx_v * dy_v, rtol=1e-15, atol=0.0)

    def test_wall_rows_keep_the_exact_cap(self, dino_iso, dino_exact):
        """The two end rows span pole-to-first-tracer-row, are fully masked,
        and sit under ``curl_vertex_cgrid``'s positivity guard.  They are
        deliberately NOT moved, so a future reader does not read their value
        as a NEMO claim."""
        _, geom = dino_iso
        _, geom_x = dino_exact
        area = np.asarray(geom.area_q, np.float64)
        area_x = np.asarray(geom_x.area_q, np.float64)
        np.testing.assert_array_equal(area[0], area_x[0])
        np.testing.assert_array_equal(area[-1], area_x[-1])
        assert area[0].min() > 0.0 and area[-1].min() > 0.0, (
            "a wall row went non-positive -- the curl's divide guard would "
            "now be load-bearing")

    def test_flag_is_inert_on_a_uniform_grid(self, _fp64_storage):
        """On a UNIFORM-dlat grid the two constructions are NOT equal (the cap
        is ``2 R^2 dlon cos(phi_v) sin(dlat/2)``, the product is
        ``R^2 dlon^2 cos^2(phi_v)``), so the flag must be GATED OFF there --
        it reproduces a Mercator mesh, not any lat-lon mesh.  Every non-DINO
        card must stay byte-identical.
        """
        from legoesm.grids.latlon import create_latlon_grid
        g = create_latlon_grid(36, 72)
        kw = dict(n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon)
        a = create_latlon_geometry(metric_convention="exact", **kw)
        b = create_latlon_geometry(metric_convention="nemo_isotropic", **kw)
        np.testing.assert_array_equal(
            np.asarray(a.area_q), np.asarray(b.area_q),
            err_msg="metric_convention moved the vertex area on a UNIFORM "
                    "grid, where NEMO's product form is not the mesh being "
                    "reproduced")
        # SCOPED: this asserts area_q ONLY.  The flag is NOT inert as a whole
        # on a uniform grid -- ``dy_v``'s override at latlon.py is gated on the
        # convention WITHOUT the variable-dlat condition, and on this very grid
        # ``dy_v`` moves by ~100%, ``dy_T`` by ~96%, ``area_T`` by ~25% and the
        # total area by ~21%.  That is pre-existing and out of scope here;
        # claiming the flag as a whole is inert would be the kind of scope word
        # this campaign has got wrong before.


def _random_closed_basin_flow(g, seed):
    """Zonally periodic, meridionally closed: no flow through either wall."""
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((g.n_lat, g.n_lon + 1)) * 0.1)
    v = np.asarray(rng.standard_normal((g.n_lat + 1, g.n_lon)) * 0.1)
    v[0] = 0.0
    v[-1] = 0.0
    return u.at[:, -1].set(u[:, 0]), jnp.asarray(v)


class TestConservationIdentities:
    """Measured on the CORRECTED area, not argued away.  Every one of these is
    the class of claim this campaign keeps having to retract."""

    def test_vertex_area_helper_returns_the_curls_own_divisor(self, dino_iso):
        """THE CONTRACT THIS FIX MADE LOAD-BEARING.  ``vertex_area_cgrid``'s
        docstring says it returns "the area by which ``curl_vertex_cgrid``
        divides the circulation", so that ``A_vertex * zeta_vertex`` IS a
        circulation and the rigid-lid island line integrals are line
        integrals.  It previously RECOMPUTED the spherical cap on the regular
        lat-lon path while the curl read the stored array; the two agreed to
        dtype roundoff while both were caps, and would now disagree by
        2.2e-05.  Pinned as an equality against the array the curl reads
        (``operators_latlon_cgrid.py:1197``), with a control below.

        Stated as what it is -- a consistency pin between two helpers, not a
        conservation law.  The vertex area is a pure DIVISOR in every active
        DINO consumer, so no identity constrains its value; what is
        constrained is that every consumer sees the SAME value.
        """
        _, geom = dino_iso
        helper = np.asarray(vertex_area_cgrid(geom), np.float64)
        curl_divisor = np.abs(np.asarray(geom.area_q, np.float64)[:, 0])
        np.testing.assert_array_equal(helper[1:-1, 0], curl_divisor[1:-1])
        assert (helper[0] == 0.0).all() and (helper[-1] == 0.0).all(), (
            "the helper's wall-row zeroing was lost")

    def test_the_stale_recompute_would_have_failed_that(self, dino_iso):
        """The control: the spherical cap the helper used to rebuild must
        DIFFER from the array the curl reads, or the pin above is vacuous."""
        _, geom = dino_iso
        curl_divisor = np.abs(np.asarray(geom.area_q, np.float64)[:, 0])
        sin_lat = np.sin(np.asarray(geom.lat, np.float64))
        sin_ext = np.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
        stale = (float(geom.radius) ** 2 * float(geom.dlon)
                 * np.abs(sin_ext[1:] - sin_ext[:-1]))
        rel = np.abs(stale[1:-1] - curl_divisor[1:-1]) / curl_divisor[1:-1]
        assert rel.max() > 1e-6, (
            "the stale spherical-cap recompute still equals the curl's "
            "divisor -- the pin above cannot fail, so it is not evidence")

    def test_the_curl_actually_moves_by_the_predicted_amount(self, dino_iso):
        """Behavioural, not structural: the fix must reach the OPERATOR.

        ONE VARIABLE.  The A/B is NOT ``nemo_isotropic`` against ``exact`` --
        those two geometries differ in the v-face WIDTH as well (the #1455
        width fix), so the circulation itself moves and nothing about the AREA
        would be isolated.  It is the corrected geometry against itself with
        ONLY ``area_q`` swapped back to the spherical cap.

        Two statements, because either alone is weak.  (a) The CIRCULATION
        ``A*zeta`` is invariant across the swap -- both runs divide the same
        circulation by their own stored area, which is only true if the curl
        reads that stored array rather than recomputing one.  (b) The
        vorticity itself moves by exactly the midpoint-rule gap.  Scored where
        ``|zeta|`` is above a tenth of its own RMS: a random flow crosses
        zero, and a RATIO at a near-zero denominator measures roundoff.
        """
        g, geom = dino_iso
        sin_lat = np.sin(np.asarray(geom.lat, np.float64))
        sin_ext = np.pad(sin_lat, (1, 1), constant_values=(-1.0, 1.0))
        cap = (float(geom.radius) ** 2 * float(geom.dlon)
               * np.abs(sin_ext[1:] - sin_ext[:-1]))
        geom_cap = geom._replace(
            area_q=jnp.asarray(cap[:, None]
                               * np.ones((1, g.n_lon + 1))))
        u, v = _random_closed_basin_flow(g, 3)
        z_iso = np.asarray(curl_vertex_cgrid(u, v, geom), np.float64)
        z_cap = np.asarray(curl_vertex_cgrid(u, v, geom_cap), np.float64)
        sl = slice(1, g.n_lat)
        A_iso = np.asarray(geom.area_q, np.float64)[sl, :1]
        A_cap = cap[sl, None]
        circ_iso = z_iso[sl, :-1] * A_iso
        circ_cap = z_cap[sl, :-1] * A_cap
        scale = np.abs(circ_iso).max()
        assert np.abs(circ_iso - circ_cap).max() / scale < 1e-13, (
            "the circulation is NOT invariant across an area-only swap -- "
            "the curl is not dividing the same circulation by its own stored "
            "area, so this fix does not reach the operator")
        lat_v = np.asarray(g.lat_v, np.float64)[sl]
        a = _RN_E1_DEG * _RAD
        gap = (a ** 2 / 12.0) * (3.0 * np.sin(lat_v) ** 2 - 1.0)
        predicted = (-gap / (1.0 + gap))[:, None]
        zi = z_iso[sl, :-1]
        big = np.abs(zi) > 0.1 * np.sqrt(np.mean(zi ** 2))
        ratio = (z_cap[sl, :-1] / zi) - 1.0
        dev = np.abs(ratio - np.broadcast_to(predicted, ratio.shape))[big]
        assert dev.max() < 1e-9, (
            f"the curl's response to the fix is not the predicted "
            f"midpoint-rule gap: max departure {dev.max():.4e} over "
            f"{int(big.sum())} cells")

    def test_een_vorticity_flux_still_does_zero_net_work(self, dino_iso):
        """The DINO card ships ``een_metric_weighting="nemo"``, whose whole
        justification is that it conserves the PHYSICAL kinetic-energy norm.
        The vertex area sets ``zeta`` and therefore ``q``; the identity is
        argued to be q-INDEPENDENT (every paired (u,v) contribution shares one
        vertex value), so the corrected area must not disturb it.

        SCOPED after review: this is a REGRESSION GUARD, not evidence for the
        fix.  Because the identity is q-independent it holds for ANY vertex
        array -- it would pass on a random one -- so it cannot discriminate
        the corrected area from the old one.  What it does catch is a future
        change that breaks the triad's pairing structure.
        """
        g, geom = dino_iso
        n_lat, n_lon = geom.n_lat, geom.n_lon
        u2, v2 = _random_closed_basin_flow(g, 11)
        u = u2[..., None]
        v = v2[..., None]
        h_u = jnp.full((n_lat, n_lon + 1, 1), 4000.0)
        h_v = jnp.concatenate(
            [jnp.zeros((1, n_lon, 1)),
             jnp.full((n_lat - 1, n_lon, 1), 4000.0),
             jnp.zeros((1, n_lon, 1))], axis=0)
        h_vtx = jnp.full((n_lat + 1, n_lon + 1, 1), 4000.0)
        u_mask = jnp.ones_like(h_u)
        v_mask = jnp.concatenate(
            [jnp.zeros((1, n_lon, 1)), jnp.ones((n_lat - 1, n_lon, 1)),
             jnp.zeros((1, n_lon, 1))], axis=0)
        vtx_mask = jnp.ones((n_lat + 1, n_lon + 1))
        zeta = curl_vertex_cgrid(u, v, geom)
        widths = (geom.dx_u, geom.dx_v, geom.dy_u, geom.dy_v)
        du, dv = pv_flux_al81_partial_cell(
            zeta, h_vtx, h_v, v, h_u, u, u_mask, v_mask, vtx_mask,
            metric_widths=widths)
        wu = (geom.dx_u * geom.dy_u)[..., None]
        wv = (geom.dx_v * geom.dy_v)[..., None]
        work = float(
            jnp.sum((wu * h_u * u * du * u_mask)[:, :-1])
            + jnp.sum(wv * h_v * v * dv * v_mask))
        ke = float(
            jnp.sum((wu * h_u * u ** 2 * u_mask)[:, :-1])
            + jnp.sum(wv * h_v * v ** 2 * v_mask))
        rel = abs(work) / ke
        assert rel < 1e-14, (
            f"the metric-weighted vorticity flux does net work {work:.6e} on "
            f"a physical KE of {ke:.6e} (relative {rel:.3e}) with the "
            f"corrected vertex area")


class TestVertexAreaHelperOnOtherGrids:
    """The helper fix is not DINO-only, and the second grid it moves is the
    one where it was silently broken (adversarial review MINOR 12)."""

    def test_cartesian_beta_plane_no_longer_gets_a_zero_vertex_area(self):
        """On the Cartesian beta-plane the spherical-cap recompute differences
        a pseudo-latitude that is not the Cartesian spacing, so it returned an
        all-but-zero vertex area -- which fed the rigid-lid diagonal and CG.
        The stored ``dx*dy`` is the correct quantity there.
        """
        from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
        dx_m = dy_m = 1.0e4
        g = create_beta_plane_cgrid_geometry(
            n_lat=20, n_lon=20, dx_m=dx_m, dy_m=dy_m, f0=1e-4, beta=2e-11)
        A = np.asarray(vertex_area_cgrid(g), np.float64)
        np.testing.assert_allclose(A[1:-1], dx_m * dy_m, rtol=1e-12)
        assert (A[0] == 0.0).all() and (A[-1] == 0.0).all()
        # The control: the recompute this replaced does NOT give dx*dy here.
        stale = np.asarray(
            _vertex_dual_area_interior(g.lat, g.radius, g.dlon), np.float64)
        assert abs(stale[5] / (dx_m * dy_m) - 1.0) > 1e-8, (
            "the spherical-cap recompute already equalled dx*dy on the "
            "beta-plane -- this test cannot fail")

    def test_helper_does_not_promote_the_storage_dtype(self):
        """It becomes ``RigidLidStaticData.A_vertex`` and enters the CG solve,
        so a silent fp32 -> fp64 promotion would widen that solve.  The stored
        ``area_q`` is built in fp64 regardless of policy, hence the cast."""
        prev = get_policy()
        set_policy(PrecisionPolicy.fp32())
        try:
            g = create_latlon_geometry(n_lat=24, n_lon=48)
            recompute = _vertex_dual_area_interior(g.lat, g.radius, g.dlon)
            assert vertex_area_cgrid(g).dtype == recompute.dtype, (
                "vertex_area_cgrid changed its return dtype when it moved to "
                "the stored array")
        finally:
            set_policy(prev)


class TestPairAnalysisPins:
    """PINS ON THE PAIR ANALYSIS'S NUMBERS -- not tests of this diff.

    STATED PLAINLY because the class name previously implied otherwise
    (adversarial review MAJOR 6): neither method below touches ``area_q``,
    ``vertex_area_cgrid`` or ``curl_vertex_cgrid``, and both pass with the
    entire fix removed.  They are closed-form arithmetic plus two medians
    carried over from NEMO's restart, pinned so that a grid change cannot
    silently move the numbers the campaign documents quote.  The MEASUREMENT
    itself lives in ``scripts/validate/ocean_fidelity/dino_1226/
    vertex_area_pair_analysis.py``, which reads the restart; this is its
    receipt, not its replacement.

    The vertex area and the vertex Coriolis are the SAME defect at the SAME
    point (a midpoint taken in latitude space rather than in Mercator index
    space), and they are the two halves of one property: whether the discrete
    curl of solid-body rotation equals the discrete ``f``.  So the fix is only
    safe to ship ALONE if the area half is negligible where the two meet.
    """

    def test_the_two_halves_have_the_predicted_analytic_forms(self, dino_iso):
        """Both halves in closed form, so the comparison below is between two
        known functions of latitude and not between two fitted numbers."""
        g, _ = dino_iso
        lat_v = np.asarray(g.lat_v, np.float64)[1:g.n_lat]
        a = _RN_E1_DEG * _RAD
        area_half = (a ** 2 / 12.0) * (3.0 * np.sin(lat_v) ** 2 - 1.0)
        cori_half = -(a ** 2 / 4.0) * np.cos(lat_v) ** 2
        # Sizes quoted in the campaign documents; pinned so a grid change
        # cannot silently move what those documents claim.  ROW SET MATTERS:
        # these are the model's own 194 interior rows.  The probe scores the
        # same closed forms over NEMO's dumped mesh, whose 2-row halo reaches
        # past the domain, and reads +1.1748e-05 / -3.9022e-05 there.  Same
        # quantity, wider rows -- both appear in the documents and neither is
        # wrong, but they must not be quoted as though they were one number.
        assert abs(np.median(area_half) - 1.0791e-05) < 1e-8
        assert abs(np.median(cori_half) + 3.9978e-05) < 1e-8
        # And the size the campaign documents quote for the UNSIGNED gap.
        assert abs(np.median(np.abs(area_half)) - 2.1872e-05) < 1e-8

    def test_area_half_is_negligible_in_the_absolute_vorticity_channel(
            self, dino_iso):
        """PRE-REGISTERED BAR: the two halves form a cancelling pair requiring
        a JOINT fix only if the area half is within a factor of 10 of the
        Coriolis half in the channel where they meet -- the F-point absolute
        vorticity ``zeta + f`` that EEN's ``q`` is built from.  The area half
        scales with ``zeta``, the Coriolis half with ``f``, and in a
        rotation-dominated basin those differ by orders of magnitude.

        Scored here on the ``|zeta|`` and ``|f|`` medians measured on NEMO's
        own day-5760 DINO restart (2.05e-08 against 1.02e-04), so the receipt
        travels without needing the restart on disk.  The bar is missed by
        three orders, which is the finding -- this assertion documents a
        settled margin, it is not a live gate.
        """
        g, _ = dino_iso
        lat_v = np.asarray(g.lat_v, np.float64)[1:g.n_lat]
        a = _RN_E1_DEG * _RAD
        d_area = (a ** 2 / 12.0) * (3.0 * np.sin(lat_v) ** 2 - 1.0)
        d_cori = -(a ** 2 / 4.0) * np.cos(lat_v) ** 2
        zeta_med, f_med = 2.0513e-08, 1.0184e-04       # NEMO restart medians
        ratio = (np.median(np.abs(d_area * zeta_med))
                 / np.median(np.abs(d_cori * f_med)))
        assert ratio < 0.1, (
            f"the vertex-area half is {ratio:.4e} of the Coriolis half in the "
            f"absolute-vorticity channel -- at or above the 0.1 bar, so the "
            f"two ARE a cancelling pair and this fix must not ship alone")
