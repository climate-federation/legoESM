"""The v-face ZONAL width under ``metric_convention="nemo_isotropic"`` (#1455).

WHAT THIS PINS.  NEMO builds DINO's horizontal metrics in
``cfgs/DINO/MY_SRC/usrdef_hgr.F90``::

    pphiv(ji,jj) = 1./rad * ASIN( TANH( rn_e1_deg *rad* zvj ) )     ! :108
    pe1v (ji,jj) = ra * rad * COS( rad * pphiv(ji,jj) ) * rn_e1_deg ! :113
    pe2v (ji,jj) = ra * rad * COS( rad * pphiv(ji,jj) ) * rn_e1_deg ! :118

with ``zvj = REAL( mjg(jj,0) - nn_jeq_s ) + 0.5`` (:98) -- the Mercator
transform evaluated at the HALF-INTEGER row index, i.e. at the V-point's own
latitude.  ``ra = 6371229 m`` and ``rad = pi/180`` come from ``phycst.F90``
(:26, :37); ``rn_e1_deg = 1`` from ``usrdef_nam.F90:30`` and DINO's
``namelist_cfg``.  Lines :113 and :118 are the SAME expression, so NEMO's mesh
is isotropic and ``e1v == e2v`` to the last bit.

THE DEFECT THIS TEST WOULD HAVE CAUGHT.  legoESM's ``"exact"`` convention
builds the v-face zonal width from the arithmetic mean of the two adjacent
TRACER latitudes.  ``asin(tanh(.))`` is nonlinear, so the midpoint of two
latitudes is not the latitude of the midpoint index, and on DINO the two
disagree by up to 0.0011 degrees -- making this one width 3.3e-05 relatively
TOO LARGE at both walls.  Same formula, same radius, same dlon: a different
LATITUDE, because the midpoint was taken in latitude space instead of in
Mercator index space.  Under ``"nemo_isotropic"`` the width is now evaluated at
the true v-face latitude, which is NEMO's ``gphiv``.

NON-VACUITY.  Every fidelity assertion below is paired with the SAME assertion
under ``metric_convention="exact"``, which must FAIL -- so the test provably
fires when the fix is removed rather than passing for an unrelated reason.

The hand-quoted NEMO values are read off NEMO's own ``domain_cfg_out.nc`` for
the DINO R1 mesh (199 x 52 including its closed walls, equator on a T-point)
at column i=26, and are carried here as literals so this test needs no NEMO
installation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids import create_latlon_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    curl_vertex_cgrid,
    pv_flux_al81_partial_cell,
    strain_rate_cgrid,
    stress_divergence_cgrid,
    vface_zonal_cos_lat,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _split_velocity_divergence,
    flux_form_vface_zonal_length,
)
from legoesm.ocean.experiments.dino import (
    dino_lat_lon_grid,
    nemo_faithful_dino_config,
)

# NEMO's own e1v [m] on the DINO R1 mesh, column i=26, keyed by the legoESM
# v-face index it corresponds to.  The correspondence is established by
# LATITUDE (asserted below), not by an assumed halo offset: NEMO's array
# carries a 2-row halo, so legoESM v-face j is NEMO's ``e1v[j+1]``.
NEMO_E1V_AT_VFACE = {
    3: (-68.9727620197, 39899.4776639535),    # first quoted wet face
    4: (-68.6110140362, 40554.0027022049),
    99: (-0.4999936539, 111194.6894417521),   # the near-equatorial face
    195: (68.6110140362, 40554.0027022049),
    196: (68.9727620197, 39899.4776639535),   # last quoted wet face
}

# The gap the fix removes, measured over the interior v-faces of this mesh.
EXACT_CONVENTION_MAX_REL_GAP = 3.3175e-05


@pytest.fixture(scope="module", autouse=True)
def _fp64_storage():
    """The oracle lane runs fp64 STORAGE, and this comparison needs it: the
    default fp32 policy stores the metric to ~6e-8 relative, which bounds any
    agreement with NEMO at ~1.7e-07 no matter how the width is built.  That is
    still 195x better than the 3.3e-05 construction gap -- the defect is a
    construction error, not a precision one -- but it is far above the bar
    asserted here, so the policy is pinned rather than inherited.
    """
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _dino_geometry(convention: str, vface_evaluation: str = "nemo_vpoint"):
    """The NEMO-faithful DINO R1 grid + its C-grid geometry."""
    # Build both metric arms on the same oracle grid.  Since aa010f143 the
    # faithful grid correctly rejects a card whose own metric convention is
    # changed to ``exact``; that convention belongs only to this A/B geometry.
    g = dino_lat_lon_grid(cfg=nemo_faithful_dino_config())
    geom = create_latlon_geometry(
        n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon,
        lat_face_1d=g.lat_v, radius=g.radius,
        metric_convention=convention,
        vface_zonal_metric_evaluation=vface_evaluation,
    )
    return g, geom


@pytest.fixture(scope="module")
def dino_iso(_fp64_storage):
    # The fp64 fixture is requested EXPLICITLY rather than relied on for
    # autouse ordering: without it the agreement with NEMO is bounded at ~7e-8
    # by the storage dtype and every 1e-12 assertion below fails.
    return _dino_geometry("nemo_isotropic")


@pytest.fixture(scope="module")
def dino_exact(_fp64_storage):
    return _dino_geometry("exact")


@pytest.fixture(scope="module")
def dino_legacy_vface(_fp64_storage):
    return _dino_geometry("nemo_isotropic", "legacy_tracer_midpoint")


class TestConstructionMatchesNemo:
    def test_attribution_selector_moves_only_dx_v(
            self, dino_iso, dino_legacy_vface):
        """The climate counterfactual changes only the diagnosed owner.

        The true faces and every metric other than e1v/dx_v must remain
        bit-identical, while dx_v must reproduce the historical midpoint gap.
        """
        g_new, new = dino_iso
        g_old, old = dino_legacy_vface
        np.testing.assert_array_equal(np.asarray(g_new.lat_v),
                                      np.asarray(g_old.lat_v))
        for name in ("dx_T", "dy_T", "area_T", "dx_u", "dy_u", "dy_v",
                     "area_q", "f_T", "f_u", "f_v"):
            np.testing.assert_array_equal(
                np.asarray(getattr(new, name)), np.asarray(getattr(old, name)),
                err_msg=f"V-face attribution selector unexpectedly moved {name}")
        assert not np.array_equal(np.asarray(new.dx_v), np.asarray(old.dx_v))
        j = 3
        nemo_width = NEMO_E1V_AT_VFACE[j][1]
        legacy_gap = abs(float(old.dx_v[j, 26]) - nemo_width) / nemo_width
        assert legacy_gap > 1e-6

    def test_vface_latitudes_are_nemos_gphiv(self, dino_iso):
        """The correspondence is by LATITUDE, so the width comparison below
        is not resting on an assumed halo offset."""
        g, _ = dino_iso
        lat_v_deg = np.degrees(np.asarray(g.lat_v, dtype=np.float64))
        for j, (nemo_lat_deg, _width) in NEMO_E1V_AT_VFACE.items():
            assert abs(lat_v_deg[j] - nemo_lat_deg) < 1e-9, (
                f"legoESM v-face {j} sits at {lat_v_deg[j]:.10f} deg but "
                f"NEMO's gphiv is {nemo_lat_deg:.10f} -- the rows being "
                f"compared are not the same physical face"
            )

    def test_width_matches_nemo_e1v_at_the_walls(self, dino_iso):
        """The registered fidelity assertion: NEMO's own array, hand-quoted."""
        _, geom = dino_iso
        dx_v = np.asarray(geom.dx_v, dtype=np.float64)
        for j, (_lat, nemo_width) in NEMO_E1V_AT_VFACE.items():
            rel = abs(dx_v[j, 26] - nemo_width) / nemo_width
            assert rel < 1e-12, (
                f"v-face {j}: legoESM {dx_v[j, 26]:.10f} m vs NEMO's e1v "
                f"{nemo_width:.10f} m, relative gap {rel:.4e}"
            )

    def test_exact_convention_still_carries_the_gap(self, dino_exact):
        """NON-VACUITY.  The same assertion under ``"exact"`` must FAIL, and
        fail by the documented 3.3e-05 -- otherwise the test above could be
        passing for a reason unrelated to the construction."""
        _, geom = dino_exact
        dx_v = np.asarray(geom.dx_v, dtype=np.float64)
        gaps = [abs(dx_v[j, 26] - w) / w
                for j, (_lat, w) in NEMO_E1V_AT_VFACE.items()]
        assert max(gaps) > 1e-6, (
            "the 'exact' convention no longer disagrees with NEMO's e1v -- "
            "this test can no longer fail, so it proves nothing"
        )
        # And it is the SIZE the campaign measured, not some other defect.
        assert abs(max(gaps) - EXACT_CONVENTION_MAX_REL_GAP) < 1e-8, (
            f"the 'exact'-convention gap is {max(gaps):.6e}, not the "
            f"{EXACT_CONVENTION_MAX_REL_GAP:.6e} recorded for this mesh"
        )

    def test_whole_interior_not_only_the_quoted_rows(self, dino_iso,
                                                     dino_exact):
        """Five rows agreeing is the reading this campaign has had to retract
        before, so the comparison is made over every interior v-face."""
        g, geom = dino_iso
        _, geom_x = dino_exact
        # NEMO's closed form, rebuilt END TO END from usrdef_hgr.F90 and
        # INDEPENDENTLY of legoESM's own latitudes -- which is the point.  An
        # earlier version of this test built the reference from ``g.lat_v``,
        # so it only checked the ``ra*rad*rn_e1_deg`` product and the
        # degree/radian round-trip; the claim that the two models' LATITUDES
        # agree across the interior still rested on the five quoted rows
        # (adversarial review finding 3).  Now the latitude comes from NEMO's
        # own transform:
        #     zvj   = (j - nn_jeq_s) + 0.5                      (:98)
        #     gphiv = 1/rad * ASIN(TANH(rn_e1_deg*rad*zvj))     (:108)
        #     e1v   = ra*rad*COS(rad*gphiv)*rn_e1_deg           (:113)
        # DINO's equator sits on a T-point, so for a grid of n_lat = 2K+1 rows
        # legoESM's v-face j carries NEMO's half-integer index j - K - 0.5.
        ra, rad, rn_e1_deg = constants.R_earth, np.pi / 180.0, 1.0
        K = (g.n_lat - 1) // 2
        zvj = np.arange(g.n_lat + 1, dtype=np.float64) - K - 0.5
        gphiv_rad = np.arcsin(np.tanh(rn_e1_deg * rad * zvj))
        # legoESM must reproduce NEMO's latitudes, not merely be consistent
        # with itself -- assert that separately so a latitude drift cannot
        # hide inside the width comparison.
        sl = slice(1, g.n_lat)          # interior faces only; ends are walls
        lat_gap = np.abs(
            np.degrees(gphiv_rad[sl])
            - np.degrees(np.asarray(g.lat_v, dtype=np.float64)[sl]))
        assert lat_gap.max() < 1e-9, (
            f"legoESM's v-face latitudes depart from NEMO's own Mercator "
            f"transform by up to {lat_gap.max():.3e} degrees")
        nemo_e1v = ra * rad * np.cos(gphiv_rad) * rn_e1_deg
        rel_iso = np.abs(
            np.asarray(geom.dx_v, np.float64)[sl, 26] - nemo_e1v[sl]
        ) / nemo_e1v[sl]
        rel_exact = np.abs(
            np.asarray(geom_x.dx_v, np.float64)[sl, 26] - nemo_e1v[sl]
        ) / nemo_e1v[sl]
        assert rel_iso.max() < 1e-12, (
            f"interior max relative gap vs NEMO's closed form "
            f"{rel_iso.max():.4e}")
        assert rel_exact.max() > 1e-6, (
            "the 'exact' convention matches NEMO's closed form too -- the "
            "non-vacuity control is dead")

    def test_isotropy_e1v_equals_e2v(self, dino_iso):
        """NEMO's :113 and :118 are one expression, so its mesh has
        ``e1v == e2v`` bit-for-bit.  Reproduce that exactly, not to a ulp."""
        _, geom = dino_iso
        dx_v = np.asarray(geom.dx_v, np.float64)
        dy_v = np.asarray(geom.dy_v, np.float64)
        np.testing.assert_array_equal(
            dx_v[1:-1], dy_v[1:-1],
            err_msg="under nemo_isotropic the two v-face scale factors must "
                    "be the SAME quantity (usrdef_hgr.F90:113 == :118)")

    def test_wall_faces_stay_hard_zeroed(self, dino_iso):
        """The #516 TRANSPORT-metric contract is unchanged by the fix: the
        two END faces remain exactly zero (no meridional flux through the
        closed wall).  Only the interior latitude moved."""
        _, geom = dino_iso
        dx_v = np.asarray(geom.dx_v, np.float64)
        assert np.all(dx_v[0] == 0.0)
        assert np.all(dx_v[-1] == 0.0)
        assert float(dx_v[1:-1].min()) > 0.0


class TestInvariantsSurviveTheCorrectedWidth:
    """THE CONSERVATION QUESTION, answered by measurement rather than by the
    claim that the invariants are value-independent.

    Each gate below is the #516 gate from
    ``test_vface_metric_consistency_mercator.py``, re-run on the CORRECTED
    ``nemo_isotropic`` geometry.  They hold because every operator SHARES one
    stored width -- and the fix moved that one width, not one operator's copy
    of it.
    """

    def test_strain_stress_adjoint_pair(self, dino_iso):
        _, geom = dino_iso
        ku, kt = jax.random.split(jax.random.PRNGKey(0))
        u = jax.random.normal(ku, (geom.n_lat, geom.n_lon + 1),
                              dtype=jnp.float64)
        v = jax.random.normal(kt, (geom.n_lat + 1, geom.n_lon),
                              dtype=jnp.float64)
        D_T, D_S = strain_rate_cgrid(u, v, geom)
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        stress_h = jax.random.normal(k1, D_T.shape, dtype=jnp.float64)
        stress_q = jax.random.normal(k2, D_S.shape, dtype=jnp.float64)
        lhs = float(jnp.sum(D_T * stress_h) + jnp.sum(D_S * stress_q))
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, geom, normalize=False)
        rhs = float(jnp.sum(u * tend_u) + jnp.sum(v * tend_v))
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
        assert rel < 1e-11, (
            f"strain/stress adjointness broken by the corrected v-face "
            f"width: lhs={lhs:.6e} rhs={rhs:.6e} rel={rel:.3e}")

    def test_divergence_advection_mass_consistency(self, dino_iso):
        _, geom = dino_iso
        dx_v_adv = flux_form_vface_zonal_length(geom)
        dx_v_div = geom.radius * vface_zonal_cos_lat(geom) * geom.dlon
        err = float(jnp.max(jnp.abs(dx_v_adv - dx_v_div)))
        assert err < 1e-9, (
            f"the flux-form advection and the continuity divergence no "
            f"longer share one v-face width: max diff {err:.3e} m")

    def test_wdivergence_reads_the_corrected_width(self, dino_iso):
        """And the width they share is the CORRECTED one -- a shared but
        stale metric would pass the consistency gate above while leaving the
        defect in place."""
        _, geom = dino_iso
        nlev = 2
        u = jnp.zeros((geom.n_lat, geom.n_lon + 1, nlev), dtype=jnp.float64)
        v = jnp.ones((geom.n_lat + 1, geom.n_lon, nlev), dtype=jnp.float64)
        _, dV_dj = _split_velocity_divergence(u, v, geom)
        area = jnp.asarray(geom.area, dtype=jnp.float64)[:, :, None]
        dx_v = jnp.asarray(geom.dx_v, dtype=jnp.float64)[:, 0]
        expected = (dx_v[1:] - dx_v[:-1])[:, None, None]
        err = float(jnp.max(jnp.abs(dV_dj * area - expected)))
        assert err < 1e-9, (
            f"the velocity-divergence split is not transporting on the "
            f"stored (corrected) v-face width: max diff {err:.3e}")


    def test_een_metric_weighting_still_conserves_the_physical_ke_norm(
            self, dino_iso):
        """The DINO oracle card ships ``een_metric_weighting="nemo"``, whose
        whole justification is that it conserves the PHYSICAL kinetic-energy
        norm (``e1*e2*h``-weighted) rather than the per-area one.  That
        identity is the one place a v-face WIDTH could plausibly be
        load-bearing by value, so it is measured on the corrected width
        instead of being argued away: the vorticity flux must still do zero
        net work.

        It holds for the same reason the #516 gates do -- the width appears in
        the transport weight and again in the energy norm, and telescopes --
        but "it should telescope" is exactly the class of claim this campaign
        keeps having to retract.
        """
        _, geom = dino_iso
        n_lat, n_lon = geom.n_lat, geom.n_lon
        rng = np.random.default_rng(11)
        u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, 1)) * 0.1)
        v = np.asarray(rng.standard_normal((n_lat + 1, n_lon, 1)) * 0.1)
        v[0] = 0.0                                   # closed south wall
        v[-1] = 0.0                                  # closed north wall
        v = jnp.asarray(v)
        u = u.at[:, -1, :].set(u[:, 0, :])           # periodic in longitude
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
            f"the metric-weighted vorticity flux does net work "
            f"{work:.6e} on a physical KE of {ke:.6e} (relative {rel:.3e}) "
            f"with the corrected v-face width -- the energy identity the "
            f"DINO card's EEN weighting is shipped for does NOT survive it")


    def test_the_model_reads_the_stored_width_not_a_recompute(self, dino_iso):
        """THE GATE THIS FIX MADE LOAD-BEARING (adversarial review finding 2).

        ``reads_stored_vface_metric`` decides whether an operator reads the
        stored width or recomputes it from cell latitudes.  Before this fix the
        two agreed, so the gate was pure redundancy and its own docstring said
        so.  They now differ by 3.3e-05, and the gate is the only thing keeping
        a single model from carrying both constructions at once.  Nothing in
        the repo asserted it, so a one-line regression could silently route the
        model onto the recompute and no test would notice.
        """
        from legoesm.grids.operators_latlon_cgrid import (
            reads_stored_vface_metric,
        )
        _, geom = dino_iso
        assert reads_stored_vface_metric(geom), (
            "the DINO geometry no longer takes the stored-width branch -- the "
            "operators would recompute the tracer-midpoint width while direct "
            "readers of grid.dx_v keep NEMO's, splitting the metric")
        # And the width they read really is the stored, corrected one.
        cos_v = np.asarray(vface_zonal_cos_lat(geom), dtype=np.float64)
        stored = (np.asarray(geom.dx_v, dtype=np.float64)[:, 0]
                  / (float(geom.radius) * float(geom.dlon)))
        np.testing.assert_allclose(cos_v, stored, rtol=0, atol=1e-15)
        # NEGATIVE CONTROL: a lean grid with no stored 2-D width must NOT take
        # that branch, or the assertion above is true of everything.
        from legoesm.grids.latlon import create_latlon_grid
        assert not reads_stored_vface_metric(create_latlon_grid(12, 24)), (
            "a lean LatLonGrid takes the stored branch too -- this gate no "
            "longer discriminates and the positive assertion is vacuous")

    def test_flag_is_inert_on_a_uniform_grid(self, _fp64_storage):
        """The convention exists to fix a NONLINEAR meridional coordinate.  On
        a uniform-dlat grid the midpoint of two tracer latitudes IS the face
        latitude, so there is nothing to correct and the flag must not move
        the width by so much as a rounding bit (adversarial review finding 1:
        it briefly did, by ~7e-07 at fp32 storage)."""
        from legoesm.grids.latlon import create_latlon_grid
        g = create_latlon_grid(36, 72)
        kw = dict(n_lat=g.n_lat, n_lon=g.n_lon, lat_1d=g.lat, lon_1d=g.lon)
        a = create_latlon_geometry(metric_convention="exact", **kw)
        b = create_latlon_geometry(metric_convention="nemo_isotropic", **kw)
        np.testing.assert_array_equal(
            np.asarray(a.dx_v), np.asarray(b.dx_v),
            err_msg="metric_convention moved the v-face zonal width on a "
                    "UNIFORM grid, where the two constructions are "
                    "mathematically identical")
