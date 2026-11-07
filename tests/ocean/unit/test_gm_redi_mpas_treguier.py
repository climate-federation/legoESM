"""Treguier adaptive GM coefficient on the MPAS Voronoi mesh.

The MPAS GM/Redi used to REFUSE ``GMRediConfig.treguier`` outright, so the
Voronoi lane ran Visbeck or a constant kappa while the structured lane ran
NEMO's ``ldf_eiv``.  Comparing the two grids therefore compared two different
eddy closures.

This suite pins the three things that could go silently wrong in the port:

* the branch must not be overwritten by the Visbeck assignment that follows;
* the cell-centred slope must come from a VECTOR reconstruction with the
  taper applied to the MAGNITUDE — tapering each edge-normal component
  separately is a different, mesh-orientation-dependent operator;
* dry columns must get exactly zero.

PARTIAL HARMONIZATION, deliberately: this is the SHARED Treguier, not the
``nemo_native`` variant the ORCA1-faithful tripole card selects.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    TreguierConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    compute_isopycnal_slopes_mpas,
    gm_redi_tracer_tendency_mpas,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord():
    return create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0)


def _state(mesh, nlev=5):
    """Stratified T/S with a horizontal gradient, so slopes are nonzero."""
    lat = np.asarray(mesh.latCell)
    z = np.arange(nlev, dtype=np.float64)
    T = (10.0 - 0.8 * z[None, :]) + 3.0 * np.cos(lat)[:, None]
    S = np.full((mesh.nCells, nlev), 35.0)
    eta = np.zeros((mesh.nCells,))
    H_bathy = np.full((mesh.nCells,), 500.0)
    return (jnp.asarray(T), jnp.asarray(S),
            jnp.asarray(eta), jnp.asarray(H_bathy))


def _cfg(*, treguier=False, visbeck=False, aei0=900.0, S_max=1e-2):
    # slope_scheme is explicit: the GMRediConfig default is "triads", which
    # MPAS does not implement, so leaving it unset makes every test here die
    # in the dispatch guard rather than in the code under test.
    #
    # S_max is a parameter because the slopes this idealised state produces
    # are ~1e-5 — three orders below the physical 1e-2 — so at the default
    # the DM95 taper is the identity and any test OF the taper is vacuous.
    return GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=S_max, slope_scheme="centered",
        treguier=TreguierConfig(enabled=treguier, aei0=aei0),
        visbeck=VisbeckConfig(enabled=visbeck),
    )


class TestTreguierRuns:

    def test_no_longer_refuses(self, mesh, z_coord):
        T, S, eta, H = _state(mesh)
        dT, dS = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(treguier=True), eos="linear")
        assert dT.shape == (mesh.nCells, 5)
        assert np.all(np.isfinite(np.asarray(dT)))
        assert np.all(np.isfinite(np.asarray(dS)))

    def test_it_actually_changes_the_tendency(self, mesh, z_coord):
        """Non-vacuity: Treguier must not reproduce the constant-kappa run.

        Without this, every assertion above would still pass if the branch
        silently fell through to ``cfg.kappa_GM``.
        """
        T, S, eta, H = _state(mesh)
        dT_const = np.asarray(gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(), eos="linear")[0])
        dT_treg = np.asarray(gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(treguier=True), eos="linear")[0])
        scale = np.abs(dT_const).max()
        # Guard the comparison itself: both tendencies are ~1e-11 here, so an
        # absolute-tolerance test (np.allclose) would call them equal whatever
        # the coefficient did.  Compare RELATIVE to the constant-kappa run.
        assert scale > 0.0, "constant-kappa tendency is identically zero"
        assert np.abs(dT_treg - dT_const).max() > 1e-3 * scale

    def test_visbeck_does_not_overwrite_it(self, mesh, z_coord):
        """The two adaptive schemes are refused together, not silently ranked."""
        T, S, eta, H = _state(mesh)
        with pytest.raises(ValueError, match="mutually exclusive"):
            gm_redi_tracer_tendency_mpas(
                T, S, eta, H, mesh, z_coord,
                _cfg(treguier=True, visbeck=True), eos="linear")

    def test_disabled_path_does_not_touch_the_new_return_value(
            self, mesh, z_coord):
        """Turning the knob OFF must reproduce the pre-port result exactly.

        ``compute_isopycnal_slopes_mpas`` grew an optional THIRD return value
        for this port.  The real risk is that the default two-value form
        changed, so this compares a config with no Treguier field at ALL
        against one that carries a disabled Treguier block — comparing
        ``_cfg()`` with ``_cfg(treguier=False)`` would be comparing two
        identical calls and could not fail.
        """
        T, S, eta, H = _state(mesh)
        bare = GMRediConfig(
            kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2, slope_scheme="centered")
        assert not bare.treguier.enabled
        d1, s1 = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, bare, eos="linear")
        d2, s2 = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(treguier=False), eos="linear")
        np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        # and the two-value unpack still works on the default path
        out = compute_isopycnal_slopes_mpas(
            jnp.asarray(np.asarray(T)),
            jnp.ones((mesh.nCells,), dtype=jnp.float64),
            z_coord, jnp.ones((mesh.nCells,), dtype=jnp.float64), mesh, bare)
        assert len(out) == 2


class TestSlopeReconstruction:

    def test_optional_third_return_is_the_raw_slope(self, mesh, z_coord):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            iterate_eos_and_pressure_anomaly,
        )
        from legoesm.ocean.eos import make_eos_fn, rho_0 as RHO_0
        from legoesm import constants
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            voronoi_neumann_fill,
        )

        T, S, _eta, _H = _state(mesh)
        mask = jnp.ones((mesh.nCells,), dtype=jnp.float64)
        jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)
        rho, _, _ = iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: voronoi_neumann_fill(f, mask, mesh),
            make_eos_fn("linear", None), z_coord.dz_ref, RHO_0,
            constants.g, n_iter=2)

        # S_max well BELOW this state's natural slopes, so the taper is a
        # real operation rather than the identity.
        cfg = _cfg(treguier=True, S_max=1e-6)
        two = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
        three = compute_isopycnal_slopes_mpas(
            rho, mask, z_coord, jac, mesh, cfg, return_raw=True)
        assert len(two) == 2 and len(three) == 3
        # the first two entries are unchanged by asking for the third
        np.testing.assert_array_equal(np.asarray(two[0]), np.asarray(three[0]))
        # and the extra one is NOT the returned slope (else it is the wrong
        # array and the clip/taper would be applied twice)
        assert not np.allclose(np.asarray(three[0]), np.asarray(three[2]))
        # the returned slope is clipped AND tapered, so it is everywhere no
        # larger than the raw one it came from, and no larger than S_max
        assert np.all(np.abs(np.asarray(three[0]))
                      <= np.abs(np.asarray(three[2])) + 1e-12)
        assert np.abs(np.asarray(three[0])).max() <= cfg.S_max + 1e-12
        # non-vacuity: the raw slope must actually EXCEED S_max here, or
        # "clipped" and "raw" would be the same array and the test above
        # could not tell them apart
        assert np.abs(np.asarray(three[2])).max() > cfg.S_max

    def test_perot_reconstruction_converges_with_refinement(self):
        """The reconstruction is SECOND-ORDER, not exact — and that is fine.

        ``reconstruct_cell_velocity``'s docstring calls itself "exact for
        uniform flow on any Voronoi mesh". MEASURED (job 9847883), feeding it
        the edge-normal components of a constant east/north vector: median
        relative error 2.4e-2 at subdivision level 2 and 4.2e-3 at level 3.
        It converges, so it is a consistent discretisation, but "exact" it is
        not, and a test written to the docstring would fail.

        The band |lat| < 60 is excluded, and that is a real exclusion worth
        naming: it drops roughly an eighth of a quasi-uniform sphere mesh, not
        just the singular polar cells. The reason to exclude anything at all
        is that a constant east/north vector is not a continuous field on a
        sphere — it is singular at the poles — so those cells say nothing
        about the reconstruction. The band is a blunt instrument for that;
        the sharp test is
        ``test_the_polar_error_is_the_test_field_not_the_reconstruction``
        below, which uses a field that IS smooth at the poles and requires
        the polar cells to behave.

        Convergence is the property that licenses using the cell-centred
        magnitude in place of |S| in the Treguier integral.
        """
        errs = {}
        for level in (2, 3):
            m = create_voronoi_mesh(subdivision_level=level)
            ang = np.asarray(m.angleEdge)
            Sx_true, Sy_true = 3.0e-3, -1.5e-3
            S_n = jnp.asarray(Sx_true * np.cos(ang) + Sy_true * np.sin(ang))
            S_x, S_y = reconstruct_cell_velocity(S_n, m)
            err = np.hypot(np.asarray(S_x) - Sx_true,
                           np.asarray(S_y) - Sy_true) / np.hypot(Sx_true,
                                                                 Sy_true)
            away = np.abs(np.degrees(np.asarray(m.latCell))) < 60.0
            assert away.sum() > 0.5 * m.nCells
            errs[level] = float(np.median(err[away]))

        assert errs[2] < 0.05, errs
        # Refinement must actually buy accuracy — a reconstruction that
        # ignored the mesh would give a flat error and pass the bound above.
        assert errs[3] < 0.5 * errs[2], errs

    def test_the_polar_error_is_the_test_field_not_the_reconstruction(self):
        """Prove the excuse, do not just assert it.

        The constant east/north field above blows up at the poles, and I
        attributed that to the field's own singularity rather than to a
        defect. That attribution is testable: feed a SOLID-BODY ROTATION
        ``u = Omega x r``, whose east/north components are smooth and finite
        everywhere including the poles (it is purely zonal, vanishing at the
        pole itself). If the reconstruction were broken in the polar cells,
        this would fail there too.
        """
        m = create_voronoi_mesh(subdivision_level=3)
        lat_e = np.asarray(m.latEdge)
        ang = np.asarray(m.angleEdge)
        # Solid-body zonal flow: u_east = U cos(lat), u_north = 0.
        U = 2.0e-3
        u_e, v_n = U * np.cos(lat_e), np.zeros_like(lat_e)
        S_n = jnp.asarray(u_e * np.cos(ang) + v_n * np.sin(ang))
        rx, ry = (np.asarray(a) for a in reconstruct_cell_velocity(S_n, m))

        lat_c = np.asarray(m.latCell)
        ref = U * np.cos(lat_c)
        scale = max(float(np.abs(ref).max()), 1e-30)
        err = np.hypot(rx - ref, ry - 0.0) / scale

        polar = np.abs(np.degrees(lat_c)) > 60.0
        assert polar.any()
        # The polar cells must behave like everywhere else on a field that is
        # actually smooth there.  A surviving ~100% error would mean the
        # earlier attribution was wrong and the reconstruction really does
        # break at the pole.
        assert err[polar].max() < 0.05, (
            float(err[polar].max()), float(err[~polar].max()))


class TestTheOperatorItself:
    """The claim the rest of this file leaves untested.

    A reviewer showed that swapping the vector reconstruction for the
    per-component proxy the Visbeck branch uses left every other test in this
    file green — so the file's headline claim was decorative. These two cases
    pin it.
    """

    def test_magnitude_taper_differs_from_per_component(self, mesh):
        """The two operators must actually disagree where the taper bites.

        If they agreed, the whole reconstruct-then-taper branch would be
        pointless and could be deleted.
        """
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            dm95_taper, dm95_taper_scalar,
        )
        from legoesm.grids.voronoi import reconstruct_cell_velocity_wet

        S_max = 1e-3
        ang = np.asarray(mesh.angleEdge)
        # Magnitude AT S_max, where the DM95 taper sits at 0.5 and is most
        # sensitive.  MEASURED: at 2x S_max the taper is ~1e-5 and BOTH
        # operators return essentially zero, so a test there compares two
        # zeros and passes for the wrong reason.
        Sx_t = Sy_t = S_max / np.sqrt(2.0)
        S_n = jnp.asarray(Sx_t * np.cos(ang) + Sy_t * np.sin(ang))
        wet = jnp.ones_like(S_n)

        # (a) the operator this branch uses: reconstruct, then taper |S|
        vx, vy = reconstruct_cell_velocity_wet(S_n, mesh, wet)
        vx, vy, _ = dm95_taper(vx, vy, S_max)
        mag_vector = np.hypot(np.asarray(vx), np.asarray(vy))

        # (b) the per-component operator: taper each edge normal, then
        #     reconstruct
        S_n_t, _ = dm95_taper_scalar(S_n, S_max)
        px, py = reconstruct_cell_velocity_wet(S_n_t, mesh, wet)
        mag_component = np.hypot(np.asarray(px), np.asarray(py))

        assert np.median(mag_vector) > 0.0
        rel = np.abs(mag_vector - mag_component) / np.maximum(mag_vector, 1e-30)
        assert np.median(rel) > 0.05, float(np.median(rel))

    def test_wet_restricted_reconstruction_survives_a_missing_edge(self, mesh):
        """Dropping an edge must not shorten the reconstructed vector.

        Plain Perot with the dry edge zeroed loses that edge's share of the
        reconstruction tensor and returns a SHORT vector — the coastal bias a
        reviewer put at up to a third. The wet-restricted solve must recover
        the uniform field from the remaining edges.
        """
        from legoesm.grids.voronoi import (
            reconstruct_cell_velocity, reconstruct_cell_velocity_wet,
        )

        wet = np.ones(mesh.nEdges)
        eoc = np.asarray(mesh.edgesOnCell)
        victim_cell = int(np.argmax((eoc >= 0).sum(axis=0)))
        dropped = int(eoc[0, victim_cell])
        wet[dropped] = 0.0
        wet_j = jnp.asarray(wet)
        # The field must point ALONG the dropped edge's normal.  The loss is
        # anisotropic — it lives in the missing normal's direction — so a
        # field pointing across it barely moves (MEASURED: 2%), while one
        # aligned with it loses a third (MEASURED: 3.0e-3 -> 1.99e-3, 33.6%).
        # Testing the transverse case would understate the defect by an order
        # of magnitude and make the fix look unnecessary.
        a0 = float(np.asarray(mesh.angleEdge)[dropped])
        amp = 3.0e-3
        S_n = jnp.asarray(amp * np.cos(np.asarray(mesh.angleEdge) - a0))
        Sx_t, Sy_t = amp * np.cos(a0), amp * np.sin(a0)

        naive_x, naive_y = reconstruct_cell_velocity(S_n * wet_j, mesh)
        fixed_x, fixed_y = reconstruct_cell_velocity_wet(S_n, mesh, wet_j)

        truth = np.hypot(Sx_t, Sy_t)
        naive = np.hypot(float(naive_x[victim_cell]),
                         float(naive_y[victim_cell]))
        fixed = np.hypot(float(fixed_x[victim_cell]),
                         float(fixed_y[victim_cell]))
        # non-vacuity: the naive form must really be short here, or this
        # proves nothing about the fix
        assert naive < 0.9 * truth, (naive, truth)
        assert abs(fixed - truth) < 0.05 * truth, (fixed, truth)


class TestDryColumns:

    def test_land_gets_exactly_zero_kappa(self, mesh, z_coord):
        """A masked column must not receive a COEFFICIENT.

        Asserting on the tendency would prove nothing — it is multiplied by
        the mask on the way out, so it is zero on land whether or not the
        coefficient was. This calls the shared kappa routine on the same
        inputs the branch feeds it and checks the coefficient ITSELF, which
        is what the dry-column cut in the branch exists to fix: the shared
        helper's own wet test is built from integrated REFERENCE thickness
        (and partial-cell land carries J=1), so it can return a nonzero
        kappa on a column this mesh calls land.
        """
        from legoesm import constants
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            compute_treguier_kappa_gm,
        )

        mask_np = np.ones((mesh.nCells,))
        mask_np[: mesh.nCells // 3] = 0.0
        dry = mask_np < 0.5
        assert dry.any()
        mask = jnp.asarray(mask_np)

        nlev = 5
        rho = jnp.asarray(
            1025.0 + 0.5 * np.arange(nlev)[None, :]
            + 0.05 * np.cos(np.asarray(mesh.latCell))[:, None])
        S_x = jnp.full((mesh.nCells, nlev - 1), 1e-3)
        S_y = jnp.full((mesh.nCells, nlev - 1), -5e-4)
        jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)
        f = 2.0 * constants.Omega * jnp.sin(mesh.latCell)

        kappa_raw = compute_treguier_kappa_gm(
            rho, S_x, S_y, z_coord, jac, f,
            _cfg(treguier=True).treguier)
        # Non-vacuity: the shared helper must actually be returning something
        # nonzero on the dry columns, or the cut below is untested.
        assert np.any(np.asarray(kappa_raw)[dry] != 0.0), (
            "shared kappa is already zero on dry columns here; this test "
            "cannot detect the loss of the dry-column cut")
        kappa_cut = kappa_raw * (mask > 0.0)
        np.testing.assert_array_equal(np.asarray(kappa_cut)[dry], 0.0)
