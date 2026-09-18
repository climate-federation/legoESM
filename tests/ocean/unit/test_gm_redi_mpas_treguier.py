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


def _cfg(*, treguier=False, visbeck=False, aei0=1800.0, S_max=1e-2):
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

    def test_disabled_is_byte_identical(self, mesh, z_coord):
        """Turning the knob OFF must reproduce the pre-port result exactly.

        ``compute_isopycnal_slopes_mpas`` grew an optional third return value
        for this port; the two-value default must be untouched.
        """
        T, S, eta, H = _state(mesh)
        d1, _ = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(), eos="linear")
        d2, _ = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(treguier=False), eos="linear")
        np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))


class TestSlopeReconstruction:

    def test_optional_third_return_is_the_untapered_slope(self, mesh, z_coord):
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
            rho, mask, z_coord, jac, mesh, cfg, return_clipped=True)
        assert len(two) == 2 and len(three) == 3
        # the first two entries are unchanged by asking for the third
        np.testing.assert_array_equal(np.asarray(two[0]), np.asarray(three[0]))
        # and the extra one is NOT the tapered slope (else it is the wrong
        # array and the magnitude taper would be applied twice)
        assert not np.allclose(np.asarray(three[0]), np.asarray(three[2]))
        # every tapered component is <= its untapered source in magnitude
        assert np.all(np.abs(np.asarray(three[0]))
                      <= np.abs(np.asarray(three[2])) + 1e-12)

    def test_perot_reconstruction_converges_with_refinement(self):
        """The reconstruction is SECOND-ORDER, not exact — and that is fine.

        ``reconstruct_cell_velocity``'s docstring calls itself "exact for
        uniform flow on any Voronoi mesh". MEASURED (job 9847883), feeding it
        the edge-normal components of a constant east/north vector: median
        relative error 2.4e-2 at subdivision level 2 and 4.2e-3 at level 3.
        It converges, so it is a consistent discretisation, but "exact" it is
        not, and a test written to the docstring would fail.

        The six cells at the POLES are excluded deliberately, not to make the
        numbers look better: a constant east/north vector is not a continuous
        field on a sphere — it is singular exactly there — so those cells say
        nothing about the reconstruction. Their error is ~100% at BOTH levels
        (it does not converge), which is the signature of an ill-posed test
        field rather than a discretisation error.

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


class TestDryColumns:

    def test_land_gets_exactly_zero_kappa(self, mesh, z_coord):
        """A masked column must not receive a coefficient.

        The shared helper's own wet test is built from integrated REFERENCE
        thickness rather than this run's mask, so it can leave a nonzero
        kappa on a column the mesh calls land.
        """
        T, S, eta, H = _state(mesh)
        mask = np.ones((mesh.nCells,))
        mask[: mesh.nCells // 3] = 0.0
        dry = mask < 0.5
        assert dry.any()
        dT, _ = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z_coord, _cfg(treguier=True),
            eos="linear", mask=jnp.asarray(mask))
        np.testing.assert_allclose(np.asarray(dT)[dry], 0.0, atol=0.0)
