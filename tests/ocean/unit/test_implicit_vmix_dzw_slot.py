"""Veros u_centered dzw slot for the implicit vertical-diffusion solves
(``config.implicit_vmix_dzw_slot``, #428).

The backward-Euler tracer (T/S) and momentum-friction vertical-diffusion solves
take a GRADIENT (center-to-center) divisor.  legoESM's default is the midpoint
reconstruction ``build_dz_half(dz_cell) = 0.5(dz_k + dz_{k+1})``; Veros uses the
coordinate's center-to-center spacing ``dzw`` (``thermodynamics.py:267
delta = dt·kappaH/dzw``), which legoESM carries as ``z_coord.dz_half_ref``.

On a midpoint z-star coordinate ``dz_half_ref == build_dz_half(dz_ref)`` so the
flag is a NO-OP.  On a u_centered z-coordinate (the Veros-faithful ACC recipe)
the two differ per level, so at identical diffusivity the discrete flux differs.

The trio of equivalence tests pins the wiring exactly:
  * flag OFF ignores ``dz_half_ref`` (depends only on ``dz_ref``);
  * flag ON uses ``dz_half_ref``;
  * the two agree iff ``dz_half_ref`` is the midpoint.

Constant background A_v/K_v (``physics=None`` ⇒ state-independent K profiles) so
the solve metric is the only thing under test.  Run in the fp64 policy.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT = 1800.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _channel(z_coord=None, n_lat=8, n_lon=16, **cfg_kw):
    """Closed channel with a thermal front and a vertically-sheared jet, on the
    supplied vertical coordinate (defaults to a midpoint z-star)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    if z_coord is None:
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=float(np.sum(np.asarray(z_coord.dz_ref))),
        land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    nlev = z_coord.n_levels
    shear = np.linspace(1.0, 0.1, nlev)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    # Vertically-sheared meridional flow so the v friction solve is nontrivial
    # too (friction on a zero field is a slot-independent no-op).
    nv = state.v.data.shape[0]
    vlat = np.linspace(-1.0, 1.0, nv)[:, None, None]
    v = 0.1 * vlat * shear
    v = np.broadcast_to(v, state.v.data.shape).copy()
    v *= np.asarray(state.v_mask.data)[..., None]
    state = state._replace(v=state.v.replace(data=jnp.asarray(v)))
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("outer_integrator", "ab2")
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4, bottom_drag_r=1.0e-3,
        n_barotropic_substeps=8, enable_runtime_checks=False, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _midpoint_zstar():
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=4, H_max=4000.0)


def _u_centered_like():
    """A z-star whose center-to-center spacing dz_half_ref is perturbed away
    from the midpoint (a controlled stand-in for the Veros u_centered grid:
    dz_half_ref != 0.5(dz_k+dz_{k+1}), dz_ref / interfaces unchanged)."""
    z = _midpoint_zstar()
    nlev = z.n_levels
    factors = jnp.asarray(
        [1.0 + 0.4 * ((-1.0) ** k) for k in range(nlev - 1)],
        dtype=z.dz_half_ref.dtype)          # alternating ±40%, like Veros dzw
    return z._replace(dz_half_ref=z.dz_half_ref * factors)


# --------------------------------------------------------------- premise

def test_midpoint_zstar_dz_half_ref_is_midpoint():
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    z = _midpoint_zstar()
    np.testing.assert_allclose(
        np.asarray(z.dz_half_ref),
        np.asarray(build_dz_half(z.dz_ref)), rtol=0, atol=1e-12)


def test_u_centered_dz_half_ref_differs_from_midpoint():
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    z = _u_centered_like()
    assert np.max(np.abs(np.asarray(z.dz_half_ref)
                         - np.asarray(build_dz_half(z.dz_ref)))) > 1.0


def test_acc_recipe_coord_is_u_centered():
    """The shipped ACC recipe coordinate is genuinely u_centered (the slot the
    flag targets in production)."""
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_z_coord
    z = build_acc_z_coord()
    assert np.max(np.abs(np.asarray(z.dz_half_ref)
                         - np.asarray(build_dz_half(z.dz_ref)))) > 1.0


# --------------------------------------------------------------- validation

def test_rejects_without_implicit_vmix():
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        _channel(implicit_vmix_dzw_slot=True, implicit_vertical_mixing=False)


# ----------------------------------------------------- slot semantics

def _solve(z_coord, flag):
    state, model = _channel(z_coord=z_coord, implicit_vmix_dzw_slot=flag)
    out = model._apply_implicit_vertical_mixing(state, _DT, None)
    return {k: np.asarray(getattr(out, k).data) for k in ("T", "S", "u", "v")}


def test_noop_on_midpoint_zstar():
    """Where dz_half_ref IS the midpoint, flag ON == OFF, bit-identical."""
    z = _midpoint_zstar()
    off, on = _solve(z, False), _solve(z, True)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(on[k], off[k])


def test_active_on_u_centered():
    """Where dz_half_ref differs from the midpoint, flag ON changes every solved
    field (guards against a silently-dead flag)."""
    z = _u_centered_like()
    off, on = _solve(z, False), _solve(z, True)
    for k in ("T", "S", "u", "v"):
        assert np.max(np.abs(on[k] - off[k])) > 0.0, f"{k} unchanged by flag"


def test_flag_off_ignores_dz_half_ref():
    """The DEFAULT slot is build_dz_half(dz_cell), a function of dz_ref only —
    overriding dz_half_ref must not move the flag-OFF result."""
    z_mid, z_uc = _midpoint_zstar(), _u_centered_like()
    a, b = _solve(z_mid, False), _solve(z_uc, False)
    for k in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(a[k], b[k])


def test_flag_on_tracks_dz_half_ref():
    """The faithful slot IS dz_half_ref: flag-ON on the perturbed coordinate
    differs from flag-ON on the midpoint coordinate (same dz_ref, different
    dz_half_ref) — i.e. flag-ON consumes dz_half_ref, not the midpoint."""
    z_mid, z_uc = _midpoint_zstar(), _u_centered_like()
    a, b = _solve(z_mid, True), _solve(z_uc, True)
    for k in ("T", "S", "u", "v"):
        assert np.max(np.abs(a[k] - b[k])) > 0.0


# ------------------------------------------------------------ robustness

def test_stability_100_steps_u_centered():
    z = _u_centered_like()
    state, model = _channel(z_coord=z, implicit_vmix_dzw_slot=True)
    for _ in range(100):
        state = model.step(state, _DT)
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(T))
    assert np.max(np.abs(u)) < 5.0
    assert -5.0 < T.min() and T.max() < 40.0


def test_differentiable():
    z = _u_centered_like()
    state, model = _channel(z_coord=z, n_lat=6, n_lon=8,
                            implicit_vmix_dzw_slot=True)

    def loss(scale):
        st = state._replace(u=state.u.replace(data=state.u.data * scale))
        s1 = model.step(st, _DT)
        s2 = model.step(s1, _DT)
        return jnp.sum(s2.u.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


# --------------------------------------------------- NEMO e3w(Kmm) divisor
# trazdf.F90:219-221 divides the implicit flux coefficient by e3w(...,Kmm),
# called from stpmlf.F90:551 as tra_zdf(kstp,Nbb,Nnn,Nrhs,ts,Naa) -- the dummy
# Kmm binds to Nnn, NEMO's NOW time level; dynzdf.F90:200-203 divides momentum
# by e3uw(...,Kmm), which zgr_lib.F90:111-112 sets equal to e3w on the zco
# branch.  Under key_qco/key_vco_3d (domzgr_substitute.h90:131,108,49) that is
#     e3w_0(i,j,k) * (1 + r3t(i,j,Nnn)),   e3w_0(k) = gdept_0(k) - gdept_0(k-1)
# i.e. the T-POINT DEPTH DIFFERENCE, not the interface midpoint.  NEMO has no
# switch here, so neither does legoESM: the divisor is unbranched inside the
# NEMO identity ``zdf_implicit_solver_evaluation="nemo_literal"``, and the
# single canonical ``nemo_e3w_kmm`` serves both the tracer and momentum solves.


def _stretched_nemo_zcoord(nlev=6, ssh=3.0):
    """A coordinate whose T points are NOT the interface midpoints, carrying
    NEMO's own mesh ``e3w_0`` -- the DINO twin's situation in miniature.

    Thicknesses grow with depth; ``gdept_0`` is deliberately pulled off the
    midpoint by a level-dependent offset, so ``diff(gdept_0)`` and
    ``0.5*(e3t_k + e3t_{k+1})`` are different arrays at every interface.
    """
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    e3t = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])[:nlev]
    gdepw = np.concatenate([[0.0], np.cumsum(e3t)])
    # Off-centre T points (a stretching function's, not the midpoint's).
    gdept = gdepw[:-1] + e3t * (0.5 + 0.06 * np.arange(len(e3t)))
    e3w = np.concatenate([[2.0 * gdept[0]], np.diff(gdept)])
    z = create_z_star_from_thicknesses(
        e3t, t_depth_ref_m=gdept,
        nemo_gdept_0_m=gdept, nemo_gdepw_0_m=gdepw[:-1],
        nemo_e3t_0_m=e3t, nemo_e3w_0_m=e3w,
        nemo_e3w_source="mesh_reference")
    return z, e3t, gdept, ssh


def test_nemo_e3w_kmm_reproduces_the_oracle_divisor_on_a_stretched_column():
    """The survivor must be e3w_0*(1+r3t) to 1e-15 relative, and must NOT be
    the midpoint it replaces (non-vacuity: the reverted expression fails).

    The two divisors differ here by 3-6% per interface, so a test that passed
    against the old midpoint arm could not also pass against this one.
    """
    from legoesm.ocean.physics.vertical_mixing import (
        build_dz_half, nemo_e3w_kmm,
    )
    from legoesm.ocean.eos import nemo_r3t_stretch

    z, e3t, gdept, ssh = _stretched_nemo_zcoord()
    H = float(np.sum(e3t))
    eta = jnp.asarray(np.full((3, 4), ssh))
    H_bathy = jnp.asarray(np.full((3, 4), H))
    stretch = nemo_r3t_stretch(z, eta, H_bathy)
    e3t_now = jnp.asarray(e3t)[None, None, :] * stretch[..., None]

    got = np.asarray(nemo_e3w_kmm(z, e3t_now, stretch))
    want = np.diff(gdept)[None, None, :] * (1.0 + ssh / H)
    assert got.shape == (3, 4, len(e3t) - 1)
    assert np.max(np.abs(got / np.broadcast_to(want, got.shape) - 1.0)) < 1e-15

    # Non-vacuity: the expression this replaced is a DIFFERENT array here.
    midpoint = np.asarray(build_dz_half(e3t_now))
    assert np.max(np.abs(midpoint / got - 1.0)) > 1e-2, (
        "synthetic column does not separate the two divisors -- the test "
        "would pass against the reverted midpoint arm")


def test_nemo_e3w_kmm_face_map_is_the_same_object_as_the_tracer_divisor():
    """``e3uw_0 == e3w_0`` (zgr_lib.F90:111-112), so the momentum solve must
    divide by the face map OF THE TRACER DIVISOR, not by a second array."""
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w_kmm
    from legoesm.ocean.eos import nemo_r3t_stretch

    z, e3t, gdept, ssh = _stretched_nemo_zcoord()
    H = float(np.sum(e3t))
    eta = jnp.asarray(np.linspace(0.0, ssh, 12).reshape(3, 4))
    H_bathy = jnp.asarray(np.full((3, 4), H))
    stretch = nemo_r3t_stretch(z, eta, H_bathy)
    e3t_now = jnp.asarray(e3t)[None, None, :] * stretch[..., None]

    cell = nemo_e3w_kmm(z, e3t_now, stretch)
    faced = nemo_e3w_kmm(z, e3t_now, stretch,
                         to_point=lambda f: 0.5 * (f + jnp.roll(f, -1, axis=1)))
    np.testing.assert_array_equal(
        np.asarray(faced),
        np.asarray(0.5 * (cell + jnp.roll(cell, -1, axis=1))))


def test_nemo_e3w_kmm_midpoint_arm_is_exactly_build_dz_half():
    """A card with no NEMO mesh ``e3w_0`` and midpoint T points takes arm 2,
    which must be BIT-IDENTICAL to the expression it replaced -- this is what
    keeps the NEMO test-case (RK3) cards unmoved."""
    from legoesm.ocean.physics.vertical_mixing import (
        build_dz_half, nemo_e3w_kmm, nemo_e3w0_reference,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    z = create_ocean_z_star(n_levels=6, H_max=3000.0)
    assert nemo_e3w0_reference(z) is None
    rng = np.random.default_rng(11)
    e3t_now = jnp.asarray(rng.uniform(5.0, 500.0, size=(3, 4, 6)))
    stretch = jnp.asarray(rng.uniform(0.9, 1.1, size=(3, 4)))
    np.testing.assert_array_equal(
        np.asarray(nemo_e3w_kmm(z, e3t_now, stretch)),
        np.asarray(build_dz_half(e3t_now)))


def test_nemo_e3w_kmm_fails_closed_on_a_stretched_ladder_without_a_mesh():
    """Arm 2 cannot build ``gdept_0(k)-gdept_0(k-1)`` from a midpoint, so a
    coordinate that declares off-midpoint T points and supplies no NEMO mesh
    ``e3w_0`` must RAISE rather than silently take the midpoint."""
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w0_reference
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    e3t = np.array([10.0, 20.0, 40.0, 80.0])
    gdepw = np.concatenate([[0.0], np.cumsum(e3t)])
    gdept = gdepw[:-1] + e3t * 0.6           # off-midpoint, no mesh supplied
    z = create_z_star_from_thicknesses(e3t, t_depth_ref_m=gdept)
    with pytest.raises(ValueError, match="interface-midpoint ladder"):
        nemo_e3w0_reference(z)


def test_dzw_slot_takes_precedence_over_the_nemo_identity():
    """Both pick the implicit-solve gradient divisor (Veros dzw vs NEMO
    e3w(Kmm)).  No card in the tree selects both, and the collapse did not
    turn that combination into a new hard error; precedence is DOCUMENTED and
    pinned here instead -- the Veros slot wins, so a Veros card can never
    silently acquire NEMO's divisor.
    """
    z = _u_centered_like()          # dz_half_ref is +-40% off the midpoint
    state, both = _channel(z_coord=z, implicit_vmix_dzw_slot=True,
                           zdf_implicit_solver_evaluation="nemo_literal")
    _, dzw_only = _channel(z_coord=z, implicit_vmix_dzw_slot=True)
    _, nemo_only = _channel(z_coord=z,
                            zdf_implicit_solver_evaluation="nemo_literal")
    out_both = both._apply_implicit_vertical_mixing(state, _DT, None)
    out_dzw = dzw_only._apply_implicit_vertical_mixing(state, _DT, None)
    out_nemo = nemo_only._apply_implicit_vertical_mixing(state, _DT, None)
    # u/v (the friction solve) carry the divisor.  Selecting both must land on
    # the VEROS divisor: the residual against the dzw-only arm is ULP-scale
    # (the nemo_literal MOMENTUM recurrence is a different summation order of
    # the same matrix), while the divisor itself is a 40% lever -- so the
    # non-vacuity leg below is what makes this test able to fail.
    for k in ("u", "v"):
        a = np.asarray(getattr(out_both, k).data)
        b = np.asarray(getattr(out_dzw, k).data)
        c = np.asarray(getattr(out_nemo, k).data)
        scale = max(float(np.max(np.abs(b))), 1e-30)
        assert np.max(np.abs(a - b)) / scale < 1e-13, (
            f"{k}: selecting both did NOT take the Veros dzw divisor")
        assert np.max(np.abs(a - c)) / scale > 1e-6, (
            f"{k}: non-vacuity failed -- the two divisors are "
            "indistinguishable on this coordinate, so the assertion above "
            "cannot detect a precedence flip")


def test_nemo_divisor_reads_the_now_eta_that_is_threaded_in():
    """The divisor is ``e3w_0*(1+r3t(Kmm))``: it must move when the NOW eta
    threaded through ``eta_now`` moves, on an otherwise identical state.

    Without the ``eta_now`` threading at the AFTER-state call sites
    (_leapfrog_step's naa_expl -- the DINO kamm_mlf production path -- plus
    _unsplit_ab2_step, _ab2_step, _step_impl) the solve would divide by the
    AFTER thickness, which is exactly the defect this arm removes.
    """
    z = _midpoint_zstar()
    state, model = _channel(
        z_coord=z, implicit_vertical_mixing=True,
        zdf_implicit_solver_evaluation="nemo_literal")
    base = model._apply_implicit_vertical_mixing(
        state, _DT, None, eta_now=state.eta.data)
    # A uniformly 20% shallower NOW column (an expanding column between NOW
    # and AFTER) -- a controlled, nonzero eta tendency.
    eta_now = state.eta.data - 0.2 * jnp.abs(state.eta.data + 10.0)
    moved = model._apply_implicit_vertical_mixing(
        state, _DT, None, eta_now=eta_now)
    for k in ("T", "S", "u", "v"):
        a = np.asarray(getattr(base, k).data)
        b = np.asarray(getattr(moved, k).data)
        assert np.max(np.abs(a - b)) > 0.0, f"{k} unchanged by the NOW eta"


def test_legacy_midpoint_arm_is_still_the_default_divisor():
    """Cards off the NEMO identity keep ``build_dz_half(dz_cell)`` -- the
    contamination control for this change: nothing outside the identity moves.
    """
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w0_reference

    z = _midpoint_zstar()
    _, model = _channel(z_coord=z, implicit_vertical_mixing=True)
    assert model.config.zdf_implicit_solver_evaluation == "shared_thomas"
    assert model.config.implicit_vmix_dzw_slot is False
    assert nemo_e3w0_reference(z) is None
