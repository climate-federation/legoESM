"""ONE NEMO qco face-thickness rule, reached by both legoESM time-stepping lanes.

NEMO builds the free-surface face ratio in two entry points that carry the
CHARACTER-IDENTICAL statement:

* ``dom_qco_r3c`` -- the Modified-Leap-Frog lane
  (NEMO 5.0.2 ``src/OCE/DOM/domqco.F90:166-169``), called from
  ``stpmlf.F90:216,265,317``;
* ``dom_qco_r3c_RK3`` -- the RK3 lane (``domqco.F90:219-222``).

They differ only in loop extent and in a ``key_qcoTest_FluxForm`` alternative
(``domqco.F90:227``) that neither configuration compiles.
``domzgr_substitute.h90:127`` then makes the face thickness
``e3u(i,j,k,t) = e3u_0(i,j,k)*(1 + r3u(i,j,t)*umask(i,j,k))``.

Because NEMO shares the routine, legoESM must have ONE implementation reached
by both lanes -- a second copy would be an artificial branch point even if both
copies were correct.  This module pins that, three ways:

1. the shared builder reproduces the analytic ``e3u_0*(1+r3u)`` on a synthetic
   tilted free surface, and the reverted ``min``-of-stretched-thicknesses rule
   is shown to be a DIFFERENT number there (the planted violation);
2. the WS-RK3 stage transport reaches the shared builder at run time
   (counted, so removing the call makes the count zero);
3. the MLF tracer transport reaches the SAME symbol, and no second builder
   survives in the stepping module.
"""

import inspect

import jax
import jax.numpy as jnp
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import vertical as vertical_module
from legoesm.ocean.vertical import (
    compute_layer_thickness,
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_vorticity_e3f_cgrid,
)


def _synthetic_operands():
    """A 1x3 periodic row, flat 2-level ladder, ssh tilted across ONE face.

    Uniform cell area and a uniform reference ladder make the analytic answer
    exact: ``r3u = 0.5*(ssh_i + ssh_{i+1})/hu_0`` and
    ``e3u = e3u_0*(1 + r3u)``.
    """
    set_policy(PrecisionPolicy.fp64())
    n_lat, n_lon, nlev = 1, 3, 2
    e3_0 = np.full((n_lat, n_lon, nlev), 10.0)
    mask = np.ones((n_lat, n_lon, nlev))
    # ssh is non-zero on exactly one column, so every face of that column has
    # a free-surface DIFFERENCE across it -- the regime where the two rules
    # disagree at first order.
    eta = np.array([[0.0, 1.0, 0.0]])
    area = np.full((n_lat, n_lon), 4.0e6)
    h_0 = e3_0.sum(axis=-1)                      # 20.0 m, uniform
    return eta, e3_0, mask, area, h_0


def _analytic_nemo_e3u(eta, e3_0, area, h_0):
    """``domqco.F90:166-169`` + ``domzgr_substitute.h90:127``, by hand."""
    weighted = area * eta
    r3u = 0.5 * (weighted + np.roll(weighted, -1, axis=1)) / h_0 / area
    return e3_0 * (1.0 + r3u[..., None]), r3u


def _reverted_min_rule_e3u(eta, e3_0, h_0):
    """The pre-fix rule: ``min`` of the two STRETCHED T-cell thicknesses."""
    stretched = e3_0 * (1.0 + (eta / h_0)[..., None])
    return np.minimum(stretched, np.roll(stretched, -1, axis=1))


def test_shared_builder_is_nemo_e3u0_times_one_plus_r3u_not_the_min_rule():
    eta, e3_0, mask, area, h_0 = _synthetic_operands()
    expected, r3u = _analytic_nemo_e3u(eta, e3_0, area, h_0)

    e3u, e3v, one_plus_r3u, one_plus_r3v = nemo_qco_live_face_geometry_cgrid(
        jnp.asarray(eta), jnp.asarray(e3_0), jnp.asarray(e3_0),
        jnp.asarray(mask), jnp.asarray(mask),
        jnp.asarray(h_0), jnp.asarray(h_0),
        jnp.asarray(area), jnp.asarray(area), jnp.asarray(area),
    )
    # fp64 all the way through (oracle-fidelity Rule 1c).
    assert np.asarray(e3u).dtype == np.float64

    # Redundant west/south layout: U gains a column, V gains a row.
    assert np.asarray(e3u).shape == (1, 4, 2)
    assert np.asarray(e3v).shape == (2, 3, 2)

    native_u = np.asarray(e3u)[:, 1:, :]
    np.testing.assert_allclose(native_u, expected, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(one_plus_r3u)[:, 1:], 1.0 + r3u, rtol=1e-15, atol=0.0)
    # The V face of a single-row domain rolls onto itself, so its ratio is
    # the column's own ssh/h_0 -- checked so the V branch is not vacuous.
    np.testing.assert_allclose(
        np.asarray(one_plus_r3v)[1:], 1.0 + eta / h_0, rtol=1e-15, atol=0.0)

    # PLANTED VIOLATION: the reverted rule is a different number here, so the
    # assertions above cannot pass with the min rule substituted.
    reverted = _reverted_min_rule_e3u(eta, e3_0, h_0)
    gap = np.abs(reverted - expected).max()
    assert gap == pytest.approx(0.25, rel=1e-12), gap
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(reverted, expected, rtol=1e-15, atol=0.0)


def test_shared_builder_carries_source_literal_live_face_reciprocal_bits():
    """Pin ``r1_hu_0/(1+r3u)`` instead of re-dividing summed live e3."""
    set_policy(PrecisionPolicy.fp64())
    rng = np.random.default_rng(2218)
    ny, nx, nz = 2, 5, 4
    eta = rng.uniform(-0.3, 0.4, (ny, nx)).astype(np.float64)
    e3u0 = rng.uniform(0.2, 250.0, (ny, nx, nz)).astype(np.float64)
    e3v0 = rng.uniform(0.2, 250.0, (ny, nx, nz)).astype(np.float64)
    umask = np.ones_like(e3u0)
    vmask = np.ones_like(e3v0)
    hu0 = np.sum(e3u0 * umask, axis=-1)
    hv0 = np.sum(e3v0 * vmask, axis=-1)
    area_t = rng.uniform(2.0e7, 1.1e10, (ny, nx)).astype(np.float64)
    area_u = rng.uniform(2.0e7, 1.1e10, (ny, nx)).astype(np.float64)
    area_v = rng.uniform(2.0e7, 1.1e10, (ny, nx)).astype(np.float64)

    weighted = area_t * eta
    numerator_u = np.float64(0.5) * (
        weighted + np.roll(weighted, -1, axis=1))
    r1_hu0 = np.float64(1.0) / hu0
    r1_area_u = np.float64(1.0) / area_u
    r3u = (numerator_u * r1_hu0) * r1_area_u
    expected = r1_hu0 / (np.float64(1.0) + r3u)

    def build(*args):
        return nemo_qco_live_face_geometry_cgrid(
            *args, include_reciprocals=True)[4]

    actual = np.asarray(jax.jit(build)(
        *map(jnp.asarray, (
            eta, e3u0, e3v0, umask, vmask, hu0, hv0,
            area_t, area_u, area_v))))[:, 1:]
    np.testing.assert_array_equal(
        actual.view(np.uint64), expected.view(np.uint64))

    # The former stage-transport path reconstructed the same real reciprocal
    # by summing live e3 and dividing again.  The awkward operands make that
    # alternative observably non-bit-identical.
    live_e3u = e3u0 * (np.float64(1.0) + r3u[..., None] * umask)
    rederived = np.float64(1.0) / np.sum(live_e3u, axis=-1)
    assert not np.array_equal(rederived.view(np.uint64), expected.view(np.uint64))


def test_live_f_thickness_uses_each_cards_own_mesh_without_bridge_operands():
    """Native and bridged GYRE cards reach one own-mesh F-thickness rule."""
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64())
    native = build_nemo_gyre_recipe()
    native_h0 = compute_layer_thickness(
        jnp.zeros_like(native.initial_state.eta.data),
        native.initial_state.H_bathy.data, native.z_coord,
        min_water_column_m=native.model_config.min_water_column_m)
    native_tmask = jnp.broadcast_to(
        native.initial_state.land_mask.data[..., None], native_h0.shape)
    native_e3f = nemo_qco_live_vorticity_e3f_cgrid(
        native.initial_state.eta.data, native.z_coord,
        native.initial_state.eta.data.dtype, grid=native.grid,
        e3t_0=native_h0, tmask=native_tmask)
    assert np.asarray(native_e3f).shape == (23, 33, 30)
    assert bool(jnp.all(jnp.isfinite(native_e3f)))

    bridged = build_nemo_testcase_card("GYRE-zco").recipe
    recorded_default = nemo_qco_live_vorticity_e3f_cgrid(
        bridged.initial_state.eta.data, bridged.z_coord,
        bridged.initial_state.eta.data.dtype, grid=bridged.grid)
    own_mesh = nemo_qco_live_vorticity_e3f_cgrid(
        bridged.initial_state.eta.data, bridged.z_coord,
        bridged.initial_state.eta.data.dtype, grid=bridged.grid,
        e3t_0=bridged.z_coord.nemo_e3t_0,
        tmask=bridged.z_coord.is_active)
    np.testing.assert_array_equal(
        np.asarray(own_mesh).view(np.uint64),
        np.asarray(recorded_default).view(np.uint64))

    # PLANTED VIOLATION: removing one consumed mask cell changes a wet F row.
    changed_mask = bridged.z_coord.is_active.at[10, 15, 0].set(False)
    changed = nemo_qco_live_vorticity_e3f_cgrid(
        bridged.initial_state.eta.data, bridged.z_coord,
        bridged.initial_state.eta.data.dtype, grid=bridged.grid,
        e3t_0=bridged.z_coord.nemo_e3t_0, tmask=changed_mask)
    assert not np.array_equal(
        np.asarray(changed).view(np.uint64), np.asarray(own_mesh).view(np.uint64))


def test_ws_rk3_stage_transport_reaches_the_shared_builder():
    """Counted at run time: zero calls if the RK3 lane stops using it."""
    from tests.ocean.unit.test_nemo_ws_qco_stage_faces import _tilted_geometry
    from tests.ocean.unit.test_nemo_ws_tracer_rk3 import _lock_model

    card, entry, *_ = _tilted_geometry()
    calls = []
    original = model_module.nemo_qco_live_face_geometry_cgrid

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    model_module.nemo_qco_live_face_geometry_cgrid = counted
    try:
        _lock_model().step(entry, dt=card.dt_s)
    finally:
        model_module.nemo_qco_live_face_geometry_cgrid = original
    assert calls, "the WS-RK3 stage transport no longer builds NEMO's e3u(Kmm)"


def test_mlf_tracer_transport_reaches_the_same_symbol():
    """The MLF lane's own stepping body calls the ONE shared builder.

    ``_step_impl`` is the function that runs on the MLF lane -- the leapfrog
    and ``nemo_mlf`` outer steps both call it -- and its
    ``wzv_call2_evaluation == "nemo_literal"`` branch is where the DINO cards
    build the tracer transport's ``e3u(Kmm)`` (``traadv.F90:329-330``).  The
    DINO cards cannot be constructed on this branch (an unrelated
    ``eos_depth="geometric"`` certification guard), so this is asserted on the
    executing function's source rather than by stepping the card.
    """
    body = inspect.getsource(model_module.LatLonCGridOceanModel._step_impl)
    # _step_impl is ~2700 lines and mentions the selector more than once, so
    # a whole-body `in` would bind on any unrelated occurrence.  Slice the
    # ONE branch that builds the tracer transport's face thickness -- it is
    # the block that assigns _h_u_tracer -- and assert inside it.
    marker = "_h_u_tracer"
    assert body.count(marker) >= 1
    start = body.index("nemo_e3t_0")
    branch = body[start:body.index(marker, start) + 400]
    assert "nemo_qco_live_face_geometry_cgrid(" in branch
    assert "nemo_qco_mesh_operands(" in branch
    # No second builder may survive on this lane.
    assert "nemo_qco_live_face_thicknesses" not in body

    assert (model_module.nemo_qco_live_face_geometry_cgrid
            is vertical_module.nemo_qco_live_face_geometry_cgrid)
    assert (inspect.getsource(model_module._nemo_ws_qco_stage_faces)
            .count("nemo_qco_live_face_geometry_cgrid(") == 1)
