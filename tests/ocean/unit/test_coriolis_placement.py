"""#1455: the vertex-Coriolis placement reaches every C-grid Coriolis path.

The option lives in ONE array, the geometry's ``f_v``, and every path reads
that array through one of exactly two helpers -- ``vertex_coriolis`` (the
barotropic EEN pre-block and the 3-D EEN/ENE vorticity flux) and
``coriolis_at_faces`` (the semi-implicit and ``explicit_ab2`` face-f
Coriolis).  These tests prove that rather than asserting it: both helpers'
output must move with the option, and the barotropic pre-block's stored
``f_vtx`` must BE ``vertex_coriolis``'s output, with no second wiring step.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import vertex_coriolis


def test_no_recipe_selects_face_latitude_before_its_owner_is_exact():
    """A selector-only arm is partial until the literal builder lands."""
    from legoesm.ocean.experiments.dino import (
        DINOConfig, DINO_RECIPES, dino_config_for_recipe)

    assert DINOConfig().coriolis_placement == "cell_average"
    selected = {
        name for name in DINO_RECIPES
        if dino_config_for_recipe(name).coriolis_placement == "face_latitude"
    }
    assert selected == set()


def test_nonfaithful_recipe_keeps_the_cell_average_byte_pin():
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    assert dino_config_for_recipe(
        "legoesm_default").coriolis_placement == "cell_average"


def _pair(**kw):
    base = dict(n_lat=12, n_lon=4, omega=1.0e-4, dtype=jnp.float64)
    base.update(kw)
    n_lat = base.pop("n_lat")
    n_lon = base.pop("n_lon")
    return (create_latlon_geometry(n_lat, n_lon, **base),
            create_latlon_geometry(n_lat, n_lon,
                                   coriolis_placement="face_latitude", **base))


def test_vertex_coriolis_carries_the_placement_option():
    """Helper 1: the EEN/ENE vorticity-flux door."""
    avg, face = _pair()
    f_avg = np.asarray(vertex_coriolis(avg))
    f_face = np.asarray(vertex_coriolis(face))
    assert f_avg.shape == f_face.shape == (13, 5)
    assert not np.array_equal(f_avg, f_face)
    # and it is exactly grid.f_v plus the periodic wrap column, both ways
    for geom, f in ((avg, f_avg), (face, f_face)):
        assert np.array_equal(f[:, :4], np.asarray(geom.f_v))
        assert np.array_equal(f[:, 4], np.asarray(geom.f_v)[:, 0])


def test_the_barotropic_EEN_preblock_stores_that_same_array():
    """The barotropic Coriolis path: its f_vtx must be vertex_coriolis's
    output, so the option reaches it with no second wiring step."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs)
    _, face = _pair()
    n_lat, n_lon = 12, 4
    h_k = jnp.ones((n_lat, n_lon, 3), dtype=jnp.float64)
    mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
    u_mask = jnp.ones((n_lat, n_lon + 1), dtype=jnp.float64)
    v_mask = jnp.ones((n_lat + 1, n_lon), dtype=jnp.float64)
    pre = _build_een_barotropic_inputs(h_k, face, mask, u_mask, v_mask,
                                       jnp.float64)
    assert np.array_equal(np.asarray(pre["f_vtx"]),
                          np.asarray(vertex_coriolis(face)))


def test_the_hand_computed_value_survives_all_the_way_to_the_preblock():
    """End to end, with a number a reader can check: at 30 degrees and
    omega = 1e-4 the vertex Coriolis is exactly 1e-4."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs)
    lat_face = jnp.deg2rad(jnp.array([-30.0, 0.0, 30.0, 60.0]))
    lat_c = 0.5 * (lat_face[:-1] + lat_face[1:])
    geom = create_latlon_geometry(
        3, 4, omega=1.0e-4, dtype=jnp.float64, lat_1d=lat_c,
        lat_face_1d=lat_face, coriolis_placement="face_latitude")
    h_k = jnp.ones((3, 4, 2), dtype=jnp.float64)
    pre = _build_een_barotropic_inputs(
        h_k, geom, jnp.ones((3, 4), dtype=jnp.float64),
        jnp.ones((3, 5), dtype=jnp.float64),
        jnp.ones((4, 4), dtype=jnp.float64), jnp.float64)
    assert float(np.asarray(pre["f_vtx"])[2, 0]) == pytest.approx(1.0e-4,
                                                                  rel=1e-12)


def test_coriolis_at_faces_carries_it_too():
    """Helper 2: the semi-implicit / explicit_ab2 face-f door.

    ``f_v`` must move with the option and ``f_u`` must NOT -- on a lat-lon
    grid the u-point shares the tracer row's latitude, so f_u is already f at
    its own point and changing it would be a second, unasked-for change.
    """
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
    avg, face = _pair()
    fu_a, fv_a = coriolis_at_faces(avg, jnp.float64)
    fu_f, fv_f = coriolis_at_faces(face, jnp.float64)
    assert np.array_equal(np.asarray(fu_a), np.asarray(fu_f))
    assert not np.array_equal(np.asarray(fv_a), np.asarray(fv_f))


# --------------------------------------------------------------------------
# The PRODUCTION branches this option added.  Adversarial review, round 2:
# four of them shipped with zero coverage, and the bridge fixtures build
# exact-NEMO-omega meshes so the tightened Coriolis bound could be reverted to
# 1e-3 with the suite staying green.  Each test below fails if its branch is
# removed or loosened.
# --------------------------------------------------------------------------
def _z():
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    return create_z_star_from_thicknesses(np.array([10.0, 20.0, 30.0]))


def _model(geom, **flat):
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    return LatLonCGridOceanModel(geom, _z(),
                                 LatLonCGridOceanConfig.from_flat(**flat))


def test_model_raises_on_an_unknown_placement():
    with pytest.raises(ValueError, match="coriolis_placement"):
        _model(create_latlon_geometry(12, 4, dtype=jnp.float64),
               coriolis_placement="f_point")


def test_model_REFUSES_a_placement_it_would_silently_ignore():
    """A pre-built geometry passes through ensure_geometry unchanged, so a
    non-default request on the config would do nothing at all.  The model must
    refuse rather than run the default while its config says otherwise."""
    geom = create_latlon_geometry(12, 4, dtype=jnp.float64)   # cell-average
    with pytest.raises(ValueError, match="silently ignored"):
        _model(geom, coriolis_placement="face_latitude")


def test_a_geometry_ALREADY_built_face_latitude_is_accepted():
    """The refusal must not fire when the request was honoured upstream --
    otherwise the oracle lane, which builds its geometry in the bridge, could
    never enable the option at all."""
    geom = create_latlon_geometry(12, 4, dtype=jnp.float64,
                                  coriolis_placement="face_latitude")
    _model(geom, coriolis_placement="face_latitude")          # must not raise


def test_the_refusal_does_NOT_fire_on_a_beta_plane():
    """Where f is LINEAR in y the two conventions are identical, so the
    setting is indistinguishable rather than ignored.  Raising there aborts a
    valid run -- measured in review as a false positive on both beta-plane and
    f-plane geometries."""
    from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
    geom = create_beta_plane_cgrid_geometry(
        8, 8, dx_m=1.0e4, dy_m=1.0e4, f0=1.0e-4, beta=2.0e-11,
        y_origin_m=0.0, cartesian_pseudo_lat=True)
    _model(geom, coriolis_placement="face_latitude")          # must not raise


def test_the_bridge_forwards_the_placement_to_the_geometry():
    """The oracle lane builds its geometry in the bridge, so the option has to
    survive that call or it is unreachable exactly where it is needed."""
    import inspect
    from legoesm.ocean.fidelity import nemo_state_bridge as B
    sig = inspect.signature(B.bridge_nemo_to_legoesm_topo)
    assert "coriolis_placement" in sig.parameters
    assert sig.parameters["coriolis_placement"].default == "cell_average"


def test_the_bridge_defaults_to_NEMOs_earth_not_legoESMs():
    """The defect that put the whole oracle lane on the wrong planet."""
    import inspect
    from legoesm import constants
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity import nemo_state_bridge as B
    got = inspect.signature(
        B.bridge_nemo_to_legoesm_topo).parameters["omega"].default
    assert got == NEMO_CONSTANTS_CONFIG.Omega
    assert got != constants.Omega
    assert abs(got - constants.Omega) / got > 1e-5


def test_the_coriolis_bound_is_tight_enough_to_catch_the_rotation_rate_gap():
    """The guard that let a 1.578e-05 rotation-rate error through was 1e-3.

    At fp64 the bound must be the tight one; at fp32 it must relax to the
    array's own resolution, because a fixed 1e-9 there would fail on rounding
    rather than on physics -- but it must STILL be well under the gap it
    exists to catch, or the relaxation has made it vacuous.
    """
    import inspect
    from legoesm import constants
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity import nemo_state_bridge as B
    f_rtol = inspect.signature(
        B.bridge_nemo_to_legoesm_topo).parameters["f_rtol"].default
    gap = abs(constants.Omega - NEMO_CONSTANTS_CONFIG.Omega) \
        / NEMO_CONSTANTS_CONFIG.Omega
    assert f_rtol <= 1e-9
    for dtype in (np.float64, np.float32):
        bound = max(f_rtol, 8.0 * float(np.finfo(dtype).eps))
        assert bound < gap / 10.0, (dtype, bound, gap)


def test_the_metric_guards_did_NOT_inherit_the_tightened_bound():
    """Tightening one guard must not silently re-scope another: the dx_T check
    tolerates a reconstruction residual the Coriolis check does not."""
    import inspect
    src = inspect.getsource(
        __import__("legoesm.ocean.fidelity.nemo_state_bridge",
                   fromlist=["x"]).bridge_nemo_to_legoesm_topo)
    assert "_metric_rtol = 1e-3" in src
    assert "dx_err > _metric_rtol" in src


def test_NEMO_omega_is_the_sidereal_expression_not_the_key_cice_literal():
    """Recomputed here from phycst.F90's own formula.  A regression to the
    key_cice literal 7.292116e-05 is otherwise caught only by a bridge guard
    that needs real NEMO dumps, i.e. never in CI."""
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    rday = 24.0 * 60.0 * 60.0
    rpi = 3.141592653589793
    rsiyea = 365.25 * rday * 2.0 * rpi / 6.283076
    rsiday = rday / (1.0 + rday / rsiyea)
    assert NEMO_CONSTANTS_CONFIG.Omega == pytest.approx(2.0 * rpi / rsiday,
                                                        rel=1e-15)
    assert abs(NEMO_CONSTANTS_CONFIG.Omega - 7.292116e-05) / \
        NEMO_CONSTANTS_CONFIG.Omega > 1e-8      # const-ok: the key_cice literal


def test_there_is_only_ONE_NEMO_constants_preset():
    """Two same-named constants blocks with one divergent field is the
    shadowing pattern that put the oracle lane on the wrong planet."""
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as canon
    from legoesm.ocean.fidelity.nemo_recipe import (
        NEMO_CONSTANTS_CONFIG as recipe_side)
    assert recipe_side is canon


# --------------------------------------------------------------------------
# THE CONSERVATION TRADE, MEASURED rather than asserted in a docstring.
# The option's main physical claim -- that the cell average IS the discrete
# curl of solid-body rotation and the face-latitude form is not -- had no
# committed measurement (adversarial review, round 2, and this repo's own
# "a throwaway probe's number is unmeasured" rule).  This is that measurement,
# made with the model's OWN curl operator, not a re-derivation.
# --------------------------------------------------------------------------
def _solid_body_curl_error(placement, n_lat=60):
    """|curl(solid-body rotation) - f_v| / 2*Omega, on a stretched mesh.

    A fluid at rest in the rotating frame has u = Omega*R*cos(phi), v = 0, and
    the curl of that field IS the planetary vorticity.  So whichever f_v the
    grid builds, the discrete curl of solid-body rotation is the answer it
    SHOULD equal -- any departure is a planetary vorticity the model would
    invent for a fluid that is not moving.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import curl_vertex_cgrid
    from legoesm import constants
    om, radius = 7.292115083046062e-05, float(constants.R_earth)  # coeff-ok: NEMO's rate, any rate works
    # Mercator-like stretching: uniform in the Mercator coordinate.
    y = np.linspace(-1.2, 1.2, n_lat + 1)
    lat_face = 2.0 * np.arctan(np.exp(y)) - np.pi / 2.0
    lat_c = 0.5 * (lat_face[:-1] + lat_face[1:])
    n_lon = 4
    lon = np.linspace(0.0, 2.0 * np.pi * (n_lon - 1) / n_lon, n_lon)
    geom = create_latlon_geometry(
        n_lat, n_lon, radius=radius, omega=om, dtype=jnp.float64,
        lat_1d=jnp.asarray(lat_c), lon_1d=jnp.asarray(lon),
        lat_face_1d=jnp.asarray(lat_face), coriolis_placement=placement)
    u = jnp.asarray(om * radius * np.cos(lat_c)[:, None]
                    * np.ones((1, n_lon + 1)))
    v = jnp.zeros((n_lat + 1, n_lon))
    zeta = np.asarray(curl_vertex_cgrid(u, v, geom))[:, 0]
    f_v = np.asarray(geom.f_v)[:, 0]
    return np.abs(zeta[1:-1] - f_v[1:-1]) / (2.0 * om)


def test_the_cell_average_IS_the_discrete_curl_of_solid_body_rotation():
    """EXACT, not merely better: this is why the default is not arbitrary."""
    err = _solid_body_curl_error("cell_average")
    assert float(np.median(err)) < 1e-13
    assert float(err.max()) < 1e-12


def test_face_latitude_BREAKS_that_identity_by_a_measurable_amount():
    """The trade the option's docstring must state: a fluid in exact
    solid-body co-rotation acquires a spurious planetary vorticity."""
    err = _solid_body_curl_error("face_latitude")
    assert float(np.median(err)) > 1e-06
    # and it is orders of magnitude worse than the default, not comparable
    ref = float(np.median(_solid_body_curl_error("cell_average")))
    assert float(np.median(err)) > 1e6 * max(ref, 1e-30)


def test_the_trade_VANISHES_on_a_uniform_grid():
    """Bounding the claim: with constant latitude spacing the two conventions
    differ by a uniform factor, so neither is 'the' discrete curl and the
    trade is not a structure change."""
    a = _solid_body_curl_error("cell_average")
    assert float(np.median(a)) < 1e-13          # stretched case, for contrast
    from legoesm.ocean.dynamics.latlon_cgrid_operators import curl_vertex_cgrid
    from legoesm import constants
    om, radius = 7.292115083046062e-05, float(constants.R_earth)  # coeff-ok: as above
    n_lat, n_lon = 60, 4
    geoms = {}
    for placement in ("cell_average", "face_latitude"):
        geoms[placement] = create_latlon_geometry(
            n_lat, n_lon, radius=radius, omega=om, dtype=jnp.float64,
            coriolis_placement=placement)
    ratio = (np.asarray(geoms["cell_average"].f_v)[1:-1, 0]
             / np.asarray(geoms["face_latitude"].f_v)[1:-1, 0])
    keep = np.abs(np.asarray(geoms["face_latitude"].f_v)[1:-1, 0]) > 1e-12
    assert float(np.ptp(ratio[keep])) < 1e-12   # a UNIFORM factor
