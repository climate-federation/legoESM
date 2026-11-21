"""Boutle et al. (2014) liquid FSD as the MG2 relvar (relvar_scheme="boutle2014").

Equation (Boutle et al. 2014, QJRMS 140, 583-594, as restated in Crnivec and
Mayer 2021, GMD 14, 3663, eq. 8): FSD = (0.45 - 0.25 C) Phi for C < 1,
0.11 Phi for C = 1, Phi = (x C)^(1/3) ((0.06 x C)^1.5 + 1)^(-0.17), x in km.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.microphysics._warm_rain import (
    boutle2014_fsd,
    boutle2014_relvar,
)

jax.config.update("jax_enable_x64", True)


def _fsd_paper(x_km, c):
    """The paper's equation written out independently, in plain floats."""
    phi = (x_km * c) ** (1.0 / 3.0) * ((0.06 * x_km * c) ** 1.5 + 1.0) ** -0.17
    return (0.11 if c >= 1.0 else 0.45 - 0.25 * c) * phi


def _area(x_km):
    return (x_km * 1.0e3) ** 2


@pytest.mark.parametrize("x_km,c", [(100.0, 0.02), (25.0, 0.02), (25.0, 0.5), (110.0, 0.999),
                                    (25.0, 1.0), (100.0, 1.0), (3.0, 0.3)])
def test_fsd_matches_the_paper(x_km, c):
    got = float(boutle2014_fsd(jnp.asarray(c), jnp.asarray(_area(x_km))))
    np.testing.assert_allclose(got, _fsd_paper(x_km, c), rtol=1e-12)


def test_fsd_reference_values_and_the_overcast_jump():
    # values recomputed independently (review, 5 November 2026)
    np.testing.assert_allclose(float(boutle2014_fsd(0.02, _area(100.0))), 0.556796, rtol=1e-5)
    np.testing.assert_allclose(float(boutle2014_fsd(0.02, _area(25.0))), 0.352886, rtol=1e-5)
    np.testing.assert_allclose(1.0 / float(boutle2014_fsd(1.0, _area(25.0))) ** 2, 13.7795, rtol=1e-4)
    # above relvarmax the relvar is MG2's clip value already (the floor of the fix)
    assert float(boutle2014_relvar(1.0, _area(25.0), 1.0)) == 10.0
    # the published jump at C = 1 is kept: just below 1 the (0.45 - 0.25 C) branch
    below, at = (float(boutle2014_fsd(c, _area(25.0))) for c in (1.0 - 1e-12, 1.0))
    assert below / at == pytest.approx(0.20 / 0.11, rel=1e-9)


def test_relvar_is_inverse_square_and_cloud_free_cells_get_relvarmax():
    c = jnp.asarray([0.0, 0.05, 0.4, 1.0])
    a = jnp.full(4, _area(110.0))
    r1 = boutle2014_relvar(c, a, 1.0)
    r2 = boutle2014_relvar(c, a, 2.0)
    np.testing.assert_allclose(r1[1:], 1.0 / np.asarray(boutle2014_fsd(c, a))[1:] ** 2, rtol=1e-12)
    np.testing.assert_allclose(r2[1:], r1[1:] / 4.0, rtol=1e-12)
    assert float(r1[0]) == 10.0 and float(r2[0]) == 10.0


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs a float64 state")
def test_cell_size_in_the_working_precision_of_the_cloud_fraction():
    # a mesh with float32 areas and a float64 state: the cell size is taken in
    # float64 from the same float32 area value, not rounded by a float32 sqrt
    c = jnp.asarray([0.03, 0.3, 0.9, 1.0], dtype=jnp.float64)
    a32 = jnp.asarray([_area(x) for x in (27.3, 61.7, 110.9, 233.1)], dtype=jnp.float32)
    got = boutle2014_fsd(c, a32)
    assert got.dtype == jnp.float64
    np.testing.assert_array_equal(got, boutle2014_fsd(c, a32.astype(jnp.float64)))


def test_gradients_in_the_factor_are_finite_and_correct():
    c = jnp.asarray([0.0, 0.03, 0.3, 0.9, 1.0])
    a = jnp.full(5, _area(60.0))
    check_grads(lambda f: boutle2014_relvar(c, a, f), (1.3,), order=2, modes=("rev",))
    g = jax.grad(lambda f: jnp.sum(boutle2014_relvar(c, a, f)))(1.0)
    assert np.isfinite(float(g)) and float(g) < 0.0
    # finite in the cloud fraction too, including C = 0
    gc = jax.grad(lambda cc: jnp.sum(boutle2014_relvar(cc, a, 1.0)))(c)
    assert np.all(np.isfinite(np.asarray(gc)))


def _physics(**morrison_kw):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none", cloud_config=CloudConfig(scheme="cam6_clubb")),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", liquid_partition=True,
                                    clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
            warm_rain_scheme="kk2000_cam6", **morrison_kw)))
    return make_physics(cfg, model_type="mpas", dt=600.0, cld_macmic_num_steps=2), cfg


def test_build_guards():
    with pytest.raises(ValueError, match="Unknown relvar_scheme"):
        _physics(warm_rain_incloud=True, relvar_scheme="boutle")
    with pytest.raises(ValueError, match="relvar_scheme='boutle2014' needs the CLUBB"):
        _physics(warm_rain_incloud=False, relvar_scheme="boutle2014")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="boutle_fsd_factor must be positive"):
            _physics(warm_rain_incloud=True, relvar_scheme="boutle2014", boutle_fsd_factor=bad)
    # every other lane refuses it instead of ignoring it
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.integration import require_relvar_scheme_lane
    mc = MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
        warm_rain_scheme="kk2000", relvar_scheme="boutle2014"))
    for lane in ("hydrostatic", "nonhydrostatic", "spectral_pe", "plane", "pipeline"):
        with pytest.raises(ValueError, match="MPAS lane only"):
            require_relvar_scheme_lane(mc, lane)
    require_relvar_scheme_lane(mc, "mpas")
    require_relvar_scheme_lane(MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig()), "plane")
    # the Boutle knobs set without boutle2014 would be inert: refused
    for kw in (dict(boutle_fsd_factor=1.5), dict(boutle_overcast_ramp_width=0.0)):
        with pytest.raises(ValueError, match="silently inert"):
            require_relvar_scheme_lane(MicrophysicsConfig(
                scheme="morrison", morrison=MorrisonConfig(**kw)), "mpas")
    # numpy scalars are checked too
    for kw, msg in ((dict(boutle_fsd_factor=np.float32(-1.0)), "must be positive"),
                    (dict(boutle_overcast_ramp_width=np.float64(1.0)), r"lie in \[0, 1\)")):
        with pytest.raises(ValueError, match=msg):
            require_relvar_scheme_lane(MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
                warm_rain_scheme="kk2000", relvar_scheme="boutle2014", **kw)), "mpas")


def test_cell_size_guard():
    from legoesm.atmosphere.physics.microphysics._warm_rain import check_boutle2014_cell_size
    check_boutle2014_cell_size(np.asarray([_area(0.5), _area(110.0), _area(500.0)]))
    with pytest.raises(ValueError, match="is areaCell in m"):
        check_boutle2014_cell_size(np.asarray([3.0e-10]))      # unit-sphere mesh cell


def test_gradients_stay_finite_at_tiny_cloud_fraction():
    """Reverse pass through MG2's clip: no 0 * inf for a vanishing positive C
    (review finding, 5 November 2026: NaN at C = 1e-300 before the fix)."""
    from legoesm.atmosphere.physics.microphysics._warm_rain import mg2_var_coef
    a = jnp.full(4, _area(110.0))

    def loss(cc, f):
        return jnp.sum(mg2_var_coef(boutle2014_relvar(cc, a, f), 2.47, jnp.float64))
    for c in (jnp.asarray([1e-300, 5e-320, 1e-185, 1e-20]), jnp.asarray([1e-30, 1e-23, 1e-12, 0.5])):
        gc, gf = jax.grad(loss, argnums=(0, 1))(c, 1.0)
        assert np.all(np.isfinite(np.asarray(gc))) and np.isfinite(float(gf))
    c32 = jnp.asarray([1e-30, 1e-23, 0.3], jnp.float32)
    g32 = jax.grad(lambda cc: jnp.sum(mg2_var_coef(boutle2014_relvar(cc, jnp.full(3, _area(110.0), jnp.float32),
                                                                     jnp.float32(1.0)), 2.47, jnp.float32)))(c32)
    assert np.all(np.isfinite(np.asarray(g32)))


def test_cloud_fraction_outside_zero_one_is_clipped():
    a = jnp.full(4, _area(110.0))
    np.testing.assert_array_equal(boutle2014_fsd(jnp.asarray([-0.2, 0.0, 1.0, 1.3]), a),
                                  boutle2014_fsd(jnp.asarray([0.0, 0.0, 1.0, 1.0]), a))


def test_c_gradient_matches_finite_differences():
    a = jnp.full(3, _area(60.0))
    check_grads(lambda cc: boutle2014_relvar(cc, a, 1.0), (jnp.asarray([0.03, 0.3, 0.9]),),
                order=1, modes=("rev",), rtol=1e-4)    # measured 2.6e-5 at the default step


@pytest.mark.parametrize("scheme", ["cam6_clubb", "boutle2014"])
def test_mg2_receives_the_selected_relvar(monkeypatch, scheme):
    """Spy on both ends: under boutle2014 Morrison gets Boutle's relvar from the
    cloud fraction CLUBB wrote in the same sub-step and the cell area; under the
    default it gets CLUBB's own relvar, unchanged."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "unit"))
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import integration as ti
    from test_physics_macmic import _moist_setup
    written_cf, written_rv, read = [], [], []
    real_upd = ti._carry_update_with_cloud_fraction
    real_micro = mi.morrison_microphysics

    def upd(carry_field, carry_val, turb_out):
        # this dry test state makes no cloud: hand on a fixed cloud fraction (0 to 1
        # across the cells) as CLUBB's, so the two relvar sources can differ
        cf0 = jnp.asarray(turb_out.cloud_fraction)
        cf = jnp.broadcast_to(jnp.linspace(0.0, 1.0, cf0.shape[0])[:, None], cf0.shape)
        turb_out = turb_out._replace(cloud_fraction=cf)
        written_cf.append(np.asarray(turb_out.cloud_fraction))
        written_rv.append(None if turb_out.relvar is None else np.asarray(turb_out.relvar))
        return real_upd(carry_field, carry_val, turb_out)

    def micro(*a, relvar=None, **k):
        read.append(None if relvar is None else np.asarray(relvar))
        return real_micro(*a, relvar=relvar, **k)
    monkeypatch.setattr(ti, "_carry_update_with_cloud_fraction", upd)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    mesh, sigma, state = _moist_setup()
    tr = dict(state.tracers)
    tr["N_s"] = state.T.replace(data=jnp.zeros_like(state.T.data), name="N_s", units="1/kg")
    state = state._replace(tracers=tr)
    fn, cfg = _physics(warm_rain_incloud=True, relvar_scheme=scheme, boutle_fsd_factor=1.7)
    ps = init_physics_state(*state.T.data.shape, cfg)
    fn(state, mesh, sigma, phys_state=ps)
    assert len(read) == len(written_cf) == 2
    for cf, rv, r in zip(written_cf, written_rv, read):
        if scheme == "boutle2014":
            # jitted, as in the scanned sub-cycle (this mesh's areaCell is float32;
            # before the cell size was cast to the state's precision, eager and
            # compiled float32 arithmetic gave values 5e-8 apart)
            width = cfg.microphysics.morrison.boutle_overcast_ramp_width
            want = jax.jit(lambda c_, a_: boutle2014_relvar(c_, a_, 1.7, width))(
                jnp.clip(cf.reshape(r.shape), 0.0, 1.0),
                jnp.asarray(mesh.areaCell).reshape(-1, 1))
            # the model fuses this inside the scanned sub-cycle; XLA's fusion
            # moves the last bit (measured 7.4e-16 relative), so compare to a
            # few ulp rather than bitwise
            np.testing.assert_allclose(np.asarray(want), r, rtol=1e-14, atol=0.0)
        else:
            np.testing.assert_array_equal(rv.reshape(r.shape), r)
    # discriminating: the two sources differ on this state
    if scheme == "boutle2014":
        assert not np.array_equal(written_rv[0].reshape(read[0].shape), read[0])


# RULE 5 smoothing of the C = 1 jump: stored per-case gap of the default
# (boutle_overcast_ramp_width = 0.01) against the published hard form, as
# fsd(default) / fsd(hard) - 1 (Phi cancels, so the gap is x independent).
_OVERCAST_GAP = {
    0.5: 0.0, 0.985: 0.0, 0.99: 0.0,
    0.992: -0.04736633663366352,
    0.995: -0.22670807453416175,
    0.998: -0.4044289276807984,
    0.9999: -0.4499346209223848,
    1.0: 0.0,
}


def test_overcast_ramp_default_is_the_config_default():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert MorrisonConfig().boutle_overcast_ramp_width == 0.01


@pytest.mark.parametrize("c", sorted(_OVERCAST_GAP))
def test_overcast_ramp_default_gap_per_case(c):
    a = _area(25.0)
    hard = float(boutle2014_fsd(jnp.asarray(c), a, 0.0))
    smooth = float(boutle2014_fsd(jnp.asarray(c), a, 0.01))
    np.testing.assert_allclose(smooth / hard - 1.0, _OVERCAST_GAP[c], rtol=1e-6, atol=1e-12)


def test_overcast_ramp_hard_limit_is_the_published_jump():
    a = _area(25.0)
    below = float(boutle2014_fsd(jnp.asarray(1.0 - 1e-12), a, 0.0))
    at = float(boutle2014_fsd(jnp.asarray(1.0), a, 0.0))
    np.testing.assert_allclose(below / at, 0.20 / 0.11, rtol=1e-9)


def test_overcast_ramp_gradient_through_the_band():
    # 100 km cells: the relvar stays below relvarmax across the band, so the
    # check is not run on the floored, flat branch
    a = jnp.full(3, _area(100.0))
    c = jnp.asarray([0.991, 0.994, 0.997])
    assert np.all(np.asarray(boutle2014_relvar(c, a, 1.0, 0.01)) < 10.0)
    check_grads(lambda cc: boutle2014_relvar(cc, a, 1.0, 0.01), (c,),
                order=2, modes=("rev",), eps=1e-6)
    g = jax.grad(lambda cc: jnp.sum(boutle2014_fsd(cc, a, 0.01)))(c)
    assert np.all(np.isfinite(np.asarray(g))) and np.all(np.asarray(g) != 0.0)


def test_driver_flags_reach_the_morrison_leaf():
    """--morrison-relvar-scheme / -boutle-fsd-factor / -boutle-overcast-ramp-width
    round-trip through ExperimentConfig onto MorrisonConfig; defaults leave the
    leaf untouched (the same object)."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args)
    base = MorrisonConfig()
    for f, leaf in (("morrison_relvar_scheme", "relvar_scheme"),
                    ("morrison_boutle_fsd_factor", "boutle_fsd_factor"),
                    ("morrison_boutle_overcast_ramp_width", "boutle_overcast_ramp_width")):
        assert ExperimentConfig._field_defaults[f] == getattr(base, leaf)
    parser = build_arg_parser()
    plain = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--microphysics", "morrison"]), parser))
    assert thread_morrison_scalars(plain, "morrison", base) is base
    on = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--microphysics", "morrison",
         "--morrison-relvar-scheme", "boutle2014",
         "--morrison-boutle-fsd-factor", "1.5",
         "--morrison-boutle-overcast-ramp-width", "0.02"]), parser))
    m = thread_morrison_scalars(on, "morrison", base)
    assert (m.relvar_scheme, m.boutle_fsd_factor, m.boutle_overcast_ramp_width) == (
        "boutle2014", 1.5, 0.02)
    with pytest.raises(ValueError, match="morrison_relvar_scheme"):
        on._replace(morrison_relvar_scheme="nope").validate_strict()
