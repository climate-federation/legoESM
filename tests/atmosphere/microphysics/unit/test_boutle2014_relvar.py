"""Boutle et al. (2014) liquid FSD as the MG2 relvar (relvar_scheme="boutle2014").

Equation (Boutle et al. 2014, QJRMS 140, 583-594, as restated in Crnivec and
Mayer 2021, GMD 14, 3663, eq. 8): FSD = (0.45 - 0.25 C) Phi for C < 1,
0.11 Phi for C = 1, Phi = (x C)^(1/3) ((0.06 x C)^1.5 + 1)^(-0.17), x in km.
"""
import sys

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
    np.testing.assert_allclose(float(boutle2014_relvar(1.0, _area(25.0), 1.0)), 13.7795, rtol=1e-4)
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
    with pytest.raises(ValueError, match="warm_rain_incloud=True"):
        _physics(warm_rain_incloud=False, relvar_scheme="boutle2014")


@pytest.mark.parametrize("scheme", ["cam6_clubb", "boutle2014"])
def test_mg2_receives_the_selected_relvar(monkeypatch, scheme):
    """Spy on both ends: under boutle2014 Morrison gets Boutle's relvar from the
    cloud fraction CLUBB wrote in the same sub-step and the cell area; under the
    default it gets CLUBB's own relvar, unchanged."""
    sys.path.insert(0, "tests/unit")
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
            want = boutle2014_relvar(jnp.clip(cf.reshape(r.shape), 0.0, 1.0),
                                     jnp.asarray(mesh.areaCell).reshape(-1, 1), 1.7)
            np.testing.assert_array_equal(np.asarray(want), r)
        else:
            np.testing.assert_array_equal(rv.reshape(r.shape), r)
    # discriminating: the two sources differ on this state
    if scheme == "boutle2014":
        assert not np.array_equal(written_rv[0].reshape(read[0].shape), read[0])
