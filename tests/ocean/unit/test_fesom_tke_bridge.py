"""legoESM (NEMO-card) TKE closure on FESOM node columns
(``vertical_mixing/fesom_integration.py`` + the fesom_jax ``vertical_mixing``
injection seam + ``FesomOceanConfig.vertical_mixing='legoesm_tke'``).

Layout contracts pinned here: fesom ``Kv[:, k]`` is the interface ABOVE layer
``k`` (``tracer_diff.py:39-40``), so ``Kv[:, 0] == 0`` (no flux through the
surface), ``Kv[:, 1:nlev]`` carries the closure's ``nlev-1`` half-levels and
everything at/below the pad slot is 0; ``Av`` is the node ``K_M`` averaged
onto elements; prognostic TKE rides in ``inner.tke`` with the same indexing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

fesom_jax = pytest.importorskip("fesom_jax")
import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.state import OceanSurfaceForcing  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_fesom import (  # noqa: E402
    FesomOceanConfig,
    FesomOceanModel,
    build_flat_bottom_mesh,
    create_rest_state,
)
from legoesm.ocean.physics.vertical_mixing.config import (  # noqa: E402
    TKEConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.fesom_integration import (  # noqa: E402
    fesom_zgeom,
    make_tke_profiles_fesom,
)

DT = 900.0
NLEV = 6
H_MAX = 600.0
TAU = 0.1


@pytest.fixture(scope="module")
def flat_mesh():
    try:
        from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
        pi = load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"packaged pi mesh unavailable: {exc}")
    return build_flat_bottom_mesh(pi, H_max=H_MAX, nlev=NLEV)


@pytest.fixture(scope="module")
def z_shim(flat_mesh):
    z = np.asarray(flat_mesh.Z, dtype=np.float64)
    return SimpleNamespace(z_full_ref=z, n_levels=int(z.size))


def _vmix():
    return VerticalMixingConfig(scheme="tke", tke=TKEConfig(prognostic=True))


def _model(flat_mesh, z_shim, mode="legoesm_tke", vmix="auto"):
    cfg = FesomOceanConfig(dt=DT, vertical_coordinate="zstar",
                           constants="fesom", vertical_mixing=mode)
    return FesomOceanModel(flat_mesh, z_shim, cfg,
                           vmix_config=(_vmix() if vmix == "auto" else vmix))


def _wind(mesh):
    n = int(mesh.nod2D)
    return OceanSurfaceForcing(tau_x=jnp.full((n,), TAU), tau_y=jnp.zeros((n,)),
                               q_net=jnp.zeros((n,)), sw_down=jnp.zeros((n,)))


def test_zgeom_ladders(flat_mesh):
    zg = fesom_zgeom(flat_mesh)
    assert zg.n_levels == NLEV
    np.testing.assert_allclose(float(jnp.sum(zg.dz_ref)), H_MAX, rtol=1e-10)
    assert zg.dz_half_ref.shape == (NLEV - 1,)
    assert float(zg.t_depth_ref[0]) > 0.0 and float(zg.w_depth_ref[0]) == 0.0


def test_profiles_layout_and_wind_response(flat_mesh, z_shim):
    model = _model(flat_mesh, z_shim)
    state = create_rest_state(flat_mesh, z_shim, stratified=True,
                              vertical_coordinate="zstar")
    zg = fesom_zgeom(flat_mesh)
    fn = make_tke_profiles_fesom(_vmix())
    Kv, Av, tke = fn(state, flat_mesh, zg, _wind(flat_mesh), dt_tke=DT)
    n, e, nl = int(flat_mesh.nod2D), int(flat_mesh.elem2D), int(flat_mesh.nl)
    assert Kv.shape == (n, nl) and tke.shape == (n, nl) and Av.shape == (e, nl)
    assert np.isfinite(np.asarray(Kv)).all() and np.isfinite(np.asarray(Av)).all()
    assert float(jnp.abs(Kv[:, 0]).max()) == 0.0          # surface: no flux
    assert float(jnp.abs(Kv[:, NLEV:]).max()) == 0.0      # at/below bottom
    assert float(Kv.min()) >= 0.0 and float(Av.min()) >= 0.0
    # wind-driven surface TKE: the first interior interface mixes harder than
    # the closure's background floor, and so does the element viscosity.
    tk = TKEConfig()
    assert float(jnp.median(Kv[:, 1])) > tk.kappaH_min * 1.5
    assert float(jnp.median(tke[:, 1])) > tk.tke_background
    # calm ocean: nothing above the floors
    Kv0, _, _ = fn(state, flat_mesh, zg, None, dt_tke=DT)
    assert float(jnp.median(Kv0[:, 1])) <= float(jnp.median(Kv[:, 1]))
    # the model carries the same function
    assert model._tke_profiles_fn is not None


def test_model_step_injects_and_carries_tke(flat_mesh, z_shim):
    model = _model(flat_mesh, z_shim)
    state = create_rest_state(flat_mesh, z_shim, stratified=True,
                              vertical_coordinate="zstar")
    assert float(jnp.abs(jnp.asarray(state.inner.tke)).max()) == 0.0
    s1 = model.step(state, DT, surface_forcing=_wind(flat_mesh))
    assert np.isfinite(np.asarray(s1.inner.T)).all()
    tke1 = np.asarray(s1.inner.tke)
    assert tke1.shape == (int(flat_mesh.nod2D), int(flat_mesh.nl))
    assert float(np.abs(tke1[:, 0]).max()) == 0.0 and float(tke1[:, 1].max()) > 0.0
    # fesom's own closure leaves inner.tke untouched (its TKE scheme is off)
    m0 = _model(flat_mesh, z_shim, mode="fesom", vmix=None)
    s0 = m0.step(state, DT, surface_forcing=_wind(flat_mesh))
    assert float(np.abs(np.asarray(s0.inner.tke)).max()) == 0.0
    assert not np.allclose(np.asarray(s0.inner.T), np.asarray(s1.inner.T))


def test_config_guards(flat_mesh, z_shim):
    with pytest.raises(ValueError, match="vmix_config"):
        _model(flat_mesh, z_shim, mode="legoesm_tke", vmix=None)
    with pytest.raises(ValueError, match="silently ignored"):
        _model(flat_mesh, z_shim, mode="fesom", vmix=_vmix())
    with pytest.raises(ValueError, match="vertical_mixing"):
        _model(flat_mesh, z_shim, mode="cvmix", vmix=None)
    with pytest.raises(NotImplementedError, match="PROGNOSTIC"):
        make_tke_profiles_fesom(
            VerticalMixingConfig(scheme="tke", tke=TKEConfig(prognostic=False)))
    with pytest.raises(ValueError, match="scheme='tke'"):
        make_tke_profiles_fesom(VerticalMixingConfig(scheme="none"))
