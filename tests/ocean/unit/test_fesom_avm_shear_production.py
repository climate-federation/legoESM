"""NEMO zdf_sh2 on FESOM element velocities
(``fesom_integration.fesom_avm_weighted_shear_production`` and its wiring
into the FESOM TKE bridge under ``tke_shear_avm_weighting="nemo_face"``)."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from legoesm.ocean.physics.vertical_mixing.fesom_integration import (
    fesom_avm_weighted_shear_production,
)

# Three triangles sharing node 2; four interfaces.
ELEM = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
AREA = np.array([1.0, 2.0, 3.0])
NOD, NLEV = 5, 5


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    uv = rng.standard_normal((3, NLEV, 2))
    dz = 1.0 + rng.random((NOD, NLEV - 1))
    wet = np.ones((3, NLEV - 1), bool)
    return uv, dz, wet


def _reference(uv, dz, wet, avm):
    """Loop transcription: per node, area-weighted mean over wet elements of
    mean-vertex-avm * |du|^2 / mean-vertex-dz^2."""
    out = np.zeros((NOD, NLEV - 1))
    for n in range(NOD):
        for k in range(NLEV - 1):
            num = den = 0.0
            for e, nodes in enumerate(ELEM):
                if n in nodes and wet[e, k]:
                    du = uv[e, k] - uv[e, k + 1]
                    s2 = float(du @ du) / np.mean(dz[nodes, k]) ** 2
                    num += AREA[e] * np.mean(avm[nodes, k]) * s2
                    den += AREA[e]
            out[n, k] = num / den if den > 0 else 0.0
    return out


def _call(uv, dz, wet, avm):
    return np.asarray(fesom_avm_weighted_shear_production(
        jnp.asarray(uv), jnp.asarray(dz), jnp.asarray(ELEM), jnp.asarray(AREA),
        jnp.asarray(wet), jnp.asarray(avm)))


def test_matches_loop_transcription_with_varying_avm_and_dry_element():
    uv, dz, wet = _inputs()
    wet[1, 2:] = False                       # element 1 dry below interface 2
    avm = 1e-3 * (1.0 + np.arange(NOD))[:, None] * np.ones((1, NLEV - 1))
    np.testing.assert_allclose(_call(uv, dz, wet, avm),
                               _reference(uv, dz, wet, avm), rtol=1e-12)


def test_uniform_avm_is_avm_times_node_mean_shear():
    uv, dz, wet = _inputs(1)
    one = np.ones((NOD, NLEV - 1))
    s2_mean = _call(uv, dz, wet, one)
    np.testing.assert_allclose(_call(uv, dz, wet, 3e-3 * one), 3e-3 * s2_mean,
                               rtol=1e-12)
    vary = one.copy(); vary[0] = 50.0        # contrast at node 0 only
    assert not np.allclose(_call(uv, dz, wet, vary), s2_mean)


def test_node_with_no_wet_element_gets_zero():
    uv, dz, wet = _inputs(2)
    wet[:, 3] = False
    assert np.all(_call(uv, dz, wet, np.ones((NOD, NLEV - 1)))[:, 3] == 0.0)


def test_nemo_face_reaches_the_fesom_kernel():
    """With sheared element velocities, nemo_face (+ nemo_ri, the card pair)
    must change the FESOM closure's K vs the tpoint card; nemo_face without
    the lane's p_sh2 would have raised in the kernel."""
    pytest.importorskip("fesom_jax")
    from fesom_jax.mesh import DEFAULT_PI_MESH_DIR, load_mesh
    from fesom_jax.pp import compute_vel_nodes
    from legoesm.ocean.dynamics.ocean_model_fesom import (
        build_flat_bottom_mesh, create_rest_state,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.vertical_mixing.fesom_integration import (
        fesom_zgeom, make_tke_profiles_fesom,
    )
    from legoesm.ocean.state import OceanSurfaceForcing
    from scripts.run.run_omip_core2 import orca1_zdftke_config
    mesh = build_flat_bottom_mesh(load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR),
                                  H_max=200.0, nlev=20)
    z = np.asarray(mesh.Z, dtype=np.float64)
    state = create_rest_state(mesh, SimpleNamespace(z_full_ref=z, n_levels=int(z.size)),
                              stratified=True, vertical_coordinate="zstar")
    rng = np.random.default_rng(3)
    uv = jnp.asarray(state.inner.uv) + 0.05 * jnp.asarray(
        rng.standard_normal(np.asarray(state.inner.uv).shape))
    uv = jnp.where(jnp.asarray(mesh.elem_layer_mask)[..., None], uv, 0.0)
    state = dataclasses.replace(
        state, inner=dataclasses.replace(state.inner, uv=uv),
        uv_node=jnp.asarray(compute_vel_nodes(mesh, uv), dtype=jnp.float64))
    n = int(mesh.nod2D)
    f = OceanSurfaceForcing(tau_x=jnp.full((n,), 0.1), tau_y=jnp.zeros((n,)),
                            q_net=jnp.zeros((n,)), sw_down=jnp.zeros((n,)))

    def run(w):
        tke = orca1_zdftke_config(prognostic=True)._replace(
            eice=0, prandtl_mode="nemo_ri", tke_shear_avm_weighting=w)
        return make_tke_profiles_fesom(VerticalMixingConfig(scheme="tke", tke=tke))(
            state, mesh, fesom_zgeom(mesh), f, dt_tke=900.0)
    Kv_t, Av_t, _ = run("tpoint")
    Kv_f, Av_f, _ = run("nemo_face")
    assert bool(jnp.all(jnp.isfinite(Kv_f)))
    assert float(jnp.max(jnp.abs(Kv_f - Kv_t))) > 1e-3 * float(jnp.max(Kv_t))
