"""zdf_sh2's step-entry r3u/r3v stored before the "1 +" (V1-selected): wiring;
and r3u formed as NEMO's (num * r1_hu_0) * r1_e1e2u, bit-for-bit under jit.

The FMA the flag removes forms only inside the compiled full step, never in
this builder compiled alone; the bit-for-bit receipt is
tests/ocean/fidelity/test_nemo_testcase_v1_sh2_r3_guard.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

pytestmark = pytest.mark.skipif(not jax.config.x64_enabled,
                                reason="binary64 receipt")

NY, NX, NZ = 30, 40, 3


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


def _case(flag, stage="step_entry", metric="nemo_qco_live_face", recip=False,
          inputs=False):
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses,
    )

    rng = np.random.default_rng(3)
    a_t, a_u, a_v = (rng.uniform(1e9, 2e10, (NY, NX)) for _ in range(3))
    hu, hv = (rng.uniform(5.0, 50.0, (NY, NX)) for _ in range(2))
    raw = create_z_star_from_thicknesses(
        [1.0, 2.0, 3.0],
        nemo_gdept_0_m=np.broadcast_to([0.5, 2.0, 4.5], (NY, NX, NZ)).copy(),
        nemo_gdepw_0_m=np.broadcast_to([0.0, 1.0, 3.0], (NY, NX, NZ)).copy(),
        nemo_e3t_0_m=np.broadcast_to([1.0, 2.0, 3.0], (NY, NX, NZ)).copy(),
        nemo_e3w_0_m=np.broadcast_to([1.0, 1.5, 2.5], (NY, NX, NZ)).copy(),
        nemo_hu_0_m=hu, nemo_hv_0_m=hv, nemo_e1e2t_m=a_t,
        nemo_e1e2u_m=a_u, nemo_e1e2v_m=a_v)
    z = create_partial_cell_coordinate(raw, jnp.full((NY, NX), raw.H_max))
    tke = TKEConfig(tke_shear_evaluation_stage=stage,
                    tke_shear_production="nemo_face_native_nbb2",
                    tke_shear_avm_weighting="nemo_face",
                    tke_shear_metric_source=metric,
                    nemo_sh2_r3_source_rounded=flag,
                    nemo_sh2_r3_stored_reciprocal=recip)
    cfg = SimpleNamespace(physics=SimpleNamespace(
        vertical_mixing=SimpleNamespace(scheme="tke", tke=tke)))
    model = SimpleNamespace(config=cfg, z_coord=z,
                            grid=SimpleNamespace(fold=None))
    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    u = jnp.asarray(rng.uniform(-1, 1, (NY, NX + 1, NZ)))
    v = jnp.asarray(rng.uniform(-1, 1, (NY + 1, NX, NZ)))
    avm = jnp.asarray(rng.uniform(0.1, 0.9, (NY, NX, NZ - 1)))

    def metrics(eta):
        state = SimpleNamespace(
            T=Field(jnp.zeros((NY, NX, NZ)), "T", d3, "degC"),
            H_bathy=Field(jnp.full((NY, NX), raw.H_max), "H_bathy", d2, "m"),
            eta=Field(eta, "eta", d2, "m"), u=Field(u, "u", d3, "m/s"),
            v=Field(v, "v", d3, "m/s"), tke_avm=Field(avm, "tke_avm", d3, "m2/s"))
        return LatLonCGridOceanModel._tke_step_entry_p_sh2(
            model, state, return_face_metrics=True)[1]

    eta = rng.uniform(-2.0, 2.0, (NY, NX))
    out = jax.jit(metrics)(jnp.asarray(eta))
    return (out, dict(a_t=a_t, a_u=a_u, a_v=a_v, hu=hu, hv=hv, eta=eta)) if inputs else out


def _nemo_e3u_now(d):
    """domqco.f90:266-267 then zdfsh2.f90:104, one rounding per operation."""
    nu = 0.5 * (d["a_t"] * d["eta"] + np.roll(d["a_t"] * d["eta"], -1, axis=1))
    wet = (d["hu"] > 0.0).astype(float)
    r3u = (nu * (wet / (d["hu"] + 1.0 - wet))) * (1.0 / d["a_u"])
    r3u = np.concatenate([r3u[:, -1:], r3u], axis=1)
    return np.asarray([1.5, 2.5]) * (1.0 + r3u[..., None])


def test_stored_reciprocal_is_nemo_bits_under_jit():
    got, d = _case(True, recip=True, inputs=True)
    assert np.array_equal(np.asarray(got[0]), _nemo_e3u_now(d))


def test_divide_form_is_not_nemo_bits_under_jit():
    got, d = _case(True, recip=False, inputs=True)
    assert np.count_nonzero(np.asarray(got[0]) != _nemo_e3u_now(d)) > 0


def test_flag_defaults_off():
    assert TKEConfig().nemo_sh2_r3_source_rounded is False
    assert TKEConfig().nemo_sh2_r3_stored_reciprocal is False


def test_flag_refuses_paths_that_would_ignore_it():
    with pytest.raises(ValueError, match="read only by the step-entry"):
        _case(True, stage="implicit_solve_state")
    with pytest.raises(ValueError, match="requires tke_shear_metric_source"):
        _case(True, metric="tpoint_jacobian")
    with pytest.raises(ValueError, match="read only by the step-entry"):
        _case(False, stage="implicit_solve_state", recip=True)
    with pytest.raises(ValueError, match="stored_reciprocal=True requires"):
        _case(False, metric="tpoint_jacobian", recip=True)
