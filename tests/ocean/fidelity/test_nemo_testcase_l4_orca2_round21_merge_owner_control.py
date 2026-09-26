from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round21_merge_owner_control as control,
)


def _fixture():
    e3t = np.array([[[2.0], [4.0]], [[6.0], [8.0]]])
    active = np.ones_like(e3t)
    raw = SimpleNamespace(
        e3f_0=np.full_like(e3t, 3.0),
        e1t=np.ones((2, 2)),
        e2t=np.ones((2, 2)),
        e1f=np.ones((2, 2)),
        e2f=np.ones((2, 2)),
        hf_0=np.ones((2, 2)),
        fe3mask=np.ones_like(e3t),
    )
    z_coord = SimpleNamespace(
        nemo_een_barotropic=raw, nemo_e3t_0=e3t, is_active=active)
    grid = SimpleNamespace(fold=None)
    return z_coord, grid


def test_bridge_control_ignores_current_card_operands_and_can_fail():
    z_coord, grid = _fixture()
    eta = jnp.zeros((2, 2), dtype=jnp.float64)
    base = np.asarray(control.bridge_carried_vorticity_e3f(
        eta, z_coord, jnp.float64, grid=grid,
        e3t_0=jnp.full((2, 2, 1), 999.0),
        tmask=jnp.zeros((2, 2, 1)),
    ))
    repeated = np.asarray(control.bridge_carried_vorticity_e3f(
        eta, z_coord, jnp.float64, grid=grid,
        e3t_0=jnp.full((2, 2, 1), -999.0),
        tmask=jnp.ones((2, 2, 1)),
    ))
    planted = np.asarray(control.bridge_carried_vorticity_e3f(
        eta, z_coord, jnp.float64, grid=grid, plant_ulp=True))

    assert base.shape == (3, 3, 1)
    assert np.array_equal(base, repeated)
    assert np.count_nonzero(base.view(np.uint64) != planted.view(np.uint64)) > 0
