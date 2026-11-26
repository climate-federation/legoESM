from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round113_een_south_e3f_walk as gate,
)


def test_southern_copy_fill_differs_from_cyclic_mesh_thickness():
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_south_copy_fill,
    )

    field = jnp.asarray([
        [[1.0], [2.0]],
        [[3.0], [4.0]],
        [[8.0], [9.0]],
    ])
    copied = np.asarray(_nemo_south_copy_fill(field))
    cyclic = np.roll(np.asarray(field), 1, axis=0)
    np.testing.assert_array_equal(copied[0], np.asarray(field)[0])
    assert not np.array_equal(copied, cyclic)


def test_literal_builder_executes_southern_mesh_thickness_copy(monkeypatch):
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    from legoesm.ocean.vertical import NemoEENBarotropicOperands

    ny, nx, nz = 3, 4, 2
    shape2, shape3 = (ny, nx), (ny, nx, nz)
    ones2 = np.ones(shape2)
    ones3 = np.ones(shape3)
    mesh_e3f = np.broadcast_to(
        np.asarray([2.0, 4.0, 9.0])[:, None, None], shape3).copy()
    raw = NemoEENBarotropicOperands(
        ff_f=jnp.asarray(ones2),
        e3u_0=jnp.asarray(ones3), e3v_0=jnp.asarray(ones3),
        e3f_0=jnp.asarray(mesh_e3f), umask=jnp.asarray(ones3),
        vmask=jnp.asarray(ones3), fmask=jnp.asarray(ones3),
        fe3mask=jnp.asarray(ones3),
        hu_0=jnp.full(shape2, nz), hv_0=jnp.full(shape2, nz),
        hf_0=jnp.full(shape2, nz),
        e1t=jnp.asarray(ones2), e2t=jnp.asarray(ones2),
        e1u=jnp.asarray(ones2), e2u=jnp.asarray(ones2),
        e1v=jnp.asarray(ones2), e2v=jnp.asarray(ones2),
        e1f=jnp.asarray(ones2), e2f=jnp.asarray(ones2),
    )
    z_coord = SimpleNamespace(
        nemo_een_barotropic=raw,
        nemo_e3t_0=jnp.asarray(ones3),
        is_active=jnp.asarray(ones3),
    )
    eta = jnp.zeros(shape2, dtype=jnp.float64)
    faithful = bt._nemo_literal_een_coefficients(
        eta, z_coord, jnp.float64, scheme="een")
    monkeypatch.setattr(
        bt, "_nemo_een_south_e3f0",
        lambda e3f0, mesh_e3f0: jnp.roll(e3f0, 1, axis=0),
    )
    cyclic = bt._nemo_literal_een_coefficients(
        eta, z_coord, jnp.float64, scheme="een")
    assert any(
        not np.array_equal(np.asarray(faithful[name]), np.asarray(cyclic[name]))
        for name in faithful
    )


def test_round113_plants_and_next_boundary_are_registered():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "scope-route")
    assert gate.r111.SOURCE_ORDER[12:18] == (
        "south_ff", "south_e3f0", "south_r3f", "south_mask",
        "south_denom", "frac_south",
    )
