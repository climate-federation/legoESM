"""``step_ponds(scheme='si3_lev')`` is one category per call, on any spatial layout."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.ice.config import SI3PondConfig  # noqa: E402
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG  # noqa: E402
from legoesm.ice.ponds import step_ponds  # noqa: E402

POND = SI3PondConfig(rn_himin=0.1, rn_apnd_min=0.15, rn_apnd_max=0.85, rn_pnd_flush=0.1, ln_pnd_lids=True)


def _fields(shape, n_layers=2, seed=0):
    rng = np.random.default_rng(seed)
    u = lambda a, b, s=shape: jnp.asarray(rng.uniform(a, b, s))  # noqa: E731
    return dict(a=u(0.2, 0.3), d=u(0.05, 0.2), h=u(1.0, 2.0), ts=u(-5.0, -1.0), hil=u(0.0, 0.01),
                dhi=-u(0.01, 0.03), dhs=-u(0.001, 0.01), sz=u(3.0, 6.0, shape + (n_layers,)),
                ti=u(263.0, 270.0, shape + (n_layers,)))


def _call(f, at=None, rain=0.0):
    a = f["a"]
    z = jnp.zeros_like(a)
    return step_ponds(
        0.2 + z, f["d"], melt_water_m=-f["dhi"], rain_water_m=rain + z, ice_mask=a > 0, h_snow=z,
        T_air=z + 272.0, dt=3600.0, drainage_timescale=1e5, refreeze_threshold=270.0,
        pond_to_ice_max_area=0.5, depth_to_area_ratio=0.8, scheme="si3_lev",
        at_i=a if at is None else at, a_i=a, h_i=f["h"], t_su=f["ts"], h_il=f["hil"], dh_i_sum=f["dhi"],
        dh_s_sum=f["dhs"], sz_i=f["sz"], t_i=f["ti"], namthd_pnd=POND, rho_w=constants.rho_water,
        gravity=constants.g_nemo, ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )


def test_cube_fields_match_face_by_face():
    f = _fields((6, 4, 4))
    cube = _call(f)
    faces = [_call({k: v[i] for k, v in f.items()}) for i in range(6)]
    for j, out in enumerate(cube):
        assert out.shape == (6, 4, 4)
        np.testing.assert_array_equal(out, jnp.stack([x[j] for x in faces]))


def test_mpas_columns_match_a_two_dimensional_grid():
    f = _fields((12,))
    grid = {k: v.reshape((3, 4) + v.shape[1:]) for k, v in f.items()}
    for flat, block in zip(_call(f), _call(grid)):
        assert flat.shape == (12,)
        np.testing.assert_array_equal(flat, block.reshape(-1))


def test_a_category_axis_is_refused():
    f = _fields((3, 4, 5))
    with pytest.raises(ValueError, match="one category per call"):
        _call(f, at=f["a"].sum(-1))


def test_layer_fields_on_another_layout_are_refused():
    f = _fields((6, 4, 4))
    with pytest.raises(ValueError, match="requires sz_i and t_i shape"):
        _call({**f, "sz": f["sz"][:, :, :3], "ti": f["ti"][:, :, :3]})


def test_level_ponds_ignore_rain():
    f = _fields((4, 5))
    for dry, wet in zip(_call(f, rain=0.0), _call(f, rain=0.01)):
        np.testing.assert_array_equal(dry, wet)
