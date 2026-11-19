"""Regression for the O12 DH-to-DA heat-flux carry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import si3_thd_pnd as pnd_mod
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG

from tests.ice.unit import test_si3_o9_zdf_dh_temp as o9_fixture
from tests.ice.unit import test_si3_o10_salinity as o10_fixture


@pytest.fixture(autouse=True)
def _libm_cpu():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    with jax.default_device(jax.devices("cpu")[0]):
        yield
    set_policy(previous)


def _fixture():
    namelist = {
        **o9_fixture._deck(),
        **o10_fixture.NML,
        "ln_icedA": True,
        "rn_beta": 0.5,
        "rn_dmin": 8.0,
    }
    categories = o9_fixture._categories(seed=19)
    rng = np.random.default_rng(20261205)
    entries, category_fluxes = [], []
    for jl in range(5):
        cells = [categories[jl][i % len(categories[jl])] for i in range(36)]
        entry, fluxes = o9_fixture._entry(cells)
        entry = {k: v.reshape((6, 6) + v.shape[1:]) for k, v in entry.items()}
        entry.update(
            at_i=jnp.full((6, 6), 0.75),
            sst=jnp.full((6, 6), 0.2),
            o_i=jnp.full((6, 6), 10.0),
            hfx_err_dif=jnp.zeros((6, 6)),
            hfx_dif=jnp.zeros((6, 6)),
        )
        entries.append(entry)
        category_fluxes.append(
            pnd_mod.SI3DHFluxes(
                **{field: getattr(fluxes, field).reshape(6, 6) for field in fluxes._fields}
            )
        )
    a_ip = jnp.asarray(rng.uniform(0.02, 0.08, (6, 6, 5)))
    h_ip = jnp.asarray(rng.uniform(0.05, 0.14, (6, 6, 5)))
    h_il = jnp.asarray(rng.uniform(0.0, 0.02, (6, 6, 5)))
    a_i = np.stack([np.asarray(entry["a_i"]) for entry in entries], axis=-1)
    zero = jnp.zeros((6, 6))
    arrays = dict(
        category_entries=tuple(entries),
        category_fluxes=tuple(category_fluxes),
        sfx_bri=zero,
        sfx_lam=zero,
        hfx_thd=zero,
        wfx_lam=zero,
        a_ip=a_ip,
        v_ip=h_ip * a_ip,
        h_ip=h_ip,
        v_il=h_il * a_ip,
        h_il=h_il,
        a_ip_frac=jnp.asarray(np.asarray(a_ip) / np.maximum(a_i, 1.0e-10)),
        wfx_pnd=zero,
    )
    static = dict(
        namelist=namelist,
        rDt_ice=o10_fixture.RDT,
        r1_Dt_ice=o10_fixture.R1_RDT,
        r1_nlay_i=o10_fixture.R1_NLAY,
        r1_nlay_s=1.0 / 5.0,
        rho_i=o10_fixture.RHO_I,
        rho_s=o10_fixture.RHO_S,
        rho_w=o10_fixture.RHO_W,
        gravity=o10_fixture.GRAVITY,
        nn_hls=2,
        constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    return arrays, static


def test_o12_da_receives_dh_updated_hfx_thd(monkeypatch):
    """DH's heat-flux update reaches DA instead of the stale category-entry value."""
    arrays, static = _fixture()
    real_da = pnd_mod.si3_ice_thd_da
    da_calls = []

    def record_da(**kwargs):
        result = real_da(**kwargs)
        da_calls.append((kwargs, result))
        return result

    monkeypatch.setattr(pnd_mod, "si3_ice_thd_da", record_da)
    monkeypatch.setattr(pnd_mod, "si3_ice_thd_pnd", lambda **kwargs: None)
    with jax.disable_jit():
        out = pnd_mod.si3_ice_thd_o12_stages(**arrays, **static)

    dh_hfx = out.categories[0].o10.o9.dh.fluxes.hfx_thd
    stale_hfx = arrays["hfx_thd"]
    assert np.any(np.asarray(dh_hfx).view(np.uint64) != np.asarray(stale_hfx).view(np.uint64))
    np.testing.assert_array_equal(
        np.asarray(da_calls[0][0]["hfx_thd"]).view(np.uint64),
        np.asarray(dh_hfx).view(np.uint64),
    )
    stale_da = real_da(**{**da_calls[0][0], "hfx_thd": stale_hfx})
    assert np.any(
        np.asarray(da_calls[0][1].hfx_thd).view(np.uint64)
        != np.asarray(stale_da.hfx_thd).view(np.uint64)
    )
