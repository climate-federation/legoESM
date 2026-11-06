"""All-g-point gas optics equal the per-g-point functions, stage by stage and end to end."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream as TS

N, L = 6, 24


def _columns(dtype=jnp.float64):
    ph = jnp.tile(jnp.linspace(225.0, 1.0e5, L + 1, dtype=dtype), (N, 1))

    pf = 0.5 * (ph[:, 1:] + ph[:, :-1])
    # The coldest columns sit below the table's 160 K, the hottest above 355 K.
    t = 150.0 + 210.0 * (pf / 1e5) * (0.6 + 0.08 * jnp.arange(N, dtype=dtype))[:, None]
    on = (jnp.arange(N) % 2 == 0)[:, None]
    return t, dict(
        p_full=pf, p_half=ph, q_v=1e-2 * (pf / 1e5) ** 3,
        cloud_path_liq=jnp.where(on & (pf > 8.5e4), 0.05, 0.0).astype(dtype),
        cloud_path_ice=jnp.where(on & (pf > 2e4) & (pf < 3e4), 0.01, 0.0).astype(dtype),
        cloud_r_eff_liq=jnp.full_like(pf, 1e-5), cloud_r_eff_ice=jnp.full_like(pf, 4e-5),
        cos_zenith=jnp.linspace(0.05, 0.95, N, dtype=dtype), sfc_albedo=0.1, sfc_emissivity=0.98)


def _solve(mode, t, kw, fp32=False, batch=16):
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        compute_fp32=fp32, overhead_layer=True, use_scan=True, include_clouds=True,
        gpoint_batch_size=batch, gas_optics_mode=mode))
    return solver.solve_columns(T=t, sfc_temperature=t[:, -1], **kw)


@pytest.fixture(scope="module")
def state():
    """The optics and cell state the solver really hands to the gas optics."""
    seen = {}
    orig = TS.solve_lw

    def spy(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields=None, *a, **k):
        seen.update(p=pressure, t=temperature, mol=molecules, lib=optics_lib, vmr=vmr_fields)
        return orig(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields, *a, **k)

    TS.solve_lw = spy
    try:
        t, kw = _columns()
        _solve("per_gpt", t, kw)
    finally:
        TS.solve_lw = orig
    return seen


@pytest.mark.parametrize("is_lw", [True, False])
def test_optical_depth_matches_every_gpoint(state, is_lw):
    lib, s = state["lib"], state
    lookup = lib.gas_optics_lw if is_lw else lib.gas_optics_sw
    n = lib.n_gpt_lw if is_lw else lib.n_gpt_sw
    allg = gas_optics.optical_depth_all_gpts(lookup, lib.vmr_lib, s["mol"], s["t"], s["p"], s["vmr"])
    ref = jax.vmap(lambda g: lib._od_fn(is_lw, g, s["mol"], s["t"], s["p"], s["vmr"]))(jnp.arange(n))
    np.testing.assert_allclose(allg, ref, rtol=1e-12, atol=1e-300)


def test_planck_fraction_and_rayleigh_match_every_gpoint(state):
    lib, s = state["lib"], state
    pf = gas_optics.planck_fraction_all_gpts(lib.gas_optics_lw, lib.vmr_lib, s["p"], s["t"], s["vmr"])
    ref = jax.vmap(lambda g: lib._pf_fn(g, s["p"], s["t"], s["vmr"]))(jnp.arange(lib.n_gpt_lw))
    np.testing.assert_allclose(pf, ref, rtol=1e-12, atol=1e-300)
    ry = gas_optics.rayleigh_optical_depth_all_gpts(
        lib.gas_optics_sw, lib.vmr_lib, s["mol"], s["t"], s["p"], s["vmr"])
    ref = jax.vmap(lambda g: lib._rayl_fn(g, s["mol"], s["t"], s["p"], s["vmr"]))(jnp.arange(lib.n_gpt_sw))
    np.testing.assert_allclose(ry, ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("batch", [16, 0])
def test_fluxes_match_end_to_end(batch):
    t, kw = _columns()
    a, b = _solve("per_gpt", t, kw, batch=batch), _solve("all_gpts", t, kw, batch=batch)
    np.testing.assert_allclose(a.heating_rate, b.heating_rate, rtol=0, atol=1e-13)


def test_gradient_matches_and_is_finite():
    t, kw = _columns()

    def grad(mode):
        return jax.grad(lambda x: jnp.sum(_solve(mode, x, kw).heating_rate))(t)

    ga, gb = grad("per_gpt"), grad("all_gpts")
    assert np.isfinite(gb).all() and np.abs(gb).max() > 0
    np.testing.assert_allclose(ga, gb, rtol=1e-9, atol=1e-13)


def test_unknown_mode_raises():
    t, kw = _columns()
    with pytest.raises(ValueError, match="gas_optics_mode"):
        _solve("nonsense", t, kw)


def test_stage_bench_inputs_and_stubs(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "bench_rrtmgp_stages", Path(__file__).parents[2] / "scripts/bench/bench_rrtmgp_stages.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    t, kw = bench.inputs(4, 8)
    assert t.shape == (4, 8) and kw["p_half"].shape == (4, 9)
    monkeypatch.setattr(gas_optics, "compute_minor_optical_depth", gas_optics.compute_minor_optical_depth)
    monkeypatch.setattr(gas_optics, "compute_major_optical_depth", gas_optics.compute_major_optical_depth)
    bench.ablate("minor")
    assert gas_optics.compute_minor_optical_depth is bench._stub_od
    assert gas_optics.compute_major_optical_depth is not bench._stub_od
