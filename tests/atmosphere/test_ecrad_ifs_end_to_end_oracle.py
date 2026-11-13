"""ecRad lane end to end with RRTMGP gas optics: budgets, gradients, jit parity.

Inputs are the 49r1 fixture's; its fluxes no longer apply, since the lane's gas
optics are RRTMGP (user decision 2026-11-06).  Fortran coverage now sits per
section: solvers, generator and cloud optics in their own oracle tests, gas
optics through test_ecrad_ifs_rrtmgp_gas.py.  fp64 (JAX_ENABLE_X64=1).
"""

import math
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad
from legoesm.atmosphere.physics.radiation.ecrad_ifs import gas_optics as go
from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import compute_heating_rate
from tests.atmosphere.test_ecrad_ifs_mcica_solver_oracle import rel_err

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(_HERE, "oracle_fixtures", "ecrad_ifs49.npz")
TOL = 1.0e-12
# Port-vs-port bound of this field, a total-minus-direct difference: measured
# 5.6e-12 with RRTMGP gas optics.
_V53_JIT_TOL = {"canopy_flux_dn_diffuse_sw_surf": 1.0e-11}
HARD = math.inf


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def inputs(z):
    f = lambda k: np.asarray(z["in:" + k], np.float64)  # noqa: E731
    return dict(
        pressure_hl=f("pressure_hl"), temperature_hl=f("temperature_hl"),
        gas_mmr=np.asarray(z["dump:in_gas_mmr"], np.float64),
        skin_temperature=f("skin_temperature"), cos_sza=f("cos_solar_zenith_angle"),
        solar_irradiance=float(z["in:solar_irradiance"]), sw_albedo=f("sw_albedo"),
        sw_albedo_direct=f("sw_albedo_direct"), lw_emissivity=f("lw_emissivity"),
        cloud_fraction=f("cloud_fraction"), q_liquid=f("q_liquid"), q_ice=f("q_ice"),
        re_liquid=f("re_liquid"), re_ice=f("re_ice"), overlap_param=f("overlap_param"),
        fractional_std=f("fractional_std"),
    )


def run(z, sharpness=HARD, jit=False):
    inp = inputs(z)
    rlw, rsw = ecrad.draw_uniforms(np.asarray(z["in:iseed"]).astype(np.int64),
                                   inp["cloud_fraction"].shape[1])
    f = lambda i, a, b: ecrad.radiation(**i, uniforms_lw=a, uniforms_sw=b, sharpness=sharpness)  # noqa: E731
    out = (jax.jit(f) if jit else f)(inp, rlw, rsw)
    return {k: np.asarray(v) for k, v in out.items()}, None


@pytest.fixture(scope="module")
def z8():
    return load(FIXTURE)


@pytest.fixture(scope="module")
def hard8(z8):
    return run(z8)


def test_sw_layers_absorb_nonnegative(hard8):
    """No layer creates SW energy: net (dn - up) flux never increases downward
    (all-sky and clear; same check and bound as the solver test)."""
    out, _ = hard8
    for key in ("", "_clear"):
        fnet = out["flux_dn_sw" + key] - out["flux_up_sw" + key]
        absorbed = fnet[:, :-1] - fnet[:, 1:]
        scale = np.max(np.abs(fnet), axis=1, keepdims=True) + 1.0
        assert np.all(absorbed >= -TOL * scale), float(absorbed.min())


def test_heating_integral_equals_net_flux_divergence(z8, hard8):
    """Column integral of the heating rate = TOA net - surface net (LW, SW, clear).
    This is the telescoping identity of the flux -> heating conversion (it checks
    the conversion, not the solver); the solver energy checks are the absorption
    sign test above and the solver test's continuity/conservative-column tests."""
    out, _ = hard8
    dp = np.diff(inputs(z8)["pressure_hl"], axis=1)
    for up, dn in (("flux_up_lw", "flux_dn_lw"), ("flux_up_sw", "flux_dn_sw"),
                   ("flux_up_lw_clear", "flux_dn_lw_clear"), ("flux_up_sw_clear", "flux_dn_sw_clear")):
        fnet = out[dn] - out[up]
        hr = compute_heating_rate(jnp.asarray(fnet)[:, None, :], None,
                                  dp=jnp.pad(jnp.asarray(dp), ((0, 0), (0, 1)), constant_values=1.0)[:, None, :])
        hr = np.asarray(hr)[:, 0, :-1]
        integral = np.sum(hr * dp, axis=1) * constants.c_pd / constants.g
        expect = fnet[:, 0] - fnet[:, -1]
        scale = np.max(np.abs(fnet), axis=1) + 1.0
        assert np.all(np.abs(integral - expect) <= TOL * scale * dp.shape[1]), up


def test_toa_and_surface_consistency(z8, hard8):
    out, _ = hard8
    inp = inputs(z8)
    mu0, s0 = inp["cos_sza"], inp["solar_irradiance"]
    day = mu0 > 0.0
    # TOA: no downwelling LW; SW down = direct = S0 * mu0 by day, zero at night.
    for k in ("flux_dn_lw", "flux_dn_lw_clear"):
        assert np.all(out[k][:, 0] == 0.0), k
    for k in ("flux_dn_sw", "flux_dn_direct_sw", "flux_dn_sw_clear", "flux_dn_direct_sw_clear"):
        assert rel_err(out[k][day, 0], s0 * mu0[day]).max() <= TOL, k
        assert np.all(out[k][~day] == 0.0), k
    # Surface: band and canopy fluxes sum to the broadband surface flux.
    surf = {"": out["flux_dn_sw"][:, -1], "_direct": out["flux_dn_direct_sw"][:, -1]}
    for sfx, ref in surf.items():
        assert rel_err(out[f"spectral_flux_dn{sfx}_sw_surf"].sum(1), ref).max() <= TOL, sfx
        clr = out[f"flux_dn{sfx}_sw_clear"][:, -1]
        assert rel_err(out[f"spectral_flux_dn{sfx}_sw_surf_clear"].sum(1), clr).max() <= TOL, sfx
    canopy = out["canopy_flux_dn_diffuse_sw_surf"] + out["canopy_flux_dn_direct_sw_surf"]
    assert rel_err(canopy.sum(1), surf[""]).max() <= TOL
    assert rel_err(out["canopy_flux_dn_direct_sw_surf"].sum(1), surf["_direct"]).max() <= TOL
    # lw_derivative is d flux_up / d surface flux_up: exactly 1 at the surface.
    assert np.all(out["lw_derivative"][:, -1] == 1.0)


# --------------------------------------------------------------------------
# Differentiability (default sharpness, production uniforms)
# --------------------------------------------------------------------------

def test_grad_flux_functional(z8):
    """d(TOA + surface net fluxes)/d(T, q, cloud fraction, q_liquid): finite
    everywhere and nonzero for each input, on two cloudy daylight columns."""
    inp = inputs(z8)
    cols = np.flatnonzero((inp["cos_sza"] > 0.0) & (inp["q_liquid"].max(1) > 0.0))[:2]
    assert cols.size == 2
    sub = {k: (v[cols] if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
    iseed = np.asarray(z8["in:iseed"])[cols].astype(np.int64)
    ulw, usw = ecrad.draw_uniforms(iseed, sub["cloud_fraction"].shape[1])

    def functional(t_hl, q, cf, ql):
        gas = jnp.asarray(sub["gas_mmr"]).at[..., go.IH2O].set(q)
        o = ecrad.radiation(**dict(sub, temperature_hl=t_hl, gas_mmr=gas, cloud_fraction=cf,
                                   q_liquid=ql), uniforms_lw=ulw, uniforms_sw=usw)
        net = lambda u, d: o[d] - o[u]  # noqa: E731
        return (jnp.sum(net("flux_up_lw", "flux_dn_lw")[:, [0, -1]])
                + jnp.sum(net("flux_up_sw", "flux_dn_sw")[:, [0, -1]]))

    args = (sub["temperature_hl"], sub["gas_mmr"][..., go.IH2O], sub["cloud_fraction"], sub["q_liquid"])
    grads = jax.grad(functional, argnums=(0, 1, 2, 3))(*map(jnp.asarray, args))
    for name, g in zip(("temperature_hl", "q", "cloud_fraction", "q_liquid"), grads):
        g = np.asarray(g)
        print(f"GRAD {name}: max|g|={np.abs(g).max():.3e} nonzero={np.count_nonzero(g)}/{g.size}")
        assert np.all(np.isfinite(g)), name
        assert np.any(g != 0.0), name


# --------------------------------------------------------------------------
# eager vs jit
# --------------------------------------------------------------------------

_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.atmosphere import test_ecrad_ifs_end_to_end_oracle as t
out, _ = t.run(t.load(t.FIXTURE), float(sys.argv[2]), jit=True)
np.savez(sys.argv[1], **out)
"""


@pytest.mark.parametrize("sharpness", [HARD, 100.0])
def test_eager_vs_jit(z8, hard8, tmp_path, sharpness):
    """jit (separate process, --xla_cpu_max_isa=AVX so XLA emits no FMA; the
    oracle is built -ffp-contract=off) == eager to 1e-12; the default-ISA
    difference is printed (KNOWN DIFFERENCE, see the solver test)."""
    eager = hard8[0] if sharpness == HARD else run(z8, sharpness)[0]
    for flags, bound in (("--xla_cpu_max_isa=AVX", TOL), ("", None)):
        path = tmp_path / f"jit_{len(flags)}.npz"
        env = dict(os.environ, XLA_FLAGS=flags, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
        subprocess.run([sys.executable, "-c", _JIT_SCRIPT, str(path), repr(sharpness)],
                       env=env, check=True)
        jit = np.load(path)
        diff = {k: float(rel_err(eager[k], jit[k]).max()) for k in ecrad.OUTPUT_KEYS}
        print(f"EAGER_VS_JIT sharpness={sharpness} XLA_FLAGS='{flags}' max_rel={max(diff.values()):.3e} "
              f"worst={max(diff, key=diff.get)}")
        if bound is not None:
            assert all(v <= _V53_JIT_TOL.get(k, bound) for k, v in diff.items()), diff  # V53
