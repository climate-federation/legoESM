"""Unit tests for the NEMO Roquet-55 EOS-80 polynomial (``eos.nemo_roquet_eos``).

The EOS-80 coefficient set is the full polynomial NEMO runs under ``ln_eos80``
(GYRE and many reference configs). These tests pin the JAX port against an
independent NumPy transcription of NEMO's ``eos_insitu`` Horner form, the
independent UNESCO surface anchor, T/S monotonicity, an at-depth regression
pin, and finite AD gradients.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import (
    _ROQUET_EOS80,
    make_eos_fn,
    nemo_roquet_eos,
    rho_0,
)

from legoesm import constants


def _numpy_reference(T, S, depth, c=_ROQUET_EOS80):
    """Independent NumPy transcription of NEMO eos_insitu (EOS-80), depth in m."""
    zh = depth * c["r1_Z0"]
    zt = T * c["r1_T0"]
    zs = np.sqrt(np.abs(S + c["rdeltaS"]) * c["r1_S0"])
    zn3 = c["EOS013"] * zt + c["EOS103"] * zs + c["EOS003"]
    zn2 = ((c["EOS022"] * zt + c["EOS112"] * zs + c["EOS012"]) * zt
           + (c["EOS202"] * zs + c["EOS102"]) * zs + c["EOS002"])
    b4 = c["EOS041"]
    b3 = c["EOS131"] * zs + c["EOS031"]
    b2 = (c["EOS221"] * zs + c["EOS121"]) * zs + c["EOS021"]
    b1 = ((c["EOS311"] * zs + c["EOS211"]) * zs + c["EOS111"]) * zs + c["EOS011"]
    b0 = ((((c["EOS401"] * zs + c["EOS301"]) * zs + c["EOS201"]) * zs
           + c["EOS101"]) * zs + c["EOS001"])
    zn1 = (((b4 * zt + b3) * zt + b2) * zt + b1) * zt + b0
    a6 = c["EOS060"]
    a5 = c["EOS150"] * zs + c["EOS050"]
    a4 = (c["EOS240"] * zs + c["EOS140"]) * zs + c["EOS040"]
    a3 = ((c["EOS330"] * zs + c["EOS230"]) * zs + c["EOS130"]) * zs + c["EOS030"]
    a2 = ((((c["EOS420"] * zs + c["EOS320"]) * zs + c["EOS220"]) * zs
           + c["EOS120"]) * zs + c["EOS020"])
    a1 = (((((c["EOS510"] * zs + c["EOS410"]) * zs + c["EOS310"]) * zs
            + c["EOS210"]) * zs + c["EOS110"]) * zs + c["EOS010"])
    a0 = ((((((c["EOS600"] * zs + c["EOS500"]) * zs + c["EOS400"]) * zs
             + c["EOS300"]) * zs + c["EOS200"]) * zs + c["EOS100"]) * zs + c["EOS000"])
    zn0 = ((((((a6 * zt + a5) * zt + a4) * zt + a3) * zt + a2) * zt + a1) * zt + a0)
    return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0


def _rho_at_depth(T, S, depth):
    """Call the port with p chosen so its Boussinesq zh recovers `depth` exactly."""
    p = rho_0 * constants.g * np.asarray(depth)
    return np.asarray(nemo_roquet_eos(jnp.asarray(T), jnp.asarray(S), jnp.asarray(p)))


def test_matches_numpy_reference_random_grid():
    rng = np.random.default_rng(0)
    T = rng.uniform(-2.0, 32.0, size=(400,))
    S = rng.uniform(20.0, 40.0, size=(400,))
    depth = rng.uniform(0.0, 5500.0, size=(400,))
    got = _rho_at_depth(T, S, depth)
    ref = _numpy_reference(T, S, depth)
    assert np.max(np.abs(got - ref)) < 1e-9, np.max(np.abs(got - ref))


def test_surface_value_vs_unesco_anchor():
    # INDEPENDENT anchor (not from our transcription): the classic UNESCO EOS-80
    # surface density rho(T=0 degC, S=35 PSU, p=0) = 1028.106 kg/m^3. Matching it
    # validates the depth-independent (zn0) coefficient block against a source
    # other than the NEMO file we transcribed from.
    rho = float(_rho_at_depth(0.0, 35.0, 0.0))
    assert abs(rho - 1028.106) < 2e-2, rho


def test_physically_sane_and_ts_monotone():
    # Ocean-range density in a sensible band.
    rho = _rho_at_depth(
        np.array([5.0, 5.0, 5.0]),
        np.array([35.0, 35.0, 35.0]),
        np.array([0.0, 1000.0, 4000.0]),
    )
    assert np.all((rho > 1015.0) & (rho < 1055.0)), rho
    # Colder is denser (robust regardless of the depth reference convention).
    assert _rho_at_depth(2.0, 35.0, 100.0) > _rho_at_depth(20.0, 35.0, 100.0)
    # Saltier is denser.
    assert _rho_at_depth(10.0, 37.0, 100.0) > _rho_at_depth(10.0, 33.0, 100.0)
    # NB: in-situ density does NOT monotonically increase with depth here. NEMO's
    # Roquet-for-Boussinesq polynomial deliberately OMITS the horizontally-uniform
    # bulk adiabatic compression (~+18 kg/m^3 at 4000 m) because it cancels in the
    # PGF horizontal density difference (dynhpg hpg_zco: rhd(i+1)-rhd(i)) and is
    # dynamically inert in a Boussinesq model; only the thermobaric (sign-varying)
    # depth correction is retained. Confirmed faithful to NEMO by adversarial
    # coefficient + usage audit (2026-07-14).


def test_depth_dependence_regression_pin():
    # Pins the NET depth behaviour (zn1/zn2/zn3) so an accidental change to any
    # depth coefficient is caught — the surface anchor exercises only zn0. Values
    # are NEMO's own polynomial (all 52 coeffs audited byte-identical), NOT a NEMO
    # RUN dump; the fully-independent at-depth oracle is the certificate step.
    assert abs(float(_rho_at_depth(10.0, 35.0, 0.0)) - 1026.9543) < 1e-3
    assert abs(float(_rho_at_depth(10.0, 35.0, 4000.0)) - 1026.4121) < 1e-3


def test_differentiable_finite_gradients():
    # drho/dT and drho/dS must be finite (thermal expansion / haline contraction).
    def rho_of(T, S):
        p = rho_0 * constants.g * 1000.0
        return nemo_roquet_eos(T, S, p)
    dT = jax.grad(lambda T: rho_of(T, 35.0))(10.0)
    dS = jax.grad(lambda S: rho_of(10.0, S))(35.0)
    assert np.isfinite(dT) and dT < 0.0          # warming expands -> lighter
    assert np.isfinite(dS) and dS > 0.0          # salting contracts -> denser


def test_make_eos_fn_dispatch():
    fn = make_eos_fn(eos="nemo_eos80")
    p = jnp.asarray(rho_0 * constants.g * 1000.0)
    rho = float(fn(jnp.asarray(10.0), jnp.asarray(35.0), p))
    assert 1015.0 < rho < 1055.0
    with pytest.raises(ValueError):
        make_eos_fn(eos="nemo_eos80_typo")
