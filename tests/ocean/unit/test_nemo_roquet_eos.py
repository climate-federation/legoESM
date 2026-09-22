"""Unit tests for the NEMO Roquet-55 EOS-80 polynomial (``eos.nemo_roquet_eos``).

The EOS-80 coefficient set is the full polynomial NEMO runs under ``ln_eos80``
(GYRE and many reference configs). These tests pin the JAX port against an
independent NumPy transcription of NEMO's ``eos_insitu`` Horner form, the
independent UNESCO surface anchor, T/S monotonicity, an at-depth regression
pin, and finite AD gradients.
"""
import hashlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import (
    _ROQUET_EOS80,
    _ROQUET_TEOS10,
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


def test_nemo_teos10_dispatch_is_the_canonical_roquet_polynomial():
    """The selectable phase-3 option must not route through Veros GSW."""
    rho_ref = 1026.0
    T = jnp.asarray([-1.0, 10.0, 25.0], dtype=jnp.float64)
    S = jnp.asarray([34.0, 35.0, 37.0], dtype=jnp.float64)
    depth = jnp.asarray([0.0, 500.0, 4000.0], dtype=jnp.float64)
    pressure = rho_ref * constants.g * depth
    got = make_eos_fn("nemo_teos10", rho0=rho_ref)(T, S, pressure)
    expected = nemo_roquet_eos(
        T, S, pressure, coeffs=_ROQUET_TEOS10, rho0=rho_ref
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_nemo_teos10_density_coefficient_table_has_ci_pin():
    """Pin all 52 EOS### values without requiring a local NEMO checkout."""
    density = {k: float(v) for k, v in _ROQUET_TEOS10.items()
               if k.startswith("EOS")}
    assert len(density) == 52

    def digest(values):
        hashed = hashlib.sha256()
        for name, value in sorted(values.items()):
            hashed.update(name.encode("ascii"))
            hashed.update(b"\0")
            hashed.update(struct.pack(">d", value))
        return hashed.hexdigest()

    expected = "dfb7fe0df632023d5f7dec65221cd2f0733cf4d80312494120384ae94e756239"
    assert digest(density) == expected
    planted = dict(density)
    planted["EOS000"] += 1.0e-11
    assert digest(planted) != expected


# ---------------------------------------------------------------------------
# The EOS-80 stratification arm.  ORCA2's namelist selects EOS-80
# (``ln_eos80 = .true.``) and NEMO's expansion-coefficient routine runs ONE
# polynomial for both forms -- ``rab_3d_t``'s ``CASE( np_teos10, np_eos80 )``
# -- while its buoyancy-frequency routine ``bn2_t`` carries no equation-of-
# state branch at all.  So the arm is a coefficient-set selection on the
# evaluator already shipped here, and these tests pin exactly that.
# ---------------------------------------------------------------------------

def _bn2_column():
    from legoesm.ocean import eos
    nlev = 12
    gdept = jnp.asarray(np.linspace(5.0, 500.0, nlev))
    gdepw = 0.5 * (gdept[:-1] + gdept[1:])
    profile = np.linspace(18.0, 4.0, nlev)
    profile[5] = profile[6] - 0.5          # one statically unstable pair
    return (eos, jnp.asarray(profile), jnp.full((nlev,), 35.0), gdept, gdepw)


def test_bn2_eos80_arm_is_the_eos80_alpha_beta_assembly():
    """The arm must use the EOS-80 coefficients, not TEOS-10's."""
    eos, T, S, gdept, gdepw = _bn2_column()
    got = eos.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, eos_form="eos80", e3w_source="depth_difference")
    alpha, beta = eos.nemo_roquet_alpha_beta(T, S, gdept, eos_form="eos80")
    expected = eos.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, e3w_source="depth_difference",
        _alpha_beta_override=(alpha, beta))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_bn2_eos80_differs_from_teos10_and_keeps_the_sign():
    """Non-vacuity: selecting EOS-80 must change the number, not just pass."""
    eos, T, S, gdept, gdepw = _bn2_column()
    eos80 = np.asarray(eos.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, eos_form="eos80", e3w_source="depth_difference"))
    teos10 = np.asarray(eos.compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, eos_form="teos10", e3w_source="depth_difference"))
    assert not np.allclose(eos80, teos10), "eos80 returned the TEOS-10 answer"
    assert eos80[5] < 0.0, "eos80 lost the statically unstable interface"
    ratio = eos80[np.arange(eos80.size) != 5] / teos10[
        np.arange(teos10.size) != 5]
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_bn2_still_refuses_an_unknown_eos_form():
    """Dispatch hardening survives the new arm."""
    eos, T, S, gdept, gdepw = _bn2_column()
    with pytest.raises(ValueError, match="eos_form"):
        eos.compute_buoyancy_frequency_nemo_bn2(
            T, S, gdept, gdepw, eos_form="eos-80",
            e3w_source="depth_difference")
