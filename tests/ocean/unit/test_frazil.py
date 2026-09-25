"""Synthetic budget and regression tests; run with JAX_ENABLE_X64=1 on CPU."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import freezing_point, potential_temperature, in_situ_temperature
from jax.experimental import checkify
from legoesm.ocean.physics.frazil import FrazilConfig, apply_frazil


def run(t, s, dz, p, config, active=None):
    """Exercise the public tracer interface; express budget assertions in situ."""
    if active is None:
        active = jnp.ones_like(t, dtype=bool)
    dtype = jnp.result_type(t, s, dz, p, jnp.float32)
    t, s = jnp.asarray(t, dtype), jnp.asarray(s, dtype)
    p = jnp.broadcast_to(jnp.asarray(p, dtype), t.shape)
    theta = potential_temperature(s, t, p * 1e-4)
    out = apply_frazil(theta, s, dz, active, p, config)
    return out._replace(T_C=in_situ_temperature(out.S_psu, out.T_C, p * 1e-4))


def liquidus(s, p, scheme):
    """Use the production EOS, including its Pa input and Kelvin output."""
    return freezing_point(s, p, scheme=scheme) - constants.T_freeze


@pytest.mark.parametrize("mode", ["direct", "remelt"])
@pytest.mark.parametrize("scheme", ["constant", "linear_S", "unesco"])
def test_heat_salt_and_water_conservation(mode, scheme):
    # Catches latent-heat sign errors, virtual salt updates, and missing water removal.
    cfg = FrazilConfig(enabled=True, rising_frazil=mode, freezing_scheme=scheme)
    s = jnp.array([[35., 34., 36.], [34., 35., 36.]], dtype=jnp.float64)
    p = jnp.array([0., 1e6, 8e6])
    dz = jnp.array([[5., 12., 20.], [8., 16., 24.]])
    t = liquidus(s, p, scheme) + jnp.array([[0.1, -0.8, -1.34], [3., -0.5, -1.]])
    out = run(t, s, dz, p, cfg)
    m0, m1 = cfg.rho_sw_kg_m3 * dz, cfg.rho_sw_kg_m3 * out.dz_m
    ice = out.ice_mass_per_area_kg_m2
    assert np.any(np.asarray(ice) > 0)
    assert np.any(np.asarray(out.dz_m) < np.asarray(dz))
    np.testing.assert_allclose(jnp.sum(m1, -1) + ice, jnp.sum(m0, -1), rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(m1 * out.S_psu, -1) + ice * cfg.ice_salinity_psu,
                               jnp.sum(m0 * s, -1), rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(m1 * cfg.c_p_sw_j_kg_k * out.T_C, -1)
                               - ice * cfg.L_f_j_kg,
                               jnp.sum(m0 * cfg.c_p_sw_j_kg_k * t, -1), rtol=2e-14)


@pytest.mark.parametrize("scheme", ["constant", "linear_S", "unesco"])
@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-4), (jnp.float64, 1e-10)])
@pytest.mark.parametrize("upper_cp_limit", [False, True])
def test_formation_equilibrates_final_salinity(scheme, dtype, tol, upper_cp_limit):
    # Catches the draft's old-salinity liquidus and first-order salt approximation.
    cfg = FrazilConfig(enabled=True, rising_frazil="direct", freezing_scheme=scheme,
                       ice_salinity_psu=0.)
    if upper_cp_limit:
        cfg = cfg._replace(c_p_sw_j_kg_k=0.02 * constants.L_f)
    s = jnp.array([[0., 10., 35., 50.]], dtype=dtype)
    p = jnp.array([[0., 1e6, 1e7, 1e8]], dtype=dtype)
    dz = jnp.array([[1., 10., 25., 50.]], dtype=dtype)
    t = liquidus(s, p, scheme).astype(dtype) - jnp.asarray(1.34, dtype=dtype)
    out = run(t, s, dz, p, cfg)
    np.testing.assert_allclose(out.T_C, liquidus(out.S_psu, p, scheme), atol=tol, rtol=0)
    fraction = 1 - out.dz_m / dz
    assert np.all(np.asarray(fraction) > 0.01)
    np.testing.assert_allclose(out.S_psu, s / (1 - fraction), atol=tol, rtol=0)
    # The planted first-order update differs appreciably in this stated regime.
    assert abs(float(out.S_psu[0, 2] - (s * (1 + fraction))[0, 2])) > 0.005


@pytest.mark.parametrize("mode", ["direct", "remelt"])
def test_above_freezing_column_unchanged(mode):
    # Catches spurious formation in warm water, including passing dbar to a Pa API.
    cfg = FrazilConfig(enabled=True, freezing_scheme="unesco", rising_frazil=mode)
    s = jnp.array([35., 35., 35.])
    p = jnp.array([0., 1e6, 8e6])
    dz = jnp.array([2., 10., 30.])
    t = liquidus(s, p, "unesco") + 0.01
    out = run(t, s, dz, p, cfg)
    np.testing.assert_allclose(out.T_C, t, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(out.S_psu, s)
    np.testing.assert_allclose(out.dz_m, dz, atol=1e-14, rtol=0)
    assert float(out.ice_mass_per_area_kg_m2) == 0
    # This warm deep water is below the zero-pressure liquidus: pressure matters.
    assert float(t[-1]) < float(liquidus(s[-1], 0., "unesco"))


@pytest.mark.parametrize("surface_excess", [0.05, 3.])
@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-4), (jnp.float64, 1e-10)])
def test_remelt_freshens_and_respects_updated_liquidus(surface_excess, dtype, tol):
    # Catches the draft's salt-creating melt and melting past the freshened liquidus.
    cfg = FrazilConfig(enabled=True, freezing_scheme="linear_S", ice_salinity_psu=0.)
    s = jnp.array([35., 35.], dtype=dtype)
    dz = jnp.array([10., 10.], dtype=dtype)
    p = jnp.array([0., 1e6], dtype=dtype)
    t = liquidus(s, p, cfg.freezing_scheme).astype(dtype) + jnp.array([surface_excess, -1.34], dtype=dtype)
    direct = run(t, s, dz, p, cfg._replace(rising_frazil="direct"))
    out = run(t, s, dz, p, cfg)
    assert float(out.S_psu[0]) < float(s[0])
    assert float(out.T_C[0]) < float(t[0])
    assert float(out.ice_mass_per_area_kg_m2) < float(direct.ice_mass_per_area_kg_m2)
    tf = liquidus(out.S_psu, 0., cfg.freezing_scheme)
    assert np.all(np.asarray(out.T_C - tf) >= -tol)
    if surface_excess < 1:
        assert float(out.ice_mass_per_area_kg_m2) > 0
        np.testing.assert_allclose(out.T_C[0], tf[0], atol=tol, rtol=0)
    else:
        np.testing.assert_allclose(out.ice_mass_per_area_kg_m2, 0., atol=tol)


def test_mask_and_zero_thickness():
    # Deliberately outside the contiguous-column contract: pin unchecked skip behavior.
    # Catches inactive-layer formation/remelting and zero-thickness divisions in AD.
    cfg = FrazilConfig(enabled=True, freezing_scheme="linear_S")
    t = jnp.array([-3., -4., -4., 2.])
    s = jnp.full_like(t, 35.)
    dz = jnp.array([10., 0., 5., 10.])
    active = jnp.array([True, True, False, True])
    out = run(t, s, dz, 0., cfg, active)
    np.testing.assert_array_equal(out.T_C[1:3], t[1:3])
    np.testing.assert_array_equal(out.S_psu[1:3], s[1:3])
    np.testing.assert_array_equal(out.dz_m[1:3], dz[1:3])
    assert float(out.ice_mass_per_area_kg_m2) > 0
    grad = jax.grad(lambda x: run(x, s, dz, 0., cfg, active).T_C.sum())(t)
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize("mode", ["direct", "remelt"])
def test_jit_mixed_dtype_and_gradient(mode):
    # Catches scan carry promotion, traced branching, and numerically wrong gradients.
    cfg = FrazilConfig(enabled=True, freezing_scheme="unesco", rising_frazil=mode)
    s = jnp.array([35., 36.], dtype=jnp.float64)
    dz = jnp.array([10., 20.], dtype=jnp.float64)
    p = jnp.array([0., 1e6], dtype=jnp.float64)
    t = jnp.array([-1.5, -3.], dtype=jnp.float32)
    fn = partial(run, s=s, dz=dz, p=p, config=cfg)
    eager, compiled = fn(t), jax.jit(fn)(t)
    for a, b in zip(eager, compiled):
        assert b.dtype == jnp.float64
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # Finite differences use float64 inputs so differencing isn't storage-limited.
    t64 = t.astype(jnp.float64)
    objective = lambda x: fn(x).ice_mass_per_area_kg_m2
    derivative = jax.jit(jax.grad(objective))(t64)
    eps = 1e-5
    finite_difference = jnp.array([
        (objective(t64 + jnp.eye(2)[i] * eps) - objective(t64 - jnp.eye(2)[i] * eps))
        / (2 * eps) for i in range(2)])
    assert np.all(np.isfinite(derivative))
    assert abs(float(derivative[-1])) > 1
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("field,value", [("freezing_scheme", "typo"),
                                         ("rising_frazil", "typo")])
def test_invalid_selectors_even_disabled(enabled, field, value):
    # Catches the draft's disabled-path freezing-scheme validation bypass.
    cfg = FrazilConfig(enabled=enabled)._replace(**{field: value})
    with pytest.raises(ValueError, match=field):
        run(jnp.array([-3.]), jnp.array([35.]), 10., 0., cfg)


@pytest.mark.parametrize("enabled", [False, True])
def test_temperature_convention_cannot_select_in_situ(enabled):
    args = (jnp.array([-3.]), jnp.array([35.]), 10., True,
            jnp.array([0.]), FrazilConfig(enabled=enabled))
    # The public API has no switch that can bypass tracer conversion.
    with pytest.raises(TypeError, match="temperature_convention"):
        apply_frazil(*args, temperature_convention="in_situ_C")


def test_disabled_is_identity():
    # Catches accidentally applying the closure when disabled to supercooled water.
    t, s, dz = jnp.array([-4.]), jnp.array([35.]), jnp.array([10.])
    out = run(t, s, dz, 0., FrazilConfig())
    for a, b in ((out.T_C, t), (out.S_psu, s), (out.dz_m, dz)):
        np.testing.assert_array_equal(a, b)
    assert float(out.ice_mass_per_area_kg_m2) == 0


def test_contracts_and_canonical_defaults():
    # Catches missing/invalid repository metadata and independent physical constants.
    from tests.test_physics_contracts import extract_contract, validate_contract
    from tests.test_param_specs import extract_param_spec, validate_param_spec
    import legoesm.ocean.physics.frazil as module

    source = Path(module.__file__).read_text()
    assert validate_contract(extract_contract(source)) == []
    fields = set(FrazilConfig.__annotations__) - {"enabled", "freezing_scheme", "rising_frazil"}
    spec = extract_param_spec(source)
    assert set(spec) == {"FrazilConfig"}
    assert set(spec["FrazilConfig"]["params"]) | set(spec["FrazilConfig"]["excluded"]) == fields
    assert validate_param_spec(spec, {}, {"FrazilConfig": fields}) == []
    cfg = FrazilConfig()
    assert cfg.rho_sw_kg_m3 == constants.rho_ocean
    assert cfg.c_p_sw_j_kg_k == constants.c_sw
    assert cfg.L_f_j_kg == constants.L_f
    assert cfg.ice_salinity_psu == constants.S_ice_bulk_default


@pytest.mark.parametrize("p", [0., jnp.array([0.])])
@pytest.mark.parametrize("compiled", [False, True])
def test_pressure_requires_full_level_axis(p, compiled):
    fn = lambda pressure: apply_frazil(jnp.array([-3., -3.]), 35., 10., True,
                                      pressure, FrazilConfig(enabled=True))
    if compiled:
        fn = jax.jit(fn)
    with pytest.raises(ValueError, match="full-depth"):
        fn(p)


@pytest.mark.parametrize("pressure", [[0., 0.], [1e7, 0.], [0., -1.],
                                      [0., 1e8 + 1.], [0., float("nan")]])
def test_checked_pressure_rejects_surface_broadcast_or_reversal(pressure):
    fn = jax.jit(checkify.checkify(lambda p: apply_frazil(
        jnp.array([2., 2.]), 35., 10., True, p, FrazilConfig(enabled=True))))
    good, _ = fn(jnp.array([0., 1e7]))
    good.throw()
    bad, _ = fn(jnp.array(pressure))
    assert bad.get() is not None
    assert "pressure" in bad.get()
    with pytest.raises(checkify.JaxRuntimeError, match="pressure"):
        bad.throw()


def test_checked_noncontiguous_active_column():
    fn = jax.jit(checkify.checkify(lambda mask: apply_frazil(
        jnp.ones(3), 35., 10., mask, jnp.array([0., 1e6, 2e6]),
        FrazilConfig(enabled=True))))
    good, _ = fn(jnp.array([True, True, False]))
    good.throw()
    bad, _ = fn(jnp.array([True, False, True]))
    assert "contiguous" in bad.get()
    with pytest.raises(checkify.JaxRuntimeError, match="contiguous"):
        bad.throw()


@pytest.mark.parametrize("bad_temperature,message", [
    (-100., "formation failed"), (float("nan"), "nonfinite active state")])
def test_checked_invalid_formation_is_signalled(bad_temperature, message):
    fn = jax.jit(checkify.checkify(lambda t: apply_frazil(
        t, 35., 10., True, jnp.array([0.]), FrazilConfig(enabled=True))))
    good, _ = fn(jnp.array([-3.]))
    good.throw()
    bad, _ = fn(jnp.array([bad_temperature]))
    assert bad.get() is not None
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        bad.throw()


def test_surface_first_order_pins_rising_ice_path():
    cfg = FrazilConfig(enabled=True, freezing_scheme="constant")
    p = jnp.array([0., 1e6])
    # Ice at depth must pass through and melt in the warm surface layer.
    out = run(jnp.array([3., -3.]), 35., 10., p, cfg)
    assert float(out.ice_mass_per_area_kg_m2) == 0
    assert float(out.S_psu[0]) < 35.
    # Ice formed at the surface cannot melt in warm water BELOW it.
    reversed_t = run(jnp.array([-3., 3.]), 35., 10., p, cfg)
    assert float(reversed_t.ice_mass_per_area_kg_m2) > 0
    np.testing.assert_allclose(reversed_t.S_psu[1], 35., atol=1e-12)


@pytest.mark.parametrize("mode", ["direct", "remelt"])
def test_potential_tracer_threshold_and_return_round_trip(mode):
    cfg = FrazilConfig(enabled=True, freezing_scheme="unesco", rising_frazil=mode)
    s = jnp.array([[35., 35.], [35., 35.]])
    p = jnp.array([0., 1e7])
    t = liquidus(s, p, "unesco") + jnp.array([[1., 0.01], [1., -0.01]])
    theta = potential_temperature(s, t, p * 1e-4)
    # Plant the original bug: potential T would classify the warm deep cell cold.
    assert float(theta[0, 1]) < float(liquidus(s, p, "unesco")[0, 1])
    fn = jax.jit(checkify.checkify(lambda x: apply_frazil(x, s, 10., True, p, cfg)))
    err, out = fn(theta)
    err.throw()
    np.testing.assert_array_equal(out.T_C[0], theta[0])
    assert float(out.S_psu[1, 1]) > 35.
    back = in_situ_temperature(out.S_psu, out.T_C, p * 1e-4)
    np.testing.assert_allclose(back[1, 1], liquidus(out.S_psu, p, "unesco")[1, 1],
                               atol=1e-10, rtol=0)
    assert abs(float(back[1, 1] - out.T_C[1, 1])) > 0.01
    np.testing.assert_allclose(potential_temperature(out.S_psu, back, p * 1e-4),
                               out.T_C, atol=1e-12, rtol=0)


@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-5), (jnp.float64, 1e-12)])
def test_inverse_eos_round_trip(dtype, tol):
    s = jnp.array([0., 35., 50.], dtype=dtype)
    t = jnp.array([-1., -3., 30.], dtype=dtype)
    p = jnp.array([0., 1000., 10000.], dtype=dtype)
    theta = potential_temperature(s, t, p)
    back = jax.jit(in_situ_temperature)(s, theta, p)
    err, checked_back = jax.jit(checkify.checkify(in_situ_temperature))(s, theta, p)
    err.throw()
    np.testing.assert_allclose(checked_back, back, atol=tol, rtol=0)
    np.testing.assert_allclose(back, t, atol=tol, rtol=0)
    derivative = jax.jit(jax.grad(lambda x: in_situ_temperature(s, x, p).sum()))(theta)
    eps = 1e-3 if dtype == jnp.float32 else 1e-5
    fd = (in_situ_temperature(s, theta + eps, p)
          - in_situ_temperature(s, theta - eps, p)) / (2 * eps)
    np.testing.assert_allclose(derivative, fd, atol=2e-3 if dtype == jnp.float32 else 1e-8)


def test_checked_nan_thickness_is_not_silently_inactive():
    fn = jax.jit(checkify.checkify(lambda dz: apply_frazil(
        jnp.array([-3.]), 35., dz, True, jnp.array([0.]), FrazilConfig(enabled=True))))
    good, _ = fn(jnp.array([10.]))
    good.throw()
    bad, _ = fn(jnp.array([float("nan")]))
    assert "nonfinite active thickness" in bad.get()
    with pytest.raises(checkify.JaxRuntimeError, match="nonfinite active thickness"):
        bad.throw()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_inverse_residual_guard_detects_failed_convergence(monkeypatch, dtype):
    import legoesm.ocean.eos as eos

    # Plant a changed EOS whose correction map oscillates rather than contracts.
    # Reverse integration gives a nonzero initial error, so this cannot pass by
    # accidentally starting at the root; each iteration preserves its magnitude.
    monkeypatch.setattr(eos, "potential_temperature",
                        lambda s, t, p, p_ref_dbar=0.: 2 * t + (p - p_ref_dbar) * 1e-3)
    fn = jax.jit(checkify.checkify(eos.in_situ_temperature))
    err, _ = fn(jnp.asarray(35., dtype), jnp.asarray(10., dtype),
                jnp.asarray(6000., dtype))
    with pytest.raises(checkify.JaxRuntimeError, match="inverse residual"):
        err.throw()


def test_checked_driver_allows_inactive_nonfinite_storage():
    fn = jax.jit(checkify.checkify(lambda t: apply_frazil(
        t, jnp.array([35., jnp.nan]), jnp.array([10., 0.]),
        jnp.array([True, False]), jnp.array([0., jnp.nan]),
        FrazilConfig(enabled=True))))
    err, out = fn(jnp.array([-3., jnp.nan]))
    err.throw()
    assert jnp.isfinite(out.T_C[0])
    assert jnp.isnan(out.T_C[1])


def test_plain_jit_value_guards_are_inert_and_pressure_is_not_unit_validation():
    fn = lambda p: apply_frazil(jnp.array([2., 2.]), 35., 10., True, p,
                               FrazilConfig(enabled=True))
    # Deliberate reversed pressure: plain JIT returns, checked driver throws.
    jax.jit(fn)(jnp.array([6000., 0.]))
    checked = jax.jit(checkify.checkify(fn))
    err, _ = checked(jnp.array([6000., 0.]))
    with pytest.raises(checkify.JaxRuntimeError, match="pressure"):
        err.throw()
    # The same number can mean a legitimate shallow pressure in Pa or wrong units.
    err, _ = checked(jnp.array([0., 6000.]))
    err.throw()
