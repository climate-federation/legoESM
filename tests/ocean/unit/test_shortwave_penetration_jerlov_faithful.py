"""Faithfulness pins for the Jerlov two-band shortwave-penetration kernel.

Target: ``shortwave_penetration_tendency`` (the ``jerlov_2band`` scheme) in
``legoesm.ocean.physics.shortwave_penetration`` — the subsurface SW absorption
heating used by the ocean.  (The chlorophyll-dependent ``rgb_chl`` NEMO port is a
separate, lookup-dependent kernel and is out of scope here.)

Most-trustful source
--------------------
Paulson & Simpson (1977), "Irradiance measurements in the upper ocean",
J. Phys. Oceanogr. 7(6), 952-956 (Table 1 Jerlov water types).  The downwelling
irradiance fraction at depth z (z <= 0) is the two-band exponential

    I(z)/Q_sw = R * exp(z/zeta1) + (1 - R) * exp(z/zeta2),

and the layer heating is the flux divergence of I, converted to a temperature
tendency by rho_0 * c_sw * dz.

The existing test_shortwave_penetration.py is behavioral (0 exact-magnitude
assertions); this pins the two-band form, the published water-type table, the
column energy conservation (with the bottom-leak boundary condition), and the
exact tendency + z-star scaling + dry-cell gating.

Certification (test-only):
1. EXACT two-band transmission I(z) vs an independent oracle; surface = 1; a
   single-band model is discriminated.
2. The Jerlov water-type table (R, zeta1, zeta2 for I/IA/IB/II/III) = Paulson &
   Simpson (1977) Table 1.
3. TRUTH-TIER column energy conservation: sum(rho_0*c_sw*dz*dT/dt) == Q_sw (the
   bottom leak is added to the last layer, so a shallow column still absorbs the
   full surface flux); per-layer absorption is the flux divergence I[k]-I[k+1].
4. Exact tendency dT/dt = Q_sw*frac/(rho_0*c_sw*dz*J); z-star Jacobian scaling;
   dry-cell (J=0) gives zero heating (not NaN); scheme dispatch raises;
   water_type plumbing; differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import legoesm.ocean.physics.shortwave_penetration as shortwave_module
import numpy as np
import pytest
from legoesm.ocean.eos import c_sw, rho_0
from legoesm.ocean.physics.shortwave_penetration import (
    JERLOV_TYPES,
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)

jax.config.update("jax_enable_x64", True)

# A 4-layer column: dz = [10, 20, 30, 40] m -> interfaces at 0, -10, -30, -60, -100.
_DZ = jnp.asarray([10.0, 20.0, 30.0, 40.0])
_Z_HALF = jnp.asarray([0.0, -10.0, -30.0, -60.0, -100.0])
_SW = 200.0            # W/m^2


def _cfg(water_type="II", scheme="jerlov_2band"):
    return ShortwavePenetrationConfig(scheme=scheme, water_type=water_type)


def _tend(sw, dz, z_half, jac, cfg):
    return np.asarray(shortwave_penetration_tendency(
        jnp.asarray(sw, dtype=jnp.float64), jnp.asarray(dz, dtype=jnp.float64),
        jnp.asarray(z_half, dtype=jnp.float64), jnp.asarray(jac, dtype=jnp.float64),
        cfg))


def _transmission_oracle(z, r_frac, z1, z2):
    """Independent Paulson-Simpson two-band irradiance fraction at depth z (<=0)."""
    return r_frac * np.exp(z / z1) + (1.0 - r_frac) * np.exp(z / z2)


# ---------------------------------------------------------------------------
# 1. Two-band transmission.
# ---------------------------------------------------------------------------
def test_transmission_two_band_exact_via_energy_budget():
    # dT/dt in each layer = Q_sw*(I[k]-I[k+1])/(rho*c*dz).  Recover I[k]-I[k+1]
    # from the tendency and check it equals the independent two-band oracle with
    # LITERAL type-II params (independent of JERLOV_TYPES; the table values
    # themselves are pinned separately in test_jerlov_water_type_table_values).
    # Bottom layer excluded: it carries the added bottom-leak term.
    r_ii, z1, z2 = 0.77, 1.50, 14.0                    # Paulson-Simpson type II, literal
    t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("II"))
    dz = np.asarray(_DZ)
    frac = t * (rho_0 * c_sw * dz) / _SW               # recovered fraction absorbed per layer
    zc = np.asarray(_Z_HALF)
    ex = np.array([_transmission_oracle(zc[k], r_ii, z1, z2)
                   - _transmission_oracle(zc[k + 1], r_ii, z1, z2)
                   for k in range(len(dz))])
    np.testing.assert_allclose(frac[:-1], ex[:-1], rtol=1e-12)   # interior layers


def test_two_band_not_single_band():
    # DEPARTURE canary: the recovered top-layer absorbed fraction matches the
    # TWO-band form, not a single-band exp(z/zeta2) with the same total.
    p = JERLOV_TYPES["II"]
    t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("II"))
    frac0 = float(t[0] * (rho_0 * c_sw * float(_DZ[0])) / _SW)
    z1 = float(_Z_HALF[1])
    two_band = 1.0 - _transmission_oracle(z1, p.R, p.zeta1, p.zeta2)
    single_band = 1.0 - np.exp(z1 / p.zeta2)           # no fast R-band
    np.testing.assert_allclose(frac0, two_band, rtol=1e-12)
    assert abs(frac0 - single_band) > 0.05             # genuinely different


def test_surface_transmission_is_unity():
    # I(0) = R + (1-R) = 1: the whole column (with bottom leak) absorbs Q_sw, so
    # the recovered total fraction is 1 (also checked as energy conservation below).
    t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("II"))
    total_frac = float(np.sum(t * (rho_0 * c_sw * np.asarray(_DZ)) / _SW))
    np.testing.assert_allclose(total_frac, 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Jerlov water-type table (Paulson & Simpson 1977, Table 1).
# ---------------------------------------------------------------------------
def test_jerlov_water_type_table_values():
    assert JERLOV_TYPES["I"] == (0.58, 0.35, 23.0)
    assert JERLOV_TYPES["IA"] == (0.62, 0.60, 20.0)
    assert JERLOV_TYPES["IB"] == (0.67, 1.00, 17.0)
    assert JERLOV_TYPES["II"] == (0.77, 1.50, 14.0)
    assert JERLOV_TYPES["III"] == (0.78, 1.40, 7.9)


# ---------------------------------------------------------------------------
# 3. Energy conservation + flux divergence (truth tier).
# ---------------------------------------------------------------------------
def test_column_energy_conservation_deep():
    # Deep column: sum(rho*c*dz*dT/dt) == Q_sw (all surface SW absorbed).
    t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("I"))
    absorbed = float(np.sum(rho_0 * c_sw * np.asarray(_DZ) * t))
    np.testing.assert_allclose(absorbed, _SW, rtol=1e-12)


def test_shallow_column_bottom_leak_added_conserves():
    # Shallow column (H=8 m << zeta2=14 m for type II): a large fraction would
    # otherwise leak past the bottom.  The kernel adds that leak to the bottom
    # layer, so the column STILL absorbs the full Q_sw (a no-leak-add impl would
    # absorb only 1 - I(-8) < 1).
    dz = jnp.asarray([2.0, 2.0, 2.0, 2.0])            # H = 8 m
    z_half = jnp.asarray([0.0, -2.0, -4.0, -6.0, -8.0])
    p = JERLOV_TYPES["II"]
    t = _tend(_SW, dz, z_half, 1.0, _cfg("II"))
    absorbed = float(np.sum(rho_0 * c_sw * np.asarray(dz) * t))
    np.testing.assert_allclose(absorbed, _SW, rtol=1e-12)          # full flux
    leak = _transmission_oracle(-8.0, p.R, p.zeta1, p.zeta2)
    assert leak > 0.05                                             # leak is non-trivial here
    # bottom layer carries its own divergence PLUS the leak:
    frac_bot = float(t[-1] * (rho_0 * c_sw * float(dz[-1])) / _SW)
    exp_bot = (_transmission_oracle(-6.0, p.R, p.zeta1, p.zeta2)
               - _transmission_oracle(-8.0, p.R, p.zeta1, p.zeta2)) + leak
    np.testing.assert_allclose(frac_bot, exp_bot, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Tendency form, z-star scaling, dry-cell gating.
# ---------------------------------------------------------------------------
def test_tendency_exact_form():
    # dT/dt[k] = Q_sw * frac[k] / (rho_0 * c_sw * dz[k]) for an interior layer.
    p = JERLOV_TYPES["II"]
    t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("II"))
    zc = np.asarray(_Z_HALF)
    frac1 = (_transmission_oracle(zc[1], p.R, p.zeta1, p.zeta2)
             - _transmission_oracle(zc[2], p.R, p.zeta1, p.zeta2))
    exp1 = _SW * frac1 / (rho_0 * c_sw * float(_DZ[1]))
    np.testing.assert_allclose(float(t[1]), exp1, rtol=1e-12)


def test_zstar_jacobian_scales_layer_thickness():
    # dz_actual = dz_ref * J: doubling J halves dT/dt (thicker layer, same
    # absorbed energy).  Absorption fraction (I depends on REFERENCE z) unchanged.
    t1 = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("II"))
    t2 = _tend(_SW, _DZ, _Z_HALF, 2.0, _cfg("II"))
    np.testing.assert_allclose(t2, 0.5 * t1, rtol=1e-12)


def test_dry_cell_zero_heating():
    # dz_actual <= 0 (jacobian = 0, and the negative-Jacobian edge) -> heating is
    # exactly 0 (not NaN/Inf); the gate is ``dz_actual > 0``, not ``== 0``.
    for jac in (0.0, -1.0):
        t = _tend(_SW, _DZ, _Z_HALF, jac, _cfg("II"))
        assert np.all(t == 0.0) and np.all(np.isfinite(t))


def test_rho_0_and_c_sw_plumbing():
    # dT/dt ~ 1/(rho_0 * c_sw): doubling BOTH quarters the tendency (a hard-coded
    # default for either would break this).
    base = np.asarray(shortwave_penetration_tendency(
        jnp.asarray(_SW), _DZ, _Z_HALF, jnp.asarray(1.0), _cfg("II")))
    scaled = np.asarray(shortwave_penetration_tendency(
        jnp.asarray(_SW), _DZ, _Z_HALF, jnp.asarray(1.0), _cfg("II"),
        rho_0=2.0 * rho_0, c_sw=2.0 * c_sw))
    np.testing.assert_allclose(scaled, base / 4.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Dispatch, config plumbing, differentiability.
# ---------------------------------------------------------------------------
def test_jerlov_kernel_rejects_non_jerlov_scheme():
    # This is the Jerlov-only kernel: it must RAISE on scheme='rgb_chl' rather
    # than silently running the wrong physics (the public dispatcher routes
    # rgb_chl to its own kernel).
    try:
        _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg(scheme="rgb_chl"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for scheme='rgb_chl'")


def test_nemo_qsr_2bd_selector_requires_extinction_initialization_timestep():
    generic = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg(scheme="jerlov_2band"))
    with pytest.raises(ValueError, match="nemo_time_step_s"):
        _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg(scheme="nemo_qsr_2bd"))
    nemo = _tend(
        _SW,
        _DZ,
        _Z_HALF,
        1.0,
        ShortwavePenetrationConfig(
            scheme="nemo_qsr_2bd", water_type="II", nemo_time_step_s=14400.0
        ),
    )
    # NEMO terminates at qsr_ext_lev rather than depositing the remaining
    # irradiance at the physical bottom like the generic conserving kernel.
    assert not np.array_equal(np.asarray(nemo), np.asarray(generic))


def test_rgb_routes_every_exponential_through_precision_policy(monkeypatch):
    """Both RGB kernels must honour the certification transcendental policy.

    The Morel--Berthon profile has three exponential call sites and the RGB
    optical-depth helper is invoked once for each of IR/R/G/B.  Replacing the
    shared policy function by a planted constant therefore has seven observable
    calls.  Leaving any one of the four source sites on ``jnp.exp`` makes this
    control fail by reducing that count or by changing the planted output.
    """
    calls = []

    def _planted_exp(value):
        value = jnp.asarray(value)
        calls.append(value.shape)
        return jnp.full_like(value, 0.75)

    monkeypatch.setattr(shortwave_module, "precision_exp", _planted_exp)
    cfg = ShortwavePenetrationConfig(
        scheme="nemo_qsr_rgb", rgb_chl_profile="morel_berthon",
        nemo_time_step_s=10800.0,
    )
    result = shortwave_module.shortwave_penetration_rgb_tendency(
        jnp.asarray([120.0], dtype=jnp.float64),
        jnp.asarray([0.2], dtype=jnp.float64),
        jnp.asarray([[5.0, 10.0]], dtype=jnp.float64),
        jnp.ones((1, 2), dtype=jnp.float64),
        cfg,
        gdepw_bottom_live=jnp.asarray([[5.0, 15.0]], dtype=jnp.float64),
        gdepw_ref=jnp.asarray([0.0, 5.0, 15.0], dtype=jnp.float64),
        e3t_ref=jnp.asarray([5.0, 10.0], dtype=jnp.float64),
    )

    assert len(calls) == 7
    assert calls[:3] == [(1,), (1,), (1, 2)]
    assert calls[3:] == [(1,)] * 4
    assert np.all(np.isfinite(np.asarray(result)))


def test_water_type_plumbing_changes_profile():
    # Type I (clear, zeta2=23) penetrates deeper than type III (turbid, zeta2=7.9)
    # -> less heating in the top layer for type I.
    t_i = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("I"))
    t_iii = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg("III"))
    assert t_i[0] < t_iii[0]
    # and both are exact per their own table row (plumbing, not a hard-coded type):
    for wt in ("I", "III"):
        p = JERLOV_TYPES[wt]
        t = _tend(_SW, _DZ, _Z_HALF, 1.0, _cfg(wt))
        zc = np.asarray(_Z_HALF)
        f0 = 1.0 - _transmission_oracle(zc[1], p.R, p.zeta1, p.zeta2)
        np.testing.assert_allclose(
            float(t[0] * (rho_0 * c_sw * float(_DZ[0])) / _SW), f0, rtol=1e-12)


def test_differentiable_including_dry_cell():
    def loss(sw):
        return jnp.sum(shortwave_penetration_tendency(
            sw, _DZ, _Z_HALF, jnp.asarray(1.0), _cfg("II")))
    g_wet = jax.grad(loss)(jnp.asarray(_SW))

    def loss_dry(jac):
        return jnp.sum(shortwave_penetration_tendency(
            jnp.asarray(_SW), _DZ, _Z_HALF, jac, _cfg("II")))
    g_dry = jax.grad(loss_dry)(jnp.asarray(0.0))     # through the dz_safe gate
    assert jnp.isfinite(g_wet) and jnp.isfinite(g_dry)
