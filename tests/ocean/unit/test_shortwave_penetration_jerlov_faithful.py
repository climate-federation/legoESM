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
import numpy as np
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


# ------------------------------------------- NEMO qsr_2BD, statement by statement
def _dino_ladder():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.experiments import dino as dm
    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    return dm, cfg, grid, z


def test_nemo_qsr_ext_lev_reproduces_the_levels_nemo_printed():
    """NEMO's ``ocean.output`` for this configuration prints, verbatim::

           level of infrared extinction       =  2  ref depth = 20.593063338905267 m
           level of visible light extinction  = 22  ref depth = 635.29067417406986 m

    and ``qsr_ext_lev`` sees ``rDt = 2*rn_Dt`` because ``dom_init`` sets that
    for the modified leap-frog (``domain.F90:309-310``) before ``tra_init``
    runs.  Both halves are asserted: at ``rn_Dt`` the infrared level comes out
    1, which is the wrong answer and the reason the factor is pinned.
    """
    import numpy as np
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.physics.shortwave_penetration import nemo_qsr_ext_lev
    _dm, cfg, _grid, z = _dino_ladder()
    tmask = np.asarray(ndm.nemo_dino_mesh().tmask) > 0.5
    kw = dict(rho_0=cfg.rho_0, c_sw=cfg.c_p)
    assert nemo_qsr_ext_lev(z, tmask, rdt=2 * 2700.0, **kw) == (2, 22)
    assert nemo_qsr_ext_lev(z, tmask, rdt=2700.0, **kw) == (1, 22)
    zh = np.asarray(z.z_half_ref)
    assert abs(-float(zh[2]) - 20.593063338905267) < 1e-9
    assert abs(-float(zh[22]) - 635.29067417406986) < 1e-8


def test_nemo_2bd_zeroes_every_level_below_nkv_and_the_default_does_not():
    """``tra_qsr``'s trend loop is ``DO jk = 1, nksr`` (``traqsr.F90:261``)
    with ``nksr = nkV`` (``:1318``), so levels below it receive NOTHING.  The
    reference kernel instead adds the whole un-absorbed remainder to the LAST
    level of the ladder, which is the statement this pins.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, shortwave_penetration_tendency)
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    _dm, cfg, _grid, z = _dino_ladder()
    wet = jnp.asarray(np.asarray(ndm.nemo_dino_mesh().tmask) > 0.5)
    q = jnp.full(wet.shape[:2], 200.0)
    stretch = jnp.ones(wet.shape[:2])
    common = dict(z_coord_dz_ref=z.dz_ref, z_coord_z_half_ref=z.z_half_ref,
                  jacobian=jnp.ones(wet.shape[:2]),
                  config=ShortwavePenetrationConfig(water_type="I"),
                  rho_0=cfg.rho_0, c_sw=cfg.c_p, z_half_stretch=stretch)
    nemo = np.asarray(shortwave_penetration_tendency(
        sw_down=q, nemo_2bd_levels=(2, 22), cell_wet=wet, **common))
    ref = np.asarray(shortwave_penetration_tendency(sw_down=q, **common))
    assert np.abs(nemo[..., 22:]).max() == 0.0
    # the reference ladder does NOT -- which is what makes the row above a
    # measurement of the statement rather than of an always-zero tail.
    assert np.abs(ref[..., 22:]).max() > 0.0


def test_nemo_2bd_drops_the_infrared_band_below_nk0():
    """``traqsr.F90:676-683``: the deeper loop's attenuation is ``zz1*EXP``
    alone.  Displacing the INFRARED extinction length must therefore leave
    every level at or below ``nk0`` untouched while moving the ones above it.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.ocean.physics.shortwave_penetration import (
        JERLOV_TYPES, JerlovParams, ShortwavePenetrationConfig,
        shortwave_penetration_tendency)
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    _dm, cfg, _grid, z = _dino_ladder()
    wet = jnp.asarray(np.asarray(ndm.nemo_dino_mesh().tmask) > 0.5)
    q = jnp.full(wet.shape[:2], 200.0)
    base = JERLOV_TYPES["I"]
    JERLOV_TYPES["_probe"] = JerlovParams(R=base.R, zeta1=base.zeta1 * 1.5,
                                          zeta2=base.zeta2)
    try:
        def run(wt):
            return np.asarray(shortwave_penetration_tendency(
                sw_down=q, z_coord_dz_ref=z.dz_ref,
                z_coord_z_half_ref=z.z_half_ref,
                jacobian=jnp.ones(wet.shape[:2]),
                config=ShortwavePenetrationConfig(water_type=wt),
                rho_0=cfg.rho_0, c_sw=cfg.c_p,
                z_half_stretch=jnp.ones(wet.shape[:2]),
                nemo_2bd_levels=(2, 22), cell_wet=wet))
        a, b = run("I"), run("_probe")
    finally:
        JERLOV_TYPES.pop("_probe")
    # levels 0 and 1 carry the IR band and must move
    assert np.abs(a[..., :2] - b[..., :2]).max() > 0.0
    # levels 2..21 are visible-only and must NOT
    assert np.abs(a[..., 2:22] - b[..., 2:22]).max() == 0.0


def test_nemo_2bd_refuses_the_operands_it_cannot_run_without():
    import jax.numpy as jnp
    import pytest as _pytest
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, shortwave_penetration_tendency)
    dz = jnp.ones((4,))
    zh = -jnp.arange(5.0)
    kw = dict(sw_down=jnp.ones((2, 2)), z_coord_dz_ref=dz,
              z_coord_z_half_ref=zh, jacobian=jnp.ones((2, 2)),
              config=ShortwavePenetrationConfig(water_type="I"))
    with _pytest.raises(ValueError, match="z_half_stretch"):
        shortwave_penetration_tendency(nemo_2bd_levels=(1, 3), **kw)
    with _pytest.raises(ValueError, match="cell_wet"):
        shortwave_penetration_tendency(
            nemo_2bd_levels=(1, 3), z_half_stretch=jnp.ones((2, 2)), **kw)
    with _pytest.raises(ValueError, match="nk0"):
        shortwave_penetration_tendency(
            nemo_2bd_levels=(3, 1), z_half_stretch=jnp.ones((2, 2)),
            cell_wet=jnp.ones((2, 2, 4), dtype=bool), **kw)


def test_the_dino_card_selects_nemo_2bd_and_an_unknown_value_raises():
    import pytest as _pytest
    dm, cfg, _grid, _z = _dino_ladder()
    assert cfg.shortwave_penetration_ladder == "nemo_2bd"
    import dataclasses
    bad = dataclasses.replace(cfg, shortwave_penetration_ladder="jerlov")
    grid = dm.dino_lat_lon_grid(bad)
    z = dm.dino_lat_lon_vertical(grid, bad)
    state = dm.dino_lat_lon_state(grid, z, bad)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, bad)
    with _pytest.raises(ValueError, match="shortwave_penetration_ladder"):
        dm.apply_dino_lat_lon_surface_forcing(state, forcing, z, bad, 2700.0,
                                              t_seconds=2700.0,
                                              return_rate=True)
