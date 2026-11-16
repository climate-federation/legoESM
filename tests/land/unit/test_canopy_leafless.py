"""A column with no leaves (ice sheet, barren ground) must not carry leaf
physics: the leaf unknowns lose their equations, the leaf fluxes are zero, and
the ground / canopy-air / atmosphere balance is solved as before.

Measured before this change (2 deg LMIP, 24 steps x 6 passes, Jan and Jul):
49-55 % of ice-sheet and 17-36 % of barren column solves unconverged (against
3-4 % of vegetated ones), leaf temperatures drifting to ~370 K, and an equal
and opposite leaf H / LE pair of 0.17 W m-2 per class on leafless columns.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.land.canopy.solver as solver_mod
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import (
    CanopyForcingBundle, canopy_forward, solve_canopy_closure)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


_ICE = dict(LAI=0.0, SZA=60.0, La=200.0, epsf=0.97, epss=0.96, fSun=0.0,
            APAR_Sun=0.0, APAR_Sh=0.0, Vcmax25_Sun=0.0, Vcmax25_Sh=0.0,
            Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0, ASW_Sun=0.0, ASW_Sh=0.0,
            ASW_Soil=100.0, Ts_bc=250.0, Ca=370.0, Ps=70000.0, Ta=248.0,
            lam=2.45e6, Cp=1005.0, rhoa=1.0, Tv_atm=248.2, q_atm=0.0005,
            m=9.0, b0=0.01, alf=0.3, TgC=-20.0, fC4=0.0, fStress_soil=1.0,
            ur=5.0, CI=0.75, z0m=0.001, displa=0.0, z0=10.0, cv=0.0135,
            d_leaf=0.025, r_soil_surface=0.0, fwet=0.0)
_BARE = {**_ICE, "Ts_bc": 315.0, "Ta": 305.0, "Tv_atm": 306.0, "q_atm": 0.004,
         "ASW_Soil": 700.0, "La": 380.0, "Ps": 95000.0, "TgC": 25.0,
         "fStress_soil": 0.05, "SZA": 30.0, "r_soil_surface": 300.0}
_VEG = {**_BARE, "LAI": 3.0, "fSun": 0.5, "APAR_Sun": 600.0, "APAR_Sh": 150.0,
        "Vcmax25_Sun": 40.0, "Vcmax25_Sh": 20.0, "ASW_Sun": 250.0,
        "ASW_Sh": 80.0, "ASW_Soil": 150.0, "fStress_soil": 0.5}


def _b(d):
    return CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in d.items()})


def _x0(d):
    return jnp.array([d["Ta"], d["Ta"], 0.7 * d["Ca"], 0.7 * d["Ca"],
                      0.5 * (d["Ta"] + d["Ts_bc"]), d["q_atm"]])


def _solve(d, cc=None):
    cc = cc or CanopyConfig()
    b = _b(d)
    x, n, conv = solve_canopy_closure(_x0(d), b, cc)
    f = canopy_forward(x, b, cc.LE_module, cc.stomatal_model, cc.le_cap_mode,
                       cc.use_ta_for_photosynthesis)
    return x, int(n), bool(conv), f


@pytest.mark.parametrize("case", [_ICE, {**_ICE, "ASW_Soil": 250.0}, _BARE,
                                  {**_BARE, "ASW_Soil": 0.0, "Ts_bc": 290.0,
                                   "Ta": 288.0, "Tv_atm": 289.0}])
def test_leafless_column_certifies_with_leaf_state_pinned(case):
    x, n, conv, _ = _solve(case)
    assert conv and n <= 5, (conv, n)
    np.testing.assert_allclose(float(x[0]), float(x[4]), atol=1e-8)   # Tf_Sun = Tc
    np.testing.assert_allclose(float(x[1]), float(x[4]), atol=1e-8)   # Tf_Sh = Tc
    np.testing.assert_allclose(np.asarray(x[2:4]), case["Ca"], atol=1e-6)  # Ci = Ca


# The last case sits just above zero leaf area with a little leaf-absorbed
# shortwave, light and capacity, so a leaf net-radiation or carbon-uptake term
# left unmasked would show.
@pytest.mark.parametrize("case", [_ICE, _BARE, {**_BARE, "LAI": 5e-7, "fSun": 0.5,
                                                "ASW_Sun": 0.2, "ASW_Sh": 0.1,
                                                "APAR_Sun": 1.0, "APAR_Sh": 0.5,
                                                "Vcmax25_Sun": 1e-5, "Vcmax25_Sh": 1e-5}])
def test_leafless_column_has_no_leaf_fluxes_and_its_ground_heat_reaches_the_air(case):
    """Leaf fluxes are exactly zero; with no leaves, all the ground's sensible
    heat crosses the canopy air to the atmosphere (H_soil = rho cp (Tc - Ta) /
    rah_above), and the ground energy balance closes (Rn_soil = H + LE + G)."""
    x, _, conv, f = _solve(case)
    assert conv
    for k in ("H_Sun", "H_Sh", "LE_Sun", "LE_Sh", "LE_wet_Sun", "LE_wet_Sh",
              "gs_Sun", "gs_Sh", "An_Sun", "An_Sh", "Agross_Sun", "Agross_Sh",
              "Rn_Sun", "Rn_Sh"):
        assert float(f[k]) == 0.0, (k, float(f[k]))
    # Tolerance = the Newton certificate (squared residual reduced 1e8x), i.e.
    # ~1e-5 relative here; the leak this pins was 0.35 W m-2 (>= 30x the margin).
    h_air = case["rhoa"] * case["Cp"] * (float(x[4]) - case["Ta"]) / float(f["rah_above"])
    np.testing.assert_allclose(float(f["H_Soil"]), h_air, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(float(f["Rn_Soil"]),
                               float(f["H_Soil"] + f["LE_Soil"] + f["G"]), atol=1e-6)


def test_vegetated_and_sparse_columns_are_untouched(monkeypatch):
    """Columns with leaves (here LAI 3 and LAI 1e-3) take the unchanged
    equations: identical to a run with the leafless switch disabled."""
    for d in (_VEG, {**_BARE, "LAI": 1e-3, "fSun": 0.01, "APAR_Sun": 5.0,
                     "Vcmax25_Sun": 0.1, "ASW_Sun": 1.0}):
        x_on, n_on, c_on, f_on = _solve(d)
        monkeypatch.setattr(solver_mod, "_LEAFLESS_LAI", -1.0)
        x_off, n_off, c_off, f_off = _solve(d)
        monkeypatch.undo()
        np.testing.assert_array_equal(np.asarray(x_on), np.asarray(x_off))
        assert (n_on, c_on) == (n_off, c_off)
        for k in ("H_Sun", "LE_Sun", "H_Soil", "LE_Soil", "G", "Agross_Sun"):
            assert float(f_on[k]) == float(f_off[k]), k


def test_wet_leaves_on_a_leafless_column_add_no_vapour():
    """Intercepted water left when the leaf area has gone to zero (fwet > 0)
    must not reach the canopy air: the solved state equals the dry case."""
    x_dry, _, c_dry, _ = _solve(_BARE)
    x_wet, _, c_wet, f_wet = _solve({**_BARE, "fwet": 0.6})
    assert c_dry and c_wet
    np.testing.assert_array_equal(np.asarray(x_wet), np.asarray(x_dry))
    assert float(f_wet["LE_wet_Sun"]) == 0.0 and float(f_wet["LE_wet_Sh"]) == 0.0


def test_gradients_through_a_leafless_column_are_finite():
    """The jnp.where switches must not leak NaN into reverse mode: the ground
    sensible heat of a converged leafless column has a finite, positive
    derivative with respect to the ground temperature.  (The aerodynamic
    forcing -- Ta, wind -- is excluded: the Monin-Obukhov scan is documented as
    not differentiable in it, independently of this change.)"""
    cc = CanopyConfig()

    def h_soil(ts):
        b = _b({**_BARE, "Ts_bc": ts})
        x, _, _ = solve_canopy_closure(_x0(_BARE), b, cc)
        f = canopy_forward(x, b, cc.LE_module, cc.stomatal_model, cc.le_cap_mode,
                           cc.use_ta_for_photosynthesis)
        return f["H_Soil"]

    g = jax.grad(h_soil)(jnp.asarray(315.0))
    assert np.isfinite(float(g)) and float(g) > 0.0   # warmer ground, more H


def test_leafless_mask_and_sif_of_a_leafless_column():
    """One definition of "leafless" (LAI <= 1e-6), and with no light and no
    uptake the two-leaf SIF is exactly zero (what the canopy passes it on a
    leafless column)."""
    from legoesm.land.canopy.sif import SIFConfig, two_leaf_canopy_sif
    from legoesm.land.canopy.solver import leafless_mask
    m = np.asarray(leafless_mask(jnp.array([0.0, 1e-6, 2e-6, 0.05, 3.0])))
    assert m.tolist() == [True, True, False, False, False]
    z = jnp.zeros(3)
    sif = two_leaf_canopy_sif(z, jnp.full(3, 370.0), jnp.full(3, 40.0), z,
                              z, jnp.full(3, 370.0), jnp.full(3, 40.0), z, SIFConfig())
    assert np.all(np.asarray(sif) == 0.0)
