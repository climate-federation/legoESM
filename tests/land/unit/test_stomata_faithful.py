"""Stomatal-conductance + coupled A-gs-Ci ORACLE-FAITHFULNESS tests.

``land/stomata`` implements the Ball-Berry (1987) and Medlyn (2011) stomatal
conductance kernels and the coupled Farquhar-stomata leaf solver.  The existing
test_stomata.py / test_stomatal_kernels.py / test_canopy_stomatal.py are
behavioral (gs floors at g0/b0, rises with A/RH, falls with VPD/CO2, VPD floor,
differentiable) — they never pin the closed FORMS or the coupled closure.

These pin the conductance ALGEBRA to round-off (rel 1e-9) against an independent
scalar reimplementation, and pin the coupled solver to its defining fixed point:
  * Ball-Berry:  gs = max(g0 + m max(A,0) RH / max(Cs,1), g0)
  * Medlyn USO:  gs = max(g0 + 1.6 (1 + g1/sqrt(max(VPD,0.05))) max(A,0)/max(Cs,1), g0)
  * Coupled closure: the returned (Ci, A, gs) is a fixed point of the A-gs-Ci
    map — gs is exactly the kernel of the returned A, and Fick's law
    Ci = Ca - 1.6 max(A,0)/gs holds (residual -> 0 in float64).  The FvCB rate
    A(Ci) itself is TAKEN AS GIVEN (pinned elsewhere), not re-verified here.

Independence: the Ball-Berry/Medlyn coefficients and the 1.6 H2O:CO2 diffusivity
ratio are local _O_* literals, canaried module == _O_* == value; the FvCB
biochemistry is NOT re-pinned here (covered by test_canopy_photosynthesis_
faithful.py).  RH/VPD in the closure check are recomputed with the SHARED,
already-pinned thermo saturation helpers (must-use, not the SUT).  Departures
(max(A,0), max(Cs,1), VPD/gs floors, Ci clip) are reproduced by the oracle.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the rel-1e-9 pins; restore the process-entry state in
    finally so selecting a single test never leaks x64 into another module."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm.land import stomata as sto                              # noqa: E402
from legoesm.land.stomata import (                                   # noqa: E402
    ball_berry_gs, medlyn_gs, solve_coupled_farquhar_ci, StomataConfig,
    DIFFUSIVITY_RATIO_H2O_CO2, _VPD_FLOOR_KPA, _CI_CA_INIT_RATIO,
)
from legoesm.thermo import (                                         # noqa: E402
    saturation_vapor_pressure, vapor_pressure_from_specific_humidity,
)

# Independent oracle literals (canaried in test_stomata_constants_match_literals).
_O_DIFF_RATIO = 1.6      # Medlyn USO prefactor / A->Ci back-calc H2O:CO2 ratio
_O_VPD_FLOOR = 0.05      # [kPa] guards 1/sqrt(VPD)
_O_CS_FLOOR = 1.0        # [umol/mol] guards A/Cs
_O_CI_CA_INIT = 0.7      # initial Ci:Ca guess


def _ball_berry_oracle(A, RH, Cs, m, b0):
    """Independent Ball-Berry (1987) gs."""
    A_pos = max(A, 0.0)
    Cs_safe = max(Cs, _O_CS_FLOOR)
    return max(b0 + m * A_pos * RH / Cs_safe, b0)


def _medlyn_oracle(A, VPD_kPa, Cs, g1, g0):
    """Independent Medlyn (2011) USO gs."""
    A_pos = max(A, 0.0)
    Cs_safe = max(Cs, _O_CS_FLOOR)
    VPD = max(VPD_kPa, _O_VPD_FLOOR)
    return max(g0 + _O_DIFF_RATIO * (1.0 + g1 / math.sqrt(VPD)) * A_pos / Cs_safe, g0)


def _arr(x):
    return jnp.array([float(x)])


# --- Ball-Berry closed-form pin ------------------------------------------------

@pytest.mark.parametrize("A,RH,Cs,m,b0", [
    (12.0, 0.70, 400.0, 9.0, 0.01),    # typical daytime
    (25.0, 0.90, 380.0, 9.0, 0.01),    # humid, high A
    (4.0, 0.40, 250.0, 4.0, 0.04),     # C4-like slope/intercept, dry
    (-5.0, 0.60, 400.0, 9.0, 0.01),    # A<0 -> floors at b0 (max(A,0))
    (10.0, 0.70, 0.5, 9.0, 0.01),      # Cs<1 -> Cs floor engages
])
def test_ball_berry_matches_oracle(A, RH, Cs, m, b0):
    """ball_berry_gs matches the independent Ball-Berry reimplementation to
    round-off, including the max(A,0) and max(Cs,1) guard branches."""
    got = float(ball_berry_gs(_arr(A), _arr(RH), _arr(Cs), m, b0)[0])
    exp = _ball_berry_oracle(A, RH, Cs, m, b0)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


# --- Medlyn USO closed-form pin ------------------------------------------------

@pytest.mark.parametrize("A,VPD,Cs,g1,g0", [
    (12.0, 1.0, 400.0, 4.0, 0.01),     # typical
    (25.0, 0.5, 380.0, 4.0, 0.01),     # low VPD, high A
    (8.0, 3.0, 300.0, 2.0, 0.01),      # high VPD
    (-3.0, 1.0, 400.0, 4.0, 0.01),     # A<0 -> floors at g0
    (10.0, 1e-6, 400.0, 4.0, 0.01),    # VPD below floor -> 1/sqrt guard engages
])
def test_medlyn_matches_oracle(A, VPD, Cs, g1, g0):
    """medlyn_gs matches the independent Medlyn USO reimplementation to round-off,
    including the 1.6 prefactor, the 1/sqrt(VPD) floor, and the max(A,0) floor."""
    got = float(medlyn_gs(_arr(A), _arr(VPD), _arr(Cs), g1, g0)[0])
    exp = _medlyn_oracle(A, VPD, Cs, g1, g0)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


def test_ball_berry_and_medlyn_are_distinct_models():
    """Non-vacuity: at identical (A, Cs, g0) the two kernels give genuinely
    different gs — Ball-Berry scales by RH/Cs, Medlyn by (1+g1/sqrt(VPD))/Cs — so
    they are not accidentally the same expression."""
    A, Cs, g0 = 15.0, 400.0, 0.01
    bb = float(ball_berry_gs(_arr(A), _arr(0.7), _arr(Cs), 9.0, g0)[0])
    med = float(medlyn_gs(_arr(A), _arr(1.0), _arr(Cs), 4.0, g0)[0])
    assert abs(bb - med) > 0.01
    assert bb > g0 and med > g0


# --- constant canaries ---------------------------------------------------------

def test_stomata_constants_match_literals():
    """The module gas-exchange constants equal both the independent oracle
    literals and their published values (module == _O_* == value)."""
    assert DIFFUSIVITY_RATIO_H2O_CO2 == _O_DIFF_RATIO == 1.6   # H2O:CO2 diffusivity
    assert _VPD_FLOOR_KPA == _O_VPD_FLOOR == 0.05
    assert _CI_CA_INIT_RATIO == _O_CI_CA_INIT == 0.7


# --- coupled A-gs-Ci closure ---------------------------------------------------

def _coupled_inputs():
    cfg = StomataConfig(enabled=True, stomata_model="ball_berry", Vc_max25=60.0,
                        g0=0.01, g1_bb=9.0, k_ext=0.5, beta_soil_min=0.01)
    return dict(T_leaf=_arr(298.15), sw_down=_arr(600.0), co2_ppmv=400.0,
                q_air=_arr(0.012), p_surface=_arr(101325.0), LAI=_arr(3.0),
                beta_soil=_arr(0.9), config=cfg)


def _recompute_rh_vpd(T_leaf, q_air, p_surface):
    e_sat = float(saturation_vapor_pressure(_arr(T_leaf))[0])
    e_air = float(vapor_pressure_from_specific_humidity(_arr(q_air), _arr(p_surface))[0])
    VPD_kPa = max(e_sat - e_air, 0.0) / 1000.0
    RH = min(max(e_air / max(e_sat, 1.0), 0.0), 1.0)
    return RH, VPD_kPa


@pytest.mark.parametrize("model", ["ball_berry", "medlyn"])
def test_coupled_solution_satisfies_fick_and_kernel(model):
    """The converged coupled state simultaneously satisfies BOTH:
      (a) Fick's stomatal-diffusion law  Ci = Ca - 1.6 max(A,0)/gs  (to the
          iteration tolerance), and
      (b) gs == the Ball-Berry/Medlyn kernel of the returned A_net (exactly).
    This is the DEFINITION of the coupled A-gs-Ci solution — the whole point of
    the solver — and is never checked by the behavioral suite."""
    kw = _coupled_inputs()
    kw["config"] = kw["config"]._replace(stomata_model=model, n_iter_ags=60)
    st = solve_coupled_farquhar_ci(**kw)
    Ca = 400.0
    gs, A_net, Ci = float(st.gs[0]), float(st.A_net[0]), float(st.Ci[0])
    RH, VPD = _recompute_rh_vpd(298.15, 0.012, 101325.0)

    # (a) Fick's law closure. The fixed point is stable in float64 so the
    # residual converges to exactly 0 by 60 iters; a tight absolute bound (1e-6
    # ppm, ~1e8x tighter than the ~1e-3 residual at 2 iters) leaves no slack.
    Ci_fick = Ca - _O_DIFF_RATIO * max(A_net, 0.0) / max(gs, kw["config"].g0)
    Ci_fick = min(max(Ci_fick, 1.0), Ca)          # same clip the solver applies
    assert Ci == pytest.approx(Ci_fick, rel=0.0, abs=1e-6)

    # (b) gs is exactly the kernel of the returned A_net (round-off).
    if model == "medlyn":
        gs_k = _medlyn_oracle(A_net, VPD, Ca, kw["config"].g1_med, kw["config"].g0)
    else:
        gs_k = _ball_berry_oracle(A_net, RH, Ca, kw["config"].g1_bb, kw["config"].g0)
    assert gs == pytest.approx(gs_k, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("model", ["ball_berry", "medlyn"])
def test_coupled_fick_residual_shrinks_with_iterations(model):
    """Non-vacuity of the closure (BOTH schemes): the Fick residual
    |Ci - (Ca - 1.6 A/gs)| genuinely SHRINKS with more iterations (it is a
    convergent fixed point, not trivially satisfied), and the 1.6 ratio is
    LOAD-BEARING — using 1.4 instead breaks the converged closure."""
    Ca = 400.0

    def residual(n, ratio=_O_DIFF_RATIO):
        kw = _coupled_inputs()
        kw["config"] = kw["config"]._replace(stomata_model=model, n_iter_ags=n)
        st = solve_coupled_farquhar_ci(**kw)
        gs, A_net, Ci = float(st.gs[0]), float(st.A_net[0]), float(st.Ci[0])
        Ci_fick = min(max(Ca - ratio * max(A_net, 0.0) / max(gs, kw["config"].g0), 1.0), Ca)
        return abs(Ci - Ci_fick)

    assert residual(60) < residual(2)             # convergent fixed point
    assert residual(60) < 1e-6                     # converged to ~0 with 1.6
    assert residual(60, ratio=1.4) > 1e-2          # wrong ratio breaks the closure


# --- AD-safety -----------------------------------------------------------------

def test_stomata_grad_finite_and_nonzero_x64_and_float32():
    """grad of both kernels and of the coupled gs is finite AND, in the interior,
    matches the EXPECTED nonzero analytic slope (finiteness alone would pass a
    broken zero-gradient path); the max(A,0)/VPD guard branches are separately
    checked finite (and zero).  Both x64 and float32."""
    def _check(rel):
        # interior: d(ball_berry)/dA = m RH / Cs; d(medlyn)/dA = 1.6(1+g1/sqrt(VPD))/Cs.
        gb = jax.grad(lambda A: ball_berry_gs(A, jnp.array(0.7), jnp.array(400.0),
                                              9.0, 0.01))(jnp.array(12.0))
        gm = jax.grad(lambda A: medlyn_gs(A, jnp.array(1.0), jnp.array(400.0),
                                          4.0, 0.01))(jnp.array(12.0))
        assert float(gb) == pytest.approx(9.0 * 0.7 / 400.0, rel=rel)
        assert float(gm) == pytest.approx(1.6 * (1.0 + 4.0 / 1.0) / 400.0, rel=rel)
        # coupled solve: grad of gs wrt sw_down (through the fixed-point + FvCB);
        # light-limited, so strictly positive.
        kw = _coupled_inputs()

        def gs_of_sw(sw):
            k = dict(kw)
            k["sw_down"] = jnp.reshape(sw, (1,))
            return solve_coupled_farquhar_ci(**k).gs[0]

        gc = jax.grad(gs_of_sw)(jnp.array(600.0))
        assert bool(jnp.isfinite(gc)) and float(gc) > 0.0, gc
        # guard branches: A<0 floors both kernels -> grad exactly 0 and finite;
        # VPD below its floor -> grad wrt VPD finite (floored).
        gb0 = jax.grad(lambda A: ball_berry_gs(A, jnp.array(0.7), jnp.array(400.0),
                                               9.0, 0.01))(jnp.array(-5.0))
        gm0 = jax.grad(lambda A: medlyn_gs(A, jnp.array(1.0), jnp.array(400.0),
                                           4.0, 0.01))(jnp.array(-3.0))
        gv = jax.grad(lambda v: medlyn_gs(jnp.array(10.0), v, jnp.array(400.0),
                                          4.0, 0.01))(jnp.array(1e-6))
        assert float(gb0) == 0.0 and float(gm0) == 0.0
        assert bool(jnp.isfinite(gv))

    _check(rel=1e-9)
    jax.config.update("jax_enable_x64", False)
    _check(rel=1e-4)   # float32; autouse fixture restores the entry state afterwards
