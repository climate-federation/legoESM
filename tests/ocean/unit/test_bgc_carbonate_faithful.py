"""Air-sea CO2 / carbonate-chemistry ORACLE-FAITHFULNESS tests.

The BGC carbonate module (``ocean/biogeochemistry/carbonate.py`` +
``gas_exchange.py``) is a stack of published closed forms — Weiss (1974) CO2
solubility, Lueker et al. (2000) K1/K2, Dickson (1990) borate K_B, Uppström
(1974) total borate, Wanninkhof (2014) Schmidt + transfer velocity, and the
Follows et al. (2006) borate-corrected quadratic pCO2/pH solver — yet the
existing suite only checks directional / range / shape / grad properties (K0 > 0,
cold > warm, undersaturated -> uptake).  These pin every closed form to
round-off against an INDEPENDENT scalar reimplementation with the published
coefficients, and cross-check the solver against a fully-converged same-chemistry
solve.

What is pinned, in order of authority:
1. Each closed form (Schmidt, k_w, K0, K1, K2, K_B, B_T) to rel 1e-12 across a
   T/S/U grid vs an independent reimplementation whose coefficients are typed
   from the papers (a coefficient typo in the module diverges from these).
2. CONVERGENCE-REFERENCE cross-check: the fixed-count Follows solver agrees with
   a fully-converged (200-iteration) solve of the SAME reduced chemistry to
   < 1 uatm over a T/S/DIC/ALK grid (135 points) — while a single un-iterated
   step (the prior form) would be > 10 uatm off at a cold high-pH column (the
   iteration does real work; a regression to one step fails here).  This shares
   the module's chemistry/clips/root choice, so it is a convergence check, NOT
   an independent-physics oracle.
3. ENVELOPE-HEALTH: across that grid the AD-safety guards do not bind (A_C,
   discriminant, [H+] clips all inactive) and the root-selection regime
   gamma = DIC/A_C > 0.5 holds, so the tests exercise the smooth physical path.
4. The air-sea flux sign convention (+ into the ocean) and assembly.
5. Departure canaries: the Schmidt T-clip, the borate-only alkalinity.
6. x64 AD-finiteness (incl. under jit) of the solver + flux gradients.

Precision: the BGC forms carry no explicit precision-policy promotion (plain
``jnp``), so they run in the input dtype; the autouse fixture enables x64 so the
float inputs are float64 for the round-off pins, and restores the entry state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from legoesm.ocean.biogeochemistry import carbonate as _carb
from legoesm.ocean.biogeochemistry.carbonate import (
    borate_equilibrium,
    carbonate_equilibria,
    co2_solubility,
    solve_carbonate_system,
    total_borate,
)
from legoesm.ocean.biogeochemistry.gas_exchange import (
    air_sea_co2_flux,
    gas_transfer_velocity,
    schmidt_number_co2,
)

from legoesm import constants


@pytest.fixture(autouse=True)
def _force_x64():
    """Enable float64 for the round-off pins; restore the entry state (captured
    per-test, not at import, so restoration is order-independent)."""
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", entry_x64)

# ---------------------------------------------------------------------------
# Independent coefficients, typed from the papers (NOT copied from the module).
# The round-off form pins below double as coefficient canaries: a module typo
# would diverge from these.
# ---------------------------------------------------------------------------
# Wanninkhof (2014) Table 1, Schmidt number of CO2 in seawater.
_O_SC = (2116.8, -136.25, 4.7353, -0.092307, 0.0007555)
_O_KW_COEFF = 0.251          # Wanninkhof (2014) k_w prefactor [cm/hr per (m/s)^2]
_O_SC_REF = 660.0            # reference Schmidt number
_O_CM_HR_TO_M_S = 1.0 / 360000.0   # exact 1 cm/hr = 1/(3600 s/hr * 100 cm/m) m/s
# Weiss (1974) Table I, K0 [mol/(kg*atm)].
_O_K0 = (-60.2409, 93.4517, 23.3585, 0.023517, -0.023656, 0.0047036)
# Lueker et al. (2000), total pH scale.
_O_PK1 = (3633.86, -61.2172, 9.6777, -0.011555, 0.0001152)
_O_PK2 = (471.78, 25.929, -3.16967, -0.01781, 0.0001122)
# Dickson (1990) borate K_B, total pH scale.
_O_KB_OVER_T = (-8966.90, -2890.53, -77.942, 1.728, -0.0996)
_O_KB_CONST = (148.0248, 137.1942, 1.62142)
_O_KB_LOGT = (-24.4344, -25.085, -0.2474)
_O_KB_SQRTS_T = 0.053105
# Uppström (1974) total boron.
_O_BT = 0.000416
_O_H_INIT = 8.1             # initial-guess pH for the borate iteration


# --- independent scalar reimplementations (plain Python math) ---------------

def _schmidt_oracle(T):
    Tc = min(max(T, 0.0), 40.0)                       # module clips to [0, 40]
    a, b, c, d, e = _O_SC
    return a + b * Tc + c * Tc**2 + d * Tc**3 + e * Tc**4


def _kw_oracle(U10, T):
    return _O_KW_COEFF * U10**2 * (_schmidt_oracle(T) / _O_SC_REF) ** (-0.5) * _O_CM_HR_TO_M_S


def _k0_oracle(T, S):
    Tk = T + float(constants.T_freeze)
    t = Tk / 100.0
    a1, a2, a3, b1, b2, b3 = _O_K0
    return math.exp(a1 + a2 / t + a3 * math.log(t) + S * (b1 + b2 * t + b3 * t**2))


def _k1k2_oracle(T, S):
    Tk = T + float(constants.T_freeze)
    a1, a2, a3, a4, a5 = _O_PK1
    b1, b2, b3, b4, b5 = _O_PK2
    pk1 = a1 / Tk + a2 + a3 * math.log(Tk) + a4 * S + a5 * S**2
    pk2 = b1 / Tk + b2 + b3 * math.log(Tk) + b4 * S + b5 * S**2
    return 10.0 ** (-pk1), 10.0 ** (-pk2)


def _kb_oracle(T, S):
    Tk = T + float(constants.T_freeze)
    rS = math.sqrt(max(S, 0.0))
    o1, o2, o3, o4, o5 = _O_KB_OVER_T
    c1, c2, c3 = _O_KB_CONST
    l1, l2, l3 = _O_KB_LOGT
    ln_kb = (
        (o1 + o2 * rS + o3 * S + o4 * S * rS + o5 * S**2) / Tk
        + c1 + c2 * rS + c3 * S
        + (l1 + l2 * rS + l3 * S) * math.log(Tk)
        + _O_KB_SQRTS_T * rS * Tk
    )
    return math.exp(ln_kb)


def _bt_oracle(S):
    return _O_BT * S / 35.0


def _follows_oracle(DIC_m3, ALK_m3, T, S, niter, rho):
    """Independent reimplementation of the iterated Follows (2006) solve
    (a cross-implementation regression oracle for the module's algebra)."""
    DICk = DIC_m3 / rho
    ALKk = ALK_m3 / rho
    K1, K2 = _k1k2_oracle(T, S)
    K0 = _k0_oracle(T, S)
    KB = _kb_oracle(T, S)
    BT = _bt_oracle(S)
    H = 10.0 ** (-_O_H_INIT)
    for _ in range(niter):
        A_B = BT * KB / (KB + H)
        A_C = max(ALKk - A_B, 1.0e-10)
        gamma = DICk / A_C
        b = K1 * (1.0 - gamma)
        c = K1 * K2 * (1.0 - 2.0 * gamma)
        disc = max(b**2 - 4.0 * c, 0.0)
        H = min(max((-b + math.sqrt(disc)) / 2.0, 1.0e-12), 1.0e-4)
    denom = max(H**2 + K1 * H + K1 * K2, 1.0e-30)
    CO2 = DICk * H**2 / denom
    pCO2 = CO2 / max(K0, 1.0e-10) * 1.0e6
    pH = -math.log10(min(max(H, 1.0e-14), 1.0e-4))
    return pCO2, pH


def _a(x):
    return jnp.array(float(x))


_RHO = float(constants.rho_ocean)

# (T [degC], S [PSU]) covering the fit ranges.
_TS = [(2.0, 33.0), (10.0, 34.0), (18.0, 35.0), (25.0, 35.0), (30.0, 36.0)]
# (DIC, ALK [umol/kg], T, S) ocean-surface states for the round-off regression.
_STATES = [(2000, 2300, 25, 35), (2100, 2300, 10, 34),
           (1950, 2250, 2, 33), (2050, 2350, 28, 36), (1900, 2200, 5, 32)]

# T/S/DIC/ALK grid (5*3*3*3 = 135 pts) for the convergence + envelope-health
# checks; ALK = DIC + dA keeps alkalinity realistically above DIC (gamma > 0.5).
_GRID = [(T, S, D, D + dA)
         for T in (2.0, 10.0, 18.0, 26.0, 34.0)
         for S in (31.0, 34.0, 37.0)
         for D in (1900, 2000, 2100)
         for dA in (200, 250, 350)]


def _replay_production_iterations(DIC_umol, ALK_umol, T, S):
    """Drive production's OWN ``_borate_quadratic_step`` for _N_BORATE_ITER passes
    from the same H_init and the same module equilibrium constants, so the per-
    pass PRE-CLIP margins (A_C, discriminant, root, gamma) are the solver's actual
    ones — the step function IS the code ``solve_carbonate_system`` runs, not a
    reimplementation.  Returns the worst-case margins at any pass + the final
    denominator/K0 + the final pCO2 (cross-checked against production in the
    test)."""
    DIC = DIC_umol * 1e-6 * _RHO
    ALK = ALK_umol * 1e-6 * _RHO
    DICk = DIC / _RHO
    ALKk = ALK / _RHO
    K1, K2 = carbonate_equilibria(_a(T), _a(S))
    KB = borate_equilibrium(_a(T), _a(S))
    BT = total_borate(_a(S))
    K0 = float(co2_solubility(_a(T), _a(S)))
    H = 10.0 ** (-_O_H_INIT) * jnp.ones(())
    min_AC = min_disc = min_H = min_gamma = math.inf
    max_H = -math.inf
    for _ in range(_carb._N_BORATE_ITER):
        H, A_C_raw, disc_raw, H_raw, gamma = _carb._borate_quadratic_step(
            H, DICk, ALKk, K1, K2, KB, BT)
        min_AC = min(min_AC, float(A_C_raw))
        min_disc = min(min_disc, float(disc_raw))
        min_H = min(min_H, float(H_raw))
        max_H = max(max_H, float(H_raw))
        min_gamma = min(min_gamma, float(gamma))
    Hf = float(H)
    K1f, K2f = float(K1), float(K2)
    denom_raw = Hf**2 + K1f * Hf + K1f * K2f      # final CO2_aq denominator, pre-clip
    pCO2 = DICk * Hf**2 / max(denom_raw, 1.0e-30) / max(K0, 1.0e-10) * 1.0e6
    return dict(pCO2=pCO2, min_AC=min_AC, min_disc=min_disc, min_H=min_H,
                max_H=max_H, min_gamma=min_gamma, denom=denom_raw, K0=K0)


# --- Wanninkhof Schmidt + transfer velocity --------------------------------

@pytest.mark.parametrize("T", [0.0, 5.0, 12.0, 20.0, 28.0, 36.0, 40.0])
def test_schmidt_matches_wanninkhof(T):
    got = float(schmidt_number_co2(_a(T)))
    assert got == pytest.approx(_schmidt_oracle(T), rel=1e-12, abs=0.0)


def test_schmidt_clips_outside_range():
    """DEPARTURE canary: the Wanninkhof fit is valid 0-40 degC; the module
    silently clips T to [0, 40] (Sc held flat, dSc/dT = 0 outside)."""
    assert float(schmidt_number_co2(_a(-5.0))) == pytest.approx(
        float(schmidt_number_co2(_a(0.0))), rel=1e-12)
    assert float(schmidt_number_co2(_a(45.0))) == pytest.approx(
        float(schmidt_number_co2(_a(40.0))), rel=1e-12)
    # gradient is exactly zero in the clipped region (not merely small)
    g = jax.grad(lambda t: schmidt_number_co2(t))(_a(45.0))
    assert float(g) == 0.0


@pytest.mark.parametrize("U10", [2.0, 5.0, 7.0, 12.0])
@pytest.mark.parametrize("T", [5.0, 15.0, 25.0])
def test_gas_transfer_velocity_matches(U10, T):
    got = float(gas_transfer_velocity(_a(U10), _a(T)))
    assert got == pytest.approx(_kw_oracle(U10, T), rel=1e-12, abs=0.0)


def test_kw_quadratic_in_wind():
    """k_w ∝ U10^2 (Wanninkhof 2014): doubling the wind quadruples the piston
    velocity exactly."""
    k1 = float(gas_transfer_velocity(_a(5.0), _a(20.0)))
    k2 = float(gas_transfer_velocity(_a(10.0), _a(20.0)))
    assert k2 / k1 == pytest.approx(4.0, rel=1e-12)


def test_kw_unit_conversion_factor():
    """The cm/hr -> m/s factor is the EXACT 1/(3600*100) used by the module
    (pinned via the ratio to the raw cm/hr rate).  Guards against a regression to
    the rounded 2.778e-6 (which biases the flux by ~8e-5 relative)."""
    U10, T = 7.0, 20.0
    k_m_s = float(gas_transfer_velocity(_a(U10), _a(T)))
    k_cm_hr = _O_KW_COEFF * U10**2 * (_schmidt_oracle(T) / _O_SC_REF) ** (-0.5)
    assert k_m_s / k_cm_hr == pytest.approx(_O_CM_HR_TO_M_S, rel=1e-12)
    assert _O_CM_HR_TO_M_S != 2.778e-6           # exact, not the rounded value


# --- Weiss K0 / Lueker K1,K2 / Dickson K_B / Uppström B_T -------------------

@pytest.mark.parametrize("T,S", _TS)
def test_co2_solubility_matches_weiss(T, S):
    got = float(co2_solubility(_a(T), _a(S)))
    assert got == pytest.approx(_k0_oracle(T, S), rel=1e-12, abs=0.0)


def test_co2_solubility_check_value():
    """Weiss (1974) K0(T=25, S=35) = 0.0283919 mol/(kg*atm) (regression anchor)."""
    assert float(co2_solubility(_a(25.0), _a(35.0))) == pytest.approx(
        0.0283918818, rel=1e-9)


@pytest.mark.parametrize("T,S", _TS)
def test_carbonate_equilibria_matches_lueker(T, S):
    K1, K2 = carbonate_equilibria(_a(T), _a(S))
    k1o, k2o = _k1k2_oracle(T, S)
    assert float(K1) == pytest.approx(k1o, rel=1e-12, abs=0.0)
    assert float(K2) == pytest.approx(k2o, rel=1e-12, abs=0.0)
    assert float(K1) > float(K2)          # first dissociation stronger


@pytest.mark.parametrize("T,S", _TS)
def test_borate_equilibrium_matches_dickson(T, S):
    got = float(borate_equilibrium(_a(T), _a(S)))
    assert got == pytest.approx(_kb_oracle(T, S), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("S", [0.0, 20.0, 35.0, 42.0])
def test_total_borate_matches_uppstrom(S):
    got = float(total_borate(_a(S)))
    assert got == pytest.approx(_bt_oracle(S), rel=1e-12, abs=0.0)


# --- Follows solver: regression oracle + converged cross-check --------------

@pytest.mark.parametrize("DIC_umol,ALK_umol,T,S", _STATES)
def test_solve_carbonate_matches_iterated_oracle(DIC_umol, ALK_umol, T, S):
    """Production solver == an independent reimplementation of the SAME iterated
    Follows algebra (cross-implementation regression oracle), to round-off."""
    DIC = DIC_umol * 1e-6 * _RHO
    ALK = ALK_umol * 1e-6 * _RHO
    pCO2, pH = solve_carbonate_system(_a(DIC), _a(ALK), _a(T), _a(S))
    p_o, ph_o = _follows_oracle(DIC, ALK, T, S, _carb._N_BORATE_ITER, _RHO)
    assert float(pCO2) == pytest.approx(p_o, rel=1e-11, abs=0.0)
    assert float(pH) == pytest.approx(ph_o, rel=1e-11, abs=0.0)


def test_solver_converges_to_reference_over_grid():
    """CONVERGENCE-REFERENCE check (NOT independent physics — same reduced
    chemistry, root choice, clips and initial guess, just more iterations): the
    fixed-count solve agrees with a fully-converged (200-step) reference to
    < 1 uatm over the whole 135-point T/S/DIC/ALK grid.  Assert the worst-case
    maximum, not a single point."""
    worst = 0.0
    for T, S, DIC_umol, ALK_umol in _GRID:
        DIC = DIC_umol * 1e-6 * _RHO
        ALK = ALK_umol * 1e-6 * _RHO
        pCO2, _ = solve_carbonate_system(_a(DIC), _a(ALK), _a(T), _a(S))
        p_conv, _ = _follows_oracle(DIC, ALK, T, S, 200, _RHO)
        worst = max(worst, abs(float(pCO2) - p_conv))
    assert worst < 1.0, f"worst |pCO2 - converged| = {worst} uatm over the grid"


def test_envelope_health_guards_inactive_and_root_regime():
    """Across the whole grid, at EVERY one of the 8 production passes, the
    AD-safety guards do NOT bind and the root-selection regime holds, so the pins
    exercise the smooth physical path (not a clipped one) and the larger (+) root
    is the unique positive [H+]:
      - pre-clip carbonate-alkalinity residual A_C > 1e-10 (A_C clip inactive),
      - pre-clip discriminant > 0 (discriminant clip inactive),
      - pre-clip [H+] strictly in (1e-12, 1e-4) (the H clip inactive),
      - final denominator > 1e-30 and K0 > 1e-10 (CO2_aq/pCO2 clips inactive),
      - gamma = DIC/A_C > 0.5 (unique positive root).
    The margins come from driving production's OWN ``_borate_quadratic_step`` (the
    same code ``solve_carbonate_system`` runs), so they ARE the solver's per-pass
    quantities; the final-pCO2 cross-check confirms the driver reproduces
    ``solve_carbonate_system`` exactly."""
    for T, S, DIC_umol, ALK_umol in _GRID:
        r = _replay_production_iterations(DIC_umol, ALK_umol, T, S)
        pCO2_prod = float(solve_carbonate_system(
            _a(DIC_umol * 1e-6 * _RHO), _a(ALK_umol * 1e-6 * _RHO), _a(T), _a(S))[0])
        tag = f"(T={T},S={S},DIC={DIC_umol},ALK={ALK_umol})"
        assert r["pCO2"] == pytest.approx(pCO2_prod, rel=1e-9), f"replay != prod {tag}"
        assert r["min_AC"] > 1.0e-10, f"A_C clip binds {tag}: {r['min_AC']}"
        assert r["min_disc"] > 0.0, f"discriminant clip binds {tag}: {r['min_disc']}"
        assert 1.0e-12 < r["min_H"] and r["max_H"] < 1.0e-4, f"H clip binds {tag}"
        assert r["denom"] > 1.0e-30 and r["K0"] > 1.0e-10, f"denom/K0 clip binds {tag}"
        assert r["min_gamma"] > 0.5, f"root regime gamma>0.5 violated {tag}: {r['min_gamma']}"


def test_iteration_matters_nonvacuous():
    """The borate iteration does real work: at a cold high-pH column a SINGLE
    un-iterated step (the prior form) is > 10 uatm off the converged solve,
    while the production fixed-count solve is < 1 uatm — so a regression to one
    step would fail ``test_solver_converges_to_reference_over_grid``."""
    DIC = 1950 * 1e-6 * _RHO
    ALK = 2250 * 1e-6 * _RHO
    T, S = 2.0, 33.0
    p_conv, _ = _follows_oracle(DIC, ALK, T, S, 200, _RHO)
    p_one, _ = _follows_oracle(DIC, ALK, T, S, 1, _RHO)
    p_prod, _ = solve_carbonate_system(_a(DIC), _a(ALK), _a(T), _a(S))
    assert abs(p_one - p_conv) > 10.0            # one step is materially wrong
    assert abs(float(p_prod) - p_conv) < 1.0     # production converged
    assert _carb._N_BORATE_ITER >= 3             # count pin


# --- air-sea flux sign convention + assembly -------------------------------

def test_air_sea_flux_sign_convention():
    """F = k_w*K0*rho*(pCO2_atm - pCO2_ocean), POSITIVE into the ocean.  With
    k_w, K0, rho > 0 the flux sign is exactly sign(pCO2_atm - pCO2_ocean)."""
    DIC = _a(2000e-6 * _RHO)
    ALK = _a(2300e-6 * _RHO)
    T, S, U = _a(25.0), _a(35.0), _a(7.0)
    pco2_ocean = float(solve_carbonate_system(DIC, ALK, T, S)[0])
    hi = air_sea_co2_flux(DIC, ALK, T, S, U, pCO2_atm=pco2_ocean + 50.0)
    lo = air_sea_co2_flux(DIC, ALK, T, S, U, pCO2_atm=pco2_ocean - 50.0)
    eq = air_sea_co2_flux(DIC, ALK, T, S, U, pCO2_atm=pco2_ocean)
    assert float(hi.flux_co2) > 0.0              # atm richer -> uptake (+)
    assert float(lo.flux_co2) < 0.0              # ocean richer -> outgas (-)
    assert float(eq.flux_co2) == pytest.approx(0.0, abs=1e-12)


def test_air_sea_flux_assembly():
    """flux == k_w * K0 * rho_sw * (pCO2_atm - pCO2_ocean) * 1e-6, exactly."""
    DIC = _a(2050e-6 * _RHO)
    ALK = _a(2300e-6 * _RHO)
    T, S, U = _a(18.0), _a(34.0), _a(9.0)
    patm = 410.0
    out = air_sea_co2_flux(DIC, ALK, T, S, U, pCO2_atm=patm)
    K0 = float(co2_solubility(T, S))
    expected = float(out.k_w) * K0 * _RHO * (patm - float(out.pCO2_ocean)) * 1e-6
    assert float(out.flux_co2) == pytest.approx(expected, rel=1e-11, abs=0.0)


# --- AD-safety --------------------------------------------------------------

def test_solver_and_flux_grads_finite_eager_and_jit():
    """grad of pCO2 through the iterated solver (wrt DIC, ALK, T, S) and of the
    flux (wrt U10, T) is finite at MULTIPLE grid states, computed BOTH eagerly
    and under jit — the unrolled borate loop (a static count) and the clip/sqrt
    guards stay AD-safe."""
    def _pco2(dic, alk, T, S):
        return solve_carbonate_system(dic, alk, T, S)[0]

    def _flux(U, T, dic, alk, S):
        return air_sea_co2_flux(dic, alk, T, S, U, pCO2_atm=420.0).flux_co2

    eager_pco2 = jax.grad(_pco2, argnums=(0, 1, 2, 3))
    jit_pco2 = jax.jit(eager_pco2)
    eager_flux = jax.grad(_flux, argnums=(0, 1))
    jit_flux = jax.jit(eager_flux)
    for T, S, DIC_umol, ALK_umol in (_GRID[0], _GRID[67], _GRID[-1]):
        dic = _a(DIC_umol * 1e-6 * _RHO)
        alk = _a(ALK_umol * 1e-6 * _RHO)
        for grad_pco2 in (eager_pco2, jit_pco2):
            gs = grad_pco2(dic, alk, _a(T), _a(S))
            assert all(bool(jnp.isfinite(g)) for g in gs)
            assert any(float(jnp.abs(g)) > 0.0 for g in gs)
        for grad_flux in (eager_flux, jit_flux):
            gf = grad_flux(_a(7.0), _a(T), dic, alk, _a(S))
            assert all(bool(jnp.isfinite(g)) for g in gf)
