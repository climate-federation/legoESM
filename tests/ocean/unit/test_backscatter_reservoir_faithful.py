"""Faithfulness pins for the Jansen-Held (2014) backscatter energy reservoir.

Target: the grid-free subgrid-KE-reservoir closed forms in
``legoesm.ocean.physics.lateral_mixing.backscatter`` — ``update_eddy_energy``
(prognostic budget), ``diagnostic_eddy_energy`` (standing-equilibrium reservoir),
``cfl_cap_eddy_energy`` (the viscous-CFL bound on the reservoir), and the
backscatter viscosity coefficient ``_A_bs_h_cgrid`` (nu_bs = c_bs*Delta*sqrt(E)).

Most-trustful source
--------------------
Jansen & Held (2014), "Parameterizing subgrid-scale eddy effects using
energetically consistent backscatter" (Ocean Modelling 80, 36-48) introduced the
negative-viscosity backscatter with a global energy budget; Jansen et al. (2015,
Ocean Modelling 94, 15-26) added the LOCAL prognostic subgrid-EKE budget used
here.  The prognostic subgrid-KE reservoir E evolves as

    dE/dt = eta * eps_dissipation - eps_backscatter - E / tau,

clamped to [E_min, E_max]; the backscatter (negative-Laplacian) viscosity is
nu_bs = c_bs * Delta * sqrt(E) (Delta = sqrt(area) the grid length).  The
diagnostic closure is a local-equilibrium APPROXIMATION that drops
eps_backscatter and sets dE/dt = 0 -> E_eq = eta * tau * eps_dissipation.

The existing test_backscatter*.py are behavioral (0 exact-magnitude assertions);
this pins the reservoir budget, the tau day->second conversion, both clamps, the
diagnostic equilibrium (and its deliberate NO-E_min-floor energetic-consistency
rule), the CFL-cap inversion + its nu_bs==nu_max truth tier, and the nu_bs
coefficient (zero at E<=0, AD-safe).  (The grid-coupled momentum tendency
-nu_bs*grad^2 u is exercised behaviorally in test_backscatter.py.)
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.physics.lateral_mixing import backscatter as bs
from legoesm.ocean.physics.lateral_mixing.backscatter import (
    BackscatterConfig,
    cfl_cap_eddy_energy,
    diagnostic_eddy_energy,
    update_eddy_energy,
)

jax.config.update("jax_enable_x64", True)

_CFG = BackscatterConfig(enabled=True, c_bs=0.01, tau_relax_days=10.0,
                         E_min=1e-4, E_max=0.1, efficiency=0.9)
_TAU_S = 10.0 * 86400.0


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# 1. Prognostic reservoir budget (update_eddy_energy).
# ---------------------------------------------------------------------------
def test_update_reservoir_exact_budget():
    # E_new = E + dt*(eta*eps_diss - eps_bs - E/tau), no clamp active.
    E = _a([1e-2, 2e-2])
    eps_d = _a([3e-8, 1e-8])
    eps_b = _a([1e-9, 2e-9])
    dt = 100.0
    out = np.asarray(update_eddy_energy(E, dt, eps_d, eps_b, _CFG))
    exp = np.asarray(E) + dt * (0.9 * np.asarray(eps_d) - np.asarray(eps_b)
                                - np.asarray(E) / _TAU_S)
    np.testing.assert_allclose(out, exp, rtol=1e-12)


def test_update_tau_days_to_seconds():
    # The damping uses tau = tau_relax_days * 86400 (not days).  Pure decay
    # (no sources): dE = -dt*E/tau.
    E = _a([5e-2])
    dt = 1000.0
    out = float(update_eddy_energy(E, dt, _a([0.0]), _a([0.0]), _CFG)[0])
    np.testing.assert_allclose(out, 5e-2 * (1.0 - dt / _TAU_S), rtol=1e-12)


def test_update_efficiency_scales_source():
    # Only the dissipation SOURCE is scaled by eta (backscatter sink is not).
    E, eps_d, dt = _a([1e-2]), _a([2e-8]), 100.0
    base = float(update_eddy_energy(E, dt, eps_d, _a([0.0]), _CFG)[0])
    hi = float(update_eddy_energy(E, dt, eps_d, _a([0.0]),
                                  _CFG._replace(efficiency=2.0 * _CFG.efficiency))[0])
    np.testing.assert_allclose(hi - base, dt * 0.9 * 2e-8, rtol=1e-9)   # extra eta*eps_d*dt


def test_update_clamps_both_bounds():
    # Huge source -> E_max ceiling; huge sink -> E_min floor.
    up = float(update_eddy_energy(_a([0.05]), 1e6, _a([1.0]), _a([0.0]), _CFG)[0])
    dn = float(update_eddy_energy(_a([0.05]), 1e6, _a([0.0]), _a([1.0]), _CFG)[0])
    np.testing.assert_allclose(up, _CFG.E_max, rtol=1e-12)
    np.testing.assert_allclose(dn, _CFG.E_min, rtol=1e-12)


def test_update_disabled_and_czero_are_noop():
    E = _a([1e-2, 2e-2])
    off = update_eddy_energy(E, 100.0, _a([1e-6, 1e-6]), _a([0.0, 0.0]),
                             _CFG._replace(enabled=False))
    cz = update_eddy_energy(E, 100.0, _a([1e-6, 1e-6]), _a([0.0, 0.0]),
                            _CFG._replace(c_bs=0.0))
    np.testing.assert_array_equal(np.asarray(off), np.asarray(E))
    np.testing.assert_array_equal(np.asarray(cz), np.asarray(E))


def test_update_differentiable():
    def loss(E):
        return jnp.sum(update_eddy_energy(E, 100.0, _a([1e-8]), _a([0.0]), _CFG))
    g = jax.grad(loss)(_a([1e-2]))
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# 2. Diagnostic (standing-equilibrium) reservoir.
# ---------------------------------------------------------------------------
def test_diagnostic_equilibrium_form():
    # E_eq = eta * tau * eps_diss (dE/dt = 0, eps_bs dropped).
    eps_d = _a([1e-8, 2e-9])
    out = np.asarray(diagnostic_eddy_energy(eps_d, _CFG))
    np.testing.assert_allclose(out, 0.9 * _TAU_S * np.asarray(eps_d), rtol=1e-12)


def test_diagnostic_clamped_zero_to_emax():
    # Large eps_diss -> E_max; the LOWER clamp is 0 (see next test), not E_min.
    big = float(diagnostic_eddy_energy(_a([1.0]), _CFG)[0])
    np.testing.assert_allclose(big, _CFG.E_max, rtol=1e-12)


def test_diagnostic_no_emin_floor_zero_sink_zero_reservoir():
    # ENERGETIC-CONSISTENCY canary: where the scale-selective dissipation
    # vanishes, the DIAGNOSTIC reservoir must be EXACTLY 0 (E_min is deliberately
    # NOT applied) -> zero sink gives zero backscatter.  A version that floored at
    # E_min (1e-4 here) would manufacture standing reservoir energy with no sink.
    out = float(diagnostic_eddy_energy(_a([0.0]), _CFG)[0])
    assert out == 0.0
    assert _CFG.E_min > 0.0                       # the floor exists but is not applied here
    # negative (garbage) eps_diss is also floored to 0.
    assert float(diagnostic_eddy_energy(_a([-1e-6]), _CFG)[0]) == 0.0


# ---------------------------------------------------------------------------
# 3. Viscous-CFL cap on the reservoir.
# ---------------------------------------------------------------------------
def test_cfl_cap_inversion():
    # E_cap = (nu_max / (c_bs*Delta))^2 ; large E is capped, small E passes.
    delta, nu_max = _a([1e4]), _a([50.0])
    E_cap = (50.0 / (_CFG.c_bs * 1e4)) ** 2
    capped = float(cfl_cap_eddy_energy(_a([1e3]), nu_max, delta, _CFG)[0])   # E huge
    passed = float(cfl_cap_eddy_energy(_a([1e-6]), nu_max, delta, _CFG)[0])  # E tiny
    np.testing.assert_allclose(capped, E_cap, rtol=1e-12)
    np.testing.assert_allclose(passed, 1e-6, rtol=1e-12)


def test_cfl_cap_truth_tier_nu_bs_equals_nu_max():
    # Compose the ACTUAL cap + coefficient fns: at E = E_cap the applied
    # coefficient _A_bs_h_cgrid (with Delta = sqrt(area) = delta) must equal
    # nu_max -- the cap's defining purpose.  (Exact up to sqrt rounding; the
    # exact pointwise ceiling is the later minimum in
    # backscatter_coefficients_cgrid.)
    delta, nu_max = 1e4, 37.0
    E_cap = float(cfl_cap_eddy_energy(_a([1e6]), _a([nu_max]), _a([delta]), _CFG)[0])
    nu_bs = float(bs._A_bs_h_cgrid(_a([E_cap]), _mock_grid(_a([delta ** 2])), _CFG.c_bs)[0])
    np.testing.assert_allclose(nu_bs, nu_max, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Backscatter viscosity coefficient nu_bs = c_bs * sqrt(area) * sqrt(E).
# ---------------------------------------------------------------------------
def _mock_grid(area):
    return types.SimpleNamespace(area=_a(area))


def test_nu_bs_coefficient_form():
    E = _a([4e-2, 1e-2])
    area = _a([1e8, 4e8])                          # Delta = sqrt(area) = 1e4, 2e4
    nu = np.asarray(bs._A_bs_h_cgrid(E, _mock_grid(area), _CFG.c_bs))
    exp = _CFG.c_bs * np.sqrt(np.asarray(area)) * np.sqrt(np.asarray(E))
    np.testing.assert_allclose(nu, exp, rtol=1e-12)
    # c_bs plumbing: 2x c_bs -> 2x nu_bs.
    nu2 = np.asarray(bs._A_bs_h_cgrid(E, _mock_grid(area), 2.0 * _CFG.c_bs))
    np.testing.assert_allclose(nu2, 2.0 * nu, rtol=1e-12)


def test_nu_bs_zero_at_nonpositive_e():
    # sqrt(max(E,0)) is EXACTLY zero at E <= 0 (no manufactured negative viscosity).
    nu = np.asarray(bs._A_bs_h_cgrid(_a([0.0, -1e-3]), _mock_grid(_a([1e8, 1e8])), _CFG.c_bs))
    assert np.all(nu == 0.0)


def test_nu_bs_differentiable_at_zero_e():
    # The double-where sqrt keeps d(nu_bs)/dE finite at E = 0 (sqrt'(0) = inf trap).
    def loss(E):
        return jnp.sum(bs._A_bs_h_cgrid(E, _mock_grid(_a([1e8])), _CFG.c_bs))
    g = jax.grad(loss)(_a([0.0]))
    # the double-where sqrt gives EXACTLY zero derivative at E=0 (not merely finite).
    assert jnp.all(jnp.isfinite(g)) and float(g[0]) == 0.0
