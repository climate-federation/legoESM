"""Carbonate chemistry vs the ORIGINAL reference code, mocsy v2.4.1 (Orr & Epitalon 2015).

The fixture ``oracle_fixtures/mocsy_v2_4_1_surface.npz`` holds mocsy's own ``constants()`` and ``vars()``
output on a 189-point surface grid (T 2-32 degC, S 31-37, DIC 1900-2100 umol/kg, ALK = DIC + 200..350),
built by a Fortran driver with optB='u74', optK1K2='l', optKf='dg', optGAS='Pzero', depth 0, Patm 1,
silicate = phosphate = 0 (provenance string stored in the fixture).

- K0, K1, K2, K_B and B_T must match mocsy to rel 1e-12.
- The Follows (2006) solver omits the water, phosphate and silicate alkalinity terms and the
  fugacity correction (module docstring).  Its pCO2 departure from mocsy's pCO2 and fCO2 is a
  KNOWN DIFFERENCE pinned per case (rel 1e-6, abs floor 1e-12): the test fails if the port drifts.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.biogeochemistry.carbonate import (
    borate_equilibrium,
    carbonate_equilibria,
    co2_solubility,
    solve_carbonate_system,
    total_borate,
)

jax.config.update("jax_enable_x64", True)

_FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "mocsy_v2_4_1_surface.npz")
_T, _S, _DIC, _ALK = (jnp.asarray(_FIX[k]) for k in ("T", "S", "DIC", "ALK"))


def _ours(name):
    K1, K2 = carbonate_equilibria(_T, _S)
    return {"K0": co2_solubility(_T, _S), "K1": K1, "K2": K2,
            "Kb": borate_equilibrium(_T, _S), "Bt": total_borate(_S)}[name]


@pytest.mark.parametrize("name", ["K0", "K1", "K2", "Kb", "Bt"])
def test_constant_matches_mocsy(name):
    np.testing.assert_allclose(np.asarray(_ours(name)), _FIX[name], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("ref,pin", [("pco2", "pin_pco2_rel_vs_mocsy_pco2"),
                                     ("fco2", "pin_pco2_rel_vs_mocsy_fco2")])
def test_solver_departure_from_mocsy_is_pinned(ref, pin):
    # Inputs are mol/kg, so rho_sw=1 makes the port's mol/m3 -> mol/kg conversion the identity.
    pco2, _ = solve_carbonate_system(_DIC, _ALK, _T, _S, rho_sw=1.0)
    gap = np.asarray(pco2) / _FIX[ref] - 1.0
    np.testing.assert_allclose(gap, _FIX[pin], rtol=1e-6, atol=1e-12)
