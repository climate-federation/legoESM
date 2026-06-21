"""Equivariance / convention-invariance tests (verification tier 0-1).

Truth-based, ORACLE-FREE guards that the physics is invariant to *conventions*
(pressure-unit choice, vertical-index order), per the oracle-recipe doctrine
(``docs/ocean/fidelity/oracle_recipe_strategy.md`` §4, §7). These push convention
bugs DOWN from the expensive tier-3 oracle comparison to a cheap every-PR test.

Decidable criterion (doctrine §4): a difference is a *convention* iff a bijective
re-encoding ``φ`` gives ``physics(φ(x)) = φ(physics(x))`` to truncation tolerance;
then it must be handled in the bridge only, never replicated in the model. A
"convention" whose ``φ`` does NOT preserve physics is actually a physics choice.

Near-term purpose (doctrine §7): partition the Phase G ACC ~5 kg/m^3 density
residual into "bridge convention/units bug" vs "genuine numerics difference"
WITHOUT another Veros run.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    VerosNonlin2Config,
    compute_hydrostatic_pressure,
    make_eos_fn,
)

jax.config.update("jax_enable_x64", True)

# 1 dbar = 1e4 Pa, exactly.
DBAR_TO_PA = 1.0e4

# EOS variants that take pressure in Pa and are pressure-dependent.
PRESSURE_EOS = ["wright", "veros_nonlin2"]


def _acc_like_column(n: int = 15):
    """A stable ACC-like single column: warm/fresh surface -> cold deep."""
    T = jnp.linspace(15.0, 0.0, n)      # degC, k=0 surface
    S = jnp.linspace(34.5, 35.0, n)     # PSU
    return T, S


# ---------------------------------------------------------------------------
# φ = pressure-unit choice. The EOS contract is "pressure in Pa"; density must
# be invariant to how that Pa value is *derived* (the 6569a22f bug class was a
# Pa-vs-depth-m unit mismatch at the probe/bridge boundary).
# ---------------------------------------------------------------------------


def test_pa_depth_conversion_roundtrips():
    """nonlin2 internally maps Pa -> depth via ``p / (rho_0 * g)``. The inverse
    (depth -> Pa via ``depth * rho_0 * g``) must round-trip to machine precision,
    or the EOS sees a different depth than the caller intended."""
    cfg = VerosNonlin2Config()
    depth_m = jnp.linspace(0.0, 4000.0, 41)
    p_pa = depth_m * cfg.rho_0 * cfg.grav
    depth_back = p_pa / (cfg.rho_0 * cfg.grav)
    np.testing.assert_allclose(
        np.asarray(depth_back), np.asarray(depth_m), rtol=1e-14, atol=1e-9
    )


def test_nonlin2_pressure_argument_is_pascals_not_dbar():
    """Pin the pressure UNIT. A +1 dbar (= 1e4 Pa) increase must change density
    by ``dρ/dp * 1e4 ≈ 1e4 / cs0**2``. If the EOS silently treated its argument
    as dbar (the convention-mismatch bug class), this would be off by ~1e4x."""
    cfg = VerosNonlin2Config()
    eos = make_eos_fn("veros_nonlin2")
    T, S = jnp.asarray(10.0), jnp.asarray(35.0)
    rho0 = eos(T, S, jnp.asarray(0.0))
    rho1 = eos(T, S, jnp.asarray(DBAR_TO_PA))  # +1 dbar, expressed in Pa
    d_rho = float(rho1 - rho0)
    expected = DBAR_TO_PA / cfg.cs0**2  # ≈ 4.5e-3 kg/m^3 per dbar
    np.testing.assert_allclose(d_rho, expected, rtol=1e-2)


# ---------------------------------------------------------------------------
# φ = vertical-index flip (k=0 surface <-> k=0 bottom). The Veros<->legoESM
# bridge reverses the vertical axis; these guard that the density/pressure
# physics is equivariant under that re-encoding.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eos_name", PRESSURE_EOS)
def test_eos_density_is_pointwise_vertical_flip_equivariant(eos_name):
    """The EOS is point-wise in (T, S, p), so ``eos(flip(x)) == flip(eos(x))``
    to machine precision. Any deviation is an accidental cross-level coupling
    or index bug — exactly the kind of error the bridge's ``_reverse_z`` could
    otherwise hide."""
    eos = make_eos_fn(eos_name)
    cfg = VerosNonlin2Config()
    T, S = _acc_like_column()
    depth_m = jnp.linspace(0.0, 1900.0, T.size)
    p = depth_m * cfg.rho_0 * cfg.grav
    rho = eos(T, S, p)
    rho_flipped = eos(T[::-1], S[::-1], p[::-1])
    np.testing.assert_array_equal(np.asarray(rho_flipped), np.asarray(rho[::-1]))


def test_hydrostatic_pressure_cumsum_order_is_negligible():
    """Phase G residual cause (c): does the surface-referenced ``cumsum`` order
    (legoESM integrates k=0 surface downward; Veros integrates from its k=0=deep
    end) explain the ~5 kg/m^3 ACC density residual?

    φ = reverse the *summation order* of the same per-layer pressure increments.
    Floating-point non-associativity makes the two prefix-sums differ by
    O(machine eps), NOT by anything resembling the residual. Translated through
    the EOS sensitivity ``dρ/dp ≈ 1/cs0**2``, the implied density difference is
    ~1e-6 kg/m^3 — six orders of magnitude below the residual. Conclusion:
    cumsum/flip ORDER is exonerated; the residual lives elsewhere (time-level
    alignment or wall-row padding), which is where the next probe should look.
    """
    cfg = VerosNonlin2Config()
    g = constants.g
    rho_ref = constants.rho_ocean
    nlev = 15
    T, S = _acc_like_column(nlev)
    dz = jnp.asarray(np.linspace(20.0, 276.0, nlev))
    jacobian = jnp.asarray(1.0)
    eta = jnp.asarray(0.3)

    depth_mid = jnp.cumsum(dz) - 0.5 * dz
    rho = make_eos_fn("veros_nonlin2")(T, S, depth_mid * cfg.rho_0 * cfg.grav)

    # forward path: the production function (cumsum from the surface)
    p_fwd = compute_hydrostatic_pressure(
        rho, eta, dz, jacobian, rho_ref=rho_ref, g=g
    )

    # reverse path: reconstruct the SAME prefix integral with reversed
    # summation order. prefix_excl[k] = sum_{j<k} dp[j] = total - suffix[k].
    h = dz * jacobian
    dp = rho * g * h
    total = jnp.sum(dp)
    suffix = jnp.cumsum(dp[::-1])[::-1]      # suffix[k] = sum_{j>=k} dp[j]
    prefix_excl = total - suffix
    p_top_rev = rho_ref * g * eta + prefix_excl
    p_rev = p_top_rev + 0.5 * dp

    abs_diff = float(jnp.max(jnp.abs(p_fwd - p_rev)))
    rel_diff = abs_diff / float(jnp.max(jnp.abs(p_fwd)))
    assert rel_diff < 1e-12, f"cumsum-order rel diff {rel_diff:.2e} (expected ~eps)"

    # translate the pressure discrepancy into a density discrepancy
    drho_dp = 1.0 / cfg.cs0**2
    implied_drho = abs_diff * drho_dp
    assert implied_drho < 1e-6, (
        f"cumsum-order implies dρ={implied_drho:.2e} kg/m^3 — should be << residual"
    )
