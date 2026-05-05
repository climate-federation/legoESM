"""End-to-end probe: KPP integration B_f sign for freshening vs brine rejection.

After the integration.py:144 fix:
- Pure freshening (Q_T=0, fw>0)            → B_f < 0 (stabilizing)
- Pure brine rejection (Q_T=0, fw<0)        → B_f > 0 (destabilizing)
- Pure cooling (q_net<0, fw=0)              → B_f > 0 (destabilizing)
- Pure warming (q_net>0, fw=0)              → B_f < 0 (stabilizing)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    rho_0 as _RHO_0,
    c_sw as _C_SW,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)


def compute_B_f(q_net, fw, T_sfc, S_sfc):
    """Replicate the production B_f computation in integration.py."""
    p_sfc = jnp.zeros_like(T_sfc)

    Q_sfc_T = q_net / (_RHO_0 * _C_SW)
    alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
    B_heat = -constants.g * alpha * Q_sfc_T

    Q_sfc_S = -S_sfc * fw / _RHO_0
    beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
    # FIXED: was -g*beta*Q_sfc_S, should be +g*beta*Q_sfc_S
    B_salt = constants.g * beta * Q_sfc_S
    return B_heat + B_salt, B_heat, B_salt


def main() -> None:
    T_sfc = jnp.asarray(20.0)   # degC
    S_sfc = jnp.asarray(35.0)   # PSU
    fail = 0

    cases = [
        # (q_net, fw, expected B_f sign, label)
        (    0.0,  1e-5, "negative", "Pure freshening (fw>0, q_net=0)"),
        (    0.0, -1e-5, "positive", "Pure brine/evap (fw<0, q_net=0)"),
        ( -100.0,   0.0, "positive", "Pure cooling (q_net<0, fw=0)"),
        ( +100.0,   0.0, "negative", "Pure warming (q_net>0, fw=0)"),
    ]
    for q_net, fw, expect, label in cases:
        B_f, B_h, B_s = compute_B_f(
            jnp.asarray(q_net), jnp.asarray(fw), T_sfc, S_sfc,
        )
        sign = "negative" if float(B_f) < 0 else ("positive" if float(B_f) > 0 else "zero")
        ok = (sign == expect)
        marker = "[PASS]" if ok else "[FAIL]"
        if not ok:
            fail += 1
        print(f"{marker} {label}")
        print(f"       B_heat = {float(B_h):+.3e}, B_salt = {float(B_s):+.3e}, B_f = {float(B_f):+.3e}")
        print(f"       expected sign = {expect}, got {sign}")
        print()
    print(f"TOTAL FAIL: {fail}")


if __name__ == "__main__":
    main()
