"""Probe: B_salt sign in KPP integration.

Hypothesis: ``integration.py:144`` has the wrong sign on ``B_salt``.

Convention (KPP / MOM6, B_f > 0 = unstable):
    B_f = -g * (alpha * Q_T - beta * Q_S)
        = -g*alpha*Q_T + g*beta*Q_S

where:
    Q_T = surface kinematic heat flux INTO ocean [K m/s]
    Q_S = surface kinematic salt flux INTO ocean [PSU m/s]

For freshening (P > E, freshwater INTO ocean):
    Q_S = -S * F_fw / rho_0 < 0   (kinematic salt flux INTO ocean is NEGATIVE,
                                    because salt is being diluted out)
    correct B_salt = +g*beta*Q_S < 0   (stabilizing — lighter water on top)

Current code: ``B_salt = -g*beta*Q_S``.
With Q_S < 0 → B_salt > 0 → KPP labels the column UNSTABLE.

This probe drives an idealised ocean column with:
- no heat flux (Q_T = 0)
- pure freshening (P > E ⇒ fw > 0)
- expected B_f < 0 (stabilizing — surface lightening)
- observed B_f > 0 if the bug is present.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    rho_0 as _RHO_0,
    haline_contraction_coeff,
)


def main() -> None:
    # Single test column at warm/saline tropical conditions.
    T_sfc = jnp.asarray(20.0)   # degC
    S_sfc = jnp.asarray(35.0)   # PSU
    p_sfc = jnp.asarray(0.0)    # Pa, surface
    fw = jnp.asarray(1e-5)      # kg/m^2/s freshwater INTO ocean (P > E)

    beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
    Q_S = -S_sfc * fw / _RHO_0

    # Current (buggy) formula:
    B_salt_buggy = -constants.g * beta * Q_S

    # Correct formula:
    B_salt_correct = +constants.g * beta * Q_S

    print(f"beta             = {float(beta):.6e}  [1/PSU]")
    print(f"Q_S              = {float(Q_S):.6e}  [PSU m/s]   (should be < 0 for freshening)")
    print(f"B_salt (current) = {float(B_salt_buggy):.6e}  [m^2/s^3]   (should be < 0 = stabilizing)")
    print(f"B_salt (correct) = {float(B_salt_correct):.6e}  [m^2/s^3]")
    print()

    if float(B_salt_buggy) > 0.0:
        print("BUG CONFIRMED: current code produces POSITIVE B_salt for freshening,")
        print("              which KPP interprets as DESTABILIZING.")
        print("              Correct sign should be NEGATIVE (stabilizing).")

    # Also probe the reverse case: brine rejection (fw < 0 = salt INTO ocean):
    fw_brine = jnp.asarray(-1e-5)  # net evaporation OR brine rejection
    Q_S_brine = -S_sfc * fw_brine / _RHO_0   # > 0 (salt INTO ocean)
    B_salt_brine_buggy = -constants.g * beta * Q_S_brine
    B_salt_brine_correct = +constants.g * beta * Q_S_brine
    print(f"\nBrine-rejection case (fw < 0):")
    print(f"Q_S              = {float(Q_S_brine):.6e}   (> 0, salt into ocean)")
    print(f"B_salt (current) = {float(B_salt_brine_buggy):.6e}   (should be > 0 = unstable)")
    print(f"B_salt (correct) = {float(B_salt_brine_correct):.6e}")


if __name__ == "__main__":
    main()
