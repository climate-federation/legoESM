# NEMO testcase L2 GYRE high resolution — round 13 admission amendment

Date: 2026-10-18

Incoming lane tip: `e4e0f9f694085c1d3e73762763356bccc60cae77`.

This amendment is written after the operator acquired the requested 180-day
full-state legoESM archive, but before computing or ranking any offline
momentum terms.  Source inspection found that round 12 registered an invalid
implication: complete daily state means are necessary inputs, but they are not
an online production-tendency budget.  This amendment retracts that admission
rule instead of silently changing the meaning of the registered measurement.

## Registered question and unchanged region

The scientific question remains the days 30–180 native-surface-V budget in
the western-boundary wet-span bin.  The requested rows remain advection,
pressure/kinetic-energy gradient, total-vorticity/Coriolis, Smagorinsky
lateral diffusion, vertical mixing, surface stress, external-mode correction,
and an independently recorded total increment.

The archive audit and the budget-observability audit are now separate:

- `STATE_ARCHIVE_PASS` requires the round-12 3-D daily state fields and all
  existing NEMO tile/cadence/placement checks.
- `BUDGET_OBSERVABILITY_PASS` additionally requires, for each model, an
  online-production record over days 30–180, the split external-mode
  correction, separation of implicit vertical mixing from surface stress, an
  independently recorded total, a passive-observer trajectory-identity
  receipt, artifact hashes, and a numeric closure bar fixed before viewing
  the term comparison.  The gate reads the term artifact itself: the day axis
  must be exactly 30–180, every registered surface-V array must share the
  `(151, j, i)` shape and be finite, the wet mask must be nonempty, and the
  component sum must close the independent total at the registered bar.

The attribution is `UNMEASURED` unless both statuses pass.  In particular, no
term may be reconstructed as `F(mean(state))` and labelled
`mean(F(state))`: nonlinear advection and state-dependent Smagorinsky
viscosity do not commute with daily averaging.  Nor may the residual be named
vertical mixing, surface stress, or external-mode pressure without observing
those production boundaries separately.

## Confirm / refute and controls

`CONFIRM_IDENTIFIABLE` requires both signed model term fields and independent
totals to close at the preregistered numeric bar.  Only then may the largest
finite `abs(mean legoESM-NEMO) / RMS(NEMO)` row be named the measured carrier.
`REFUTE_IDENTIFIABLE` is any missing term, a combined implicit term, missing
external correction, failed passive identity, missing total, or absent
numeric bar.  That verdict is `ACQUISITION_NEEDED`, not a zero-filled budget.

The non-vacuity control uses two subdaily pairs with identical provider means
but mean products +1 and −1.  The 2.0 ambiguity must fire.  A synthetic online
contract must pass, while plants in every semantic contract field and all three
artifact digests must refuse.  A correctly hashed active-term closure
violation must also refuse.  The existing missing-field and non-finite
state-provider plants remain active.

No physics, deck, fidelity bar, or shared trajectory changes in this round.
