# NEMO testcase L2 GYRE round 60 — TKE closure walk

Date: 2026-09-11. Governing preregistration: round 57 plus the committed
continuations in the round-60 pre-code review. Record producer:
`1a695951be1396abdd4d7a85f57b31de9c0917df`; target
`GYRE_OMIP_L2_P3_SM_R59TKE`; CPU/fp64. **HOLD: calibration and closure walk
PASS, but P1/P2 are refuted and the certified trajectory comparison FAILS.**

## Round 60 — compiled-source evidence

The record's own compiled program sets the surface boundary, applies the
Langmuir source, forms inverse Prandtl plus matrix/RHS, solves and floors `en`,
constructs the bounded mixing lengths and closure, then copies the result
before EVD at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:277-281`,
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:318-380`,
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433`,
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:466-483`,
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589-701`, and
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354`.

Re-executing those statements from the R59 record gives zero unequal for
`en` 0/17,400, `zmxlm` 0/18,000, `zmxld` 0/18,000, Prandtl 0/17,400, and
`avm`/`avt`/`dissl` each 0/18,000; the pre-EVD copy is exact. The consumed-RHS
plant exits 1.

## Substitution verdict and changes

The model-path cumulative `avt` walk is: carried production/buoyancy/
dissipation 744/17,400 unequal; matrix/RHS/sweep/`en` floor 0; surface input
0; raw mixing length/floor/bounds 0; Prandtl 0; `avt` derivation 0; zdfphy copy
0. Thus the first exact boundary is the recorded matrix/RHS/sweep result,
whose compiled statements are
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:416-433` and
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:466-483`.

The candidate corrects the shared `nn_mxl=3` raw expression and terminal
bottom-up update per
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:589-682`, and applies
NEMO's stored inverse-Prandtl factor directly per
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:394-412` and
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:691-701`.
The literal-Langmuir sole-owner prediction was refuted (744→645 unequal), and
the linked-sine continuation was refuted (→1,049); both were reverted.
The after-code audit also confined direct inverse-Prandtl multiplication to
step-entry NEMO cards, leaving historical callers bit-associated as before.

## Controlled trajectories

The certified decision-35 after arm is the sole before arm. All 390 moved
day-gap statistics are registered in `round60/after_day_gap.json`. Day-30 RMS
before→candidate is T 0.014241019261565466→0.014241019260383367; S
0.002234450969141919→0.002234450967451659; u
0.000596199573487925→0.0005961995734964586; v
0.0008332994329995481→0.0008332994333897992; ssh
0.00045333980700886135→0.0004533398073483128.

The kt=1..10 card keeps first-over-bar at kt=2 (`u`,`v`) and has no status
changes, but its exact 954-row comparison FAILS: 52 rows exceed the two-ULP
worsening limit (maximum 64,639 row-scale ULP). kt3 T/S are
3.722344242989895e-4 K and 3.6832457463108026e-5 g/kg, not preregistered
1.6275031290e-4 / 6.3278533133e-6: P2 is REFUTED. Consequently P1 (derived
floor as sole owner) is also REFUTED; the combined decision-35→round-56 floor
effect is measured here, while its one-ULP subcomponent remains below this
trajectory's resolution.

## Rule 12 and scope

| card | disposition |
|---|---|
| GYRE | HOLD: closure downstream of `en` exact; trajectory comparison FAIL. |
| LOCK_EXCHANGE | Unchanged: resolved `ln_zdfcst=.true.` constant mixing (`round33_lock_zdf_matrix/namelist_cfg:131`). |
| OVERFLOW | Unchanged: resolved `ln_zdfcst=.true.` constant mixing (`round33_overflow_zdf_matrix/namelist_cfg:128`). |
| DINO | HOLD/risk: separate branch shares `tke.py`; candidate statements require its own gates. |
| ORCA2 | UNMEASURED-with-spec: replay compiled TKE operands, then trajectory gate. |
| legacy `nemo_v1` | UNMEASURED-with-spec; its registered non-ULP floor move remains. |

ASKED: all work above, including the optional year control. UNASKED: none.
No NEMO source/run, year harness, reconciliation gate, freshwater pair, or
`#1484` guard was changed. Open: the 744-cell pre-`en` owner and the failed
trajectory comparison prevent landing the candidate physics.

Focused final tests pass 113/113; provenance/citation/stamp ratchets pass
39/39. The citation-shift and consumed-walk plants both exit nonzero.
The clean optional day-360 control (1,411.6 s CPU) scores RMS gaps against
`year_fromrest/nemo_seed0` of T 0.4071294150317232, S 0.06928546755392434,
u 0.004079444056363985, v 0.004060765080710654, and ssh
0.007611610571267302.
