# ORCA2 round 235 follow-up preregistration — back-interpolation alpha replay

Date: 2026-10-10. Frozen after the two-label source table and before this
replay. The original round-235 predictions remain frozen in
`PREREG_nemo_testcases_l4_orca2_round235.md`; R235-P2's expected forcing owner
is already refuted because both labels first leave the interior floor at
substep 3 `eta_pgf`, not at `slow_u/slow_v`.

## Compiled statement and one-variable hypothesis

The executing OMT-4 deck states `nn_bt_flt=3` and `rn_bt_alpha=0.09` in its
resolved `namelist_cfg`. NEMO computes substep-3-and-later coefficients from
that value at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:1522-1557` and applies
them to `ssha_e`, `sshn_e`, `sshb_e`, and `sshbb_e` at `:604-612`. legoESM's
shared `nemo_ab3am4_coeff_arrays` currently defaults to the GYRE value `0.07`
and has no configuration operand for the deck's `0.09`.

The replay uses only NEMO's admitted substep states: at substep 3,
`ssha_e=eta_exit[2]`, `sshn_e=eta_entry[2]`, `sshb_e=eta_entry[1]`, and
`sshbb_e=eta_entry[0]`. It calls the existing coefficient helper with exactly
one changed operand, alpha `0.07 -> 0.09`, and evaluates the recorded source
association. No JAX trajectory and no package change are measured.

## Frozen predictions

- **R235-A1:** the resolved namelist parser returns exactly filter `3` and
  alpha `0.09`; changing either in a plant refuses.
- **R235-A2:** the `alpha=0.09` replay matches NEMO's recorded substep-3
  `eta_pgf` bit-for-bit on all 13,320 rank-0 cells. Any unequal cell refutes
  source-exact ownership.
- **R235-A3:** the otherwise identical `alpha=0.07` replay is over the
  `2e-10` floor and reproduces the table's `0.0031271104752883805 m` maximum.
  A value at the floor, or a different maximum by more than one binary64 ULP,
  refutes the one-variable attribution.
- **R235-A4:** if A1-A3 hold, the statement is named but not landed: adding a
  configurable NEMO alpha is a configuration/API choice. Ask whether to add
  an explicit field with ORCA2 `0.09` and preserve GYRE `0.07` before changing
  packages. If any replay input is absent, report the exact record gap.

Status remains `HELD`; Decision 114 remains pending and untouched.
