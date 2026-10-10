# Preregistration — ORCA2 round 216 vector boundary association

Date: 2026-10-09. Frozen base: `568e4bde5`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round216/`.
Every OMT-1 number remains separately labelled **independent OMT-1** or
**given NEMO's entry OMT-1**. Rung-0 numbers are **independent** and rung-10
numbers are **given NEMO's entry**. This round changes no configuration,
carried state, stabiliser, sea-ice selector, or `unmeasured_features` tuple.

## Frozen source boundary

Round 215 proved the slow-V/raw-`ssvmask` pair bit-exact at NEMO's pre-LBC
`va_e` boundary but held it because independent rung 0's kt=10 stage-3 SSH
maximum worsened by `2.401964303011539e-7 m`. The compiled vector update is
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`.
The next executed statement is the one seven-array `lbc_lnk` association at
`dynspg_ts.f90:738-756`; the executable then rotates `va_e` into `vn_e` and
accumulates weighted velocity/SSH at `dynspg_ts.f90:807-839`. The active
T-pivot V sign/permutation is compiled at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718`.

The repository already has the complete seven-array private association arm
and its rank-complete round-96 record gate. Search-before-build result: reuse
`nemo_testcase_l4_orca2_round146_boundary_association_gate.py` and the
`barotropic_external_mode_association` hook; add no second association.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R216-P1 | The round-215 pair plus NEMO's complete seven-array association closes the next downstream V boundary. | Candidate post-association V is bit-exact against the admitted round-96 target; all seven fields are present and the replay reproduces the private-arm trajectory. | Any missing field, non-passive replay, or post-V difference: **HELD_INSTRUMENT**; read no trajectory verdict. |
| R216-P2 | The missing association is the compensating partner behind round 215's final-SSH veto. | Against the frozen base, independent rung 0 has a majority of RMS-moved rows toward NEMO, first-over-bar toward/unchanged, no exact-row loss, and kt=10 stage-3 SSH maximum no worse than `0.42832517646246693 m`. | Any predicate fails: **HELD**; restore production and name the first downstream statement still non-bit. |
| R216-P3 | OMT-1 keeps round 215's qualifying direction with the complete unit. | Both labelled kt=1..8 ladders have majority toward, no exact loss, first-over-bar toward/unchanged, and final SSH maximum no worse. | Failure: **HELD**; do not start OMT-2. |
| R216-P4 | The complete unit is inert or admissible on other certified cards. | Rung 10, GYRE year, DINO month, DINO/lock-exchange/overflow gates meet their standing predicates; every moved row is registered. | Any non-Decision-96 red: restore production and **HELD** at the exact row. |
| R216-P5 | Every new classifier is non-vacuous. | Boundary-V, seven-field registry, final-SSH veto, exact-loss, and false-majority plants each refuse. | Any plant stays green: no round-216 claim or landing. |

## Landing predicate

The round-215 slow-V/raw-mask pair and NEMO's immediately following complete
seven-array association are one candidate unit. No pair half and no one-field
association may land. The unit lands only if R216-P1 through R216-P5 pass the
full standing gates. Otherwise production is restored and the round is held.
OMT-2 waits for this disposition.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
