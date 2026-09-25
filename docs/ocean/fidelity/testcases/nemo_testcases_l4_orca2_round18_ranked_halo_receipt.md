# NEMO testcase Lane 4 — ORCA2 card round 18 ranked-halo receipt

Date: 2026-09-25

Parent: `75d1ac4ad64b98b9d7d13ac35839135492af85b6`

Measurement tip: `b8110bb4d9c453251f652c820c8c6c2e876a938f`

Status: **STOPPED_FOR_RECORD.** Given NEMO's entry, the admitted ranked halo
record moves the first non-bit statement upstream from substep-2
`continuity_du` to its `ua_e` operand. The difference is born between
substep-1 midpoint and substep-2 midpoint, but the existing record does not
carry the ranked velocity histories and update operands needed to walk that
interval. No production statement lands.

No file under `packages/` changed. The six sea-ice selectors and the card's
`unmeasured_features` tuple remain unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every solver number below is labelled **given NEMO's entry**. The whole-card
ladder is separately labelled **independent with Decision-52 SSH**.

## 1. Record reconciliation and admission

The existing run is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round17/acquisition/orca1ice_bt_halo_ranked_np2`.
Both intended streams exist: rank 0 is 1,143,092 bytes with SHA-256
`c63057b2242fcfe865f3b0e111af1486040a084a48c7d5e66b0805aaabd5b029`;
rank 1 has the same byte count and SHA-256
`9e1a0321b3572edb4963735ca4e2e44333a92b8637b4d3a6e8b51507cc1421e4`.
Their headers name ranks 0 and 1 and each contains substeps 1 and 2.

The writer opens and identifies one rank-tagged file per MPI rank at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:445-455`,
writes all five direct operand arrays at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:571-574`,
and closes and reports the file at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:845-848`.
The compiled output manager declares rank-0-only `lwp` at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/in_out_manager.f90:180-180`.
Consequently rank 0's marker is in `ocean.output`, while rank 1's unguarded
marker is correctly captured in `run.user.stdout.log`. The earlier admission
refusal was an instrument expectation defect, not a missing-record defect.

The repaired admission passes against the existing run without rebuild or
rerun. Its swapped-rank plant fires.

## 2. First non-bit operand

The inherited boundary reproduces exactly: substep 1 is bit-exact, and
substep-2 `continuity_du` remains first non-bit at 64 / 8,794 wet T cells,
maximum `7.705384632572532e-07`.

NEMO constructs face depth and transport in the order recorded at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:539-544`,
then subtracts adjacent U transports and updates sea surface height at
`ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:562-569`.
Direct recorded right-minus-left `zhU` replays NEMO's `continuity_du`
bit-exactly on all 8,794 cells, and candidate operands replay the candidate
subtraction bit-exactly. The rank-0 direct right-face mapping is also exact on
all 8,794 cells.

The 64 differing T cells touch 128 U faces. On those faces, substep 1 is exact
for `e2u`, `ssumask`, `ua_e`, `zhup2_e`, and `zhU`. At substep 2, `e2u`,
`ssumask`, and `zhup2_e` remain exact, while `ua_e` differs on 64 / 128 faces
with maximum `1.2223159767330016e-15` and `zhU` differs on 64 / 128 with
maximum `7.705384632572532e-07`.

Both candidate and recorded compiled products replay their own `zhU`
bit-exactly. Replacing only candidate `ua_e` with recorded `ua_e` makes both
the product and all 64 visible continuity cells bit-exact; replacing only
`zhup2_e` leaves all 64. The first recorded non-bit operand is therefore
`ua_e`, before multiplication by exact face depth and width.

The candidate's substep-1 U exit differs from recorded substep-2 `ua_e` on the
same 64 / 128 faces with the same maximum. This localizes birth to the
source-ordered velocity update and exchange between those boundaries; it does
not yet attribute a production expression.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R18-P1 | **CONFIRMED** | the existing run admits with both distinct expected-size ranked streams, run-completion stamps, correct headers/substeps, and split rank markers. |
| R18-P2 | **CONFIRMED** | the inherited first boundary, count, maximum, and substep-1 identity reproduce exactly. |
| R18-P3 | **CONFIRMED** | direct recorded `zhU` replays all 8,794 recorded continuity cells bit-exactly. |
| R18-P4 | **CONFIRMED** | `e2u` and `ssumask` are bit-exact on all 128 mismatch-adjacent faces. |
| R18-P5 | **CONFIRMED** | `ua_e` is first non-bit; its isolated substitution removes the full support, while face depth does not. |
| R18-P6 | **CONFIRMED** | both the swapped-rank admission plant and one-ULP payload plant fire. |
| R18-P7 | **CONFIRMED** | no `packages/` diff; the independent ladder remains measured with the same kt=10 entry-T boundary. |

No prediction was rewritten after measurement.

## 4. Acquisition needed

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round18_u_history_acquisition/run.sh`.
It creates isolated target `ORCA2_ORCA1ICE_OMIP_L4_R18UHIST` and new run
`round18/acquisition/orca1ice_u_history_ranked_np2`. For both ranks and the
first two substeps its WRITE-only patch records `jn`, `za1`, `za2`, `za3`,
entry `un_e`, `ub_e`, `ubb_e`, midpoint `ua_e`, then `rDt_e`, `zu_spg`,
`zu_trd`, `zu_frc`, `ssumask`, and exit `ua_e` before history rotation. It
does not assign any model value. Preflight passes pinned source, CPP, deck,
input, binary, patch, dry-apply, and tool checks. Admission refuses wrong
rank sizes, failed-run stamps, missing rank markers, or absent STOP 0.

## 5. Ladder, review, and tests

The final-tip independent ORCA2 kt=1..10 ladder exits 0 with
`LADDER_MEASURED`. Its trajectory remains
`MEASURED_INDEPENDENT_WITH_DECISION52_SSH`; kt=10 entry T remains maximum
`3.9430791763114783` on 430,552 cells. No GYRE rerun is required because
`git diff 75d1ac4ad..HEAD -- packages/` is empty.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed acquisition tip. It failed before reading the diff because
the app-server client could not initialize on a read-only filesystem.
Verdict: **independent review unavailable in-sandbox**.

The clean operand gate exits 0 with `WALKED_TO_FIRST_NON_BIT_HALO_OPERAND`;
both plants fire and the plant run exits 1. The receipt citation gate passes
all six rendered compiled citations; a rigid two-line shift of every citation
fails. The focused round-17/18 and citation battery passes **20 / 20**.

The required `tests/ocean/fidelity -n 12` battery was launched exactly once:
it collected 1,675 tests, recorded 1,658 passes and seven skips, and reached
99% before its final worker again stopped producing output. After the same
bounded tail used by round 17, it was interrupted with exit 130 and is not
represented as green. Its three failures are the same pre-existing reds named
in the round-17 receipt: the GYRE round-129 certified-record stamp says the
stepping gate moved; the SI3 scalar-math gate says `A MY_SRC is not verbatim`;
and the GYRE round-51 assertion expects the stale final six trace fields. No
file in those implementation/test paths, and no file under `packages/`,
changed in round 18.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round18/codex`.

## Choices

ASKED: admit the existing ranked halo record and walk its compiled operands to
the first non-bit statement.

UNASKED: none. Decisions 54, 57, and 58 remain pending and untouched. No
configuration choice, carried-state change, stabilizer, sea-ice change, or
production statement was made.

## OPEN

1. Run and admit the reported ranked U-history acquisition. Walk the direct
   `un_e`, `ub_e`, `ubb_e`, predictor coefficients, and source-ordered vector
   update before attributing or changing production code.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part `dyn_ldf` scoping), 57 (`r1_rho0` spelling), and
   58 (second continuity solve) remain pending and untouched.
4. The independent ORCA2 year remains dependent on the scheduled independent
   initial-state completion; this solver result is only given NEMO's entry.
5. The wide ocean-fidelity battery's hung final tail remains an operator
   action; this interrupted run is not represented as green.
