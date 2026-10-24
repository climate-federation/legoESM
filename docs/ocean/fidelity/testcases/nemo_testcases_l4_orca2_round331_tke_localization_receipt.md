# ORCA2 round 331 — remaining TKE residual requires a source-order record

Date: 2026-10-24

Base: `484b60f5b7da`; preregistration: `a69fa0005d23`; acquisition tool:
`ba88fcbdb48e`.

Disposition: **STOP_PROPOSAL; ACQUISITION_NEEDED; no model or configuration
change**.

## Answer

**CONFIRMED.**  The landed round-330 card reproduces the admitted residual on
the CPU fp64 production-JIT path: returned TKE energy has 901 unequal wet
interfaces, post-TKE viscosity has 931, and the viscosity consumed after DDM
and IWM has 925.  The repaired deepest level remains exact in every score.
The remaining returned-energy family is 279 surface plus 622 interior plus
zero deepest interfaces.

**CONFIRMED.**  No literal surface-boundary transcription difference is
visible.  The record build sets the held surface value and its identity row at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:273-276`; the card
selects `nemo_z0` plus its literal matrix and solver
(`nemo_testcase_recipe.py:416,421-422`) and constructs the held virtual row in
`tke.py:1845-1857`.  That source read does not exclude a differing input or a
later source-order statement from the surface family.

**CONFIRMED.**  The old records cannot distinguish the live shear output from
the boundary, RHS/matrix, solve, or post-solve penetration statements.  No
single cited statement owns all or part of the residual, so Decision 122 is
not opened and no physics change is eligible this round.

## Production-JIT census

The committed round-328 replay was run unchanged against the landed round-330
card on CPU with fp64 and production JIT.  Its exact per-level census is:

| zero-based `k` | returned `en` | post-TKE `avm` | consumed `avm` |
|---:|---:|---:|---:|
| 0 | 279 | 284 | 282 |
| 1 | 111 | 116 | 116 |
| 2 | 67 | 67 | 67 |
| 3 | 55 | 58 | 58 |
| 4 | 53 | 56 | 56 |
| 5 | 43 | 46 | 46 |
| 6 | 23 | 28 | 28 |
| 7 | 15 | 18 | 18 |
| 8 | 15 | 18 | 18 |
| 9 | 11 | 13 | 13 |
| 10 | 13 | 12 | 10 |
| 11 | 13 | 13 | 13 |
| 12 | 17 | 17 | 16 |
| 13 | 12 | 12 | 12 |
| 14 | 17 | 16 | 16 |
| 15 | 15 | 15 | 15 |
| 16 | 15 | 15 | 15 |
| 17 | 3 | 3 | 3 |
| 18 | 5 | 5 | 5 |
| 19 | 48 | 48 | 48 |
| 20 | 11 | 11 | 11 |
| 21 | 5 | 5 | 5 |
| 22 | 26 | 26 | 25 |
| 23 | 4 | 4 | 4 |
| 24 | 4 | 4 | 4 |
| 25 | 6 | 6 | 6 |
| 26 | 9 | 9 | 9 |
| 27 | 6 | 6 | 6 |
| 28 | 0 | 0 | 0 |
| **total** | **901** | **931** | **925** |

The family totals are respectively surface/interior/deepest = `279/622/0`,
`284/647/0`, and `282/643/0`.  Returned `en` first differs at global
`[j,i,k]=[1,119,19]`; its maximum absolute difference is
`5.153991021727283e-9` at `[117,55,3]`.  The entry `rn2` operand remains exact
at all 414,119 wet interfaces.  This confirms R331-P1 and exactly reconciles
the round-330 totals; nothing is retracted.

Evidence:
`orca2_rounds/round331/replay/production_jit.json`, SHA-256
`fddba8b2921325e2f2237dac83b4f2896841173dfc2b4448461e820147c0333c`.

## Source-order discriminator

**CONFIRMED.**  `zdf_phy` computes shear, calls bottom then top drag, computes
mixed-layer outputs, and only then calls TKE
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:316-336`).

- `zdf_sh2` is live.  The executing no-Stokes arm forms U and V shear and
  horizontally averages them into `p_sh2`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90:98-112`).  TKE
  consumes that value in both the Richardson-number calculation and the RHS;
  the RHS/matrix record boundary is therefore needed
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428`).
- The deck uses linear bottom drag.  The bottom call therefore leaves the
  initialized coefficient unchanged, while the top call writes implicit ice
  drag because there is no cavity
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfdrg.f90:165-187`).  The
  coefficients are still recorded to verify their handoff and to localize any
  shallow bottom-boundary cells; source reading alone is not promoted to a
  numerical exclusion.
- `zdf_mxl` does compute `nmln` and `hmlp`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:109-123`), but
  the resolved `nn_etau=1` arm applies penetration over every interior level.
  The only adjacent arm that reads `nmln` is inactive `nn_etau=2`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:500-507`).  Thus
  the mixed-layer outputs are **CONFIRMED inactive** for this TKE update, while
  their record remains a control on the source-order acquisition.
- The compiled matrix/RHS and Thomas solve remain the candidate boundaries
  after the input handoff
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428` and
  `:459-477`).  Existing post-TKE records skip all of those boundaries.

R331-P2 and P3 are confirmed.  R331-P4 is also confirmed: the old records do
not select the first unequal source-order boundary.

## Operator acquisition

The round-328 acquisition harness now has a `tke331` mode selected by the thin
launcher at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition/run.sh`.
It imports the reviewed round-328 helpers and preserves the record build's
existing round-326 `zdftke` observer.  Additions-only observers capture both
ranks at kt 1:

1. `zdf_sh2`, bottom/top `zdf_drg`, and `zdf_mxl` outputs, followed by a
   second copy at TKE entry to prove exact handoff, plus entry `avm_k` and
   `dissl` operands;
2. `en` after the ordinary top/bottom boundary conditions;
3. RHS `en` and all three tridiagonal coefficients after matrix construction
   and any wave-boundary override; the active Langmuir term lies between the
   boundary and matrix records and therefore falls inside this after-matrix
   segment;
4. `en` immediately after the Thomas solve and floor; and
5. returned `en`, `avm_k`, `avt_k`, and `wmask` for exact reconciliation with
   the admitted round-328 post-TKE record.

Admission requires the inherited `dyn_zdf` records to remain bit-identical,
the new drag fields to equal those admitted inputs, exact output-to-input
handoffs, exact surface continuity across each internal boundary, and exact
wet-cell returned fields against the admitted post-TKE record.  Eleven planted
violations cover header, order, truncation, stamp, shear, drag, mixing length,
boundary, matrix, solve, and final output; the semantic plants each move one
binary64 word by one ULP.

The clean read-only preflight passed at `ba88fcbdb48e`: 45 tests passed, all 11
preflight plants fired, all 140 inherited deterministic files matched, all six
writer/patched Fortran units passed syntax, and the same-program census found
410 identical preprocessed units and 422 identical normalized objects.  Its
non-observer source plant refused.  The first preflight attempt correctly
refused because it would have replaced the record build's already instrumented
`MY_SRC/zdftke.F90` with pristine source; the tool now extends that exact
record unit and the same-program proof passes.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r331-322489136/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round331/acquisition/orca1ice_tke331_kt1_np2`.
No NEMO executable was run by Codex.

## Gates and disposition

- Preregistered predictions R331-P1 through P4: all **CONFIRMED**.
- Focused compiled-layout, admission, control, and same-program battery:
  20 passed; the imported clean preflight battery is 45 passed.
- Per-receipt citation gate: PASS, 10 citations, 0 unmapped, 0 failures, and
  0 map-audit failures; the required citation-gate pytest is 17 passed.
- Default ORCA2 landing battery: 148 passed in 943.01 s, including the
  six-test private-import ratchet.
- No configuration value changed: **ASKED**.
- No model change, seven-pair Decision-122 score, cross-card measurement, or
  gradient claim is made because ownership is still unmeasured: **ASKED**.
- Stop for the source-order record: **ASKED conditional** and now satisfied.

independent review: operator (Claude)

The next admissible action is the operator-only `--run`.  Its first unequal
boundary will either identify the live input family (`sh2`/drag), the surface
and bottom BC boundary, the matrix/RHS, the solve, or the post-solve
penetration.  Until that record exists, every such owner is **PLAUSIBLE**, not
confirmed.
