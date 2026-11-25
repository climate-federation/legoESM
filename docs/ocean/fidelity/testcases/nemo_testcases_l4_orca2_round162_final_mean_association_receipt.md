# ORCA2 round 162 — final external-mode association scope

Date: 2026-10-06. Base `afa4d0bbd`; preregistration `4ec4b3cdc`;
measurement tip `459c7021f`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round162/`.
Verdict: **HELD**. The proposed final U/V association is dead code on ORCA2's
vector-invariant branch. The already-held per-substep halo pair still fails
the rung-0 salinity veto. No production selector or physics statement lands.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. Decision 52's
given-NEMO-entry bridge is not used. Sea ice, all six sea-ice selectors and
the shipped card's `unmeasured_features` tuple are unchanged.

## Loud correction to the preregistered source premise

The preregistration called the final association the next executed statement.
That is wrong. NEMO does associate the live external-mode fields inside every
substep at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`, and its
vector-invariant arm accumulates the already-associated U/V *velocities* at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:854-856`. But the
later conversion and U/V `lbc_lnk` are jointly guarded by
`NOT(ln_dynadv_vec OR lk_linssh)` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:924-937`.
The rung-0 card resolves `momentum_advection="vector_invariant"`, i.e.
`ln_dynadv_vec=.true.`. NEMO therefore never executes this final association
on ORCA2. The committed gate records that resolved selector and refuses with
`REFUSE_R162_DEAD_ARM`; its source-scope plant makes the branch live and
fires. This corrects the preregistration rather than silently editing it.

## Dead-arm discriminator

The false-default private hook is passive. Applied hypothetically after the
completed mean, it moves 90 `uu_b` and 180 `vv_b` cells. Against NEMO at kt=1,
the production unequal counts are 16,495/16,546 U/V and the final-only counts
are 16,443/16,438. This numerical movement is not a fidelity improvement
claim because the cited NEMO branch is dead.

The source-exact per-substep arm and the pair both have exactly 16,443/16,438
unequal U/V cells. Their complete rung-0 ladder rows are array-equal; only the
artifact's private-arm label differs. Thus the final association is also
idempotent after the live per-substep association. R162-P2 is **REFUTED**.

The pair moves 195/200 ladder rows, loses zero bit-exact rows and leaves the
first debt at kt=1 stage-1 T. By RMS, 10 rows move toward NEMO and 185 away;
by maximum, 65 move toward, 75 away and 55 are equal. The kt=10 stage-3 S
maximum still increases from `0.4156673855238111` to
`0.41567240155913865`, by `5.0160353275430225e-6`. The gate therefore also
retains the prior salinity refusal. The independent month candidate was not
run, exactly as preregistered after a ten-step refusal.

## GYRE shared-path control

The default hook is false. The 70-row base/tip comparison passes with zero
worsening ULPs and the first-over-bar boundary unchanged at kt=3. The base and
tip residual archives have identical SHA-256
`c7879296c7b9cedaea48b214475d4d5088cab1264afeb572209697095d86a93c`.
An exact-base `afa4d0bbd` 30-day rerun and the tip rerun have 30/30
byte-identical daily snapshots; day 30 is
`461f6647104ef9f2ad83dfe42b7a3fd5fd50c7489c0e1664290fd25871d6157a`.
An initial comparison to round 158 was discarded because round 158 is not the
current base; no number from that confounded comparison is used here.

R162-P5 is **CONFIRMED**. The shared production trajectory is unchanged.

## Prediction disposition

| prediction | disposition |
|---|---|
| R162-P1 final association is a missing live statement | **REFUTED**: the hypothetical hook moves 270 cells, but the compiled statement is dead under `ln_dynadv_vec`. The false default is passive. |
| R162-P2 pair reduces kt=1 barotropic disagreement | **REFUTED**: pair and per-substep-only counts are identical at 16,443/16,438 U/V. |
| R162-P3 pair removes the salinity veto | **REFUTED**: the arm is outside the executed source branch and S maximum also increases by `5.0160353275430225e-6`. |
| R162-P4 month advances beyond step 36 | **UNMEASURED_BY_PROTOCOL** after R162-P3 refuted. |
| R162-P5 GYRE default path unchanged | **CONFIRMED**: 70 rows, residual arrays and 30 daily snapshots are identical. |

## Validation and choices

The four focused gate files pass 27/27. All four non-vacuity controls
(passivity, boundary-bit, source-scope and salinity) fire. The round receipt
citation gate passes all 3 citations with zero unmapped spans, and its compiled
line shift fires. The cumulative receipt gate passes all 274 citations with
zero unmapped spans.

The prescribed separate read-only review was attempted, but `codex exec`
could not initialize its app-server client on the read-only filesystem:
**independent review unavailable in-sandbox**.

The single `tests/ocean/fidelity -n 12` battery collected 2,675 tests. Its
wrapper was interrupted after it stopped reporting progress at 99% with no
live pytest process: 2,645 passes, 7 skips, 4 failures and 19 unfinished tests
were recorded, but interruption prevented pytest from printing the four test
IDs. No failure appeared in the round-162 focused files, which independently
pass 27/27. This incomplete battery is reported, not promoted to green.

ASKED choices: none. UNASKED choices: empty. No configuration, forcing,
carried-state policy, stabiliser, sea-ice selector or production card changed.

## OPEN

1. Resume the source-ordered rung-0 growth walk at the first stage consumer of
   the already-associated vector-invariant external mode. Do not revisit the
   per-substep association or this dead flux-form final association.
2. The independent production month remains first non-finite at step 36 T
   `[86,159,0]`; the candidate month is unmeasured by protocol.
3. Keep the U-cyclic/V-fold halo pair separate and **HELD**. Its local live
   statement is exact, but its salinity compensation remains unresolved.
4. The rung-7 kt=10 stage-3 V maximum regression remains separately
   unattributed.
