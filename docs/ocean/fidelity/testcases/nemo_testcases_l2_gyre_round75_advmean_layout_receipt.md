# NEMO-testcases L2 GYRE round 75: advective-mean layout repair receipt

Date: 2026-09-12. Final disposition: **STOPPED FOR RECORD; no production
physics or scientific configuration changed**. A new fail-closed acquisition
card is ready, but no new oracle payload was produced in this round.

## Verdict and first non-bit statement

The round-74 record-size refusal is explained exactly. The failed record is
3,041,176 bytes; its declared layout is 3,789,976 bytes. The 748,800-byte
deficit equals two 36 by 26 float64 fields at each of 50 external substeps:
`50 * 2 * 36 * 26 * 8 = 748800`.

The failed compiled writer opens kt=2 while allocating only the two accumulator
snapshots at
`GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:442-450`.
The two velocity snapshots are allocated only by the kt=1 arm at the same
compiled source's `:390-393` and populated only by its kt=1 predicate at
`:503-506`. Nevertheless, every kt=2 substep attempts to write those two
snapshots among ten named fields at `:581-584`. They are unallocated on kt=2,
so those two slots emit no bytes. The exact observed deficit confirms the
field count and precision; no model output was interpreted to invent a
scientific cause.

Round 73's first non-bit statement remains `un_adv`: 580 of 580 wet U cells
differ, with maximum absolute difference `0.00012029895814569258`. The live
compiled source still assigns the substep weight and updates the U accumulator
before V at
`GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:572-579`.
No valid kt=2 producer record exists yet, so that ordered walk remains
**UNREACHED** rather than inferred.

## Repair and fail-closed acquisition

The replacement acquisition remains in the existing
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round73_advmean/`
card. It selects the new target `GYRE_OMIP_L2_P3_SM_R75ADV3` and new evidence
directory `round75/oracle_advmean_kt2`, both absent when preflight ran. It still
clones GYRE_PISCES, copies the R72 EXP00 and MY_SRC files one by one, applies an
additive config-local patch, compares `namelist_cfg` byte-for-byte, builds a
new target, and stamps the clean commit that actually invokes the full run.

The WRITE-only repair preserves the kt=1 `l2_u_mid` and `l2_v_mid` slots. Its
kt=2 arm writes the allocated live `ua_e` and `va_e` values into the same two
stream positions. The reader now owns one ordered ten-field registry, and the
record-size gate derives its 3,789,976-byte expectation from that registry.
The preflight separately requires the four writer clauses, ten fields, the
same byte count on both sides, one header, and the existing resolved run rows.
No numerical expression, namelist, default, carried state, or production file
changed.

Exact preprocessing with the R72 keys and includes followed by
`gfortran -fsyntax-only` exits zero with empty stdout and stderr. The final
clean-tree preflight at commit `5f3f2d3e422c` prints
`ROUND75_ADVMEAN_PREFLIGHT_READY` and exits zero before `makenemo` or `mpirun`.
Its log SHA-256 is
`405a2cc4b7190ecd99931d5a433e46759f3faabbb389ba10939bba52bea113ec`.

The operator's full run must still emit exactly 3,789,976 bytes; replay all 100
U/V entry-plus-increment associations and both normalization associations
bit-for-bit; keep the exact header, EOF, stamp, final restart, and mesh mask;
admit only the named new record; and make the stamp, header, truncation,
replay-ULP, consumed-field, and layout plants exit nonzero. This round did not
invoke `makenemo` or `mpirun`.

## Preregistered failures and controls

Two preregistered predictions were initially **REFUTED** and remain recorded:

- The first committed preflight stopped at its own parent-line guard because
  the old combined WRITE's continuation was not classified as instrumentation.
  It exited 66; log SHA-256
  `1cbcfc37a2779e0283de367de27688e82a45c4c699ded9b8a46dd3915934d34e`.
  The guard now admits only that exact continuation while retaining the
  six-line census and refusal of non-writer parent lines.

- The first missing-field plant printed `layout plant stayed green`. Its
  predicate ran inside a shell `if`, so a failed middle comparison was
  overwritten by the final comparison's success. The log SHA-256 is
  `977e4cafb61d164bf8e9a39dd17de628b9a63f161367dd17b3bd06df6576d9fd`.
  The four clauses are now chained; no failed clause can be overwritten.

The corrected layout plant removes one kt=2 velocity slot, names that removal,
and exits 69. Its log SHA-256 is
`9f0f199cf5ff5b2e6cae2f434bb43368f279b6f24a60558200705dd87cee5dae`.
The inherited resolved-row plant changes `nn_e` from 50 to 49, refuses the
missing 50 row, and exits 65; log SHA-256
`32f9dedbb636839a7c6a95fbe63f52371ede4e43a2323570e798e2dcf8e8b544`.
Both final controls are nonzero and non-vacuous.

## Rule 12 card

| card | changed statement | disposition |
|---|---|---|
| GYRE | none in production; WRITE-only acquisition layout only | **STOPPED FOR RECORD** at the already non-bit `un_adv` input. The kt1--10 ladder and days 1--30 are **UNREACHED**. No registered or AT-BAR row moved, and first-over-bar cannot move. |
| LOCK_EXCHANGE | none | No production statement changed; tank gating is **UNREACHED** and the shared implementation is unchanged. |
| OVERFLOW | none | No production statement changed; tank gating is **UNREACHED** and the shared implementation is unchanged. |
| DINO | none | **UNREACHED**. The external-mode accumulator remains shared-statement risk, and DINO's 96--98% per-row cancellation requires its own row and pair gates before any future production edit. |
| ORCA2 | none | **UNMEASURED WITH SPEC**: resolve its compiled external-mode card; record every substep entry, weight, transport, reciprocal metric, exit, normalization, and boundary handoff for kt1--10; replay in compiled order; register every moved row; retain every AT-BAR row; and forbid an earlier first-over-bar boundary. |

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest changed. The round-70 patch remains held.

## Review and focused checks

The required separate Codex pass used `codex exec --sandbox read-only` against
the complete committed round-75 instrument diff and explicitly tried to refute
the byte cause, writer/reader layout, plants, commit stamp, and Rule 12 table.
It exited before review because the client could not initialize in this
filesystem sandbox. Its terminal verdict, quoted verbatim, is: **“Error:
failed to initialize in-process app-server client: Read-only file system (os
error 30)”**. There is no SHIP or DO NOT SHIP verdict; review is
**UNMET/BLOCKED**, not treated as approval. The log SHA-256 is
`bb1f71537a57c448673b7081b1523197c0da0e3574b4dea9ac7003eb4a047e84`.

The first focused citation run was **REFUTED**: it found that the new header
WRITE anchor occurred twice, and dirty-tree provenance correctly failed two
receipt tests. After pinning the second occurrence and committing the map, the
final focused run reports 18 passed in 1.55 seconds. Its JUnit SHA-256 is
`1eb934b947768c1ec2220efe2824012e3f55e8cec9f1ff8f9d04e30031fe5165`.
The receipt citation gate found five mapped citations and no failures at clean
receipt commit `a7d651a76ba8`; its JSON SHA-256 is
`e5d9e43c57d2873d62586864ae9e590df9e0eb188781ef4a34b4f206858aa577`.
Shifting the failed writer citation by two lines produced
`SYMBOL-NOT-AT-LINE` and exited 1; its log SHA-256 is
`894712ce1426e03b9a85a7ef609048772ebb8018156a47e8e9c258ef03248b72`.
Shell parsing, Python compilation, exact preprocessing, Fortran syntax, and
`git diff --check` pass.

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | diagnose the byte mismatch and issue a new acquisition target | two absent kt=2 velocity fields repaired; round-75 target prepared exactly as requested |
| UNASKED | scientific configuration, carried state, stabilizer, scoring, or production choice | none performed |

OPEN for round 76: the operator must run the round-75 acquisition and admit its
output. Then read and cite the new target's compiled writer and walk the kt=2 U
accumulator in actual substep order: entry `un_adv`, `wgtbtp2`, `zhU`,
`r1_e2u`, the left-associated increment, and exit `un_adv`. Stop at the first
non-bit input or association; continue through all 50 substeps only while
exact, then check normalization and pre/post-LBC boundaries. Walk V only after
the U owner is named. Do not infer from kt=1, resume the held round-70 patch,
or edit production until an independently computable shared legoESM statement
is bit-exact on NEMO inputs and has a preregistered full Rule-12 causal card.
