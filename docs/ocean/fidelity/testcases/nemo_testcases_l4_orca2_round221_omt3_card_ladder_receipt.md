# ORCA2 round 221 — OMT-3 card and atomic-unit ladder

Date: 2026-10-10. Frozen base: `adef2eefc`. Preregistration commit:
`2ab18e857`. Status: **LANDED** (gate-local OMT-3 record/card; no production
physics change). Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round221/`.

This round changes no package model file, shipped rung-0/rung-10 card, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` tuple. Every
number below is labelled either **independent** or **given NEMO's entry**.

## Record disposition and correction

The operator's smoke, uninstrumented ten-step calibration, and both
rank-complete ten-step P3 twins all completed with `STOP 0`. Admission parses
80 self-describing frames per twin, both ranks, four checkpoints, and five
fp64 fields; makes 400 twin field comparisons; and compares four kt=10
terminal restart files byte-for-byte. The admitted status is
`PASS_R220_OMT3_ENTRY_STAGE_AND_MONTH_RECORD`.

NEMO's month run reached the compiled stability boundary at kt=15: |ssh| max
3.680 m, |U| max 1.975 m/s, |V| max 10.49 m/s, salinity 21.60--36.50 PSU. The
compiled predicate is
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:243-250`, and its emitted
diagnostic and stop are
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-310`. This **REFUTES**
R220-P3's 96-step completion prediction; it does not invalidate the complete
kt=1..10 record.

The acquisition's exit 123 came from Bash's global `ERR` trap at the month
pipeline, before the compiled-boundary classifier could run. The repaired
pipeline is syntactically guarded by an OR-list and preserves both
`PIPESTATUS` values. Its regression test failed before the repair and passes
after it. Admission used `--admit-existing`; NEMO was not rebuilt or rerun.

## Exact OMT-3 card edge

Search-before-build reused the admitted OMT-2 builder and the rung-0 card.
Exactly one numerical model-config field changes: lateral viscosity `A_h`
0.0 -> 100000.0 m2/s. The inherited resolved selection is
`nemo_div_curl`, `nemo_e3`, `nemo_ahm_3d_file`, and `free_slip`; every other
OMT-2 field is equal. A planted `A_h=0` edge refuses.

The compiled deck reads `namdyn_ldf` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-184` and selects one
OFF/laplacian/bilaplacian arm at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-228`. The runtime
dispatcher calls the iso-level Laplacian at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90:81-85`; its live div-curl
statements are
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.

## Baseline ladders

Both labels complete 40 checkpoints and 200 scored rows on CPU/JIT/fp64/libm.
Their RMS and maximum scores are numerically identical; the separate artifacts
remain authoritative because their bitwise unequal counts differ at the
labelled entry.

| labelled state | row | rms | maximum |
|---|---|---:|---:|
| independent | kt1 stage1 T | 1.4205672679174248e-05 K | 0.001294884324850809 K |
| independent | kt1 stage1 ssh | 0.006238271974498368 m | 0.1310917645484672 m |
| independent | kt10 stage3 T | 0.0002998871533285411 K | 0.05029513503569216 K |
| independent | kt10 stage3 S | 0.0028811546601132726 PSU | 0.3634585334278668 PSU |
| independent | kt10 stage3 ssh | 0.024416301389167186 m | 0.36632729531730257 m |
| given NEMO's entry | kt1 stage1 T | 1.4205672679174248e-05 K | 0.001294884324850809 K |
| given NEMO's entry | kt1 stage1 ssh | 0.006238271974498368 m | 0.1310917645484672 m |
| given NEMO's entry | kt10 stage3 T | 0.0002998871533285411 K | 0.05029513503569216 K |
| given NEMO's entry | kt10 stage3 S | 0.0028811546601132726 PSU | 0.3634585334278668 PSU |
| given NEMO's entry | kt10 stage3 ssh | 0.024416301389167186 m | 0.36632729531730257 m |

For both labels the first non-bit checkpoint remains kt=1 stage1 T.

## Complete vector-unit arm

The complete round-217 vector unit is replayed atomically: Ve_rhs/ssvmask fold
pair, seven-array association, and V transport. It completes kt=1..10 on
OMT-3 for both labels. This **REFUTES** preregistered R221-P5: restoring lateral
momentum diffusion does not recreate rung-0's kt=8 live-W-thickness refusal,
so momentum LDF is exonerated as the switch carrying that compensating partner.

| labelled state | row | before rms / max | atomic-unit rms / max |
|---|---|---:|---:|
| independent | kt1 stage1 T | 1.4205672679174248e-05 / 0.001294884324850809 | 2.2271934337297844e-06 / 0.00027876686458938593 |
| independent | kt1 stage1 ssh | 0.006238271974498368 / 0.1310917645484672 | 0.00023180348917108855 / 0.006277375309180683 |
| independent | kt10 stage3 T | 0.0002998871533285411 / 0.05029513503569216 | 0.00011496970296420367 / 0.010734380541439492 |
| independent | kt10 stage3 S | 0.0028811546601132726 / 0.3634585334278668 | 0.00042038194335974844 / 0.04372048071099499 |
| independent | kt10 stage3 ssh | 0.024416301389167186 / 0.36632729531730257 | 0.003435914940642178 / 0.055099784353873416 |
| given NEMO's entry | kt1 stage1 T | 1.4205672679174248e-05 / 0.001294884324850809 | 2.2271934337297844e-06 / 0.00027876686458938593 |
| given NEMO's entry | kt1 stage1 ssh | 0.006238271974498368 / 0.1310917645484672 | 0.00023180348917108855 / 0.006277375309180683 |
| given NEMO's entry | kt10 stage3 T | 0.0002998871533285411 / 0.05029513503569216 | 0.00011496970296420367 / 0.010734380541439492 |
| given NEMO's entry | kt10 stage3 S | 0.0028811546601132726 / 0.3634585334278668 | 0.00042038194335974844 / 0.04372048071099499 |
| given NEMO's entry | kt10 stage3 ssh | 0.024416301389167186 / 0.36632729531730257 | 0.003435914940642178 / 0.055099784353873416 |

For each label the Decision-96 census is: 195/200 RMS-score rows moved, all
195 toward NEMO, zero away, five unchanged; maximum census 181 toward, 14
away, zero equal; first-over-bar row toward; zero exact rows lost; final kt10
stage3 ssh maximum 0.36632729531730257 -> 0.055099784353873416 m. The unit
therefore qualifies locally on OMT-3. It is **not** promoted again: production
already contains this exact unit, and rung 0 still reaches its registered kt=8
refusal.

Artifacts are `omt3_{independent,given}_{before,after}.json` and
`omt3_{independent,given}_decision96.json`. Their SHA-256 values are,
respectively, `f9e9943eb079526b45f9e8b0e78f8bc4c96a458d2414d827e8b3351ef296ac98`,
`9833f7facecb8cde871a32101362e8457678c76857d689b1343f612e6d620c15`,
`82a5d2d671f20f6a7e13c616c479fa7c15962f9960623e269dcc83ece80e057c`,
`ab623aa1d00307b76b6df3f6b4b4ae1583774dfc9cdd4b4fcfff036e89062655`,
`d40ae0176f9f449b9783ff7ba4a6101ad230b7c4b91ed42cae5debbf3581945b`,
and `9df2426afd53999c18eb8f781b3a87919106ee38bf2552244f6ded708eae4ca2`.
Pair-closure, exact-loss, and false-majority plants all refuse.

## Preregistration dispositions

| ID | disposition |
|---|---|
| R221-P1 | **CONFIRMED**: the operator output is complete; exit 123 was only the launcher's premature shell trap. |
| R221-P2 | **CONFIRMED**: the existing ten-step record admits without rerunning NEMO. |
| R221-P3 | **CONFIRMED**: the card edge changes only `A_h` and resolves the shipped momentum-LDF selections. |
| R221-P4 | **CONFIRMED**: both labelled baseline ladders complete 40 checkpoints / 200 rows. |
| R221-P5 | **REFUTED**: the complete unit remains finite through kt=10; momentum LDF is not the refusal-carrying switch. |
| R221-P6 | **CONFIRMED**: all registered plants fire. |

## Validation and review

Focused acquisition/card/citation tests pass 11/11. Independent review was attempted
with `codex exec --sandbox read-only` and exited 1 before reading the diff:
`failed to initialize in-process app-server client: Read-only file system (os
error 30)`. **Independent review unavailable in-sandbox**; this is not a PASS.
The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

No `packages/` file changed, so no production GYRE/DINO/tank trajectory can
move in this round. On clean implementation commit `cbc64c340`, the round
citation gate passes six citations and the cumulative default gate passes 274;
both report zero failures, unmapped citations, or map-audit failures. The
explicit two-line shift plant on the mapped dynldf range exits 1 and makes the
gate fail. Round/default/plant SHA-256 values are
`25d01a542c6069058912fdd7f25ef426388282af2ef5804fcf618b220999d245`,
`6f0a08c5a411885077e6e0824cbc99fd686de70f0ef865c231fad0de798ffd83`,
and `4173c61a45f73de90e65a7ef3e5e02a5edf11a3738517956b2c8582c0681b3b6`.

The prescribed `tests/ocean/fidelity -n 12` battery collected 3,072 tests and
reached 99%; 3,037 passed and seven skipped. Its four failures are the same
registered pre-existing reds reported in round 220: the GYRE round-129
spread-floor record stamp, allow-dirty scope, worktree-stamp ratchet, and SI3
scalar-math provenance gate. The remaining 24 tests were unclassified when
the pytest processes disappeared without a terminal summary; the idle wrapper
was then interrupted. The battery is not called PASS and was not relaunched.
Its log SHA-256 is
`8e35c1843ef30a184e4be649027aa6978d5c7356e0b5348170628e4002f5bf33`.

## OPEN

OMT-4 is next under Decision 109: OMT-3 plus tracer advection, using rung-0's
resolved tracer-advection settings and otherwise changing no field. Copy the
last working binary-reuse acquisition file by file, change only that deck edge
and run names, run the identical two-step smoke first, acquire the rank-complete
record and boundary, build the gate-local card, and score the same atomic unit.
If the kt=8 refusal first returns on OMT-4, tracer advection carries the
compensating partner and its first statement is walked offline before OMT-5.
No constituent is promoted separately. Sea ice and its `unmeasured_features`
tuple remain untouched.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
