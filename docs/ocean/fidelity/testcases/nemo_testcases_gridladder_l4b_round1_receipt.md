# Grid ladder rung L4b — round 1 receipt (gridladder_r54, Codex)

Date: 2026-10-20

Starting lane-A commit: `b4f70973d0d4148fe4390a3a7071568ad892f623`

Merged lane-B commit: `fb02c4b29f397fbd1e8bdbcb10239ec9a1a2102f`

Merge commit: `e68604baf0b8f78c8df6ab91bbd934a9fdbe8709`

Preregistration commit: `7d30dc3162`

Implementation and acquisition-tool commits: `f0549b581d`, `376a6e6ee6`

Citation re-anchor commit: `6b6c43efe6`

Independent-review repair commits: `ff080c094c`, `b8c68d344b`

Resume repair commit: `288311ef9b`

Status: **L4b CERTIFIED.** The operator acquisition completed steps 1--10 in
three byte-identical twins. This resume admitted that existing record without
running NEMO, repaired three review controls without changing a model value,
and scored both labels, every available full-depth row, wet/dry partitions,
kt-1 stages, and the geothermal trend bitwise exact. The earlier stopped-build
status below is retained as history and is superseded by section 6.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- scoring: not run; no L4b record exists

## 1. Lane merge and exact preservation

The merge has two parents: lane A at `b4f70973d0` and lane B at
`fb02c4b29f`. The four textual conflicts were resolved as follows:

1. The historical GYRE receipt kept lane B's already re-anchored citations.
2. The recipe now keeps the union of lane A's geothermal and G9 selectors and
   lane B's V1--V5 and L1--L3 selectors. Independent review found that the
   merge had dropped lane A's `stage transport stored r3u/r3v` census row even
   though card values remained unchanged; the row and a G9 flip plant are now
   restored. An AST audit compared all 77 literal selector rows from each
   parent. Lane B's rows are all present. The only remaining lane-A label
   differences are four OFF-only rows generalized by lane B into conditional
   ON/OFF rows (`ln_traldf`, `ln_zdfcst`, `ln_zdfevd`, and `ln_trabbl`); their
   existing plants cover both arms.
3. The ladder scorer kept the geothermal, TKE/mixing-length, and DDM arms and
   the union of their case tables. A merge-resolution omission of the
   geothermal arm's `return out` was found by the targeted L4 scorer test and
   restored inside `f0549b581d`, not before that implementation commit;
   without it the arm fell through into unrelated later scoring logic.
4. The citation map kept the union. Before this receipt, its 82 source-shifted
   entries were re-anchored and its one shadowed duplicate key was removed.
   The map audit now reports zero failures.

The committed merge-identity probe first measured the merge before any L4b
model change. It ran two production steps for all 18 inherited cards: G1--G9
and L4 against the lane-A parent, and V1--V5 plus L1--L3 against the lane-B
parent. Across 700 state leaves, the result was zero differences. A planted
one-bit digest change was refused. The evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/merge_identity`.

After implementing L4b, the same probe compared all 18 inherited cards with
the pre-implementation merged snapshots. All 700 leaves were again
bit-identical, with zero differences. The snapshot stamps are
`376a6e6ee67c6508769bef92eae39b72be1e6b7c` and
`6b6c43efe62243ff950e47b51b5dc7ec0ca13bf6`; the latter changes only tests,
receipts, and citation anchors, not model code. A planted one-bit change in
L3 was refused with exit status 2 and exactly one reported differing leaf.
The post-implementation snapshots and comparison reports are in the same
evidence root under `post_l4b` and `post_l4b_reports`.

## 2. L4b card and ordered stage-3 composition

The new card is `ORCA2L4b-iso-bbc`. Its rendered deck is certified L2 plus
exactly four geothermal entries: `ln_trabbc=T`, `nn_geoflx=2`, `cn_dir='./'`,
and the variable `heatflow` descriptor. EIV, BBL, interior tracer damping,
tracer advection, and altered vertical mixing remain off.

The executed Oracle source calls `tra_ldf`, `tra_bbc`, `tra_bbl`, then
`tra_dmp` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`. L4b therefore
materializes the certified isoneutral T/S additions first, then the geothermal
temperature addition. Only L4b selects the new default-false order flag; L4
and every inherited card retain their previous path.

The exact x64 kernel test is JIT-compiled and receives traced inputs. With stage
temperature `1e16`, lateral-diffusion increment `-1e16`, and geothermal
increment `1`, the selected LDF-then-BBC association returns exactly `1`; the
reversed association returns `0`. Its salt control returns exactly `7`.
Gradients with respect to stage temperature, the LDF increment, and the BBC
increment are finite and exactly one. The implementation adds no division or
square root.

The production one-step gate now compares L2 and L4b through the same live
WS-RK3 observer. Their stage-1 and stage-2 sources, all stage thicknesses,
stage-3 salt source, and L4b's pre-geothermal stage-3 temperature source are
bit-identical. The consumed L4b temperature source differs only at wet bottom
cells, and the resulting explicit stage-3 temperature differs from L2 at
exactly those wet bottom cells, bit for bit. The full-step S/u/v/eta leaves
remain bit-identical; the implicit
vertical solve then carries the bottom heat upward within heated wet columns,
so final temperature remains identical at every dry cell and differs at every
heated bottom. A plant that adds geothermal both before and after LDF is
refused by the pre-geothermal identity check.

The validator accepts the complete L2-plus-geothermal selection and rejects
both direction errors: geothermal plus LDF without the order flag, and the
order flag without geothermal. Plants also reject missing/non-L2 GM/Redi,
nonzero scalar lateral diffusion, EIV-like nonzero bolus diffusivity, the
written FCT Krhs tuple, and BBL. The exact deck test separately binds EIV,
BBL, and damping off.

## 3. Acquisition boundary and seven-item preflight

The committed acquisition entry point defaults to read-only preflight and has
separate `--run` and `--admit-existing` modes. Admission rechecks the rendered
deck, numeric log tokens, and observer log for both the two-step smoke and
writer-free reference, but runs acting-record checks on the smoke only; the
reference intentionally writes no `oracle_*` files. It repeats the deck, log,
and acting checks for each twin, checks raw observer identity, and requires
restart passivity plus its planted violation. Numeric deck/log comparisons
parse binary64 values. Each logical uses a whitespace-tolerant end-of-line
regex, including `ln_trabbc`; the source contains no `cmd | grep -q` construct.

The first preflight attempt refused before acquisition because its Oracle
order matcher expected single-spaced source text. Commit `376a6e6ee6` changed
that check to match the actual preprocessed whitespace. The later clean-tree
result quoted below is withdrawn: its preflight checked only that the checker
function names appeared and did not execute either one. The real padded L2
round-17 log makes that old numeric checker fail, and the writer-free round-17
and round-53 references make the old admission raise on missing records.

The superseded preflight had printed:

```text
L4B_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=true
L4B_PREFLIGHT_OK commit=376a6e6ee67c6508769bef92eae39b72be1e6b7c reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L4B_ISO_BBC_R54
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/oracle_l4b_r54
```

The saved stdout remains
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/l4b_preflight.log`.
It is withdrawn history, not a valid launch authorization.

The corrected preflight executes the numeric checker on a committed verbatim
excerpt of the real L2 round-17 `twin_a/ocean.output` (full-file SHA-256
`6208350f4314d68247d2bee38ec424f58d9f19c32afa53cc89518fee4f708be6`)
and rejects a one-flag flip. It executes the acting checker on a scratch smoke
directory backed by pinned real L2 acting records, proves that invoking it on a
synthetic writer-free reference fails, and passes the shared smoke/reference
scope only while the reference has no `oracle_*` file. A planted reference
record is refused. The corrected preflight result is recorded below after the
review-fix commit. It creates no acquisition directory; neither `--run` nor
`--admit-existing` is invoked by Codex.

The first corrected invocation failed closed inside item 5 because the
preflight pre-created the synthetic acting directory before its writer did.
No acquisition path was created. Commit `b8c68d344b` corrected that internal
scratch ordering and changed the scratch root to collision-safe `mktemp`. From
a clean tree, with both evidence and target independently confirmed absent,
the default invocation then produced:

```text
L2_ISO_ACTING ldf_wet_nonzero=860708 ldf_dry_nonzero=0 completed_tracer_unequal=861013 entry_T_unequal=430464 entry_S_unequal=430549 entry_u_unequal=0 entry_v_unequal=0 entry_ssh_unequal=0 barotropic_checked=False barotropic_unequal=0 pass=True
L4B_PREFLIGHT_7_OF_7_PASS plants=7/7 checker_functions=2/2 evidence_parent_absent=true target_absent=true
L4B_PREFLIGHT_OK commit=b8c68d344b72fc3a15999a821cdd3c4fc061f49a reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L4B_ISO_BBC_R54
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/oracle_l4b_r54
```

## 4. Tests and gates

The merge probe's two unit tests passed. The L4b contract passed 6 / 6 tests:
exact deck delta, acquisition entry-point structure, exact card delta,
bidirectional validation plants, and exact ordered JIT/gradient behavior. The
combined L4b/L4/L1/L2/L3 suite passed 72 selected tests, including the two
slow L2 tests run separately; there were no failures.

The review reproductions were positive before repair: the real padded L2 log
made the numeric checker return 1; both inspected writer-free references made
the acting checker raise for absent records; the G9 stored-r3u/r3v flip
returned an empty failure list; and the exact double-add source plant passed
all 22 then-existing L4b/L4 tests. After repair, the focused L4b/L4/observer/G9
run passed 27 / 27 in 463.66 seconds. The final strengthened one-step gate and
observer-layout run passed 4 / 4 in 348.50 seconds. With the double-add plant
reapplied, the new gate fails at the bitwise pre-geothermal stage-3 comparison
(16,433 unequal words), so the plant is no longer invisible.

The citation repair moved 82 shifted pins and removed one duplicate. Its
shift-sensitive audit reports zero failures, the citation-gate test file
passes 17 / 17, and the private-import ratchet passes 6 / 6.

The exact ORCA2 default landing suite from `claude_rounds/land.sh` passed
145 / 145 in 1037.36 seconds. The explicit gate on this receipt found one
citation, zero failures, zero unmapped citations, zero failing map-audit
entries, and all nine self-test plants fired. The default receipt gate found
274 citations with the same zero-failure result and all nine plants fired.

No trajectory scorer or fidelity bar was invoked because there is no admitted
L4b record.

## 5. ASKED / UNASKED (pre-acquisition history)

ASKED and completed: merge lane B; list and resolve every conflict; measure
all inherited two-step cards before and after L4b; add the L2-plus-geothermal
card in NEMO order; retain L4 and existing-card programs; plant both order
directions and unsupported combinations; commit the ladder-standard
acquisition tool; run its seven-item read-only preflight; stop for acquisition;
run citation, default, and private-import gates; reproduce and repair every
independent-review finding; audit the selector union; and prove the corrected
checker functions and double-add plant execute.

UNASKED and not done: execute NEMO; create, alter, admit, or score an L4b
record; claim a bar verdict or certification; begin Decision 146 interior T/S
damping; relax any fidelity bar; push; or post an external review. Review is
reserved for the operator named above.

## 6. Resume: admitted record and review repairs

The operator reported
`L4B_ACQUISITION_PASS steps=1..10 twins=3` for
`gridladder_rounds/round54/oracle_l4b_r54`. Codex did not build or run NEMO.
From clean commit `288311ef9ba02aff02e4cc47e29fb5379e96edcc`,
`--admit-existing` reran the exact deck and log checks, acting checks, all
three record inventories, raw twin identity, and restart passivity. It ended:

```text
L4_GEOTHERMAL_RECORDS_PASS steps=1..10 sha256=875764008fa6e2dfb021b125df9e23cf349a55f45ac26d948274bbbf4abef2df
PASS_G5_ADMISSION unequal_rows=None
L4B_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/oracle_l4b_r54
```

The repaired admission calls the smoke's acting check as its own statement,
outside any conditional command. A temporary smoke holding the real deck,
log, and timestep but no `oracle_*` records was refused with status 66 and did
not print an acquisition pass. Preflight scratch now has an `EXIT` cleanup
trap. Two consecutive deliberate preflight refusals both reached the occupied
target check and each left no `.l4b-preflight.*` directory.

The stage-3 observer now exposes the live compiled post-LDF source and the live
geothermal rate. The production test requires the composed temperature source
to equal `nemo_source_round(post_ldf + qgh_rate)` bit for bit. This binds the
executed order at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`, not merely the set of
cells that changed. Moving geothermal before LDF at the call site while leaving
the helper intact made the test fail at 697 words; the correct order passes.
The trace-only repair does not alter the production state.

The admitted-record test runs the geothermal statement through an actual JIT,
records that the live `e3t_Kmm` and `heatflow_mwm2` arrays are runtime
arguments, requires both to be binary64, and matches the recorded trend with
zero unequal cells. Removing the geothermal production wire changes all
16,428 nonzero-flux wet columns. Removing JIT, closing over either input, or
using binary32 makes the same test fail its execution-wiring assertions.

## 7. Clean-tree L4b score

All four score files stamp commit
`288311ef9ba02aff02e4cc47e29fb5379e96edcc` and packages tree
`c7d5d819ea76c18c34a90b285e940cfae2acd149`. CPU, x64, production JIT, the
admitted three restart roots, and the preregistered entries were used.

| label / record | coverage | exact | unequal |
|---|---:|---:|---:|
| INDEPENDENT surface | kt 1--10 | 70 / 70 | 0 |
| INDEPENDENT full/partial/dry depth | restarts kt 2--10 | 36 / 36 | 0 |
| GIVEN-NEMO-ENTRY surface | direct + two-step outputs kt 1--10 | 119 / 119 | 0 |
| GIVEN-NEMO-ENTRY full/partial/dry depth | every available restart row | 64 / 64 | 0 |
| stage surface | kt 1, stages 1--3 | 21 / 21 | 0 |
| stage full/partial/dry depth | kt 1, stages 1--3 | 12 / 12 | 0 |
| geothermal wet-bottom trend | kt 1--10, 16,433 wet columns each | 164,330 / 164,330 cells | 0 |
| geothermal dry slots | kt 1--10 | all canonical zero | 0 |

Each geothermal step has 16,428 nonzero-flux wet columns and five zero-flux
wet columns. Every maximum absolute difference is 0.0. Score SHA-256 hashes:

```text
52256fa6092ae38a56fd479ba61777d72022ec4e91cb18792e9c7381a63c6d09  l4b_independent_kt1_10.json
581b0bbd8c762ea795a504292ef8589d36a6a515b524530b3ae4266625d7fbe0  l4b_given_entry_kt1_10.json
ceaa304fa4ea8ece0edf4bd4db15f5aa1e71245365839dcee32ac9d55cf9a081  l4b_stages_kt1.json
325dad32e95a20a2f8c2e9bde3a477c45a8e045a07e6c5dd2ee8cd810caa6d75  l4b_geothermal_kt1_10.json
```

The operator's post-fix review flipped every boolean on all 18 inherited cards
(1,818 flips, every guard still firing) and measured 36/36 two-step identities
on 5 cards (G9, L4, L1, L2, L3). This resume changes only a private write-only observer payload on the
L4b-selected branch; inherited selectors and state returns are unchanged. The
merge-identity unit probe and its one-bit plant remain passing.

L4b therefore clears the exact bar and is **CERTIFIED: geothermal after
standard isoneutral diffusion in NEMO's order**. Decision 144's ladder
precondition for a later ORCA2 import is satisfied. Decision 146 interior T/S
damping is not started in this round.

## 8. Resume gates and disposition

- codex review: operator
- independent review: operator (Claude)
- NEMO execution in this resume: none
- focused tests: the admitted-record JIT/traced/x64 gate passed 1 / 1 in
  11.05 seconds; the observer-layout gate passed; and the correct production
  one-step gate passed 1 / 1 in 377.89 seconds. The moved-call plant failed
  that same bitwise composition assertion at 697 words.
- receipt citation gate: PASS from clean commit
  `ac1c27101edae4b2062226c052e5056e45253c73`; one citation found, zero
  failures, zero unmapped citations, zero failing map-audit entries, and all
  nine self-test plants fired. The observer insertion required 77 mechanical
  line-pin moves plus one expanded-extent and one unique-endpoint repair.
- default gate and private-import ratchet: the exact ORCA2 test list from
  `claude_rounds/land.sh` passed 145 / 145 in 1580.29 seconds from the same
  clean commit. Its six private-import tests all passed.

ASKED and completed: repair all three reviewed controls with plants; admit the
operator record; score both labels through kt 10 at full available depth,
wet/dry, stages, and geothermal trend; keep inherited-card identity; add a
JIT/traced/x64 admitted-record test; and certify only on exact rows.

UNASKED and not done: rerun NEMO; alter a model value; change any fidelity bar;
import geothermal into the full ORCA2 card; start Decision 146 damping; push.

UNVERIFIED: none in the L4b certification claim or repository gates.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r54/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L4b CERTIFIED: both labels, every recorded depth, wet/dry cells, kt-1 stages, and geothermal trends are bitwise exact
