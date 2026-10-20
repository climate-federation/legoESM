# Grid ladder rung L4b — round 1 receipt (gridladder_r54, Codex)

Date: 2026-10-20

Starting lane-A commit: `b4f70973d0d4148fe4390a3a7071568ad892f623`

Merged lane-B commit: `fb02c4b29f397fbd1e8bdbcb10239ec9a1a2102f`

Merge commit: `e68604baf0b8f78c8df6ab91bbd934a9fdbe8709`

Preregistration commit: `7d30dc3162`

Implementation and acquisition-tool commits: `f0549b581d`, `376a6e6ee6`

Citation re-anchor commit: `6b6c43efe6`

Status: **STOPPED FOR RECORD — L4b is implemented and preregistered, but is
not scored or certified.** The default acquisition preflight passed and
stopped at its writer-free `WOULD_RUN` boundary. Codex did not build or run
NEMO. An operator must run the exact `--run` command in the final machine
lines, after which a later round may admit and score the resulting record.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- scoring: not run; no L4b record exists

## 1. Lane merge and exact preservation

The merge has two parents: lane A at `b4f70973d0` and lane B at
`fb02c4b29f`. The four textual conflicts were resolved as follows:

1. The historical GYRE receipt kept lane B's already re-anchored citations.
2. The recipe kept the union of lane A's geothermal selector and lane B's
   V1--V5 and L1--L3 selectors. Lane B's repaired L1 end-stage pivot-fold
   selector was retained.
3. The ladder scorer kept the geothermal, TKE/mixing-length, and DDM arms and
   the union of their case tables. A merge-resolution omission of the
   geothermal arm's `return out` was found by the targeted L4 scorer test and
   restored before the L4b implementation commit; without it the arm fell
   through into unrelated later scoring logic.
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

The exact x64 test is JIT-compiled and receives traced inputs. With stage
temperature `1e16`, lateral-diffusion increment `-1e16`, and geothermal
increment `1`, the selected LDF-then-BBC association returns exactly `1`; the
reversed association returns `0`. Its salt control returns exactly `7`.
Gradients with respect to stage temperature, the LDF increment, and the BBC
increment are finite and exactly one. The implementation adds no division or
square root.

The validator accepts the complete L2-plus-geothermal selection and rejects
both direction errors: geothermal plus LDF without the order flag, and the
order flag without geothermal. Plants also reject missing/non-L2 GM/Redi,
nonzero scalar lateral diffusion, EIV-like nonzero bolus diffusivity, the
written FCT Krhs tuple, and BBL. The exact deck test separately binds EIV,
BBL, and damping off.

## 3. Acquisition boundary and seven-item preflight

The committed acquisition entry point defaults to read-only preflight and has
separate `--run` and `--admit-existing` modes. Admission rechecks the rendered
deck, numeric log tokens, observer log, and record inventory for both the
two-step smoke and writer-free reference; it then repeats those checks for
each of three twins, checks raw observer identity, and requires restart
passivity plus its planted violation. All numeric deck/log comparisons parse
binary64 values. The source contains no `cmd | grep -q` construct.

The first preflight attempt refused before acquisition because its Oracle
order matcher expected single-spaced source text. Commit `376a6e6ee6` changed
that check to match the actual preprocessed whitespace. The new clean-tree
run then produced:

```text
L4B_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=true
L4B_PREFLIGHT_OK commit=376a6e6ee67c6508769bef92eae39b72be1e6b7c reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L4B_ISO_BBC_R54
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/oracle_l4b_r54
```

The saved stdout is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/l4b_preflight.log`.
The successful run created no acquisition directory. `--run` and
`--admit-existing` were not invoked.

## 4. Tests and gates

The merge probe's two unit tests passed. The L4b contract passed 5 / 5 tests:
exact deck delta, acquisition entry-point structure, exact card delta,
bidirectional validation plants, and exact ordered JIT/gradient behavior. The
combined L4b/L4/L1/L2/L3 suite passed 72 selected tests, including the two
slow L2 tests run separately; there were no failures.

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

## 5. ASKED / UNASKED

ASKED and completed: merge lane B; list and resolve every conflict; measure
all inherited two-step cards before and after L4b; add the L2-plus-geothermal
card in NEMO order; retain L4 and existing-card programs; plant both order
directions and unsupported combinations; commit the ladder-standard
acquisition tool; run its seven-item read-only preflight; stop for acquisition;
and run citation, default, and private-import gates.

UNASKED and not done: execute NEMO; create, alter, admit, or score an L4b
record; claim a bar verdict or certification; begin Decision 146 interior T/S
damping; relax any fidelity bar; push; or post an external review. Review is
reserved for the operator named above.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r54-749527799
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r54-749527799/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4b_orca2iso_bbc/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L4b implemented and preflighted; operator acquisition is required before admission, scoring, or certification
