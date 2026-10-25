# ORCA2 round 334 receipt — CI ratchet cleanup (DECISION 166)

Round: 334 (Claude). Parent `072e4aa1b1c8`, fix commit `11943cc1a076`.
Disposition: **LANDED-pending** (operator review + push gate + DINO month gate).
codex review: operator.

## Ratchet suite, before

Ten ratchet files, fresh detached worktrees, `JAX_ENABLE_X64=1`, CPU.
Lane tip `072e4aa1b1c8`: 8 failed, 12275 passed. Main `b090be7b18d9`: 4 failed, 11412 passed.

| failing test ID | lane tip | main | class |
|---|---|---|---|
| hardcoded constants `[tests/unit/test_jra55_do.py]` | F | F | on main |
| inline coeffs `[packages/land/legoesm/land/restart.py]` | F | F | on main |
| param specs `[.../physics/shortwave_penetration.py]` | F | F | on main (lane lists 3 of main's 4 bad names) |
| no private cross imports | pass | F | main only |
| dispatch `test_no_dispatch_guard_removed` (slab land `step_land`) | F | pass | inherited at the fork point, fixed on main since |
| inline coeffs `[.../physics/geothermal.py]` | F | pass | **lane** (round 320 import) |
| inline coeffs `[.../lateral_mixing/mle.py]` | F | pass | **lane** (round 323 import) |
| inline coeffs `[.../lateral_mixing/mle_latlon_cgrid.py]` | F | pass | **lane** (round 323 import) |
| param specs `[.../lateral_mixing/mle.py]` | F | pass | **lane** (round 323 import) |

The dispatch failure: main commit `642481601` (2026-10-03) moved slab land's
unknown-scheme raises from `step_land` into `step_land_with_diagnostics`. That
was a deliberate move, not a removal. Main's `f9cde047d6` (2026-10-05) then
updated the baseline. The lane forked at `cffa2ab79` (2026-10-05), which
contains the first commit but not the second. The lane never touched either
file. The fix arrives with the next main merge, so it is not fixed here.

## Fixes (lane items only)

- Geothermal NEMO path: the inline `1.0e-3` is NEMO's mW to W conversion
  (`trabbc.F90:179`). It is now the module constant `_MW_TO_W`, with its
  provenance written next to it.
- MLE NEMO coefficient: the inline `3.141592653589793 / 180.0` is NEMO's `rad`
  (`phycst.F90:26`). It is now the module constant `_NEMO_RAD`.
- Lat-lon MLE NEMO branch: the code re-derived `5.0 / 21.0` (NEMO
  `tramle.F90:51`) even though `mle.py` already defines it. It now imports that
  constant, made public as `R5_21`. One test assertion follows the rename.
- MLE spec: the spec listed the boolean `nemo_association` under `excluded`.
  Only float fields can carry a spec, so the entry was removed. No float field
  changed tier or bounds.

No ratchet was loosened, no baseline entry was added, and no escape comment was
used. No configuration value or default changed.

## Ratchet suite, after

Fix tree: 4 failed, 12279 passed. The four failures are exactly the
main/fork-point rows above. Every lane-introduced item is green.

## Bit-identity (parent `072e4aa1b1c8` vs candidate `11943cc1a076`)

The shipped rung-10 card selects both touched NEMO paths: L8 geothermal and
L5 MLE with `nemo_association=True`.

- **D122 ladders, all seven card/label pairs, CONFIRMED.** These are rung 0,
  OMT-1..4, and rung 10 under both the independent and given-NEMO-entry labels.
  They were run with round 330's measurement recipe (round-329 gate `--measure
  recordice`, same deck roots), on clean worktrees at the two commits. On every
  pair the ladder row objects are byte-identical between parent and candidate,
  and no other key differs apart from the worktree stamp. Rows: rung 0 200,
  OMT-1 160, OMT-2..4 200 each, rung 10 200 per label.
  The registered D122 classifier refuses same-arm pairs ("control is not
  admitted"). Byte-identical rows make every one of its counts zero anyway.
- **DINO faithful grid, three production steps, CONFIRMED.** The round-330
  probe in `values` mode gives 66/66 leaf-step SHA-256 equal on both trees, and
  the whole file is byte-identical (`5f977ac14552…`). It also equals round 332
  resume 9's hashes.
- **DINO from-rest month gate (GPU), CONFIRMED.** Day-30 wet 3-D T rms is
  2.056821683e-03 K on both trees (bar 2.244317642e-03 K), PASS.
- **Touched routines on fixed inputs, CONFIRMED.** The routines are the MLE
  coefficient (both branches, three latitudes), the MLE vertical structure, and
  the NEMO geothermal tendency. Output hashes and float hex values are
  identical on both trees.

Other callers of the touched code are the MPAS MLE and the generic lat-lon MLE.
They reach it only through the renamed `R5_21`, which has the same Python float
(`0x1.e79e79e79e79ep-3`), and the routine-level check covers that.

Evidence: `orca2_rounds/round334/` (`d122/`, `dino/`, `unit/`, `ratchet_*.log`).

## Gates

- Ratchet suite: lane items green (above).
- Default gate battery (land.sh set) on the candidate: green. The fast files
  passed in one run. The three slow files had 12 passed under `-n 12`, because
  they exceed 10 minutes serially.
- Focused MLE/geothermal unit tests: 114 passed, 5 failed. The same 5 fail on
  the parent: two from SPEC_MODULES registry drift, and three because ORCA1
  geothermal input files are missing from the archive. These are pre-existing
  and out of scope.
- Per-receipt citation gate: PASS, 3 citations, 0 failures, 0 unmapped (three NEMO
  entries added to `CITATION_MAP`); default citation-gate test 17 passed.

## CHOICES

- Removing `nemo_association` from the MLE spec instead of adding a float spec
  for it — ASKED (task text; the rule says bools are not spec-eligible; the task orders
  honest specs).
- Making `_R5_21` public instead of copying it — ASKED (task text: no
  re-derivation; repo rule: promote, never import privately).
- Not fixing the inherited slab-land dispatch baseline — ASKED (task text says
  fix only lane-introduced items).
- Running the DINO month gate myself on idle GPUs 0/1 — UNASKED. The task named
  the month gate, but the standing note says CPU only. Offered for re-run by
  the operator.
- Running the three slow default-gate files with `-n 12` — UNASKED (to fit the
  10-minute foreground limit; it does not change the results).
- Starting the ladders with nohup, recording their pids, and waiting on them in
  the foreground — UNASKED (each rung-10 arm takes about 14 minutes, longer than
  the foreground limit).

## OPEN

- Main-side failures (jra55 literal, land restart coefficients, shortwave spec,
  private imports) belong to main, not this lane.
