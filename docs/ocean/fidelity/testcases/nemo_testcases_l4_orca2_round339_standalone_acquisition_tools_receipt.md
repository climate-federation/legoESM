# ORCA2 round 339 — standalone ice-cover and K0 acquisition tools

Date: 2026-10-26. Base `94ec2462614a`; preregistration `735499281`.
Disposition: **ACQUISITION_NEEDED**. This round produced and preflighted two
operator-only tools. It did not build or run NEMO and therefore produced no
ice-cover record, K0 smoke result, or physics score.

## Source basis and frozen experiment

Tool A observes the admitted round-5 full-SI3 program. The call it follows is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:501`, and the
observed value has just been assigned by
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceupdate.f90:212`.
The observer is additions-only, immediately after `ice_stp`, outside every
NEMO loop. It records the declared local `fr_i(jpi,jpj)` on both ranks at the
1,460 odd forcing timesteps from 1 through 2919.

The offline producer stitches each rank's owned native region, proves exact
one-time coverage of the 180 by 148 grid, and accumulates binary64 values in
increasing `kt` order with one final division by 1,460. It writes exactly one
binary64 `ice_cover(time_counter,y,x)` record, including native land and halo
content needed by the reader. The requested NEMO record is declared at
`cfgs/SHARED/namelist_ref:418`.

Tool B rebuilds the round-5 deck with `key_si3`, NEMO's unmodified source and
an empty `MY_SRC`, and stages
only the Decision-175 K0 selectors: `nn_ice=1`, `ln_drgice_imp=F`,
`ln_mxl0=T`, `nn_mxlice=0`, `ln_icebergs=F`, with `sn_ice` naming Tool A's
digest-pinned file. It runs ten steps only after Tool A admission. Its result
is explicitly a boot/smoke result and never scores S4 physics.

## Delivered operator tools

- Tool A:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_ice_cover_producer/run.sh`.
  Default and `--preflight-only` are read-only. `--run` performs the retained
  build, a ten-step observer passivity twin, the one-year production run and
  offline admission. `--admit-existing` rechecks a retained in-place record
  against its producer commit, binary and recursive relative-path manifest.
- Tool B:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_k0_smoke/run.sh`.
  Default and `--preflight-only` are read-only. `--run` refuses until Tool A's
  manifest and climatology verify, then builds and executes the ten-step K0
  smoke. `--admit-existing` pins the retained producer and binary manifests.

Both scripts refuse a dirty repository, check at least 150 GB available memory
immediately before any NEMO launch, use a foreground timeout, and retain failed
runs and builds under the round-339 evidence tree. The documented retry is to
move a retained build to a unique retained name and start again; it is never
deleted. Tool A's estimate is about 55 minutes of production stepping plus a
short ten-step passivity run and build time; the operator should reserve 90
minutes plus build. Tool B reserves a 15-minute smoke timeout plus build.

## Post-review repair

The operator's Claude review returned RUN; Argon returned HOLD on acquisition
source isolation, the iceberg banner, plant specificity, and manifest timing.
The resumed implementation now builds Tool A from a separate observer-free
same-key baseline and keeps only `sbcmod.F90` plus its `fr_i` writer in the
observed `MY_SRC`. Tool B's `MY_SRC` is empty, so no SI3 field can be touched by
a configuration-local observer. Both manifests are written before the first
NEMO launch and digest the complete generated ppsrc and `MY_SRC` snapshot.

## Acquisition checklist disposition

1. **Passivity census:** Tool A compiles the actual ppsrc at the round-5
   `-O3 -funroll-all-loops -fno-tree-vectorize` flags. The parsed nonzero
   floating-point count for `sbc` is 355 in both arms. An injected add changes
   it to 356 and refuses. The observer-free admitted ten-step run is also
   compared byte-for-byte with the instrumented run's four terminal restarts;
   the restart-byte plant refuses.
2. **Declared extents:** the writer receives `fr_i(:,:)` from the actual
   declared `jpi,jpj` object and records those dimensions in every header.
   Admission reads exactly those extents and extracts only the declared owned
   range.
3. **Filled range and signed zero:** SI3 has filled `fr_i` through its halo
   assignment before the call. The raw full local field is preserved; only
   owned cells enter the native-grid mean. The bounds fixture initializes the
   array to `-0.0` and proves that bit survives.
4. **Exact expression/macros:** this observer copies `fr_i` and evaluates no
   replacement physics expression or macro. Nothing is simplified or replayed.
5. **Bounds fixture:** `-fcheck=bounds` compiles the real patched and
   preprocessed `sbcmod` source plus the exact writer. Dead-section linking
   lets two rank-distinct fixtures call the wrapper in that compiled NEMO unit;
   both execute and retain `-0.0`.
6. **Time level:** not applicable: `fr_i` has no selected time level here. The
   header instead pins and validates the actual `kt`, rank and forcing stride.
7. **Placement:** preflight reproduces the admitted real ppsrc, checks line
   order around `ice_stp` and the iceberg block, and fires independent removed
   call, removed import and removed wrapper plants. The production build is
   checked again from its generated ppsrc.
8. **Corruption controls:** checksum, truncation, header, rank, nonfinite and
   one-ULP controls refuse. Value-changing controls select a positive
   ice-covered wet cell when available; every real value must be finite and in
   `[0,1]`.
9. **Admission and retention:** the producer manifest is written before the
   first NEMO launch, so every refusal already has its producer commit, binary,
   staged inputs and complete build-source snapshot. The short passivity run requires `STOP 0`, no
   `E R R O R`, and byte-identical restarts against the admitted observer-free
   record. Both tools retain relative-path SHA-256 manifests, executables,
   inputs, resolved namelists, logs, restarts, outputs and admission reports.
   Existing-run admission verifies the recorded producer commit and binary,
   including retained paths inside the refusal trap.
10. **Score scope:** Tool A checks all 2,920 rank-record pairs and every native
    output cell. Tool B checks every numeric variable in each ocean restart.
    Neither claims or computes an S4 physics score.
11. **Like-with-like proof:** the reused round-328 helper compares Tool A's real
    preprocessed source and objects after stripping only the declared observer
    and writer additions against its separately built observer-free baseline.
    A changed unrelated `iceupdate` ppsrc plant refuses. Tool B uses the
    observer-free variant against that same clean baseline.
12. **Namelists and errors:** the K0 preflight validates every staged name in
    its exact `namelist_ref` group. Moved-group and misspelled-name plants
    refuse. Resolved `ocean.output` selectors are parsed with whitespace
    tolerance; `ln_icebergs=.false.` is checked through NEMO's real
    `==>>>   No icebergs used` banner, and either log containing `E R R O R`
    refuses. A committed excerpt from the admitted round-5 `ocean.output`
    exercises that banner.
13. **Rerun after refusal:** both READMEs and traps retain builds and failed
    runs, name the move-aside retry procedure, and provide manifest-pinned
    `--admit-existing` paths. Each preflight moves a synthetic manifested run
    under `refused/` and admits it through the launcher's actual
    `--admit-existing` route. No cleanup path deletes acquired evidence.

## Preflight result

Tool A preflight: **PASS**. Placement-call, placement-import,
placement-wrapper, arithmetic-census, whole-build same-program, restart-byte,
memory and disk plants all fired. The retained-run admission route passed. The
real patched-source bounds fixture ran on two rank-distinct inputs. Focused
Python tests passed 9 of 9. Final status:
`ORCA2_ROUND339_ICE_COVER_PREFLIGHT_READY`.

Tool B preflight: **PASS**. Three wrong-group plants, the misspelled-name
plant, staged-filename plant, empty-`MY_SRC` plant and memory plant all fired.
The retained-run admission route passed. Focused Python tests passed 9 of 9.
Final status: `ORCA2_ROUND339_K0_PREFLIGHT_READY`.

No NEMO executable was built or run by the author. The next action is strictly
Tool A `--run`, followed only after its admission by Tool B `--run`.

## Choices and review

- ASKED: Decision 171 native-grid yearly mean, Decision 181 one 365-day
  round-5 year and all 1,460 SBC samples, Decision 175 K0 selectors, ten-step
  smoke, no NEMO run by Codex, reuse of the round-5/333/336 helpers, and an
  operator-only acquisition stop.
- UNASKED: none.
- independent review: operator Claude RUN; Argon HOLD findings repaired and
  re-preflighted

## Verification

- Per-receipt citation gate: **PASS**, 3 citations, zero failures, zero
  unmapped citations, zero map-audit failures, all 9 self-tests fired.
- Campaign-default citation gate: **PASS**, 274 citations, zero failures,
  zero unmapped citations, zero map-audit failures, all 9 self-tests fired.
- Default ORCA2 landing inventory, including the required
  `test_nemo_testcase_receipt_citation_gate.py`: **152 / 152 PASS** in
  1,469.76 seconds.
- Required standalone citation-gate pytest on the clean pre-final tree:
  **17 / 17 PASS**.
- Focused round-339 tests: **9 / 9 PASS**.
- Dedicated three-invocation ratchet: **12,314 PASS, 4 SKIP, 3 inherited
  FAIL**. The failures are the unchanged JRA55 test literal, land restart
  coefficient and shortwave-penetration parameter spec; none names a file in
  this round's diff.

NEMO acquisition is deliberately absent.
