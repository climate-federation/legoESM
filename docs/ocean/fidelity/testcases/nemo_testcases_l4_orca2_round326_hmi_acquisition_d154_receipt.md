# ORCA2 round 326 — no record carries SI3's ice thickness for TKE's under-ice mixing length (DECISION 154): acquisition written

Date: 2026-10-23 (Claude round; codex review: operator).

- Resume gates (after the codex-HOLD fixes): citation-gate test +
  private-import ratchet + round-326 gate test 45 passed
  (`default_gate2_1.log`); TKE terms + recipe 96 passed; freshwater + stage
  face mask 18 passed; prognostic barotropic 8 passed = 167 passed, 0
  failed. Receipt and preregistration citation gates: PASS, 0 failures.

Status: **STOPPED — ACQUISITION_NEEDED** (acquisition committed, preflight green).

## Verdict

- No rung-10 record carries hm_i, so the comparison cannot run nn_mxlice = 2
  yet. This round commits the acquisition and stops (prompt step 1).
- No model code, card or comparison harness changed; steps 2-5 (harness
  selection, refusal plant, D122, gradients) wait for the record.
- Rung 10's kt-10 numbers are therefore unchanged from round 325 (drag on,
  given entry, full domain: max sea level 0.00706750 m, max T 0.19674038 K,
  RMS T 1.10230e-3 K, max u/v 0.0472/0.0516 m/s; independent: 0.02256209 m,
  0.19673128 K, 1.10251e-3 K, 0.0472/0.0516 m/s). No "after" exists.

## What NEMO does (compiled ORCA2 build; the recorded R5FULLENTRY build has the same statements)

- Under ice, with ln_mxl0 = T and nn_mxlice = 2, the surface mixing length is
  ((1 - fr_i) zraug taum + 2 fr_i hm_i) tmask:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:621-622`. Waves are off
  (ln_wave = F), so this branch runs.
- hm_i is assigned in one statement, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:186`
  (ice_var_agg(2)), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:260`
  on ice steps only (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186`;
  nn_fsbc = 2: kt 1, 3, 5, 7, 9).
- sbc, which runs the ice step, precedes zdf_phy every step
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:151`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`). So TKE at kt reads the
  hm_i of the latest odd step <= kt.
- The record's ocean.output states nn_mxlice = 2, ln_mxl0 = T, nn_mxl = 3,
  nn_fsbc = 2.

## What the records hold

- Rung-10 surface frames (round 5, 20 frames): fr_i, taum, rCdU_ice; no hm_i.
- Phase-2u TKE walk: hm_i at kt 2, rank 0 only, from a different run
  (`parallel/orca2/oracle_phase2v_tke_boundary_np2`).
- SI3 frames in the rung-10 run (Prather, zdf inputs, thd): odd steps,
  mid-ice-step or compressed ice columns; none is hm_i after ice_var_agg(2).
  Rebuilding hm_i from them would be a proxy; not done.

## Acquisition (`1abfce93d`, `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round326_hmi_acquisition/`)

- Re-runs round 5's ten-step record with one write-only hook in zdf_tke just
  before tke_avn: hm_i, fr_i, taum, surface mask, inner domain, kt 1..10,
  both ranks (`oracle_tke_hmi_kt*_r*.bin`). Patch is additions only.
- Admission: four restarts and 40 surface / step-entry frames byte-identical
  to round 5; five gate plants fire; the gate then reports fr_i/taum against
  round 5's surface frames and the even-step hm_i identity.
- Preflight (`--preflight-only`, run here, rc 0, `gates/preflight.log`): all
  seven checklist items, each with a plant that fired (continuation insert,
  non-reference config, nn_itend 9, nn_mxlice 0 in a real ocean.output copy,
  disk, existing target); gfortran syntax proof of the patched zdftke and the
  writer; whitespace-tolerant echo checks on the real ocean.output; no
  `grep -q` in pipelines.
- Gate test (writer's byte layout from its Fortran source, five plants,
  missing frame, cross-check on the real round-5 frames incl. a 1-point
  perturbation found): 9 passed.
- Predictions A1-A4 frozen in `PREREG_nemo_testcases_l4_orca2_round326.md`.

## Resume: operator review before any NEMO run (Argon RUN, codex HOLD)

All seven codex findings verified as real, then fixed; NEMO still not run.

- Same program: the build stays fresh (round 5's tree is not reused or
  touched), then refused unless arch, cpp keys and bld.cfg match round 5's
  (name substituted) and all 412 other preprocessed sources and 424 other
  objects are byte-identical; zdftke may differ only by the patch's two
  inserted lines, plus the new writer. Preflight shows today's cpp and
  compiler reproduce round 5's zdftke.f90 and zdftke.o byte for byte; a -O2
  compile and a planted icevar.f90 are refused. Unit test: seven planted
  differences refused.
- Admission now refuses on A2 (fr_i/taum), A3 (even-step hm_i) and A4
  (all-zero or >= 20 m hm_i), each with a plant; nine gate plants in all.
  Prereg amendment records it (predictions unchanged).
- Log checks run on a committed verbatim excerpt of round 5's ocean.output,
  pinned by that file's SHA-256; the new run's own echo is checked too.
- EXIT trap removes only this invocation's scratch (beside the evidence,
  never /tmp) and, on a failed `--run`, the target config and run directory
  it created. The three /tmp directories from the first preflight removed.
- makenemo's edit of the shared cfgs/work_cfgs.txt is stated in the README;
  the run refuses if any line but the target's changed.
- Refuses below 150 GB available memory (preflight and just before mpirun);
  mpirun timeout 900 s (round 5: 13 s).
- No-argument invocation is the read-only preflight; `--run` is explicit.
- Removing the A2, A3 or object-comparison refusal makes its test fail
  (shown, then restored).
- Preflight re-run: rc 0, 11 plants fired, PASS_SAME_PROGRAM on the dry
  tree, no scratch left (`gates/preflight2.log`). Gate test 22 passed.

## Resume 2: operator `--run` refused at the same-program gate (no NEMO run)

- The build finished; all 412 other preprocessed sources matched round 5;
  the gate then refused on the first object compared (ablmod) and the
  failed run removed the build.
- Cause: the config's absolute path is stored inside the objects, not a code
  difference. gfortran writes each source's full path into its runtime-error
  messages; 311 of round 5's 425 objects carry round 5's config name, and the
  new name is 4 characters shorter.
- Evidence (gfortran only, no makenemo, no NEMO;
  `object_path_probe.sh`, log `gates/object_path_probe.log`): today's
  compiler with round 5's flags reproduces all 425 round-5 objects byte for
  byte at round 5's path (so no timestamp, flag or compiler drift); under a
  renamed config root 311 differ, and in each only the merged-string
  section (path text plus its 8-byte re-padding) and the string-label
  offsets in the symbol table change. Code, data and relocations are
  byte-identical. Every compile line in the operator's build log used round
  5's flags.
- Gate change: objects are now compared section by section with the config
  root normalised (strings compared as strings, string-label symbols as the
  string they name, everything else byte-exact), and every differing file is
  listed. Under it 425 of 425 match; an -O2 build and a flipped code byte
  are refused.
- New test compiles a unit under two config roots: passes; a changed string
  literal, a different optimisation level and a flipped code byte refuse.
  Removing the path substitution, the string-sequence compare, the symbol
  resolution, or the whole section compare each makes it fail (shown, then
  restored).
- Not checked: the operator's actual new objects (deleted by the failed
  run's cleanup); the gate checks them on the next `--run`.
- Resume-2 gates: preflight rc 0, 11 plants fired, PASS_SAME_PROGRAM on the
  dry tree (`gates/preflight3.log`); citation-gate test + private-import
  ratchet + round-326 gate test 46 passed; TKE terms + recipe 96 passed;
  freshwater + stage face mask + ORCA2 parallel receipt 20 passed;
  prognostic barotropic 8 passed = 170 passed, 0 failed
  (`gates/default_gate3_*.log`). Receipt citation gate: 6 citations,
  0 failures (`gates/citation_gate_receipt3.log`).

## Round-314 audit row

- nn_mxlice stays DIFFERS (b:D135); reason now names DECISION 154's pending
  record. Counts unchanged (180/3 vs record). Audit test 9 passed.

## Choices made

- Stop at ACQUISITION_NEEDED instead of building steps 2-5 ahead of data —
  ASKED (prompt step 1).
- Record hm_i at TKE's call site (just before tke_avn) rather than at
  icevar.f90:186 — UNASKED, instrument choice (it is the value TKE reads);
  offered for revert.
- Also record fr_i, taum and the surface mask as cross-checks — UNASKED,
  instrument choice; offered for revert.
- fr_i/taum mismatch, even-step identity and A4 now refuse — ASKED
  (operator, resume item 2).
- Fresh build plus byte-level comparison, rather than reusing round 5's
  build tree — ASKED (resume item 1 offered either; reuse would write into
  round 5's evidence tree).
- mpirun timeout 900 s — UNASKED value (operator asked for a timeout);
  offered for change.
- Object comparison normalises only the config root (strings and
  string-label offsets), code bytes exact — ASKED (resume 2: "compare the
  code-bearing content ... after normalising embedded paths").
- Audit reason text names D154 — text only, no configuration.

## OPEN

- Operator re-runs `run.sh --run` (fresh build, about 6 min, then one
  round-79b-sized NEMO run); the next round does steps 2-5 on the admitted
  record.

## Gates (x64, CPU; logs `orca2_rounds/round326/gates/`)

- New gate test 9 passed; audit test 9 passed; preflight rc 0.
- Default gate (land.sh list), split for the 10-minute cap: citation-gate
  test + private-import ratchet 23 passed (`default_gate_1.log`); TKE terms +
  recipe 96 passed (`default_gate_2.log`); freshwater + stage face mask +
  ORCA2 parallel receipt 20 passed (`default_gate_3.log`); prognostic
  barotropic 8 passed (`default_gate_4.log`) = 147 passed, 0 failed.
- Receipt citation gate on this receipt: PASS, 6 citations, 0 failures
  (`citation_gate_receipt.log`); on the preregistration: PASS
  (`citation_gate_prereg.log`).

- Resume gates (after the codex-HOLD fixes): citation-gate test +
  private-import ratchet + round-326 gate test 45 passed
  (`default_gate2_1.log`); TKE terms + recipe 96 passed; freshwater + stage
  face mask 18 passed; prognostic barotropic 8 passed = 167 passed, 0
  failed. Receipt and preregistration citation gates: PASS, 0 failures.

codex review: operator
