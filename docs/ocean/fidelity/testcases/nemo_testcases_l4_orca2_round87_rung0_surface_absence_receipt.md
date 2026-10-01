# ORCA2 round 87 — rung-0 surface ABSENT repair

**Verdict: STOPPED_FOR_RECORD.**  The round-86 SIGSEGV is source-resolved and
the additions-only recorder repair, self-describing checker, plants, and new
optimized acquisition launcher are committed.  The repaired NEMO acquisition
has not been run, so no frame, restart-identity, ladder, or physics claim is
made.  Every result in this receipt is labelled **independent**.

Base: `448fd3323e1f4788dbf610a927bbcb099709eb32`.
Preregistration: `PREREG_nemo_testcases_l4_orca2_round87.md`, committed before
the source census or repair was measured.  This round changes no `packages/`
file, ORCA2 card, hierarchy switch, carried state, NEMO physics statement,
threshold, or sea-ice selector.  The GYRE trajectory is therefore unchanged by
construction; no GYRE result is claimed from an unrun gate.

## Independent diagnostic: the crash and its owner

The round-86 debug-only run reproduced the SIGSEGV before a model STOP.  Its
symbolized backtrace resolves to the canonicalizer in
`ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:443-473`, called by the
runoff writes in
`ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:424-438`, which are
entered from the step program at
`ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:149-154`.  The log is
`round86/acquisition/orca2_rung0_frame_debug_resume_np2/run.user.stdout.log`.
Those line numbers point into the inherited surface-input writer, not the new
round-84 entry/stage frame writer.  Debug output is diagnostic only.

The compiled rung-0 source gives the exact allocation boundary:

| Recorded fields | Owner and resolved switch | Compiled statement | Rung-0 representation |
|---|---|---|---|
| `rnf`, `rnf_b` | runoff, `ln_rnf=.false.` | the only allocation is guarded by `IF(ln_rnf)` in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_oce.f90:195-216` | ABSENT |
| `rnf_tsc`, `rnf_tsc_b` | runoff, `ln_rnf=.false.` | the runoff allocator owns both arrays in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:146-152` | ABSENT |
| `utau_icb`, `vtau_icb` | icebergs, `ln_icebergs=.false.` | initialization returns before their allocation in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbini.f90:133-140` | ABSENT |
| `berg_calving`, `berg_calv_hflx`, `berg_float_melt`, `berg_stored_heat` | icebergs, `ln_icebergs=.false.` | the skipped `icb_alloc` owns the gridded payloads in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icb_oce.f90:161-174` | ABSENT |
| `snwice_mass`, `snwice_mass_b`, `snwice_fmass`, `rCdU_ice` | no live SI3, `nn_ice=0` | the no-ice branch nevertheless calls `sbc_ice_alloc` in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:376-383`, which allocates these arrays in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_ice.f90:91-121` | PRESENT |
| the other 21 named surface operands | surface core | the core arrays are allocated independently of runoff in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_oce.f90:195-216`; `fr_i` is set to zero for no ice in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:274-280` | PRESENT |

Thus the defect was instrumentation dereferencing storage that the resolved
deck intentionally never allocates.  Writing plausible zero arrays would
erase that semantic distinction, so the repair records an explicit header-only
ABSENT entry.

## Independent implementation and mechanical controls

The committed patch composes at fuzz zero with the round-84 frame patch and
removes no NEMO source line.  It emits magic `NEMO_L4_SBCIN_2`; each of the 35
ordered fields has a 16-byte name and four header integers.  A present field's
payload length is derived from its header dimensions.  An absent field has
`(ndim,n1,n2,n3)=(0,0,0,0)` and no payload.  The writer contains exactly ten
explicit ABSENT calls and four inherited entry/stage frame calls.

The checker reads the self-describing header rather than predicting byte
counts.  It accepts ABSENT only when both `namelist_cfg` and `ocean.output`
resolve the corresponding owner off.  It refuses trailing bytes and plants for
bad magic, changed field name, truncation, a non-finite payload, replacing an
ABSENT entry with a plausible zero payload, and turning an absent field's owner
on.  Offline synthetic records prove all six plants fire and the unplanted
25-present/10-absent record passes.

`run.sh --preflight-only` passed from a clean committed tree after applying both
patches and compiling both Fortran units with the record build's include tree.
It printed:

```text
ORCA2_ROUND87_RUNG0_FRAMES_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round87/acquisition/orca2_rung0_entry_stage_fixed_10step_np2
ROUND87_WORKTREE_CLEAN_AFTER_PREFLIGHT
```

The launcher creates the new target `ORCA2_OMIP_L4_R87FRAMES`, runs the pinned
round-83 rung-0 deck, requires 80 frame shards and 20 terminal restarts, runs
every old and new plant, and compares each restart byte-for-byte with the
admitted round-83 record.  It contains no `/usr/bin/time` use and every input
artifact it reads is committed and repository-relative.

## Frozen prediction ledger

| Prediction | Verdict | Evidence |
|---|---|---|
| 35 fields = 25 PRESENT + the preregistered 10 ABSENT | UNMEASURED for the NEMO record; schema CONFIRMED offline | exact writer-call census and synthetic parser test; acquisition not run |
| ABSENT is conditional on two resolved owner records; all-zero and owner-on plants refuse | CONFIRMED offline | six planted violations fire; unplanted synthetic record passes |
| optimized run emits 80 frames and 20 restarts | UNMEASURED | operator acquisition required |
| all 20 terminal restarts are byte-identical to round 83 | UNMEASURED | admission is fail-closed but has no target record yet |
| round-84 parser reaches EOF and every inherited plant fires on all frames | UNMEASURED | acquisition required |

No prediction is silently promoted from synthetic control evidence to a NEMO
record claim.

## Validation and review

- Focused round-84/87 acquisition and clean-worktree tests: **6 passed**.
- The one permitted `tests/ocean/fidelity -n 12` battery selected 2,212 tests
  and reached 98% before a no-output stall was interrupted, so it has no suite
  PASS claim.  Two failures were visible.  The round-87 preflight initially
  leaked `stprk3.mod` into the worktree; the compiler output was redirected to
  its temporary directory and the exact clean-worktree test then passed in
  isolation.  The remaining isolated failure is the listed pre-existing SI3
  scalar-math provenance red (`test_full_v2_gate_and_plants`: `A MY_SRC is not
  verbatim`).
- The citation gate passes this receipt and the default cumulative receipt with
  no unmapped citations.  Its planted shift of the runoff allocation citation
  fails as required.
- Review verdict: **independent review unavailable in-sandbox**.  The separate
  read-only `codex exec` failed before reading the diff with `failed to
  initialize in-process app-server client: Read-only file system (os error
  30)`.

## OPEN

1. The operator must run the committed launcher below.  This is the only
   acquisition request; do not use the round-86 debug build as evidence.
2. Admission must prove 80 self-describing entry/stage frames, the exact
   25-present/10-absent surface census on the resolved rung-0 deck, all plants,
   and 20/20 byte-identical terminal restarts against round 83.
3. Only after admission: name the true rung-0 step-entry state, walk the first
   non-bit boundary in compiled stage order, then build and score the rung-0
   card's given-entry ladder, independent ladder, and independent month.

Acquisition launcher:
`/tmp/autopilot-orca2-2086828217/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round87_rung0_frames/run.sh`.
Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round87/acquisition/orca2_rung0_entry_stage_fixed_10step_np2`.

## UNVERIFIED

- The optimized NEMO target has not been built or run in this round.
- The surface record's actual payloads and 25/10 census are not measured.
- The 80 frame shards, 20 terminal restarts, restart identity, rung-0 first
  non-bit statement, ladders, and month score remain unmeasured.
