# NEMO testcases L2 GYRE round 79 — absolute barotropic history receipt

Date: 2026-09-13

Starting commit: 7590a8eb218387b6a940fa0d851fa4a7485423de

Writable clone: /tmp/autopilot-work-Jyrwcr5f

Evidence root: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round79

## Verdict

**HOLD.** The physical candidate is strongly supported and improves every
registered one-to-thirty-day RMS field on every measured day, but this round
cannot issue SHIP for two mechanical reasons:

1. the mandatory separate `codex exec --sandbox read-only` review never
   reached a model because this environment's in-process app-server attempted
   a forbidden write; there is therefore no independent reviewer verdict to
   quote; and
2. the exact immediately-pre-change GYRE kt=1..10 ladder was not captured
   before the implementation commit.  The after ladder is complete, the
   source-order before arm is the admitted round-78 record, and the recorded
   year before arm is complete, but a post-hoc toggle is expressly forbidden
   and cannot substitute for the missing parent ladder.

No configuration choice was made.  Decision 37 was already authorized YES by
the user.  No NEMO source, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held manifest patch was modified.

## Frozen preregistration and commits

- Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round79.md,
  committed before measurement as 0078f9cc851c921176afd8e30660373129321712.
- Candidate: c47f9c33dce7ebcdec96dcb3f6269bf7145bdd58.
- Gate-only correction: add5cbd555b99ebe32ce4e46ae69538f67951883.
  The first gate run retained the correct measured rows but mislabeled them
  because it expected the inherited round-66 status CONFIRMED rather than its
  established value MEASURED.  The correction changed only that acceptance
  literal; the clean-commit rerun passed.
- Citation-map retraction: e290dfa42073790047b3f74c87ee64537f64235a.
  The first citation run correctly failed because five cumulative map entries
  still named the removed deviation representation or line-shifted neighbors;
  the obsolete claims were removed from the tool before the final audit.

## What changed

The one shared lat-lon C-grid implementation now persists and consumes NEMO's
six absolute AB3/AM4 histories in this order: ub_e, ubb_e, vb_e, vbb_e,
sshb_e, sshbb_e.  The previous representation persisted final-minus-history
deviations and reconstructed each array by subtraction at the next window,
which was algebraically equivalent but changed bits.

Continuation with bt_hist now also requires the paired prognostic uu_b/vv_b
boundary state.  That pair and the post-vertical-solve reconciliation owner
were already in the shared production path, so this is the authorized paired
Decision-37 landing, not the refuted round-51 raw-history-only arm.

Run-restart format 4 writes the absolute representation.  Format 3 remains
readable only through an explicit one-time migration against persisted uu_b,
vv_b and eta.  A format-3 archive with history but without those anchors fails
with the stable phrase "cannot migrate deviation-form bt_hist".  Format 2
continues to fail before payload reconstruction.

The preimplementation search found no second production AB3/AM4 history owner:
state declaration, restart serialization, the single shared barotropic
implementation, unit tests, and the held round-51 manifest were the complete
set.  The held manifest was not changed.

## First non-bit statement

Round 78 entered this round with the first U non-bit row at kt=2, external
substep 1, ubb_e: 6/580 wet cells, maximum absolute difference
8.470329472543003e-22.  The associated midpoint ua_e differed in 2/580 cells,
maximum 2.117582368135751e-22.

The clean round-79 gate makes every registered substep-1 row bit-identical:
all three coefficients, un_e, ub_e, ubb_e, and ua_e.  The first non-bit row is
now external substep 2 un_e: 580/580 wet cells, maximum absolute difference
3.032539284029834e-09.  Thus the first remaining statement lies after the
now-exact substep-1 midpoint and before/at the compiled swap into the next
substep current value.  The next walk must instrument the remaining
substep-1 pressure/tendency/update sequence through that swap; it must not
return to the now-exact history association.

Evidence:

- round79_raw_history_final.json, SHA-256
  637f95e876c0f1b21618cd12f0233f324017b7d1b73af575c07da350372f117f.
- round79_raw_history_plant.json, SHA-256
  702256cc3122859e317f8a3f58498851551d482272fb0cb7d62b5e4badc6fe62.
- The one-ULP ubb_e plant exited 1 and printed exactly
  `ROUND79 HISTORY PLANT FIRED`.

## Rule-12 table

| Lane | Registered movement | Result |
|---|---:|---|
| GYRE source-order kt=2 substeps | substep-1 ubb_e and ua_e; later rows allowed downstream | **CONFIRMED.** Both promoted rows became bit-exact. First non-bit moved to substep-2 un_e. |
| GYRE kt=1..10 | kt1 fixed; first possible whole-step movement kt3; kt3..10 all fields registered | **AFTER MEASURED / parent comparison HOLD.** kt1 remains exact/uninformative; kt2 T and S remain AT-BAR, SSH remains exact AT-BAR, and first-over-bar remains kt2 U/V. The after artifact is complete, but the exact parent artifact was not captured. |
| GYRE days 1..30 | every day after day 1 registered | **IMPROVED.** Against the recorded before arm, all five RMS fields improve on all 30 days. Day-1 T is 2.6994481725e-3 to 3.4102733936e-4 K (ratio 0.12633); day-30 T is 1.4241019262e-2 to 1.2397568272e-2 K (ratio 0.87055). Day-30 U/V ratios are 0.88932/0.67601. No year row was at the 1e-15 bar before. |
| LOCK_EXCHANGE kt=1..10 | kt1 fixed; kt2+ allowed | **PASS.** Fifty rows compared; no status change; first-over-bar remains kt4 U; maximum worsening is 4.76837158203125e-7 row-scale ULP. |
| OVERFLOW kt=1..10 | predicted bit-identical because boxcar reinitializes | **STATEMENT NOT EXECUTED.** The registered card uses nemo_boxcar1_ab3, so the changed cross-window nemo_ab3am4 continuation branch is unreachable. Its after run retains first-over-bar kt2 T/U. |
| DINO | explicit shared-statement risk | **STATEMENT NOT EXECUTED / restart risk bounded.** DINO uses nemo_boxcar_ab3 with its separate leapfrog carry, not bt_hist continuation. A v3 DINO archive with bt_hist absent passes through unchanged; the repository-wide format bump remains a compatibility surface covered by migration/failure tests. No DINO numerical claim is made. |
| ORCA2 | required spec | **UNMEASURED WITH SPEC.** Input: independently produced ORCA2 NEMO restart plus legoESM NEMO-identity state at the same instantaneous Kbb frame. Fields: T, S, u, v, SSH and the six absolute barotropic histories. Masks: NEMO tmask/umask/vmask at native staggering. Statistic: elementwise float64 bit equality and normalized L-infinity per row. Bar: 1e-15. Falsifier: any pre-existing AT-BAR row becomes DEBT, first-over-bar becomes earlier, or any persisted history differs on a wet face. |

GYRE after-ladder SHA-256:
330b71aed037131f0be38965e67dedd9489bd135d47771e669f477d7e914c728.
Recorded-before year score SHA-256:
11333cb9eeec57ce15a080b09e2864f991744e258fe8ac41a7abf5d9f41db96f.
Candidate year score SHA-256:
c68e0fac5de29183817c993f7cd1f9f73c07a2c25d3828c753afacf3afefc422.
LOCK comparison SHA-256:
4c69873bf45b3c89cecf94fa024fede0d3d54b7a8908e5fe4faea1b92224e7f2.
OVERFLOW after SHA-256:
d2c0a985c0beec500e2b575ea971753a8a6d4d80f8a1f2e947f0bafcd6b71c26.

## Tests

- `git diff --check`: pass.
- Python compilation of the changed modules and new gate: pass.
- `test_nemo_ab3am4_filter.py` plus `test_ocean_run_restart.py`:
  **87 passed in 116.88 s** under CPU/fp64.
- Restart-focused new/adjacent controls: **4 passed in 2.51 s**.
- Receipt citation gate: **PASS**, 4/4 receipt citations mapped, zero map-audit
  failures.  Shifting the compiled initialization citation by two lines exited
  1 with SYMBOL-NOT-AT-LINE.
- Known unrelated red test was not encountered in the focused suites.

## Separate Codex review

Three required read-only invocations were attempted: ordinary, ephemeral, and
ephemeral with apps/plugins/hooks/browser/image features disabled.  All failed
before model startup with the same terminal line, quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore there is no reviewer verdict.  This is a process blocker and one
reason for HOLD.  The next round must rerun the exact adversarial prompt in an
environment where read-only Codex can initialize, and must quote its verdict.

## REFUTED / retained findings

- **Round-80 correction (2026-09-13):** the preregistered whole-step kt=2 U/V
  movement is **REFUTED**.  The recovered exact-parent comparison found zero
  movement in all 954 registered rows, so the Rule-12 table's phrase "first
  possible whole-step movement kt3" is withdrawn rather than relabeled.  The
  candidate remains HOLD; the source-order substep-2 U observation remains a
  distinct direct measurement.

- Round 51's raw-history-only arm remains refuted and held.  It predates the
  separately prognostic uu_b/vv_b plus stage reconciliation owner and is not
  evidence against this paired candidate.
- The first round-79 gate report labeled the otherwise correct prediction
  REFUTED solely because of the inherited-status literal described above.  It
  remains in the evidence directory as round79_raw_history.json; it was not
  overwritten or cited as the final verdict.
- The preregistered expectation that the first remaining U row would be
  substep-2 un_e is confirmed.  No direction was preregistered for whole-step
  maxima, and no post-hoc direction claim is made.

## OPEN — exact handoff to round 80

Acquisition script: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round79/run.sh.
It performs no checkout; it requires a clean clone already at the frozen parent
commit, captures that ladder, and reruns the read-only Codex review.

1. **Unblock independent review first.** Run the mandated separate Codex
   read-only adversarial review on commits 0078f9cc, c47f9c33 and add5cbd5.
   DO NOT SHIP if it says DO NOT SHIP; disposition every finding explicitly.
2. **Close the missing parent ladder comparison.** The discriminating
   measurement is the same fixed GYRE kt=1..10 phase-3 gate at the exact
   preregistration commit 0078f9cc, compared mechanically with
   round79_gyre_after.json.  This round was forbidden to check out another
   commit and did not fabricate a toggle.  Require every registered movement,
   no AT-BAR loss, and no earlier first-over-bar.
3. If both blockers pass, promote the paired Decision-37 candidate.  Preserve
   restart format 4, the tested v3 migration, and the loud missing-anchor
   failure.  Do not resurrect the round-51 raw-only manifest.
4. The next magnitude-ranked source walk starts at kt=2 external substep 1
   immediately after the now-exact midpoint and ends at the swap that creates
   substep-2 un_e.  Pre-register and record pressure-gradient operands,
   Coriolis/drag/forcing trends, the actual updated ua_e, and the swap output;
   name the first non-bit statement in compiled order.
5. Keep DINO as explicit non-execution/restart-format risk and ORCA2 as the
   full UNMEASURED-with-spec lane until independent trajectories exist.

## Compiled source citations

The running instrumented GYRE compiled branch initializes and seeds the six
absolute external-mode arrays directly from persistent state; only the
accumulators are zeroed: `GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`.

Its midpoint consumes current, b and bb absolute velocities in the written
association; there is no deviation reconstruction:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509`.

At each external substep it rotates absolute b/bb/current velocity and SSH
arrays, including the assignment that creates the next substep current value:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:783-795`.

NEMO restart I/O reads and writes those absolute arrays by name, which is the
compiled-source basis for run-restart format 4 and the explicit format-3
migration: `GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:991-1018`.
