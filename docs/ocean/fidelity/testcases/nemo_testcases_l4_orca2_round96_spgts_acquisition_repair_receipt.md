# ORCA2 round 96 — rung-0 SPG acquisition deck-staging repair

Date: 2026-10-02. Base `90abce744`; validated implementation tip
`2217a5e42`. Scope is ocean only. Every future payload from this acquisition
is labelled **independent**. This round produces no ocean-model number.

## Verdict

**STOPPED_FOR_RECORD.** The operator-run round-95 launcher did not reach the
instrument patch, custom rebuild, run-directory creation, or MPI launch. It
failed while applying the Decision-83 namelist patch to the wrong copy of the
deck. The repaired, committed launcher uses a new target name, explicitly
stages and pins the admitted round-93 rung-0 namelist before patching it, and
is preflight-clean. Both the writer-layout plant and a new stale-source-deck
plant fire. The operator must run the new target before the split-explicit
walk can continue.

No `packages/` file, selector, configuration choice, carried-state convention,
stabilizer, sea-ice field, threshold, or `unmeasured_features` entry changes.
The held round-94 slow-depth candidate is not present.

## Round-95 refusal diagnosis

The decisive operator log ends with:

```text
Compilation successful
2 out of 5 hunks FAILED -- saving rejects to file namelist_cfg.rej
REFUSE: round-95 SPG acquisition failed at line 210 (exit 1)
```

The launcher had correctly dry-run the patch against the admitted run's
`namelist_cfg`, SHA-256
`d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c`.
During target construction it instead copied
`ORCA2_OMIP_L4_R93SLOW/EXP00/namelist_cfg`, SHA-256
`c7d350404fff3eaf68673e052955cbad9decafe4007b2b2e415e7d058724f69d`,
then applied the run-deck patch to that stale shipped-deck copy. The two files
differ in the rung-0 module switches. The failed hunks were `ln_spc_dyn` and
the explicit vertical-mixing-selector removal; three earlier hunks had already
modified the target file before `patch` returned 1.

This failure happened in the initial target setup. The round-95 evidence
directory is empty, its `MY_SRC/dynspg_ts.F90` still has the pinned unpatched
source digest, and the compiled source contains no `r95_spg_open` call. There
is therefore no hidden record to admit and no model result to interpret.
R96-P1 is **CONFIRMED**.

The partly created `ORCA2_OMIP_L4_R95SPG` target is retained as failed evidence
and never reused or deleted.

## Repair and fail-closed controls

The same committed launcher now targets fresh config
`ORCA2_OMIP_L4_R96SPG` and evidence directory
`orca2_rounds/round96/acquisition/orca2_rung0_spgts_ranked_10step_np2`.
After cloning the source config it overwrites only the target's
`EXP00/namelist_cfg` with the admitted round-93 run file, requires the source
digest above, applies Decision 83 at fuzz zero, and requires the unchanged
harmonized digest
`5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8`.
The target run is still assembled from the complete admitted deck and input
manifests. Admission now also pins the Decision-83 patch itself to the producer
commit.

Preflight on clean commit `2217a5e42` reports:

```text
SYNTAX_PROOF_PASS l4_r95_spgts_frames.f90 dynspg_ts.f90
ORCA2_ROUND96_RUNG0_SPGTS_PREFLIGHT_READY .../round96/acquisition/orca2_rung0_spgts_ranked_10step_np2
STATUS PLANT-FIRED layout
STATUS PLANT-FIRED source-deck
PREFLIGHT_RC=0 LAYOUT_RC=69 SOURCE_DECK_RC=69
```

The source-deck plant deliberately substitutes the stale config-directory
namelist for the admitted deck and proves the source digest rejects it before
patch, build, or run. R96-P2 is **CONFIRMED** at preflight. R96-P3 remains
**UNMEASURED** until the operator-run target supplies all twenty restart
shards. R96-P4 is **CONFIRMED**: no scientific number is claimed from either
the failed target or preflight.

## Scientific boundary retained

The requested record still observes the compiled rung-0 program in source
order: forcing setup at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:287-320`, barotropic
initialization at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:339-381`, and the
substep loop at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:446`. Within each
substep the first two boundaries remain mid-step extrapolation and face depth
at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:460-519`, followed by
transport and after-SSH at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:530-557`.

Round 95's predictions about the first non-bit substep and the `e3f_0vor`
cross test remain frozen and unmeasured. This round repairs only the producer
provenance needed to obtain that record.

## Gates, tests, and review

The focused record/launcher plus citation-gate suite passes 25/25. The default
citation gate passes 274 citations and this receipt passes all five of its
compiled citations, with zero unmapped citations, failures, or map-audit
failures. All nine built-in non-vacuity controls fire; shifting the cited
mid-step range by two lines makes the receipt gate fail with exit 1.

The prescribed single `tests/ocean/fidelity -n 12` battery reached 99%. The
round-96 launcher regression test passed in that run. It displayed the known
worktree-stamp ratchet, recipe case-board ratchet, and SI3 scalar-math
provenance failures, then reproduced the established silent tail: every
`python -m pytest` process disappeared without a summary while the launcher
remained open. After confirming the process census was empty, only the
stranded launcher was interrupted. This receipt does not represent that wide
battery as PASS and does not assign an unseen failure.

The required separate `codex exec --sandbox read-only` review did not reach
the diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

The package tree is identical to base, so GYRE's certified trajectory is
unchanged by construction. No ORCA2 trajectory or observer-passivity claim is
made without the record.

## OPEN

1. Operator runs the committed launcher with `--run`; it uses the fresh
   `ORCA2_OMIP_L4_R96SPG` target and round-96 evidence directory.
2. Admit both self-describing rank streams and prove all twenty terminal
   restarts byte-identical to round 93; every record and restart plant must
   fire.
3. Walk the independent rung-0 substeps in the compiled order above and name
   the first active non-bit statement.
4. If the walk reaches `dyn_cor_2D`, execute the preregistered one-variable
   `e3f_0vor` cross-operator test.
5. Keep the round-94 slow-depth statement held until its GYRE cancelling pair
   is identified.

## UNVERIFIED

- No round-96 payload, terminal-restart comparison, or admission JSON exists.
- The first non-bit statement inside `dyn_spg_ts` remains unnamed.
- ORCA2's `e3f_0vor` cross-operator result remains unmeasured.
- The rung-0 card, its two ten-step ladders, and independent month remain
  later hierarchy work.
