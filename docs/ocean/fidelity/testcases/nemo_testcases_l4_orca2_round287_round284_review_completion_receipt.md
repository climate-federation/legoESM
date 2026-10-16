# ORCA2 round 287 — round-284 review completion

Date: 2026-10-16

Status: **LANDED**.  This round completes only the four independent-review
items ordered after round 284.  It changes no physics, card selector, NEMO
deck, NEMO source, record, runoff setting, or sea-ice setting.  The sole
`packages/` edit is a documentation citation correction.

## Scope and provenance

- Lane base: `3322696b6`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round287.md`, commit
  `db5a581d1`.
- Review-control implementation: `19c97fca5`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round287/`.
- Independent review: **operator**.  No nested review was attempted in this
  sandbox, per the current lane instruction.

The executing ORCA2 statement is the combined Coriolis plus relative-vorticity
branch inside `SUBROUTINE vor_een`, at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`.  The compiled routine starts
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:692-694`.  These are the statements
whose citation and behavioral controls were repaired; no scientific result was
remeasured or changed in this round.

## Frozen predictions and review dispositions

1. **R287-P1 — CONFIRMED.**  The new small-grid behavioral test executes
   statement A through the live baroclinic-tendency caller.  With the supplied
   NEMO F-cell area exactly twice the generic area, the selected relative
   vorticity is array-exactly one half of the unselected result, on 234 nonzero
   values.  Removing the `een_nemo_zeta` dispatch clause fails 234/234 values;
   replacing the live `nemo_f_area` operand with `None` independently fails
   234/234.  The two known-answer logs have SHA256
   `b4d85a9ca9fea634fd6d1296c33861e676c4bfc966300bb9466a3ffc153387b7`
   and
   `b9a3a966674a0b0dcf298afe150e27ac2fd9230bfe00236deaddd37e40e1c33d`.
2. **R287-P2 — CONFIRMED.**  The first 81 lines of the round-282
   preregistration are byte-identical to commit `9cc8d2ba3`; the original
   dead-arm `:648-652` text is restored.  A dated correction below the frozen
   body retracts it and records the executing `:762-766` EEN span.  No frozen
   prediction was edited in place.
3. **R287-P3 — CONFIRMED.**  The round-284 CLI now gives a fired known-answer
   plant `STATUS PLANT-FIRED` and exit 2.  A plant that stays green instead
   gives `STATUS PLANT-STAYED-GREEN` and the distinct non-plant exit 3.  A
   focused test exercises both paths and proves the markers cannot be
   interchanged.
4. **R287-P4 — CONFIRMED.**  The first 64 lines of the round-284
   preregistration are byte-identical to commit `e45b2d525`; a dated appended
   correction records that the current compiled routine starts at line 692,
   rather than the stale prose locator 694.  The package docstring now cites
   the live `:762-766` branch.  The executable code is unchanged.

The behavioral and review-gate focused tests pass 10/10.  `git diff --check`
is clean.

## Shared GYRE proof

Because the citation-only edit touches a shared model file, the full shared
implementation predicate was measured rather than inferred:

- The base and tip ten-step GYRE gates each produce 70 rows.  The offline
  comparison reports zero worsening ULPs and the same first-over-bar checkpoint
  (`kt=3`); the two residual archives are byte-identical with SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
- The base and tip 30-day member-0 runs used daily snapshots.  All 30 snapshot
  files compare byte-for-byte; day 30 has SHA256
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b` on
  both trees.

The ten-step comparison artifact has SHA256
`4a3acb051a3da00977c8d2c754dd63e96f55e0dceea324d38017654d8ead1e61`.
This proves the documentation-only package edit leaves both certified GYRE
trajectories unchanged.

## Fidelity battery

The single permitted `tests/ocean/fidelity -n 12` invocation collected 3,605
tests and reached 99%.  One G1 worker then aborted inside the JAX compiler and
the replacement controller eventually stopped progressing.  The nine G1 IDs
reported after that worker entered failure state all pass in fresh serial
processes (`-p no:xdist`), in 34.01–108.52 seconds each; the full battery was
not rerun.  The retained full log SHA256 is
`2156e78f40c6b91870069e55fe4888ca4e733d5ea9bb4f2e48a378a7e0de2d09`.

The other eight displayed failures are the branch-tip reds already registered
before this round: the OMT-1 module-edge ratchet; SI3 scalar-math provenance;
the round-129 GYRE harness hash; two round-33 source-string anchors; the
round-228 citation-map entry; dirty-escape scope; and worktree stamping.  None
names a round-287 file.  No unexpected assertion reproduces in isolation.

## Configuration choices

- **ASKED:** none.  Decision 126 remains pending and is not re-raised.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange remains off.
- Sea ice remains exactly as the ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## OPEN

1. Resume Decision 125's matched stage-1 vector-form V-update diagnosis from
   round 286; the completed-stage top-V exchange stays off.
2. Keep the resolved stage-mask candidate held after its rung-0 trajectory
   refusal.
3. Leave shipped runoff unchanged while Decision 126 remains pending.
