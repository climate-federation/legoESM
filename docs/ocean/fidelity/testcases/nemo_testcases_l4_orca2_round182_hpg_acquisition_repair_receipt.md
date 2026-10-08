# ORCA2 round 182 — HPG fold acquisition repair

Date: 2026-10-08. Base `817d75fb5`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round182/`.
Status: **STOPPED_FOR_RECORD**.

Every scientific statement remains **independent hierarchy rung 0**. This
round changes no `packages/` file, card, configuration, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` entry. It reads no new
scientific payload.

## Verdict

The operator did run the round-181 launcher. NEMO's base configuration built
successfully, but the launcher failed before applying the writer and before
creating a run directory. The decisive line was:

```text
error: affected file '.../ORCA2_OMIP_L4_R181HPGFOLD/MY_SRC/dynhpg.F90' is beyond a symbolic link
```

The owner was `git apply --directory=...`: the configured
`/home/dbalwada/oracle-builds` path is itself a symlink to the writable data
filesystem. The compiled base target remained uninstrumented. Its
`dynhpg.F90` SHA256 is
`0fca1d9b74dec033f96737e4ab6fe883f68d4df10c432ce67daff63bd4d449f6`,
equal to the pinned round-180 source, and no round-181 target run directory or
HPG-fold record exists.

The repair uses the repository's established exact staging pattern: GNU
`patch --fuzz=0`, then a byte comparison against the source already patched by
`git apply` in an ordinary scratch directory and syntax-compiled. The original
writer patch lacked the normal trailing context GNU `patch` requires; it was
mechanically regenerated from the unchanged base and unchanged instrumented
source. Both patch engines now produce the same file, SHA256
`599014973c8a2c8146dda421136e8af71c1e63b7c0d41f5c79df9a76645ae0d6`.
The patch remains additions-only.

The repair targets the fresh configuration `ORCA2_OMIP_L4_R182HPGFOLD` and
fresh evidence directory `orca2_rounds/round182/acquisition`. The failed
round-181 target is never deleted, reused, or overwritten. Its source and
binary hashes remained identical before and after all round-182 preflights.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R182-P1 failure was after base build but before writer/run | **CONFIRMED**: the log says `Compilation successful`; target source equals the pinned base; the target run directory is absent. |
| R182-P2 exact GNU patch is faithful | **CONFIRMED**: GNU patch and git-apply outputs are byte-identical; the staged-source one-line plant fires. |
| R182-P3 repaired launcher is ready under a fresh target | **CONFIRMED**: symlink-parent proof, syntax proof, layout census and both plants pass; the fresh target remains absent before operator execution. |
| R182-P4 no scientific payload is read | **CONFIRMED**: no new record exists and no HPG operand is interpreted. |

## Oracle citation and scope

The pending instrument still observes the same compiled independent-rung-0
HPG component sequence and changes no executed statement:
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`. The first non-bit
component remains round 181's admitted `zhpj` boundary; north/current operand
ownership remains **UNMEASURED-with-spec** until the repaired record admits.

## Mechanical checks and review

The clean-tip launcher prints `PATCH_SYMLINK_PARENT_PROOF_PASS`,
`SYNTAX_PROOF_PASS`, and
`ORCA2_ROUND182_HPG_FOLD_PREFLIGHT_READY`. The layout plant and staged-source
identity plant each exit 69 with `STATUS PLANT-FIRED`. The focused parser and
launcher battery passes 6/6.

The separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

No model file changed, so ORCA2/GYRE/DINO/tank trajectory gates are ineligible
in this acquisition-repair round. Citation and wide-battery results are added
at the clean final tip.

## OPEN

1. The operator runs the reported launcher. It builds only the fresh round-182
   target, runs the existing ten-step rung-0 deck, and admits or refuses the
   two exact fold-operand rank files.
2. After admission, replay `zhpj` top-down and stop at the first north/current
   product or accumulation statement above the fixed floor.
3. Analyse raw HPG, `e3v`, `vmask`, and `r1_hv0` atomically before any landing;
   then resume the independent month's step-96 live-thickness refusal.

The independent 240-step month score remains **UNMEASURED-with-spec**.
ASKED choices are the exact mechanical repair under a fresh target. UNASKED
choices are empty.
