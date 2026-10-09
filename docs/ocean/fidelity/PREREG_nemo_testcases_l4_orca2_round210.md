# Preregistration — ORCA2 round 210 OMT-1 acquisition repair

Date: 2026-10-09. Base: `c368e8b0d`. Scope: repair the failed round-209
OMT-1 record-build launcher without changing the frozen deck, card, frame
writer, admission gate, or scoring protocol. No NEMO trajectory is measured in
the sandbox; the operator runs the committed launcher.

## Frozen observation

The round-209 two-step OMT-1 smoke run completed with `STOP 0`. The subsequent
record build failed before a binary existed. Its log printed an empty reference
configuration and empty component list, attempted to write `/work_cfgs.txt`,
and then failed preprocessing because `do_loop_substitute.h90` was absent.

The cause is frozen from `makenemo`: a new configuration's `-r` operand is
resolved only against `cfgs/ref_cfgs.txt` (`makenemo:263-280`). Round 209 passed
the work configuration `ORCA2_OMIP_L4_R90FRAMES`, which is registered only in
`cfgs/work_cfgs.txt`. The admitted round-166, round-169, and round-172
launchers instead create from reference `ORCA2_OMIP_L4`, then copy the pinned
instrumented configuration's `EXP00`, `MY_SRC`, and CPP card before rebuilding.

## Frozen predictions and falsifiers

| ID | prediction | falsifier |
|---|---|---|
| R210-P1 | A new launcher created from registered reference `ORCA2_OMIP_L4` and then populated from the pinned R90 configuration passes its static preflight. | The reference/work registrations are absent, any pinned source digest moves, or the reconstructed source inventory differs. |
| R210-P2 | The repaired launcher changes no OMT-1 deck assignment, frame-writer byte, admission rule, or ladder rule. | Any content other than target/evidence labels and the build bootstrap differs from round 209. |
| R210-P3 | An operator-run build produces a P3 binary whose compiled `stprk3` contains the four admitted frame calls. | Build failure, missing binary, or missing compiled call. |
| R210-P4 | The smoke, calibration, twin, month and admission outcomes remain **UNMEASURED_WITH_SPEC** until the operator runs the repaired launcher. | No in-sandbox result can confirm them. |
| R210-P5 | A planted invalid bootstrap using the work configuration as `-r` is rejected before `makenemo`; a source-inventory plant also fires. | Either plant stays green. |

## Controls and stop conditions

- Use a new target `ORCA2_OMIP_L4_R210OMT1_P3` and a new round-210 evidence
  directory; never delete or reuse the failed round-209 target or smoke run.
- The mandatory two-step smoke uses the same rendered OMT-1 namelist physics.
- The record build is reconstructed exactly as the last admitted ORCA2
  acquisition pattern: create from `ORCA2_OMIP_L4`, copy the pinned R90
  `EXP00` and `MY_SRC` inventories plus CPP card, touch the copied Fortran
  sources, rebuild, and verify the compiled calls.
- Any target collision, digest mismatch, unregistered bootstrap, incomplete
  frame stream, non-identical twins, terminal-restart mismatch, or plant that
  stays green is a named `REFUSE`.
- Sea ice and the shipped rung-10 card are untouched.

