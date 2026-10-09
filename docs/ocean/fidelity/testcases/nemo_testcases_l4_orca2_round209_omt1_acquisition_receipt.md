# ORCA2 round 209 — OMT-1 vector-form acquisition handoff

Date: 2026-10-09. Base: `9884a20d9`. Preregistration commit: `f80e35c38`.
Implementation commits: `9e5417c98` and `6e6bc3970`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round209/`.
No trajectory number was measured. Future scores are separately labelled
**independent OMT-1** and **given NEMO's entry OMT-1**.

## Result

Decision 109's OMT-1 rung is preflight-ready. Its NEMO deck differs from the
admitted OMT-0 deck in exactly two mutually-exclusive assignments:

| assignment | OMT-0 | OMT-1 |
|---|---:|---:|
| `namdyn_adv.ln_dynadv_OFF` | true | false |
| `namdyn_adv.ln_dynadv_vec` | false | true |

The compiled selector maps these to linear dynamics and the vector KEG + ZAD
+ VOR program, and refuses any non-unique selection at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190`. The split-explicit solver's
executing OMT-1 update is the direct vector U/V statement at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`. Bottom drag,
momentum lateral diffusion, tracer advection, and tracer lateral diffusion
remain OFF. No surface forcing, ice selector, carried state, stabiliser, or
unmeasured-feature tuple changed.

The explicit legoESM OMT-1 card is also committed. It starts from the admitted
OMT-0 card and restores exactly four configuration fields from the instantiated
rung-0 card: vector-invariant momentum advection, its flux-scheme and vertical-
momentum dispatch values, and total EEN vorticity. Its test proves every other
model-config field remains representation-identical to OMT-0. This is a
prepared gate, not a measured card landing; admission and both 200-row ladders
must run after the oracle record exists.

## Round-208 acquisition disposition

The operator ran round 208's requested flux-form stream. Its own checker
refused before any operand was read:
`oracle_r208_midv_rank0000_kt00000001.bin: zv_frc dimensions moved`.
Decision 109's handoff explicitly retires that flux-form-only walk, so this
round neither repairs nor interprets the refused stream. It remains an
instrument refusal, not physics evidence.

## Acquisition handoff

The committed launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round209_omt1_frames_acquisition/run.sh`.
It copies the admitted round-203/204 protocol and changes only the two deck
selectors, OMT-1 labels, target names, and the required month boundary arm.

The launcher:

- pins the complete 17-file `ORCA2_OMIP_L4_R90FRAMES/MY_SRC` inventory and
  CPP card by content;
- runs the two-step smoke with the identical OMT-1 physics before building;
- builds fresh target `ORCA2_OMIP_L4_R209OMT1_P3` from the admitted passive
  frame source;
- runs one uninstrumented ten-step calibration and two instrumented twins;
- admits 80 self-describing rank/stage frames per twin only after twin-array
  identity and four terminal-restart byte identities;
- runs the independent 96-step OMT-1 month with the frozen ten-entry restart
  list and admits either STOP 0 or an exact later `stp_ctl` boundary; a stop at
  or before OMT-0's kt=11 boundary refuses.

`--preflight-only` passes. The rendered namelist diff contains exactly the two
rows in the table. The passive-writer preflight reports four frame call sites,
zero removed source lines, 80 expected frames per twin, two ranks, ten steps,
four stages, and five fields. Both deck plants fire. Evidence:
`round209/acquisition_preflight.log`.

## Frozen predictions

| ID | disposition |
|---|---|
| R209-P1 | **CONFIRMED at preflight**: the deck delta is exactly the two `namdyn_adv` assignments. Runtime resolution awaits the record. |
| R209-P2 | **UNMEASURED_WITH_SPEC**: smoke, twin payloads, and terminal restart identity require operator acquisition. |
| R209-P3 | **CONFIRMED structurally**: the instantiated card selects the four exact rung-0 vector fields and retains the other four OMT modules OFF. Runtime deck/card agreement awaits the record. |
| R209-P4 | **UNMEASURED_WITH_SPEC**: both labelled ten-step ladders require the 160 oracle frames. |
| R209-P5 | **UNMEASURED_WITH_SPEC**: the OMT-1 month boundary requires NEMO. |
| R209-P6 | **PARTIAL**: both deck plants and focused synthetic plants fire; record plants run only after acquisition. |

## Validation and review

Focused round-209 tests pass 8/8. Shell syntax, Python compilation, exact deck
rendering, full record-source manifest, and the committed acquisition preflight
pass. No `packages/` file changed, so no GYRE, DINO, tank, rung-0, or shipped
rung-10 trajectory can move in this round.

Independent review was attempted with `codex exec --sandbox read-only` and
failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system
(os error 30)`. Independent review is unavailable in-sandbox; this is not a
PASS verdict. The full fidelity-battery result is recorded below after its one
permitted invocation. Its log reached 95% with 2,848 passed, 7 skipped, 3
failed, and 0 errors before the worker/session tail stopped producing output;
there is no final pytest summary, so this is **INCOMPLETE**, not a pass. The
three observed failures are the registered pre-existing reds
`test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`,
`test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it`,
and
`test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`.
The last emitted node was
`test_nemo_testcase_l1_vortex_round204_stage1_split.py::test_the_split_hook_at_its_default_changes_nothing`;
the corrected process census was zero after interruption. Evidence:
`orca2_rounds/round209/ocean_fidelity_pytest.log`.

## OPEN

1. The operator runs the reported launcher. The next round admits the record,
   runs both labelled OMT-1 ladders, records the month boundary, and names the
   first non-bit vector-branch statement.
2. The card remains prepared but unlanded until those gates complete.

No configuration or sea-ice decision is pending. The failed round-208
flux-form acquisition is not an OPEN item.
