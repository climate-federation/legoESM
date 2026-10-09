# ORCA2 round 211 — OMT-1 eight-step record recovery

Date: 2026-10-09. Base: `a6e08ab99`. Preregistration commit:
`dcb2dd2f3`. Instrument commits: `fbb3bf45e`, `f51cb5040`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round211/`.
No legoESM trajectory was measured. Future ladder numbers remain separately
labelled **independent OMT-1** and **given NEMO's entry OMT-1**.

## Result and correction

Round 210's ten-step OMT-1 protocol is retracted. NEMO completed kt=8, then
its own stability control stopped at kt=9 with `|V| max = 10.13 m/s` at local
`(22,85,27)`. The compiled control prints this exact diagnostic family at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-298`. The committed boundary gate now proves the preserved run
uses the one-module OMT-1 deck, exits through MPI error 123 and compiled
`stp_ctl`, and stops at exactly kt=9. Its wrong-boundary plant fires.

The admissible OMT-1 record is therefore kt=1..8. The recovery launcher expects
64 self-describing frames per instrumented twin: two ranks × eight steps ×
four stages. It stages one uninstrumented calibration and two instrumented
twins with `nn_itend=8`, `nn_stock=8`, and restart list `2,4,6,8`; all three
step-8 rank restarts must be byte-identical. Header, field-name, truncation,
non-finite, missing-frame, twin-ULP, terminal-byte, changed-binary and cadence
plants remain fail-closed.

No NEMO build is requested. The launcher pins and reuses the completed
round-210 record binary, SHA256
`5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895`.
The executing vector-form U/V update is compiled at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`. The OMT-1 deck hash remains
`d0cccebd76c9c4631c91e51a1af3a9b552d97f307c7c523020160c883fb5e73f`;
its only OMT-0 differences remain the mutually-exclusive momentum-advection
selectors.

## Frozen-prediction disposition

| ID | disposition |
|---|---|
| R211-P1 | **CONFIRMED:** the deck and completed record binary retain their pinned hashes; the deck delta remains exactly two selector assignments. |
| R211-P2 | **UNMEASURED_WITH_SPEC:** the operator must run the eight-step uninstrumented calibration. |
| R211-P3 | **UNMEASURED_WITH_SPEC:** the two 64-frame instrumented twins do not exist yet. |
| R211-P4 | **UNMEASURED_WITH_SPEC:** additions-only terminal restart identity requires those twins. |
| R211-P5 | **CONFIRMED:** the mechanical boundary gate reports `PASS_R211_OMT1_STABILITY_BOUNDARY`, kt=9 and `|V|=10.13 m/s`; its plant fires. |
| R211-P6 | **PARTIAL:** all preflight and boundary plants fire; payload and terminal plants require acquisition. |

## Validation and review

The committed launcher reports
`ORCA2_ROUND211_OMT1_FRAMES_PREFLIGHT_READY`. The frame preflight reports 64
expected frames per twin, four call sites, and zero removed source lines. The
focused round-209/210/211 tests pass 15/15; shell syntax and Python compilation
pass. No `packages/` file changed, so GYRE, DINO, tanks, rung 0, and shipped
rung 10 cannot move in this round.

Independent review was attempted with `codex exec --sandbox read-only` and
failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Independent review is unavailable in-sandbox; this is not a PASS verdict.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round211_omt1_frames_acquisition/run.sh --run`.
2. The next round admits the 64-frame twins, runs both labelled 160-row OMT-1
   ladders through kt=8, and names the first non-bit vector-branch statement.
3. The OMT-1 card remains prepared but unlanded until those gates complete.

No configuration or sea-ice decision is pending. ASKED choices: Decisions 103
and 109 and the binding eight-step recovery. UNASKED choices: empty.
