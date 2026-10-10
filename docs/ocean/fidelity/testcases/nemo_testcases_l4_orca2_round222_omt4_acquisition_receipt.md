# ORCA2 round 222 — OMT-4 tracer-advection acquisition handoff

Date: 2026-10-10. Frozen base: `9363ce511`. Preregistration commit:
`fc29e5ad7`. Acquisition implementation commit: `4e2efad4f`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/`.

This round changes no package model file, shipped rung-0/rung-10 card,
carried state, stabiliser, sea-ice selector, or `unmeasured_features` tuple.
It creates the Decision-109 OMT-4 deck and a fail-closed operator acquisition.
No NEMO run was attempted in the sandbox. OMT-4 trajectories are therefore
**UNMEASURED WITH SPEC**, and no independent/given-entry numbers are mixed or
reported.

## Exact OMT-3 to OMT-4 edge

The canonical deck is rendered from the admitted round-220 OMT-3 deck,
SHA-256 `a4c023c1b9a667cb0ba6ea839b62fe38059cb4050e8cec9a057fd710c6382715`.
The gate reports exactly one changed assignment and one removed override:

| assignment | admitted OMT-3 | OMT-4 | provenance |
|---|---:|---:|---|
| `namtra_adv.ln_traadv_fct` | `.false.` | `.true.` | rung-0 shipped module |
| `namtra_adv.ln_traadv_off` | explicit `.true.` | absent, resolving `.false.` from `namelist_ref` | rung-0 shipped module |

Vector momentum advection, linear implicit bottom drag, lateral momentum
diffusion, and tracer-diffusion OFF remain unchanged. The rendered deck
SHA-256 is
`a55981c9c3aa5fab9d98760f64e7229950c0d974b5356f128494bf868773bf99`.
The deck preflight JSON SHA-256 is
`41f8e7def32b3e602ee5771ccdfc42312f18fca447e7914497d684dd1931d98f`.
Both extra-delta and wrong-selector plants exit nonzero.

The compiled build declares and reads the tracer-advection selectors at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:592-602`, requires one
and only one arm and validates the FCT order at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:630-647`, and prints the
resolved FCT selection at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:668-674`. The live stage
program calls tracer advection at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:633-643`; the
dispatcher uses centred advection at stages 1 and 2 and FCT at stage 3 at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540`.

Search-before-build found and reused the round-220 OMT-3 binary-reuse
launcher, deck gate, frame-record gate, and the admitted P3 writer. The
record gate gained a backward-compatible deck-validator parameter so the
same header-driven inventory and plant suite serves OMT-3 and OMT-4. No
numerical implementation or NEMO source patch was added.

## Acquisition contract

The launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round222_omt4_frames_acquisition/run.sh`.
It reuses the admitted uninstrumented binary SHA-256
`c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343`
and additions-only P3 binary SHA-256
`5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895`;
it does not call `makenemo`.

The committed run order is:

1. legal two-step smoke with `nn_fsbc=2`, `nn_stock=2`;
2. uninstrumented ten-step calibration;
3. two independent rank-complete P3 twins through kt=10;
4. uninstrumented 96-step boundary run with the compiled ten-entry restart
   list `10,20,...,90,95`.

Admission requires 80 self-described frames per twin, both ranks, four stage
boundaries, five fp64 fields, exact twin equality, and byte-identical kt=10
terminal restarts against the uninstrumented calibration. The month either
completes kt=96 or records an exact compiled `stp_ctl` boundary strictly after
kt=10; an earlier boundary refuses the record. Expected `stp_ctl` exit 123 is
classified inside an OR-list before the global shell trap. The frame preflight
JSON SHA-256 is
`04bf164ffc79072ada8d2446ca6da89fb3992a55b92a6ee176a3a7f6c1324d4d`;
the full preflight log SHA-256 is
`e6d60d0c9b9acff57531e46cf85cf71f3bfc6ad4ae45f3c2230ba997ecb56d8c`.

## Preregistered predictions

| ID | disposition |
|---|---|
| R222-P1 | **CONFIRMED**: exact one-module deck edge; both deck plants fire. |
| R222-P2 | **UNMEASURED WITH SPEC**: ten-step NEMO calibration/twins await the operator. |
| R222-P3 | **UNMEASURED WITH SPEC**: the month boundary awaits the operator. |
| R222-P4 | **UNMEASURED WITH SPEC**: the complete vector unit cannot be scored without the OMT-4 record. |
| R222-P5 | **PARTLY CONFIRMED**: deck and inventory plants fire in preflight; record-content and future trajectory plants await real output. |

## Validation and review

Focused round-220 compatibility plus round-222 tests pass 14/14. Independent
review was attempted separately with `codex exec --sandbox read-only` and
exited 1 before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. **Independent review unavailable
in-sandbox**; this is not a PASS. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

No `packages/` file changed, so no production GYRE/DINO/tank trajectory can
move in this round. On clean receipt/map commit `756605a36`, the round
citation gate passes five citations and the cumulative default gate passes
274; both report zero failures, unmapped citations, or map-audit failures. The
explicit two-line shift plant on the mapped `traadv.f90:592-602` range exits 1
and makes the gate fail. Round/default/plant JSON SHA-256 values are
`c00f3ebe6b133d39ca27076f45775316db944cd0bbd5004fd99bab532e662e20`,
`88df3f34246d1d61780d5ab80372b16e5780c01343803e6e4758f662a848e9eb`,
and `6de539a46ef2362d84e3122c42423edd94bbc92e34063d025e7d029de3efecaf`.

Focused acquisition/compatibility/citation tests pass 15/15 (log SHA-256
`5ccecaa324164f02c67856e15b39f111e185137dd7166f2f25bd1da0d2f48829`).
The prescribed `tests/ocean/fidelity -n 12` battery collected 3,079 tests and
reached 99%; 3,055 passed and seven skipped. Its four failures are the same
registered pre-existing reds reported in rounds 220 and 221: the GYRE
round-129 spread-floor record stamp, allow-dirty scope, worktree-stamp ratchet,
and SI3 scalar-math provenance gate. The remaining 13 tests were unclassified
when the pytest processes disappeared without a terminal summary. The battery
is not called PASS and was not relaunched. Its log SHA-256 is
`22fa97e0f85aeb304d797ff557a71a5f893b1708c764a82e4d70025bafb8e2b5`.

## OPEN

1. The operator runs the committed launcher with `--run`. The expected target
   is
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
2. Admit the record and disposition R222-P2/P3. For a checker mismatch, repair
   the checker against the self-described record and use `--admit-existing`;
   never rerun NEMO for a check-only refusal.
3. Build the gate-local OMT-4 card, run both labelled ladders, and score the
   complete round-217 vector unit atomically. If the kt=8 live-W refusal first
   returns here, walk tracer advection to its first cited non-bit statement
   before OMT-5. If it remains clean, OMT-5 is next.
4. Do not promote any constituent of the unit or touch the shipped rung-10
   card. Sea ice and its `unmeasured_features` tuple remain unchanged.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
