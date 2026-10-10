# ORCA2 round 219 — OMT-2 card and atomic vector-unit ladder

Date: 2026-10-10. Frozen base: `618a2eec6`. Preregistration commit:
`1da75ed32`. Gate commits: `eb9708227`, `840008627`. Status: **LANDED**
(gate-local OMT-2 card and record; no production physics change). Evidence
root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round219/`.

The final tree changes no package model file, shipped ORCA2 card, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` entry. Every
trajectory number below is separately labelled **independent OMT-2** or
**given NEMO's entry OMT-2**; the two labels happen to have the same scores,
but were run and retained independently.

## Existing record admitted; 96-step prediction refuted

No NEMO run was repeated. The round-218 smoke, uninstrumented ten-step
calibration, and both P3 twins had already completed. The refusal came only
from the launcher's shortened literal diagnostic. NEMO selects the extrema
stop in compiled
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:243-250` and prints the
full diagnostic plus abort state at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-310`.

The repaired gate admits 80 self-described frames per twin, 400 field
comparisons, and four byte-identical terminal-restart comparisons. Every
record-content plant, including the planted full-stop-line mismatch, fires.
The month boundary is `STP_CTL` at kt=11: `|ssh|max=3.574 m`, `|U|max=2.838
m/s`, and `|V|max=10.54 m/s`. Only the step-10 restart exists before that
boundary. Admission JSON SHA-256 is
`224ea17fd4ea47f9f79fa39d4b89873fdd4c58d0c03c94923c90d63800caf9d4`;
the admission log SHA-256 is
`80d647e45fda889e85b86ee6c7333c5fc3af8af825f61df17402a12cae7a5570`.

This confirms R218-P2 and refutes R218-P3. The round-218 receipt now carries a
loud correction; its original preregistered claim remains visible.

## Exact OMT-2 card edge

Search-before-build found the already-shared seamount SMT-2 composition and
the OMT-1 ladder; round 219 reuses both. The gate reports only three values
that actually change from OMT-1:

| field | OMT-1 | OMT-2 |
|---|---:|---:|
| `bottom_drag_scheme` | `legacy` (zero drag) | `nemo_linear` |
| `zdf_drag_in_matrix` | false | true |
| `barotropic_drag_substep` | false | true |

The resolved `rn_Cd0=1.e-3`, `rn_Uc0=0.4`, and
`zdf_baroclinic_only=True` were already carried by OMT-1 and remain unchanged.
No other model-config field moves. NEMO constructs and selects linear drag at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:258-284` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:540-547`, removes and
restores the barotropic component at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:158-169`, inserts the
partial-cell U/V diagonals at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:303-312` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:470-474`, and builds the
split-explicit drag coefficients/residual at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463`.

Both independent and given-entry card entries retain the admitted bit-exact
classification. The card-module and entry-bit plants refuse.

## Baseline OMT-2 ladders

Both labels complete 40 checkpoints / 200 rows through kt=10, finite. Their
scores are identical. The first non-bit row is kt=1 stage-1 T. Selected
baseline rows are:

| row | RMS | maximum |
|---|---:|---:|
| kt1 stage1 T | `1.4205672679174248e-05 K` | `0.001294884324850809 K` |
| kt1 stage1 SSH | `0.006238271974498368 m` | `0.1310917645484672 m` |
| kt10 stage3 T | `0.0003405479310316469 K` | `0.0653356552458737 K` |
| kt10 stage3 S | `0.003243784118955996 PSU` | `0.39326251184436245 PSU` |
| kt10 stage3 SSH | `0.027845928328538504 m` | `0.3851267259362673 m` |

Independent/given-entry baseline JSON SHA-256 values are
`25586264076ee040c82779d99f14ee06886333fde5a865e09c8e24005711439b` and
`cb0e2e747710ed5d5864ec541d4cc8642ccd0bd1a117121f6f2c39bb19c6634f`.

## Complete vector unit qualifies on OMT-2

The round-217 unit remains indivisible: slow-V/raw-mask pair, raw reference
face depths, complete seven-array association, and unmasked/materialised V
transport. Both candidate ladders complete kt=1..10. On each label, 195/200
rows move and all 195 RMS-moved rows move toward NEMO; zero move away or remain
score-equal, no exact row leaves the bar, and the first debt moves toward.
Maximum-error votes are 184 toward / 11 away / 0 equal.

The kt=1 stage-1 SSH RMS/maximum improves
`0.006238271974498368 / 0.1310917645484672` to
`0.00023180348917108853 / 0.006277375309180697 m`. The kt=10 stage-3 rows
move as follows:

| field | RMS before → after | maximum before → after |
|---|---:|---:|
| T | `3.405479310316469e-4 → 9.490239678117918e-5 K` | `0.0653356552458737 → 0.00924936616763361 K` |
| S | `0.003243784118955996 → 0.0004425351036623843 PSU` | `0.39326251184436245 → 0.03901689756581561 PSU` |
| SSH | `0.027845928328538504 → 0.0034307860095954 m` | `0.3851267259362673 → 0.06761657785137581 m` |

Independent/given-entry candidate JSON SHA-256 values are
`5eb19bd2186d104a073f097c0ee880a69abb0f506e224218ec65ba4a67baad82` and
`3e4156ef541d9eba28f67803f9a42d4a37b9ca4dca7a566ea6f8ce08bab916ac`.
Their Decision-96 report hashes are
`ba93c15bcd0dc6cc39f409f92dbc1639048e0b4474b001b0a0852b1fb801c136`
and `5b6f5e7b1fa824ece7a926536f9cf03756882e6d3e63daae90929ea4fb32935b`.
The pair-closure, exact-loss, and false-majority plants all refuse.

The unit therefore qualifies on OMT-2 under Decision 96, but this does not
authorise a production landing: independent rung 0 still refuses with the same
unit, as round 217 recorded. The result says the missing compensating partner
is not linear drag; it first enters at OMT-3 (+ momentum LDF) or later.

## Preregistered predictions

| ID | disposition |
|---|---|
| R219-P1 | **CONFIRMED**: existing twin/restart/frame record admits without rerun; boundary is kt=11. |
| R219-P2 | **CONFIRMED**: R218-P3 is REFUTED by the printed `|V|=10.54 m/s` kt=11 boundary. |
| R219-P3 | **CONFIRMED**: exact card edge, no extra config field, both entries retain their admitted classification. The three actually changed fields are listed above; inherited resolved values are explicit. |
| R219-P4 | **CONFIRMED**: both baseline labels complete 40/200 and remain finite. |
| R219-P5 | **CONFIRMED on OMT-2 only**: both candidate labels complete and pass Decision 96, but no rung-0 landing is authorised. |
| R219-P6 | **CONFIRMED**: all record, card, entry, pair, exact-loss, majority, and citation plants fire. |

## Validation and review

Focused round-218/219 and citation-map tests pass 11/11 (log SHA-256
`cd0dee9f98a367c58116b136d718e17d80581b1a239049c1fc806fcc4bf40aed`).

The one prescribed `tests/ocean/fidelity -n 12` battery collected 3,060 tests.
It reached 99% and emitted 3,016 PASS, eight SKIP and 27 FAIL lines before a
bounded interrupt left nine tests unclassified. It is not called PASS. The two
EOS failures were replayed alone and both are the registered clean-worktree
stamp refusal against this round's uncommitted receipt/map; their isolated log
SHA-256 is
`74be9d4ddd8b3bfc8e4ead77ca342c8ff563a1760f6be8b091f4b96b328f505f`.
All 27 emitted failing IDs are retained in `pytest_failed_ids.txt` for the
post-commit clean replay. Full battery log SHA-256:
`4c8ebfb262691eae2f3f4c02257b099c596b4758ccb052419a5f9ba428fc80eb`.

Independent review was attempted separately with `codex exec --sandbox
read-only` and exited 1 before reading the diff: `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
Independent review is unavailable in-sandbox; this is not a PASS. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

On clean receipt/map commit `65321007c`, the round-219 citation gate passes all
eight compiled citations with zero failures/unmapped entries, the cumulative
default gate passes 274 citations with zero failures/unmapped entries, and the
explicit two-line `stpctl` shift plant fails. Receipt/default JSON SHA-256
values are `fd29d6715db916d4931bba153916cb8632f1304a28c9e1076460aca818b268f7`
and `0f642c762a6ef91191d70bde6d12bd849f1c8e2ed273ec9bff6e49a581e54ab9`.

The 27 emitted failure IDs were replayed on that clean tree: 23 pass and four
remain as the registered pre-existing reds named in the campaign brief — SI3
scalar-math provenance, GYRE round-129 spread-floor record stamp, allow-dirty
scope, and worktree-stamp ratchet. Clean-replay log SHA-256:
`f3bdc09963bcabba2bd5c0d7f18781a2a17efa7ba1252a278ff43623a0ef5e02`.
No round-219 test is red.

## OPEN

1. OMT-3 is OMT-2 plus the rung-0 lateral momentum-diffusion module. Copy the
   proven round-218 binary-reuse acquisition, change only that deck edge and
   target names, smoke first, and acquire its per-stage kt=1..10 record plus
   month boundary.
2. Build the gate-local OMT-3 card, run both labels, and score the same complete
   vector unit atomically. The first rung where it recreates the kt=8
   live-W-thickness refusal identifies the module containing the compensating
   partner; walk that module there before OMT-4.
3. Do not promote any constituent of the unit or touch the shipped rung-10
   card. Sea ice and its `unmeasured_features` tuple remain unchanged.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
