# Grid ladder rung G9 — round 4 resume receipt (gridladder_r49)

Status: **LANDED (tests only), G9 NOT CERTIFIED.**  The two review HOLDs on
round 4's tests are answered; no model code, card, record or score changes.
Decision: the stage-3 statement stays selected (standing procedure, option 1).

- Record: round 46's, unchanged.  No acquisition.
- codex review: operator.
- Evidence: `gridladder_rounds/round49/plants/r49b/`, `round49/resume/`.

## 1. The content count becomes a value test

Removed: the record test's "≤ 1,300 T / ≤ 800 S cells off" bound (a measured
residual turned into a limit).  The record test now checks only the bar.

New `test_g9_stage3_krhs_and_content_are_nemos_statements`: synthetic
operands (24×24×3, magnitudes spread so the order of the sums shows), all
traced, jitted, x64, through the tracer stage program with a stubbed FCT
rate.  Against a numpy transcription of NEMO's lines — Krhs zeroed
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:604`), plus the FCT rate
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637`), plus qns/sfx and runoff
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:323`), plus qsr
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:717`), then the content
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289`) — bit-equal on every cell,
T and S.

## 2. The 1% cap is dropped

Removed from the stage-wzv test: the 1% cap on excluded columns and the
extra one-cell widening around them.  What is left is wzv's own stencil, the
horizontal divergence of a column's four faces
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298`): a column is excluded only
if one of its four faces' zFu/zFv differs from NEMO's.  It still passes.

## 3. Plants (jit, traced inputs, x64; `round49/plants/r49b/`)

| test | plant | result |
|---|---|---|
| value test | stage-3 Krhs branch removed | failed: 1,728 / 1,728 cells |
| value test | FCT rate not requested | failed: wiring assert |
| value test | FCT rate added last | failed: 11 / 1,728 cells |
| value test | restored | passed |
| stage wzv | stage-2 clock pair removed | failed: 165,851 cells |
| stage wzv | stage-1 clock pair removed | failed: 355,104 cells |
| stage wzv | restored (G9 file) | passed |

The value test does not see the model passing the stage-3 terms (the
round-4 plant "terms not passed"); the record test's bar check does not
either.  Not covered by a test.

## 4. Gates (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH; `round49/resume/`)

| battery | result |
|---|---|
| G9 file | 20 passed |
| citation-gate test, private-import ratchet, TKE | 91 passed |
| recipe, freshwater, barotropic state | 49 passed |
| face-mask rank | UNVERIFIED: killed at the 10-minute cap three times, no summary line (`-n 2` printed both dots once); file not touched by this diff |

## 5. Receipt citation gate

`--from-heading` this title, at `dd7a72501`: `PASS`, 8 citations, 0 failures,
0 unmapped, 0 map entries failing the audit.  No map entries added
(`round49/resume/citation_gate.json`).

## Choices

ASKED: both edits (review HOLD).  UNASKED, offered for revert: dropped the
extra one-cell widening in the wzv test along with the cap.
