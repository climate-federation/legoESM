# NEMO testcase Lane 4 — ORCA2 card round 35 `r3f` reciprocal refutation

Date: 2026-09-26

Parent: `d28759f36fe032526eb78a253f593c49e56663e2`

Measurement tip: `8f330278bdd8faddbb68ae2e9d63a2e79e93328a`

Status: **HELD — THE STORED-RECIPROCAL SPELLING IS NOT THE OWNER.**  The
one-variable source-order arm changes 3,810 `r3f` cells, but not one scored
lateral-diffusion tendency bit.  The same read-out exposes an earlier and
larger difference: the F-area array used by the live `r3f` builder is not the
card's native NEMO `e1f*e2f` field.  Nothing in `packages/` changed.

All measurements below are **given NEMO's entry (kt=2 recorded state)**.
No independent trajectory was run because the proposed statement is
tendency-inert and no model change was eligible.  The six sea-ice selectors
and the card's `unmeasured_features` tuple are unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round35/`.

## Compiled statements

The executing build forms the F area and stores its reciprocal at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157`.
The executing QCO routine multiplies the bracketed surface sum by the stored
F-column and F-area reciprocals at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer reads the live F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

## Given-NEMO-entry result

The parent replay reproduces Round 34 exactly:

| field | unequal / scored | maximum, m/s2 | L2 ratio |
|---|---:|---:|---:|
| U | 410,460 / 411,736 | `3.181628207426175e-09` | `1.1072530302454793e-04` |
| V | 407,570 / 412,537 | `2.9702048395431957e-09` | `8.952453804424378e-05` |

Replacing only division by the F area with multiplication by its stored
reciprocal reproduces every number in that table exactly.  The planted arm
changes one additional U cell and exits nonzero, so the tendency comparison
is live even though the proposed arithmetic change is inert.

The boundary read-out separates the two reasons the prediction failed:

| boundary | unequal / cells | maximum absolute difference |
|---|---:|---:|
| T area, card grid versus native `e1t*e2t` | 0 / 26,640 | `0.0 m2` |
| F area, card grid slice versus native `e1f*e2f` | 26,456 / 26,640 | `2.856475828212061e10 m2` |
| `r3f`, division versus stored reciprocal | 3,810 / 26,640 | `3.469446951953614e-18` |

Thus the reciprocal-order difference exists at `r3f`, but rounding the
result into the live F thickness removes it before the LDF consumer.  The
first remaining differing input is the F-area field itself.  Its exact cause
and correction are not claimed in this round; Round 36 must compare the
native-F-to-vertex index map before any implementation.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R35-P1 | **REFUTED** | `r3f` moves on 3,810 cells, but the prerequisite F-area equality fails on 26,456 cells. |
| R35-P2 | **REFUTED** | The reciprocal arm retains the parent's exact U/V maxima and unequal counts. |
| R35-P3 | **UNMEASURED** | No implementation was eligible, so no ORCA2 trajectory arm was run. |
| R35-P4 | **UNMEASURED** | No `packages/` file changed, so no GYRE arm was run or claimed. |
| R35-P5 | **CONFIRMED FOR THE MEASUREMENT DIFF** | Focused tests pass; the wide battery reproduces only the five inherited failures and the inherited final-tail stall; the measurement plant exits nonzero. |

Failed predictions are retained rather than rewritten.

## Gates, review, and tests

The focused Round 31/34/35 files pass **15/15 in 3.75 s**.  The required
`tests/ocean/fidelity -n 12` battery reached 99% and the inherited final-tail
stall before a bounded interrupt.  It emitted the same five failures as
Rounds 33-34: SI3 MY_SRC provenance, the stale GYRE member/gate stamp, the
Round-51 trace suffix, three unstamped legacy emitters, and the missing
`hires_lane_surface` case-board row.  Their exact node IDs were rerun in
isolation and reproduce **5/5 inherited failures**; no Round-35 test failed.

The required `codex exec --sandbox read-only` review was attempted against
the committed measurement diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.  The complete log is
preserved as `codex_readonly_review.log`.

The receipt citation gate passes all three compiled citations with zero
failures, zero unmapped citations, and zero map-audit failures.  Its real
rigid-shift plant exits 1 with `SYMBOL-NOT-AT-LINE`.  The first gate run
caught the preregistration's one-line-high `domhgr` range; the three-line
extent was shifted rigidly by +1 and the correction is retained in the
preregistration.  The final push battery, including the citation gate,
NEMO recipe, TKE, freshwater, parallel-receipt, and Round-35 tests, passes
**130/130 in 360.54 s**.

The live GitHub issue could not be read: the local `gh` credential is invalid
and the public issue URL is unavailable without authentication.  No issue
comment was posted or claimed.

## Choices

ASKED: Round 34's OPEN section authorizes the one-variable walk through the
remaining `r3f`/metric operands.

UNASKED: none.  No model, configuration, forcing, state, NEMO source,
sea-ice selector, score, or acquisition changed.

## OPEN

1. Round 36 must preregister the native F-area index-map discriminator.  The
   live builder's F-area slice differs from NEMO on 26,456 / 26,640 cells,
   maximum `2.856475828212061e10 m2`; substitute only the card's native
   `e1f*e2f` before considering a model change.
2. The given-NEMO-entry LDF replay remains at
   `3.181628207426175e-09` / `2.9702048395431957e-09` m/s2.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
