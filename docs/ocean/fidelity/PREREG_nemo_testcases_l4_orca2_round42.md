# NEMO testcase Lane 4 — ORCA2 card round 42 preregistration

Date: 2026-09-27

Parent: `e5bdf138c1346d0105b4490223ced9a524d1a644`

Status: **PREREGISTERED BEFORE ROUND-42 SCIENTIFIC SCORING.**

All scientific numbers will be **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.

Round 41 acquired and admitted one passive per-rank stream containing the
stage-1 `hpg_sco` inputs (`rhd`, live `e3w`, live `gdept_z0`, and stored metric
reciprocals), the two compiled statement components (`zhpi`/`zhpj` and
`zuap`/`zvap`), and their final sum.  Round 42 walks those boundaries on the
64 registered rank-1 U rows.  It reuses the existing production literal
recurrence and the Round-41 support-row selector; no second HPG implementation
is permitted.  No configuration, state, physics, or threshold changes.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R42-P1 | The admitted Round-41 stream remains observationally passive and complete. | Both record digests, both inherited RHS-family streams, the inherited stage-2 HPG stream, and all four restarts remain bit-exact; all five acquisition plants retain firing markers. | Any moved byte, missing field, or failed admission control; stop for record drift. |
| R42-P2 | Round 41 reproduces exactly at this parent. | HPG remains first on the 64 disputed rank-1 U rows at 1,753 / 1,754 wet layers and maximum `1.4862887125471208e-17` m/s2; LDF/VOR/KEG/ZAD add zero residual cells. | Any count, maximum, first boundary, or residual-identity movement; stop for instrument drift. |
| R42-P3 | The existing shared literal recurrence replays NEMO's recorded stage-1 HPG statement boundaries bit-for-bit from NEMO's recorded inputs. | Recorded-input `zhpi`, `zuap`, and sum rows are 0 unequal on every scored owned wet cell on both ranks. | Any nonzero row; the scorer or its layout is wrong, so no candidate attribution is reportable. |
| R42-P4 | On the 64 registered rank-1 U rows, `rhd`, `e3w`, `gdept_z0`, and `r1_e1u` are bit-exact; the first non-bit boundary is the compiled `zhpi` accumulation, before `zuap` and the final sum. | All four inputs are exact and `zhpi` is the first unequal registered statement. | Any input is unequal, or `zhpi` is exact; the first observed boundary replaces this prediction and the failed prediction remains REFUTED. |
| R42-P5 | No production statement lands unless one isolated recorded operand or statement substitution closes all 64 final HPG rows and the full ORCA2/GYRE gates pass. | One-variable substitution reaches 0/64 and all required trajectory gates pass. | Any residual remains, more than one substitution is required, or a gate worsens; hold at the first measured statement. |

The gate must refuse missing or non-finite fields; score only owned active
cells; prove the recorded-input replay and first-boundary selector with
independent one-ULP plants; and print CPU, fp64, scalar-libm, record digests,
and the exact 64-row support census.

## Choices

ASKED: walk the admitted stage-1 HPG inputs and compiled statements in order.

UNASKED: none.
