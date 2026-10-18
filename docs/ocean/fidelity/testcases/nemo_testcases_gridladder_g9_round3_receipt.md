# Grid ladder rung G9 — round 3 receipt (gridladder_r48)

Status: **LANDED, G9 NOT CERTIFIED.**  Two NEMO statements, G9-selected
only.  G9's kt-1 velocity debt is gone: u and v are exact at kt 1, and
on every given-NEMO-entry step (kt 1, 3, 5, 7, 9) at every level.  What
remains is the kt-1 stage-3 T/S last place (under the bar) carried into
the next step.

- Record: round 46's (`gridladder_rounds/round46/oracle_g9_r46/twin_a`),
  unchanged.  No acquisition: the record decided every question.
- Preregistration `1d2604c18`, before any measurement.  Landing `d2d43d493`,
  tests `93a8f07eb`, re-anchor `02fad8056`.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.
- Evidence: `gridladder_rounds/round48/` (manifest `manifest.sha256`).
- NEMO lines are cited in ORCA2_OMIP_L4; the record build
  ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 has `stprk3_stg.f90`, `sshwzv.f90`,
  `lbcnfd.f90`, `domain.f90` byte-identical (`cmp`).

## 1. Item 0 — NEMO's end-of-stage fold

G9 runs `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795` with T and S on
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:796`: one `lbc_lnk` of uu (U, −1),
vv (V, −1), T and S (T, +1), `ldfull`, after every stage.  On the T-pivot
fold it rewrites the right half of the stored pivot row from its left half
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:625` for T,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/lbcnfd.f90:670` for U).

| field | card before this round | now |
|---|---|---|
| vv | folded (top V row = image) | unchanged |
| uu, T, S | not folded | pivot half-row after stages 1, 2, 3 |

Landed with the existing shared pivot half-row helper, one new model
function, new field `rk3_end_of_stage_pivot_fold` (default False, G9 only).

Pivot-row self-mirror breaks, wet pairs, all levels (independent label,
NEMO's surface input).  NEMO: 0 at stages 1, 2, 3 and in every restart,
every kt.

| kt | card T / S / U before | card T / S / U now |
|---|---|---|
| 1, 2 | 2 / 0 / 0 | 0 / 0 / 0 |
| 3–5 | 0 / 4 / 0 | 0 / 0 / 0 |
| 6 | 2 / 4 / 0 | 0 / 0 / 0 |
| 7 | 4 / 6 / 0 | 0 / 0 / 0 |
| 8 | 102 / 14 / 96 | 0 / 0 / 0 |
| 9 | 126 / 26 / 96 | 0 / 0 / 0 |
| 10 | 146 / 34 / 142 | 0 / 0 / 0 |

The before arm reproduces round 47's T/S counts exactly (instrument check).
"Now" is measured at the landing commit (both statements).  The fold alone
was not scored separately; section 4 scores both together.

## 2. Preregistration outcome

| prediction | outcome |
|---|---|
| P0a card mirror breaks 0 every kt | CONFIRMED (table above) |
| P0b no status worsens | CONFIRMED (section 4) |
| P0c kt-1 v debt unmoved by the fold | not separable: scored with both statements |
| link: NEMO's stage-1 T/S make stage-2 v exact | CONFIRMED: 483 → 0 cells |
| first statement = stage-1 `zFw` | CONFIRMED |
| mechanism = XLA reciprocal/FMA contraction | **REFUTED**: an algebraic shortcut in the card |

## 3. The walk (kt 1, offline replays and one-variable substitutions)

1. Link.  Stage 2 entered with the card's own stage-1 u, v, ssh and NEMO's
   stage-1 T, S: stage-2 u 394 → 0, v 483 → 0 cells.  Control (card's own
   T, S substituted): reproduces 394 / 483 bit for bit.  One G9 flag that
   refuses the stage override is tracer-only; turning it off leaves stage-2
   u, v bit-identical (checked).
2. Stage-1 tracer update.  Card zFu, zFv exact; zFw off on 355,104 cells
   (last place).  NEMO's zFw alone in the card's stage-1 tracer update:
   stage-1 T 584 → 0, S 115 → 0.  Control (card's own triplet): unchanged.
3. zFw.  The card's two shared wzv helpers, jitted, on NEMO's own operands
   equal NEMO's literal statements on every cell: no XLA defect.  The card
   fed them a different pair.  NEMO's stage 1 uses rDt = rn_Dt/3
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:134`) and
   r3t(Kaa) = r2_3·r3t(Kbb) + r1_3·r3ta
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`), read by
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297`.  The card used the N+1
   ratio over rn_Dt: equal as real numbers.
   Offline literal replay: NEMO's pair matches NEMO's zFw off 4,709
   columns (my replay omits runoff; 4,649 of them carry runoff).  The card's
   pair misses 352,652 cells (card itself: 355,104).
   Stage 2 is the same statement: rn_Dt/2
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:188`) with
   r1_2·(r3t(Kbb) + r3ta)
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`, the np_HYB branch).
   G6's existing stored-ratio option does not fix it (355,104 → 352,010).

Landed in the ONE shared wzv: new argument `stage_after_level` takes the
stored blend through the existing `nemo_stage_ratio`; the stage transport
passes that stage's rDt.  New field `rk3_stage_wzv_nemo_clock_pair`
(default False, G9 only).  Searched first: `nemo_stage_ratio`,
`stored_kmm_level`, `half_step_after_ratio`, `stored_r3_ratios`,
`source_stage_wzv_clock_arm` — the blend helper existed; no stage-clock
pair existed for the transport wzv (only a private hook for the clock).

| kt-1 quantity (cells off NEMO) | before | now |
|---|---|---|
| stage-1 zFw | 355,104 | 0 |
| stage-2 zFw | 165,986 | 73 |
| stage-1 T / S | 584 / 115 | 0 / 0 |
| stage-2 u / v / T / S | 394 / 483 / — / — | 0 / 0 / 0 / 0 |
| step-end u / v | 66 / 59 | 0 / 0 |
| step-end T / S (≤4.7e-16) | — | 60,401 / 1,414 |

Next owner, located: the stage-3 tracer update.  Stage-3 transports are
off (zFw 102,463 cells) but NEMO's triplet substituted leaves step-end T/S
at 60,162 / 1,342 of 60,401 / 1,414, so they do not own it.

## 4. Score at `d2d43d493` (both labels, kt 1..10)

Row = ssh, uu_b, vv_b, u, v, T, S (level 1 or 2-D).  Normalised max.
Flushed subnormals 0; dry-cell mismatches 0.

| label | kt | exact / AT-BAR / DEBT | worst row | u, v rows | round 47 worst |
|---|---|---|---|---|---|
| independent | 1 | 5 / 2 / 0 | T 2.4e-16 | exact | v 1.46e-15 |
| independent | 2..5 | 0 / 0 / 7 | 3.1e-14 → 8.2e-14 | ≤3.7e-14 | 3.4e-14 → 8.2e-14 |
| independent | 6..10 | 0 / 0 / 7 | v 7.6e-14 → 2.0e-13 | — | v 7.4e-14 → 3.5e-13 |
| given-NEMO-entry | 1, 3, 5, 7, 9 | 5 / 2 / 0 | ≤3.5e-16 | exact | u, v ≤8.7e-15 |
| given-NEMO-entry | two-step 2..10 | 0 / 0 / 7 | ≤3.1e-14 | — | ≤4.0e-14 |

All-level rows on the given entries (kt 3, 5, 7, 9 restarts): u, v exact
(0 cells), T ≤4.7e-16, S ≤3.5e-16, all AT-BAR.

Against round 47's resume score, 212 rows: statuses AT-BAR 41 → 58, DEBT
171 → 154, no status worsened; 18 rows identical, 87 lower, 59 count-only,
48 higher.  Every higher row is a chained row (independent kt ≥6 or a
two-step), DEBT both sides; largest independent kt-8 u 5.7e-14 → 1.0e-13.
The first over-bar step moves kt 1 → kt 2.

Score SHA-256 (first 16): `indep_kt1_5.json` `95e9956f19818050`,
`indep_kt6_10.json` `649a85c5d4130699`, `given_e135.json` `f61b6ba16aaa154b`,
`given_e79.json` `37cf5692244c6c3c`.  Row diff `score/cmp.txt`.

**Not certified.**  So this does not yet certify, as a chain, the stage-1 V
transport mask, the two round-46 exchanges and the round-47 FCT fold
statements.  What it does show: with them and this round's two statements,
every single step from NEMO's state reproduces NEMO's u and v exactly at
every level and T, S within the bar, fold rows included.

Other cards: traced step jaxpr, addresses masked, unchanged — G5
`339a8016d262c4f6`, G8 `03ccdb94113eab61` (`round48/jaxpr/`).

## 5. Tests and plants (CPU, fp64, jit, state traced)

- `test_g9_every_stage_ends_with_a_self_mirrored_pivot_row`: kt-1 step from
  an entry with an asymmetric pivot row (right half T, S ×(1+1e-6), u +1e-3);
  after stages 1, 2, 3 the T, S, U pivot row is self-mirrored, indices
  written from NEMO's loops.
- `test_g9_stage_transport_wzv_reads_nemos_stage_clock_pair`: kt-1 stage-1
  zFw equals NEMO's record on every cell; stage 2 at most 100 cells.

Plants: scratch worktrees at `93a8f07eb`, one wiring line removed
(`round48/plants/*.log`, diffs beside them):

| copy | result |
|---|---|
| restored | 2 passed |
| stage-1 fold call off | failed at stage 1 |
| stage-2 fold call off | failed at stage 2 |
| stage-3 fold call off | failed at stage 3 (T 1,366 / S 1,366 / U 1,294) |
| U arm off | failed (U 1,294) |
| T arm off | failed (T 1,366) |
| S arm off | failed (S 1,366) |
| stage-1 pair off | failed (stage 1: 355,104 cells) |
| stage-2 pair off | failed (stage 2: 165,986 cells) |
| clock only removed | failed (stage 1: 430,552) |
| blend only removed | failed (stage 1: 430,552) |

## 6. Gates (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH; `round48/final/`)

| battery | result |
|---|---|
| G9 file | 18 passed |
| G2 ladder file + recipe unit tests | 36 passed |
| citation-gate test, private-import ratchet, TKE | 91 passed |
| freshwater, barotropic state | 24 passed |
| face-mask rank (`-n 2`) | 2 passed |

Before the citation-gate test passed: the code edits shifted 98 lines of
older citations (77 map lines, 21 in one older receipt), re-anchored with
the existing tool.  Receipt citation gate on this receipt: section 7.

## Choices

ASKED: the four items (brief); G9's record; the bar and labels (D127).

UNASKED, offered for revert:
- both new fields default False, only G9 selects them (campaign
  no-silent-default rule).  One-line question for the ORCA2 lane under
  Decision 122: import both, or keep them G9-only?
- the stage-2 wzv pair lands with stage 1 (same statement at stage 2;
  leaving it would knowingly keep 165,986 stage-2 zFw cells off);
- the fold and the clock pair were scored together, not separately;
- the link test turned off one tracer-only G9 flag that refuses the
  stage override (stage-2 u, v shown bit-identical with it on and off);
- the fold test's input is constructed (NEMO's entries are self-mirrored).

UNVERIFIED: the stage-3 tracer statement owning the remaining kt-1 T/S
last place; the 73 residual stage-2 zFw cells (stage-2 transports zFu 28 /
zFv 9 cells were already off in round 46, not walked).

## 7. Receipt citation gate

`--from-heading` this title, at `b08ef9373`: `PASS`, 9 citations, 0 failures,
0 unmapped, 0 map entries failing the audit (`round48/final/`).  Its shift
plant on the rn_Dt/3 citation exits 1.  Two map entries added (the stage
rDt lines).  Citation-gate test, private-import ratchet, TKE re-run after
them: 91 passed.

Evidence manifest `gridladder_rounds/round48/manifest.sha256` (pickles and
arrays excluded).
