# Grid ladder rung G2 — round 4 preregistration (gridladder_r13)

Frozen before the candidate statement is measured. Base commit 8b17016ed.
Record: round10/oracle_g2_r10 (twins a, b), round11 twin c, under
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/.

## 0. Exploratory facts measured before this file (not predictions)

Step 1, entry = the exact initial state, card at 8b17016ed, scored against
NEMO's solver exit frame (ssh, uu_b, vv_b) and step frames (level-1 u, v, T,
S). Uncommitted probe; evidence round13/explore/*.json.

| arm | ssh | uu_b | vv_b | u1 | v1 | T1 | S1 |
|---|---|---|---|---|---|---|---|
| production | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| solver traced (exit only) | 0 | 0 | 0 | - | - | - | - |
| hand back the loop's entry forcing | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the forcing before the Coriolis subtraction | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the depth-mean forcing | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the 3-D right-hand side | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| production, constant folding off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| the last three hand-backs, constant folding off | all 0 | | | | | | |
| production, algebraic simplifier off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, fused instructions off (max ISA AVX) | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, new fusion emitters off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, scalar code only (prefer vector width 64) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Reading: every hand-back at or after the 3-D right-hand side restores
equality, so the cut cannot separate statements downstream of it (each
hand-back also changes what the compiler sees). Scalar-only code restores
equality with no hand-back, so the class is a VECTORISED REDUCTION ORDER,
not fusion and not an algebraic rewrite.

The first reduction on that path: NEMO's depth mean of the 3-D right-hand
side, `Ue_rhs = SUM(e3u_3d*uu(Krhs)*umask over 1:jpkm1) * r1_hu_0`
(ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179), a rank-1 SUM that
gfortran accumulates in ascending k. The card computes it with `jnp.sum`,
whose order XLA chooses. The shared ascending-k helper already exists
(barotropic_common.py, used for the stage correction only).

## 1. Candidate

Option `barotropic_rhs_ascending_level_sum` (default False, appended last),
selected by the G2 card only through its own rung table: the three sums of
the card's native depth mean (u, v, and the fold-unit v) use the existing
ascending-k helper. G1 not selected (one level: order cannot matter).

## 2. Predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| R4-P1 | with the option, production step 1 equals NEMO bit for bit on ssh, uu_b, vv_b, u1, v1, T1, S1, no XLA flag | any cell unequal |
| R4-P2 | without the option, an OFFLINE numpy replay of the statement in ascending k from the traced operands equals the traced (NEMO-equal) depth mean on every owned face, and a 4-lane split sum of the same operands differs on at least one face | ascending replay differs anywhere; or lane order never differs |
| R4-P3 | 100 steps INDEPENDENT: first step over the bar later than 16 on every field (round 3: 16) | 16 or earlier on any field |
| R4-P4 | GIVEN-NEMO-ENTRY rows stay at the bar (90 of 90) | any row over |
| R4-P5 | G1 unchanged: its 100-step certification test passes | fails |
| R4-P6 | not decided before measuring: whether the 100-step record reaches the bar on every field (certification). No direction predicted | — |

Diagnostic census (item 4): remaining non-bit rows after landing, with and
without fused instructions (`--xla_cpu_max_isa=AVX`), at steps 1, 2, 10.
The flag stays a diagnostic; no card selects it.

## 3. ASKED / choices

ASKED: name section 6's statement, land it G2-card-selected if a
transcription defect, re-score 100 steps, certify if at the bar, the stage-1
right-hand side exposure, the fusion census, one codex review. Choices of
mine, each offered for revert: the hand-back arms and XLA flags above as
diagnostics; the option name; promoting the private ascending-sum helper to
public (a cross-module private import is banned by a ratchet).
