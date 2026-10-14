# Grid ladder rung G2 — round 3 preregistration (gridladder_r12)

Frozen before any comparison of the card with the record in this round.
Base commit 52c1d185046d. Record: round10/oracle_g2_r10 (twins a, b) and
round11/oracle_g2_r11_consecutive/p3_c_10step (twin c) under
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/.

## 0. Admission fact found before this file (not a prediction)

The round brief says twin c's `ORCA2G1_00000001_restart.nc` is NEMO's state
after step 1. Read before anything else: it is not. NEMO's own log opens that
file at kt = 10; the file has 0 time records (215,260 bytes against
30,265,184 for a written restart). So the record holds NO all-level state
after step 1, and no consecutive pair of all-level states exists. This round
therefore cannot do "one-step given-entry rows on all levels". The check's
refusal stands; the reason it gives is corrected.

## 1. What this round measures instead (no NEMO run)

All from the existing record, entry kt = 1 (the exact initial state), scored
at kt = 2 on all levels against twin b's restart of step 2 and on level 1
against twin a's frames. The card's state after step 1 is varied in ONE way
per arm before the second step:

- A0 baseline: nothing varied (round 2: about 1.4e-13 on U).
- A1 re-entry: the state is rebuilt from the restart set only (T, S, u, v,
  sea level, barotropic pair, six histories); everything else the card
  carries is reset to its initial value.
- A2 substitution: level 1 of u, v, T, S and the 2-D fields are overwritten
  by NEMO's own entry frames of step 2 (bit-exact).
- A3 sensitivity: T on every wet cell moved by one unit in the last place,
  random sign, fixed seed; scored against A0, not against NEMO.

## 2. Predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| R3-P1 | A1 equals A0 to 1e-16 or better on u, v, sea level: nothing outside the restart set carries the term | any of them differs by more than 1e-15 |
| R3-P2 | A2 leaves the step-2 U error at 3e-14 or more: the term does not enter through level 1 or the 2-D fields | error under 1e-14 |
| R3-P3 | A3 moves step-2 U by 1e-14 or more: a last-place difference in T is enough to make the term, through the pressure gradient of a 5 km column | movement under 1e-15 |
| R3-P4 | the card's level-1 T after step 1 differs from NEMO's in some cells by exactly one unit in the last place (3.55e-15) and by no more | bit-identical, or a larger difference |
| R3-P5 | the 100-step record is not at the bar at the end of this round | at the bar |
| R3-P6 | G1 unchanged (its 100-step certification test passes) | fails |

If R3-P3 holds, the first non-bit statement is in the tracer step of step 1
and the walk goes there (stage frames of kt = 1 hold level-1 T per stage).
If it fails, the term is in u, v at depth and needs NEMO's all-level state
at an odd step: a new acquisition (restart list 3,5,7,9, which NEMO can
open; step 1 it cannot).

## 3. ASKED / choices

ASKED by the brief: fix the check, re-admit, measure the second-step term,
land a cited transcription defect G2-card-selected, re-score 100 steps, one
codex review. NOT followed, with reason: "admit the step-1 file" (section 0).
Choices of mine, each offered for revert: the four arms above; one-ulp
perturbation with seed 0.
