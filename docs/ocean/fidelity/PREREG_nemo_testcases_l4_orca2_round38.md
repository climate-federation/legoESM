# NEMO testcase Lane 4 — ORCA2 card round 38 preregistration

Date: 2026-09-26

Parent: `4cda04817736a4bc48f40d4bea54e1c17a83a0c4`

Status: **PREREGISTERED BEFORE ROUND-38 SCIENTIFIC SCORING.**

Round 37 closed the given-entry lateral-diffusion replay.  This round returns
to the actual first non-bit statement, kt=1 stage-1 temperature, by resuming
round 20's earlier ranked slow-forcing producer walk.  The operator-completed
ranked record exists at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round20/acquisition/orca1ice_slow_forcing_ranked_np2`.
Before freezing this file, only filenames, byte counts, completion stamps, the
writer patch, and the compiled statements were inspected; no floating-point
payload was parsed.

Every producer number is labelled **given NEMO's entry**.  Whole-card ladder
numbers remain separately labelled **independent with Decision-52 SSH**.  The
six sea-ice selectors and the card's `unmeasured_features` tuple are frozen.

## Compiled order

The executing build first vertically averages the three-dimensional momentum
RHS with `e3u_3d`, `umask`, and `r1_hu_0` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:189-205`, then
applies baroclinic drag at `:216-222`, and finally adds wind using the stored
density reciprocal, face stress, and live inverse face depth at `:224-236`.
The WRITE-only instrument records every operand and boundary in that order on
both MPI ranks.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R38-P1 | The existing acquisition admits without repair or rerun. | Both files are 22,814,424 bytes, carry distinct rank headers/digests, completion stamps, `STOP 0`, and unique split-rank markers. | Any missing/wrong field, header, size, digest relation, or completion marker. Stop for record reconciliation. |
| R38-P2 | Round 20's boundary reproduces before the new payload is used. | The first live non-bit rank-1 source remains substep-1 `zu_frc` on 64/64 cells, maximum `7.356481146903598e-18`, and recorded `zu_frc` alone closes exit U bit-for-bit. | Any changed support, magnitude, order, or causal replay. Stop for instrument drift. |
| R38-P3 | The record replays its own compiled arithmetic bit-for-bit. | NEMO's recorded vertical-average operands replay recorded depth mean; recorded post-drag plus recorded wind operands replay final `Ue_rhs`, on both ranks. | Any replay bit differs. The reader/transcription is invalid and no producer row is quotable. |
| R38-P4 | On the 64 disputed rank-1 east-source cells, mesh thickness and mask are bit-exact; the completed 3-D U RHS is the first non-bit producer operand. | `e3u_3d` and `umask` are exact, then `uu(Krhs)` is the first differing row in compiled order. | Any earlier difference, or an exact `uu(Krhs)`. The measured first row owns the walk instead. |
| R38-P5 | Substituting only the first differing recorded boundary into the literal downstream replay closes final rank-1 `Ue_rhs` bit-for-bit. | The one-variable replay has 0/64 unequal final cells while the parent reproduces round 20. | Any residual cell or need for a second substituted family. No production statement is eligible. |
| R38-P6 | This round is HELD unless one cited production statement reaches zero unequal cells and passes the full ORCA2/GYRE landing bar. | A single-statement arm is exact, ORCA2 loses no AT-BAR row, and GYRE is byte-identical. | A multi-statement owner, nonzero residual, earlier bar loss, unregistered movement, or GYRE movement. |

Failed predictions remain **REFUTED** in the receipt.  The walk stops at the
first non-bit compiled statement.  Every comparison uses fp64/libm and the
production JIT path; one-ULP plants must make both admission-derived and
scientific rows fail.

## Choices

ASKED: resume round 20's ranked producer walk in compiled order.

UNASKED: none.  No configuration value, carried state, stabilizer, sea-ice
selector, score, NEMO source, or acquisition changes.
