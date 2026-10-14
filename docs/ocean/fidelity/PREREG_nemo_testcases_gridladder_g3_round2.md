# Grid ladder rung G3 — round 2 preregistration (gridladder_r15)

Base: e4e8e32df (G1, G2 certified; G3 round 1). Written BEFORE any card
number is compared with NEMO's G3 record. Decision 120 (damping switch)
is pending; this round works on the as-built deck (damping off, no hand
edits of the initial state).

## 1. The record in hand (operator's acquisition, round14/oracle_g3_r14)

- The 2-step smoke finished (G3_SMOKE_OK): NEMO's restart after step 2,
  its mesh file, its log.
- The 100-step reference run stopped inside NEMO at kt = 11. The check
  that fired is the meridional-velocity one, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stpctl.f90:245`,
  on the maximum taken at `:181`: |V| = 10.24 m/s at (ji, jj, jk) =
  (22, 84, 27), 122.0 °E 7.4 °N, a partial bottom face (e3v_0 = 183 m
  against 493 m full). |ssh| 3.85 m and |U| 3.04 m/s were inside their
  limits. Before stopping, the run wrote its listed restart at step 10.
- Neither the record binary nor the frame twins ran, so there are no step
  frames and no consecutive restarts: this round has the INDEPENDENT label
  only, at kt = 2 and kt = 10, on every level. The GIVEN-NEMO-ENTRY label
  needs the next acquisition.

## 2. The boundary (campaign practice for OMT-0 / OMT-1)

G3's record is the steps NEMO completes, kt = 1..10. kt = 11 is NEMO's
own boundary (real density, no momentum advection, no lateral friction,
no drag: nothing dissipates), registered, not a failure. G3 is certified
when kt = 1..10 is at the bar on every field, both labels.

## 3. Measurements and frozen predictions

- G3-R2-P1 (geometry). Every mesh row of the G3 card equals NEMO's G3
  mesh file bit for bit on all levels (same shipped domain file as G2).
  Falsifier: one unequal row.
- G3-R2-P2 (step 2, all levels). Every field (u, v, T, S on all levels,
  sea level, the two barotropic velocities) is at the bar (relative
  1e-15) at kt = 2 in both depth classes, and NOT bit-identical in u and v.
  Falsifier for the first half: a row over the bar. Falsifier for the
  second: u and v bit-identical.
- G3-R2-P3 (tracers follow sea level). No advection and constant
  diffusivity: T and S can move only through the vertical solve's
  thicknesses, which follow sea level. Every unequal T or S cell at kt = 2
  lies in a column whose sea level is unequal. Falsifier: one unequal T/S
  cell in a column with bit-identical sea level.
- G3-R2-P4 (step 10). At the bar on every field. PLAUSIBLE only: with no
  dissipation a last-place difference may grow; no rate predicted.
  Falsifier: a row over the bar.
- G3-R2-P5 (where). The worst u/v cells sit at partial bottom faces.
  PLAUSIBLE, no mechanism measured. Falsifier: the worst cell of u or v at
  kt = 2 is in the full class.
- Hand-edit cells (Decision 120): every unequal-cell count is also given
  inside the 1,283 T / 720 S cells ORCA2's edits would change.

## 4. What lands

Only a cited NEMO statement whose offline replay on the card's own
operands differs from the card AND whose selection moves the kt = 2 /
kt = 10 rows toward NEMO. Without stage frames a statement can be named
only PLAUSIBLE; if none is replayable from what is in hand, nothing lands
and the next acquisition (frame twins and the record to kt = 10) is the
round's product.
