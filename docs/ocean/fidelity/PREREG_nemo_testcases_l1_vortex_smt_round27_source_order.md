# Preregistration — VORTEX_SMT round 27 (lane round 239): HPG source order

Frozen before changing production or scoring a candidate trajectory. Base:
`fef682255` (round 238 / VORTEX_SMT round 26). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round239/`.

## Compiled statement read first

The admitted SMT-4 build is
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3`. Its compiled stage program calls HPG first,
then VOR, then vector advection at `stprk3_stg.f90:328-344`; the vector
advection dispatcher calls KEG before ZAD at `dynadv.f90:134-139`. The z-level
HPG routine assigns, rather than accumulates, `Krhs` at
`dynhpg.f90:314-335`. Round 238's production-JIT walk measured the consequence:
ordinary legoESM has already associated KEG with the HPG frame, while its
existing private NEMO-source-order arm makes both HPG rows bit-exact.

Pre-implementation search found the existing source-order implementation,
the common 50-row trajectory gate, the common 100-day scorer, and round 193's
one-variable candidate diff. This round reuses those paths. The candidate is
exactly round 193's two-line routing change: stages that use NEMO's vector
velocity update select the already-tested source-order accumulator. No card,
scheme, coefficient, state field, or public option is added.

## Frozen predictions and falsifiers

* **R27-P1 — one variable and local proof.** The candidate diff changes only
  the selection of the existing HPG -> VOR -> KEG -> ZAD accumulator for
  vector WS-RK3 stages. The round-238 production-JIT walk remains bit-exact at
  HPG under the candidate; its one-ULP HPG plant moves exactly one scored cell,
  prints `STATUS PLANT-FIRED`, and exits nonzero. Any other production diff or
  a non-bit HPG candidate row refuses the arm.
* **R27-P2 — predicted trajectory inertness.** As on the flat vector card in
  round 193, the SMT-4 kt=2 U/V normalized maxima change by less than one
  percent from `3.215977248394175e-09` / `6.826891959729742e-10`; first-over-bar
  remains kt=2 on T/u/v/ssh. The 100-day T RMS changes by less than one percent
  from `2.552708052e-04 K`. A larger move refutes inertness and is retained.
* **R27-P3 — landing gate.** The statement lands only if the first-over-bar
  row does not move earlier, all five kt=1 rows stay AT-BAR, no certified row
  is lost beyond the floor, and a strict majority of every moved 50-row
  aggregate moves toward NEMO. Every movement and the cellwise ULP ratchet are
  registered. Failure of any condition keeps production unchanged and the
  candidate HELD.
* **R27-P4 — 100-day direction.** A landing candidate must not worsen day-100
  T RMS by more than the campaign's `2e-10 K` run-to-run floor. A larger
  worsening keeps it HELD even if P3 passes. The full day-1..100 curves and
  days 1/2/5/10/20/30/60/100 table are registered.
* **R27-P5 — scope.** If the candidate lands, run the full shared-statement
  blast radius: GYRE ladder and year, the generic GYRE recipe, all VORTEX/SMT
  registries, tanks, and the private DINO month gate. If it is HELD, restore
  production and run focused source-order, trajectory, scorer, and citation
  controls only; no unchanged-card claim is made from the rejected arm.

Production JIT is authoritative. The existing scorer will be extended, not
duplicated, so a candidate 100-day run can bind its sanity check to the
candidate's own just-produced ten-step ladder. Its default certified behavior
must remain unchanged and a wrong candidate reference must fail.

## No hidden choices

No configuration, card, coefficient, timestep, run length, threshold,
stabiliser, carried state, record source, default, or acceptance bar changes.
If the measured arm requires any such choice, the round stops with
`DECISION_NEEDED`.
