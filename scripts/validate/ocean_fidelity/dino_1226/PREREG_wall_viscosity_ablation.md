# PRE-REGISTRATION — the one-variable lateral-viscosity ablation at the southern wall

Written 2026-08-23, BEFORE any number from either arm existed.
Registered by: #1455 wall-friction lane. Issue: #1455.

## The question

`docs/ocean/fidelity/dino_basin_budget_result.md` Part 4 measured that legoESM
piles ~4.4 mm too much sea surface against DINO's southern free-slip wall,
decaying with an e-folding scale of 2.6 rows ~ 112 km, and that the westward
circulation lobe in the four wall rows (69.5S..68.4S) is 34% too strong while
the rest of the basin is right to 3%. The lobe is 91% geostrophic, so geostrophy
names a target, not an owner. The only discriminating number so far is the
LENGTH SCALE: 112 km is not the grid scale (1 row = 43 km) and not the
barotropic deformation radius (~1500 km), but it is within 30% of the Munk
frictional boundary-layer width computed with the oracle's own viscosity at
these rows (5387 m2/s -> 87 km = 2.0 rows). That is one coincidence and one
number, so the friction hypothesis is PLAUSIBLE and untested.

This arm tests it by perturbation, which is the only thing that can promote it.

## The one variable

`DINOConfig.U_M` (NEMO `rn_Uv`), 0.27 -> 0.54 m/s, through the new
`kamm_twin_90d.py --u-m` flag. `U_M` is read in exactly one place on this lane,
`dino.py:3020` `A_h_base = 0.5*U_M*R*dlon`, so doubling it doubles the lateral
viscosity coefficient everywhere and changes nothing else: the card sets
`B_h = 0` (the biharmonic raises if nonzero under this operator),
`barotropic_diffusion_alpha = 0`, and `A_h_eq_boost`/`A_h_cap_boost`/`A_h_floor`
are all at their no-op defaults. The runner's `smoke-check` mode asserts the
doubling reaches the built `lateral_viscosity.A_h`, so the knob cannot be
vacuous.

Everything else is byte-identical between the two arms: same commit, same
restart (`RUN_STEPDUMP/DINO_00005760_restart.nc`, NEMO day 180), same vertical
ladder (`LEGOESM_NEMO_E3T=both`), same 90 days, same snapshot days, same
`--bridge-before --surface-tendency-placement leapfrog_rhs`, same fp64 policy.
Both arms are re-run at THIS commit; no arm from an earlier commit is reused.

## (a) The number each arm produces

1. `G4` = the day-90 southern-basin zonal-transport gap (legoESM - NEMO) summed
   over the four wall rows 69.5S..68.4S (grid rows j=1..4; row j=0 is entirely
   dry). Baseline expectation from the parent document: the whole-basin day-90
   gap is -0.43 Sv and the wall rows carry ~95% of it.
2. `A_wall` = the day-90 sea-surface excess against the wall [mm] and `L_e`, its
   meridional e-folding scale [rows], from the same exponential fit
   `southern_wall_balance.py` used to report 4.4 mm / 2.6 rows.
3. The 5-metric 90-day acceptance gate (`acceptance_gate_90d.py`), including the
   circumpolar channel transport, which is the campaign's crown jewel and is
   reported for BOTH arms whatever the verdict.

## (b) What confirms and what refutes

Munk scaling: the frictional boundary-layer width goes as the cube root of the
viscosity, so doubling widens it by 2^(1/3) = 1.26x. A boundary current whose
transport is set upstream and whose width is set by friction then has its peak
amplitude scaled by 1/1.26, i.e. -21%; the parent document registered "-26% for
a doubling" from the same cube root. Both readings live in the same band, so the
registered bands below are deliberately wider than the gap between them.

* **CONFIRM (friction owns the wall scale)**: `|G4|` moves by **>= 15%**
  between the arms, OR `L_e` moves by **>= 15%**. Direction is recorded but is
  not part of the pass condition: a shrink toward zero is the Munk-predicted
  sign, a growth means the lateral operator still owns the lobe with the
  opposite sign, and either way friction is the live lever.
* **REFUTE (friction exonerated)**: BOTH `|G4|` and `L_e` move by **< 5%**.
  Then the wall lobe is not set by the lateral-friction layer, Task 2 (a wall
  treatment fix) does NOT run, and the barotropic free-surface solve returns to
  the top of the candidate list. This is to be reported loudly.
* **NO VERDICT**: any outcome between 5% and 15% on both quantities. Reported as
  no verdict; no candidate is promoted or retired on it.

Noise floor for the comparison: the #1492 item-2.1 micro-ensemble floor at day
90 is 0.091 Sv on the circumpolar transport; the wall-row gap has no published
floor of its own, so the arm reports `|G4_2x - G4_1x|` next to 0.091 Sv as an
order-of-magnitude sanity bound and does not claim a move smaller than that.

## (c) Why no cheaper offline test answers it

The quantity under test is an EQUILIBRATED boundary-layer width, not a
single-step tendency. An offline tendency comparison on a saved state (which
this lane also runs, as the alignment table's numeric row) measures the
operator; it cannot measure the layer the operator sets up over days, and the
parent document's own stage and per-term tables are closed by a 310:1
cancellation, so no bookkeeping decomposition of the existing states can
substitute. The cost is two 90-day twins at ~215 s of wall clock each,
i.e. ~0.12 GPU-hours total, which is below the 1 GPU-hour threshold that would
require this justification at all; it is written because the rule is that every
run is pre-registered, not because the cost is large.

## Recorded before the fact

Nothing about either arm's outcome is known at the time of writing. If the
result lands in the NO VERDICT band it will be reported as such rather than
re-cut against a new threshold.
