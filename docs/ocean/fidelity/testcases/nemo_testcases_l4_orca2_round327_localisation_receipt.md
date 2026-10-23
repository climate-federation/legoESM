# ORCA2 round 327 — the independent sea-level lead is inherited from entry

Date: 2026-10-23

Base: `6a465f349e367478bdee4bcdab2a5de570be931c`

Disposition: **LOCALISATION COMPLETE; no model, card, or configuration
change**

## Answer

**CONFIRMED.**  On the current comparison card (implicit ice-ocean top drag,
L3/L9 eddy-induced velocity, and `nn_mxlice = 2`), the independent label first
leaves the exact numerical bar at **kt 1, entry, SSH**.  T/S/u/v are exact;
SSH differs on `16,433 / 26,640` cells, maximum
`0.015479333813968585 m`, RMS `0.012157480547162937 m`.

At that identical row the given-NEMO-entry label is exact in all five fields:
T/S/u/v each have `0 / 799,200` unequal cells and SSH has `0 / 26,640`.
Its first positive-magnitude departure is later, at kt 1 stage 3 T:
`8,843 / 399,600`, maximum `1.973531867094369e-07 K`, RMS
`1.237023269820231e-09 K` (first zero-based cell `[5,67,0]`).  Signed-zero
bit differences before that row do not leave the numerical bar.

The independent excess is therefore **INHERITED_ENTRY**, not created by an
ocean-step statement.  Replacing only the card SSH with the passive completed
NEMO entry SSH makes T/S/u/v bit-identical on all `799,200` cells apiece and
SSH bit-identical on all `26,640` cells.  No executable observer and no NEMO
run entered this replay.

## Cell localisation

**CONFIRMED.**  All and only the `16,433` wet T cells differ, by one uniform
signed value: candidate minus NEMO is
`+0.015479333813968585 m`.  The first cell is also the maximum cell because
the offset is uniform:

| property | measured value |
|---|---:|
| zero-based `(j,i)` | `[1,49]` |
| latitude / longitude | `-77.77420179896262 deg`, `177.9998921261904 deg` |
| depth | 2-D surface SSH; no vertical index |
| recorded ice fraction / covered | `0.7914190462629991` / yes |
| coastal | yes |
| fold row / zonal seam | no / no |

The full mismatch partitions are: ice-covered `3,629`, not ice-covered
`12,804`; coastal `1,627`, interior `14,806`; fold row `70`, off-fold
`16,363`; zonal seam `136`, off-seam `16,297`.  “Coastal” is preregistered as
a wet T cell with a dry four-neighbour under periodic-i and tripole-north
connectivity.  These categories localise the uniform global adjustment; they
do not imply a local ice, coast, fold, or seam mechanism.

## First owner

**CONFIRMED.**  The first owner is the record-producing NEMO build's
pre-step, domain-global levitating-ice mass adjustment.  It sums category
snow/ice/pond mass, computes one global `zsshadj`, and subtracts it from both
SSH time levels.  legoESM independently constructs a different initial ice
identity and its own global SSH adjustment.  The one-field owner-output replay
closes all five entry fields exactly, and the given-entry arm is already exact
there.

Certified rung: **none**.  G5 certifies the ice-ocean drag statement after it
receives a completed ice/ocean state; it does not construct SI3's initial
category state or this pre-step SSH adjustment.  Thus the source attribution
is confirmed by the source-order boundary and exact replay, while the physical
reason the two initial ice identities have different total mass is
**PLAUSIBLE** rather than newly decomposed here.

## Remaining rung-10 owner queue by size

The ranking below is the maximum `kt=10` full-domain entry error for each
field family across both labels.  Magnitudes are **CONFIRMED**.  Only SSH has
a confirmed statement owner in this round; assigning the compounded T/u/v/S
endpoints to a particular statement would be inference, so those owner labels
remain **PLAUSIBLE / UNMEASURED**.

| rank | family | independent max / RMS / unequal | given-entry max / RMS / unequal | owner status |
|---:|---|---|---|---|
| 1 | SSH | `0.018533089817820425 m` / `0.012157723718584366 m` / `16,433 / 26,640` | `3.734978452696325e-05 m` / `1.1036518664249012e-06 m` / `16,433 / 26,640` | **CONFIRMED** initial ice-mass SSH adjustment for the independent carry; later given residual unwalked |
| 2 | T | `0.011429686050284715 K` / `2.107759120115402e-05 K` / `430,552 / 799,200` | `7.703006966558457e-05 K` / `3.4648985521996326e-07 K` / `430,508 / 799,200` | **PLAUSIBLE / UNMEASURED** compounded tracer family |
| 3 | u | `0.005295100146905263 m/s` / `1.4207404570554471e-05 m/s` / `415,083 / 799,200` | `0.003046847350500703 m/s` / `4.908301491048144e-06 m/s` / `415,083 / 799,200` | **PLAUSIBLE / UNMEASURED** compounded momentum family |
| 4 | v | `0.0031473714762187688 m/s` / `1.1837433234154354e-05 m/s` / `419,256 / 799,200` | `0.003149083070336161 m/s` / `7.115095569073884e-06 m/s` / `419,256 / 799,200` | **PLAUSIBLE / UNMEASURED** compounded momentum family |
| 5 | S | `0.0018793894167785652` / `4.226871881693567e-06` / `430,551 / 799,200` | `1.2237551899829668e-05` / `6.621158794775114e-08` / `429,500 / 799,200` | **PLAUSIBLE / UNMEASURED** compounded tracer family |

This table is deliberately not a statement attribution for ranks 2--5.  The
given-entry arm's exact state through stage 2 makes its kt 1 stage 3 T row the
clean next source-order boundary.

## Fix-round options and pick

1. **Bridge the independent entry.**  Reuse the recorded NEMO SSH, which
   closes this row but changes the method into the existing given-entry label.
2. **Represent the one-category SI3 initial state.**  Add sufficient card
   state/configuration to reconstruct the record's ice mass independently,
   then rerun both labels.  This is the physical fix but expands model and
   configuration scope.
3. **Keep both labels and walk the first created row next.**  Round 328 starts
   from the passive exact given-entry stage-2 state and replays kt 1 stage 3 T
   in NEMO source order until the first owning statement is identified.

Pick: **option 3**.  It preserves the meaning of both labels and attacks an
ocean-created residual without silently turning “independent” into a bridge.
Option 2 remains the later initial-state round if the operator brings SI3
initialisation into scope.

## Probe, controls, and evidence

The preregistration is commit `d80a2aa68`; the committed probe is
`b11d105cf`.  It ran CPU fp64/libm on a clean stamped tree and returned
`PASS_R327_LOCALISATION`, with R327-P1 through R327-P5 all **CONFIRMED**.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round327/`.
The result JSON SHA-256 is
`d00987c8297d36f7b098f71b69fc4e4ea2e320c3ce197b05e42cc1a0055ab140`;
the independent/given input JSON SHA-256 values are respectively
`3e122f9ad46c6c0d0758bbc3397e81ae82d74b77f1fc44cddadde030a3ea6e2a`
and `7f8905159ebd79e35cfd1ee4f6a35138d134d2b3973936ff39a566391841c062`.

All four controls refused: row-order (“independent first departure moved”),
owner-replay (“one-field owner-output replay did not close”), locality
partition (“does not partition the mismatch”), and ranking (“field-family
ranking moved”).  Their log SHA-256 values are, in that order,
`3d2eceae67a2f506d54329c96a3942baed7a997538bee1cc44bfcb5b0c9bc130`,
`edd626f21af34364c30487401b495ef7fb5b6e09128557dfbf5babf2fe504ac8`,
`5d9037e6d9653d675ad8331c28a53d7ce539dd2bef34d16c45bd8ddb3fc4f8c6`,
and `e17f9472e43d1f7b556b471802898b422fc703b6a95e6aed9101659d7adcde53`.

The first attempted measurement refused before state-array access because the
probe expected an obsolete completion-status string.  That metadata assertion
was corrected in `b11d105cf`; a subsequent invocation with an abbreviated
expected commit also refused before array access.  Neither produced a
numerical artifact.  No NEMO run occurred.

## Validation and review

The per-receipt citation gate found `2` citations, `0` unmapped citations,
`0` failures, and `0` map-audit failures.  Shifting either new citation by
two lines made its plant exit nonzero.  The landing environment's default
ORCA2 battery passed `147 / 147` in `1117.62 s`: receipt citation, private
cross-import, TKE terms, ORCA2 recipe, freshwater closure, stage-face rank,
prognostic barotropic state, and ORCA2 parallel-receipt tests.  Logs and JSON
are under `orca2_rounds/round327/gates/`.  No `packages/` file changed, so
model trajectories cannot move and the DINO month trajectory gate is not
applicable.

independent review: operator (Claude)

## Mechanically cited finding

The admitted trajectory's exact producer computes category-summed snow,
ice, and pond mass, forms a domain-global levitating-ice adjustment, and
subtracts it from both SSH time levels at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:442-459`.
The independent legoESM card constructs its initial T/S and five-category
ice-mass SSH state at `nemo_testcase_recipe.py:1834-1839`.  The exact
owner-output replay measured above confirms that this pre-step boundary owns
the first independent excess; no certified ladder rung constructs that
initial SI3 identity.

## Operator amendments after independent review (Claude LAND, Argon LAND)

- **Why the two initial ice masses differ — now CONFIRMED (was PLAUSIBLE).** The rung-10 record's NEMO run uses a
  one-category, ponds-off initial ice (its ocean.output: jpl = 1 at :702, ln_pnd = F at :785, rn_hts_ini 0.2/0.1,
  rn_hti_ini 2.5/1.0, rn_ati_ini 0.997/0.8 at :835-837, "sea level adjusted by -0.108914 m" at :855), while the
  card hard-codes ORCA2's five-category, ponds-on reference values (concentration 0.9, thickness 2.0/1.0, snow 0.2,
  ponds). Recomputing ati*(rho_i*h + rho_s*hs) on the card's ice mask gives -0.09343505872684969 m with the card's
  values and -0.10891439254081828 m with the record's, bit-exact (Claude review). Density, area, mask and category
  count are not the cause.
- **Next target corrected.** At given-entry kt 1 stage 3, u (74,603 cells, max 9.07e-6 m/s) and v (77,790 cells,
  2.46e-5 m/s) depart together with T (8,843 cells, 1.97e-7 K) and S (6,194 cells, 2.43e-8); "T first" above is only
  the field-order tie-break. NEMO's stage 3 runs dyn_zdf (stprk3_stg.f90:506) before the tracer calls (:555 on), so
  the next walk starts at dyn_zdf in source order.
- **Controls.** The owner label and the SSH replay are partly tautological and the plants mutate only the
  classifier's input; the result rests on the measured entry rows, a single unique signed offset, and the
  all-and-only-wet partition (recomputed independently by both reviewers).
