# NEMO testcase Lane 4 — ORCA2 card round 12 transcription receipt

Date: 2026-09-23

Starting tip: `4dfdd5a1cebc080e23eb38ff1ad31cbd57c19264`

Preregistration: `ac8b2539f`

Status: **LANDED-READY.**

Two things happened, and the second is not the one the round's order expected.

**A — the lateral-viscosity operator is now FULLY explained.**  Round 11 left
2 to 4 percent of the disagreement with no owner.  It has one: legoESM's
stored VERTEX-CELL AREA, which is the exact spherical-cap area while NEMO
stores `e1f*e2f`, and on ORCA2 the two differ by a MEDIAN of 5.4 percent.
With that and one smaller statement added to round 11's two, the compiled
loops reproduce the production operator to **3.1e-16 (u) and 2.7e-16 (v)** in
L2 — machine precision.  Nothing from statement A is landed: Decision 54 is
pending, and the decision is now a THREE-way one.

**B — the river-runoff tracer source is transcribed, gated bit-exact, and
LANDED — and it is NOT the owner of the ladder's first disagreement.**  The
ladder's attribution of its whole stage-1 temperature row to this statement
was a hard-coded mapping, and it is too broad: 191,282 of the 233,341
disagreeing cells lie off every runoff column, and adding the channel leaves
that count unchanged.  Round 13's order follows from that.

No configuration, selector default, tunable, threshold, cadence, resolution,
timestep, carried state, data source, NEMO source or sea-ice registry entry
changed.  Sea ice remains out of scope and the six-entry `unmeasured_features`
tuple is unchanged.

Independent review **not run (codex is paused)**; a fresh adversarial reviewer
was run in its place (section 8).

## 1. What the record fixes, and what nothing here may choose

| resolved setting | value | where it is printed |
|---|---|---|
| lateral viscosity operator family | div-rot, `nn_dynldf_typ = 0`, laplacian, iso-level | run `ocean.output:1176-1180,1193,1195` |
| viscosity coefficient source | `ahmt_3d`/`ahmf_3d` read whole from `eddy_viscosity_3D.nc` | run `ocean.output:1184,1197,1200-1201` |
| lateral momentum boundary condition | no-slip, `rn_shlat = 2.0` | run `ocean.output:339` |
| river runoff | ON | run `ocean.output:534` |
| runoff river-mouth treatment | `ln_rnf_mouth = T`, `rn_hrnf = 15 m`, `rn_avt_rnf = 1e-3` | run `ocean.output:648-650` |
| runoff multiplier | `rn_rfact = 1.0` | run `ocean.output:651` |
| runoff source | `runoff_core_monthly`, variable `sorunoff`, monthly climatology | run `ocean.output:657-660` |
| runoff DEPTH spreading | NOT selected | neither `ln_rnf_depth`'s nor `ln_rnf_depth_ini`'s banner appears in the run's `ocean.output` |
| runoff temperature / salinity from a file | NOT selected | neither banner appears |
| ocean time step | `rn_Dt = 10800` s | run `ocean.output:217` |

The record is the pinned ORCA1-ice reference run
`orca1ice_surface_entry_every_step_a_np2`.  **No new NEMO run was made or
needed**: `rnf`, `rnf_b`, `rnf_tsc` and `rnf_tsc_b` are recorded in every
surface frame the round-5 acquisition wrote.

A correction to the round's own order: there is **no `dynldf_lap_blp.f90`** in
this build.  The compiled owner of both the laplacian and the bilaplacian
level operators is `dynldf_lev.f90`.

## 2. Statement A — the last few percent, named

### 2a. The three candidates, and why only three

Reading the two sides statement by statement, exactly three legoESM behaviours
were left unsubstituted by round 11.  A fourth candidate round 11 listed —
NEMO's stage-three `Kbb`/`Kmm` split — is **excluded by construction and not
measured**: the gate drives both sides with ONE recorded sea surface, so no
such difference can exist inside it.

| ID | statement | compiled owner | result |
|---|---|---|---|
| A-e3t | the OUTER DIVISOR of the divergence term | `dynldf_lev.f90:127` | **VACUOUS on the scored set** |
| A-curl-edge | the zonal edge length of the circulation loop | `dynldf_lev.f90:123-125` | **small** — 0.2 percent of the residual |
| A-metric | the stored horizontal metrics | `domain.f90:197-205,212-215` | **THE OWNER** — 93 percent of it |
| A-slope | the slope-foot factor on the operator's output | (no NEMO statement) | **VACUOUS** — the factor is the scalar 1.0 |

### 2b. The numbers

Every row is scored against the PRODUCTION operator, on ONE cell set, at kt=2
(kt=1 is rest and cannot score a momentum operator — round 11's finding).
Round 11's rows were re-run with the new rows added and **reproduce exactly**,
so the extension did not change the instrument.

| row | u, L2 | v, L2 |
|---|---|---|
| round 11: the production operator against the compiled loops | 0.7681 | 0.6136 |
| round 11 closure — the extra vertex mask and the min-rule thicknesses | 0.02669 | 0.01904 |
| + A-e3t (legoESM's live cell thickness as the divisor) | 0.02669 | 0.01904 |
| + A-curl-edge (the cell metric on the circulation's zonal edges) | 0.02665 | 0.01902 |
| **+ A-metric (legoESM's stored horizontal metrics)** | **0.001850** | **0.001405** |
| **all of them together** | **3.06e-16** | **2.72e-16** |

**The operator's disagreement is therefore 100 percent attributed.**  That
also certifies, in one measurement, the transcription, the index map, the
coefficient read and all four named statements together: nothing is left over.

### 2c. Which metric, and by how much

Five of the six metrics the compiled statements read are **bitwise** the
record's own mesh-mask arrays.  Exactly one is not.

| metric role | cells unequal | max relative |
|---|---|---|
| cell area `e1t*e2t` | 0 / 26,640 | 0 |
| u-face zonal length `e1u` | 0 / 26,640 | 0 |
| u-face meridional length `e2u` | 0 / 26,640 | 0 |
| v-face zonal length `e1v` | 0 / 26,640 | 0 |
| v-face meridional length `e2v` | 0 / 26,640 | 0 |
| **vertex-cell area `e1f*e2f`** | **26,456 / 26,640** | **14,007** |

The vorticity bracket divides by that area, so the error passes straight into
the tendency.  It is not a corner case: the **median** relative difference is
**5.4 percent**, 26,376 cells exceed 0.1 percent (16,269 of them on a wet
column) and 3,335 exceed 10 percent.  legoESM builds the vertex area as the
EXACT spherical cap
(`packages/core/legoesm/grids/operators_latlon_cgrid.py` `_vertex_dual_area_interior`);
NEMO stores the product of two midpoint metrics.  The module's own docstring
already recorded that the two conventions differ — it estimated the gap at
4.1e-05, which is right for a regular lat-lon grid and wrong by three orders
of magnitude for this one.

### 2d. The two that are vacuous, and what that means

**A-e3t.** legoESM's live cell thickness differs from NEMO's
`e3t_0*(1+r3t*tmask)` on 368,648 of 799,200 cells, by up to 500 m — and
substituting it changes the operator on NOT ONE cell.  Every cell where the
two disagree carries a viscosity coefficient of exactly zero, so the divisor
never multiplies anything.  Reported as vacuous ON THE SCORED SET, not as
agreement.

**A-slope.** `slope_foot_alpha` is 0.0 on this card, so both factors are the
scalar 1.0 and the operator's output is returned unmultiplied.  Settled by
reading the value, not by a substitution, because at alpha = 0 there is no
array to substitute.

### 2e. Nothing from statement A lands, and the decision has grown

Round 11 raised Decision 54 as mask-only versus both halves.  It is now a
THREE-part fix, and the third part is the largest of the three that can be
scoped:

| part | can it be scoped to ORCA2? | why |
|---|---|---|
| drop the second zero/one vertex mask | yes | only the file-sourced coefficient already carries NEMO's mask, and only the ORCA2 card selects that source |
| use NEMO's stored face and vertex thicknesses | no | GYRE selects the same `nemo_e3` weighting through the same operator |
| use NEMO's `e1f*e2f` as the vertex area | no | every lat-lon card's vorticity divides by the stored vertex area |

## 3. Statement B — the river-runoff tracer source

### 3a. What NEMO executes

| statement | compiled owner | what it says |
|---|---|---|
| runoff is on | `sbcrnf.f90:184` | `IF( ln_rnf ) THEN` |
| the temperature content | `sbcrnf.f90:221` | the runoff enters at the sea surface temperature, floored at 0 |
| the salinity content | `sbcrnf.f90:227` | `zrnf_sal * rnf * r1_rho0`, and `zrnf_sal` is 0 (`sbcrnf.f90:175`) |
| how deep it goes | `sbcrnf.f90:487-488` | the surface arm: one level, over the LIVE top-cell thickness |
| the deposit | `trasbc.f90:318-326` | the reciprocal of the depth is formed FIRST, then multiplied |
| which stages | `trasbc.f90:278` | the block sits OUTSIDE the stage switch, so all THREE stages |

### 3b. What legoESM already had, before anything was written (Rule 4)

Searched `runoff` and `rnf` across `packages/ocean/legoesm/ocean/`.  Found:
the runoff MASS channel into the horizontal divergence
(`ocean_pe_latlon_cgrid.py:1563-1565`, cited to `sbcrnf.F90:253-260`), the
NEMO depth-spreading virtual-salt helper and its argument resolver
(`freshwater.py:485` and `:442`), and the MPAS and lat-lon callers of both.
NOT found: any channel carrying the runoff's TRACER content.  The deposit is
therefore an extension of the existing surface-forcing channel set, not a
second runoff path.  **The mass side is deliberately NOT switched on**: it
forces the sea surface and the horizontal divergence and is a separate
statement, recorded in OPEN.

### 3c. The gate, and why it is not circular

The gate runs the PRODUCTION step twice on the record's own kt=1 state — once
with the recorded runoff content supplied and once with it withheld — and
reads the per-stage tracer source rates the production stage helper consumes
out of the live-operand trace, which now carries them.  The compiled statement
says the first must equal the second PLUS `rnf_tsc * (1/h_rnf)`, with
`rnf_tsc` read from the RECORD and `h_rnf` the live top thickness that same
stage divided by.  The only legoESM-supplied operand is that thickness, and it
is gated separately.

| row | result |
|---|---|
| stage 1, temperature and salinity | **0 / 799,200 unequal each** |
| stage 2, temperature and salinity | **0 / 799,200 unequal each** |
| stage 3, temperature and salinity | **0 / 799,200 unequal each** |
| the runoff depth operand — legoESM's live top thickness against NEMO's own | **0 / 16,433 wet cells unequal**, max 0.0 |
| control — the stage-2 thickness fed to the stage-1 row | **REFUSES**, 3,400 cells |
| the gate at the round's BASE commit | **REFUSES** — "the ladder's surface forcing carries no runoff tracer content" |
| one-representable-value plant | **FIRES** — "PLANT FIRED: the gate refuses a one-representable-value move" |
| gate exit at the tip | **0 — AT BAR** |

Label: **given NEMO's entry** (the kt=1 recorded state and the recorded
runoff frames).

### 3d. What the record says about the statement itself

| quantity | value |
|---|---|
| surface cells carrying a non-zero runoff | 4,649 of 26,640 |
| of those, cells whose runoff carries HEAT | 3,400 — the other 1,249 sit where the sea surface temperature is at or below 0 degC, which the compiled `MAX(sst_m, 0)` floors away |
| largest runoff temperature content | 1.5424e-05 K m/s |
| largest runoff SALINITY content | **exactly 0.0** — `zrnf_sal = 0` |

So the runoff adds heat at 3,400 surface cells and **no salt anywhere**.
