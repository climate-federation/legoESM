# PRE-REGISTRATION — the nonlinear terms at the wall: KE gradient, pressure gradient, vertical advection

Written 2026-08-23, BEFORE any number in it existed. Issue #1455.
Parents: `PREREG_een_e3f_mechanism.md` (the EEN decomposition), Part 5-6 of
`docs/ocean/fidelity/dino_wall_ldf_alignment.md` (the rescore).

## Why these three, and why now

Every LINEAR candidate is closed. Bottom drag is refuted over-determined.
Lateral friction is refuted on the magnitude leg under both weightings. The EEN
vorticity flux's promotion is void — it failed its own registered enrichment
leg (1.86× / 2.81× against a 3× bar) before any correction, and closing 96-98%
of it moved every 90-day gated metric by less than one noise floor.

The forcing is identical to 2.2e-19 Pa. The deficit is a pure recirculation
(99.98% internal). So what is left is the NONLINEAR momentum terms and their
wall behaviour.

## What NEMO runs, read from the compiled source

DINO sets `ln_dynadv_vec = .true.` and `nn_dynkeg = 1` (`namelist_cfg:322-323`),
so `dyn_adv` dispatches to the VECTOR form: the horizontal advection is carried
entirely by `dyn_keg` (kinetic-energy gradient) plus `dyn_zad` (vertical
advection), with the rotational part in `dyn_vor` — already decomposed.

**`dyn_keg`, Hollingsworth branch (`dynkeg.F90:83-107`).** The KE at a T-point
is `r1_48*(zu+zv)` where

    zu = 8*( u(i-1,j)^2 + u(i,j)^2 )
       + ( u(i-1,j-1) + u(i-1,j+1) )^2 + ( u(i,j-1) + u(i,j+1) )^2

**This branch has a MERIDIONAL stencil that the standard `nkeg_C2` branch does
not.** At the first wet row it therefore reads row `j-1`, which in DINO is an
entirely dry land row whose stored `puu` is exactly zero — so the cross term
degenerates to `(0 + u(j+1))^2` rather than a symmetric pair. legoESM's
Hollingsworth (`ocean_pe_latlon_cgrid.py:1523-1546`) builds the same stencil but
fills `j±1` by EDGE REPLICATION (`concatenate([u_l[:1], u_l[:-1]])`), a Neumann
condition, not by reading a masked zero.

Those two agree only because DINO's row 0 is dry and its stored velocity is
exactly zero — which is a property of the STATE, not of the code, and is
therefore measured here rather than assumed. The gradient itself is
`-(zhke(i+1,j) - zhke(i,j))*r1_e1u` with **no mask applied to the difference**
(`dynkeg.F90:105`), so a u-face at the wall sees KE only from its own row.

**`dyn_zad` (`dynzad.F90:47-80`).** Levels 1..jpk-2 use
`-0.25*r1_e1e2u/e3u * ( zWdzU(jk) + zWdzU(jk+1) )` built from the transport
`mi(e1e2t*ww)` times `dk(u)`. The **bottom level `jpkm1` is a separate loop**
that reuses the last computed `zWdzU` alone — i.e. only the TOP face of the
deepest wet cell contributes and nothing is advected through the sea floor.
That special case fires on EVERY column, but the wall rows are the shallowest
in the basin (30-31 wet levels against 35 in the interior), so the fraction of
the column governed by it is largest there.

## The state

NEMO's `RUN_D180_1STEP_1R`, one step from the day-180 restart, single rank,
legoESM bridged from the same restart on the same vertical ladder. Identical
state on both sides; any difference is the term. No model run.

All three terms are dumped in isolation by the model itself — `keg_dump_du`,
`hpg_dump_du`, `zad_dump_du` — and legoESM exposes the KE and pressure
gradients separately from `_bc_ke_and_pressure_gradients` and the vertical
advection as its own diagnostic. So the partition is THREE-WAY, not the
two-way union the residual list assumed.

## The instrument controls, which must pass before any leg is scored

1. **The three dumps must be mutually self-consistent** — the stage-3 chain
   dump equals the sum of the two isolated dumps, bar 1e-15 relative to the
   field RMS.
   **CORRECTED AFTER REVIEW: this is NOT a known-answer check on the dump
   reader, as originally registered.** `dynadv.F90:92` dumps the RHS after
   `dyn_keg` and `:101` forms the vertical-advection dump by *subtracting* it,
   so the sum is an algebraic identity that holds for any reader, any layout
   and regardless of whether the vector-form assumption is right. It prints
   exactly zero by construction. What validates the reader is the NEMO-side
   corruption controls (shift by a row / column / level, zero the field,
   mispair the terms), each of which must land at or above the carry bar.
2. **The dry-face gate** (`wall_term_discriminators._gate_dry_faces`): every
   scored term reads land, and the Hollingsworth stencil reads it by design, so
   the stored velocity on every dry face must be exactly zero. Measured on the
   scoring path.
3. **The reassembly must reproduce the model's own diagnostics.** legoESM's
   pieces are recomputed here; they must match the published diagnostic
   fields to bit-identity before any argument is varied.

## THE THREE LEGS — scored on the statistic each bar was calibrated for

This is the lesson from the rescore, applied in advance: **a bar belongs to a
STATISTIC.** Scoring the enrichment leg with the magnitude statistic is what
turned a 1.86× into an 8.74× and promoted a candidate that never qualified.

* **MAGNITUDE** — the SIGNED, **MASS-weighted**, zonally-averaged difference on
  wall rows 1-4. This is the depth-mean acceleration the bar was derived for.
  Clears at **>= 5.9e-11 m/s²**; **< 5.9e-12** refutes the term as owner.
  The level-mean is printed beside it and is NOT scored.
* **ENRICHMENT** — `row_report`'s row-mean **absolute** difference over rows
  1-4 divided by the same over the interior, which is the statistic the
  pre-registration defines and the 3× bar was calibrated against (2.4 for the
  viscosity thickness weighting, 0.57 for the implicit control volume).
  Reported on BOTH interior sets; if they straddle the bar the leg is
  **undecided by the data** and is reported as such, not re-cut.
* **SIGN** — the wall-row difference `NEMO - legoESM`, mass-weighted, must be
  **POSITIVE (eastward) on at least 3 of the 4 rows**. legoESM's recirculating
  lobe is too strong WESTWARD, so a term that owns it must show legoESM
  accelerating more westward than NEMO, i.e. NEMO minus legoESM positive.

**CONFIRM requires all three.** Any leg failing REFUTES that term as owner.
Terms are scored INDEPENDENTLY; there is no "best of three".

## What happens next, decided in advance

* **If a term clears all three legs**: an offline shape test against the
  measured deficit fingerprint (34% amplitude on four rows, depth-uniform,
  geostrophic), then the option + one 90-day A/B under the SAME joint
  transport-and-density gate as before — the channel and all four density
  metrics must not move past their floors.
* **If all three terms match**: the honest output is the ranked residual list
  with every number measured, and the campaign escalates: **basin owner not
  found at operator level — candidates exhausted; next tier is
  trajectory-feedback experiments.** That is a result, and it is registered as
  an acceptable outcome here so it cannot be reported as a failure to find
  something.

Nothing about any outcome is known at the time of writing.


---

# ADDENDUM, written after dual adversarial review

Both reviewers reproduced the three numbers exactly and independently tried to
break them. The refutations stand; two things in this registration do not.

**The motivating mechanism was FALSE.** This registration argued that NEMO's
Hollingsworth meridional stencil reads a masked zero at the first wet row while
legoESM fills it by edge replication, so that the two agree only because the
land row stores zero. legoESM's fill returns the TRUE row `j-1` for every
scored row; it deviates only at index 0 and the last index, both entirely dry.
The stencils agree for any land value. The measurement is unaffected — it never
depended on the motivation — but the premise is withdrawn.

**A leg was missing, and it changes what two rows MEAN.** Registering
magnitude / enrichment / sign says whether a term's DIFFERENCE is big enough,
never whether the TERM is. Scored with the headroom added afterwards, the
kinetic-energy gradient and the vertical advection have less headroom at the
wall than the carry bar (0.93x and 0.62x) — they could not have owned the
deficit at 100% error, so their agreement is not load-bearing. Only the
pressure gradient (259x headroom) genuinely tests the two models. **Any future
term-comparison registration must include the headroom leg**; without it a
table cannot distinguish "the models agree" from "this term is negligible
here".

**The non-cancelling maximum is now a GATE, not a printout.** The magnitude leg
cancels in both the vertical and the zonal; a term may only be called refuted
if its largest single cell in the wall rows is also under the carry bar.

**Scope, restated because the escalation drafted from this probe was wrong.**
This is a single-state, one-step operator comparison from an identical bridged
state. It bounds the DIRECT forcing difference at day 180. It cannot bound
rectification once the trajectories separate, and it says nothing about the
terms outside the tendency function — the leap-frog recombination, the
barotropic solve, the implicit vertical solve, the after-level reconciliation,
the time filter. "Candidates exhausted" does not follow and has been retracted.
