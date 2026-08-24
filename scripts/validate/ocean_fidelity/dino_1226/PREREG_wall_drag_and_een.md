# PRE-REGISTRATION — the two wall-row term comparisons: bottom drag, and the EEN vorticity triad

Written 2026-08-23, BEFORE either number existed. Issue #1455.

## Why these two, and why now

`docs/ocean/fidelity/dino_wall_ldf_alignment.md` closed lateral friction end to
end: the operator matches the oracle to 1e-4, its free-slip corner mask matches
at every one of 372528 corner points, the leap-frog composition delivers the
increment at 2*dt on the before level with the depth mean arriving at the wall
rows to 0.13%, and raising the coefficient makes the basin error worse rather
than better. Its ranked residual list put bottom drag first and the EEN
vorticity triads second, sorted on whether the mechanism carries a 1/H (the wall
rows are about 2260 m against 4000 m in the interior, so every 1/H term is
amplified 1.77x there). These are those two, both offline.

## The state, and why it is the right one

NEMO's `RUN_D180_1STEP_1R`: ONE step (kt 5761) from `DINO_00005760_restart.nc`,
single rank, the SAME restart the 90-day twins start from. Both quantities are
dumped there by the model itself. legoESM is bridged from the same restart with
the same vertical ladder the twins run (`e3t_mode="both"`). So the two sides see
a bit-identical state and any difference is the term, not the trajectory.

## (1) BOTTOM DRAG — the drag-only increment to the barotropic forcing

**What NEMO computes** (`MY_SRC/dynspg_ts.F90`, `dyn_drg_init`):

    pu_RHSi += r1_hu(Kmm) * 0.5*(rCdU_bot(i+1,j) + rCdU_bot(i,j)) * (u(ikbu,Kbb) - u_b(Kbb))

with `ikbu = mbku` the bottom level, the `Kbb` branch selected because DINO sets
`ln_bt_fw = .false.` (`namelist_cfg:353`). Units m/s^2. It is dumped in
isolation as `drg_dump_zu_frc_inc.bin` (the routine snapshots `zu_frc` before
and after the call precisely so the increment can be read alone), interior
`(Nis0:Nie0, Njs0:Nje0)` = (52,199), i-fastest. `rCdU_bot` itself is dumped as
`drg_dump_rCdU_bot.bin`, full `(jpi,jpj)` = (56,203) with the nn_hls=2 halo.

**What legoESM computes** (`ocean_model_latlon_cgrid.py:3596-3610`):

    F_slow_u -= (r_u_bt / H_u_pre) * (u_bot - U_bar)

with `r_u_bt`, `isb_u` from `nemo_bottom_drag_rate_faces`
(`ocean_pe_latlon_cgrid.py:3188`) and `u_src = u_before` because the card sets
`barotropic_forcing_centred = True`. Same quantity, same units, opposite sign
convention on the coefficient (NEMO's `rCdU_bot` is negative and is added;
legoESM's rate is positive and is subtracted) — which is itself one of the
things to check rather than assume.

**Three sub-measurements, reported separately** so a composition error and a
coefficient error cannot hide in each other:
  a. `rCdU_bot` at T-points, NEMO's dump against legoESM's rate field.
  b. the bottom-level index: legoESM's `isb_u` against the mesh's `mbku`.
  c. the assembled increment, NEMO's dump against legoESM's expression.

## (2) EEN VORTICITY TRIAD — the numeric value at the wall

**NEMO**: the `dyn_vor` contribution is the Krhs chain dump difference,
`stp_dump_04_dynvor_du.bin` minus `stp_dump_03_dynadv_du.bin`
(`stpmlf.F90:314` and `:318`), each `(jpi,jpj)` per level over `jpkm1 = 35`
levels, i-fastest, halo included. DINO runs `ln_dynvor_een` with
`ln_dynvor_msk = .false.` (`namelist_cfg:333`), so the coastal relative
vorticity stays LIVE in the flux, and `nn_e3f_typ = 1`.

**legoESM**: `MomentumTendencyDiagnostics.vortcor_u` (`ocean_pe_latlon_cgrid.py:4606`),
which under `vorticity_scheme="een_total"` + `coriolis_scheme="explicit_ab2"`
is the combined `(f+zeta)` EEN flux — the same grouping — with
`een_q_boundary="nemo_live"` and `een_e3f_scheme="nemo_avg"` matching the two
namelist settings above.

## The registered SHAPE test, identical for both

A term can only carry the defect if its difference is BOTH wall-enriched AND big
enough. The fingerprint to match: a 34% amplitude error on the four wall rows
(grid rows 1-4, 69.50S..68.43S) with the rest of the basin right to 3%, carried
by a depth-uniform velocity error of 4.6e-4 m/s.

* **Wall enrichment**: the row-mean absolute difference over rows 1-4, divided by
  the same over the sampled interior rows, must be **>= 3**. (The lateral
  viscosity's thickness weighting scored 2.4 and was still three orders too
  small; the implicit-solve control volume scored 0.57 and was anti-enriched.)
* **Magnitude**: a persistent depth-mean acceleration difference `da` produces a
  velocity error of about `da * tau` over the window. With `tau` = 90 days,
  reproducing the measured 4.6e-4 m/s needs
  **`da >= 5.9e-11 m/s^2`** at the wall rows. Below **5.9e-12** (a tenth of
  that) the term cannot carry the lobe even if every bit of it accumulated
  coherently, and is refuted as owner.
* Between those, or wall-enriched but too small, or big but not wall-enriched:
  **reported as neither**, with the numbers, and no candidate promoted.

`tau = 90 days` is deliberately generous — it assumes the difference accumulates
without cancelling for the whole window, which no real tendency difference does.
It is an UPPER bound on what the term could deliver, so failing it is a strong
refutation and passing it is weak evidence.

**Whichever term clears the shape test becomes the fix candidate.** If both
clear, the larger one goes first. If neither clears, the next item is the
advection + kinetic-energy/pressure-gradient union partition, which the report's
residual list already establishes is separable today.

## The traps that apply

* **The dry-row trap** (0be305459): row 0 is entirely dry and both models store
  exact zeros there. Every reduction is over wet cells only, and the probe
  plants a value on the dry row and requires no scored number to move, and on a
  wet wall row and requires one to move.
* **Masked-stencil contamination**: both terms read a bottom-level index, so the
  probe reports the index comparison (1b) BEFORE the assembled increment, since
  an off-by-one there would make (1c) meaningless.
* **Halo and axis conventions differ between the two dump families** (one is
  interior-only, one carries the nn_hls=2 halo, both are i-fastest while
  legoESM is (j,i)). The probe asserts every shape after slicing rather than
  trusting the arithmetic.

Nothing about either outcome is known at the time of writing.

---

# ADDENDUM, written after dual review, recording what the registration got wrong

Both reviewers read the probe and the results. Between them they overturned the
reduction, the interior sampling, and four claims in the ranking that promoted
these two candidates in the first place. What follows is recorded here rather
than reinterpreted quietly.

## The reduction was wrong, and it decided both verdicts

The registration named a bar — 5.9e-11 m/s² — derived for **a persistent
depth-mean acceleration** (4.6e-4 m/s over 90 days). The probe then scored
`mean |difference|` over every wet cell and every level. Those are not the same
quantity: the mean of absolute values is an **upper bound** on the depth mean,
and a loose one for any term whose difference changes sign with depth or with
longitude. Measured coherence (|signed| ÷ |mean of magnitudes|) is 0.5-0.8 for
the drag increment but as low as 0.07 for the vorticity flux at some rows.

**Corrected**: both terms are now scored as the **signed, thickness-weighted,
zonally-averaged** difference per row — the quantity the bar was derived for —
with the old `mean|·|` printed beside it and the coherence ratio between them,
so no term can be credited with a difference that cancels the moment it is
projected onto the mode the defect lives in.

## The interior sample contained a degenerate row

Row 99 is the **equator**, where `f = 0` and the vorticity flux structurally
collapses (its own magnitude there is 2.6e-9 against ~1e-6 at every other
sampled row). Putting it in the denominator of an enrichment ratio inflated
that ratio. It is excluded from every interior set and reported separately.

## The before level was silently substituted

`dyn_drg_init` reads the **before** velocity (`ln_bt_fw=.false.`). The bridge
call the probe used leaves `u_before` as `None`, and the probe fell back to the
**now** velocity — a fallback that produced entirely plausible numbers about
100× too large. The before level is now bridged explicitly and its absence
**raises** instead of defaulting.

## Four claims in the ranking that promoted these candidates are RETRACTED

1. **"Rows 1-4 are a zonally periodic band."** FALSE. Those rows have land at
   columns 0 and 51; the re-entrant channel is rows **14-48**. Rows 1-13 are a
   **closed sub-basin**. The zonal-integral argument that made bottom drag the
   top candidate — that the pressure gradient drops out, leaving drag — does not
   hold on a blocked row, and neither does the vanishing of bottom form stress.
   **The argument that promoted drag to #1 is void.**
2. **"A 1.77× differential H/r gives 0.19-0.34 e-folds over 90 days."** FALSE.
   Those are the *absolute* e-folds (0.176 at the wall, 0.107 in the interior);
   the **differential is 0.07 e-folds**, about 7% against a 34% target. Drag was
   promoted on a number roughly 5× too large.
3. **"The vorticity flux agrees to 1.4e-5 relative."** FALSE — that was a
   max-versus-max statistic on a single outlier cell. Pointwise it is **2.1e-3
   at the wall rows** and 4.9e-4 in the interior, two orders *worse* than this
   campaign's matched-operator floor. It is the **worst-matched operator the
   campaign has reported**, which inverts the framing that number was used for.
4. **The drag site list mislabelled one site and omitted the one that
   matters.** `dynzdf.F90:172-177` is the implicit path's barotropic re-add, not
   an explicit bottom-stress source; the genuinely explicit site
   (`dynzdf.F90:119`) is **dead code** under `ln_drgimp=.true.`. And the site
   that damps the **depth mean** — the barotropic substep drag at
   `dynspg_ts.F90:820-824` — was never in the list at all.

## Drag is refuted anyway, and by something better than the measurement

Summing all active drag sites at the wall rows on the coherent reduction gives
about **1.3e-10 m/s²**, so the defect would need **45% of the entire bottom-drag
term**. The coefficient is bit-exact, the bottom-level index is exact, and the
column depth agrees to 0.5%. No composition error in the unmeasured sites
delivers 45%. That bound refutes drag over-determined, independently of which
single site was measured.
