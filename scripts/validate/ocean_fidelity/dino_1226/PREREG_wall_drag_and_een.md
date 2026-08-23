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
