# PRE-REGISTRATION — the EEN F-point thickness (`e3f`) as the owner of the wall-row vorticity-flux mismatch

Written 2026-08-23, BEFORE any number in it existed. Issue #1455.
Parent: `docs/ocean/fidelity/dino_wall_ldf_alignment.md` (the EEN vorticity flux
is the sole surviving fix candidate) and `PREREG_wall_drag_and_een.md` (which
measured it).

## What is already measured, and what is not

The EEN vorticity flux clears both registered shape legs at the wall rows:
**3.10e-10 m/s²** on the signed thickness-weighted zonal mean (5.25× the
magnitude bar), **8.74×** wall-enriched, and a **2.1e-3** pointwise relative
disagreement against a 1.5e-5 matched-operator floor. That names the TERM.
It does not name the MECHANISM inside the term.

The campaign's standing lore — "the count-normalised e3f mean matches, so the
wall corner gets full thickness" — was established by READING the two codes.
The numeric value was never compared. This registration settles it by
measurement.

## The mechanism, as read from the oracle (to be confirmed or refuted by number)

DINO compiles with `bld::tool::fppkeys key_qco key_vco_3d`
(`cfgs/DINO/cpp_DINO.fcm`). Under `key_qco`, `vor_een`'s runtime
`SELECT CASE(nn_e3f_typ)` block sits inside the `#else` arm of
`#if defined key_qco || defined key_linssh` (`src/OCE/DYN/dynvor.F90:718-745`)
and is therefore **dead code in this build**. What executes is the one-line
`z1_e3f(ji,jj) = 1._wp / e3f_vor(ji,jj,jk)` at `dynvor.F90:719`, with

    e3f_vor(i,j,k) = e3f_0vor(i,j,k) * ( 1 + r3f(i,j) * fe3mask(i,j,k) )

(`src/OCE/DOM/domzgr_substitute.h90:130` with `E3fv_0 -> e3f_0vor` at `:118`
and `Tmskf` at `:48`). Its three pieces:

* `e3f_0vor` — built ONCE at init from the **static** `e3t_0`, masked-averaged
  over the wet count (`nn_e3f_typ = 1`, `cfgs/DINO/.../namelist_cfg:337`;
  `dynvor.F90:927-939`), with zeros overwritten by `e3f_0` (`:950`).
* `r3f` — the sea-surface ratio at the F-point, an **area-weighted, entirely
  unmasked** four-point average (`src/OCE/DOM/domqco.F90:177-181`).
* `fe3mask` — set to `fmask` (`src/OCE/DOM/dommsk.F90:198`), and `fmask` is the
  **product of the four surrounding `tmask`** (`dommsk.F90:152-153`), i.e. it
  is **0 at every vertex with a dry neighbour**.

So NEMO applies **no free-surface stretching at all** to the F-point thickness
at a wall vertex, and the static wet-count mean of `e3t_0` alone.

legoESM (`ocean_pe_latlon_cgrid.py::een_e3f_h_vtx`, `een_e3f_scheme="nemo_avg"`)
averages the **live** thickness over the wet count, so the stretching IS applied
at the wall vertex. The two constructions therefore differ by approximately the
local free-surface ratio η/H, depth-uniformly, exactly on the rows where the
column is shallowest — which is the shape of the measured defect.

Nothing above is a claim about magnitude. That is what is being measured.

## The state

NEMO's `RUN_D180_1STEP_1R`: one step (kt 5761) from `DINO_00005760_restart.nc`,
single rank — the same state `wall_term_discriminators.py` scored the 3.10e-10
on, bridged identically (`e3t_mode="both"`, before level bridged explicitly).
No model run is needed for any part of this registration.

`r3f` is produced by exactly one call in DINO's step (`MY_SRC/stpmlf.F90:378`),
which runs **after** `dyn_vor` (`:315`); the `ln_dynspg_exp` call at `:252` is
dead (`ln_dynspg_ts = .true.`). So the `r3f` that `dyn_vor` reads at kt 5761 is
the one `dom_qco_init` built from the restart's **now** ssh
(`domqco.F90:131`) — the same ssh the bridge hands legoESM.

## The instrument, and the control that licenses it

`r3f` is not dumped. It is reconstructed offline from the restart ssh and the
mesh metrics, which makes the reconstruction untrusted code.

**Control (registered, must pass before any e3f number is quoted):** the same
transcribed `dom_qco_r3c` arithmetic produces `r3u` and `r3v`, which ARE dumped
(`r3c_dump_r3u.bin`, `r3c_dump_r3v.bin`, written from the AFTER ssh, which is
also dumped as `sshnxt_dump_ssh_after.bin`). Feeding the dumped after-ssh
through the transcription must reproduce the dumped `r3u`/`r3v` to **1e-12
relative** over wet points. If it does not, the transcription is wrong and no
`r3f` number is reported. This is the proxy-against-a-known-answer rule: the
f-point branch is the same routine, three lines below the u/v branch.

## LEG 1 — the registered mechanism test

Compute both F-point thicknesses on the same state, diff them, then propagate
each through the SAME production triad code
(`pv_flux_al81_partial_cell`, one argument swapped, everything else identical)
and diff the resulting `vortcor_u`.

The mechanism is **CONFIRMED** only if ALL THREE hold:

  a. **Magnitude** — the propagated tendency difference on the four wall rows,
     under the same signed thickness-weighted zonal reduction the parent used,
     is within a factor of **2** of **3.10e-10 m/s²** (i.e. in
     [1.55e-10, 6.20e-10]).
  b. **Shape** — that difference is wall-enriched **≥ 3×** against the far
     interior (rows 20, 40, 60, 150; the equator row 99 is excluded, it is
     degenerate for this term).
  c. **Sign** — it has the sign that strengthens legoESM's westward lobe
     relative to NEMO, i.e. the same sign as the measured NEMO-minus-legoESM
     wall-row difference.

Any leg failing **REFUTES** the mechanism as the owner.

## LEG 1b — the decisive form of the same test, registered alongside

Legs (a)-(c) ask whether the mechanism's difference LOOKS like the measured one.
The stronger question is whether it CLOSES it. Registered in advance:

  **Residual reduction** — with `R_before = NEMO_dump − legoESM_baseline` and
  `R_after = NEMO_dump − legoESM_with_NEMO_e3f`, both on the wall rows under the
  same reduction, the mechanism OWNS the mismatch if
  `|R_after| / |R_before| < 0.4` (a >60% close), is a CONTRIBUTOR if the ratio
  is in [0.4, 0.9], and is REFUTED as owner above 0.9.

Where legs (a)-(c) and 1b disagree, **1b governs the verdict** and the
disagreement is reported as the finding.

## LEGS 2-5 — the decomposition, run in the SAME pass, reported whatever leg 1 does

The parent left ONE operator; exhaustive decomposition is tractable, and the
substitutions are one argument each into the same function, so they cost
nothing extra. Each factor is replaced, one at a time, by its NEMO-faithful
construction, and the residual-reduction statistic above is reported for each:

  2. **`f_vtx`** — legoESM's analytic `vertex_coriolis(grid)` against the mesh's
     own `ff_f`.
  3. **`zeta`** — legoESM's `curl_vertex_cgrid` against `dynvor.F90:757-758`
     transcribed on the mesh metrics (`e2v`, `e1u`, `r1_e1e2f`).
  4. **Mass fluxes `F_u`, `F_v`** — legoESM's min-rule face thicknesses against
     NEMO's `e3u = e3u_0·(1+r3u·umask)`, `e3v = e3v_0·(1+r3v·vmask)`.
  5. **Everything at once** — all four substitutions together. If the combined
     residual does not go to the matched-operator floor, the remainder is the
     triad assembly itself (weights, pairing, seam), and that is the finding.

The ranking of legs 2-5 by residual reduction is the deliverable if leg 1
refutes.

## Traps that apply, and how each is handled

* **The dry-row trap** (0be305459). Row 0 is entirely dry. Every reduction is
  over wet cells only; the probe plants a value on the dry row and requires no
  scored wall number to move where no stencil crosses, and plants on a wet wall
  row and requires movement. Note the EEN flux reads row 0 BY DESIGN on both
  sides (`ln_dynvor_msk = .false.`), so for leg 1 that plant is reported with
  its size, not forbidden — as the parent probe already does.
* **The dry-face gate.** Both constructions read land; the scored numbers only
  mean anything if the stored velocity on every dry face is exactly zero. The
  parent's `_gate_dry_faces` runs on the scoring path.
* **Index conventions.** F-points, T-points and the two dump families carry
  different halos and axis orders. Every shape is asserted after slicing.
* **NaN is fatal.** No `nanmean` anywhere.
* **No verdict is printed by the probe.** It prints values; the verdict text is
  computed from those values against the bars registered above.

## What is NOT registered here

No fix. No model run. Phase 2 (a selectable NEMO-faithful `e3f` option and a
single 90-day A/B) is conditional on leg 1b CONFIRMING, and will be
pre-registered separately with its own joint transport-and-density gate.

Nothing about any outcome is known at the time of writing.
