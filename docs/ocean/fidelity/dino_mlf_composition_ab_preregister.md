# DINO MLF composition A/B vs NEMO — preregistration

Row M-01 (`nemo_branch_isomorphism_map.md`) measured legoESM's two `stp_MLF`
compositions against EACH OTHER (commit `80eaf9b33`, `mlf_step_mechanism_ab.py`):
T differs by 4.963e-4 degC at step 1, momentum bit-unchanged by a GM/Redi
ablation. That measurement never touched NEMO. This asks the next question:
which arm actually sits closer to NEMO's own trajectory, and is the arm-vs-arm
gap the same SIZE as either arm's distance from NEMO — the condition that
decides whether this composition choice is a candidate mechanism for the
20-year DINO climate divergence (`dino_multi_year_climate_equivalence_result.md`,
attribution-open).

## Rule 0 — oracle source, `file:line`

Checked against BOTH `src/OCE/stpmlf.F90` and DINO's
`cfgs/DINO/MY_SRC/stpmlf.F90` override (the latter only adds write-only
#1226/#1455 dump instrumentation around the SAME calls with the SAME
arguments — diffed line-by-line, zero physics `CALL` lines changed except
`mlf_baro_corr` gaining a `kstp` arg for its own dump, `:392` vs MY_SRC `:575`).

* `stpmlf.F90:196-199` — slope of lateral mixing. DINO selects the standard
  (non-triad) operator (`ln_traldf_triad=.false.`), so
  `CALL eos(ts,Nbb,rhd)` / `CALL ldf_slp(kstp,rhd,rn2b,Nbb,Nnn)`: the GM/Redi
  slopes are built from the BEFORE (`Nbb`) density and BEFORE Brunt-Vaisala
  (`rn2b`, itself `CALL bn2(ts(:,:,:,:,Nbb),rab_b,rn2b,Nnn)` at `:186`).
* `stpmlf.F90:250` — `CALL dyn_ldf(kstp,Nbb,Nnn,uu,vv,Nrhs)`. DINO's
  `namdyn_ldf` (`cfgs/DINO/EXP00/namelist_cfg:365-366`) sets
  `ln_dynldf_lap=.true.`, `ln_dynldf_lev=.true.` -> iso-level Laplacian
  (`np_lap`), dispatched (`dynldf.F90:68-70`) to `dynldf_lev_lap`
  (`dynldf_lev.F90:45`), whose own docstring (`:55`) states the harmonic
  operator is "applied on pu(Kbb), pv(Kbb)" — confirmed in the included
  scheme body (`dynldf_lev_rot_scheme.h90:24-29`): the curl/div differencing
  AND its `e3t/e3u/e3v` metrics are all at `Kbb`; `Kmm` enters only the final
  flux-divergence normalisation (`:41,51`, `e3u/e3v(...,Kmm)`). DINO's
  momentum LDF is iso-level (`np_lap`, not the rotated `np_lap_i`), so it
  does NOT consume the GM/Redi slopes at all.
* `stpmlf.F90:368` — `CALL tra_ldf(kstp,Nbb,Nnn,ts,Nrhs)`. DINO's
  `namtra_ldf` (`EXP00/namelist_cfg:257,263`) sets `ln_traldf_lap=.true.`,
  `ln_traldf_iso=.true.` -> rotated/iso-neutral Laplacian, dispatched to
  `traldf_iso_lap` (`traldf_iso.F90:54`), whose docstring (`:101`) states
  `pt` is "tracers, in: at kbb; out: at Krhs" — this operator DOES consume
  the Nbb-built GM/Redi slopes from the `ldf_slp` call above.

So: ONE traversal of `stp_MLF` per step. `dyn_ldf`/`tra_ldf` (and the slope
precompute feeding `tra_ldf`) read `Nbb` as their OWN call argument; every
OTHER term in the SAME traversal (`dyn_adv`, `dyn_vor`, `dyn_hpg`, `dyn_spg`,
`tra_adv`, forcing) reads `Nnn`.

## Which legoESM composition matches

* **`_nemo_mlf_step`** (`ocean_model_latlon_cgrid.py:10422`) — ONE
  `_step_impl` call; `_ldf_state=(T_before,S_before,u_before,v_before)` is
  read ONLY by the tra_ldf operand (`ocean_pe_latlon_cgrid.py:4570-4577`,
  `_T_ldf_local`/`_S_ldf_local`) and the dyn_ldf operand (`:4612-4626`,
  `_u_ldf_local`/`_v_ldf_local`); every other tendency in the SAME call reads
  the pass's own (Nnn) `T`/`S`/`u`/`v`. A per-term substitution inside one
  traversal — structurally the same shape as `stpmlf.F90`.
* **`_leapfrog_step`** (`:9885`) — TWO full `_step_impl` calls: one entirely
  at Nnn (`state`, dissipation withheld via `_ab2_scope_override="advective"`),
  one entirely at Nbb (`state_before`, only its dissipative increment kept),
  combined additively (docstring residual #4, `:9974-9989`). NEMO's source
  never evaluates a full second pass at Nbb — this is a MECHANISM
  SUBSTITUTION whose fidelity depends on the two passes' non-LDF machinery
  (metrics, masks, EOS/slopes, layer thickness, barotropic solve) being
  inert between the Nnn and Nbb evaluations, which `stpmlf.F90` never
  requires and this repo has not verified. (The isomorphism map's
  2026-09-02 entry already names exactly this candidate mechanism for the
  bit-unchanged momentum gap under GM/Redi ablation — logged UNVERIFIED.)

**Verdict: `_nemo_mlf_step` matches `stpmlf.F90`'s structural ordering;
`_leapfrog_step` does not — it substitutes a different mechanism.**

## Prediction (recorded before measuring)

`_nemo_mlf_step` sits closer to NEMO's own trajectory than `_leapfrog_step`,
on the fields where the two arms are already known to disagree (T, S — the
GM/Redi channel measured in `80eaf9b33`).

## Metric and protocol

Harness: `kamm_twin_90d._build_twin_state("nemo_dino_kamm_mlf", ...)` (the
bridge already proven in `80eaf9b33`), `LEGOESM_NEMO_E3T=both`, fp64 policy
set explicitly (`PrecisionPolicy.fp64()`), CPU. Certified `mc` UNCHANGED
(`outer_integrator="leapfrog"`, `implicit_vmix_e3t_now_divisor=False`) for
BOTH arms — `_nemo_mlf_step` is called directly (a private method, bypassing
`model.step()`'s dispatch), which is how `80eaf9b33` already avoided the
divisor confound (`outer_integrator="nemo_mlf"` hard-requires
`implicit_vmix_e3t_now_divisor=True` at construction time,
`ocean_model_latlon_cgrid.py::_validate_config`, ~`:3008-3021`): call the
method, do not flip the dispatch field. Byte-identical protocol on both arms
(same IC, same forcing, same jit pattern, same `mc`).

Two INDEPENDENT 160-step (5-day, dt=2700s) trajectories from the SAME bridged
day-180 state: arm A repeatedly calls `_leapfrog_step`; arm B repeatedly
calls `_nemo_mlf_step`. At every step: worst |diff| T/S/u/v/eta between arm A
and arm B (the arm-vs-arm curve, all 160 steps, wet cells).

**Gap found and preregistered before running, not discovered after**: NEMO
ground truth in this environment exists as exact per-step restarts ONLY for
kt=5761..5764 (steps 1-4 past the day-180 IC, `RUN_D180_STEP1_1R`; its
step-5764 file is cited as bit-identical to the certified trajectory by
`d180_step_walk.py:405-410`) and at 10-day cadence from day 190 on
(`RUN_90D_TWIN`, MPI-decomposed per-rank files, not read here). NOTHING
exists at day 185 (step 160) itself — confirmed by listing every DINO run
directory in the oracle tree. So "vs NEMO" is measured at steps 1-4 (both
arms against `read_nemo_restart` on those 4 files, masked by
`read_nemo_mesh_mask`'s tmask/umask/vmask), and the literal "day 5" number is
reported as arm-vs-arm only, explicitly labelled as having no NEMO anchor.
This is a WAIVED gap (oracle-fidelity skill Rule 1: enumerate, verify or
waive with a written reason), not a silent substitution — no new NEMO run is
launched to manufacture a day-185 reference (that would be an unasked choice
about where data comes from).

Determinism control: `_leapfrog_step` on the identical input state, twice —
must be exactly 0.0 on every field.

## Pass / refute

**CONFIRMED** if, at every one of steps 1-4, `_nemo_mlf_step`'s worst |diff|
vs NEMO on T (and S) is smaller than `_leapfrog_step`'s.
**REFUTED** if `_nemo_mlf_step` is not closer on T at any of steps 1-4, or is
closer only by a margin at or below the determinism-control floor (0.0).

**SCALE-COMPATIBLE** (candidate mechanism for the 20-year divergence) if the
CLOSER arm's own distance from NEMO at step 4 is LESS THAN HALF the FARTHER
arm's distance from NEMO at step 4 (factor >2 reduction). **NOT
SCALE-COMPATIBLE** otherwise — the composition difference would be real but
too small relative to either arm's total NEMO-distance to be a candidate for
the climate-scale gap.
