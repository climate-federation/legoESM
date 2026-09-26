# OVERFLOW-zps census owner: preregistration

Frozen before either one-variable arm and before computing any new legoESM
trajectory metric.  The starting revision is `e4263111f`; all runs are CPU,
with `JAX_ENABLE_X64=1` and `PrecisionPolicy.fp64()` set before card
construction unless the registered fp32 floor is named explicitly.

The target is the sole full-duration `OUTSIDE` row at the starting revision:
`final_water_mass_census = 0.009437984098628803`, against NEMO's FCT2/FCT4
scheme spread `0.003362090947063974` and the measured fp32/fp64 floor
`0.0014131684158793859`.  The reduction is the scorer's max absolute
difference among live-volume fractions in `[10,12)`, `[12,18)`, and `[18,20]`
over wet columns with `500 m < H < 2000 m`, at completed step 6120.

## Pre-implementation search: searched and found

Searched `scripts/validate/ocean_fidelity/testcases/`,
`packages/ocean/legoesm/ocean/fidelity/`, the canonical ocean advection,
vertical-mixing and BBL modules, and `tests/ocean/unit/` for an existing slope
census, term budget, tendency capture, and candidate implementations.

Found and reused:

* `nemo_testcase_census_map_probe.py` owns the certified scorer loaders,
  halo/staggering map, wet-mask intersection, slope mask, live-volume weights,
  bin edges, and the planted scorer-reproduction control.  This round extends
  that tool; it does not create a second census implementation.
* `nemo_testcase_full_statistics.py` owns the 6120-step runner and statistical
  verdict.  Arm states will be written in its existing NPZ/JSON format and
  scored by it unchanged.
* `_NEMOWSRK3TestHooks` is the established private A/B surface.  No arm becomes
  a public selector and no unattested composition is constructible.
* `ocean.physics.bbl_adv` is the one canonical Campin--Goosse arithmetic;
  `nemo_wicker_aimp_partition_transport` is the one canonical Shchepetkin
  adaptive split.  Repairs, if confirmed, change operands supplied to these
  shared kernels rather than duplicate either scheme.
* `ocean.fidelity.box_heat_budget` and `tendency_probe` are DINO/general budget
  accumulators.  They do not expose the private WS-RK3 stage-3 FCT/BBL/Aimp
  split, so they are not silently repurposed as if they did.

## Rule-0 coverage of the resolved tracer/bottom physics

The executed namelist selects FCT2 and its two-step implicit treatment
(`overflow_zps/namelist_cfg:67-72`), no tracer lateral diffusion (`:75-79`),
flux-form UP3 momentum advection (`:81-87`), no momentum lateral diffusion
(`:106-110`), `ln_zad_Aimp=.true.`, constant vertical closure with
`rn_avm0=1e-4 m2/s` and `rn_avt0=0` (`:123-131`), and Campin--Goosse BBL
with `nn_bbl_ldf=0`, `nn_bbl_adv=2`, `rn_gambbl=20 s` (`:134-140`).  The
resolved `ocean.output:634-698` additionally proves EVD, TKE, GLS, OSM, MFC,
NPC, tracer LDF, and momentum LDF are off.  Thus they are `VERIFIED-OFF`, not
unmeasured owner candidates.

NEMO's executed order is one source-defined package: stage 3 computes the BBL
coefficients (`stprk3_stg.F90:463-468`), runs FCT advection
(`stprk3_stg.F90:508-519`), adds the BBL tendency (`:583-589`), then solves
the implicit Aimp/ZDF matrix (`:596-599`).  There is no NEMO switch that
permutes those operations; this round adds none.

## Budget instrument, frozen before its output

The extended census probe advances one faithful baseline.  At every registered
sample it writes the scorer's own slope census and, on the same entering state,
one-step counterfactuals made only with private hooks:

1. BBL omitted (`disable_bbl=True`): isolates the stage-3 BBL contribution.
2. stage vertical tracer transport omitted
   (`disable_tracer_vertical_transport=True`): bounds explicit vertical
   advection plus its Aimp partition/solve.
3. the FCT two-step low-order predictor replaced by the one-step diagnostic
   arm (`two_step_fct_predictor=False`): isolates the predictor.  This is
   instrumentation, not a reference model, and is never used as a production
   configuration.

For each local counterfactual, both sides begin from the identical faithful
state.  The probe records max/rms `T_full-T_ablation`, slope heat-content
change, and the scorer-identical census movement.  It also records faithful
census/variance at a fixed cadence through 6120.  NEMO comparisons use the
existing certified Nbb/restart fields at completed steps 0, 3059 and 6120;
there is no interpolation and no claim between oracle samples.  “First” means
the first registered sample at which the legoESM-minus-NEMO census separation
is nonzero above the fp64 evaluation floor.

Planted controls must (a) perturb one wet slope cell across 18 C and make the
census check fail, and (b) add an unregistered budget term and make registry
coverage fail.  The probe JSON stamps git SHA, card/resolved-namelist hashes,
precision, backend, cadence, frame, and every input artifact hash.

## Arm 1 -- NEMO BBL reference geometry

### Source reading and discrepancy

NEMO constructs the static BBL geometry in `TRA/trabbl.F90:507-533`:

* `mbku_d=max(mbkt(i),mbkt(i+1))` (`:507-515`);
* `mgrhu` is the sign of the difference between **reference** bottom T depths
  `gdept_0(i,mbkt(i))`, and remains zero when those depths are equal
  (`:517-527`);
* `e3u_bbl_0` is the minimum of the U-face `e3u_0` evaluated at each adjacent
  bottom index (`:529-533`).

At run time option 2 reads Kbb bottom T/S and the Kmm geometric bottom depth
before `eos_rab` (`trabbl.F90:342-353`), computes the Campin--Goosse transport
(`:415-454`), and applies the closed three-leg exchange using Kmm `e3t`
(`:207-290`).  OVERFLOW zps deliberately leaves `gdept_0` and `e3w_0` on the
one-dimensional 20 m reference ladder while thinning only T/U/V/F bottom
cells (`tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:157-180`).

The current legoESM BBL geometry instead signs the continuous cumulative
partial-cell centroid, uses the minimum of adjacent bottom **T-cell**
thicknesses, and supplies a cumulative live bottom centroid to `eos_rab`
(`bbl_adv.py:117-165,176-229`; `ocean_model_latlon_cgrid.py:1322-1339`).
The static census-domain count is decisive and computed without a model arm:
NEMO has 29 wet interior nonzero i-slope faces; legoESM has 141, including 112
same-bottom-level faces on which NEMO's `mgrhu` is exactly zero.  On shared
faces the BBL thickness differs by as much as `9.960723613572725 m`.

### One-variable arm and predictions

Arm 1 changes only the operands of the existing canonical BBL kernel to the
three NEMO rules above.  A private hook retains the old operands solely for
the causal comparison.  The card has no selector: if the arm is faithful it
becomes the unbranched `rk3_ws + bbl_adv_option=2` behavior.

Before integrating, a replay on the existing midpoint/final faithful states
must print current versus NEMO-geometry `utr_bbl`, active-face counts, and
their induced one-step T tendency.  This is the scaling check.

* **CONFIRMED census owner:** the fp64 full-run row moves from
  `0.009437984098628803` to `<=0.003362090947063974`, a movement of at least
  `0.006075893151564829` toward NEMO, and no other statistical row regresses
  from `WITHIN-SCHEME-SPREAD` to `OUTSIDE`.
* **CONFIRMED contributor, not owner:** movement is toward NEMO by at least the
  measured precision floor `0.0014131684158793859`, but the row remains above
  scheme spread.
* **REFUTED as census owner:** movement is less than one tenth of the starting
  gap (`0.0009437984098628803`) or has the wrong sign.  Intermediate movement
  is `PLAUSIBLE`, not promoted post hoc.
* **Regression guard:** stage sweep kt=1..10, kt=60, and the 19-frame kt=1..4
  comparison retain their existing bar classifications.  A source-faithful
  correction may move a prior debt but may not silently invalidate geometry,
  frame, or time-level controls.

## Arm 2 -- NEMO W metric in the adaptive split (only if Arm 1 does not own)

`wAimp_RK3_t` divides each interface's vertical transport by
`e3w(Kmm)` (`sshwzv.F90:812-843`) and uses the donor cell's horizontal Courant
number (`:776-799,816-827`).  OVERFLOW's zps definition keeps `e3w_0=20 m` at
every level (`usrdef_zgr.F90:157-168`).  The current WS stage code instead
passes `0.5*(e3t_k+e3t_{k+1})`, padded with T thickness at top/bottom
(`ocean_model_latlon_cgrid.py:1215-1223`); at a partial bottom cell that is
not NEMO's W metric and can inflate the Courant number by up to `2x`.

Arm 2 supplies the card's reference W ladder times the stage Kmm qco stretch
to the existing canonical partition and retains the old midpoint only in a
private hook.  Its owner/contributor/refute bands and regression gates are
identical to Arm 1.  It is not run before Arm 1 is dispositioned.

## Registered stopping rule and prohibited changes

Stop at the first `CONFIRMED census owner`, land that source-defined package,
and rerun all gates plus fp64/fp32 statistics.  If both arms are refuted, the
register is exhausted for this dispatch and the remaining FCT bottom-flux
question stays `UNMEASURED`; no threshold, damping, clip, or limiter may be
added.  FCT's bottom flux is already structurally zero where NEMO's masked
`pW` is zero (`traadv_fct.F90:493-524,560-594`), while its two-step predictor
and Aimp adjustments are source-defined at `:526-557,596-625`; only the
budget instrument may promote a more specific mismatch.

## Committed addendum: keep the raw W-metric consumers one-variable

The first Arm-2 implementation attempt exposed a hidden coupling before it
completed 612 steps: populating `z_coord.nemo_e3w_0` changed both
`wAimp_RK3_t` **and** the active literal vertical-viscosity solve.  That run
was interrupted and wrote no artifact.  Commit `8ace6513aa` records the
retraction in executable code and restores a genuinely one-variable Aimp arm.

This matters because NEMO uses the same raw `e3w_0*(1+r3t(Kmm))` object in
both consumers (`domzgr_substitute.h90:131`): `wAimp_RK3_t` divides the
vertical Courant number by it (`sshwzv.F90:812-843`), while `dyn_zdf` divides
the constant-viscosity gradient by `e3uw(Kmm)` (`dynzdf.F90:190-205`).  The
resolved OVERFLOW values are `rn_avm0=1e-4`, `rn_avt0=0`; therefore the ZDF
arm is momentum-only directly and can affect T only through the evolved
velocity.  It is a live source mismatch, not a tracer-diffusion hypothesis.

The remaining frozen sequence is:

1. **Arm 2a:** raw W metric in the Aimp partition only; legacy midpoint in
   ZDF.
2. **Arm 2b:** raw W metric in ZDF only; legacy midpoint in Aimp.
3. **Arm 2c:** both consumers use the one NEMO W metric.  This is the actual
   reference package and is run only if neither one-variable arm owns; it is
   not a Frankenstein combination.

Each arm starts from the now-source-faithful BBL baseline whose measured
fp64 census distance is `0.01003241135670166`.  The frozen classifications are:

* **CONFIRMED owner:** final distance `<=0.003362090947063974`, movement
  toward NEMO `>=0.006670320409637686`, and no new OUTSIDE row;
* **CONFIRMED contributor:** movement toward NEMO
  `>=0.0014131684158793859` but final distance remains above spread;
* **REFUTED:** wrong-sign movement or absolute movement
  `<0.001003241135670166`;
* otherwise **PLAUSIBLE**.

The same stage, kt=60, frame, full-statistics, no-damping, and first-owner
stopping rules above apply.  If the separate arms are inert but their combined
reference package moves the result, ownership is **CONFIRMED INTERACTION**,
not assigned to either constituent post hoc.
