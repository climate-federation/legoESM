# Pre-registration — DINO equatorial-undercurrent mechanism probe

Registered before computing any legoESM-versus-NEMO vertical-mixing statistic
for this probe and before stepping either arm.

Upstream finding: `dino_regional_audit_result.md` reports a 28--32% deficit in
the 5--26 m equatorial zonal-velocity shear and a wind-driven-layer zero
crossing 9--11% (3.0--3.4 m) deeper in legoESM at days 90/180/270/360.  The
audit labels vertical mixing only PLAUSIBLE and prescribes the matched-state
coefficient comparison performed here.

## State, code, and provenance

Both arms start from the control member's NEMO day-180 restart
`DINO_00005760_restart.nc`, the same restart used by the verdict-year campaign.
The restart's T, S, u, v, before levels, TKE `en`, and closure-only `avm_k` are
read through the committed DINO twin loader.  The legoESM recipe is the shipped
`nemo_dino_kamm_mlf` card, with NEMO's own ladders (`LEGOESM_NEMO_E3T=both`),
fp64, the bridged start, and the corrected seasonal clock.  The NEMO comparison
target is the control run from this same restart.

The artifact must record, after reading rather than merely requesting them:

* legoESM launch SHA and dirty state;
* SHA-256 of every imported sibling reducer/loader;
* NEMO source/config SHA and dirty state, DINO namelist SHA-256, NEMO binary
  SHA-256, and day-180 restart SHA-256;
* observed recipe, ladder mode, seasonal phase, state dtype, and CUDA device.

A tree or input that changes between launch and artifact write is fatal.

## Leg O — offline matched-state coefficient discriminator

At the equator row (T-row 99), compare legoESM's closure-only vertical
viscosity computed from the bridged NEMO day-180 state with NEMO restart
`avm_k`.  The shared interface mapping is legoESM interface `i` to NEMO W-level
`jk=i+2`.  Report both the equator row and the nested E10 rows 89--109, but the
equator row decides.

The scored depth interval is the interfaces with NEMO `gdepw_1d` in
`[10, 40] m`, spanning the observed 28--35 m undercurrent top.  Every aggregate
uses interface control-volume thickness weights; no layer-count average may
decide a verdict.  Let

```
NRMS = sqrt(sum(w * (A_lego - avm_k_NEMO)^2)
            / sum(w * avm_k_NEMO^2))
```

over wet equatorial cells and scored interfaces.  Also report each interface's
thickness-weighted mean coefficient and ratio.

* **CONFIRM_CLOSURE_DIFFERENCE**: `NRMS >= 0.25` and at least two consecutive
  scored interfaces have a mean ratio outside `[0.75, 1.25]` in the same
  direction.
* **REFUTE_CLOSURE_DIFFERENCE**: `NRMS <= 0.10` and every scored-interface mean
  ratio lies in `[0.90, 1.10]`.
* **UNRESOLVED**: everything between those bars.

This leg names only whether the closures deliver materially different
coefficients at a matched state.  It does not by itself claim that viscosity
owns the trajectory difference.

## Leg R — short substitution response

Run one 10-day legoESM pair on `CUDA_VISIBLE_DEVICES=0` from the identical
day-180 state:

* **BASE**: shipped card, unchanged;
* **NEMO_AVM**: identical, except the momentum-viscosity output passed to the
  implicit vertical solve is replaced at every step by NEMO's day-180
  closure-only `avm_k`, mapped to legoESM interfaces.  Tracer diffusivity,
  prognostic TKE update, EVD trigger, forcing, and every configuration field
  remain BASE.  The replacement is a fixed coefficient field, so the result is
  explicitly a short impulse/retention discriminator, not a production model.

Daily outputs are scored against the NEMO control member at the same elapsed
day.  The primary statistic is the equator-row undercurrent-top depth: the
shallowest linearly interpolated sign change of zonal-mean `u` in the top
100 m.  The co-primary guard is the 5--26 m shear used by the regional audit.
Profile summaries and integrated differences use the campaign thicknesses.

At day 10 define core-depth closure

```
Cz = 1 - abs(z_NEMO_AVM - z_NEMO) / abs(z_BASE - z_NEMO).
```

Define shear closure analogously from absolute shear errors.

* **CONFIRM_VISCOSITY_DRIVER**: `Cz >= 0.50`, the core moves toward NEMO, and
  the shear error does not worsen by more than 10% of BASE's absolute error.
* **REFUTE_VISCOSITY_DRIVER**: `Cz <= 0.10`, or the core moves away from NEMO.
* **UNRESOLVED**: `0.10 < Cz < 0.50`, or a core-depth confirmation fails the
  shear guard.

If BASE's day-10 core-depth error is below 0.5 m, the response denominator is
too small and Leg R is **UNMEASURABLE**, not a pass.  No transport integral can
override either profile verdict.

## Controls

All controls run before a verdict is emitted and are demonstrated able to fail.

1. Exact day-0 identity of T, S, u, v and bridged before/TKE state on the shared
   wet mask; any mismatch is fatal.
2. BASE and NEMO_AVM inputs/configs hash-identical before the coefficient hook.
3. The hook must leave K_v and returned TKE bit-identical while changing A_v;
   a deliberate wrong interface shift must fail the known-depth mapping.
4. Wet/dry mask plant: `1e6` on a dry coefficient cell moves every scored
   profile by exactly zero; the same plant on a wet equatorial cell must move
   it.
5. BASE day-10 u must reproduce the previously committed/saved verdict-year
   control trajectory within its storage precision; otherwise no response
   verdict is emitted.
6. Profile reduction uses wet-cell zonal means and refuses an empty level.
7. A deliberate 2x coefficient plant must alter the first-step implicit
   momentum result; this proves the substitution path is live.

## Review-mandated reissue (A3/A4; registered before decomposition)

The original two-consecutive-interface confirmation clause is retained above
for provenance but is retracted as physically mis-specified for this metric.
The regional shear readout spans 5--26 m, so the 10.14 m interface is itself a
shear-setting level. Requiring the adjacent 20.59 m interface to fail too
would make a surface-anchored viscosity mechanism impossible to confirm.

The corrected offline bar is:

* **CONFIRM_VISCOSITY_PRIME_SUSPECT** if thickness-weighted 10--40 m NRMS is at
  least 0.25 and either (a) two consecutive ratios lie outside [0.75, 1.25] in
  the same direction, or (b) the 10.14 m shear-setting interface alone lies
  outside that interval and its sign predicts the observed weak 5--26 m shear
  and deeper core.
* **REFUTE_VISCOSITY_PRIME_SUSPECT** keeps the original refutation bar: NRMS
  at most 0.10 and all level ratios within [0.90, 1.10].
* otherwise **UNRESOLVED_VISCOSITY_PRIME_SUSPECT**.

This reissue is compelled by the adversarial review and is not represented as
having preceded the first measurement. It does precede the attribution
measurement below.

## 10.14 m TKE-closure attribution bar

At equator row 99 and the declared lego interface 0 / NEMO jk=2 alignment,
decompose each positive, wet, non-floor column using the executed avm branch:

```
log(avm_L / avm_N) = log(Ck_L / rn_ediff_N)
                    + log(mxl_L / zmxlm_N)
                    + 0.5 * log(en_L / en_N).
```

NEMO's `rn_ediff` is the viscosity coefficient in the executed
`zav=rn_ediff*zmxlm*sqrt(en)` branch. If no independent multiplicative
stability function acts on avm, the coefficient/stability contribution is
explicitly zero rather than being silently folded into another piece.

Controls and eligibility:

* both direct avm values must exceed their background floors;
* the three-factor reconstruction must reproduce each direct avm within
  `1e-10` relative; otherwise abort attribution;
* at least 75% of wet equatorial columns must be eligible;
* the dry-cell and one-level-shift plants must fail as in the parent probe;
* all component dumps must resolve through the shared time-level registry and
  their SHA-256 values must be recorded and read back.

For columns with direct `avm_L/avm_N > 1.25`, define each piece's band score as
the median absolute log contribution. A closure piece is labelled
**CARRIES_10M_EXCESS** when it supplies at least 60% of the sum of the three
scores and its signed contribution agrees with the total excess in at least
75% of excess columns. If no piece clears both bars, attribution is
**DISTRIBUTED_OR_UNRESOLVED**. Also report geometric-mean factors,
interquartile ranges, and the exact per-column reconstruction control; those
diagnostics do not alter the registered label.

## Follow-up: single-root energy propagation test

Registered before reading the following ratios. The population is the same
10.14 m equatorial excess set used above (`avm_L/avm_N > 1.25`, positive,
wet, non-floor, exact-factorization columns).

The executed NEMO raw buoyancy limb is
`MAX(rmxl_min,SQRT(2*en/MAX(rn2,rsmall)))` at
`cfgs/DINO/MY_SRC/zdftke.F90:757-760`; the DINO `nn_mxl=3` distance-bounding
sweeps then apply at `:799-812`.

Measurements:

1. Per-column energy ratio `en_L/en_N`. Report geometric mean, median and IQR.
2. Per-column normalized-length ratio
   `(mxl_L/sqrt(en_L))/(mxl_N/sqrt(en_N))`.
3. Active limb on both models. A column is `BUOYANCY_LIMITED` when final mixing
   length agrees with its raw buoyancy length within `1e-8` relative. It is
   `DISTANCE_BOUNDED` when final length is smaller by more than `1e-8`; an
   impossible final length above the raw limb aborts. NEMO's raw limb uses the
   same-step `tke_dump_en` and `tke_dump_rn2`; legoESM's uses the final captured
   `en`, `N2`, and the production `mxl_min`.

Verdict bars:

* **CONFIRM_TKE_ENERGY_SINGLE_ROOT** if all hold: geometric-mean `en_L/en_N`
  is in [2.03, 2.48] (within 10% of 2.256); geometric-mean normalized length
  is in [0.95, 1.05]; at least 75% of excess columns individually have
  normalized-length ratio in [0.90, 1.10]; and at least 75% of excess columns
  are buoyancy-limited in each model.
* **REFUTE_TKE_ENERGY_SINGLE_ROOT** if the normalized-length geometric mean is
  outside [0.90, 1.10], or fewer than 50% of excess columns are
  buoyancy-limited in NEMO.
* otherwise **UNRESOLVED_TKE_ENERGY_SINGLE_ROOT**.

The limb classification and normalized ratio are required together: equal
algebraic factors do not establish a single energy root if a distance bound is
actually active.

## Follow-up: TKE-equation owner decomposition

Registered before loading or reducing `tke_dump_sh2.bin`,
`tke_dump_dissl.bin`, the restart `avt_k`, or `sbc_dump_utau.bin`, and before
running any one-term replay.  The population is frozen to the same 47
equatorial columns whose direct 10.14 m `avm_L/avm_N` exceeds 1.25 in the
single-root test.  The response is final `en` at legoESM interface 0 / NEMO
`jk=2`, after the `nn_etau=1` addition.  Surface diagnostics additionally use
NEMO `jk=1`, the held z=0 row.

The executed NEMO equation is the matrix/RHS branch at
`cfgs/DINO/MY_SRC/zdftke.F90:499-516`: positive shear production is `p_sh2`,
buoyancy work is `-p_avt*rn2`, and the carried `dissl` is split between the
1.5 diagonal and 0.5 RHS terms.  The held surface row is
`en(ji,jj,1)=MAX(rn_emin0,(rn_ebb/rho0)*taum(ji,jj))` at `:334,356-365`.
The DINO stress modulus is `ABS(utau)`, multiplied by 1.3 only for positive
`utau`, at `MY_SRC/usrdef_sbc.F90:380-383`.

For each candidate define a causal-direction factor `q` (values above one
predict excess legoESM energy):

* production: `q = sh2_L / sh2_N` on columns where both are positive;
* stable buoyancy sink: `q = (avt_N*rn2_N)/(K_H_L*N2_L)` on columns where both
  sink magnitudes are positive;
* dissipation: `q = dissl_N/dissl_L` on positive columns (the two closures use
  the same `rn_ediss=0.7` split coefficient);
* surface-BC class: `q = en_surface_L/en_surface_N`, accompanied by the
  independently reconstructed `taum` and surface `zmxlm`/`avm` ratios.

A factor **MATCHES_EN_EXCESS** only when its geometric mean is in [2.03, 2.48],
the geometric mean of `(en_L/en_N)/q` is in [0.90, 1.10], and at least 75% of
its eligible columns have that normalized ratio in [0.80, 1.20].  A factor is
**NEAR_UNITY** when its geometric mean lies in [0.90, 1.10].  Zero/sign-mixed
terms are labelled **INELIGIBLE_SIGN_OR_FLOOR**, never silently discarded;
every term reports its eligible count, geometric mean, median and IQR.

As the causal guard, replay legoESM's captured production TKE solve four
times, changing one equation input only:

1. replace `P_s` by NEMO `p_sh2`;
2. replace the explicit `K_H*N2` product by restart `avt_k * tke_dump_rn2`;
3. replace the carried `sqrt(en_old)/l_eps` by NEMO `dissl`;
4. replace the held surface `en` and surface-face viscosity by NEMO
   `tke_dump_en(jk=1)` and the corresponding dumped `zmxlm(jk=1)` value.  Its
   `nn_etau=1` addition is scaled by the substituted held surface energy.

All other captured operands remain legoESM BASE.  BASE replay must reproduce
the production captured solve and final energy within `1e-10` relative or the
decomposition aborts.  For arm `t`, per column define log-gap closure

```
C_t = 1 - abs(log(en_t/en_N)) / abs(log(en_BASE/en_N)).
```

A term **CARRIES_TKE_EN_EXCESS** only if it both MATCHES_EN_EXCESS and its
replay has median `C_t >= 0.60`, moves energy toward NEMO in at least 75% of
eligible columns, and no other qualifying term reaches median `C_t >= 0.25`.
If more than one term clears 0.25, label **DISTRIBUTED_TKE_EQUATION_OWNER**;
if none clears the full owner bar, label **UNRESOLVED_TKE_EQUATION_OWNER**.
The surface boundary is a distinct candidate class and must also pass direct
stress/BC reconstruction: NEMO surface `en` must match its quoted formula and
legoESM's imposed surface `en` must match `_surface_tke_dirichlet` within
`1e-10` relative.  A failed reconstruction aborts rather than refuting the
surface class.

Controls proven able to fail: the fixed 47-column mask is read back from the
single-root result; a one-level NEMO shift must worsen BASE energy agreement;
a 2x production plant must alter replayed energy; and each dump is accepted
only through the committed time-level registry with SHA-256 recorded and
read back.  The analysis is thickness-aware by retaining the campaign's
10.14 m interface control-volume provenance; its deciding population is a
single interface, so no unweighted vertical average is introduced.

### Residual structural discriminator (registered after the term null)

The registered term pass above returned no matching input factor and no arm
response at 10.14 m.  Its non-deciding response ledger, inspected only after
that verdict, showed legoESM's matrix result equal to `tke_surface_min=1e-4`
in all 47 columns while NEMO's solved `jk=2` energy is below `1e-4`.  Source
inspection supplies a single structural hypothesis: legoESM applies its
surface-minimum clamp unconditionally to interior interface 0 after creating
the virtual `nemo_z0` row, whereas NEMO applies `rn_emin0` only to held `jk=1`
and applies `rn_emin` to solved `jk=2..jpkm1`.

Before computing the following arm, register one replay changing only
`tke_surface_min` from `1e-4` to `tke_background=1e-6` in the already captured
legoESM matrix solve.  This emulates the faithful branch condition without
editing production physics; all equation operands and the post-solve etau
addition remain BASE.

* **CONFIRM_MISPLACED_SURFACE_MIN_CLAMP** if at least 95% of BASE columns are
  exactly pinned to `1e-4`, at least 75% of corresponding NEMO `jk=2` values
  are below `1e-4`, the arm closes at least 60% of the median log-energy gap,
  and it moves toward NEMO in at least 75% of columns.
* **REFUTE_MISPLACED_SURFACE_MIN_CLAMP** if fewer than 50% of BASE columns are
  pinned or median log-gap closure is at most 10%.
* otherwise **UNRESOLVED_MISPLACED_SURFACE_MIN_CLAMP**.

This diagnostic cannot retroactively change the registered four-input verdict.
If it confirms, the combined physics finding may name the surface-BC placement
clamp as the owner and specify (but not implement) a branch-conditional fix:
apply `tke_surface_min` only for `interior_pinned`; under `nemo_z0`, clamp the
solved first interior interface only to `tke_background`, because the separate
virtual row already carries the surface Dirichlet value.
