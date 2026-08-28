# DINO day-180 ZDF execution-chain sweep: first divergence

Date: 2026-08-28.  Lane: CPU-only, one-rank matched day-180 state
(`RUN_SEQDUMP_D180_1R`, `kt=5761`).

## Round-3 result: carried coefficients fixed; row 4 stops again

The registered `carried_previous_step` construction is now the production
default on the complete DINO NEMO cards, `nemo_dino_kamm` and its inherited
`nemo_dino_kamm_mlf` card.  A NEMO restart bridge reads `avm_k` and `avt_k`
without inference.  During the step the carried pair feeds face-weighted shear,
the Prandtl numerator, TKE matrix diagonals, the stratification RHS, and (when
active) the wave denominator.  Only the post-solve closure output is stored as
the next step's pair; EVD and other enhancements remain downstream composition
and cannot leak into the carry.

This follows NEMO's active lifetime: `zdf_phy` calls
`zdf_sh2(Kbb,Kmm,avm_k)` before `zdf_tke(...,avm_k,avt_k)` at
`cfgs/DINO/WORK/zdfphy.F90:268,286`; `zdftke.F90:489,503-506,514,538` consumes
the incoming pair and `zdftke.F90:832-844` overwrites it after the solve.

The generic `TKEConfig` default remains `current_subiteration`.  Therefore the
partial DINO `nemo_paper` card, the DINO `veros` card, the NEMO-recipe/ORCA
path, every Veros ACC/global recipe, and MPAS when configured with this common
TKE kernel keep their previous numerical path and array values.  No non-oracle
card selects the new behavior.  In particular, `fidelity/nemo_recipe.py` is a
partial, non-DINO NEMO-oriented recipe with its own certificate; it remains on
`current_subiteration` rather than borrowing this DINO matched-state result.
The common state pytree necessarily gains
three optional carry slots, but they remain `None` and are not read on those
cards; this is a schema extension, not a numerical-path change.

The red-capable regression set includes a hand-computed two-interface case
with deliberately different carried/current coefficients.  It proves that
carried `avm=[3,5]` is observed as shear production and the momentum matrix
operand, carried `avt=[7,11]` is the stratification operand, and the post-solve
`en=4` produces the next `avm=[2,2]`, `avt=[1,0.5]`.  Missing carry, an ignored
carry on the legacy selector, and selector/surface shape violations fail
loudly.  Restart axis/halo identity, DINO-card selection, JIT, and finite
gradients are also covered.  The final focused CPU/fp64 run passed 145 tests.

### Ordered rerun

The requested composite target was **not reached**.  The corrected carried
`p_avm` operand itself is `VERIFIED` at exactly 0/9,920 failures, zero maximum
column error, and 4/4 southern focus columns passing.  Continuing to the next
operand in the same row shows that row 4 as a whole is still `DIVERGED`:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920 failed; alpha max `3.087467e-16` | 4/4 pass |
| 2 | `bn2(Nbb)` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4a | carried `p_avm` operand | `VERIFIED` | **0/9,920; max `0`** | **4/4 pass** |
| 4 | complete `zdf_sh2` | `DIVERGED` | **9,920/9,920; max `73.29360955`** | **4/4 fail** |
| 5 onward | remaining TKE terms through implicit solves | `UNMEASURED` | ordered stop at row 4 | ordered stop |

All planted controls remain live: the baseline passes and the wet-cell
perturbation, one-i roll, and nonfinite injection each fire.

### New first operand: step-entry NOW velocities

After exact carried viscosity, NEMO's next factors are

```fortran
* ( uu(ji,jj,jk-1,Kmm) - uu(ji,jj,jk,Kmm) )
* ( uu(ji,jj,jk-1,Kbb) - uu(ji,jj,jk,Kbb) )
```

with the analogous `vv` factors at
`cfgs/DINO/WORK/zdfsh2.F90:81-82,86-87`.  NEMO evaluates `zdf_phy` at step
entry.  legoESM currently reaches the TKE closure after its explicit
dynamics/advection update, so its NOW factors are post-explicit and
pre-implicit-solve.

The directly scored BEFORE vertical-difference operands are bit-identical
(0 failed U or V face columns, zero maximum error).  The NOW difference
operand is not: 9,758/9,758 wet U columns and 9,868/9,868 wet V columns fail.
Scoring the literal `velocity(k-1)-velocity(k)` gives maxima `12.59807570`
(U) and `4.530529912` (V).  Each southern T-column focus score is the maximum
over its two surrounding wet faces; all four fail both operands.  U focus
errors are `1.1925297`, `0.9549570`, `1.1168380`, and `1.1182779`; V errors
are `0.1037876`, `0.1000737`, `0.1000737`, and `0.4359260`.  Substituting the
exact step-entry NOW velocities reduces the
complete shear maximum to `0.03098204` but does not close it (9,920/9,920
still fail), so later metric/four-face operands remain deliberately
unattributed.  The ordered walk stops at the first failing velocity operand.

Next-round design: add the static option `tke_shear_evaluation_stage`, with
`step_entry` as the faithful default on the two complete DINO NEMO cards and
`implicit_solve_state` as explicit legacy opt-in.  Evaluate and freeze
`p_sh2` from step-entry NOW/BEFORE velocities before explicit dynamics and
advection, carry that field to the closure, and use the same frozen field for
both the TKE shear RHS and Prandtl denominator.  Re-run row 4 before examining
the next `e3uw/e3vw` operand; the present substitution proves velocity timing
is first, not that it is the only remaining row-4 defect.

### Frozen climate prediction and GPU handoff

The prospective registration remains frozen even though the ordered
matched-state sweep found a later row-4 defect.  Baseline southern-basin
day-90 MLD RMS is `22.479491 m` (`22.4795 m` headline).

- `CONFIRM`: faithful RMS `<=11.2397455 m`; legacy control within `0.001 m` of
  `22.479491 m`; no acceptance error worsens versus control by more than its
  one-floor value; the 5x pass tally does not decrease; and at least one of the
  two southern surface-density errors improves by one floor.
- `REFUTE`: faithful RMS `>=20.2775 m`, or the legacy baseline control fails,
  or any acceptance metric worsens by more than one floor, or the 5x pass tally
  decreases.
- Between the RMS bands with valid controls is `PARTIAL/INDETERMINATE`.

The frozen one-floor values are ACC `0.091 Sv`, upper density contrast
`1.1e-4 kg m-3`, deep contrast `4.5e-5 kg m-3`, southern surface sigma maximum
`9.5e-5 kg m-3`, and mean `9.5e-5 kg m-3`.

Exact arm commands:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_carry_faithful_d90.npz --days 90 --save-3d \
  --bridge-tke

CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_carry_legacy_d90.npz --days 90 --save-3d \
  --bridge-tke --tke-preclosure-coeff-source current_subiteration

python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  /tmp/zdf_carry_faithful_d90.npz --level 5
python scripts/validate/ocean_fidelity/dino_1226/acceptance_gate_90d.py \
  /tmp/zdf_carry_legacy_d90.npz --level 5
```

The exact MLD scorer remains the audit's symmetric NOW-state criterion with
its NEMO area/mask and southern region.  This round ran no GPU arm.

Round-3 artifact:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round3_artifact.json`, SHA256
`695efe3c8f1f65634795be7ddf99b82ea90afc7ae3b990ccf7d495e03191ab43`.
Probe/tree SHA: `57c6b21e3bbb61fccd915c8575c2cadcc8ac6de0`.

### Round-3 adversarial review

Two independent read-only rereviews ended `NON-HOLD`.  The measurement
reviewer reproduced the artifact byte-for-byte, checked all 32 provenance
hashes, the exact difference/focus populations, live controls, and the ordered
stop.  The physics reviewer checked the active NEMO lifetime against source,
the partial-depth cold-start mask, restart/pass-through and partial-carry
failures, resolved-card scope, legacy identity, surface consumption, JIT/AD,
and separation of the post-solve closure carry from EVD.  `dissl` remains a
later ordered operand; this round makes no claim that it is closed.

## Round-2 result (supersedes the row-2 stop below)

The registered raw-mesh fix is now production code.  A NEMO bridge preserves
`mesh_mask.nc:e3w_0`, and the default `mesh_reference` construction supplies
`e3w_0*(1+r3t)` to every `nemo_bn2` path and to the paired `zdf_mxl`
multiplication.  The former `diff(live_gdept)` construction remains available
only through the explicit `depth_difference` option.  Missing, malformed, or
nonpositive native geometry fails closed.

The row-2 rerun is `VERIFIED`: 0/9,920 wet columns fail, maximum column error
`5.968673256056548e-16`, and all four southern focus columns pass.  The
explicit legacy control reproduces the accepted baseline exactly: 4,630/9,920
fail with maximum `6.366584806460317e-15`.  Row 3
`eos_rab/bn2(Nnn)` is also `VERIFIED`, 0/9,920 failures and maximum
`5.968545325803916e-16`.

The next and therefore current first divergence is row 4, `zdf_sh2`:

| Row | Operation | Disposition | Whole-domain per-column result | Southern focus |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` | `VERIFIED` | 0/9,920 failed | 4/4 pass |
| 2 | `bn2(Nbb)` native `e3w` | `VERIFIED` | 0/9,920; max `5.968673e-16` | 4/4 pass |
| 3 | `eos_rab/bn2(Nnn)` | `VERIFIED` | 0/9,920; max `5.968545e-16` | 4/4 pass |
| 4 | `zdf_sh2` | `DIVERGED` | 9,920/9,920; max `7.329361e+01` | 4/4 fail (`3.036e-4` to `3.988e-3`) |
| 5 onward | bottom drag through implicit solves | `UNMEASURED` | ordered stop at row 4 | ordered stop |

### Row-4 first operand

NEMO starts its active no-Stokes expression with the carried pre-step
viscosity:

```fortran
zsh2u(ji,jj) = ( p_avm(ji+1,jj,jk) + p_avm(ji,jj,jk) ) &
```

Source: NEMO 5.0.2 `src/OCE/ZDF/zdfsh2.F90:80`; the full face products and
four-face assembly are at lines 80-94.  legoESM instead supplies the newly
reconstructed current sub-iteration `K_M_curr`.  That operand comparison is
already `DIVERGED` in 9,920/9,920 wet columns (max column error
`3.3486302069727913`, correlation `0.9984439417185291`, RMS ratio
`1.0146340961467346`); all four focus columns fail (`1.0224e-5` through
`6.8126e-5`).  The day-0 now/before velocity bridge is bit-identical, so the
ordered operand walk stops at `p_avm` before considering the later velocity,
`e3uw/e3vw`, and four-face operands.  Substituting NEMO's carried `p_avm`
alone does not close the composite because later operands remain non-identical;
that is recorded rather than misreported as a failed localization.

Next-round design: add `tke_preclosure_coeff_source` with faithful default
`carried_previous_step` and explicit legacy `current_subiteration`.  Carry
`avm/avt` closure fields across steps and seed a bridged run from restart
`avm/avt`.  Before `tke_avn`, carried `p_avm` must feed `zdf_sh2`, the
`rn2b*p_avm` Prandtl numerator, TKE matrix diagonals (`zdftke.F90:503-506`),
and wave surface denominator (`:538`); carried `p_avt` must feed the
`-p_avt*rn2` RHS (`:514`).  Only the post-solve `tke_avn` overwrite at line
621 and below constructs the coefficients carried into the next step.
Required red tests distinguish carried/current arrays at every consumer,
prove restart identity and next-step carry, lock post-solve sequencing and
legacy bits, and keep JIT/grad finite.  Per the ordered discipline, this is
design only; row 4 is not fixed in this round.

Round-2 machine-readable result:
`docs/ocean/fidelity/dino_zdf_chain_sweep_round2_artifact.json`, SHA256
`bab31d7a8e322e855b187dcfdf0ad19cdb36b4232f1782c15be3c64730234384`.
The stamped probe/tree SHA is
`a58c33d5c7b7d4788f921d8626d9f499e29ff445`.  All planted controls fired.

### Round-2 adversarial review

Two independent final read-only reviews ended `NON-HOLD`.  The measurement
reviewer reran the committed CPU/fp64 sweep from the final production delta and
reproduced the artifact byte-for-byte (SHA256 above), verified every source and
input stamp, and passed 92 delta-focused tests.  The physics reviewer passed 25
prior-regression tests and mutation-tested the production MLD helper: replacing
only its multiplier with reconstructed `diff(gdept)` changes the doubled-native
case from 20 m/base 1 to 40 m/base 2, so the shared-e3w control is red-capable.
Review HOLDs on generic MPAS/ORCA geometry, the flat NEMO bridge, native operand
validation, row-4 provenance, and the full pre-`tke_avn` carry design were all
fixed before sign-off.

The remainder of this document preserves the accepted round-1 evidence and
design history; its statement that the sweep stopped at row 2 is historical.

## Verdict

The sweep stopped at row 2, `bn2(Nbb)`.  `eos_rab(Nbb)` is `VERIFIED`; `bn2`
is `DIVERGED` under its preregistered POINTWISE per-column bar.  The first
failing operand is construction/evaluation order for NEMO's live
`e3w(Kmm)` divisor, not different physical geometry.

The result is not visible in the aggregate statistics: `bn2` has correlation
`1.0` and RMS ratio `1.0`, but 4,630 of 9,920 wet columns exceed the
`1e-15` per-column bar.  This is exactly why the reset requires a whole-domain
column census rather than an aggregate verdict.

The complete 32-row execution-order table, active/dead branch evidence,
measurement classes, fixed bars, focus registry, controls, and stop rule were
committed before measurement in
`scripts/validate/ocean_fidelity/dino_1226/PREREG_zdf_chain_sweep.md`
(preregistration commit `b6c11c309ec5d42e3fa48645849e278724dca38d`).

## Rows reached

| Row | Operation | Disposition | Whole-domain per-column result | Focus-column result |
|---:|---|---|---|---|
| 1 | `eos_rab(Nbb)` alpha/beta | `VERIFIED` | alpha max `3.087467e-16`, beta max `0`; 0/9,920 failed | all four pass |
| 2 | `bn2(Nbb)` | `DIVERGED` | max `6.366585e-15`; 4,630/9,920 failed | all four pass (`1.321191e-17` to `2.020645e-17`) |
| 3 onward | `eos_rab/bn2(Nnn)` through TKE, EVD, `ldf_slp`, and the implicit solves | `UNMEASURED` | ordered stop at row 2 | ordered stop at row 2 |

The aggregate gates pass for both reached rows.  They do not override the
per-column failure.  Row 1 covers 342,134 wet T cells; row 2 covers 332,214
wet W interfaces.  The row-2 reference RMS is `6.811828981750718e-05 s-2`.

## Focus registry

The focus rule was derived from the committed MLD audit maps before this
measurement: all wet southern-basin day-90 columns where
`basin_legacy_base_index_day90 != nemo_base_index_day90`.  The map SHA256 is
`9fb7344d1e6f92232d211f6a52ff8636022f0d9b0b05acea2b0bae6e6afd9bf0`.
The resulting zero-based, halo-stripped `(j,i)` registry is

```text
(11,1), (12,1), (13,1), (13,23)
```

Those four output columns do not expose the `bn2` failure—their output scores
remain below the bar—while the whole-domain census does.  Conversely, the
underlying derived-`e3w` operand fails in all 9,920 wet columns, including all
four focus columns.  This is a direct demonstration that the MLD pattern audit
is useful for targeting but cannot decide term fidelity.

## Operand localization

The active NEMO expression is

```fortran
pn2 = grav * (zaw * dT - zbw * dS) / e3w(ji,jj,jk,Kmm) * wmask(ji,jj,jk)
```

Source: NEMO 5.0.2 `src/OCE/TRA/eosbn2.F90:1465-1467`; DINO calls it with
BEFORE T/S and NOW geometry at `cfgs/DINO/MY_SRC/stpmlf.F90:206`.

Substitutions were applied one at a time, holding all later arithmetic fixed:

| Substitution | Failed wet columns | Max column error | Bar result |
|---|---:|---:|---|
| `mesh_mask:gdepw_0 * (1+r3t)` in `zrw` | 4,630 | `6.366585e-15` | fail |
| NEMO dumped `gdept(Kmm)` in `zrw` | 4,639 | `6.366585e-15` | fail |
| NEMO dumped alpha/beta | 4,632 | `6.366585e-15` | fail |
| `diff(gdept_0) * (1+r3t)` divisor | 1,361 | `1.790602e-15` | fail |
| raw mesh `diff(gdept_0) * (1+r3t)` divisor | 0 | `5.968673e-16` | pass |
| `mesh_mask:e3w_0 * (1+r3t)` divisor | 0 | `5.968673e-16` | pass |
| NEMO dumped live `e3w(Kmm)` divisor | 0 | `5.968673e-16` | pass |

The independently scored operand confirms the distinction:

- the bridged BEFORE T/S are bit-identical to the raw restart `tb/sb` at every
  registered wet point, and the live `gdepw` is at-bar against
  `mesh_mask:gdepw_0*(1+r3t)`; substituting that
  independently reconstructed `gdepw` leaves all 4,630 baseline failures;
- raw mesh `diff(gdept_0)` and raw mesh `e3w_0` are bit-identical over the
  scored interfaces; both raw-mesh constructions close the row after the live
  stretch;
- current `diff(gdept(Kmm))` versus the live dump: all 9,920 columns fail,
  maximum `5.641628e-15`;
- `diff(gdept_0)*(1+r3t)` versus the live dump: all 9,920 columns fail,
  maximum `3.173416e-15`;
- the mesh's independent `e3w_0*(1+r3t)` versus the live dump: 0 columns fail,
  maximum `3.526017e-16`.

Therefore this is not the already-closed question “is the divisor live?”—all
candidates above are live.  The remaining defect is operand construction and
floating evaluation order.  The bridge canonicalizes `gdept_0` by a wet-cell
horizontal mean (`nemo_state_bridge.py:360-364`), then legoESM differences the
already-live depth.  NEMO's raw `gdept_0` difference is exactly its raw
`e3w_0`, and it stretches that reference spacing directly.  Preserving either
raw mesh construction closes this row; preserving `e3w_0` is the most direct
transcription of the operand NEMO actually reads.  The old claim that the
bridged reconstruction is exact has been retracted in the measuring tool and
in `fidelity_bar_gate.py`; the gate can no longer print this row `AT BAR`.

This roundoff-tier whole-domain divergence is **not** claimed to explain the
22.5 m southern MLD pattern.  All four targeted southern `bn2` output columns
already pass before substitution.  The sweep must repair/reverify row 2 and
then continue to row 6 before any MLD-causality claim is possible.

## Next-round fix design (not implemented here)

Add one statically validated bridge/geometry selector, for example
`bn2_e3w_source`, with values `"mesh_reference"` and `"depth_difference"`.
It is one shared geometry choice, not independently selectable TKE/EVD flags
and not an environment variable.

- `"mesh_reference"` is the correct-by-default value for a NEMO bridge.  It
  consumes the raw reference `e3w_0` carried from `mesh_mask.nc` and forms live
  `e3w=e3w_0*(1+r3t)`.  Preserve raw `gdept_0` as well; validate that its
  reference difference agrees with `e3w_0` where the grid promises that
  identity.  If a NEMO-fidelity configuration lacks the selected operand or
  encounters a bad shape/nonpositive spacing, fail closed.
- `"depth_difference"` preserves the legacy `diff(gdept)` behavior and must be
  selected explicitly.  It remains useful for generic/synthetic coordinates
  that do not claim NEMO operand identity.

Implementation shape:

1. extend `NemoGrid`/the NEMO bridge and both vertical-coordinate wrappers to
   preserve raw `mesh_mask:gdept_0` and `e3w_0` as optional array pytree leaves;
   do not horizontally average the bit-uniform reference used for this arm;
2. add a helper returning the live native `e3w` without changing
   `nemo_bn2_live_ladders`' existing two-value API;
3. add an explicit canonical `e3w_int` operand to
   `compute_buoyancy_frequency_nemo_bn2` and thread the selector/operand through
   all TKE, EVD, MLD, GM/Redi, C-grid, and MPAS `nemo_bn2` consumers;
4. use that same exact `e3w_int` wherever the next operation multiplies by
   `e3w` (notably `zdf_mxl`) so the NEMO cancelling pair remains paired;
5. add reader halo/axis, shape/positivity, both coordinate-wrapper propagation,
   JIT, finite `jax.grad` with respect to T/S/eta, pytree, selector-typo,
   missing-mesh-operand, C-grid/MPAS parity, bit-identity legacy, planted
   last-bit divergence, and exact shared-`e3w` bn2-to-MLD cancellation tests.
   The decisive
   regression is the day-180 census: 4,630 failures must become zero and the
   maximum must be no larger than `1e-15` before the sweep may continue.

No production physics fix is included in this round.

## Controls and provenance

The focus set and map SHA were checked mechanically.  The alpha baseline
passes; a planted nonzero wet-cell perturbation makes it fail, and a one-cell
zonal roll also makes it fail.  A planted NaN at a registered wet point is
counted and fails rather than being censored.  Backend is CPU, JAX x64 is enabled, and the
artifact stamps source/dump/restart/mesh SHA256 values and time levels.  No new
NEMO dump slot or NEMO rerun was necessary: the existing PR #1689 dump family
already contained `eiv_dump_e3w.bin`, `eiv_dump_gdept.bin`, alpha/beta, and
`tke_dump_rn2b.bin`.

Machine-readable result:
`docs/ocean/fidelity/dino_zdf_chain_sweep_artifact.json`, SHA256
`949604577cf3900c259e033a3f53c3067ef3740cc50df2100621f50d0eda095b`.
The committed probe SHA stamped inside it is
`88d53850d3f2e9ca95210010caba9d39dbbc1af0`.

The host worktree's administrative Git directory is read-only in this
sandbox.  Commits were therefore made, in order, in writable shadow metadata
cloned from parent `782b0d7887277c88bcaa9c1be24eedad5447d9ae`; the final bundle
is created from that metadata and is the authoritative reachable branch ref.

GitHub issue #1455 could not be read or updated from this sandbox: the `gh`
request failed to connect to `api.github.com`.  This document is formatted as
the evidence record to post when connectivity is available; no claim of issue
publication is made.

## Adversarial review

Two independent read-only reviews ended `NON-HOLD`.  The measurement reviewer
independently reran the final CPU/fp64 probe byte-for-byte, verified every
artifact SHA, and confirmed the preregistration is an ancestor of the stamped
probe commit.  The design reviewer confirmed the source attribution and shared
geometry design.  Findings raised during review—missing `gdepw`/T/S operand
checks, nonfinite censoring, raw-mesh construction discrimination, paired MLD
geometry, exact row-5 lines, and overbroad dry-cell identity wording—were all
dispositioned in the committed probe/table/result before sign-off.
