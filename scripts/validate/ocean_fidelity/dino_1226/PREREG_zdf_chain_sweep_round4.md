# Preregistration: row-4 shear stage and operand peel

Date: 2026-08-28. Status at commit: **no round-4 implementation, focused
test, or post-fix row-4 measurement has run**.

This round resumes the ordered ZDF sweep at its first open row, row 4
(`zdf_sh2`). The accepted round-3 substitution proved that using the
step-entry NOW velocity reduces the composite maximum column error to
`0.03098203766767776`, but all `9,920/9,920` wet columns and all four
southern-basin focus columns still fail the registered pointwise `1e-15` bar.
No later ZDF row may be measured until row 4 passes.

Fixed provenance:

- DINO `WORK/zdfsh2.F90` SHA256
  `c31f062e282b8710a0b565be7aecbc35b8e741b05a96a393402fcd5a737f28f8`;
- committed row-4 probe SHA256
  `b223d6afe5e5232debef64b0d69e31fdcea40992ffb2a95162b227250ecbab07`;
- round-3 artifact SHA256 (recorded after this preregistration by the result
  writer; its path is fixed as
  `docs/ocean/fidelity/dino_zdf_chain_sweep_round3_artifact.json`).

## Production option and scope contract

`TKEConfig.tke_shear_evaluation_stage` has exactly two values:

- `step_entry`: construct `p_sh2` once from the matched step-entry NOW and
  carried BEFORE face velocities and the carried previous-step `p_avm`, then
  freeze that array. The same frozen array feeds the TKE shear-production RHS
  and every `nemo_ri` Prandtl denominator during the implicit TKE iteration.
- `implicit_solve_state`: the historical path, evaluating shear from the state
  presented to the implicit solve. This remains the generic default and is the
  explicit legacy-control selector.

Only the complete DINO NEMO cards `nemo_dino_kamm` and its inherited
`nemo_dino_kamm_mlf` select `step_entry` by default. `nemo_paper`, every DINO
Veros card, the partial ORCA/NEMO fidelity recipe, all Veros recipes, and MPAS
retain `implicit_solve_state`. Their resolved cards are pinned and the legacy
kernel path must be bit-identical to explicit `implicit_solve_state`.

The step-entry mode is valid only for prognostic, face-native NEMO shear with
face-weighted `p_avm`; it requires all NOW/BEFORE face operands and must reject
missing or shape-incompatible frozen `p_sh2`. The legacy mode rejects a frozen
operand rather than silently ignoring it.

Red-capable tests are registered for:

1. a hand-computed matched-step column in which the implicit-solve velocity
   differs from step entry, proving the frozen `p_sh2` feeds both the first TKE
   RHS and the Prandtl denominator;
2. a call-count/poisoned-current-state control proving `step_entry` does not
   re-evaluate the implicit-solve velocity;
3. invalid selector, missing operand, shape mismatch, and legacy silent-no-op
   rejection;
4. resolved-selector pins for every changed and unchanged reachable card;
5. exact-array equality between the omitted/default selector and explicit
   `implicit_solve_state`, plus JIT and finite-gradient coverage.

## Ordered remaining-operand substitutions

After the stage fix, the row-4 composite is rerun first. If it is still
`DIVERGED`, the following operands are substituted **one at a time**, holding
all later operands at the legoESM value, in literal NEMO evaluation order:

| Subrow | Operand | Oracle line | Bar and disposition |
|---|---|---|---|
| 4c | live NOW `e3uw/e3vw(Kmm)` | `zdfsh2.F90:83,88` | pointwise `1e-15`; 0/9,920 plus all focus columns => VERIFIED, otherwise DIVERGED |
| 4d | live BEFORE `e3uw/e3vw(Kbb)` | `zdfsh2.F90:83,88` | same |
| 4e | `wumask/wvmask` multiplication | `zdfsh2.F90:84,89` | same |
| 4f | literal `0.25` four-face sum and wet-coast doubling/indexing | `zdfsh2.F90:92-94` | same |

The exact metric candidate is the raw-mesh `e3uw_0/e3vw_0` multiplied by the
QCO live face stretch at each time level (`domzgr_substitute.h90:132-133`),
using the already committed `sh2_walk.py` reconstruction. A failure at 4c/4d
implies a separately selectable `nemo_qco_live_face` metric source, faithful by
default only on the two complete DINO NEMO cards; legacy T-point Jacobian
spacing remains opt-in there and unchanged everywhere else. A failure at 4e/4f
implies a separately selectable literal NEMO wet-face/coastal assembly with the
same scope rule. No such production fix is authorized until its substitution
is measured.

Row 4 passes only at `0/9,920` failing wet columns, all four southern focus
columns passing, maximum column error `<=1e-15`, and all perturbation,
horizontal-roll, and nonfinite controls firing. A miss stops the sweep here.
On a pass, rows 5 onward resume exactly in
`PREREG_zdf_chain_sweep.md`, retaining their registered bars and focus-column
scorecard through avm/avt assembly, EVD, and the implicit tracer/momentum
applications.

## Dated climate-prediction amendment (before GPU execution)

The numerical CONFIRM/REFUTE bands frozen in the round-3 preregistration do
not change. The ownership hypothesis now names the accumulated row-4 repairs
(carried preclosure coefficients plus frozen step-entry `p_sh2`, and any later
row-4 operand fix that is separately measured and implemented), not the carry
alone. Baseline remains `22.479491 m`; CONFIRM remains `<=11.2397455 m`,
REFUTE remains `>=20.2775 m`, with the same legacy-baseline, acceptance-floor,
pass-tally, and southern-density requirements.

The faithful arm remains the new defaults:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_row4_faithful_d90.npz --days 90 --save-3d \
  --bridge-tke
```

The byte-reproduction control must opt out of both row-4 defaults:

```bash
CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf /tmp/zdf_row4_legacy_d90.npz --days 90 --save-3d \
  --bridge-tke --tke-preclosure-coeff-source current_subiteration \
  --tke-shear-evaluation-stage implicit_solve_state
```

The climate arms are **not yet authorized**: row 4 must first be clean at its
registered bar, and the ordered sweep must either reach the implicit-solve
application or stop at a newly localized later divergence with no unmeasured
upstream operand.

### 2026-08-28 amendment after registered 4c/4d measurement, before fix/GPU

The sequential substitution measured 4c (NOW face metric) at `9,920/9,920`
failures, maximum `1.549657e-02`; adding 4d (BEFORE face metric) produced
`0/9,920`, maximum `0.0`, with every focus column passing. Therefore the
preregistered `nemo_qco_live_face` metric option is now authorized for
implementation and becomes the default on the same two complete DINO NEMO
cards. The legacy selector is `tpoint_jacobian`; all other cards retain it.
The numerical climate bands remain unchanged. The legacy GPU command gains
one additional required opt-out:

```text
--tke-shear-metric-source tpoint_jacobian
```

Omitting that flag would no longer reproduce the old row-4 chain and invalidates
the control. The faithful command remains option-free.

### 2026-08-28 row-8 localization amendment, before operand measurement

The corrected row-4 composite and rows 5--7 have crossed their registered
bars.  The first subsequent output checked by the existing row table, the TKE
surface Dirichlet boundary at row 8, fails narrowly (`154/9,920` columns,
maximum normalized column error `1.638670e-15`).  Rows 9 onward are therefore
not citable measurements from that exploratory run and remain unmeasured.

Before rerunning, row 8 is split in NEMO's literal order
(`MY_SRC/zdftke.F90:334,360-364`):

1. infer NEMO's live `taum` from unfloored dumped surface `en` using the
   separately evaluated `zbbrau`, and compare it to legoESM's supplied
   `SurfaceForcing.taum`;
2. evaluate `zbbrau = rn_ebb / rho0` with the resolved double-precision
   operands and compare the inferred input/output relation;
3. evaluate the product and `MAX(rn_emin0, zbbrau*taum)` with literal NEMO
   association.

Every comparison retains the pointwise per-column `1e-15` bar and the four
southern focus scores.  A failed input identity localizes row 8 to the forcing
operand; an input pass followed by an output failure localizes it to arithmetic
association.  A planted one-cell perturbation and nonfinite injection must
fail.  The ordered sweep stops at row 8 unless the literal construction passes
at `0/9,920`; no row 9+ result may be promoted across a row-8 miss.

The first registered peel above localizes the miss to supplied `taum` in the
same 154 unfloored columns.  Before looking inside that operand, its source is
therefore split in the literal `usrdef_sbc.F90:221-223` order: `gphiu`, nearest
knot/interval selection, `zs`, the left-associated cubic expression
`val_s + (val_n-val_s)*(3-2*zs)*zs**2`, `ABS(utau)`, then the conditional
`*1.3`.  `sbc_dump_utau.bin` is the oracle input.  The NEMO-literal
reconstruction must match that dump at the same `1e-15` column bar, and
substituting the dump-derived `taum` into row 8 must give `0/9,920` failures;
otherwise attribution stops at the earliest failed suboperand.

### 2026-08-28 review amendment, before the final rerun

Adversarial review found three specification/execution gaps; this amendment is
committed before their corrected measurement:

- The `nemo_dino_kamm` forward-Euler card retains its registered
  `squared_centered` shear discretization and has no leapfrog BEFORE velocity.
  Its `step_entry` stage therefore freezes *that configured formulation* at
  entry, `carried_avm * squared_centered_shear(NOW)`.  The MLF card alone
  freezes the measured face-native NOW x BEFORE formulation.  Both feed their
  frozen `p_sh2` to RHS and Prandtl.  This is a scope clarification, not a
  selector change.  An actual resolved-FE-card helper test and a JIT/gradient
  test of the MLF live-QCO helper are required before rerun.
- Row 6 is an accumulating/exact-index row.  It is rescored at its registered
  `1e-12` A bar **and** exact integer equality, with an off-by-one planted
  control.  The prior `1e-15` float presentation is retracted from the final
  artifact (its exact-zero result did not change the disposition).
- `tke_dump_en.bin` is post-`tke_tke`, so it cannot be row 8's primary oracle.
  The corrected row-8 reference is independently constructed from the
  registered `sbc_dump_utau.bin` and resolved `rn_ebb/rho0/rn_emin0` at
  `zdftke.F90:334,361`.  Equality to poststage `en(:,:,1)` is retained only as
  a separately labeled downstream-invariance check.  The final probe also
  rejects all effective ablation/precision environment overrides, derives its
  git/probe SHAs rather than accepting arbitrary stamps, and includes the NEMO
  state-bridge source in provenance.

The bars, focus registry, first-divergence stop rule, climate prediction, and
GPU commands remain unchanged.
