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
