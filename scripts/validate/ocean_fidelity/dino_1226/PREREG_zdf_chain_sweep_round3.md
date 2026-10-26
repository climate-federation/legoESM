# Preregistration: carried TKE coefficients and ZDF continuation

Date: 2026-08-28. Status at commit: **no round-3 implementation, test, or
post-fix measurement has run**.

This round accepts row 4 of the ordered day-180 sweep as the first divergence.
It implements the already-registered `tke_preclosure_coeff_source` design,
re-verifies row 4, and then resumes the table in
`PREREG_zdf_chain_sweep.md` at row 5. All earlier bars, focus columns, controls,
and the first-divergence stop rule remain unchanged.

## Oracle lifetime and fixed source

NEMO calls `zdf_sh2(...,avm_k,sh2)` before `zdf_tke(...,avm_k,avt_k)`
(`src/OCE/ZDF/zdfphy.F90:268,286`). Inside `zdf_tke`, the incoming pair is
read by the Richardson/Prandtl expression, TKE matrix, stratification RHS, and
wave denominator (`cfgs/DINO/MY_SRC/zdftke.F90:489,503-506,514,538`). Only
after the solve does `tke_avn` overwrite the pair for the next step
(`zdftke.F90:264,621,832-844`). EVD and later enhancements act on the separate
composed `avm/avt`, after `avm_k/avt_k` have been copied
(`zdfphy.F90:311-336`); they are not part of this carry.

Source SHA256 at preregistration:

- DINO `MY_SRC/zdftke.F90`:
  `5623a7444048927c766b0fd23ba4cc5854028e2ead67f1750febd789105c025e`
- upstream `src/OCE/ZDF/zdfphy.F90`:
  `76eeec2a280d56c55bec9e59b579e5290623c408c03962d04652e406334b45fd`
- matched restart `DINO_00005760_restart.nc`:
  `0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e`
- row-4 `tke_dump_avm_in.bin`:
  `f05299d26b198ffad1449bf29bebfa8f2035de0b339be030bbf1c4eba0bce5bd`

The restart carries `en`, `avm_k`, `avt_k`, and `dissl`, exactly the four
fields read by NEMO `tke_rst` (`MY_SRC/zdftke.F90:1027-1041`). A bridged
restart therefore seeds all closure memory from those fields; no inferred
`avt` is permitted.

## Fix and scope contract

`TKEConfig.tke_preclosure_coeff_source` has exactly two values:

- `carried_previous_step`: use carried closure `avm_k/avt_k` for face-weighted
  shear, the `rn2b*avm/(sh2+rn_bshear)` Prandtl operand, both TKE matrix
  off-diagonals, `-avt*rn2`, and the wave Neumann denominator. The post-solve
  closure output becomes the next step's carry.
- `current_subiteration`: explicit historical reproduction; derive those
  operands from the current TKE sub-iteration as before.

The generic `TKEConfig` and every non-oracle recipe retain
`current_subiteration`, byte-identically. The complete DINO oracle cards
`nemo_dino_kamm` and inherited `nemo_dino_kamm_mlf` select
`carried_previous_step`; the partial diagnostic `nemo_paper` card stays on its
existing non-prognostic path. Other NEMO-oriented configurations are inventoried
in the result; none is silently changed without a matched-state certificate.

Unknown selectors, a carried selector without prognostic TKE, missing or
shape-incompatible carried fields, and a carried surface coefficient missing
when the NEMO z=0 matrix row needs it must fail loudly. The legacy selector
must reject supplied carry operands so a supposedly controlled arm cannot
silently ignore them.

## Red-capable tests and row-4 bar

Before implementation, at least one focused test must fail because the option
or carry is absent. After implementation the tests must cover:

1. a hand-computed two-level matched-step case with distinct carried and
   current `avm/avt`, checking shear, Prandtl, both matrix neighbours,
   stratification RHS, and surface coefficient consumption;
2. restart `avm_k/avt_k` halo stripping, axis order, bridge identity, and
   next-step overwrite by the post-solve closure rather than composed EVD;
3. faithful DINO-card defaults and byte-identical resolved configurations for
   every non-oracle DINO recipe;
4. selector/shape/missing-carry failures, JIT, and finite gradients;
5. a legacy arm equal bit-for-bit to the pre-fix current-subiteration path.

Row 4 is `VERIFIED` only at `0/9,920` diverged whole-domain columns, all four
southern focus columns passing, and maximum column error `<=1e-15`. The
perturbation, horizontal-roll, and nonfinite controls must still fail. Any miss
stops the sweep at row 4.

## Ordered continuation

After row 4 passes, measure rows 5 onward in the committed call table, without
reordering or using downstream composites to skip an operation. Every row
retains its registered class/bar and reports whole-domain plus the four focus
columns. Stop at the next `DIVERGED` row and localize its first operand in NEMO
evaluation order. Missing dump slots require the existing write-only stream
instrument pattern plus dump-off/dump-on one-step CPU output/restart identity.

## Frozen 90-day climate prediction and handoff

This prediction is prospective and is not used to accept the matched-step
fix. Baseline southern-basin day-90 MLD RMS is `22.479491 m` (rounded campaign
headline `22.4795 m`). The control is the same committed code and card with
only `tke_preclosure_coeff_source=current_subiteration`.

If the carried-coefficient defect owns the basin MLD error:

- **CONFIRM ownership:** faithful-arm RMS `<=11.2397455 m` (at least 50% below
  baseline), legacy-control RMS within `0.001 m` of `22.479491 m`, no one of
  the five acceptance-gate absolute errors worsens versus control by more than
  its one-floor value, the 5x pass tally does not decrease, and at least one of
  the two southern surface-density errors improves by at least one floor.
- **REFUTE ownership:** faithful-arm RMS `>=20.2775 m` (the audit's registered
  DIFF boundary), or the legacy control misses its baseline band, or any
  acceptance metric worsens by more than one floor, or the 5x pass tally falls.
- Results between the RMS bands, with all controls valid, are
  `PARTIAL/INDETERMINATE`, not promoted to ownership.

The unmodified acceptance-gate one-floor values are: ACC `0.091 Sv`, upper
contrast `1.1e-4 kg m-3`, deep contrast `4.5e-5 kg m-3`, southern surface
sigma max `9.5e-5 kg m-3`, and mean `9.5e-5 kg m-3`.

Exact GPU handoff commands (GPU index is the operator's allocation):

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

The MLD score must reuse the committed audit's exact symmetric NOW-state
criterion, NEMO area/mask, region rows, and day-90 baseline; no native-online
`hmlp` claim is made because the twin NPZ still lacks BEFORE T/S snapshots.
