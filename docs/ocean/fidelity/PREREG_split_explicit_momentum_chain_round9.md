# Preregistration: row-1.3 QCO continuity operand ladder, round 9

Date: 2026-08-29. Frozen before the SLOT build and before any new operand
measurement.

## Ordered stop and active oracle path

Round 8 localized the held-chain first-substep SSH divergence to the continuity
composition at active DINO `cfgs/DINO/MY_SRC/dynspg_ts.F90:651-725`.
`zsshp2_e` is fed by bit-exact held inputs, while the algebraically implied
continuity divergence is the campaign's largest normalized divergence so far:
`E=0.453077140495226`, correlation `0.9141355599848835`, and mean-absolute
ratio `1.0523503032867603` on 9,920 wet T cells.

DINO defines `key_qco` and `key_vco_3d`, but not
`key_qcoTest_FluxForm`.  Therefore the executed source order is:

| subrow | operand | active NEMO expression | population | arithmetic bar |
|---:|---|---|---:|---|
| 9.1 | `zsshp2_e` | AB3 SSH extrapolation, `dynspg_ts.F90:651-657` | 9,920 wet T | POINTWISE `1e-15` |
| 9.2 | `zhup2_e` | QCO area-weighted U-face depth, `:663-681` | 9,758 wet U | POINTWISE `1e-15` |
| 9.3 | `zhvp2_e` | QCO area-weighted V-face depth, `:663-686` | 9,868 wet V | POINTWISE `1e-15` |
| 9.4 | `zhU` | face-local `e2u*ua_e*zhup2_e`, `:698-701` | 9,758 wet U | POINTWISE `1e-15` |
| 9.5 | `zhV` | face-local `e1v*va_e*zhvp2_e`, `:702-704` | 9,868 wet V | POINTWISE `1e-15` |
| 9.6 | `zhdiv` | adjacent-face flux divergence, `:718-724` | 9,920 wet T | ACCUMULATING `1e-12` |
| 9.7 | `ssha_e` | `sshn_e-rDt_e*(ssh_frc+zhdiv)`, `:724-725` | 9,920 wet T | ACCUMULATING `1e-12` |

For every operand, **AT BAR** requires all three standing axes: correlation
`>=1-1e-9`, `abs(mean_abs_ratio-1)<=1e-6`, and maximum absolute difference
divided by oracle RMS no larger than the registered arithmetic-class bar.
Anything finite outside a bar is **DEBT**.  The scorer stops ownership at the
first DEBT subrow; later scores are targeting context only.

The six new streams are direct, write-only snapshots after the unchanged
production assignments; `zhU/zhV` are copied after any active AGRIF/W&D
commit, so they are the literal inputs consumed by continuity. They are full `jpi*jpj` fp64 streams
(`56*203*8 = 90,944` bytes), with cells outside each production loop's exact
bounds zero-filled before copying. The writer fires only at `kt==nit000` and
`jn==1` through the existing `ll_spg_dump` guard. No oracle value is injected
into NEMO.

## Offline held-substitution ladder

The half-step `ua_e/va_e` operands remain independently bit-exact from round 8.
Using those same existing dumps, the scorer evaluates the following nested
arms in production order. Each arm changes only the named operand and retains
legoESM's production code for every subsequent available operation.

1. **BASE:** legoESM production `nemo_ssh_avg_face_depth`, product, and
   `divergence_cgrid` on the exact oracle eta and half-step velocities.
2. **Q (depth held):** replace only the production face depths by the direct
   NEMO `zhup2_e/zhvp2_e`; retain the exact velocities, production product,
   and production divergence.
3. **F (metric-flux held):** replace Q's product by direct NEMO `zhU/zhV`,
   converted to the velocity-like input of `divergence_cgrid` with that
   operator's exact production face widths. This makes its subsequent
   multiplication recover the dumped NEMO metric flux while retaining
   legoESM's cell-area division, differencing, and boundary representation.
4. **B (NEMO divergence commit):** replace the subsequent divergence/boundary
   result by the independently dumped NEMO `zhdiv`. This is a closure arm, not
   by itself proof of a single boundary statement.

This is an ordered substitution ladder, not a fictitious `2^k` matrix: F
contains Q's depth in its observed product, and B contains F. Interactions
between nested arms are therefore not identifiable. The first arm whose
prediction uniquely meets the ownership bars owns the earliest bounded
composition:

- Q confirms: QCO face-thickness weighting owns the row;
- Q refutes and F confirms: the depth/velocity/face-metric product composition owns;
- F refutes and B confirms: the remaining divergence/boundary representation
  composition owns, with no narrower statement claim;
- no unique confirmation: `OPEN_UNRESOLVED`.

For each arm, define residual `R = NEMO_zhdiv - BASE_zhdiv`, prediction
`P = ARM_zhdiv - BASE_zhdiv`, normalized prediction error
`RMS(R-P)/RMS(R)`, explained fraction
`1-RMS(R-P)/RMS(R)`, and correlation `corr(P,R)`.

**CONFIRMS** only when normalized prediction error `<=0.10`, explained
fraction `>=0.90`, and correlation `>=0.99`. **REFUTES** only when normalized
error `>=0.90`, explained fraction `<=0.10`, and correlation `<=0.20`.
Everything between is **UNRESOLVED**. The whole 9,920-cell wet T domain is
primary. Companion scores are printed separately on the two end-wall rows
used by the basin preregistration (`j=1` and `j=197` in its convention); a
whole-domain confirmation may not be narrowed to a boundary statement if the
wall-row prediction contradicts it.

These prediction bars and the planted vector-sign/halo-offset control are
reused verbatim from
`PREREG_endwall_barotropic_source.md` on
`fidelity/dino-basin-rectification-codex`. That preregistration's uniform-eta
fixer is excluded here because it runs after the solver and cannot precede
substep 1. Its temporary Kmm transport-mean candidate is also excluded by the
`dyn_zdf`/`mlf_baro_corr` execution order. NEMO's second post-loop LBC on
`un_adv/vn_adv` remains tracer bookkeeping and is not row-1.3 eta ownership.

## Controls and disposition

The run is admissible only if all pre-existing `.bin` streams are byte-exact
between instrumentation ON and the frozen OFF run, the six and only six new
streams appear, every new stream has 90,944 bytes, all SHA slots are bound,
and the one-bit-file and missing-stream controls fire. The scorer must also
make a planted U halo offset and a planted V vector-sign error fail their raw
operand gates, and a zero prediction must not confirm the nonzero residual.

No production fix is authorized by this preregistration. A confirmed Q/F/B
owner is a localization result to be reconciled with basin candidate A before
any selector or free run. Rows 1.4 and 2--6 remain ordered-blocked.
