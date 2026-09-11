# NEMO testcase L2 GYRE round 55 — TKE floor interim receipt

Date: 2026-09-11. Base `3a8d94ea1985`; CPU/fp64. **INTERIM: the replacement
oracle record does not exist, and no trajectory number from this tree is
citable.** The repository Git metadata is read-only in this environment, so
the mandatory clean commit stamp cannot yet be made.

## Source verdict and transcription

GYRE resolves `nn_pdl=1`, `nn_mxl=3`, `ln_mxl0=T`, raw `rn_mxl0=0.04`,
`nn_etau=0`, `nn_htau=1`, `ln_lc=T`, `rn_lc=0.15`, and (because there is no
ice model) `nn_eice=0` in R41ADVSP `ocean.output:568-600`. The compiled card
derives `rmxl_min=1.e-6/(rn_ediff*SQRT(rn_emin))` and then assigns
`rn_mxl0=rmxl_min` (`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:
810-816,828-831`). `tke_avn` uses that effective value in the surface bound
(`:580,603-611`), then derives `avm/avt/dissl` at `:681-694`.

The shared `_mxl0_surface_anchor` and its no-explicit-anchor fallback now use
`cfg.mxl_min`, not the dead raw `cfg.mxl0_min_m`. A red unit control proves
the prior `0.04` result differs from the required `0.01`. The separate
`bottom_level == N` pin debt was not touched.

## Replacement acquisition

The repaired operator script gates the *resolved* source `ocean.output`
instead of grepping only `namelist_cfg`; this removes the false refusal on the
inherited `nn_pdl=1`. The WRITE-only record now includes the resolved scalars,
`taum`, masks/backgrounds, entry `en/avm/avt/dissl`, `rn2/rn2b/sh2`, matrix,
RHS, solved `en`, both mixing lengths, Prandtl factor, closure `avm/avt/dissl`,
and `zdfphy`'s copied pre-EVD `avm/avt`. The latter boundary is compiled
`zdfphy.f90:334-351`, before EVD at `:359`; `nn_pdl` itself branches only in
compiled `zdftke.f90:187-193,386-405,690-694`. Source patches dry-apply with
zero shipped lines removed. Six independent plants (header, truncation, NaN,
resolved config, closure-copy, stamp) are required to exit nonzero.

Calibration and the substitution ladder remain **UNMEASURED** until the
operator runs the newly emitted round-55 script and supplies the record.

## Rule 12 and gates

| card | disposition |
|---|---|
| GYRE | Executes the changed shared anchor; kt=1..10 and days 1–30 are blocked on a clean commit stamp. Frozen kt3 prediction: max T/S `1.6275031290e-4` / `6.3278533133e-6`, from `8.7413164119e-3` / `1.3568885211e-3`. |
| LOCK_EXCHANGE | Does not execute TKE: tank `namelist_cfg:131` resolves `ln_zdfcst=.true.`. |
| OVERFLOW | Does not execute TKE: tank `namelist_cfg:128` resolves `ln_zdfcst=.true.`. |
| DINO | Both NEMO cards execute this shared code but already resolve `mxl_min=mxl0_min_m=0.01`; trajectory unchanged by this statement. Its leapfrog branch is separate and remains at risk/unmeasured here. |
| ORCA2 | UNMEASURED-with-spec: build its card and bit-score closure `avm/avt` from that card's own compiled operands. |

Focused tests: `176 passed`; the expanded reader plus floor control rerun:
`7 passed`. The kt=1..10 gate then correctly refused before writing a report:
the working tree was dirty and commit `3a8d94ea1985` did not identify it.

ASKED: writer correction, shared floor, Rule-12 audit. UNASKED: bypass the
stamp, change the bottom pin, change any card/configuration, or run NEMO.
