# Preregistration — ORCA2 round 182 HPG fold acquisition repair

Date: 2026-10-08. Frozen base: `817d75fb5`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round182/`.

This round repairs only the failed round-181 acquisition launcher. Every
scientific claim remains **independent hierarchy rung 0**. No model file,
configuration, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry may change.

## Observed failure and frozen repair

The operator's round-181 run compiled `ORCA2_OMIP_L4_R181HPGFOLD`
successfully, then refused before patching or running NEMO:

```text
error: affected file '.../ORCA2_OMIP_L4_R181HPGFOLD/MY_SRC/dynhpg.F90' is beyond a symbolic link
REFUSE: round-181 acquisition failed at line 0 (exit 1)
```

`/home/dbalwada/oracle-builds` resolves through a filesystem symlink. The
launcher's `git apply --directory=<target MY_SRC>` rejects that parent path.
The established acquisition pattern in this repository is `patch --fuzz=0`
followed by a byte comparison with the syntax-proved scratch source. The
repair uses that pattern and a fresh target/config/run name; it never deletes,
reuses, or overwrites the failed target.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R182-P1 | The operator failure occurred after the base build but before the writer patch and NEMO run. | Build log says `Compilation successful`; no target run directory or HPG-fold record exists; target `dynhpg.F90` equals the pinned source hash. | A target record exists, the writer is already present, or the source hash moved. |
| R182-P2 | Exact-zero-fuzz `patch` is a faithful replacement for `git apply` here. | The patched target source is byte-identical to the already syntax-proved scratch source; a one-line perturbation makes the equality control fail. | Patch uses fuzz, changes a source line outside the committed additions-only patch, or differs from scratch. |
| R182-P3 | The repaired launcher is preflight-ready under a fresh target. | Syntax proof and writer-layout census pass; every layout/content plant fires; the failed target remains untouched. | Any preflight/control stays green or the old target changes. |
| R182-P4 | This round reads no scientific payload. | No new rank record is admitted and no operand result is reported. | Any HPG operand is interpreted before admission. |

## Terminal rule

If preflight and plants pass, stop with `ACQUISITION_NEEDED` naming the fresh
round-182 launcher. After the operator run, the next round admits the exact
evaluated fold operands, replays `zhpj` top-down, and then analyses raw HPG,
`e3v`, `vmask`, and `r1_hv0` atomically. Failed predictions remain in the
receipt.

ASKED choices: exact mechanical repair of the failed acquisition under a fresh target.
UNASKED choices: empty.
