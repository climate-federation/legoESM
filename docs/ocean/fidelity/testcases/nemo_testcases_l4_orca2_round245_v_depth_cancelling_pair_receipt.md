# ORCA2 round 245 — midpoint V-depth cancelling pair

Date: 2026-10-11. Frozen base: `a702e3a9d`. Preregistration commit:
`67c2a16ba`. Final V-only measurement commit: `b028af351`. Status:
**HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round245/`.

This round changes no effective package/model file, deck, selector, carried
state, stabiliser, sea-ice setting, or `unmeasured_features` entry. All results
are reported separately as **independent** and **given NEMO's entry**; the two
labels have identical counts and maxima. No NEMO acquisition was required.

## Compiled statement and instrument

The executing OMT-4 build writes the V-face midpoint depth as

```
zhvp2_e = hv_0 + 0.5 * r1_e1e2v
         * (e1e2t(j) * zsshp2_e(j) + e1e2t(j+1) * zsshp2_e(j+1))
         * ssvmask
```

at `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519`,
then writes `zhV=e1v*va_e*zhvp2_e` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:530-535`.
The same compiled tree builds `vmask` from adjacent T masks, associates it as
a V point, and reduces it vertically to `ssvmask` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dommsk.f90:206-249`. Its T-pivot
T/W boundary maps the halo row from the mirrored row two positions south at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:945-959`.

The committed gate reuses the admitted round-222 twins and round-244 passive
trace. It replays the written arithmetic in source order, then runs two causal
arms: raw NEMO `ssvmask` on V faces only, and that V-only mask atomically with
the T-pivot north-neighbour association. The U mask stays fixed. The ordinary
and traced completed states are array-identical. Both record twins are exact
on every defined payload. Execution is production JIT on CPU with fp64, x64,
and libm. The committed instrument is
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round245_v_depth_split.py;
the same file contains the fail-closed pair classifier.

## Frozen predictions

| prediction | result |
|---|---|
| R245-P1: admitted, passive, fp64 measurement under both labels | **CONFIRMED** |
| R245-P2: D114 already makes `hv_0` and `ssvmask` exact | **REFUTED**: `hv_0` is exact, but current `ssvmask` differs on 35 V faces |
| R245-P3: first debt is a horizontal-area operand | **REFUTED**: both area operands are exact; the first direct non-bit operand is `ssvmask` |
| R245-P4: the first operand alone closes the chain | **REFUTED**: raw `ssvmask` alone leaves 35 depth cells non-bit at substep 3 |
| R245-P5: a cited statement is eligible only after an atomic score and full trajectory gates | **HELD**: the direct pair is exact, but the complete private unit has not been rescored through kt=8 and the full gates |
| R245-P6: every control fires | **CONFIRMED** |

## Source-ordered split — both labels

At substep 3, raw `hv_0`, `r1_e1e2v`, `e1e2t`, and the owned midpoint SSH
are bit-exact. The current compact `ssvmask` is the first non-bit operand:
35/13,320 cells, maximum 1 at `[147,29]`. The baseline depth is non-bit on
35 cells, maximum 0.0033989498219852976 m at `[147,49]`.

The V-only raw-mask arm makes `ssvmask` exact without touching U. It reduces
the substep-3 depth maximum to 0.002204577618613257 m but leaves the same 35
cells unequal. The residual is the north T-point operand that the old zero
mask hid. This is the measured cancelling-pair signature, not single-operand
closure.

| substep | raw-V-mask-only midpoint V unequal / max | depth unequal / max (m) | `zhV` unequal / max (m3/s) |
|---:|---:|---:|---:|
| 3 | 0 / 0 | 35 / 0.002204577618613257 | 35 / 0.10368071639095433 |
| 4 | 66 / 1.5767403506970756e-10 | 68 / 0.004414719408487144 | 68 / 0.3772472336422652 |
| 5 | 100 / 6.083023629081807e-10 | 97 / 0.005797299485493568 | 100 / 0.8289895471534692 |
| 6 | 129 / 1.3378321328354204e-09 | 119 / 0.005326659807451506 | 129 / 1.2236731614684686 |

With raw V-only `ssvmask` **and** the T-pivot north-neighbour association,
midpoint V, midpoint V depth, and `zhV` are each bit-exact at substeps 3, 4,
5, and 6: 0 unequal and maximum 0 under both labels. The earlier two-mask
diagnostic gives identical rows, proving that its U-mask substitution was
inert; the final claim rests only on the V-only runs.

The pair classifier binds every report to its clean measurement commit,
requires the mask-only arm to remain non-exact, and requires all three direct
outputs to close at all four substeps. Its label-coverage, single-arm-control,
and pair-closure plants each exit red. The earlier classifier's source-order,
one-bit, replay-closure, production-route, and passivity plants also each exit
red. `vonly_pair_classification.json` reports
`HELD_R245_MASK_T_HALO_CANCELLING_PAIR_NAMED`; hashes are in the evidence
`SHA256SUMS`.

## Landing disposition

Nothing lands in production this round. The cited two-statement unit closes
the immediate statement and its registered substep-6 consumer exactly, but
Decision 96 and preregistration R245-P5 require the complete round-217 private
unit to survive its former kt=8 live-thickness refusal and then pass the
OMT-1..4, rung-0, rung-10, GYRE, DINO, tank, moved-row, and test gates. A
direct-boundary closure is not a trajectory landing predicate.

## Evidence, tests, and review

Focused round-245 tests pass 4/4 and cover source-order classification, all
baseline plants, V-depth replay sensitivity, native-to-compact mask mapping,
the single-arm control, the two-label atomic closure, and every pair plant.
The citation gate, once-only fidelity battery, and independent review are
reported below after their final executions.

## OPEN

Build one private production-shaped candidate containing raw NEMO V surface
mask routing plus the T-pivot north-neighbour association, atomically with the
complete round-217 unit. First require the direct 0/0/0 substep-3..6 table to
reproduce. Then score OMT-4 through kt=8: whether the former live-W-thickness
refusal disappears, its first boundary if not, and the complete five-operand
`zFv` table. Only if that atomic candidate survives does it proceed to the
full Decision-96 landing gates. No acquisition or user choice is currently
needed; Decision 116 remains pending separately and is not part of this unit.
