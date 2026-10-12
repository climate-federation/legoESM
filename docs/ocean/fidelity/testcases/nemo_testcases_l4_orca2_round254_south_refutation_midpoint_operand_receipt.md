# ORCA2 round 254: south-row refutation and kt=8 midpoint operand split

Date: 2026-10-12. This is an ocean-only OMT-4 measurement round. The shipped
ORCA2 card, SI3, and its `unmeasured_features` tuple are unchanged. No file
under `packages/` changes in this round.

## Frozen question and record

The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round254.md`. Every number below was measured
separately as **independent** (OMT-4's corrected initial state) and **given
NEMO's entry**. The two reports agree in every quoted value; they were not
pooled.

The operator-run round-253 acquisition is admitted as
`PASS_R253_OMT4_SPG8_ADMISSION`. Its rank-complete record contains 65 external
substeps on each of two ranks, and its kt=10 restart is byte-identical to the
admitted OMT-4 reference. This round uses that record read-only. Both probes
require production JIT, CPU, fp64/libm, a clean exact tree, the production
D114 resolved-mesh route, and both claim labels.

## Source order

The compiled OMT-4 oracle associates `ua_e`, `va_e`, `hu_e`, `hv_e`,
`hur_e`, `hvr_e`, and `ssha_e` in one `lbc_lnk` call at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:728-730`.
The T-pivot implementation writes the northern fold rows at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:584-638`; it does not
rewrite the southern closed boundary.
legoESM's private complete-unit image enters its V association at
`barotropic_latlon_cgrid.py:688-742`.

After each external substep, NEMO rotates the current and two before SSH
histories at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:791-793`. At the
next substep it evaluates

`zsshp2_e = za1*sshn_e + za2*sshb_e + za3*sshbb_e`

at `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493`;
`ts_bck_interp` supplies the coefficients at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1496-1531`.
legoESM's corresponding literal interpolation is
`barotropic_latlon_cgrid.py:2006-2025`.

## The suspected south-row writer is refuted

The complete private unit was rolled through kt=1..7. At kt=8, all other unit
parts were retained while the seven-array association was replayed offline at
the first inspected substep. The exact south-row census is:

| field | unit ON versus OFF, unequal cells on south row | unit ON versus NEMO |
|---|---:|---:|
| U velocity | 0 | 0 |
| V velocity | 0 | 0 |
| U depth | 0 | 0 |
| V depth | 0 | 0 |
| U inverse depth | 0 | 0 |
| V inverse depth | 0 | 0 |
| SSH | 0 | 0 |

Rows 1 and 2 are likewise unchanged where the closed-boundary contract
requires it. The control is non-vacuous: the same replay changes 271 northern
fold cells (91 U-family and 180 V-family values). Therefore R254-P2 is
**REFUTED and retained**: neither scalar padding nor any other member of the
seven-array association writes the southern boundary. R254-P3 through P5 are
not activated, and no repair or partial held association is landed.

All seven unique south-audit plants fire: label coverage, admission,
source-order, south support, old-behaviour sentinel, north-fold non-vacuity,
and one-bit detection. The classifier is
`REFUTED_R254_SOUTH_ASSOCIATION_HYPOTHESIS` under both labels.

## First non-bit midpoint operand

The admitted record permits the preregistered fallback split. At substep 3,
the NEMO histories are reconstructed in compiled rotation order as the
substep-2 exit, substep-1 exit, and initial current field. Candidate and NEMO
midpoint arithmetic each reproduce their own recorded midpoint bit-for-bit
(0/26,640 unequal); therefore the interpolation statement and its observer are
exonerated.

| operand, in NEMO statement order | unequal / 26,640 | maximum absolute difference | RMS difference | argmax `[j,i]` |
|---|---:|---:|---:|---:|
| `sshn_e` | 16,433 | 3.9615338373618567e52 m | 3.4325056231964855e50 m | `[147,48]` |
| `sshb_e` | 16,433 | 7.060633126433089 m | 0.09934199315608172 m | recorded |
| `sshbb_e` | 16,433 | 7.091307702123102 m | 0.0990932191663574 m | recorded |
| `za1` | 0 | 0 | 0 | none |
| `za2` | 0 | 0 | 0 | none |
| `za3` | 0 | 0 | 0 | none |

The complete midpoint output differs on 16,433 cells, with maximum
`7.05590772539439e52 m`, RMS `6.113652928003376e50 m`, and argmax
`[147,48]`. Substituting only NEMO's `sshn_e` reduces the maximum to
`5.5064730616231765 m` and RMS to `0.07768376106648628 m`; substituting an
older history or an already-exact coefficient alone leaves the explosive
maximum unchanged. The first non-bit midpoint input is therefore the current
history `sshn_e`, not a coefficient or the midpoint arithmetic.

R254-P1 and P6 are **CONFIRMED**. R254-P2 is **REFUTED**. R254-P3 through P5
are correctly not activated by their own predicates. All seven unique
midpoint plants fire: label coverage, south-refutation dependency,
source-order, NEMO replay, candidate replay, first-nonbit, and one-bit
detection. The final classifier is `HELD_R254_MIDPOINT_OPERAND_NAMED`.

## Gates, tests, and review

This round changes no model or recipe file, so no trajectory can move and the
OMT/rung/GYRE/DINO/tank landing gates are not activated. The round-253
certified states remain the baseline; this receipt makes no new fidelity claim
about them. Focused round-252/253/254 tests pass 40/40.

The once-only `tests/ocean/fidelity -n 12` battery collected 3,309 tests,
printed 3,237 passes and five failures, then stopped returning worker output at
98%; it was terminated and was not rerun. The five printed IDs were rerun
serially with xdist disabled. Four reproduce the registered pre-existing
conditions: SI3 scalar-math `MY_SRC` provenance, the GYRE round-129 certified-
year harness stamp, the round-35 process-global allow-dirty scoping ratchet,
and the report-emitter worktree-stamp ratchet. The fifth, the GYRE round-179
source-layout plant, refused only because the round's receipt and test were
then uncommitted; it is rerun from the clean committed tree below. The full,
focused, and first isolated log SHA-256 values are
`1565d5b706a08d0c9b729c787ed1570f79a5a9180b8608b9d83ae2aca325aea7`,
`1e0f8fc3db2c7e2bf5bc62b7880a7d4a5245f110b46888d468b65d68543f546a`,
and `aaba6078db14c6c7b9a4217a27ea6115ad263c3be2ea6dafdfd6c307de569022`.

The receipt citation gate, its planted stale-line control, the default-receipt
unmapped-citation check, the clean-tree round-179 disposition, the required
read-only review, and retained evidence hashes are reported here after
completion: **FINAL_GATE_PLACEHOLDER**.

The south-classification and midpoint-classification JSON SHA-256 values are
`da33c9320e592bd49e8a2944d42769298db6af0e75680891fc0713832ef06ee5`
and `d706f20e35a77847b802277300ef7af48dff166170be087f041c5cf984116ba5`.
The independent/given south measurements are
`1e192cd223af8610d2497f514a9eca334eb13052dac309037c1b0e6d87cee631` /
`375b1ebe1c7d63705ea8eff964ed73194888b27d75881263ce6e665224364234`;
the independent/given midpoint measurements are
`d6f3fcaf74fdc8ba855302d88ac399b0cc9a6d3204429f7a52a60ac7d3cc5c08` /
`b746941ef9a2541faa954c38284fef4e7fce4ffc90814b16dfb28ef774d7400b`.

## OPEN

Continue one boundary upstream, without a new acquisition. At kt=8 stage 1,
split the substep-2 current SSH in compiled order: compare the candidate
continuity output before association, its value after the seven-array
association, and its value after the history rotation against the admitted
`j002_ssha_e`. The first boundary that turns that field non-bit owns the next
walk. If continuity owns it, split that statement's recorded operands; if the
association or rotation owns it, transcribe only the cited statement and
score the complete private unit atomically. The complete fold/transport unit
and both held associations remain private. No stabilizer, floor, clip,
configuration change, or acquisition is authorized.

Round disposition: **HELD**.
