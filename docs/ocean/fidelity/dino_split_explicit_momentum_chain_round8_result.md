# DINO split-explicit / momentum-commit chain: rounds 4--8 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

Row 1.1 is now **fully disposed component-wise, but remains a production
composition debt**.  The requested joint substitutions prove that no single
lateral-friction owner exists.  U closes with lateral friction + total EEN
vorticity + pre-loop Coriolis removal; V closes only when hydrostatic pressure
gradient is added.  Every other forcing term and the explicitly reconstructed
assembly remainder is below its preregistered 0.10 ownership bound.

The ordered chain does not cross row 1.2.  Its literal SSH seed is bit-exact,
while U/V have only operation-order residue (`E=2.2962596892e-16` /
`2.2888243354e-16`) but miss the POINTWISE class bar at maxima
`5.5743163337e-15` / `5.3302843870e-15`.  Exact oracle holding makes all three
seed fields bit-exact for targeting, but cannot promote the literal row.

With row 1.1 forcing and row 1.2 seed held exactly, row 1.3 still diverges at
the first substep: SSH/U/V `E=6.2048763005e-7` / `6.2120556997e-7` /
`1.1172636476e-6`.  The existing `ssh_frc` candidate is identically zero and
is refuted.  All three runtime-captured held seed fields and the independently
dumped half-step U/V operands are bit-exact, whereas the algebraically recovered
continuity flux divergence is DEBT (`E=0.4530771405`, correlation
`0.9141355600`, mean-absolute ratio `1.0523503033`).  Inverting and reapplying
the same continuity equation closes to `5.8233515124e-22`, but that replay is
explicitly an algebraic identity, not a control or independent ownership fact.
The independent exact runtime inputs followed by the first divergent executed
output localize the held-chain difference to DINO's active `key_qco`
variable-volume face-depth/transport/divergence composition at
`cfgs/DINO/MY_SRC/dynspg_ts.F90:651-725`.

## Row 1.1 complete term disposition

The source ladder and time levels are retained in execution order.

| Term | Active NEMO source | U | V |
|---|---|---|---|
| kinetic-energy gradient | `dynadv.F90:89-95`; `stpmlf.F90:309-314` | bounded (`3.86e-13`) | bounded (`2.34e-3`) |
| vertical advection | `dynadv.F90:97-103`; `dynzad.F90:81-119`; `stpmlf.F90:309-314` | bounded (`1.00e-2`) | bounded (`1.52e-3`) |
| total EEN vorticity, Kmm/NOW | `dynvor.F90:143-194`; `stpmlf.F90:315-318` | owned canceller | owned positive contributor |
| lateral friction, Kbb/BEFORE | `dynldf.F90:73-115`; `stpmlf.F90:319-322` | owned positive contributor | owned positive contributor |
| hydrostatic pressure gradient, Kmm/NOW | `dynhpg.F90:348-413`; `stpmlf.F90:324-328` | bounded (`7.94e-7`) | owned positive contributor |
| pre-loop 2-D Coriolis removal, Kmm/NOW | `dynspg_ts.F90:358-370` | owned positive contributor | owned canceller |
| baroclinic-residual drag | `dynspg_ts.F90:372-400` | bounded (`4.14e-4`) | bounded (`1.88e-5`) |
| centred wind | `dynspg_ts.F90:423-459` | **FIXED-BY-#1695**, then bounded (`1.90e-2`) | exact-zero/bounded |
| explicit assembly remainder | reconstruction of `dynspg_ts.F90:316-459` | bounded (`1.58e-10`) | bounded (`9.97e-12`) |

The values in parentheses are normalized RMS gains relative to each component's
total residual, not fidelity scores.  The bound is 0.10.

### U three-axis matrix

For axes `L=lateral friction`, `V=vorticity`, `C=pre-loop Coriolis`, the full
`L+V+C` arm removes `0.978442231409` of residual RMS, has correlation
`0.999779920403` with the total residual, and leaves normalized squared energy
`4.64737386616e-4`.  Shapley allocation is L `+0.859608329870`, V
`-0.137439313324`, C `+0.277366246067`.  Material energy interactions are
L×V `+0.662547447803`, L×C `-0.713816151557`, and V×C
`+46.368079455360`; the triple is zero.  Field interaction contrasts are at
most `6.94e-18`, so these are cancellation/RMS interactions rather than a
nonlinear physical operator.

### V four-axis matrix

The full `L+V+C+P` arm, with `P=pressure gradient`, removes
`0.997227273491` of residual RMS, correlates `0.999998160346` with the total,
and leaves normalized squared energy `7.68801229187e-6`.  Shapley allocation is
L `+0.133701275958`, V `+0.064497260189`, C `-0.043244995596`, and P
`+0.845038771437`.  Material interactions are V×C `+0.082515069391`, V×P
`-0.124451136797`, and C×P `+0.083128126635`; all higher interactions are
bounded.  Maximum field contrast is `1.3814402479e-17`.

This follows the basin reconciliation's preregistered EEN×Omega four-corner
precedent (`19acf16b2f0`, scorer `820e3500bf3`) and extends it to complete
`2^3` and `2^4` matrices.  Ownership does not authorize a composite production
change.

## #1695 carry and merge order

The centred-wind term is **FIXED-BY-#1695**.  This branch carries PR #1695's
restart-stress face-placement repair as cherry-picks `8327ffb10bc` and
`7061c59af51`, plus local integration commit `b698eac4d8d`.  Its 99.28695527%
removal receipt confirms the same coastal-donor-doubling defect; it is not an
independent implementation.  Merge this lane after #1695, or retain the
equivalent carry commits while resolving overlap.  Do not merge a pre-#1695
bridge over this result.

## Ordered registry disposition

| Row | Executed NEMO site | Status after rounds 4--8 |
|---:|---|---|
| 1.1 | forcing assembly `dynspg_ts.F90:316-459` | **OWNERSHIP COMPLETE; PRODUCTION DEBT IS COMPOSITE** |
| 1.2 | centred BEFORE seed `dynspg_ts.F90:561-580` | **DIVERGED, NEAR-CLASS U/V operation-order residue** |
| 1.3 | first recurrence `dynspg_ts.F90:614-850` | **TARGETING DIVERGED at key_qco continuity composition `:651-725`** |
| 1.4 | final `puu_b/pvv_b/pssh/un_adv/vn_adv`; rewrite `:1170-1174` | ORDERED-BLOCKED; targeting DEBT under held forcing/seed |
| 2 | second `div_hor`, `stpmlf.F90:349-376` | ORDERED-BLOCKED; existing dumps inventoried |
| 3 | second `dom_qco_r3c`, `stpmlf.F90:378-394` | ORDERED-BLOCKED; existing dumps inventoried |
| 4 | `dyn_zdf`, `stpmlf.F90:396-409` | ORDERED-BLOCKED; independent completed ZDF receipt retained, not promoted here |
| 5 | second `wzv`, `stpmlf.F90:411-412` | ORDERED-BLOCKED; call-2 dump inventoried |
| 6 | `mlf_baro_corr`, `stpmlf.F90:578,709-765` | ORDERED-BLOCKED; before/after dumps inventoried |

The held row-1.4 targeting scores are `E=1.1889091711e-5` U,
`1.3675897110e-5` V, `4.8049595655e-6` SSH, `8.3103365733e-6` U transport,
and `1.1501070989e-5` V transport.  They are not chain verdicts across row 1.2
or row 1.3.  Rows 2--6 were therefore not post-hoc scored or promoted.

## Measurement and instrumentation disposition

All accepted calculations used CPU/JAX fp64, `DINO_1226_LANE=d180`,
`LEGOESM_NEMO_E3T=both`, a clean detached commit, and existing
`RUN_SEQDUMP_D180_1R` dumps.  No GPU, `mpirun`, NEMO build/run, or writer was
used.  Consequently no SLOT block was acquired.  The two round-6 pre-score
compatibility exits and the round-7 zero-RMS scorer exit are recorded as
`INVALID`; none supplied evidence.  The first nominal receipts are also
withdrawn because adversarial review proved their editable install imported
production modules from another worktree.  The replacement receipts fail
closed unless all production imports resolve beneath the clean measured
checkout, stamp those paths, and assert every monkeypatch restoration.  Exact
forcing reconstruction, runtime exact held seed, exact-zero SSH forcing,
planted gate traversal, zero-shift alignment, interception/restoration
receipts, populations, runtime settings, sources, dumps, restart, mesh,
namelists, and executable are hashed.

Machine receipts:

- `dino_split_explicit_momentum_chain_round6_artifact.json`
- `dino_split_explicit_momentum_chain_round7_artifact.json`
- `dino_split_explicit_momentum_chain_round8_artifact.json`

The next admissible discriminator needs direct eta-dependent
`zhup2_e/zhvp2_e` and `zhU/zhV` operands around `dynspg_ts.F90:651-704`.
Because those arrays are not in the existing stack, that work is
designed-not-run and must use a preregistered SLOT-protocol held writer block
before any measurement.
