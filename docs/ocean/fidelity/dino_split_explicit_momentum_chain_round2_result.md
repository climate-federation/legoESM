# DINO split-explicit / momentum-commit chain: round-2 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Clean measurement commit:
`1c736e5f5123d9894d4c6603847b8fac9ae7a337`.

## Verdict

The day-180 U slow-forcing debt is localized to the centred-wind term.  On
the original 9,758-face residual, its error has correlation
`0.9999746989`, RMS gain `1.0002839648`, and removal `0.9928767443`; it
therefore **CONFIRMS_CARRY** under the frozen three-axis bar.  Every other U
term refutes carrying the original residual.  The signed term-error ledger
reconstructs the assembled residual to `1.0738730e-12` of its RMS (U) and
`1.0718442e-11` (V).

The independently preregistered basin correction changes only the initial
previous-stress carry from the historical NEMO-U-as-lego-T representation to
the analytic DINO T-point stress at the restart clock.  It reduces U
`zu_frc` error from `2.7403117433e-4` to `1.9539648559e-6`, predicts the
negative legacy residual with correlation `0.9999746551`, normalized
prediction error `0.0071304473`, and removes `99.28695527%` of it.  This
**CONFIRMS_BRIDGE_WIND_SOURCE**.  The equalized-east-donor plant changes the
prediction error from `0.0071304473` to `0.9792034895`, so the ownership path
can fail.

This reuses, rather than repeats, the basin lane's donor alignment and
faithful harness work: coastal-donor measurement `6572f6e7469`, T-point gate
`73ad090d411`, and faithful harness default `f862e155491` on
`fidelity/dino-basin-rectification-codex`.  No second bridge implementation
was introduced here.

Row 1.1 is nevertheless still **OPEN** under the exact campaign gate.  The
faithful U field remains DEBT at `1.9539648559e-6`; V is unchanged and DEBT at
`2.9393366602e-5`.  Therefore rows 1.2/1.3 and chain rows 2--6 remain
ordered-blocked.  The result closes ownership of the original `2.7e-4` U
debt, not the entire assembled-forcing row.

## Reused source-ordered term ladder

All NEMO operands came from existing `RUN_SEQDUMP_D180_1R` dumps at
`kt=5761`.  `E` is each legoESM term's RMS error normalized by that NEMO
term's own RMS.  `corrR`, gain, and removal compare the signed term error with
the original assembled residual.

| Order | Constituent and active NEMO source | U `E` / gate | U `corrR`, gain, removal | Original-residual verdict | V `E` / gate |
|---:|---|---|---|---|---|
| 1 | pressure + KE gradient: `dynadv.F90:89-95`, `dynhpg.F90:348-413`; called at `stpmlf.F90:309-314,324-328` | `1.5712e-12` / NEAR-CLASS | `-0.004242`, `5.66e-9`, `-2.38e-11` | REFUTES_CARRY | `2.69387e-5` / DEBT |
| 2 | vertical advection: `dynadv.F90:97-103`; `stpmlf.F90:309-314` | `6.42855e-5` / DEBT | `+0.009092`, `7.16e-5`, `6.15e-7` | REFUTES_CARRY | `1.01641e-4` / DEBT |
| 3 | total EEN vorticity/Coriolis: `dynvor.F90:147-155`; `stpmlf.F90:315-318` | `2.97729e-5` / DEBT | `-0.112407`, `0.034481`, `-0.004485` | REFUTES_CARRY | `2.04328e-5` / DEBT |
| 4 | lateral friction: `dynldf.F90:69-85,118-119`; `stpmlf.F90:319-322` | `7.62755e-4` / DEBT | `+0.006114`, `0.006513`, `0.000020` | REFUTES_CARRY | `3.26464e-3` / DEBT |
| 5 | REST depth mean: `dynspg_ts.F90:316-339` | base reconstruction control | — | exact-ledger prerequisite | base reconstruction control |
| 6 | pre-loop 2-D Coriolis removal: `dynspg_ts.F90:358-370` | `3.01037e-5` / DEBT | `+0.104576`, `0.034268`, `0.003018` | REFUTES_CARRY | `1.98758e-5` / DEBT |
| 7 | baroclinic-residual drag: `dynspg_ts.F90:372-400` | `1.41810e-5` / DEBT | `+0.061012`, `2.95e-6`, `1.81e-7` | REFUTES_CARRY | `2.82178e-6` / DEBT |
| 8 | atmospheric pressure: `dynspg_ts.F90:404-421` | structural zero (`ln_apr_dyn=F`) | — | not an operand | structural zero |
| 9 | centred wind: `dynspg_ts.F90:423-459` | `3.86709e-2` / DEBT | `+0.999975`, `1.000284`, `0.992877` | **CONFIRMS_CARRY** | exact-zero oracle and model |
| 10 | assembled forcing: `dynspg_ts.F90:507-525` | legacy `2.74031e-4`; faithful `1.95396e-6`, both DEBT | — | original U owner localized | legacy/faithful `2.93934e-5`, DEBT |

Vertical friction is explicitly accounted for as
`STRUCTURAL_ZERO_NOT_AN_OPERAND`: `stpmlf.F90:332` calls `dyn_spg` before the
`stpmlf.F90:396` `dyn_zdf` call; the active solve is
`dynzdf.F90:134-337`.  Both legoESM component maxima are exact zero.  NEMO's
V wind reference and legoESM V wind are also exact zero, so their normalized
error is correctly reported as `UNMEASURED_ZERO_REFERENCE`, not divided by a
zero RMS.

## Post-correction tail and ordered continuation

Rescoring every term against the faithful U residual leaves no confirming
single-term owner.  Pressure/KE, vertical advection, vorticity, Coriolis
removal, drag, and faithful wind all REFUTE_CARRY.  Lateral friction is the
largest remaining single term but is **UNRESOLVED**: correlation `0.9426393`,
gain `0.9133374`, removal `0.6612415`.  V is also composite: pressure/KE is
the strongest single candidate (`corrR=0.9293190`, gain `0.9085035`, removal
`0.6322386`) but remains UNRESOLVED; lateral friction, vorticity, and the 2-D
Coriolis removal are also unresolved.

The next row-1.1 design is therefore a preregistered cancellation peel of the
faithful tail, beginning in source order and treating the EEN-vorticity /
pre-loop-Coriolis pair together before any physics edit.  Their separate U
errors are each about `4.8x` the faithful residual and neither carries it,
which is the campaign's cancelling-pair signature.  Lateral friction is then
rescored with that pair held.  For V, pressure/KE is the first unresolved
source-ordered candidate and must be tested against the same pair.  Existing
dumps remain sufficient for that design; no later execution-chain row is
opened and no coefficient or production physics is changed here.

## Controls, provenance, and attempt audit

- The NEMO `keg + zad + vor + ldf + hpg` reconstruction matches the existing
  stage-06 accumulated RHS within the registered `1e-12` normalized bar.
- The inferred 2-D Coriolis ledger reconstructs final forcing inside an
  explicit fp64 operation-count envelope: maximum differences are
  `6.7763e-21` U and `3.3881e-21` V; a material planted offset breaches it.
- Actual alignment selects zero shift; identity and planted campaign-gate
  controls pass.  Every non-stress entry leaf is bit-identical between the
  legacy and faithful arms.
- The accepted receipt hashes 18 existing dumps, seven run inputs, the active
  NEMO sources, all production/scorer modules, the probe, and preregistration.
- Invalid attempts were rejected before admission for terminal-level mask
  shape, an over-strict bit-exact inverse-ledger assertion, and zero-reference
  normalization.  A later clean run exposed and retracted use of `phys_u` as
  the standalone wind term; the accepted probe reuses the prior campaign's
  production `surface_stress_faces` spy and exact F-slow thickness reduction.
  A clean interim receipt was superseded by the final post-faithful rescore.

No new NEMO writer was needed, so no SLOT block was allocated.  This lane ran
no GPU, MPI process, or NEMO integration.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round2_artifact.json`,
SHA-256 `cf76649114f9dc646ee125ca0c1b146b6ea1dad81f2b6615f5d330e1e75a8476`.
Accepted run log SHA-256:
`56ba98fbe884b47e0c25815a5f40958b491d57a0cb9c98eca7d1cad2287fd34c`.
