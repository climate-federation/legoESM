# PREREGISTRATION — split-explicit momentum chain round 2

Frozen before the first round-2 numerical execution.  This round peels row
1.1 of `dino_split_explicit_momentum_chain_round1_result.md`: the assembled
`zu_frc`/`zv_frc` slow-momentum forcing at the matched day-180 entry.

## Reused source-order ladder

This is the existing `PHASE0_zu_frc_assembly.md`,
`fslow_stage_decomp.py`, `zu_frc_momentum_row_reconstruction.py`, and
`zu_frc_budget_completion.py` ladder, refreshed against the active DINO
sources rather than re-derived from a new decomposition.

| Order | Constituent | Active NEMO statement and existing dump |
|---:|---|---|
| 1 | kinetic-energy gradient plus vertical advection | `cfgs/DINO/MY_SRC/stpmlf.F90:269-270,309-314`; split dumps at `dynadv.F90:89-103` (`keg_dump_*`, `zad_dump_*`) |
| 2 | total EEN vorticity/Coriolis | `stpmlf.F90:315-318`; active planetary and relative EEN dispatch, accumulation, and dump at `dynvor.F90:147-193` (`vor_dump_*`) |
| 3 | lateral friction | `stpmlf.F90:319-322`; active level-Laplacian snapshot, dispatch, dump, and trend bracket at `dynldf.F90:69-119` (`ldf_dump_*`) |
| 4 | hydrostatic pressure gradient | `stpmlf.F90:324-328`; active SCO increment bracket at `dynhpg.F90:348-413` (`hpg_dump_*`) |
| 5 | REST-weighted depth mean of rows 1-4 | `cfgs/DINO/MY_SRC/dynspg_ts.F90:316-339` |
| 6 | remove pre-loop 2-D Coriolis | `dynspg_ts.F90:358-370`; inferred exactly as `base + drag + wind - final` because `ln_apr_dyn=F` |
| 7 | baroclinic-residual implicit drag | `dynspg_ts.F90:372-400` (`drg_dump_*_frc_inc.bin`) |
| 8 | atmospheric pressure | `dynspg_ts.F90:404-421`; inactive by the run's own `ln_apr_dyn=F` echo |
| 9 | centred wind | `dynspg_ts.F90:423-459` (`wnd_dump_*_frc_inc.bin`) |
| 10 | assembled loop-entry forcing | `dynspg_ts.F90:507-525` (`spg_dump_z[uv]_frc.bin`) |

`dyn_zdf` vertical friction is deliberately a registered structural-zero row,
not silently omitted: `stpmlf.F90:332` calls `dyn_spg` before
`stpmlf.F90:396` calls `dyn_zdf`; the latter constructs the after velocity and
implicit vertical-friction solve at `cfgs/DINO/MY_SRC/dynzdf.F90:134-168,
193-337`.  It cannot contribute to the already-frozen `zu_frc`/`zv_frc`.

The NEMO base is independently reconstructed two ways from existing dumps:
the sum of `keg + zad + vor + ldf + hpg`, and the running-RHS
`stp_dump_06_dynhpg_kt00005761_d[uv].bin`.  They must agree at the fp64
rounding scale before any ownership result is admitted.

## Exact measurements and frozen bars

All fields use the day-180 `kt=5761` state, CPU, JAX fp64,
`LEGOESM_NEMO_E3T=both`, the native U/V mappings, full 3-D wet masks inside
the REST-thickness reduction, and the same 9,758 U / 9,868 V wet-column
populations as round 1.  Each term records correlation, campaign gate,
RMS-normalized error relative to its own NEMO term, and its signed error
field's relationship to the assembled residual.

For assembled residual `R = F_slow_legacy - zu_frc` and one term-error field
`T`, report:

- `gain = RMS(T) / RMS(R)`;
- `removal = 1 - RMS(R - T) / RMS(R)`;
- `corr = corr(T, R)`.

A term **CONFIRMS CARRY** only if `corr >= 0.99`, `0.90 <= gain <= 1.10`, and
`removal >= 0.90`.  It **REFUTES CARRY** only if `abs(corr) <= 0.20` and
`removal <= 0.10`; otherwise it is **UNRESOLVED**.  These are attribution
bars, not replacements for `fidelity_bar_gate.classify`, which supplies the
ordinary AT-BAR/DEBT verdict for every directly comparable field.

The prior basin lane's fourth-wind-premise closure supplies one already-built
same-input counterfactual: replace only the bridged NEMO U/V-point prior
stress stored as a T-point carry with the existing analytic DINO T-point
stress at the restart's own clock.  Let
`P = F_slow_faithful - F_slow_legacy`.  This bridge-wind source
**CONFIRMS** only if

- `RMS(P + R) / RMS(R) <= 0.10`;
- `corr(P, -R) >= 0.99`;
- `1 - RMS(F_slow_faithful - zu_frc) / RMS(R) >= 0.90`;
- every non-stress entry field is bit-identical between arms.

It **REFUTES** only if the three numerical axes are respectively `>=0.90`,
`<=0.20`, and `<=0.10`; otherwise it is **UNRESOLVED**.  U and V are scored
separately.  A donor-equalization plant must change this verdict path.

## Cancellation and continuation rules

The signed sum of all measured constituent-error fields is reported before
any owner is named.  Pairwise sums involving the leading term are rescored;
a confirming single term is downgraded to **CANCELLING PAIR / UNRESOLVED** if
another measured term reverses at least half its removal.  No scalar
`substep-1 × n` extrapolation is causal evidence.

If the faithful-carry assembled forcing reaches the campaign bar, row 1.1 is
closed and the chain advances to the existing row 1.2/1.3 seed and first-
substep dumps.  If it remains DEBT, the first source-ordered remaining DEBT
constituent is the next row-1.1 peel.  No later chain row may receive an
ownership verdict while row 1.1 remains open.

## Controls, provenance, and instrumentation

- The actual U/V alignment scan must select `(dj,di)=(0,0)`.
- The exact base reconstruction must have normalized error `<=1e-12` for U
  and V; adding `1e-6*RMS(stage06)` to the U reconstruction must fail.  The
  inferred-Coriolis ledger must close within its explicit fp64 roundoff
  envelope.
- The run's own `ocean.output` must echo `ln_apr_dyn=F`, `ln_bt_fw=F`,
  `ln_isfcav=F`, and `ln_drgice_imp=F` before the inferred-Coriolis identity
  is evaluated; missing or inconsistent echoes stop the probe.
- The signed constituent-error sum must reconstruct each assembled residual
  with normalized closure `<=1e-10`.  Adding `1e-6*RMS(residual)` to the U
  sum must fail that same bar.
- Synthetic identity and orthogonal error fields must traverse the same
  attribution classifier and return `CONFIRMS_CARRY` and `REFUTES_CARRY`,
  respectively.
- Identical arrays must classify AT BAR; a planted `1e-6` RMS-scale offset
  must traverse the same campaign classifier and classify DEBT.
- A planted east-donor equalization must alter the wind prediction/owner
  score; a zero perturbation is not a control.
- The receipt hashes every dump, restart, mesh, executable, namelist, active
  cited NEMO source, production module, inherited probe, scorer, and this
  preregistration.  It stamps the effective config and environment.

Round 2 requires no new NEMO instrumentation: all operands listed above
already exist in `RUN_SEQDUMP_D180_1R`.  Therefore no SLOT block is allocated.
If an exact pre-loop term cannot be recovered from the registered ledger, the
probe stops `UNMEASURED`; any later held writer must be preregistered and use
the campaign SLOT protocol before it is built.

Source-citation amendment before the accepted run: independent review found
that the original `dynvor.F90:147-155` and
`dynldf.F90:69-85,118-119` shorthand omitted the second EEN pass/dump and the
LDF dump write, respectively.  The table now gives the complete active
ranges.  No operand, statistic, population, threshold, or interpretation bar
changed.

Instrument-control amendment before the accepted run: independent review
required the oracle-arm, signed-closure, attribution-path, zero-reference,
donor-verdict, and strict-JSON checks above to fail closed.  These controls
were committed before the accepted receipt was produced.  No science bar or
ownership operand changed; the `1e-10` closure threshold is an arithmetic
ledger-validity bar, not a fidelity classification.
