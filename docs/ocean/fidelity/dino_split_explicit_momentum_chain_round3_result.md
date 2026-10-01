# DINO split-explicit / momentum-commit chain: round-3 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Clean measurement commit:
`1c5c8414fceacb156983ebf6e3fc0881668d4ece`.

## Verdict

The centred-wind owner is **FIXED-BY-#1695**.  This branch cherry-picks the
existing T-point restart-stress reconstruction and its default-on follow-up;
it does not add a second implementation.  The admitted day-180 receipt
reproduces the round-2 correction exactly: prediction correlation
`0.9999746551`, normalized prediction error `0.0071304473`, and residual
reduction `0.9928695527` (`99.28695527%`).  The historical U-as-T bridge
remains an explicit reproduction-only CLI arm.

The faithful U `zu_frc` field nevertheless remains DEBT at
`E=1.9539648559e-6`.  Continuing in registered source order clears vertical
advection and total EEN vorticity as standalone owners, then stops at lateral
friction.  Replacing only legoESM's live NOW lateral-friction operand with its
production-diagnostics BEFORE evaluation removes `65.27883822%` of the
faithful residual, but its prediction correlation is only `0.9393166` and
normalized prediction error is `0.3472116`.  It is therefore
**UNRESOLVED_LDF_BEFORE_SOURCE**, not a fix authorization.

Pair analysis is load-bearing.  Adding either the vorticity error or the
pre-loop Coriolis-removal error to lateral friction reverses its improvement;
the vorticity and Coriolis errors themselves largely cancel.  A lateral-
friction-only production edit would therefore violate the campaign's
cancelling-pair rule.  Row 1.1 remains **OPEN at lateral friction** and rows
1.2--6 remain ordered-blocked.

## #1695 import and merge-order dependency

The fetched source branch was
`fidelity/dino-basin-rectification-codex` at `f862e155491`.  Imported commits:

| Dependency order | Upstream commit | Local cherry-pick | Disposition |
|---:|---|---|---|
| 1 | `73ad090d411` | `8327ffb10bc` | T-point reconstruction, content/time/stagger receipts, selector, gate, tests |
| 2 | `f862e155491` | `7061c59af51` | faithful carry becomes default; legacy U-as-T becomes explicit opt-in |

Independent review found two import-integration gaps after the first accepted
execution.  Local follow-up `b698eac4d8d` imports the exact latest committed
`PREREG_endwall_wind_placement.md` required by the reused gate (SHA-256
`6dd29cd67621f13c9d0c6229d38ea4fdeaef8f45f9a6800f0214cd1f2a65dfb5`)
and makes the faithful default conditional on a bridged start.  Thus the
ordinary bridged CLI still selects corrected T-point carry, while
`--legacy-euler-start` selects neither a before-level bridge nor prior-stress
carry; an explicitly incompatible request still fails closed.  This is an
integration follow-up, not a second carry implementation.

Both patch-equivalent local commits and the integration follow-up are
ancestors of the clean measurement commit.  The default receipt is
`build=True`, `run=True`, `CLI=True`; the legacy U-as-T flag selects `False`;
the legacy-Euler receipt is `bridge_before=False`, `stress=False`.  The
focused fp64 CPU tests pass (`5 passed`).  Contextual resolution retained this
branch's newer harness arguments.  For the default-on follow-up, its
harness/test changes were imported; basin-only result files were not
transplanted, while the gate's named preregistration dependency was imported
by the reviewed integration follow-up.

This creates an explicit merge-order dependency.  If PR #1695 lands first,
the two patch-equivalent local cherry-picks must be dropped before this branch
lands.  If this branch lands first, #1695 must rebase or omit those two
changes.  In neither order should a second carry implementation be merged.

## Source-ordered faithful-tail peel

All NEMO operands remain the existing `RUN_SEQDUMP_D180_1R` dumps at
`kt=5761`.  Attribution is against the faithful U assembled residual, using
the frozen round-2 correlation/gain/removal classifier.

| Order | Constituent and active NEMO source | Term `E` / gate | `corrR`, gain, removal | Ordered disposition |
|---:|---|---|---|---|
| 1a | kinetic-energy gradient: `dynadv.F90:89-95`; `stpmlf.F90:309-314` | `2.22344e-16` / NEAR-CLASS | `-0.069879`, `3.86e-13`, `-2.73e-14` | CLEARED_NEAR_CLASS |
| 1b | vertical advection: `dynadv.F90:97-103`; `dynzad.F90:81-119`; `stpmlf.F90:309-314` | `6.42855e-5` / DEBT | `+0.021117`, `0.010037`, `0.000169` | **CLEARED_REFUTES_CARRY** |
| 2 | total EEN vorticity/Coriolis: `dynvor.F90:147-193`; `stpmlf.F90:315-318` | `2.97729e-5` / DEBT | `-0.029166`, `4.835751`, `-3.965154` | **CLEARED_REFUTES_CARRY** |
| 3 | lateral friction: `dynldf.F90:69-119`; `stpmlf.F90:319-322` | `7.62755e-4` / DEBT | `+0.942639`, `0.913337`, `0.661242` | **OPEN_UNRESOLVED** |

The vertical-advection result reuses the committed ZAD operand walk and
production fixes instead of decomposing the term again.  The
flux-vs-advective correction is `84c169eb770`; the bottom/straddling-face mask
correction is `5bdcf219edc`.  Their surviving depth-mean error refutes carrying
the faithful assembled residual, so source order advances.  Vorticity also
refutes despite its large gain, because its direction is wrong and removing
it makes the residual nearly five times worse.

## Lateral-friction time level and cancellation

NEMO evaluates `dyn_ldf` at Kbb/BEFORE.  legoESM's live first barotropic pass
evaluates lateral friction at NOW; its later BEFORE pass keeps only the 3-D
dissipative increment and discards that pass's barotropic result
(`ocean_model_latlon_cgrid.py:8984-9064`).  The registered offline
counterfactual uses the existing `ldf_state=BEFORE` production diagnostics
hook and substitutes only this depth-mean forcing operand.

The BEFORE term itself improves from `E=7.62755e-4` to `5.0057633e-5` against
NEMO, with field correlation `0.9999999989`, but remains DEBT.  Its forcing
substitution gives:

- prediction correlation `0.9393166157`;
- prediction normalized error `0.3472116178`;
- faithful-residual reduction `0.6527883822`;
- verdict `UNRESOLVED_LDF_BEFORE_SOURCE`.

The registered pair scores explain why this is not promoted:

| Error combination | `corrR` | Gain | Removal | Verdict |
|---|---:|---:|---:|---|
| lateral + vertical advection | `0.943002` | `0.913209` | `0.662236` | UNRESOLVED |
| lateral + vorticity | `0.148398` | `4.853466` | `-3.806762` | REFUTES_CARRY |
| lateral + pre-loop Coriolis removal | `0.229440` | `4.964252` | `-3.835115` | UNRESOLVED |
| vorticity + pre-loop Coriolis removal | `0.412686` | `0.334979` | `0.086524` | UNRESOLVED |

The next design must hold the lateral time-level substitution together with
the vorticity/pre-loop-Coriolis cancellation group and then peel that group's
operands in NEMO execution order.  It must not ship a lateral-only physics
change from the `65.28%` partial improvement.  V remains descriptively DEBT at
`E=2.9393366602e-5`; it receives no independent ownership while U row 1.1 is
open.

## Ordered chain state

| Row | Active NEMO execution | Status after round 3 |
|---:|---|---|
| 1.1 | `dyn_spg_ts` assembled `zu_frc/zv_frc` | **OPEN_LATERAL_FRICTION_UNRESOLVED** |
| 1.2 | split-explicit loop seed | ORDERED_BLOCKED |
| 1.3 | first split-explicit substep | ORDERED_BLOCKED |
| 2 | second `div_hor` | ORDERED_BLOCKED |
| 3 | second `dom_qco_r3c` | ORDERED_BLOCKED |
| 4 | `dyn_zdf` momentum commit | ORDERED_BLOCKED |
| 5 | second `wzv` | ORDERED_BLOCKED |
| 6 | `mlf_baro_corr` | ORDERED_BLOCKED |

## Controls and provenance

- The clean run regenerated the full round-2 receipt, including exact signed
  term closure (`1.0764e-12` U, `9.9735e-12` V), oracle flags, base/inferred
  ledgers, strict JSON, and all planted controls.
- Raw-term interception reproduced every serialized faithful metric exactly;
  the capture sequence and 9,758-face U population are fail-closed.
- Perfect and zero lateral-correction plants traverse the new classifier as
  CONFIRM and REFUTE, respectively.
- The receipt hashes the round-2/round-3 probes and preregistrations, imported
  harness, existing ZAD tests, all 18 dumps, and seven run inputs.
- Independent review held the first accepted artifact for the two import-
  integration gaps above.  The preregistration was amended before extending
  the selector receipt; that superseded artifact is not packaged.
- The accepted rerun used CPU, JAX fp64, `LEGOESM_NEMO_E3T=both`, and exited
  zero.  A second clean invocation produced a byte-identical artifact.

No new NEMO writer, integration, GPU, MPI process, or SLOT block was used.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round3_artifact.json`,
SHA-256 `f102167ea9784644e5c86a2f761caad2eccba85f03b26ef8580f1e5c689334f9`.
Accepted run log SHA-256:
`13ed76f5dcf09ef34176a96dc2e775869173e86108fb10a202aa1d3b482cd6e5`.
