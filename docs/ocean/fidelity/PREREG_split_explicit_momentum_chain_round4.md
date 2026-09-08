# PREREGISTRATION — split-explicit momentum chain round 4

Date: 2026-08-29.  Frozen before computing the previously unread `111`
corner of the lateral-friction × vorticity × pre-loop-Coriolis substitution
matrix.

## Question and precedent

Round 3 stopped row 1.1 at a composition: the faithful U slow-forcing
residual is `E=1.9539648558722173e-6`; substituting legoESM's production
BEFORE lateral-friction evaluation removes `0.6527883822`, below the frozen
ownership bars, while the NEMO-term error for lateral friction removes
`0.6612415045` alone and reverses when combined with either vorticity or the
pre-loop Coriolis removal.  No single-term edit is authorized.

This round follows the basin reconciliation's EEN × bridge-Omega four-corner
precedent, preregistered at `19acf16b2f0` in
`scripts/validate/ocean_fidelity/dino_1226/PREREG_tcarry_een_omega_interaction.md`
and scored at `820e3500bf3` by `tcarry_bridge_omega_score.py`.  That precedent
locked the already-read corners, computed the missing corner only after the
preregistration, separated interaction from combined ownership, and required
planted INVALID/CONFIRM/REFUTE paths.  Here the same design is extended from
two axes to the complete `2^3` matrix.

## Active sources, time levels, and oracle substitutions

The axes are the three material U-term errors already captured by the
committed round-2 path from the existing `RUN_SEQDUMP_D180_1R` files at
`kt=5761`:

| Axis | legoESM operand | NEMO oracle operand and active source |
|---|---|---|
| `L` lateral friction | production diagnostics lateral increment evaluated on the live NOW state | Kbb/BEFORE increment written by `cfgs/DINO/MY_SRC/dynldf.F90:73-115`, called at `stpmlf.F90:319-322` |
| `V` total EEN vorticity/Coriolis | production `vortcor_u/v` at NOW | total EEN increment at Kmm/NOW, `cfgs/DINO/MY_SRC/dynvor.F90:143-194`, called at `stpmlf.F90:315-318` |
| `C` pre-loop 2-D Coriolis removal | negative production `barotropic_coriolis_een_pre_step` result at Kmm/NOW | negative `dyn_cor_2D` result inferred by the round-2 exact forcing ledger, `cfgs/DINO/MY_SRC/dynspg_ts.F90:358-370` |

`C` remains an inferred oracle term, not a new dump.  Admission therefore
requires the round-2 base reconstruction and final forcing ledger to retain
their fp64 roundoff envelopes and planted failures.  `L` in this matrix is
the NEMO dumped Kbb term itself; it is not the separate legoESM-BEFORE
diagnostic arm from round 3.

For component `x` in `{u,v}`, let `r0_x = F_faithful_x - F_NEMO_x` on the
registered wet population and `e_i,x = term_lego_i,x - term_NEMO_i,x`.
For each subset `S` of `{L,V,C}`, the oracle-substituted residual is

`r_S,x = r0_x - sum(e_i,x for i in S)`.

The eight arms are `000`, `100`, `010`, `001`, `110`, `101`, `011`, and
`111`; a set bit means that term alone is replaced by its NEMO operand.  U
has 9,758 wet faces and V has 9,868.  Shapes, masks, residual identity across
term calls, exact arm order, dump hashes, and the full signed eight-term
closure are fail-closed.

## Locked corners and unread corner

The seven already-measured U corners are bound to the round-2/round-3
receipts; their `1 - RMS(r_S)/RMS(r0)` values are:

| Arm | Substitutions | Locked RMS removal |
|---|---|---:|
| `000` | none | `0.0` |
| `100` | `L` | `0.6612415045308856` |
| `010` | `V` | `-3.9651538510810838` |
| `001` | `C` | `-3.8528100525195788` |
| `110` | `L+V` | `-3.8067621779483583` |
| `101` | `L+C` | `-3.8351151874225913` |
| `011` | `V+C` | `0.08652382878366294` |
| `111` | `L+V+C` | **UNMEASURED** |

The probe must reproduce every locked removal within `1e-12` before it may
read or classify `111`.  V corners are descriptive new outputs under the
same matrix and may not independently promote row 1.1 unless their endpoint
also clears the registered rules below.

## Main effects, interactions, and ownership bars

For each component and arm define normalized squared residual energy
`Q_S = mean(r_S^2) / mean(r0^2)` and benefit `B_S = 1 - Q_S`, with `B_000=0`.
The source-anchored Möbius decomposition is:

- main effect `M_i = B_i`;
- pair interaction `I_ij = B_ij - B_i - B_j`;
- triple interaction
  `I_LVC = B_LVC - B_LV - B_LC - B_VC + B_L + B_V + B_C`;
- Shapley ownership allocation
  `phi_i = M_i + 0.5*sum_j(I_ij) + I_LVC/3`.

The identities `sum(phi_i)=B_111` and
`B_111=M_L+M_V+M_C+I_LV+I_LC+I_VC+I_LVC` must hold within `1e-12`.
Because slow-forcing assembly is a linear sum, the field contrasts
`r_ij-r_i-r_j+r_000` and
`r_111-r_110-r_101-r_011+r_100+r_010+r_001-r_000` must be zero to
`1e-12*RMS(r0)`.  A nonzero energy interaction therefore means cancellation
under the squared-error reducer, not a nonlinear model interaction.

Retain the campaign attribution classifier for each correction
`p_S=-sum(e_i for i in S)`:

- CONFIRM iff `corr(p_S,-r0) >= 0.99`, `0.90 <= RMS(p_S)/RMS(r0) <= 1.10`,
  and `1-RMS(r_S)/RMS(r0) >= 0.90`;
- REFUTE iff `abs(corr) <= 0.20` and removal `<=0.10`;
- otherwise UNRESOLVED.

The full trio is `CONFIRMED_COMPOSITE_OWNER` for a component only if arm
`111` CONFIRMS and `Q_111 <= 0.01`.  The `0.01` energy remainder is exactly
the squared form of the established 90% RMS-removal ownership bar.  Under a
confirmed trio:

- `phi_i >= +0.01` → `OWNED_POSITIVE_CONTRIBUTOR`;
- `phi_i <= -0.01` → `OWNED_CANCELLER`;
- `abs(phi_i) < 0.01` → `BOUNDED_BELOW_OWNERSHIP_REMAINDER`;
- an interaction is `MATERIAL` iff its absolute energy coefficient is at
  least `0.01`, otherwise `BOUNDED`.

Every noncandidate term from the exact eight-term ledger is also scored by
`RMS(e_i)/RMS(r0)`.  It is individually bounded only at `<=0.10`, the same
maximum correction amplitude left by the ownership bar.  Row 1.1 U is fully
disposed only if the trio confirms and every noncandidate U term is so
bounded.  V is evaluated identically; any unbounded V remainder names the
next source-ordered V term and prevents claiming the whole U/V row closed.

## Endpoint and continuation

The machine artifact reports all eight arms for U and V, the energy
decomposition, Shapley allocation, field-linearity contrasts, per-term
bounds, and the exact all-eight-term closure.  It must not recommend a
lateral-only, vorticity-only, or Coriolis-only physics edit.

If both U and V are fully disposed, a later committed preregistration may
use the existing NEMO `spg_dump_zu_frc.bin`/`spg_dump_zv_frc.bin` as a held
row-1.1 forcing control and continue rows 1.2--6 in execution order.  If V
or the U trio remains unresolved, STOP at the first named unbounded term and
design its next matrix before any downstream promotion.  No post-hoc bar may
be introduced to force continuation.

## Controls, provenance, and SLOT

The scorer must run from a clean commit on CPU/fp64 with
`LEGOESM_NEMO_E3T=both`, `DINO_1226_LANE=d180`, exact dump populations and
hashes, strict JSON, and unchanged round-2 controls.  Additional plants must:

1. perturb one locked corner so the locked-corner gate rejects it;
2. perturb one residual arm so a field-linearity contrast breaches its bar;
3. exercise confirmed, cancelling, and bounded Shapley labels;
4. make the full-group classifier reach CONFIRM and REFUTE;
5. make a `0.11*RMS(r0)` noncandidate term fail the `0.10` bound.

No NEMO execution, writer, or new dump is required.  Therefore no SLOT block
is allocated.  Any later need for a NEMO writer must first use the held-SLOT
protocol in the host conventions.

## Instrument amendment after invalid attempt 1

Attempt 1 reran the inherited round-2 operand loader but exited at the
total-field interception lookup before `_component_matrix` was called; no
matrix arm, `111` corner, energy coefficient, Shapley allocation, or row
verdict was computed or written.  The interceptor retained the metric dict
by reference.  The inherited probe subsequently appended `campaign_gate` to
that same object, so lookup against the pre-gate metric found zero matches.
Before attempt 2, store a shallow copy of the metric at interception and make
the lookup ignore `campaign_gate` on both sides.  Add a uniqueness check and
retain the existing exact residual-array identity check.  No operand, arm,
bar, locked corner, ownership rule, control, or continuation rule changes.

## Receipt amendment after withheld attempt 2

Attempt 2 completed the matrix, but its artifact is withheld after an
internal receipt audit.  Two issues are repaired before the accepted rerun:

1. The preregistration required the eight-term closure to be fail-closed but
   used the word "exact" without an executable fp64 association envelope.
   Bind it now to the pointwise operation-count bound
   `32*eps*(abs(r0)+sum(abs(e_i)))`.  This is derived from the eight additions,
   subtractions, and separately associated total/term paths; it is not fitted
   to the observed aggregate closure.  Every wet point must pass.  Adding
   `1e-12*RMS(r0)` to one wet closure element must breach the same bound.
2. Shapley values remain computable for a nonconfirming group, but the frozen
   ownership labels apply only "under a confirmed trio."  For an unconfirmed
   component, stamp every Shapley label
   `DESCRIPTIVE_UNADMITTED_GROUP_UNRESOLVED` and retain
   `UNRESOLVED_IN_COMPOSITE` in the term table.  This changes no number.

Attempt 2's science values have been seen, so this amendment makes no new
prediction and changes no operand, matrix arm, locked corner, ownership or
interaction threshold, group classifier, term bound, or continuation rule.
Only an artifact satisfying both executable receipt requirements may be
accepted and cited.

## Retraction after invalid attempt 3

Attempt 3 falsified the operation-count premise before writing an artifact:
the faithful assembled forcing and the independently reconstructed eight
term fields do not share one fp64 association tree, so their pointwise
closure cannot honestly be certified as addition/subtraction roundoff.  The
operation-count envelope above is withdrawn and the scorer must stop using
or printing it.

Before the next attempt, promote
`assembly_remainder = r0 - sum(e_i, i in the eight-term ledger)` to an
explicit ninth residual component.  Score its
`RMS(assembly_remainder)/RMS(r0)` through the already-frozen noncandidate
bound `<=0.10`; label it `BOUNDED_MAGNITUDE` or `UNBOUNDED_REMAINDER` exactly
like a named term, include it in `fully_disposed`, and retain its signed
field in the all-term endpoint.  The inherited round-2 legacy term-closure
control and planted failure remain mandatory and unchanged.  The existing
`0.11*RMS(r0)` bound plant exercises this new component too.

This is a retraction and explicit accounting entry, not a relaxed closure
bar: an unclosed remainder now consumes the same ownership budget as any
other term and can block the row.  No matrix arm, measured value, ownership
threshold, interaction threshold, group classifier, named-term bound, or
continuation rule changes.
