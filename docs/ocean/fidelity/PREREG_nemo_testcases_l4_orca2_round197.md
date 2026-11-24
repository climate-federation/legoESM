# ORCA2 round 197 preregistration — substep-2 vector V update

Date: 2026-10-09. Frozen base:
`60534edc6f0f95300c39c617fc81644024279238`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round197/`.

Every number is **independent hierarchy rung 0**. The card starts from its
corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this round. The shipped ORCA2
card, sea ice, its six selectors and its `unmeasured_features` tuple remain
unchanged. This is an offline, measurement-only split; no executable observer,
model/configuration change or carried-state change is permitted.

## Frozen source order and protocol

The resolved rung-0 run takes NEMO's vector-form external-velocity update
(`ORCA2_R96SPG/ocean.output:765`). At external substep 2 NEMO evaluates
`va_e = (vn_e + rDt_e * (zv_spg + zv_trd + zv_frc)) * ssvmask` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727`, then associates
that V field with sign `-1` in the seven-array `lbc_lnk` call at
`dynspg_ts.f90:770-779`. The admitted round-96 record supplies the current
velocity (`j001_va_new`), `j002_zv_spg`, `j002_trd_v`, the frozen slow forcing
(`i000_zv_frc`), `rDt_e` (`i000_entry_sc`), and the post-association target
(`j002_va_new`). The candidate V mask and the raw NEMO `vmask` are compared
explicitly before either is used.

Extend the round-196 offline gate pattern. Reuse `nemo_source_round` and the
already-certified private boundary-association helper; do not transcribe a
second boundary operator. First prove that a record-only literal update plus
association reproduces `j002_va_new` bit-for-bit, and that the same replay on
the passive candidate operands reproduces the passive substep-2 `v_exit`
bit-for-bit. Then compare, in source order: `vn_e`, `rDt_e`, `zv_spg`,
`zv_trd`, `zv_frc`, the inner partial sums, the increment, raw `va_e`,
`ssvmask`, and post-association `va_e`. Substitute each recorded operand into
the candidate replay one variable at a time; print cumulative rows only after
the single-variable rows.

The gate parses the self-describing record, requires both ranks exactly once,
requires every named stream and its expected shape, reproduces round 196's
16,506-cell current-V debt, and proves the passive traced/untraced state is
array-identical. Plants independently reorder the input registry, perturb one
exact timestep bit, break the record-only association, and remove one required
stream; every plant must refuse.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R197-P1 | The admitted round-96 record is sufficient; no raw pre-association acquisition is needed. | Record-only update plus association closes `j002_va_new` bit-for-bit, and candidate replay closes its passive `v_exit`. | Either replay residual or a missing/malformed stream: **REFUTED**; write a fail-closed acquisition and stop. |
| R197-P2 | Substep-2 entry `vn_e` is bit-exact. | Candidate `v_entry[1]` equals `j001_va_new` on all 26,640 record cells. | Non-bit entry: **REFUTED**; stop at `vn_e`. |
| R197-P3 | The first non-bit update operand is `zv_spg`. | `vn_e` and `rDt_e` exact, then `zv_spg` non-bit; replacing only `zv_spg` moves raw and post-associated V toward or to NEMO. | Exact `zv_spg`: **REFUTED**; continue in source order through `zv_trd`, then `zv_frc`, then `ssvmask`. |
| R197-P4 | One recorded operand substitution closes post-associated substep-2 V. | A source-ordered single-variable replay reaches 0 / 26,640 unequal. | Residual after every single substitution: **REFUTED**; report the cancelling unit and stop before midpoint depth, reciprocal or trajectory. |
| R197-P5 | The association itself adds no new debt once its input is exact. | Record raw update associated by the certified helper is bit-identical to `j002_va_new`. | Non-bit result: **REFUTED**; boundary association is the first unresolved boundary and a raw-target acquisition is needed. |
| R197-P6 | Controls bind. | Registry-order, timestep-bit, association and missing-stream plants each refuse. | Any plant stays green: invalid instrument; report no operand claim. |

If P1, P5 or P6 fails, no source attribution is reported. If P2 fails, it is
the first boundary. Otherwise the first non-bit operand or cancelling unit is
round 198's OPEN item. No depth, reciprocal or trajectory census follows this
measurement. The final package tree must equal the frozen base.

ASKED choices: round 196's OPEN source-ordered substep-2 vector V update and
post-update boundary association.  
UNASKED choices: empty.
