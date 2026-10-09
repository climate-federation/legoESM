# ORCA2 round 196 preregistration — substep-3 AB3 midpoint-V inputs

Date: 2026-10-09. Frozen base:
`776389945b6aa09b148f0712bacc16e4160f401e`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round196/`.

Every number is **independent hierarchy rung 0**. The card starts from its
corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this round. The shipped ORCA2
card, sea ice, its six selectors and its `unmeasured_features` tuple remain
unchanged. This is an offline, measurement-only split; no executable observer
or model/configuration change is permitted.

## Frozen source order and protocol

NEMO selects the full-AB3 coefficients at substep 3 and then evaluates the
left-associated midpoint V velocity
`va_e = za1*vn_e + za2*vb_e + za3*vbb_e` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:486-511`. It rotates the
three V histories only after the completed substep update at
`dynspg_ts.f90:817-819`. Therefore the admitted round-96 record supplies the
substep-3 inputs without a new acquisition: `j002_va_new` is current `vn_e`,
`j001_va_new` is prior `vb_e`, and `i000_vn_e` is prior-prior `vbb_e`; the
recorded `j003_ext_coef` supplies `za1/za2/za3`. The history rotation itself is
at `dynspg_ts.f90:842-844`. The mapping is accepted only if replaying those
four recorded fields reproduces recorded `j003_va_ext` bit-for-bit.

Extend the existing round-195 offline gate rather than writing a second
solver or product implementation. Reuse the shared literal midpoint helper.
Run the already-passive private arm on CPU with JIT, fp64 and libm, but read
only substep 3. Compare in source order: the three coefficients, current
`vn_e`, prior `vb_e`, prior-prior `vbb_e`, each materialised product, the two
left-associated partial sums, and completed `va_e`. Substitute each recorded
input into the candidate replay one variable at a time; cumulative replay is
allowed only after all individual rows are printed.

The gate parses the self-describing record, requires both ranks exactly once,
requires all five named fields and their expected shapes, and reproduces round
195's 15,943-cell `va_e` debt. Plants independently reorder the input registry,
perturb one exact coefficient bit, break the history rotation mapping, and
remove one required stream; every plant must refuse.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R196-P1 | The admitted round-96 record is sufficient. | Both rank shards expose `j003_ext_coef`, `j002_va_new`, `j001_va_new`, `i000_vn_e`, and `j003_va_ext`; their record-only replay closes `va_e` bit-for-bit. | Any missing/malformed stream or replay residual: **REFUTED**; write a fail-closed acquisition and stop. |
| R196-P2 | The three AB3 coefficients are bit-exact. | Candidate and recorded `za1/za2/za3` are bit-identical. | Any coefficient differs: **REFUTED**; stop at the first coefficient. |
| R196-P3 | Current `vn_e` is the first non-bit velocity input. | Coefficients exact and `vn_e` non-bit; replacing only `vn_e` moves completed `va_e` toward or to NEMO. | Exact `vn_e`: **REFUTED**; continue in source order through `vb_e`, then `vbb_e`. A null substitution is recorded. |
| R196-P4 | One recorded input substitution closes substep-3 `va_e`. | A source-ordered single-variable replay reaches 0 / 26,640 unequal. | Residual after every single substitution: **REFUTED**; report the cancelling unit and stop before depth, reciprocal or trajectory. |
| R196-P5 | Controls bind. | Registry-order, coefficient-bit, rotation-map and missing-stream plants each refuse. | Any plant stays green: invalid instrument; report no input claim. |

If P1, P2, P4 or P5 fails, no trajectory gate can reverse the source-order
result. The final package tree must equal the frozen base. The newly named
input or cancelling unit becomes round 197's OPEN item.

ASKED choices: round 195's OPEN source-ordered AB3 input split.  
UNASKED choices: empty.
