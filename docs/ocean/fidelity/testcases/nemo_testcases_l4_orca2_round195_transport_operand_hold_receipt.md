# ORCA2 round 195 — substep-3 V-transport operand hold

Date: 2026-10-09. Incoming tip:
`a7928d44c06c9f9ef8b52e4354f9604778afa282`. Preregistration:
`8a1a581a7`. Measurement instrument commits: `a63b43c71` and
`0ca01f1a7`.

## Result

**HELD.** Every number is **independent hierarchy rung 0**. The card starts
from its corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this result. The shipped
ORCA2 card, sea ice, all six ice selectors and its `unmeasured_features`
tuple are unchanged. `git diff
a7928d44c06c9f9ef8b52e4354f9604778afa282..HEAD -- packages` is empty.

Round 194's substep-3 materialised V transport is a two-operand cancelling
unit. The static `e1v` is bit-exact on all 26,640 record cells. The first
non-bit operand in compiled source order is midpoint `va_e`: 15,943 unequal
cells, maximum absolute difference `1.604736666251539e-06 m s-1`, first at
`[j=1,i=49]`. Its producer is NEMO's AB3 midpoint extrapolation
`va_e = za1*vn_e + za2*vb_e + za3*vbb_e` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:502-511`.
This round names that producer but does not attribute the difference to its
arithmetic: its three inputs have not yet been split.

The midpoint depth `zhvp2_e` is independently non-bit on 68 northern-fold
cells, maximum `0.0033989498219852976 m`, first at `[147,29]`. NEMO builds it
from `hv_0` and the neighbouring area-weighted midpoint sea surfaces at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:542-545`.
Replacing only `va_e` reduces completed-`zhV` debt from 15,943 to 68 cells and
the maximum from `188.93279014341533` to `0.2086417620885186 m3 s-1`, but is
not exact. Replacing only `zhvp2_e` leaves the 15,943-cell debt unchanged.
Replacing both in source order closes NEMO's left-associated
`zhV = e1v * va_e * zhvp2_e` statement at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:564-570`
bit-for-bit: `0 / 26,640` unequal.

No reciprocal or trajectory census follows this result. Prediction R195-P4
is **REFUTED** because no single substitution closes `zhV`; the next round
must walk the earlier `va_e` producer before the 68-cell depth partner can be
revisited.

## Frozen source-ordered table

| boundary | unequal / 26,640 | maximum absolute | disposition |
|---|---:|---:|---|
| `e1v` | 0 | 0.0 | exact; exonerated |
| `va_e` | 15,943 | 1.604736666251539e-06 m/s | **first non-bit operand** |
| `e1v * va_e` | 15,943 | 0.2698720398399246 m2/s | carries `va_e` debt |
| `zhvp2_e` | 68 | 0.0033989498219852976 m | separate fold debt |
| completed `zhV` | 15,943 | 188.93279014341533 m3/s | non-bit |
| substitute `va_e` only | 68 | 0.2086417620885186 m3/s | not exact |
| substitute `zhvp2_e` only | 15,943 | 188.93279014341533 m3/s | not exact |
| substitute `va_e` + `zhvp2_e` | 0 | 0.0 | exact |

The admitted round-96 records contain both ranks exactly once and 65
substeps. Their self-describing fields `j003_va_ext`, `j003_hvp2_e`, and
`j003_zhV` are present. The passive traced and untraced solver states are
array-identical in SSH, U, V, both external modes and both transport averages.
The round-194 prerequisite reproduces exactly: completed `zhV` is bit-exact
at substeps 1 and 2 and non-bit at substep 3.

The measurement artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round195/transport_operands.json`
(SHA256 `10085183093b47f264a0648ea41ffdbc1e717f314c971de69ebfdf176a6de0b8`;
log SHA256 `158adf92ca7ed30bac9ddaf9429eeb9bcd9ed5abe7ca7f14fb9f201f98dd2efb`).

## Instrument correction and controls

The first invocation produced no measurement. It copied round 194's temporary
unmasked-reciprocal hook after production had restored that private hook, so
the current function signature rejected the unknown keyword before JIT. The
downstream reciprocal is not an operand of `zhV`; removing the argument
changes no registered quantity. A direct signature regression test now pins
every private hook this measurement actually calls. The refused transcript
was overwritten by the successful run and is therefore not cited as an
artifact; the correction is retained in commit `0ca01f1a7` and here as a
first-class retraction.

The gate reuses the round-146 product splitter, extended by a substep argument;
there is no second formula. Registry-order, one-bit `e1v`, and missing-stream
plants each refuse with exit 2 (log SHA256
`246c2e32d506dc8fd60214825c4b94d11ef724018bfad7b54f97d65312dda725`).

## Prediction ledger

| prediction | verdict |
|---|---|
| R195-P1 record sufficient | **CONFIRMED**: both shards and all three substep-3 streams present. |
| R195-P2 static `e1v` exact | **CONFIRMED**: 0 / 26,640 unequal. |
| R195-P3 `va_e` first non-bit | **CONFIRMED**: 15,943 unequal before the depth operand. |
| R195-P4 one substitution closes `zhV` | **REFUTED**: the cumulative `va_e` + `zhvp2_e` pair is required. |
| R195-P5 controls bind | **CONFIRMED**: all three plants refuse. |

## Validation and choices

The focused round-146 plus round-195 battery passed 28/28 before measurement;
the corrected signature-focused battery passed 6/6. Final focused tests,
citation gates and the required single fidelity battery are recorded in the
closing commit.

The separate `codex exec --sandbox read-only` review could not initialize its
in-process app-server client on the read-only filesystem. Verdict:
**independent review unavailable in-sandbox** (SHA256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

No configuration choice, carried-state change, stabiliser, tolerance, NEMO
source change or executable observer was introduced. ASKED choices: round
194's OPEN source-ordered split. UNASKED choices: empty.

## OPEN

Round 196 keeps the same private arm and splits the substep-3 midpoint
velocity producer in compiled order: coefficients `za1/za2/za3`, current
`vn_e`, prior `vb_e`, and prior-prior `vbb_e`, against the admitted round-96
record. It must name the first non-bit input before changing the arithmetic or
revisiting the 68-cell `zhvp2_e` partner, the reciprocal, or any trajectory
census. If the record lacks any input stream, write a fail-closed acquisition
launcher and stop with `ACQUISITION_NEEDED`.
