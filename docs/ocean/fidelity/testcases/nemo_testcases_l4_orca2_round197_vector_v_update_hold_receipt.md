# ORCA2 round 197 — substep-2 vector V update hold

Date: 2026-10-09. Incoming tip:
`60534edc6f0f95300c39c617fc81644024279238`. Preregistration:
`6dfeed70f`. Measurement instrument: `7be00d4a4`.

## Result

**HELD.** Every number is **independent hierarchy rung 0**. The card starts
from its corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this result. The shipped
ORCA2 card, sea ice, all six ice selectors and its `unmeasured_features` tuple
are unchanged. `git diff 60534edc6..HEAD -- packages` is empty.

The substep-2 entry V and external timestep are bit-exact. The first non-bit
operand of NEMO's vector-form V update is the surface-pressure-gradient
`zv_spg`: 1,226 of 26,640 record cells differ, maximum absolute difference
`5.879156869962787e-07 m s-2`, first at `[j=0,i=49]`. Its next operand,
the completed `zv_trd`, is also non-bit on 16,055 cells, maximum
`4.064540562696444e-08 m s-2`, first at `[j=1,i=49]`. The frozen slow forcing
is exact.

Neither operand closes the completed post-associated V alone. Replacing only
`zv_spg` leaves 15,943 unequal cells and maximum
`9.009781378703638e-07 m s-1`; replacing only `zv_trd` leaves 563 signed-zero
differences (maximum absolute 0.0). Replacing both in compiled source order
closes every one of the 26,640 cells bit-for-bit. Therefore `zv_spg` plus
`zv_trd` is a two-operand cancelling unit and no partial statement is eligible
to land.

The cited statement is NEMO's
`va_e = (vn_e + rDt_e * (zv_spg + zv_trd + zv_frc)) * ssvmask` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727`. The subsequent
seven-array association is at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`. Replaying only
the recorded update and that already-shared association reaches the recorded
`j002_va_new` bit-for-bit, so the association is exonerated at this boundary
and no raw pre-association acquisition is needed.

## Frozen source-ordered table

| boundary | unequal / 26,640 | maximum absolute | disposition |
|---|---:|---:|---|
| entry `vn_e` | 0 | 0.0 | exact |
| `rDt_e` | 0 / 1 | 0.0 | exact |
| `zv_spg` | 1,226 | 5.879156869962787e-07 m/s2 | **first non-bit operand** |
| `zv_trd` | 16,055 | 4.064540562696444e-08 m/s2 | second non-bit operand |
| `zv_frc` | 0 | 0.0 | exact |
| `ssvmask` | 68 | 1.0 | known fold-mask difference; null after exact update |
| raw completed `va_e` | 16,505 | 7.619236634995991e-04 m/s | non-bit |
| post-associated `va_e` | 16,506 | 9.009781378703638e-07 m/s | reproduces round 196 |
| substitute `zv_spg` only | 15,943 | 9.009781378703638e-07 m/s | not exact |
| substitute `zv_trd` only | 563 | 0.0 | signed-zero debt remains |
| substitute `zv_spg` + `zv_trd` | 0 | 0.0 | exact |

The raw reference update differs from its post-associated target on 109
signed-zero cells, so the association control is non-vacuous. Applying the
association closes all 109. The candidate offline replay reproduces the
passive candidate output bit-for-bit. The passive traced and untraced solver
states remain array-identical in SSH, U, V, both external modes and both
transport averages.

The measurement artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round197/vector_v_update.json`
(SHA256 `a12f59586cbf8dcaf684432b5db2c608c3c0202c46ccc04804ea586e28594476`).

## Prediction ledger

| prediction | verdict |
|---|---|
| R197-P1 record sufficient | **CONFIRMED**: record-only update plus association and candidate replay are exact. |
| R197-P2 entry `vn_e` exact | **CONFIRMED**: 0 / 26,640 unequal. |
| R197-P3 `zv_spg` first non-bit | **CONFIRMED**: 1,226 cells before `zv_trd`. |
| R197-P4 one substitution closes | **REFUTED**: the cumulative `zv_spg` + `zv_trd` pair is required. |
| R197-P5 association adds no debt | **CONFIRMED**: reference replay is exact after association. |
| R197-P6 controls bind | **CONFIRMED**: registry, timestep, association and missing-stream plants refuse. |

## Validation and choices

The four known-answer plants refuse with exit 2. The pre-measurement focused
battery passed 22/22 in 1.37 s. The final focused set (rounds 146, 195, 196,
197 and the citation gate) passed 62/62 in 4.26 s (JUnit SHA256
`77d0d2be204b08019df552ca39449e3ed03b201361aba7d118dd1426800cb5bc`).

The default citation audit passes 274 citations with zero failures, unmapped
citations or failing map entries (SHA256
`fc2f596428cc3d6cfc94f2ff404aaa99be70313df1ef4596dda291b890ca6315`).
This receipt passes two citations with the same zero counts (SHA256
`ec3993427cee3ca6c33f478cd0cd7e25f5b0ebbaf9dc73b11a20536071ed9876`).
Shifting the vector-update span by two lines fails as required (exit 1,
`SYMBOL-NOT-AT-LINE`).

The required single `tests/ocean/fidelity -n 12` battery collected 2,946
tests and reached 97%. It displayed four reds before it stopped producing
progress: the registered stamp-scope ratchet, worktree-stamp ratchet and GYRE
spread-floor record gate, plus round 179's acquisition layout plant. The
round-179 test's isolated error is `REFUSE: acquisition requires a clean
committed tree`, caused by this receipt still being untracked during that
battery rather than by its layout plant. The battery emitted no terminal XML
and was interrupted once; it was not rerun. This is an incomplete battery,
not a green claim. The four displayed IDs were rerun alone as required; the
three registered reds reproduced, while round 179 is rerun again after the
receipt commit on the clean tree.

The independent-review verdict and clean-tree round-179 result are recorded
by the round-closing commit.

No configuration choice, carried-state change, stabiliser, tolerance, NEMO
source change or executable observer was introduced. ASKED choices: round
196's OPEN source-ordered substep-2 vector update and association split.
UNASKED choices: empty.

## OPEN

Round 198 keeps the same complete private arm and splits `zv_spg` and
`zv_trd` as one cancelling unit, without landing either half. Walk `zv_spg`
first in source order through the back-interpolated SSH and its V-gradient;
then walk the already-completed EEN-plus-drag `zv_trd`. Re-score the pair only
after both operands are source-exact. Do not revisit the midpoint-depth,
reciprocal or trajectory census before this unit closes. The admitted round-96
record already carries `j002_sshp2_bck`, `j002_zv_spg`, `j002_cor_v`,
`j002_trd_v`, and the drag operands; no acquisition is expected unless an
unrecorded pre-gradient or EEN intermediate becomes the first required target.
