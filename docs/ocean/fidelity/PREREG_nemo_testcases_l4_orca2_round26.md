# NEMO testcase Lane 4 — ORCA2 card round 26 preregistration

Date: 2026-09-26

Parent: `7f382b7ccd33aa8c8785d52493192057c5d60dcc`

Status: **PREREGISTERED BEFORE ROUND-26 SCIENTIFIC SCORING.**

Round 26 executes round 25's first OPEN item.  It splits the live-thickness
bundle into the three thickness positions written by the compiled level
operator, with the mask and metric paths held at the parent:

1. `f_curl`: NEMO's carried live F thickness is used only in the curl term;
2. `kbb_divergence`: NEMO's carried T/U/V Kbb thicknesses are used only in
   the divergence term;
3. `kmm_divisor`: NEMO's carried U/V Kmm thicknesses are used only in the
   outer momentum divisors.

Every arm starts from the clean parent and is reverted before the next arm.
Every ladder number is **independent with Decision-52 SSH**.  No
given-NEMO-entry operator score is mixed into a trajectory table.  The six
sea-ice selectors and the card's `unmeasured_features` tuple remain frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round26/`.

## Compiled statements and executed branch

The admitted build forms the F curl with live `e3f_3d*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`.
It forms the divergence with live T/U/V Kbb thicknesses at `:127-129`, then
divides the U and V curl increments by their live Kmm face thicknesses at
`:132-140`.  These are three distinct positions in the compiled arithmetic;
the experiment does not infer them from the trajectory.

Each arm uses the same ORCA2 card, admitted NEMO record, surface forcing, CPU
backend, fp64 policy, ten-step maximum, and Decision-52 sea-surface entry.  No
arm changes a resolved configuration leaf.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R26-P1 | Each arm changes exactly its named compiled thickness position and no metric, mask, config, carried-state, or sea-ice boundary. | The clean arm diff and instantiated card audit show one isolated boundary and zero resolved-card changes. | Any arm changes another boundary or resolved leaf. |
| R26-P2 | Each non-vacuous arm first moves at kt=1 stage-2 momentum and leaves the first NEMO mismatch at kt=1 stage-1 T. | The first moved row is stage-2 U or V and the first non-bit statement is unchanged. | An earlier row moves, an arm is vacuous, or the first NEMO boundary changes. |
| R26-P3 | The Kbb-divergence arm alone reproduces the round-25 non-positive raw-`e3w` refusal while entering kt=4. | Kbb completes kt=1..3 and refuses at the registered boundary; F-curl and Kmm reach kt=10. | F-curl or Kmm also refuses, Kbb reaches kt=4, or any arm refuses earlier/differently. |
| R26-P4 | No arm moves a formerly bit-identical row off the bar. | Every arm has an empty AT-BAR-left set through its measured extent. | Any formerly exact row becomes non-bit. |
| R26-P5 | The three arms recompose the round-25 live-thickness first movement when their stage-1 tendencies are combined. | The combined isolated operands reproduce the published kt=1 stage-2 U/V scores. | The scores differ, proving an interaction or an incomplete split. |
| R26-P6 | The comparison instrument can fail. | A one-ULP exact-row plant and a rigid citation shift are refused. | Either planted violation passes. |

Failed predictions remain **REFUTED**.  If more than one subcomponent exposes
the refusal, all are named.  If none does, the pair interaction becomes the
OPEN item.  A vacuous arm remains measured and is not promoted to a match.

## Landing and stop rules

- Commit this preregistration before running any scientific arm.
- Run each arm from the same clean parent and retain its clean-worktree stamp.
- Register every moved field row and direction against a fresh parent ladder;
  never synthesize rows beyond a refusal.
- Do not relax the raw-`e3w` refusal or add a stabilizer.
- This is an attribution round.  No model statement lands; all experimental
  model changes are reverted before the receipt.
- No NEMO acquisition, configuration choice, carried-state change, sea-ice
  edit, or NEMO source edit is authorized.

## Choices

ASKED: Decision 54 and round 25's OPEN item authorize this compiled-order
thickness split.

UNASKED: none.
