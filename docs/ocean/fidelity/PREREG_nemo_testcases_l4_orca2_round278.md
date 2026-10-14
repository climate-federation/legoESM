# ORCA2 round 278 preregistration — Decision 122 landing

Frozen before any round-278 ladder runs.  Decision 122 (user, 2026-10-14):
a statement certified bit-exact on a geometry-ladder rung lands on the
production ORCA2 card when (a) its ladder receipt is cited, (b) the same
compiled NEMO line runs on the full card, (c) no exact row is lost on rung 0,
OMT-1..4 or rung 10 (both labels), (d) the first over-bar row is unchanged
or toward on every card.  The toward/away/score-equal census (early kt=1..3,
late kt>=4) is registered, never a veto.

Protocol: round 277's measurement and round 276's classifier, driven by
`nemo_testcase_l4_orca2_round278_d122_landing_gate.py` (one arm per unit,
`--decide` applies (c)+(d) over the seven card/labels).  Each unit is
measured singly against the same-tree control, then the final set together.

## (b) — compiled-line check (deck/source read, not a measurement)

Every cited ORCA2G1 build line was searched verbatim in the ORCA2 build
`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo` (script
`orca2_rounds/round278/line_map.py`).  All are present; branch selectors read
from the ORCA2 resolved namelist (`ln_eos80=T`, `ln_drgimp=T`,
`ln_dynspg_ts=T`, `ln_dynadv_vec=T`, `ln_dynvor_een=T`).  One statement fails
(b): G1's EEN vorticity written order certifies the `np_COR` line
`zwz = ff_f * z1_e3f` (ORCA2 build dynvor.f90:745), which ORCA2's vector-form
call (`np_CRV`) does not execute.  It is not a candidate (D96 governs it).

Candidates (15 units, 16 statements): G2 divisors; G2 tracer diagonal; G2
ascending sum; G3 stored r3u (with its divisor host, as one arm); G1 EEN
fold pair (atomic); G1 EEN ratio written order; stage-1 RHS; stage velocity
order; ZDF barotropic removal; completed-stage top-V fold exchange; stage
F ratio; stage T ratio; stored T-ratio product; EOS depth product; stage
U/V ratio.

## Predictions and falsifiers

- **R278-P1.** No single unit loses an exact row on any card (rounds
  275-277 reported zero everywhere).  Falsifier: any loss.
- **R278-P2.** Every single unit keeps the first over-bar row unchanged or
  toward on every card.  Falsifier: any away.
- **R278-P3.** The top-V fold exchange reproduces round 275's growth on
  rung 0 (kt=10 SSH max > 8 m).  If it nonetheless meets (c)+(d), it is NOT
  landed in this round: a model that grows ~20x in ten steps is reported as
  a decision for the user, the selector stays off meanwhile.
- **R278-P4.** The set of all units meeting (c)+(d) singly (P3's exception
  removed) also meets (c)+(d) together.  Falsifier: the set fails; then the
  largest passing subset is found by removing the unit whose single census
  is most away on the failing card, one at a time.
- **R278-P5.** GYRE ten-step trajectory 0 ULP worsening and the CPU 30-day
  member byte-identical: the selectors flip only on the ORCA2 card family.

No configuration, stabiliser, carried-state or sea-ice change beyond the
landed selectors.
