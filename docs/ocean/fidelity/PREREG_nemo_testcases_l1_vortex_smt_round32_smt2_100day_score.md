# Preregistration — VORTEX_SMT round 32 (lane round 244): SMT-2 100-day score

Frozen before parsing or scoring the operator-created SMT-2 record. Base:
`b937f824e` (round 243 / VORTEX_SMT round 31). Scientific evidence belongs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round244/`; the admitted
NEMO trajectory remains the operator-created round-243 record.

## Continuation contract

Round 243 preregistered the complete measurement and stopped for the daily
NEMO record. The operator has now run that committed acquisition and reported
exit 0 with both the common-driver oracle-ready marker and the round-local
ready marker. This continuation makes no new scientific prediction and does
not revise a falsifier after seeing a score.

The frozen round-243 predictions remain binding:

1. The record contains exactly 100 daily restarts at steps 30 through 3000,
   all required fields are finite, and the common driver's self-describing
   admission checks pass. Malformed or missing input refuses the score.
2. Its first ten step-entry frames reproduce the round-237 certified SMT-2
   50-row registry exactly before any 100-day number is accepted.
3. Day-100 wet three-dimensional temperature RMS is within 2x of SMT-1's
   `4.3321114781972461e-05 K`, and NEMO's day-100 maximum absolute U is below
   SMT-1's. Either failure is retained as a refutation; no bound is revised.
4. Days 1/2/5/10/20/30/60/100 are registered for T/u/v/ssh RMS and maxima and
   compared with SMT-1 through the same scorer. The curve does not establish
   a causal owner.
5. The shared renderer produces 100 frames and the day-1/30/60/100 montage
   with SMT-2's bathymetry, fixed day-100 SSH-difference scale, and fixed
   day-1 velocity-arrow scale. Missing input or a zero/non-finite scale
   refuses rendering.
6. This round is measurement only: no model, card, configuration, carried
   state, or certified short-run registry changes.

## Execution and controls

The committed round-210 scorer runs the production JIT closure on CPU in
fp64/libm and writes daily legoESM snapshots plus the score JSON and curve.
The committed renderer consumes those exact snapshots and NEMO restarts. A
post-run gate must check admission, the 100-day count, short-run reproduction,
finiteness, checkpoint completeness, both inherited predictions, and the
visual artefacts. Its planted missing checkpoint and bound violation must each
print `STATUS PLANT-FIRED` and exit nonzero.

No hidden option, threshold, interpolation, stabiliser, or alternate record is
introduced. If the operator record fails admission or calibration, the round
stops rather than re-baselining it.
