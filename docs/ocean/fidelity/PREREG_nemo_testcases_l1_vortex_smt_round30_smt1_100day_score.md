# Preregistration — VORTEX_SMT round 30 (lane round 242): SMT-1 100-day score

Frozen before scoring or running legoESM. Base: `2c6007d46` (round 241 /
VORTEX_SMT round 29). Scientific evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round242/`; the admitted NEMO
trajectory remains the operator-created round-241 record.

## Continuation contract

Round 241 already preregistered the complete measurement and stopped for the
daily NEMO record. The operator has now run that committed acquisition. Its
log ends with `STATUS ADMITTED`, a byte-identical restart control, a firing
header plant, and the requested ready markers. This continuation makes no new
scientific prediction and does not revise a falsifier after seeing a score.

The frozen round-241 predictions remain binding:

1. The admitted record contains exactly 100 daily restarts at steps 30 through
   3000 and all required finite fields; malformed or missing input refuses.
2. Its first ten entries reproduce the round-237 certified SMT-1 registry
   exactly before any 100-day number is accepted.
3. Every daily T/u/v/ssh RMS and maximum is finite, and day-100 T RMS is below
   `1e-3 K`; a larger result is retained as a refutation.
4. Days 1/2/5/10/20/30/60/100 are registered and compared with the existing
   SMT-0 vector curve using the same scorer. The curve does not establish a
   causal owner.
5. The shared renderer produces 100 frames and the day-1/30/60/100 montage
   with the SMT-1 bathymetry, fixed day-100 SSH-difference scale, and fixed
   day-1 velocity scale. Missing data or a zero/non-finite scale refuses.
6. This round is measurement only: no model, card, configuration, carried
   state, or certified short-run registry changes.

## Execution and controls

The committed round-210 scorer runs the production JIT closure on CPU in
fp64/libm and writes daily legoESM snapshots plus the score JSON and curve.
The committed renderer consumes those exact snapshots and NEMO restarts. A
post-run gate must check the 100-day count, short-run reproduction, finiteness,
checkpoint completeness, the frozen bound, and expected visual artefacts; its
planted missing checkpoint must print `STATUS PLANT-FIRED` and exit nonzero.

No hidden option, threshold, interpolation, stabiliser, or alternate record is
introduced. If the admitted record fails calibration, the round stops rather
than re-baselining it.
