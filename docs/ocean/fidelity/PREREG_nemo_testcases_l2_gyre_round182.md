# Preregistration — round 182, DINO positive live-W geometry and carried-N2 landing

Committed before reading numerical mesh/restart values or running a DINO,
GYRE, tank, generic-card, or trajectory measurement.  Round 181 established a
source-exact and trajectory-beneficial carried step-entry `rn2b` candidate, but
both DINO arms stopped before it executed because the complete raw-mesh
`e3w_int` positivity assertion fired.  This round first discriminates the
geometry defect, repairs only that upstream bridge path, proves the repair is
non-vacuous, and only then requalifies the held candidate.

No configuration value, carried-state schema, threshold, cadence, timestep,
or NEMO source is changed.  The production candidate lands only if the full
Decisions 43/45/55/59 gate and every executing card accept it.

## Compiled DINO statements

The compiled DINO z-coordinate branch fills the full horizontal `pe3w` field
from a maximum-depth `zflat` before it applies the land/bottom indices at
`DINO/BLD/ppsrc/nemo/usrdef_zgr.f90:123-135`.  Its compiled MI96 routine forms
`pe3w` from the depth arrays and then reconstructs the depths from those scale
factors at `DINO/BLD/ppsrc/nemo/zgr_lib.f90:198-209`.  NEMO therefore retains
positive raw W geometry outside the wet tracer mask; land and below-bottom
cells are not represented by zero thickness.

The compiled domain separately forms the wet reference depth with `tmask` and
sets its reciprocal to zero on land at
`DINO/BLD/ppsrc/nemo/domain.f90:194-216`.  The QCO program then evaluates
`r3t = ssh*r1_ht_0` at `DINO/BLD/ppsrc/nemo/domqco.f90:205-207`, so dry cells
use zero stretch anomaly.  Finally, compiled `bn2` divides by the positive
live `e3w_3d*(1+r3t)` and only then applies `wmask` at
`DINO/BLD/ppsrc/nemo/eosbn2.f90:1508-1517`.  A bridge that masks raw `e3w_0`,
or lets a decoded dry restart sentinel contaminate its dry stretch, is not this
program.

The pre-implementation search found the single shared geometry producer
`nemo_bn2_live_geometry`/`nemo_e3w_from_live_gdept` in
`packages/ocean/legoesm/ocean/eos.py`, the single DINO topology bridge in
`packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py`, and existing
raw-mesh coverage in `nemo_dino_mesh_gate.py` and `test_nemo_bn2.py`.  This
round extends those paths; it does not create a second geometry formula.

## Frozen predictions and falsifiers

1. The DINO coordinate's stored full-grid `nemo_e3w_0` has **zero** non-finite
   or non-positive values.  Any invalid raw cell refutes the bridge-input
   hypothesis and stops the round before a repair.
2. The failing complete live `e3w_int` has at least one invalid cell, while its
   wet-interface subset has **zero** invalid cells.  Invalid cells must overlap
   only dry/below-bottom/decoded-sentinel locations.  A wet invalid cell, or
   an invalid cell with finite raw geometry and finite dry stretch, refutes the
   proposed bridge repair and requires a different owner.
3. Reconstructing the live divisor with NEMO's compiled scope preserves every
   wet-interface value bit-for-bit, produces **zero** invalid full-grid cells,
   and leaves the raw positive dry/halo reference geometry in place.  Any wet
   bit movement or remaining invalid cell refuses the fix.
4. A planted zero, negative, NaN, and decoded dry-sentinel cell each makes the
   new wet/dry/halo gate print `STATUS PLANT-FIRED` and exit nonzero.  A plant
   that reaches a success marker or fails before the planted condition is
   checked invalidates the test.
5. After the repair, both one-day DINO arms reach and execute the mixed-layer
   recurrence.  The carried arm is the production DINO route and must move no
   candidate-selected DINO value relative to the repaired production control;
   the explicit recompute counterfactual may differ but must finish.  If the
   candidate arm worsens any certified DINO row, the GYRE candidate does not
   land.
6. Reapplying only Round 181's carried-N2 route reproduces its local proof:
   developed GYRE `nmln` and `hmlp` are **0 cells unequal** under the complete
   production JIT step and eager execution.  Any unequal cell refuses the
   candidate.
7. The trajectory predictions are the Round-181 controlled values: first over
   bar remains kt=3; kt2 T/S/U/V remain at bar; day-30 T RMS is
   `2.3276772050683987e-06 K`, day-240 is
   `6.5861718814795174e-05 K`, and day-360 is
   `2.6709923853294689e-03 K`, subject only to exact same-tip reproduction.
   Failure to reproduce is reconciled before any direction claim.
8. Recipe-derived census predictions: GYRE changes; both DINO Kamm cards
   execute the carried route and are measured; LOCK_EXCHANGE, OVERFLOW,
   ORCA2-zps, and the generic NEMO-GYRE recipe do not execute it and retain
   their certified outputs.  Any additional executing card is measured before
   landing, never waived.

If the DINO geometry defect cannot be repaired from the existing compiled mesh
and restart without a configuration choice, this round stops with an explicit
`DECISION_NEEDED`; no other walk starts.
