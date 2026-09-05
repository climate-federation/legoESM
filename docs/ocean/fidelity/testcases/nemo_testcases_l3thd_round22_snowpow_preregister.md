# Lane 3b round 22 addendum — certified SI3 snowfall POW

Date: 2026-09-05

Parent implementation: `f056785ff519`

The required post-change native-site inventory found a fourth live fractional
power on the already-certified ORCA1 SI3 path.  NEMO 5.0.2
`icevar.F90:1619-1628` implements both its 2-D bulk and 1-D thermodynamics
interfaces as `pout = 1._wp - pin**rn_snwblow`; `sbcblk.F90:1294` and
`icethd_dh.F90:98` call those two interfaces.  legoESM currently repeats the
formula at `ice/sea_ice.py:722-726` and `ice/bitz_lipscomb.py:727-731`, and both
powers remain native.  This is not an unmeasured operator: both consumers are
inside the C1D gates requested in the main preregistration.

Leaving this row native would make a broader “SI3 certified path uses scalar
libm” statement false.  The one-variable change is therefore preregistered
before measurement: centralize the source statement in one private shared snow
helper, route only its power through `core.transcendentals.pow`, preserve the
existing per-statement `nemo_source_round` identity, and replace both callers.
No card selector or second snowfall implementation is added.

Predictions:

- SI3 bulk remains 271,560 / 271,560 bit-identical;
- the thermodynamics step/year JSONs remain byte-identical to Round 19;
- exchange, slab, LOCK, OVERFLOW, and GYRE rows remain as measured in the main
  round because this source statement is either already accounted for in the
  first two gates or is not executed there;
- a private input-dependent poisoned POW return changes a scored snowfall/
  freshwater downstream row and makes the SI3 bulk gate exit nonzero.

Any changed unpoisoned row refutes the prediction and will be reported as a
finding.  This addendum is an ASKED-scope completion discovered by the required
inventory, not multi-category work and not permission to convert any unmeasured
operator.
