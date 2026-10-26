# PREREGISTRATION — coastal donor source of the one-cell wall discrepancy

Status: frozen after mechanism review identified that the earlier classifier
tested the receiving wet face rather than its bridged upstream donor, before
the donor-aware execution.

## Retraction and source chain

The two receiving-support coastal verdicts are **RETRACTED**. A receiving wet
face cannot itself have `umask=0`, and material-residual overlap at receiving
coordinates does not test the source. The measured receiver `(j=1,i=49)`, its
delta, one-cell support, and material-support overlaps remain descriptive.

NEMO restart `utau_b` is a U-point array (`sbcmod.F90:554-558`).
`read_nemo_restart_before` loads it into `NemoBeforeState.tau_x`, but the current
comments/types call it T-point and `bridge_before_state_topo` stores it directly
as lego `tau_x_prev` (`nemo_io.py:237,264`;
`nemo_state_bridge.py:802-833`). `_step_impl` then averages that array with
current T-point stress before `surface_stress_faces` interpolates the mixture
to U faces (`ocean_model_latlon_cgrid.py:3489-3498`;
`ocean_pe_latlon_cgrid.py:3619-3622`). Thus the prior U field is interpolated a
second time.

## Exact donor-aware numbers and gates

Using the same committed kt=5761 bridge and production interpolation, print:

1. the wet direct-residual argmax receiver `(j,i)` and the east input index
   mixed into its lego face by `interp_cell_to_uface`;
2. raw ocean-sign bridged prior stresses at receiver and east donor, their
   ratio, and each point's `umask`, adjacent T masks, and exact
   `sbcmod.F90:543-544` multiplier;
3. `tau_production`, captured from the production helper, and
   `tau_stagger_correct = 0.5*(utau_b_U + current_tau_U)` at the receiver;
4. predicted direct-increment delta
   `rDt*(tau_production-tau_stagger_correct)/(rho0*dz0_u)` and its relative
   error against the measured lego-minus-NEMO direct delta;
5. full wet-domain peak-equivalent faces and material-support Jaccard between
   that predicted delta and the measured direct residual; and
6. Jaccard of predicted-delta support with the measured `F_slow` residual.

The review's pre-run expected values are receiver `(1,49)`, east donor
`(1,50)`, prior stresses about `2.326325658e-4` and `4.652651316e-4 Pa`, donor
ratio 2, receiver wet/multiplier 1, donor dry-coastal/multiplier 2, and a
predicted 25% receiver excess.

**CONFIRMS_COASTAL_DONOR_DOUBLE_INTERPOLATION** iff receiver is `(1,49)`, its
east donor is coastal-unmasked, donor/receiver stress differs from 2 by at
most `1e-8`, receiver predicted-delta relative error is `<=1e-6`, and both
predicted-support Jaccards are `>=0.90`. **REFUTES** iff the donor is not
coastal, its ratio differs from 2 by `>=0.10`, prediction error is `>=0.10`, or
either Jaccard is `<=0.10`; otherwise **UNRESOLVED**.

Controls must (a) shift the donor index by one and fail the donor/coastal
condition, (b) replace the donor stress with the receiver stress and make the
predicted receiver anomaly zero, and (c) move a synthetic predicted support
from known coastal donors to known noncoastal points and make the overlap gate
switch from CONFIRM to REFUTE. Any failure invalidates the verdict.

The probe must print every earlier retraction, Git SHA, this preregistration
commit, input/flag/content hashes, and controls. The earlier impossible
receiving-face and receiving-support coastal labels must not appear as current
verdicts. No production physics edit and no GPU run is authorized here; the
faithful-fix candidate is the U-stagger-aware representation of bridged
`utau_b/vtau_b`, to be implemented only under a separate preregistered arm.
