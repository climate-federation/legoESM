# OVERFLOW-zps barotropic substep walk preregistration

Status: **PREREGISTERED; no frame comparison or causal-arm metric has been
computed at this revision.**  This document freezes the Lane-1 OVERFLOW kt=2
initiator experiment before the instrumented oracle rerun.

## Resolved reference program

The reference is NEMO 5.0.2, certified `OVERFLOW-zps`, `key_qco + key_RK3`.
The resolved log reports `ln_dynspg_ts=T`, `ln_bt_fw=T`, `nn_bt_flt=1`,
`rn_bt_alpha=0`, `nn_e=3`, and `rDt_e=10/3 s`.  RK3 requires forward mode at
`src/OCE/DYN/dynspg_ts.F90:270-278`; its frozen SSH/U/V slow forcings are read
at `:280-300`; the loop starts from Kmm at `:484-493`.  The namelist selects
`ln_dynadv_up3=T` and `ln_dynadv_vec=F`, so the executed external-mode update is
the flux-form branch at `dynspg_ts.F90:731-761`.  The resolved `namdrg` has
`ln_drg_OFF=T`, making both drag operands exact zero.  Rotation is also exact
zero because the case geometry has `f=0`.

NEMO extrapolates the midpoint U/V/SSH at `dynspg_ts.F90:534-562`, builds the
qco face depths and transports at `:573-609`, applies continuity at `:623-630`,
back-interpolates SSH and forms the surface PGF at `:672-687`, then advances
the flux-form face transport at `:731-761`.  In particular, the U update is

`(hu_e*un_e + rDt_e*(zhu_bck*zu_spg + zhup2_e*zu_trd + hu(Kmm)*zu_frc))*z1_hu`.

The current legoESM implementation instead uses the velocity-form composition
`U + dt*(PGF + drag + Fslow)` at
`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:1413-1419`.
This is a source-attested candidate, not yet an owner.

## Frozen 19-frame registry

The WRITE-only NEMO instrument and legoESM harness capture the following
ordered fields for every external substep.  `T`, `U`, and `V` mean native
NEMO T/U/V staggering after two-cell halo removal; comparison uses the
intersection of the certified wet masks, with no reduction before pointwise
subtraction.  Every row is fp64.

1. `eta_entry` (T, instantaneous Kmm/substep carry)
2. `u_entry` (U, instantaneous substep carry)
3. `v_entry` (V, instantaneous substep carry)
4. `eta_mid` (T, AB3 midpoint operand)
5. `u_mid` (U, AB3 midpoint operand)
6. `v_mid` (V, AB3 midpoint operand)
7. `transport_u` (U, midpoint volume transport divided by `e2u`)
8. `transport_v` (V, midpoint volume transport divided by `e1v`)
9. `eta_continuity` (T, instantaneous post-continuity SSH)
10. `eta_pgf` (T, back-interpolated SSH operand)
11. `pgf_u` (U, acceleration)
12. `pgf_v` (V, acceleration)
13. `slow_u` (U, frozen acceleration)
14. `slow_v` (V, frozen acceleration)
15. `drag_u` (U, acceleration; expected exact zero)
16. `drag_v` (V, acceleration; expected exact zero)
17. `u_exit` (U, instantaneous post-update velocity)
18. `v_exit` (V, instantaneous post-update velocity)
19. `eta_exit` (T, same instantaneous field as post-continuity SSH, retained
    as the loop-boundary control)

The oracle writer is permitted to WRITE these arrays only when `kt=nit000`.
The overlap control requires the pre-existing `oracle_step_entry_kt00000001`
and the newly instrumented run's step-entry file to be byte-identical.  The
reader is fail-closed on magic, version, dimensions, substep count, field count,
and EOF.  Planted controls independently perturb an oracle entry frame and an
oracle exit frame and must make the gate red.

## Bar and first-divergence rule

For each wet native field, the score is
`max(abs(lego-oracle))/max(max(abs(oracle)), 1)`.  EXACT means byte identity;
AT-BAR requires normalized error `<=1e-15`; otherwise the row is DEBT.  The
first DEBT in registry order is the only eligible initiating boundary.  A
downstream field cannot own an upstream mismatch.

## Pre-stated predictions

Baseline prediction: substep-1 entry, midpoint, transport, continuity, PGF,
slow, and zero-drag operands are AT-BAR; `u_exit` is the first DEBT.  A result
earlier than `u_exit` REFUTES the proposed update-composition boundary and the
walk stops there.

Scaling prediction: if the missing flux-form composition owns the boundary,
the directly reconstructed NEMO-minus-velocity-form exit increment has the
same location and sign as the measured `u_exit` residual and a magnitude ratio
between 0.1 and 10.  Outside that range REFUTES an owner label.

Causal-arm prediction: replacing the NEMO WS-RK3 flux-form identity's external
update by the literal transport update above reduces substep-1 `u_exit` by at
least 100x and moves the first boundary downstream.  It must also reduce the
certified kt=2 instantaneous-U normalized error from `3.082388670642283e-6` by
at least 10x without worsening T or SSH by more than 10x.  Otherwise the arm is
REFUTED.  No public micro-selector is permitted: NEMO exposes no switch inside
this scheme identity, so the literal composition is unbranched for the NEMO
WS-RK3 + flux-form package; legacy schemes retain their existing path.

Owner labels remain forbidden until the frame boundary, scaling check, and
one-variable causal movement all agree.  After a landing, kt=2..10 and the
6,120-step fp64 statistical scorer are mandatory regressions.

## Frozen-predicate outcome (post-run addendum)

The first-boundary prediction is confirmed only at the registered frame
level: substep-1 `u_exit` is the first DEBT (`1.87350135e-15`).  The preceding
`slow_u` row is AT-BAR but not exact (`5.62917768e-16`); multiplying it by the
external `10/3 s` step predicts the exit tail with ratio `0.99846`.  Entry,
midpoint transport, continuity, PGF, and zero drag are exonerated through that
boundary.

The substep-1 100x causal prediction is **REFUTED**: the literal NEMO
flux-form update leaves its roundoff-scale exit unchanged.  It does remove the
downstream legacy recurrence, improving substep-2--4 U exits by
`7.23e5`, `2.60e7`, and `9.54e7` respectively, so it is the confirmed owner of
that separate structural debt.

The whole-step predicate is also **REFUTED**.  kt=2 instantaneous U moves from
`3.08238867e-6` to `3.31108168e-6` instead of falling 10x; T is effectively
unchanged and SSH improves from `1.23723132e-7` to `1.04916076e-14`.
Therefore the transport update is not the kt=2 root initiator.  The external
substep register is exhausted and the root operand remains **UNMEASURED** in
the post-external RK3 stage composition.  These labels are emitted by the
machine gate, not only by this addendum.
