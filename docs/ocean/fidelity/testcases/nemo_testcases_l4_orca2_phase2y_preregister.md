# NEMO testcase Lane 4 — ORCA2 Phase-2y preregistration

Date: 2026-09-06

Parent: `bcccc890e31c`

Status: **PREREGISTERED BEFORE LOCAL ADMISSION, ZPELC SCORING, OR TKE WALK.**

## P2Y-1 — ORCA1-ice variant admission

Treat `variant_orca1ice_phase2x_a_10step_np2` and twin B as a new oracle
family because their ice namelist differs from `VARIANT_ORACLE_V2`.  Admission
requires: both launchers report zero and `RUN DONE`; `time.step=10`; no NEMO
error; every stream decodes to exact EOF from its header; all schema plants
exit nonzero; A/B oracle records are 116/116 byte-identical; restart/history
payloads and ordinary outputs satisfy the existing identity rules.  Pin A as
`VARIANT_ORACLE_ORCA1ICE` and B only as its reproducibility witness.  Keep the
Phase-2w initialization failures rejected and retained.

Compare every common record to the pinned `VARIANT_ORACLE_V2` root in NEMO
execution order.  Report the first ocean record that differs.  A difference
after the first SI3 call is expected from the deliberately different ice
state/operators; it is evidence that the roots must remain separate, not a
WRITE-only failure.

The receipt must inventory every SI3 handoff stream by path, SHA-256, magic,
frame order, field order/extents, time level, and category/layer axes, with
capture-site citations.

## P2Y-2 — `zpelc` operand discriminator

At kt=2, score the three NEMO operands of `zdftke.F90:339-345` independently
on the rank-0 owned W cells:

1. `rn2b`, copied from `rn2` by `stprk3.F90:160` after the EOS/bn2 chain;
2. live `gdepw(:,:,:,Kmm)`;
3. live `e3w(:,:,:,Kmm)`.

Use production JIT, fp64, explicit scalar-libm policy, cellwise equality and
row-scale ULP.  Each row is AT_BAR only at `0 / n`; each has a binding
one-variable plant.  Any non-exact operand is Lane-4 owned (EOS-80/bn2,
partial-cell depth or QCO geometry) and must be walked before the recurrence.
Only three exact operand rows can promote the `zpelc` arithmetic handoff to
`CONFIRMED_GYRE_OWNER_SHARED_TKE`.

## P2Y-3 — ordered continuation

If the three operands and source-literal recurrence close, continue through
the recorded TKE statements in NEMO order: Langmuir `en` source; matrix build;
tridiagonal solve; `en_etau`; mixing-length branch; `avm/avt`; `dissl`.  Then
enter EVD and IWM only if TKE closes.  Stop at the first non-bit statement and
assign its owner: shared arithmetic to GYRE with the reproducer, ORCA2 forcing,
partial-cell or selector operands to Lane 4.  Never modify shared TKE here.

## ASKED / UNASKED

| action | classification | preregistered disposition |
|---|---|---|
| admit successful user-shell twins | ASKED | exact gates only; distinct ORCA1-ice root family |
| pin twin B | UNASKED | retain as witness, do not pin as a second root |
| score all three `zpelc` operands | ASKED | 0 / n individually, then ownership decision |
| infer shared ownership with unscored operands | forbidden | do not do it |
| change shared TKE/EVD arithmetic | Lane-4-forbidden | register and hand to GYRE |
| preserve `nn_iceini_file=0` disclosure | ASKED | explicit; assess whether inert for exact-input certification |
| edit shipped NEMO, delete, run MPI in sandbox, or push | forbidden | none |
