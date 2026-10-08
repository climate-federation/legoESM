# ORCA2 round 171 — kt=8 RHS observer hold

Date: 2026-10-08. Base `cea3bdd83`; preregistration `8734a8ab6` with
instrument addenda through `a709d1e47`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round171/`.
Verdict: **HELD**. The operator's round-170 RHS record is admissible, but every
legoesm component observer changed the executable it observed. No component
value is citable, no HPG/LDF/VOR/KEG/ZAD owner is named, and no physics or
configuration changed.

Every ORCA2 statement in this receipt is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity and zero sea surface.
No given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple are
unchanged.

## Admitted record and source order

The round-170 admission was rerun from the record itself. Both self-describing
rank files parse, cover 148 x 180 exactly once, contain the ten registered
accumulators, and all 20 kt=1..10 terminal restarts are byte-identical to the
round-169 baseline. The header, field-name, field-dimension, truncation,
swapped-rank and restart-byte plants all refuse.

The compiled rung-0 vector program calls HPG, LDF and VOR in that order at
`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-157`, then calls KEG and
ZAD at `ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:168-179`. Every NEMO
accumulator is finite. Across the assembled owned domain, the maximum absolute
U value is `8.452807413638579e-4 m s^-2` and the maximum absolute V value is
`9.310706505661398e-4 m s^-2` at every recorded boundary. Thus R171-P1 and
R171-P4 are **CONFIRMED**.

## Observer refusals

Four progressively narrower observers were attempted, and none emitted a
scientific report:

| observer | terminal result |
|---|---|
| full-step host callback | moved at least one completed kt=1..7 state bit |
| barotropic-prefix host callback | moved at least one kt=1..7 post-barotropic bit |
| bundled return-only component trace | moved at least one kt=1..7 post-barotropic bit |
| one static return-only component pair per executable | exact at kt=1; moved the observed boundary at kt=2 |

The last discriminator names the kt=2 movement. LDF and KEG move the completed
V RHS; VOR moves the completed U RHS; all three move ssh, u and v. The exact
keys are retained in `instrument_refusal.json` and
`rhs_accumulator_walk.log`. Returning HPG or ZAD alone did not appear in that
first failing key set, but the frozen predicate requires all five selectors;
this is not evidence that either returned component is faithful.

R171-P2 is **REFUTED and retained**. R171-P3 is **UNMEASURED**: the prediction
that VOR is the first finite-to-explosive boundary cannot be classified from a
perturbing observer. R171-P5 is **CONFIRMED**: all experimental package code
was removed, the failed measurement path now refuses explicitly, and the net
`packages/` diff against `cea3bdd83` is empty. The round-170 completed U RHS
maximum `1.5835360371918837e51 m s^-2` therefore remains the first unresolved
upstream boundary, not an attributed statement.

## Controls and validation

The round gate retains rank-placement, source-order, observer-closure,
explosive-classification and passivity plants plus a one-ULP known-answer
control. The retired measurement entry point has its own unit test and cannot
silently print a result from any rejected observer. Focused tests pass 9/9.
The citation gate passes both cited spans with zero failures or unmapped
citations; shifting the HPG/LDF/VOR span by two lines makes the plant fail.
The cumulative citation gate also passes.

The single required `tests/ocean/fidelity -n 12` invocation collected 2,744
tests and reached 99%, then stopped emitting with no pytest process left to
poll. Its retained log contains 2,723 terminal outcomes: 2,712 passed, seven
skipped and four failed. The four failures are the registered pre-existing
reds from rounds 166-170: the unscoped dirty-tree escape ratchet, unstamped
report emitters, the moved GYRE spread record and SI3 scalar-math provenance.
No round-171 test failed; the battery was not run a second time.

The separate `codex exec --sandbox read-only` review attempt returned
**independent review unavailable in-sandbox** before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.

Because the final tree has no model or configuration diff, the GYRE trajectory
and year, both ORCA2 ladders, DINO and tank trajectories cannot move; their
shared-model rerun predicate is not invoked. No configuration decision and no
NEMO acquisition are needed.

ASKED choices: continue the independent rung-0 source walk. UNASKED choices:
empty.

## OPEN

1. Build a passive legoESM-side RHS boundary instrument whose observed
   executable reproduces the unobserved completed RHS and post-barotropic
   state bit-for-bit at kt=1..8. A callback or extra JAX return is not an
   admissible design on this explosive trajectory.
2. Only after that control passes, compare HPG, LDF, VOR, KEG and ZAD in the
   compiled order above. Keep R171-P3 frozen and retain its eventual
   CONFIRMED or REFUTED disposition.
3. Keep the V halo/mask unit and the round-169 1/41 face-average residual as
   separate HELD debts. Do not land the adverse complete halo unit.
