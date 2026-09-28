# Preregistration — round 170, the MPAS TKE surface-boundary red left by the merge

Written and committed BEFORE any source or test edit.  The read-only
diagnosis described below (probes that import the model and print numbers,
changing nothing) ran first, as the round's brief requires
("diagnose before patching"); every edit comes after this file is committed.

## What is red, and the decision that governs it

`tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_eice3_quarter_ice_maps_to_full_attenuation`
is red on the merged lane tip `d3631f884`.  Decision 60 (standing brief note
AX) keeps `main`'s NEMO-faithful surface boundary on the mesh-based OMIP card
— the surface turbulent energy is held at the z=0 water surface, not one
level down — and asks for the lane side to be reconciled to it.  The GYRE and
ORCA2 certified numbers must not move.

## The two candidate mechanisms, and the measurement that discriminates them

**M1 — a closure defect.**  The lane's turbulence closure answers the z=0
boundary differently from `main`'s, so under that boundary the lane flattens
the mixing coefficients onto their background floors while `main` does not.
This is the mechanism the merge receipt's OPEN entry assumed, and it is what
the brief's "name the diverging statement and fix it to NEMO's" presumes.

**M2 — a collision between two versions of the same test.**  Both parents
changed this test.  The union merge kept the lane's assertion and `main`'s
card value, and the lane's assertion is only satisfiable when the boundary is
placed one level too deep.

**Discriminator (pre-registered):** run the SAME read-only probe — the mesh
card, the same fixture, the same forcing, the ice modes the test uses — in a
clean worktree of EACH merge parent, and compare the printed mixing
coefficients.

- M1 is confirmed if the two parents print DIFFERENT profiles under the z=0
  boundary.
- M1 is REFUTED, and M2 is the mechanism, if the two parents print the SAME
  profile under the z=0 boundary while the two parents' copies of the test
  assert different things.

Whichever wins, the receipt records the numbers.  If M1 wins, the fix is the
diverging closure statement, transcribed from the compiled `zdftke.f90` and
cited by line.  If M2 wins, NO closure statement may be edited — inventing
one would be a fit to the test — and the reconciliation is to resolve the
test-vs-test collision the way Decision 60 resolved the card, while keeping
every protection the lane's stricter assertion was added to provide, asserted
where the background floor cannot mask it.

## What must be true at the end

| invariant | how it is checked |
|---|---|
| the red test is green | `pytest tests/ocean/unit/test_mpas_tke.py` quoted in full |
| the fix is not vacuous | the same test shown RED with the change reverted |
| GYRE certified ladder, 70 rows, 0 moved | digest `cf06a8fc7d0e90f2`; day 30 `6.572574374770603e-05` K, day 240 `1.644836070117868e-02` K, day 360 `1.1225660018551306e-02` K.  A diff that touches no file the GYRE run imports cannot move them, and the file list is quoted as the proof |
| the other cards' gates unchanged | the same five files the merge round ran: `tests/ocean/unit/test_dino_experiment.py`, `tests/ocean/fidelity/test_nemo_testcase_l1_tanks_round33_zdf_rule12.py`, `tests/ocean/fidelity/test_nemo_testcase_lock_slow_forcing_owner.py`, `tests/ocean/fidelity/test_nemo_testcase_overflow_barotropic_gate.py`, `tests/ocean/fidelity/test_nemo_testcase_round34_tank_zdf_removal.py` — 170 passed |
| the GYRE push gate | the six files in `phase3/autopilot/autopilot_max.sh`, binary64 on, lane import path |
| the citation gate | `status PASS`, every plant fired, quoted |
| ORCA2 does not execute the changed line | stated from its resolved configuration |

## Abort rule

If any certified GYRE row moves at all, restore the tree, report which row,
and HOLD.  If neither M1 nor M2 explains the numbers, report that and HOLD
rather than edit anything.
