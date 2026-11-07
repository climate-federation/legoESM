# NEMO testcase L2 GYRE phase 3 — round 88 Kaa-scratch receipt

## Outcome

**HELD; no production physics landed.** Decision 39 made NEMO's pre-solve
RK3 Kaa SSH scratch an eligible carried state.  The candidate carried,
restart-loaded, consumed, and rotated that slot, and combined it with the
Round-87 source-ordered W/WZV/ZAD correction.  On NEMO's own kt2 inputs the
complete local chain was bit-exact: Kaa SSH, direct shared W, production-
captured W, and both ZAD faces all had zero unequal active cells.

The composition gate nevertheless rejects the candidate.  Relative to the
Round-85 recorded baseline, first-over-bar expands at kt2 from U/V to
T/S/U/V.  kt2 U/V worsen from `2.7377110452773967e-12` /
`3.284922138989399e-12` to `6.733005735178965e-07` /
`1.3183569256688065e-06`; kt2 T/S leave AT-BAR for
`3.0652394795183113e-03 K` / `4.835710022078388e-03 PSU`.  kt3 T/S worsen
from `1.627497246303733e-04 K` / `6.327735185607253e-06 PSU` to
`1.566544749833554e-02 K` / `4.324733116938262e-03 PSU`.  The frozen
improvement prediction is therefore **REFUTED and retained**.

The required half-bisection also rejects both halves independently.  The
scratch-carry/consumer half and the source-order/removal half each promote
kt2 T/S into first-over-bar and each has 88 cellwise Rule-12 violations.
This is another measured cancelling-pair exposure: local exactness is real,
but neither exact half can be inserted into the present shared program without
exposing a larger compensated error.

Candidate commit `391a8f8b0170` and bisection commits `29ae111faf7e` and
`f2f1765ba4dd` were reverted.  The complete locally proven candidate is
preserved, not applied, as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round88_held_kaa_wzv_bundle.patch`.
The executable production files are byte-identical to branch input
`66212eace1d1`; only the preregistration, held patch, citation mapping, and this
receipt remain.

## Frozen registration and local proof

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round88.md`, committed as
`ed97577a5251` before the candidate was measured.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round88/`.  All scientific
executions used CPU, JAX fp64/libm, production JIT, and clean fail-closed
commit stamps.

The decisive local artifact is `kaa_scratch.json`, SHA-256
`b1428611fa697ddcd44bd13f484aa142b38a28e24409fcea458779a095ce2abd`:

| registered boundary | unequal / active | maximum | verdict |
|---|---:|---:|---|
| carried Kaa SSH | 0 / 600 | 0 | bit-exact |
| direct shared W | 0 / 18,000 | 0 | bit-exact |
| captured stage-one W | 0 / 18,000 | 0 | bit-exact |
| ZAD U after-state | 0 / 17,400 | 0 | bit-exact |
| ZAD V after-state | 0 / 17,100 | 0 | bit-exact |

All three registered one-ULP controls print `PLANT_FIRED` and exit 1:

- Kaa scratch: SHA-256
  `9a46d1de0aad03b5a5839861764d32de8df69830b5237a8612de8ad6c61c80c7`;
- captured W: SHA-256
  `aab44cafd27a675428b282855325abe78635ba03f6faac74c87070e97c77ff4f`;
- downstream ZAD-U: SHA-256
  `7ea4f7b3a1abd8d4257fa38b5d4c96940ffeb6d47693e1cb31e166eda0cf0c7c`.

## Rule-12 adjudication

The full candidate ladder is `gyre_ladder.json`, SHA-256
`e9062e3e5a86635c1a2766c75aa4536852caf693e8ba240c19a8d6f4cc6b103d`.
The shared offline comparison checks 954 certified rows and fails with 88
violations; `gyre_rule12.json` has SHA-256
`f130b04edbc7698443268b6b507e30a2d97dd1b67385b302f9f913b94e8e60fd`.

| arm | kt2 U max | kt2 V max | kt2 T max | kt2 S max | kt3 T max | kt3 S max | first-over-bar | Rule 12 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Round-85 before | `2.7377e-12` | `3.2849e-12` | `1.4211e-14` | `2.1316e-14` | `1.6275e-04` | `6.3277e-06` | kt2 U/V | baseline |
| full exact bundle | `6.7330e-07` | `1.3184e-06` | `3.0652e-03` | `4.8357e-03` | `1.5665e-02` | `4.3247e-03` | kt2 T/S/U/V | FAIL |
| scratch half | `6.7330e-07` | `1.3184e-06` | `3.0652e-03` | `4.8357e-03` | `1.5665e-02` | `4.3247e-03` | kt2 T/S/U/V | FAIL |
| WZV/order half | `6.7330e-07` | `1.3184e-06` | `3.0652e-03` | `4.8357e-03` | `1.5665e-02` | `4.3247e-03` | kt2 T/S/U/V | FAIL |

The bisection reports are `bisect_scratch_rule12.json` (SHA-256
`4abe962a3bcda369972e727e90904fc236c2684f4404e215e7011f2656365983`)
and `bisect_wzv_rule12.json` (SHA-256
`8a0edfdfae5573a5c0ef7295af0cb81dbcae2c2b12490f5cbae12cae416962b3`).
The largest registered cellwise worsening is respectively
`7.101005938018881e13` and `7.100994257679322e13` row-scale oracle ULPs.

No AT-BAR loss is permitted and first-over-bar may not move earlier or gain
fields.  Thus none of the three arms is eligible to reach the day-30 or tank
acceptance lanes.  The recorded day-30 before value remains
`1.2397011295506804e-02 K`; candidate day-30 is **UNMEASURED because the
candidate failed the earlier mandatory ladder**, not carried as an implied
improvement.  LOCK_EXCHANGE and OVERFLOW execute the same WS-RK3 statement,
so they cannot be claimed not to execute it; their candidate lanes were
likewise withheld after GYRE rejection.  DINO uses the MLF program and would
not allocate or consume the new Kaa slot, but the source-round WZV helper was
shared and therefore represented an explicit DINO risk; no DINO candidate
claim is made.  ORCA2 remains **UNMEASURED-with-spec**: an eventual arm must
prove its selected integrator and restart schema, then run its existing
fidelity gate before any cross-card claim.

## Compiled-source basis

The admitted GYRE program calls the external-mode routine before entering RK3
stage one
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:188-201`).  Inside that
routine, the compiled branch reads the already-present Kaa SSH, constructs W,
and consumes it in ZAD
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`).  The active
QCO W recurrence adds the Kaa-minus-Kbb stretch bottom-up
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`).

After the third stage, the compiled program swaps Naa with Nbb and writes the
next scratch as twice new Kbb minus old Kbb
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226`).  The compiled
restart writer stores Kbb as `sshn` and the separate Kaa slot as `ssha`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:176-184`); its reader
loads both on an RK3 continuation
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:354-370`) and initializes
Kaa from Kbb on a cold start
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:395-405`).  These are
the source statements implemented by the held patch; the user-required legoESM
restart guard is stricter than NEMO's MLF fallback and fails by name instead of
silently substituting Kbb.

With the local Kaa/W/ZAD chain exact, the first remaining certified non-bit
statement is the stage-one velocity update: `5.421010862427522e-20` on both
U and V, still AT-BAR, at the compiled vector-branch assignments
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674`).  The
first over-bar boundary is stage three, where the exact bundle enlarges the
residual to `6.733005735178965e-07` U and `1.3183569256688065e-06` V.  The
next round must walk forward from that stage-one update and identify the
compensating statement before retrying any held Kaa/WZV member.

## Review and verification

The required separate read-only Codex review failed before a reviewer model
started.  Its terminal verdict is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued.  No `SHIP` verdict is claimed.  The review artifact SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Before candidate measurement, 96 restart/recipe tests passed and the five
new structural/control tests passed.  The candidate's direct instrument and
tests are retained inside the held patch.  After rejection, production was
restored and checked against input commit `66212eace1d1`; no production/test
diff remains.  The known unrelated RK3-WS test failure was not encountered.
Final focused verification passes 137 tests covering the phase-3 gate,
Round-87 WZV controls, citation machinery, run restarts, and testcase recipes;
the log SHA-256 is
`3ff5529321bc90af75d538291b5d99015b84c1a42f553c147af1ee23adea19d0`.
The receipt citation gate passes all 8/8 compiled-source citations with no
unmapped citation, failed anchor, or global map-audit failure (SHA-256
`cb422541a56219b55f5336c98d5d3544f8312683620fc99dd9e64afd725ba05d`).
Shifting the scratch-rotation citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE` (SHA-256
`2fa63eaeb24ca603a0b56fc3056f6fdd3874e76dc7797f001fd41e85f9794234`).
GitHub CLI authentication remains invalid and no GitHub connector is
available, so this receipt could not be posted to issue #1455; no external
state was mutated.

## OPEN — next round

1. Start from the restored Round-85 production baseline, not any Round-88
   scratch arm.
2. Pre-register a source-order walk from the already-AT-BAR stage-one vector
   update through the stage-two update and into the first stage-three
   over-bar row.  Rank candidate owners by their ability to explain the
   `6.7330e-07` / `1.3184e-06` U/V exposure and the resulting kt2 T/S jump.
3. Use the existing kt1 stage records and the held-patch arms to measure the
   compensation vector.  Do not re-land the Kaa scratch, W source-rounding, or
   ZAD-removal members until the paired statement is locally proven and the
   certified ladder passes.
4. The standing magnitude targets remain kt3 T `1.627497246303733e-04 K` and
   day-30 T `1.2397011295506804e-02 K`; no day-30 candidate measurement from
   this rejected arm may be inferred.
5. Once a composed candidate passes GYRE Rule 12, run both WS-RK3 tanks, the
   DINO shared-helper check, and ORCA2's stated specification before landing.
