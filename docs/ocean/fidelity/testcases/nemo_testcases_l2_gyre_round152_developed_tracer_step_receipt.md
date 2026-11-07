# NEMO testcase L2 GYRE round 152 — developed tracer-step ranking

Date: 2026-09-22

Status: **HELD**.  No production physics, configuration, or carried state
changed.  From NEMO's admitted day-180 entry, the complete production-JIT
temperature step was scored through free-surface geometry, advection, surface
forcing, shortwave, lateral diffusion, and the implicit vertical solve.  The
largest isolated one-step mismatch is vertical diffusion at
`2.1834089000437955e-5 K` RMS, followed by lateral diffusion at
`1.0655217366755294e-5 K`.  The first non-bit boundary remains inherited
free-surface geometry; the first directly scored process call is the active
FCT advection call at `1.0974591404090626e-8 K` RMS.

The record does not contain NEMO's developed-state FCT faces, coefficients, or
limiter activity.  Therefore the FCT call boundary is named, but no internal
FCT statement is claimed as owner and no candidate is eligible for the month
or year gates.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round152.md` at
`92721047e`.  The authoritative measurement was emitted from clean commit
`e3fea8b1aa9be4eb4769f143442ff14dda275a52`.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round152/`.

## Compiled program and first statement boundary

The compiled stage program calls and records advection, surface forcing,
shortwave, lateral diffusion, and vertical diffusion in that order at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`,
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`, and
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.
The first inherited non-bit statement is the T-point QCO write
`pr3t = pssh*r1_ht_0` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90:237-257`.

For the first process call, the record's resolved output selects FCT2 with
optimized implicit treatment at
`round123/oracle_process_budget/ocean.output:756-764`, and the compiled
dispatcher enters `CALL tra_adv_fct` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv.f90:358-364`.
This is the first directly scored active-branch statement boundary.  It is not
an internal ownership claim: an input transport or any earlier FCT write may
already differ, and this record cannot discriminate them.

The compiled implicit solver forms the thickness-weighted tracer content and
then executes its forward/back recurrences at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560` and
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578`.  Thus the
large vertical row is the complete response of this solve to its accumulated
input plus its matrix; it is not automatically a vertical-closure owner.

## Entry, passivity, and complete boundary table

The daily-record gate admitted all 360 daily boundaries and all 12 monthly
overlaps.  Every restart-backed live state leaf was mapped, the entry
temperature differs from NEMO's process `Tbb` in `0 / 18,000` wet cells, and
the observer changes `0` bytes of the separately compiled ordinary returned
state.

| compiled boundary | cumulative unequal / 18,000 | cumulative max abs (K) | cumulative RMS (K) |
|---|---:|---:|---:|
| geometry | 18,000 | `2.843236757144041e-11` | `6.281725340556424e-12` |
| advection | 17,906 | `4.53254078e-7` | `1.09742738e-8` |
| surface boundary | 17,906 | `4.53254078e-7` | `1.09742739e-8` |
| shortwave | 17,908 | `2.46803727e-6` | `2.51643679e-7` |
| lateral diffusion | 18,000 | `3.56214152e-4` | `1.06577317e-5` |
| vertical diffusion | 18,000 | `7.58804000e-4` | `1.92439497e-5` |

The geometry split reproduces exact step-entry `q_Kbb`, while all 600 wet
columns differ in `q_Kmm` and `q_Kaa`; their maxima are
`6.540323838066797e-13` and `1.307953745310897e-12`.  The model therefore
enters the actual process calls with inherited geometry debt, which the
isolated contribution table below keeps separate.

## One-step magnitude ranking

Each row is the model-minus-NEMO difference of that isolated one-step
temperature contribution, not the cumulative state.  Effective tendency is
the contribution RMS divided by the compiled `14400 s` stage timestep.

| rank | isolated contribution | unequal / 18,000 | max abs (K) | RMS (K) | effective RMS (K/s) |
|---:|---|---:|---:|---:|---:|
| 1 | vertical diffusion | 17,994 | `6.652851977140273e-4` | `2.1834089000437955e-5` | `1.516256180585969e-9` |
| 2 | lateral diffusion | 18,000 | `3.5621425651477523e-4` | `1.0655217366755294e-5` | `7.399456504691177e-10` |
| 3 | shortwave | 9,701 | `2.4680393178044824e-6` | `2.513690760498398e-7` | `1.745618583679443e-11` |
| 4 | advection | 17,999 | `4.532680648594578e-7` | `1.0974591404090626e-8` | `7.62124403061849e-13` |
| 5 | geometry | 18,000 | `2.843236757144041e-11` | `6.281725340556424e-12` | `4.362309264275294e-16` |
| 6 | surface boundary | 578 | `9.237055564881302e-14` | `4.6577410196032515e-15` | `3.2345423747244802e-19` |

The preregistered magnitude order is **CONFIRMED**.  The table is a one-step
response ranking, not a day-240 causal decomposition.  Rounds 124--128 already
showed that vertical diffusion responds to the coupled temperature/EVD state;
this round does not retract that attribution limit or rename the response as
the year owner.

## Active branches and claim boundary

| branch family | NEMO evidence | legoESM on NEMO entry | disposition |
|---|---|---:|---|
| FCT nonosc limiter | `UNMEASURED` by this record | 502 active wet cells | active locally; internal NEMO statement OPEN |
| EVD replacement | 1,123 projected active cells | 1,123; zero interface disagreements | active and selection agrees |
| TKE diffusivity floor | 9,973 interfaces | 9,973 interfaces | active and partial NEMO row agrees |
| TKE viscosity / energy floors | `UNMEASURED` by NEMO record | 9,973 / 15,058 interfaces | model activity only |

The compiled limiter calculates its beta branches and later applies the
sign-selected coefficient at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:849-876` and
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:905-931`.
Spatial overlap with the model's 502 active cells does not prove that NEMO
took the same branch, so neither range is promoted to the first internal
non-bit statement.

## Prediction ledger

1. **CONFIRMED:** entry T is BIT and the production observer is passive.
2. **CONFIRMED:** geometry remains non-bit; among actual process calls,
   advection is first non-bit.
3. **CONFIRMED:** the complete isolated-contribution ranking is vertical
   diffusion, lateral diffusion, shortwave, advection, geometry, surface.
4. **CONFIRMED with the preregistered provenance limit:** all three model-side
   branch families are active; NEMO FCT activity remains unmeasured.
5. **CONFIRMED stopping rule:** no internal FCT operand sequence exists for
   developed step 1081, so no source-exact statement or candidate is claimed.

The legacy `predictions.first_non_bit_is_advection=false` field in the JSON is
Round 136's full-boundary prediction and remains false because geometry comes
first.  Round 152 separately registers
`first_non_bit_process_call=advection`; it does not relabel the inherited row.

## Campaign surfaces and required headlines

No numerical candidate exists, so no trajectory row moved and no new ladder,
month, or year arm was run.  The current admitted Round-149/150 values are
reported as inherited, not remeasured:

| headline | inherited value | disposition |
|---|---:|---|
| kt2 T max abs | `1.4210854715202004e-14 K` | AT-BAR |
| kt2 S max abs | `2.1316282072803006e-14 g/kg` | AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12 m/s` | first-over-bar |
| kt2 V RMS | `3.2849219221489645e-12 m/s` | first-over-bar |
| kt3 T RMS | `8.659371033559182e-7 K` | DEBT |
| kt3 S RMS | `7.027288972949464e-8 g/kg` | DEBT |
| day-30 T3D RMS | `6.888193513796918e-5 K` | DEBT |
| day-240 T3D RMS | `1.644671864406711e-2 K` | DEBT |
| day-360 T3D RMS | `1.1223450861560211e-2 K` | DEBT |

GYRE alone was observed by a private write-only gate.  Shared production code
is unchanged, so DINO, LOCK_EXCHANGE, OVERFLOW, and the generic NEMO-GYRE card
execute no changed statement.  ORCA2 is **UNMEASURED-WITH-SPEC**: bridge an
ORCA2 developed entry through its own production step and score this same
complete boundary registry before transferring the ranking.

## Controls, review, and verification

At final instrument commit `e3fea8b1aa9b`, the missing-day,
missing-process-row, missing-branch, and missing-ranking-row plants each print
`STATUS PLANT-FIRED` and exit 1.  The consumed-temperature one-ULP plant moves
one cell in all six registered boundaries, prints `STATUS PLANT-FIRED`, and
exits 1.  The final authoritative artifact SHA-256 values are:

| artifact | SHA-256 |
|---|---|
| `developed_step_walk_final.json` | `36a04a4311e48e57df8bd841ca579f2197f0cef5cdea29bb5b24ae203b871547` |
| `developed_branch_maps_e3fea8b1aa9b.npz` | `02eec865afda3680e6f8dd1be0e6eff8a15715ed82d48052b9d283a8446c6e10` |
| `daily_record_audit_final.json` | `ac5dc516f2c781104b52535be06562d628920ce089a7f0cbff22d62f2fa44659` |
| `entry-temperature-ulp_plant_final.json` | `9b0215374171088ff3164209b980e927360df5351ea16e7924ba0a682f5e7fca` |

The required read-only Codex review was attempted after measurement.  It did
not start a reviewer and emitted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
codex_exit=1
```

Independent review was unavailable in-sandbox.  It emitted neither `SHIP` nor
`DO NOT SHIP`; no approval is fabricated.

The clean-tree receipt citation gate reports `PASS`: 10 citations found, zero
unmapped, zero failures, and zero failing map entries.  Shifting the compiled
FCT dispatch citation by two lines reports `FAIL` and exits 1.

Focused test summaries are:

* developed process, citation, and stamp files: `1 failed, 56 passed in
  35.24s`; the only failure is the registered pre-existing worktree-stamp
  ratchet, with exactly the Round-146 RHS-family and Round-50 LDF-association
  offenders;
* live-geometry guard, prognostic barotropic state, and complete generic NEMO
  recipe files: `35 passed in 886.24s (0:14:46)`;
* the changed year-owner file alone before the evidence run: `30 passed, 1
  skipped in 5.41s`.

No combined full-ocean tree was launched: this round changes no production
path and already ran the requested production-JIT scientific step plus all
three prior-landing production-path files.  No new failure occurs in a changed
path.

## OPEN — round 153

Stay on developed step 1081 and do not year-score a response bucket.  The first
directly scored active branch is FCT advection, but the admitted record begins
only at its completed `Krhs`.  Reuse the Round-111 self-describing FCT writer
layout for this day-180 entry and preregister a passive operator-run acquisition
that records, in compiled order, the stage-3 transports, first upwind faces,
first divergence, midpoint tracer, limiter coefficients/activity, averaged
faces, final divergence, divisor, and direct `Krhs` write.  Compare the new
run's restart and every inherited process boundary before admitting it.  The
first internal non-bit row then owns the next statement walk; only a
source-exact production-JIT candidate proceeds to Decisions 43 and 45.
