# NEMO testcase L2 GYRE round 149 — developed LDF live geometry

Date: 2026-09-22

Status: **LANDED**.  The RK3-WS NEMO identity path now supplies the live QCO
T/U/V/F thicknesses, at NEMO's Kbb/Kmm time levels, to the divergence/curl
lateral-momentum diffusion.  The day-180 production-JIT LDF U/V terms become
BIT, day-30 T RMS improves by `2.2379740289906955e-8 K`, day-240 by
`2.158910828234384e-8 K`, and day-360 by `1.203862834524283e-7 K`.

The two frozen preregistrations are
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round149.md` at
`1fb64d6ca` and
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round149_ldf_live_geometry_candidate.md`
at `262c98458`.  Candidate physics was measured clean at `94b7761bc`; all
evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round149/`.

## Round-148 record admission

The acquisition's exit 71 was a gate defect, not an active instrument.  The
old check compared unowned diagnostic halos byte-for-byte.  The ownership-aware
admission finds both completed three-dimensional RHS arrays, both restarts, and
all five other inherited records BIT.  Only excluded two-dimensional halos
move: `cd_u` 189 cells (max `0.1`), `cd_v` 150
(`8.192533274826719e-4`), wind-U 120 (`21.353930564131815`), and wind-V 120
(`26.870429785765168`); every owned cell in those rows is BIT.  The direct
pre/post U/V boundaries are BIT to the same-run family record.  The admitted
record is 4,717,612 bytes with SHA-256
`4596cf743d1c683354f0361381b502d108f334c86a6e387c34b7a41d04ccf033`.

This classification follows the compiled writer: it allocates and writes the
direct operands at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:111-140`, copies
only the assigned curl/divergence extents and then writes the post accumulators
at `GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:142-197`, while
the inherited parent record writes its full arrays at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:204-217`.  All eight
admission plants exit nonzero with a named marker.  The stored binary and
manifest were admitted in place; NEMO was not rebuilt or rerun.

## First non-bit statement

The existing Round-50 walker was extended, not duplicated.  Starting from
NEMO's admitted day-180 state through the production JIT step, `ahmf`, all
metrics, U/V, and the T-point Kbb thickness are BIT.  The first non-bit factor
is the live F-point thickness in NEMO's curl statement: 16,530/16,530 wet
cells, max `3.872632379170682e-3 m`.  The subsequent Kbb U/V thicknesses differ
in 17,400/17,100 cells, maxima `3.210403464038336e-3` /
`3.5332818579263403e-3 m`.

The compiled statement multiplies curl by live `e3f` and divergence by live
Kbb T/U/V thickness, then divides the final U/V terms by the live Kmm U/V
thickness at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176`.
NEMO constructs the exact U/V/F free-surface ratios, including the F-point
association, at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:256-286`.
The external stage calls the operator with Kbb=Kmm at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:158-164`; stage 3 calls
it with distinct Kbb/Kmm at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90:704-739`.

The landed transcription routes the step-entry Kbb tuple at
`ocean_model_latlon_cgrid.py:5328-5363` and the stage Kmm F/U/V tuple at
`ocean_model_latlon_cgrid.py:5863-5922`.  It adds no selector, configuration,
state, stabiliser, or alternate formula.

## Local production proof

| production-JIT row | before unequal / cells | before max | after unequal / cells | after max |
|---|---:|---:|---:|---:|
| live F thickness | 16,530 / 16,530 | `3.872632379170682e-3` | 0 / 16,530 | 0 |
| live U Kbb thickness | 17,400 / 17,400 | `3.210403464038336e-3` | 0 / 17,400 | 0 |
| live V Kbb thickness | 17,100 / 17,100 | `3.5332818579263403e-3` | 0 / 17,100 | 0 |
| isolated LDF U | 17,400 / 17,400 | `8.227285494346176e-12` | 0 / 17,400 | 0 |
| isolated LDF V | 17,100 / 17,100 | `1.2612924221587602e-11` | 0 / 17,100 | 0 |
| full accumulator after LDF U | 17,400 / 17,400 | `1.0690316927004515e-5` | 17,400 / 17,400 | `1.0690316464779957e-5` |
| full accumulator after LDF V | 17,100 / 17,100 | `9.419267861525294e-6` | 17,100 / 17,100 | `9.4192651923183e-6` |

Thus the operator term closes exactly under production JIT.  The frozen
prediction that the production post-LDF accumulator itself would become BIT is
**REFUTED**: it still contains upstream RHS debt.  This does not cross the
preregistered local veto, which was any non-bit row *before* that accumulator.
The literal replay's direct post-LDF rows are BIT, separating that inherited
debt from this operator.  A 65,536-ULP entry-SSH plant moves 57 consumed LDF-U
cells, prints `STATUS PLANT-FIRED`, and exits 1.

## Certified ladder

| headline | exact starting tip | candidate | result |
|---|---:|---:|---|
| kt2 T max abs | `1.4210854715202004e-14` | `1.4210854715202004e-14` | unchanged, AT-BAR |
| kt2 S max abs | `2.1316282072803006e-14` | `2.1316282072803006e-14` | unchanged, AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12` | `2.7377110452773967e-12` | unchanged |
| kt2 V RMS | `3.2849219221489645e-12` | `3.2849219221489645e-12` | unchanged |
| kt3 T RMS | `8.659373840202989e-7` | `8.659371033559182e-7` | improved |
| kt3 S RMS | `7.027291104577671e-8` | `7.027288972949464e-8` | improved |

The first-over-bar remains kt2 U/V and no kt1 AT-BAR row leaves the bar.  All
58 moved rows are registered exactly in `round149/moved_rows.tsv`: kt2
`after.uu_b/vv_b`; kt3 through kt10 `after.uu_b/vv_b`; and kt3 through kt10
each of `before.T/S/u/v/ssh`.  Decision 43 permits later-row worsening; the
full improved/worsened cell census is in `round149/ladder_comparison.json`.

## Month and year gate

| day | before T3D RMS (K) | after T3D RMS (K) | after - before |
|---:|---:|---:|---:|
| 30 | `6.890431487825909e-5` | `6.888193513796918e-5` | `-2.2379740289906955e-8` |
| 60 | `1.932998392040720e-4` | `1.9325579916772937e-4` | `-4.4040036342637414e-8` |
| 90 | `1.8645017233187848e-3` | `1.8644545401142745e-3` | `-4.7183204510354085e-8` |
| 120 | `1.0501248765795105e-3` | `1.0500500515487367e-3` | `-7.482503077381464e-8` |
| 180 | `3.5805502932896326e-3` | `3.5805134076684307e-3` | `-3.688562120194097e-8` |
| 240 | `1.6446740233175394e-2` | `1.644671864406711e-2` | `-2.158910828234384e-8` |
| 300 | `1.359740243654708e-2` | `1.359738174115693e-2` | `-2.0695390149857995e-8` |
| 360 | `1.1223571247843664e-2` | `1.1223450861560211e-2` | `-1.203862834524283e-7` |

The exact starting-tip 30-day run reproduces the immutable baseline bit for
bit.  The year control is the most recent same-production base run from round
145; its eight numbers equal the round-130 immutable arm.  The Decision-43/45
gate reports PASS: all eight year checkpoints improve, the month/year day-30
rows agree, all 58 ladder rows are registered, first-over-bar is not earlier,
and kt1 retains the bar.  Its day-240-worse and missing-registry plants print
`STATUS PLANT-FIRED` and exit 1.

These candidate artifacts are now the immutable before arms for the next
landing: `phase3/round149/candidate_ladder_certified.json`,
`phase3/round149/candidate_month_gap.json` with its `candidate_month/` member,
and `phase3/round149/candidate_year_gap.json` with its `candidate_year/`
member.  No older arm may be used as the next landing's before trajectory.

## Blast radius, review, and evidence

The execution census is derived from every resolved recipe.  The route
requires RK3-WS momentum plus `nemo_div_curl` and `nemo_e3`; only GYRE executes
it.  The generic NEMO-GYRE, LOCK_EXCHANGE, and OVERFLOW recipes use the vector
Laplacian; both DINO cards use Euler momentum.  The complete DINO real-card
suite reports `128 passed, 9 warnings` on both the exact base and candidate.
ORCA2 remains **UNMEASURED-WITH-SPEC**: its future ocean identity card must
select this operator and weighting before this route can execute.

The required read-only Codex review was attempted and emitted verbatim:
`Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`.  It produced no SHIP/DO NOT SHIP verdict; this is
**independent review unavailable in-sandbox**, not approval.

| evidence | SHA-256 |
|---|---|
| `candidate_local.json` | `297fa122e623931fb8892c0d998ed13a92ef533c532a683c4b1ec3c5b31c5278` |
| `candidate_local_plant.json` | `edc8699ef85578aa3d208453e03f7949071377b8b6c7a231bc6b7919203a3764` |
| `ladder_comparison.json` | `951214303558595f50fd1667f52342560445c7bc47aed6bd2aea935c5aeaa7b5` |
| `base_month_gap.json` | `8e725221ab6526cd8cafdeda5e65edc74381d9f3715510bec2283148b1dad5a2` |
| `candidate_month_gap.json` | `2d4eb45ed7ec041a5c57f50294666236bdadca44a804f5bd160b151462c7d8c0` |
| `candidate_year_gap.json` | `afbbe90026b5f8798bb4fd20d5f3b51b40df958e722da5bc24efb9c261cc0418` |
| `decision43_45.json` | `d8a87286010db6e8175d83ccbc3fc0c7fceeb77dfccf6fa161d435ce20ecdd12` |
| `moved_rows.tsv` | `de3b39aee07f8f84288bda763287e5efaa4eb4725dc5acc4eb7aa07459a409bb` |

The final candidate-sensitive four-file suite reports exactly `60 passed in
24.86s`.  The DINO real-card base and candidate suites report respectively
`128 passed, 9 warnings in 108.56s` and `128 passed, 9 warnings in 111.82s`.

The required combined `tests/ocean/fidelity tests/ocean/unit -n 12` run reached
98%, then stopped emitting progress after six JAX/XLA worker aborts.  It was
bounded with exit 130 and emitted **no terminal pytest summary**, so it is not
represented as a pass and cannot supply a complete failure-ID diff against the
frozen known-red inventory.  The six traceback nodes were rerun in fresh serial
processes and report, in order, `1 passed in 8.82s`, `1 passed in 13.94s`, `1
passed in 8.30s`, `1 passed in 13.24s`, `1 passed in 31.86s`, and `1 passed in
20.78s`.  They cover distributed barotropic PCG differentiation, additive
momentum-friction differentiation, stored mass flux, the z-star PGF self-test,
model-step differentiation, and first-step TKE advection.  The incomplete
large-suite log is retained as evidence, not promoted to a green result.

The receipt citation gate reports PASS with 9 citations, zero unmapped, zero
failures, and an empty map audit.  The two preregistrations independently PASS
with 3 and 4 citations.  Shifting the compiled LDF citation by two lines
reports FAIL and exits 1.

## OPEN — round 150

The live LDF operator is now source-exact at the admitted developed state, but
its year benefit is only `2.16e-8 K` at the `1.64e-2 K` day-240 gap.  Re-run
the existing Round-146 day-180 family ranking from this landed tip.  Among the
remaining HPG, VOR, KEG, and ZAD families, identify the largest non-bit incoming
slow-forcing term under production JIT, name its first compiled non-bit
statement, and year-score only a source-exact candidate.  Do not return to the
closed LDF term or to rest-state last-bit walks.  No acquisition is currently
needed.
