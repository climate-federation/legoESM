# NEMO testcase L2 GYRE phase 3 — round 126 vertical sub-owner receipt

Date: 2026-09-20

Incoming tip: `4cac617cd928007506f2de7ccb098f87e04204d1`

Status: **HELD — the day-180-to-240 vertical carry closes to
`+2.4168271578053416e-2 K`; its frozen source-order telescope is dominated by
a cancelling `avt`/EVD coefficient response (`+7.158989942001176e-1 K`) and
the live column it acts on (`-6.975158045602758e-1 K`).  Driving the production
closure from NEMO's entry state removes essentially 100 percent of the `avt`
RMS error and all EVD-trigger disagreement, so mixing responds to upstream
state rather than owning the error intrinsically.  No physics lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round126/`

## Outcome first

The acquired Round-125 NEMO record is admitted: 362 exact frames, including
all steps 1081--1440, both passive restart hashes, the Round-123 process
trajectory, and all eight compiled-arithmetic calibration rows.  The matching
independent legoESM run traversed the production CPU/fp64 step from rest and
recorded 360 vertical solves.  The write-only observer changed zero carried
state bytes and reproduced every inherited Round-124 process boundary.

The source-order sub-owner budget closes the inherited vertical-diffusion
carry to `8.569533971325427e-16 K`.  It is not a collection of independently
additive causal effects: changing the live column before changing its mixing
coefficient creates a `58.7216564845` absolute-carry cancellation ratio.
Under the preregistered order, the two leading rows are nevertheless
mechanically unambiguous:

- TKE plus enhanced-diffusion heat coefficient: `+7.158989942001176e-1 K`;
- live temperature column/gradient acted on: `-6.975158045602758e-1 K`;
- isoneutral vertical coefficient: `+5.785612304013637e-3 K`;
- all geometry, matrix association, solve association and interaction rows
  together are below `5.31e-7 K` in signed carry.

The large coefficient row is a **response**, not an intrinsic TKE-closure
defect.  At days 180 and 210 the model-own heat-diffusivity RMS errors are
`2.93525982529855` and `2.5136487632185793 m2/s`, with maxima near the active
`100 m2/s` EVD replacement.  Re-running the full production closure under
`jax.jit` from NEMO's recorded entry reduces those RMS errors to
`8.260934106155803e-15` and `2.4586546006968085e-17 m2/s`.  EVD-trigger mask
disagreements fall from 15 and 11 cells to zero at both checkpoints.

This establishes **UPSTREAM-STATE-RESPONSE** and sends the next magnitude
walk upstream into the state/stratification that crosses EVD's threshold,
not into TKE arithmetic.  It does not establish the narrower jet-position
story: the preregistered western upper-100-m jet-centroid vectors reverse
direction between days 180 and 210.

No production physics, card, configuration, restart schema, carried state or
stabilizer changes.  Decision-43/45 landing gates therefore do not run and the
immutable before arms do not move.

## Frozen preregistration ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round126.md`, committed
before measurement as full commit
`a54475a0cd4cb8b6acab1d6c00922bc6ed4b7e95`.

- **P0 CONFIRMED.** The record has the frozen frame count, byte layout,
  producer and restart hashes.  Every trajectory/calibration row is bit-exact.
  Wrong-stamp, truncation, consumed-matrix-ULP and trajectory-ULP controls
  each printed `STATUS PLANT-FIRED` and exited 1.
- **P1 INITIAL PREDICTION REFUTED; repaired instrument CONFIRMED.** The first
  completed trace changed 78 inherited `Bpre` cells because adding the larger
  observer to the same returned graph changed production-JIT fusion.  It was
  refused and never scored.  Separate process and vertical observer graphs
  then reproduced all inherited process rows and all carried-state bytes.
  Both final trace plants exit 1, and the production K-effect control reaches
  the matrix and solved temperature without moving an upstream row.
- **P2 MAGNITUDE PREDICTIONS REFUTED.** The prediction
  `avt in [+1.5e-2,+3.5e-2] K` measured `+7.158989942001176e-1 K`; live-column
  `abs <=1.0e-2 K` measured `6.975158045602758e-1 K`; isoneutral
  `abs <=5.0e-3 K` measured `5.785612304013637e-3 K`.  The geometry, matrix,
  solve and boundary upper bounds were confirmed.  The measured ranking wins.
- **P3 UPSTREAM PREDICTION CONFIRMED; JET-SPECIFIC PREDICTION REFUTED.** Both
  conditional RMS reductions exceed 0.999999999999997 and both EVD trigger
  masks become exact.  The two centroid displacement vectors have opposite
  directions, so the verdict remains upstream state, not jet position.
- **P4 CONFIRMED.** This is diagnostic-only.  Only the private observer,
  owner gate, controls, tests, citation mappings, preregistration and receipt
  land.

## Compiled execution order

The cited program is the compiled branch that produced the admitted record.
Stage 3 completes its optional tracer processes and then calls `tra_zdf` with
`Kbb`, `Kmm`, `Krhs` and output slot `Kaa` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90:937-944`.

For temperature, the compiled implicit routine chooses `avt`, adds the
standard isoneutral `ah_wslp2` contribution, zeros the surface mixing
coefficient, and forms the lower/upper/diagonal matrix from live `e3w(Kmm)`
and `e3t(Kaa)` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480`.
It then forms the LU diagonal, thickness-weighted content RHS, forward sweep
and backward solution at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582`.

The compiled vertical-physics driver calls the selected TKE closure, copies
its `avt_k` into `avt`, and then calls enhanced vertical diffusion at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359`.
The TKE closure's coefficient itself is the energy/mixing-length expression
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`.
The later EVD statement replaces `avt` when the minimum of `rn2` and `rn2b`
crosses `-1e-12` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`.
The record's resolved card enables both TKE and EVD and sets `rn_evd=100`
and background `rn_avt0=1.2e-5` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:205-213`.

These compiled statements explain why an 11--15-cell instability-mask
difference has order-100 coefficient magnitude.  The conditional experiment,
not the source text alone, establishes that legoESM executes the same
coefficient statements when it receives NEMO's state.

## Oracle-record admission

The normal report is `round126/nemo_vertical_record_validation.json`.

| control | measured result |
|---|---:|
| record set | 362 frames: steps 1, 2 and 1081--1440 |
| bytes per frame / total | 6,965,416 / 2,521,480,592 |
| interval frames / bytes | 360 / 2,507,549,760 |
| process-alignment rows unequal | 0 in all 7 rows |
| compiled reconstruction rows unequal | 0 in all 8 rows |
| step-1080 restart SHA-256 | `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976` |
| step-1440 restart SHA-256 | `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a` |

The persisted plant reports are
`plant_vertical-{stamp,truncation,matrix-ulp,trajectory-ulp}.json`.  Each exits
1 and prints `STATUS PLANT-FIRED`; none prints PASS while firing.

## legoESM production trace and retained refusals

The trace producer is clean commit
`34070a99e43227412a6118ecdcf31180b0e1c6d3`.  Its authoritative root is
`round126/lego_vertical_trace_v2/`; it contains 360 frames and took
`2432.789155960083 s`.  The final validation is
`round126/lego_vertical_trace_v2_validation_final.json`.

| passivity / control row | measured result |
|---|---:|
| carried-state unequal bytes | 0 |
| inherited Round-124 process fields unequal | 0 in all 11 fields |
| effective K versus heat plus isoneutral K | 0 unequal interfaces |
| solved T versus production process `Taa` | 0 unequal cells |
| K-effect plant upstream moved cells | 0 in all 10 fields |
| K-effect plant heat/effective K moved | 1 / 1 cells |
| K-effect plant lower/diagonal/upper moved | 1 / 2 / 1 cells |
| K-effect plant solved T moved | 4 cells |

Two refusals are preserved rather than rewritten:

1. `round126/lego_vertical_trace_v1_refusal.json` rejects the first 360-step
   run because the enlarged single observer graph moved 78 `Bpre` last bits.
   The repair runs the unchanged Round-124 process observer and the vertical
   observer as separate compiled graphs from the same ordinary entry and
   carries only their independently compiled ordinary state.
2. `round126/lego_vertical_trace_v2_validation_v1_refusal.json` rejects
   `1,443,898` diagonal cells when an isolated NumPy matrix reconstruction is
   incorrectly treated as the production boundary.  Lower and upper rows are
   bit-exact.  Per the production-JIT rule, the repaired validator reports
   this isolated-association diagnostic but admits the directly returned
   production matrix by manifest/hash, exact production solve output and a
   full-step K-to-matrix-to-solution effect plant.  The source-order budget
   retains the discrepancy in `matrix_content_association`; it does not erase
   or relabel it as production arithmetic.

The final wrong-stamp and stored-matrix-bit controls are
`plant_lego-vertical-stamp.json` and
`plant_lego-vertical-matrix-ulp.json`; both print `STATUS PLANT-FIRED` and
exit 1.

## Day-240 vertical sub-owner ranking

The authoritative machine result is
`round126/day240_vertical_subowners.json`, generated at clean commit
`16f0cec758fb2fa8b3febf590543a1f0a460d46d`.  Carries are signed projections
onto the independently measured day-240 temperature error.  The rows are a
fixed source-order telescope; their individual values are order-dependent,
while their sum and the endpoint are not.

| rank | frozen substitution boundary | signed carry (K) | component RMS (K) | strongest ten-day birth | strongest depth / longitude / latitude |
|---:|---|---:|---:|---|---|
| 1 | TKE/EVD heat diffusivity `avt` | `+7.158989942001176e-1` | `1.0535048764054324` | days 230--240, `+1.7492888906312726e-1 K` | 0--100 m / west third / south <=37.2 N |
| 2 | live column/gradient acted on | `-6.975158045602758e-1` | `1.014884430148211` | days 230--240, `-1.7228664821028808e-1 K` | 0--100 m / west third / south <=37.2 N |
| 3 | isoneutral vertical diffusivity | `+5.785612304013637e-3` | `4.7362018464761096e-2` | days 200--210, `+1.1570365215460635e-3 K` | 100--1000 m / east third / north >37.2 N |
| 4 | free-surface weighting | `-5.303658013767689e-7` | `8.131299395571062e-7` | days 230--240, `-1.435678625900588e-7 K` | 0--100 m / west third / south <=37.2 N |
| 5 | matrix/content association | `+2.2195371546170447e-16` | `7.742977051894698e-13` | days 210--220, `-4.834080294003923e-16 K` | 0--100 m / interior third / south <=37.2 N |
| 6 | interaction/order residual | `+3.3982113394762173e-17` | `1.6970072448573172e-14` | days 210--220, `+1.7982919296720384e-16 K` | 100--1000 m / east third / north >37.2 N |
| 7 | implicit-solve association | `0.0` | `0.0` | no nonzero block | none |

Budget controls:

| control | value |
|---|---:|
| day-240 wet-T3D RMS | `1.6446741930292448e-2 K` |
| inherited vertical carry | `+2.4168271578053416e-2 K` |
| signed sub-owner sum | `+2.4168271578054273e-2 K` |
| sum minus inherited carry | `+8.569533971325427e-16 K` |
| maximum per-step telescope residual | `0.0 K` |
| NEMO offline solve unequal cells | 0 |
| legoESM recorded-matrix solve versus production unequal cells | 0 |
| explicit surface/bottom nonzero visits, NEMO | 0 / 0 |
| explicit surface/bottom nonzero visits, legoESM | 0 / 0 |

The matrix row's small signed projection does not mean the isolated diagonal
was bit-exact: its component RMS is `7.742977051894698e-13 K`, and the raw
isolated diagonal diagnostic remains 1,443,898 unequal last-bit cells.  The
implicit solve row is exactly zero because the source-ordered offline solve of
the **recorded production matrix** reproduces the production solved column.

## Closure versus upstream state

| checkpoint | own `avt` RMS / max error (m2/s) | conditional-on-NEMO RMS / max (m2/s) | RMS reduction | EVD trigger mismatch own / conditional | western jet-centroid displacement `(j,i)` cells |
|---:|---:|---:|---:|---:|---:|
| day 180, step 1081 | `2.93525982529855` / `99.99982684066974` | `8.260934106155803e-15` / `1.0896838986695911e-12` | `0.9999999999999972` | 15 / 0 | `(-2.2107554412720276e-4,-1.2251627060858362e-4)` |
| day 210, step 1261 | `2.5136487632185793` / `99.99719888565417` | `2.4586546006968085e-17` / `2.706168622523819e-15` | `1.0` | 11 / 0 | `(+1.1337693336876953e-4,+2.1132161430958973e-5)` |

The remaining conditional unequal cells are last-bit coefficient differences,
not EVD classification errors; their RMS is 14--17 orders smaller than the
model-own coefficient error.  Since the centroid displacement reverses sign,
the preregistered classifier returns `UPSTREAM-STATE-RESPONSE`, not
`UPSTREAM-JET-POSITION-RESPONSE`.  This directs Round 127 to the state and
stratification inputs of EVD; it does not authorize a TKE or EVD formula fix.

## Landing and cross-card scope

No physical statement lands.  The admitted headlines remain:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The new observer is a private test hook, statically false for ordinary model
construction.  GYRE production, generic NEMO-GYRE, DINO, LOCK_EXCHANGE and
OVERFLOW execute no new production arm, so their certified rows do not move.
ORCA2 remains **UNMEASURED-WITH-SPEC**: repeat a native interval matrix record,
independent production trace, source-order endpoint projection and
conditional-on-NEMO closure/EVD-trigger test before transferring this ranking.

No configuration or carried-state choice is exposed.  `DECISION_NEEDED` is
`NONE`.

## Independent adversarial review

The required command was run against the complete incoming-tip-to-receipt
diff with `codex exec --sandbox read-only`.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

The command exited 1.  Independent review unavailable in-sandbox.  It emitted
neither `SHIP` nor `DO NOT SHIP`; no verdict is fabricated.  The log is
`round126/codex_review.log`.

## Verification

The owner self-check prints `self-check: all checks passed`; its log is
`round126/self_check.log`.  The final clean-tree focused suite covering the
literal vertical solver, year-owner/vertical gate, compiled-source citation
gate and worktree-stamp ratchet reported:

```text
============================= 57 passed in 49.54s ==============================
```

Its log and JUnit are `round126/focused_tests.log` and
`round126/focused_tests.xml`.  Python byte compilation and `git diff --check`
also pass.

The receipt citation gate found seven citations, zero unmapped citations,
zero failures and zero map-audit failures in `round126/citation_gate.json`.
Shifting the compiled EVD trigger citation by two lines produced
`SYMBOL-NOT-AT-LINE`, printed no PASS verdict and exited 1; its report and log
are `round126/citation_gate_shift_plant.json` and
`round126/citation_gate_shift_plant.log`.  Because the private observer added
lines to an already cited production file, every affected historical citation
was rigidly shifted with its extent unchanged; the full campaign gate then
passed 274 citations with zero map-audit failures in
`round126/citation_gate_default_after_trace.json`.

## OPEN — round 127

Do not fix TKE, EVD, the implicit matrix or the solve: the conditional
production experiment rules those out as the magnitude owner on NEMO's state.

1. Preregister a magnitude decomposition of the EVD trigger's upstream
   `rn2/rn2b` state over days 180--240.  The headline controls are the admitted
   15 and 11 own-trajectory trigger disagreements at days 180 and 210 and zero
   disagreements on NEMO entry.
2. At those trigger cells, separate temperature, salinity and live-depth/free-
   surface contributions to the `rn2/rn2b <= -1e-12` crossing, then locate
   when, depth band, longitude third and latitude band the mismatching state
   first appears.  Use the production-JIT closure and a trigger-bit plant; do
   not substitute an isolated N2 formula for the production result.
3. Because the jet-centroid vectors reverse, call the result upstream state
   unless a new discriminating measurement directly establishes coherent jet
   displacement.  Do not carry “jet position” forward as an established fact.
4. The largest upstream state/process row becomes the next candidate.  A
   production landing still requires the Decision-43 ladder/month and
   Decision-45 year gates against a same-tip before arm, with DINO measured if
   the statement is shared.

No NEMO acquisition is needed for this next decomposition: the admitted
Round-125 interval record contains `avt`, the matrix and the endpoint state,
and the year/process records contain the independent trajectory.  Request a
new record only if the preregistered trigger operands cannot be reconstructed
or read from those admitted artifacts.
