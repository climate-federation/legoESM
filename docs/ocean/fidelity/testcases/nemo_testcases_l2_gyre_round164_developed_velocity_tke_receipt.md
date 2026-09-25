# Round 164 receipt — developed stage-1 velocity and tracer-ZDF/TKE boundary

**Status: STOPPED_FOR_RECORD.** No production physics, card choice, state
schema, or configuration changed.  The stage-2 authority reproduces the
55.06% entry-velocity share exactly, the stage-1 walk localizes the material
velocity difference to the barotropic correction boundary, and the developed
tracer walk localizes the largest remaining one-step process error to the heat
diffusivity supplied by TKE.  The admitted records stop at `avt_k`; they do not
carry the developed TKE operands required to name the upstream statement.  A
fail-closed operator acquisition is therefore the result of this round.

Preregistration was commit `6adb940ee`.  The measurement extensions were
committed at `ac81cf9db`; shape-only corrections that let the stage-1 plant
reach its assertion are `1c71fc96a` and `81e4570d8`.  Evidence is under
`phase3/round164/`.  The clean production measurements were made at
`0eb4e5ee39ad63b20b4d61c6cc449ea070c29476`; the later commits add only the
acquisition card and repair the plant's score-mask dimensionality.

## Frozen predictions and results

| preregistered item | result |
|---|---|
| velocity-form authority `2.334682468902387e-13` m/s | **CONFIRMED**, exact stored digits |
| authority with NEMO entry velocity `1.0492082366460718e-13` m/s | **CONFIRMED**, 55.059917114153% removed |
| HPG and LDF BIT; VOR first non-bit | **REFUTED**: HPG is BIT on active faces, but the cross-build LDF boundary is already non-bit at `8.17e-22`/`8.35e-22` m s-2 |
| current process ranking reproduces corrected private arm | **CONFIRMED** to all registered digits; current no-hook arm is BIT-identical to explicit corrected arm |
| vertical diffusion remains largest one-step process row | **CONFIRMED**, `2.1834094362625713e-05` K rms |
| first observed vertical boundary is the heat coefficient | **CONFIRMED**: `avt`/`heat_K`, 5,721/17,400 cells, max `1.594045423436441e-09` m2 s-1 |
| round-125 lacks the upstream TKE discriminator | **CONFIRMED**; acquisition required |
| ORCA2 resolves route, explicit production choice false | **CONFIRMED**; route true, at-tip execution false |
| ORCA2 certified trajectory executes | **REFUTED before trajectory**, honestly UNMEASURED on six selected arms |

Failed predictions are retained here; none was rewritten after measurement.

## Stage-2 authority and the stage-1 output walk

The authority run is `wzv/wzv_authority.json` and
`wzv_authority_current.log`.  Its tagged velocity-indicator call gives:

| arm | active rms (m/s) | max (m/s) | cells unequal |
|---|---:|---:|---:|
| current production | 1.2326857042024439e-08 | 4.896576158832036e-08 | 18,000/18,000 |
| null, own entry reinstalled | 1.2326857042027658e-08 | 4.8965761588036394e-08 | 18,000/18,000 |
| velocity form | 2.334682468902387e-13 | 1.195710882770948e-12 | 18,000/18,000 |
| velocity form + NEMO entry velocity | 1.0492082366460718e-13 | 3.9066402463515133e-13 | 18,000/18,000 |

The null differs from production by only `2.61e-13` relative in the scored
rms, while the NEMO entry velocity removes
`1.2854742322563152e-13` m/s, **55.059917114153%**, from the velocity-form
residual.  This is the round-159 JSON result the campaign summary had
misquoted, now reproduced at the landed two-solve tip.  NEMO selects the
velocity-indicator continuity call in the vector arm at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360`; the walk tags
that call and does not infer it from call count.

The stage-1 walk is `stage1/stage1_output_walk.json`.  NEMO's compiled
external-step order is HPG, LDF, VOR, KEG and ZAD in
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192`.

| boundary | U active unequal | U max / rms | V active unequal | V max / rms |
|---|---:|---:|---:|---:|
| HPG | 0/17,400 | 0 / 0 | 0/17,100 | 0 / 0 |
| LDF | 15,327/17,400 | 8.1725e-22 / 6.5984e-23 | 14,849/17,100 | 8.3483e-22 / 7.4666e-23 |
| VOR | 10,591/17,400 | 8.4703e-22 / 9.7311e-23 | 9,501/17,100 | 1.6941e-21 / 9.8205e-23 |
| KEG | 17,347/17,400 | 8.4703e-22 / 9.8960e-23 | 17,067/17,100 | 8.4207e-22 / 9.7661e-23 |
| ZAD | 17,377/17,400 | 8.3111e-22 / 9.6860e-23 | 17,094/17,100 | 8.4203e-22 / 9.6675e-23 |
| completed RHS | 4,388/17,400 | 1.6941e-21 / 1.7153e-22 | 4,725/17,100 | 1.6941e-21 / 1.5489e-22 |
| raw stage update | 2,697/17,400 | 1.3878e-17 / 9.5006e-19 | 2,471/17,100 | 2.7756e-17 / 1.0865e-18 |
| barotropic correction | 17,400/17,400 | 6.1342e-10 / 1.9012e-10 | 17,100/17,100 | 5.8001e-10 / 1.6745e-10 |
| final stage-1 output | 17,400/17,400 | 6.1342e-10 / 1.9012e-10 | 17,100/17,100 | 5.8001e-10 / 1.6745e-10 |

The first cross-record non-bit boundary is therefore LDF, refuting P2.  It is
**not** assigned to LDF physics: the family record and production trace come
from different instrumented compilations, note AS makes such inherited
last-bit differences informational, and round 149's same-build developed LDF
proof was BIT.  The measured scale here (`<1e-21` m s-2) is that compiled
scheduling floor.  NEMO's raw vector update is the branch at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:721` and its U
write is at `:723`.  The first material boundary is the barotropic correction:
NEMO computes `zub` at `:787` and adds it to every level at `:810`.  Thus the
stage-1 output inherited by stage 2 is magnitude-owned by the barotropic
target/correction, not by the 3-D RHS operator rounding.  This round does not
turn that attribution into a candidate.

The repaired `stage1-entry-u-ulp` control exits 1 with `STATUS PLANT-FIRED`:
one active output cell moves by `2.7755575615628914e-17` m/s.  The two earlier
attempts are retained in the evidence directory: both failed on instrument
shape defects before reaching the assertion and are not counted as plants.

## Current-arm process ranking

`process_rank/process_rank_split.json` reruns the historical production arm,
the explicit corrected arm, and the current no-hook landed arm from the same
admitted day-180 entry.  Every current row is BIT-identical to the explicit
corrected arm (zero unequal cells for all six processes).

| rank | current landed process | T contribution rms (K) | max (K) |
|---:|---|---:|---:|
| 1 | vertical diffusion | 2.1834094362625713e-05 | 6.652851999202625e-04 |
| 2 | lateral diffusion | 1.0655217366762800e-05 | 3.562142565147752e-04 |
| 3 | shortwave | 2.513690760778605e-07 | 2.468039319580839e-06 |
| 4 | geometry | 6.281725340556424e-12 | 2.843236757144041e-11 |
| 5 | advection | 6.181193168123800e-12 | 4.748912374452630e-11 |
| 6 | surface boundary | 4.680755513457375e-15 | 9.592326932761353e-14 |

The preregistered advection and vertical values reproduce.  Their corrected
combined rms is `2.1834094298886284e-05` K.  The cancellation hypothesis
remains REFUTED: Pearson `+0.0026539162702729348`, uncentred cosine
`+0.0031034599597189708`, and an exact zero-residual decomposition over all
18,000 cells.

## First observed non-bit vertical statement

`vertical/vertical_walk.json` extends the existing developed process walk; it
does not introduce a second bridge or a local-only closure.  The compiled
dataflow is closure selection and `zdf_tke`, copying `avt_k` into `avt`, then
EVD in
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359`;
EVD's active branch is exactly
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`.

| compiled dataflow row | cells unequal / scored | max abs | rms |
|---|---:|---:|---:|
| heat diffusivity `avt` | 5,721 / 17,400 | 1.594045423436441e-09 m2/s | 1.380750544672839e-11 m2/s |
| isoneutral coefficient | 17,400 / 17,400 | 2.731312717075364e-03 m2/s | 1.270216333966077e-04 m2/s |
| effective coefficient | 17,400 / 17,400 | 2.731312717075364e-03 m2/s | 1.270216333377450e-04 m2/s |
| after thickness | 18,000 / 18,000 | 3.932996150979307e-10 m | 8.159729031748704e-11 m |
| current W thickness | 17,400 / 17,400 | 1.964508555829525e-10 m | 4.058309524120698e-11 m |
| content RHS | 18,000 / 18,000 | 3.769802907231679e-02 | 7.046353169351283e-04 |
| lower / diagonal / upper | 17,400 / 18,000 / 17,400 | 8.769312921405801e-01 / 1.598792846143311 / 8.769312921405801e-01 | 3.673574232652890e-02 / 6.922797760420521e-02 / 3.673574232652889e-02 |
| solved temperature | 18,000 / 18,000 | 7.588041023360859e-04 K | 1.924391457462549e-05 K |

The coefficient feeds the matrix exactly as shown by
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480`, and the
compiled three recurrences are `:527-582`.  The first **directly scored**
non-bit statement is therefore the `avt_k -> avt` copy within the cited
`zdfphy` range.  Its arithmetic is not accused: its operand is already
non-bit.  NEMO produces that operand through the TKE energy solve and
mixing-length conversion; the final conversion is compiled at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`.

The admitted vertical record contains none of `en`, `sh2`, `rn2`, `rn2b`,
`avm_k`, `avt_k`, or the TKE sweep boundaries at step 1081.  Consequently no
upstream TKE statement can be named honestly from this record.  The
`vertical-heat-ulp` control exits 1 before success output because the planted
coefficient changes an upstream process boundary; this is the intended
fail-closed result, not a passing plant.

## Acquisition requested

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round164_developed_tke/run.sh`
clones the exact Round-132 source card file by file into the new target
`GYRE_OMIP_L2_P3_SM_R164TKEDEV`, adds the previously calibrated round-54 and
round-101 WRITE-only instruments, and records their operands and boundaries
at step 1081.  It changes only `nn_itend=1081`, `nn_stock=1080`, and
`nn_write=0`; it uses no `/usr/bin/time`.

Passivity is fail-closed on the uninstrumented Round-132 step-1080 restart:
the candidate restart must be byte-identical.  Both binary records have
fixed byte counts, header shapes and commit/digest stamps.  `--preflight-only`
passes four `gfortran -fsyntax-only` proofs (`l2_r54_tke`,
`l2_r101_tke_walk`, `zdftke`, `zdfphy`); `--plant-layout` removes the
post-Langmuir boundary and exits 69 with a named REFUSE line.  Because PMIx
sockets are unavailable in the sandbox, the script was not run here.

## ORCA2 blast radius

`orca2_census_and_execution.json` is built from the real deck, not a nominal
config.  ORCA2 resolves the shared stage-momentum two-solve route
(`executes_route=true`) while its own explicit setting keeps it out at this
tip (`executes_at_this_tip=false`).  Its certified execution validator then
refuses before trajectory integration:

> ORCA2-zps is not execution-ready; unresolved selected arms: staged_gm_eiv,
> linear_implicit_bottom_drag, internal_wave_mixing,
> spatial_lateral_viscosity, freshwater_budget_carry,
> si3_jpl5_layered_prather_state

The ORCA2 route is therefore **UNMEASURED**, not inert and not waived.
Decision 58 remains pending and was not answered by changing the card.  The
new census row is non-vacuously asserted from `build_orca2_zps_card` in the
Decision-43 gate test.

## Certified production headline and landing table

No candidate physics was formed, so no month/year integration or trajectory
comparison was rerun.  The immutable round-163 after arm remains production:

| certified row | current value | status |
|---|---:|---|
| kt2 T max abs | 1.4210854715202004e-14 K | AT-BAR |
| kt2 S max abs | 2.1316282072803006e-14 | AT-BAR |
| kt2 U max abs | 8.326672684688674e-17 m/s | AT-BAR |
| kt2 V max abs | 9.714451465470120e-17 m/s | AT-BAR |
| kt3 T max abs | 4.940071072212504e-07 K | DEBT |
| kt3 S max abs | 4.008629872487290e-08 | DEBT |
| day-30 T rms | 6.572574374770603e-05 K | immutable after arm |
| day-240 T rms | 1.644836070117868e-02 K | immutable after arm |
| day-360 T rms | 1.122566001855131e-02 K | immutable after arm |

Decision 43/45/55 criterion table: **NOT RUN / no candidate**.  No row moved,
no AT-BAR row left, no first-over-bar moved, and there is no DINO, tank,
generic-card, or ORCA2 production blast radius because no shared physics was
modified.  This is not a landing claim.

## Controls, tests, citation gate and review

To be finalized after the independent read-only review and final batteries.

## OPEN — Round 165

1. Operator runs the acquisition script below.  Admit it only if the
   step-1080 restart is byte-identical to Round 132 and both record
   header/commit stamps pass.  If it refuses, do not reuse a partial record.
2. From the admitted step-1081 record, walk TKE in compiled order under the
   production step: entry `en`; boundary treatment; Langmuir source; shear,
   buoyancy and dissipation RHS; tridiagonal sweep; mixing length; `avm_k`
   and `avt_k`.  Name the first non-bit statement that produces the observed
   `avt_k` mismatch.  Eager rows may accompany but never replace production
   JIT rows.
3. Only a one-variable compiled NEMO statement may become a candidate.  It
   must then pass the full Decision 43/45 gate as amended by Decision 55;
   measure DINO/generic/tanks if the statement is shared.
4. ORCA2 remains UNMEASURED on six selected arms and Decision 58 remains
   pending.  Do not flip its explicit `False` until a certified executable
   card exists and its own trajectory is measured.
5. The unrelated round-160 compiled-scheduling-floor caveat remains open.
