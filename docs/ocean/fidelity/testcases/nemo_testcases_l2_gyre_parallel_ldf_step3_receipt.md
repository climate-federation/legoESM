# NEMO-testcases L2 GYRE receipt: parallel LDF step-3 candidate held

Date: 2026-09-16. Delivery branch `held/gyre-ldf-step3`; requested starting
tip `3e7a15c1e64e036e2e066dcfceaa663f25406f24`.

## Verdict

**HELD; NO PRODUCTION PHYSICS LANDED.** The stage-3 lateral-diffusion content
route remains source-faithful and reproduces its current-tip private causal
arm exactly, but its old Round-69 absolute target is stale, it makes no
given-NEMO-entry stage output exact, and the canonical Rule-12 ladder fails:
53 of 954 registered rows move and every moved row has at least one cell that
worsens against NEMO by more than two row-scale binary64 ULPs. No row changes
class and first-over-bar remains kt2 U/V.

Decision 41 independently forbids landing this stage-3 change while the
autopilot owns the earlier kt=1 stage-1 walk. The measured candidate exists
only as scratch commit `6e0386627c29b1eab2e284fcee673a5523712df4` and as
the held manifest patch
`scripts/validate/ocean_fidelity/testcases/manifests/
nemo_testcase_l2_gyre_parallel_ldf_step3_held.patch`.

## Compiled statement and candidate

The compiled GYRE stage clears tracer `Krhs`, then accumulates advection and
SBC at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950`.
Stage 3 adds QSR, calls `tra_ldf`, and then calls `tra_zdf` at the same
compiled file's `:917-965`. The selected LDF implementation reads Kbb tracer
gradients at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`,
forms its Kmm-metric fluxes at `:230-246`, and adds their divergence with a
positive sign to `Krhs` at `:287-305`. The ZDF recurrence consumes Kbb
content plus Kmm thickness times that `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

At the requested tip, the private one-variable route is declared at
`ocean_model_latlon_cgrid.py:1350-1353`
and adds the already-computed signed `dT_gm/dS_gm` arrays to the WS helper's
stage-3 source tuple at `:7723-7731`. The candidate removes that private
selector and executes the identical route when the already selected `rk3_ws`
path has a configured GM/Redi operator. It does not add a card selector,
coefficient, timestep, carry, stabilizer, state field, second FCT call, public
API, or alternate helper.

No NEMO source, executable, build, or record was changed or run. The cited
oracle tree remained read-only at
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`.

## Local proof: historical target refuted, same-tip equivalence confirmed

The first candidate invocation used the existing Round-67/69 gate, the
Round-64 `oracle_krhs_split` record, the Round-46 `oracle_kt2_stage` records,
and producer commit `3b3b045bd9e03b60330204e7590e4c4470b7a0ca`. Record
admission reproduced 43 exact and 20 classified-changed records and admitted
all 132 consumed values. The clean candidate artifact is
`parallel/ldfstep3/candidate/local_proof.json`, SHA-256
`c51ad1e7a3aed0821997193e3d6a6076ce0747283883f9efbe5cde849846a60f`.

That gate correctly reports **`REFUTED`** against Round 69's frozen absolute
dictionaries. Both frozen content comparisons, both frozen kt3 comparisons,
and the temperature association census are unequal. The old and current
maxima are:

| quantity | Round-69 frozen T / S | current-tip candidate T / S |
|---|---:|---:|
| stage-3 content vs NEMO | `5.954039670541533e-5` / `7.651457053725608e-6` | `5.743498263655056e-5` / `7.387909136014059e-6` |
| post-solve kt3 vs NEMO | `8.916073106490785e-7` / `7.235656340753849e-8` | `8.600420500215478e-7` / `6.979443156751586e-8` |

The old proof therefore does **not** survive as literal equality to its old
artifact. That target predates intervening upstream production changes; this
receipt does not assign the numerical delta to one intervening commit.

The preregistered addendum then used the existing Round-70 mode of the same
gate on the clean current tip. Its private `ldf_only` arm used the same
Round-64/Round-46 records. The complete `content_vs_oracle` and
`kt3_vs_oracle` dictionaries—not rounded maxima—equal the clean candidate's
dictionaries exactly for both tracers, and both candidate source-injection
rows have zero unequal cells:

| tracer | content unequal / max / RMS | kt3 unequal / max / RMS | candidate equals current private arm |
|---|---:|---:|---|
| T | `18000` / `5.743498263655056e-5` / `4.449124265960819e-6` | `17999` / `8.600420500215478e-7` / `8.802189293022496e-8` | exact complete dictionaries; source injection `0` unequal |
| S | `18000` / `7.387909136014059e-6` / `5.370401527093586e-7` | `17265` / `6.979443156751586e-8` / `6.51849324799198e-9` | exact complete dictionaries; source injection `0` unequal |

The private-arm artifact is
`parallel/ldfstep3/baseline/current_tip_private_pair.json`, SHA-256
`4d3496ee73a0cf24774d1525f11aec3f4ae9c4308d68157cd395ab7a79c3ba45`.
Its broader Round-70 pair verdict remains `REFUTED`; only its preregistered
`ldf_only` arm is used here. Thus the current-tip implementation equivalence
is bit-exact even though the historical absolute dictionary is not.

## Existing stage twin

The existing
`nemo_testcase_l2_gyre_round46_kt2_stage_gate.py --mode stage-twin` was run
to completion at both clean commits; no second stage harness was written.
Both invocations print `STATUS PASS`, retain all 98 entry rows, and contain
124 literal output rows in each of the given-NEMO-entry and model-chained
tables. Baseline/candidate artifacts and SHA-256 values are:

- `parallel/ldfstep3/baseline/stage_twin.json`:
  `ab6075ea9dc9935c749ec1c2910eff710192f40cef6f9cb70670f23c0568552a`;
- `parallel/ldfstep3/candidate/stage_twin.json`:
  `7afff473d08eff1dba288a2fd3e8ceb5b0f25292b8660e8864dc795ccf06eb83`.

All 98 entry rows are unchanged. In the given-NEMO-entry table, only kt2
stage-3 T/S move, and neither becomes exact:

| entry mode | output | tip class / unequal / max | candidate class / unequal / max | disposition |
|---|---|---:|---:|---|
| NEMO recorded, kt1 s3 | T | DEBT / `8851` / `1.2830175028447854e-2` | same | unchanged |
| NEMO recorded, kt1 s3 | S | DEBT / `8394` / `5.196377745448899e-4` | same | unchanged |
| NEMO recorded, kt2 s3 | T | DEBT / `17988` / `7.97034157194787e-3` | DEBT / `17644` / `7.975082879060125e-3` | moves; not exact; maximum worsens |
| NEMO recorded, kt2 s3 | S | DEBT / `15488` / `1.0668958712400922e-3` | DEBT / `14234` / `1.064355106770165e-3` | moves; not exact; maximum improves |

Therefore the answer to “which stage-3 output rows does it make exact given
NEMO's entry?” is **none**. Existing BIT stage-3 handoff/TKE rows remain BIT
but are not made exact by this candidate.

In the model-chained table, only kt2 stage-3 T/S move; the other 122 output
rows are identical:

| chained output | tip class / unequal / max | candidate class / unequal / max | disposition |
|---|---:|---:|---|
| kt1 s3 T | AT-BAR / `6858` / `1.4210854715202004e-14` | same | unchanged |
| kt1 s3 S | AT-BAR / `6457` / `2.1316282072803006e-14` | same | unchanged |
| kt2 s3 T | DEBT / `18000` / `1.627497246303733e-4` | DEBT / `18000` / `8.600419718618468e-7` | moves; maximum improves 189.2x; not exact |
| kt2 s3 S | DEBT / `17378` / `6.327735185607253e-6` | DEBT / `17379` / `6.97944244620885e-8` | moves; maximum improves 90.7x; not exact |

The first owned non-bit stage remains kt=1 stage 1 U, AT-BAR in 7,620 cells
at `5.421010862427522e-20`. This downstream candidate does not change that
Decision-41 boundary.

## Full kt=1..10 Rule-12 ladder

The canonical gate was run without `--trajectory-only` at both clean commits.
Each report contains 954 registered residual rows. The candidate invocation
used the gate's own `--compare-to` and `--comparison-output` implementation.
Artifacts are:

- tip ladder `parallel/ldfstep3/baseline/ladder_full.json`, SHA-256
  `ad698f14fce4bf80a586be4b50149d006273f359a7e077be557c8becd091e28a`;
- tip residual sidecar, SHA-256
  `d5ce82ed1fa06b210e7adb16bab27503aa176b51403921c35623853abef31893`;
- candidate ladder `parallel/ldfstep3/candidate/ladder_full.json`, SHA-256
  `594c861f37b33f7dafa60fb917c48c4d3ee144f82cf7ecd36c500ba3091db742`;
- candidate residual sidecar, SHA-256
  `da637c28041051b02c0a94a99974cd09e97a8302c114f2f92795fc6b4ec661fb`;
- full comparison `parallel/ldfstep3/candidate/ladder_comparison.json`,
  SHA-256
  `7f2c2309f5f7ee55de3ce8c5a180570195cc607ab4038a3cf566529681743ff5`.

The comparison is **FAIL**: 53 rows move and the same 53 rows violate the
two-ULP oracle-relative movement bar. There are zero status changes, no
report-status change, and first-over-bar is kt2 U/V in both arms. The largest
oracle-residual worsening is `101838520042.21875` row-scale ULPs. The first
causal trajectory move is kt3-before T/S; there is no kt1 or kt2 move.

This is the complete nonzero moved-row table. “Moved” is the maximum absolute
candidate-minus-tip field movement; improved/worsened counts are cellwise
against NEMO, and `>2 ULP` is the gate's violation census.

| row | tip max | candidate max | moved | improved | worsened | >2 ULP |
|---|---:|---:|---:|---:|---:|---:|
| `GYRE-zco.kt3.after.uu_b` | `1.178525769766816e-7` | `1.1804872147619556e-7` | `5.2205883659030977e-10` | 269 | 311 | 311 |
| `GYRE-zco.kt3.after.vv_b` | `7.089790817762457e-8` | `7.0736399775160955e-8` | `5.370232187542831e-10` | 250 | 320 | 320 |
| `GYRE-zco.kt3.before.S` | `6.3277351856072528e-6` | `6.97944244620885e-8` | `6.3265926044664411e-6` | 7809 | 6230 | 5824 |
| `GYRE-zco.kt3.before.T` | `1.6274972463037329e-4` | `8.6004197186184683e-7` | `1.626617136771813e-4` | 9665 | 8320 | 8240 |
| `GYRE-zco.kt4.after.uu_b` | `1.4040031106644923e-7` | `1.4078950055025803e-7` | `1.1519657200798755e-9` | 256 | 324 | 324 |
| `GYRE-zco.kt4.after.vv_b` | `9.4137505911892454e-8` | `9.3848313686929405e-8` | `1.2803270101014938e-9` | 299 | 271 | 271 |
| `GYRE-zco.kt4.before.S` | `1.48574231531029e-5` | `4.7252279244958117e-7` | `1.4809267817383898e-5` | 12031 | 5132 | 4452 |
| `GYRE-zco.kt4.before.T` | `3.7880763850139942e-4` | `5.8269268095045845e-6` | `3.7764030215114985e-4` | 12746 | 5253 | 5238 |
| `GYRE-zco.kt4.before.ssh` | `4.577542667915345e-7` | `4.6933513020257223e-7` | `1.0319142079370791e-7` | 327 | 273 | 273 |
| `GYRE-zco.kt4.before.u` | `2.0557060249691561e-5` | `2.0552956753669416e-5` | `3.070252775218274e-6` | 8951 | 8449 | 8449 |
| `GYRE-zco.kt4.before.v` | `1.8319409220590721e-5` | `1.7604221225930861e-5` | `4.5753190759112439e-6` | 8755 | 8345 | 8345 |
| `GYRE-zco.kt5.after.uu_b` | `2.9752690469843487e-7` | `1.4927768608556657e-7` | `2.9280481009953511e-7` | 300 | 280 | 280 |
| `GYRE-zco.kt5.after.vv_b` | `2.1959943558032106e-7` | `7.9384013595496308e-8` | `2.1048770219388288e-7` | 264 | 306 | 306 |
| `GYRE-zco.kt5.before.S` | `2.5487148408984694e-3` | `4.2859174698151037e-7` | `2.5487124126541971e-3` | 12681 | 4782 | 4184 |
| `GYRE-zco.kt5.before.T` | `5.7537968257086902e-2` | `7.5037487690110538e-6` | `5.7537701686193543e-2` | 13177 | 4823 | 4821 |
| `GYRE-zco.kt5.before.ssh` | `7.7419854865911145e-7` | `3.6889919135260976e-7` | `8.1852636741523536e-7` | 312 | 288 | 288 |
| `GYRE-zco.kt5.before.u` | `9.1535810057503421e-3` | `3.709912415888637e-5` | `9.1640821477342593e-3` | 9220 | 8180 | 8180 |
| `GYRE-zco.kt5.before.v` | `3.8977704462847096e-2` | `1.8740728287947039e-5` | `3.8971982805023064e-2` | 9078 | 8022 | 8022 |
| `GYRE-zco.kt6.after.uu_b` | `2.1650266523456311e-7` | `1.277922236908851e-7` | `2.0327608187033501e-7` | 317 | 263 | 263 |
| `GYRE-zco.kt6.after.vv_b` | `1.3081666796051173e-7` | `1.0840287114981771e-7` | `1.2811341321859119e-7` | 283 | 287 | 287 |
| `GYRE-zco.kt6.before.S` | `3.7495788822639042e-4` | `9.2775697879687868e-7` | `3.749699157040709e-4` | 13296 | 4318 | 3855 |
| `GYRE-zco.kt6.before.T` | `1.0066457774108528e-2` | `1.1337056044169458e-5` | `1.0066541800345163e-2` | 13734 | 4266 | 4265 |
| `GYRE-zco.kt6.before.ssh` | `3.2036937495085946e-6` | `5.5872750230872431e-7` | `3.092578268455964e-6` | 409 | 191 | 191 |
| `GYRE-zco.kt6.before.u` | `3.5817005120260779e-3` | `4.1657379212887286e-5` | `3.5722317366217761e-3` | 10361 | 7039 | 7039 |
| `GYRE-zco.kt6.before.v` | `1.2604558083863959e-2` | `3.5678782147365463e-5` | `1.2620542226610576e-2` | 9621 | 7479 | 7479 |
| `GYRE-zco.kt7.after.uu_b` | `1.9830559538162851e-7` | `1.0695948667645977e-7` | `1.7944533999139168e-7` | 253 | 327 | 327 |
| `GYRE-zco.kt7.after.vv_b` | `1.4977735369276976e-7` | `1.4349374444027306e-7` | `9.5211220686698076e-8` | 281 | 289 | 289 |
| `GYRE-zco.kt7.before.S` | `1.3330682220242807e-3` | `7.5376051000830557e-7` | `1.3332084358808061e-3` | 13859 | 3838 | 3425 |
| `GYRE-zco.kt7.before.T` | `3.1855998174336264e-2` | `1.1442120992910532e-5` | `3.1859709106175416e-2` | 14189 | 3811 | 3809 |
| `GYRE-zco.kt7.before.ssh` | `6.7956931858242572e-6` | `8.8270114146108899e-7` | `6.8765744583526925e-6` | 406 | 194 | 194 |
| `GYRE-zco.kt7.before.u` | `7.8155829836336378e-3` | `3.0582514325500076e-5` | `7.8144640850866844e-3` | 11926 | 5474 | 5474 |
| `GYRE-zco.kt7.before.v` | `4.2142807016476158e-3` | `2.1825903039044857e-5` | `4.2210284481331017e-3` | 11529 | 5571 | 5571 |
| `GYRE-zco.kt8.after.uu_b` | `1.6195036391092711e-7` | `1.3367887091427031e-7` | `1.1690827294148732e-7` | 316 | 264 | 264 |
| `GYRE-zco.kt8.after.vv_b` | `1.4934223134142541e-7` | `1.4235700537693421e-7` | `8.4215916099149128e-8` | 282 | 288 | 288 |
| `GYRE-zco.kt8.before.S` | `2.6539122494284584e-4` | `1.0231593563503338e-6` | `2.6543671366852095e-4` | 14494 | 3249 | 2903 |
| `GYRE-zco.kt8.before.T` | `7.3109445962309394e-3` | `9.4578707994230626e-6` | `7.312428259734105e-3` | 14883 | 3117 | 3116 |
| `GYRE-zco.kt8.before.ssh` | `1.0411528457225996e-5` | `6.398441143645392e-7` | `1.1051372571590536e-5` | 417 | 183 | 183 |
| `GYRE-zco.kt8.before.u` | `3.7893641214670377e-3` | `4.1569950654324339e-5` | `3.7936121307289572e-3` | 12738 | 4662 | 4662 |
| `GYRE-zco.kt8.before.v` | `2.4124980342377228e-3` | `3.5962302005912959e-5` | `2.4299388593012061e-3` | 12176 | 4924 | 4924 |
| `GYRE-zco.kt9.after.uu_b` | `1.1302896615634096e-7` | `9.8408899412618517e-8` | `6.2822867539244574e-8` | 332 | 248 | 248 |
| `GYRE-zco.kt9.after.vv_b` | `2.324981445752522e-7` | `2.2524999864306262e-7` | `6.6733983447705893e-8` | 251 | 319 | 319 |
| `GYRE-zco.kt9.before.S` | `2.8746065442675217e-4` | `8.60353651432888e-7` | `2.8740233189239461e-4` | 14629 | 3152 | 2874 |
| `GYRE-zco.kt9.before.T` | `8.5224492466018376e-3` | `7.0207176605663335e-6` | `8.5207644814424555e-3` | 14947 | 3053 | 3053 |
| `GYRE-zco.kt9.before.ssh` | `1.4383345478020948e-5` | `9.0001226450741115e-7` | `1.432010122422514e-5` | 441 | 159 | 159 |
| `GYRE-zco.kt9.before.u` | `2.0269816331771329e-3` | `2.9207515585017666e-5` | `2.0189026910879468e-3` | 12899 | 4501 | 4501 |
| `GYRE-zco.kt9.before.v` | `1.5707397392859292e-3` | `3.3940539035597073e-5` | `1.5751977751836642e-3` | 11600 | 5500 | 5500 |
| `GYRE-zco.kt10.after.uu_b` | `8.7336337465434427e-8` | `8.994843005063341e-8` | `3.8616853447812744e-8` | 345 | 235 | 235 |
| `GYRE-zco.kt10.after.vv_b` | `1.9001900318678377e-7` | `1.8475275007307293e-7` | `5.3474132103771034e-8` | 251 | 319 | 319 |
| `GYRE-zco.kt10.before.S` | `2.6118831114274599e-4` | `1.2956015993381698e-6` | `2.610928079320729e-4` | 15026 | 2779 | 2525 |
| `GYRE-zco.kt10.before.T` | `7.7397267823400284e-3` | `8.9813184160902892e-6` | `7.7369481754985259e-3` | 15287 | 2713 | 2713 |
| `GYRE-zco.kt10.before.ssh` | `1.8935128029820558e-5` | `6.6429214714679857e-7` | `1.898721693397061e-5` | 437 | 163 | 163 |
| `GYRE-zco.kt10.before.u` | `1.5136670604975304e-3` | `2.8633491321540007e-5` | `1.5064928224836297e-3` | 12663 | 4737 | 4737 |
| `GYRE-zco.kt10.before.v` | `5.8258010362488045e-4` | `3.2166251310261802e-5` | `5.9033626368157864e-4` | 11979 | 5121 | 5121 |

The headline rows make the cancellation visible:

| row | tip | candidate | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | same | AT-BAR retained |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR retained |
| kt2 U | `2.7377110452773967e-12` | same | first-over-bar unchanged |
| kt2 V | `3.284922138989399e-12` | same | first-over-bar unchanged |
| kt3 T | `1.627497246303733e-4` | `8.600419718618468e-7` | maximum improves, but 8,240 cells violate Rule 12 |
| kt3 S | `6.327735185607253e-6` | `6.97944244620885e-8` | maximum improves, but 5,824 cells violate Rule 12 |

The fixed day-30 member was deliberately not rerun: the requested experiment
was the full kt=1..10 ladder. The immutable tip reference remains day-30 T RMS
`1.2397011295506804e-2 K`; no candidate day-30 claim is made.

## Manifest, tests, review, and scope

The held manifest patch contains only the two-file measured candidate diff:
the WS source route and its private-selector-removal test. `git apply --check`
passes at the delivered documentation tip. The production worktree retains
the private causal hook; no candidate source line was landed.

The final clean-tree suite passed: `65 passed in 8.35s` across the Round-67
LDF-order, Round-46 stage-twin, phase-3 Rule-12, and receipt-citation test
modules. The receipt citation gate separately reports `PASS`: all eight cited
ranges are mapped, no citation fails, and every planted shift fires.

Independent review unavailable in-sandbox. The requested `codex exec
--sandbox read-only` command exited 1 with the verbatim terminal result:
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. Absence of an independent verdict is not approval;
the canonical Rule-12 gate already requires HOLD.

No new numerical claim is made for LOCK_EXCHANGE, OVERFLOW, DINO, or ORCA2.
DINO's regional cancellation warning remains explicit. ORCA2 remains
unmeasured for this candidate. No configuration, carried-state policy,
coefficient, timestep, stabilizer, freshwater pair, year harness,
reconciliation gate, #1484 guard, NEMO source, or NEMO executable changed.

Choices made: none. The user named the candidate, records, existing stage
harness, canonical Rule-12 gate, held disposition, and earlier-stage ownership
constraint.
