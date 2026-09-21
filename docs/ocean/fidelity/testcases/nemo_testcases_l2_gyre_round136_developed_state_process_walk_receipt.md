# NEMO testcase L2 GYRE phase 3 — round 136 developed-state process walk receipt

Date: 2026-09-21

Incoming tip: `dda3f3257dad5a7f86e1177f934afc531e6567d9`

Measurement commit: `98ea0010e411d01ed1b4436f7a6f2232de8e1024`

Status: **HELD — the frozen advection prediction is REFUTED. With NEMO's
bit-exact day-180 entry temperature, the first recorded non-bit boundary is
already the free-surface geometry term, before tracer advection: 17,975 of
18,000 wet cells, maximum `2.7704061267286306e-11 K`. The entry `q_Kbb` is
BIT, but the developed `q_Kmm` and `q_Kaa` ratios differ in the same 599 wet
columns. The first recorded non-bit compiled statement is therefore the
T-point QCO write fed by the external solve, not an FCT statement. Its
`pssh` input is not recorded at compiled-operator resolution, so this round
does not claim that the ratio multiplication is wrong and lands no physics,
configuration, or carried state.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round136/`

## Outcome first

Round 136 drove one full `LatLonCGridOceanModel.step` through production JIT
from the admitted NEMO restart at completed step 1080 and compared the stage-3
temperature boundaries against NEMO's recorded step 1081. Every restart-backed
state leaf was mapped, the entry temperature was bit-identical in all 18,000
scored wet cells, and the diagnostic returned state was byte-identical to a
separately compiled ordinary production call.

The preregistered expectation that geometry would be BIT and combined
advection would be first non-bit is false. The geometry boundary is non-bit
before `CALL tra_adv`. Splitting its three recorded QCO operands gives:

| QCO operand | wet columns unequal | max absolute | RMS | verdict |
|---|---:|---:|---:|---|
| `q_Kbb` | 0 / 600 | `0` | `0` | BIT |
| `q_Kmm` | 599 / 600 | `6.253886297713507e-13` | `2.0334388882559797e-13` | non-bit |
| `q_Kaa` | 599 / 600 | `1.2506662372402388e-12` | `4.066795963906242e-13` | non-bit |

The approximately two-to-one `q_Kaa`/`q_Kmm` scale is consistent with the
compiled half-step construction, but it is not promoted to an ownership proof.
The after-level SSH consumed by the QCO write is the output of the same step's
external solve; this record does not bracket that solve statement by statement.

## Record availability: four requested days, one measurable entry

The Round-123 process stream really contains only steps 1081–1440. It records
stage-3 temperature `Tbb`, `r3t(Kbb/Kmm/Kaa)`, four cumulative `Krhs` writes,
and `Taa`; it does not contain salinity process terms, per-axis
`ttrd_xad/yad/zad`, FCT coefficients, or process frames near days 30 and 90.
The compiled writer declares the 1081–1440/stage-3 window and writes exactly
those geometry operands at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`.

| requested day | exact entry | process frame | result |
|---:|---:|---:|---|
| 30 | step 180 | absent | `UNAVAILABLE_BY_RECORD` |
| 90 | step 540 | absent | `UNAVAILABLE_BY_RECORD` |
| 180 | step 1080 | step 1081 | `MEASURED` |
| 240 | step 1440 | step 1441 absent | `UNAVAILABLE_BY_RECORD` |

Frame 1440 begins from an unrecorded full step-1439 state, so it was not
relabeled as a day-240 equal-input step. The day-180 trends required by the
operator's acquisition exception exist; no new acquisition is requested.

The daily record was freshly admitted at the measurement commit: 360/360
boundaries, all 18 required restart variables, and 12/12 monthly overlaps
bit-identical. NEMO swaps the accepted state into `Nbb` before the restart
write at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:220-229`, invokes the
writer at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260`, and writes
the tracer, velocity, SSH, and separate `ssha` values at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184`. The next
restart load restores the distinct RK3 after slot at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:354-370`.

## Complete compiled-order temperature table

Every row scores 18,000 wet temperature cells. “Cumulative” is the value at
the recorded boundary; “term” is that boundary minus its predecessor in the
compiled sequence.

| boundary | cumulative unequal | cumulative max abs (K) | cumulative RMS (K) | term unequal | term max abs (K) | term RMS (K) |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 17,975 | `2.7704061267286306e-11` | `5.800694710939682e-12` | 17,975 | `2.7704061267286306e-11` | `5.800694710939682e-12` |
| combined advection | 17,909 | `4.542535023688288e-7` | `1.0994255449824732e-8` | 18,000 | `4.542699869602984e-7` | `1.0994767489428778e-8` |
| surface boundary | 17,909 | `4.542535023688288e-7` | `1.0994255544793108e-8` | 569 | `8.881784197001252e-14` | `4.1177392215656916e-15` |
| shortwave | 17,912 | `2.468036477409896e-6` | `2.5164072570739504e-7` | 9,660 | `2.4680393178044824e-6` | `2.513690755150022e-7` |
| lateral diffusion | 18,000 | `3.5621416268050154e-4` | `1.0657752615890357e-5` | 18,000 | `3.5621425651477523e-4` | `1.0655217366759954e-5` |
| vertical diffusion | 18,000 | `7.588040097612492e-4` | `1.9243952894431772e-5` | 17,993 | `6.652851978756757e-4` | `2.1834089071459892e-5` |

The source order is mechanically anchored. NEMO clears `Krhs`, calls
`tra_adv`, writes the first cumulative RHS, then calls and writes the surface
boundary row at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
Shortwave and lateral diffusion are accumulated and recorded at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`;
`tra_zdf` produces and records `Taa` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.

## First recorded non-bit statement and the ownership boundary

The compiled stage program saves the external-mode after SSH as `ssha` and
passes it to `dom_qco_r3c_RK3` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:150-180`.
Inside that compiled routine the T-point statement is
`pr3t = pssh * r1_ht_0` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90:237-257`.
Stage 2 constructs the half-step ratio from entry plus after, and stage 3
places the full `r3ta` in `Kaa`, at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:223-255`.

Those statements explain the measured layout: exact `q_Kbb`, non-bit
`q_Kaa`, and a half-sized non-bit `q_Kmm`. They do **not** prove the multiply
is mistranscribed. The statement's `pssh` operand is an unrecorded output of
the developed-flow external solve. The calibrated conclusion is therefore:

* first recorded non-bit statement: T-point QCO write `pr3t = pssh*r1_ht_0`;
* first demonstrated non-bit operand: the after-level QCO ratio in 599 wet
  columns;
* arithmetic owner: **OPEN upstream at or before `pssh`**;
* advection/FCT owner claim: **REFUTED for this step**, because geometry is
  already non-bit.

No downstream tracer candidate can land from this record until the developed
external-mode `pssh` path is bracketed.

## Active-branch census

The branch maps are spatial diagnostics against the first unequal geometry
cells. These operators execute later, so overlap is **not causal attribution**.

| family | NEMO | legoESM on NEMO entry | overlap with 17,975 geometry-unequal cells |
|---|---|---|---:|
| FCT nonosc limiter | `UNMEASURED`: Round-123 stores no coefficients | 502 active wet cells | 502 (2.7928%) |
| EVD replacement | 1,123 projected active cells | 1,123; zero interface-selection disagreements | 1,122 (6.2420%) on each side |
| TKE floors | diffusivity only: 9,973 interfaces / 10,731 projected cells; viscosity and energy unmeasured | diffusivity 9,973, viscosity 9,973, post-solve energy 15,058 interfaces; union 16,025 projected cells | NEMO partial 10,717 (59.6217%); legoESM union 16,001 (89.0181%) |

The compiled FCT routine computes its beta branches at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:849-876` and applies
the sign-selected `MIN(1, beta...)` coefficients at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:905-931`.
Because its coefficients were not recorded, the table deliberately labels
NEMO FCT activity unmeasured rather than treating the model's 502-cell map as
NEMO evidence.

The admitted vertical record resolves NEMO's EVD condition and replacement at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`, the
post-solve energy floor at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:473-474`, and the
viscosity/diffusivity floors at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`. Only `avt`
is present in the NEMO vertical record, hence the explicitly partial NEMO TKE
row.

## Frozen prediction ledger

* **P0 CONFIRMED.** The process record readmitted all 360 frames, steps
  1081–1440, and the availability registry contains exactly days 30, 90, 180,
  and 240 with only day 180 measurable. The raw count in P1 saying 17,400 wet
  T cells was a preregistration error: the certified T mask has 18,000; the
  exactness prediction itself is confirmed on all 18,000.
* **P1 PARTLY REFUTED.** Entry T is BIT and the ordinary/observer returned
  states differ in zero bytes. Geometry BIT is **REFUTED**: 17,975 unequal
  cells at `2.7704061267286306e-11 K` maximum.
* **P2 REFUTED as to owner.** Combined advection is not first. FCT activity
  and nonzero spatial overlap are confirmed, but cannot own a boundary that
  is already non-bit. EVD and TKE-floor activity are confirmed.
* **P3 CONFIRMED.** The complete registries and all plants fired. No normal
  card, physics statement, configuration, or carried state changed.

## Production trajectory and campaign surfaces

The private observer is default-off. Its larger production-JIT return graph
moved zero bytes of the ordinary returned state and zero cells in every
previously admitted process/vertical observer row. A fresh certified ladder
at the measurement commit remained `DEBT` for the campaign's existing rows
and reproduced the current headlines:

| row | value | status |
|---|---:|---|
| kt2 T | `1.4210854715202004e-14 K` absolute | AT-BAR |
| kt2 S | `2.1316282072803006e-14 g/kg` absolute | AT-BAR |
| kt2 U | `2.7377110452773967e-12 m/s` | first-over-bar |
| kt2 V | `3.2849219221489645e-12 m/s` | first-over-bar |
| kt3 T | `8.659373840202989e-7 K` absolute | DEBT |
| kt3 S | `7.027291104577671e-8 g/kg` absolute | DEBT |

The 30-day control reproduced Round 135's day-30 values in all five saved
fields. T3D RMS is `6.890431487825909e-5 K`; S, u, v, and SSH RMS are
`1.178122248084361e-5 g/kg`, `5.580098025916742e-6 m/s`,
`4.620078091147412e-6 m/s`, and `6.8027817620663075e-6 m`.
No first-over-bar row moved, no kt1 at-bar row left the bar, and there is no
candidate/year arm to assess under Decisions 43 and 45.

| campaign surface | disposition |
|---|---|
| GYRE | diagnostic only; ladder and day-30 control measured |
| DINO | shared production implementation unchanged; private hook absent |
| LOCK_EXCHANGE / OVERFLOW / tanks | shared production implementation unchanged; private hook absent |
| ORCA2 | `UNMEASURED-WITH-SPEC`; no ORCA2/SI3 selector or card touched |
| configuration / carried state | none changed |

## Evidence, controls, and verification

Authoritative artifacts and SHA-256 digests:

| artifact | SHA-256 |
|---|---|
| `developed_step_walk.json` | `926558f289eed794ddd9fa0bba4ec1a7c476c19b97becdb10a00a380fa5811db` |
| `developed_branch_maps_98ea0010e411.npz` | `7547fb5212d697ce514d5b8f2a65f5aecb09f48a7a10c8cece0da8c9d496ac7d` |
| `daily_record_audit.json` | `6d43332d98ad2852547b975bc7fbda6f3bdd836fe22a3027f8df711fb1361d5c` |
| `entry-temperature-ulp_plant.json` | `ad694120c4d792e5cb21872efeeab20401456b2c79de498e896c156e8abc671d` |
| `ladder.json` | `d5e9ec81fc3e7962c36702680b884203e001bcbce4061065e49d9e94abd05418` |
| `day_gap.json` | `61960524951a23e7d26b16891a39298bf39479bb503c89eb81e83588ef5c6f96` |

The earlier `developed_branch_maps_72645fc86fbc.npz` came from the pre-operand
split measurement commit. It is retained under the no-delete rule, is
byte-identical to the authoritative map, and is superseded by the stamped
`98ea0010e411` artifact.

Every planted violation exited nonzero and printed `STATUS PLANT-FIRED`:

| plant | decisive result |
|---|---|
| missing requested day | availability registry incomplete |
| missing process row | process registry incomplete |
| missing branch family | branch registry incomplete |
| process stamp | producer commit mismatch |
| truncated process record | 1,415,292 bytes vs 1,415,300 expected |
| raw surface-RHS ULP | raw increment changed; decoded row correctly remained below resolution |
| effect-scale surface RHS | one decoded surface-boundary and shortwave cell moved |
| entry-temperature ULP | geometry/advection/SBC/QSR/ZDF boundaries moved; command exited 1 |

The final clean-tree receipt citation pass found all 16 mapped citations with
zero failures or unmapped citations. Shifting the compiled QCO citation by two
lines exited 1 with `SYMBOL-NOT-AT-LINE`. The final normal and planted reports
are `citation_gate_final_receipt.json` and
`citation_gate_final_receipt_shifted_plant.json` under the evidence root.

Focused and broad test summaries are recorded below after the final clean-tree
pass:

* developed-state/year-owner gate: `30 passed in 30.71s`;
* FCT advection unit file: `27 passed in 26.19s`;
* receipt citation unit file: `16 passed in 1.95s`;
* the first combined year-owner/advection invocation reported `1 failed, 56
  passed`; its only failure was the pre-existing `1.7826e-10` versus `1e-10`
  conservation threshold, and the advection file then passed alone as the
  preceding `27 passed` line records;
* a clean-tree rerun of the 16 fidelity tests that had correctly refused an
  uncommitted receipt reported `72 passed in 17.05s`;
* the worktree-stamp/RK3/MPAS check reported `1 failed, 2 passed, 1 warning in
  93.38s`: the stamp ratchet and named RK3 test passed; the unrelated MPAS
  bottom-drag test retained its float32/float64 CG-carry mismatch.

The required combined `tests/ocean/fidelity tests/ocean/unit -n 12` sweep was
run once. It reached 94% but nine workers aborted in JAX compilation and the
xdist parent stopped making progress; interruption emitted no summary line.
A second fidelity-only `-n 4` sweep reached 95% and likewise emitted no summary
before its remaining long-running gates were bounded. This is the exact
large-suite compiler-limit failure mode warned about in `AGENTS.md`, so the
run was split into fresh processes rather than assigned a false pass/fail
verdict.

The completed 20-file split invocations emitted these exact summary lines:

| group | exact pytest summary |
|---:|---|
| 01 | `348 passed, 1 skipped in 19.19s` |
| 02 | `155 passed, 4 skipped in 85.51s` |
| 03 | `7 failed, 192 passed in 145.21s` |
| 04 | `5 failed, 202 passed in 173.41s` |
| 05 | `6 failed, 151 passed, 1 skipped in 52.02s` |
| 06 | timeout after ten minutes; no summary emitted |
| 07 | `130 passed, 2 skipped in 17.14s` |
| 08 | `4 failed, 279 passed, 1 xfailed, 1 warning in 275.01s` |
| 09 | `1 failed, 293 passed, 1 xfailed, 4 warnings in 539.82s` |
| 10 | `1 failed, 304 passed, 1 skipped in 281.79s` |
| 11 | timeout after ten minutes; no summary emitted |

Sixteen of the group-03/04/05 failures were the intended dirty-tree stamp
refusal; after the receipt was committed those exact tests are the `72 passed`
clean-tree line above. Of the remaining completed-split failures, five match
the supplied 87-node baseline exactly (the four float32 advection-gradient
cases and baroclinic decomposition). The SI3 scalar-math failure is the later
operator-registered pre-existing `A MY_SRC is not verbatim` debt. The sole
failure absent from that old list is
`test_mpas_bbl_keeps_thin_partial_cell_run_stable`; an isolated rerun reproduces
it in `barotropic_implicit_mpas` before any changed Round-136 path executes.

The timeout groups were then run one file per fresh process. Files 101--120
emitted, in order:

```text
2 passed in 0.27s
3 passed in 0.27s
3 passed in 0.80s
5 passed in 0.58s
18 passed in 0.45s
timeout after 180s; no summary (OVERFLOW barotropic gate, four tests completed)
10 passed in 9.24s
3 passed in 6.35s
8 passed in 0.30s
timeout after 180s; no summary (phase-3 stage-sweep gate, one test completed)
9 passed in 0.34s
16 passed in 2.02s
21 passed in 1.42s
10 passed in 7.39s
8 passed in 0.47s
50 passed in 27.32s
17 passed in 3.88s
8 passed in 65.20s
7 passed in 0.46s
5 passed in 0.40s
```

Files 201--220 emitted, in order:

```text
8 passed in 0.13s
9 passed in 14.37s
2 passed in 0.09s
5 passed in 0.34s
4 passed in 6.87s
4 passed in 0.09s
6 passed in 0.69s
25 passed in 0.39s
28 passed in 0.41s
3 passed in 0.21s
5 passed in 0.34s
4 passed in 2.52s
timeout after 180s; no summary (CORE2 corrected-cache file, four tests completed)
4 passed in 1.17s
5 passed in 3.19s
19 passed in 9.00s
16 passed in 66.15s
10 passed, 2 warnings in 49.31s
39 passed in 2.16s
31 passed in 17.68s
```

Finally, the nine selectors interrupted by the combined xdist run were all
rerun in fresh processes. Seven correctly named invocations emitted
`1 passed in 6.46s`, `1 passed in 12.37s`, `1 passed in 7.74s`, `5 passed in
3.59s`, `1 passed in 1.22s`, `1 passed in 43.32s`, and `1 passed in 75.35s`.
Two initially incomplete node selectors each emitted `no tests ran`; their
class-qualified correction emitted `2 passed in 66.32s`. Thus all nine
interrupted selectors pass outside the exhausted workers. The full-tree sweep
is honestly recorded as infrastructure-incomplete, not green; every test that
directly exercises this round's changed paths is green.

After measurement, the source statement that constructs one flux pair was
consolidated from a parenthesized multiline assignment to the same one-line
assignment. This was an AST-neutral formatting change made solely to preserve
the citation gate's rigid historical source extent; no measurement code path
or arithmetic expression changed.

## Independent adversarial review

The required separate command was invoked, but the read-only sandbox prevented
the in-process app-server client from initializing. Its output was, verbatim:

```text
Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review was unavailable in-sandbox; the command emitted no verdict.

## OPEN — round 137

1. Stay upstream of tracer advection. Reuse the developed step-1081 entry and
   walk the external-mode program that produces stage-1 `ssha`/`r3ta` in
   compiled order. The first target is the earliest boundary at which the 599
   wet-column `q_Kaa` set appears. Search the existing barotropic records
   first; if no developed-state operand stream brackets it, preregister a
   passive day-180 external-step acquisition rather than guessing from the
   final ratio.
2. Compare `pssh` itself before testing `pssh*r1_ht_0`. Only if `pssh` is BIT
   may the QCO multiplication become an arithmetic candidate. If `pssh` is
   already non-bit, continue upstream through the external solve; do not land
   a ratio rewrite.
3. Do not continue the FCT/EVD/TKE walk while the geometry boundary is owned
   upstream. Their developed branches are active, but their overlap is not an
   ownership measurement.
4. Days 30, 90, and 240 remain unavailable for equal-input process ordering.
   Do not interpolate them or relabel frame 1440. A future four-day comparison
   requires a separately preregistered record.

`ACQUISITION_NEEDED`: `NONE`

`DECISION_NEEDED`: `NONE`

`ROUND_STATUS`: `HELD`
