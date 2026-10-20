# Grid ladder rung L4 — round 1 receipt (gridladder_r53, Codex)

Date: 2026-10-20

Resume base: `bdc16249cbb0`

Implementation and independent recheck commit: `588c0e3da2affac10cf958e811a85bdb4c0d75a3`

Status: **L4 CERTIFIED.** The admitted operator record was not regenerated and
NEMO was not run in this session. After correcting the loader, production-step
coverage, expression-association coverage, validation guards, admission
passivity check, and this receipt, both score labels, all admitted depths,
wet/dry summaries, stages, and all ten geothermal-trend rows remain bitwise
exact. The fixes do not change L4 numerical values.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- preregistration commit: `50ea9550d0`
- acquisition implementation commit: `763a8625f3`
- operator pipefail correction commit: `1bd536e885`

## 1. NEMO source and remeasured artifacts

The executed deck selects `ln_trabbc=.true.`, `nn_geoflx=2`, and the variable
`heatflow` field at `ORCA2_OMIP_L4/EXP00/namelist_cfg:256,257,259,263`.
NEMO forms `rho0*rcp` and then its reciprocal at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:2504-2507`. The variable-file arm is
selected at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:228`, and its written
association is `(r1_rho0_rcp * heatflow) * 1.e-3` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:239-243`. The bottom-cell tendency is
then added using live Kmm thickness at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90:156-160`. Stage 3 calls tracer LDF
before BBC at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`.

These values were remeasured with this exact command:

```bash
sha256sum \
  /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe \
  /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2L4_BBC_R53/BLD/bin/nemo.exe \
  /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90 \
  /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz \
  /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/geothermal_heating.nc \
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2bbc/trabbc_round53.patch
stat -c '%s %n' \
  /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/geothermal_heating.nc
```

```text
e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7  ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
9ff4e9c05120f16412419542277bd27f0ed66bda8f21bab2dd73bdcafa1bf5ae  ORCA2L4_BBC_R53/BLD/bin/nemo.exe
65e0d2f111f675d08e1224f4336bd65b0f0911c151ea2ae8b306b66937581ac6  ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbc.f90
5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4  ORCA2_ICE_v5.0.0.tar.gz
e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb  geothermal_heating.nc
dd98dfca4051cec8dd4814edd71c3bc2717ad599af3a4a9b7e086495f2f68453  trabbc_round53.patch
116499 geothermal_heating.nc
```

The patch file is deliberately unchanged from the acquisition evidence. Its
first line is a stray conda error message. GNU patch ignores that leading
garbage, and the following check against the pinned base source returned 0:

```bash
task_tmp=$(mktemp -d /tmp/l4-patch-check.XXXXXX)
cp /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/trabbc.F90 "$task_tmp/trabbc.F90"
patch --dry-run -s -d "$task_tmp" -p0 < \
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2bbc/trabbc_round53.patch
```

```text
PATCH_DRY_RUN_RC=0
```

The old receipt's tarball and compiled-source hashes were incorrect and are
withdrawn. Its claimed successful preflight at `763a8625f3` is also withdrawn:
that source used the pipefail-unsafe `nm | grep -q` form, and the operator
reported status 141 in six of six attempts before fixing it in `1bd536e885`.
This session does not relabel that historical operator report as a new
measurement. On the current filesystem the old command happened to return 0
in six of six attempts, illustrating why the receipt now cites only the
current corrected admission output below.

## 2. Production wiring and expression identity

The G5 forcing route now reads `geothermal_heating.nc`, disables netCDF
mask/scale conversion, requires native float32, and widens with NumPy
float32-to-float64 conversion. Across the file's 183 distinct values, direct
widening has zero unequal values; decimal `.8g` re-parsing changes 170.

The NEMO-named helper now requires both `rho_0` and `c_sw`. The association
test uses the card constants 1026.0 and 3991.86795711963 and all 183 distinct
file values. The mutation `r1*(q*1e-3)` differs on 57 values and fails the
test; the former 1025/3994 single-value test at 86.4 could not distinguish it.

The production-step test loads the real file through the G5/L4 route, runs an
actual JIT-compiled model step with real flux and zero flux, and observes:

- S, u, v, and ssh are bit-identical;
- all 16,428 wet bottom cells with nonzero heat flow warm;
- dry cells remain zero-difference;
- column heat gain divided by `dt*flux` ranges from 0.9999999892843886 to
  1.0000000102528694, within relative tolerance 1.1e-8.

Deleting the stage-3 geothermal block made this test fail with 0 rather than
16,428 warmed acting cells. Mutating the helper association made the
57-of-183 assertion fail. Removing either required helper argument makes its
TypeError test fail. These mutations were temporary and were restored before
commit.

## 3. Latent-order guards

The selected geothermal path now refuses two combinations whose NEMO order is
not implemented: G9's written stage-3 tracer-Krhs list (which otherwise drops
BBC), and tracer lateral diffusion via `K_h`, `K_bih`, or GM/Redi (which would
otherwise place diffusion after BBC). The validation test constructs both
formerly accepted configurations and requires rejection. L4/G5 selects
neither, so this validation-only change leaves its values unchanged.

## 4. Existing-evidence admission and passivity

`--admit-existing` now runs the preregistered G5 passivity validator: the three
instrumented twins' restart files equal the writer-free reference at matching
steps. A one-ULP plant must fail. The existing evidence passed without running
NEMO:

```bash
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2bbc/run.sh --admit-existing
```

```text
L4_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=false
L4_PREFLIGHT_OK commit=588c0e3da2affac10cf958e811a85bdb4c0d75a3 reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L4_BBC_R53
L4_GEOTHERMAL_RECORDS_PASS steps=1..10 sha256=0b184a56aca2c3c7c871cc545a6df057f7318c09d2695f8c412ac34712c93ab4
PASS_G5_ADMISSION unequal_rows=None
L4_WRITER_PASSIVITY_PASS reference=ref_100step twins=A/B/C plant=fired
L4_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round53/oracle_l4_r53
```

The producer-list file was remeasured with:

```bash
sha256sum /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round53/oracle_l4_r53/manifest/producers.sha256
```

```text
d666a29a4accc44ec6acf6582d2e15fadd81cd8b6f18e27a771c33039e69394d  producers.sha256
```

## 5. Independent scoring recheck

The scorer was rerun from the clean implementation commit with CPU x64 and
the repository PYTHONPATH used by `claude_rounds/land.sh`. INDEPENDENT used
kt 1--10 and all three restart roots. GIVEN-NEMO-ENTRY used the preregistered
entries `1,3,4,5,6,7,8,9,10`, so kt 2 is the two-step result from kt 1.
Stage scoring covered kt 1 stages 1--3. Geothermal scoring covered kt 1--10.

| label / record | coverage | exact | unequal |
|---|---:|---:|---:|
| INDEPENDENT surface | kt 1--10 | 70 / 70 | 0 |
| INDEPENDENT full/partial/dry depth | restarts kt 2--10 | 36 / 36 | 0 |
| GIVEN-NEMO-ENTRY surface | direct + two-step outputs kt 1--10 | 119 / 119 | 0 |
| GIVEN-NEMO-ENTRY full/partial/dry depth | every available restart row | 64 / 64 | 0 |
| stage surface | kt 1, stages 1--3 | 21 / 21 | 0 |
| stage full/partial/dry depth | kt 1, stages 1--3 | 12 / 12 | 0 |
| geothermal wet-bottom trend | kt 1--10, 16,433 wet columns each | 164,330 / 164,330 cells | 0 |
| geothermal dry slots | kt 1--10 | all canonical zero | 0 |

Each geothermal step has 16,428 nonzero-flux wet columns and five zero-flux
wet columns. Every maximum absolute difference is exactly 0.0.

The exact score commands used `python
scripts/validate/ocean_fidelity/testcases/nemo_testcase_g2_orca2bathy_ladder.py`
with `--case ORCA2L4-bbc`, `--kt-from 1 --kt-to 10`, and output paths named
below. Their arm-specific arguments were:

```text
--arm independent --root twin_a --restart-root twin_b --restart-root twin_c
--arm given_entry --root twin_a --restart-root twin_b --restart-root twin_c --entries 1,3,4,5,6,7,8,9,10
--arm stages --root twin_a --restart-root twin_b --restart-root twin_c --kt-from 1 --kt-to 1
--arm geothermal --root twin_a --restart-root twin_b --restart-root twin_c
```

The output hashes were measured with `sha256sum` on the four final files:

```text
d1d6ea623f8287bdfa6879e4a6bdff873c8e920326c45ca5f5e73117fc5a905f  final_recheck_independent.json
1fee694cb263f1dd4df2a9ae29dfec245aeb1dff4853c4da6b09a4ba89e7011f  final_recheck_given_entry.json
7dbe3e8a6a3d9b7d19964a2245e074f33855bd938c4b7c41b86fb81f87e90b8c  final_recheck_stages_kt1.json
27619deb2226791a35cfdd145ae38a5999e74ca921ba2173f5916ec3c81b51c0  final_recheck_geothermal.json
```

After recursively deleting only provenance/timing keys
(`legoesm_git_sha`, `packages_tree`, and `seconds`) and canonicalizing with
`jq -cS`, old and final scientific payload hashes matched pairwise:

```text
independent old/final c990bd7a289d49ab8ff98cfbdbe598b3db88956b53204ffd360f32a8483fe54e
given_entry old/final 3394c326d3eac4b977160cb26a0ea250ea9bfb7fdb79505bfeb918ab89aaaa03
stages_kt1 old/final f59828ce40364f7bbb8100d7b0bb63a11ff600e3fe8e88d15be086bc5e6ecf7a
geothermal old/final f6498933ea9dbe6a9ad95e2af956eb207196055ca54103cb6b6cd47c28f835a1
```

## 6. Tests, gates, and control scope

The focused CPU/x64 suite passed 31 / 31. Its new tests bind exact binary
widening, association, required constants, both order guards, existing-record
passivity plus plant, the real production model step, and deletion of the
actual stage-3 model block. The direct receipt citation gate and its 17-test
suite pass with all citations mapped; the private-import ratchet passes 6 / 6.
The prescribed default gate passes 142 / 142 when split at the documented
compiler limit: citation 17, private imports 6, TKE/recipe/freshwater 109,
stage-face-mask 2, and prognostic barotropic state 8.

The preregistered full-card JAXPR controls for ORCA2, DINO, GYRE, SMT, VORTEX,
OMT, rung 0, and LOCK_EXCHANGE were **not measured in this resume**. The
preregistered G1--G9 two-step value controls were also **not measured in this
resume**. The prior receipt's inference that they were unchanged from static
flag checks is withdrawn. This is an explicit unmeasured-control declaration,
not a numerical claim. L4 itself was remeasured as described above and its
scientific score payloads are unchanged.

ASKED and completed: measure every review hypothesis before fixing; correct
receipt measurements; load and exactly widen float32 heat flow; bind the
production model step and association with mutation-sensitive tests; require
card constants; reject the two latent orders; admit existing evidence with
restart passivity and its plant; rerun both score labels, full admitted depth,
wet/dry, stages, and geothermal trend; run citation/default/private gates; and
certify only after independent recheck.

UNASKED and not done: run NEMO; alter or regenerate acquisition evidence;
change `trabbc_round53.patch`; fetch or vendor data; relax a bar; push.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r53/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L4 CERTIFIED: all admitted trajectory, full-depth, stage, wet/dry, and geothermal-trend rows are bit-identical
