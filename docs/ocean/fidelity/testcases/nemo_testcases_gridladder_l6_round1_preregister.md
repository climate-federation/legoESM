# NEMO testcase grid ladder L6 — round 1 preregistration

Date: 2026-10-21

Starting commit: `43ac6db0c8ff`; L4b is certified at this commit.  Status:
**PREREGISTERED BEFORE L6 IMPLEMENTATION, MODEL-VALUE MEASUREMENT, OR ORACLE
ACQUISITION.**  Decision 146 defines L6 as L4b plus ORCA2's interior T/S
damping, with no BBL on this rung.  This round may build and test the card and
prepare a WRITE-only acquisition, but Codex may not execute NEMO or score L6.

## Frozen survey

The exact ORCA2 deck selection is `ln_tradmp=T`, `nn_zdmp=0`, and inherited
`cn_resto='resto.nc'` (`ORCA2_OMIP_L4/EXP00/namelist_cfg:326-330` and
`ORCA2_OMIP_L4/EXPREF/namelist_ref:1012-1019`).  The damping target is the
same monthly potential-temperature and salinity climatology already used by
G3 and above: `votemper` and `vosaline`, cyclic yearly, with record-time
interpolation enabled (`ORCA2_OMIP_L4/EXP00/namelist_cfg:44-55`).

The executed source initializes `tra_dmp`, reads the target fields, and reads
`resto` from `cn_resto` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:256-310`).
At each call it updates every real level and both tracers as
`Krhs = Krhs + resto * (target - Kbb)` for `nn_zdmp=0`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:142-195`).  `dta_tsd` obtains the
two surrounding climatology records through `fld_read`, applies ORCA_R2's
hand alterations, and masks the copied fields
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:186-263,307-310`); `fld_read`
forms the before/after weights from the model clock and preserves the written
multiply-add order (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/fldread.f90:200-246`).

The stage-3 order is `tra_ldf`, `tra_bbc`, `tra_bbl`, then `tra_dmp`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`).  L6 selects LDF,
BBC, and DMP; BBL remains off.  Therefore the production source must
materialize certified L4b first and then damping as the final selected
stage-3 write.

`resto.nc` is not part of the ladder's inherited 16-file acquisition set.  It
is already present in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` and in its existing
extraction; no network fetch or vendoring is permitted.  The frozen SHA-256 is
`f4b6aec4695f984eab3a3ad9f2efdcf80d19180cf5f55aa4a93e3f45639b8634`
for both the extracted file and the streamed tar member
`ORCA2_ICE_v5.0.0/resto.nc`.  The archive SHA-256 is
`5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4`.
The file contains one binary64 `resto(z=31,y=148,x=180)` variable; only the
30 real levels consumed by NEMO may enter the card, transposed to the model's
`(y,x,z)` layout.

legoESM already owns both required mechanisms.  The climatology reader and
ORCA_R2 alterations are centralized in
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1418-1490`,
and the shared full-rank tracer-relaxation kernel is
`packages/ocean/legoesm/ocean/dynamics/ocean_tendency_common.py:654-721`.
L6 must extend those paths; it may not add a second damping formula or a
second climatology implementation.

## Frozen L6 measurement and exact controls

The card name is `ORCA2L6-dmp`.  Its deck is L4b's rendered deck plus exactly
the resolved `namtra_dmp` block above.  EIV, BBL, tracer advection, and altered
vertical mixing remain off.  Only L6 may select a new default-false stage-3
damping-order flag.  The validator must refuse damping without certified L4b,
missing/full-rank-incompatible forcing, BBL, EIV, and any attempt to let the
ordinary stage-10c sponge path apply the same source a second time.

The exact one-step discriminator is the live stage-3 source.  For both T and
S, L6 must equal
`nemo_source_round(L4b_source + resto * (target - Kbb))` bit for bit.  A
rounding-sensitive planted operand makes moving damping before geothermal
fail.  The gate runs JIT-compiled, takes state, target, and restoring rate as
traced binary64 inputs, and checks finite gradients.  The new implementation
introduces no division or square root and therefore no new zero-operand
gradient site.

Before and after implementation, the committed merge-identity probe compares
two production steps of every inherited card with its frozen owner snapshot.
Every state leaf and program fingerprint must be identical; a planted bit or
selector mutation must fail.  L6 is excluded from this preservation set.

CONFIRM means: exact deck delta; exact L4b-plus-damping source association;
nonzero damping on at least one defined wet cell; zero movement in every
inherited card; every unsupported-selection and wiring-removal plant fires.
REFUTE means any missing/extra deck assignment, any unequal exact source word,
any inherited-card movement, any undefined-cell write, or a surviving plant.
No tolerance, trajectory score, or post-hoc substitute is allowed.

## Frozen acquisition boundary

The committed entry point will live at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l6_orca2dmp/run.sh`.
Its default is the seven-item read-only preflight; `--run` is operator-only;
`--admit-existing` revalidates an operator record.  The deck is L4b plus only
damping.  Input staging adds the SHA-pinned existing `resto.nc` to L4b's
inputs.  The operator plan is a two-step acting smoke, a writer-free boundary
run, and three independently staged twins to one common endpoint no greater
than ten steps.  Observers may write only cells NEMO defines.

The preflight must execute whitespace-tolerant numeric/logical checks against
the committed excerpt of a real `ocean.output` and against a STAMPED rendered
L6 deck.  `check_acting` runs on the smoke only and as its own statement.  The
script may contain no `cmd | grep -q` under `pipefail`, and preflight scratch
is removed by an `EXIT` trap.  Each of the seven items carries a planted
violation that must fire.

Frozen outcome: invoke `--run`, prove that the operator boundary stops before
NEMO execution, and report `ACQUISITION_NEEDED`.  Do not score or certify L6.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| survey and SHA-pin on-disk `resto.nc` | ASKED | verified locally; never fetch/vendor |
| add L4b + interior damping | ASKED, Decision 146 | one L6-only ordered card |
| enable BBL on L6 | forbidden | remains off |
| alter inherited card programs or values | forbidden control | exact two-step identity |
| execute NEMO in this sandbox | operator-only | never |
| score or certify L6 | forbidden this round | no admitted record exists |
