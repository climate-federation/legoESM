# Grid ladder rung G9 — round 7 acquisition receipt (gridladder_r52)

Status: **STOPPED FOR RECORD.**  This round adds only the operator-executed
acquisition requested by G9 round 6.  No NEMO build or run was performed in
the agent session.

- codex review: operator
- independent review: operator (Claude)
- acquisition: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run_r52_trazdf_walk.sh`
- evidence destination: `gridladder_rounds/round52/oracle_g9_r52/`

## 1. Exact acquisition boundary

The new script defaults to read-only `--preflight-only`.  The operator runs
`--run` to build the new config `ORCA2G9_OMIP_L4_R52TRAZDF` from the pinned
round-46 producer base and to execute the committed G9 deck for two steps.
`--admit-existing` reruns admission without building or running NEMO.

The only source change is an additive, WRITE-only observer in `tra_zdf_imp`.
At kt 1, stage 3, it records the top-level content immediately after the
assignment and the level 2 through `jpkm1` scalar `zrhs` before the forward
recurrence consumes it.  These are the compiled assignments at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290`.

The resulting file is exactly
`oracle_g9_trazdf_zrhs_kt00000001.bin`.  Its 16-byte magic is followed by
13 native integers: version, kt, Kbb, Kmm, Krhs, Kaa, jpi, jpj, jpk, halo,
real precision in bits, frame count, and array count.  Four 16-byte-tagged
frames follow:

1. `zrhs_content`: the two requested T/S three-dimensional content arrays;
2. `matrix_diags`: lower, diagonal, upper, and factored diagonal arrays;
3. `forward_state`: T/S after the second recurrence;
4. `solution`: final T/S after the back substitution.

All arrays use NEMO's native `wp` stream layout.  The observer never reads a
captured value back into model state.

## 2. Admission and controls

Admission fails unless the run completed two steps, exactly one new record is
present, all four frames are read completely, every value is finite, and no
trailing byte remains.  It also requires the 42 kt-1/kt-2 records from round
46 to be present in the candidate and byte-identical to their admitted source.

Preflight compiles and runs a small Fortran writer using the record's exact
magic, 13-integer header, 16-byte frame tags, and array order, then feeds that
file to the production admission reader.  This is the direct guard against
the round-51 15-byte-magic/16-byte-tag mistake.

The missing-frame plant truncates the synthetic `solution` frame; the reader
refuses it with `missing frame solution`.  The non-additive-patch plant changes
one original NEMO call; the source differ rejects the removed line.  Both
plants exit nonzero before the preflight success marker.

## 3. Read-only preflight result

The default preflight ran from committed acquisition commit
`320bba0140caba5a1ef41b50441a219c1fb95d82` and printed:

`G9R52_PREFLIGHT_OK base=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 target=ORCA2G9_OMIP_L4_R52TRAZDF commit=320bba0140caba5a1ef41b50441a219c1fb95d82`

It also printed `TRAZDF_OBSERVER_ADDITIVE_OK`,
`G9R52_NONADDITIVE_PATCH_PLANT_REFUSED`,
`G9R52_TRAZDF_RECORD_OK profile=synthetic frames=4 finite_arrays=10 bytes=1092`,
`G9R52_MISSING_FRAME_PLANT_REFUSED`, and
`G9R52_LAYOUT_SELFTEST_OK magic_bytes=16 frame_tag_bytes=16 arrays=10`.

## 4. Gates and disposition

The receipt-specific citation gate passes with 1 citation, 0 failures, 0
unmapped citations, and 0 failing map entries.  The default gate was split to
respect the foreground limit: citation, private-import, TKE, recipe, and
freshwater tests **132 passed in 444.39 s**; face-mask tests **2 passed in
565.81 s**; prognostic barotropic-state tests **8 passed in 70.43 s**.

ASKED: write the committed preflight / `--run` / `--admit-existing`
acquisition, capture the round-6 section-3 content record and downstream solve
boundaries, validate exact layout and passivity, plant missing-frame and
non-additive failures, preflight only, then stop.

UNASKED, offered for revert: none.

UNVERIFIED: the operator-only `--run` and `--admit-existing` paths.  The NEMO
record does not exist yet and no scientific comparison is claimed.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r52-262047163
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r52-262047163/scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run_r52_trazdf_walk.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — acquisition is committed and preflighted; the operator must run and admit the kt-1 stage-3 trazdf record.
