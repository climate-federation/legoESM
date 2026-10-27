# ORCA2 round 336 kt=2 TKE acquisition

This operator-only tool records the `kt=2` TKE statement boundaries with a
new production-arithmetic observer. Unlike the round-331/332 recorders, its
only three procedure calls are in outer `zdf_tke`; the arithmetic routines use
direct copy-outs after each completed state-changing loop nest. The record
separates the state-changing Langmuir `en` nest, every Thomas-solve loop, and
the `lup`, `ldown`, and merge sweeps. Langmuir-local scratch values are
replay-derived from the complete entry inputs: directly reading those locals
changed the production `zdf_tke` FP census, so they are deliberately not
observed. It does not copy the scalar `zav` inside the coefficient loop.
The preflight proves the pristine
and observed per-function FP instruction censuses identical at NEMO's real
flags, compiles and runs the actual patched routines over each array's real
NEMO allocation extent, and reports the non-passive historical 331/332 censuses.
The 51-field record includes `htau`, bottom velocity at `Kbb`, evaluated
`gdepw/e3t/e3w` at `Kmm`, masks, bottom/top levels, and background viscosity
inputs so the recorded active TKE blocks have a complete replay read set. The
full-halo `entry_avm_k` includes the north halo consumed by `zdf_sh2`.
The drag fields use exactly NEMO's `DO_2D(1,1,1,1)` filled range
`ntsi-1:ntei+1,ntsj-1:ntej+1`; the unwritten outer ring is neither recorded nor
checked. This follows `zdfdrg.F90:122,172-190,215-221` and
`do_loop_substitute.h90:117-125`.

For `tke336`, the shared driver copies no inherited `MY_SRC` file: the target is
pristine NEMO plus only the round-336 `zdftke` observer and its writer. The
preflight compares the FP opcode census of every shared function to a pristine
425-object baseline; `stp_rk3_stg` is pinned at 1,175 instructions and a plant
that restores the inherited 1,367-instruction object refuses. It separately
audits the pinned binaries
that produced admitted records in rounds 5, 326, 328, 331, 332, and 333 against
observer-free objects compiled at the record flags; that audit is historical
evidence only and does not add those observers to the new build.

The preflight also syntax-plants the executable driver, runs a stub NEMO through
success, quoted `E R R O R`, crash retention, in-place admission, and retained
admission through the real gate and all 19 admission plants, including the
one-halo drag-range check and source-value checks for ice cover and linear
bottom drag. Its drag overlap comes from the admitted round-328 NEMO files and
is checked against the admitted NEMO mask before the records are written; the
stub cannot manufacture a full-domain drag field. The stub also supplies the
inherited oracle/restart files so the byte-for-byte deterministic passivity
check runs. Its
default mode is read-only; it never runs NEMO unless the operator supplies
`--run`.

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh --run
```

The run uses the unchanged round-326 ten-step deck and creates the distinct
`ORCA2_ORCA1ICE_OMIP_L4_R336TKE` build and
`orca1ice_tke_internal_kt2_np2` record directory. A refusal keeps both the
build and run through the inherited and wrapper `EXIT` traps. Before retrying,
move the retained build directory aside; do not delete it. A retained run is
rechecked with `--admit-existing`, setting `R328_EXISTING_RUN` to the path
printed by the refusal.
