# ORCA2 round 336 kt=2 TKE acquisition

This operator-only tool records the `kt=2` TKE statement boundaries with a
new production-arithmetic observer.  Unlike the round-331/332 recorders, all
procedure calls are outside NEMO's arithmetic loops; the in-loop additions are
direct copy-outs after the statement blocks.  The preflight proves the pristine
and observed per-function FP instruction censuses identical at NEMO's real
flags, compiles and runs the actual patched routines over real `A2D(0)` versus
full-halo extents, and reports the non-passive historical 331/332 censuses.

The preflight also syntax-plants the executable driver, runs a stub NEMO through
success, quoted `E R R O R`, crash retention, in-place admission, and retained
admission, and exercises the binary gate's layout and semantic plants. Its
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
