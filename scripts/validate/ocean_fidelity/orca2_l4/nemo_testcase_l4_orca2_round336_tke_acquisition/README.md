# ORCA2 round 336 kt=2 TKE acquisition

This operator-only wrapper extends the reviewed round-331 TKE solver recorder
and round-332 mixing-length discriminator from `kt=1` to `kt=2`. Its default
mode is a read-only preflight; it never runs NEMO unless the operator supplies
`--run`.

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh --run
```

The run uses the unchanged round-326 ten-step deck and creates a distinct NEMO
build and record directory. A refusal keeps both the build and run through the
inherited retention trap. Before retrying, move the retained build directory
aside; do not delete it. A retained run is rechecked with the same command and
`--admit-existing`, setting `R328_EXISTING_RUN` to the path printed by the
refusal.
