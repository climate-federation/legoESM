# ORCA2 round-336 `tra_zdf` acquisition

This operator-only tool extends the reviewed round-333 acquisition without
changing any ORCA2 configuration value. Its default action is a preflight;
that path does not build or execute NEMO:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --preflight-only
```

The extension targets kt 2, RK stage 3. `r336_begin` copies `Krhs` immediately
before `tra_zdf_imp`, because this deck assigns the same RK slot to `Krhs` and
`Kaa` and the solve overwrites it. `r336_capture` is called once after
`tra_zdf_imp`, outside all of NEMO's arithmetic loops. It records the final
`avt`, `avs`, `akz`, raw geometry and RK levels, then replays the exact
source-order matrix construction, eliminated diagonal, forward sweep and back
substitution into a passive record. Admission requires the replayed
back-substitution to be bit-identical to the live post-solve `Kaa` on every
executed cell; it therefore cannot admit a reconstruction with changed
association or order.

Each rank writes
`oracle_r336_trazdf_kt00000002_r0000.bin` or `_r0001.bin`. The 25 fields retain
declared extents and IEEE signed zeros and carry source-order names, explicit
rank/origin/extent metadata and XOR payload checksums. The gate also re-admits
the inherited round-333 tracer and internal-LDF records and cross-checks the
new `akz`, geometry, RK stretches, `Kbb`, pre-solve `Krhs`, and live `Kaa`
against them.

Preflight applies the additions-only patch with zero fuzz to both pristine
source and the record-build ppsrc. It audits the resolved DDM/EVD/IWM,
mass-flux, adaptive-advection, and MSC selectors, and plants a changed value
for each. It compiles the actual record-build `trazdf.f90` with
`-O3 -funroll-all-loops -fno-tree-vectorize`; the normalized floating-point
instruction census must be unchanged by the observer calls, while a planted
matrix arithmetic change must fail. A two-rank, signed-zero fixture runs the
writer with those production arithmetic flags plus `-fcheck=bounds` and
compares the passive replay bitwise with an independent live solve. Structural,
checksum, finite-value, wet-coverage, signed-zero, replay, and live-result
plants must all fire. Every round-333 helper, plant, syntax/same-program proof,
deterministic rerun, admission identity check, and refused-output retention
path remains active.

Only the operator may request the NEMO run:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --run
```

The expected evidence directory is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_trazdf_internal_kt2_np2`.
Set `R328_EXISTING_RUN` to a retained refusal and use `--admit-existing` to
repeat admission without rebuilding or running NEMO. Admission is pinned to
the retained producer commit and binary SHA, and every refusal log remains
with the evidence.
