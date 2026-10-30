# Preregistration: ORCA2 round 348 — fast exact-libm and native long runs

Date: 2026-10-30. Base: `6b9723445516`. This registration covers only the
three CPU acquisitions requested for round 348. They run one at a time. No NEMO
executable is run.

## A. Fast exact-libm 80-step identity and timing

Command:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh --steps 80 --allow-full-year --wait --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/libm_fast_80step
```

Required run output: `"status": "PASS_R342_S4_K0_LEGOESM_RUN"`,
`"transcendentals": "libm"`, and a saved `kt=0080/0080` line. Refuse before
launch if the tracked tree is dirty, available RAM is below 150 GiB, the output
directory already contains evidence, the backend is not CPU fp64, or the
resolved mode is not `libm`.

Compare the data arrays in every saved snapshot at steps 8, 10, 16, 24, 32,
40, 48, 56, 64, 72, and 80, and compare total seconds per completed step with
the certified year manifest:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py --reference-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/libm_fast_80step --geometry-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2 --days 10 --bitwise-steps 8,10,16,24,32,40,48,56,64,72,80 --require-bitwise-exact --reference-manifest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step/run_manifest.json --candidate-manifest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/libm_fast_80step/run_manifest.json --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/libm_fast_80step_comparison
```

Required comparison lines are `PASS_R348_BITWISE_EXACT snapshots=11 arrays=99`
and `PASS_R348_ARM_COMPARISON days=10`. Bitwise means equal dtype, shape, field
inventory, and raw C-order bytes for every array; `-0.0` and `+0.0` are distinct
and therefore fail. Any missing/extra field, byte difference, non-CPU run stamp,
or nonzero exit refutes identity. Timing has no pass bar: report candidate total
wall seconds/80 next to the certified manifest's total wall seconds/2920
(previously 24.07 s/step).

## B. Native-transcendental one-year arm

Command:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh --steps 2920 --allow-full-year --wait --transcendentals native --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_year1_2920step
```

Required output: `"status": "PASS_R342_S4_K0_LEGOESM_RUN"`,
`"transcendentals": "native"`, and saved `kt=2920/2920`. Refuse under the same
tree/RAM/output/backend conditions as A, if any daily snapshot is absent, or if
the manifest records any mode other than `native`.

## C. Native vs NEMO and native vs libm

Run the existing round-342 reporter unchanged, with the admitted round-345
twin-B envelope unchanged:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_metrics.py --nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2 --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_year1_2920step --nemo-twin-b-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/orca2_k0_year1_sst_plus_1ulp_np2 --days 365 --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_vs_nemo_metrics
```

Required line: `"status": "EVALUATED_REFERENCE_ONLY"`. Refuse on a missing
daily state, changed sampling/weights/bars, non-finite metric, or a failed
planted control.

Apply the existing round-345 Decision-188 gate unchanged. A scientific PASS or
FAIL is admissible; any other result is refusal:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_decision188_gate.py --candidate-metrics /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_vs_nemo_metrics/round342_s4_candidate_vs_nemo_metrics.json --envelope-metrics /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_vs_nemo_metrics/round342_s4_nemo_twin_b_vs_a_metrics.json --envelope-admission /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/round345_s4_year_sst_ulp_admission.json --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_vs_nemo_metrics/decision188_verdict.json
```

The gate must emit `"bar_mode": "DECISION_188_PASS_FAIL"` and exactly one
`"verdict": "PASS"` or `"verdict": "FAIL"`. It refuses a changed twin-B
identity, absent/non-positive monthly envelope, mismatched day windows, or
changed fixed bars.

Compare native directly with the certified libm arm:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py --reference-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_year1_2920step --geometry-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2 --days 365 --reference-manifest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step/run_manifest.json --candidate-manifest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_year1_2920step/run_manifest.json --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r348/native_vs_libm
```

Required line: `PASS_R348_ARM_COMPARISON days=365`. The JSON and PNG must carry
daily area-weighted SST RMS and 3-D wet T RMS. Refuse on missing snapshots,
wrong shapes, non-finite data, empty wet support, dirty provenance, or a planted
control that does not move its score.
