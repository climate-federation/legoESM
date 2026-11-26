# Round 344 fast-literal GPU completion

Run only on a host where the selected physical GPU reports exactly 0 MiB used
and 0% utilization:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round344_fast_literal_gpu/run.sh --run GPU_INDEX
```

The runner refuses a missing/busy GPU, reuses round 343's benchmark, writes the
fp64 GPU arm beside the completed CPU arm, and assembles the round-344 report.
It does not run or modify NEMO.
