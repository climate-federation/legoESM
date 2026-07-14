# Derecho runbook — legoESM GPU scaling (July 2026)

Task-ordered guide for running the legoESM scaling campaign on NCAR
Derecho (4× A100-40 per GPU node, Slingshot-11). Everything here already
exists on `main`; the deep reference is
`scripts/cluster/scaling_derecho/README.md` (environments, fabric notes,
every job). This page is the short "what to run, in what order, what to
check, what to send back".

---

## 0. One-time setup (login node)

1. Clone the repo and follow **README Step 1** in
   `scripts/cluster/scaling_derecho/README.md` to create the `legoesm-gpu`
   conda env. (Step 1b — the route-A mpi4jax overlay — is only needed for
   the optional route-A comparison lane; skip unless asked.)
2. Edit `scripts/cluster/scaling_derecho/_env.sh`: set your project code,
   repo path, and scratch dir.
3. **Build the NCCL/Slingshot plugin** (required for production numbers —
   without it NCCL silently falls back to TCP sockets, 2–3× slower comm):

   ```bash
   bash scripts/cluster/scaling_derecho/build_nccl_ofi.sh
   ```

   It prints the `LEGOESM_NCCL_OFI_LIB=...` path to pass at submit time.

**First-run verification (every new machine/env, non-negotiable):**
- The job log's `NCCL_DEBUG=INFO` lines must say
  `Using network AWS Libfabric`. If they say `Using network Socket`, the
  plugin is not loaded — numbers are shakeout-only, do not report them.
- Every job runs its own parity/conservation gates and exits nonzero on
  failure. A red job = stop and send the log; never keep the timing row.
- Result JSON/JSONL rows are self-describing: check
  `"transport"` (should be `nccl`/route-B for these lanes) and
  `"virtual_cpu_devices": false`.

---

## 1. Main scaling lanes (route-B NCCL, 2 nodes × 4 GPUs)

```bash
qsub -v LEGOESM_NCCL_OFI_LIB=/glade/work/$USER/nccl-ofi/<tag>/lib \
     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
```

Runs, in one job: lane A (cube cs-spmd, 6 GPU), lane C (atm lat-lon
multicontroller, 8 GPU), lane D (ocean multicontroller, 8 GPU), lane E
(icosahedral/MPAS multicontroller, 6 GPU). Larger ladders: resubmit with
more nodes and `GPU_RANKS`/`LL_RES` etc. per the README.

**Production knob (measured 2026-07-09, use it):** add
`JAX_ENABLE_PGLE=true JAX_PGLE_PROFILING_RUNS=3` to the environment of
lat-lon-class route-B lanes — measured **+8.5%** (5.97 vs 6.48 ms/step at
8 GPU). Do **not** set `LEGOESM_LATLON_SPMD_FUSED_HALO=1` (measured −12%
on latlon at this size) and do not add the CP-combine/pipelined-p2p XLA
flags (measured −10% combined). Full A/B table:
`docs/performance/scaling/spmd_message_census_2026-07-08.md`.

**Resolution floors (or the curves will look "broken" and are not):**
keep ≥ ~30k columns per GPU. Concretely: lat-lon LL512+ and icosahedral
L7+ for 8–16 GPUs; coarse grids (156 km latlon, 112 km ico, C48 cube) do
NOT strong-scale on GPUs anywhere — per-device saturation, expected.

## 2. Cube at 24 GPUs — the new capability (6 nodes × 4 GPUs)

```bash
qsub -v LEGOESM_NCCL_OFI_LIB=... \
     scripts/cluster/scaling_derecho/cube_tiled_step.pbs
```

Stage A runs two CPU-virtual parity gates (single-shot + closed-loop) —
must both pass before the timed stages run. Stage B times C192/L60 at
24 GPUs twice: the single-shot lane and the **closed-loop lane**
(`--closed-loop`: a real multi-step integration, state resident
tile-sharded across steps — this is the number that matters). Both rows
land in one JSONL with `solver_variant` distinguishing them.

The same 24-GPU decomposition is also reachable through the standard
harness for ladder comparisons (dry and moist):

```bash
mpiexec -n 24 --ppn 4 <pin-shim> python scripts/bench/run_cpu_mpi_scaling.py \
    --grid cubed-sphere --cs-spmd --device gpu --physics moist \
    --mode strong --resolution 192 --n-levels 60 --output-dir ...
```

(see `gpu_multinode_scaling.pbs` lane A for the exact pin shim).

**Known scope limits at real multi-node GPU counts** (refused loudly,
not silent): the ModelDriver's tiled *unified-physics* lane
(gray/RRTMGP/TKE-class configs at 6·kt² devices) is single-process
multi-device only for now — it is CPU-virtual/parity-validated, and the
cross-process (multicontroller) variant is the next increment. The
dynamics-only and Kessler-moist tiled lanes, and everything in the two
jobs above, run multi-node today.

## 3. Optional: remaining lane-T tuning arms

The 8-GPU fused/xla/pgle A/B is DONE (verdicts above — do not re-run at
the same size). Still informative if you have spare allocation:

```bash
qsub -v RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0,LEGOESM_NCCL_OFI_LIB=... \
     scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
```

- the cube base/xla arms (never collected),
- the latlon arms at np=16/24 (fused might flip sign where latency
  dominates),
- if send/recv-bound after PGLE: `NCCL_NCHANNELS_PER_NET_PEER` 4→8/16.

A/B outputs land under `_ab_tuning/` and are excluded from the scaling
curves automatically.

## 4. Aggregate + plot + what to send back

```bash
python scripts/bench/aggregate_bcw_scaling.py \
    --root $SCRATCH/legoesm_scaling --out $SCRATCH/legoesm_scaling/all_tidy.csv
python scripts/plot/plot_cpu_vs_gpu_scaling.py \
    --csv $SCRATCH/legoesm_scaling/all_tidy.csv \
    --out $SCRATCH/legoesm_scaling/plots
```

The aggregator now ingests the SPMD lanes' `.jsonl` directly (SYPD panels
fill in for lat-lon; no manual merging). Send back:

1. `all_tidy.csv` + the `plots/*.png`,
2. the raw `*.jsonl`/`*.json` result dirs (they carry the transport/
   parity metadata we audit),
3. each job's `run.log` (first ~100 lines contain the NCCL net line and
   the gate receipts),
4. anything that exited nonzero — full log, don't retry first.

## 5. Quick triage

| symptom | meaning | action |
|---|---|---|
| `Using network Socket` in log | NCCL plugin missing | rebuild step 0.3, resubmit with `LEGOESM_NCCL_OFI_LIB` |
| job exits rc=5, "parity ... MISMATCH" | correctness gate tripped | send log; do not use the timing rows |
| "contains full-cube all-gathers … refusing" | tiled step compiled replicated | send log (compiler/env issue) |
| coarse grid flat/anti-scaling on GPUs | per-device floor | expected; use the §1 resolution floors |
| `NotImplementedError: … refused` from a driver run | config outside the tiled envelope (land, writers, w-grid convection, multicontroller op-split) | run the face-only lane (≤6 GPUs) or drop the feature; the message names the exact knob |
| every step as slow as step 0 | per-step retrace | send log — that is a bug on our side, not a tuning issue |
