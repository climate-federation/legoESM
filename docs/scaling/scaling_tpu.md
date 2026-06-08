# TPU scaling — legoESM (Google Cloud TPU)

Single-host Cloud TPU testing for legoESM. Status: provisioning + emulation
validation done; real-hardware numbers pending (Phase 3).

## Scope and hard constraints

- **Single-host slice only** (v5e-8 / v6e-8): 8 chips in one host, pure SPMD.
  No multi-host pods, no MPI, no `jax.distributed.initialize()`.
- **float32 only.** TPUs have no efficient native float64, so the spectral
  dycore (x64 + complex128) is excluded. Targets are the float32-capable
  finite-volume grids — cubed-sphere FV3 first.
- **Cubed-sphere uses 6 of the 8 chips.** Face sharding requires a device
  count that divides the 6-face layout (1, 2, 3, 6). 8 does not, so:
  - **6 chips** → true face sharding (2 chips idle). This is the supported,
    validated path.
  - **8 chips** → would need the Issue-#273 "level-parallel fallback", but
    the cubed-sphere dycore's halo exchange (`explicit_pad_halo_4d` in
    `parallel/cubesphere_exchange.py`) runs a `shard_map` over a hardcoded
    6-face mesh regardless of config. On an 8-device mesh it raises
    `Received incompatible devices`. Using all 8 chips for cubed-sphere is a
    **known gap requiring dycore work** (a replicated/local `pad_halo`
    fallback), not a config flag.

## Phase 0 — emulation validation (done, free)

Before any TPU spend, the exact sharded code paths were validated on emulated
CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=N`):

```
python scripts/validate/validate_tpu_emulation_sharding.py --verbose
```

Findings:
- 6-chip face sharding is **numerically identical** to single-device — in
  float64 the diff is roundoff (~5e-10 on a ~1e5 Pa surface-pressure field);
  in float32 it is ~1e-6 relative (float32 epsilon over reordered global
  reductions in collectives). Sharding is a numerical no-op, as expected.
- 8-chip cubed-sphere (level fallback) is unsupported (see constraint above).

The production bench driver also runs end-to-end under emulation across the
full device sweep (1, 2, 3, 6):

```
XLA_FLAGS="--xla_force_host_platform_device_count=6" JAX_PLATFORMS=cpu \
  PYTHONPATH="$(pwd)" \
  python scripts/bench/run_levante_gpu_scaling.py \
    --grid cubed-sphere --mode strong --precision float32 \
    --n-gpus 6 --n-levels 10 --strong-resolutions 24 \
    --n-warmup 2 --n-timing 10 --no-plot --no-timestamp
```

Emulation proves correctness only — scaling *efficiency* numbers from
emulated CPU devices are meaningless (collectives are pure overhead with no
added compute). Real efficiency comes from TPU hardware (Phase 3).

## Phase 2/3 — provisioning and running on a TPU VM

Scripts live in `scripts/cluster/gcp_tpu/`. Accelerator types and runtime
versions change across TPU generations — verify current values with
`gcloud compute tpus accelerator-types list` /
`gcloud compute tpus tpu-vm versions list`.

```bash
# 1. Create the VM (from your laptop). v5e-8 default; USE_SPOT=1 for cheap.
GCP_PROJECT=my-proj TPU_ZONE=us-east5-a \
  bash scripts/cluster/gcp_tpu/create_tpu_vm.sh
#    For v6e: ACCELERATOR_TYPE=v6e-8 RUNTIME_VERSION=v2-alpha-tpuv6e ...

# 2. SSH in.
gcloud compute tpus tpu-vm ssh legoesm-tpu --zone=us-east5-a --project=my-proj

# 3. On the VM: get the repo. Prefer git clone — 'scp .' would drag the
#    multi-GB .venv/.git/results along. (scp fallback if no git access:
#    gcloud compute tpus tpu-vm scp --recurse . legoesm-tpu:~/legoESM ...)
git clone <your-legoESM-remote-url> ~/legoESM

# 4. On the VM: set up the env (installs repo + jax[tpu] + libtpu, verifies TPU).
cd ~/legoESM && bash scripts/cluster/gcp_tpu/setup_env.sh

# 5. On the VM: run the benchmark (cubed-sphere, float32, 6 of 8 chips).
bash scripts/cluster/gcp_tpu/run_bench.sh

# 6. Copy results back, then DELETE the VM (it bills while it exists).
GCP_PROJECT=my-proj TPU_ZONE=us-east5-a \
  bash scripts/cluster/gcp_tpu/delete_tpu_vm.sh
```

Notes:
- `setup_env.sh` pins `jax[tpu]==0.10.1` (the locally validated version).
  Keep jax and the TPU jaxlib at the same version.
- `run_bench.sh` exports `PYTHONPATH=<repo root>` because the bench driver
  imports `tests.test_cases.baroclinic_wave`; without it every run silently
  reports `FAILED: No module named 'tests'`.

### Cheap smoke test (v5litepod-4)

Before paying for a full v5e-8 sweep, a 4-chip VM gives a fast, low-cost
"does it run on real TPU silicon at all" check. On 4 chips the valid
cubed-sphere device counts are **1 and 2 only** — face sharding needs a
divisor of the 6-face layout, and neither 4 nor 8 divides 6, so a 4-chip
slice runs face sharding on 2 of its 4 chips (the other 2 idle).

```bash
# 1. Create a cheap 4-chip Spot VM (same runtime version as v5e-8).
GCP_PROJECT=my-proj TPU_ZONE=us-east5-a \
  ACCELERATOR_TYPE=v5litepod-4 USE_SPOT=1 \
  bash scripts/cluster/gcp_tpu/create_tpu_vm.sh

# 2. ssh + clone + setup_env.sh as above, then:
# 3. On the VM: cheap C24/L10 sweep over 1 -> 2 chips (few timing iters).
bash scripts/cluster/gcp_tpu/run_smoke.sh

# 4. Delete the VM.
```

`run_smoke.sh` just lowers the `run_bench.sh` defaults (`N_GPUS=2`,
`STRONG_RESOLUTIONS=24`, `N_LEVELS=10`, short warmup/timing) and writes to
`results/scaling_tpu_smoke/`. It is a correctness/liveness check only —
its timing numbers are **not** a meaningful scaling benchmark.

## Phase 3 results

_Pending real-hardware run._ Fill in SYPD / ms-step / Mcells-s for device
counts 1, 2, 3, 6 at C48 and C96, float32. Compare the 1→6 face-sharding
curve (the clean scaling story) against the GPU numbers in `scaling_gpu.md`.