# TPU scaling — legoESM (Google Cloud TPU)

Single-host Cloud TPU testing for legoESM. Status: provisioning + emulation
validation done; **single-chip v6e baseline measured** (Phase 3, see results
below); the multi-chip 1→6 face-sharding sweep is pending an 8-chip slice.

## Scope and hard constraints

- **Single-host slice only** — default **v6e-8** (Trillium, 2x4 topology);
  v5e-8 also works. 8 chips in one host, pure SPMD. No multi-host pods, no
  MPI, no `jax.distributed.initialize()`. v6e and v5e are both counted in
  chips (so `-8` = 8 chips); v5p is counted in *cores* (2/chip), so a v5p
  8-chip slice is `v5p-16` — avoid it for this single-host workload.
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
# 1. Create the VM (from your laptop). v6e-8 default; USE_SPOT=1 for cheap.
GCP_PROJECT=my-proj TPU_ZONE=<your-v6e-zone> \
  bash scripts/cluster/gcp_tpu/create_tpu_vm.sh
#    For v5e instead: ACCELERATOR_TYPE=v5litepod-8 RUNTIME_VERSION=v2-alpha-tpuv5-lite ...

# 2. SSH in.
gcloud compute tpus tpu-vm ssh legoesm-tpu --zone=<your-v6e-zone> --project=my-proj

# 3. On the VM: get the repo. Prefer git clone — 'scp .' would drag the
#    multi-GB .venv/.git/results along. (scp fallback if no git access:
#    gcloud compute tpus tpu-vm scp --recurse . legoesm-tpu:~/legoESM ...)
git clone <your-legoESM-remote-url> ~/legoESM

# 4. On the VM: set up the env (installs repo + jax[tpu] + libtpu, verifies TPU).
cd ~/legoESM && bash scripts/cluster/gcp_tpu/setup_env.sh

# 5. On the VM: run the benchmark (cubed-sphere, float32, 6 of 8 chips).
bash scripts/cluster/gcp_tpu/run_bench.sh

# 6. Copy results back, then DELETE the VM (it bills while it exists).
GCP_PROJECT=my-proj TPU_ZONE=<your-v6e-zone> \
  bash scripts/cluster/gcp_tpu/delete_tpu_vm.sh
```

Notes:
- **Install is via uv, not bare pip.** The repo is a uv workspace: the
  federated members (`legoesm-core`/`-atmosphere`/`-ocean`/`-land`/`-ice`/
  `-coupler`/`-ml`/`-tools`) live in-tree under `packages/*` and resolve via
  `[tool.uv.sources]`. A plain `pip install -e .` tries to fetch
  `legoesm-atmosphere~=0.1.0` from PyPI (not published there) and fails with
  *"No matching distribution"*. `setup_env.sh` installs `uv` (if missing) and
  runs `uv sync`, which installs every member editable from `uv.lock`.
- **Python >= 3.11 is handled automatically.** legoESM requires Python >=3.11,
  but TPU VM images often default to 3.10 (Ubuntu 22.04). `uv sync --python
  3.11` provisions a managed CPython 3.11 itself, so no `deadsnakes`/system
  Python install is needed. Override with `PY_VERSION=3.12 bash ...` if desired.
- `setup_env.sh` pins `jax[tpu]==0.10.1` (the locally validated version) on top
  of the sync; `uv.lock` pins the CPU build at 0.10.0, so the override swaps in
  the TPU jaxlib + libtpu. Keep jax and the TPU jaxlib at the same version.
  It uses `uv pip install` (not `uv run`/`uv sync`) for the override so it isn't
  silently reverted to the locked CPU build.
- `run_bench.sh` exports `PYTHONPATH=<repo root>` because the bench driver
  imports `tests.test_cases.baroclinic_wave`; without it every run silently
  reports `FAILED: No module named 'tests'`.

### Cheap smoke test (4-chip slice: v6e-4)

Before paying for a full 8-chip sweep, a 4-chip VM gives a fast, low-cost
"does it run on real TPU silicon at all" check. On 4 chips the valid
cubed-sphere device counts are **1 and 2 only** — face sharding needs a
divisor of the 6-face layout, and neither 4 nor 8 divides 6, so a 4-chip
slice runs face sharding on 2 of its 4 chips (the other 2 idle).

```bash
# 1. Create a cheap 4-chip Spot VM (v6e default; v5litepod-4 also works).
GCP_PROJECT=my-proj TPU_ZONE=<your-v6e-zone> \
  ACCELERATOR_TYPE=v6e-4 RUNTIME_VERSION=v2-alpha-tpuv6e USE_SPOT=1 \
  bash scripts/cluster/gcp_tpu/create_tpu_vm.sh
#    For v5e instead: ACCELERATOR_TYPE=v5litepod-4 RUNTIME_VERSION=v2-alpha-tpuv5-lite

# 2. ssh + clone + setup_env.sh as above, then:
# 3. On the VM: cheap C24/L10 sweep over 1 -> 2 chips (few timing iters).
bash scripts/cluster/gcp_tpu/run_smoke.sh

# 4. Delete the VM.
```

`run_smoke.sh` is generation-agnostic — it only lowers the `run_bench.sh`
defaults (`N_GPUS=2`, `STRONG_RESOLUTIONS=24`, `N_LEVELS=10`, short
warmup/timing) and writes to `results/scaling_tpu_smoke/`, running on
whatever VM you created. It is a correctness/liveness check only — its
timing numbers are **not** a meaningful scaling benchmark.

### Single-chip baseline (ct6e-standard-1t)

v6e (and v5e) 8-chip slices are frequently blocked by capacity/quota, so you
may only be able to get a **1-chip** VM. The Compute-Engine-style machine
type `ct6e-standard-Nt` encodes the chip count in the `Nt` suffix:
`-1t` = 1 chip, `-4t` = 4 chips, `-8t` = 8 chips. A `ct6e-standard-1t`
(44 vCPU / 176 GB host) is a single v6e chip.

One chip means **single-device only** — no sharding, no scaling sweep (that
needs ≥2 chips; the clean 1→6 cubed-sphere curve needs 8). But it still
proves the cubed-sphere FV3 path runs on real v6e silicon and gives a
single-chip SYPD baseline. Every script defaults to a multi-chip count
(2 or 6) for the 8-chip slice, so on one chip you **must** override
`N_GPUS=1` or the run tries to grab devices that aren't there.

```bash
# On the 1-chip VM, after setup_env.sh (prints "device count: 1"):
N_GPUS=1 bash scripts/cluster/gcp_tpu/run_smoke.sh   # quick C24 liveness
N_GPUS=1 bash scripts/cluster/gcp_tpu/run_bench.sh   # C48/C96 baseline
```

When an 8-chip slice (`ct6e-standard-8t` / `v6e-8`) becomes available, the
same scripts produce the full 1→6 curve with no changes.

## Phase 3 results

### Single-chip v6e baseline (ct6e-standard-1t, float32)

First real-hardware run, on a single v6e chip (the 8-chip slice was
capacity/quota-blocked). Cubed-sphere FV3, L26, float32, `backend: TPU`,
jax 0.10.1 / Python 3.11. `Mcells/s = total_cells / step_time` where
`total_cells` already includes levels (same definition as `scaling_gpu.md`).

| res | total cells | dt (s) | ms/step | SYPD  | Mcells/s | compile |
|-----|-------------|--------|---------|-------|----------|---------|
| C48 | 359,424     | 210    | 4.48    | 128.4 | 80.3     | 10.5 s  |
| C96 | 1,437,696   | 90     | 20.6    | 11.9  | 69.7     | 12.2 s  |

(A C24/L10 smoke run first confirmed liveness at 7.5 Mcells/s — too small to
fill the chip; throughput rises ~10× by C48 as the chip saturates.)

**vs GPU (RTX 5090, same cubed-sphere FV3).** TPU is fp32-only, so compare to
the GPU *fp32* figure, not the fp64 table in `scaling_gpu.md`:

| res | v6e fp32 (1 chip) | RTX 5090 fp32 | ratio |
|-----|-------------------|---------------|-------|
| C48 | 80.3 Mcells/s     | ~299 Mcells/s | ~0.27× |

A single v6e chip is ~3.7× slower than one RTX 5090 on this dycore. That is
expected: FV3 (SSP-RK3 × PPM, elementwise + stencil + gather) is
memory/stencil-bound and barely uses the TPU's matmul units (MXU), where the
v6e's advantage lives. The TPU value proposition here is **multi-chip
scaling**, not single-chip peak — 6 chips at ~linear face-sharding efficiency
would land in RTX-5090 territory. Confirming that needs the 1→6 sweep.

### 1→6 face-sharding sweep — pending 8-chip slice

Still to do once a `ct6e-standard-8t` / `v6e-8` (2x4) slice is available
(blocked so far by v6e capacity/quota). `run_bench.sh` produces it unchanged;
fill in device counts 1, 2, 3, 6 at C48 and C96, float32, and compare the
scaling-efficiency curve against the GPU numbers in `scaling_gpu.md`.

| n_chips | C48 ms/step | C48 SYPD | C48 eff | C96 ms/step | C96 SYPD | C96 eff |
|---------|-------------|----------|---------|-------------|----------|---------|
| 1       | 4.48        | 128.4    | 100%    | 20.6        | 11.9     | 100%    |
| 2       | _pending_   |          |         |             |          |         |
| 3       | _pending_   |          |         |             |          |         |
| 6       | _pending_   |          |         |             |          |         |