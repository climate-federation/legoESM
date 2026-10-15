# SWG online learning, round 9: independent review and GPU pilot receipt

Date: 2026-10-15. Base `e0b4229a8`; implementation commit `f8c14fc81`.
No GPU, pilot training, or long free run was launched. All model execution in
this review used CPU fp64 with libm; the longest test integration was four
model days. The committed pilot remains the user-decided experiment.

## Verdict

Rounds 7 and 8 are correct on the loss, split/window construction, optimiser,
checkpoint/resume path, and hardened free-run admission. Their 13 published
reversions all failed independently in round 9. One operational gap was
reproduced: the one-seed pilot had no command that could score its 20-year run,
because the only score command correctly requires at least three seeds. The
new `score-run` reports all frozen-window metrics and explicitly withholds a
gate verdict.

The pilot is ready for the operator sequence below. CUDA/native free runs are
explicit, disable XLA autotuning, record execution provenance, and cannot be
scored without a byte-bound PASS device-check record from both no-closure GPU
controls. Neither the frozen records nor any pass-mark tolerance moved.

## Review findings, dispositions, evidence, reversions

| Finding or review hypothesis | Disposition | Evidence | Reversion result |
|---|---|---|---|
| Loss could differ from Maddison's deterministic template objective or include the excluded first wall ring | **Not reproduced.** Model and target pass through the same selected diagnostic, then the Eq. 7 squared-vorticity reduction (`swg_online.py:1352`); the ring choice has no default and is supplied to the point weights (`swg_online.py:1303`). The pilot explicitly says `template` and `excluded` (`pilot_template_ring_excluded.json:6`, `:7`). | The end-to-end test recomputes the first logged loss from `deterministic_trajectory_loss`; the committed-config test pins every optimiser and curriculum field (`test_run_swg_online_training.py:90`). | Existing objective/ring plants remain in the lane battery; no code fix was warranted. |
| Training windows could cross splits, reuse held-out sampling, or use the wrong two years | **Not reproduced.** Training draws a seeded offset then non-overlapping windows; held-out splits use the fixed chronological lattice (`swg_online.py:917`). Bounds and ten-window batch membership are checked (`swg_online.py:971`). The pilot train split is days 731–1461, batch 10, windows 1/2, five epochs (`pilot_template_ring_excluded.json:26`, `:31`, `:33`). | `check-data` on the committed pilot: `DATA_CHECKED sources=67 batches=6 contract_sha256=bf97d77a...`; the exact pilot-vs-candidate equality test is `test_run_swg_online_training.py:120`. | Existing split leakage, relabelling and incomplete-record plants remain in the battery; no fix was warranted. |
| Optimiser settings or window transition state could drift from the template | **Not reproduced.** The strict wrapper passes constant-rate AdamW, Keras epsilon-hat, no warm-up/decay and no clipping (`swg_online.py:1521`). The driver reinitialises optimiser state at the first epoch of each window while carrying the network (`run_swg_online_training.py:844`). | The config test pins rate `1e-3`, decay `0.004`, betas, epsilon and no clipping (`test_run_swg_online_training.py:94`); the two-window test checks reset optimiser counts and carried weights. | Existing optimiser constructor and carried/reset plants remain in the battery; no fix was warranted. |
| Resume might accept changed inputs or a changed checkpoint | **Not reproduced.** Resume compares the complete stored run identity (`run_swg_online_training.py:780`) and hashes the last checkpoint before loading it (`:821`). | Killed-write resume reproduces the next update bit for bit; changed config, execution, commit, seed and checkpoint tests all refuse. | Existing resume/checkpoint plants remain in the battery; no fix was warranted. |
| Round 8's unified manifest, daily means, execution binding, checkpoint binding, test-year chunk label and digest refusal might be vacuous | **Not reproduced.** Shared schema and exact closure keys are at `swg_online.py:1560`; checkpoint existence and bytes are checked at `:1625`. The tiny end-to-end path trains two steps, writes four dual days, admits and actually scores days 1–3 (`test_run_swg_online_training.py:666`), then plants the obsolete block, missing file and changed file (`:753`). | Baseline review: 83 targeted tests passed. Round 9 independently ran all 13 published round-8 mutants; all failed for the named reason (schema writer, schema reader, snapshot-only, fake mean, no closure, execution bypass, device flag, arbitrary execution, unchecked file, unchecked digest, three-seed pilot, underscore label, listed digests). | Each of the 13 exact reversions failed; no round-8 fix was weakened. |
| The four required refusal plants might reach scoring | **Not reproduced.** The old checkpoint block, missing checkpoint and changed checkpoint are planted at `test_swg_online_evaluation.py:609`; snapshot-only output is one of the independently rerun round-8 mutants. | All refuse before a candidate score. The obsolete checkpoint block reports exact-key mismatch; missing and changed files report their respective checkpoint errors; snapshot-only reports no admitted per-file sampling label. | The corresponding round-8 reversions all failed. |
| The driver might omit the repository no-inert rule | **Not reproduced.** Every trainable gradient leaf goes through `assert_no_inert` (`swg_online.py:1530`) before the first optimiser update (`run_swg_online_training.py:863`). | The success log records nine live leaves. A zero-gradient first step aborts before an epoch is logged. | `r9_no_first_step_inert_gate` removed the call: the zero-gradient plant failed with `DID NOT RAISE`. |
| **Reproduced:** a one-seed pilot could be admitted but not scored, because `score` requires three seeds | Fixed in `f8c14fc81`: `score_one_trained_run` computes the same frozen-window rows and medians but writes `gate_verdict: null` (`nemo_testcase_l1_swg_online_evaluation.py:1332`); CLI `score-run` exposes it (`:1410`). The three-seed gate is unchanged. | `test_swg_online_evaluation.py:515` checks the rows and withheld verdict; `:844` checks both new commands. | `r9_single_seed_uses_gate` routed it back through the three-seed gate and failed with `held-out gate requires at least three network seeds`. |
| CUDA free runs could silently use another math policy or autotuning state | Fixed in `f8c14fc81`: free-run configs require exactly `cpu+libm` or `cuda+native` (`run_swg_online_training.py:156`); CUDA free runs append `--xla_gpu_autotune_level=0` and reject conflicts (`:306`); the manifest records device, library, flags, JAX version and GPU name. | Selection/provenance tests cover missing keys, illegal pairs, backend mismatch, flag insertion and conflict. | `r9_autotune_not_set` failed because only the unrelated pre-existing XLA flag remained. |
| The GPU controls could use a new deck or initial state | Fixed in `f8c14fc81` with exact no-closure configs: nu=10 `dg50` and nu=80 `dg50-nu80`, both 7,300 days from rest with dual daily sampling (`control_nu10_gpu.json:5`, `control_nu80_gpu.json:5`). The same free-run function handles trained and no-closure paths (`run_swg_online_training.py:955`). | On CPU/libm the command reproduced every SSH/U/V word of frozen days 1–2 for **both** controls (`test_run_swg_online_training.py:822`). | `r9_nu80_uses_nu10_deck` failed the exact config/deck binding test. |
| A CUDA trained run could be scored before GPU/CPU comparability is shown | Fixed in `f8c14fc81`. Device check scores both GPU controls through `score_free_run`, compares every metric's median by strict truth half-range, and requires both controls still fail the unchanged pass mark (`nemo_testcase_l1_swg_online_evaluation.py:1102`). The record binds both manifests and its own bytes (`:1167`); CUDA admission requires it (`:1220`). | Synthetic PASS and mechanically valid FAIL records, no-record refusal, byte tampering and content tampering are tested at `test_swg_online_evaluation.py:473`. | `r9_cuda_without_device_check` produced `DID NOT RAISE`; `r9_device_digest_unchecked` changed the expected byte-binding failure into a later content mismatch. |

## New refusal surface

The following inputs now raise before integration or scoring: a free-run
configuration missing either device or math-library key; any device/library
pair except CPU/libm or CUDA/native; CUDA without a non-negative physical GPU,
with mismatched `CUDA_VISIBLE_DEVICES`, or with a conflicting autotune flag;
a free-run manifest missing execution provenance, naming the wrong backend,
lacking the CUDA autotune flag or GPU name, or giving a CPU run a GPU name; a
control config with closure other than none or the wrong viscosity deck; a
device check without exactly nu10 and nu80, with changed manifest bindings,
wrong objective/schema, malformed evidence, FAIL verdict, changed bytes or
non-reproducible content; and any CUDA trained run scored without an explicit
bound PASS record. CPU/libm trained runs require no device record.

## Operator commands

The example uses physical GPU 0. Change both `N` and
`CUDA_VISIBLE_DEVICES=$N` together for another physical GPU. Run from the
committed bundle head.

```bash
export REPO=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/codex_swgonl_r9/round_202610142155/repo
export OUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round9_pilot
export N=0
export PY=/home/dbalwada/legoESM/.venv/bin/python
export PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
cd "$REPO"
mkdir -p "$OUT"
```

1. Two-day nu=10 CUDA smoke, then print the largest snapshot difference from
   frozen CPU days 1–2. It must be finite and tiny; it is not expected to be
   bit zero across devices.

```bash
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py free-run \
  --config config/swg_online/control_nu10_gpu_smoke_2d.json \
  --physical-gpu $N --run-root "$OUT/smoke"

export SMOKE_ROOT=$OUT/smoke/swgonl_control_nu10_gpu_smoke_2d
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "$PY" - <<'PY'
import os
from pathlib import Path
import numpy as np
gpu = Path(os.environ["SMOKE_ROOT"])
cpu = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round1/dg50_cpu_7300d_no_closure")
largest = 0.0
finite = True
for day in (1, 2):
    with np.load(gpu / f"day{day:04d}.npz") as left, np.load(cpu / f"day{day:04d}.npz") as right:
        for field in ("ssh_m", "u_m_s", "v_m_s"):
            delta = np.abs(left[field] - right[field])
            finite &= bool(np.all(np.isfinite(delta)))
            largest = max(largest, float(np.max(delta)))
print(f"largest_snapshot_difference={largest:.17g} finite={finite}")
if not finite:
    raise SystemExit(1)
PY
```

2. The two 7,300-day CUDA controls, sequentially on GPU 0.

```bash
export CONTROL_ROOT=$OUT/controls
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py free-run \
  --config config/swg_online/control_nu10_gpu.json \
  --physical-gpu $N --run-root "$CONTROL_ROOT"
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py free-run \
  --config config/swg_online/control_nu80_gpu.json \
  --physical-gpu $N --run-root "$CONTROL_ROOT"
export NU10=$CONTROL_ROOT/swgonl_control_nu10_gpu
export NU80=$CONTROL_ROOT/swgonl_control_nu80_gpu
export NU10_SHA=$(sha256sum "$NU10/manifest.json" | awk '{print $1}')
export NU80_SHA=$(sha256sum "$NU80/manifest.json" | awk '{print $1}')
```

3. Mechanical device check. Continue to trained CUDA scoring only when this
   prints `PASS device-check` and the JSON has `"verdict": "PASS"`.

```bash
export DEVICE=$OUT/device_check_template.json
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$PY" scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py \
  device-check --objective template \
  --control nu10 "$NU10" "$NU10_SHA" \
  --control nu80 "$NU80" "$NU80_SHA" \
  --output "$DEVICE"
export DEVICE_SHA=$(sha256sum "$DEVICE" | awk '{print $1}')
```

4. Pilot training, seed 20261013, batch 10 split into one device call of ten
   windows. The measured profile is 6.2/12.4 s per 1-/2-day training step and
   1.2/1.7 GiB peak device memory.

```bash
export PILOT_ROOT=$OUT/pilot
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py train \
  --config config/swg_online/pilot_template_ring_excluded.json \
  --device cuda --physical-gpu $N --windows-per-call 10 \
  --seed 20261013 --run-root "$PILOT_ROOT"
export PILOT_RUN=$PILOT_ROOT/swgonl_pilot_template_ring_excluded/seed20261013
```

5. One 7,300-day CUDA/native free run for each trained window length.

```bash
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py free-run \
  --config config/swg_online/pilot_template_ring_excluded.json \
  --physical-gpu $N --seed 20261013 --window-days 1 \
  --run-root "$PILOT_ROOT"
env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$N \
  "$PY" scripts/run/run_swg_online_training.py free-run \
  --config config/swg_online/pilot_template_ring_excluded.json \
  --physical-gpu $N --seed 20261013 --window-days 2 \
  --run-root "$PILOT_ROOT"
export W1=$PILOT_RUN/free_runs/w01
export W2=$PILOT_RUN/free_runs/w02
export W1_SHA=$(sha256sum "$W1/manifest.json" | awk '{print $1}')
export W2_SHA=$(sha256sum "$W2/manifest.json" | awk '{print $1}')
```

6. Score each pilot run. These are diagnostic single-seed scores, not the
   unchanged three-seed pass-mark verdict.

```bash
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$PY" scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py \
  score-run --objective template --window-days 1 \
  --run-root "$W1" --manifest-sha256 "$W1_SHA" \
  --device-check "$DEVICE" "$DEVICE_SHA" --output "$OUT/score_w01.json"
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  "$PY" scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py \
  score-run --objective template --window-days 2 \
  --run-root "$W2" --manifest-sha256 "$W2_SHA" \
  --device-check "$DEVICE" "$DEVICE_SHA" --output "$OUT/score_w02.json"
```

The pilot learned if `training_log.json` shows the validation-loss sequence
for all five epochs of each window length responding to optimisation. The same
log must contain `gates.first_step_no_inert = {passed: true, leaves: 9}`. Both
free-run manifests must report `status=complete`, `completed_days=7300`, and
`numerical_health.all_finite=true`. Print those decisive outputs with:

```bash
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "$PY" - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["PILOT_RUN"])
log = json.loads((run / "training_log.json").read_text())
print("validation_loss", [(e["window_days"], e["epoch"], e["validation_loss"]) for e in log["epochs"]])
print("first_step_no_inert", log["gates"]["first_step_no_inert"])
for window in ("w01", "w02"):
    manifest = json.loads((run / "free_runs" / window / "manifest.json").read_text())
    print(window, manifest["status"], manifest["completed_days"], manifest["numerical_health"]["all_finite"])
PY
```

## Validation record

- Round-8 baseline review: `83 passed in 241.64s`.
- Round-9 targeted driver/scorer suite before final additions:
  `87 passed in 245.94s`; new evaluator-focused run: `39 passed in 28.50s`.
- Pilot `check-data`: 67 sources, six split/window batches, contract digest
  `bf97d77a639f80549adf1d03ef00c9840ca5ed0137ad7ac1900aceeaeda0eb5a`.
- Round-8 reversions: 13/13 failed. Round-9 reversions: 6/6 failed.
- Ruff on every touched Python file: `All checks passed!`.

No new scientific choice was made in round 9. The explicit device policy,
control reruns, device criterion, pilot seed/windows/epochs/batch/data/free-run
length, and template/excluded objective are the user's stated decisions.

## Amendment 2026-10-15: device-check tolerance (user decision)

The first real device check (2026-10-14, `device_check_template.json`)
returned FAIL under the round-9 criterion (each metric's GPU-minus-CPU
difference of window medians strictly below the TRUTH's half-range), although
every difference was 0.01-0.64 of the CPU control's own half-range over its
eight scoring windows. The truth half-range is the pass mark's allowance for
the truth's internal spread and is far tighter than two runs of the same model
differ by sampling; four noise-metric differences of order 1e-4 failed against
truth half-ranges of order 1e-6 to 1e-5.

**User decision (2026-10-15):** the device check compares each metric
difference against the CPU control's own half-range over its eight frozen
scoring windows (half of max minus min of its frozen per-window values).
Strict inequality and "both GPU controls still fail the pass mark" are
unchanged; nothing about the pass mark itself (frozen bars, thresholds, noise
allowance) changes. The record carries both tolerances (the control one
decides; the truth one is recorded for information) and the ratio
difference / control half-range. Schema bumped to
`legoesm.swg_online_device_check.v2`; `admit_device_check` refuses v1 records
and re-evaluates every record under the new criterion, so the 2026-10-14 FAIL
record is refused outright ("schema must be ...v2").

Re-run on the same GPU controls (same manifests and digests as step 3 above),
CPU, written to `device_check_template_v2.json` (SHA-256
`8b310ddb09b1313c7884832a07ee632856b86ee0044b39f7fb09e19c60d0bf97`):
**PASS**; both controls still fail the pass mark. Ratios of difference to
control half-range, nu10 / nu80: mean face transport 0.19 / 0.40, KE spectrum
0.33 / 0.25, EKE map 0.34 / 0.25, basin EKE ratio 0.18 / 0.64, top-octave KE
0.29 / 0.08, top-octave enstrophy 0.44 / 0.04, native-grid noise 0.01 / 0.04.
Largest ratio 0.64 (nu80 basin EKE ratio). Use this record and digest as
`$DEVICE` / `$DEVICE_SHA` in the trained-run scoring commands above.

Tests: synthetic PASS between the two tolerances, FAIL just above and exactly
at the control half-range, a v1 record refused, and the ratio arithmetic on a
tiny synthetic frozen gate; all three new tests fail against the round-9
scorer (`3 failed, 2 passed` on a scratch copy). Full scorer suite
`42 passed`; ruff `All checks passed!`.
