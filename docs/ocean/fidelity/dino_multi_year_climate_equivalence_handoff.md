# 20-year DINO climate equivalence: exact SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. **PREREGISTERED, NOT RUN.** These
blocks create the raw ensemble registered by
`PREREG_multi_year_climate_equivalence.md`. They do not score or interpret it;
the frozen scorer and its direct tests/plants must be committed before a
result is read.

The legoESM producer is SHA-pinned. Every Python invocation carries the
checkout `PYTHONPATH`; every block changes into its own directory; porcelain
gates are tracked-only. The NEMO blocks run outside the sandbox and are the
only blocks containing `mpirun`. They copy only named namelists and restart
files, never `RUN_*`, `BLD`, or an oracle source tree wholesale.

## Block 1 — common checkout and admissions (CPU only)

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
producer=ddd3a8476afd87da4afa5747eb7e5057490b893c
run_root=/tmp/dino-climate-equivalence-20y-01a04e34
checkout="$run_root/producer-checkout"
dino=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_cert="$dino/BLD/bin/nemo.exe.certified_d3cf9242"
nemo_start="$dino/RUN_90D_TWIN/DINO_00005760_restart.nc"

cd "$repo"
test ! -e "$run_root"
free_kb=$(df -Pk /tmp | awk 'NR==2 {print $4}')
test "$free_kb" -ge 31457280 || {
  echo "STOP: legoESM capstone root requires at least 30 GiB free in /tmp" >&2
  exit 1
}
git cat-file -e "$producer^{commit}"
mkdir -p "$run_root/arms" "$run_root/logs" "$run_root/manifests"
git worktree add --detach "$checkout" "$producer"
test "$(git -C "$checkout" rev-parse HEAD)" = "$producer"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
checkout_kb=$(du -sk "$checkout" | awk '{print $1}')
test "$checkout_kb" -le 5242880 || {
  echo "STOP: checkout exceeds the 5 GiB lean-tree guard" >&2
  exit 1
}
test "$(sha256sum "$nemo_cert" | awk '{print $1}')" = \
  decd157807f992566a7c6184cd3610a0d5021d87d109ad2abd08b4f9c0528661
test "$(sha256sum "$nemo_start" | awk '{print $1}')" = \
  0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e
test "$(sha256sum "$dino/RUN_90D_TWIN/namelist_cfg" | awk '{print $1}')" = \
  cc239f92d9f613f7507e14c7b9ed6c1f8d2afa587e678e95fa29c3343dcbed2b
test "$(sha256sum "$dino/RUN_90D_TWIN/namelist_ref" | awk '{print $1}')" = \
  b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd
test "$(sha256sum "$dino/RUN_TRAJ/mesh_mask.nc" | awk '{print $1}')" = \
  3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622
printf '%s\n' "$producer" > "$run_root/producer_commit.txt"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
cd "$checkout"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  /home/dbalwada/legoESM/.venv/bin/python -m pytest -q \
  tests/ocean/unit/test_dino_1226_instruments.py \
  tests/ocean/unit/test_dino_1455_verdict360.py \
  tests/ocean/unit/test_dino_1492_acceptance_gate_90d.py \
  -k 'snapshot or reduced or perturb or bridge_before'
printf 'producer=%s checkout_kb=%s free_tmp_kb=%s session=%s\n' \
  "$producer" "$checkout_kb" "$free_kb" "$CODEX_SESSION_ID"
```

Expected time: 3--8 minutes CPU.

## Block 2 — NEMO prefix gates and phase-A setup (outside sandbox, no run)

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-climate-equivalence-20y-01a04e34
checkout="$run_root/producer-checkout"
producer=ddd3a8476afd87da4afa5747eb7e5057490b893c
dino=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_root="$dino/RUN_EQ20Y_01a04e34"
cert="$dino/BLD/bin/nemo.exe.certified_d3cf9242"

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$nemo_root"
free_kb=$(df -Pk "$dino" | awk 'NR==2 {print $4}')
test "$free_kb" -ge 167772160 || {
  echo "STOP: NEMO capstone requires at least 160 GiB free" >&2
  exit 1
}
expected=(
  104f520b0294d7b87342e6721488baffd82774983a493f4430ab1858a103c6f7
  d90bf4bf98f4517fceb45619d0e03484dbd88e1417d2323ab120212647801d00
  d91085b350f5a87df396482c02b88e17982968c7a1f5d1700472584175fbf560
)
for m in 0 1 2; do
  src="$dino/RUN_VERDICT360_M$m"
  test "$(find "$src" -maxdepth 1 -name 'DINO_00017280_restart_*.nc' | wc -l)" -eq 16
  got=$(cd "$src" && sha256sum DINO_00017280_restart_*.nc | sha256sum | awk '{print $1}')
  test "$got" = "${expected[$m]}"
  grep -Fx 'STOP 0' "$src/run_verdict360_m$m.log"
done

pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" /home/dbalwada/legoESM/.venv/bin/python - \
  "$dino" "$nemo_root" "$cert" <<'PY'
from pathlib import Path
import json
import os
import re
import shutil
import sys

dino, root, cert = map(Path, sys.argv[1:])
root.mkdir(parents=True)
src_nml = dino / "RUN_90D_TWIN" / "namelist_cfg"
src_ref = dino / "RUN_90D_TWIN" / "namelist_ref"

def rewrite(dst, *, it000, itend, stock, restart):
    replacements = {
        "nn_it000": f"   nn_it000    =   {it000}",
        "nn_itend": f"   nn_itend    =   {itend}",
        "nn_stock": f"   nn_stock    =   {stock}",
        "cn_ocerst_in": f'      cn_ocerst_in = "{restart}"',
    }
    hits = {k: 0 for k in replacements}
    out = []
    for line in src_nml.read_text().splitlines(True):
        stripped = line.lstrip()
        changed = False
        for key, value in replacements.items():
            if not stripped.startswith("!") and re.match(rf"{key}\\s*=", stripped):
                out.append(value + "\n")
                hits[key] += 1
                changed = True
                break
        if not changed:
            out.append(line)
    if hits != {k: 1 for k in replacements}:
        raise SystemExit(f"active namelist hit mismatch: {hits}")
    (dst / "namelist_cfg").write_text("".join(out))
    shutil.copy2(src_ref, dst / "namelist_ref")
    os.symlink(cert, dst / "nemo")

sys.path.insert(0, str(Path.cwd() / "scripts/validate/ocean_fidelity/dino_1226"))
import perturb_nemo_tn_90d as perturb

receipts = {}
for m, seed in enumerate((None, 1, 2, 3, 4, 5)):
    dst = root / f"m{m}" / "phase_a"
    dst.mkdir(parents=True)
    if m <= 2:
        prefix = dino / f"RUN_VERDICT360_M{m}"
        for src in sorted(prefix.glob("DINO_00017280_restart_*.nc")):
            shutil.copy2(src, dst / src.name)
        it000, restart = 17281, "DINO_00017280_restart"
        receipts[str(m)] = {"prefix": str(prefix), "seed": seed}
    else:
        rep = perturb.perturb(seed, dst)
        it000, restart = 5761, "DINO_00005760_restart"
        receipts[str(m)] = {"prefix": None, "seed": seed, "perturb": rep}
    rewrite(dst, it000=it000, itend=178560, stock=5760, restart=restart)
(root / "setup_receipt.json").write_text(json.dumps(receipts, indent=2, sort_keys=True))
PY

for m in 0 1 2 3 4 5; do
  d="$nemo_root/m$m/phase_a"
  test -x "$d/nemo"
  grep -E '^ *nn_it000|^ *nn_itend|^ *nn_stock|^ *cn_ocerst_in' "$d/namelist_cfg"
done
printf 'SLOT __MEASURED_NEMO_SETUP_RECEIPT_SHA256__ VALUE=%s\n' \
  "$(sha256sum "$nemo_root/setup_receipt.json" | awk '{print $1}')"
printf 'nemo_root=%s free_kb=%s session=%s\n' \
  "$nemo_root" "$free_kb" "$CODEX_SESSION_ID"
```

Expected time: 5--15 minutes, mostly copying named restart files. The 160-GiB
guard is against the complete six-member output, not this setup block.

## Block 3 — NEMO phase A, through year 15 (outside sandbox, `mpirun`)

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
dino=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_root="$dino/RUN_EQ20Y_01a04e34"
mpirun=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpirun

cd "$nemo_root"
test -s setup_receipt.json
for m in 0 1 2 3 4 5; do
  d="$nemo_root/m$m/phase_a"
  cd "$d"
  test ! -e ocean.output
  "$mpirun" -np 16 ./nemo > "run_m${m}_phase_a.log" 2>&1
  grep -Fx 'STOP 0' "run_m${m}_phase_a.log"
  test "$(find . -maxdepth 1 -name 'DINO_00178560_restart_*.nc' | wc -l)" -eq 16
  (sha256sum DINO_00178560_restart_*.nc) > "kt178560_sha256.txt"
  printf 'SLOT __MEASURED_NEMO_M%s_PHASE_A_SHA256__ VALUE=%s\n' \
    "$m" "$(sha256sum kt178560_sha256.txt | awk '{print $1}')"
done
```

Expected time: about 13--16 hours serial total (members 0--2 run 14 years;
members 3--5 run 15 years). Do not parallelize these filesystem-heavy NEMO
members; the registered four-wide trial was slower in aggregate.

## Block 4 — NEMO phase-B setup and monthly years 16--20 (outside sandbox, `mpirun`)

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-climate-equivalence-20y-01a04e34
checkout="$run_root/producer-checkout"
producer=ddd3a8476afd87da4afa5747eb7e5057490b893c
dino=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_root="$dino/RUN_EQ20Y_01a04e34"
cert="$dino/BLD/bin/nemo.exe.certified_d3cf9242"
mpirun=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpirun

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" /home/dbalwada/legoESM/.venv/bin/python - \
  "$nemo_root" "$cert" <<'PY'
from pathlib import Path
import os
import re
import shutil
import sys

root, cert = map(Path, sys.argv[1:])
for m in range(6):
    src = root / f"m{m}" / "phase_a"
    dst = root / f"m{m}" / "phase_b"
    if dst.exists():
        raise SystemExit(f"refusing existing phase-B directory: {dst}")
    dst.mkdir()
    for f in sorted(src.glob("DINO_00178560_restart_*.nc")):
        shutil.copy2(f, dst / f.name)
    shutil.copy2(src / "namelist_ref", dst / "namelist_ref")
    replacements = {
        "nn_it000": "   nn_it000    =   178561",
        "nn_itend": "   nn_itend    =   236160",
        "nn_stock": "   nn_stock    =   960",
        "cn_ocerst_in": '      cn_ocerst_in = "DINO_00178560_restart"',
    }
    hits = {k: 0 for k in replacements}
    out = []
    for line in (src / "namelist_cfg").read_text().splitlines(True):
        stripped = line.lstrip()
        changed = False
        for key, value in replacements.items():
            if not stripped.startswith("!") and re.match(rf"{key}\\s*=", stripped):
                out.append(value + "\n")
                hits[key] += 1
                changed = True
                break
        if not changed:
            out.append(line)
    if hits != {k: 1 for k in replacements}:
        raise SystemExit(f"m{m}: active namelist hit mismatch: {hits}")
    (dst / "namelist_cfg").write_text("".join(out))
    os.symlink(cert, dst / "nemo")
PY

for m in 0 1 2 3 4 5; do
  d="$nemo_root/m$m/phase_b"
  cd "$d"
  test ! -e ocean.output
  "$mpirun" -np 16 ./nemo > "run_m${m}_phase_b.log" 2>&1
  grep -Fx 'STOP 0' "run_m${m}_phase_b.log"
  test "$(find . -maxdepth 1 -name 'DINO_00236160_restart_*.nc' | wc -l)" -eq 16
  monthly_count=0
  for kt in $(seq 179520 960 236160); do
    n=$(find . -maxdepth 1 -name "DINO_$(printf '%08d' "$kt")_restart_*.nc" | wc -l)
    test "$n" -eq 16
    monthly_count=$((monthly_count + 1))
  done
  test "$monthly_count" -eq 60
  find . -maxdepth 1 -name 'DINO_*_restart_*.nc' -print0 | sort -z | \
    xargs -0 sha256sum > "phase_b_restart_sha256.txt"
  printf 'SLOT __MEASURED_NEMO_M%s_PHASE_B_SHA256__ VALUE=%s\n' \
    "$m" "$(sha256sum phase_b_restart_sha256.txt | awk '{print $1}')"
done
```

Expected time: 4--5 hours serial total. Phase-B output is the storage-heavy
piece. Keep every raw restart until the compact scorer artifact and its
manifest have passed; no cleanup command is included here.

## Block 5 — six fresh legoESM members, three two-GPU waves

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export FP64=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both
run_root=/tmp/dino-climate-equivalence-20y-01a04e34
checkout="$run_root/producer-checkout"
producer=ddd3a8476afd87da4afa5747eb7e5057490b893c
run_traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
run_stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
run_fp64="$checkout/scripts/validate/ocean_fidelity/dino_1226/run_fp64.py"
twin="$checkout/scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"
snap_days=$(PYTHONPATH="$pythonpath" /home/dbalwada/legoESM/.venv/bin/python - <<'PY'
days = list(range(360, 5401, 360)) + list(range(5430, 7201, 30))
assert len(days) == 75 and days[-1] == 7200 and len(set(days)) == 75
print(",".join(map(str, days)))
PY
)

run_member() {
  gpu=$1
  member=$2
  seed=$3
  out="$run_root/arms/m${member}.npz"
  log="$run_root/logs/m${member}.log"
  test ! -e "$out"
  test ! -e "$log"
  extra=()
  if [ "$seed" != control ]; then
    extra=(--perturb-seed "$seed")
  fi
  PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES="$gpu" \
    /home/dbalwada/legoESM/.venv/bin/python "$run_fp64" "$twin" \
    nemo_dino_kamm_mlf "$out" \
    --days 7200 --save-3d --snap-days "$snap_days" --fp64-3d --daily-acc \
    --run-traj "$run_traj" --run-stepdump "$run_stepdump" \
    --bridge-tke --bridge-before --bridge-before-stress-tpoint \
    "${extra[@]}" > "$log" 2>&1
}

seeds=(control 1 2 3 4 5)
for first in 0 2 4; do
  second=$((first + 1))
  run_member 0 "$first" "${seeds[$first]}" & p0=$!
  run_member 1 "$second" "${seeds[$second]}" & p1=$!
  rc=0
  wait "$p0" || rc=1
  wait "$p1" || rc=1
  test "$rc" -eq 0
done

for m in 0 1 2 3 4 5; do
  grep -F "SAVED $run_root/arms/m$m.npz  stable=True" "$run_root/logs/m$m.log"
  printf 'SLOT __MEASURED_LEGO_M%s_SHA256__ VALUE=%s\n' \
    "$m" "$(sha256sum "$run_root/arms/m$m.npz" | awk '{print $1}')"
done
```

Expected time: 12--15 hours wall in three two-GPU waves. Each artifact is
expected to be roughly 1--3 GiB. The block never resumes a partial artifact;
an absent final `.npz` is rerun as the same member.

## Block 6 — raw-data bracket and SLOT manifest (CPU only)

This block validates dates, dtypes, producer/config equality, and all required
NEMO files. It intentionally does not compute a climate verdict.

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
run_root=/tmp/dino-climate-equivalence-20y-01a04e34
checkout="$run_root/producer-checkout"
producer=ddd3a8476afd87da4afa5747eb7e5057490b893c
dino=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO
nemo_root="$dino/RUN_EQ20Y_01a04e34"
out="$run_root/manifests/raw_capstone_manifest.json"

cd "$checkout"
test "$(git rev-parse HEAD)" = "$producer"
test "$(cat "$run_root/producer_commit.txt")" = "$producer"
test -z "$(git status --porcelain --untracked-files=no)"
pythonpath="$checkout/packages/core:$checkout/packages/ocean:$checkout/packages/atmosphere:$checkout/packages/coupler:$checkout/packages/ice:$checkout/packages/land:$checkout/packages/ml:$checkout/packages/tools:$checkout:$checkout/scripts/validate/ocean_fidelity/dino_1226"
PYTHONPATH="$pythonpath" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python - \
  "$run_root" "$nemo_root" "$producer" "$CODEX_SESSION_ID" "$out" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

run_root, nemo_root = map(Path, sys.argv[1:3])
producer, session, out = sys.argv[3], sys.argv[4], Path(sys.argv[5])
days = np.array(list(range(360, 5401, 360)) + list(range(5430, 7201, 30)))
assert days.size == 75
members = []
config0 = None
for m in range(6):
    path = run_root / "arms" / f"m{m}.npz"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with np.load(path, allow_pickle=False) as z:
        got_days = np.asarray(z["snap_days"], dtype=np.int64)
        red_days = np.asarray(z["reduced_days"], dtype=np.int64)
        if not np.array_equal(got_days, days) or not np.array_equal(red_days, days):
            raise SystemExit(f"m{m}: snapshot/reduced dates differ from prereg")
        cfg = json.loads(str(z["run_config"]))
        if int(cfg["n_days"]) != 7200 or cfg["recipe"] != "nemo_dino_kamm_mlf":
            raise SystemExit(f"m{m}: wrong horizon or recipe")
        if str(z["producer_git_sha"]) != producer:
            raise SystemExit(f"m{m}: wrong producer {z['producer_git_sha']}")
        storage = json.loads(str(z["storage_dtypes"]))
        if any(storage[k] != "float64" for k in ("T3d", "S3d", "eta3d", "u3d", "v3d", "reduced")):
            raise SystemExit(f"m{m}: non-fp64 storage/reduction {storage}")
        cfg_cmp = dict(cfg)
        cfg_cmp.pop("perturb_seed", None)
        if config0 is None:
            config0 = cfg_cmp
        elif cfg_cmp != config0:
            raise SystemExit(f"m{m}: resolved config differs beyond seed")
        members.append({"member": m, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

nemo = []
for m in range(6):
    phase_a = nemo_root / f"m{m}" / "phase_a"
    phase_b = nemo_root / f"m{m}" / "phase_b"
    for day in range(360, 5401, 360):
        kt = 5760 + 32 * day
        source = (Path(f"/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_VERDICT360_M{m}")
                  if m <= 2 and day == 360 else phase_a)
        if len(list(source.glob(f"DINO_{kt:08d}_restart_*.nc"))) != 16:
            raise SystemExit(f"m{m} day {day}: missing NEMO tile set in {source}")
    for day in range(5430, 7201, 30):
        kt = 5760 + 32 * day
        if len(list(phase_b.glob(f"DINO_{kt:08d}_restart_*.nc"))) != 16:
            raise SystemExit(f"m{m} day {day}: missing NEMO phase-B tile set")
    for phase in (phase_a, phase_b):
        log = phase / f"run_m{m}_{phase.name}.log"
        if "STOP 0" not in log.read_text().splitlines():
            raise SystemExit(f"m{m}: no exact STOP 0 in {log}")
    nemo.append({"member": m, "phase_a": str(phase_a), "phase_b": str(phase_b)})

receipt = {"schema": "dino-climate-equivalence-20y-raw-v1",
           "producer": producer, "session_id": session,
           "days": days.tolist(), "lego_members": members,
           "nemo_members": nemo, "verdict": "NOT_SCORED"}
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(f"RAW_CAPSTONE_ADMITTED members=6+6 snapshots=75 verdict=NOT_SCORED")
PY

sha256sum "$out" "$nemo_root/setup_receipt.json" "$run_root"/arms/*.npz
printf 'SLOT __MEASURED_CAPSTONE_RAW_MANIFEST_SHA256__ VALUE=%s\n' \
  "$(sha256sum "$out" | awk '{print $1}')"
printf 'producer=%s session=%s\n' "$producer" "$CODEX_SESSION_ID"
```

Expected time: 2--6 minutes plus hashing the six legoESM artifacts. A later
score block must consume this manifest by hash, run all preregistered plants,
and write the sole climate verdict JSON. `NOT_SCORED` is deliberate: raw-arm
completion is not a scientific result.
