# Round-37 scoped U/V momentum-RHS SLOT handoff

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. **CORRECTED AFTER THE CLOSURE
STOP; NOT RERUN.** Producer
`9f1e3dfec4e59891ce40b7cc6089e31b4451a138`.

The stopped capture is not retained: it emitted no metadata. The corrected
producer adds the missing explicit surface-stress diagnostic, admits it as a
coverage-only row outside D03--D06, and requires two new full-stagger files.
All four blocks must rerun because both production diagnostics and script
hashes changed. Each fresh capture now contains 17 files.

Enumeration found that the retained deterministic-writer stack already has
all D03--D06 U/V cumulative streams and the dedicated KEG/ZAD/VOR/LDF/HPG
increments. Therefore this cascade does not copy or build NEMO and applies no
Fortran patch: a redundant writer would violate the existing-dumps-first
rule. Block 1 instead builds/admit-checks the Python producer; block 2 makes
two fresh independent CPU captures; block 3 brackets them byte-exactly; block
4 scores the registered term ladder.

The NEMO inputs are full-halo deterministic streams: each is
`56*203*35*8 = 3,183,040` bytes. Only the manifest-admitted
`kt00005761` cumulative names are used; the old unsuffixed aliases are not
admitted. Twin capture arrays are the model's native full staggered layouts,
U `(199,53,36)` and V `(200,52,36)`. The scorer's cited interior alignment is
U `[:,1:,:35]` and V `[1:,:,:35]`, the established bridge convention used by
`carry_injection_discriminator.py:433-434` and the round-35 raw-dispatch
capture. No loader silently reshapes an interior stream as full halo.

The scorer emits the file:line citations registered for each exact row:
ZAD (`dynadv.F90:97-103`), vorticity/Coriolis
(`dynvor.F90:143-179`), lateral friction (`dynldf.F90:79-115`), the
KE-gradient+HPG group (`dynadv.F90:89-96`; `dynhpg.F90:117-133`), and the
D06 accumulator (`stpmlf.F90:269-270,309-328`). The new coverage row binds
the twin deposit (`ocean_pe_latlon_cgrid.py:3696-3761`) to NEMO's later
surface-stress deposit (`dynzdf.F90:353-363`) and cannot own D03--D06.

Every block changes to its own absolute checkout, uses checkout-first
`PYTHONPATH`, exports the session ID, pins the producer SHA, and uses a
tracked-only cleanliness/model-diff gate. No block pushes, invokes a GPU,
uses `mpirun`, or assumes a retained capture.

## Block 1 — producer admission and Python build

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
producer=9f1e3dfec4e59891ce40b7cc6089e31b4451a138
git cat-file -e "${producer}^{commit}"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
test -z "$(git diff --name-only "$producer" HEAD -- \
  packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_capture.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_bracket.py)"

capture="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_capture.py"
bracket="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_bracket.py"
scorer="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37.py"
prereg="$repo/docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round37.md"
manifest="$repo/docs/ocean/fidelity/dino_zdf_row21_coeff_assembly_artifact.json"
test "$(sha256sum "$capture" | awk '{print $1}')" = fed0b64c3b98910bfe2f8f426c4417a197d1025461b55322e9eac4a7cf69d2ab
test "$(sha256sum "$bracket" | awk '{print $1}')" = 6fc13399285911b68295d20f32a7c1d93e8f1d9aa2225c5f37ff119c22fddfb2
test "$(sha256sum "$scorer" | awk '{print $1}')" = ef6fea962cf4932688b5c075b57b21a5328193231ae87a430244d04893be30f4
test "$(sha256sum "$prereg" | awk '{print $1}')" = 479485589b9dc8852ecc01aad1640f9707b1111bd2610d1838d77cd9d8890564
test "$(sha256sum "$manifest" | awk '{print $1}')" = 84885e45ecc149082606c0b44b411271f40942a99497996b0d4b35e051b6a97a
test "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round36.json | awk '{print $1}')" = f680200f2a3733558d1de7be7f97f5a80e575c003cbebf554fc1aa80e4428270

export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export PYTHONPYCACHEPREFIX=/tmp/dino-row37-pycache
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python -m py_compile "$capture" "$bracket" "$scorer"
/home/dbalwada/legoESM/.venv/bin/python "$capture" --help >/dev/null
/home/dbalwada/legoESM/.venv/bin/python "$bracket" --help >/dev/null
/home/dbalwada/legoESM/.venv/bin/python "$scorer" --help >/dev/null
printf 'SLOT __MEASURED_ROW37_PRODUCER__ VALUE=%s\n' "$producer"
printf 'SLOT __MEASURED_ROW37_CAPTURE_SCRIPT_SHA256__ VALUE=%s\n' "$(sha256sum "$capture" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW37_BRACKET_SCRIPT_SHA256__ VALUE=%s\n' "$(sha256sum "$bracket" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW37_SCORER_SHA256__ VALUE=%s\n' "$(sha256sum "$scorer" | awk '{print $1}')"
```

## Block 2 — two fresh deterministic CPU captures

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW37_PRODUCER__
CAPTURE_SHA=__MEASURED_ROW37_CAPTURE_SCRIPT_SHA256__
for value in "$PRODUCER" "$CAPTURE_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 9f1e3dfec4e59891ce40b7cc6089e31b4451a138
capture="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_capture.py"
test "$(sha256sum "$capture" | awk '{print $1}')" = "$CAPTURE_SHA"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
traj=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
test -f "$traj/DINO_00005760_restart.nc"
test -f "$stepdump/mesh_mask.nc"
A=$(mktemp -d /tmp/dino-row37-capture-A.XXXXXX)
B=$(mktemp -d /tmp/dino-row37-capture-B.XXXXXX)
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
for out in "$A" "$B"; do
  cd "$repo"
  /home/dbalwada/legoESM/.venv/bin/python "$capture" \
    --run-stepdump "$stepdump" --run-traj "$traj" \
    --restart-file DINO_00005760_restart.nc \
    --round36 /tmp/dino_split_explicit_momentum_chain_round36.json \
    --output-dir "$out"
  test "$(find "$out" -maxdepth 1 -type f | wc -l)" -eq 17
done
printf '%s\n' "$A" > /tmp/row37-capture-a.txt
printf '%s\n' "$B" > /tmp/row37-capture-b.txt
printf 'SLOT __MEASURED_ROW37_CAPTURE_A_SHA256__ VALUE=%s\n' "$(sha256sum "$A/capture.json" | awk '{print $1}')"
printf 'SLOT __MEASURED_ROW37_CAPTURE_B_SHA256__ VALUE=%s\n' "$(sha256sum "$B/capture.json" | awk '{print $1}')"
```

## Block 3 — exact 17/17 duplicate-capture bracket

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
BRACKET_SCRIPT_SHA=__MEASURED_ROW37_BRACKET_SCRIPT_SHA256__
CAPTURE_A_SHA=__MEASURED_ROW37_CAPTURE_A_SHA256__
CAPTURE_B_SHA=__MEASURED_ROW37_CAPTURE_B_SHA256__
for value in "$BRACKET_SCRIPT_SHA" "$CAPTURE_A_SHA" "$CAPTURE_B_SHA"; do case "$value" in __*) exit 2;; esac; done
A=$(cat /tmp/row37-capture-a.txt)
B=$(cat /tmp/row37-capture-b.txt)
test "$(sha256sum "$A/capture.json" | awk '{print $1}')" = "$CAPTURE_A_SHA"
test "$(sha256sum "$B/capture.json" | awk '{print $1}')" = "$CAPTURE_B_SHA"
bracket="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_bracket.py"
test "$(sha256sum "$bracket" | awk '{print $1}')" = "$BRACKET_SCRIPT_SHA"
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
/home/dbalwada/legoESM/.venv/bin/python "$bracket" \
  --capture-a "$A" --capture-b "$B" \
  --output /tmp/dino_split_explicit_momentum_chain_round37_bracket.json
printf 'SLOT __MEASURED_ROW37_BRACKET_SHA256__ VALUE=%s\n' "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round37_bracket.json | awk '{print $1}')"
```

## Block 4 — scoped U/V D03--D06 score

```bash
set -euo pipefail
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
repo=/tmp/codex-zdf-sweep
cd "$repo"
PRODUCER=__MEASURED_ROW37_PRODUCER__
SCORER_SHA=__MEASURED_ROW37_SCORER_SHA256__
BRACKET_SHA=__MEASURED_ROW37_BRACKET_SHA256__
for value in "$PRODUCER" "$SCORER_SHA" "$BRACKET_SHA"; do case "$value" in __*) exit 2;; esac; done
test "$PRODUCER" = 9f1e3dfec4e59891ce40b7cc6089e31b4451a138
test -z "$(git diff --name-only "$PRODUCER" HEAD -- packages/core packages/ocean \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_capture.py \
  scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37_bracket.py)"
scorer="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round37.py"
test "$(sha256sum "$scorer" | awk '{print $1}')" = "$SCORER_SHA"
test "$(sha256sum /tmp/dino_split_explicit_momentum_chain_round37_bracket.json | awk '{print $1}')" = "$BRACKET_SHA"
A=$(cat /tmp/row37-capture-a.txt)
stepdump=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_D180_1R
export PYTHONPATH="$repo/packages/core:$repo/packages/ocean:$repo:$repo/scripts/validate/ocean_fidelity/dino_1226"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both DINO_1226_LANE=d180
/home/dbalwada/legoESM/.venv/bin/python "$scorer" \
  --capture "$A" --run-stepdump "$stepdump" \
  --round36 /tmp/dino_split_explicit_momentum_chain_round36.json \
  --bracket /tmp/dino_split_explicit_momentum_chain_round37_bracket.json \
  --bracket-sha "$BRACKET_SHA" \
  --manifest-artifact "$repo/docs/ocean/fidelity/dino_zdf_row21_coeff_assembly_artifact.json" \
  --nemo-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2 \
  --output /tmp/dino_split_explicit_momentum_chain_round37.json
sha256sum /tmp/dino_split_explicit_momentum_chain_round37.json
```
