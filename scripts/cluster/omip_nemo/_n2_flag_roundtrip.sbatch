#!/bin/bash
#SBATCH --job-name=n2flag
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n2flag_%j.out
#
# --tke-n2-mode / --tke-n2-eos-form must REACH the config, not merely parse.
# A flag that is accepted and then dropped is the dispatch footgun this repo
# has a standing rule about, and it is exactly what was just found in
# compare_tendencies_nemo.py's Stage A2.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/run"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u - <<'PY'
from run_omip_core2 import (orca1_zdftke_config, build_tripole_vmix_config,
                            _validate_tke_card_grid)

# 1. the card's own defaults
c = orca1_zdftke_config()
print(f"[card]      n2_mode={c.n2_mode!r}  n2_eos_form={c.n2_eos_form!r}")
assert c.n2_mode == "nemo_bn2" and c.n2_eos_form == "teos10"

# 2. the override reaches TKEConfig through the builder the tripole uses
v = build_tripole_vmix_config("tke", tke_n2_mode="insitu")
print(f"[override]  n2_mode={v.tke.n2_mode!r}  n2_eos_form={v.tke.n2_eos_form!r}")
assert v.tke.n2_mode == "insitu", "the flag did not reach TKEConfig"

# 3. and it does not disturb the rest of the card
base = build_tripole_vmix_config("tke")
for f in base.tke._fields:
    if f == "n2_mode":
        continue
    a, b = getattr(base.tke, f), getattr(v.tke, f)
    assert a == b, f"override changed an unrelated field: {f} {a!r} -> {b!r}"
print(f"[isolation] all {len(base.tke._fields) - 1} other TKE fields unchanged")

# 4. unknown values raise rather than silently defaulting
for bad in ("nemo_bn3", "TEOS10", ""):
    try:
        build_tripole_vmix_config("tke", tke_n2_mode=bad)
    except ValueError:
        pass
    else:
        raise SystemExit(f"FATAL: n2_mode={bad!r} did not raise")
print("[dispatch]  unknown n2_mode raises ValueError")

# 5. the flag is rejected where it would be silently discarded
for grid, vmix in (("mpas", "none"), ("tripole", "none"), ("tripole", "kpp")):
    try:
        _validate_tke_card_grid(grid, vmix, tke_n2_mode="insitu")
    except SystemExit:
        pass
    else:
        raise SystemExit(f"FATAL: --tke-n2-mode accepted on {grid}/{vmix}")
_validate_tke_card_grid("tripole", "tke", tke_n2_mode="insitu")
print("[guard]     rejected off tripole+tke, accepted on it")
print("\nALL ROUNDTRIP CHECKS PASSED")
PY
echo "=== rc=$?"
