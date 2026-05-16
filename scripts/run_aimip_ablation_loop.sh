#!/usr/bin/env bash
# AIMIP classical greedy ablation -- one subprocess per variant so the
# JAX compilation-cache footprint never accumulates across runs.
#
# Stages: GWD -> convection -> turbulence -> microphysics -> cloud.
# Each stage trains every alternative on top of the previous stage's
# winner; the lowest (T_RMSE + |T_bias|) attempt is the winner.

set -uo pipefail

OUT_DIR="results/aimip_001/ablation"
mkdir -p "$OUT_DIR"

SUITE="config/aimip/aimip_suite.yaml"

# Initial scheme set (matches AIMIP baseline).
GWD="mcfarlane"
CONV="tiedtke"
TURB="louis"
MICRO="none"
CLOUD="xu_randall"

run_variant() {
  local label="$1"
  local out_path="$OUT_DIR/${label}.json"
  if [[ -f "$out_path" ]]; then
    echo "[skip] $label already complete -> $out_path"
    return 0
  fi
  echo "================================================================"
  echo "[run] $label  schemes: gwd=$GWD conv=$CONV turb=$TURB micro=$MICRO cloud=$CLOUD"
  echo "================================================================"
  JAX_ENABLE_X64=1 .venv/bin/python -u scripts/_aimip_ablation_single.py \
    --label "$label" \
    --gwd-scheme "$GWD" \
    --convection-scheme "$CONV" \
    --turbulence-scheme "$TURB" \
    --microphysics-scheme "$MICRO" \
    --cloud-scheme "$CLOUD" \
    --suite "$SUITE" \
    --out "$out_path"
}

pick_winner() {
  python3 - "$OUT_DIR" "$@" <<'PYEOF'
import json, sys, os
out_dir = sys.argv[1]
labels = sys.argv[2:]
best = None
best_obj = float("inf")
for lab in labels:
    path = os.path.join(out_dir, f"{lab}.json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if d.get("status") != "ok":
        continue
    T = d.get("metrics", {}).get("T", {})
    obj = T.get("rmse", float("inf")) + abs(T.get("bias", 0.0))
    if obj < best_obj:
        best, best_obj = lab, obj
print(best or labels[0])
PYEOF
}

scheme_of() {
  local label="$1"; local dim="$2"
  python3 - "$OUT_DIR" "$label" "$dim" <<'PYEOF'
import json, sys, os
out_dir, label, dim = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(os.path.join(out_dir, f"{label}.json")))
print(d["schemes"][dim])
PYEOF
}

# --------------------------------------------------------------------
# Baseline (cached from prior run if available).
# --------------------------------------------------------------------
run_variant "baseline"

# --------------------------------------------------------------------
# Stage 1: Gravity wave drag.
# --------------------------------------------------------------------
for cand in lindzen hines rayleigh; do
  GWD="$cand"
  run_variant "gwd_${cand}"
done
GWD=$(scheme_of "$(pick_winner baseline gwd_lindzen gwd_hines gwd_rayleigh)" gwd_scheme)
echo "[winner stage gwd] -> $GWD"

# --------------------------------------------------------------------
# Stage 2: Convection.
# --------------------------------------------------------------------
for cand in sbm emanuel; do
  CONV="$cand"
  run_variant "conv_${cand}"
done
CONV=$(scheme_of "$(pick_winner baseline conv_sbm conv_emanuel)" convection_scheme)
echo "[winner stage convection] -> $CONV"

# --------------------------------------------------------------------
# Stage 3: Turbulence.
# --------------------------------------------------------------------
for cand in tke smagorinsky; do
  TURB="$cand"
  run_variant "turb_${cand}"
done
TURB=$(scheme_of "$(pick_winner baseline turb_tke turb_smagorinsky)" turbulence_scheme)
echo "[winner stage turbulence] -> $TURB"

# --------------------------------------------------------------------
# Stage 4: Microphysics.
# --------------------------------------------------------------------
for cand in sundqvist; do
  MICRO="$cand"
  run_variant "micro_${cand}"
done
MICRO=$(scheme_of "$(pick_winner baseline micro_sundqvist)" microphysics_scheme)
echo "[winner stage microphysics] -> $MICRO"

# --------------------------------------------------------------------
# Stage 5: Cloud fraction.
# --------------------------------------------------------------------
for cand in sundqvist; do
  CLOUD="$cand"
  run_variant "cloud_${cand}"
done
CLOUD=$(scheme_of "$(pick_winner baseline cloud_sundqvist)" cloud_scheme)
echo "[winner stage cloud] -> $CLOUD"

echo "================================================================"
echo "[FINAL] gwd=$GWD conv=$CONV turb=$TURB micro=$MICRO cloud=$CLOUD"
echo "================================================================"
