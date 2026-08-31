#!/usr/bin/env bash
set -euo pipefail
producer=$1
output=${2:-/tmp/dino_split_explicit_momentum_chain_round88.json}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$repo"
git cat-file -e "$producer^{commit}"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git diff --name-only "$producer" HEAD -- packages/core packages/ocean scripts/validate/ocean_fidelity/dino_1226)"
export CODEX_SESSION_ID=01a04e34-d1fb-73e0-b25a-177641f0a246
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$repo/scripts/validate/ocean_fidelity/dino_1226:$repo/packages/core:$repo/packages/ocean:$repo"
scorer="$repo/scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round55.py"
args=(
  --held-dir /tmp/RUN_LATERAL_ROW8_PU_ON.Qln5u8
  --run-traj /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ
  --round54 /tmp/dino_split_explicit_momentum_chain_round54.json
  --round56 /tmp/dino_split_explicit_momentum_chain_round58_unheld.json
  --round59 /tmp/dino_split_explicit_momentum_chain_round59_held.json
)
for n in $(seq 60 77); do args+=(--round$n /tmp/dino_split_explicit_momentum_chain_round$n.json); done
args+=(
  --round78 /tmp/dino_split_explicit_momentum_chain_round78_rescore.json
  --round79 /tmp/dino_split_explicit_momentum_chain_round79.json
  --round80 /tmp/dino_split_explicit_momentum_chain_round80.json
  --round81 /tmp/dino_split_explicit_momentum_chain_round81_rescore.json
)
for n in $(seq 82 87); do args+=(--round$n /tmp/dino_split_explicit_momentum_chain_round$n.json); done
args+=(
  --hold-slow-forcing --oracle-transport --capture-cycle --direct-cycle-entry
  --live-thickness-entry --capture-bolus-operands --capture-kappa-operands
  --literal-kappa-reduction --capture-kappa-geometry --coupled-kappa-carry
  --surface-kmm-carry --exact-surface-kmm-carry --post-chain-factorial
  --rossby-factorial --zn-sqrt-factorial --exact-sqrt-production
  --capture-redi-tail --redi-e3w-factorial --redi-flux-ladder
  --redi-zfu-operand-ladder --redi-zfu-postfix --redi-zfu-kmm-postfix
  --redi-zfu-slope-kmm-postfix --redi-zfu-kmm-operator-postfix
  --redi-zfu-bolus-stage-split-postfix --redi-zfu-ahtu-postfix
  --redi-zfw-association-factorial --redi-zfw-component-score
  --redi-zfw-skew-factorial
  --redi-run-dir /tmp/RUN_BN2_CERT_1R
  --redi-flux-run-dir /tmp/dino-redi-flux-round78-01a04e34/on
  --redi-flux-bracket /tmp/dino-redi-flux-round78-01a04e34/redi_flux_bracket.json
  --redi-flux-bracket-sha f860de191c12970ca6726f17579f9c30190937decc907b0a8cf7f5f68a392890
  --redi-zfw-component-dir /tmp/dino-redi-zfw-round87-01a04e34/on
  --redi-zfw-component-bracket /tmp/dino_redi_zfw_round87_bracket.json
  --redi-zfw-component-bracket-sha af97ff4f3d4aa95dd3595edbb0d1a690d2b115c7c1a02e8a9517655d94d80776
  --raw-artifact /tmp/dino_lateral_row8_pu_held_artifact.json
  --nemo-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
  --output "$output"
)
python3 "$scorer" "${args[@]}"
