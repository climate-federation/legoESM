#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pcell_grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pcellgrad_%j.out

# Pragmatic stable-run attempt: partial-cell fix + gradual spin-up.
# Plain-z* nudge(60)+drag(1,120) reached day 90 then blew up; with the partial-cell
# fix (Indonesian super-exponential seed removed) it may go further / stay stable.
#   ARM A: --partial-cell + nudge-WOA tau60 + drag tau1/120d, 180 days, smc03 PGF.
#   ARM B: same but drag tau5/longer + RELEASE nudge at day 120 (toward free-run).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM A: --partial-cell smc03 + nudge60 + drag(1,120), 180d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --pgf-scheme smc03 \
  --nudge-woa-tau-days 60 --spinup-drag-tau-days 1 --spinup-drag-days 120 \
  --years 0.5 --dt 600 --diag-every-days 5 \
  --output results/omip_nemo/pcell_grad_A 2>&1 \
  | grep -vE "Constant folding|This isn|If you|reduce-window|slow_operation|^$"
echo "[armA_rc]=${PIPESTATUS[0]}"

echo; echo "### ARM B: --partial-cell smc03 + nudge60(release d120) + drag(5,120), 180d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --pgf-scheme smc03 \
  --nudge-woa-tau-days 60 --nudge-release-day 120 \
  --spinup-drag-tau-days 5 --spinup-drag-days 120 \
  --years 0.5 --dt 600 --diag-every-days 5 \
  --output results/omip_nemo/pcell_grad_B 2>&1 \
  | grep -vE "Constant folding|This isn|If you|reduce-window|slow_operation|^$"
echo "[armB_rc]=${PIPESTATUS[0]}"
echo; echo "=== DONE ==="; date
