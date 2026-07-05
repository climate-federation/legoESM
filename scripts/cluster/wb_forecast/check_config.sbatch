#!/bin/bash
#SBATCH --job-name=wb_cfgcheck
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_wbforecast/results/wb_sweep_stage1/slurm_logs/cfgcheck_%j.out
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:15:00
# Non-smoke config-build check: merge the real T63 baseline + a combo overlay and
# call run_aimip._build_spectral_config (CPU, no training) to confirm the T63
# config parses (catches the nlev/n_levels KeyError that --smoke masks).
set -euo pipefail
WORKTREE=/burg-archive/glab/users/pg2328/legoESM_wbforecast
cd "$WORKTREE"
source scripts/cluster/wb_forecast/env.sh
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
mkdir -p results/wb_sweep_stage1/slurm_logs
"$WB_PYTHON" - <<'PY'
import importlib.util, yaml
suite = yaml.safe_load(open("config/wb/sweep/stage1/combo_baseline/suite.yaml"))
base = yaml.safe_load(open(suite["base"]))
overlay = yaml.safe_load(open("config/wb/sweep/stage1/combo_baseline/variant_classical.yaml"))
cfg = {**base, **overlay}
cfg["output_dir"] = suite["output_dir"]   # run_aimip adds this from the suite
cfg["aimip_variant"] = "classical"        # run_aimip sets this from --variants
spec = importlib.util.spec_from_file_location("run_aimip", "scripts/run/run_aimip.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
sc = ra._build_spectral_config(cfg)
print("CONFIG OK: n_max=%s n_levels=%s dt=%s" % (sc.n_max, sc.n_levels, sc.dt))
lc = getattr(sc, "loss_config", getattr(sc, "loss", None))
print("loss multi_step_hours=%s" % (getattr(lc, "multi_step_hours", "?"),))
PY
echo "config-build check passed"
