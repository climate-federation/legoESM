#!/bin/bash
#SBATCH --account=glab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg/glab/users/pg2328/enso_logs/swap/swap_%j.log
set -e
export WT=/burg-archive/glab/users/pg2328/wt_fix_scm_lat CLUBB_ROOT=/burg/glab/users/pg2328/clubb_release_oracle
LEGO_PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg/glab/users/pg2328/enso_logs/swap
SET=$D/settings; mkdir -p $SET
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu $LEGO_PY $WT/scripts/validate/oracle/clubb_core/dump_cam_settings.py $SET
cd $CLUBB_ROOT/run_scripts
[ -f $D/F.npz ] || $CLUBB_ROOT/.venv-python/bin/python $D/swap_core.py fortran $SET $D/F.npz
cd $D
PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $LEGO_PY $D/swap_core.py jax $D/F.npz $D/J.npz
echo DONE
