#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --job-name=modonfbg
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_fb_gpu_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
# faithful cross-face divg ghost = the certified k2e ring map (mode 2);
# recorded here so the run's config is reproducible (codex r9 P0)
export LEGOESM_SW_FB_CROSS_FACE_HALO=${LEGOESM_SW_FB_CROSS_FACE_HALO:-2}
echo "CROSS_FACE_HALO=$LEGOESM_SW_FB_CROSS_FACE_HALO"
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/fv3_recon/modon_fb_gpu
rm -rf "$OUT" && mkdir -p "$OUT"
cd $REPO
$P -c "import jax; print('JAX devices:', jax.devices())"
# FB core (JAX forward-backward) modon run on GPU, C48, ext-native ED grid
$P scripts/matrix/run_atmosphere_test_matrix.py \
  --test =colliding_modons --grid cubed_sphere --resolution C48 \
  --sw-core fb --fv3-native-grid -o "$OUT" 2>&1 | tail -30
echo "MODON_FB_GPU_DONE"
