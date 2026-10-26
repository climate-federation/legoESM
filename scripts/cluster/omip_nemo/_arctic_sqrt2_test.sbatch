#!/bin/bash
#SBATCH --job-name=arcsq2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcsq2_%j.out
#
# HOW MUCH OF THE 4.8x IS THE sqrt(2) DOUBLE-COUNT?  Free CPU test on NEMO's
# own state -- no model run, no GPU.
#
# MEASURED (job 9389262, Stage A, arctic/calm): our K_H 4.048e-2 vs NEMO avt
# 8.410e-3 = 4.8x.  Our config docstring (config.py:418-424) says
# kappa_convention="gaspar_sqrt2e" DOUBLE-COUNTS the sqrt(2) on the
# buoyancy-length path (tke_mxl_choice=3, which the runs use), so K_M is
# x1.414 too large.
#
# PRE-REGISTERED: with "veros_sqrte" the arctic/calm rms(le) should fall to
# about 4.048e-2 / 1.414 = 2.86e-2, i.e. the ratio 4.8 -> ~3.4.
#   MATCHES ~2.86e-2  => the double-count is real AND reaches K_H; the
#     remaining ~3.4x is a SEPARATE defect still to be found.
#   NO CHANGE       => the convention does not reach K_H on this path; the
#     docstring is describing an inactive branch and the whole 4.8x is
#     elsewhere.
#   FALLS FURTHER   => the convention couples into more than the amplitude
#     (e.g. through P_s or the Prandtl chain); investigate before claiming.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps --kappa-convention veros_sqrte \
  --output-dir $REPO/results/omip_nemo/arctic_sqrt2_test
echo "=== rc=$? ==="
