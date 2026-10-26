#!/bin/bash
#SBATCH --job-name=fesom_lvl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesom_lvl_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RAW=/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2/mesh_ic/core2_mesh_ic/mesh_core2_raw
$PY scripts/validate/ocean_fidelity/fesom_level_rule.py --raw $RAW --nemo-domain-cfg /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc --out /burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2/mesh_nemo75_raw
echo DONE
