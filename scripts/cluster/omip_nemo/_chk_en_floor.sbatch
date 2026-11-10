#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=enfloor
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/enfloor_%j.out
#
# IS THE SEEDED TKE REAL, OR A MASK VALUE?
#
# The length/energy decomposition (job 9877115) says the Antarctic shelf
# excess is ENERGY, not length: l_eps is 0.635 m there, the smallest of any
# band, while our one-hour step multiplies the TKE by about 693 from a seed of
# exactly 1.000e-10.
#
# That seed is the problem with the finding. NEMO floors en at rn_emin = 1e-6
# (namelist_ref:1228), so a wet cell should never carry 1e-10 — which is
# avtb, a DIFFUSIVITY background, not a TKE floor. If those cells are masked
# in NEMO's own accounting then the 693x growth is our closure climbing away
# from an artificial zero, and the 20x diffusivity ratio is measured against
# values NEMO never intended as physical.
#
# So: what fraction of NEMO's own en sits at or below each threshold, and what
# discrete values does it take below rn_emin?
set -uo pipefail
cd /burg-archive/glab/users/pg2328/_omip_merge || exit 1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python scripts/tmp/_chk_en_floor.py
