#!/bin/bash
#SBATCH --job-name=cdxharm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxharm_%j.out
# ONE REVIEW of the two harmonized cards, before they are submitted.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $R "ONE REVIEW, no follow-up questions.

Two new sbatch cards claim to give the MPAS and FESOM2 ocean lanes the SAME
physics as the tripole production card:
  scripts/cluster/omip_nemo/_mpas_orca1faithtint_d30.sbatch
  scripts/cluster/omip_nemo/_fesom_orca1faithtint_d30.sbatch
Reference card: scripts/cluster/omip_nemo/orca1_faithful_card.sbatch
Driver: scripts/run/run_omip_core2.py

Check ONLY these, answer in under 300 words:
1. SILENT DROP: is any flag in the two new cards accepted by argparse but never
   threaded to that grid's builder or host-loop consumer? Name the flag and the
   call site.
2. MISSED FLAG: is any flag in the tripole card actually wired for mpas (or in
   _FESOM_WIRED_DESTS for fesom) that the new card fails to carry?
3. REFUSAL: would either card exit at validation (a flag refused for that grid,
   a missing companion flag, a mutually-exclusive pair)?
4. SHELL: is the exit status of the python run actually checked, and would the
   card fail loudly if the run died?
Reply with a numbered list of concrete defects, or 'no defects' per item." < /dev/null 2>&1 | tail -60
echo "=== codex rc=$? ==="
echo DONE
