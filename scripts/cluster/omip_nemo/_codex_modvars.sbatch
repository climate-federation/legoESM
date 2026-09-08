#!/bin/bash
#SBATCH --job-name=cdx_mod
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_mod_%j.out

# Review the claim that our CORE-II forcing uses the WRONG netCDF variables
# BEFORE rebuilding nyf.zarr. Every OMIP run to date used this file, so a
# wrong rebuild is expensive and a wrong diagnosis is worse.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json   "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Review a claim about CORE-II forcing in a JAX ocean model that is being
matched against NEMO ORCA1. Do NOT restate what the code does; find defects in
the CLAIM and in the fix it implies.

THE CLAIM. scripts/data/build_core2_nyf_zarr.py builds our forcing cache from
the BASE variables of the CORE-II files (U_10, V_10, T_10, Q_10, SWDN, LWDN,
RAIN+SNOW). NEMO ORCA1's namelist_cfg at
/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/namelist_cfg
names the _MOD variants instead: sn_wndi=U_10_MOD, sn_wndj=V_10_MOD,
sn_tair=T_10_MOD, sn_humi=Q_10_MOD, sn_qsr=SWDN_MOD, sn_qlw=LWDN_MOD,
sn_prec=PRC_MOD, sn_snow=SNOW.

MEASURED over the nino3 box (5S-5N, 210-270E), CORE-II records 100-119
(simulated days 25-30), cos-lat weighted, from the SAME file
INPUTS/u_10.15JUNE2009.nc:
  U_10      -> 4.736 m/s   (what we use)
  U_10_MOD  -> 6.279 m/s   (what NEMO reads)   i.e. +33%
  NEMO's own windsp 5-day-mean diagnostic -> 6.064 m/s
  our wind-stress magnitude / NEMO taum   -> 0.67
U_10_MOD's long_name is
'U_TEMP*COS(WINDDF1*DEGTORAD)-V_TEMP*SIN(WINDDF2*DEGTORAD)'.

We conclude our equatorial trades are ~33% too weak, that this is why our
Equatorial Undercurrent is 29% weak and nino3 is +3.03 C too warm, and that
rebuilding the cache from the _MOD variants is the fix.

ATTACK THESE SPECIFICALLY:
1. Is _MOD actually the right choice, or is it a PRE-PROCESSED field that
   NEMO expects to combine with something else (rotation onto the model grid,
   a weights file, an on-the-fly correction)? The long_name is a rotation
   formula with TWO different angles WINDDF1 and WINDDF2. Read the file's
   variables and attributes and say what U_10_MOD actually is. If it is a
   rotation to a different reference frame, feeding it to a bulk formula that
   assumes earth-relative winds could be WRONG even though the magnitude looks
   closer to NEMO's diagnostic.
2. Magnitude vs direction: a pure rotation preserves |U|. Ours differs by 33%
   in magnitude, so it is NOT a pure rotation. What else differs? Check
   whether U_10 and U_10_MOD have different units, missing-value handling,
   land masking, or time conventions that would corrupt a box mean.
3. Is the box comparison sound? Both were read from the same file on the same
   grid with the same weights, but confirm the record indices 100-119 really
   correspond to days 25-30 given the file's own TIME axis, and that a
   cos(lat) weight is adequate on this regular T62 grid (94x192).
4. Does NEMO apply anything else to the wind before the bulk -- relative
   winds (wind minus surface current), gustiness, a minimum wind speed -- that
   would make windsp 6.064 not directly comparable to a raw file mean of
   6.279? Which direction does each push?
5. If we DO rebuild from _MOD, what else must change together? Check whether
   the other channels' _MOD variants have different units or time axes
   (SWDN_MOD/LWDN_MOD are daily, PRC_MOD is monthly per the namelist's -1
   frequency) and whether build_core2_nyf_zarr.py's repeat/interpolation logic
   would still be correct. Name every place the swap could silently break.
6. Is there any evidence in the repo that the base variables were chosen
   DELIBERATELY for a reason we are about to discard? The builder comment says
   'Base (non-_MOD) variables are the raw CORE data.'

For each finding give severity (CRITICAL/MAJOR/MINOR), file:line, why it is
wrong, and the smallest correct fix. Refuting the claim is the most valuable
output. State plainly at the end whether the rebuild should proceed." 2>&1
rc=$?
echo "=== codex exit rc=$rc ==="
