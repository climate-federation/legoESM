#!/bin/bash
#SBATCH --job-name=orca1_faith
#SBATCH --account=glab
# glab1 holds exactly ONE GPU node (g049) and it sits 'planned' behind
# higher-priority reservations, so a job queued there can wait indefinitely.
# 'short' has 36 GPU nodes at a 12 h cap, which this 30-day run fits.
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/orca1_faith_%j.out
#
# THE ORCA1-FAITHFUL PRODUCTION CARD.
#
# User decision, 2026-09-11: adopt ALL FOUR NEMO-faithfulness sets that landed
# as selectable capability during the 20-gap programme. Until now the
# production configuration selected NONE of them and every gap shipped with
# its default preserved.
#
# THIS IS A NEW BASELINE, NOT AN ARM. Four sets change together, which is the
# ONE case the house rules permit a multi-change configuration: a deliberate
# baseline, labelled as such. Its output is therefore NOT attributable term by
# term against any pre-adoption run, and no single lever below may be credited
# for a difference. Any single-lever claim from here needs a fresh
# one-variable arm off THIS card.
#
# Every scientific selector is named EXPLICITLY below, including several whose
# default is already what we want. A default is not a record.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1

DAYS=${DAYS:-30}
# --years is the runner's length knob; express the target in days.
YEARS=$($PY -c "print(${DAYS}/365.0)")
FORCING=${FORCING:-/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod}
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faith_d${DAYS}}

$PY -u scripts/run/run_omip_core2.py \
  `# --- geometry, resolution, timestep: stated even where defaulted ---` \
  --grid tripole \
  --mesh /burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc \
  --nlev 75 \
  --dt 150 \
  --years "$YEARS" \
  `# dt=150 s is CORRECT and was re-derived on 2026-09-11: tracers are` \
  `# explicit, so the VERTICAL tracer CFL binds first -- max|w| 2.25e-03 m/s` \
  `# against a 1.02 m top layer allows ~456 s, eight times below the 3488 s` \
  `# horizontal bound. 150 s sits at vertical Courant ~0.33.` \
  \
  `# --- initial condition and forcing ---` \
  --woa-init \
  --forcing-path "$FORCING" \
  \
  `# --- SET 1: the measured bug fixes -------------------------------------` \
  `# Thin-snow albedo (the surface went toward BLACK as snow thinned) and the` \
  `# salinity-dependent freezing point are corrections, not preferences.` \
  --prognostic-sea-ice \
  --ice-freeze-scheme unesco \
  `# NOT nemo_teos10: that liquidus consumes ABSOLUTE salinity and our tracers` \
  `# are PRACTICAL. Choosing it here would be the same unit error the --eos` \
  `# selector refuses by name. 'unesco' delivers the salinity DEPENDENCE that` \
  `# gap 6 was about, on the salinity scale we actually carry. THIS IS THE ONE` \
  `# JUDGEMENT CALL IN THIS CARD -- flagged rather than buried.` \
  \
  `# --- SET 2: ocean physics (GM/Redi + lateral operator) -----------------` \
  --gm-treguier \
  --gm-kappa-min 0 \
  `# kappa_min=0 is REQUIRED by nemo21 and is the oracle's own behaviour: NEMO` \
  `# tapers its GM coefficient to zero at the equator too (ldftra.F90:702,` \
  `# "tropical decrease"). A 200 floor would not be faithful.` \
  --gm-slope-scheme nemo_iso_lap \
  --gm-slope-positions nemo_native \
  --gm-bolus-advection through_fct \
  --gm-msc-stabilize \
  --redi-coefficient nemo21 \
  --redi-aht0 900 \
  `# nemo21 reads the mesh's exact gphif for ff_f (verified present in this` \
  `# mesh) and applies NEMO ldftra.F90:428-438: max(0.2*aht0, GM_face) plus an` \
  `# ADDITIVE tropical term, which is the only construction that reaches the` \
  `# equator once GM has tapered to zero there.` \
  \
  `# EOS: DELIBERATELY NOT nemo_teos10, so this card is NOT fully faithful on` \
  `# the dynamics density. ORCA1 runs ln_teos10=.true., but NEMO's TEOS-10` \
  `# branch consumes Conservative Temperature and ABSOLUTE salinity` \
  `# (eosbn2.F90:1924 sets l_useCT=.TRUE.) and converts nothing -- its inputs` \
  `# are pre-converted offline. Ours are potential temperature and practical` \
  `# salinity. Converting the IC, the SSS restoring target and the forcing is` \
  `# the remaining work for gap 10. Measured cost of the mismatch if it were` \
  `# selected anyway: 0.13 kg/m3 in density but only 0.47% in stratification.` \
  `# The runner REFUSES --eos nemo_teos10 for this reason, so it is omitted` \
  `# here rather than set: the dynamics runs 'wright'.` \
  \
  `# --- SET 3: ice (exchange coefficients, snow, optics) ------------------` \
  --ice-exchange nemo_si3 \
  `# ORCA1 namelist_cfg:140-142 and namelist_ice_ref:136 -- 1.0e-3 for all` \
  `# three air-ice coefficients and 5.0e-3 ice-ocean, against our 1.5/1.3/5.5.` \
  `# namelist_CFG overrides the 1.4e-3 in namelist_REF; quoting the reference` \
  `# file would have been wrong by 40%.` \
  --ice-snow bulk \
  --ice-snow-flooding on \
  \
  `# --- SET 4: forcing time interpolation ---------------------------------` \
  --forcing-time-interp linear \
  --chl-remap nemo_scrip \
  `# ORCA1 remaps chlorophyll with its own weights_reg05_bilinear.nc` \
  `# (namelist_cfg:170), not with inverse-distance weighting. Chlorophyll` \
  `# sets the shortwave penetration depth, so this feeds upper-ocean heating.` \
  `# ORCA1 carries ln_tint=.true. on exactly the three MONTHLY climatologies` \
  `# (sn_chl:170, sn_sss:185, sn_rnf:199) and ln_tint=.FALSE. on the 6-hourly` \
  `# winds and daily radiation (148-152), which both models therefore HOLD.` \
  `# Note: linear weights month k by (M[k-1]+2M[k]+M[k+1])/4 days, so February` \
  `# integrates as 29.5 rather than 28 -- the annual runoff total shifts by` \
  `# well under a percent. Stated, not assumed away.` \
  \
  --output "$OUT" \
  --snapshot-every-days 5 \
  2>&1 | tail -60

echo "=== card exit: $? ==="
echo "OUT=$OUT"
echo DONE
