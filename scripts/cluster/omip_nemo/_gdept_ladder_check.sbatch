#!/bin/bash
#SBATCH --job-name=gdeptchk
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gdeptchk_%j.out
#
# Quantify the ladder defect the Claude adversarial review named: with
# t_depth_ref=None on the OMIP z-coordinate, nemo_bn2_depth_ladders falls back
# to |z_full_ref| -- the MIDPOINTS of the thickness ladder -- where NEMO uses
# its own analytic gdept_1d. bn2 divides by e3w = diff(t_depth), so the error
# lands directly on N2.
#
# This measures the size before deciding whether it is worth fixing.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u - <<'PY'
import numpy as np, xarray as xr
# The path run_omip_core2._NEMO_DOMAIN_CFG actually reads. The RUN_TRD2
# copy is a symlink to RUN_REF and is currently broken.
p=('/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/'
   'orca1_inputs/data_repository/input_fields/domain_cfg.nc')
ds=xr.open_dataset(p, decode_times=False)
print('1-D vars present:', [v for v in ds.variables if v.endswith('_1d')])
# This domain_cfg carries e3t_1d and e3w_1d but NO gdept_1d. That is fine:
# e3w_1d IS the divisor bn2 uses (eosbn2 pn2 = grav*(...)/e3w), so it can be
# compared directly against what our fallback ladder produces, with no need
# to reconstruct gdept at all.
e=np.asarray(ds['e3t_1d'].values).ravel()
ew=np.asarray(ds['e3w_1d'].values).ravel()
print(f'  {e.size} levels, e3t {e[0]:.3f} -> {e[-1]:.1f} m')

# What the code does today: t_depth_ref is None on the OMIP z-coordinate, so
# nemo_bn2_depth_ladders falls back to |z_full_ref|, the MIDPOINTS of the
# thickness ladder. bn2 then divides by diff(t_depth).
mid = np.cumsum(e) - 0.5 * e
ew_ours = np.diff(mid)              # = 0.5*(e3t[k] + e3t[k+1])
ew_nemo = ew[1:]                    # NEMO's own e3w at the same interfaces

rel = np.abs(ew_ours - ew_nemo) / np.abs(ew_nemo)
print(f'\n[e3w] the divisor N2 actually uses -- ours vs NEMO:')
print(f'  median rel {np.median(rel):.3e}   mean {rel.mean():.3e}   '
      f'max {rel.max():.3e} at interface {rel.argmax()}')
k = int(rel.argmax())
print(f'  worst interface {k}: ours {ew_ours[k]:.4f} m vs NEMO {ew_nemo[k]:.4f} m')
print(f'  N2 is biased by that same fraction with the OPPOSITE sign '
      f'(N2 ~ 1/e3w)')
top = rel[:30]
print(f'  upper 30 interfaces (thermocline): median {np.median(top):.3e}  '
      f'max {top.max():.3e}')

# Put it next to the effect the whole card change is chasing: the signed-N2
# switch moved the one-step K ratio by 0.16-0.26 (16-26%).
print(f'\n[scale] compare to the signed-N2 lever, which moved the one-step '
      f'K ratio by 0.16-0.26.')
print(f'  a {np.median(rel):.2e} median e3w error is '
      f'{0.16/max(np.median(rel),1e-30):.0f}x SMALLER than that lever')
PY
echo "=== rc=$?"
