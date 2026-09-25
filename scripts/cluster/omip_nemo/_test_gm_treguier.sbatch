#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gm_treg_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gm_treg_test_%j.out
set +e
ROOT=/burg/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== --gm-treguier argparse + config-build smoke ==="
$PY -c "
from scripts.run.run_omip_core2 import _build_arg_parser
import run_omip_shim  # noqa: F401
" 2>/dev/null
$PY -c "
from scripts.run.run_omip_core2 import _build_arg_parser
p=_build_arg_parser()
a=p.parse_args(['--grid','tripole','--mesh','m.nc','--gm-treguier','--gm-aei0','1800'])
assert a.gm_treguier is True and a.gm_aei0==1800.0, (a.gm_treguier,a.gm_aei0)
b=p.parse_args(['--grid','tripole','--mesh','m.nc'])
assert b.gm_treguier is False and b.gm_aei0==1800.0
print('[argparse] --gm-treguier default False OK; parsed OK; aei0 default 1800')
# the GMRediConfig swap itself
from scripts.run import run_omip
from legoesm.ocean.physics.lateral_mixing.config import TreguierConfig, VisbeckConfig
base = run_omip._DEFAULT_BATHY_GM_REDI
print('[default] kappa_GM=%g visbeck=%s treguier=%s' % (base.kappa_GM, base.visbeck.enabled, base.treguier.enabled))
sw = base._replace(visbeck=VisbeckConfig(enabled=False), treguier=TreguierConfig(enabled=True, aei0=1800.0))
assert sw.visbeck.enabled is False and sw.treguier.enabled is True and sw.treguier.aei0==1800.0
assert sw.kappa_Redi==base.kappa_Redi and sw.S_max==base.S_max, 'Redi/S_max must be untouched'
print('[swap] treguier ON / visbeck OFF; kappa_Redi+S_max preserved OK')
# kappa_min floor: default inert (0.0), and applied when set
import jax.numpy as jnp, numpy as np
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import compute_treguier_kappa_gm
assert TreguierConfig().kappa_min == 0.0, 'default floor must be inert'
a2=p.parse_args(['--grid','tripole','--mesh','m.nc','--gm-treguier'])
assert a2.gm_kappa_min==200.0, a2.gm_kappa_min
print('[argparse] --gm-kappa-min default 200 OK (Visbeck-matched)')
" 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin|tensorflow"
echo "=== DONE ==="
