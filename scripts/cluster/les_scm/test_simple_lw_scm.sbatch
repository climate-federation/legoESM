#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=lwscm
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/lwscm_%j.out
set +e
: "${LW_WT:?set LW_WT}"
WT=$LW_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== simple_lw through the SCM radiation pipeline ==============="
$PY - <<'PYEOF' 2>&1 | tail -40
import numpy as np, jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import _call_radiation_backend
from legoesm.atmosphere.physics.radiation.simple_lw import SimpleLWConfig

ncol, nlev = 1, 60
p_half = jnp.linspace(2.0e4, 1.0e5, nlev + 1)[None, :]
p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
T = jnp.full((ncol, nlev), 288.0)
q_v = jnp.full((ncol, nlev), 6.0e-3)
# A stratocumulus deck: cloud in a band a few hundred metres thick.
q_c = jnp.where((p_full > 8.5e4) & (p_full < 9.2e4), 4.0e-4, 0.0)

cfg = RadiationConfig(scheme="simple_lw", simple_lw=SimpleLWConfig())
out = _call_radiation_backend(
    radiation_config=cfg, T=T, p_full=p_full, p_half=p_half,
    sfc_temperature=jnp.full((ncol,), 290.0), lat=jnp.zeros((ncol,)),
    q_v=q_v, insolation=jnp.zeros((ncol,)), q_cloud=q_c)
h = np.asarray(out.heating_rate)[0]
f = np.asarray(out.lw_flux_up)[0]
print("heating shape", h.shape, "flux shape", f.shape)
print("heating min %.4e max %.4e K/s" % (h.min(), h.max()))
print("net upward LW: sfc %.2f  TOA %.2f W/m^2" % (f[-1], f[0]))
assert np.all(np.isfinite(h)) and np.all(np.isfinite(f))
qc = np.asarray(q_c)[0]
kc = np.flatnonzero(qc > 0)
# TOA-first arrays: the SMALLEST index in the cloud band is its TOP.
top, base = kc.min(), kc.max()
print("cloud top idx %d heating %.4e ; base idx %d heating %.4e"
      % (top, h[top], base, h[base]))
assert h[top] < 0.0, "cloud top must cool"
assert h[base] > 0.0, "cloud base must warm"
# clear column: no cloud -> no heating gradient beyond the clear-sky term
out0 = _call_radiation_backend(
    radiation_config=RadiationConfig(scheme="simple_lw",
                                     simple_lw=SimpleLWConfig(divergence_s=0.0)),
    T=T, p_full=p_full, p_half=p_half,
    sfc_temperature=jnp.full((ncol,), 290.0), lat=jnp.zeros((ncol,)),
    q_v=q_v, insolation=jnp.zeros((ncol,)), q_cloud=jnp.zeros_like(q_c))
assert np.allclose(np.asarray(out0.heating_rate), 0.0, atol=1e-14)
print("clear column with D=0 gives exactly zero heating")
print("SCM RADIATION PIPELINE OK")
PYEOF
echo "pipeline rc=$?"

echo "=============== unknown scheme still raises ==============="
$PY -c "
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import _get_radiation_fn
for bad in ('simple-lw', 'simplelw', 'stevens'):
    try:
        _get_radiation_fn(RadiationConfig(scheme=bad)); raise SystemExit('NO RAISE for %r'%bad)
    except ValueError: pass
print('unknown radiation schemes raise')
"
echo "dispatch rc=$?"

echo "=============== tuner accepts the stratocumulus cases ==============="
$PY -c "
import importlib.util, sys
from pathlib import Path
sp = importlib.util.spec_from_file_location('drv', 'scripts/run/run_scm_les_turbulence_tuning.py')
d = importlib.util.module_from_spec(sp); sys.modules['drv']=d; sp.loader.exec_module(d)
print('ALL_CASES:', d.ALL_CASES)
assert 'dycoms' in d.ALL_CASES and 'astex' in d.ALL_CASES and 'wangara' in d.ALL_CASES
assert d._RADIATION_MISMATCH == {}, d._RADIATION_MISMATCH
cfg = d.build_physics_config('louis', prescribed_fluxes=True, simple_lw=True)
assert cfg.radiation.scheme == 'simple_lw', cfg.radiation.scheme
cfg0 = d.build_physics_config('louis', prescribed_fluxes=True)
assert cfg0.radiation.scheme == 'none', cfg0.radiation.scheme
print('tuner selects simple_lw for stratocumulus, none elsewhere')
"
echo "tuner rc=$?"
echo "DONE $(date +%T)"
