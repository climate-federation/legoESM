#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=seedlogic
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/seedlogic_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM; cd "$WT" || exit 1
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("drv", "scripts/run/run_scm_les_turbulence_tuning.py")
drv = importlib.util.module_from_spec(spec); sys.modules["drv"] = drv
spec.loader.exec_module(drv)
import jax.numpy as jnp
base = drv._initial_params("louis", "extended")
s0   = drv._initial_params("louis", "extended", seed=0, jitter=0.25)
s0b  = drv._initial_params("louis", "extended", seed=0, jitter=0.25)
s1   = drv._initial_params("louis", "extended", seed=1, jitter=0.25)
none = drv._initial_params("louis", "extended", seed=None, jitter=0.25)
def raw(p): return {k: float(jnp.ravel(v)[0]) for k,v in p.raw_values.items()}
b,a0,a0b,a1,an = raw(base),raw(s0),raw(s0b),raw(s1),raw(none)
k0 = sorted(b)[0]
ok = True
ok &= abs(an[k0]-b[k0])<1e-12          # seed=None -> deterministic
ok &= abs(a0[k0]-a0b[k0])<1e-12        # reproducible
ok &= abs(a0[k0]-a1[k0])>1e-6          # seeds differ
print(f"seed=None==base:{abs(an[k0]-b[k0])<1e-12}  reproducible:{abs(a0[k0]-a0b[k0])<1e-12}  seeds-differ:{abs(a0[k0]-a1[k0])>1e-6}")
c1 = drv.build_physics_config("smagorinsky", prescribed_fluxes=False, smag_stability_form="deardorff")
c2 = drv.build_physics_config("smagorinsky", prescribed_fluxes=False)
print("deardorff form:", c1.turbulence.smagorinsky.stability_form, "| default:", c2.turbulence.smagorinsky.stability_form)
ok &= c1.turbulence.smagorinsky.stability_form=="deardorff"
ok &= c2.turbulence.smagorinsky.stability_form=="lilly"
print("ALL OK" if ok else "FAILED")
PYEOF
echo "rc=$?"
echo "DONE $(date +%T)"
