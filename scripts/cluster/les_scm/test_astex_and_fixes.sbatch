#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=astexfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/astexfix_%j.out
set +e
: "${AX_WT:?set AX_WT to the pinned worktree}"
WT=$AX_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== all new/changed instruments ==============="
$PY -m pytest -q \
    tests/unit/test_stratocumulus_les_case_selector.py \
    tests/atmosphere/test_prescribed_surface_flux_to_closure.py \
    tests/atmosphere/test_simple_lw.py \
    tests/unit/test_scm_les_joint_aggregation.py \
    tests/unit/test_scm_les_score_decomposition.py \
    tests/unit/test_plot_scm_les_turbulence_case_select.py 2>&1 | tail -30
echo "instruments rc=${PIPESTATUS[0]}"

echo "=============== ASTEX deck loads and builds a column ==============="
# The deck has never been read by this driver. Prove it parses and produces a
# finite initial column BEFORE spending a GPU on an LES.
$PY - <<'PYEOF' 2>&1 | tail -25
import importlib.util, sys
from pathlib import Path
import numpy as np
p = Path("scripts/run/run_dycoms_les.py")
sp = importlib.util.spec_from_file_location("drv", p)
d = importlib.util.module_from_spec(sp); sys.modules["drv"] = d; sp.loader.exec_module(d)
for case in ("dycoms", "astex"):
    sys.argv = ["x", "--case", case, "--nx", "8", "--ny", "8", "--nz", "24"]
    a = d.parse_args()
    print(f"{case}: dir={Path(a.case_dir).name} f_cor={a.f_cor:.3e} "
          f"D_sub={a.subsidence_divergence_s:.3e} Lz={a.Lz} Nc={a.n_c_m3:.3e}")
    import jax.numpy as jnp
    g, st, ref, forc = d.build(a, jnp.float64)
    tr = np.asarray(st.tracers)
    print(f"   grid nz={g.z_c.shape[0]} z_top={float(g.z_f[-1]):.1f} m")
    print(f"   q_v  range {tr[...,0].min():.4e} {tr[...,0].max():.4e} kg/kg")
    print(f"   q_c  range {tr[...,1].min():.4e} {tr[...,1].max():.4e} kg/kg")
    print(f"   theta range {float(np.asarray(st.theta).min()):.2f} "
          f"{float(np.asarray(st.theta).max()):.2f} K")
    assert np.all(np.isfinite(tr)), f"{case}: non-finite tracers in the IC"
    assert np.all(np.isfinite(np.asarray(st.theta))), f"{case}: non-finite theta"
    assert tr[...,1].max() > 0.0, f"{case}: no cloud in the initial column"
print("BOTH CASES BUILD A FINITE CLOUDY COLUMN")
PYEOF
echo "astex build rc=$?"
echo "DONE $(date +%T)"
