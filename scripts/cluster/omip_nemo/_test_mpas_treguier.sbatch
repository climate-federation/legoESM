#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=mpastreg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/mpastreg_%j.out
#
# Gate for the Treguier adaptive GM coefficient on the MPAS Voronoi lane.
# Must be able to FAIL: every section feeds fail() and the job exits non-zero.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RC=0
fail() { echo "FAIL[$1]: $2"; RC=1; }

run_pytest() {   # run_pytest <label> <cmd...>
  local label="$1"; shift
  local log=/tmp/mpastreg.$$.$label
  JAX_PLATFORMS=cpu $@ 2>&1 | tee "$log" | tail -30
  local st=${PIPESTATUS[0]}
  grep -qE "[0-9]+ (failed|error)" "$log" && fail "$label" "pytest reported failures/errors"
  grep -qE "[0-9]+ passed" "$log" || fail "$label" "nothing PASSED (all skipped? collection error?)"
  [ "$st" -eq 0 ] || fail "$label" "pytest exited $st"
  rm -f "$log"
}

echo "=== 1. the new Treguier-on-MPAS units (x64) ==="
run_pytest s1 env JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_gm_redi_mpas_treguier.py

echo "=== 2. the MPAS GM/Redi suites this edit could break ==="
run_pytest s2 env JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_gm_redi_mpas.py \
  tests/ocean/unit/test_gm_redi_mpas_seafloor_slope.py \
  tests/ocean/unit/test_gm_redi_adjoint_stabilization.py

echo "=== 3. the structured lane must be untouched (shared helper reused, not edited) ==="
run_pytest s3 env JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_gm_redi_latlon_cgrid.py \
  tests/ocean/unit/test_gm_redi_eady_physics.py \
  tests/ocean/unit/test_omip_gm_redi_coefficients.py

echo "=== 4. DIAGNOSTIC: how exact IS the Perot reconstruction on this mesh? ==="
# reconstruct_cell_velocity's docstring claims it is "exact for uniform flow on
# any Voronoi mesh".  That is a testable claim and the first run contradicted
# it on 58 of 162 cells, some by 100%.  Characterise the error before writing
# a tolerance, rather than tuning the tolerance until it passes.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -c "
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
for lev in (2, 3):
    m = create_voronoi_mesh(subdivision_level=lev)
    ang = np.asarray(m.angleEdge)
    sx, sy = 3.0e-3, -1.5e-3
    S_n = jnp.asarray(sx*np.cos(ang) + sy*np.sin(ang))
    rx, ry = (np.asarray(a) for a in reconstruct_cell_velocity(S_n, m))
    err = np.hypot(rx-sx, ry-sy) / np.hypot(sx, sy)
    lat = np.degrees(np.asarray(m.latCell))
    deg = (np.asarray(m.edgesOnCell) >= 0).sum(axis=0)
    print(f'level {lev}: nCells={m.nCells} median={np.median(err):.3e} '
          f'p90={np.percentile(err,90):.3e} max={err.max():.3e}')
    bad = err > 0.1
    print(f'  cells with >10% error: {bad.sum()} of {len(err)}')
    if bad.any():
        print(f'  their |lat| range: {np.abs(lat[bad]).min():.1f}..{np.abs(lat[bad]).max():.1f}')
        print(f'  their edge counts: {sorted(set(deg[bad].tolist()))}')
        print(f'  all-cell edge counts: {sorted(set(deg.tolist()))}')
    for lo, hi in ((0,30),(30,60),(60,90)):
        s = (np.abs(lat)>=lo)&(np.abs(lat)<hi)
        if s.any():
            print(f'  |lat| {lo:2d}-{hi:2d}: median {np.median(err[s]):.3e} max {err[s].max():.3e} n={s.sum()}')
" 2>&1 | tail -30
[ ${PIPESTATUS[0]} -eq 0 ] || fail s4 "the Perot diagnostic itself failed to run"

echo "DONE rc=$RC"
exit $RC
