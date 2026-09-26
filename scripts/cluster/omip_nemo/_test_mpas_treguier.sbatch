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
# test_nemo_match_recipe pins the recipe-level refusal this commit re-words,
# and test_dispatch_hardening pins the grow-only set of scheme guards — the two
# suites a change that removes a NotImplementedError is most likely to break.
run_pytest s2 env JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_gm_redi_mpas.py \
  tests/ocean/unit/test_gm_redi_mpas_seafloor_slope.py \
  tests/ocean/unit/test_gm_redi_adjoint_stabilization.py \
  tests/ocean/unit/test_nemo_match_recipe.py \
  tests/test_dispatch_hardening.py \
  tests/unit/test_run_omip_core2_gm_treguier_mpas.py \
  tests/unit/test_run_omip_core2_gm_treguier.py

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
    polar = np.abs(lat) > 85.0
    if polar.any():
        print(f'  |lat|>85 ({polar.sum()} cells): median {np.median(err[polar]):.3e} '
              f'max {err[polar].max():.3e}')
    away = np.abs(lat) < 60.0
    print(f'  |lat|<60 ({away.sum()} cells): median {np.median(err[away]):.3e}')
    print(f'  ALL cells: median {np.median(err):.3e}')
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

echo "=== 5. DIAGNOSTIC: why did the wet-restricted reconstruction return zero? ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -c "
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from legoesm.grids.voronoi import (create_voronoi_mesh,
                                   reconstruct_cell_velocity,
                                   reconstruct_cell_velocity_wet)
m = create_voronoi_mesh(subdivision_level=2)
print('dtypes: areaCell', m.areaCell.dtype, 'dvEdge', m.dvEdge.dtype,
      'angleEdge', m.angleEdge.dtype)
eoc = np.asarray(m.edgesOnCell); valid = eoc >= 0
dv = np.asarray(m.dvEdge)[np.maximum(eoc,0)] * valid
dc = np.asarray(m.dcEdge)[np.maximum(eoc,0)] * valid
A  = np.asarray(m.areaCell)[None, :]
w  = dv*dc/(2.0*A)
print('per-edge weight: median', np.median(w[valid]), 'sum per cell: median',
      np.median(w.sum(axis=0)))
ang = np.asarray(m.angleEdge)[np.maximum(eoc,0)]
c, s_ = np.cos(ang), np.sin(ang)
Mxx=(w*c*c).sum(0); Mxy=(w*c*s_).sum(0); Myy=(w*s_*s_).sum(0)
det = Mxx*Myy - Mxy*Mxy
print('M trace: median', np.median(Mxx+Myy), ' det: median', np.median(det),
      'min', det.min(), 'max', det.max())
print('cells failing det>1e-6:', int((det<=1e-6).sum()), 'of', m.nCells)
sx, sy = 3.0e-3, -1.5e-3
S_n = jnp.asarray(sx*np.cos(np.asarray(m.angleEdge)) + sy*np.sin(np.asarray(m.angleEdge)))
ax, ay = (np.asarray(v) for v in reconstruct_cell_velocity(S_n, m))
bx, by = (np.asarray(v) for v in reconstruct_cell_velocity_wet(S_n, m, jnp.ones_like(S_n)))
print('plain  : median |v| =', np.median(np.hypot(ax,ay)), ' truth', np.hypot(sx,sy))
print('wet-fix: median |v| =', np.median(np.hypot(bx,by)))
# worst-case single-edge drop: field ALIGNED with the dropped normal
vc = int(np.argmax(valid.sum(axis=0))); de = int(eoc[0, vc])
a0 = float(np.asarray(m.angleEdge)[de])
S2 = jnp.asarray(3.0e-3*np.cos(np.asarray(m.angleEdge)-a0))
wet = np.ones(m.nEdges); wet[de] = 0.0; wet = jnp.asarray(wet)
nx, ny = (np.asarray(v) for v in reconstruct_cell_velocity(S2*wet, m))
fx, fy = (np.asarray(v) for v in reconstruct_cell_velocity_wet(S2, m, wet))
print('ALIGNED drop, cell', vc, ': truth 3.0e-3  naive', np.hypot(nx[vc],ny[vc]),
      ' wetfix', np.hypot(fx[vc],fy[vc]))
" 2>&1 | tail -20
[ ${PIPESTATUS[0]} -eq 0 ] || fail s5 "the reconstruction diagnostic failed to run"

echo "DONE rc=$RC"
exit $RC
