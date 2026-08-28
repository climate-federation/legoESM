#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_prec
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_prec_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
DESIGN/FEASIBILITY REVIEW (no code changes) -- how to make the FV3 "duo"
cubed-sphere dycore support fp32 + mixed precision like the other grids.

READ THE ACTUAL CODE, cite file:line:
- The fp64 hard-pin: packages/core/legoesm/core/fv3_phase3d_common.py
  `require_f64_jax` and EVERY call site of it (grep across packages/core/
  legoesm/core/fv3_*.py and the atmosphere duo model). List every phase
  it gates.
- The existing precision framework other grids use:
  packages/core/legoesm/core/precision.py -- `PrecisionPolicy`,
  `resolve_dtype(module, role)`, `_ATMOSPHERE_OVERRIDES` (spectral forces
  fp64 for spectral_transform/atm_pressure_gradient/semi_implicit/
  gravity_wave). How does a dycore OPT INTO a policy? Trace how the
  cd-grid / spectral kernels read resolve_dtype (grep resolve_dtype usage
  in packages/core/legoesm/core/ and driver/compiled_segments.py).
- How the driver threads --precision into a model (grep precision /
  PrecisionPolicy in packages/coupler/legoesm/driver/, and how
  create_atmosphere_dycore for OTHER dycores receives it).

ANSWER CONCRETELY:
1. Enumerate every require_f64_jax call site and, for each, name the
   physical operation it protects (pressure diag, pt->theta_v, energy
   fixer, remap, acoustic substep, tracer, halo). Which are pure data
   movement (fp32-trivially-safe) vs numerically sensitive?
2. What is the MINIMAL code change to give fv3_duo a coarse global-policy
   hook (all-fp32 or all-fp64 selected by --precision), matching the
   cd-grid's coarse behavior -- i.e. relax require_f64_jax to
   require_dtype(policy.storage) and thread the policy into
   FV3DuoConfig / the factory? Cite the exact structs.
3. What would a PER-OP mixed split look like (fp64 overrides only for the
   sensitive ops via a new `_ATMOSPHERE_OVERRIDES` entry like
   `fv3_duo_pgrad`/`fv3_duo_energy_fixer`, everything else fp32)? Which
   ops MUST be listed?
4. The set_eta/ak/bk constants, the ctx grid metrics (areas, cos/sin,
   edge lengths), and the DCMIP16 IC are built np.float64 -- do they need
   downcast for an fp32 run, and where? Any dtype-mismatch that would
   make jit fail or silently upcast the whole step back to fp64?
5. The consv_te energy fixer and the p_var/geopotential integrals -- are
   there specific lines where fp32 would lose catastrophic-cancellation
   bits? Cite them.
6. Differentiability + jit: does anything in the fp64 path assume x64
   (jax_enable_x64) globally in a way that an fp32 run would break, or
   any np.float64 literal that pins dtype? grep for jnp.float64 /
   dtype=np.float64 hardcodes in the step path.
7. The cleapest correct increment + the validation to certify it (fp32
   vs fp64 DCMIP16 trajectory divergence over N steps; mass/energy
   conservation).

Output as: (A) call-site inventory table, (B) minimal-coarse plan, (C)
per-op mixed plan, (D) risks/blockers, (E) recommended increment.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_PREC_DONE
