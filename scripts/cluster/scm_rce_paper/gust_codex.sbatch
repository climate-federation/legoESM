#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gust_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gust_cdx_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91
cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a new physics term and its plumbing. Break it. Read the
code; do not take this description on trust.

WHAT WAS ADDED (commits fb001b9f, 44e16307, 9088d446 on this branch):
Cold-pool gustiness for bulk surface fluxes. In tropical RCE a CRM with NO mean
wind evaporates ~2.7 mm/day; our single column reaches 0.70 once an artificial
5 m/s wind pin is removed, and COARE 3.0's internal w* gustiness recovers only
0.87 (measured) because w* scales with the SURFACE buoyancy flux, which
collapses in the warm moist near-neutral state the term exists to break.

  packages/core/legoesm/core/bulk_flux.py  -- convective_gust_wind():
      u_gust = coeff * (P / 1mm-day)^(1/3), double-where for the AD hazard,
      optional cap, coeff=0 returns exact zeros via a static Python branch.
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/config.py --
      SurfaceLayerConfig.convective_gustiness_coeff / _cap + __param_spec__.
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py --
      applied in compute_surface_fluxes BEFORE the bulk-scheme branch by
      SCALING (u, v) so stress direction is preserved and |U| carries the gust
      in quadrature. NOT applied on the tiled path (_single_tile_flux).
  packages/atmosphere/legoesm/atmosphere/physics/physics_state.py --
      new sfc_precip carry slot (ncol,).
  packages/atmosphere/legoesm/atmosphere/physics/combined.py --
      _store_sfc_precip writes this step's precip_accum into phys_updates.
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py --
      _read_sfc_precip reads it and passes sfc_precip= to louis only.
  tests/core/unit/test_convective_gust_wind.py -- 12 tests.

ATTACK, each with file:line:
A. CORRECTNESS OF THE APPLICATION. Scaling (u,v) by
   sqrt(|U|^2+u_gust^2)/|U| inflates the MOMENTUM stress as well as the
   scalar fluxes -- tau ~ Cd*|U|*u becomes ~Cd*|U|_eff*u_eff, i.e. it grows
   like |U|_eff^2 not |U|_eff. Is that right? Standard practice (Beljaars 1995,
   COARE) applies gustiness to the SCALAR exchange and to the stress
   MAGNITUDE, but a spurious momentum SOURCE would be a real defect: with a
   uniform-wind column the extra drag changes the wind, which changes the
   fluxes. Compare with how compute_most_fluxes uses its own gustiness
   (gustiness_w_zi / wg2) around bulk_flux.py:990-1015 -- does COARE add it to
   the wind used for stress, or only for scalars? If they differ, ours is
   inconsistent with the scheme it sits next to.
B. THE ONE-STEP LAG AND THE CARRY. Does _store_sfc_precip run on EVERY path
   that has precipitation (grep every physics_fn variant in combined.py -- I
   only edited one; there are ~4 update_physics_state call sites). If some
   lanes never write the slot, the term silently reads zeros forever and the
   feature is dead on those models. Also: is precip_accum a RATE [kg/m^2/s] or
   an ACCUMULATION over the step at every producer? A units error here scales
   the gust by dt.
C. SHAPE AND LANE SAFETY. sfc_precip is flattened to (-1,); for MPAS it is
   (nCells,), for hydrostatic/SCM the native horizontal is (face,x,y). Does
   _read_sfc_precip's reshape+slice do the right thing on every lane, and does
   the lax.scan carry type stay stable across steps? rad_heating had exactly
   this bug historically (see its comment).
D. AD AND JIT. Is the double-where genuinely NaN-safe under reverse mode at
   precip=0, and does the static coeff==0 branch prevent any retrace? Is
   PhysicsState still a valid pytree with the new field (any code that builds
   it positionally, unpacks it, or asserts a field count)?
E. TEST HONESTY. Do the tests in tests/core/unit/test_convective_gust_wind.py
   actually FAIL if the feature is broken -- specifically, would
   test_surface_fluxes_unchanged_when_the_term_is_off still pass if the gust
   were applied unconditionally? Name any assertion that cannot fail.
F. WHAT I DID NOT WIRE. Name every consumer that should get this term but
   does not (tiled/mosaic path, coupler, ocean, other turbulence schemes,
   the SCM campaign's own config builder) and say which omissions are
   defensible scope and which are latent inconsistencies.
G. PHYSICS SANITY. Is a (P)^(1/3) precipitation scaling defensible for
   cold-pool gust wind, and is a coefficient of order 1-3 m/s at 1 mm/day in
   the right range? If you know Redelsperger et al. (2000) or the IFS
   formulation, say how ours differs.

Numbered findings, each BLOCKER / SHOULD-FIX / NIT, with the one-line fix. Say
explicitly where you checked and found nothing wrong.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
