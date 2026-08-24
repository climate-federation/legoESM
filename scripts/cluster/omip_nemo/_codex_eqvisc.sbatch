#!/bin/bash
#SBATCH --job-name=cdx_eqvisc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_eqvisc_%j.out

# Adversarial review of commit c35c1ebc2 (equatorial A_h reduction) BEFORE the
# two 12h arms finish, so any defect is known before their numbers are read.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarially review commit c35c1ebc2 (git show c35c1ebc2), which changes
lateral-viscosity equatorial shaping in a JAX ocean model. Scope: only this
commit's diff plus enough surrounding code to judge it.

CONTEXT. NEMO ORCA1 reads momentum viscosity from a 3-D file: 20000 m2/s
midlatitude reduced to ~1000 within ~2 deg of the equator. Our card ran
A_h=100000 everywhere plus Laplacian Smagorinsky C=3. The commit makes
A_h_eq_boost<1 act as a REDUCTION (previously values<=1 were a SILENT no-op
in equatorial_boost_factor) and wires --A-h-eq-sigma-deg through both
builders of scripts/run/run_omip_core2.py.

ATTACK THESE SPECIFICALLY:
1. The eight '> 1.0' -> '!= 1.0' gate flips in
   packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py. Do ALL
   branches (A_h+B_h combined, A_h-only, lat_scaling on/off, kdiss capture,
   flux_divergence / nemo_div_curl guards) now treat a reduction
   consistently? Is there any remaining path where boost<1 is silently
   dropped (the old helper no-op is removed, but a caller may still guard)?
2. equatorial_boost_factor in latlon_cgrid_operators.py: with boost=0.05 and
   sigma=7 the factor must be ~0.05 at lat=0, ~1 beyond ~20 deg, strictly
   positive everywhere. Any dtype/tracer issue? It raises on boost<=0 -- is
   that raise reachable under jit with a static Python float (it should be;
   confirm no traced value reaches it)?
3. run_omip_core2.py wiring: the latlon builder's _ovr previously DROPPED
   A_h_eq_boost silently; the commit adds both fields there. Verify no OTHER
   silently-dropped kwargs were introduced, and that replace_flat routes
   A_h_eq_sigma_deg into LateralViscosityConfig (state.py).
4. The K_diss_h / _ah_scale_center capture paths: with a reduction active,
   is the dissipation diagnostic still consistent with the tendency actually
   applied (same A_h x scale field)?
5. Stability logic the commit may have broken: smag_cfl_safety caps SMAG
   only; with A_h reduced at the equator is any cap/floor silently assuming
   boost>=1?
6. The paired arm script scripts/cluster/omip_nemo/_ab_eqvisc_d30.sbatch:
   are the two arms actually one-variable against the stated control command
   (results/omip_nemo/nemolev_trp_signedn2_d30/run_manifest.json)? Name any
   flag that differs beyond the declared viscosity block.

For each finding: file:line, severity (BLOCKER/MAJOR/MINOR), the failure
scenario, and the minimal fix. End with a one-line verdict: SHIP or FIX-FIRST."
rc=$?
echo "=== CODEX RC=$rc ==="
