#!/bin/bash
#SBATCH --job-name=alphacdx
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/alphacdx_%j.out
#
# Card audit + tests + MANDATORY codex adversarial review for alpha_tke=1.0.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "FIELD-BY-FIELD CARD AUDIT (the systematic finding)" \
  $PY -u scripts/validate/ocean_fidelity/audit_orca1_tke_card.py --iwm \
     --out-json $WT/results/omip_nemo/card_audit/orca1_tke_iwm.json

run "card pins + CLI tests" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kappa_convention_cli.py \
      tests/unit/test_run_omip_core2_kpp_override.py \
      tests/unit/test_omip_config_parity.py

run "params reachability + dispatch ratchets" \
  $PY -m pytest -q tests/unit/test_params_reachability_audit.py \
      tests/test_dispatch_hardening.py tests/test_param_specs.py 2>&1 | tail -20

echo; echo "########## CODEX ADVERSARIAL REVIEW"
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="

codex exec --sandbox read-only -C $WT \
  "ATTACK THIS. I want it refuted if wrong, not confirmed.

CHANGE: \`orca1_zdftke_config\` (scripts/run/run_omip_core2.py) now pins
\`alpha_tke=1.0\`; TKEConfig's default is 30.0.  alpha_tke multiplies the TKE
vertical-diffusion coefficient in \`_solve_tke_backward_euler\`
(packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py, look for
\`K_tke_face = cfg.alpha_tke * 0.5 * (K_M_old[..., :-1] + K_M_old[..., 1:])\`).

MY CLAIM: NEMO diffuses \`en\` with the PLAIN viscosity.  In zdftke.F90 the
tridiagonal is zzd_up = zcof * MAX(p_avm(jk+1)+p_avm(jk), 2e-5) /
(e3t(jk)*e3w(jk)) with zcof = zfact1*tmask and zfact1 = -0.5*rn_Dt, so the
face coefficient is -rn_Dt*mean(avm) and alpha_tke is 1.

EVIDENCE (job 9405026, our TKE kernel fed NEMO's OWN restart state and \`en\`,
one 3600 s step, scored against NEMO's \`avm\` because ORCA1 sets nn_evdm=0 so
EVD never touches momentum; Antarctic band, 491743 interfaces on every row):
    zero step                     K_M/avm 0.861
    alpha_tke=30                  K_M/avm 1.846,  our TKE grows 3.20x
    alpha_tke=1.0                 K_M/avm 0.634,  our TKE grows 0.95x
    dissipation nemo_1p5_split    K_M/avm 1.673
    lc=False / etau=none / nn_mxl=2 / nemo_z0: 1.81 / 1.77 / 1.73 / 1.80

QUESTIONS. Quote sources.
 1. Read zdftke.F90's tke_tke tridiagonal IN FULL. Is the TKE diffusivity
    really avm x1, or is there a factor I have dropped -- in zfact1, in the
    e3t*e3w denominators (note it is e3t*e3w, NOT e3w^2 -- does that make the
    operator something other than d/dz(avm dE/dz)?), or in the 2e-5 floor?
    Does NEMO diffuse \`en\` with avm or with avm/sigma_e for some Schmidt
    number?
 2. Is legoESM's \`K_tke_face\` the SAME operator? Compare the discrete forms
    including the metric slots (dz_half vs dz_cell) and the Veros
    veros_dz_slots branch. If our operator differs structurally, alpha_tke=1
    may not be equivalent to NEMO even though the coefficient matches.
 3. THE OVERSHOOT. With alpha_tke=1 our avm goes 1.331e-2 -> 1.033e-2 over the
    hour (DOWN 22%) while NEMO's goes 1.5468e-2 -> 1.6281e-2 (UP 5%). So the
    pin does not land on NEMO. What is the most likely remaining term? Rank:
    the shear production (we run tke_shear_production='squared_centered', NEMO
    uses face-native zdf_sh2 with coastal doubling), the buoyancy sink, the
    surface Dirichlet's interaction with the solve, the missing avm_k
    (previous-step) viscosity in the tridiagonal, our N2 (n2_mode='insitu' vs
    NEMO rn2), or the dissipation split.
 4. IS THE 0.861 ZERO-STEP OFFSET REAL OR AN ARTIFACT? At zero steps our K_M
    is 0.76-0.95x NEMO's avm_k with the SAME en and the same c_k, so it is
    purely a mixing-length difference. Where does a uniform ~15% short length
    come from? Check the lup/ldown sweeps against zdftke.F90:658-690, the
    surface anchor, and whether our probe's global-max e3t reference profile
    (nemo_zero_step_closure.py builds dz_ref as nanmax over columns) biases it.
 5. BLAST RADIUS. alpha_tke is a tier-2 __param_spec__ tunable with bounds
    (1.0, 90.0), so 1.0 sits ON the lower bound. Does anything -- the
    trainable-parameter collector, a sigmoid transform, a test -- break or
    become degenerate at the bound? Does any committed baseline, DINO recipe
    or scorecard depend on 30.0? The MPAS branch reaches the same card.
 6. WHAT AM I OVER-READING? The scoring uses one hour from one restart in
    January. Name what that cannot support."
rc=$?
echo "=== CODEX RC=$rc ==="
echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
