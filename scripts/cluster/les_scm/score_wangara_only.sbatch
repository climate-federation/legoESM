#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=wang_score
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/wang_score_%j.out
set +e
# JUST the Wangara score at defaults. The full verify sbatch runs the wide
# hydrostatic unit tier first (~1.5 h), which was worth paying once for two
# new PhysicsState fields and is not worth paying again for a one-case
# initial-condition change -- that tier's 16 failures are already established
# as pre-existing by the BASE-vs-HEAD A/B (job 9419722: byte-identical failure
# summaries on one node).
#
# PRE-REGISTERED, same criterion as the first attempt, which this repeats
# because the first attempt REFUTED the flux-BC fix as the cure:
#   the two BROKEN arms measured 8.6-9.9 and 11.9-15.5 on every closure;
#   every other case scores 0.06-3.4 at defaults.
#   CONFIRMS: wangara lands in 0.06-3.4 and the per-variable u and v terms
#     stop being identical across closures.
#   REFUTES: still >= 8, i.e. a third defect in this arm.
#   AMBIGUOUS: 3.4-8. Say so; do not spin.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
PIN=$(git rev-parse HEAD); echo "PINNED sha: $PIN  host: $(hostname)"
[ -n "$EXPECT_SHA" ] && [ "$PIN" != "$EXPECT_SHA" ] && { echo "pin mismatch"; exit 4; }
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=${OUT:-$R/scm_les_turbulence/_wangara_lean}
rm -rf "$OUT"
$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --cases "wangara:$R/les_ref/wangara" --outdir "$OUT" \
    --nlev 48 --skip-tuning --surface-flux-to-closure \
    2>&1 | grep -Ev "^E[0-9]{4}|slow_operation|XLA has|If you|^$"
echo "rc=$?"
cat "$OUT/ranking.csv"
$PY - "$OUT" <<'PYEOF'
import json, sys
import numpy as np
d = json.load(open(f"{sys.argv[1]}/tuned_parameters.json"))
print(f"{'scheme':18s} " + "  ".join(f"{v:>9s}" for v in ("theta", "u", "v")))
for e in d["schemes"]:
    c = (e.get("components_default") or {}).get("wangara", {})
    print(f"{e['scheme']:18s} " + "  ".join(f"{c.get(v, float('nan')):9.3f}"
                                           for v in ("theta", "u", "v")))
z = np.load(f"{sys.argv[1]}/profiles_wangara.npz", allow_pickle=True)
m = z["mask"].astype(bool)
for v in ("theta", "u", "v"):
    les = z[f"les_scmlev_{v}"][m]
    print(f"--- {v}: LES {np.nanmin(les):8.3f}..{np.nanmax(les):8.3f}  "
          f"spread {np.nanstd(les):.4f}")
    for s in ("holtslag_boville", "ysu", "louis"):
        k = f"scm_{s}_{v}"
        if k in z.files:
            a = z[k][m]
            print(f"    {s:18s} SCM {np.nanmin(a):8.3f}..{np.nanmax(a):8.3f}"
                  f"   bias {np.nanmean(a - les):+8.3f}")
PYEOF
echo "DONE $(date +%T)"
