#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=gradcase
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gradcase_%j.out
set +e
# The seven-case campaign evaluated all nine closures finite and then tuned
# NONE of them: every scheme came back `no_active_gradient` with essentially
# every parameter frozen for "non-finite gradient". The preflight LOSS is
# finite, so this is a NaN in the BACKWARD pass, and it cannot be any one
# closure because it hits all nine.
#
# The joint loss is a mean over cases, so ONE case with a NaN gradient poisons
# every scheme's gradient. The five-case campaign that tuned fine (job 9333698:
# bomex, rico, gabls1, cbl, ekman) did not include astex or dycoms -- the two
# arms that select the Stevens (2005) `simple_lw` radiation kernel -- and did
# not include wangara.
#
# So: run the SAME preflight, one case at a time, and see which case's arm
# produces the NaN. One cheap closure is enough because the defect is shared;
# louis has six spec'd parameters and the shortest graph.
#
# `--steps 1` reaches the preflight gradient and stops. The driver reports
# `no_active_gradient` with a per-parameter reason, which is the measurement.
#
# ---------------------- 2026-08-21 REUSED, for smagorinsky ------------------
# Same instrument, second question. In the eight-case fit at pin 5d3ae37cf,
# smagorinsky reports preflight loss = 1 (exactly 1 by construction under
# `default_relative`) with |grad| of 9.27e17 / 1.75e17 / 1.31e17 on C_s / Pr_t
# / l_mix_max -- all three within 7x of each other. A single near-singular
# denominator would show up in ONE parameter, so the flat spread points at the
# objective rather than at a coefficient, and the earlier "the mixing-length
# cutoff is non-smooth" reading does not account for it.
#
# The joint loss is a mean, so per-case gradients ADD and one bad case sets the
# scale for every parameter at once -- which is exactly how CLUBB's missing
# surface BC was finally localised (#1508). Enumerate before theorising.
#
# PRE-REGISTERED:
#   CONFIRMS case-localised -> one or two cases carry |grad| >= 1e10 while the
#                              rest sit at <= 1e4.
#   REFUTES it              -> all eight cases come back near 1e17, i.e. the
#                              defect is a shared term in the closure and not a
#                              property of any regime.
# The per-case LOSS carries no information here (default_relative normalises
# each case by its own default, so a single-case arm is 1 by construction).
# The GRADIENT is the measurement.
# ---------------------------------------------------------------------------
: "${G_WT:?set G_WT to the pinned worktree}"
WT=$G_WT
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
SCHEME=${G_SCHEME:-louis}
CASE_LIST=${G_CASES:-"bomex ekman cbl gabls1 rico astex dycoms"}
OUT=${G_OUT:-$R/scm_les_turbulence/_gradcase}
rm -rf "$OUT"

for c in $CASE_LIST; do
  echo "=================== $c ==================="
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$c:$R/les_ref/$c" --outdir "$OUT/$c" \
      --nlev 48 --steps 1 --schemes "$SCHEME" \
      --joint-aggregation default_relative --surface-flux-to-closure \
      2>&1 | grep -E "preflight loss|largest:|^\[tune\]|^\[eval\]|joint|status|FAILED|Traceback|Error" | tail -10
  echo "  --- frozen reasons ---"
  $PY - "$OUT/$c/tuned_parameters.json" <<'PYEOF'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
except Exception as e:                                    # noqa: BLE001
    print(f"  no report: {e}"); raise SystemExit(0)
for s in p["schemes"]:
    fr = s.get("frozen") or {}
    nonfinite = [k.rsplit('.', 1)[-1] for k, v in fr.items()
                 if "non-finite" in v]
    zero = [k.rsplit('.', 1)[-1] for k, v in fr.items() if "non-finite" not in v]
    print(f"  {s['scheme']}: status={s['status']} trained={s['n_trained']} "
          f"nonfinite_grad={len(nonfinite)} zero_grad={len(zero)}")
    if nonfinite:
        print(f"    non-finite: {nonfinite}")
PYEOF
done

if [ "${G_RAD_DISCRIM:-1}" = 1 ]; then
echo "=================== the two stratocumulus arms WITHOUT simple_lw ==============="
# If astex/dycoms are the culprits, the discriminator is the RADIATION, not the
# case: same deck, same reference, kernel off. This is a confounded score (the
# LES has the longwave and the SCM would not), so it is a GRADIENT probe only
# and its joint number must not be quoted as a scheme result.
for c in astex dycoms; do
  echo "--- $c, radiation forced off ---"
  $PY - <<PYEOF 2>&1 | tail -12
import importlib.util, sys, pathlib
spec = importlib.util.spec_from_file_location(
    "drv", "scripts/run/run_scm_les_turbulence_tuning.py")
drv = importlib.util.module_from_spec(spec); sys.modules["drv"] = drv
spec.loader.exec_module(drv)
# Same arm, but with the case dropped from the simple_lw set, which is the ONE
# thing that differs.
drv._SIMPLE_LW_CASES = frozenset()
rc = drv.main(["--cases", "$c:$R/les_ref/$c", "--outdir", "$OUT/${c}_norad",
               "--nlev", "48", "--steps", "1", "--schemes", "louis",
               "--joint-aggregation", "default_relative",
               "--surface-flux-to-closure"])
import json
p = json.load(open("$OUT/${c}_norad/tuned_parameters.json"))
for s in p["schemes"]:
    fr = s.get("frozen") or {}
    nf = [k for k, v in fr.items() if "non-finite" in v]
    print(f"  NO-RAD {s['scheme']}: status={s['status']} trained={s['n_trained']} "
          f"nonfinite_grad={len(nf)}")
PYEOF
done
fi
echo "DONE $(date +%T)"
