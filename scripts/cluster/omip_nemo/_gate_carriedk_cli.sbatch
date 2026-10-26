#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carrkgate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/carrkgate_%j.out
# Gate for --tke-preclosure-coeff-source before the arm is submitted:
#   1. the new CLI round-trip module (default unchanged, flag reaches the
#      config, unknown value raises, refused on grids with no carried memory);
#   2. the closure's own carried-coefficient unit module, unchanged by this
#      diff but the thing the flag now selects in production;
#   3. NON-VACUITY: --help must print the flag, and the arm card's exact flag
#      string must parse, so a typo in the card cannot reach the queue.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

echo "=== 1. CLI round-trip ==="
$PY -m pytest -q tests/unit/test_run_omip_core2_preclosure_coeff_cli.py 2>&1 | tail -5

echo "=== 2. closure carried-coefficient unit module ==="
JAX_ENABLE_X64=1 $PY -m pytest -q tests/ocean/unit/test_tke_carried_coefficients.py 2>&1 | tail -5

echo "=== 3. the card's own flag string parses ==="
# grep the RUN LINE, not the header: the pre-registration prose names the flag
# too, and matching both fed argparse the flag twice with no value.
FLAG=$(grep -v '^ *#' scripts/cluster/omip_nemo/_trp_carriedk_d30.sbatch \
       | grep -o -- "--tke-preclosure-coeff-source [a-z_][a-z_]*")
echo "card flag: $FLAG"
[ -n "$FLAG" ] || { echo "FATAL: the arm card carries no coeff-source flag"; exit 1; }
$PY - <<PYEOF
import shlex, sys
sys.argv = ["run_omip_core2.py"]
import scripts.run.run_omip_core2 as core2
p = core2._build_arg_parser()
args = p.parse_args(shlex.split("""$FLAG"""))
print("parsed ->", args.tke_preclosure_coeff_source)
assert args.tke_preclosure_coeff_source == "carried_previous_step"
cfg = core2.orca1_zdftke_config(
    preclosure_coeff_source=args.tke_preclosure_coeff_source)
print("config  ->", cfg.tke_preclosure_coeff_source)
assert cfg.tke_preclosure_coeff_source == "carried_previous_step"
base = core2.orca1_zdftke_config()
print("default ->", base.tke_preclosure_coeff_source)
assert base.tke_preclosure_coeff_source == "current_subiteration"
# The arm must differ from the baseline in this field and NOTHING else.
diff = [f for f in base._fields
        if getattr(base, f) != getattr(cfg, f)]
print("fields differing between arm and baseline card:", diff)
assert diff == ["tke_preclosure_coeff_source"], diff
print("ONE VARIABLE CONFIRMED")
PYEOF
echo "=== 4. the two cards differ in exactly one flag line ==="
diff <(grep -oE -- "--[a-z0-9-]+" scripts/cluster/omip_nemo/_trp_base2_d30.sbatch | sort -u) \
     <(grep -oE -- "--[a-z0-9-]+" scripts/cluster/omip_nemo/_trp_carriedk_d30.sbatch | sort -u)
echo "(a single '> --tke-preclosure-coeff-source' line above is the expected diff)"
echo DONE
