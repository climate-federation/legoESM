#!/bin/bash
#SBATCH --job-name=cardchk
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cardchk_%j.out
# A card whose flags do not parse is worthless. Extract the flag list from the
# committed card and push it through the runner's own parser plus its
# cross-flag gates -- WITHOUT running the model.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY -u - <<'PY'
import importlib.util, pathlib, re, shlex
RUN = pathlib.Path("scripts/run/run_omip_core2.py")
CARD = pathlib.Path("scripts/cluster/omip_nemo/orca1_faithful_card.sbatch")
spec = importlib.util.spec_from_file_location("_r", RUN)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# Pull the argument lines out of the card: real flags only, skipping the
# backtick comment blocks the shell treats as no-ops.
argv = []
for raw in CARD.read_text().splitlines():
    line = raw.strip().rstrip("\\").strip()
    if not line.startswith("--"):
        continue
    if "`" in line:
        continue
    for tok in shlex.split(line):
        argv.append(tok)
argv = [{"$YEARS": "0.0822", "$OUT": "/tmp/x", "$FORCING": "/tmp/f"}.get(a, a) for a in argv]
print("CARD FLAGS:", " ".join(argv))
print()
p = m._build_arg_parser()
args = p.parse_args(argv)
print("PARSED OK")
for k in ("grid", "dt", "nlev", "gm_treguier", "gm_kappa_min",
          "gm_slope_scheme", "gm_slope_positions", "gm_bolus_advection",
          "gm_msc_stabilize", "redi_coefficient", "redi_aht0", "eos",
          "ice_exchange", "ice_snow", "ice_freeze_scheme",
          "forcing_time_interp", "prognostic_sea_ice"):
    print(f"  {k} = {getattr(args, k, '<<MISSING>>')}")
assert args.eos is None, "card must NOT select an EOS: teos10 is refused"
assert args.redi_coefficient == "nemo21"
assert args.gm_kappa_min == 0
assert args.forcing_time_interp == "linear"
assert args.ice_exchange == "nemo_si3"
assert args.ice_freeze_scheme == "unesco"
print("\nALL ADOPTED SELECTORS PRESENT AND CORRECT")
PY
echo DONE
