#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_plot
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_plot_%j.out
set +e
# Render the tuning figures from a campaign directory that is STILL BEING
# WRITTEN. The driver flushes its whole report after every scheme, so a
# mid-flight directory is a valid partial campaign -- but it is also a moving
# target, so copy it first and plot the copy. Plotting in place would mix a
# ranking.csv from scheme N with a profiles npz from scheme N+1.
#
# CPU only: both plotters are numpy + matplotlib over small npz files.
SRC=${SRC:-/burg-archive/glab/users/pg2328/legoESM/results/scm_les_turbulence/all_seven}
DST=${DST:-${SRC}_snapshot}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd /burg-archive/glab/users/pg2328/legoESM || exit 1
echo "sha: $(git rev-parse HEAD)  host: $(hostname)"

rm -rf "$DST"; mkdir -p "$DST"
cp -f "$SRC"/*.npz "$SRC"/*.csv "$SRC"/*.json "$SRC"/*.md "$DST"/ || exit 2
echo "snapshot taken $(date +%T) from $SRC"
$PY - "$DST" <<'PYEOF'
import json, sys
d = json.load(open(f"{sys.argv[1]}/tuned_parameters.json"))
print("run_complete:", d.get("run_complete"),
      " missing:", d.get("schemes_missing"))
PYEOF

$PY scripts/plot/plot_scm_les_tuning_summary.py "$DST"
echo "  summary rc=$?"
for c in $(python3 -c "
import json,sys
d=json.load(open('$DST/tuned_parameters.json'))
print(' '.join(d['protocol']['cases']))"); do
  $PY scripts/plot/plot_scm_les_turbulence.py "$DST" --case "$c" \
      --out "$DST/profiles_vs_les_$c.png" && echo "  plotted $c"
done
echo "=== ranking ==="; cat "$DST/ranking.csv"
ls -la "$DST"/*.png
echo "DONE $(date +%T)"
