#!/bin/bash
#SBATCH --job-name=fesomdl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomdl_%j.out
#
# Fetch the FESOM2-JAX CORE2 data package (Zenodo 10.5281/zenodo.21324319,
# from arXiv:2608.01546) -- the mesh + IC the FESOM2 hindcast used, and the
# portable source for a global FESOM2-mesh arm of the cross-grid comparison
# (docs/ocean/fidelity/fesom2_gap_analysis.md; the Levante pool paths are not
# reachable from Ginsburg).  Files land in data/fesom2_core2/ (gitignored
# runtime data).
set -uo pipefail
OUT=/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2
mkdir -p "$OUT"
cd "$OUT"

REC=21324319
echo "[zenodo] querying record $REC"
curl -sfL "https://zenodo.org/api/records/$REC" -o record.json \
  || { echo "FATAL: zenodo API unreachable or record missing"; exit 1; }
python3 - <<'EOF'
import json
r = json.load(open("record.json"))
files = r.get("files", [])
print(f"[zenodo] {len(files)} files, total "
      f"{sum(f['size'] for f in files)/1e9:.2f} GB")
for f in files:
    print(f"  {f['key']}  {f['size']/1e6:.1f} MB  {f['links']['self']}")
with open("filelist.txt", "w") as out:
    for f in files:
        out.write(f"{f['links']['self']}\t{f['key']}\n")
EOF
[ -s filelist.txt ] || { echo "FATAL: no files listed"; exit 1; }

NFAIL=0
while IFS=$'\t' read -r url key; do
  if [ -s "$key" ]; then echo "[skip] $key exists"; continue; fi
  echo "[get ] $key"
  curl -sfL --retry 3 -o "$key.part" "$url" && mv -f "$key.part" "$key" \
    || { echo "FAILED: $key"; NFAIL=$((NFAIL+1)); }
done < filelist.txt
echo; ls -la "$OUT" | head -30
echo "=== DONE (failed downloads: $NFAIL) ==="
exit $((NFAIL > 0))
