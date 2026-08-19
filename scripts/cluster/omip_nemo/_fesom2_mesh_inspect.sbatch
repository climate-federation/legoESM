#!/bin/bash
#SBATCH --job-name=fesomins
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomins_%j.out
#
# Verify + unpack the fesom-jax CORE2 mesh/IC archive and print the mesh
# inventory: file list, formats, node/element counts, vertical levels, IC
# variables.  The ARRAY LAYOUT IS AN API -- read it before writing a loader.
set -uo pipefail
D=/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2
cd "$D" || exit 1

echo "[sha256] core2_mesh_ic.zip"
want=997b7372988536ba32616e9485f6ea931fa61514cee19412c26b7ebfe176dd92
got=$(sha256sum core2_mesh_ic.zip | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "FATAL: sha mismatch $got"; exit 1; }
echo "OK"

mkdir -p mesh_ic
unzip -o -q core2_mesh_ic.zip -d mesh_ic
echo "[unzip] contents:"
find mesh_ic -type f | head -40
echo
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'EOF'
from pathlib import Path
import numpy as np
root = Path("mesh_ic")
for p in sorted(root.rglob("*")):
    if p.is_dir():
        continue
    sz = p.stat().st_size
    print(f"-- {p} ({sz/1e6:.1f} MB)")
    if p.suffix == ".nc":
        try:
            import netCDF4 as nc
            d = nc.Dataset(p)
            print("   dims:", {k: len(v) for k, v in d.dimensions.items()})
            for v in list(d.variables)[:25]:
                print(f"   var {v}: {d.variables[v].shape} {d.variables[v].dtype}")
            d.close()
        except Exception as e:
            print("   nc read failed:", e)
    elif p.suffix == ".out" or p.name in ("nod2d.out", "elem2d.out", "aux3d.out"):
        with open(p) as f:
            head = [next(f, "").strip() for _ in range(3)]
        print("   head:", head)
    elif p.suffix == ".npz":
        z = np.load(p)
        for k in list(z.files)[:25]:
            print(f"   npz {k}: {z[k].shape} {z[k].dtype}")
EOF
echo "=== DONE ==="
