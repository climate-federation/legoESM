#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=kprofsmk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kprofsmk_%j.out
#
# PROVE THE DUMP EXECUTES BEFORE SPENDING TWO GPU ARMS ON IT.
#
# `--kprofile-snapshots` asks the model for one tendency evaluation at the
# snapshot's state and stores the K_v/A_v it carries.  If that call raises on
# the production card -- a missing vertex mask, a config the tendency entry
# point will not take -- the run still completes and prints SKIPPED, which is
# the right behaviour for an integration but means a 24-hour arm would produce
# no measurement at all.  A short smoke on the same closure settles it for
# minutes of queue time.
#
# PASS  : the final snapshot carries K_M_diag and K_H_diag, both finite and
#         positive somewhere, with the interface level count (nlev-1).
# FAIL  : "SKIPPED" in the log, or the keys absent -- do NOT launch the arms.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=$ROOT/results/omip_nemo/_smoke_kprofile
rm -rf $OUT

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  --smoke --dt 600 --diag-every-days 1 --kprofile-snapshots \
  --nemo-vertical --partial-cell \
  --tripole-vmix tke --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 \
  --tke-prognostic --tke-kappa-convention veros_sqrte \
  --output $OUT 2>&1 | tail -30
echo "run_rc=${PIPESTATUS[0]}"

$PY - <<'PY'
import glob
import numpy as np
f = sorted(glob.glob(
    "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
    "_smoke_kprofile/snapshot_*.npz"))
if not f:
    raise SystemExit("FAIL: no snapshot written")
z = np.load(f[-1])
print("snapshot:", f[-1])
print("keys:", sorted(z.files))
for k in ("K_M_diag", "K_H_diag"):
    if k not in z:
        raise SystemExit(f"FAIL: {k} absent -- diagnose_vertical_K did not run")
    a = z[k]
    fin = np.isfinite(a)
    print(f"{k}: shape {a.shape}  finite {fin.mean():.3f}  "
          f"max {np.nanmax(np.where(fin, a, np.nan)):.4g}  "
          f"median>0 {np.nanmedian(a[fin & (a > 0)]) if (fin & (a > 0)).any() else float('nan'):.4g}")
    if not (fin & (a > 0)).any():
        raise SystemExit(f"FAIL: {k} has no finite positive value")
if "z_interface_ref" not in z:
    raise SystemExit("FAIL: z_interface_ref absent -- the depth axis is a guess")
nlev = z["T"].shape[-1]
nk = z["K_M_diag"].shape[-1]
zi = z["z_interface_ref"]
print(f"levels: T {nlev}, K {nk}, z_interface_ref {zi.size}  "
      f"(faces {zi[0]:.2f}..{zi[-1]:.2f} m)")
if nk != nlev - 1 or zi.size != nk:
    raise SystemExit(f"FAIL: K has {nk} levels, T {nlev}, z_interface_ref "
                     f"{zi.size}; expected interior interfaces nlev-1")
# The Prandtl number the whole campaign hangs on: viscosity over diffusivity,
# in the upper ocean, must be a sane O(1-10) where both are closure-set.
avm, avt = z["K_M_diag"], z["K_H_diag"]
ok = np.isfinite(avm) & np.isfinite(avt) & (avt > 1e-6) & (avm > 1e-6)
if ok.any():
    pr = np.median(avm[ok] / avt[ok])
    print(f"global median Pr where both closure-set: {pr:.3f}")
print("PASS")
PY
echo DONE
