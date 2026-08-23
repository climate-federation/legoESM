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
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
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
        raise SystemExit(f"FAIL: {k} absent -- the dump did not execute")
    a = z[k]
    fin = np.isfinite(a)
    print(f"{k}: shape {a.shape}  finite {fin.mean():.3f}  "
          f"max {np.nanmax(np.where(fin, a, np.nan)):.4g}  "
          f"median>0 {np.nanmedian(a[fin & (a > 0)]) if (fin & (a > 0)).any() else float('nan'):.4g}")
    if not (fin & (a > 0)).any():
        raise SystemExit(f"FAIL: {k} has no finite positive value")
nlev = z["T"].shape[-1]
print(f"levels: T {nlev}, K {z['K_M_diag'].shape[-1]} "
      f"({'interfaces' if z['K_M_diag'].shape[-1] == nlev - 1 else 'centres'})")
print("PASS")
PY
echo DONE
