#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=les_ref
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_ref_%x_%j.out
set +e
# ---------------------------------------------------------------------------
# PRODUCTION LES reference for the turbulence-closure tuning campaign.
#
# One case per submission ($LES_CASE). Each run produces the time-mean profile
# reference (theta, q_v, u, v + the resolved w'theta'/w'q_v' fluxes) that the
# SCM turbulence schemes are scored against.
#
# WALLTIME IS SIZED FROM MEASUREMENT, not a template. The smoke job 9277065 ran
# every case at a known small size and reported wall seconds; the production
# cost is that throughput scaled by the cell-count ratio:
#
#   case     smoke (cells, steps, wall)      production (cells, steps)   est.
#   bomex    40960,    900,  103 s  ->  307200,  21600   ~2 h
#   dycoms   65536,   1800,   65 s  -> 1769472,  28800   ~8 h
#   gabls1   49152,   9000,  134 s  ->  393216,  32400   ~3 h
#   wangara  49152,   1800,  ~60 s  ->  884736,  28800   ~6 h
#
# 36 h walltime therefore has >=4x margin on the largest estimate, which is the
# right direction to err: these decks feed everything downstream and a run
# killed at the wall produces nothing usable.
#
# RICO is deliberately NOT launched here. It is the most expensive case
# (1.64M cells x 43200 steps, ~15 h) and adds precipitating cumulus, which is
# not needed to prove the pipeline. It gets submitted once the tuning campaign
# has run end-to-end on the cheaper cases.
# ---------------------------------------------------------------------------
WT=${LES_WT:-/burg/glab/users/pg2328/les_scm_wt}
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
echo "case: ${LES_CASE:?set LES_CASE}"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/les_ref/$LES_CASE
# Clear the directory: the drivers overwrite only MATCHING frame indices, so a
# shorter rerun would leave higher-index frames from the previous run behind
# and the reference would silently average two runs together.
rm -rf $OUT
mkdir -p $OUT

t0=$SECONDS
case "$LES_CASE" in
  bomex)
    # Siebesma et al. 2003: 6.4 km x 6.4 km x 3 km, dx=100 m, dz=40 m, 6 h.
    # --record-frames 36 = every 10 min, so the trailing 2 h analysis window
    # holds 12 frames (the reference loader refuses a 1-frame "time mean").
    $PY scripts/run/run_bomex_les.py \
      --nx 64 --ny 64 --nz 75 --Lx 6400 --Ly 6400 --Lz 3000 \
      --hours 6 --dt 1.0 --record-frames 36 --print-every 2000 \
      --output $OUT
    ;;
  dycoms)
    # Stevens et al. 2005 RF01: 3.36 km x 3.36 km x 1.5 km, dx=35 m, dz~8 m, 4 h.
    $PY scripts/run/run_dycoms_les.py \
      --nx 96 --ny 96 --nz 192 --Lx 3360 --Ly 3360 --Lz 1500 \
      --hours 4 --dt 0.5 --record-frames 24 --print-every 4000 \
      --output $OUT
    ;;
  gabls1)
    # GABLS1 stable BL: 400 m cube, 9 h, surface cooling 0.25 K/h.
    $PY scripts/run/run_spectral_sbl.py \
      --nx 64 --ny 64 --nz 96 --Lx 400 --Ly 400 --Lz 400 \
      --hours 9 --dt 0.1 --record-frames 36 --print-every 20000 \
      --output $OUT
    ;;
  cbl)
    # Nieuwstadt CBL_N91 dry convective BL: 3.2 km x 3.2 km x 1.6 km, 4 h,
    # constant +0.06 K m/s surface flux, no Coriolis, theta0 = 300 K.
    #
    # THIS CASE WAS LABELLED "wangara" AND IT IS NOT WANGARA. run_spectral_cbl.py
    # is run at its DEFAULTS here, which are the Nieuwstadt profile; Wangara
    # Day 33 has theta ~ 277 K, a cosine surface flux peaking at 13:00 local,
    # and a height-dependent easterly geostrophic wind, none of which this
    # driver can express. The mislabel produced results/les_ref/wangara/, whose
    # contents are byte-for-byte the CBL run (Q0 = 0.06, theta 301.07-306.21 K).
    # Scoring an SCM "Wangara" case against it would have compared two
    # different problems under a name that hid the fact.
    $PY scripts/run/run_spectral_cbl.py \
      --nx 96 --ny 96 --nz 96 --Lx 3200 --Ly 3200 --Lz 1600 \
      --hours 4 --dt 0.5 --record-frames 24 --print-every 5000 \
      --output $OUT
    ;;
  wangara)
    # Wangara Day 33 (Clarke et al. 1971), DRY: diurnal surface heat flux
    # peaking at 13:00 local from a 09:00 start, southern-hemisphere Coriolis,
    # and a height-dependent easterly geostrophic wind. The forcing comes from
    # legoesm.atmosphere.forcing.wangara_day33, which the SCM case reads too,
    # so the two sides cannot be driven differently.
    #
    # 8 h covers 09:00-17:00, i.e. the growth of the convective layer through
    # the 13:00 flux maximum and into the afternoon decay. Lz = 2000 m clears
    # the depth the layer reaches; the sponge sits above 1700 m, which is where
    # the SCM case stops scoring.
    # --case-label is now derived from --case, but it is passed explicitly
    # here anyway: the first production run of this case omitted it, the flag
    # then defaulted to the literal "cbl", and eight hours of a genuine
    # Wangara integration were stamped as the other case. The reference loader
    # refused it and the eight-case tuning campaign died on the spot.
    $PY scripts/run/run_spectral_cbl.py --case wangara --case-label wangara \
      --nx 96 --ny 96 --nz 128 --Lx 4000 --Ly 4000 --Lz 2000 \
      --z0 0.01 --hours 8 --dt 0.5 --record-frames 48 --print-every 5000 \
      --output $OUT
    ;;
  astex)
    # ASTEX flight 209 stratocumulus. Same simple-LW engine as DYCOMS (both
    # decks set doradsimple), but no Coriolis, a stronger subsidence
    # divergence read off its lsf, and a deeper domain: its sounding runs to
    # 1637 m against DYCOMS's 840 m inversion. Lz = 2000 m clears it at
    # dz = 10.4 m, which still resolves the 612-637 m inversion.
    # The deck is a 6 h run (prm: nstop = 7200 at dt = 3 s).
    $PY scripts/run/run_dycoms_les.py --case astex \
      --nx 96 --ny 96 --nz 192 --Lx 4800 --Ly 4800 --Lz 2000 \
      --hours 6 --dt 0.5 --record-frames 36 --print-every 4000 \
      --output $OUT
    ;;
  ekman)
    # Neutral Ekman layer: 2 km x 2 km x 1 km, f = 1e-4 (inertial period
    # 17.5 h), geostrophic 10 m/s. Run 6 h so the spiral is established --
    # the driver's 1.5 h default is a smoke length, not a reference.
    $PY scripts/run/run_spectral_les.py \
      --nx 96 --ny 96 --nz 96 --Lx 2000 --Ly 2000 --Lz 1000 \
      --ekman --Ug 10 --fcor 1.0e-4 --z0 0.1 \
      --hours 6 --dt 0.4 --record-frames 36 --print-every 10000 \
      --case-label ekman --output $OUT
    ;;
  rico)
    # van Zanten et al. 2011 precipitating trade cumulus, 24 h. Expensive.
    $PY scripts/run/run_rico_les.py \
      --nx 128 --ny 128 --nz 100 --Lx 12800 --Ly 12800 --Lz 4000 \
      --hours 24 --dt 2.0 --record-frames 48 --print-every 5000 \
      --output $OUT
    ;;
  *)
    echo "unknown LES_CASE=$LES_CASE"; exit 2 ;;
esac
rc=$?
echo "CASE $LES_CASE rc=$rc wall=$((SECONDS - t0))s"

# The reference is only usable if the frames carry finite, nonzero fluxes.
$PY - "$OUT" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
frames = sorted((out / "profiles").glob("prof_*.npz"))
if not frames:
    print("REFERENCE UNUSABLE: no profile frames written"); sys.exit(1)
with np.load(frames[-1], allow_pickle=True) as f:
    have = set(f.files)
    need = {"z", "theta", "wth"}
    missing = need - have
    if missing:
        print(f"REFERENCE UNUSABLE: last frame missing {sorted(missing)}")
        sys.exit(1)
    wth = f["wth"]
    if not np.all(np.isfinite(wth)):
        print("REFERENCE UNUSABLE: wth non-finite in last frame"); sys.exit(1)
    print(f"frames={len(frames)} nz={wth.size} "
          f"t_last={float(f['t_hours']):.3f} h wth|max|={np.abs(wth).max():.4e}")
    if "wqv" in have:
        print(f"  wqv|max|={np.abs(f['wqv']).max():.4e}")
print("REFERENCE OK")
PYEOF
echo "verify rc=$?"
echo "DONE $(date +%T)"
