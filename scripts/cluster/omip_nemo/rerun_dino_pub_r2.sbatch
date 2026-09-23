#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dinopub_r2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_pub_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
flt() { grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin|tensorflow|TpuPlatform|Updating files"; }
RD="$PY scripts/run/run_dino.py"; R=$WT/results
echo "########## RUNS (reproduce on main: feecd6a67) ##########"
echo "== latlon wright 365d =="; $RD --grid latlon --eos wright   --days 365 --snapshot-every-days 15 --output-dir $R/dino_latlon_wright_r2 2>&1 | flt | tail -3
echo "== latlon seos 365d ==";   $RD --grid latlon --eos nemo_seos --days 365 --snapshot-every-days 15 --output-dir $R/dino_latlon_seos_r2   2>&1 | flt | tail -3
echo "== mpas wright 90d ==";    $RD --grid mpas   --eos wright   --days 90  --snapshot-every-days 5  --output-dir $R/dino_mpas_wright_r2   2>&1 | flt | tail -3
echo "== mpas seos 90d ==";      $RD --grid mpas   --eos nemo_seos --days 90  --snapshot-every-days 5  --output-dir $R/dino_mpas_seos_r2     2>&1 | flt | tail -3
echo "########## CANONICAL FIGURES ##########"
echo "== publication cross-grid (wright) =="; $PY scripts/plot/plot_dino_publication.py $R/dino_latlon_wright_r2 $R/dino_mpas_wright_r2 --day 90 --outdir $R 2>&1 | flt | tail -4
echo "== ACC spin-up (latlon wright) =="; $PY scripts/plot/plot_dino_acc.py $R/dino_latlon_wright_r2 --out $R/dino_acc_wright_r2.png 2>&1 | flt | tail -2
echo "== ACC spin-up (latlon seos) =="; $PY scripts/plot/plot_dino_acc.py $R/dino_latlon_seos_r2 --out $R/dino_acc_seos_r2.png 2>&1 | flt | tail -2
echo "== vs NEMO equilibrium =="; $PY scripts/plot/plot_dino_vs_paper.py data/dino_paper/Reference_experiments/EXP_R1/restart_prod_1deg.nc $R/dino_latlon_wright_r2 $R/dino_mpas_wright_r2 --day 90 --out $R/dino_vs_paper_r2.png 2>&1 | flt | tail -3
echo "########## HEADLINE: ACC overshoot + T(z) vs NEMO (4yr) ##########"
$PY scripts/tmp/_pub_dino_nemo_comparison.py 4 $R/dino_pub_nemo_comparison.png 2>&1 | flt | tail -8
echo "=== ALL DONE ==="; ls -lh $R/*.png 2>/dev/null
