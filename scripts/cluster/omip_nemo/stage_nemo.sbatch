#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_stage
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/stage_%j.out

# OMIP-faithful step B (STAGE): pull morays container, clone NEMO 5.0.1 + ORCA1,
# and probe the forcing-download endpoints BEFORE the expensive deploy/compile.
# Runs on a compute node (has internet, confirmed by probe 8089676). All build
# artifacts live OUTSIDE the git repo at $WS.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1
SIF=$WS/morays_env_amd64.sif
mkdir -p $WS && cd $WS
echo "=== HOST ==="; hostname; date
module load singularity/3.7.1 2>&1 | tail -1

echo "=== 1. PULL MORAYS CONTAINER (1.68 GB) ==="
if [ ! -f "$SIF" ]; then
  time wget -q --show-progress -O "$SIF" \
    https://github.com/morays-community/morays-doc/releases/download/containers/morays_env_amd64.sif
  echo "wget_sif_rc=$?"
else echo "SIF already present"; fi
ls -lh "$SIF" 2>/dev/null

echo "=== 2. CLONE NEMO 5.0.1 (shallow) ==="
if [ ! -d "$WS/nemo_5.0.1" ]; then
  time git clone --branch 5.0.1 --depth 1 \
    https://forge.nemo-ocean.eu/nemo/nemo.git nemo_5.0.1
  echo "clone_nemo_rc=$?"
else echo "nemo_5.0.1 already present"; fi
echo "=== 3. CLONE ORCA1 into cfgs ==="
if [ ! -d "$WS/nemo_5.0.1/cfgs/ORCA1" ]; then
  git clone --depth 1 https://github.com/alexis-barge/ORCA1.git \
    nemo_5.0.1/cfgs/ORCA1 ; echo "clone_orca1_rc=$?"
else echo "ORCA1 cfg already present"; fi

echo "=== 4. PROBE FORCING ENDPOINTS ==="
echo "-- COREv2 FTP (gfdl) --"
timeout 60 wget --spider --tries=1 --timeout=40 \
  "ftp://data1.gfdl.noaa.gov/1/users/Niki.Zadeh/COREv2/data_1948_2006/NormalYearForcing/CORRECTED/slp.15JUNE2009.nc" 2>&1 | tail -6
echo "ftp_spider_rc=$?"
echo "-- Zenodo inputs (https) --"
timeout 40 curl -sIL --max-time 35 \
  "https://zenodo.org/records/14041098/files/data_repository.zip?download=1" 2>&1 | grep -iE "HTTP/|content-length" | head -4
echo "-- GH release inputs (https) --"
timeout 30 curl -sIL --max-time 25 \
  "https://github.com/alexis-barge/ORCA1/releases/download/inputs/Ice_initialization.nc" 2>&1 | grep -iE "HTTP/|content-length" | head -4

echo "=== 5. INSPECT CONTAINER (XIOS, compilers, netcdf) ==="
singularity exec "$SIF" bash -c '
  echo "XIOS:"; ls -d /home/jdoe/XIOS 2>&1; ls /home/jdoe/XIOS/lib 2>&1 | head -3
  echo "compilers:"; which mpif90 gfortran gcc 2>&1
  echo "netcdf libs:"; ls /usr/lib/$(uname -m)-linux-gnu/ 2>/dev/null | grep -iE "netcdf|hdf5" | head
  echo "uname -m:"; uname -m' 2>&1 | head -25
echo "container_inspect_rc=$?"

echo "=== DISK ==="; du -sh $WS/* 2>/dev/null; df -h $WS | tail -1
echo "=== DONE ==="
