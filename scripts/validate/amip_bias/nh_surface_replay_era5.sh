#!/bin/bash
# ERA5 references for nh_surface_replay.py, Dec 12-31 2001 (model days 345-364), 30-90N.
#   fc/1D  daily SUMS [J/m2 or N m-2 s] of ssrd 169, ssr 176, strd 175, str 177, sshf 146, slhf 147
#          -> e5d_<p>.nc (divide by 86400 for the daily-mean flux)
#   fc/1H  ONE-HOUR accumulations; the record stamped YYYY-MM-DDT00:00 covers 23-24 UTC of
#          the previous day: strd 175, str 177, ewss 180, nsss 181, sshf 146 -> e5h_<p>.nc (/3600)
#   an/1H  00Z analyses: u10 165, v10 166, fsr 244 -> e5a_<p>.nc
#   ml/an/1H 00Z u 131 / v 132 on model levels 133-137 (spectral T639 -> F320 Gaussian) -> e5ml_<p>.nc
# The checked control: the hourly 1H values are 1-hour accumulations (global mean ssrd
# 6.7e5 J/m2 per record = 186 W/m2) and the 1D values are daily sums (1.68e7 = 194 W/m2).
set -euo pipefail
source /usr/share/Modules/init/bash; module load cdo/2.6.0-gcc-11.2.0
OUT=${1:-/scratch/b/b381103/nhcold/era5rep}; cd "$OUT"
P=/pool/data/ERA5/E5
DATES=$(python3 -c "
import datetime as d
print(' '.join((d.date(2001,1,1)+d.timedelta(n)).isoformat() for n in range(345,365)))")
BOX="-sellonlatbox,0,360,30,90 -setgridtype,regular"
for p in 169 176 175 177 146 147; do
  [ -f e5d_$p.nc ] || cdo -s -f nc $BOX -seldate,2001-12-12,2001-12-31 $P/sf/fc/1D/$p/E5sf12_1D_2001-12_$p.grb e5d_$p.nc &
done; wait
hour() { k=$1; p=$2; dt=$3; o=tmp_${k}_${p}_${dt}.nc; [ -f $o ] && return
  case $k in
    h) cdo -s -f nc -sellonlatbox,0,360,30,90 -setgridtype,regular -seltime,00:00:00 $P/sf/fc/1H/$p/E5sf12_1H_${dt}_$p.grb $o ;;
    a) cdo -s -f nc -sellonlatbox,0,360,30,90 -setgridtype,regular -seltime,00:00:00 $P/sf/an/1H/$p/E5sf00_1H_${dt}_$p.grb $o ;;
    m) cdo -s -f nc -sellonlatbox,0,360,30,90 -sp2gp -sellevel,133,134,135,136,137 -seltimestep,1 $P/ml/an/1H/$p/E5ml00_1H_${dt}_$p.grb $o ;;
  esac; }
export -f hour; export P
{ for dt in $DATES; do
    for p in 175 177 180 181 146; do echo "h $p $dt"; done
    for p in 165 166 244; do echo "a $p $dt"; done
    for p in 131 132; do echo "m $p $dt"; done
  done; } | xargs -P 12 -n 3 bash -c 'hour "$0" "$1" "$2"'
for p in 175 177 180 181 146; do cdo -s -O mergetime tmp_h_${p}_*.nc e5h_$p.nc; done
for p in 165 166 244; do cdo -s -O mergetime tmp_a_${p}_*.nc e5a_$p.nc; done
for p in 131 132; do cdo -s -O mergetime tmp_m_${p}_*.nc e5ml_$p.nc; done
ls -la e5*.nc
