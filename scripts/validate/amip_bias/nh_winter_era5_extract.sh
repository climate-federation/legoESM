#!/bin/bash
# ERA5 00Z instants matching model checkpoint days (day N = 2001-01-01 + N days, 00Z), 30-90N.
source /usr/share/Modules/init/bash; module load cdo/2.6.0-gcc-11.2.0
cd /scratch/b/b381103/nhcold/era5
DATES=$(python3 -c "
import datetime as d
b=d.date(2001,1,1); days=sorted(set(list(range(5,346,5))+list(range(330,367))))
print(' '.join((b+d.timedelta(n)).isoformat() for n in days))")
one() { p=$1; dt=$2; o=tmp_${p}_${dt}.nc; [ -f $o ] && return
  cdo -s -f nc -sellonlatbox,0,360,30,90 -setgridtype,regular -seltime,00:00:00 /pool/data/ERA5/E5/sf/an/1H/$p/E5sf00_1H_${dt}_$p.grb $o; }
export -f one
for p in 141 139 170 183 236 235 167 243 033; do
  for dt in $DATES; do echo "$p $dt"; done
done | xargs -P 32 -n 2 bash -c 'one "$0" "$1"'
for p in 141 139 170 183 236 235 167 243 033; do cdo -s -O mergetime tmp_${p}_*.nc e5_${p}.nc; done
cdo -s -f nc -sellonlatbox,0,360,30,90 -setgridtype,regular /pool/data/ERA5/E5/sf/an/IV/172/E5sf00_IV_INVARIANT_172.grb e5_172.nc
ls -la e5_*.nc
