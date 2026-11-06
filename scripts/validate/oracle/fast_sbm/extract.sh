#!/bin/bash
# Extract the WRF v4.8.0 FAST_SBM routines the oracle driver calls, VERBATIM by
# line range from phys/module_mp_fast_sbm.F
# (wrf-model/WRF @ 06d4240ae989cc3e50af412bb472df3d9048783c).
# Usage: extract.sh <WRF checkout> <out dir>
set -euo pipefail
F=$1/phys/module_mp_fast_sbm.F; O=$2
test "$(git -C "$1" rev-parse HEAD)" = 06d4240ae989cc3e50af412bb472df3d9048783c
mkdir -p "$O"
sed -n '447,1691p' "$F" > "$O/sbm_collision.F90"      # module module_mp_SBM_Collision
sed -n '1694,3010p' "$F" > "$O/sbm_auxiliary.F90"     # module module_mp_SBM_Auxiliary
{
cat <<'HDR'
! Extraction of FAST_SBM main-module routines (WRF v4.8.0 06d4240a phys/module_mp_fast_sbm.F)
! ONECOND1 l.6748-7143, FREEZ l.6560-6636, J_W_MELT l.6638-6746, FALFLUXHUCM_Z l.5324-5382,
! ecoalmass..vTBeard l.8784-9048. Only the module-variable declarations they need are added.
module sbm_extract
  use module_mp_SBM_Auxiliary, only: JERRATE_KS, JERTIMESC_KS, JERSUPSAT_KS, JERDFUN_KS, JERDFUN_NEW_KS, POLYSVP
  implicit none
  INTEGER, PARAMETER, PRIVATE:: R8SIZE = 8
  INTEGER, PARAMETER, PRIVATE:: R4SIZE = 4
  INTEGER, PARAMETER :: NKRP = 33
  REAL (KIND=R4SIZE) :: FR_LIM(NKRP) = 0.0, FRH_LIM(NKRP) = 0.0   ! oracle l.3997 (never initialised there)
contains
HDR
sed -n '5324,5382p' "$F"
sed -n '6560,6746p' "$F"
sed -n '6748,7143p' "$F"
sed -n '8784,9048p' "$F"
echo 'end module sbm_extract'
} > "$O/sbm_extract.F90"
