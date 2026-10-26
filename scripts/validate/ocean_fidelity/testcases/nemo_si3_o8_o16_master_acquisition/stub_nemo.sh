#!/usr/bin/env bash
# Checklist-15 executable: exercises run.sh control flow without running NEMO.
set -Eeuo pipefail
if [[ ${O8O16_STUB_FAIL:-0} == 1 ]]; then
  printf 'STUB_REQUESTED_FAILURE\n' >&2
  exit 7
fi
cat > ocean.output <<'EOF'
 AAAAAAAA
 AAAAAAAA
 Ni0glo = 180
 Nj0glo = 148
 nn_hls = 2
 jpl = 5
 nlay_i = 10
 nlay_s = 5
 ln_virtual_itd = F
 ln_icedyn = T
 ln_icethd = T
 ln_cat_hfn = T
 rn_himean = 2.0
 ln_pnd = T
 ln_pnd_alb = T
 ln_blk = T
 ln_mixcpl = F
 ln_Cx_ice_cst = T
 ln_rstart = F
 nn_it000 = 1
 nn_itend = 100
 nn_fsbc = 1
 jpni = 1
 jpnj = 1
EOF
cat > output.namelist.dyn <<'EOF'
&namdrg
   ln_drgice_imp = .true.
/
EOF
printf 'stub restart, observer independent\n' > ORCA2_00000100_restart.nc
if [[ -d o8_o16_master_records ]]; then
  : "${O8O16_GATE:?O8O16_GATE is required by stub NEMO}"
  : "${O8O16_PY:?O8O16_PY is required by stub NEMO}"
  "$O8O16_PY" "$O8O16_GATE" --write-stub-records o8_o16_master_records >/dev/null
fi
printf 'STOP 0\n'
