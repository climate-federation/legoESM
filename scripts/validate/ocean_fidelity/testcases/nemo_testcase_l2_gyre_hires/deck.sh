# Shared deck resolver for the GYRE stage-entry acquisition.

gyre_hires_select_deck() {
  local deck=$1 here=$2
  BASE_DECK=$here/../configs/gyre_omip_l2_phase3_namelist_cfg
  case "$deck" in
    shipped)
      DECK_TAG=gyre_r1
      RUN_NAME=GYRE_OMIP_L2_P3
      DECK_PATCH=
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round1/oracle_gyre_r1_stage
      ;;
    r4)
      DECK_TAG=gyre_r4
      RUN_NAME=GYRE_OMIP_L2_P3_R4
      DECK_PATCH=$here/namelist_cfg_r4_kt10.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round1/oracle_gyre_r4_stage
      ;;
    r4_smag)
      DECK_TAG=gyre_r4_smag
      RUN_NAME=GYRE_OMIP_L2_P3_R4_SMAG
      DECK_PATCH=$here/namelist_cfg_r4_smag_kt10.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round2/oracle_gyre_r4_smag_stage
      ;;
    *)
      printf 'REFUSE_DECK: expected shipped, r4, or r4_smag\n' >&2
      return 64
      ;;
  esac
}

gyre_hires_namelist_value() {
  local path=$1 key=$2
  awk -v key="$key" '
    BEGIN { IGNORECASE=1 }
    /^[[:space:]]*!/ { next }
    {
      line=$0
      sub(/!.*/, "", line)
      if (line ~ "^[[:space:]]*" key "[[:space:]]*=") {
        sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", line)
        gsub(/[[:space:]]/, "", line)
        print line
        exit
      }
    }' "$path"
}

gyre_hires_verify_deck() {
  local deck=$1 namelist=$2
  local nn=1 dt=14400. ld=100.e+3 lv=100.e+3 runstat=
  if [[ "$deck" == r4 || "$deck" == r4_smag ]]; then
    nn=4 dt=3600. ld=25000. lv=25000. runstat=.TRUE.
  fi
  [[ "$(gyre_hires_namelist_value "$namelist" nn_it000)" == 1 \
      && "$(gyre_hires_namelist_value "$namelist" nn_itend)" == 10 \
      && "$(gyre_hires_namelist_value "$namelist" nn_stock)" == 10 \
      && "$(gyre_hires_namelist_value "$namelist" nn_write)" == 10 \
      && "$(gyre_hires_namelist_value "$namelist" nn_GYRE)" == "$nn" \
      && "$(gyre_hires_namelist_value "$namelist" rn_Dt)" == "$dt" \
      && "$(gyre_hires_namelist_value "$namelist" rn_Ld)" == "$ld" ]] || {
    printf 'REFUSE_RESOLVED_DECK: %s does not match %s\n' "$namelist" "$deck" >&2
    return 68
  }
  if [[ "$deck" == r4 ]]; then
    [[ "$(gyre_hires_namelist_value "$namelist" rn_Lv)" == "$lv" ]] || {
      printf 'REFUSE_LAPLACIAN: r4 deck must carry rn_Lv=25000\n' >&2
      return 68
    }
  fi
  if [[ "$deck" == r4 || "$deck" == r4_smag ]]; then
    [[ "$(gyre_hires_namelist_value "$namelist" 'sn_cfctl%l_runstat')" == "$runstat" \
        && "$(gyre_hires_namelist_value "$namelist" 'sn_cfctl%ptimincr')" == 1 ]] || {
      printf 'REFUSE_RUNSTAT: r4 deck must carry the operator run.stat controls\n' >&2
      return 68
    }
  fi
  if [[ "$deck" == r4_smag ]]; then
    [[ "$(gyre_hires_namelist_value "$namelist" ln_dynldf_lap)" == .false. \
        && "$(gyre_hires_namelist_value "$namelist" ln_dynldf_blp)" == .true. \
        && "$(gyre_hires_namelist_value "$namelist" ln_dynldf_lev)" == .true. \
        && "$(gyre_hires_namelist_value "$namelist" nn_ahm_ijk_t)" == 32 \
        && "$(gyre_hires_namelist_value "$namelist" rn_csmc)" == 3.5 \
        && "$(gyre_hires_namelist_value "$namelist" rn_minfac)" == 1.0 \
        && "$(gyre_hires_namelist_value "$namelist" rn_maxfac)" == 1.0 \
        && -z "$(gyre_hires_namelist_value "$namelist" rn_Uv)" \
        && -z "$(gyre_hires_namelist_value "$namelist" rn_Lv)" ]] || {
      printf 'REFUSE_SMAG: r4_smag must be the exact mode-32 operator block\n' >&2
      return 68
    }
  fi
}

gyre_hires_write_resolved_namelist() {
  local namelist=$1 output=$2
  grep -nE 'nn_GYRE|rn_Dt|nn_it000|nn_itend|nn_stock|nn_write|rn_Ld|rn_Lv|rn_Ud|rn_Uv|ln_dynldf_|nn_ahm_ijk_t|rn_csmc|rn_minfac|rn_maxfac|ln_meshmask|l_runstat|ptimincr' \
    "$namelist" | grep -vE '^\S*:\s*!' >"$output"
}
