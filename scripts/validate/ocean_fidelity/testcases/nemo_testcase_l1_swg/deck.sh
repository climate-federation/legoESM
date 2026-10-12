# Shared deck resolver for the SWG stage and month acquisition tools.

swg_select_deck() {
  local mode=$1 deck=$2 here=$3
  SOURCE_PATCH=$here/usrdef_zgr_domsiz.patch
  SOURCE_PATCH_SHA256=5021a71143943a34943497f40b72e86a43073ab7ff089646fdb3418cb029ba9e
  case "$deck:$mode" in
    shipped:stage)
      SOURCE_CASE_NAME=SWG
      RUN_NAME=SWG_OMIP_L1_P3
      DECK_PATCH=$here/namelist_cfg_kt10.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round1/oracle_swg_r1
      ;;
    shipped:month)
      SOURCE_CASE_NAME=SWG
      RUN_NAME=SWG_OMIP_L1_P3D
      DECK_PATCH=$here/namelist_cfg_30d.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6/oracle_swg_r6_30d
      ;;
    maddison:stage)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG_OMIP_L1_P3
      DECK_PATCH=$here/namelist_cfg_maddison_kt10.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1
      ;;
    maddison:month)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG_OMIP_L1_P3D
      DECK_PATCH=$here/namelist_cfg_maddison_30d.patch
      DEFAULT_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d
      ;;
    *)
      printf 'REFUSE_DECK: expected shipped or maddison\n' >&2
      return 64
      ;;
  esac
}

swg_namelist_value() {
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

swg_verify_source_case() {
  local nemo_root=$1 deck=$2 scratch=$3
  local shipped=$nemo_root/tests/SWG
  local source_case=$nemo_root/tests/$SOURCE_CASE_NAME
  local cpp=$source_case/cpp_${SOURCE_CASE_NAME}.fcm
  [[ -d "$source_case/MY_SRC" && -f "$cpp" ]] || {
    printf 'REFUSE_SOURCE_CASE: incomplete %s\n' "$source_case" >&2
    return 68
  }
  local keys
  keys=$(sed -E 's/.*fppkeys *//' "$cpp" | xargs)
  if [[ "$deck" == shipped ]]; then
    [[ "$keys" == 'key_xios key_qco key_vco_3d key_RK3' ]] || {
      printf 'REFUSE_SHIPPED_KEYS: found "%s"\n' "$keys" >&2
      return 68
    }
    return 0
  fi
  [[ "$keys" == 'key_qco key_vco_3d key_RK3' ]] || {
    printf 'REFUSE_SWG_DG_KEYS: found "%s"\n' "$keys" >&2
    return 68
  }
  [[ "$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')" == "$SOURCE_PATCH_SHA256" ]] || {
    printf 'REFUSE_SOURCE_PATCH_HASH: %s\n' "$SOURCE_PATCH" >&2
    return 68
  }
  mkdir -p "$scratch/expected_my_src"
  cp -r "$shipped/MY_SRC/." "$scratch/expected_my_src/"
  patch -s --fuzz=0 "$scratch/expected_my_src/usrdef_zgr.F90" <"$SOURCE_PATCH"
  diff -qr "$scratch/expected_my_src" "$source_case/MY_SRC" >/dev/null || {
    printf 'REFUSE_SWG_DG_SOURCE: MY_SRC is not shipped SWG plus %s\n' \
      "$SOURCE_PATCH_SHA256" >&2
    return 68
  }
  local compiled=$source_case/BLD/ppsrc/nemo/usrdef_zgr.f90
  local binary=$source_case/BLD/bin/nemo.exe
  [[ -f "$compiled" && -x "$binary" && "$binary" -nt "$compiled" ]] || {
    printf 'REFUSE_SWG_DG_STALE_BUILD: rebuild %s after its compiled source\n' \
      "$source_case" >&2
    return 68
  }
  [[ "$(grep -c 'rn_domsiz + 10000._wp.*SWG_DG' "$compiled")" -eq 2 ]] || {
    printf 'REFUSE_SWG_DG_COMPILED_SOURCE: patched limits absent from %s\n' \
      "$compiled" >&2
    return 68
  }
}

swg_require_daily_clock() {
  local namelist=$1
  local dt_raw dt_int itend stock write
  dt_raw=$(swg_namelist_value "$namelist" rn_Dt)
  dt_int=${dt_raw%%.*}
  [[ "$dt_int" =~ ^[0-9]+$ && "$dt_int" -gt 0 \
        && $((86400 % dt_int)) -eq 0 ]] || {
    printf 'REFUSE_DAILY_CLOCK: rn_Dt=%s does not divide 86400 s\n' "$dt_raw" >&2
    return 68
  }
  STEPS_PER_DAY=$((86400 / dt_int))
  itend=$(swg_namelist_value "$namelist" nn_itend)
  stock=$(swg_namelist_value "$namelist" nn_stock)
  write=$(swg_namelist_value "$namelist" nn_write)
  [[ "$itend" == $((30 * STEPS_PER_DAY)) ]] || {
    printf 'REFUSE_30D_LENGTH: nn_itend=%s expected %s\n' \
      "$itend" "$((30 * STEPS_PER_DAY))" >&2
    return 68
  }
  [[ "$stock" == "$STEPS_PER_DAY" && "$write" == "$STEPS_PER_DAY" ]] || {
    printf 'REFUSE_DAILY_CADENCE: stock=%s write=%s expected %s\n' \
      "$stock" "$write" "$STEPS_PER_DAY" >&2
    return 68
  }
}

swg_write_resolved_namelist() {
  local namelist=$1 output=$2
  grep -nE 'nn_SWG|rn_domsiz|rn_tau|rn_modified_grav|rn_rfr|rn_beta|rn_f0|rn_Uv|rn_Lv|rn_Dt|nn_itend|nn_write|nn_stock|ln_meshmask|l_runstat|ptimincr' \
    "$namelist" | grep -vE '^\S*:\s*!' >"$output"
}
