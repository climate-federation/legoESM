# Shared deck resolver for the SWG stage and month acquisition tools.

swg_select_deck() {
  local mode=$1 deck=$2 here=$3
  MAKENEMO_SOURCE_CASE=SWG
  SOURCE_PATCH=$here/usrdef_zgr_domsiz.patch
  SOURCE_PATCH_SHA256=c3af0880a1beb44e7ab40027aa8012d11f1e43fc97127d6cc27d50a44fcab08b
  case "$deck:$mode" in
    shipped:stage)
      SOURCE_CASE_NAME=SWG
      RUN_NAME=SWG_OMIP_L1_P3
      DECK_PATCH=$here/namelist_cfg_kt10.patch
      ;;
    shipped:month)
      SOURCE_CASE_NAME=SWG
      RUN_NAME=SWG_OMIP_L1_P3D
      DECK_PATCH=$here/namelist_cfg_30d.patch
      ;;
    maddison:stage)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG_OMIP_L1_P3
      DECK_PATCH=$here/namelist_cfg_maddison_kt10.patch
      ;;
    maddison:month)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG_OMIP_L1_P3D
      DECK_PATCH=$here/namelist_cfg_maddison_30d.patch
      ;;
    dg50:stage)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG50_OMIP_L1_P3
      DECK_PATCH=$here/namelist_cfg_dg50_kt10.patch
      ;;
    dg50:month)
      SOURCE_CASE_NAME=SWG_DG
      RUN_NAME=SWG_DG50_OMIP_L1_P3D
      DECK_PATCH=$here/namelist_cfg_dg50_30d.patch
      ;;
    *)
      printf 'REFUSE_DECK: expected shipped, maddison, or dg50\n' >&2
      return 64
      ;;
  esac
}

swg_verify_makenemo_source_case() {
  local nemo_root=$1 source_case=$2
  local registry=$nemo_root/tests/demo_cfgs.txt
  [[ "$source_case" == SWG ]] || {
    printf 'REFUSE_MAKENEMO_SOURCE: -a must name shipped test case SWG, not %s\n' \
      "$source_case" >&2
    return 68
  }
  [[ -d "$nemo_root/tests/SWG" && -f "$registry" ]] || {
    printf 'REFUSE_MAKENEMO_SOURCE: shipped SWG or demo_cfgs.txt is missing\n' >&2
    return 68
  }
  grep -qE '^SWG[[:space:]]' "$registry" || {
    printf 'REFUSE_MAKENEMO_SOURCE: SWG is not registered in %s\n' \
      "$registry" >&2
    return 68
  }
}

swg_verify_target_cpp_keys() {
  local cfg=$1 run_name=$2
  local cpp=$cfg/cpp_${run_name}.fcm
  [[ -f "$cpp" ]] || {
    printf 'REFUSE_TARGET_KEYS: missing %s\n' "$cpp" >&2
    return 70
  }
  local keys
  keys=$(sed -E 's/.*fppkeys *//' "$cpp" | xargs)
  [[ "$keys" == 'key_qco key_vco_3d key_RK3' ]] || {
    printf 'REFUSE_TARGET_KEYS: found "%s" in %s\n' "$keys" "$cpp" >&2
    return 70
  }
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
  local namelist=$1 python=$2
  local dt_raw clock nint exact ratio itend stock write
  dt_raw=$(swg_namelist_value "$namelist" rn_Dt)
  clock=$("$python" -I -c '
import math
import sys

try:
    dt = float(sys.argv[1].replace("D", "e").replace("d", "e"))
except ValueError:
    raise SystemExit(2)
if not math.isfinite(dt) or dt <= 0.0:
    raise SystemExit(2)
ratio = 86400.0 / dt
# For this positive quotient, Fortran NINT is floor(x + 0.5).
nint = math.floor(ratio + 0.5)
print(nint, int(ratio == float(nint)), format(ratio, ".17g"))
' "$dt_raw") || {
    printf 'REFUSE_DAILY_CLOCK: rn_Dt=%s is not positive finite real\n' \
      "$dt_raw" >&2
    return 68
  }
  read -r nint exact ratio <<<"$clock"
  [[ "$exact" == 1 ]] || {
    printf 'REFUSE_DAILY_CLOCK: rn_Dt=%s gives 86400/rn_Dt=%s; NINT cadence=%s; exact integer required\n' \
      "$dt_raw" "$ratio" "$nint" >&2
    return 68
  }
  STEPS_PER_DAY=$nint
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
