#!/usr/bin/env bash
# SI3 ladder Rung O2 ACQUISITION (DECISION 159) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  bash run.sh [--preflight-only] [json_out]
#   acquire (operator):   bash run.sh --run [evidence_dir]
#   re-admit existing:    bash run.sh --admit-existing [evidence_dir]
#
# The agent that wrote this tool ran only --preflight-only; it never invoked
# makenemo or mpirun.
set -Eeuo pipefail

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG_SOURCE=ORCA2_SAS_ICE
readonly TARGET_REF_CFG=${TARGET_REF_CFG:-ORCA2_SAS_ICE_O2_REF}
readonly TARGET_REC_CFG=${TARGET_REC_CFG:-ORCA2_SAS_ICE_O2_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7}
EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o2_orca2_sas_ice}
REFUSED_DIR=${REFUSED_DIR:-$EVIDENCE_ROOT/refused}

readonly ORCA2_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz
readonly ORCA2_BUNDLE_SHA256=5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4
readonly SAS_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz
readonly SAS_BUNDLE_SHA256=ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf

readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-8388608}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-1800}

readonly PY=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly NEMO_INC=/home/dbalwada/miniconda3/envs/nemo-build/include
readonly FPP_KEYS='-Dkey_nosignedzero -Dkey_si3 -Dkey_linssh -Dkey_vco_1d3d -Dkey_RK3'

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

readonly WRITER_F90=$here/o2_si3_state_writer.F90
readonly PRA_PATCH=$here/icedyn_adv_pra_o2_public.patch
readonly IST_PATCH=$here/iceistate_o2_writer.patch
readonly STP_PATCH=$here/icestp_o2_writer.patch
readonly STUBS_F90=$here/o2_si3_fixture_stubs.F90
readonly FIXTURE_F90=$here/o2_si3_layout_fixture.F90
readonly NML_CFG=$here/namelist_cfg
readonly NML_ICE_CFG=$here/namelist_ice_cfg
readonly LOG_EXCERPT=$here/ocean.output.excerpt
readonly GATE_PY=$here/../nemo_si3_o2_orca2_sas_gate.py

readonly SHIPPED_PRA=$NEMO_ROOT/src/ICE/icedyn_adv_pra.F90
readonly SHIPPED_IST=$NEMO_ROOT/src/ICE/iceistate.F90
readonly SHIPPED_STP=$NEMO_ROOT/src/ICE/icestp.F90
readonly SHIPPED_ITD=$NEMO_ROOT/src/ICE/iceitd.F90
readonly SHIPPED_DOM=$NEMO_ROOT/src/OCE/DOM/domain.F90
readonly SHIPPED_SAS_GCM=$NEMO_ROOT/src/SAS/nemogcm.F90
readonly SHIPPED_NML_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ref
readonly SHIPPED_NML_ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly SHIPPED_EXPREF_NML=$NEMO_ROOT/cfgs/$REF_CFG_SOURCE/EXPREF/namelist_cfg
readonly SHIPPED_EXPREF_ICE=$NEMO_ROOT/cfgs/$REF_CFG_SOURCE/EXPREF/namelist_ice_cfg

mode=${1:---preflight-only}
arg2=${2:-}
preflight_json_out=""
case "$mode" in
  --preflight-only)
    preflight_json_out="$arg2"
    ;;
  --run|--admit-existing)
    if [[ -n "$arg2" ]]; then
      EVIDENCE="$arg2"
    fi
    ;;
  *)
    printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing] [json_out|evidence_dir]\n' "$0" >&2
    exit 64
    ;;
esac

ACTIVE_RUN_DIR=""
ACTIVE_SCRATCH_DIR=""
CREATED_REF_CFG=0
CREATED_REC_CFG=0
CREATED_EVIDENCE=0
RUN_COMPLETED=0
TARGET_REF_ROOT="$NEMO_ROOT/cfgs/$TARGET_REF_CFG"
TARGET_REC_ROOT="$NEMO_ROOT/cfgs/$TARGET_REC_CFG"

cleanup_on_exit() {
  local rc=$?
  if [[ -n "$ACTIVE_SCRATCH_DIR" && -d "$ACTIVE_SCRATCH_DIR" ]]; then
    rm -rf -- "$ACTIVE_SCRATCH_DIR"
  fi
  if [[ "$rc" -ne 0 ]]; then
    if [[ "$CREATED_REF_CFG" -eq 1 && "$RUN_COMPLETED" -eq 0 && -d "$TARGET_REF_ROOT" ]]; then
      rm -rf -- "$TARGET_REF_ROOT"
      printf 'REFUSE_CLEANUP: removed incomplete target cfg %s\n' "$TARGET_REF_ROOT" >&2
    fi
    if [[ "$CREATED_REC_CFG" -eq 1 && "$RUN_COMPLETED" -eq 0 && -d "$TARGET_REC_ROOT" ]]; then
      rm -rf -- "$TARGET_REC_ROOT"
      printf 'REFUSE_CLEANUP: removed incomplete target cfg %s\n' "$TARGET_REC_ROOT" >&2
    fi
    if [[ -n "$ACTIVE_RUN_DIR" && -d "$ACTIVE_RUN_DIR" ]]; then
      mkdir -p "$REFUSED_DIR"
      local ts
      ts=$(date -u +%Y%m%dT%H%M%SZ)
      if [[ "$RUN_COMPLETED" -eq 1 || -f "$ACTIVE_RUN_DIR/RUN_COMPLETE" ]]; then
        local dest="$REFUSED_DIR/$(basename "$ACTIVE_RUN_DIR")_admission_refused_${ts}_$$"
        mv "$ACTIVE_RUN_DIR" "$dest"
        printf 'REFUSE_CLEANUP: retained completed run in %s\n' "$dest" >&2
      elif [[ "$CREATED_EVIDENCE" -eq 1 ]]; then
        local dest="$REFUSED_DIR/$(basename "$ACTIVE_RUN_DIR")_incomplete_${ts}_rc${rc}_$$"
        mv "$ACTIVE_RUN_DIR" "$dest"
        printf 'REFUSE_CLEANUP: moved incomplete run %s -> %s\n' "$ACTIVE_RUN_DIR" "$dest" >&2
      fi
    fi
  fi
  exit "$rc"
}
trap cleanup_on_exit EXIT

check_free_ram_gb() {
  local floor_gb=$1
  local avail_gb
  avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [[ -z "$avail_gb" || "$avail_gb" -lt "$floor_gb" ]]; then
    printf 'REFUSE: available RAM %s GB is below floor %s GB\n' "${avail_gb:-unknown}" "$floor_gb" >&2
    return 1
  fi
  return 0
}

check_free_disk_kb() {
  local target_path=$1
  local floor_kb=$2
  local probe=$target_path
  while [[ ! -e "$probe" ]]; do
    probe=$(dirname "$probe")
  done
  local avail_kb
  avail_kb=$(df -Pk "$probe" | awk 'NR==2 {print $4}')
  if [[ -z "$avail_kb" || "$avail_kb" -lt "$floor_kb" ]]; then
    printf 'REFUSE: available disk %s KB on %s (for %s) is below floor %s KB\n' \
      "${avail_kb:-unknown}" "$probe" "$target_path" "$floor_kb" >&2
    return 1
  fi
  return 0
}

check_ref_cfg_in_registry() {
  local cfg_name=$1
  local ref_file=$2
  grep -E "^[[:space:]]*${cfg_name}([[:space:]]|$)" "$ref_file" >/dev/null
}

check_target_absent() {
  local path=$1
  if [[ -e "$path" ]]; then
    printf 'REFUSE: target path already exists: %s\n' "$path" >&2
    return 1
  fi
  return 0
}

check_ascii_only() {
  local file=$1
  if LC_ALL=C grep -n '[^ -~	]' "$file" >/dev/null; then
    printf 'REFUSE: non-ASCII characters found in %s\n' "$file" >&2
    return 1
  fi
  return 0
}

check_intrinsic_use() {
  local file=$1
  if grep -E '^[[:space:]]*USE[[:space:]]+iso_fortran_env' "$file" >/dev/null; then
    printf 'REFUSE: %s uses bare USE iso_fortran_env without USE, INTRINSIC\n' "$file" >&2
    return 1
  fi
  grep -E '^[[:space:]]*USE,[[:space:]]*INTRINSIC[[:space:]]*::[[:space:]]*iso_fortran_env' "$file" >/dev/null
}

verify_work_cfgs_guard() {
  local before_file=$1
  local after_file=$2
  local s_before=$3
  local s_after=$4
  grep -v -E "^(${TARGET_REF_CFG}|${TARGET_REC_CFG})([[:space:]]|$)" "$before_file" > "$s_before" || true
  grep -v -E "^(${TARGET_REF_CFG}|${TARGET_REC_CFG})([[:space:]]|$)" "$after_file" > "$s_after" || true
  if ! cmp -s "$s_before" "$s_after"; then
    printf 'REFUSE: STATUS PLANT-FIRED: cfgs/work_cfgs.txt changed outside %s/%s\n' \
      "$TARGET_REF_CFG" "$TARGET_REC_CFG" >&2
    return 1
  fi
  return 0
}

render_rst_namelists() {
  local src_oce=$1
  local src_ice=$2
  local dst_oce=$3
  local dst_ice=$4
  cp "$src_oce" "$dst_oce"
  cp "$src_ice" "$dst_ice"
  sed -E -i 's/^( *nn_itend *= *).*/\11/' "$dst_oce"
  # cn_icerst_in is declared in &nampar (src/ICE/icestp.F90:341-342);
  # nn_iceini_file is declared in &namini (src/ICE/iceistate.F90:465-470).
  sed -E -i 's|^(&nampar.*)|\1\n   cn_icerst_in     = "ORCA2_SAS_00000001_restart_ice"|' "$dst_ice"
  sed -E -i 's|^(&namini.*)|\1\n   nn_iceini_file   = 2|' "$dst_ice"
}

verify_namelists_and_stems() {
  local nml_cfg=$1
  local nml_ice=$2
  local toc_file=$3
  local plant=${4:-none}
  local deck_kind=${5:-2step}
  "$PY" - "$nml_cfg" "$nml_ice" "$SHIPPED_NML_REF" "$SHIPPED_NML_ICE_REF" \
    "$SHIPPED_EXPREF_NML" "$SHIPPED_EXPREF_ICE" "$toc_file" "$plant" "$deck_kind" <<'PY'
import pathlib
import re
import sys

(
    nml_cfg_p,
    nml_ice_p,
    ref_oce_p,
    ref_ice_p,
    exp_oce_p,
    exp_ice_p,
    toc_p,
    plant,
    deck_kind,
) = sys.argv[1:]

nml_cfg = pathlib.Path(nml_cfg_p).read_text()
nml_ice = pathlib.Path(nml_ice_p).read_text()
ref_oce = pathlib.Path(ref_oce_p).read_text()
ref_ice = pathlib.Path(ref_ice_p).read_text()
exp_oce = pathlib.Path(exp_oce_p).read_text()
exp_ice = pathlib.Path(exp_ice_p).read_text()
toc_lines = pathlib.Path(toc_p).read_text().splitlines()
archive_basenames = {pathlib.PurePosixPath(line.strip()).name for line in toc_lines if line.strip()}

if plant == "unknown-group":
    nml_cfg = nml_cfg + "\n&namrun_typo\n/\n"
elif plant == "wrong-group-var":
    nml_ice = re.sub(r"(&namini)", '&namini\n   cn_icerst_in = "ORCA2_SAS_00000001_restart_ice"', nml_ice, count=1)
elif plant == "missing-stem":
    nml_cfg = nml_cfg.replace("u_10.15JUNE2009_fill", "missing_forcing_stem")
    exp_oce = exp_oce.replace("u_10.15JUNE2009_fill", "missing_forcing_stem")
elif plant == "bad-fsbc-multiple":
    nml_cfg = re.sub(r"(nn_fsbc\s*=\s*)\d+", r"\g<1>2", nml_cfg)
    nml_cfg = re.sub(r"(nn_itend\s*=\s*)\d+", r"\g<1>3", nml_cfg)
elif plant == "unapproved-deviation":
    nml_ice = re.sub(r"(&nampar)", "&nampar\n   rn_amax_n = 0.5", nml_ice, count=1)

def parse_grouped_assignments(text: str) -> tuple[set[str], dict[str, dict[str, str]], dict[str, str]]:
    group_names: set[str] = set()
    by_group: dict[str, dict[str, str]] = {}
    flat: dict[str, str] = {}
    current_group: str | None = None
    for raw_line in text.splitlines():
        s = raw_line.strip()
        if not s or s.startswith("!"):
            continue
        code = s.split("!")[0].strip()
        if not code:
            continue
        m_grp = re.match(r"^&([A-Za-z0-9_]+)(.*)$", code)
        if m_grp:
            current_group = m_grp.group(1).lower()
            group_names.add(current_group)
            by_group.setdefault(current_group, {})
            rest = m_grp.group(2).strip()
            if rest == "/" or rest.endswith("/"):
                current_group = None
            continue
        if code == "/" or code.startswith("/"):
            current_group = None
            continue
        if current_group is not None and "=" in code:
            k, v = code.split("=", 1)
            k_clean = re.sub(r"\(.*\)$", "", k.strip()).strip()
            v_clean = v.strip().rstrip("/").strip()
            by_group[current_group][k_clean] = v_clean
            flat[k_clean] = v_clean
            if code.endswith("/"):
                current_group = None
    return group_names, by_group, flat

oce_groups, oce_by_group, cfg_oce_vars = parse_grouped_assignments(nml_cfg)
ice_groups, ice_by_group, cfg_ice_vars = parse_grouped_assignments(nml_ice)
ref_oce_groups, ref_oce_by_group, _ = parse_grouped_assignments(ref_oce)
ref_ice_groups, ref_ice_by_group, _ = parse_grouped_assignments(ref_ice)
_, _, exp_oce_vars = parse_grouped_assignments(exp_oce)
_, _, exp_ice_vars = parse_grouped_assignments(exp_ice)

unknown_oce = sorted(oce_groups - ref_oce_groups)
unknown_ice = sorted(ice_groups - ref_ice_groups)
if unknown_oce or unknown_ice:
    raise SystemExit(f"STATUS PLANT-FIRED: unknown namelist groups oce={unknown_oce} ice={unknown_ice}")

for grp, gvars in oce_by_group.items():
    ref_vars_lower = {k.lower() for k in ref_oce_by_group.get(grp, {})}
    for var_name in gvars:
        if var_name.lower() not in ref_vars_lower:
            owner_groups = [
                g for g, rv in ref_oce_by_group.items()
                if var_name.lower() in {rk.lower() for rk in rv}
            ]
            raise SystemExit(
                f"STATUS PLANT-FIRED: ocean variable {var_name} in group &{grp} "
                f"is not declared in &{grp} (belongs to {owner_groups})"
            )

for grp, gvars in ice_by_group.items():
    ref_vars_lower = {k.lower() for k in ref_ice_by_group.get(grp, {})}
    for var_name in gvars:
        if var_name.lower() not in ref_vars_lower:
            owner_groups = [
                g for g, rv in ref_ice_by_group.items()
                if var_name.lower() in {rk.lower() for rk in rv}
            ]
            raise SystemExit(
                f"STATUS PLANT-FIRED: ice variable {var_name} in group &{grp} "
                f"is not declared in &{grp} (belongs to {owner_groups})"
            )

nn_fsbc = int(cfg_oce_vars.get("nn_fsbc", "1"))
nn_itend = int(cfg_oce_vars.get("nn_itend", "0"))
nn_stock = int(cfg_oce_vars.get("nn_stock", "0"))
if nn_fsbc <= 0 or nn_itend <= 0 or nn_stock <= 0 or (nn_itend % nn_fsbc != 0) or (nn_stock % nn_fsbc != 0):
    raise SystemExit(
        f"STATUS PLANT-FIRED: nn_itend={nn_itend} or nn_stock={nn_stock} not a positive multiple of nn_fsbc={nn_fsbc}"
    )

oce_diff_keys = {
    k for k in (set(cfg_oce_vars) | set(exp_oce_vars))
    if cfg_oce_vars.get(k) != exp_oce_vars.get(k)
}
ice_diff_keys = {
    k for k in (set(cfg_ice_vars) | set(exp_ice_vars))
    if cfg_ice_vars.get(k) != exp_ice_vars.get(k)
}
allowed_oce_diff = {"nn_itend", "nn_stock", "nn_write"}
if deck_kind == "2step":
    allowed_ice_diff = {"ln_icedyn", "ln_icethd"}
elif deck_kind == "rst_1step":
    allowed_ice_diff = {"ln_icedyn", "ln_icethd", "cn_icerst_in", "nn_iceini_file"}
else:
    raise SystemExit(f"STATUS PLANT-FIRED: unknown deck_kind={deck_kind!r}")

if oce_diff_keys != allowed_oce_diff or ice_diff_keys != allowed_ice_diff:
    raise SystemExit(
        f"STATUS PLANT-FIRED: deck deviation mismatch ({deck_kind}) "
        f"oce_diff={sorted(oce_diff_keys)} ice_diff={sorted(ice_diff_keys)}"
    )
if cfg_ice_vars.get("ln_icedyn", "").lower() != ".false." or cfg_ice_vars.get("ln_icethd", "").lower() != ".false.":
    raise SystemExit("STATUS PLANT-FIRED: O2 requires ln_icedyn=.false. and ln_icethd=.false.")

if deck_kind == "rst_1step":
    if nn_itend != 1:
        raise SystemExit(f"STATUS PLANT-FIRED: rst_1step requires nn_itend=1, got {nn_itend}")
    if cfg_ice_vars.get("nn_iceini_file") != "2":
        raise SystemExit("STATUS PLANT-FIRED: rst_1step requires nn_iceini_file=2 in &namini")
    if cfg_ice_vars.get("cn_icerst_in", "").strip("\"'") != "ORCA2_SAS_00000001_restart_ice":
        raise SystemExit("STATUS PLANT-FIRED: rst_1step requires cn_icerst_in='ORCA2_SAS_00000001_restart_ice' in &nampar")

domcfg_stem = cfg_oce_vars["cn_domcfg"].strip("\"'")
if f"{domcfg_stem}.nc" not in archive_basenames:
    raise SystemExit(f"STATUS PLANT-FIRED: cn_domcfg {domcfg_stem}.nc missing from bundles")

for k, v in list(cfg_oce_vars.items()) + list(cfg_ice_vars.items()):
    if k.startswith("sn_"):
        parts = [p.strip().strip("\"'") for p in v.split(",")]
        stem = parts[0]
        if stem and stem != "NOT USED":
            if f"{stem}.nc" not in archive_basenames:
                raise SystemExit(f"STATUS PLANT-FIRED: {k} stem {stem}.nc missing from bundles")
        if len(parts) >= 7:
            weights = parts[6]
            if weights and f"{weights}.nc" not in archive_basenames:
                raise SystemExit(f"STATUS PLANT-FIRED: {k} weights {weights}.nc missing from bundles")
PY
}

verify_excerpt_provenance_and_deck_independence() {
  local excerpt_file=$1
  local scratch=$2
  local plant=${3:-none}
  "$PY" - "$excerpt_file" "$SHIPPED_SAS_GCM" "$SHIPPED_DOM" "$SHIPPED_STP" "$SHIPPED_ITD" "$scratch" "$plant" <<'PY'
import hashlib
import pathlib
import re
import sys

excerpt_p, gcm_p, dom_p, stp_p, itd_p, scratch_p, plant = sys.argv[1:]
lines = pathlib.Path(excerpt_p).read_text(encoding="utf-8").splitlines()
headers = [l for l in lines if l.startswith("#")]
body = [l for l in lines if not l.startswith("#")]

src_match = next((re.search(r"^#\s*Source path:\s*(.+)$", l) for l in headers if "Source path:" in l), None)
sha_match = next((re.search(r"^#\s*Complete source SHA-256:\s*([0-9a-f]{64})\s*$", l) for l in headers if "SHA-256:" in l), None)
rng_match = next((re.search(r"^#\s*Source lines:\s*(.+)$", l) for l in headers if "Source lines:" in l), None)

if not src_match or not sha_match or not rng_match:
    raise SystemExit("STATUS PLANT-FIRED: ocean.output.excerpt missing provenance header comments")

src_path = pathlib.Path(src_match.group(1).strip())
expected_sha = sha_match.group(1).strip()
if plant == "excerpt-sha-mismatch":
    expected_sha = "0" * 64

if src_path.is_file():
    actual_sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"STATUS PLANT-FIRED: excerpt source SHA-256 {actual_sha} != {expected_sha}"
        )
    line_nums: list[int] = []
    for part in rng_match.group(1).rstrip(".").split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            line_nums.extend(range(int(a), int(b) + 1))
        else:
            line_nums.append(int(part))
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    extracted = [src_lines[n - 1] for n in line_nums]
    if body != extracted:
        raise SystemExit("STATUS PLANT-FIRED: ocean.output.excerpt body does not match source lines verbatim")

gcm_txt = pathlib.Path(gcm_p).read_text(encoding="utf-8")
dom_txt = pathlib.Path(dom_p).read_text(encoding="utf-8")
stp_txt = pathlib.Path(stp_p).read_text(encoding="utf-8")
itd_txt = pathlib.Path(itd_p).read_text(encoding="utf-8")

required_source_anchors = [
    (gcm_txt, "Ocean General Circulation Model", "nemogcm.F90"),
    (gcm_txt, "AAAAAAAA", "nemogcm.F90"),
    (dom_txt, "Ni0glo", "domain.F90"),
    (dom_txt, "Nj0glo", "domain.F90"),
    (stp_txt, "CALL par_init", "icestp.F90"),
    (stp_txt, "CALL ice_itd_init", "icestp.F90"),
    (stp_txt, "number of ice  categories                           jpl       =", "icestp.F90"),
    (stp_txt, "number of ice  layers                               nlay_i    =", "icestp.F90"),
    (stp_txt, "number of snow layers                               nlay_s    =", "icestp.F90"),
    (stp_txt, "Ice dynamics       (T) or not (F)                   ln_icedyn =", "icestp.F90"),
    (stp_txt, "Ice thermodynamics (T) or not (F)                   ln_icethd =", "icestp.F90"),
    (itd_txt, "ln_cat_hfn =", "iceitd.F90"),
    (itd_txt, "rn_himean  =", "iceitd.F90"),
]
for txt, token, label in required_source_anchors:
    if token not in txt:
        raise SystemExit(f"STATUS PLANT-FIRED: missing source anchor {token!r} in {label}")

o2_lines = []
for l in body:
    l2 = re.sub(r"(ln_icedyn\s*=\s*)T", r"\g<1>F", l)
    l2 = re.sub(r"(ln_icethd\s*=\s*)T", r"\g<1>F", l2)
    o2_lines.append(l2)
(pathlib.Path(scratch_p) / "ocean.output.o2_rendered").write_text("\n".join(o2_lines) + "\n", encoding="utf-8")
PY
}

build_orca2_sas_ice_syntax_mods() {
  local scratch=$1
  local mod_dir="$scratch/orca2_sas_ice_mods"
  local src_dir="$scratch/orca2_sas_ice_src"
  mkdir -p "$mod_dir" "$src_dir"

  # Link ORCA2_SAS_ICE's own source tree (IOIPSL, PPR, OCE, ICE, NST, SAS) -- never ORCA2_OMIP_L4.
  for f in $(find "$NEMO_ROOT/ext/IOIPSL/src" "$NEMO_ROOT/ext/PPR/src" \
                  "$NEMO_ROOT/src/OCE" "$NEMO_ROOT/src/ICE" \
                  "$NEMO_ROOT/src/NST" "$NEMO_ROOT/src/SAS" \
                  -name '*.F90' -o -name '*.f90' -o -name '*.h90'); do
    ln -sf "$f" "$src_dir/$(basename "$f")"
  done
  rm -f "$src_dir/icedyn_adv_pra.F90" "$src_dir/iceistate.F90" "$src_dir/icestp.F90" "$src_dir/o2_si3_state_writer.F90"
  cp "$SHIPPED_PRA" "$src_dir/icedyn_adv_pra.F90"
  cp "$SHIPPED_IST" "$src_dir/iceistate.F90"
  cp "$SHIPPED_STP" "$src_dir/icestp.F90"
  cp "$WRITER_F90" "$src_dir/o2_si3_state_writer.F90"
  patch -s --fuzz=0 "$src_dir/icedyn_adv_pra.F90" < "$PRA_PATCH"
  patch -s --fuzz=0 "$src_dir/iceistate.F90" < "$IST_PATCH"
  patch -s --fuzz=0 "$src_dir/icestp.F90" < "$STP_PATCH"

  "$PY" - "$src_dir" "$mod_dir" "$FC" "$NEMO_INC" "$FPP_KEYS" <<'PY'
import pathlib
import re
import subprocess
import sys

src_dir = pathlib.Path(sys.argv[1])
mod_dir = pathlib.Path(sys.argv[2])
fc = sys.argv[3]
nemo_inc = sys.argv[4]
fpp_keys = sys.argv[5].split()

mod_to_file: dict[str, pathlib.Path] = {}
file_uses: dict[pathlib.Path, set[str]] = {}
for p in list(src_dir.glob("*.F90")) + list(src_dir.glob("*.f90")):
    txt = p.read_text(errors="ignore")
    for m in re.finditer(r"^\s*MODULE\s+([A-Za-z0-9_]+)", txt, flags=re.M | re.I):
        name = m.group(1).lower()
        if name != "procedure":
            mod_to_file[name] = p
    uses = {
        m.group(1).lower()
        for m in re.finditer(r"^\s*USE(?:\s*,\s*INTRINSIC\s*::|\s+)([A-Za-z0-9_]+)", txt, flags=re.M | re.I)
    }
    file_uses[p] = uses

needed: set[pathlib.Path] = set()
stack = [
    src_dir / "icedyn_adv_pra.F90",
    src_dir / "o2_si3_state_writer.F90",
    src_dir / "iceistate.F90",
    src_dir / "icestp.F90",
]
while stack:
    curr = stack.pop()
    if curr in needed:
        continue
    needed.add(curr)
    for u in file_uses.get(curr, ()):
        if u in mod_to_file and mod_to_file[u] not in needed:
            stack.append(mod_to_file[u])

ordered: list[pathlib.Path] = []
visited: set[pathlib.Path] = set()
def visit(node: pathlib.Path) -> None:
    if node in visited:
        return
    visited.add(node)
    for u in sorted(file_uses.get(node, ())):
        dep = mod_to_file.get(u)
        if dep in needed and dep != node:
            visit(dep)
    ordered.append(node)

for n in sorted(needed):
    visit(n)

for f in ordered:
    pp = mod_dir / (f.stem + ".f90")
    subprocess.run(
        ["cpp", *fpp_keys, "-P", "-traditional", "-I", str(src_dir), "-I", nemo_inc, str(f), "-o", str(pp)],
        check=True,
    )
    subprocess.run(
        [
            fc,
            "-fsyntax-only",
            "-fdefault-real-8",
            "-fcray-pointer",
            "-ffree-line-length-none",
            "-fallow-argument-mismatch",
            "-I",
            str(src_dir),
            "-I",
            str(mod_dir),
            "-I",
            nemo_inc,
            "-J",
            str(mod_dir),
            str(pp),
        ],
        check=True,
    )
PY
}

run_preflight() {
  local target_check_mode=${1:-require_absent}
  mkdir -p "$EVIDENCE_ROOT"
  ACTIVE_SCRATCH_DIR=$(mktemp -d "$EVIDENCE_ROOT/.o2-scratch.XXXXXX")
  local scratch=$ACTIVE_SCRATCH_DIR
  export TMPDIR=$scratch

  for f in "$WRITER_F90" "$PRA_PATCH" "$IST_PATCH" "$STP_PATCH" "$STUBS_F90" "$FIXTURE_F90" \
           "$NML_CFG" "$NML_ICE_CFG" "$LOG_EXCERPT" "$GATE_PY" \
           "$SHIPPED_PRA" "$SHIPPED_IST" "$SHIPPED_STP" "$SHIPPED_ITD" "$SHIPPED_DOM" "$SHIPPED_SAS_GCM" \
           "$SHIPPED_NML_REF" "$SHIPPED_NML_ICE_REF" \
           "$SHIPPED_EXPREF_NML" "$SHIPPED_EXPREF_ICE" \
           "$ORCA2_BUNDLE" "$SAS_BUNDLE" "$NEMO_ROOT/makenemo" \
           "$NEMO_ROOT/cfgs/ref_cfgs.txt" "$NEMO_ROOT/cfgs/work_cfgs.txt"; do
    [[ -f "$f" ]] || { printf 'REFUSE: missing required input %s\n' "$f" >&2; exit 65; }
  done

  local actual_orca2_sha actual_sas_sha
  actual_orca2_sha=$(sha256sum "$ORCA2_BUNDLE" | awk '{print $1}')
  actual_sas_sha=$(sha256sum "$SAS_BUNDLE" | awk '{print $1}')
  [[ "$actual_orca2_sha" == "$ORCA2_BUNDLE_SHA256" ]] || {
    printf 'REFUSE: ORCA2 bundle SHA-256 %s != %s\n' "$actual_orca2_sha" "$ORCA2_BUNDLE_SHA256" >&2
    exit 65
  }
  [[ "$actual_sas_sha" == "$SAS_BUNDLE_SHA256" ]] || {
    printf 'REFUSE: SAS bundle SHA-256 %s != %s\n' "$actual_sas_sha" "$SAS_BUNDLE_SHA256" >&2
    exit 65
  }
  tar -ztf "$ORCA2_BUNDLE" > "$scratch/bundles.toc"
  tar -ztf "$SAS_BUNDLE" >> "$scratch/bundles.toc"

  # ---- Checklist Item 1: gfortran syntax proof against ORCA2_SAS_ICE's own modules + plants ----
  if grep -q '^-[^-]' "$PRA_PATCH" "$IST_PATCH" "$STP_PATCH"; then
    printf 'REFUSE: source patches must be strictly additive\n' >&2
    exit 66
  fi
  for f90 in "$WRITER_F90" "$STUBS_F90" "$FIXTURE_F90"; do
    check_intrinsic_use "$f90"
  done
  # Plant 1c: bare USE iso_fortran_env without INTRINSIC
  sed 's/USE, INTRINSIC :: iso_fortran_env/USE iso_fortran_env/' "$WRITER_F90" > "$scratch/bad_intrinsic.F90"
  if check_intrinsic_use "$scratch/bad_intrinsic.F90" 2>/dev/null; then
    printf 'REFUSE: bare USE iso_fortran_env plant did not fire\n' >&2
    exit 72
  fi

  build_orca2_sas_ice_syntax_mods "$scratch"
  local mod_dir="$scratch/orca2_sas_ice_mods"
  local src_dir="$scratch/orca2_sas_ice_src"

  preprocess_nemo() {
    cpp $FPP_KEYS -P -traditional -I "$src_dir" -I "$mod_dir" -I "$NEMO_INC" "$1" -o "$2"
  }

  # Plant 1a: insertion immediately after a continuation line ending in '&'
  awk '{print} /IF\( ln_icedyn .AND. .NOT.ln_c1d \)[[:space:]]*&/ {print "         CALL o2_write_si3_state_snapshot( '\''step'\'', kt, lrst_ice )"}' \
    "$src_dir/icestp.F90" > "$scratch/icestp_amp_plant.F90"
  preprocess_nemo "$scratch/icestp_amp_plant.F90" "$scratch/icestp_amp_plant.f90"
  if "$FC" -fsyntax-only -fdefault-real-8 -fcray-pointer -ffree-line-length-none -I "$src_dir" -I "$mod_dir" -I "$NEMO_INC" -J "$scratch" "$scratch/icestp_amp_plant.f90" 2>/dev/null; then
    printf 'REFUSE: Plant 1a (insertion after continuation &) did not fail gfortran -fsyntax-only\n' >&2
    exit 72
  fi

  # Plant 1b: undeclared identifier in MY_SRC
  sed 's/CALL o2_write_si3_state_snapshot( '\''entry'\'', 0/CALL o2_write_si3_state_snapshot( '\''entry'\'', o2_undeclared_identifier/' "$mod_dir/icestp.f90" > "$scratch/icestp_undef_plant.f90"
  if "$FC" -fsyntax-only -fdefault-real-8 -fcray-pointer -ffree-line-length-none -I "$src_dir" -I "$mod_dir" -I "$NEMO_INC" -J "$scratch" "$scratch/icestp_undef_plant.f90" 2>/dev/null; then
    printf 'REFUSE: Plant 1b (undeclared identifier) did not fail gfortran -fsyntax-only\n' >&2
    exit 72
  fi

  # ---- Checklist Item 2: makenemo -r in cfgs/ref_cfgs.txt + work_cfgs.txt guard + plants ----
  check_ref_cfg_in_registry "$REF_CFG_SOURCE" "$NEMO_ROOT/cfgs/ref_cfgs.txt"
  if check_ref_cfg_in_registry "NONEXISTENT_WORK_CFG_PLANT" "$NEMO_ROOT/cfgs/ref_cfgs.txt" 2>/dev/null; then
    printf 'REFUSE: Plant 2a (non-reference config in ref_cfgs.txt) did not fire\n' >&2
    exit 72
  fi
  cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.before"
  cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.after_ok"
  printf '%s OCE ICE NST SAS\n' "$TARGET_REF_CFG" >> "$scratch/work_cfgs.after_ok"
  verify_work_cfgs_guard "$scratch/work_cfgs.before" "$scratch/work_cfgs.after_ok" \
    "$scratch/wc_b.txt" "$scratch/wc_a.txt"
  cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.after_bad"
  printf 'UNEXPECTED_OTHER_CFG OCE ICE\n' >> "$scratch/work_cfgs.after_bad"
  if verify_work_cfgs_guard "$scratch/work_cfgs.before" "$scratch/work_cfgs.after_bad" \
       "$scratch/wc_b.txt" "$scratch/wc_a.txt" 2>/dev/null; then
    printf 'REFUSE: Plant 2b (work_cfgs.txt unexpected mutation) did not fire\n' >&2
    exit 72
  fi

  # ---- Checklist Item 3: namelist groups, variable-group membership, nn_fsbc multiples, sn_* stems, rendered rst_1step deck + plants ----
  verify_namelists_and_stems "$NML_CFG" "$NML_ICE_CFG" "$scratch/bundles.toc" "none" "2step"
  render_rst_namelists "$NML_CFG" "$NML_ICE_CFG" "$scratch/rst_namelist_cfg" "$scratch/rst_namelist_ice_cfg"
  verify_namelists_and_stems "$scratch/rst_namelist_cfg" "$scratch/rst_namelist_ice_cfg" "$scratch/bundles.toc" "none" "rst_1step"
  for nml_plant in unknown-group wrong-group-var missing-stem bad-fsbc-multiple unapproved-deviation; do
    if verify_namelists_and_stems "$NML_CFG" "$NML_ICE_CFG" "$scratch/bundles.toc" "$nml_plant" "2step" >"$scratch/nml_${nml_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 3 (%s) did not fail namelist verification\n' "$nml_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/nml_${nml_plant}.log" >/dev/null
  done

  # ---- Checklist Item 4: compiled gfortran multi-rank layout fixture + plants ----
  mkdir -p "$scratch/fixture_run/o2_si3_records"
  "$FC" -O2 -ffree-line-length-none -J "$scratch" \
    "$STUBS_F90" "$WRITER_F90" "$FIXTURE_F90" \
    -o "$scratch/o2_si3_layout_fixture.exe"
  # Plant 4a: missing o2_si3_records output directory must fail the writer
  mkdir -p "$scratch/fixture_missing_dir"
  if ( cd "$scratch/fixture_missing_dir" && "$scratch/o2_si3_layout_fixture.exe" ) >/dev/null 2>&1; then
    printf 'REFUSE: Plant 4a (missing writer output directory) did not fail\n' >&2
    exit 72
  fi
  ( cd "$scratch/fixture_run" && "$scratch/o2_si3_layout_fixture.exe" )
  "$PY" "$GATE_PY" --verify-layout-fixture "$scratch/fixture_run/o2_si3_records" > "$scratch/layout_fixture.json"

  # ---- Checklist Item 5: all expected frames written + ASCII-only check + plants ----
  for lf_plant in rank-collapse wrong-offset truncation bad-magic missing-frame field-swap; do
    if "$PY" "$GATE_PY" --verify-layout-fixture "$scratch/fixture_run/o2_si3_records" --plant "$lf_plant" >"$scratch/lf_${lf_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 4/5 (%s) did not fail layout fixture verification\n' "$lf_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/lf_${lf_plant}.log" >/dev/null
  done

  for ascii_target in "$WRITER_F90" "$PRA_PATCH" "$IST_PATCH" "$STP_PATCH" "$STUBS_F90" "$FIXTURE_F90" "$NML_CFG" "$NML_ICE_CFG"; do
    check_ascii_only "$ascii_target"
  done
  printf '! UTF-8 em-dash plant: —\n' > "$scratch/utf8_plant.F90"
  if check_ascii_only "$scratch/utf8_plant.F90" 2>/dev/null; then
    printf 'REFUSE: Plant 5 (UTF-8 character) did not fail check_ascii_only\n' >&2
    exit 72
  fi

  # ---- Checklist Item 6: disk check when parent does not exist yet + free -g >= 150 GB + plants ----
  check_free_disk_kb "$EVIDENCE/nonexistent_parent_a/nonexistent_parent_b/run" "$MIN_FREE_DISK_KB"
  if check_free_disk_kb "$EVIDENCE/nonexistent_parent_a/run" 999999999999999 2>/dev/null; then
    printf 'REFUSE: Plant 6a (impossible disk floor) did not fail\n' >&2
    exit 72
  fi
  check_free_ram_gb "$MIN_FREE_RAM_GB"
  if check_free_ram_gb 999999 2>/dev/null; then
    printf 'REFUSE: Plant 6b (impossible RAM floor) did not fail\n' >&2
    exit 72
  fi

  # ---- Checklist Item 7: target absence + excerpt provenance + log check + same-program proof + EXIT trap ----
  if [[ "$target_check_mode" == "require_absent" ]]; then
    check_target_absent "$NEMO_ROOT/cfgs/$TARGET_REF_CFG"
    check_target_absent "$NEMO_ROOT/cfgs/$TARGET_REC_CFG"
    check_target_absent "$EVIDENCE"
  fi
  mkdir -p "$scratch/existing_target_plant"
  if check_target_absent "$scratch/existing_target_plant" 2>/dev/null; then
    printf 'REFUSE: Plant 7a (existing target directory) did not fail\n' >&2
    exit 72
  fi

  verify_excerpt_provenance_and_deck_independence "$LOG_EXCERPT" "$scratch" "none"
  if verify_excerpt_provenance_and_deck_independence "$LOG_EXCERPT" "$scratch" "excerpt-sha-mismatch" >"$scratch/exc_plant.log" 2>&1; then
    printf 'REFUSE: Plant 7b (excerpt SHA mismatch) did not fail\n' >&2
    exit 72
  fi
  grep -F "STATUS PLANT-FIRED" "$scratch/exc_plant.log" >/dev/null

  "$PY" "$GATE_PY" --verify-log "$LOG_EXCERPT" --expect-icedyn T --expect-icethd T >/dev/null
  "$PY" "$GATE_PY" --verify-log "$scratch/ocean.output.o2_rendered" --expect-icedyn F --expect-icethd F > "$scratch/log_check.json"
  sed -E 's/=/   =    /g' "$scratch/ocean.output.o2_rendered" > "$scratch/ocean.output.spaced"
  "$PY" "$GATE_PY" --verify-log "$scratch/ocean.output.spaced" --expect-icedyn F --expect-icethd F >/dev/null
  if "$PY" "$GATE_PY" --verify-log "$LOG_EXCERPT" --expect-icedyn F --expect-icethd F >"$scratch/log_wrong_dyn.log" 2>&1; then
    printf 'REFUSE: Plant 7c (wrong ln_icedyn/ln_icethd in log) did not fail\n' >&2
    exit 72
  fi
  grep -F "STATUS PLANT-FIRED" "$scratch/log_wrong_dyn.log" >/dev/null
  for log_plant in error-token missing-aaa wrong-jpl wrong-flag; do
    if "$PY" "$GATE_PY" --verify-log "$scratch/ocean.output.o2_rendered" --plant "$log_plant" >"$scratch/log_${log_plant}.log" 2>&1; then
      printf 'REFUSE: Plant 7d (log plant %s) did not fail\n' "$log_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/log_${log_plant}.log" >/dev/null
  done

  # Same-program proof preflight on synthetic reference vs instrumented BLD trees + plants
  mkdir -p "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo" "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/obj"
  mkdir -p "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj"
  for bcfg in arch_nemo.fcm cpp.fcm bld.cfg; do
    printf 'dir::root %s/sp/ORCA2_SAS_ICE_O2_REF/BLD\n%%FCFLAGS -O3\n' "$scratch" > "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/$bcfg"
    printf 'dir::root %s/sp/ORCA2_SAS_ICE_O2_REC/BLD\n%%FCFLAGS -O3\n' "$scratch" > "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/$bcfg"
  done
  cpp $FPP_KEYS -P -traditional -I "$src_dir" "$SHIPPED_PRA" -o "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90"
  cpp $FPP_KEYS -P -traditional -I "$src_dir" "$SHIPPED_IST" -o "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90"
  cpp $FPP_KEYS -P -traditional -I "$src_dir" "$SHIPPED_STP" -o "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90"
  cp "$mod_dir/icedyn_adv_pra.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/icedyn_adv_pra.f90"
  cp "$mod_dir/iceistate.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/iceistate.f90"
  cp "$mod_dir/icestp.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/icestp.f90"
  cp "$mod_dir/o2_si3_state_writer.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/o2_si3_state_writer.f90"
  cp "$mod_dir/icevar.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90"
  cp "$mod_dir/icevar.f90" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/icevar.f90"
  "$FC" -c -O3 -fdefault-real-8 -fcray-pointer -ffree-line-length-none -I "$mod_dir" -J "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD" \
    "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90" -o "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/obj/icevar.o"
  "$FC" -c -O3 -fdefault-real-8 -fcray-pointer -ffree-line-length-none -I "$mod_dir" -J "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD" \
    "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/icevar.f90" -o "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj/icevar.o"
  for u in icedyn_adv_pra iceistate icestp; do
    cp "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/obj/icevar.o" "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD/obj/${u}.o"
    cp "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj/icevar.o" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj/${u}.o"
  done
  cp "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj/icevar.o" "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/obj/o2_si3_state_writer.o"
  "$PY" "$GATE_PY" --same-program "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD" \
    --reference-bld "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD" \
    --pra-patch "$PRA_PATCH" --ist-patch "$IST_PATCH" --stp-patch "$STP_PATCH" > "$scratch/same_program_check.json"
  # Plant on same-program: mutate non-observer ppsrc/nemo/icevar.f90
  printf '\n! planted mutation\n' >> "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD/ppsrc/nemo/icevar.f90"
  if "$PY" "$GATE_PY" --same-program "$scratch/sp/ORCA2_SAS_ICE_O2_REC/BLD" \
       --reference-bld "$scratch/sp/ORCA2_SAS_ICE_O2_REF/BLD" \
       --pra-patch "$PRA_PATCH" --ist-patch "$IST_PATCH" --stp-patch "$STP_PATCH" >"$scratch/sp_plant.log" 2>&1; then
    printf 'REFUSE: Plant 7e (same-program non-observer mutation) did not fail\n' >&2
    exit 72
  fi
  grep -F "STATUS PLANT-FIRED" "$scratch/sp_plant.log" >/dev/null

  # Plant 7f: verify EXIT trap cleans created target cfgs on failed build AND moves completed-but-refused run to refused/
  mkdir -p "$scratch/trap_test/cfgs/REF_MOCK" "$scratch/trap_test/cfgs/REC_MOCK" \
           "$scratch/trap_test/incomplete_run" "$scratch/trap_test/completed_run" "$scratch/trap_test/refused"
  touch "$scratch/trap_test/completed_run/RUN_COMPLETE"
  if (
    trap cleanup_on_exit EXIT
    REFUSED_DIR="$scratch/trap_test/refused"
    TARGET_REF_ROOT="$scratch/trap_test/cfgs/REF_MOCK"
    TARGET_REC_ROOT="$scratch/trap_test/cfgs/REC_MOCK"
    CREATED_REF_CFG=1
    CREATED_REC_CFG=1
    CREATED_EVIDENCE=1
    RUN_COMPLETED=0
    ACTIVE_RUN_DIR="$scratch/trap_test/incomplete_run"
    ACTIVE_SCRATCH_DIR=""
    exit 91
  ) 2>/dev/null; then
    printf 'REFUSE: Plant 7f subshell did not exit non-zero\n' >&2
    exit 72
  fi
  if [[ -e "$scratch/trap_test/cfgs/REF_MOCK" || -e "$scratch/trap_test/cfgs/REC_MOCK" ]]; then
    printf 'REFUSE: Plant 7f EXIT trap did not clean created target cfgs after failed build\n' >&2
    exit 72
  fi
  if [[ -e "$scratch/trap_test/incomplete_run" ]] || [[ -z "$(ls -A "$scratch/trap_test/refused")" ]]; then
    printf 'REFUSE: Plant 7f EXIT trap did not move incomplete run to refused/\n' >&2
    exit 72
  fi
  if (
    trap cleanup_on_exit EXIT
    REFUSED_DIR="$scratch/trap_test/refused"
    TARGET_REF_ROOT="$scratch/trap_test/cfgs/REF_MOCK"
    TARGET_REC_ROOT="$scratch/trap_test/cfgs/REC_MOCK"
    CREATED_REF_CFG=0
    CREATED_REC_CFG=0
    CREATED_EVIDENCE=1
    RUN_COMPLETED=1
    ACTIVE_RUN_DIR="$scratch/trap_test/completed_run"
    ACTIVE_SCRATCH_DIR=""
    exit 92
  ) 2>/dev/null; then
    printf 'REFUSE: Plant 7f completed subshell did not exit non-zero\n' >&2
    exit 72
  fi
  if ! find "$scratch/trap_test/refused" -name RUN_COMPLETE | grep -q RUN_COMPLETE; then
    printf 'REFUSE: Plant 7f EXIT trap did not retain completed run in refused/\n' >&2
    exit 72
  fi

  # Also run the full O2 gate self-test and its plants P2-P6
  "$PY" "$GATE_PY" --self-test > "$scratch/gate_self_test.json"
  for gate_plant in 1ulp-istate 1ulp-ssh eos-wrong-branch tbo-halo-leak pond-unscaled hemisphere-swap ssh-sign-or-div dom-uniq-fold north-fold-sign 1ulp-state 1ulp-moment agg-formula stop-gradient; do
    if "$PY" "$GATE_PY" --self-test --plant "$gate_plant" >"$scratch/gp_${gate_plant}.log" 2>&1; then
      printf 'REFUSE: gate self-test plant %s did not fail\n' "$gate_plant" >&2
      exit 72
    fi
    grep -F "STATUS PLANT-FIRED" "$scratch/gp_${gate_plant}.log" >/dev/null
  done

  if [[ -n "$preflight_json_out" ]]; then
    mkdir -p "$(dirname "$preflight_json_out")"
    "$PY" - "$preflight_json_out" "$ORCA2_BUNDLE_SHA256" "$SAS_BUNDLE_SHA256" \
      "$scratch/layout_fixture.json" "$scratch/log_check.json" \
      "$scratch/same_program_check.json" "$scratch/gate_self_test.json" <<'PY'
import json
import pathlib
import sys

out_p, orca2_sha, sas_sha, lf_p, log_p, sp_p, gst_p = sys.argv[1:]
report = {
    "status": "PREFLIGHT_OK",
    "rung": "O2",
    "reference_config": "ORCA2_SAS_ICE",
    "orca2_bundle_sha256": orca2_sha,
    "sas_bundle_sha256": sas_sha,
    "checklist_items_1_to_7": "PASS",
    "plants_verified": [
        "1a_continuation_ampersand",
        "1b_undeclared_identifier",
        "1c_bare_use_iso_fortran_env",
        "2a_work_config_not_in_ref_cfgs",
        "2b_work_cfgs_txt_guard",
        "3a_unknown_namelist_group",
        "3b_wrong_group_var_cn_icerst_in_in_namini",
        "3c_missing_sn_stem",
        "3d_bad_fsbc_multiple",
        "3e_unapproved_deck_deviation",
        "4a_missing_writer_output_dir",
        "4b_rank_collapse",
        "4c_wrong_offset",
        "4d_truncation",
        "4e_bad_magic",
        "5a_missing_frame",
        "5b_field_swap",
        "5c_utf8_non_ascii",
        "6a_disk_floor_nonexistent_parent",
        "6b_ram_floor_150gb",
        "7a_existing_target_dir",
        "7b_excerpt_provenance_sha",
        "7c_log_wrong_icedyn_icethd",
        "7d_log_error_token",
        "7e_log_missing_aaa",
        "7f_log_wrong_jpl",
        "7g_log_wrong_flag",
        "7h_same_program_non_observer_mutation",
        "7i_exit_trap_failed_build_cleanup_and_completed_retained_in_refused",
        "gate_1ulp_istate_first_wet_cell",
        "gate_1ulp_ssh_first_wet_cell",
        "gate_eos_wrong_branch",
        "gate_tbo_halo_leak",
        "gate_pond_unscaled",
        "gate_hemisphere_swap",
        "gate_ssh_sign_or_div",
        "gate_dom_uniq_fold",
        "gate_north_fold_sign",
        "gate_1ulp_state_first_wet_cell",
        "gate_1ulp_moment_first_wet_cell",
        "gate_agg_formula",
        "gate_stop_gradient",
    ],
    "layout_fixture": json.loads(pathlib.Path(lf_p).read_text()),
    "log_verifier": json.loads(pathlib.Path(log_p).read_text()),
    "same_program_check": json.loads(pathlib.Path(sp_p).read_text()),
    "gate_self_test": json.loads(pathlib.Path(gst_p).read_text()),
}
pathlib.Path(out_p).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
  fi

  printf 'PREFLIGHT_OK: checklist items 1-7 and all plants passed\n'
}

admit_all() {
  local expect_commit=$1
  if [[ -d "$TARGET_REF_ROOT/BLD" && -d "$TARGET_REC_ROOT/BLD" ]]; then
    "$PY" "$GATE_PY" --same-program "$TARGET_REC_ROOT/BLD" \
      --reference-bld "$TARGET_REF_ROOT/BLD" \
      --pra-patch "$PRA_PATCH" --ist-patch "$IST_PATCH" --stp-patch "$STP_PATCH" \
      --new-cfg "$TARGET_REC_CFG" --reference-cfg "$TARGET_REF_CFG" \
      --json "$EVIDENCE/manifest/same_program.json"
  fi
  "$PY" "$GATE_PY" --admit-evidence "$EVIDENCE" \
    --expect-commit "$expect_commit" \
    --json "$EVIDENCE/o2_admission.json"
  for adm_plant in stamp same-program binary-sha input-sha missing-complete missing-stop0 ref-rec-diff 1ulp-istate 1ulp-ssh 1ulp-state rst-missing rst-1ulp; do
    if "$PY" "$GATE_PY" --admit-evidence "$EVIDENCE" \
         --expect-commit "$expect_commit" \
         --plant "$adm_plant" >/dev/null 2>&1; then
      printf 'REFUSE: admission plant %s did not fail\n' "$adm_plant" >&2
      exit 72
    fi
  done
}

if [[ "$mode" == --preflight-only ]]; then
  run_preflight require_absent
  exit 0
fi

if [[ "$mode" == --admit-existing ]]; then
  run_preflight skip_target_absent
  [[ -f "$EVIDENCE/manifest/producer_commit.txt" ]] || {
    printf 'REFUSE: missing %s/manifest/producer_commit.txt\n' "$EVIDENCE" >&2
    exit 65
  }
  existing_commit=$(tr -d '[:space:]' < "$EVIDENCE/manifest/producer_commit.txt")
  admit_all "$existing_commit"
  printf 'ACQUISITION_ADMITTED %s\n' "$EVIDENCE"
  exit 0
fi

# ---- Operator-only --run mode ----
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: --run requires a clean committed git worktree\n' >&2
  exit 63
}
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)

run_preflight require_absent

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/inputs_scratch"
CREATED_EVIDENCE=1
ACTIVE_RUN_DIR="$EVIDENCE"
cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$EVIDENCE/manifest/work_cfgs.before"
printf '%s\n' "$COMMIT" > "$EVIDENCE/manifest/producer_commit.txt"
sha256sum "$ORCA2_BUNDLE" "$SAS_BUNDLE" "$WRITER_F90" "$PRA_PATCH" "$IST_PATCH" "$STP_PATCH" \
  "$NML_CFG" "$NML_ICE_CFG" "$GATE_PY" > "$EVIDENCE/manifest/inputs.sha256"

tar -ztf "$ORCA2_BUNDLE" > "$EVIDENCE/manifest/bundles.toc"
tar -ztf "$SAS_BUNDLE" >> "$EVIDENCE/manifest/bundles.toc"
tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/inputs_scratch"
tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/inputs_scratch"

build_target() {
  local cfg_name=$1
  local instrumented=$2
  local cfg_root=$NEMO_ROOT/cfgs/$cfg_name
  check_free_ram_gb "$MIN_FREE_RAM_GB"
  if [[ "$cfg_name" == "$TARGET_REF_CFG" ]]; then
    CREATED_REF_CFG=1
  elif [[ "$cfg_name" == "$TARGET_REC_CFG" ]]; then
    CREATED_REC_CFG=1
  fi
  cd "$NEMO_ROOT"
  ./makenemo -r "$REF_CFG_SOURCE" -n "$cfg_name" -m conda-scalarmath del_key 'key_xios'
  if [[ "$instrumented" -eq 1 ]]; then
    cp "$SHIPPED_PRA" "$cfg_root/MY_SRC/icedyn_adv_pra.F90"
    cp "$SHIPPED_IST" "$cfg_root/MY_SRC/iceistate.F90"
    cp "$SHIPPED_STP" "$cfg_root/MY_SRC/icestp.F90"
    cp "$WRITER_F90" "$cfg_root/MY_SRC/o2_si3_state_writer.F90"
    patch --fuzz=0 "$cfg_root/MY_SRC/icedyn_adv_pra.F90" < "$PRA_PATCH"
    patch --fuzz=0 "$cfg_root/MY_SRC/iceistate.F90" < "$IST_PATCH"
    patch --fuzz=0 "$cfg_root/MY_SRC/icestp.F90" < "$STP_PATCH"
    touch "$cfg_root/MY_SRC/"*.F90
    ./makenemo -n "$cfg_name" -m conda-scalarmath del_key 'key_xios'
  fi
  verify_work_cfgs_guard "$EVIDENCE/manifest/work_cfgs.before" "$NEMO_ROOT/cfgs/work_cfgs.txt" \
    "$EVIDENCE/manifest/work_cfgs.filtered_before" "$EVIDENCE/manifest/work_cfgs.filtered_after"
  [[ -x "$cfg_root/BLD/bin/nemo.exe" ]] || {
    printf 'REFUSE: %s did not produce nemo.exe\n' "$cfg_name" >&2
    exit 68
  }
  local dyn_syms
  dyn_syms=$(nm -D "$cfg_root/BLD/bin/nemo.exe")
  if grep '_ZGV' <<<"$dyn_syms" >/dev/null; then
    printf 'REFUSE: vector-math symbol present in %s\n' "$cfg_name" >&2
    exit 68
  fi
}

stage_and_run() {
  local cfg_name=$1
  local run_dir=$2
  local is_rst_run=$3
  local cfg_root=$NEMO_ROOT/cfgs/$cfg_name

  check_free_ram_gb "$MIN_FREE_RAM_GB"
  check_free_disk_kb "$run_dir" "$MIN_FREE_DISK_KB"
  mkdir -p "$run_dir/o2_si3_records"
  cp -L "$SHIPPED_NML_REF" "$run_dir/namelist_ref"
  cp -L "$SHIPPED_NML_ICE_REF" "$run_dir/namelist_ice_ref"
  if [[ "$is_rst_run" -eq 1 ]]; then
    render_rst_namelists "$NML_CFG" "$NML_ICE_CFG" "$run_dir/namelist_cfg" "$run_dir/namelist_ice_cfg"
    verify_namelists_and_stems "$run_dir/namelist_cfg" "$run_dir/namelist_ice_cfg" "$EVIDENCE/manifest/bundles.toc" "none" "rst_1step"
    cp "$EVIDENCE/rec_2step/ORCA2_SAS_00000001_restart_ice.nc" "$run_dir/ORCA2_SAS_00000001_restart_ice.nc"
  else
    cp "$NML_CFG" "$run_dir/namelist_cfg"
    cp "$NML_ICE_CFG" "$run_dir/namelist_ice_cfg"
    verify_namelists_and_stems "$run_dir/namelist_cfg" "$run_dir/namelist_ice_cfg" "$EVIDENCE/manifest/bundles.toc" "none" "2step"
  fi
  for f in "$EVIDENCE/inputs_scratch"/*.nc; do
    ln -sf "$f" "$run_dir/$(basename "$f")"
  done
  cp "$cfg_root/BLD/bin/nemo.exe" "$run_dir/nemo"
  (
    cd "$run_dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > run.user.time.log
    timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1
    printf 'MPIRUN_RC=0\nRUN_FINISHED_UTC=%s\nRUN DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> run.user.time.log
  )
  "$PY" "$GATE_PY" --verify-log "$run_dir/ocean.output" --expect-icedyn F --expect-icethd F > "$run_dir/log_check.json"
  touch "$run_dir/RUN_COMPLETE"
}

build_target "$TARGET_REF_CFG" 0
build_target "$TARGET_REC_CFG" 1
sha256sum "$NEMO_ROOT/cfgs/$TARGET_REF_CFG/BLD/bin/nemo.exe" \
          "$NEMO_ROOT/cfgs/$TARGET_REC_CFG/BLD/bin/nemo.exe" > "$EVIDENCE/manifest/binaries.sha256"

stage_and_run "$TARGET_REF_CFG" "$EVIDENCE/ref_2step" 0
stage_and_run "$TARGET_REC_CFG" "$EVIDENCE/rec_2step" 0
stage_and_run "$TARGET_REC_CFG" "$EVIDENCE/rst_1step" 1
RUN_COMPLETED=1
touch "$EVIDENCE/RUN_COMPLETE"

admit_all "$COMMIT"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
