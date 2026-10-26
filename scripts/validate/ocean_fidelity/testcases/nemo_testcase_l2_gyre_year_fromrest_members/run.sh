#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  NEMO and legoESM, both from rest on the
# certified GYRE card, have never been compared beyond TEN time steps.  This
# acquires NEMO's side of the YEAR: a four-member from-rest ensemble whose own
# spread is the noise floor against which the model-vs-model gap is judged.
# The design, the scored rows, the floor, the verdict rule and the
# preregistered expectations are in
#   docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md
# and nothing here re-derives any of them.
#
# ONE BINARY, FIVE RUN DIRECTORIES.  The members differ ONLY in a namelist
# integer, so they share one build.  Four config copies would be four binaries
# and four chances of a build difference inside what is supposed to be a
# controlled comparison; a namelist line cannot do that.
#
#   $TARGET_RUN/nemo_seed0..3/   the four members, nn_pert_seed = 0..3
#   $TARGET_RUN/nemo_pristine/   the SAME year on the UNPATCHED certified
#                                binary, with NO nn_pert_seed in its namelist
#
# The pristine run is the instrument's own check and the reason the patch is
# allowed to exist: "seed 0 is the unperturbed path" is not argued, it is
# measured, by requiring every one of seed 0's twelve restarts to be
# BYTE-IDENTICAL to the pristine run's.
#
# THE PATCH.  GYRE, unlike DINO, does not ship nn_pert_seed.  Two additive
# MY_SRC patches add it:
#   usrdef_istate_yearpert.patch  copies cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183
#                                 verbatim, after the existing profile loop
#   usrdef_nam_yearpert.patch     declares nn_pert_seed, prints it, and extends
#                                 the NAMELIST/namusr_def/ statement
# The istate patch removes ZERO lines.  The nam patch removes exactly ONE, the
# NAMELIST statement, because Fortran has no way to extend a NAMELIST group
# without rewriting its line.  Both properties are CHECKED below, not asserted.
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have; makenemo
# then fails with "Can't locate Text/Balanced.pm" long before it compiles
# anything.  The line below puts the conda build environment first, which is
# the PATH the round-33..47 builds actually used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_YRPERT
readonly ARCH=${ARCH:-conda-scalarmath}
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly TARGET_RUN=$L2/year_fromrest
readonly PHASE0=$TARGET_RUN/phase0_floor.json
readonly SEEDS=${SEEDS:-"0 1 2 3"}
# The year, and the restart cadence.  Both are fixed by the preregistration.
readonly NN_ITEND=2160          # 360 days at rn_Dt = 14400 s, nn_leapy = 30
readonly NN_STOCK=180           # a restart every 30 days
readonly NN_WRITE=2160          # one output file; scoring reads restarts

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly IST_PATCH=$here/usrdef_istate_yearpert.patch
readonly NAM_PATCH=$here/usrdef_nam_yearpert.patch
readonly HARNESS=$here/../nemo_testcase_l2_gyre_year_fromrest.py
readonly SHIPPED_IST=$NEMO_ROOT/src/OCE/USR/usrdef_istate.F90
readonly SHIPPED_NAM=$NEMO_ROOT/src/OCE/USR/usrdef_nam.F90

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG
pristine_binary=$source_cfg/BLD/bin/nemo.exe

# ------------------------------------------------------------- preconditions
[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$IST_PATCH" && -f "$NAM_PATCH" ]]
[[ -f "$SHIPPED_IST" && -f "$SHIPPED_NAM" && -f "$HARNESS" ]]
[[ -x "$pristine_binary" ]] || { printf 'REFUSE: no certified binary at %s\n' "$pristine_binary" >&2; exit 64; }
[[ -f "$NEMO_ROOT/arch/arch-$ARCH.fcm" ]] || { printf 'REFUSE: no arch/arch-%s.fcm\n' "$ARCH" >&2; exit 64; }

# ------------------------------------------------------------- the PHASE-0 gate
# PHASE 0 costs no NEMO time and its ONLY job here is to say whether the floor
# can see anything at all.  A zero floor would be read as "the models are
# distinguishable", which is the exact false positive this harness exists to
# avoid, so the members are not spent until it has spoken.  The decision is
# READ from phase 0's own artifact and never re-derived here: a constant copied
# into two files is a constant that goes stale in one of them.
if [[ ! -f "$PHASE0" ]]; then
  printf 'REFUSE: %s does not exist.\n' "$PHASE0" >&2
  printf '  Run PHASE 0 first (legoESM only, no NEMO time):\n' >&2
  printf '    for s in 0 1 2 3; do python %s --member $s; done\n' "$HARNESS" >&2
  printf '    python %s --score-phase0\n' "$HARNESS" >&2
  exit 2
fi
python3 - "$PHASE0" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
gate = report.get("vacuity_gate")
if gate is None:
    raise SystemExit("REFUSE: %s predates the vacuity gate; re-run "
                     "--score-phase0" % sys.argv[1])
if not gate["phase1_may_run"]:
    raise SystemExit("REFUSE: the legoESM floor is zero on days %s.  %s"
                     % (gate["zero_floor_days"], gate["reason"]))
shape = report.get("floor_shape", {}).get("classification")
print("phase-0 gate: PROCEED (floor positive on every scored day; "
      "floor shape = %s)" % shape)
PY

# ------------------------------------------------------------------- refusals
guard() {                       # guard <path being written> [cfgcopy]
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$source_cfg")"/*)
      printf 'REFUSE: %s resolves inside the certified card (read-only)\n' "$1" >&2; exit 2 ;;
    "$(readlink -m "$NEMO_ROOT/src")"/*)
      printf 'REFUSE: %s resolves inside src/ (read-only oracle)\n' "$1" >&2; exit 2 ;;
  esac
  if [[ "${2:-}" != "cfgcopy" ]]; then
    case "$real/" in
      "$(readlink -m "$NEMO_ROOT")"/*)
        printf 'REFUSE: %s resolves inside the NEMO checkout\n' "$1" >&2; exit 2 ;;
    esac
  fi
}
guard "$target_cfg" cfgcopy
guard "$TARGET_RUN"
if [[ -e "$target_cfg" ]]; then
  printf 'REFUSE: %s already exists; delete it by hand if it is stale.\n' "$target_cfg" >&2
  exit 64
fi
for s in $SEEDS; do
  if [[ -e "$TARGET_RUN/nemo_seed$s" ]]; then
    printf 'REFUSE: %s already exists\n' "$TARGET_RUN/nemo_seed$s" >&2; exit 64
  fi
done
[[ ! -e "$TARGET_RUN/nemo_pristine" ]] || { printf 'REFUSE: %s already exists\n' "$TARGET_RUN/nemo_pristine" >&2; exit 64; }
# The premise the patches rest on: this card does not already override these
# two files, so the shipped source is what gets patched.
for f in usrdef_istate.F90 usrdef_nam.F90; do
  if [[ -e "$source_cfg/MY_SRC/$f" ]]; then
    printf 'REFUSE: %s overrides %s; the shipped-source premise is false\n' "$SOURCE_CFG" "$f" >&2
    exit 66
  fi
done

# A WRITE-only instrument may ADD lines.  `diff | grep -q` would return diff's
# own status under pipefail and could never fire, so count removed lines
# instead; '^-[^-]' would MISS a deleted BLANK line, which a unified diff emits
# as a bare '-'.
removed() { echo $(( $(grep -c '^-' "$1") - $(grep -c '^---' "$1") )); }
if [[ "$(removed "$IST_PATCH")" -ne 0 ]]; then
  printf 'REFUSE: %s removes a shipped line; the perturbation must only ADD\n' "$IST_PATCH" >&2
  exit 67
fi
# Exactly one removed line in the nam patch, and it must be the NAMELIST
# statement.  Fortran cannot extend a NAMELIST group without rewriting its
# line; anything else removed is a different change wearing this one's clothes.
if [[ "$(removed "$NAM_PATCH")" -ne 1 ]]; then
  printf 'REFUSE: %s removes %s lines, expected exactly 1\n' "$NAM_PATCH" "$(removed "$NAM_PATCH")" >&2
  exit 67
fi
if ! grep '^-' "$NAM_PATCH" | grep -v '^---' | grep -q 'NAMELIST/namusr_def/'; then
  printf 'REFUSE: the one removed line in %s is NOT the NAMELIST statement\n' "$NAM_PATCH" >&2
  exit 67
fi
dry=$(mktemp -d /tmp/gyre-yearpert-dry.XXXXXX)
cp "$SHIPPED_IST" "$dry/usrdef_istate.F90"
cp "$SHIPPED_NAM" "$dry/usrdef_nam.F90"
if ! patch -s "$dry/usrdef_istate.F90" <"$IST_PATCH" >/dev/null 2>&1 \
   || ! patch -s "$dry/usrdef_nam.F90" <"$NAM_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the patches do not apply cleanly to the shipped source\n' >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"

# Space.  Five runs x twelve restarts x ~2.6 MB, plus a build tree.
for mount in /tmp "$TARGET_RUN" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 4194304 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 4 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

work=$(mktemp -d /tmp/gyre-yearpert-prov.XXXXXX)
printf 'provenance directory (retained): %s\n' "$work"
( cd "$source_cfg" && find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum ) \
  >"$work/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-$ARCH.fcm" "$source_cfg/cpp_${SOURCE_CFG}.fcm" \
  "$SHIPPED_IST" "$SHIPPED_NAM" "$IST_PATCH" "$NAM_PATCH" "$pristine_binary" \
  >"$work/toolchain.sha256"

# ---------------------------------------------------------------------- build
cd "$NEMO_ROOT"
# No del_key: the cpp key file is copied wholesale from the source card below,
# so the target inherits exactly its keys (key_qco key_vco_1d3d key_RK3) and
# does NOT inherit key_xios, which this card does not use.
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m "$ARCH"
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
( cd "$target_cfg" && find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum ) \
  >"$work/copied_cfg_before_patch.sha256"
cmp "$work/source_cfg.sha256" "$work/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
cp "$SHIPPED_IST" "$target_cfg/MY_SRC/usrdef_istate.F90"
cp "$SHIPPED_NAM" "$target_cfg/MY_SRC/usrdef_nam.F90"
cmp "$SHIPPED_IST" "$target_cfg/MY_SRC/usrdef_istate.F90"
cmp "$SHIPPED_NAM" "$target_cfg/MY_SRC/usrdef_nam.F90"
patch "$target_cfg/MY_SRC/usrdef_istate.F90" <"$IST_PATCH"
patch "$target_cfg/MY_SRC/usrdef_nam.F90" <"$NAM_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/usrdef_istate.F90" "$target_cfg/MY_SRC/usrdef_nam.F90" \
  "$IST_PATCH" "$NAM_PATCH" >"$work/yearpert_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m "$ARCH"
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The perturbation must be IN the binary, not merely in a file next to it.
grep -q 'nn_pert_seed' "$target_cfg/BLD/ppsrc/nemo/usrdef_istate.f90" \
  || { printf 'REFUSE: nn_pert_seed is absent from the COMPILED usrdef_istate; every member would be the SAME run and the floor would be 0\n' >&2; exit 69; }
grep -q 'nn_pert_seed' "$target_cfg/BLD/ppsrc/nemo/usrdef_nam.f90" \
  || { printf 'REFUSE: nn_pert_seed is absent from the COMPILED usrdef_nam\n' >&2; exit 69; }
# The card's OWN pre-existing writers must still be compiled, or this is a
# different model from the one the ten-step ladder certified.
for writer in stprk3 stprk3_stg; do
  grep -q 'oracle_' "$target_cfg/BLD/ppsrc/nemo/$writer.f90" \
    || { printf 'REFUSE: MY_SRC writer %s not compiled (stale build)\n' "$writer" >&2; exit 69; }
done
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s; this is not the scalar-math arch\n' "$target_binary" >&2
  exit 65
fi
sha256sum "$target_binary" >>"$work/binary.sha256"

# ------------------------------------------------------------------ namelists
# One python writer for every run directory, so the five namelists cannot drift
# apart.  It PRINTS the diff against the certified namelist and REFUSES if more
# than the preregistered rows changed.
write_namelist() {            # write_namelist <dir> <seed|none>
  python3 - "$source_cfg/EXP00/namelist_cfg" "$1/namelist_cfg" "$2" \
           "$NN_ITEND" "$NN_STOCK" "$NN_WRITE" <<'PY'
import re, sys
src, dst, seed, itend, stock, write = sys.argv[1:7]
text = open(src).read()
for key, value in (("nn_itend", itend), ("nn_stock", stock), ("nn_write", write)):
    text, n = re.subn(r"^(\s*%s\s*=\s*)(\S+)" % key, r"\g<1>%s" % value,
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("REFUSE: %s not found exactly once in %s" % (key, src))
if seed != "none":
    # Insert at the END of &namusr_def, just before its closing '/', so the
    # group's existing rows and comments keep their order.
    block = re.search(r"^&namusr_def\b.*?^(/)\s*$", text, re.M | re.S)
    if block is None:
        raise SystemExit("REFUSE: no &namusr_def block in %s" % src)
    text = (text[:block.start(1)]
            + "   nn_pert_seed = %s   ! from-rest ensemble member seed\n" % seed
            + text[block.start(1):])
open(dst, "w").write(text)

# The check that this namelist differs from the certified one ONLY in the
# preregistered rows.  It compares PARSED ASSIGNMENTS, not lines: a line-by-line
# zip shifts by one at the first insertion and then reports every subsequent
# line as changed, which is exactly what the first version of this check did --
# it refused a correct namelist with "197 changed rows".  Caught by dry-running
# the writer against the certified namelist before the operator ever saw it.
ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*(?:!.*)?$")

def rows(blob):
    out = {}
    for line in blob.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("&"):
            continue
        match = ASSIGN.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out

before, after = rows(open(src).read()), rows(text)
changed = sorted(k for k in before if k in after and before[k] != after[k])
added = sorted(set(after) - set(before))
removed = sorted(set(before) - set(after))
print("  namelist rows changed vs the certified card:")
for key in changed:
    print("    %-14s %s -> %s" % (key, before[key], after[key]))
for key in added:
    print("    %-14s (new) -> %s" % (key, after[key]))
allowed_changed = {"nn_itend", "nn_stock", "nn_write"}
allowed_added = {"nn_pert_seed"} if seed != "none" else set()
if removed or set(changed) - allowed_changed or set(added) - allowed_added:
    raise SystemExit(
        "REFUSE: changed=%s added=%s removed=%s; the preregistration allows "
        "changed %s and added %s"
        % (changed, added, removed, sorted(allowed_changed),
           sorted(allowed_added)))
if set(changed) != allowed_changed:
    raise SystemExit(
        "REFUSE: only %s changed; all of %s must change or this is not the "
        "year run" % (changed, sorted(allowed_changed)))
PY
}

stage_run_dir() {             # stage_run_dir <dir> <binary> <seed|none>
  local dir=$1 binary=$2 seed=$3
  guard "$dir"
  mkdir -p "$dir"
  for name in namelist_ref namelist_top_cfg namelist_top_ref \
              namelist_pisces_cfg namelist_pisces_ref context_nemo.xml \
              file_def_nemo.xml iodef.xml axis_def_nemo.xml \
              domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml \
              field_def_nemo-pisces.xml; do
    cp -L "$source_cfg/EXP00/$name" "$dir/$name"
  done
  write_namelist "$dir" "$seed"
  cp "$binary" "$dir/nemo"
  cp "$work"/*.sha256 "$dir/"
}

run_dir() {                   # run_dir <dir>
  ( cd "$1"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
    { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
      2>>run.user.time.log
    test "${PIPESTATUS[0]}" -eq 0
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.time.log )
}

mkdir -p "$TARGET_RUN"
printf '\n=== pristine control: the certified binary, NO nn_pert_seed ===\n'
stage_run_dir "$TARGET_RUN/nemo_pristine" "$pristine_binary" none
run_dir "$TARGET_RUN/nemo_pristine"

for s in $SEEDS; do
  printf '\n=== member seed %s ===\n' "$s"
  stage_run_dir "$TARGET_RUN/nemo_seed$s" "$target_binary" "$s"
  run_dir "$TARGET_RUN/nemo_seed$s"
done

# ------------------------------------------------------- the instrument's checks
# 1. seed 0 must be BYTE-IDENTICAL to the pristine run.  This is the whole
#    licence for the patch: it proves the added block is a no-op when off, and
#    it proves the extended NAMELIST statement changed nothing else.
printf '\n=== seed-0 byte identity against the unpatched binary ===\n'
identical=1
for f in "$TARGET_RUN/nemo_pristine"/*_restart.nc; do
  base=${f##*/}
  if [[ ! -e "$TARGET_RUN/nemo_seed0/$base" ]]; then
    printf 'REFUSE: seed 0 did not write %s\n' "$base" >&2; exit 71
  fi
  if cmp -s "$f" "$TARGET_RUN/nemo_seed0/$base"; then
    printf '  IDENTICAL %s\n' "$base"
  else
    printf '  DIFFERS   %s\n' "$base"; identical=0
  fi
done
if [[ "$identical" -ne 1 ]]; then
  printf 'REFUSE: the seed-0 member is NOT byte-identical to the unperturbed run,\n' >&2
  printf '  so the patch is not write-only with respect to the unperturbed path.\n' >&2
  exit 71
fi
printf 'seed 0 is byte-identical to the unpatched certified binary on every restart\n'

# 2. The members must actually DIFFER.  Four bit-identical members would give a
#    floor of exactly zero, which would then be read as "the models are
#    distinguishable" -- the false positive this whole harness exists to avoid.
printf '\n=== members differ ===\n'
python3 - "$TARGET_RUN" "$NN_ITEND" $SEEDS <<'PY'
import glob, hashlib, os, sys
root, itend, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
digests = {}
for seed in seeds:
    matches = sorted(glob.glob(os.path.join(
        root, "nemo_seed%s" % seed, "*_%08d_restart.nc" % itend)))
    if len(matches) != 1:
        raise SystemExit("REFUSE: member %s has %d final restarts"
                         % (seed, len(matches)))
    digests[seed] = hashlib.sha256(open(matches[0], "rb").read()).hexdigest()
    print("  seed %s final restart sha256 = %s" % (seed, digests[seed][:24]))
if len(set(digests.values())) != len(seeds):
    raise SystemExit(
        "REFUSE: two or more members are bit-identical, so nn_pert_seed did "
        "not reach the initial state and the measured floor would be 0.")
print("all members differ, as the perturbation requires")
PY

# 3. Every scored day must have a restart, or the verdict would silently be
#    computed on fewer days than the preregistration names.
python3 - "$TARGET_RUN" "$NN_STOCK" "$NN_ITEND" $SEEDS <<'PY'
import glob, os, sys
root, stock, itend, seeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
missing = []
for seed in seeds:
    for step in range(stock, itend + 1, stock):
        if not glob.glob(os.path.join(root, "nemo_seed%s" % seed,
                                      "*_%08d_restart.nc" % step)):
            missing.append((seed, step))
if missing:
    raise SystemExit("REFUSE: missing restarts %s" % missing)
print("every scored day has a restart on every member")
PY

( cd "$TARGET_RUN" && find nemo_seed* nemo_pristine -name '*_restart.nc' -print0 \
    | sort -z | xargs -0 sha256sum > nemo_year_fromrest_restarts.sha256 )

printf '\nNEMO members written to %s\n' "$TARGET_RUN"
printf 'score with:\n'
printf '  python %s --score\n' "$HARNESS"
printf '  python %s --figures\n' "$HARNESS"
printf 'GYRE_YEAR_FROMREST_NEMO_READY %s\n' "$TARGET_RUN"
