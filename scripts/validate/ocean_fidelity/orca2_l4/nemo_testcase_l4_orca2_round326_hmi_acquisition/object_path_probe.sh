#!/usr/bin/env bash
# Why does a fresh build's object differ from round 5's?  Recompiles every
# round-5 object from round 5's own sources with round 5's arch flags, (a) at
# round 5's paths and (b) with its config root copied under OUT with the
# round-326 config name, then counts objects raw-identical / identical under
# the gate's objects_equal (config root substituted) / different.  Plants: a
# flipped .text byte and an -O2 recompile must be refused.  gfortran only; no
# makenemo, no NEMO.  Usage: object_path_probe.sh OUT   (OUT must not exist)
set -Eeuo pipefail
export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/bin:/usr/bin:/bin
readonly OUT=${1:?usage: $0 OUT}
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REF=$NEMO_ROOT/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly NEW=$OUT/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
[[ ! -e "$OUT" ]] || { echo "REFUSE: $OUT exists" >&2; exit 64; }
mkdir -p "$NEW/BLD/obj" "$OUT/obj_ref" "$OUT/mod_ref" "$OUT/mod_new"
cp -a "$REF/BLD/ppsrc" "$NEW/BLD/ppsrc"
cp -a "$REF/WORK" "$NEW/WORK"
# Round 5's compile line (as in makenemo's log): %FC %FCFLAGS -I conda -c SRC.
FCLINE="$(awk '$1 == "%FC" {$1 = ""; print}' "$REF/BLD/arch_nemo.fcm") -I$REF/BLD/inc $(awk '$1 == "%FCFLAGS" {$1 = ""; print}' "$REF/BLD/arch_nemo.fcm") -I/home/dbalwada/miniconda3/envs/nemo-build/include"
export FCLINE

src_of() {  # source a round-5 object was compiled from, under config root $2
  local f u=$1
  # Two objects are named after their first program unit (paths embedded in them).
  case $u in exampl) u=module_example ;; subcalledbyagrif) u=agrif_user ;; esac
  for f in "$2"/BLD/ppsrc/*/"$u.f90" "$2/WORK/$u.f90" "$NEMO_ROOT/ext/IOIPSL/src/$u.f90"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  echo "REFUSE: no source for $1" >&2; return 1
}
jobs=$OUT/jobs.txt; : >"$jobs"
for o in "$REF"/BLD/obj/*.o; do
  u=$(basename "$o" .o)
  a=$(src_of "$u" "$REF"); b=$(src_of "$u" "$NEW")
  echo "$a $OUT/obj_ref/$u.o $OUT/mod_ref" >>"$jobs"
  echo "$b $NEW/BLD/obj/$u.o $OUT/mod_new" >>"$jobs"
done
# shellcheck disable=SC2016
xargs -P 32 -L 1 bash -c '$FCLINE -o "$1" -J "$2" -c "$0" 2>>"$2.err"' <"$jobs"
ablsrc=$(src_of ablmod "$NEW")
$FCLINE -O2 -o "$OUT/ablmod_O2.o" -J "$OUT/mod_new" -c "$ablsrc"
cp "$NEW/BLD/obj/ablmod.o" "$OUT/ablmod_textflip.o"
read -r off size < <(readelf -SW "$OUT/ablmod_textflip.o" | awk '{for (i = 1; i < NF; i++) if ($i == ".text") print strtonum("0x"$(i+3)), strtonum("0x"$(i+4))}')
printf '\x90' | dd of="$OUT/ablmod_textflip.o" bs=1 seek=$((off + size / 2)) conv=notrunc status=none

"$PY" -I - "$here/../nemo_testcase_l4_orca2_round326_hmi_gate.py" "$REF" "$NEW" "$OUT" <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", sys.argv[1]); gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
ref, new, out = (Path(a) for a in sys.argv[2:])
rp, np_ = f"{ref}/".encode(), f"{new}/".encode()
units = sorted(p.name for p in (ref / "BLD/obj").glob("*.o"))
rows = {"reproduced_at_round5_path": 0, "raw_identical_new_path": 0,
        "identical_after_root_substitution": 0, "differ": []}
for u in units:
    r = (ref / "BLD/obj" / u).read_bytes()
    rows["reproduced_at_round5_path"] += (out / "obj_ref" / u).read_bytes() == r
    n = (new / "BLD/obj" / u).read_bytes()
    rows["raw_identical_new_path"] += n == r
    if gate.objects_equal(r, n, rp, np_):
        rows["identical_after_root_substitution"] += 1
    else:
        rows["differ"].append(u)
print(f"OBJECTS {len(units)}")
for k, v in rows.items():
    print(f"{k} {v}")
a = (ref / "BLD/obj/ablmod.o").read_bytes()
for plant in ("ablmod_O2.o", "ablmod_textflip.o"):
    refused = not gate.objects_equal(a, (out / plant).read_bytes(), rp, np_)
    print(f"PLANT {plant} {'REFUSED' if refused else 'STAYED_GREEN'}")
    if not refused:
        sys.exit(3)
sys.exit(0 if not rows["differ"] and rows["reproduced_at_round5_path"] == len(units) else 2)
EOF
