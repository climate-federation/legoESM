#!/bin/bash
# Acquire NEMO's DINO kt=1 SLOPE record -- uslp/vslp/wslpi/wslpj, ah_wslp2 and
# akz as tra_ldf's own frame sees them -- as a CONFIG COPY.  The pristine
# cfgs/DINO and src/ trees are never written.
#
# WHY.  RUN_FROMREST_KT1's uslp_stg/vslp_stg/wslpi_stg/wslpj_stg are
# identically ZERO on all sixteen tiles while rhd_stg/tn_stg/rn2_stg from the
# same snapshot carry data.  tra_ldf cannot have run on zero slopes (its A33
# stage is 1.07x the whole tendency), so the DUMP is empty, not the physics --
# and the consequence is that legoESM's slope routine has never been scored on
# this card.  Every tra_ldf number on this branch is currently a number for the
# OPERATOR fed legoESM's own slopes; nothing separates a correct operator on
# wrong slopes from the converse.
#
# WHAT IT PRODUCES
#   $OUT/ldfslp_at_traldf_r<rank>.bin   16 self-describing tiles, each holding
#                                       uslp, vslp, wslpi, wslpj, ah_wslp2, akz
#                                       at tra_ldf's own call site at kt=1
#   $OUT/DINO_00000001_restart_*.nc     the twin restart, checked byte-for-byte
#                                       against RUN_FROMREST_KT1's
#
# The patch is WRITE-ONLY and ADDITIVE: it reads nothing and changes nothing,
# which the twin check then PROVES rather than asserts.
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt1_slopes/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
# ~5 minutes: one makenemo build plus a ONE TIME STEP run.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_KT1_SLOPES
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_slopes}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
guard() {                       # guard <path being written> [cfgcopy]
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$NEMO/cfgs/DINO")"/*)
      echo "REFUSING: $1 resolves inside cfgs/DINO (read-only oracle)" >&2
      exit 2 ;;
    "$(readlink -m "$NEMO/src")"/*)
      echo "REFUSING: $1 resolves inside src/ (read-only oracle)" >&2
      exit 2 ;;
  esac
  if [ "${2:-}" != "cfgcopy" ]; then
    case "$real/" in
      "$(readlink -m "$NEMO")"/*)
        echo "REFUSING: $1 resolves inside the NEMO checkout $NEMO; the" >&2
        echo "  record must be written outside the read-only oracle." >&2
        exit 2 ;;
    esac
  fi
}
guard "$COPY" cfgcopy
guard "$OUT"
case "$OUT" in
  /|/tmp|/home|/data) echo "REFUSING: OUT=$OUT is a system directory" >&2
                      exit 2 ;;
esac
case "$(readlink -m "$NEMO")/" in
  "$OUT"/*) echo "REFUSING: OUT=$OUT CONTAINS the NEMO checkout" >&2
            exit 2 ;;
esac
if [ -e "$COPY" ]; then
  echo "REFUSING: $COPY already exists." >&2
  echo "  Delete it by hand if you are sure it is stale -- this script will" >&2
  echo "  not decide that for you, because a stale copy still compiles." >&2
  exit 2
fi
if [ ! -d "$SRCREF" ]; then
  echo "REFUSING: $SRCREF not found (the namelists this record must reuse)" >&2
  exit 2
fi
if [ ! -f "$NEMO/arch/arch-$ARCH.fcm" ]; then
  echo "REFUSING: no arch/arch-$ARCH.fcm; set ARCH=<your makenemo -m arch>" >&2
  exit 2
fi

# ---------------------------------------------------------------- build
cd "$NEMO"
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 0
# traldf_iso.F90 is NOT one of DINO's MY_SRC overrides, so the copy has to
# take it from src/ FIRST.  Copied, never moved; the original is untouched.
guard "$COPY/MY_SRC/traldf_iso.F90" cfgcopy
cp -f "$NEMO/src/OCE/TRA/traldf_iso.F90" "$COPY/MY_SRC/traldf_iso.F90"
# The copy must be byte-identical to the oracle's before it is patched, or the
# record would be about a file nobody has read.
cmp "$COPY/MY_SRC/traldf_iso.F90" "$NEMO/src/OCE/TRA/traldf_iso.F90"
python3 "$HERE/slopes_patch.py" "$COPY/MY_SRC/traldf_iso.F90"
./makenemo -n "$CFGNAME" -m "$ARCH" -j 8

# The PREPROCESSOR decides whether the writer is in the binary.  A patch that
# compiled away would otherwise produce an empty record that looks successful
# -- which is precisely the failure this whole record exists to repair.
PP=$COPY/BLD/ppsrc/nemo/traldf_iso.f90
for token in "ldfslp_at_traldf_r" "cl_slp" "7281"; do
  if ! grep -qF "$token" "$PP"; then
    echo "REFUSING: '$token' is not in the COMPILED source $PP -- the writer" >&2
    echo "  did not survive preprocessing, so the record would be empty." >&2
    exit 3
  fi
  if grep -qF "$token" "$NEMO/src/OCE/TRA/traldf_iso.F90"; then
    echo "REFUSING: token '$token' also occurs in the PRISTINE source, so" >&2
    echo "  finding it above proved nothing. Pick a token the patch adds." >&2
    exit 3
  fi
done
echo "ppsrc check: the per-rank slope writer is in the compiled source, and"
echo "  each token checked is absent from the pristine source"

# ---------------------------------------------------------------- run
mkdir -p "$OUT"
cd "$OUT"
cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_kt1_slopes.log

n=$(ls "$OUT"/ldfslp_at_traldf_r*.bin 2>/dev/null | wc -l)
if [ "$n" -ne "$NPROC" ]; then
  echo "REFUSING: $n slope tiles written, expected $NPROC" >&2
  exit 3
fi
echo "slope record: $n tiles"

# ---------------------------------------------------------------- twin check
# The instrumentation must not have changed the PHYSICS.  Proven, not asserted:
# every variable of every restart tile against the untouched record.
python3 "$HERE/../nemo_dino_kt1_rankdump/read_rankdump.py" \
    --twin-check "$OUT" --reference "$SRCREF"

echo
echo "record written to $OUT"
echo "then score legoESM's slope routine against it in ONE command:"
echo "  CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\"
echo "    python scripts/validate/ocean_fidelity/dino_1226/ldfslp_kt1_gate.py \\"
echo "      --run-dir $OUT"
