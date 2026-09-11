#!/bin/bash
# Acquire NEMO's DINO kt=2 record, as a CONFIG COPY.  The pristine cfgs/DINO
# and src/ trees are never written.
#
# WHY kt=2 EXISTS AT ALL.  Everything about the leap-frog that is INTERESTING
# is degenerate on the from-rest first step:
#
#   * rDt.  `domain.f90:288` sets rDt = 2*rn_Dt for the modified leap-frog,
#     but `stpmlf.f90:131-133` reduces it to rn_Dt while `l_1st_euler`, and
#     `:617-620` restores it at the END of that first step.  So at kt=1 a card
#     that wrongly used rn_Dt everywhere would score IDENTICALLY.  The
#     isoneutral operator's stabilising threshold reads rDt twice
#     (`traldf_iso.f90:829` `zcoef0 = rDt*(pakz + pah_wslp2/ze3w_2)`, `:830`
#     `* r1_Dt`), so kt=2 is where that statement can be wrong.
#
#   * ssh_atf.  `stpmlf.f90:425` Asselin-filters ssh(:,:,Nnn) BEFORE tra_ldf
#     at `:504`, while r3t(:,:,Nnn) is not refreshed until `:571`.  That split
#     is guarded `IF( .NOT.l_1st_euler )` (`sshwzv.f90:443`), so at kt=1 the
#     two "now" heights coincide and the Kmm-stretch transcription cannot be
#     tested.  At kt=2 they deliberately disagree.
#
#   * akz.  On the kt=1 record the stabiliser fires on 0 of 342134 wet cells,
#     so `traldf_iso_a33`'s own e3w and the implicit half of the explicit/
#     implicit split are multiplied by nothing any gate can see.
#
# WHAT IT PRODUCES
#   $OUT/DINO_00000002_restart_*.nc   the kt=2 restart, carrying the Kbb/Kmm
#                                     levels AND the per-operator ttrd_*/strd_*
#                                     trends (ln_tra_trd/ln_dyn_trd are already
#                                     .true. in the namelist this copies)
#   $OUT/DINO_00000001_restart_*.nc   the kt=1 restart from the SAME run, which
#                                     is the BEFORE/NOW state the kt=2 gate
#                                     feeds legoESM -- so the two sides start
#                                     from one object, not from two runs
#   $OUT/*_kt00000002*.bin            the per-step streams this build writes
#
# The ONLY namelist edits are nn_itend 1 -> 2 and nn_stock 1 -> 2.  No source
# patch: this record needs no instrumentation the certified binary lacks.
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt2_trends/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
# ~5 minutes: one makenemo build plus a TWO TIME STEP run.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_KT2_TRENDS
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# Identical guard to nemo_dino_kt1_rankdump/run.sh, for the identical reason:
# a relative OUT, a symlinked NEMO or a CFGNAME containing '..' must not be
# able to resolve inside the read-only oracle.
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
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 8

# ---------------------------------------------------------------- namelist
mkdir -p "$OUT"
cd "$OUT"
cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
# Two edits, both on lines whose CURRENT value is checked first, so a namelist
# that has already moved cannot be silently re-edited into something else.
python3 - "$OUT/namelist_cfg" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for key, want_old, new in (("nn_itend", "1", "2"), ("nn_stock", "1", "2")):
    m = re.search(r"^(\s*%s\s*=\s*)(\S+)(.*)$" % key, s, re.M)
    if m is None:
        raise SystemExit("REFUSING: %s not found in %s" % (key, p))
    if m.group(2).rstrip() != want_old:
        raise SystemExit(
            "REFUSING: %s is %r, expected %r -- this namelist is not the "
            "kt=1 record's, so editing it would produce a record that is "
            "not what this script claims" % (key, m.group(2), want_old))
    s = s[:m.start()] + m.group(1) + new + m.group(3) + s[m.end():]
open(p, "w").write(s)
print("namelist: nn_itend 1 -> 2, nn_stock 1 -> 2")
PY
grep -E "^\s*(nn_itend|nn_stock|rn_Dt|ln_rstart|ln_tra_trd|ln_dyn_trd)\s*=" namelist_cfg

# ---------------------------------------------------------------- run
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_kt2_trends.log

# ------------------------------------------------------------- twin check
# The kt=1 restart written by THIS run must be bit-identical to the certified
# RUN_FROMREST_KT1, or the kt=2 state is not the continuation of the record
# every other gate on this branch is scored against.  Proven, not asserted.
python3 "$HERE/../nemo_dino_kt1_rankdump/read_rankdump.py" \
    --twin-check "$OUT" --reference "$SRCREF"

echo
echo "record written to $OUT"
echo "  kt=1 restart (the BEFORE/NOW state the gate feeds legoESM):"
echo "    $OUT/DINO_00000001_restart_*.nc"
echo "  kt=2 restart (the AFTER state and the per-operator trends):"
echo "    $OUT/DINO_00000002_restart_*.nc"
echo
echo "then score it in ONE command:"
echo "  CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\"
echo "    python scripts/validate/ocean_fidelity/dino_1226/kt2_leapfrog_gate.py \\"
echo "      --run-dir $OUT"
