#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nhwsub
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_wsubstep_%j.log
# THE NEXT CUT OF THE NH DRY GAP. The 6.6116e-04 was localised to the
# ACOUSTIC LOOP (job 9448215): 1.3003e-07 m/s of w is already present
# when dyn_core returns, and pt/delp are at the parity floor there, so
# the defect is w-specific and born inside `do it=1,n_split`. The loop
# runs 8 sub-steps; a single end-of-loop dump cannot tell
#
#   error full-size at sub-step 1  -> ONE STAGE is wrong
#   error growing over 8 sub-steps -> a COEFFICIENT is wrong
#
# so this build dumps w/pt/delp at the END OF EVERY SUB-STEP, inside
# dyn_core, and KEEPS the pre-remap dump from the previous instrument.
# Keeping both in ONE binary is what makes the cross-check free: nothing
# between the loop's end and dyn_core's return writes w
# (dyn_core.F90:1736-1830), so sub-step 8 must equal the pre-remap dump
# BITWISE -- and if it does not, one of the two dumps is not where it
# says it is, with no build drift available as an excuse.
#
# delz IS DUMPED PER SUB-STEP TOO, and it is the field that decides the
# question. w starts at EXACTLY zero on both sides, and the only routine
# that can make it non-zero is Riem_Solver3, which is strictly column
# local (nh_core.F90:42-206 -- every array indexed (i,j) inside the j/i
# loops, no horizontal stencil). So a corner-localised w error out of a
# column-local solver is a statement about that solver's INPUTS, and the
# only input with a horizontal stencil is the layer height, written by
# update_dz_d (nh_utils.F90:194-311). Scoring w without scoring delz
# measures the symptom and calls it the cause.
#
# A REVIEWER ASKED FOR A NO-DUMP CONTROL BINARY (codex MAJOR, job
# 9450542): the worry is that inserting a call in the hot loop changes
# what the optimizer may do, so the eight readings would describe the
# instrumented program rather than the pinned one. DECLINED, with the
# reason, rather than silently: this deck runs ONE step, so the restart
# is a deterministic function of everything all eight sub-steps did, and
# the probe's control 3 now requires that restart to be BITWISE the
# certified one on EVERY prognostic. Any perturbation large enough to
# move a sub-step would have to cancel to the last bit by the end of the
# same step to hide from that, which floating-point arithmetic does not
# do. If this instrument is ever pointed at a multi-step deck, or if
# control 3 is ever relaxed to a tolerance, build the control binary.
#
# WHY A NEW BUILD. dyncore_dump2d exists only in fv3_recon/duo_model,
# which is compiled -DSW_DYNAMICS: under that define fv_dynamics sets
# akap = 1 and excludes the whole 3-D block, so it cannot produce a 3-D
# NH dump at all. The pinned tree is chmod a-w and must not be touched,
# so this copies it and patches the copy.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
NCF=/burg/opt/netcdf-fortran-4.5.3
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
SRC0=$PIN/atmos_cubed_sphere-symmetryclean
W=/burg-archive/glab/users/pg2328/fv3_wsubstep
SRC=$W/src
B=$W/build
FMSB=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/build_gfs
FMSSRC=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/fms-src
FMSLIB=$(find $FMSB -name "libfms_r8.a" | head -1)
FMSINC=$FMSB/include_r8
echo "host: $(hostname)"; echo "FMSLIB=$FMSLIB"

if [ "${REBUILD:-1}" = "1" ]; then
  rm -rf $W && mkdir -p $SRC $B
  cp -r $SRC0/. $SRC/
  chmod -R u+w $SRC
  # THE PATCH: dump w (level 1) right after dyn_core returns and before
  # Lagrangian_to_Eulerian. Asserted to apply exactly once.
  python3 - "$SRC/model/fv_dynamics.F90" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
anchor = "                                           call timing_off('DYN_CORE')"
if s.count(anchor) != 1:
    raise SystemExit(f"anchor count {s.count(anchor)}, expected 1")
dump = anchor + """

! LEGOESM w-DUMP: the state as dyn_core leaves it, before the remap.
      if ( .not. hydrostatic ) then
         call legoesm_wdump(n_map, 'w', w(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_wdump(n_map, 'pt', pt(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_wdump(n_map, 'delp', delp(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_wdump(n_map, 'delz', delz(is:ie, js:je, 1), is, js)
      endif
"""
s = s.replace(anchor, dump, 1)

# the writer, appended before the module's final `end module`
tail = "\nsubroutine legoesm_wdump(blk, name, arr, ilo, jlo)\n" \
       "  character(len=*), intent(in) :: name\n" \
       "  integer, intent(in) :: blk, ilo, jlo\n" \
       "  real, intent(in) :: arr(:, :)\n" \
       "  character(len=256) :: fn\n" \
       "  integer :: un, i, j\n" \
       "  write (fn, '(A,I0,A,A,A,I0,A)') 'wdump_b', blk, '_', trim(name), &\n" \
       "      '_t', mpp_pe() + 1, '.dat'\n" \
       "  open (newunit=un, file=trim(fn), status='replace', action='write')\n" \
       "  write (un, '(4I8)') ilo, ilo + size(arr, 1) - 1, jlo, jlo + size(arr, 2) - 1\n" \
       "  do j = 1, size(arr, 2)\n" \
       "    do i = 1, size(arr, 1)\n" \
       "      write (un, '(2I8,ES26.17E3)') ilo + i - 1, jlo + j - 1, arr(i, j)\n" \
       "    end do\n" \
       "  end do\n" \
       "  close (un)\n" \
       "end subroutine legoesm_wdump\n\n"
m = list(re.finditer(r"^end module .*$", s, re.M))
if not m:
    raise SystemExit("no end module")
i = m[-1].start()
s = s[:i] + tail + s[i:]
open(p, "w").write(s)
print("patched fv_dynamics.F90: w-dump after dyn_core")
PYEOF
  [ $? -eq 0 ] || { echo "PATCH FAILED"; exit 8; }
  # COUNT THE CALL SITES, not the routine. `grep -q legoesm_wdump` is
  # satisfied by the appended writer alone, so a replace() that silently
  # no-opped would still have reported success (codex MAJOR, job 9450542).
  n_call=$(grep -c "call legoesm_wdump(" $SRC/model/fv_dynamics.F90)
  n_def=$(grep -c "^subroutine legoesm_wdump(" $SRC/model/fv_dynamics.F90)
  [ "$n_call" = "4" ] && [ "$n_def" = "1" ] || {
    echo "PATCH DID NOT LAND: fv_dynamics calls=$n_call (want 4) defs=$n_def (want 1)"; exit 8; }

  # THE SECOND PATCH: dump w/pt/delp (level 1) at the END of every
  # acoustic sub-step, inside dyn_core's `do it=1,n_split` loop.
  python3 - "$SRC/model/dyn_core.F90" <<'PYEOF2'
import re, sys
p = sys.argv[1]
s = open(p).read()
anchor = "  enddo   ! time split loop"
if s.count(anchor) != 1:
    raise SystemExit(f"anchor count {s.count(anchor)}, expected 1")
dump = """
! LEGOESM SUB-STEP DUMP: the state at the end of acoustic sub-step `it`.
      if ( .not. hydrostatic ) then
         call legoesm_sdump(it, 'w', w(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_sdump(it, 'pt', pt(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_sdump(it, 'delp', delp(isd:ied, jsd:jed, 1), isd, jsd)
         call legoesm_sdump(it, 'delz', delz(is:ie, js:je, 1), is, js)
      endif
""" + anchor
s = s.replace(anchor, dump, 1)

tail = "\nsubroutine legoesm_sdump(it, name, arr, ilo, jlo)\n" \
       "  character(len=*), intent(in) :: name\n" \
       "  integer, intent(in) :: it, ilo, jlo\n" \
       "  real, intent(in) :: arr(:, :)\n" \
       "  character(len=256) :: fn\n" \
       "  integer :: un, i, j\n" \
       "  write (fn, '(A,I0,A,A,A,I0,A)') 'sdump_it', it, '_', trim(name), &\n" \
       "      '_t', mpp_pe() + 1, '.dat'\n" \
       "  open (newunit=un, file=trim(fn), status='replace', action='write')\n" \
       "  write (un, '(4I8)') ilo, ilo + size(arr, 1) - 1, jlo, jlo + size(arr, 2) - 1\n" \
       "  do j = 1, size(arr, 2)\n" \
       "    do i = 1, size(arr, 1)\n" \
       "      write (un, '(2I8,ES26.17E3)') ilo + i - 1, jlo + j - 1, arr(i, j)\n" \
       "    end do\n" \
       "  end do\n" \
       "  close (un)\n" \
       "end subroutine legoesm_sdump\n\n"
m = list(re.finditer(r"^end module .*$", s, re.M))
if not m:
    raise SystemExit("no end module")
i = m[-1].start()
s = s[:i] + tail + s[i:]
open(p, "w").write(s)
print("patched dyn_core.F90: per-sub-step dump")
PYEOF2
  [ $? -eq 0 ] || { echo "DYNCORE PATCH FAILED"; exit 8; }
  n_call=$(grep -c "call legoesm_sdump(" $SRC/model/dyn_core.F90)
  n_def=$(grep -c "^subroutine legoesm_sdump(" $SRC/model/dyn_core.F90)
  [ "$n_call" = "4" ] && [ "$n_def" = "1" ] || {
    echo "DYNCORE PATCH DID NOT LAND: calls=$n_call (want 4) defs=$n_def (want 1)"; exit 8; }

  # THE THIRD PATCH: the layer height over the FULL PADDED BOX -- halo
  # rings included, which nothing in this campaign has dumped for this
  # field -- immediately BEFORE and immediately AFTER update_dz_d. What
  # remains after the metric families were exhausted is exactly two
  # hypotheses, and one capture pair separates them:
  #    A. the height already differs on ENTRY  -> the duo halo FILL
  #    B. same on entry, differs on EXIT       -> the update OPERATOR
  # TWO interfaces, not one: delz(:,1) spans zh(:,1) and zh(:,2), so a
  # single interface cannot attach a height error to a layer. zh is
  # allocated (isd:ied, jsd:jed, npz+1) at dyn_core.F90:299, so the
  # padded slice below is its own declared extent.
  #
  # NAMED BY THE ACOUSTIC SUB-STEP, not by n_map. update_dz_d sits INSIDE
  # `do it=1,n_split` (dyn_core.F90:339 .. :1737), so an n_map-keyed name
  # lets sub-steps 2..8 overwrite sub-step 1 and leaves one file that
  # LOOKS complete. It matters beyond a time offset: `if (it == 1)` at
  # :537 means only the first sub-step takes its height halo from the
  # gz exchange at :543, so the surviving file would test a DIFFERENT
  # exchange than the hypothesis names -- and the artifact (~1e-2 m over
  # seven sub-steps) is ~300x the signal being hunted (~3e-5 m) while
  # sitting under the probe's abort threshold. It would have read as a
  # confirmation.
  python3 - "$SRC/model/dyn_core.F90" <<'PYEOF3'
import re, sys
p = sys.argv[1]
s = open(p).read()
head = "call update_dz_d(nord_v, damp_vt, flagstruct%hord_tm"
tail_a = "flagstruct%lim_fac)"
if s.count(head) != 1:
    raise SystemExit(f"update_dz_d head count {s.count(head)}, expected 1")
if s.count(tail_a) != 1:
    raise SystemExit(f"update_dz_d tail count {s.count(tail_a)}, expected 1")
pre = ("! LEGOESM HALO PROBE: the two interfaces bracketing model layer 1,\n"
       "! full padded box, BEFORE update_dz_d writes them.\n"
       "      call legoesm_hdump(it, 'pre', 'zh1', "
       "zh(isd:ied, jsd:jed, 1), isd, jsd)\n"
       "      call legoesm_hdump(it, 'pre', 'zh2', "
       "zh(isd:ied, jsd:jed, 2), isd, jsd)\n      ") + head
s = s.replace(head, pre, 1)
post = tail_a + ("\n"
       "! LEGOESM HALO PROBE: the same two interfaces, AFTER update_dz_d.\n"
       "      call legoesm_hdump(it, 'post', 'zh1', "
       "zh(isd:ied, jsd:jed, 1), isd, jsd)\n"
       "      call legoesm_hdump(it, 'post', 'zh2', "
       "zh(isd:ied, jsd:jed, 2), isd, jsd)")
s = s.replace(tail_a, post, 1)

writer = "\nsubroutine legoesm_hdump(blk, when, name, arr, ilo, jlo)\n" \
       "  character(len=*), intent(in) :: when, name\n" \
       "  integer, intent(in) :: blk, ilo, jlo\n" \
       "  real, intent(in) :: arr(:, :)\n" \
       "  character(len=256) :: fn\n" \
       "  integer :: un, i, j\n" \
       "  write (fn, '(A,I0,A,A,A,A,A,I0,A)') 'hdump', blk, '_', trim(when), &\n" \
       "      '_', trim(name), '_t', mpp_pe() + 1, '.dat'\n" \
       "  open (newunit=un, file=trim(fn), status='replace', action='write')\n" \
       "  write (un, '(4I8)') ilo, ilo + size(arr, 1) - 1, jlo, jlo + size(arr, 2) - 1\n" \
       "  do j = 1, size(arr, 2)\n" \
       "    do i = 1, size(arr, 1)\n" \
       "      write (un, '(2I8,ES26.17E3)') ilo + i - 1, jlo + j - 1, arr(i, j)\n" \
       "    end do\n" \
       "  end do\n" \
       "  close (un)\n" \
       "end subroutine legoesm_hdump\n\n"
m = list(re.finditer(r"^end module .*$", s, re.M))
if not m:
    raise SystemExit("no end module")
i = m[-1].start()
s = s[:i] + writer + s[i:]
open(p, "w").write(s)
print("patched dyn_core.F90: pre/post update_dz_d full-pad zh dump")
PYEOF3
  [ $? -eq 0 ] || { echo "HDUMP PATCH FAILED"; exit 8; }
  n_call=$(grep -c "call legoesm_hdump(" $SRC/model/dyn_core.F90)
  n_def=$(grep -c "^subroutine legoesm_hdump(" $SRC/model/dyn_core.F90)
  [ "$n_call" = "4" ] && [ "$n_def" = "1" ] || {
    echo "HDUMP PATCH DID NOT LAND: calls=$n_call (want 4) defs=$n_def (want 1)"; exit 8; }

  cd $B
  FF="-O2 -fdefault-real-8 -fdefault-double-8 -fcray-pointer -ffree-line-length-none -fno-range-check -fdec -cpp"
  # NO -DSW_DYNAMICS: that define excludes the entire 3-D fv_dynamics
  # block and sets akap = 1.
  DEFS="-DSPMD -Duse_libMPI -Duse_netCDF -DINTERNAL_FILE_NML"
  INC="-I$FMSINC -I$NCF/include -I$SRC/tools -I$FMSSRC/include"
  FILES=$(ls $SRC/tools/global_grid/*.F90 $SRC/tools/*.F90 $SRC/model/*.F90 $SRC/driver/solo/*.F90 2>/dev/null)
  DRV=$(ls /burg-archive/glab/users/pg2328/fv3_recon/duo_model/atmos_drivers/solo/atmos_model.F90 2>/dev/null)
  FILES="$FILES $DRV"
  echo "compiling $(echo "$FILES" | wc -w) files (fixpoint passes)"
  prev=999
  for pass in 1 2 3 4 5 6 7 8 9 10; do
    remaining=""; n_ok=0
    for f in $FILES; do
      o=$(basename ${f%.F90}).o
      [ -f $o ] && { n_ok=$((n_ok+1)); continue; }
      if mpif90 $FF $DEFS $INC -I$B -c $f -o $o 2> ${o%.o}.err; then n_ok=$((n_ok+1))
      else remaining="$remaining $f"; fi
    done
    cur=$(echo $remaining | wc -w)
    echo "pass $pass: ok=$n_ok remaining=$cur"
    [ -z "$remaining" ] && break
    if [ "$cur" = "$prev" ]; then
      echo "FIXPOINT STALL"
      for f in $remaining; do echo "== $(basename $f)"; head -12 $(basename ${f%.F90}).err; done
      exit 1
    fi
    prev=$cur
  done
  [ -n "${remaining:-}" ] && { echo "BUILD_INCOMPLETE"; exit 1; }
  mpif90 $FF -o fv3_wsubstep.exe *.o $FMSLIB -L$NCF/lib -lnetcdff \
    -L/burg/opt/netcdf-c-4.7.4/lib -lnetcdf || { echo "LINK_FAILED"; exit 1; }
  echo "BUILD OK: $B/fv3_wsubstep.exe"
fi

# RUN the NH deck with the instrumented binary
R=$W/run_nh_1step
rm -rf $R && mkdir -p $R/RESTART
cp -r $PIN/run_nh_1step_gfs/INPUT $R/
for f in input.nml diag_table field_table; do /bin/cp -f $PIN/run_nh_1step_gfs/$f $R/; done
cd $R
mpirun -np 6 $B/fv3_wsubstep.exe > run_out.txt 2>&1
rc=$?
echo "run rc=$rc"
[ "$rc" -eq 0 ] || { tail -30 run_out.txt; echo "RUN FAILED"; exit 6; }
grep -i "FMSConstants" logfile.000000.out | head -1
grep -iE "^ *(HYDROSTATIC|ADIABATIC|NPZ|K_SPLIT|N_SPLIT) " logfile.000000.out | head -6
echo "--- pre-remap dumps ---"; ls wdump_*.dat 2>/dev/null | wc -l
echo "--- sub-step dumps ---"; ls sdump_*.dat 2>/dev/null | head -6
# ASSERT the count, do not print it for a human. A crash three sub-steps
# in used to leave a partial set, report success, and surface only as a
# cryptic missing-file exit inside the probe (GLM m3, job 9450546).
NSP=$(grep -iE "^ *n_split *=" logfile.000000.out | head -1 \
      | awk -F= '{gsub(/[^0-9]/, "", $2); print $2}')
NW=$(ls wdump_*.dat 2>/dev/null | wc -l)
NS=$(ls sdump_*.dat 2>/dev/null | wc -l)
WANT_S=$((NSP * 4 * 6))
echo "n_split=$NSP  wdump=$NW  sdump=$NS (want $WANT_S)"
[ -n "$NSP" ] && [ "$NS" -eq "$WANT_S" ] && [ "$NW" -eq 24 ] || {
  echo "DUMP COUNT WRONG: the run did not complete every sub-step"; exit 7; }
# `hdump*`, not `hdump_*`: the writer emits hdump<it>_<when>_<name>_t<pe>
# with no underscore after the stem, so the first version of this glob
# matched nothing and failed a run that had produced every file.
NH=$(ls hdump*.dat 2>/dev/null | wc -l)
WANT_H=$((NSP * 24))
echo "hdump=$NH (want $WANT_H: n_split x 2 capture points x 2 interfaces x 6 tiles)"
[ "$NH" -eq "$WANT_H" ] || {
  echo "HDUMP COUNT WRONG: the update_dz_d dumps are incomplete"; exit 7; }
echo NH_WSUBSTEP_DONE
