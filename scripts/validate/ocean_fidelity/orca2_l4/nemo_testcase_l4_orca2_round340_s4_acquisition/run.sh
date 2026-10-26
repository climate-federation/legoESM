#!/usr/bin/env bash
# Operator-only exact S4 acquisition. Default is read-only preflight.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in --preflight-only|--run|--admit-existing) ;; *) exit 64;; esac
refuse(){ printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse(){ local n=$1; shift; if "$@"; then refuse "plant stayed green: $n"; fi; printf 'PREFLIGHT_PLANT_FIRED %s\n' "$n"; }
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P); readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly K0_CFG=ORCA2_STANDALONE_IIF_OMIP_L4_R339K0
readonly TARGET_CFG=ORCA2_STANDALONE_IIF_OMIP_L4_R340S4
readonly K0_BUILD=$NEMO_ROOT/cfgs/$K0_CFG
readonly TARGET_BUILD=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly K0_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/k0_smoke/orca2_k0_10step_np2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly ICE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/ice_cover_producer/orca2_full_si3_year1_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round340/s4_acquisition
readonly RUN_A=$EVIDENCE/orca2_s4_exact_a_np2
readonly RUN_B=$EVIDENCE/orca2_s4_exact_b_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly MPIFC=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpif90
readonly GATE=$here/../nemo_testcase_l4_orca2_round340_standalone_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round340_standalone.py
readonly SURFACE_PATCH=$here/sbcmod_round340.patch
readonly STATE_PATCH=$here/stprk3_round340.patch
readonly STAGE_PATCH=$here/stprk3_stg_round340.patch
readonly FLD_PATCH=$here/fldread_round340.patch
readonly IIF_PATCH=$here/sbcice_if_round340.patch
readonly SURFACE_WRITER=$here/round340_surface_writer.F90
readonly STATE_WRITER=$here/round340_state_writer.F90
readonly SURFACE_FIXTURE=$here/round340_surface_fixture.F90
readonly NAMELIST_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ref
readonly ADMISSION_IDENTITY_ONLY=${R340_ADMISSION_IDENTITY_ONLY:-0}
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for f in "$GATE" "$TEST" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$SURFACE_WRITER" "$STATE_WRITER" "$SURFACE_FIXTURE" \
 "$K0_RUN/producer_manifest.json" "$ICE_RUN/producer_manifest.json" "$NAMELIST_REF"; do [[ -f "$f" ]] || refuse "missing $f"; done
if [[ "$ADMISSION_IDENTITY_ONLY" == 1 ]]; then
 [[ "$MODE" == --admit-existing && -n ${R340_ADMISSION_TEST_ROOT:-} && -n ${R340_EXISTING_RUN:-} ]] \
   || refuse 'identity-only admission needs --admit-existing and explicit test/run roots'
 test_root=$(realpath -e -- "$R340_ADMISSION_TEST_ROOT")
 retained=$(realpath -e -- "$R340_EXISTING_RUN")
 [[ "$retained" == "$test_root/refused/"* ]] || refuse 'retained test run is outside refused/'
 "$PY" - "$retained/producer_manifest.json" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
p=c.verify_manifest(Path(sys.argv[1]),expected_commit=sys.argv[2])
print('R340_RETAINED_IDENTITY_ADMISSION_PASS',p['binary_sha256'])
PY
 exit 0
fi
[[ "$ADMISSION_IDENTITY_ONLY" == 0 ]] || refuse 'R340_ADMISSION_IDENTITY_ONLY must be 0 or 1'

check_placement(){
 local f=$1
 [[ $(grep -Ec 'CALL r340_surface_frame\(kt,[0-7]\)' "$f") -eq 8 ]] || return 1
 awk '/CALL sbc_ssm/{m=NR}/SELECT CASE\( nsbc \)/{s1=NR}/SELECT CASE\( nn_ice \)/{s3=NR}/CALL sbc_rnf/{s2=NR}/CALL r340_surface_frame\(kt,7\)/{x=NR} END{exit !(m<s1&&s1<s3&&s3<s2&&s2<x)}' "$f"
}
check_other_placements(){
 local d=$1
 [[ $(grep -Fc 'CALL r340_state_frame' "$d/stprk3.F90") -eq 1 ]] || return 1
 [[ $(grep -Fc 'CALL r340_state_frame' "$d/stprk3_stg.F90") -eq 1 ]] || return 1
 [[ $(grep -Fc 'CALL r340_fld_frame' "$d/fldread.F90") -eq 1 ]] || return 1
 [[ $(grep -Fc 'CALL r340_iif_frame' "$d/sbcice_if.F90") -eq 4 ]] || return 1
 awk '/^[[:space:]]*IF.*CALL isf_stp/{a=NR}/CALL r340_state_frame/{b=NR}/CALL sbc        /{c=NR} END{exit !(a<b&&b<c)}' "$d/stprk3.F90" || return 1
 awk '/CALL r340_state_frame/{a=NR}/DEALLOCATE\( zFu/{b=NR} END{exit !(a<b)}' "$d/stprk3_stg.F90" || return 1
 awk '/end loop over field/{a=NR}/CALL r340_fld_frame/{b=NR} END{exit !(a<b)}' "$d/fldread.F90" || return 1
 awk '/CALL fld_read/{f=NR}/r340_iif_frame\(kt,0/{a=NR}/CALL eos_fzp/{e=NR}/r340_iif_frame\(kt,1/{b=NR}/r340_iif_frame\(kt,2/{c=NR}/r340_iif_frame\(kt,3/{d=NR} END{exit !(f<a&&a<e&&e<b&&b<c&&c<d)}' "$d/sbcice_if.F90"
}

scratch=$(mktemp -d /tmp/orca2-r340-preflight.XXXXXX)
cleanup(){ local rc=$?; trap - EXIT; printf 'R340_SCRATCH_RETAINED %s rc=%s\n' "$scratch" "$rc" >&2; exit "$rc"; }
trap cleanup EXIT
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcmod.f90" "$scratch/sbcmod.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3.f90" "$scratch/stprk3.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3_stg.f90" "$scratch/stprk3_stg.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/fldread.f90" "$scratch/fldread.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcice_if.f90" "$scratch/sbcice_if.F90"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$SURFACE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$STATE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$STAGE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$FLD_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$IIF_PATCH"
"$PY" - "$scratch" "$K0_RUN/build_sources/ppsrc/nemo" \
  "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
scratch, reference = Path(sys.argv[1]), Path(sys.argv[2])
units = dict(zip(('sbcmod','stprk3','stprk3_stg','fldread','sbcice_if'), map(Path,sys.argv[3:]), strict=True))
for unit, patch in units.items():
    baseline=(reference/f'{unit}.f90').read_text().splitlines()
    observed=(scratch/f'{unit}.F90').read_text().splitlines()
    inserted=[line[1:] for line in patch.read_text().splitlines()
              if line.startswith('+') and not line.startswith('+++')]
    g._observer_only(baseline,observed,inserted,unit)
planted=observed.copy(); planted.insert(-1,'      CALL r340_unregistered_physics()')
try: g._observer_only(baseline,planted,inserted,'sbcice_if')
except g.GateError: print('PREFLIGHT_PLANT_FIRED same-program-ppsrc')
else: raise AssertionError('unregistered real-ppsrc insertion stayed green')
print('R340_SAME_PROGRAM_PPSRC_PASS units=5')
PY
check_placement "$scratch/sbcmod.F90" || refuse 'surface observers moved'
check_other_placements "$scratch" || refuse 'state/fld/ice-if observers moved'
cp "$scratch/sbcmod.F90" "$scratch/placement.plant"
sed -i '/r340_surface_frame(kt,5)/d' "$scratch/placement.plant"
plant_must_refuse placement check_placement "$scratch/placement.plant"
for unit in stprk3 stprk3_stg fldread sbcice_if; do
  mkdir "$scratch/$unit.plant"
  cp "$scratch"/{stprk3,stprk3_stg,fldread,sbcice_if}.F90 "$scratch/$unit.plant/"
  sed -i '0,/CALL r340_/d' "$scratch/$unit.plant/$unit.F90"
  plant_must_refuse "placement-$unit" check_other_placements "$scratch/$unit.plant"
done
mkdir "$scratch/mod"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -fsyntax-only "$SURFACE_WRITER" "$STATE_WRITER"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$SURFACE_WRITER" -o "$scratch/surface-writer.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$STATE_WRITER" -o "$scratch/state-writer.o"
for unit in sbcmod stprk3 stprk3_stg fldread sbcice_if; do
  "$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$K0_RUN/build_sources/ppsrc/nemo/$unit.f90" -o "$scratch/$unit.base.o"
  "$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$scratch/mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$scratch/$unit.F90" -o "$scratch/$unit.observed.o"
done
# Exercise the imported round-328 whole-build proof on a dry K0 build made
# from the real ppsrc/objects plus exactly these observers.
mkdir "$scratch/dry-bld"
cp -as "$K0_BUILD/BLD/ppsrc" "$scratch/dry-bld/ppsrc"
cp -as "$K0_BUILD/BLD/obj" "$scratch/dry-bld/obj"
for name in arch_nemo.fcm cpp.fcm bld.cfg; do
  sed "s/$K0_CFG/$TARGET_CFG/g" "$K0_BUILD/BLD/$name" >"$scratch/dry-bld/$name"
done
for unit in sbcmod stprk3 stprk3_stg fldread sbcice_if; do
  unlink "$scratch/dry-bld/ppsrc/nemo/$unit.f90"
  unlink "$scratch/dry-bld/obj/$unit.o"
  cp "$scratch/$unit.F90" "$scratch/dry-bld/ppsrc/nemo/$unit.f90"
  cp "$scratch/$unit.observed.o" "$scratch/dry-bld/obj/$unit.o"
done
cp "$SURFACE_WRITER" "$scratch/dry-bld/ppsrc/nemo/round340_surface_writer.f90"
cp "$STATE_WRITER" "$scratch/dry-bld/ppsrc/nemo/round340_state_writer.f90"
cp "$scratch/surface-writer.o" "$scratch/dry-bld/obj/round340_surface_writer.o"
cp "$scratch/state-writer.o" "$scratch/dry-bld/obj/round340_state_writer.o"
"$PY" - "$scratch/dry-bld" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),{'sbcmod':Path(sys.argv[3]),'stprk3':Path(sys.argv[4]),'stprk3_stg':Path(sys.argv[5]),'fldread':Path(sys.argv[6]),'sbcice_if':Path(sys.argv[7])},('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[8],ref_cfg=sys.argv[9])
print('R340_SAME_PROGRAM_DRY_BUILD_PASS')
PY
unlink "$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
cp "$K0_BUILD/BLD/ppsrc/nemo/iceupdate.f90" "$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
printf '! unregistered plant\n' >>"$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
plant_must_refuse same-program-build "$PY" - "$scratch/dry-bld" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),{'sbcmod':Path(sys.argv[3]),'stprk3':Path(sys.argv[4]),'stprk3_stg':Path(sys.argv[5]),'fldread':Path(sys.argv[6]),'sbcice_if':Path(sys.argv[7])},('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[8],ref_cfg=sys.argv[9])
PY
cp "$scratch/sbcmod.F90" "$scratch/sbcmod.arithmetic-plant.F90"
sed -i 's/zwdht = ssh/zwdht = 1.0001_wp * ssh/' "$scratch/sbcmod.arithmetic-plant.F90"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$scratch/mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$scratch/sbcmod.arithmetic-plant.F90" -o "$scratch/sbcmod.arithmetic-plant.o"
"$PY" - "$scratch" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1])
for name,pattern in [('sbcmod','MOD_sbc$'),('stprk3','MOD_stp_rk3$'),('stprk3_stg','MOD_stp_rk3_stg$'),('fldread','MOD_fld_read$'),('sbcice_if','MOD_sbc_ice_if$')]:
    census=c.compare_fp_census(r/f'{name}.base.o',r/f'{name}.observed.o',function_pattern=pattern)
    print('R340_PRODUCTION_FP_CENSUS',name,census)
try: c.compare_fp_census(r/'sbcmod.base.o',r/'sbcmod.arithmetic-plant.o',function_pattern='MOD_sbc$')
except c.GateError: print('PREFLIGHT_PLANT_FIRED arithmetic-census')
else: raise AssertionError('arithmetic census plant stayed green')
PY
# Execute an actual wrapper compiled from the patched K0 sbcmod with bounds
# checking.  Linker section GC drops sbcmod's unrelated production callees;
# the stubs supply storage only, while both scientific units compile against
# the admitted K0 module interfaces.
cat >"$scratch/fixture_stubs.F90" <<'EOF'
MODULE par_oce
 USE,INTRINSIC::iso_fortran_env,ONLY:real64
 INTEGER,PARAMETER::wp=real64
 INTEGER::jpi=6,jpj=5
END MODULE
MODULE lib_mpp; INTEGER::mpprank=0; END MODULE
MODULE sbc_oce
 USE par_oce,ONLY:wp
 REAL(wp),ALLOCATABLE::sst_m(:,:),sss_m(:,:),ssu_m(:,:),ssv_m(:,:),ssh_m(:,:),e3t_m(:,:),frq_m(:,:)
 REAL(wp),ALLOCATABLE::utau(:,:),vtau(:,:),utauU(:,:),vtauV(:,:),qsr(:,:),qns(:,:),emp(:,:),sfx(:,:),taum(:,:),wndm(:,:)
END MODULE
EOF
mkdir "$scratch/stub-mod" "$scratch/check-mod"
"$FC" -c -J "$scratch/stub-mod" "$scratch/fixture_stubs.F90" -o "$scratch/fixture-stubs.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections \
  -I "$K0_BUILD/BLD/inc" -J "$scratch/check-mod" -c "$SURFACE_WRITER" -o "$scratch/surface-check.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections \
  -I "$scratch/check-mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/check-mod" \
  -c "$scratch/sbcmod.F90" -o "$scratch/sbcmod-check.o"
"$FC" -fdefault-real-8 -fcheck=bounds -I "$scratch/check-mod" -I "$scratch/stub-mod" \
  -J "$scratch/check-mod" -c "$SURFACE_FIXTURE" -o "$scratch/fixture-check.o"
"$MPIFC" -Wl,--gc-sections "$scratch/fixture-stubs.o" "$scratch/surface-check.o" \
  "$scratch/sbcmod-check.o" "$scratch/fixture-check.o" -o "$scratch/surface-fixture"
for rank in 0 1; do
  (cd "$scratch" && ./surface-fixture "$rank") >"$scratch/fixture-$rank.log"
  grep -E "R340_SURFACE_FIXTURE_PASS[[:space:]]+$rank$" "$scratch/fixture-$rank.log" >/dev/null \
    || refuse "surface bounds fixture failed for rank $rank"
done
"$PY" - "$scratch" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round340_standalone_gate as g
root=Path(sys.argv[1])
for path in sorted(root.glob('oracle_r340_surface_*.bin')): g.inspect_exact_record(path)
print('R340_PATCHED_SOURCE_BOUNDS_FIXTURE_PASS ranks=2 signed_zero=true')
PY
"$PY" - "$ICE_RUN" "$K0_RUN" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round340_standalone_gate as g
g.verify_round339_prerequisites(Path(sys.argv[1]),Path(sys.argv[2]))
print('R340_PREREQUISITES_PASS')
PY
"$PY" - "$NAMELIST_REF" "$K0_RUN/staged.namelist_cfg" "$K0_RUN/ocean.output" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_k0_smoke_gate as k0
reference, staged, output = map(Path,sys.argv[1:])
c.validate_group_membership(reference,k0.GROUPS)
c.require('E R R O R' not in output.read_text(errors='replace'),'K0 parent contains E R R O R')
planted=dict(k0.GROUPS); planted['nn_mxlice']='namzdf'
try: c.validate_group_membership(reference,planted)
except c.GateError: print('PREFLIGHT_PLANT_FIRED namelist-group')
else: raise AssertionError('misplaced namelist variable stayed green')
c.require(staged.is_file(),'K0 staged namelist is absent')
print('R340_K0_NAMELIST_SCHEMA_PASS')
PY
"$PY" -m pytest -q -p no:cacheprovider "$TEST"

# Produce every driver used by --run, syntax-check it, then exercise its real
# refusal/retention/success flow while replacing only the NEMO executable.
cat >"$scratch/execute.sh" <<'DRIVER'
#!/usr/bin/env bash
set -Eeuo pipefail
run=$1; nemo=$2; source_run=$3; k0_run=$4
mkdir "$run"
armed=1
trap 'rc=$?; if [[ $rc -ne 0 && $armed -eq 1 ]]; then mkdir -p "$(dirname "$run")/refused"; mv "$run" "$(dirname "$run")/refused/$(basename "$run").failed.$$"; fi; exit $rc' EXIT
while read -r _ name; do cp -a "$source_run/$name" "$run/$name"; done <"$source_run/deck_files.sha256"
while read -r _ name; do cp -a "$source_run/$name" "$run/$name"; done <"$source_run/input_files.sha256"
cp "$k0_run/staged.namelist_cfg" "$run/namelist_cfg"
cp "$k0_run/ice_cover_clim.nc" "$run/ice_cover_clim.nc"
cp "$nemo" "$run/nemo"
if [[ ${R340_PREFLIGHT_DIRECT:-0} == 1 ]]; then
  (cd "$run" && { ./nemo; ./nemo; } >run.user.stdout.log 2>&1)
else
  (cd "$run" && timeout --kill-after=10 900 mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1)
fi
grep 'STOP 0' "$run/run.user.stdout.log" >/dev/null
! grep -Eq 'E R R O R|STOP [1-9]' "$run/run.user.stdout.log"
armed=0
DRIVER
chmod +x "$scratch/execute.sh"; bash -n "$scratch/execute.sh"
cat >"$scratch/nemo-stub" <<'STUB'
#!/usr/bin/env bash
printf 'STOP 0\n'
STUB
chmod +x "$scratch/nemo-stub"
mkdir "$scratch/source-template" "$scratch/k0-template"
: >"$scratch/source-template/deck_files.sha256"; : >"$scratch/source-template/input_files.sha256"
: >"$scratch/k0-template/staged.namelist_cfg"; : >"$scratch/k0-template/ice_cover_clim.nc"
R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-success" "$scratch/nemo-stub" "$scratch/source-template" "$scratch/k0-template"
sed 's/printf '\''STOP 0/printf '\''STOP 1/' "$scratch/nemo-stub" >"$scratch/nemo-refuse"; chmod +x "$scratch/nemo-refuse"
plant_must_refuse generated-flow env R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-refuse" "$scratch/nemo-refuse" "$scratch/source-template" "$scratch/k0-template"
find "$scratch/refused" -maxdepth 1 -type d -name 'stub-refuse.failed.*' -print -quit | grep . >/dev/null \
  || refuse 'generated driver did not retain its refused run'
sed 's/set -Eeuo pipefail/set -Eeuo pipefail\nif then/' "$scratch/execute.sh" >"$scratch/syntax.plant"
plant_must_refuse generated-syntax bash -n "$scratch/syntax.plant"
synthetic=$scratch/admission-control
mkdir -p "$synthetic/run/build_sources/MY_SRC" "$synthetic/run/build_sources/ppsrc/nemo" "$synthetic/refused"
printf 'synthetic nemo\n' >"$synthetic/run/nemo"
printf 'program synthetic\n' >"$synthetic/run/build_sources/ppsrc/nemo/synthetic.f90"
"$PY" - "$synthetic/run" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1]); files=[p for p in r.rglob('*') if p.is_file()]
c.write_manifest(r/'producer_manifest.json',producer_commit=sys.argv[2],binary=r/'nemo',files=files,root=r,build_source_root=r/'build_sources')
PY
mv "$synthetic/run" "$synthetic/refused/run.failed.control"
R340_ADMISSION_IDENTITY_ONLY=1 R340_ADMISSION_TEST_ROOT=$synthetic \
 R340_EXISTING_RUN=$synthetic/refused/run.failed.control "$0" --admit-existing \
 >"$scratch/retained-admission.log"
grep -F R340_RETAINED_IDENTITY_ADMISSION_PASS "$scratch/retained-admission.log" >/dev/null \
 || refuse 'generated retained --admit-existing flow failed'
printf 'PREFLIGHT_RETAINED_ADMISSION_PASS route=refused\n'
printf 'checksum plant\n' >>"$synthetic/refused/run.failed.control/nemo"
plant_must_refuse retained-checksum env R340_ADMISSION_IDENTITY_ONLY=1 \
 R340_ADMISSION_TEST_ROOT=$synthetic R340_EXISTING_RUN=$synthetic/refused/run.failed.control \
 "$0" --admit-existing

admit(){
 local root=$1
 local expected_commit
 expected_commit=$(git -C "$REPO" rev-parse HEAD)
 "$PY" - "$root" "$expected_commit" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round340_standalone_gate as g
r=Path(sys.argv[1]); c.verify_manifest(r/'producer_manifest.json',expected_commit=sys.argv[2])
inventory=g.inspect_acquisition_records(r)
print('ORCA2_ROUND340_S4_ACQUISITION_PASS',r,inventory)
PY
}
if [[ "$MODE" == --preflight-only ]]; then printf 'ORCA2_ROUND340_S4_PREFLIGHT_READY\n'; exit 0; fi
if [[ "$MODE" == --admit-existing ]]; then admit "${R340_EXISTING_RUN:-$RUN_A}"; exit 0; fi

[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse 'run requires a clean committed tree'
[[ ! -e "$TARGET_BUILD" && ! -e "$RUN_A" && ! -e "$RUN_B" ]] || refuse 'retain/move existing build and runs before retry'
[[ $(free -g | awk '/^Mem:/{print $7}') -ge 150 ]] || refuse 'under 150 GB available memory'
mkdir -p "$EVIDENCE"
cd "$NEMO_ROOT"
./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -z $(find "$TARGET_BUILD/MY_SRC" -mindepth 1 -maxdepth 1 -print -quit) ]] \
  || refuse 'new K0-derived MY_SRC is not empty'
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcmod.f90" "$TARGET_BUILD/MY_SRC/sbcmod.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3.f90" "$TARGET_BUILD/MY_SRC/stprk3.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3_stg.f90" "$TARGET_BUILD/MY_SRC/stprk3_stg.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/fldread.f90" "$TARGET_BUILD/MY_SRC/fldread.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcice_if.f90" "$TARGET_BUILD/MY_SRC/sbcice_if.F90"
cp "$SURFACE_WRITER" "$STATE_WRITER" "$TARGET_BUILD/MY_SRC/"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$SURFACE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$STATE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$STAGE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$FLD_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$IIF_PATCH"
cp -a "$K0_BUILD/EXP00/." "$TARGET_BUILD/EXP00/"
cp "$K0_BUILD/cpp_$K0_CFG.fcm" "$TARGET_BUILD/cpp_$TARGET_CFG.fcm"
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
obs=$TARGET_BUILD/BLD/bin/nemo.exe; [[ -x "$obs" ]] || refuse 'instrumented build failed'
for run in "$RUN_A" "$RUN_B"; do
  "$scratch/execute.sh" "$run" "$obs" "$SOURCE_RUN" "$K0_RUN"
  cp "$K0_RUN/producer_manifest.json" "$run/k0_parent_manifest.json"
  mkdir -p "$run/build_sources/MY_SRC" "$run/build_sources/ppsrc"
  cp -a "$TARGET_BUILD/BLD/ppsrc/." "$run/build_sources/ppsrc/"
  cp -a "$TARGET_BUILD/MY_SRC/." "$run/build_sources/MY_SRC/"
  "$PY" - "$run" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1]); files=[p for p in r.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
c.write_manifest(r/'producer_manifest.json',producer_commit=sys.argv[2],binary=r/'nemo',files=files,root=r,build_source_root=r/'build_sources')
PY
done
admit "$RUN_A"
admit "$RUN_B"
# The generic round-328 proof strips exactly the five observer patches and two writers.
"$PY" - "$TARGET_BUILD/BLD" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),{'sbcmod':Path(sys.argv[3]),'stprk3':Path(sys.argv[4]),'stprk3_stg':Path(sys.argv[5]),'fldread':Path(sys.argv[6]),'sbcice_if':Path(sys.argv[7])},('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[8],ref_cfg=sys.argv[9])
PY
cmp -s "$RUN_A/ORCA2_00000010_restart_0000.nc" "$K0_RUN/ORCA2_00000010_restart_0000.nc" || refuse 'observer changed rank-0 restart'
cmp -s "$RUN_A/ORCA2_00000010_restart_0001.nc" "$K0_RUN/ORCA2_00000010_restart_0001.nc" || refuse 'observer changed rank-1 restart'
cmp -s "$RUN_B/ORCA2_00000010_restart_0000.nc" "$K0_RUN/ORCA2_00000010_restart_0000.nc" || refuse 'twin observer changed rank-0 restart'
cmp -s "$RUN_B/ORCA2_00000010_restart_0001.nc" "$K0_RUN/ORCA2_00000010_restart_0001.nc" || refuse 'twin observer changed rank-1 restart'
diff <(cd "$RUN_A" && sha256sum oracle_r340_*.bin | sed 's|  .*\/|  |') <(cd "$RUN_B" && sha256sum oracle_r340_*.bin | sed 's|  .*\/|  |') >/dev/null || refuse 'reproducibility twins differ'
printf 'ORCA2_ROUND340_S4_TWINS_ADMITTED\n'
