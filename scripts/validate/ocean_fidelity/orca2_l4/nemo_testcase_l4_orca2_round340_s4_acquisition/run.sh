#!/usr/bin/env bash
# Operator-only exact S4 acquisition. Default is read-only preflight.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
refuse(){ printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
failure_step=initialization
report_failure(){
 local rc=$? command=$BASH_COMMAND
 trap - ERR
 printf 'REFUSE: %s failed (rc=%s; command=%s)\n' "$failure_step" "$rc" "$command" >&2
 exit "$rc"
}
trap report_failure ERR
case "$MODE" in --preflight-only|--run|--admit-existing) ;; *) refuse "unknown mode: $MODE";; esac
plant_must_refuse(){ local n=$1; shift; if "$@"; then refuse "plant stayed green: $n"; fi; printf 'PREFLIGHT_PLANT_FIRED %s\n' "$n"; }
quote_ocean_errors(){
 local log=$1 lines
 lines=$(grep -nF 'E R R O R' "$log" 2>/dev/null || true)
 [[ -z $lines ]] || printf 'R340_OCEAN_ERROR %s %s\n' "$log" "$lines" >&2
}
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
readonly BLK_PATCH=$here/sbcblk_round340.patch
readonly FWB_PATCH=$here/sbcfwb_round340.patch
readonly SURFACE_WRITER=$here/round340_surface_writer.F90
readonly STATE_WRITER=$here/round340_state_writer.F90
readonly SURFACE_FIXTURE=$here/round340_surface_fixture.F90
readonly NAMELIST_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ref
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for f in "$GATE" "$TEST" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$BLK_PATCH" "$FWB_PATCH" "$SURFACE_WRITER" "$STATE_WRITER" "$SURFACE_FIXTURE" \
 "$K0_RUN/producer_manifest.json" "$ICE_RUN/producer_manifest.json" "$NAMELIST_REF"; do [[ -f "$f" ]] || refuse "missing $f"; done

admit_one(){
 local root=$1 parent=${2:-$K0_RUN} expected_commit
 failure_step="admission of $root"
 expected_commit=$(git -C "$REPO" rev-parse HEAD)
 "$PY" - "$root" "$parent" "$expected_commit" "$REPO" <<'PY'
import json,subprocess,sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round340_standalone_gate as g
r,parent=map(Path,sys.argv[1:3])
producer=json.loads((r/'producer_manifest.json').read_text()).get('producer_commit','')
c.verify_manifest(r/'producer_manifest.json',expected_commit=producer)
c.require(subprocess.run(
    ['git','-C',sys.argv[4],'merge-base','--is-ancestor',producer,sys.argv[3]]
).returncode == 0,'manifest producer is not an ancestor of the admission commit')
for log_name in ('run.user.stdout.log','ocean.output'):
    text=(r/log_name).read_text(errors='replace')
    if 'E R R O R' in text:
        for line_number,line in enumerate(text.splitlines(),start=1):
            if 'E R R O R' in line:
                print(f'R340_OCEAN_ERROR {r/log_name}:{line_number}:{line}',file=sys.stderr)
    g.require('E R R O R' not in text and not any(
        f'STOP {code}' in text for code in range(1,10)),
        f'admission log contains an error marker: {log_name}')
stdout=(r/'run.user.stdout.log').read_text(errors='replace')
g.require('STOP 0' in stdout,'admission stdout has no clean STOP 0')
inventory=g.inspect_acquisition_records(r)
for rank in (0,1):
    name=f'ORCA2_00000010_restart_{rank:04d}.nc'
    g.require((r/name).read_bytes()==(parent/name).read_bytes(),f'restart changed: {name}')
print('ORCA2_ROUND340_S4_ACQUISITION_PASS',r,inventory)
PY
}

finalize_interrupted_manifest(){
 local root=$1 expected_commit
 failure_step="finalize interrupted manifest for $root"
 expected_commit=$(git -C "$REPO" rev-parse HEAD)
 "$PY" - "$root" "$expected_commit" "$REPO" <<'PY'
import json,subprocess,sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1]); current=sys.argv[2]; repo=sys.argv[3]
manifest=r/'producer_manifest.json'
payload=json.loads(manifest.read_text())
files=payload.get('files',{})
if 'run.user.stdout.log' in files and 'ocean.output' in files:
    return_code=0
else:
    prior=payload.get('producer_commit','')
    c.verify_manifest(manifest,expected_commit=prior)
    return_code=subprocess.run(
        ['git','-C',repo,'merge-base','--is-ancestor',prior,current]).returncode
    c.require(return_code == 0,'interrupted manifest producer is not an ancestor')
    actual=[p for p in r.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
    c.write_manifest(manifest,producer_commit=current,binary=r/'nemo',files=actual,
                     root=r,build_source_root=r/'build_sources')
    print('R340_INTERRUPTED_MANIFEST_FINALIZED',r)
PY
}

admit_pair(){
 local left=$1 right=$2 parent=${3:-$K0_RUN}
 admit_one "$left" "$parent"
 admit_one "$right" "$parent"
 diff <(cd "$left" && sha256sum oracle_r340_*.bin | sed 's|  .*\/|  |') \
      <(cd "$right" && sha256sum oracle_r340_*.bin | sed 's|  .*\/|  |') >/dev/null \
   || refuse 'reproducibility twins differ'
 printf 'ORCA2_ROUND340_S4_TWINS_ADMITTED\n'
}

admit_existing_route(){
 local existing_a=${R340_EXISTING_RUN_A:-$RUN_A}
 local existing_b=${R340_EXISTING_RUN_B:-$RUN_B}
 local parent=${R340_EXISTING_PARENT:-$K0_RUN}
 [[ -d $existing_a ]] || refuse "retained arm A is absent: $existing_a"
 finalize_interrupted_manifest "$existing_a"
 if [[ -d $existing_b ]]; then
  finalize_interrupted_manifest "$existing_b"
  admit_pair "$existing_a" "$existing_b" "$parent"
  return
 fi
 admit_one "$existing_a" "$parent"
 if [[ $existing_a != "$RUN_A" ]]; then
  [[ ! -e $RUN_A ]] || refuse "canonical arm A already exists: $RUN_A"
  mv "$existing_a" "$RUN_A"
 fi
 printf 'ORCA2_ROUND340_S4_ARM_A_ADMITTED %s\n' "$RUN_A"
}

check_placement(){
 local f=$1
 [[ $(grep -Ec 'CALL r340_surface_frame\(kt,([0-9]|1[01])\)' "$f") -eq 12 ]] || return 1
 awk '/CALL sbc_ssm/{m=NR}/SELECT CASE\( nsbc \)/{s1=NR}/SELECT CASE\( nn_ice \)/{s3=NR}/CALL sbc_rnf/{r=NR}/CALL sbc_ssr/{q=NR}/CALL sbc_fwb/{w=NR}/CALL r340_surface_frame\(kt,11\)/{x=NR} END{exit !(m<s1&&s1<s3&&s3<r&&r<q&&q<w&&w<x)}' "$f"
}
check_other_placements(){
 local d=$1
 [[ $(grep -Fc 'CALL r340_state_frame' "$d/stprk3.F90") -eq 2 ]] || return 1
 [[ $(grep -Fc 'CALL r340_state_frame' "$d/stprk3_stg.F90") -eq 2 ]] || return 1
 [[ $(grep -Fc 'CALL r340_fld_frame' "$d/fldread.F90") -eq 2 ]] || return 1
 [[ $(grep -Fc 'CALL r340_iif_frame' "$d/sbcice_if.F90") -eq 4 ]] || return 1
 [[ $(grep -Ec 'CALL r340_s1_(turb|flux)_frame' "$d/sbcblk.F90") -eq 4 ]] || return 1
 awk '/^[[:space:]]*IF.*CALL isf_stp/{a=NR}/CALL r340_state_frame/{if(!b)b=NR}/CALL sbc        /{c=NR} END{exit !(a<b&&b<c)}' "$d/stprk3.F90" || return 1
 awk '/CALL r340_state_frame/{if(!a)a=NR}/DEALLOCATE\( zFu/{b=NR} END{exit !(a<b)}' "$d/stprk3_stg.F90" || return 1
 awk '/end loop over field/{a=NR}/CALL r340_fld_frame/{b=NR} END{exit !(a<b)}' "$d/fldread.F90" || return 1
 awk '/CALL fld_read/{f=NR}/r340_iif_frame\(kt,0/{a=NR}/CALL eos_fzp/{e=NR}/r340_iif_frame\(kt,1/{b=NR}/r340_iif_frame\(kt,2/{c=NR}/r340_iif_frame\(kt,3/{d=NR} END{exit !(f<a&&a<e&&e<b&&b<c&&c<d)}' "$d/sbcice_if.F90"
}

scratch=$(mktemp -d /tmp/orca2-r340-preflight.XXXXXX)
active_runs=()
cleanup(){
 local rc=$?; trap - EXIT
 if [[ $rc -ne 0 ]]; then
  for active_run in "${active_runs[@]}"; do
   if [[ -d $active_run ]]; then
    mkdir -p "$EVIDENCE/refused"
    mv "$active_run" "$EVIDENCE/refused/$(basename "$active_run").failed.$$"
    printf 'R340_REFUSED_RUN_RETAINED %s\n' "$EVIDENCE/refused/$(basename "$active_run").failed.$$" >&2
   fi
  done
 fi
 printf 'R340_SCRATCH_RETAINED %s rc=%s\n' "$scratch" "$rc" >&2
 exit "$rc"
}
trap cleanup EXIT
if [[ ${R340_PREFLIGHT_ADMIT_DIRECT:-0} == 1 && "$MODE" == --admit-existing ]]; then
 admit_existing_route
 exit 0
fi
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcmod.f90" "$scratch/sbcmod.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3.f90" "$scratch/stprk3.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3_stg.f90" "$scratch/stprk3_stg.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/fldread.f90" "$scratch/fldread.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcice_if.f90" "$scratch/sbcice_if.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcblk.f90" "$scratch/sbcblk.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcfwb.f90" "$scratch/sbcfwb.F90"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$SURFACE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$STATE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$STAGE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$FLD_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$IIF_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$BLK_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$FWB_PATCH"
"$PY" - "$scratch" "$K0_RUN/build_sources/ppsrc/nemo" \
  "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$BLK_PATCH" "$FWB_PATCH" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
scratch, reference = Path(sys.argv[1]), Path(sys.argv[2])
units = dict(zip(('sbcmod','stprk3','stprk3_stg','fldread','sbcice_if','sbcblk','sbcfwb'), map(Path,sys.argv[3:]), strict=True))
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
print('R340_SAME_PROGRAM_PPSRC_PASS units=7')
PY
check_placement "$scratch/sbcmod.F90" || refuse 'surface observers moved'
check_other_placements "$scratch" || refuse 'state/fld/ice-if observers moved'
cp "$scratch/sbcmod.F90" "$scratch/placement.plant"
sed -i '/r340_surface_frame(kt,5)/d' "$scratch/placement.plant"
plant_must_refuse placement check_placement "$scratch/placement.plant"
for unit in stprk3 stprk3_stg fldread sbcice_if sbcblk; do
  mkdir "$scratch/$unit.plant"
  cp "$scratch"/{stprk3,stprk3_stg,fldread,sbcice_if,sbcblk}.F90 "$scratch/$unit.plant/"
  sed -i '0,/CALL r340_/d' "$scratch/$unit.plant/$unit.F90"
  plant_must_refuse "placement-$unit" check_other_placements "$scratch/$unit.plant"
done
mkdir "$scratch/base-mod" "$scratch/mod"
for unit in sbcmod stprk3 stprk3_stg fldread sbcice_if sbcblk sbcfwb; do
  "$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/base-mod" -c "$K0_RUN/build_sources/ppsrc/nemo/$unit.f90" -o "$scratch/$unit.base.o"
done
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$scratch/sbcfwb.F90" -o "$scratch/sbcfwb.observed.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$scratch/mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$SURFACE_WRITER" -o "$scratch/surface-writer.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$STATE_WRITER" -o "$scratch/state-writer.o"
for unit in fldread sbcblk sbcice_if sbcmod stprk3 stprk3_stg; do
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
for unit in sbcmod stprk3 stprk3_stg fldread sbcice_if sbcblk sbcfwb; do
  unlink "$scratch/dry-bld/ppsrc/nemo/$unit.f90"
  unlink "$scratch/dry-bld/obj/$unit.o"
  cp "$scratch/$unit.F90" "$scratch/dry-bld/ppsrc/nemo/$unit.f90"
  cp "$scratch/$unit.observed.o" "$scratch/dry-bld/obj/$unit.o"
done
cp "$SURFACE_WRITER" "$scratch/dry-bld/ppsrc/nemo/round340_surface_writer.f90"
cp "$STATE_WRITER" "$scratch/dry-bld/ppsrc/nemo/round340_state_writer.f90"
cp "$scratch/surface-writer.o" "$scratch/dry-bld/obj/round340_surface_writer.o"
cp "$scratch/state-writer.o" "$scratch/dry-bld/obj/round340_state_writer.o"
"$PY" - "$scratch/dry-bld" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$BLK_PATCH" "$FWB_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),dict(zip(('sbcmod','stprk3','stprk3_stg','fldread','sbcice_if','sbcblk','sbcfwb'),map(Path,sys.argv[3:10]),strict=True)),('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[10],ref_cfg=sys.argv[11])
print('R340_SAME_PROGRAM_DRY_BUILD_PASS')
PY
unlink "$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
cp "$K0_BUILD/BLD/ppsrc/nemo/iceupdate.f90" "$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
printf '! unregistered plant\n' >>"$scratch/dry-bld/ppsrc/nemo/iceupdate.f90"
plant_must_refuse same-program-build "$PY" - "$scratch/dry-bld" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$BLK_PATCH" "$FWB_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),dict(zip(('sbcmod','stprk3','stprk3_stg','fldread','sbcice_if','sbcblk','sbcfwb'),map(Path,sys.argv[3:10]),strict=True)),('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[10],ref_cfg=sys.argv[11])
PY
cp "$scratch/sbcmod.F90" "$scratch/sbcmod.arithmetic-plant.F90"
sed -i 's/zwdht = ssh/zwdht = 1.0001_wp * ssh/' "$scratch/sbcmod.arithmetic-plant.F90"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -I "$scratch/mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/mod" -c "$scratch/sbcmod.arithmetic-plant.F90" -o "$scratch/sbcmod.arithmetic-plant.o"
"$PY" - "$scratch" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1])
for name,pattern in [('sbcmod','MOD_sbc$'),('stprk3','MOD_stp_rk3$'),('stprk3_stg','MOD_stp_rk3_stg$'),('fldread','MOD_fld_read$'),('sbcice_if','MOD_sbc_ice_if$'),('sbcblk','MOD_(sbc_blk|blk_oce_[12])$'),('sbcfwb','MOD_sbc_fwb$')]:
    census=c.compare_fp_census(r/f'{name}.base.o',r/f'{name}.observed.o',function_pattern=pattern)
    print('R340_PRODUCTION_FP_CENSUS',name,census)
try: c.compare_fp_census(r/'sbcmod.base.o',r/'sbcmod.arithmetic-plant.o',function_pattern='MOD_sbc$')
except c.GateError: print('PREFLIGHT_PLANT_FIRED arithmetic-census')
else: raise AssertionError('arithmetic census plant stayed green')
PY
# Execute an actual wrapper compiled from the patched K0 sbcmod with bounds
# checking.  Linker section GC drops unrelated production callees; explicit
# probes keep and execute one section from each of the seven patched objects.
# The stubs supply storage only; scientific units use admitted K0 interfaces.
"$PY" - "$K0_RUN/build_sources/ppsrc/nemo/sbc_oce.f90" "$K0_RUN/ocean.output" "$scratch/fixture_extents.F90" <<'PY'
import re,sys
from pathlib import Path
source,output,target=map(Path,sys.argv[1:])
allocation=re.search(
    r'ALLOCATE\(\s*qns\s*\(Nis0-\(0\):Nie0\+\(0\),Njs0-\(0\):Nje0\+\(0\)\)',
    source.read_text())
if allocation is None:
    raise AssertionError('sbc_oce A2D(0) allocation moved')
matches=re.findall(r'halo width .* nn_hls\s*=\s*(\d+)',output.read_text())
if matches != ['2']:
    raise AssertionError(f'deck nn_hls moved: {matches}')
h=int(matches[0])
target.write_text(f'''MODULE dom_oce
 USE par_oce,ONLY:nn_hls,Nis0,Nie0,Njs0,Nje0
END MODULE
''')
print('R340_SOURCE_DERIVED_EXTENTS_PASS allocation=A2D(0) nn_hls=2 bounds=3:jpi-2,3:jpj-2')
PY
cat >"$scratch/fixture_stubs.F90" <<'EOF'
MODULE par_oce
 USE,INTRINSIC::iso_fortran_env,ONLY:real64
 INTEGER,PARAMETER::wp=real64
 INTEGER,PARAMETER::jpi=6,jpj=5,jpk=3,jpts=2
 INTEGER,PARAMETER::nn_hls=2
 INTEGER::Nis0=1+nn_hls,Nie0=jpi-nn_hls,Njs0=1+nn_hls,Nje0=jpj-nn_hls
END MODULE
MODULE lib_mpp; INTEGER::mpprank=0; END MODULE
MODULE sbc_oce
 USE par_oce,ONLY:wp
 REAL(wp),ALLOCATABLE::sst_m(:,:),sss_m(:,:),ssu_m(:,:),ssv_m(:,:),ssh_m(:,:),e3t_m(:,:),frq_m(:,:)
 REAL(wp),ALLOCATABLE::utau(:,:),vtau(:,:),utauU(:,:),vtauV(:,:),qsr(:,:),qns(:,:),emp(:,:),sfx(:,:),taum(:,:),wndm(:,:),rnf(:,:),fr_i(:,:)
END MODULE
MODULE sbcrnf; USE par_oce,ONLY:wp; REAL(wp),ALLOCATABLE::rnf_tsc(:,:,:); END MODULE
MODULE sbcssr; USE par_oce,ONLY:wp; REAL(wp),ALLOCATABLE::erp(:,:),qrp(:,:),coefice(:,:); END MODULE
EOF
mkdir "$scratch/stub-mod" "$scratch/check-mod"
"$FC" -c -J "$scratch/stub-mod" "$scratch/fixture_stubs.F90" -o "$scratch/fixture-stubs.o"
"$FC" -c -I "$scratch/stub-mod" -J "$scratch/stub-mod" "$scratch/fixture_extents.F90" -o "$scratch/fixture-extents.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections \
  -I "$scratch/stub-mod" -J "$scratch/check-mod" -c "$SURFACE_WRITER" -o "$scratch/surface-check.o"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections \
  -I "$scratch/stub-mod" -J "$scratch/check-mod" -c "$STATE_WRITER" -o "$scratch/state-check.o"
# Checklist 5: every patched production unit is compiled under bounds checking
# against the admitted K0 interfaces; the linked fixture executes both writer
# modules with rank-distinct and signed-zero data.
mkdir "$scratch/bounds-mod"
"$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections -I "$K0_BUILD/BLD/inc" -J "$scratch/bounds-mod" -c "$scratch/sbcfwb.F90" -o "$scratch/sbcfwb-bounds.o"
for unit in fldread sbcblk sbcice_if sbcmod stprk3 stprk3_stg; do
  "$FC" -fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize -fcheck=bounds -ffunction-sections -I "$scratch/bounds-mod" -I "$scratch/mod" -I "$K0_BUILD/BLD/inc" -J "$scratch/bounds-mod" -c "$scratch/$unit.F90" -o "$scratch/$unit-bounds.o"
done
"$FC" -fdefault-real-8 -fcheck=bounds -I "$scratch/bounds-mod" -I "$scratch/check-mod" -I "$scratch/stub-mod" \
  -J "$scratch/check-mod" -c "$SURFACE_FIXTURE" -o "$scratch/fixture-check.o"
"$MPIFC" -Wl,--gc-sections "$scratch/fixture-stubs.o" "$scratch/fixture-extents.o" "$scratch/surface-check.o" \
  "$scratch/state-check.o" "$scratch/fldread-bounds.o" "$scratch/sbcblk-bounds.o" \
  "$scratch/sbcice_if-bounds.o" "$scratch/sbcmod-bounds.o" \
  "$scratch/stprk3-bounds.o" "$scratch/stprk3_stg-bounds.o" \
  "$scratch/sbcfwb-bounds.o" "$scratch/fixture-check.o" -o "$scratch/surface-fixture"
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
for path in sorted(root.glob('oracle_r340_state_*.bin')): g.inspect_exact_record(path)
for path in sorted(root.glob('oracle_r340_s1[tf]_*.bin')): g.inspect_exact_record(path)
fld=sorted(root.glob('oracle_r340_fld_*.bin'))
iif=sorted(root.glob('oracle_r340_iif_*.bin'))
for path in fld+iif: g.inspect_exact_record(path)
g.require(len(list(root.glob('oracle_r340_surface_*.bin')))==24,'surface probe inventory moved')
g.require(len(list(root.glob('oracle_r340_state_*.bin')))==4,'state probe inventory moved')
g.require(len(list(root.glob('oracle_r340_s1[tf]_*.bin')))==4,'S1 probe inventory moved')
g.require(len(fld)==2 and len(iif)==8,'patched fld/iif probe inventory moved')
print('R340_PATCHED_SOURCE_BOUNDS_FIXTURE_PASS units=7 linked=7 executed=7 writers=6 surface_kids=12 iif_kids=4 ranks=2 signed_zero=true')
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
ORCA2_R340_REAL_SMOKE=1 "$PY" -m pytest -q -p no:cacheprovider "$TEST" \
 -k real_rung10_composed_card_ten_steps_cpu
ORCA2_R340_REAL_GRADIENT=1 "$PY" -m pytest -q -p no:cacheprovider "$TEST" \
 -k real_composed_card_n1_reverse_equals_forward

# Produce every driver used by --run, syntax-check it, then exercise its real
# refusal/retention/success flow while replacing only the NEMO executable.
cat >"$scratch/execute.sh" <<'DRIVER'
#!/usr/bin/env bash
set -Eeuo pipefail
run=$1; nemo=$2; source_run=$3; k0_run=$4; target_build=$5; commit=$6; py=$7
step=setup
refuse(){ printf 'REFUSE: execute/%s: %s\n' "$step" "$*" >&2; exit 70; }
report_failure(){
 rc=$?; command=$BASH_COMMAND; trap - ERR
 printf 'REFUSE: execute/%s failed (rc=%s; command=%s)\n' "$step" "$rc" "$command" >&2
 exit "$rc"
}
trap report_failure ERR
quote_ocean_errors(){
 local log=$1 lines
 lines=$(grep -nF 'E R R O R' "$log" 2>/dev/null || true)
 [[ -z $lines ]] || printf 'R340_OCEAN_ERROR %s %s\n' "$log" "$lines" >&2
}
mkdir "$run"
armed=1
trap 'rc=$?; if [[ $rc -ne 0 && $armed -eq 1 ]]; then mkdir -p "$(dirname "$run")/refused"; mv "$run" "$(dirname "$run")/refused/$(basename "$run").failed.$$"; fi; exit $rc' EXIT
while read -r _ name; do cp -a "$source_run/$name" "$run/$name"; done <"$source_run/deck_files.sha256"
while read -r _ name; do cp -a "$source_run/$name" "$run/$name"; done <"$source_run/input_files.sha256"
cp "$k0_run/staged.namelist_cfg" "$run/namelist_cfg"
cp "$k0_run/ice_cover_clim.nc" "$run/ice_cover_clim.nc"
cp "$nemo" "$run/nemo"
cp "$k0_run/producer_manifest.json" "$run/k0_parent_manifest.json"
mkdir -p "$run/build_sources/MY_SRC" "$run/build_sources/ppsrc"
cp -a "$target_build/MY_SRC/." "$run/build_sources/MY_SRC/"
cp -a "$target_build/BLD/ppsrc/." "$run/build_sources/ppsrc/"
"$py" - "$run" "$commit" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1]); files=[p for p in r.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
c.write_manifest(r/'producer_manifest.json',producer_commit=sys.argv[2],binary=r/'nemo',files=files,root=r,build_source_root=r/'build_sources')
PY
step='NEMO launch'
if [[ ${R340_PREFLIGHT_DIRECT:-0} == 1 ]]; then
  (cd "$run" && { ./nemo; ./nemo; } >run.user.stdout.log 2>&1)
else
  (cd "$run" && timeout --kill-after=10 900 mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1)
fi
step='stdout STOP 0 check'
grep -F 'STOP 0' "$run/run.user.stdout.log" >/dev/null
step='ocean.output existence check'
[[ -f "$run/ocean.output" ]]
step='runtime error scan'
if grep -Eq 'E R R O R|STOP [1-9]' "$run/run.user.stdout.log" "$run/ocean.output"; then
  quote_ocean_errors "$run/run.user.stdout.log"
  quote_ocean_errors "$run/ocean.output"
  refuse 'error marker found'
fi
step='final manifest write'
"$py" - "$run" "$commit" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
r=Path(sys.argv[1]); files=[p for p in r.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
c.write_manifest(r/'producer_manifest.json',producer_commit=sys.argv[2],binary=r/'nemo',files=files,root=r,build_source_root=r/'build_sources')
PY
DRIVER
chmod +x "$scratch/execute.sh"; bash -n "$scratch/execute.sh"
cat >"$scratch/nemo-stub" <<'STUB'
#!/usr/bin/env bash
printf 'STOP 0\n'
printf 'STOP 0\n' >ocean.output
STUB
chmod +x "$scratch/nemo-stub"
mkdir -p "$scratch/source-template" "$scratch/k0-template" "$scratch/target-template/MY_SRC" "$scratch/target-template/BLD/ppsrc"
: >"$scratch/source-template/deck_files.sha256"; : >"$scratch/source-template/input_files.sha256"
: >"$scratch/k0-template/staged.namelist_cfg"; : >"$scratch/k0-template/ice_cover_clim.nc"
: >"$scratch/k0-template/producer_manifest.json"
printf 'program fixture\nend program fixture\n' >"$scratch/target-template/MY_SRC/fixture.F90"
printf 'program fixture_pp\nend program fixture_pp\n' >"$scratch/target-template/BLD/ppsrc/fixture.f90"
commit=$(git -C "$REPO" rev-parse HEAD)
R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-success" "$scratch/nemo-stub" "$scratch/source-template" "$scratch/k0-template" "$scratch/target-template" "$commit" "$PY"
cat >"$scratch/nemo-no-stop" <<'STUB'
#!/usr/bin/env bash
printf 'run ended without marker\n'
printf 'normal ocean log without a stop marker\n' >ocean.output
STUB
chmod +x "$scratch/nemo-no-stop"
if R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-no-stop" "$scratch/nemo-no-stop" "$scratch/source-template" "$scratch/k0-template" "$scratch/target-template" "$commit" "$PY" >"$scratch/no-stop.log" 2>&1; then
 refuse 'missing-STOP plant stayed green'
fi
grep -F 'REFUSE: execute/stdout STOP 0 check failed' "$scratch/no-stop.log" >/dev/null \
 || refuse 'missing-STOP failure was silent'
printf 'PREFLIGHT_PLANT_FIRED generated-named-refusal\n'
sed 's/printf '\''STOP 0/printf '\''STOP 1/' "$scratch/nemo-stub" >"$scratch/nemo-refuse"; chmod +x "$scratch/nemo-refuse"
plant_must_refuse generated-flow env R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-refuse" "$scratch/nemo-refuse" "$scratch/source-template" "$scratch/k0-template" "$scratch/target-template" "$commit" "$PY"
find "$scratch/refused" -maxdepth 1 -type d -name 'stub-refuse.failed.*' -print -quit | grep . >/dev/null \
  || refuse 'generated driver did not retain its refused run'
cat >"$scratch/nemo-ocean-error" <<'STUB'
#!/usr/bin/env bash
printf 'STOP 0\n'
printf 'E R R O R\nSTOP 0\n' >ocean.output
STUB
chmod +x "$scratch/nemo-ocean-error"
plant_must_refuse generated-ocean-error env R340_PREFLIGHT_DIRECT=1 "$scratch/execute.sh" "$scratch/stub-ocean-error" "$scratch/nemo-ocean-error" "$scratch/source-template" "$scratch/k0-template" "$scratch/target-template" "$commit" "$PY"
find "$scratch/refused" -maxdepth 1 -type d -name 'stub-ocean-error.failed.*' -print -quit | grep . >/dev/null \
  || refuse 'ocean.output error run was not retained'
sed 's/set -Eeuo pipefail/set -Eeuo pipefail\nif then/' "$scratch/execute.sh" >"$scratch/syntax.plant"
plant_must_refuse generated-syntax bash -n "$scratch/syntax.plant"
synthetic=$scratch/admission-control
mkdir -p "$synthetic/a/build_sources/MY_SRC" "$synthetic/a/build_sources/ppsrc/nemo" \
 "$synthetic/b/build_sources/MY_SRC" "$synthetic/b/build_sources/ppsrc/nemo" "$synthetic/parent" "$synthetic/refused"
"$PY" - "$synthetic" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import shutil,struct,sys
from pathlib import Path
import numpy as np
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round340_standalone_gate as g
root=Path(sys.argv[1]); commit=sys.argv[2]
for rank in (0,1): (root/'parent'/f'ORCA2_00000010_restart_{rank:04d}.nc').write_bytes(b'restart'+bytes([rank]))
def write(run):
    (run/'nemo').write_bytes(b'synthetic nemo\n')
    (run/'run.user.stdout.log').write_text('STOP 0\n')
    (run/'ocean.output').write_text('STOP 0\n')
    (run/'build_sources/ppsrc/nemo/synthetic.f90').write_text('program synthetic\n')
    z=np.zeros
    for kt in range(1,11):
      for rank in (0,1):
       for kid in range(12):
        h=b'NEMO_R340_S4_3  '+struct.pack('=7i',3,kt,kid,rank,2,1,64)
        (run/f'oracle_r340_surface_k{kt:08d}_b{kid:02d}_r{rank:04d}.bin').write_bytes(h+z(2*len(g.OUTER_FIELDS_BY_KID[kid]),'=f8').tobytes())
       for stage,level in ((0,1 if kt%2 else 3),(1,3 if kt%2 else 1),(2,2),(3,3 if kt%2 else 1)):
        h=b'NEMO_R340_STATE1'+struct.pack('=10i',1,kt,stage,level,rank,6,5,3,2,64)
        (run/f'oracle_r340_state_k{kt:08d}_s{stage}_r{rank:04d}.bin').write_bytes(h+z(390,'=f8').tobytes())
       for kid in ((0,1,2,3) if kt%2 else (0,3)):
        h=b'NEMO_R340_IIF_2 '+struct.pack('=9i',2,kt,kid,1 if kt%2 else 3,1 if kt%2 else 3,rank,2,1,64)
        (run/f'oracle_r340_iif_k{kt:08d}_b{kid}_r{rank:04d}.bin').write_bytes(h+z(2*len(g.IIF_FIELDS_BY_KID[kid]),'=f8').tobytes())
       if kt%2:
        for kind,kid,n,name in ((b'NEMO_R340_S1T_1 ',0,19,'s1t'),(b'NEMO_R340_S1F_1 ',1,6,'s1f')):
         h=kind+struct.pack('=7i',1,kt,kid,rank,2,1,64)
         (run/f'oracle_r340_{name}_k{kt:08d}_r{rank:04d}.bin').write_bytes(h+z(2*n,'=f8').tobytes())
       h=b'NEMO_R340_FLDRD1'+struct.pack('=9i',1,kt,1,1,rank,0,1,2,64)+b'tair'.ljust(34)+struct.pack('=4i',1,2,3,4)+z(1,'=f8').tobytes()+struct.pack('=7i',1,1,1,2,1,1,2)+z(2,'=f8').tobytes()+struct.pack('=10i',0,0,0,0,0,0,0,0,0,0)
       (run/f'oracle_r340_fld_k{kt:08d}_c001_f001_r{rank:04d}.bin').write_bytes(h)
       shutil.copy2(root/'parent'/f'ORCA2_00000010_restart_{rank:04d}.nc',run/f'ORCA2_00000010_restart_{rank:04d}.nc')
    files=[p for p in run.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
    c.write_manifest(run/'producer_manifest.json',producer_commit=commit,binary=run/'nemo',files=files,root=run,build_source_root=run/'build_sources')
write(root/'a'); shutil.copytree(root/'a',root/'b',dirs_exist_ok=True)
PY
mv "$synthetic/a" "$synthetic/refused/a.failed.control"
mv "$synthetic/b" "$synthetic/refused/b.failed.control"
R340_PREFLIGHT_ADMIT_DIRECT=1 R340_EXISTING_RUN_A="$synthetic/refused/a.failed.control" \
 R340_EXISTING_RUN_B="$synthetic/refused/b.failed.control" R340_EXISTING_PARENT="$synthetic/parent" \
 "$here/run.sh" --admit-existing >"$scratch/retained-admission.log"
grep -F ORCA2_ROUND340_S4_TWINS_ADMITTED "$scratch/retained-admission.log" >/dev/null || refuse 'retained admission failed'
printf 'PREFLIGHT_RETAINED_ADMISSION_PASS route=refused full=true\n'
printf 'checksum plant\n' >>"$synthetic/refused/a.failed.control/nemo"
plant_must_refuse retained-checksum admit_one "$synthetic/refused/a.failed.control" "$synthetic/parent"

if [[ "$MODE" == --preflight-only ]]; then printf 'ORCA2_ROUND340_S4_PREFLIGHT_READY\n'; exit 0; fi
if [[ "$MODE" == --admit-existing ]]; then
 admit_existing_route
 exit 0
fi

failure_step='run preparation'
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse 'run requires a clean committed tree'
resume_arm_b=0
if [[ -d $RUN_A && ! -e $RUN_B && -x $TARGET_BUILD/BLD/bin/nemo.exe ]]; then
 admit_one "$RUN_A"
 resume_arm_b=1
else
 [[ ! -e "$TARGET_BUILD" && ! -e "$RUN_A" && ! -e "$RUN_B" ]] || refuse 'retain/move existing build and runs before retry'
fi
[[ $(free -g | awk '/^Mem:/{print $7}') -ge 150 ]] || refuse 'under 150 GB available memory'
mkdir -p "$EVIDENCE"
cd "$NEMO_ROOT"
if [[ $resume_arm_b -eq 0 ]]; then
 failure_step='NEMO build'
 ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -z $(find "$TARGET_BUILD/MY_SRC" -mindepth 1 -maxdepth 1 -print -quit) ]] \
  || refuse 'new K0-derived MY_SRC is not empty'
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcmod.f90" "$TARGET_BUILD/MY_SRC/sbcmod.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3.f90" "$TARGET_BUILD/MY_SRC/stprk3.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/stprk3_stg.f90" "$TARGET_BUILD/MY_SRC/stprk3_stg.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/fldread.f90" "$TARGET_BUILD/MY_SRC/fldread.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcice_if.f90" "$TARGET_BUILD/MY_SRC/sbcice_if.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcblk.f90" "$TARGET_BUILD/MY_SRC/sbcblk.F90"
cp "$K0_RUN/build_sources/ppsrc/nemo/sbcfwb.f90" "$TARGET_BUILD/MY_SRC/sbcfwb.F90"
cp "$SURFACE_WRITER" "$STATE_WRITER" "$TARGET_BUILD/MY_SRC/"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$SURFACE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$STATE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$STAGE_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$FLD_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$IIF_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$BLK_PATCH"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_BUILD/MY_SRC" <"$FWB_PATCH"
cp -a "$K0_BUILD/EXP00/." "$TARGET_BUILD/EXP00/"
cp "$K0_BUILD/cpp_$K0_CFG.fcm" "$TARGET_BUILD/cpp_$TARGET_CFG.fcm"
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
fi
obs=$TARGET_BUILD/BLD/bin/nemo.exe; [[ -x "$obs" ]] || refuse 'instrumented build failed'
if [[ $resume_arm_b -eq 1 ]]; then runs=("$RUN_B"); else runs=("$RUN_A" "$RUN_B"); fi
for run in "${runs[@]}"; do
  active_runs+=("$run")
  failure_step="execution of $run"
  "$scratch/execute.sh" "$run" "$obs" "$SOURCE_RUN" "$K0_RUN" "$TARGET_BUILD" "$(git -C "$REPO" rev-parse HEAD)" "$PY"
  admit_one "$run"
done
admit_pair "$RUN_A" "$RUN_B"
# The generic round-328 proof strips exactly the five observer patches and two writers.
"$PY" - "$TARGET_BUILD/BLD" "$K0_BUILD/BLD" "$SURFACE_PATCH" "$STATE_PATCH" "$STAGE_PATCH" "$FLD_PATCH" "$IIF_PATCH" "$BLK_PATCH" "$FWB_PATCH" "$TARGET_CFG" "$K0_CFG" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round328_dynzdf_gate as g
g.same_program_observers(Path(sys.argv[1]),Path(sys.argv[2]),dict(zip(('sbcmod','stprk3','stprk3_stg','fldread','sbcice_if','sbcblk','sbcfwb'),map(Path,sys.argv[3:10]),strict=True)),('round340_surface_writer','round340_state_writer'),new_cfg=sys.argv[10],ref_cfg=sys.argv[11])
PY
cmp -s "$RUN_A/ORCA2_00000010_restart_0000.nc" "$K0_RUN/ORCA2_00000010_restart_0000.nc" || refuse 'observer changed rank-0 restart'
cmp -s "$RUN_A/ORCA2_00000010_restart_0001.nc" "$K0_RUN/ORCA2_00000010_restart_0001.nc" || refuse 'observer changed rank-1 restart'
cmp -s "$RUN_B/ORCA2_00000010_restart_0000.nc" "$K0_RUN/ORCA2_00000010_restart_0000.nc" || refuse 'twin observer changed rank-0 restart'
cmp -s "$RUN_B/ORCA2_00000010_restart_0001.nc" "$K0_RUN/ORCA2_00000010_restart_0001.nc" || refuse 'twin observer changed rank-1 restart'
active_runs=()
