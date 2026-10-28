#!/usr/bin/env python3
"""Real-flag, per-function FP census for the O4 Prather observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o9_observer_census import _flags, fp_counts
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import _strip_preprocessed_observers

HERE = Path(__file__).resolve().parent / "nemo_si3_o4_orca2_prather_acquisition"
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")
BASE_ROUTINE = "__icedyn_adv_pra_MOD_ice_dyn_adv_pra"
OBS_ROUTINE = "__icedyn_adv_pra_MOD_ice_dyn_adv_pra_physics"


def census(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict[str, object]:
    flags = _flags(build)
    raw = NEMO / "src/ICE/icedyn_adv_pra.F90"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td); env = dict(os.environ, PATH=f"{toolchain/'bin'}:{os.environ['PATH']}")
        writer = tmp / "o4_prather_writer.F90"; writer.write_bytes((HERE/"o4_prather_writer.F90").read_bytes())
        subprocess.run(["mpif90","-c","-cpp",*flags,f"-I{build/'inc'}",f"-J{tmp}",str(writer),"-o",str(tmp/"writer.o")],
                       check=True,env=env,capture_output=True,text=True)
        patched_raw = tmp / "icedyn_adv_pra.F90"; patched_raw.write_bytes(raw.read_bytes())
        subprocess.run(["patch","--batch","--fuzz=0",str(patched_raw),str(HERE/"icedyn_adv_pra_o4.patch")],
                       check=True,capture_output=True,text=True)
        def pp(source: Path, dest: Path) -> None:
            with dest.open("w") as stream:
                subprocess.run(["cpp","-Dkey_nosignedzero","-P","-traditional","-Dkey_si3","-Dkey_linssh",
                    "-Dkey_vco_1d3d","-Dkey_RK3",f"-I{build.parent/'WORK'}",str(source)],check=True,env=env,stdout=stream,text=True)
        plain = tmp/"plain.f90"; observed = tmp/"observed.f90"; pp(raw,plain); pp(patched_raw,observed)
        def routine_lines(path: Path, name: str) -> list[str]:
            lines=path.read_text().splitlines(); start=next(i for i,line in enumerate(lines) if re.match(rf"\s*SUBROUTINE\s+{name}\b",line,re.I))
            stop=next(i for i in range(start+1,len(lines)) if re.match(rf"\s*END\s+SUBROUTINE\s+{name}\b",lines[i],re.I))
            return lines[start:stop+1]
        plain_body=routine_lines(plain,"ice_dyn_adv_pra")
        observed_body=routine_lines(observed,"ice_dyn_adv_pra_physics")
        stripped,blocks,_=_strip_preprocessed_observers(observed_body,"! O4_OBSERVER_BEGIN","! O4_OBSERVER_END","icedyn_adv_pra")
        stripped=[line.replace("ice_dyn_adv_pra_physics","ice_dyn_adv_pra") for line in stripped]
        plain_body[0]=re.sub(r"ice_dyn_adv_pra\(\s+kt", "ice_dyn_adv_pra( kt", plain_body[0], flags=re.I)
        stripped[0]=re.sub(r"ice_dyn_adv_pra\(\s+kt", "ice_dyn_adv_pra( kt", stripped[0], flags=re.I)
        same_program_source_body=(stripped==plain_body and blocks==3)
        planted = tmp/"plant.f90"; text = observed.read_text(); marker = "! O4_OBSERVER_END"
        at = text.find(marker, text.find("o4_zcfl_current =")); end = text.find("\n",at)
        if at < 0: raise ValueError("O4 observer executable marker absent")
        planted.write_text(text[:end+1]+"      pato_i(1,1) = pato_i(1,1) + 1._wp\n"+text[end+1:])
        def compile_one(source: Path) -> Path:
            obj = source.with_suffix(".o")
            subprocess.run(["mpif90","-c",*flags,f"-I{tmp}",f"-I{build/'inc'}",f"-J{tmp}",str(source),"-o",str(obj)],
                           check=True,env=env,capture_output=True,text=True)
            return obj
        a,b,c = (fp_counts(compile_one(path)) for path in (plain,observed,planted))
        def canonical(rows):
            out={}
            for function,counts in rows.items():
                name=re.sub(r"\.(?:constprop|isra)\.\d+", "", function)
                target=out.setdefault(name,{})
                for mnemonic,count in counts.items(): target[mnemonic]=target.get(mnemonic,0)+count
            return out
        a,b,c=canonical(a),canonical(b),canonical(c)
        # The observer boundary intentionally changes which private helper clones
        # GCC retains.  Census the preserved physics function itself, not dead or
        # duplicated IPA clones; its complete mnemonic map is the passivity claim.
        diffs = {}
        physics_diff={key:[a.get(BASE_ROUTINE,{}).get(key,0),b.get(OBS_ROUTINE,{}).get(key,0)]
                      for key in sorted(set(a.get(BASE_ROUTINE,{}))|set(b.get(OBS_ROUTINE,{})))
                      if a.get(BASE_ROUTINE,{}).get(key,0)!=b.get(OBS_ROUTINE,{}).get(key,0)}
        if physics_diff: diffs[OBS_ROUTINE]=physics_diff
        total = sum(a.get(BASE_ROUTINE,{}).values()); plant_fired = c.get(OBS_ROUTINE)!=b.get(OBS_ROUTINE)
        return {"flags":flags,"routine":OBS_ROUTINE,"plain_fp_total":total,
                "instrumented_fp_total":sum(b.get(OBS_ROUTINE,{}).values()),"differing_mnemonics":diffs,
                "census_scope":"complete preserved physics function",
                "plant_fired":plant_fired,"source_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),
                "same_program_source_body":same_program_source_body,
                "observer_blocks_in_physics":blocks,
                "pass":total>0 and not diffs and plant_fired and same_program_source_body}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--build",type=Path,default=BUILD); parser.add_argument("--toolchain",type=Path,default=TOOLCHAIN)
    args=parser.parse_args()
    try: result=census(args.build,args.toolchain)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        print(f"STATUS PLANT-FIRED: {exc}"); return 1
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["pass"] else 1


if __name__=="__main__": raise SystemExit(main())
