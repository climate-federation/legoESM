#!/usr/bin/env python
"""Generate the STAGED VERBATIM copies of dyn_core.F90 + fv_dynamics.F90.

ORACLE STAGE-STATE instrument, step 1 of the panel-boundary-floor hunt:
the staged copies are byte-identical to the pinned tree EXCEPT for a
mechanically applied, fully enumerated deviation list:

  D1  module renames:  dyn_core_mod    -> dyn_core_staged_mod
                       fv_dynamics_mod -> fv_dynamics_staged_mod
      (subroutine names are unchanged; module name mangling separates
      them from the real ones at link time)
  D2  fv_dynamics's  `use dyn_core_mod, only: ...`  becomes
      `use dyn_core_staged_mod, only: ...`  -- the ONE wiring change
      that makes the staged shell drive the staged core.
  D3  one `use stage_dump_mod` line per module.
  D4  single-line dump hooks (each tagged `! STAGED-HOOK`) inserted
      AFTER pinned anchor lines.  Pure I/O; inert until the driver
      calls stage_dump_open, so the REAL certification arm is untouched.
  D5  NOTHING ELSE.  Every non-hook line is byte-identical, enforced by
      strip_hooks() round-trip in this file's --check mode and by
      tests/grids/test_fv3_dyncore_stage_instrument.py.

Anchors are (1-based line number, exact expected content).  The pinned
sources are immutable (chmod a-w + sha256 asserted below), so line
numbers are stable; a changed pin fails LOUDLY here instead of silently
hooking the wrong statement.

Stage names (hydro duo lane, substep it==1 only; oracle anchors in the
pinned dyn_core.F90):

  S00_entry      dyn_core entry state (post p_var / theta conversion)
  S01_extdp      :437-438  ext_scalar(delp), ext_scalar(pt)   [it==1]
  S02_extuv      :471      ext_vector(u, v)
  S03_csw        :489      c_sw (k loop)
  S04_geopkC     :533      geopk CG=.true.
  S05_pgradc     :629      p_grad_c
  S06_extdivgd   :652      ext_scalar(divgd, 1,1)
  S07_extucvc    :655      ext_vector(uc, vc, 1,0,0,1)
  S08_dsw1       :831      d_sw1 (k loop)
  S09_fluxavg    :853-900  BARRIER 1 flux average (CGRID_NE)
  S10_dsw23      :950/:961 d_sw2 + d_sw3 (k loop)
  S11_b2         :969-1011 BARRIER 2 (BGRID_NE) on ubb/vbbtemp
  S12_kee        :1015-1020 corner KE assembly
  S13_dsw45      :1102/:1107 d_sw4 + d_sw5 (k loop)
  S14_dsw6       :1256     d_sw6 (k loop)
  S15_extdp2     :1336-1337 ext_scalar(delp), ext_scalar(pt)
  S16_geopkD     :1401     geopk CG=.false.
  S17_onegradp   :1531     one_grad_p
  S90_postdyncore  fv_dynamics, after dyn_core returns
  S95_postremap    fv_dynamics, after Lagrangian_to_Eulerian
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

PIN = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned"
SRC = f"{PIN}/atmos_cubed_sphere-symmetryclean"

DYN_CORE_SHA = "2cdbdf1fd30773b7ce26e23cbc22b8bbc9099cdab4f1df82eef74468e4d82835"
FV_DYNAMICS_SHA = "f862fb8ab77f3569689e558d2977e5125a2432d9d9f949020c9e5fe8ad7b2870"

HOOK_TAG = "! STAGED-HOOK"

USE_LINE = ("  use stage_dump_mod, only: stage_dumps_active, stage_dump2, "
            f"stage_dump3, stage_dump4   {HOOK_TAG}")


def _h3(stage: str, var: str, guard_it: bool = True) -> str:
    g = "stage_dumps_active() .and. it == 1" if guard_it \
        else "stage_dumps_active()"
    return (f"      if ({g}) call stage_dump3('{stage}_{var}', {var})"
            f"   {HOOK_TAG}")


def _h4(stage: str, var: str) -> str:
    return (f"      if (stage_dumps_active() .and. it == 1) "
            f"call stage_dump4('{stage}_{var}', {var})   {HOOK_TAG}")


def _hooks3(stage: str, names, guard_it: bool = True):
    return [_h3(stage, v, guard_it) for v in names]


# ---------------------------------------------------------------------------
# dyn_core.F90 deviations
# ---------------------------------------------------------------------------
# (line_no, expected exact content, replacement or None)
DYN_EDITS = [
    (22, "module dyn_core_mod", "module dyn_core_staged_mod"),
    (3128, "end module dyn_core_mod", "end module dyn_core_staged_mod"),
]
# (line_no, expected exact content, [hook lines inserted AFTER])
DYN_INSERTS = [
    (64, "  use fv_grid_utils_mod,  only: g_sum", [USE_LINE]),
    (316, "    call init_ijk_mem(isd, ied, js,  je+1, npz, cy, 0.)",
     _hooks3("S00_entry", ["u", "v", "delp", "pt", "uc", "vc"],
             guard_it=False)),
    (440, "                                      call timing_off('COMM_TOTAL')",
     _hooks3("S01_extdp", ["delp", "pt"])),
    (482, "                                                     "
          "call timing_off('COMM_TOTAL')",
     _hooks3("S02_extuv", ["u", "v"])),
    (497, "                                                     "
          "call timing_off('c_sw')",
     _hooks3("S03_csw", ["delpc", "ptc", "uc", "vc", "ut", "vt", "divgd",
                         "ua", "va", "omga"])),
    (534, "                      gridstruct%bounded_domain, "
          "gridstruct%dg%is_initialized, .false., npx, npy, "
          "flagstruct%a2b_ord, bd)",
     _hooks3("S04_geopkC", ["pkc", "gz"])),
    (629, "      call p_grad_c(dt2, npz, delpc, pkc, gz, uc, vc, bd, "
          "gridstruct%rdxc, gridstruct%rdyc, hydrostatic)",
     _hooks3("S05_pgradc", ["uc", "vc"])),
    (652, "                        if(duogrid        .and. "
          "flagstruct%nord > 0) call ext_scalar(divgd, gridstruct%dg, bd, "
          "domain, 1,1)",
     _hooks3("S06_extdivgd", ["divgd"])),
    (655, "                        if (duogrid)     call ext_vector(uc, vc, "
          "gridstruct%dg, bd, domain,gridstruct, flagstruct,1,0,0,1)",
     _hooks3("S07_extucvc", ["uc", "vc"])),
    (848, "enddo",
     _hooks3("S08_dsw1", ["crx", "cry", "xfx", "yfx", "mfx", "mfy",
                          "cx", "cy", "utt", "vtt", "rax", "ray"])
     + [_h4("S08_dsw1", "allflux_x"), _h4("S08_dsw1", "allflux_y")]),
    (900, "endif !if duo",
     [_h4("S09_fluxavg", "allflux_x"), _h4("S09_fluxavg", "allflux_y")]),
    (966, "enddo",
     _hooks3("S10_dsw23", ["delp", "pt", "ptc", "utt", "vtt", "ubbtemp",
                           "vbbtemp", "ubb", "vbb"])),
    (1011, "endif !if duo",
     _hooks3("S11_b2", ["ubb", "vbbtemp"])),
    (1020, "enddo",
     _hooks3("S12_kee", ["kee"])),
    (1123, "enddo",
     _hooks3("S13_dsw45", ["u", "v", "vt", "kee", "wkk", "vortfluxx",
                           "vortfluxy", "ua", "va", "divgd",
                           "utt", "vtt"])),
    (1288, "    enddo           ! end openMP k-loop",
     _hooks3("S14_dsw6", ["u", "v"])),
    (1348, "                                       "
           "call timing_off('COMM_TOTAL')",
     _hooks3("S15_extdp2", ["delp", "pt"])),
    (1402, "                     gridstruct%bounded_domain, "
           "flagstruct%duogrid, .true., npx, npy, flagstruct%a2b_ord, bd)",
     _hooks3("S16_geopkD", ["pe", "peln", "pkc", "gz", "pkz"])),
    (1531, "          call one_grad_p(u, v, pkc, gz, divg2, delp, dt, ng, "
           "gridstruct, bd, npx, npy, npz, ptop, hydrostatic, "
           "flagstruct%a2b_ord, flagstruct%d_ext)",
     _hooks3("S17_onegradp", ["u", "v"])),
]

# ---------------------------------------------------------------------------
# fv_dynamics.F90 deviations
# ---------------------------------------------------------------------------
FVD_EDITS = [
    (22, "module fv_dynamics_mod", "module fv_dynamics_staged_mod"),
    (24, "   use dyn_core_mod,        only: dyn_core, del2_cubed, "
         "init_ijk_mem",
     "   use dyn_core_staged_mod, only: dyn_core, del2_cubed, "
     "init_ijk_mem   " + HOOK_TAG),
    (1309, "end module fv_dynamics_mod", "end module fv_dynamics_staged_mod"),
]
FVD_INSERTS = [
    (24, None,  # content asserted via FVD_EDITS above; insert after it
     [USE_LINE]),
    (507, "                                           "
          "call timing_off('DYN_CORE')",
     [f"      if (stage_dumps_active()) call stage_dump3('S90_{v}', {v})"
      f"   {HOOK_TAG}" for v in ("u", "v", "delp", "pt")]),
    (629, "                     flagstruct%moist_phys, flagstruct%w_limiter)",
     [f"      if (stage_dumps_active()) call stage_dump3('S95_{v}', {v})"
      f"   {HOOK_TAG}" for v in ("u", "v", "delp", "pt")]),
]


def read_pinned(path: str, want_sha: str) -> list:
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != want_sha:
        raise SystemExit(
            f"{path}: sha256 {got} != pinned {want_sha}. The pinned tree "
            f"changed (or the wrong tree is mounted); every line anchor "
            f"below would be untrustworthy. Refusing to generate.")
    return raw.decode(errors="replace").splitlines()


def apply(lines: list, edits, inserts) -> list:
    lines = list(lines)
    for ln, want, repl in edits:
        got = lines[ln - 1]
        if got != want:
            raise SystemExit(
                f"anchor mismatch at line {ln}:\n  want {want!r}\n"
                f"  got  {got!r}")
        lines[ln - 1] = repl
    # apply inserts bottom-up so earlier line numbers stay valid
    for ln, want, hook_lines in sorted(inserts, reverse=True):
        if want is not None:
            got = lines[ln - 1]
            # the line may have been rewritten by an edit at the same anchor
            if got != want and HOOK_TAG not in got:
                raise SystemExit(
                    f"insert-anchor mismatch at line {ln}:\n"
                    f"  want {want!r}\n  got  {got!r}")
        lines[ln - 1:ln - 1] = []  # no-op, clarity
        for h in reversed(hook_lines):
            lines.insert(ln, h)
    return lines


def strip_hooks(staged_lines: list) -> list:
    """Drop hook lines and un-apply renames -> must equal the pinned file."""
    out = []
    for s in staged_lines:
        if HOOK_TAG in s:
            continue
        s = s.replace("module dyn_core_staged_mod", "module dyn_core_mod")
        s = s.replace("module fv_dynamics_staged_mod",
                      "module fv_dynamics_mod")
        out.append(s)
    return out


def generate(outdir: str, check: bool = True) -> None:
    os.makedirs(outdir, exist_ok=True)
    for src, sha, edits, inserts, name in (
            (f"{SRC}/model/dyn_core.F90", DYN_CORE_SHA, DYN_EDITS,
             DYN_INSERTS, "staged_dyn_core.F90"),
            (f"{SRC}/model/fv_dynamics.F90", FV_DYNAMICS_SHA, FVD_EDITS,
             FVD_INSERTS, "staged_fv_dynamics.F90")):
        pinned = read_pinned(src, sha)
        staged = apply(pinned, edits, inserts)
        if check:
            # VERBATIM GUARANTEE: hooks out + renames undone == pinned.
            # The FVD use-line edit carries a HOOK_TAG so strip_hooks drops
            # it; re-add the original there.
            recovered = strip_hooks(staged)
            expect = [ln for ln in pinned]
            if name == "staged_fv_dynamics.F90":
                # line 24 was rewritten AND tagged -> stripped; reinsert.
                recovered.insert(23, expect[23])
            if recovered != expect:
                for i, (a, b) in enumerate(zip(recovered, expect)):
                    if a != b:
                        raise SystemExit(
                            f"{name}: verbatim round-trip FAILED at "
                            f"pinned line {i + 1}:\n  pinned {b!r}\n"
                            f"  recovered {a!r}")
                raise SystemExit(
                    f"{name}: verbatim round-trip FAILED (length "
                    f"{len(recovered)} vs {len(expect)})")
        out = os.path.join(outdir, name)
        with open(out, "w") as fh:
            fh.write("\n".join(staged) + "\n")
        n_hooks = sum(1 for s in staged if HOOK_TAG in s)
        print(f"wrote {out}  ({len(staged)} lines, {n_hooks} hook/deviation "
              f"lines, source sha {sha[:12]})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the verbatim round-trip (never in CI)")
    a = ap.parse_args(argv)
    generate(a.outdir, check=not a.no_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
