"""Make an INSTRUMENTED copy of WRF v4.8.0 phys/module_cu_tiedtke.F.

Insertions are (a) `use tdbg` lines, (b) plain assignments into module tdbg,
(c) cpp blocks selected at compile time:
  -DT89     : skip section 5.1 (Nordeng/ECHAM4 CAPE closure) for deep (ktype=1)
              convection, so the deep cloud-base mass flux stays the Tiedtke-1989
              sub-cloud moisture-budget value computed in section 3(c).
  -DLEGOSAT : saturation curve = the legoESM Tetens curve (legoesm.thermo
              saturation_vapor_pressure coefficients, T_freeze reference) on BOTH
              the liquid and the ice branch (legoESM has no ice saturation),
              i.e. water-only thermodynamics.
No other oracle statement is altered or removed. Anchors asserted unique.
Output: module_cu_tiedtke_dbg.F (+ tiedtke_patch.diff).
"""
import difflib

import sys

SRC = sys.argv[1]   # <WRF v4.8.0 checkout>/phys/module_cu_tiedtke.F
OUT = sys.argv[2]
src = open(SRC).read()
L = src.splitlines(keepends=True)

# legoESM Tetens curve as injected into the Fortran oracle (mirrors
# legoesm.thermo.saturation_vapor_pressure; verified numerically in harness.py
# stage `inputs`, which asserts the oracle formula reproduces legoesm.thermo).
LEGO_C1ES = "611.2"     # satcurve-ok: value injected into Fortran oracle, checked vs legoesm.thermo
LEGO_C3 = "17.67"       # satcurve-ok: same
LEGO_C4 = "29.65"       # satcurve-ok: 273.15-243.5, same
LEGO_TMELT = "273.15"   # const-ok: legoesm constants.T_freeze injected into Fortran, checked in harness


def idx(pred, start=0, end=None, unique=True, what=""):
    end = len(L) if end is None else end
    hits = [i for i in range(start, end) if pred(L[i])]
    assert hits, what
    if unique:
        assert len(hits) == 1, (what, hits)
    return hits[0]


def ins(i, text):
    L[i:i] = [text]


# --- LEGOSAT parameter swaps
i_tm = idx(lambda l: l.strip().startswith("tmelt=273.16"), what="tmelt")  # const-ok: oracle anchor
L[i_tm] = ("#ifdef LEGOSAT\n      tmelt=" + LEGO_TMELT + ",                 &\n#else\n" + L[i_tm] + "#endif\n")
lego_vals = {"c1es=": LEGO_C1ES, "c3les=": LEGO_C3, "c4les=": LEGO_C4,
             "c3ies=": LEGO_C3, "c4ies=": LEGO_C4}
for nm, val in lego_vals.items():
    i = idx(lambda l, nm=nm: l.strip().startswith(nm), what=nm)
    rest = L[i].split("=", 1)[1]
    tail = rest[rest.index(","):]
    L[i] = ("#ifdef LEGOSAT\n      " + nm + val + tail + "#else\n" + L[i] + "#endif\n")

# --- tiecnv: use tdbg + dump just before the state update loop
i_tie = idx(lambda l: l.strip().startswith("subroutine tiecnv("), what="tiecnv")
i_imp = idx(lambda l: l.strip() == "implicit none", i_tie, i_tie + 20, what="tiecnv implicit")
ins(i_imp, "      use tdbg\n")
i_up = idx(lambda l: l.strip() == "pt(j,k)=ztp1(j,k)+ptte(j,k)*ztmst", what="pt update")
assert L[i_up - 2].strip() == "do k=1,km" and L[i_up - 1].strip() == "do j=1,lq"
ins(i_up - 2,
    "      d_ptte(1:lq,1:km)=ptte(1:lq,1:km); d_qte(1:lq,1:km)=pqte(1:lq,1:km)-zqq(1:lq,1:km)\n"
    "      d_cte(1:lq,1:km)=pcte(1:lq,1:km); d_vom(1:lq,1:km)=pvom(1:lq,1:km)\n"
    "      d_vol(1:lq,1:km)=pvol(1:lq,1:km); d_prec(1:lq)=prsfc(1:lq)+pssfc(1:lq)\n"
    "      d_ktype(1:lq)=ktype(1:lq)\n")

# --- cumastr_new
i_cm = idx(lambda l: l.strip().startswith("subroutine cumastr_new"), what="cumastr")
i_cm_end = idx(lambda l: l.strip() == "end subroutine cumastr_new", what="cumastr end")
i_imp = idx(lambda l: l.strip() == "implicit none", i_cm, i_cm_end, what="cm implicit")
ins(i_imp, "      use tdbg\n")
i_cm_end += 1
i_z2 = idx(lambda l: l.strip() == "zcons2=1./(g*ztmst)", i_cm, i_cm_end, what="zcons2")
ins(i_z2 + 1,
    "      d_mfub0=0.; d_ktype0=0; d_zdqpbl=0.; d_zdqcv=0.; d_zqumqe=0.; d_kcbot0=0; d_ldcum0=0\n"
    "      d_zcape=0.; d_zheat=0.; d_zrelh=0.; d_mfub=0.; d_kcbot=0; d_kctop=0; d_ldcum=0\n"
    "      d_mfu=0.; d_mfd=0.; d_tu=0.; d_qu=0.; d_lu=0.; d_lude=0.; d_dmfup=0.; d_idtop=0; d_lodd=0\n"
    "      d_uu=0.; d_vu=0.; d_ud=0.; d_vd=0.; d_icum=-1\n")
i_cm_end += 1
i_mb = idx(lambda l: l.strip() == "zmfub(jl)=min(zmfub(jl),zmfmax)", i_cm, i_cm_end, what="mfub0")
ins(i_mb + 1,
    "      d_mfub0(jl)=zmfub(jl); d_ktype0(jl)=ktype(jl); d_zdqpbl(jl)=zdqpbl(jl)\n"
    "      d_zdqcv(jl)=zdqcv(jl); d_zqumqe(jl)=zqumqe; d_kcbot0(jl)=kcbot(jl)\n"
    "      if(ldcum(jl)) d_ldcum0(jl)=1\n")
i_cm_end += 1
i_51 = idx(lambda l: l.strip() == "if(ldcum(jl).and.ktype(jl).eq.1) then", i_cm, i_cm_end, what="5.1")
assert L[i_51 + 1].strip() == "ktop0=max(12,kctop(jl))"
L[i_51] = ("#ifdef T89\n      if(.false.) then\n#else\n" + L[i_51] + "#endif\n")
i_52 = idx(lambda l: l.strip().startswith("!*  5.2   recalculate"), i_cm, i_cm_end, what="5.2")
ins(i_52, "      d_zcape(1:klon)=zcape(1:klon); d_zheat(1:klon)=zheat(1:klon); d_zrelh(1:klon)=zrelh(1:klon)\n")
i_cm_end += 1
i_ic = idx(lambda l: l.strip() == "if(icum.eq.0) return", i_cm, i_cm_end, what="icum")
ins(i_ic, "      d_icum=icum\n")
i_cm_end += 1
assert L[i_cm_end - 1].strip() == "return", L[i_cm_end - 1]
ins(i_cm_end - 1,
    "      d_mfub(1:klon)=zmfub(1:klon); d_kcbot(1:klon)=kcbot(1:klon); d_kctop(1:klon)=kctop(1:klon)\n"
    "      d_idtop(1:klon)=idtop(1:klon)\n"
    "      do jl=1,klon\n"
    "        d_ldcum(jl)=0; if(ldcum(jl)) d_ldcum(jl)=1\n"
    "        d_lodd(jl)=0; if(loddraf(jl)) d_lodd(jl)=1\n"
    "      end do\n"
    "      d_mfu(1:klon,1:klev)=pmfu; d_mfd(1:klon,1:klev)=pmfd; d_tu(1:klon,1:klev)=ptu\n"
    "      d_qu(1:klon,1:klev)=pqu; d_lu(1:klon,1:klev)=plu; d_lude(1:klon,1:klev)=plude\n"
    "      d_dmfup(1:klon,1:klev)=zdmfup; d_uu(1:klon,1:klev)=zuu; d_vu(1:klon,1:klev)=zvu\n"
    "      d_ud(1:klon,1:klev)=zud; d_vd(1:klon,1:klev)=zvd\n")

out = "".join(L)
open(OUT, "w").write(out)
open("tiedtke_patch.diff", "w").writelines(
    difflib.unified_diff(src.splitlines(keepends=True), out.splitlines(keepends=True),
                         "module_cu_tiedtke.F", OUT))
removed = [l for l in difflib.ndiff(src.splitlines(), out.splitlines()) if l.startswith("- ")]
print("patched; removed lines (must be 0):", len(removed))
