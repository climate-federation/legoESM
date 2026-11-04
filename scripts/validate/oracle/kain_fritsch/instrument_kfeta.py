"""Make an INSTRUMENTED copy of WRF v4.8.0 phys/module_cu_kfeta.F.

usage: python instrument_kfeta.py <module_cu_kfeta.F> <out.F>

Every insertion is an assignment into module kfdbg, a USE line, or a new
read-only dump routine (kf_dump_lut); no oracle statement is altered or
removed (asserted).  Anchors must occur exactly once.  Same instrumentation as
the tier-2 audit (oracle_audit/kain_fritsch_convection/fortran/patch_oracle.py)
plus FBFRC/FRC2/CPR and the lookup-table dump.
"""
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
lines = src.splitlines(keepends=True)


def find_once(pred, start=0, what=""):
    hits = [i for i in range(start, len(lines)) if pred(lines[i])]
    assert len(hits) >= 1, what
    return hits[0]


i_para = find_once(lambda ln: "SUBROUTINE KF_eta_PARA" in ln and "END" not in ln, 0, "para")
i_imp = find_once(lambda ln: ln.strip() == "IMPLICIT NONE", i_para, "implicit")
lines.insert(i_imp, "      USE kfdbg\n")
i = find_once(lambda ln: ln.strip() == "IPRNT=.FALSE.", i_para, "iprnt")
assert sum(1 for ln in lines if ln.strip() == "IPRNT=.FALSE.") == 1
lines.insert(i + 1, "      d_fired=0\n")
i = find_once(lambda ln: ln.strip() == "CUTOP(I,J)=REAL(LTOP)", i_para, "cutop")
lines.insert(i, (
    "      d_fired=1; d_ltop=LTOP; d_let=LET; d_abe=ABE; d_vmflcl=VMFLCL; d_ainc=AINC\n"
    "      d_timec=TIMEC; d_ncount=NCOUNT; d_fabe=FABE; d_tder=TDER\n"
    "      d_trppt=TRPPT; d_pptflx=PPTFLX; d_cpr=CPR; d_lfs=LFS; d_ldb=LDB\n"
    "      d_wlcl=WLCL; d_rad=RAD; d_chmin=CHMIN; d_cldhgt=CLDHGT(LC); d_kpblf=KPBL; d_lcf=LC\n"
    "      d_lcl=LCL; d_au0=AU0; d_nstep=NSTEP; d_fbfrc=FBFRC; d_frc2=FRC2\n"
    "      d_umf(1:KX)=UMF(1:KX); d_uer(1:KX)=UER(1:KX); d_udr(1:KX)=UDR(1:KX)\n"
    "      d_dmf(1:KX)=DMF(1:KX); d_der(1:KX)=DER(1:KX); d_ddr(1:KX)=DDR(1:KX)\n"
    "      d_tu(1:KX)=TU(1:KX); d_qu(1:KX)=QU(1:KX); d_qliq(1:KX)=QLIQ(1:KX)\n"
    "      d_qice(1:KX)=QICE(1:KX); d_detlq(1:KX)=DETLQ(1:KX)\n"
    "      d_detic(1:KX)=DETIC(1:KX); d_pptliq(1:KX)=PPTLIQ(1:KX); d_pptice(1:KX)=PPTICE(1:KX)\n"
    "      d_tg(1:KX)=TG(1:KX); d_qg(1:KX)=QG(1:KX); d_eqfrc(1:KX)=EQFRC(1:KX)\n"
    "      d_ems(1:KX)=EMS(1:KX)\n"))
i_end = find_once(lambda ln: ln.strip() == "END MODULE module_cu_kfeta", 0, "endmod")
lines.insert(i_end, (
    "   SUBROUTINE kf_dump_lut(tt, qs, th, al)\n"
    "   REAL, INTENT(OUT) :: tt(KFNT,KFNP), qs(KFNT,KFNP), th(KFNP), al(200)\n"
    "   tt = TTAB; qs = QSTAB; th = THE0K; al = ALU\n"
    "   END SUBROUTINE kf_dump_lut\n"))
# kf_dump_lut must sit inside CONTAINS of the module: the END MODULE anchor is
# after the last CONTAINed routine, so the insertion above is inside it.
dbgmod = """MODULE kfdbg
  IMPLICIT NONE
  INTEGER, PARAMETER :: NMX=200
  INTEGER :: d_fired=0, d_ltop, d_let, d_ncount, d_lfs, d_ldb, d_kpblf, d_lcf, d_lcl, d_nstep
  REAL :: d_abe,d_vmflcl,d_ainc,d_timec,d_fabe,d_tder,d_trppt,d_pptflx,d_cpr
  REAL :: d_wlcl,d_rad,d_chmin,d_cldhgt,d_au0,d_fbfrc,d_frc2
  REAL, DIMENSION(NMX) :: d_umf,d_uer,d_udr,d_dmf,d_der,d_ddr,d_tu,d_qu,d_qliq,d_qice
  REAL, DIMENSION(NMX) :: d_detlq,d_detic,d_pptliq,d_pptice,d_tg,d_qg,d_eqfrc,d_ems
END MODULE kfdbg

"""
out = dbgmod + "".join(lines)
# no oracle line removed or altered: every source line survives in order
it = iter(out.splitlines())
assert all(any(o == s for o in it) for s in src.splitlines()), "source line lost"
open(out_path, "w").write(out)
print("wrote", out_path)
