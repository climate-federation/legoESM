"""Instrument convect43c.f (CONVECT 4.3c) for the oracle audit.

Patches (all by original 1-based line number, each asserted against the
expected original text so a different source file fails loudly):

P1 constants CPD,CPV,CL,RV,RD,LV0,G (CONVECT l.210-216, TLIFT l.1010-1015)
   read from COMMON /EMCONST/ (driver sets legoESM or stock values).
P2 COEFFR/COEFFS (l.196-197) read from COMMON (no-evaporation variant).
P3 IDDOFF flag: l.717 ``IF(EP(INB).LT.0.0001)GOTO 405`` also skips the
   precipitating downdraft when IDDOFF=1 (mixing-only variant).
P4 ICOLD flag: TLIFT l.1061 cold branch (TC<0) es formula used only when
   ICOLD=1 (stock); ICOLD=0 uses the warm Tetens branch everywhere
   (diagnostic variant isolating that divergence).
P5 debug copies into COMMON /EMDBG/ (assignments only, no arithmetic
   change): after l.377 (NK), l.408 (ICB,PLCL), l.567 (closure, INB,
   profiles), l.712 (mixing matrix), l.938 (FT/FQ pre-FRAC, downdraft),
   l.965 (FT/FQ pre-ENTS).
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
L = open(src).read().split("\n")


def expect(n, text):
    assert text in L[n - 1], (n, L[n - 1], text)


DECL = ["      INTEGER IDDOFF,ICOLD",
        "      COMMON /EMCONST/ XCPD,XCPV,XCL,XRV,XRD,XLV0,XG",
        "      COMMON /EMCONST/ XCOEFFR,XCOEFFS,IDDOFF,ICOLD"]
DBGINC = ["      INCLUDE 'emdbg.inc'"]

rep = {}   # line -> replacement list (replaces the line)
ins = {}   # line -> list inserted AFTER the line

# P1/P2 CONVECT constants
for n, old, new in [(196, "COEFFR=1.0", "        COEFFR=XCOEFFR"),
                    (197, "COEFFS=0.8", "        COEFFS=XCOEFFS"),
                    (210, "CPD=1005.7", "      CPD=XCPD"),
                    (211, "CPV=1870.0", "      CPV=XCPV"),
                    (212, "CL=2500.0", "      CL=XCL"),
                    (213, "RV=461.5", "      RV=XRV"),
                    (214, "RD=287.04", "      RD=XRD"),
                    (215, "LV0=2.501E6", "      LV0=XLV0"),
                    (216, "G=9.8", "      G=XG"),
                    (1010, "CPD=1005.7", "        CPD=XCPD"),
                    (1011, "CPV=1870.0", "        CPV=XCPV"),
                    (1012, "CL=2500.0", "        CL=XCL"),
                    (1013, "RV=461.5", "        RV=XRV"),
                    (1014, "RD=287.04", "        RD=XRD"),
                    (1015, "LV0=2.501E6", "        LV0=XLV0")]:
    expect(n, old)
    rep[n] = [new]

expect(140, "REAL LV(NA),LVCP(NA),LV0")
ins[140] = DECL + DBGINC
expect(1006, "REAL T(ND),Q(ND),QS(ND),TVP(ND),LV0")
ins[1006] = DECL

# P3
expect(717, "IF(EP(INB).LT.0.0001)GOTO 405")
rep[717] = ["        IF(EP(INB).LT.0.0001.OR.IDDOFF.EQ.1)GOTO 405"]
# P4
expect(1061, "IF(TC.GE.0.0)THEN")
rep[1061] = ["          IF(TC.GE.0.0.OR.ICOLD.EQ.0)THEN"]

# P5 debug copies
expect(377, "42   CONTINUE")
ins[377] = ["        DNK=NK", "        DIHMIN=IHMIN", "        DSTAGE=1"]
expect(408, "50   CONTINUE")
ins[408] = ["        DICB=ICB", "        DPLCL=PLCL", "        DSTAGE=2"]
expect(567, "CBMF=MAX(CBMF,0.0)")
ins[567] = ["      DDTMA=DTMA", "      DCBMF=CBMF", "      DTVPPLCL=TVPPLCL",
            "      DTVAPLCL=TVAPLCL", "      DDTPBL=DTPBL", "      DINB=INB",
            "      DINB1=INB1", "      DFRAC=FRAC", "      DSTAGE=3",
            "      DO 9001 I=1,ND",
            "       DTVP(I)=TVP(I)", "       DTP(I)=TP(I)", "       DEP(I)=EP(I)",
            "       DCLW(I)=CLW(I)", "       DHP(I)=HP(I)", "       DTV(I)=TV(I)",
            "       DGZ(I)=GZ(I)", "       DH(I)=H(I)",
            " 9001 CONTINUE"]
expect(712, "200   CONTINUE")
ins[712] = ["        DSTAGE=4", "        DO 9002 I=1,ND", "         DM(I)=M(I)",
            "         DO 9002 J=1,ND", "          DMENT(I,J)=MENT(I,J)",
            "          DQENT(I,J)=QENT(I,J)", "          DELIJ(I,J)=ELIJ(I,J)",
            "          DSIJ(I,J)=SIJ(I,J)", " 9002   CONTINUE"]
expect(938, "500   CONTINUE")
ins[938] = ["        DSTAGE=5", "        DO 9003 I=1,ND", "         DFT1(I)=FT(I)",
            "         DFQ1(I)=FQ(I)", "         DMP(I)=MP(I)",
            "         DEVAP(I)=EVAP(I)", "         DWATER(I)=WATER(I)",
            "         DQP(I)=QP(I)", " 9003   CONTINUE"]
expect(966, "Very slightly adjust")
ins[965] = ["        DSTAGE=6", "        DO 9004 I=1,ND", "         DFT2(I)=FT(I)",
            "         DFQ2(I)=FQ(I)", " 9004   CONTINUE"]

out = []
for n, line in enumerate(L, start=1):
    out.extend(rep.get(n, [line]))
    out.extend(ins.get(n, []))
open(dst, "w").write("\n".join(out))
print("patched", len(rep), "replacements,", len(ins), "insertions")
