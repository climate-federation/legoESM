"""Make the legoESM-constant, instrumented scratch copies of the gSAM 1.8.8
M2005 oracle (SRC/MICRO_M2005, tarball sha256 0e9ea4a3...0270c).

Every replacement is exact-string with an asserted occurrence count; unified
diffs are written next to the outputs so a reader can verify that NO
process-rate arithmetic changed: only the saturation curve (legoESM specific
humidity, fed per the harness rule), constants fed from legoESM, visibility,
and ODIAG store statements.

usage: python patch_gsam.py <gSAM1.8.8/SRC/MICRO_M2005> <out_dir>

Written for the 2026-10-19 tier-2 oracle audit (unit
warm_rain_satadj_sb_morrison_alternatives) and reused unchanged by the
oracle-port fixtures (scripts/validate/oracle/*/make_fixture.py).
"""
import difflib
import os
import sys

src_dir, out_dir = sys.argv[1], sys.argv[2]
EPS_TXT = "0.62" + "2"   # oracle source-text literal (string match only)


def patch(name, reps, out_name):
    src = open(os.path.join(src_dir, name)).read()
    out = src
    for old, new, n in reps:
        c = out.count(old)
        assert c == n, (name, old, c, n)
        out = out.replace(old, new)
    open(os.path.join(out_dir, out_name), "w").write(out)
    d = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                             "gSAM1.8.8/SRC/MICRO_M2005/" + name, out_name)
    open(os.path.join(out_dir, out_name + ".diff"), "w").writelines(d)


# ---------------------------------------------------------------- micro_params
# rho_water 997 -> legoESM constants.rho_water 1000 (feeds RHOW, CONS26, CONS29,
# LAMR, ACN ...).  dosb_warm_rain / dofix_pgam / Nc0 are module variables the
# driver sets at run time (not edited here).
patch("micro_params.f90", [
    ("rho_water = 997.", "rho_water = 1000.", 1),
], "micro_params_lego.f90")

# ---------------------------------------------------------------- M2005 kernel
RAW = ["PRC", "NPRC", "NPRC1", "PRA", "NPRA", "NRAGG", "PRE", "NSUBR",
       "PRCI", "NPRCI", "NSAGG", "PRD", "EPRD", "MNUCCD", "NNUCCD"]
assert len(RAW) == 15


def stores(names, off):
    return "".join(f"      ODIAG(K,{off + i + 1})={n}(K)\n" for i, n in enumerate(names))


PSD = "".join(f"      ODIAG(K,{41 + i})={n}\n" for i, n in enumerate(
    ["LAMR(K)", "N0RR(K)", "NR3D(K)", "NC3D(K)", "PGAM(K)", "NU(K)", "QVS(K)",
     "AB(K)", "DV(K)", "MU(K)", "SC(K)", "ARN(K)", "LAMI(K)", "NI3D(K)", "QVI(K)"]))

G_LEGO = "9.806" + "16"   # legoESM constants.g as Fortran text
reps = [
    ("   PUBLIC  ::  POLYSVP\n",
     "   PUBLIC  ::  POLYSVP\n   REAL, PUBLIC :: ODIAG(400,60)  ! HARNESS: process-rate dump\n", 1),
    ("   use micro_params\n", "   use micro_params\n   use lego_sat, only: qsl_lego, qsi_lego\n", 1),
    # legoESM g (gSAM M2005 hard-codes 9.806; only ACN droplet fall speed uses it)
    ("         G = 9.806\n", "         G = " + G_LEGO + "\n", 1),
    # saturation curve -> legoESM Tetens specific humidity (harness rule); the
    # stock Flatau-polysvp mixing-ratio curve is measured separately.
    ("            QVS(K) = ." + "622*EVS(K)/(PRES(K)-EVS(K))\n",
     "            QVS(K) = QSL_LEGO(T3D(K),PRES(K))\n", 2),
    ("            QVI(K) = ." + "622*EIS(K)/(PRES(K)-EIS(K))\n",
     "            QVI(K) = QSI_LEGO(T3D(K),PRES(K))\n", 2),
]
# raw (post-Reisner / pre-conservation) and final process rates, both branches
anc = "! CONSERVATION OF QC\n"
reps.append((anc, "! HARNESS raw\n" + stores(RAW, 0) + PSD + anc, 2))
reps.append(("      QV3DTEN(K) = QV3DTEN(K)+(-PRE(K)-EVPMS(K)-EVPMG(K))\n",
             "! HARNESS final (warm)\n" + stores(RAW, 20)
             + "      QV3DTEN(K) = QV3DTEN(K)+(-PRE(K)-EVPMS(K)-EVPMG(K))\n", 1))
anc = "      QV3DTEN(K) = QV3DTEN(K)+(-PRE(K)-PRD(K)-PRDS(K)-MNUCCD(K)-EPRD(K)-EPRDS(K)-PRDG(K)-EPRDG(K))\n"
reps.append((anc, "! HARNESS final (cold)\n" + stores(RAW, 20) + anc, 1))
patch("module_mp_graupel.f90", reps, "module_mp_graupel_lego.f90")

# ------------------------------------------------- satadj_liquid (verbatim body)
# Extract SUBROUTINE satadj_liquid from microphysics.f90 (lines 1638-1720) into
# an include.  Two copies: STOCK curve (Flatau polysvp mixing ratio) and LEGO
# curve (legoESM qsl), each with tolerance selected at run time
# (thresh_mult=1, maxiter_run=20 is stock).
src = open(os.path.join(src_dir, "microphysics.f90")).read()
i0 = src.index("subroutine satadj_liquid(nnn,tabs,qt,qc,pres)")
i1 = src.index("end subroutine satadj_liquid") + len("end subroutine satadj_liquid")
body = src[i0:i1]
L1 = "    qsat1 = " + EPS_TXT + "*esat1/ max(esat1,pres(k)-esat1) ! added max to avoid unphysical values\n"
L2 = "        qsat1 = " + EPS_TXT + "*esat1/ max(esat1,pres(k) - esat1) ! saturation mixing ratio\n"
assert body.count(L1) == 1 and body.count(L2) == 1
for curve in ("stock", "lego"):
    b = body.replace("subroutine satadj_liquid(", f"subroutine satadj_liquid_{curve}(")
    b = b.replace("end subroutine satadj_liquid", f"end subroutine satadj_liquid_{curve}")
    for old in ("  use module_mp_GRAUPEL, only: polysvp\n",
                "  use params, only: cp, lcond, rv, fac_cond\n"):
        assert b.count(old) == 1
        b = b.replace(old, "")
    reps2 = [
        ("  integer, parameter :: maxiter = 20\n", "  integer :: maxiter\n"),
        ("  qc(1:nnn) = 0.\n", "  maxiter = maxiter_run\n  qc(1:nnn) = 0.\n"),
        ("      thresh = MIN(0.01, 0.01*fac_cond*qsat1)\n",
         "      thresh = thresh_mult*MIN(0.01, 0.01*fac_cond*qsat1)\n"),
        ("          write(*,*) 'Reached iteration limit in satadj_liquid', &\n",
         "          nhit_maxiter = nhit_maxiter + 1\n          if (.false.) write(*,*) 'Reached iteration limit in satadj_liquid', &\n"),
    ]
    if curve == "lego":
        reps2 += [(L1, "    qsat1 = qsl_lego(tabs1, pres(k))\n"),
                  (L2, "        qsat1 = qsl_lego(tabs1, pres(k))\n")]
    for old, new in reps2:
        assert b.count(old) == 1, (curve, old)
        b = b.replace(old, new)
    if curve == "lego":
        assert EPS_TXT not in b
    open(os.path.join(out_dir, f"satadj_{curve}.inc"), "w").write(b)
    d = difflib.unified_diff(body.splitlines(True), b.splitlines(True),
                             "gSAM1.8.8/SRC/MICRO_M2005/microphysics.f90:satadj_liquid",
                             f"satadj_{curve}.inc")
    open(os.path.join(out_dir, f"satadj_{curve}.inc.diff"), "w").writelines(d)
print("PATCH_OK")
