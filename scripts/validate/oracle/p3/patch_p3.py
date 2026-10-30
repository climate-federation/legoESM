"""Write the legoESM-fed scratch copy of WRF v4.8.0 phys/module_mp_p3.F (P3 v4.5.2).

Usage: PYTHONPATH=<legoESM packages> python3 patch_p3.py <WRF phys/module_mp_p3.F> <out_dir>

Writes <out_dir>/module_mp_p3_lego.F and a unified diff next to it.  The
patch is the tier-1 oracle-fixture glue (legoESM constants and saturation fed
to the oracle, as scripts/validate/oracle/clubb_core and morrison_mg2_warm_rain
do), plus instrumentation:

  * p3_init physical constants -> legoesm.constants: cp -> c_pd, g -> g,
    rd -> R_d (WRF P3's rv already equals R_v; rhow = rho_water, cpw = c_pw).
  * latent heats: the WRF P3 constant xxlv (3.1484e6-2370*T0) -> L_v;
    xxls = xxlv + 0.3337e6 is kept (0.3337e6 equals L_f exactly).
  * qv_sat -> legoESM saturation SPECIFIC humidity (lego_sat stub in
    stubs_lego.F90, liquid AND ice; checked against legoesm.thermo by
    make_fixture.py).  polysvp1 is NOT patched: its only other use is
    e0 = polysvp1 at the freezing point (melting / wet growth qsat0).
  * process-rate dumps into diag_3d (instrumentation only, no arithmetic change).

diag_3d slots (rates AFTER every limiter, i.e. the values that update the
state; [kg/kg/s] or [#/kg/s]):
  1 qccon 2 qcevp 3 qrcon 4 qrevp 5 qcaut 6 qcacc 7 ncautr 8 nrslf 9 qccol
  10 qrcol 11 nrcol 12 qidep 13 qisub 14 qinuc 15 ninuc 16 nislf 17 qimlt
  18 nimlt 19 qcheti 20 qrheti 21 qcshd 22 qwgrth 23 rhorime_c 24 nccol
  25 nrevp 26 qcnuc 27 ncheti 28 qc homogeneously frozen / dt 29 qr
  homogeneously frozen / dt 30 nrheti 31 ncautc 32 ncacc 33 nisub 34 epsi_tot
  (stale on levels without hydrometeors) 35 nrshdr 36 ncshdc
"""
import difflib
import os
import sys

from legoesm import constants as C

src, out = sys.argv[1], sys.argv[2]
orig = open(src).read()

WRF_CP = " cp     = 1005.\n"
WRF_G = " g      = 9.816\n"
WRF_RD = " rd     = 287.15\n"
WRF_RV = " rv     = 461.51\n"  # const-ok: WRF P3 source text (anchor), equals legoESM R_v
WRF_XXLV = "       xxlv(i,k)    = 3.1484e6-2370.*273.15 !t(i,k), use constant Lv\n"  # const-ok: WRF source text
WRF_XXLS = "       xxls(i,k)    = xxlv(i,k)+0.3337e6\n"
assert orig.count(WRF_RV) == 1 and abs(float(WRF_RV.split("=")[1]) - C.R_v) == 0.0
assert orig.count(WRF_XXLS) == 1 and float("0.3337e6") == C.L_f  # latent-ok: WRF P3 oracle constant

REPL = [
    (WRF_CP, f" cp     = {C.c_pd!r} !ORACLE-FIXTURE legoESM c_pd (WRF P3: 1005.)\n"),
    (WRF_G, f" g      = {C.g!r} !ORACLE-FIXTURE legoESM g (WRF P3: 9.816)\n"),
    (WRF_RD, f" rd     = {C.R_d!r} !ORACLE-FIXTURE legoESM R_d (WRF P3: 287.15)\n"),
    (WRF_XXLV, f"       xxlv(i,k)    = {C.L_v!r} !ORACLE-FIXTURE legoESM L_v "  # latent-ok: WRF P3 oracle constant
               "(WRF P3: 3.1484e6-2370.*T0)\n"),
    ("#else\n  e_pres = polysvp1(t_atm,i_wrt)\n  qv_sat = ep_2*e_pres/max(1.e-3,(p_atm-e_pres))\n",
     "#else\n!ORACLE-FIXTURE: legoESM saturation SPECIFIC humidity (thermo.py) fed to the oracle\n"
     " if (i_wrt == 1) then\n    qv_sat = qsi_lego(t_atm, p_atm)\n"
     " else\n    qv_sat = qsl_lego(t_atm, p_atm)\n endif\n"),
    (" implicit none\n\n private\n public :: p3_main, polysvp1, p3_init\n",
     " use lego_sat, only: qsl_lego, qsi_lego !ORACLE-FIXTURE\n"
     " implicit none\n\n private\n public :: p3_main, polysvp1, p3_init\n"),
    ("       iice_loop2: do iice = 1,nCat\n",
     "!ORACLE-FIXTURE begin: process-rate dumps (instrumentation only)\n"
     "       diag_3d(i,k,1) = qccon;  diag_3d(i,k,2) = qcevp;  diag_3d(i,k,3) = qrcon\n"
     "       diag_3d(i,k,4) = qrevp;  diag_3d(i,k,5) = qcaut;  diag_3d(i,k,6) = qcacc\n"
     "       diag_3d(i,k,7) = ncautr; diag_3d(i,k,8) = nrslf;  diag_3d(i,k,9) = sum(qccol)\n"
     "       diag_3d(i,k,10) = sum(qrcol); diag_3d(i,k,11) = sum(nrcol); diag_3d(i,k,12) = sum(qidep)\n"
     "       diag_3d(i,k,13) = sum(qisub); diag_3d(i,k,14) = sum(qinuc); diag_3d(i,k,15) = sum(ninuc)\n"
     "       diag_3d(i,k,16) = sum(nislf); diag_3d(i,k,17) = sum(qimlt); diag_3d(i,k,18) = sum(nimlt)\n"
     "       diag_3d(i,k,19) = sum(qcheti); diag_3d(i,k,20) = sum(qrheti); diag_3d(i,k,21) = sum(qcshd)\n"
     "       diag_3d(i,k,22) = sum(qwgrth); diag_3d(i,k,23) = rhorime_c(1); diag_3d(i,k,24) = sum(nccol)\n"
     "       diag_3d(i,k,25) = nrevp; diag_3d(i,k,26) = qcnuc; diag_3d(i,k,27) = sum(ncheti)\n"
     "       diag_3d(i,k,30) = sum(nrheti); diag_3d(i,k,31) = ncautc; diag_3d(i,k,32) = ncacc\n"
     "       diag_3d(i,k,33) = sum(nisub); diag_3d(i,k,34) = epsi_tot\n"
     "       diag_3d(i,k,35) = sum(nrshdr); diag_3d(i,k,36) = sum(ncshdc)\n"
     "!ORACLE-FIXTURE end\n"
     "       iice_loop2: do iice = 1,nCat\n"),
    ("       qc_not_small_2: if (qc(i,k).ge.qsmall .and. t(i,k).lt.233.15) then\n",
     "       qc_not_small_2: if (qc(i,k).ge.qsmall .and. t(i,k).lt.233.15) then\n"
     "          diag_3d(i,k,28) = qc(i,k)*odt !ORACLE-FIXTURE\n"),
    ("       qr_not_small_2: if (qr(i,k).ge.qsmall .and. t(i,k).lt.233.15) then\n",
     "       qr_not_small_2: if (qr(i,k).ge.qsmall .and. t(i,k).lt.233.15) then\n"
     "          diag_3d(i,k,29) = qr(i,k)*odt !ORACLE-FIXTURE\n"),
]
lego = orig
for a, b in REPL:
    assert lego.count(a) == 1, a
    lego = lego.replace(a, b)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "module_mp_p3_lego.F"), "w") as f:
    f.write(lego)
with open(os.path.join(out, "module_mp_p3_lego.F.diff"), "w") as f:
    f.writelines(difflib.unified_diff(orig.splitlines(True), lego.splitlines(True),
                                      "module_mp_p3.F", "module_mp_p3_lego.F"))
print("PATCH_OK")
