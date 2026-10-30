"""Make the patched oracle copy zm_conv_patched.F90 from CAM cam_cesm2_1_rel_60.

Patches (all recorded in README.md):
 P1  drop zm_mphy from the zm_microphysics use list and delete the
     `call zm_mphy(...)` statement: dead code (zmconv_microp=.false. in CAM6),
     removed only so the file compiles without the convective-microphysics stack.
 P2  ientropy Brent tolerances become module variables zm_ient_tol / zm_ient_eps
     (defaults = stock 0.001 K / 3e-8), so the harness can tighten the root
     solve to isolate the Brent-tolerance effect.  Stock values reproduce CAM.
 P4  diagnostic counter dbg_nobracket(rcall): ientropy calls whose initial
     [Tfg-10, Tfg+10] bracket has no sign change (Brent then cannot find the root).
 P3  diagnostic copies (dbg_*) of internal per-column quantities (mb, lcl, lel,
     mx, tl, jd, j0, jlcl, tp, lengath).  Assignments only; no numerics change.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
out = []
i = 0
n_p1 = n_p2 = n_p3 = n_p4 = 0
while i < len(lines):
    ln = lines[i]
    if ln.strip() == "use zm_microphysics, only: zm_mphy, zm_aero_t, zm_conv_t":
        out.append(ln.replace("zm_mphy, ", ""))
        n_p1 += 1
        i += 1
        continue
    if ln.strip().startswith("call  zm_mphy("):
        while lines[i].rstrip().endswith("&"):
            i += 1
        i += 1  # last continuation line
        out.append("         ! [patch P1] call zm_mphy(...) removed (dead: zmconv_microp=.false.)")
        n_p1 += 1
        continue
    if ln.strip().startswith("real(r8),parameter ::  tiedke_add = 0.5_r8"):
        out.append(ln)
        out += [
            "   ! [patch P2] ientropy tolerances (stock values)",
            "   real(r8), public :: zm_ient_tol = 0.001_r8",
            "   real(r8), public :: zm_ient_eps = 3.e-8_r8",
            "   ! [patch P3] diagnostic exports",
            "   real(r8), public :: dbg_mb(pcols), dbg_tl(pcols), dbg_tp(pcols,pver)",
            "   integer,  public :: dbg_lcl(pcols), dbg_lel(pcols), dbg_mx(pcols), dbg_jd(pcols), &",
            "                       dbg_j0(pcols), dbg_jlcl(pcols), dbg_lengath",
            "   integer,  public :: dbg_nobracket(pcols,4) = 0   ! [patch P4] per column, per ientropy call site; ientropy calls whose +-10 K bracket has no sign change",
        ]
        n_p2 += 1
        i += 1
        continue
    if ln.strip() == "real(r8), parameter :: EPS = 3.e-8_r8":
        out.append("  real(r8) :: EPS   ! [patch P2]")
        n_p2 += 1
        i += 1
        continue
    if ln.strip() == "fb = entropy(b, p, qt) - s" and n_p4 == 0:
        out.append(ln)
        out.append("  if (fa*fb > 0._r8) dbg_nobracket(icol,rcall) = dbg_nobracket(icol,rcall) + 1   ! [patch P4]")
        n_p4 += 1
        i += 1
        continue
    if ln.strip() == "tol=0.001_r8":
        out.append("  tol=zm_ient_tol   ! [patch P2]")
        out.append("  EPS=zm_ient_eps   ! [patch P2]")
        n_p2 += 1
        i += 1
        continue
    if ln.strip() == "if (lengath.eq.0) return":
        out += [
            "   ! [patch P3]",
            "   dbg_mb = 0._r8; dbg_jd = 0; dbg_j0 = 0; dbg_jlcl = 0",
            "   dbg_lcl(:ncol) = lcl(:ncol); dbg_lel(:ncol) = lel(:ncol); dbg_mx(:ncol) = maxi(:ncol)",
            "   dbg_tl(:ncol) = tl(:ncol); dbg_tp(:ncol,:) = tp(:ncol,:); dbg_lengath = lengath",
        ]
        out.append(ln)
        n_p3 += 1
        i += 1
        continue
    if ln.strip() == "call q1q2_pjr(lchnk   , &":
        out += [
            "   ! [patch P3]",
            "   do i = 1,lengath",
            "      dbg_mb(ideep(i)) = mb(i); dbg_jd(ideep(i)) = jd(i)",
            "      dbg_j0(ideep(i)) = j0(i); dbg_jlcl(ideep(i)) = jlcl(i)",
            "   end do",
        ]
        out.append(ln)
        n_p3 += 1
        i += 1
        continue
    out.append(ln)
    i += 1

assert (n_p1, n_p2, n_p3, n_p4) == (2, 3, 2, 1), (n_p1, n_p2, n_p3, n_p4)
open(dst, "w").write("\n".join(out))
print("patched", (n_p1, n_p2, n_p3))
