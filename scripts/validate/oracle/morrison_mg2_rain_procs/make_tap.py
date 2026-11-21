"""Write a probe COPY of CAM's micro_mg2_0.F90 that records MG2's rain
process rates and rain fall-speed inputs in public module arrays.  No MG2
arithmetic changes: only assignments to the new arrays are inserted, at
anchors that must each match exactly once (checked).  Python text insertion
instead of sed so it runs with the BSD sed of macOS.

Recorded (per column i, level k):
  rtap_proc(:,:,1:12) just before the "administration" loop
    (micro_mg2_0.F90:1855, every limiter applied): precip_frac, lcldm, qric,
    nric (size-adjusted), lamr (pre-process), prc, nprc, pra, npra, nragg,
    pre, nsubr (in-cloud / in-precip rates, before the fraction weights)
  rtap_fall(:,:,1:2) before the post-process rain slope (:2158-2161): the
    in-precip dumr, dumnr fed to size_dist_param_basic
  rtap_fall(:,:,3:5) after the 'final' rain fall speeds (:2171-2175): unr,
    lamr, rhof
  rtap_nstep(i) the rain sedimentation sub-step count (before :2395)

usage: python make_tap.py <CAM micro_mg2_0.F90> <out .F90>
"""
import sys

DECL = """real(r8), public :: rtap_proc(1,512,12) = 0._r8
real(r8), public :: rtap_fall(1,512,5) = 0._r8
integer, public :: rtap_nstep(1) = 0
contains
"""
PROC = ["precip_frac", "lcldm", "qric", "nric", "lamr", "prc", "nprc", "pra", "npra",
        "nragg", "pre", "nsubr"]
EDITS = [
    ("contains\n", DECL, "replace_first_line"),
    ('     ! Big "administration" loop enforces conservation, updates variables\n',
     "".join(f"     rtap_proc(1:mgncol,k,{j + 1}) = {v}(:,k)\n" for j, v in enumerate(PROC)),
     "before"),
    ("        ! fallspeed for rain\n",
     "        rtap_fall(1:mgncol,k,1) = dumr(:,k)\n        rtap_fall(1:mgncol,k,2) = dumnr(:,k)\n",
     "after"),
    ("           fnr(i,k) = g*rho(i,k)*unr(i,k)\n",
     "           rtap_fall(i,k,3) = unr(i,k)\n           rtap_fall(i,k,4) = lamr(i,k)\n"
     "           rtap_fall(i,k,5) = rhof(i,k)\n",
     "after"),
    ("        faloutr  = fr(i,:)  * dumr(i,:)\n", "        rtap_nstep(i) = nstep\n", "before"),
]


def main(src, dst):
    lines = open(src).read().splitlines(keepends=True)
    out = list(lines)
    for anchor, text, mode in EDITS:
        hits = [n for n, l in enumerate(out) if l == anchor]
        if mode == "replace_first_line":
            hits = hits[:1]
        assert len(hits) == 1, (anchor, len(hits))
        n = hits[0]
        if mode == "before":
            out[n:n] = [text]
        elif mode == "after":
            out[n + 1:n + 1] = [text]
        else:
            out[n] = text
    open(dst, "w").write("".join(out))
    added = sum(t.count("\n") for _, t, _ in EDITS) - 1
    print("TAP_OK", dst, "lines added", added)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
